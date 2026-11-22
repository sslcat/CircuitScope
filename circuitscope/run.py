"""Run discovery, held-out interventions and the paired statistical analysis."""
import argparse
import csv
import json
import platform
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from transformers import GPT2LMHeadModel

from .analysis import enrich, select_heads, summarize
from .checkpoint import MODEL, REVISION, WEIGHT_HASH, verify
from .model import Trace, forward
from .task import batches, metrics


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf8", newline="\n")


def discover(model, cfg, device, out):
    total = np.zeros((model.config.n_layer, model.config.n_head))
    saved = {}
    for batch in batches(cfg, cfg["discovery_seed"], cfg["discovery_n"], device):
        first = not saved
        for name, tokens, sign in (("clean", batch.clean, 1), ("swap", batch.swap, -1)):
            _, trace = forward(model, tokens, patterns=True)
            for layer, p in trace.patterns.items():
                target_index = batch.target_pos[:, None, None].expand(-1, p.shape[1], 1)
                weights = p[:, :, -1].gather(2, target_index)
                total[layer] += sign * weights.float().sum(0).flatten().cpu().numpy()
                if first:
                    saved[f"{name}_{layer}"] = p[0].float().cpu().numpy()
        if first:
            saved["positions"] = [batch.target_pos[0].item(), batch.foil_pos[0].item()]
            saved["clean_ids"] = batch.clean[0].cpu().numpy()
            saved["swap_ids"] = batch.swap[0].cpu().numpy()
    scores = total / cfg["discovery_n"]
    np.savez_compressed(out / "discovery.npz", scores=scores, **saved)
    return select_heads(scores, cfg["control_seed"])


def remove_source(model, tokens, trace, head, pos):
    """Subtract one source's weighted value, preserving the original softmax."""
    layer, h = head
    row = torch.arange(len(tokens), device=tokens.device)
    weight = trace.patterns[layer][row, h, -1, pos].float()
    value = trace.values[layer][row, pos, h].float()
    heads = trace.heads[layer].clone()
    heads[:, h] = heads[:, h].float() - weight[:, None] * value
    donor = Trace(heads={layer: heads})
    return forward(model, tokens, [head], donor)[0]


def evaluate_batch(model, batch, groups, seed, offset, qc):
    clean, source = forward(model, batch.clean, capture=True, patterns=True)
    swap, other = forward(model, batch.swap, capture=True, patterns=True)
    neutral, _ = forward(model, batch.neutral)
    base = {k: metrics(x, batch) for k, x in (("clean", clean), ("swap", swap), ("neutral", neutral))}
    values = {f"{k}_{metric}": v for k, series in base.items() for metric, v in series.items()}
    gap = base["clean"]["margin"] - base["swap"]["margin"]
    values["gap"] = gap
    values["target_logp_gain"] = base["clean"]["logp"] - base["swap"]["logp"]
    values["gap_nonpositive"] = (gap <= 0).float()
    if offset == 0:
        identity, _ = forward(model, batch.swap, groups["candidate_pair"], other)
        restored, _ = forward(model, batch.swap, donor=source, patch_residual=True)
        for key, actual, expected in (("self_patch", identity, swap), ("residual_patch", restored, clean)):
            error = (actual - expected).abs().max().item()
            qc[key] = max(qc.get(key, 0.0), error)
            if not torch.allclose(actual, expected, rtol=1e-5, atol=1e-5):
                raise RuntimeError(f"{key} failed at seed {seed}, max error {error}")
    for label, heads in groups.items():
        margins = [metrics(forward(model, tokens, heads)[0], batch)["margin"]
                   for tokens in (batch.clean, batch.swap, batch.neutral)]
        values[f"ablate_{label}"] = gap - (margins[0] - margins[1])
        values[f"neutral_shift_{label}"] = base["neutral"]["margin"] - margins[2]
        if label.startswith("layer_"):
            continue
        patched, _ = forward(model, batch.swap, heads, donor=source)
        gain = metrics(patched, batch)["margin"] - base["swap"]["margin"]
        values[f"patch_{label}"] = gain
        if label == "candidate_pair":
            values["patch_nonpositive"] = (gain <= 0).float()
    for label in ("candidate_1", "candidate_2", "same_1", "same_2"):
        head = groups[label][0]
        for link, clean_pos, swap_pos in (("bound", batch.target_pos, batch.foil_pos),
                                          ("other", batch.foil_pos, batch.target_pos)):
            clean_cut = remove_source(model, batch.clean, source, head, clean_pos)
            swap_cut = remove_source(model, batch.swap, other, head, swap_pos)
            cut_gap = metrics(clean_cut, batch)["margin"] - metrics(swap_cut, batch)["margin"]
            values[f"source_{link}_{label}"] = gap - cut_gap
        values[f"source_specificity_{label}"] = (values[f"source_bound_{label}"]
                                                - values[f"source_other_{label}"])
    fields = list(values)
    matrix = torch.stack(list(values.values()), dim=1).cpu().tolist()
    rows = [dict(zip(fields, row)) for row in matrix]
    for i, row in enumerate(rows):
        row.update(seed=seed, trial=offset+i, target=batch.target[i].item(), foil=batch.foil[i].item(),
                   target_pos=batch.target_pos[i].item(), foil_pos=batch.foil_pos[i].item())
    return rows


def validate(cfg):
    for key in ("batch_size", "discovery_n", "test_n", "bootstrap_n"):
        if cfg[key] <= 0:
            raise ValueError(f"{key} must be positive")
    seeds = cfg["test_seeds"]
    if not seeds or len(set(seeds)) != len(seeds) or cfg["discovery_seed"] in seeds:
        raise ValueError("Discovery and test seeds must be distinct")
    if cfg["prefix_length"] + 2 > 1024:
        raise ValueError("Prompt exceeds GPT-2 context window")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--weights", type=Path, default=Path(".weights"))
    parser.add_argument("--out", type=Path, default=Path("reproduction"))
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--precision", choices=["auto", "float32", "float16"], default="auto")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    validate(cfg)
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError("Choose a fresh output directory to keep runs separate")
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    torch.set_num_threads(min(6, torch.get_num_threads()))
    torch.manual_seed(cfg["discovery_seed"])
    torch.set_float32_matmul_precision("highest")
    has_cuda = torch.cuda.is_available()
    free = torch.cuda.mem_get_info()[0] / 2**20 if has_cuda else 0
    device = args.device
    if device == "auto":
        device = "cuda" if has_cuda and free >= 500 else "cpu"
    dtype = args.precision
    if dtype == "auto":
        dtype = "float16" if device == "cuda" and free < 1300 else "float32"
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    verify(args.weights)
    model, info = GPT2LMHeadModel.from_pretrained(
        args.weights, local_files_only=True, use_safetensors=True, attn_implementation="eager",
        dtype=getattr(torch, dtype), output_loading_info=True)
    if set(info["missing_keys"]) - {"lm_head.weight"} or info["mismatched_keys"] or info["error_msgs"]:
        raise RuntimeError(f"Incomplete checkpoint: {info}")
    model.eval().to(device)
    print(f"Running on {device}, {dtype}; {free:.0f} MiB initially free", flush=True)
    groups = discover(model, cfg, device, args.out)
    write_json(args.out / "groups.json", groups)
    print(f"Discovery candidates {groups['candidate_pair']}", flush=True)
    rows, qc = [], {}
    for seed in cfg["test_seeds"]:
        offset = 0
        for batch in batches(cfg, seed, cfg["test_n"], device):
            new_rows = evaluate_batch(model, batch, groups, seed, offset, qc)
            rows.extend(new_rows)
            offset += len(new_rows)
        print(f"Finished held-out seed {seed} in {time.perf_counter()-start:.1f} s cumulative", flush=True)
    enrich(rows)
    with (args.out / "trials.csv").open("w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    write_json(args.out / "summary.json", summarize(rows, cfg["bootstrap_n"], cfg["bootstrap_seed"]))
    metadata = dict(model=MODEL, revision=REVISION, sha256=WEIGHT_HASH, params=model.num_parameters(),
                    config=cfg, device=device, dtype=dtype, platform=platform.system(),
                    python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
                    packages={p: version(p) for p in ("transformers", "numpy", "safetensors", "matplotlib")},
                    qc_max_abs_logit_error=qc, threads=torch.get_num_threads())
    if device == "cuda":
        torch.cuda.synchronize()
        metadata.update(gpu=torch.cuda.get_device_name(), initial_free_mib=free,
                        gpu_total_mib=torch.cuda.get_device_properties(0).total_memory / 2**20,
                        peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                        peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20)
    metadata["runtime_seconds"] = time.perf_counter() - start
    write_json(args.out / "run.json", metadata)


if __name__ == "__main__":
    main()
