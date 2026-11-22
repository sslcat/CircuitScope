"""Repeat the first held-out batch on CPU in FP32 with the frozen head choices."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from transformers import GPT2LMHeadModel

from .checkpoint import verify
from .model import forward
from .run import write_json
from .task import batches, metrics


def compare(path, weights):
    cfg = json.loads((path / "run.json").read_text())["config"]
    heads = json.loads((path / "groups.json").read_text())["candidate_pair"]
    seed = cfg["test_seeds"][0]
    b = next(batches(cfg, seed, cfg["test_n"], "cpu"))
    verify(weights)
    torch.set_num_threads(4)
    model = GPT2LMHeadModel.from_pretrained(weights, local_files_only=True, use_safetensors=True,
                                            attn_implementation="eager", dtype=torch.float32).eval()
    clean, donor = forward(model, b.clean, capture=True)
    swap, _ = forward(model, b.swap)

    def margin(logits):
        return metrics(logits, b)["margin"]

    gap = margin(clean) - margin(swap)
    clean_zero, _ = forward(model, b.clean, heads)
    swap_zero, _ = forward(model, b.swap, heads)
    patched, _ = forward(model, b.swap, heads, donor)
    current = {"gap": gap, "ablate_candidate_pair": gap - (margin(clean_zero) - margin(swap_zero)),
               "patch_candidate_pair": margin(patched) - margin(swap)}
    with (path / "trials.csv").open() as f:
        rows = [r for r in csv.DictReader(f) if int(r["seed"]) == seed][:len(clean)]
    if [int(r["target"]) for r in rows] != b.target.tolist():
        raise ValueError("Sample mismatch")
    result = {"seed": seed, "n": len(rows), "metrics": {}}
    for key, values in current.items():
        ref = np.array([float(r[key]) for r in rows])
        diff = values.numpy() - ref
        result["metrics"][key] = {"cpu_fp32_mean": float(values.mean()), "original_mean": ref.mean(),
                                  "max_abs_difference": float(np.abs(diff).max())}
    write_json(path / "precision_check.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--weights", type=Path, default=Path(".weights"))
    args = parser.parse_args()
    compare(args.run, args.weights)
