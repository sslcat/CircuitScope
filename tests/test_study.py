import json
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from circuitscope.analysis import enrich, select_heads, summarize
from circuitscope.model import forward
from circuitscope.run import discover, evaluate_batch, remove_source, validate
from circuitscope.task import batches, metrics


@pytest.fixture
def cfg():
    return json.loads(Path("config.json").read_text()) | {"batch_size": 3, "discovery_n": 5}


@pytest.fixture
def model():
    torch.manual_seed(13)
    torch.set_num_threads(2)
    config = GPT2Config(n_layer=2, n_head=4, n_embd=16, n_positions=40, vocab_size=50257,
                        attn_implementation="eager")
    return GPT2LMHeadModel(config).eval()


def test_alignment_and_frequency(cfg):
    for b in batches(cfg, 41, 40, "cpu"):
        for clean, swap, neutral, t, f in zip(b.clean, b.swap, b.neutral, b.target_pos, b.foil_pos):
            assert clean.sort().values.equal(swap.sort().values)
            assert (clean != swap).sum() == 2
            assert clean[-1] == clean[t-1] == swap[f-1] == swap[-1]
            assert clean[t] == swap[t] and clean[f] == swap[f]
            assert clean[t] != clean[f] and clean[t] != clean[-1]
            assert clean[1:-1].unique().numel() == cfg["prefix_length"]
            assert clean[:-1].equal(neutral[:-1]) and neutral[-1] not in clean
            assert clean[0] == 50256


def test_seed_and_batch_invariance(cfg):
    def generate(seed, size):
        return torch.cat([b.clean for b in batches(cfg | {"batch_size": size}, seed, 7, "cpu")])
    assert generate(41, 1).equal(generate(41, 4))
    assert not generate(41, 1).equal(generate(43, 1))


def test_splits(cfg):
    validate(cfg)
    with pytest.raises(ValueError):
        validate(cfg | {"test_seeds": [cfg["discovery_seed"]]})


def test_metrics(cfg):
    b = next(batches(cfg, 1, 2, "cpu"))
    logits = torch.zeros(2, 50257)
    logits[torch.arange(2), b.target] = torch.tensor([2.0, -1.0])
    result = metrics(logits, b)
    torch.testing.assert_close(result["margin"], torch.tensor([2., -1.]))
    assert result["top1"].tolist() == result["wins"].tolist() == [1., 0.]


def test_hooks_match_native_and_attention(model, cfg):
    b = next(batches(cfg, 1, 2, "cpu"))
    y, cache = forward(model, b.clean, capture=True, patterns=True)
    with torch.inference_mode():
        native = model(b.clean, use_cache=False).logits[:, -1]
    torch.testing.assert_close(y, native)
    p = cache.patterns[0]
    assert torch.count_nonzero(torch.triu(p, diagonal=1)) == 0
    torch.testing.assert_close(p.sum(-1), torch.ones_like(p.sum(-1)))


@pytest.mark.parametrize("patch", [False, True])
def test_only_requested_head_and_position_change(model, cfg, patch):
    b = next(batches(cfg, 1, 2, "cpu"))
    _, source = forward(model, b.clean, capture=True)
    # A forward hook sees the actual projection input after all pre-hooks.
    seen = []
    projection = model.transformer.h[0].attn.c_proj
    with projection.register_forward_hook(lambda m, args, out: seen.append(args[0].clone())):
        forward(model, b.swap)
        forward(model, b.swap, [(0, 2)], source if patch else None)
        forward(model, b.swap)
    orig, actual, repeated = [s.reshape(2, -1, 4, 4) for s in seen]
    expected = orig.clone()
    expected[:, -1, 2] = source.heads[0][:, 2] if patch else 0
    assert actual.equal(expected) and not orig.equal(expected)
    assert repeated.equal(orig)


def test_source_term_reconstructs_query_and_cut(model, cfg):
    b = next(batches(cfg, 43, 2, "cpu"))
    _, tr = forward(model, b.clean, capture=True, patterns=True)
    weights = tr.patterns[0][:, 1, -1]
    values = tr.values[0][:, :, 1]
    terms = weights.unsqueeze(-1) * values
    torch.testing.assert_close(terms.sum(1), tr.heads[0][:, 1])
    expected = terms.clone()
    expected[torch.arange(2), b.target_pos] = 0
    seen = []
    with model.transformer.h[0].attn.c_proj.register_forward_hook(
            lambda m, args, out: seen.append(args[0].clone())):
        remove_source(model, b.clean, tr, (0, 1), b.target_pos)
    actual = seen[0][:, -1].reshape(2, 4, 4)[:, 1]
    torch.testing.assert_close(actual, expected.sum(1))


def test_controls_and_bootstrap():
    scores = np.arange(24).reshape(3, 8)
    groups = select_heads(scores, 208)
    assert groups["candidate_pair"] == [(2, 7), (2, 6)]
    six = groups["candidate_pair"] + groups["same_pair"] + groups["random_pair"]
    assert len(set(six)) == 6 and all(h[0] == 2 for h in groups["same_pair"])
    rows = [{"seed": seed, "constant": float(seed)} for seed in [1, 1, 3, 3]]
    stat = summarize(rows, 100, 10)["constant"]
    assert stat["mean"] == stat["lo"] == stat["hi"] == 2.


def test_cpu_pipeline(model, cfg, tmp_path):
    groups = discover(model, cfg, "cpu", tmp_path)
    b = next(batches(cfg, 47, 2, "cpu"))
    qc = {}
    rows = evaluate_batch(model, b, groups, 47, 0, qc)
    enrich(rows)
    assert qc == {"self_patch": 0., "residual_patch": 0.}
    assert len(rows) == 2 and all(np.isfinite(list(row.values())).all() for row in rows)
    for row in rows:
        assert row["ablate_pair_minus_same"] == pytest.approx(
            row["ablate_candidate_pair"] - row["ablate_same_pair"])
