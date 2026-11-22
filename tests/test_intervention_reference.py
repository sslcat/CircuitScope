"""Compare the production pre-hook against a separate post-projection calculation."""
from contextlib import ExitStack

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from circuitscope.model import Trace, forward


@torch.inference_mode()
def reference_logits(model, tokens, sites, donors=None, intervene=False):
    """Read native QKV, then add a projected head delta to the attention write."""
    vectors = {}
    width = model.config.n_embd // model.config.n_head

    def read_qkv(site):
        def hook(module, args, output):
            _, head = site
            # Slice the flattened projection coordinates directly.
            start = head * width
            q, k, v = (part[..., start:start + width] for part in output.chunk(3, -1))
            scores = torch.einsum("bd,btd->bt", q[:, -1], k) / width**0.5
            # The final query can see every token in these unpadded prompts.
            vectors[site] = torch.einsum("bt,btd->bd", scores.softmax(-1), v)
        return hook

    def replace_write(site):
        def hook(proj, args, output):
            current = vectors[site]
            replacement = torch.zeros_like(current) if donors is None else donors[site]
            start = site[1] * width
            delta = (replacement - current) @ proj.weight[start:start + width]
            edited = output.clone()
            edited[:, -1] += delta
            return edited
        return hook

    with ExitStack() as stack:
        for site in sites:
            attn = model.transformer.h[site[0]].attn
            stack.enter_context(attn.c_attn.register_forward_hook(read_qkv(site)))
            if intervene:
                stack.enter_context(attn.c_proj.register_forward_hook(replace_write(site)))
        logits = model(tokens, use_cache=False).logits
    return logits, vectors


def check_reference(model, clean, swap, sites):
    base, _ = reference_logits(model, swap, sites)
    _, donors = reference_logits(model, clean, sites)
    _, trace = forward(model, clean, capture=True)
    errors = {}
    for label, use_donor in (("patch", True), ("zero", False)):
        expected, _ = reference_logits(model, swap, sites, donors if use_donor else None, True)
        actual, _ = forward(model, swap, sites, donor=trace if use_donor else None)
        torch.testing.assert_close(expected[:, :-1], base[:, :-1], atol=0, rtol=0)
        torch.testing.assert_close(actual, expected[:, -1], atol=1.5e-4, rtol=1e-5)
        errors[label] = (actual - expected[:, -1]).abs().max().item()
    return errors


def test_multilayer_patch_against_native_projected_delta():
    torch.manual_seed(11)
    cfg = GPT2Config(n_layer=3, n_head=4, n_embd=32, n_positions=16, vocab_size=101,
                     attn_implementation="eager")
    model = GPT2LMHeadModel(cfg).eval()
    clean = torch.tensor([[0, 8, 6, 4, 2, 8], [0, 2, 5, 3, 7, 2]])
    swap = clean[:, [0, 3, 2, 1, 4, 5]]
    check_reference(model, clean, swap, [(0, 3), (2, 1)])


def test_raising_hook_leaves_native_model_intact():
    cfg = GPT2Config(n_layer=1, n_head=2, n_embd=8, vocab_size=16, attn_implementation="eager")
    model = GPT2LMHeadModel(cfg).eval()
    x = torch.tensor([[1, 2, 1]])
    expected, _ = forward(model, x)
    with pytest.raises(KeyError):
        forward(model, x, [(0, 0)], donor=Trace())
    proj = model.transformer.h[0].attn.c_proj
    assert not proj._forward_pre_hooks
    actual, _ = forward(model, x)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
