"""Instrument the native GPT-2 attention output before head mixing."""
from collections import defaultdict
from contextlib import ExitStack
from dataclasses import dataclass, field

import torch


@dataclass
class Trace:
    heads: dict = field(default_factory=dict)
    patterns: dict = field(default_factory=dict)
    values: dict = field(default_factory=dict)
    residual: torch.Tensor | None = None


@torch.inference_mode()
def forward(model, tokens, heads=(), donor=None, capture=False, patterns=False,
            patch_residual=False):
    """Return final-position logits and requested query activations.

    A selected head is zeroed unless a donor trace supplies its replacement.
    Only the final position is edited. Context managers remove hooks on failure.
    """
    trace = Trace()
    by_layer = defaultdict(list)
    for layer, head in heads:
        if not 0 <= layer < model.config.n_layer or not 0 <= head < model.config.n_head:
            raise ValueError("Head index outside GPT-2 configuration")
        by_layer[layer].append(head)
    n_heads = model.config.n_head

    def pre_projection(layer):
        def hook(module, args):
            x = args[0]
            split = x.reshape(*x.shape[:2], n_heads, -1)
            if capture:
                trace.heads[layer] = split[:, -1].detach().clone()
            if layer in by_layer:
                split = split.clone()
                ids = by_layer[layer]
                split[:, -1, ids] = 0 if donor is None else donor.heads[layer][:, ids]
                return (split.reshape_as(x),)
        return hook

    def record_attention(layer):
        def hook(module, args, output):
            trace.patterns[layer] = output[1].detach()
        return hook

    def record_values(layer):
        def hook(module, args, output):
            v = output.chunk(3, dim=-1)[2]
            trace.values[layer] = v.reshape(*v.shape[:2], n_heads, -1).detach()
        return hook

    def final_residual(module, args):
        residual = args[0]
        if capture:
            trace.residual = residual[:, -1].detach().clone()
        if patch_residual:
            new = residual.clone()
            new[:, -1] = donor.residual
            return (new,)

    with ExitStack() as stack:
        for layer, block in enumerate(model.transformer.h):
            if capture or layer in by_layer:
                stack.enter_context(block.attn.c_proj.register_forward_pre_hook(pre_projection(layer)))
            if capture and patterns:
                stack.enter_context(block.attn.c_attn.register_forward_hook(record_values(layer)))
            if patterns:
                stack.enter_context(block.attn.register_forward_hook(record_attention(layer)))
        if capture or patch_residual:
            stack.enter_context(model.transformer.ln_f.register_forward_pre_hook(final_residual))
        state = model.transformer(tokens, use_cache=False, output_attentions=patterns).last_hidden_state
        logits = model.lm_head(state[:, -1]).float()
    return logits, trace
