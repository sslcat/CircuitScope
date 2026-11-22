"""Generate paired prompts directly in GPT-2 vocabulary space."""
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Batch:
    clean: torch.Tensor
    swap: torch.Tensor
    neutral: torch.Tensor
    target_pos: torch.Tensor
    foil_pos: torch.Tensor

    @property
    def target(self):
        return self.clean.gather(1, self.target_pos[:, None]).flatten()

    @property
    def foil(self):
        return self.clean.gather(1, self.foil_pos[:, None]).flatten()


def batches(cfg, seed, count, device):
    rng = np.random.default_rng(seed)
    length = cfg["prefix_length"]
    low, high = cfg["token_min"], cfg["token_max"]
    if length < 4 or not (0 <= low < high <= 50256) or high - low < length + 1:
        raise ValueError("Need two disjoint pairs and a spare distinct non-BOS token")
    slots = [(i, j) for i in range(1, length) for j in range(1, length) if abs(i - j) >= 2]
    for offset in range(0, count, cfg["batch_size"]):
        rows = []
        for _ in range(min(cfg["batch_size"], count - offset)):
            ids = rng.choice(high - low, size=length + 1, replace=False) + low
            i, j = slots[rng.integers(len(slots))]
            clean = np.r_[50256, ids[:-1], ids[i - 1]]
            swap = clean.copy()
            swap[i], swap[j] = clean[j], clean[i]
            neutral = clean.copy()
            neutral[-1] = ids[-1]
            rows.append((clean, swap, neutral, i + 1, j + 1))
        yield Batch(*(torch.as_tensor(np.stack(items), device=device, dtype=torch.long)
                      for items in zip(*rows)))


def metrics(logits, batch):
    lp = logits.float().log_softmax(-1)
    row = torch.arange(len(lp), device=lp.device)
    target = lp[row, batch.target]
    foil = lp[row, batch.foil]
    return {"margin": target - foil, "logp": target, "foil_logp": foil,
            "top1": (lp.argmax(-1) == batch.target).float(),
            "wins": (target > foil).float() + 0.5 * (target == foil).float()}
