"""Held-out comparisons and paired, seed-stratified bootstrap."""
import numpy as np


def select_heads(scores, seed):
    rng = np.random.default_rng(seed)
    population = list(np.ndindex(scores.shape))
    candidates = sorted(population, key=lambda h: (-scores[h], h))[:2]
    used = list(candidates)

    def pick(pool):
        head = pool[rng.integers(len(pool))]
        used.append(head)
        return head

    same = [pick([v for v in population if v[0] == h[0] and v not in used]) for h in candidates]
    random = [pick([v for v in population if v not in used]) for _ in candidates]
    groups = {}
    for label, pair in (("candidate", candidates), ("same", same), ("random", random)):
        groups[label + "_1"] = [pair[0]]
        groups[label + "_2"] = [pair[1]]
        groups[label + "_pair"] = pair
    for layer in sorted({h[0] for h in candidates}):
        groups[f"layer_{layer}"] = [v for v in population if v[0] == layer]
    return groups


def enrich(rows):
    for row in rows:
        for op in ("ablate", "patch"):
            for part in ("1", "2", "pair"):
                candidate = row[f"{op}_candidate_{part}"]
                for ref in ("same", "random"):
                    row[f"{op}_{part}_minus_{ref}"] = candidate - row[f"{op}_{ref}_{part}"]
            row[f"{op}_interaction"] = (row[f"{op}_candidate_pair"]
                                        - row[f"{op}_candidate_1"] - row[f"{op}_candidate_2"])


def summarize(rows, draws, bootstrap_seed):
    keys = [key for key in rows[0] if key not in ("seed", "trial", "target", "foil",
                                                   "target_pos", "foil_pos")]
    rng = np.random.default_rng(bootstrap_seed)
    seeds = sorted({r["seed"] for r in rows})
    means, samples = [], []
    for seed in seeds:
        x = np.array([[r[k] for k in keys] for r in rows if r["seed"] == seed])
        means.append(x.mean(0))
        idx = rng.integers(len(x), size=(draws, len(x)))
        # Reuse the same paired resamples across metrics without a large 3D allocation.
        samples.append(np.stack([x[:, j][idx].mean(1) for j in range(len(keys))], axis=1))
    ci = np.quantile(np.mean(samples, axis=0), [0.025, 0.975], axis=0)
    return {k: {"mean": float(np.mean(means, axis=0)[j]), "lo": float(ci[0, j]),
                "hi": float(ci[1, j]), "by_seed": {str(s): means[i][j] for i, s in enumerate(seeds)}}
            for j, k in enumerate(keys)}
