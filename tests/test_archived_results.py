"""Independent checks of the committed study, using its saved rows and discovery matrix."""
import csv
import json
from pathlib import Path

import numpy as np

from circuitscope.task import batches


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def test_archived_split_and_choices():
    cfg = json.loads((ROOT / "config.json").read_text())
    groups = json.loads((RESULTS / "groups.json").read_text())
    with (RESULTS / "trials.csv").open() as f:
        rows = {(int(r["seed"]), int(r["trial"])): r for r in csv.DictReader(f)}
    unique_prefix_sets = set()
    with np.load(RESULTS / "discovery.npz") as archive:
        scores = archive["scores"]
        best = np.argsort(-scores, axis=None, kind="stable")[:2]
        ranked = np.column_stack(np.unravel_index(best, scores.shape)).tolist()
        assert ranked == groups["candidate_pair"]
        first = next(batches(cfg, cfg["discovery_seed"], 1, "cpu"))
        np.testing.assert_array_equal(first.clean[0], archive["clean_ids"])
        np.testing.assert_array_equal(first.swap[0], archive["swap_ids"])
    for seed in [cfg["discovery_seed"], *cfg["test_seeds"]]:
        discovery = seed == cfg["discovery_seed"]
        count = cfg["discovery_n"] if discovery else cfg["test_n"]
        trial = 0
        for b in batches(cfg, seed, count, "cpu"):
            for i, clean in enumerate(b.clean):
                prefix_set = tuple(sorted(clean[1:-1].tolist()))
                assert prefix_set not in unique_prefix_sets
                unique_prefix_sets.add(prefix_set)
                swap = b.swap[i]
                t, f = b.target_pos[i], b.foil_pos[i]
                assert clean[-1] == swap[-1] == clean[t-1] == swap[f-1]
                np.testing.assert_array_equal((clean != swap).nonzero().flatten().sort().values,
                                              sorted([int(t-1), int(f-1)]))
                if not discovery:
                    row = rows.pop((seed, trial))
                    assert [int(row[k]) for k in ("target", "foil", "target_pos", "foil_pos")] == [
                        int(clean[t]), int(clean[f]), int(t), int(f)]
                trial += 1
    assert not rows and len(unique_prefix_sets) == 240


def test_archived_paired_intervals():
    cfg = json.loads((ROOT / "config.json").read_text())
    saved = json.loads((RESULTS / "summary.json").read_text())
    with (RESULTS / "trials.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 192
    keys = list(saved)
    rng = np.random.default_rng(cfg["bootstrap_seed"])
    replicates, means = [], []
    for seed in cfg["test_seeds"]:
        subset = [r for r in rows if int(r["seed"]) == seed]
        matrix = np.asarray([[float(r[k]) for k in keys] for r in subset])
        # Each row is a complete prompt pair with every intervention and control.
        # Frequency weights give a separate implementation of paired resampling.
        n = len(matrix)
        indices = rng.integers(n, size=(cfg["bootstrap_n"], n))
        weights = np.stack([np.bincount(draw, minlength=n) for draw in indices])
        replicates.append((weights @ matrix) / n)
        means.append(matrix.mean(0))
        np.testing.assert_allclose(means[-1], [saved[k]["by_seed"][str(seed)] for k in keys])
    bounds = np.percentile(np.mean(replicates, axis=0), [2.5, 97.5], axis=0)
    for values, field in ((np.mean(means, axis=0), "mean"), (bounds[0], "lo"), (bounds[1], "hi")):
        np.testing.assert_allclose(values, [saved[k][field] for k in keys], rtol=0, atol=1e-10)
