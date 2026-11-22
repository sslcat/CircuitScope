"""Render figures from saved measurements without rerunning GPT-2."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


INK = "#203346"
BLUE = "#17647b"
ORANGE = "#ba5830"
GRAY = "#6d7780"


def name(head):
    return f"L{head[0]}H{head[1]}"


def export(fig, out, filename):
    fig.savefig(out / filename, dpi=155, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_attention(data, groups, out):
    candidates = groups["candidate_pair"]
    score = data["scores"]
    fig, ax = plt.subplots(figsize=(7.7, 5.3), layout="constrained")
    lim = max(abs(score.min()), abs(score.max()), 1e-8)
    im = ax.imshow(score, cmap="RdBu_r", vmin=-lim, vmax=lim)
    for layer, head in candidates:
        ax.add_patch(plt.Rectangle((head-.48, layer-.48), .96, .96, fill=False,
                                   ec="black", lw=2))
    ax.set(xticks=range(score.shape[1]), yticks=range(score.shape[0]),
           xlabel="Head", ylabel="Layer", title="Which heads follow the cue?")
    fig.colorbar(im, ax=ax, label="Clean − swap attention to the target position")
    export(fig, out, "head-screen.png")

    fig, axs = plt.subplots(2, 2, figsize=(8.3, 7.1), layout="constrained", sharex=True, sharey=True)
    t, f = data["positions"]
    for row, (layer, head) in enumerate(candidates):
        for col, condition in enumerate(("clean", "swap")):
            ax = axs[row, col]
            weights = data[f"{condition}_{layer}"][head]
            im = ax.imshow(weights, cmap="Blues", vmin=0, vmax=1)
            for pos, color in ((t, ORANGE), (f, "#19805c")):
                ax.scatter(pos, len(weights)-1, s=75, marker="s", facecolors="none",
                           edgecolors=color, lw=1.6)
            ax.set_title(f"{name((layer, head))}  ·  {condition}")
            if col == 0:
                ax.set_ylabel("Query position")
            if row == 1:
                ax.set_xlabel("Source position")
    handles = [Line2D([], [], marker="s", linestyle="none", mfc="none", mec=c,
                      label=txt) for c, txt in ((ORANGE, "Clean target"), ("#19805c", "Foil"))]
    fig.legend(handles=handles, loc="outside lower center", ncols=2)
    fig.colorbar(im, ax=axs, shrink=.6, label="Attention weight")
    export(fig, out, "attention-example.png")


def plot_effects(summary, groups, out):
    labels = ["candidate_1", "candidate_2", "candidate_pair", "same_pair", "random_pair"]
    names = [name(groups[k][0]) for k in labels[:2]] + ["Candidate pair", "Same-layer pair", "Random pair"]
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True, layout="constrained")
    for ax, op, title in zip(axs, ("ablate", "patch"), ("Query ablation", "Clean → swap patch")):
        ax.axvline(0, color="#adb9bf", lw=1)
        for j, label in enumerate(labels):
            m = summary[f"{op}_{label}"]
            ax.plot([m["lo"], m["hi"]], [j, j], lw=2.0, color=BLUE if j < 3 else GRAY)
            ax.scatter(m["mean"], j, s=42, color=BLUE if j < 3 else GRAY, zorder=3)
        ax.set(yticks=range(len(labels)), yticklabels=names, title=title,
               xlabel=("Reduction in paired gap (nats)" if op == "ablate" else "Gain in swap margin (nats)"))
        ax.grid(axis="x", alpha=.15)
    axs[0].invert_yaxis()
    export(fig, out, "causal-effects.png")


def plot_baseline(rows, out):
    x = np.array([float(r["swap_margin"]) for r in rows])
    y = np.array([float(r["clean_margin"]) for r in rows])
    fig, ax = plt.subplots(figsize=(5.9, 4.9), layout="constrained")
    low, high = min(x.min(), y.min())-.3, max(x.max(), y.max())+.3
    ax.plot([low, high], [low, high], "--", color=GRAY, lw=1)
    ax.axhline(0, color="#b5bfc5", lw=.7)
    ax.axvline(0, color="#b5bfc5", lw=.7)
    ax.scatter(x, y, s=20, alpha=.67, color=BLUE, edgecolors="none")
    ax.set(xlabel="Swap margin (nats)", ylabel="Clean margin (nats)", aspect="equal",
           xlim=(low, high), ylim=(low, high), title="One point per held-out prompt pair")
    export(fig, out, "paired-baseline.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    args = p.parse_args()
    result = args.run
    out = result / "figures"
    out.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.labelcolor": INK,
                         "axes.spines.top": False, "axes.spines.right": False})
    groups = json.loads((result / "groups.json").read_text())
    with np.load(result / "discovery.npz") as data:
        plot_attention(data, groups, out)
    plot_effects(json.loads((result / "summary.json").read_text()), groups, out)
    with (result / "trials.csv").open() as f:
        plot_baseline(list(csv.DictReader(f)), out)
