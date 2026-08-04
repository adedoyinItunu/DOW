#!/usr/bin/env python3
"""
Produce the Chapter 3 exemplar heat-map figure from an existing dataset .npz.

Renders a 2x4 grid: columns are the four traffic classes, rows are the two
attack intensities (leech / flood). The `normal` class has no intensity
variants, so two independent normal samples are shown instead.

All panels share a single colour scale. This is deliberate: the fixed
log1p(400) divisor means absolute brightness is comparable across images, and
the figure should show that, because Sections 4.7 and 4.8 turn on it.

Usage:
    python make_exemplar_figure.py --data data_lramp.npz --out heatmap_exemplars.png
    python make_exemplar_figure.py --data data_lconst.npz --out heatmap_exemplars_lconst.png
"""

import argparse
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLASS_NAMES = ["normal", "linear", "geometric", "random"]

# Candidate key names, in order of preference. The script prints whatever it
# finds if none of these match, so you can pass the right ones explicitly.
X_KEYS = ["X", "images", "x", "data", "heatmaps"]
Y_KEYS = ["y", "labels", "label", "classes"]
I_KEYS = ["intensity", "intensities", "intensity_tag", "tag", "z"]


def pick_key(npz, candidates, what, override=None):
    if override is not None:
        if override not in npz:
            sys.exit(f"Key '{override}' not in file. Available: {list(npz.keys())}")
        return override
    for k in candidates:
        if k in npz:
            return k
    sys.exit(
        f"Could not find the {what} array. Available keys: {list(npz.keys())}\n"
        f"Re-run with --{what}-key <name>."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_lramp.npz")
    ap.add_argument("--out", default="heatmap_exemplars.png")
    ap.add_argument("--seed", type=int, default=42,
                    help="Which exemplar to pick within each group (0 = first).")
    ap.add_argument("--x-key", default=None)
    ap.add_argument("--y-key", default=None)
    ap.add_argument("--intensity-key", default=None)
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    npz = np.load(args.data)
    xk = pick_key(npz, X_KEYS, "x", args.x_key)
    yk = pick_key(npz, Y_KEYS, "y", args.y_key)
    ik = pick_key(npz, I_KEYS, "intensity", args.intensity_key)

    X = np.asarray(npz[xk])
    y = np.asarray(npz[yk]).ravel()
    tag = np.asarray(npz[ik]).ravel()

    # Accept (N,24,30) or (N,1,24,30)
    if X.ndim == 4:
        X = X[:, 0]
    if X.shape[1:] != (24, 30):
        sys.exit(f"Expected images of shape (24,30), got {X.shape[1:]}.")

    rng = np.random.default_rng(args.seed)

    def pick(cls, intensity=None, exclude=()):
        mask = (y == cls)
        if intensity is not None:
            mask &= (tag == intensity)
        idx = np.flatnonzero(mask)
        idx = np.array([i for i in idx if i not in exclude])
        if idx.size == 0:
            sys.exit(f"No samples for class {cls}, intensity {intensity}.")
        return int(rng.choice(idx))

    # Two independent normal samples for the first column.
    n1 = pick(0)
    n2 = pick(0, exclude=(n1,))

    grid = [
        [(n1, "normal"),
         (pick(1, 1), "linear (leech)"),
         (pick(2, 1), "geometric (leech)"),
         (pick(3, 1), "random (leech)")],
        [(n2, "normal"),
         (pick(1, 2), "linear (flood)"),
         (pick(2, 2), "geometric (flood)"),
         (pick(3, 2), "random (flood)")],
    ]

    vmin = 0.0
    vmax = max(X[i].max() for row in grid for i, _ in row)

    fig, axes = plt.subplots(2, 4, figsize=(11.0, 5.2), constrained_layout=True)

    im = None
    for r, row in enumerate(grid):
        for c, (idx, title) in enumerate(row):
            ax = axes[r, c]
            im = ax.imshow(
                X[idx], cmap="viridis", vmin=vmin, vmax=vmax,
                aspect="auto", origin="upper",
                extent=[0.5, 30.5, 23.5, -0.5],
                interpolation="nearest",
            )
            ax.set_title(title, fontsize=9)
            ax.set_xticks([1, 10, 20, 30])
            ax.set_yticks([0, 6, 12, 18, 23])
            ax.tick_params(labelsize=7)
            if c == 0:
                ax.set_ylabel("hour of day", fontsize=8)
            else:
                ax.set_yticklabels([])
            if r == 1:
                ax.set_xlabel("day of month", fontsize=8)
            else:
                ax.set_xticklabels([])

    cbar = fig.colorbar(im, ax=axes, location="right", shrink=0.85, pad=0.015)
    cbar.set_label("normalised invocations", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"wrote {args.out}")
    print(f"  shared colour scale: [{vmin:.3f}, {vmax:.3f}]")
    print(f"  sample indices: {[i for row in grid for i, _ in row]}")


if __name__ == "__main__":
    main()
