#!/usr/bin/env python3
"""
plot_lr_coefficients.py  --  produce Figure 4.2 and verify the claims about it.

Fits the multinomial logistic-regression baseline on the same split as the CNN,
reshapes its 720 per-class coefficients to the 24x30 grid, and saves them as one
diverging-scale panel per class.

It also prints the three quantities Section 4.2.3 asserts about this figure, so
they can be checked rather than taken on trust:

  * share of each class's absolute coefficient mass in the final week (days 24-30)
    -- thesis says 47.4% for `geometric` against 21.6-29.7% for the others
  * the three highest-weighted hours per class
    -- thesis says nocturnal (0-3, 22-23) for normal/geometric/random
  * the same for `linear`
    -- thesis says hours 11, 12 and 14, at the diurnal peak

Usage:
    python plot_lr_coefficients.py --data data_lramp.npz --seed 0
"""

import argparse
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

from dow_data import load_and_split

warnings.filterwarnings("ignore")

HOURS, DAYS = 24, 30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_lramp.npz")
    ap.add_argument("--seed", type=int, default=0,
                    help="split seed; use the one quoted for the baseline in 4.2.3")
    ap.add_argument("--out", default="lr_coefficients.png")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    (Xtr, ytr, _), _, _, names = load_and_split(args.data, args.seed)
    lr = LogisticRegression(max_iter=2000).fit(Xtr.reshape(len(Xtr), -1), ytr)

    coef = lr.coef_.reshape(len(names), HOURS, DAYS)
    vmax = np.abs(coef).max()

    fig, axes = plt.subplots(1, len(names), figsize=(4.0 * len(names), 3.6),
                             constrained_layout=True)
    im = None
    for k, (ax, name) in enumerate(zip(axes, names)):
        im = ax.imshow(coef[k], cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                       aspect="auto", origin="upper",
                       extent=[0.5, 30.5, 23.5, -0.5], interpolation="nearest")
        ax.set_title(name, fontsize=11)
        ax.set_xticks([1, 10, 20, 30])
        ax.set_yticks([0, 6, 12, 18, 23])
        ax.tick_params(labelsize=8)
        ax.set_xlabel("day of month", fontsize=9)
        if k == 0:
            ax.set_ylabel("hour of day", fontsize=9)
        else:
            ax.set_yticklabels([])

    cbar = fig.colorbar(im, ax=axes, location="right", shrink=0.85, pad=0.015)
    cbar.set_label("fitted coefficient", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"wrote {args.out}  (data={args.data}, split seed={args.seed})\n")

    # ---- claims asserted in Section 4.2.3 -------------------------------
    a = np.abs(coef)

    print("Final-week share of absolute coefficient mass (days 24-30)")
    print("  thesis: geometric 47.4%, others 21.6-29.7%")
    for k, name in enumerate(names):
        share = a[k, :, 23:].sum() / a[k].sum() * 100
        print(f"    {name:10s} {share:5.1f}%")

    print("\nThree highest-weighted hours per class")
    print("  thesis: nocturnal (0-3, 22-23) for normal/geometric/random;")
    print("          11, 12, 14 for linear")
    for k, name in enumerate(names):
        top = np.argsort(a[k].sum(axis=1))[::-1][:3]
        print(f"    {name:10s} {sorted(top.tolist())}")

    print("\nIf any figure above differs from Section 4.2.3, update the thesis "
          "to match this output.")


if __name__ == "__main__":
    main()
