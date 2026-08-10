"""
plot_multiseed_sweep.py

Regenerates e5_leech_sweep.png from the five-seed sweep results,
with error bars, for both the CNN and the logistic-regression baseline.

No re-running of experiments needed - the numbers below are the ones
the sweeps already printed. If re-run with different seeds,
just update the six arrays.

Usage:
    python plot_multiseed_sweep.py
    python plot_multiseed_sweep.py --out e5_leech_sweep.png --dpi 300
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")          # no display needed
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# Results from the five-seed runs. Update if re-run.
# ---------------------------------------------------------------
SCALES = np.array([1.00, 0.70, 0.50, 0.30, 0.20, 0.10])

CNN_ACC       = np.array([0.635, 0.484, 0.364, 0.185, 0.098, 0.056])
CNN_ACC_STD   = np.array([0.215, 0.255, 0.256, 0.177, 0.127, 0.108])
CNN_NORM      = np.array([0.186, 0.311, 0.414, 0.549, 0.672, 0.812])
CNN_NORM_STD  = np.array([0.157, 0.182, 0.199, 0.225, 0.264, 0.289])

LR_ACC        = np.array([0.922, 0.781, 0.552, 0.330, 0.216, 0.000])
LR_ACC_STD    = np.array([0.010, 0.007, 0.019, 0.006, 0.011, 0.000])
LR_NORM       = np.array([0.077, 0.200, 0.389, 0.633, 0.657, 0.708])
LR_NORM_STD   = np.array([0.009, 0.007, 0.017, 0.006, 0.005, 0.011])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="e5_leech_sweep.png")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--width", type=float, default=10.0)
    p.add_argument("--height", type=float, default=4.2)
    args = p.parse_args()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(args.width, args.height))

    # ---- greyscale-safe styling (theses get printed in black and white) ----
    cnn_style = dict(color="black",  marker="o", linestyle="-",
                     linewidth=1.6, markersize=5, capsize=3)
    lr_style  = dict(color="dimgray", marker="s", linestyle="--",
                     linewidth=1.6, markersize=5, capsize=3)

    # ---------------- Panel 1: accuracy ----------------
    ax1.errorbar(SCALES, CNN_ACC, yerr=CNN_ACC_STD,
                 label="DoWNetCNN", **cnn_style)
    ax1.errorbar(SCALES, LR_ACC, yerr=LR_ACC_STD,
                 label="Logistic regression", **lr_style)

    ax1.axhline(0.5, color="gray", linewidth=0.8, linestyle=":")
    ax1.text(0.98, 0.515, "half detection", fontsize=7.5,
             color="gray", va="bottom", ha="left")

    ax1.set_xlabel("Leech intensity scale")
    ax1.set_ylabel("Detection accuracy")
    ax1.set_title("(a) Detection accuracy", fontsize=10)
    ax1.set_ylim(-0.05, 1.05)
    ax1.invert_xaxis()          # quieter leech to the right
    ax1.grid(alpha=0.25, linewidth=0.5)
    ax1.legend(fontsize=8, loc="upper right")

    # ---------------- Panel 2: misread as normal ----------------
    ax2.errorbar(SCALES, CNN_NORM, yerr=CNN_NORM_STD,
                 label="DoWNetCNN", **cnn_style)
    ax2.errorbar(SCALES, LR_NORM, yerr=LR_NORM_STD,
                 label="Logistic regression", **lr_style)

    ax2.axhline(0.5, color="gray", linewidth=0.8, linestyle=":")
    ax2.text(0.98, 0.515, "majority misread", fontsize=7.5,
             color="gray", va="bottom", ha="left")

    ax2.set_xlabel("Leech intensity scale")
    ax2.set_ylabel("Proportion misread as normal")
    ax2.set_title("(b) Misclassified as legitimate traffic", fontsize=10)
    ax2.set_ylim(-0.05, 1.05)
    ax2.invert_xaxis()
    ax2.grid(alpha=0.25, linewidth=0.5)
    ax2.legend(fontsize=8, loc="upper left")

    fig.tight_layout()
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"wrote {args.out} at {args.dpi} dpi")
    print("\nSuggested caption:\n")
    print(r"""\caption{Leech detection versus intensity across five seeds,
for the CNN and the logistic-regression baseline. Error bars show one
standard deviation. Panel (a) gives detection accuracy; panel (b) gives
the proportion of leech samples misclassified as legitimate traffic. The
linear model traces the same collapse as the CNN with roughly twenty times
smaller variance. The leech grows quieter from left to right.}""")


if __name__ == "__main__":
    main()
