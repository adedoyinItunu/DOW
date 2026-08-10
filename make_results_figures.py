"""
make_results_figures.py
Generates three results figures from the values reported in the paper.
No data files needed: the numbers are the ones already in Tables 4, 6 and 8.

    python make_results_figures.py

Writes fig_sweep.png, fig_stability.png, fig_attention.png at 200 dpi.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

NAVY, AMBER, GREY, TEAL = "#1E2761", "#D99A00", "#8A93A8", "#2E9E8F"
plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#555555", "axes.labelcolor": "#222222",
    "xtick.color": "#555555", "ytick.color": "#555555",
})

# ----------------------------------------------------------------- Table 6
scale = [1.00, 0.70, 0.50, 0.30, 0.20, 0.10]
cnn25 = [0.635, 0.484, 0.364, 0.185, 0.098, 0.056]
cnn25_sd = [0.215, 0.255, 0.256, 0.177, 0.127, 0.108]
conv = [0.986, 0.900, 0.637, 0.164, 0.017, 0.000]
conv_sd = [0.005, 0.038, 0.055, 0.042, 0.012, 0.000]
lr = [0.922, 0.781, 0.552, 0.330, 0.216, 0.000]
lr_sd = [0.010, 0.007, 0.019, 0.006, 0.011, 0.000]

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.9),
                              gridspec_kw={"width_ratios": [2.1, 1]})

ax.axvspan(0.10, 0.50, color=AMBER, alpha=0.10, zorder=0)
ax.text(0.30, 1.02, "transition region", ha="center", va="bottom",
        fontsize=8, color="#8A6A00")

for y, sd, lab, col, ls in [
    (conv,  conv_sd,  "CNN, converged",        NAVY,  "-"),
    (lr,    lr_sd,    "Logistic regression",   AMBER, "-"),
    (cnn25, cnn25_sd, "CNN, 25 epochs",        GREY,  "--"),
]:
    ax.errorbar(scale, y, yerr=sd, label=lab, color=col, ls=ls,
                marker="o", ms=3.5, lw=1.6, capsize=2, elinewidth=0.8)

ax.set_xlim(1.05, 0.05)
ax.set_ylim(-0.03, 1.05)
ax.set_xticks(scale)
ax.set_xlabel("leech intensity scale")
ax.set_ylabel("leech detection accuracy")
ax.grid(axis="y", color="#E6E9F0", lw=0.7)
ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=8, loc="lower left")

# binary-task panel
x = np.arange(2)
w = 0.36
ax2.bar(x - w/2, [0.689, 0.366], w, label="volume threshold", color=TEAL)
ax2.bar(x + w/2, [0.395, 0.185], w, label="CNN, converged", color=NAVY)
for xi, v in zip(x - w/2, [0.689, 0.366]):
    ax2.text(xi, v + 0.02, f"{v:.3f}", ha="center", fontsize=7.5)
for xi, v in zip(x + w/2, [0.395, 0.185]):
    ax2.text(xi, v + 0.02, f"{v:.3f}", ha="center", fontsize=7.5)
ax2.set_xticks(x)
ax2.set_xticklabels(["scale 0.30", "scale 0.10"])
ax2.set_ylim(0, 0.85)
ax2.set_ylabel("attack flagged (binary task)")
ax2.grid(axis="y", color="#E6E9F0", lw=0.7)
ax2.set_axisbelow(True)
ax2.legend(frameon=False, fontsize=7.5, loc="upper right")

fig.tight_layout()
fig.savefig("fig_sweep.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# ----------------------------------------------------------------- Table 4
conds = ["Logistic regression\non raw pixels",
         "DoWNet-style head\n25 ep",
         "Positional head\n25 ep",
         "Original head\n200 ep, converged",
         "Original head\n25 ep, selected",
         "Original head\n25 ep, final"]
mean = [0.972, 0.996, 0.996, 0.996, 0.968, 0.759]
sd   = [0.009, 0.012, 0.009, 0.004, 0.038, 0.169]
cols = [AMBER, NAVY, NAVY, NAVY, GREY, GREY]

fig, ax = plt.subplots(figsize=(5.4, 2.5))
y = np.arange(len(conds))
ax.errorbar(mean, y, xerr=sd, fmt="o", ms=5, color="none",
            ecolor="#9AA5C4", elinewidth=1.6, capsize=3)
for yi, m, c in zip(y, mean, cols):
    ax.plot(m, yi, "o", ms=5.5, color=c, zorder=3)
ax.set_yticks(y)
ax.set_yticklabels(conds, fontsize=7.5)
ax.set_xlim(0.45, 1.03)
ax.set_xlabel("macro-F1 (mean $\\pm$ 1 SD over ten seeds)")
ax.grid(axis="x", color="#E6E9F0", lw=0.7)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig("fig_stability.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# ----------------------------------------------------------------- Table 8
groups = ["legitimate,\nmisclassified", "legitimate,\ncorrect",
          "genuine\nlinear", "leech", "flood"]
gc   = [0.268, 0.294, 0.363, 0.314, 0.322]
gc_e = [0.017, 0.016, 0.019, 0.011, 0.040]
sal  = [0.243, 0.266, 0.267, 0.285, 0.263]
sal_e= [0.012, 0.012, 0.014, 0.015, 0.013]
shap = [0.267, 0.391, 0.351, 0.432, 0.398]
shap_e=[0.020, 0.033, 0.017, 0.024, 0.012]

fig, ax = plt.subplots(figsize=(6.4, 2.6))
x = np.arange(len(groups)); w = 0.26
ax.bar(x - w, gc,   w, yerr=gc_e,   label="Grad-CAM", color=GREY,
       capsize=2, error_kw={"elinewidth": 0.8})
ax.bar(x,     sal,  w, yerr=sal_e,  label="Saliency", color=TEAL,
       capsize=2, error_kw={"elinewidth": 0.8})
ax.bar(x + w, shap, w, yerr=shap_e, label="SHAP", color=NAVY,
       capsize=2, error_kw={"elinewidth": 0.8})
ax.axhline(0.233, color=AMBER, lw=1.4, ls="--", zorder=4)
ax.text(len(groups) - 0.35, 0.243, "uniform attention, 7/30",
        fontsize=7.5, color="#8A6A00", ha="right")
ax.set_xticks(x); ax.set_xticklabels(groups, fontsize=7.5)
ax.set_ylim(0, 0.50)
ax.set_ylabel("late-week attention share")
ax.grid(axis="y", color="#E6E9F0", lw=0.7)
ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper left")
fig.tight_layout()
fig.savefig("fig_attention.png", dpi=200, bbox_inches="tight")
plt.close(fig)

print("wrote fig_sweep.png, fig_stability.png, fig_attention.png")
