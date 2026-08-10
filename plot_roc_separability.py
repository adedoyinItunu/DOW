"""
plot_roc_separability.py  --  ROC curves for the research report
==========================================================
test_auc_separability.py reports the AUC as a scalar per intensity. Table 4.8
gives those numbers but the thesis has no figure for them, and the shape of the
curve is what carries the argument: at scale 0.30 the curve still hugs the
top-left corner while the trained detector is assigning 55% of those samples to
the legitimate class.

Panel (a) plots the ROC at each leech intensity, with the fixed operating point
(Youden's J, chosen once at scale 1.00 and held fixed) marked on every curve so
the reader can see it sliding down as the leech quietens. Panel (b) plots AUC
and fixed-threshold detection against intensity, alongside the trained
detectors' binary detection rates from Table 4.8.

    python plot_roc_separability.py --out roc_separability.png

Numbers are regenerated from the same GenParams path test_auc_separability.py
uses, so they reproduce Table 4.8 rather than restating it.
"""
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

from dow_data import GenParams, generate_dataset, normalize

SCALES = [1.0, 0.7, 0.5, 0.3, 0.2, 0.1]

# binary attack-detection rates for the fitted detectors, the research report.
# only the two intensities quoted in the thesis are available.
CNN_BINARY = {0.30: 0.451, 0.10: 0.188}
LR_BINARY = {0.30: 0.367, 0.10: 0.292}


def scores_at_scale(scale, per_class, seed):
    """Mean normalised pixel value per image, with normal/leech labels."""
    params = GenParams(leech_scale=scale)
    X, y, inten = generate_dataset(per_class=per_class, seed=seed, params=params)
    score = normalize(X).mean(axis=(1, 2, 3))
    is_normal, is_leech = (y == 0), (inten == 1)
    s = np.concatenate([score[is_normal], score[is_leech]])
    lab = np.concatenate([np.zeros(is_normal.sum()), np.ones(is_leech.sum())])
    return s, lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=300)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", default="roc_separability.png")
    args = ap.parse_args()

    # ---- fixed threshold, calibrated per seed at full intensity, matching
    # ---- fixed_threshold_control.py, which produced Table 4.8
    thr_by_seed = {}
    for sd in args.seeds:
        s1, l1 = scores_at_scale(1.0, args.per_class, sd)
        fpr1, tpr1, thr1 = roc_curve(l1, s1)
        thr_by_seed[sd] = thr1[np.argmax(tpr1 - fpr1)]     # Youden's J
    print(f"per-seed Youden thresholds at scale 1.00: mean "
          f"{np.mean(list(thr_by_seed.values())):.4f}\n")

    aucs, det, fprs = {}, {}, {}
    curves = {}

    for sc in SCALES:
        a, d, f = [], [], []
        for sd in args.seeds:
            s, lab = scores_at_scale(sc, args.per_class, sd)
            a.append(roc_auc_score(lab, s))
            d.append(float((s[lab == 1] >= thr_by_seed[sd]).mean()))
            f.append(float((s[lab == 0] >= thr_by_seed[sd]).mean()))
            if sd == args.seeds[0]:
                curves[sc] = roc_curve(lab, s)[:2]
        aucs[sc], det[sc], fprs[sc] = (np.mean(a), np.std(a)), np.mean(d), np.mean(f)
        print(f"scale {sc:.2f}  AUC {aucs[sc][0]:.3f} +/- {aucs[sc][1]:.3f}"
              f"   fixed-threshold detection {det[sc]:.3f}   FPR {fprs[sc]:.4f}")

    # ---- figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    cmap = plt.cm.viridis(np.linspace(0, 0.88, len(SCALES)))

    for c, sc in zip(cmap, SCALES):
        fpr, tpr = curves[sc]
        ax1.plot(fpr, tpr, color=c, lw=1.6,
                 label=f"scale {sc:.2f}  (AUC {aucs[sc][0]:.3f})")
        # mark where the fixed threshold sits on this curve
        ax1.plot(fprs[sc], det[sc], "o", color=c, ms=5, mec="k", mew=0.5)

    ax1.plot([0, 1], [0, 1], "k--", lw=0.8, label="chance")
    ax1.set_xlabel("false-positive rate on legitimate traffic")
    ax1.set_ylabel("leech detection rate")
    ax1.set_title("(a) Separability by mean pixel value")
    ax1.set_xlim(-0.02, 1.0); ax1.set_ylim(0, 1.02)
    ax1.legend(fontsize=7.5, loc="lower right", framealpha=0.9)
    ax1.grid(alpha=0.25)

    x = SCALES
    ax2.errorbar(x, [aucs[s][0] for s in x], yerr=[aucs[s][1] for s in x],
                 marker="o", color="k", lw=1.6, capsize=3, label="AUC (threshold-free)")
    ax2.plot(x, [det[s] for s in x], marker="s", color="0.45", lw=1.6,
             ls="--", label="fixed threshold, never refitted")
    ax2.plot(sorted(CNN_BINARY), [CNN_BINARY[s] for s in sorted(CNN_BINARY)],
             marker="^", color="C3", lw=1.4, ls=":", label="CNN (binary)")
    ax2.plot(sorted(LR_BINARY), [LR_BINARY[s] for s in sorted(LR_BINARY)],
             marker="v", color="C0", lw=1.4, ls=":", label="LR (binary)")
    ax2.axhline(0.5, color="0.7", lw=0.8, ls=":")
    ax2.invert_xaxis()
    ax2.set_xlabel("leech intensity scale")
    ax2.set_ylabel("AUC / detection rate")
    ax2.set_title("(b) Separability against fitted detectors")
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=7.5, loc="lower left", framealpha=0.9)
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
