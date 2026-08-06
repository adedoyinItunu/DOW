#!/usr/bin/env python3
"""
Fixed-threshold control for the leech-intensity sweep (Section 4.5.1, Table 4.10).

The AUC column currently reported is threshold-free, so it implicitly grants an
optimal operating point at every intensity, while the CNN and LR detectors are
fitted once at full intensity and never refitted. This script removes that
asymmetry: it chooses ONE threshold on mean normalised pixel value at scale 1.00
(Youden's J on the ROC), then applies that same threshold unchanged down the
sweep, exactly as a deployed detector would.

Output is a new column for Table 4.10: "fixed-threshold detection rate".

------------------------------------------------------------------------------
ADAPTATION POINT: fill in generate() to match your dow_data.py API. Everything
else is generator-agnostic. See the two options in the function body.
------------------------------------------------------------------------------
"""

import argparse
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

SCALES = [1.00, 0.70, 0.50, 0.30, 0.20, 0.10]
SEEDS = [0, 1, 2, 3, 4]


def generate(scale, seed, per_class=300, linear_const=False):
    """Generate a dataset with the leech intensity scaled by `scale`."""
    from dow_data import GenParams, generate_dataset
    params = GenParams(leech_scale=scale, linear_const=linear_const)
    return generate_dataset(per_class=per_class, seed=seed, params=params)


def mean_pixel_scores(X):
    """Mean NORMALISED pixel value, matching test_auc_separability.py and the
    description in Section 4.5.1. Scoring the raw counts instead changes the
    ranking, because the mean of log1p is not log1p of the mean, and gives a
    fixed-threshold detection rate 0.21 lower at scale 0.30."""
    from dow_data import normalize
    X = np.asarray(normalize(X))
    if X.ndim == 4:
        X = X[:, 0]
    return X.reshape(len(X), -1).mean(axis=1)


def youden_threshold(scores, labels):
    fpr, tpr, thr = roc_curve(labels, scores)
    return float(thr[np.argmax(tpr - fpr)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=300)
    ap.add_argument("--linear-const", action="store_true")
    args = ap.parse_args()

    rows = {s: {"auc": [], "fixed": [], "oracle": []} for s in SCALES}
    fpr_at_fixed = []

    for seed in SEEDS:
        # --- calibrate once, at full intensity -------------------------------
        X, y, tag = generate(1.00, seed, args.per_class, args.linear_const)
        s = mean_pixel_scores(X)
        keep = (tag == 0) | (tag == 1)
        thr = youden_threshold(s[keep], (tag[keep] == 1).astype(int))

        normal_scores = s[tag == 0]
        fpr_at_fixed.append(float((normal_scores > thr).mean()))

        # --- apply that same threshold down the sweep ------------------------
        for scale in SCALES:
            if scale == 1.00:
                Xs, ys, tags = X, y, tag
                ss = s
            else:
                Xs, ys, tags = generate(scale, seed, args.per_class,
                                        args.linear_const)
                ss = mean_pixel_scores(Xs)

            leech = ss[tags == 1]
            norm = ss[tags == 0]

            lab = np.r_[np.zeros(len(norm), int), np.ones(len(leech), int)]
            sc = np.r_[norm, leech]

            rows[scale]["auc"].append(roc_auc_score(lab, sc))
            rows[scale]["fixed"].append(float((leech > thr).mean()))
            rows[scale]["oracle"].append(
                float((leech > youden_threshold(sc, lab)).mean())
            )

    cond = "L-const" if args.linear_const else "L-ramp"
    print(f"\nFixed-threshold control ({cond}), {len(SEEDS)} seeds, "
          f"threshold calibrated at scale 1.00")
    print(f"false-positive rate on normal at the fixed threshold: "
          f"{np.mean(fpr_at_fixed):.3f} +/- {np.std(fpr_at_fixed):.3f}\n")
    print(f"{'scale':>6}  {'AUC':>15}  {'fixed-thr detect':>17}  "
          f"{'refitted-thr':>15}")
    for scale in SCALES:
        r = rows[scale]
        def ms(k):
            a = np.asarray(r[k])
            return f"{a.mean():.3f} +/- {a.std():.3f}"
        print(f"{scale:>6.2f}  {ms('auc'):>15}  {ms('fixed'):>17}  "
              f"{ms('oracle'):>15}")

    print("\nLaTeX column for Table 4.10 (fixed-threshold detection rate):")
    for scale in SCALES:
        a = np.asarray(rows[scale]["fixed"])
        print(f"  {scale:.2f} & ${a.mean():.3f} \\pm {a.std():.3f}$ \\\\")


if __name__ == "__main__":
    main()
