"""
test_auc_separability.py  --  ITEM 9: class-distribution separability
=====================================================================
Answers the objection recorded in Section 5.6 of the thesis: the signal-to-noise
analysis compares injected signal against WITHIN-IMAGE noise, whereas the
classification task requires separating the leech and normal DISTRIBUTIONS.

This computes, at each leech intensity, the ROC AUC obtained by thresholding a
single scalar -- the mean pixel value of the heat-map -- to separate normal
images from leech images. No model is involved. If AUC stays high while the
detectors collapse, the information is present and the detectors fail to use it.
If AUC falls with the detectors, the classes genuinely overlap.

    python test_auc_separability.py
    python test_auc_separability.py --per-class 300 --seed 42
"""
import argparse
import numpy as np
from sklearn.metrics import roc_auc_score

from dow_data import GenParams, generate_dataset, normalize

SCALES = [1.0, 0.7, 0.5, 0.3, 0.2, 0.1]


def auc_at_scale(scale, per_class, seed):
    """Generate data at this leech scale; AUC of mean-pixel-value, normal vs leech."""
    params = GenParams(leech_scale=scale)
    X, y, inten = generate_dataset(per_class=per_class, seed=seed, params=params)
    Xn = normalize(X)

    score = Xn.mean(axis=(1, 2, 3))          # one scalar per image

    is_normal = (y == 0)
    is_leech = (inten == 1)                  # attack samples at low intensity

    s = np.concatenate([score[is_normal], score[is_leech]])
    lab = np.concatenate([np.zeros(is_normal.sum()), np.ones(is_leech.sum())])

    return roc_auc_score(lab, s), is_normal.sum(), is_leech.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=300)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = ap.parse_args()

    print("Separability of normal vs leech by mean pixel value alone")
    print("(no model; AUC 0.5 = indistinguishable, 1.0 = perfectly separable)\n")
    print(f"{'leech scale':>12} | {'AUC mean':>9} | {'std':>6} | n_normal | n_leech")
    print("-" * 60)

    for sc in SCALES:
        aucs = []
        for sd in args.seeds:
            a, nn, nl = auc_at_scale(sc, args.per_class, sd)
            aucs.append(a)
        print(f"{sc:>12.2f} | {np.mean(aucs):>9.3f} | {np.std(aucs):>6.3f} | "
              f"{nn:>8d} | {nl:>7d}")

    print("\nReading:")
    print("  AUC stays high as scale falls -> the leech/normal distributions remain")
    print("    separable on a single scalar, so the detectors' collapse is a failure")
    print("    to use available information (supports the Section 5.3 reading).")
    print("  AUC falls toward 0.5 alongside detection -> the distributions genuinely")
    print("    overlap, and the evasion threshold reflects the task, not the detectors.")
    print("\nEither outcome resolves the limitation stated in Section 5.6.")


if __name__ == "__main__":
    main()
