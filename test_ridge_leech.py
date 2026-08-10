"""
test_ridge_leech.py

Tests the claim: "logistic regression fails on the quiet leech because 720
coefficients cannot be estimated from 840 training samples."

That claim implies the failure is an ESTIMATION problem, not a
SEPARABILITY problem. If it is estimation error, penalising the weights
should help. This sweeps the L2 regularisation strength at the collapse
intensity and reports whether leech detection recovers.

  * Recovery with stronger regularisation  -> estimation error confirmed.
    The signal is linearly separable; the fit was just badly conditioned.
    The design recommendation (global evidence integration, properly
    regularised) stands and is now tested.

  * No recovery at any C                   -> the explanation is wrong.
    Soften the sentence in the research report to state only what was measured.

Also fits an oracle matched filter -- the theoretically optimal linear
detector, computed from the class means and noise rather than learned --
which upper-bounds what ANY linear model could achieve here. If the oracle
succeeds where LR fails, that is direct evidence for estimation error.

Usage:
    python test_ridge_leech.py
    python test_ridge_leech.py --scales 0.5 0.3 0.2
"""

import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from dow_data import GenParams, generate_dataset, normalize, CLASS_NAMES

NORMAL, LEECH = 0, 1


def build(scale, per_class, seed):
    """Generate a dataset with the leech scaled, return normalised arrays."""
    p = GenParams(leech_scale=scale)
    X, y, inten = generate_dataset(per_class=per_class, seed=seed, params=p)
    return normalize(X), y, inten


def leech_accuracy(pred, y, inten):
    m = inten == 1
    return (pred[m] == y[m]).mean() if m.any() else float("nan")


def as_normal(pred, inten):
    m = inten == 1
    return (pred[m] == NORMAL).mean() if m.any() else float("nan")


def oracle_matched_filter(Xtr, ytr, Xte, yte, ite):
    """
    The optimal linear detector, computed rather than learned.

    For each class, the template is the difference between that class's mean
    image and the normal mean, weighted by the inverse of the per-cell
    variance of the normal class. Score each test image against every
    template and take the argmax. This is what a linear model would do with
    perfectly estimated weights, so it upper-bounds any linear approach.
    """
    ntr = Xtr.reshape(len(Xtr), -1)
    nte = Xte.reshape(len(Xte), -1)

    mu_norm = ntr[ytr == NORMAL].mean(axis=0)
    var_norm = ntr[ytr == NORMAL].var(axis=0) + 1e-8

    templates, biases = [], []
    for c in range(len(CLASS_NAMES)):
        mu_c = ntr[ytr == c].mean(axis=0)
        w = (mu_c - mu_norm) / var_norm
        templates.append(w)
        biases.append(-0.5 * np.dot(w, mu_c + mu_norm))

    W = np.stack(templates)
    b = np.array(biases)
    scores = nte @ W.T + b
    pred = scores.argmax(axis=1)

    return leech_accuracy(pred, yte, ite), f1_score(yte, pred, average="macro")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scales", type=float, nargs="+",
                    default=[0.5, 0.3, 0.2])
    ap.add_argument("--per-class", type=int, default=300)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--Cs", type=float, nargs="+",
                    default=[100.0, 1.0, 0.1, 0.01, 0.001, 0.0001])
    args = ap.parse_args()

    print("=" * 74)
    print("DOES REGULARISATION RECOVER THE QUIET LEECH?")
    print("=" * 74)
    print("Lower C = stronger L2 penalty. C=100 is close to unregularised.")
    print(f"seeds: {args.seeds}   per_class: {args.per_class}\n")

    for scale in args.scales:
        print("=" * 74)
        print(f"LEECH SCALE {scale}")
        print("=" * 74)
        print(f"{'C':>10} | {'leech acc':>18} | {'as normal':>18} | {'macro-F1':>16}")
        print("-" * 74)

        best = (None, -1)
        for C in args.Cs:
            accs, norms, macros = [], [], []
            for s in args.seeds:
                X, y, inten = build(scale, args.per_class, seed=s)
                Xf = X.reshape(len(X), -1)
                Xtr, Xte, ytr, yte, itr, ite = train_test_split(
                    Xf, y, inten, test_size=0.30, random_state=s, stratify=y)

                lr = LogisticRegression(C=C, max_iter=3000, random_state=s)
                lr.fit(Xtr, ytr)
                pred = lr.predict(Xte)

                accs.append(leech_accuracy(pred, yte, ite))
                norms.append(as_normal(pred, ite))
                macros.append(f1_score(yte, pred, average="macro"))

            am, asd = np.mean(accs), np.std(accs)
            if am > best[1]:
                best = (C, am)
            print(f"{C:>10.4f} | {am:>10.3f} +/- {asd:.3f} | "
                  f"{np.mean(norms):>10.3f} +/- {np.std(norms):.3f} | "
                  f"{np.mean(macros):>9.3f} +/- {np.std(macros):.3f}")

        # ---- oracle ----
        orc_acc, orc_macro = [], []
        for s in args.seeds:
            X, y, inten = build(scale, args.per_class, seed=s)
            Xtr, Xte, ytr, yte, itr, ite = train_test_split(
                X, y, inten, test_size=0.30, random_state=s, stratify=y)
            a, m = oracle_matched_filter(Xtr, ytr, Xte, yte, ite)
            orc_acc.append(a); orc_macro.append(m)

        print("-" * 74)
        print(f"{'ORACLE':>10} | {np.mean(orc_acc):>10.3f} +/- "
              f"{np.std(orc_acc):.3f} | {'':>18} | "
              f"{np.mean(orc_macro):>9.3f} +/- {np.std(orc_macro):.3f}")
        print(f"\n  best learned C = {best[0]}  (leech acc {best[1]:.3f})")
        print(f"  oracle matched filter: {np.mean(orc_acc):.3f}")
        print()

    print("=" * 74)
    print("HOW TO READ THIS")
    print("=" * 74)
    print("""
Compare three things at each scale:

  1. Unregularised LR (C=100)  -- roughly the reported baseline
  2. Best regularised LR       -- does penalising the weights help?
  3. Oracle matched filter     -- the best any linear detector could do

  * Regularised LR >> unregularised, and approaching the oracle
      -> ESTIMATION ERROR CONFIRMED. The signal is linearly separable at
         this intensity; the unregularised fit was badly conditioned. The
         claim is now tested and the design recommendation follows.

  * Regularisation makes little difference, but the ORACLE succeeds
      -> the signal IS separable, but neither a learned linear model nor
         the CNN finds it. That is still a strong finding: the limit is in
         the fitting, not the representation.

  * Neither regularised LR nor the oracle recovers the leech
      -> the estimation explanation is WRONG. Delete that sentence from
         the research report and state only what the SNR analysis measured.
""")


if __name__ == "__main__":
    main()
