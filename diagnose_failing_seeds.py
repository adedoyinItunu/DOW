"""
diagnose_failing_seeds.py  --  what actually happens on the collapsed seeds
===========================================================================
An F1 of 0.000 for a class does not establish that the class was never
predicted; it establishes only that no positive prediction for it was correct.
This script distinguishes the four possibilities for each collapsed
(seed, class) pair:

  (a) the class was never predicted at all           -> predicted count == 0
  (b) all its true samples went to one other class   -> row concentrated
  (c) it was predicted, but only for wrong samples   -> predicted count > 0,
                                                        diagonal == 0
  (d) the run did not converge                       -> validation trajectory

It retrains under the registered protocol (25 epochs, final-epoch parameters,
no selection) so the runs reproduce Table 4.2's first column exactly, prints a
full confusion matrix per seed, classifies each collapse into (a)-(d), and
saves a figure of the failing seeds' matrices for the report.

    python diagnose_failing_seeds.py --seeds 2 6 7 9 --out failing_seeds.png

Default seeds are the four that collapse under L-ramp at 25 epochs. Pass
--seeds 0 1 2 3 4 5 6 7 8 9 to scan all ten and let the script find them.
"""
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from dow_data import load_and_split
from dow_model import set_seed
from architecture_controls import train_one


def classify_collapse(cm, k, names):
    """Return a one-line diagnosis for class k given the confusion matrix."""
    predicted = cm[:, k].sum()
    true_row = cm[k, :]
    if predicted == 0:
        return f"(a) never predicted; its {true_row.sum()} samples went to " \
               f"{names[int(np.argmax(true_row))]} ({true_row.max()} of {true_row.sum()})"
    if cm[k, k] == 0 and predicted > 0:
        wrong_from = int(np.argmax(cm[:, k] * (np.arange(len(names)) != k)))
        return f"(c) predicted {predicted} times but never correctly; " \
               f"those predictions were mostly true {names[wrong_from]}"
    return "not a collapse"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_lramp.npz")
    ap.add_argument("--seeds", type=int, nargs="+", default=[2, 6, 7, 9])
    ap.add_argument("--protocol", default="fixed25",
                    choices=["fixed25", "fixed25_sel", "converged"])
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--out", default="failing_seeds.png")
    args = ap.parse_args()

    print(f"=== failing-seed diagnosis, {args.protocol} on {args.data} ===\n")
    results = []

    for s in args.seeds:
        set_seed(s)
        (Xtr, ytr, _), (Xv, yv, _), (Xte, yte, _), names = \
            load_and_split(args.data, s)
        model, trace = train_one(Xtr, ytr, Xv, yv, len(names),
                                 "minimal", args.protocol, epochs=args.epochs)
        model.eval()
        with torch.no_grad():
            pred = model(torch.tensor(Xte)).argmax(1).numpy()

        cm = confusion_matrix(yte, pred, labels=range(len(names)))
        _, _, f1, _ = precision_recall_fscore_support(
            yte, pred, labels=range(len(names)), zero_division=0)

        print(f"--- seed {s}  (macro-F1 {f1.mean():.3f}) ---")
        w = max(len(n) for n in names) + 1
        print(" " * (w + 8) + "predicted")
        print(" " * (w + 2) + "  ".join(f"{n[:6]:>6}" for n in names))
        for i, n in enumerate(names):
            print(f"true {n:<{w}}" + "  ".join(f"{v:>6}" for v in cm[i])
                  + f"   F1={f1[i]:.3f}")

        for k, n in enumerate(names):
            if f1[k] == 0.0:
                diag = classify_collapse(cm, k, names)
                print(f"  COLLAPSE in {n}: {diag}")
                results.append((s, n, diag))

        tail = trace[-6:]
        swing = max(abs(tail[i + 1] - tail[i]) for i in range(len(tail) - 1))
        print(f"  final six validation accuracies: "
              + ", ".join(f"{v:.3f}" for v in tail))
        print(f"  largest swing among them: {swing:.3f}"
              + ("   <- (d) trajectory not settled" if swing > 0.15 else ""))
        print()

        results.append((s, None, cm, f1, names))

    # ---- figure: one confusion matrix per failing seed
    mats = [r for r in results if len(r) == 5]
    fig, axes = plt.subplots(1, len(mats), figsize=(3.3 * len(mats), 3.6))
    if len(mats) == 1:
        axes = [axes]
    for ax, (s, _, cm, f1, names) in zip(axes, mats):
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n[:4] for n in names], rotation=45, ha="right")
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels([n[:4] for n in names] if ax is axes[0] else [])
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_title(f"seed {s}  (macro-F1 {f1.mean():.2f})", fontsize=9)
        ax.set_xlabel("predicted", fontsize=8)
    axes[0].set_ylabel("true", fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")

    print("\n=== summary of collapses ===")
    for r in results:
        if len(r) == 3:
            print(f"seed {r[0]}, class {r[1]}: {r[2]}")


if __name__ == "__main__":
    main()
