"""
plot_e1_confusion.py

Draws the E1 confusion matrix from a SAVED checkpoint, without training.

experiment1_baseline.py has no --model flag: it trains a fresh model on
every run, which is how the thesis ended up with figures from different
checkpoints. Use this instead for the figure.

Usage:
    python plot_e1_confusion.py --model dow_cnn_locked.pt --seed 42
"""

import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from dow_data import load_and_split
from dow_model import DoWNetCNN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dow_cnn_locked.pt")
    ap.add_argument("--data", default="data.npz")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="e1_confusion_matrix.png")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    (_, _, _), (_, _, _), (Xte, yte, ite), names = load_and_split(
        args.data, args.seed)

    model = DoWNetCNN(n_classes=len(names))
    state = torch.load(args.model, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        pred = model(torch.tensor(Xte)).argmax(1).numpy()

    cm = confusion_matrix(yte, pred, labels=list(range(len(names))))
    acc = (pred == yte).mean()
    macro = f1_score(yte, pred, average="macro")

    print(f"model: {args.model}   seed: {args.seed}   n={len(yte)}")
    print(f"accuracy: {acc:.3f}   macro-F1: {macro:.3f}\n")
    print(classification_report(yte, pred, target_names=names,
                                digits=3, zero_division=0))
    print("\nconfusion matrix (rows=true, cols=pred):")
    print(cm)

    # ---- figure ----
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticklabels(names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=11)

    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"\nwrote {args.out}")

    print("\nSuggested caption:")
    print(r"\caption{Confusion matrix on the held-out test split for the "
          r"locked checkpoint (\texttt{" + args.model.replace("_", r"\_") +
          r"}, seed 42). Accuracy " + f"{acc:.3f}" +
          r", macro-F1 " + f"{macro:.3f}" + r".}")


if __name__ == "__main__":
    main()
