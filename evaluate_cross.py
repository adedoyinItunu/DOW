"""
evaluate_cross.py  --  the generator-parameter robustness test
==============================================================
Loads a CNN trained on one generator configuration and evaluates it on a dataset
generated with DIFFERENT parameters. Reports per-class precision/recall/F1, the
confusion matrix, and a leech-vs-flood breakdown -- the additional checks required
for (per-class rather than aggregate, with attention to the low-rate leech).

    # 1. train on Config A (default generator)
    python dow_data.py --per-class 300 --out data_configA.npz
    python experiment1_baseline.py --data data_configA.npz --seed 0 --out cnn_configA.pt

    # 2. generate a DIFFERENT configuration (do NOT retrain on it)
    python dow_data.py --per-class 300 --peak-hour 10 --width 5.0 --amp 45 \
                       --out data_configB.npz

    # 3. cross-evaluate: model from A, data from B
    python evaluate_cross.py --model cnn_configA.pt --data data_configB.npz

Honesty note: if accuracy drops on Config B (especially on the leech), that is a
VALID finding -- report it as a limitation. Do not tune Config B until the numbers
improve; that would be the post-hoc adjustment to avoid.
"""
import argparse
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from dow_data import load_all
from dow_model import DoWNetCNN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="trained model, e.g. cnn_configA.pt")
    ap.add_argument("--data", required=True, help="DIFFERENT-config dataset, e.g. data_configB.npz")
    ap.add_argument("--out-cm", default="cross_confusion_matrix.png")
    args = ap.parse_args()

    X, y, inten, names = load_all(args.data)
    Xt = torch.tensor(X)

    model = DoWNetCNN(n_classes=len(names))
    model.load_state_dict(torch.load(args.model, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        logits = model(Xt)
        pred = logits.argmax(1).numpy()

    print(f"=== CROSS-EVALUATION ===")
    print(f"model: {args.model}   evaluated on: {args.data}")
    print(f"samples: {len(y)}\n")

    print("Per-class metrics (trained on A, tested on B):")
    print(classification_report(y, pred, target_names=names, digits=3, zero_division=0))
    macro = f1_score(y, pred, average="macro")
    print(f"Macro-F1 (cross): {macro:.3f}\n")

    print("Confusion matrix (rows = true, cols = predicted):")
    cm = confusion_matrix(y, pred)
    print("        " + "  ".join(f"{n[:6]:>6}" for n in names))
    for i, n in enumerate(names):
        print(f"{n[:8]:>8} " + "  ".join(f"{v:6d}" for v in cm[i]))
    print()

    # leech vs flood breakdown (the research focus)
    print("Leech (low-rate) vs flood (high-rate) on the cross set:")
    for lvl, label in [(1, "leech (low) "), (2, "flood (high)")]:
        m = inten == lvl
        if m.sum() == 0:
            continue
        acc = (pred[m] == y[m]).mean()
        as_normal = (pred[m] == 0).mean()   # class 0 == normal
        print(f"  {label}: n={int(m.sum()):3d}  accuracy={acc:.3f}  "
              f"misread-as-normal={as_normal:.3f}")

    # optional confusion-matrix figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4.5, 4))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.set_title("Cross-config confusion (train A / test B)")
        fig.colorbar(im); fig.tight_layout()
        fig.savefig(args.out_cm, dpi=120); plt.close(fig)
        print(f"\nsaved {args.out_cm}")
    except Exception as e:
        print(f"(confusion-matrix figure skipped: {e})")


if __name__ == "__main__":
    main()
