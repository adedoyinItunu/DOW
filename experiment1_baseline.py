"""
experiment1_baseline.py  --  EXPERIMENT 1 (baseline accuracy)
============================================================
Trains the CNN on the four-class heat-maps and reports accuracy, precision,
recall and F1 on the held-out test set. This is the DoWNet replication.

    python experiment1_baseline.py --data data.npz --epochs 25 --out dow_cnn.pt
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from dow_data import load_and_split
from dow_model import DoWNetCNN, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data.npz")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="dow_cnn.pt")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from sklearn.metrics import classification_report, confusion_matrix, f1_score
    import matplotlib.pyplot as plt

    set_seed(args.seed)
    (Xtr, ytr, _), (Xv, yv, _), (Xte, yte, _), names = load_and_split(args.data, args.seed)
    print(f"train {len(ytr)}  val {len(yv)}  test {len(yte)}  classes {names}")

    def loader(X, y, shuffle):
        ds = TensorDataset(torch.tensor(X), torch.tensor(y))
        return DataLoader(ds, batch_size=args.batch, shuffle=shuffle)

    tr, va = loader(Xtr, ytr, True), loader(Xv, yv, False)

    model = DoWNetCNN(n_classes=len(names))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()

    best_state, best_val = None, -1.0
    for ep in range(1, args.epochs + 1):
        model.train()
        for xb, yb in tr:
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward(); opt.step()
        # validation accuracy
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in va:
                pred = model(xb).argmax(1)
                correct += (pred == yb).sum().item(); total += len(yb)
        val_acc = correct / total
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        print(f"epoch {ep:2d}  val_acc {val_acc:.3f}")

    model.load_state_dict(best_state)            # restore best

    # ---- test evaluation ----
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(Xte))
        y_pred = logits.argmax(1).numpy()

    print("\n=============== EXPERIMENT 1: TEST RESULTS ===============")
    print(classification_report(yte, y_pred, target_names=names, digits=3))
    print(f"Macro F1: {f1_score(yte, y_pred, average='macro'):.3f}")

    cm = confusion_matrix(yte, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title("E1 confusion matrix")
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, fraction=0.046, pad=0.04); fig.tight_layout()
    fig.savefig("e1_confusion_matrix.png", dpi=130, bbox_inches="tight"); plt.close(fig)
    print("saved e1_confusion_matrix.png")

    torch.save(model.state_dict(), args.out)
    print(f"saved model -> {args.out}")


if __name__ == "__main__":
    main()
