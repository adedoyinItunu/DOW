"""
reconcile_experiments.py  --  enforce ONE checkpoint across E1, E3, E4
======================================================================
The thesis states "Experiments 1-4 and 6 use a single trained checkpoint at
seed 42." That claim must be TRUE. This script makes it true: it trains once,
saves the checkpoint, and then runs every evaluation by LOADING that exact
file. No experiment retrains. The numbers it prints are, by construction,
mutually consistent, so E1's confusion matrix, E3's leech/flood table and
E4's baseline row will agree on how many samples land where.

Run this, then replace the E1 / E3 / E4 numbers and figures in the thesis with
what it prints. Do NOT mix these with numbers from any older checkpoint.

    python reconcile_experiments.py --seed 42

What it does, in order:
  1. Build the dataset at seed 42 (fixed split).
  2. Train ONE model, save to dow_cnn_locked.pt.
  3. E1: full-test confusion matrix + macro-F1.
  4. E3: leech vs flood accuracy on the SAME test split, from the SAME model.
  5. E4 baseline row: overall test accuracy, from the SAME model.
  6. Cross-check: verify E1's "attack samples predicted normal" count equals
     what E3 reports, so the contradiction cannot recur.
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, confusion_matrix

from dow_data import load_and_split
from dow_model import DoWNetCNN, set_seed

CKPT = "dow_cnn_locked.pt"


def train_once(Xtr, ytr, n_classes, seed, epochs=25, lr=1e-3, batch=32):
    set_seed(seed)
    ds = TensorDataset(torch.tensor(Xtr), torch.tensor(ytr))
    dl = DataLoader(ds, batch_size=batch, shuffle=True)
    model = DoWNetCNN(n_classes=n_classes)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for xb, yb in dl:
            opt.zero_grad(); crit(model(xb), yb).backward(); opt.step()
    model.eval()
    torch.save(model.state_dict(), CKPT)
    print(f"[trained once] saved {CKPT}")
    return model


def load_locked(n_classes):
    model = DoWNetCNN(n_classes=n_classes)
    model.load_state_dict(torch.load(CKPT))
    model.eval()
    return model


def predict(model, X):
    with torch.no_grad():
        return model(torch.tensor(X)).argmax(1).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data.npz")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    (Xtr, ytr, _), (Xva, yva, _), (Xte, yte, ite), names = load_and_split(
        args.data, args.seed)
    n = len(names)
    normal_idx = names.index("normal") if "normal" in names else 0

    # ---- 1 & 2: train once ----
    train_once(Xtr, ytr, n, args.seed)

    # ---- everything below LOADS the same file ----
    model = load_locked(n)
    pred = predict(model, Xte)

    # ---- 3: E1 confusion matrix + macro-F1 ----
    cm = confusion_matrix(yte, pred, labels=list(range(n)))
    macro = f1_score(yte, pred, average="macro")
    print("\n================ E1 (same locked model) ================")
    print("labels:", names)
    print("confusion matrix (rows=true, cols=pred):")
    print(cm)
    print(f"test accuracy: {(pred==yte).mean():.3f}  macro-F1: {macro:.3f}")

    # attack samples (true != normal) that were predicted normal
    attack_mask = yte != normal_idx
    attack_to_normal = int(((pred == normal_idx) & attack_mask).sum())
    print(f"attack samples predicted normal (E1 view): {attack_to_normal}")

    # ---- 4: E3 leech vs flood on the SAME split & model ----
    # intensity tag: 1 = leech, 2 = flood, 0 = normal (adjust if the tag differs)
    leech_mask = ite == 1
    flood_mask = ite == 2
    leech_acc = (pred[leech_mask] == yte[leech_mask]).mean() if leech_mask.any() else float("nan")
    flood_acc = (pred[flood_mask] == yte[flood_mask]).mean() if flood_mask.any() else float("nan")
    leech_to_normal = int((pred[leech_mask] == normal_idx).sum())
    flood_to_normal = int((pred[flood_mask] == normal_idx).sum())
    print("\n================ E3 (same locked model) ================")
    print(f"leech: n={int(leech_mask.sum())}  acc={leech_acc:.3f}  "
          f"misread_as_normal={leech_to_normal}")
    print(f"flood: n={int(flood_mask.sum())}  acc={flood_acc:.3f}  "
          f"misread_as_normal={flood_to_normal}")

    # ---- 5: E4 baseline overall accuracy ----
    print("\n================ E4 baseline (same locked model) ================")
    print(f"overall test accuracy: {(pred==yte).mean():.3f} "
          f"({(pred==yte).sum()}/{len(yte)})")

    # ---- 6: consistency cross-check ----
    print("\n================ CONSISTENCY CHECK ================")
    print(f"E1 attack->normal count:        {attack_to_normal}")
    print(f"E3 leech->normal + flood->normal: {leech_to_normal + flood_to_normal}")
    if attack_to_normal == leech_to_normal + flood_to_normal:
        print("CONSISTENT: E1 and E3 agree on attack->normal. "
              "Numbers reconcile.")
    else:
        print("MISMATCH REMAINS: check that the intensity tag (ite) values are "
              "1=leech, 2=flood and that E1/E3 use this same test split.")
    print("\nUse ONLY these numbers in E1/E3/E4. Regenerate the E1 confusion-"
          "matrix figure and the E3 table from this run.")


if __name__ == "__main__":
    main()
