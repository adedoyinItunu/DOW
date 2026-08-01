"""
test_normalisation.py

Tests the thesis's central causal claim:

    "The transfer failure is a property of the representation, specifically
     its fixed-constant normalisation."

Currently that is inferred from two facts -- that base amplitude alone
collapses the normal class, and that the linear baseline fails on the same
class. It has never been tested directly.

The test: retrain on Configuration A and evaluate on Configuration B, twice.
Once with the fixed divisor log1p(400), once with per-image normalisation.
If the normal-class F1 recovers under per-image normalisation, the claim is
demonstrated. If it does not, the cause is the diurnal-shape sensitivity you
also observed, and Sections 4.7, 4.8 and 5.1 need rewriting.

This script does NOT modify dow_data.py. It loads the raw counts from the
.npz files and applies each normalisation itself.

Prereqs:
    data_configA.npz and data_configB.npz already exist.

Usage:
    python test_normalisation.py --seeds 0 1 2 3 4
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from dow_data import CLASS_NAMES, _NORM_REF
from dow_model import DoWNetCNN, set_seed


def norm_fixed(X):
    """The current scheme: log1p then divide by a fixed constant."""
    return (np.log1p(np.clip(X, 0, None)) / _NORM_REF).astype("float32")


def norm_per_image(X):
    """Per-image: log1p then divide by that image's own maximum."""
    L = np.log1p(np.clip(X, 0, None))
    m = L.reshape(L.shape[0], -1).max(axis=1)
    m = np.where(m > 0, m, 1.0).reshape(-1, 1, 1, 1)
    return (L / m).astype("float32")


def norm_robust(X):
    """Per-image robust: log1p, subtract median, divide by IQR."""
    L = np.log1p(np.clip(X, 0, None))
    flat = L.reshape(L.shape[0], -1)
    med = np.median(flat, axis=1).reshape(-1, 1, 1, 1)
    q75 = np.percentile(flat, 75, axis=1).reshape(-1, 1, 1, 1)
    q25 = np.percentile(flat, 25, axis=1).reshape(-1, 1, 1, 1)
    iqr = np.where((q75 - q25) > 1e-6, q75 - q25, 1.0)
    return ((L - med) / iqr).astype("float32")


SCHEMES = {
    "fixed log1p(400)": norm_fixed,
    "per-image max":    norm_per_image,
    "per-image robust": norm_robust,
}


def train_and_eval(Xa_raw, ya, Xb_raw, yb, normfn, seed, epochs=25, batch=32):
    Xa = normfn(Xa_raw)
    Xb = normfn(Xb_raw)

    Xtr, _, ytr, _ = train_test_split(
        Xa, ya, test_size=0.30, random_state=seed, stratify=ya)

    set_seed(seed)
    ds = TensorDataset(torch.tensor(Xtr), torch.tensor(ytr))
    dl = DataLoader(ds, batch_size=batch, shuffle=True)
    model = DoWNetCNN(n_classes=len(CLASS_NAMES))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        for xb, yb_ in dl:
            opt.zero_grad(); crit(model(xb), yb_).backward(); opt.step()
    model.eval()

    with torch.no_grad():
        pred = model(torch.tensor(Xb)).argmax(1).numpy()

    per_class = f1_score(yb, pred, average=None,
                         labels=list(range(len(CLASS_NAMES))), zero_division=0)
    macro = f1_score(yb, pred, average="macro", zero_division=0)
    return macro, per_class, pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-a", default="data_configA.npz")
    ap.add_argument("--config-b", default="data_configB.npz")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = ap.parse_args()

    da = np.load(args.config_a, allow_pickle=True)
    db = np.load(args.config_b, allow_pickle=True)
    Xa_raw, ya = da["X"], da["y"]
    Xb_raw, yb = db["X"], db["y"]

    print("=" * 72)
    print("DOES PER-IMAGE NORMALISATION FIX THE TRANSFER FAILURE?")
    print("=" * 72)
    print(f"train on {args.config_a} ({len(ya)} samples)")
    print(f"test on  {args.config_b} ({len(yb)} samples)")
    print(f"seeds: {args.seeds}\n")

    summary = {}

    for scheme, fn in SCHEMES.items():
        macros, normals = [], []
        last_pred = None
        for s in args.seeds:
            macro, per_class, pred = train_and_eval(
                Xa_raw, ya, Xb_raw, yb, fn, s)
            macros.append(macro)
            normals.append(per_class[0])       # class 0 = normal
            last_pred = pred

        summary[scheme] = (np.mean(macros), np.std(macros),
                           np.mean(normals), np.std(normals))

        print("-" * 72)
        print(f"SCHEME: {scheme}")
        print(f"  cross-config macro-F1 : {np.mean(macros):.3f} "
              f"+/- {np.std(macros):.3f}")
        print(f"  NORMAL-class F1       : {np.mean(normals):.3f} "
              f"+/- {np.std(normals):.3f}")
        print(f"  last-seed confusion matrix (rows=true, cols=pred):")
        cm = confusion_matrix(yb, last_pred,
                              labels=list(range(len(CLASS_NAMES))))
        print("   " + "".join(f"{n[:8]:>9}" for n in CLASS_NAMES))
        for i, row in enumerate(cm):
            print(f"{CLASS_NAMES[i][:8]:>8} " +
                  "".join(f"{v:>9}" for v in row))
        print()

    # ---- verdict ----
    print("=" * 72)
    print("VERDICT")
    print("=" * 72)
    base_normal = summary["fixed log1p(400)"][2]
    print(f"{'scheme':>20} | {'macro-F1':>16} | {'normal-class F1':>18}")
    print("-" * 62)
    for scheme, (mm, ms, nm, ns) in summary.items():
        print(f"{scheme:>20} | {mm:>7.3f} +/- {ms:.3f} | {nm:>9.3f} +/- {ns:.3f}")

    best_alt = max((k for k in summary if k != "fixed log1p(400)"),
                   key=lambda k: summary[k][2])
    alt_normal = summary[best_alt][2]

    print()
    if alt_normal > base_normal + 0.20:
        print(f"CLAIM SUPPORTED. Normal-class F1 recovers from "
              f"{base_normal:.3f} to {alt_normal:.3f} under '{best_alt}'.")
        print("The fixed-constant normalisation is demonstrably the cause of")
        print("the transfer failure. Sections 4.7 and 5.1 can state this as a")
        print("tested result rather than an inference.")
    elif alt_normal > base_normal + 0.05:
        print(f"PARTIAL. Normal-class F1 rises from {base_normal:.3f} to "
              f"{alt_normal:.3f} under '{best_alt}'.")
        print("Normalisation contributes but does not fully account for the")
        print("failure. The diurnal-shape sensitivity is also implicated.")
        print("Reword Sections 4.7 and 5.1 to reflect a partial cause.")
    else:
        print(f"CLAIM NOT SUPPORTED. Normal-class F1 stays at "
              f"{alt_normal:.3f} versus {base_normal:.3f}.")
        print("Per-image normalisation does not fix the transfer failure, so")
        print("the fixed divisor is NOT the cause. The remaining candidate is")
        print("the diurnal-shape (width) sensitivity reported in Section 4.7.")
        print("Sections 4.7, 4.8 and 5.1 must be rewritten accordingly.")


if __name__ == "__main__":
    main()
