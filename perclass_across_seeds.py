"""
perclass_across_seeds.py  --  per-class metrics across random seeds
==================================================================
Reports per-class results across the existing seeds, because an
aggregate macro-F1 can mask weaker or more variable performance on a specific
class. Trains the baseline across seeds and reports, per class, the mean and
standard deviation of precision, recall and F1 over the seeds.

    python perclass_across_seeds.py --data data.npz --seeds 0 1 2 3 4 5 6 7 8 9
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support

from dow_data import load_and_split
from dow_model import DoWNetCNN, set_seed


def train_one(Xtr, ytr, n_classes, epochs=25, lr=1e-3, batch=32):
    ds = TensorDataset(torch.tensor(Xtr), torch.tensor(ytr))
    dl = DataLoader(ds, batch_size=batch, shuffle=True)
    model = DoWNetCNN(n_classes=n_classes)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for xb, yb in dl:
            opt.zero_grad(); loss = crit(model(xb), yb); loss.backward(); opt.step()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data.npz")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    args = ap.parse_args()

    # collect per-class P/R/F for each seed
    P, R, F = [], [], []
    names = None
    for s in args.seeds:
        set_seed(s)
        (Xtr, ytr, _), (_, _, _), (Xte, yte, _), names = load_and_split(args.data, s)
        model = train_one(Xtr, ytr, len(names))
        model.eval()
        with torch.no_grad():
            pred = model(torch.tensor(Xte)).argmax(1).numpy()
        p, r, f, _ = precision_recall_fscore_support(
            yte, pred, labels=range(len(names)), zero_division=0)
        P.append(p); R.append(r); F.append(f)
        print(f"seed {s}: macro-F1 {f.mean():.3f}  per-class F1 " +
              " ".join(f"{n[:4]}={fi:.3f}" for n, fi in zip(names, f)))

    P, R, F = np.array(P), np.array(R), np.array(F)
    print("\n=============== PER-CLASS METRICS ACROSS "
          f"{len(args.seeds)} SEEDS (mean \u00b1 std) ===============")
    print(f"{'class':>10} | {'precision':>15} | {'recall':>15} | {'f1-score':>15}")
    print("-" * 66)
    for i, n in enumerate(names):
        print(f"{n:>10} | {P[:, i].mean():.3f} \u00b1 {P[:, i].std():.3f}   | "
              f"{R[:, i].mean():.3f} \u00b1 {R[:, i].std():.3f}   | "
              f"{F[:, i].mean():.3f} \u00b1 {F[:, i].std():.3f}")
    print("-" * 66)
    print(f"{'macro':>10} | {'':>15} | {'':>15} | "
          f"{F.mean(axis=1).mean():.3f} \u00b1 {F.mean(axis=1).std():.3f}")
    print("\nReading: watch for any class whose F1 is systematically lower or more")
    print("variable (larger std) than the others -- especially where the leech-heavy")
    print("attack classes are concerned.")


if __name__ == "__main__":
    main()
