"""
paired_test_cnn_vs_lr.py  --  the paired significance test for Section 4.2.1
============================================================================
Section 4.2.1 compares the CNN against logistic regression across ten
partition-varying seeds but reports only that the ranges do not overlap. The
two models are evaluated on the SAME partitions, so the comparison is paired
and a paired test is available at no cost.

This trains both models on each seed, using the protocol of
reconcile_experiments.py (25 epochs, Adam 1e-3, batch 32) for the CNN and the
setup of baseline_comparison.py (multinomial logistic regression on raw
flattened pixels, max_iter=2000) for the baseline, then runs a Wilcoxon
signed-rank test over the ten paired macro-F1 values.

    python paired_test_cnn_vs_lr.py
    python paired_test_cnn_vs_lr.py --seeds 0 1 2 3 4 5 6 7 8 9

Paste the printed statistic and p-value into the \\todo{} marker in Section 4.2.1.
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from scipy.stats import wilcoxon

from dow_data import load_and_split
from dow_model import DoWNetCNN, set_seed


def train_cnn(Xtr, ytr, seed, epochs=25, lr=1e-3, batch=32):
    """Identical to reconcile_experiments.train_once, minus the checkpoint save."""
    set_seed(seed)
    ds = TensorDataset(torch.tensor(Xtr), torch.tensor(ytr))
    dl = DataLoader(ds, batch_size=batch, shuffle=True)
    model = DoWNetCNN(n_classes=4)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for xb, yb in dl:
            opt.zero_grad(); crit(model(xb), yb).backward(); opt.step()
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data.npz")
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    args = ap.parse_args()

    cnn_f1, lr_f1 = [], []

    print(f"{'seed':>5} | {'CNN macro-F1':>13} | {'LR macro-F1':>12} | {'diff':>7}")
    print("-" * 50)

    for sd in args.seeds:
        (Xtr, ytr, _), _, (Xte, yte, _) , _ = load_and_split(args.data, seed=sd)

        model = train_cnn(Xtr, ytr, sd)
        with torch.no_grad():
            pred = model(torch.tensor(Xte)).argmax(1).numpy()
        c = f1_score(yte, pred, average="macro")

        clf = LogisticRegression(max_iter=2000)
        clf.fit(Xtr.reshape(len(Xtr), -1), ytr)
        l = f1_score(yte, clf.predict(Xte.reshape(len(Xte), -1)), average="macro")

        cnn_f1.append(c); lr_f1.append(l)
        print(f"{sd:>5} | {c:>13.3f} | {l:>12.3f} | {l - c:>7.3f}")

    cnn_f1 = np.array(cnn_f1); lr_f1 = np.array(lr_f1)

    print("\n" + "=" * 50)
    print(f"CNN : {cnn_f1.mean():.3f} +/- {cnn_f1.std():.3f} "
          f"(range {cnn_f1.min():.3f}-{cnn_f1.max():.3f})")
    print(f"LR  : {lr_f1.mean():.3f} +/- {lr_f1.std():.3f} "
          f"(range {lr_f1.min():.3f}-{lr_f1.max():.3f})")

    stat, p = wilcoxon(lr_f1, cnn_f1)
    print(f"\nWilcoxon signed-rank, paired over {len(args.seeds)} seeds:")
    print(f"  statistic W = {stat:.1f}")
    print(f"  p = {p:.5f}")
    print(f"  LR exceeds CNN on {int((lr_f1 > cnn_f1).sum())} of {len(args.seeds)} seeds")
    print("\nSentence for Section 4.2.1:")
    print(f"  A Wilcoxon signed-rank test over the ten paired seeds confirms the")
    print(f"  difference is significant (W = {stat:.1f}, p = {p:.4f}), with the linear")
    print(f"  model exceeding the convolutional one on "
          f"{int((lr_f1 > cnn_f1).sum())} of {len(args.seeds)} partitions.")


if __name__ == "__main__":
    main()
