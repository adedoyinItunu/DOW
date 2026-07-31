"""
experiment5_multiseed_sweep.py  --  leech-intensity sweep across seeds
=====================================================================
The single-seed sweep (experiment5_leech_sweep.py) establishes the evasion
threshold from one model. Because the thesis argues that single-seed figures
are unreliable, the headline finding should itself be multi-seed. This runs the
sweep across several seeds and reports mean +/- std accuracy (and misread-as-
normal) at each intensity level.

For each seed it trains a fresh model on the default data, then evaluates that
model on freshly generated leech data at each intensity level.

    python experiment5_multiseed_sweep.py --seeds 0 1 2 3 4 \
        --levels 1.0 0.7 0.5 0.3 0.2 0.1

Note: this trains one model per seed, so it is the slowest script here
(~a few minutes per seed on CPU). Five seeds is enough to report mean +/- std;
ten matches the stability analysis if time permits.
"""
import argparse
import subprocess
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from dow_data import load_and_split, load_all
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
            opt.zero_grad(); crit(model(xb), yb).backward(); opt.step()
    model.eval()
    return model


def gen_leech(level, seed, per_class=300):
    out = f"_ms_{seed}_{level}.npz"
    subprocess.run([sys.executable, "dow_data.py", "--per-class", str(per_class),
                    "--seed", str(seed), "--leech-scale", str(level), "--out", out],
                   check=True, capture_output=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data.npz")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--levels", type=float, nargs="+",
                    default=[1.0, 0.7, 0.5, 0.3, 0.2, 0.1])
    args = ap.parse_args()

    # acc[level] = list of per-seed accuracies; norm[level] = list of misread-as-normal
    acc = {lvl: [] for lvl in args.levels}
    norm = {lvl: [] for lvl in args.levels}

    for s in args.seeds:
        set_seed(s)
        (Xtr, ytr, _), _, _, names = load_and_split(args.data, s)
        model = train_one(Xtr, ytr, len(names))
        print(f"seed {s}: model trained", flush=True)
        for lvl in args.levels:
            path = gen_leech(lvl, s)
            X, y, inten, _ = load_all(path)
            with torch.no_grad():
                pred = model(torch.tensor(X)).argmax(1).numpy()
            m = inten == 1
            acc[lvl].append((pred[m] == y[m]).mean())
            norm[lvl].append((pred[m] == 0).mean())

    print("\n=============== LEECH-INTENSITY SWEEP ACROSS "
          f"{len(args.seeds)} SEEDS ===============")
    print(f"{'scale':>6} | {'accuracy (mean +/- std)':>26} | {'as_normal (mean +/- std)':>26}")
    print("-" * 66)
    for lvl in args.levels:
        a = np.array(acc[lvl]); n = np.array(norm[lvl])
        print(f"{lvl:>6.2f} | {a.mean():>10.3f} +/- {a.std():<11.3f} | "
              f"{n.mean():>10.3f} +/- {n.std():<11.3f}")
    print("\nReading: report these mean +/- std figures in place of the single-seed")
    print("sweep table. The evasion threshold (accuracy < 0.5, majority read as")
    print("normal) can then be stated with the same evidential standard as the rest")
    print("of the results chapter.")

    # LaTeX-ready rows
    print("\n--- LaTeX table rows (accuracy mean +/- std, as_normal mean) ---")
    for lvl in args.levels:
        a = np.array(acc[lvl]); n = np.array(norm[lvl])
        print(f"{lvl:.2f} & ${a.mean():.3f} \\pm {a.std():.3f}$ & "
              f"${n.mean():.3f} \\pm {n.std():.3f}$ \\\\")


if __name__ == "__main__":
    main()
