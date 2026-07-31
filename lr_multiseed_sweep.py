"""
lr_multiseed_sweep.py  --  logistic-regression leech sweep across seeds
======================================================================
Companion to experiment5_multiseed_sweep.py. Runs the same leech-intensity
sweep on the logistic-regression baseline across seeds, so the VARIANCE of the
two models can be compared fairly. LR trains in seconds, so this is fast.

If LR stays tight across seeds while the CNN is highly variable, the evasion
threshold is a stable property of the task/representation, and the CNN's
variance around it is the CNN's own instability.

    python lr_multiseed_sweep.py --seeds 0 1 2 3 4 --levels 1.0 0.7 0.5 0.3 0.2 0.1
"""
import argparse
import subprocess
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression

from dow_data import load_and_split, load_all


def gen_leech(level, seed, per_class=300):
    out = f"_lrms_{seed}_{level}.npz"
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

    acc = {lvl: [] for lvl in args.levels}
    norm = {lvl: [] for lvl in args.levels}

    for s in args.seeds:
        (Xtr, ytr, _), _, _, names = load_and_split(args.data, s)
        lr = LogisticRegression(max_iter=2000)
        lr.fit(Xtr.reshape(len(Xtr), -1), ytr)
        for lvl in args.levels:
            path = gen_leech(lvl, s)
            X, y, inten, _ = load_all(path)
            pred = lr.predict(X.reshape(len(X), -1))
            m = inten == 1
            acc[lvl].append((pred[m] == y[m]).mean())
            norm[lvl].append((pred[m] == 0).mean())
        print(f"seed {s}: done", flush=True)

    print("\n=============== LR LEECH SWEEP ACROSS "
          f"{len(args.seeds)} SEEDS ===============")
    print(f"{'scale':>6} | {'LR accuracy (mean +/- std)':>28} | {'as_normal (mean +/- std)':>26}")
    print("-" * 66)
    for lvl in args.levels:
        a = np.array(acc[lvl]); n = np.array(norm[lvl])
        print(f"{lvl:>6.2f} | {a.mean():>12.3f} +/- {a.std():<11.3f} | "
              f"{n.mean():>10.3f} +/- {n.std():<11.3f}")
    print("\nCompare the std column with the CNN multi-seed sweep. If LR's std is")
    print("much smaller, the threshold is stable in the linear model and the CNN's")
    print("variance is the CNN's instability, not the task's.")

    print("\n--- LaTeX rows ---")
    for lvl in args.levels:
        a = np.array(acc[lvl])
        print(f"{lvl:.2f} & ${a.mean():.3f} \\pm {a.std():.3f}$ \\\\")


if __name__ == "__main__":
    main()
