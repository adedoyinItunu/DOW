"""
baseline_extended.py  --  is the CNN's behaviour a property of the task or the model?
=====================================================================================
Following the review, this runs the logistic-regression baseline in the same
three settings as the CNN, for a fair comparison:

  (1) across the same ten partition-varying seeds  -> LR mean +/- std vs CNN 0.756 +/- 0.149
  (2) through the leech-intensity sweep             -> does LR also collapse at low intensity?
  (3) on the Config-A -> Config-B transfer          -> does LR also fail to generalise?

Interpretation:
  - If LR is both higher and more stable than the CNN across seeds, the CNN's
    instability is a property of the architecture/training, not the data.
  - If LR ALSO collapses at low leech intensity, the evasion threshold is a
    property of the TASK, not the CNN (a more general finding).
  - If LR ALSO fails on Config-B, the transfer failure is a property of the
    representation/task, not the CNN specifically.

    python baseline_extended.py --data data.npz
"""
import argparse
import subprocess
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from dow_data import load_and_split, load_all


def fit_lr(Xtr, ytr):
    lr = LogisticRegression(max_iter=2000)
    lr.fit(Xtr.reshape(len(Xtr), -1), ytr)
    return lr


def gen(level_flag, seed, per_class=300):
    out = f"data_lrtmp.npz"
    cmd = [sys.executable, "dow_data.py", "--per-class", str(per_class),
           "--seed", str(seed)] + level_flag + ["--out", out]
    subprocess.run(cmd, check=True, capture_output=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data.npz")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    args = ap.parse_args()

    # (1) across partition-varying seeds
    print("=============== (1) LOGISTIC REGRESSION ACROSS SEEDS ===============")
    f1s = []
    for s in args.seeds:
        (Xtr, ytr, _), (_, _, _), (Xte, yte, _), names = load_and_split(args.data, s)
        lr = fit_lr(Xtr, ytr)
        f = f1_score(yte, lr.predict(Xte.reshape(len(Xte), -1)), average="macro")
        f1s.append(f)
        print(f"  seed {s}: macro-F1 {f:.3f}")
    print(f"\n  Logistic regression: {np.mean(f1s):.3f} +/- {np.std(f1s):.3f} "
          f"(range {min(f1s):.3f}-{max(f1s):.3f})")
    print(f"  CNN (for comparison): 0.756 +/- 0.149 (range 0.525-0.960)")

    # train one LR on the default data for the sweep + transfer tests (seed 42)
    (Xtr, ytr, _), _, _, names = load_and_split(args.data, 42)
    lr_main = fit_lr(Xtr, ytr)

    # (2) leech sweep
    print("\n=============== (2) LOGISTIC REGRESSION THROUGH THE LEECH SWEEP ===============")
    print(f"{'leech_scale':>11} | {'accuracy':>8} | {'as_normal':>9}")
    print("-" * 36)
    for lvl in [1.0, 0.7, 0.5, 0.3, 0.2, 0.1]:
        path = gen(["--leech-scale", str(lvl)], 42)
        X, y, inten, _ = load_all(path)
        pred = lr_main.predict(X.reshape(len(X), -1))
        m = inten == 1
        acc = (pred[m] == y[m]).mean()
        as_normal = (pred[m] == 0).mean()
        print(f"{lvl:>11.2f} | {acc:>8.3f} | {as_normal:>9.3f}")
    print("  (compare with the CNN sweep: 0.967 -> 0.000 as scale falls 1.0 -> 0.1)")

    # (3) Config-B transfer
    print("\n=============== (3) LOGISTIC REGRESSION ON CONFIG-B TRANSFER ===============")
    pathB = gen(["--peak-hour", "10", "--width", "5.0", "--amp", "45"], 42)
    Xb, yb, ib, _ = load_all(pathB)
    predB = lr_main.predict(Xb.reshape(len(Xb), -1))
    from sklearn.metrics import classification_report
    print(classification_report(yb, predB, target_names=names, digits=3, zero_division=0))
    print("  (compare with the CNN cross-config macro-F1 of 0.549, normal F1 0.000)")

    print("\nReading:")
    print("  - LR higher AND more stable than CNN across seeds -> instability is the CNN's, not the data's.")
    print("  - LR ALSO collapses at low leech intensity -> evasion threshold is a property of the TASK.")
    print("  - LR ALSO fails on Config-B -> transfer failure is a property of the representation, not the CNN.")


if __name__ == "__main__":
    main()
