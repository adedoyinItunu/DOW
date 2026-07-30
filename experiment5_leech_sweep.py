"""
experiment5_leech_sweep.py  --  controlled leech-intensity analysis
==================================================================
Makki's priority #1: examine leech detection across CONTROLLED intensity levels,
rather than at a single setting. Generates datasets with the leech scaled to
several intensities, evaluates the trained model on each, and reports for the
LEECH subset: accuracy, misread-as-normal rate, and per-class breakdown.

The point: find the intensity at which the low-rate leech stops being detected
(and, ideally, starts being confused with normal traffic). This is central to the
research question and also gives Experiment 4 the headroom it lacked.

    python experiment5_leech_sweep.py --model dow_cnn.pt --seed 42 \
        --levels 1.0 0.7 0.5 0.3 0.2 0.1
"""
import argparse
import subprocess
import sys
import numpy as np
import torch

from dow_data import load_all
from dow_model import DoWNetCNN


def gen_leech_data(level, seed, per_class=300):
    """Generate a dataset with leech intensity scaled to `level`, return its path."""
    out = f"data_leech_{level}.npz"
    cmd = [sys.executable, "dow_data.py", "--per-class", str(per_class),
           "--seed", str(seed), "--leech-scale", str(level), "--out", out]
    subprocess.run(cmd, check=True, capture_output=True)
    return out


def eval_leech(model, path, names):
    X, y, inten, _ = load_all(path)
    with torch.no_grad():
        pred = model(torch.tensor(X)).argmax(1).numpy()
    m = inten == 1                       # leech subset only
    acc = (pred[m] == y[m]).mean()
    as_normal = (pred[m] == 0).mean()    # class 0 == normal
    # where do misclassified leech samples go?
    wrong = pred[m][pred[m] != y[m]]
    dist = {names[k]: int((wrong == k).sum()) for k in range(len(names))}
    return acc, as_normal, dist, int(m.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dow_cnn.pt")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--per-class", type=int, default=300)
    ap.add_argument("--levels", type=float, nargs="+",
                    default=[1.0, 0.7, 0.5, 0.3, 0.2, 0.1])
    args = ap.parse_args()

    # class names (load from any generated file)
    _, _, _, names = load_all(gen_leech_data(args.levels[0], args.seed, args.per_class))

    model = DoWNetCNN(n_classes=len(names))
    model.load_state_dict(torch.load(args.model, map_location="cpu"))
    model.eval()

    print("=============== EXPERIMENT 5: leech-intensity sweep ===============")
    print(f"model: {args.model}   seed: {args.seed}\n")
    print(f"{'leech_scale':>11} | {'n':>4} | {'accuracy':>8} | {'as_normal':>9} | misclassified-as")
    print("-" * 72)

    results = []
    for lvl in args.levels:
        path = gen_leech_data(lvl, args.seed, args.per_class)
        acc, as_normal, dist, n = eval_leech(model, path, names)
        results.append((lvl, acc, as_normal))
        # show only non-zero misclassification destinations
        dest = ", ".join(f"{k}:{v}" for k, v in dist.items() if v > 0) or "(none)"
        print(f"{lvl:>11.2f} | {n:>4} | {acc:>8.3f} | {as_normal:>9.3f} | {dest}")

    print("\nReading: accuracy should fall and 'as_normal' should rise as the leech")
    print("gets quieter (lower scale). The level where the leech starts being read as")
    print("normal is the point at which a low-rate attack becomes genuinely evasive.")

    # optional plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        lv = [r[0] for r in results]
        plt.figure(figsize=(6, 4))
        plt.plot(lv, [r[1] for r in results], "o-", label="leech accuracy")
        plt.plot(lv, [r[2] for r in results], "s--", label="misread as normal")
        plt.gca().invert_xaxis()          # quieter leech to the right
        plt.xlabel("leech intensity scale (lower = quieter attack)")
        plt.ylabel("rate")
        plt.title("Leech detection vs intensity")
        plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig("e5_leech_sweep.png", dpi=120)
        print("\nsaved e5_leech_sweep.png")
    except Exception as e:
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
