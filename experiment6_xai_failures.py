"""
experiment6_xai_failures.py  --  explainability of the failure mode
==================================================================
REWRITTEN. Changes from the previous version:

1. Targets the DOMINANT failure mode rather than a hardcoded one.
   On the locked checkpoint, Config-B normals are misclassified as
   `geometric` 273 times and as `linear` only 27 times. The old script
   analysed the 27 and ignored the 273.

2. Uses all available samples per group (capped by --k, default 50)
   instead of a fixed 8, so the comparison has statistical weight.

3. Reports mean +/- standard deviation per group, and n, which the
   previous version did not - a specific criticism raised in review.

4. States the late-column definition explicitly in the output so it can
   be quoted in the thesis.

Run:
    python experiment6_xai_failures.py --model dow_cnn_locked.pt \
        --config-a data_configA.npz --config-b data_configB.npz
"""

import argparse
import numpy as np
import torch
from captum.attr import LayerGradCam
from dow_data import load_all
from dow_model import DoWNetCNN

NORMAL, LINEAR, GEOMETRIC, RANDOM = 0, 1, 2, 3
LATE_START = 23          # columns 23-29 inclusive = final week of the month


def gradcam_map(model, x, target):
    gc = LayerGradCam(model, model.block2)
    a = gc.attribute(x.unsqueeze(0), target=target)
    a = torch.nn.functional.interpolate(a, size=(24, 30), mode="bilinear",
                                        align_corners=False)
    a = np.abs(a.squeeze().detach().numpy())
    return a / a.max() if a.max() > 0 else a


def late_share(m):
    """Proportion of Grad-CAM attention falling in the final week."""
    return m[:, LATE_START:].sum() / (m.sum() + 1e-9)


def group_stats(model, X, idx, target):
    """Per-sample late-column shares, so we can report mean AND std."""
    if len(idx) == 0:
        return np.array([]), np.zeros((24, 30))
    maps = [gradcam_map(model, torch.tensor(X[i]), target) for i in idx]
    shares = np.array([late_share(m) for m in maps])
    return shares, np.mean(maps, axis=0)


def pick(y, pred, true_cls, pred_cls, k):
    idx = np.where((y == true_cls) & (pred == pred_cls))[0]
    return idx[:k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dow_cnn_locked.pt")
    ap.add_argument("--config-a", default="data_configA.npz")
    ap.add_argument("--config-b", default="data_configB.npz")
    ap.add_argument("--out", default="e6_xai_failures.png")
    ap.add_argument("--k", type=int, default=50,
                    help="max samples per group (default 50)")
    args = ap.parse_args()

    Xa, ya, ia, names = load_all(args.config_a)
    Xb, yb, ib, _ = load_all(args.config_b)

    model = DoWNetCNN(n_classes=len(names))
    state = torch.load(args.model, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        pa = model(torch.tensor(Xa)).argmax(1).numpy()
        pb = model(torch.tensor(Xb)).argmax(1).numpy()

    # ---- identify the DOMINANT failure mode, do not assume it ----
    print("=" * 66)
    print("EXPERIMENT 6: XAI of the failure mode")
    print("=" * 66)
    print(f"model: {args.model}")
    print(f"late columns defined as {LATE_START}-29 (final week of the month)\n")

    print("Config-B normal-class misclassification breakdown:")
    norm_mask = yb == NORMAL
    counts = {}
    for c in range(len(names)):
        n = int(((pb == c) & norm_mask).sum())
        counts[c] = n
        marker = "  <-- dominant" if n == max(
            [int(((pb == k) & norm_mask).sum()) for k in range(len(names))
             if k != NORMAL]) and c != NORMAL else ""
        print(f"  normal -> {names[c]:10s} : {n:4d}{marker}")

    fail_cls = max((c for c in counts if c != NORMAL), key=lambda c: counts[c])
    print(f"\nAnalysing the dominant mode: normal -> {names[fail_cls]} "
          f"({counts[fail_cls]} samples)\n")

    # ---- comparison groups ----
    fail_idx = pick(yb, pb, NORMAL, fail_cls, args.k)
    oknorm_idx = pick(ya, pa, NORMAL, NORMAL, args.k)
    genuine_idx = pick(ya, pa, fail_cls, fail_cls, args.k)
    leech_idx = np.where((ia == 1) & (ya != NORMAL))[0][:args.k]
    flood_idx = np.where((ia == 2) & (ya != NORMAL))[0][:args.k]

    groups = [
        (f"B normal->{names[fail_cls]} (FAIL)", Xb, fail_idx, fail_cls),
        ("A normal (correct)",                  Xa, oknorm_idx, NORMAL),
        (f"genuine {names[fail_cls]}",          Xa, genuine_idx, fail_cls),
        ("leech (any attack)",                  Xa, leech_idx, fail_cls),
        ("flood (any attack)",                  Xa, flood_idx, fail_cls),
    ]

    print(f"{'group':>30} | {'n':>4} | {'late-col share (mean +/- std)':>30}")
    print("-" * 72)

    heat, table = {}, []
    for label, X, idx, tgt in groups:
        shares, mean_map = group_stats(model, X, idx, tgt)
        heat[label] = mean_map
        if len(shares) == 0:
            print(f"{label:>30} | {0:>4} | {'(no samples)':>30}")
            continue
        m, s = shares.mean(), shares.std()
        table.append((label, len(shares), m, s))
        print(f"{label:>30} | {len(shares):>4} | {m:>17.3f} +/- {s:.3f}")

    # ---- LaTeX rows ----
    print("\n--- LaTeX table rows ---")
    for label, n, m, s in table:
        safe = label.replace("_", r"\_").replace("->", r"$\rightarrow$")
        print(f"{safe} & {n} & ${m:.3f} \\pm {s:.3f}$ \\\\")

    print("\nREADING")
    print("  If the FAILURE group's late-column attention resembles the")
    print("  correct-normal group rather than the genuine-attack group, the")
    print("  model is reacting to baseline intensity rather than to the")
    print("  temporal structure that defines the attack class.")
    print("  Check whether the standard deviations overlap before claiming a")
    print("  difference - with n around 50 these are meaningful.")

    # ---- figure ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(groups), figsize=(2.4 * len(groups), 3))
        for ax, (label, *_) in zip(axes, groups):
            ax.imshow(heat[label], origin="lower", aspect="auto", cmap="jet")
            ax.set_title(label, fontsize=7)
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f"Mean Grad-CAM by group "
                     f"(diagnosing the normal->{names[fail_cls]} failure)",
                     fontsize=11, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.9])
        fig.savefig(args.out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"\nsaved {args.out}")
    except Exception as e:
        print(f"(figure skipped: {e})")


if __name__ == "__main__":
    main()
