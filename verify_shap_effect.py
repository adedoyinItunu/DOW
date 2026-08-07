"""
verify_shap_effect.py

Verifies the rank-biserial effect sizes reported for the group-level attention
analysis. A value of exactly -1.00 on every comparison means U = 0 in every case,
that is, no sample in the failure group exceeds any sample in any other group.
That is a strong claim and it should be checked directly rather than trusted.

This script recomputes the late-column attention shares, then for each comparison
reports:

  * the full distribution of each group (n, mean, sd, min, quartiles, max)
  * the raw pair counts used to form U, computed by hand rather than by library
  * the number of tied pairs, which the rank-biserial formula treats as neither
    win nor loss
  * whether the two ranges actually overlap, which is what -1.00 denies
  * scipy's U and the hand-computed U side by side, as an implementation check
  * Holm-Bonferroni adjusted p-values across the four comparisons

    python verify_shap_effect.py --method shap
    python verify_shap_effect.py --method grad-cam
    python verify_shap_effect.py --method saliency
"""
import argparse
import itertools

import numpy as np
import torch
from scipy.stats import mannwhitneyu

from dow_data import CLASS_NAMES, load_all
from dow_model import DoWNetCNN, set_seed

LATE_START = 23
NORMAL = 0


# ───────────────────────────────────────────────────────────── attribution maps
def map_gradcam(model, x, target):
    from captum.attr import LayerGradCam
    a = LayerGradCam(model, model.block2).attribute(x.unsqueeze(0), target=int(target))
    a = torch.nn.functional.interpolate(a, size=(24, 30), mode="bilinear",
                                        align_corners=False)
    return np.abs(a.squeeze().detach().numpy())


def map_saliency(model, x, target):
    from captum.attr import Saliency
    xi = x.unsqueeze(0).clone().requires_grad_(True)
    return np.abs(Saliency(model).attribute(xi, target=int(target))
                  .squeeze().detach().numpy())


def make_shap_mapper(model, background, n_classes):
    import shap
    ex = shap.DeepExplainer(model, background)

    def mapper(_m, x, target):
        vals = ex.shap_values(x.unsqueeze(0), check_additivity=False)
        if isinstance(vals, list):
            arr = vals[int(target)]
        else:
            arr = np.asarray(vals)
            if arr.shape[-1] == n_classes:
                arr = arr[..., int(target)]
            elif arr.shape[0] == n_classes:
                arr = arr[int(target)]
        return np.abs(np.squeeze(arr))
    return mapper


def late_share(m):
    tot = m.sum()
    return float(m[:, LATE_START:].sum() / tot) if tot > 0 else 0.0


# ──────────────────────────────────────────────────────────────────── statistics
def hand_U(a, b):
    """U for sample a, counted directly. Ties contribute 0.5 each."""
    wins = ties = 0
    for x in a:
        for y in b:
            if x > y:
                wins += 1
            elif x == y:
                ties += 1
    return wins + 0.5 * ties, ties


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        running = max(running, val)
        adj[idx] = min(running, 1.0)
    return adj


def summary(v):
    q1, q2, q3 = np.percentile(v, [25, 50, 75])
    return (f"n={len(v):3d}  mean={v.mean():.4f}  sd={v.std(ddof=1):.4f}  "
            f"min={v.min():.4f}  Q1={q1:.4f}  med={q2:.4f}  Q3={q3:.4f}  "
            f"max={v.max():.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dow_cnn_locked.pt")
    ap.add_argument("--config-a", default="data_configA.npz")
    ap.add_argument("--config-b", default="data_configB.npz")
    ap.add_argument("--method", default="shap",
                    choices=["shap", "grad-cam", "saliency"])
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()
    set_seed(args.seed)

    names = list(CLASS_NAMES)
    model = DoWNetCNN(n_classes=len(names))
    model.load_state_dict(torch.load(args.model, map_location="cpu"))
    model.eval()

    Xa, ya, ia, _ = load_all(args.config_a)
    Xb, yb, _, _ = load_all(args.config_b)
    with torch.no_grad():
        pa = model(torch.tensor(Xa)).argmax(1).numpy()
        pb = model(torch.tensor(Xb)).argmax(1).numpy()

    dest = np.bincount(pb[yb == NORMAL], minlength=len(names))
    fail_cls = int(dest.argmax())
    uniform = (30 - LATE_START) / 30

    print("=" * 78)
    print(f"METHOD: {args.method}   region: columns {LATE_START}-29   "
          f"uniform reference: {uniform:.4f}")
    print(f"dominant failure: normal -> {names[fail_cls]} "
          f"({dest[fail_cls]} of {(yb == NORMAL).sum()})")
    print("=" * 78)

    if args.method == "shap":
        bg = torch.tensor(Xa[np.random.default_rng(0)
                             .choice(len(Xa), 50, replace=False)])
        mapper = make_shap_mapper(model, bg, len(names))
    else:
        mapper = map_gradcam if args.method == "grad-cam" else map_saliency

    def pick(y, pred, tc, pc, k):
        return np.where((y == tc) & (pred == pc))[0][:k]

    groups = [
        (f"FAIL: B normal->{names[fail_cls]}", Xb, pick(yb, pb, NORMAL, fail_cls, args.k)),
        ("A normal (correct)",                 Xa, pick(ya, pa, NORMAL, NORMAL, args.k)),
        (f"genuine {names[fail_cls]}",         Xa, pick(ya, pa, fail_cls, fail_cls, args.k)),
        ("leech (any attack)",                 Xa, np.where((ia == 1) & (ya != NORMAL))[0][:args.k]),
        ("flood (any attack)",                 Xa, np.where((ia == 2) & (ya != NORMAL))[0][:args.k]),
    ]

    shares = {}
    for label, X, idx in groups:
        shares[label] = np.array([late_share(mapper(model, torch.tensor(X[i]), fail_cls))
                                  for i in idx])

    print("\nDISTRIBUTIONS (target class fixed to the failure class)\n" + "-" * 78)
    for label in shares:
        print(f"{label:<34}{summary(shares[label])}")

    fail = groups[0][0]
    a = shares[fail]
    others = [g[0] for g in groups[1:]]

    print(f"\nCOMPARISONS: {fail} against each group\n" + "-" * 78)
    raw_p, rows = [], []
    for label in others:
        b = shares[label]
        u_hand, n_ties = hand_U(a, b)
        u_sp, p = mannwhitneyu(a, b, alternative="two-sided")
        r = 2 * u_hand / (len(a) * len(b)) - 1
        overlap = a.max() >= b.min()
        rows.append((label, u_hand, u_sp, n_ties, r, overlap, a.max(), b.min()))
        raw_p.append(p)

    adj = holm(np.array(raw_p))
    for (label, uh, us, ties, r, ov, amax, bmin), p, pa_ in zip(rows, raw_p, adj):
        flag = "" if abs(uh - us) < 1e-9 else "   <-- SCIPY MISMATCH"
        print(f"\n  vs {label}")
        print(f"     U (hand) = {uh:>8.1f}   U (scipy) = {us:>8.1f}{flag}")
        print(f"     tied pairs = {ties} of {len(a)*len(b)}")
        print(f"     rank-biserial r = {r:+.4f}"
              + ("   (perfect separation)" if abs(r) == 1.0 else ""))
        print(f"     ranges overlap? {'YES' if ov else 'NO'}   "
              f"max(fail)={amax:.4f}  min(other)={bmin:.4f}")
        print(f"     p = {p:.3e}   Holm-adjusted = {pa_:.3e}   "
              f"{'significant' if pa_ < args.alpha else 'NOT significant'} "
              f"at alpha={args.alpha}")
        if abs(r) == 1.0 and ov:
            print("     *** INCONSISTENT: r = -1 requires no overlap. Investigate. ***")

    print("\n" + "=" * 78)
    print("A rank-biserial of exactly -1.00 is only correct when max(failure group)")
    print("is strictly below min(comparison group). Where the ranges overlap, the")
    print("effect size must be strictly greater than -1. Report the values printed")
    print("above rather than the earlier figures if they differ.")


if __name__ == "__main__":
    main()
