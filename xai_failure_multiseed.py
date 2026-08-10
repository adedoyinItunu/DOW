"""
xai_failure_multiseed.py

The group-level attention analysis in experiment6_xai_failures.py is computed on
a single trained model. Its bootstrap intervals quantify variability across
samples within that model's attributions; they say nothing about whether a model
trained from a different seed reproduces the ordering. Because the same protocol
produces a macro-F1 spread of 0.169 across seeds, that distinction matters.

This script trains a fresh model per seed, locates the dominant failure mode for
that model, computes the late-week attention share for the five groups, and
reports whether the ordering survives. It answers one question: is the ordering a
property of the phenomenon, or of one trained model?

    python xai_failure_multiseed.py --seeds 0 1 2 3 4 --method grad-cam
    python xai_failure_multiseed.py --seeds 0 1 2 3 4 --method shap --k 25

Grad-CAM is the default because SHAP is roughly an order of magnitude slower;
run SHAP with a smaller --k once the Grad-CAM result is known.
"""
import argparse
import json

import numpy as np
import torch
from scipy.stats import mannwhitneyu

from dow_data import CLASS_NAMES, GenParams, generate_dataset, normalize, load_and_split
from dow_model import DoWNetCNN, set_seed
from architecture_controls import train_one

LATE_START = 23
NORMAL = 0
UNIFORM = (30 - LATE_START) / 30


# ─────────────────────────────────────────────────────────── attribution
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


def make_shap(model, bg, n_classes):
    import shap
    ex = shap.DeepExplainer(model, bg)

    def m(_m, x, target):
        v = ex.shap_values(x.unsqueeze(0), check_additivity=False)
        if isinstance(v, list):
            arr = v[int(target)]
        else:
            arr = np.asarray(v)
            arr = arr[..., int(target)] if arr.shape[-1] == n_classes else arr[int(target)]
        return np.abs(np.squeeze(arr))
    return m


def late_share(m):
    tot = m.sum()
    return float(m[:, LATE_START:].sum() / tot) if tot > 0 else 0.0


def rb(a, b):
    u, p = mannwhitneyu(a, b, alternative="two-sided")
    return 2 * u / (len(a) * len(b)) - 1, p


# ─────────────────────────────────────────────────────────── one seed
def run_seed(seed, args, names):
    set_seed(seed)
    (Xtr, ytr, _), (Xv, yv, _), _, _ = load_and_split(args.config_a, seed)
    model, _ = train_one(Xtr, ytr, Xv, yv, len(names), "minimal",
                         args.protocol, epochs=args.epochs)
    model.eval()

    da = np.load(args.config_a, allow_pickle=True)
    db = np.load(args.config_b, allow_pickle=True)
    Xa, ya, ia = da["X"], da["y"], da["intensity"]
    Xb, yb = db["X"], db["y"]
    Xan, Xbn = normalize(Xa), normalize(Xb)
    with torch.no_grad():
        pa = model(torch.tensor(Xan)).argmax(1).numpy()
        pb = model(torch.tensor(Xbn)).argmax(1).numpy()

    dest = np.bincount(pb[yb == NORMAL], minlength=len(names))
    fail_cls = int(dest.argmax())
    n_fail = int(dest[fail_cls])

    if args.method == "shap":
        bg = torch.tensor(Xan[np.random.default_rng(seed).choice(len(Xan), 50, False)])
        mapper = make_shap(model, bg, len(names))
    else:
        mapper = map_gradcam if args.method == "grad-cam" else map_saliency

    def pick(y, pred, tc, pc):
        return np.where((y == tc) & (pred == pc))[0][:args.k]

    spec = [
        ("failure",     Xbn, pick(yb, pb, NORMAL, fail_cls)),
        ("normal-ok",   Xan, pick(ya, pa, NORMAL, NORMAL)),
        ("genuine",     Xan, pick(ya, pa, fail_cls, fail_cls)),
        ("leech",       Xan, np.where((ia == 1) & (ya != NORMAL))[0][:args.k]),
        ("flood",       Xan, np.where((ia == 2) & (ya != NORMAL))[0][:args.k]),
    ]
    shares = {}
    for label, X, idx in spec:
        if len(idx) == 0:
            shares[label] = np.array([np.nan]); continue
        shares[label] = np.array([late_share(mapper(model, torch.tensor(X[i]), fail_cls))
                                  for i in idx])
    return fail_cls, n_fail, shares


# ─────────────────────────────────────────────────────────── main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-a", default="data_configA.npz")
    ap.add_argument("--config-b", default="data_configB.npz")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--method", default="grad-cam",
                    choices=["grad-cam", "saliency", "shap"])
    ap.add_argument("--protocol", default="converged",
                    choices=["fixed25", "fixed25_sel", "converged"])
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    names = list(CLASS_NAMES)
    labels = ["failure", "normal-ok", "genuine", "leech", "flood"]

    print("=" * 76)
    print(f"MULTI-SEED GROUP ATTENTION   method={args.method}  "
          f"protocol={args.protocol}  n per group={args.k}")
    print(f"uniform reference = {UNIFORM:.4f}")
    print("=" * 76)

    per_seed, orderings, rows = [], [], []
    for s in args.seeds:
        fail_cls, n_fail, sh = run_seed(s, args, names)
        means = {k: float(np.nanmean(v)) for k, v in sh.items()}
        order = sorted(means, key=lambda k: -means[k])
        lowest = order[-1] == "failure"
        orderings.append(tuple(order))
        per_seed.append({"seed": s, "failure_class": names[fail_cls],
                         "n_failed": n_fail, "means": means,
                         "failure_is_lowest": lowest})
        eff = {k: rb(sh["failure"], sh[k])[0] for k in labels[1:]}
        rows.append((s, names[fail_cls], n_fail, means, lowest, eff))

        print(f"\nseed {s}: normal -> {names[fail_cls]} ({n_fail} of 300)")
        print("   " + "  ".join(f"{k}={means[k]:.3f}" for k in labels))
        print(f"   failure group lowest? {'YES' if lowest else 'NO'}   "
              f"vs uniform: {means['failure'] - UNIFORM:+.4f}")
        print("   effect sizes vs failure: " +
              "  ".join(f"{k} r={eff[k]:+.2f}" for k in labels[1:]))

    # ---- aggregate
    n_low = sum(r[4] for r in rows)
    same_order = len(set(orderings)) == 1
    print("\n" + "=" * 76)
    print("AGGREGATE")
    print("-" * 76)
    print(f"  failure group ranked lowest on {n_low} of {len(rows)} seeds")
    print(f"  identical full ordering on all seeds? {'YES' if same_order else 'NO'}")
    if not same_order:
        for o in sorted(set(orderings)):
            c = orderings.count(o)
            print(f"     {c}x  {' > '.join(o)}")
    for k in labels:
        v = np.array([r[3][k] for r in rows])
        print(f"  {k:<10} across seeds: {v.mean():.4f} +/- {v.std(ddof=1):.4f}"
              f"  (min {v.min():.4f}, max {v.max():.4f})")
    for k in labels[1:]:
        e = np.array([r[5][k] for r in rows])
        print(f"  effect vs {k:<10} r = {e.mean():+.3f} +/- {e.std(ddof=1):.3f}"
              f"   all negative? {'YES' if (e < 0).all() else 'NO'}")

    print("\n" + "=" * 76)
    print("READING: if the failure group ranks lowest on every seed and every")
    print("effect size is negative, the ordering is a property of the phenomenon")
    print("rather than of one trained model, and the single-seed result in the")
    print("report can be replaced by these mean +/- sd figures. If it does not")
    print("hold, report the proportion of seeds on which it does and withdraw the")
    print("stronger claim.")

    if args.out:
        json.dump({"method": args.method, "protocol": args.protocol,
                   "k": args.k, "uniform": UNIFORM, "per_seed": per_seed},
                  open(args.out, "w"), indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
