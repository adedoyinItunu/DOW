"""
xai_failure_stats.py  --  uncertainty and cross-method checks for the research report
==============================================================================
experiment6_xai_failures.py reports group means and standard deviations for the
late-column attention share. That supports an ordering but not an inference, and
it leaves two things unstated that a reader needs:

  * WHICH TARGET CLASS the Grad-CAM map was computed for. In the existing script
    the target varies by group: the failure group and the attack groups use the
    failure class, while the correctly classified Config-A normals use the normal
    class. Because global average pooling makes Grad-CAM equivalent to CAM on
    this network (the research report), the map for class c is a fixed weighting
    w_c of the channel activations, so a different target gives a genuinely
    different map and the two are not obviously comparable. This script reports
    BOTH conventions: 'predicted' (as originally run) and 'fixed' (the failure
    class for every group), so the reader can see whether the ordering depends on
    the choice.

  * UNCERTAINTY. Group means are reported with a bootstrap confidence interval,
    and the failure group is compared against every other group with a
    Mann-Whitney U test and a rank-biserial effect size. Mann-Whitney is used
    rather than a t-test because the shares are bounded in [0,1] and there is no
    reason to assume normality at n = 50.

It then repeats the whole analysis with SALIENCY and SHAP in place of Grad-CAM.
Agreement across methods strengthens the interpretation; disagreement is itself a
reportable explainability result, and the research report already establishes that
saliency and Grad-CAM agree at close to chance on individual maps. Whether they
agree at GROUP level on this ordering is a different question and is the one this
script answers.

    python xai_failure_stats.py --k 50 | tee out_xai_failure_stats.txt
"""
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import mannwhitneyu

from dow_data import load_all, CLASS_NAMES
from dow_model import DoWNetCNN, set_seed

LATE_START = 23          # columns 23-29 inclusive: final week of the month
NORMAL = 0
N_BOOT = 10000


# ---------------------------------------------------------------- attributions
def map_gradcam(model, x, target):
    from captum.attr import LayerGradCam
    a = LayerGradCam(model, model.block2).attribute(x.unsqueeze(0), target=target)
    a = F.interpolate(a, size=(24, 30), mode="bilinear", align_corners=False)
    return a.squeeze().detach().numpy()


def map_saliency(model, x, target):
    from captum.attr import Saliency
    xi = x.unsqueeze(0).clone().requires_grad_(True)
    return Saliency(model).attribute(xi, target=target).squeeze().detach().numpy()


def make_shap_mapper(model, background, n_classes):
    """SHAP returns the class axis first or last depending on version."""
    import shap
    explainer = shap.DeepExplainer(model, background)

    def mapper(_model, x, target):
        vals = explainer.shap_values(x.unsqueeze(0), check_additivity=False)
        if isinstance(vals, list):                  # older SHAP: one array per class
            arr = vals[target]
        else:
            arr = np.asarray(vals)
            if arr.shape[-1] == n_classes:          # (batch, C, H, W, classes)
                arr = arr[..., target]
            elif arr.shape[0] == n_classes:         # (classes, batch, C, H, W)
                arr = arr[target]
        return np.abs(np.squeeze(arr))
    return mapper


METHODS = {"grad-cam": map_gradcam, "saliency": map_saliency}


def late_share(m):
    m = np.abs(m)
    total = m.sum()
    return float(m[:, LATE_START:].sum() / total) if total > 0 else 0.0


# ------------------------------------------------------------------ statistics
def boot_ci(v, n=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n, len(v)), replace=True).mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def rank_biserial(a, b):
    """Effect size for Mann-Whitney: +1 means every a exceeds every b."""
    u, _ = mannwhitneyu(a, b, alternative="two-sided")
    return 2 * u / (len(a) * len(b)) - 1


def pick(y, pred, true_cls, pred_cls, k):
    return np.where((y == true_cls) & (pred == pred_cls))[0][:k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dow_cnn_locked.pt")
    ap.add_argument("--config-a", default="data_configA.npz")
    ap.add_argument("--config-b", default="data_configB.npz")
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-shap", action="store_true",
                    help="skip SHAP (slowest of the three)")
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

    # dominant destination of the shifted normal samples
    dest = np.bincount(pb[yb == NORMAL], minlength=len(names))
    fail_cls = int(dest.argmax())
    print(f"late columns: {LATE_START}-29 (final week)   n per group: {args.k}")
    print(f"dominant failure: normal -> {names[fail_cls]} "
          f"({dest[fail_cls]} of {(yb == NORMAL).sum()})")
    uniform = (30 - LATE_START) / 30
    print(f"uniform-attention reference: {uniform:.3f} "
          f"({30 - LATE_START} of 30 columns carry no preference)\n")

    groups = [
        (f"B normal->{names[fail_cls]} (FAIL)", Xb, pick(yb, pb, NORMAL, fail_cls, args.k), fail_cls),
        ("A normal (correct)",                  Xa, pick(ya, pa, NORMAL, NORMAL, args.k),   NORMAL),
        (f"genuine {names[fail_cls]}",          Xa, pick(ya, pa, fail_cls, fail_cls, args.k), fail_cls),
        ("leech (any attack)",                  Xa, np.where((ia == 1) & (ya != NORMAL))[0][:args.k], fail_cls),
        ("flood (any attack)",                  Xa, np.where((ia == 2) & (ya != NORMAL))[0][:args.k], fail_cls),
    ]

    methods = dict(METHODS)
    if not args.no_shap:
        bg = torch.tensor(Xa[np.random.default_rng(0).choice(len(Xa), 50, False)])
        methods["shap"] = make_shap_mapper(model, bg, len(names))

    for method, mapper in methods.items():
        for convention in ("predicted", "fixed"):
            print("=" * 76)
            print(f"METHOD: {method}    TARGET CLASS: {convention}"
                  + (f" (= {names[fail_cls]} for every group)" if convention == "fixed"
                     else " (per-group, as originally run)"))
            print("=" * 76)

            shares = {}
            for label, X, idx, tgt in groups:
                target = fail_cls if convention == "fixed" else tgt
                v = np.array([late_share(mapper(model, torch.tensor(X[i]), target))
                              for i in idx])
                shares[label] = v

            order = sorted(shares, key=lambda k: -shares[k].mean())
            print(f"{'group':>32} | {'mean':>6} | {'std':>5} | {'95% CI':>16}")
            print("-" * 76)
            for label in order:
                v = shares[label]
                lo, hi = boot_ci(v)
                print(f"{label:>32} | {v.mean():>6.3f} | {v.std():>5.3f} | "
                      f"[{lo:.3f}, {hi:.3f}]")

            fail_label = groups[0][0]
            print(f"\n  Mann-Whitney U, {fail_label} vs each group:")
            for label in order:
                if label == fail_label:
                    continue
                u, p = mannwhitneyu(shares[fail_label], shares[label],
                                    alternative="two-sided")
                r = rank_biserial(shares[fail_label], shares[label])
                flag = "" if p < 0.05 else "   (not significant)"
                print(f"    vs {label:<30} U={u:>7.0f}  p={p:<9.2e} "
                      f"rank-biserial={r:+.2f}{flag}")
            print()

    print("=" * 76)
    print("READ THIS AS: the claim is about the RELATIVE ORDERING of group means,")
    print("not about which cells drive any individual prediction. If the ordering")
    print("holds under both target conventions and across all three attribution")
    print("methods, the interpretation in the research report is supported independently")
    print("of the choice of method. If it does not, report the disagreement: it is")
    print("a meaningful explainability result either way (the research report).")


if __name__ == "__main__":
    main()
