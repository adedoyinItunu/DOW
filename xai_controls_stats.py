"""
xai_controls_stats.py

Same two controls as xai_controls.py, plus the paired significance test
for the fidelity comparison.

The two fidelity measurements are PAIRED -- guided and random occlusion are
applied to the same images -- so a paired test is appropriate. Probability
drops are bounded in [-1, 1] and are not normally distributed, so the
Wilcoxon signed-rank test is used rather than a paired t-test. The t-test is
reported alongside for completeness.

Note the direction: the hypothesis being tested is whether the guided drop is
LOWER than the random drop, which is the opposite of what a fidelity metric
is supposed to show.

Usage:
    python xai_controls_stats.py --model dow_cnn_locked.pt --seed 42
    python xai_controls_stats.py --n 100 --n-random 30
"""

import argparse
import numpy as np
import torch
import torch.nn.functional as F
from captum.attr import Saliency, LayerGradCam
from scipy.stats import wilcoxon, ttest_rel
from dow_data import load_and_split
from dow_model import DoWNetCNN


def saliency_map(model, x, target):
    a = Saliency(model).attribute(x.unsqueeze(0), target=int(target))
    a = np.abs(a.squeeze().detach().numpy())
    return a / a.max() if a.max() > 0 else a


def gradcam_raw(model, x, target):
    gc = LayerGradCam(model, model.block2)
    a = gc.attribute(x.unsqueeze(0), target=int(target))
    a = np.abs(a.squeeze().detach().numpy())
    return a / a.max() if a.max() > 0 else a


def gradcam_upsampled(model, x, target):
    gc = LayerGradCam(model, model.block2)
    a = gc.attribute(x.unsqueeze(0), target=int(target))
    a = F.interpolate(a, size=(24, 30), mode="bilinear", align_corners=False)
    a = np.abs(a.squeeze().detach().numpy())
    return a / a.max() if a.max() > 0 else a


def topk_mask(m, frac=0.10):
    k = max(1, int(round(m.size * frac)))
    idx = np.argpartition(m.ravel(), -k)[-k:]
    out = np.zeros(m.size, dtype=bool)
    out[idx] = True
    return out.reshape(m.shape)


def iou(a, b):
    u = (a | b).sum()
    return (a & b).sum() / u if u else 0.0


def prob_of(model, x, cls):
    with torch.no_grad():
        p = torch.softmax(model(x.unsqueeze(0)), dim=1)
    return float(p[0, cls])


def occlude(x, mask, fill):
    y = x.clone()
    y[0][torch.tensor(mask)] = fill
    return y


def downsample(m, out_h, out_w):
    t = torch.tensor(m)[None, None]
    fh, fw = m.shape[0] // out_h, m.shape[1] // out_w
    return F.avg_pool2d(t, kernel_size=(fh, fw)).squeeze().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dow_cnn_locked.pt")
    ap.add_argument("--data", default="data.npz")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--n-random", type=int, default=20)
    ap.add_argument("--frac", type=float, default=0.10)
    ap.add_argument("--fill", default="zero", choices=["zero", "mean"],
                    help="occlusion fill value (default zero)")
    args = ap.parse_args()

    (_, _, _), (_, _, _), (Xte, yte, _), names = load_and_split(
        args.data, args.seed)

    model = DoWNetCNN(n_classes=len(names))
    state = torch.load(args.model, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()

    rng = np.random.default_rng(args.seed)
    n_img = min(args.n, len(Xte))

    guided, random_ = [], []
    native, matched = [], []
    cam_shape = None

    for i in range(n_img):
        x = torch.tensor(Xte[i])
        cls = int(yte[i])
        p0 = prob_of(model, x, cls)
        fill = 0.0 if args.fill == "zero" else float(x.mean())

        # ---- fidelity ----
        cam_up = gradcam_upsampled(model, x, cls)
        m = topk_mask(cam_up, args.frac)
        guided.append(p0 - prob_of(model, occlude(x, m, fill), cls))

        k = int(round(m.size * args.frac))
        drops = []
        for _ in range(args.n_random):
            flat = np.zeros(m.size, dtype=bool)
            flat[rng.choice(m.size, size=k, replace=False)] = True
            drops.append(p0 - prob_of(model, occlude(x, flat.reshape(m.shape),
                                                     fill), cls))
        random_.append(np.mean(drops))

        # ---- IoU ----
        sal = saliency_map(model, x, cls)
        cam_raw = gradcam_raw(model, x, cls)
        cam_shape = cam_raw.shape
        native.append(iou(topk_mask(sal, args.frac), topk_mask(cam_up, args.frac)))
        sal_ds = downsample(sal, *cam_raw.shape)
        matched.append(iou(topk_mask(sal_ds, args.frac),
                           topk_mask(cam_raw, args.frac)))

    g = np.array(guided); r = np.array(random_)
    nat = np.array(native); mat = np.array(matched)

    print("=" * 72)
    print("XAI CONTROLS WITH SIGNIFICANCE TESTING")
    print("=" * 72)
    print(f"model: {args.model}   images: {n_img}   "
          f"top fraction: {args.frac:.0%}   fill: {args.fill}\n")

    # ---------------- fidelity ----------------
    print("-" * 72)
    print("CONTROL 1: fidelity against random occlusion (paired)")
    print("-" * 72)
    print(f"  Grad-CAM guided : {g.mean():.3f} +/- {g.std(ddof=1):.3f}")
    print(f"  random          : {r.mean():.3f} +/- {r.std(ddof=1):.3f}")

    d = g - r
    print(f"  paired difference (guided - random): "
          f"{d.mean():.3f} +/- {d.std(ddof=1):.3f}")
    print(f"  images where guided > random: {(d > 0).sum()} of {n_img}")

    # Wilcoxon signed-rank, two-sided and one-sided
    try:
        w_stat, w_p2 = wilcoxon(g, r, alternative="two-sided")
        _, w_p_less = wilcoxon(g, r, alternative="less")
        print(f"\n  Wilcoxon signed-rank (two-sided): "
              f"W = {w_stat:.1f}, p = {w_p2:.2e}")
        print(f"  Wilcoxon, H1: guided < random  : p = {w_p_less:.2e}")
    except ValueError as e:
        print(f"\n  Wilcoxon could not be computed: {e}")
        w_p2 = float("nan")

    t_stat, t_p = ttest_rel(g, r)
    print(f"  paired t-test (two-sided)      : "
          f"t = {t_stat:.3f}, p = {t_p:.2e}")

    # effect size: rank-biserial for Wilcoxon
    n_nonzero = (d != 0).sum()
    if n_nonzero:
        pos = (d > 0).sum()
        rbc = 2 * (pos / n_nonzero) - 1
        print(f"  rank-biserial correlation      : {rbc:+.3f}")

    print()
    if not np.isnan(w_p2) and w_p2 < 0.05:
        if d.mean() < 0:
            print("  => The guided drop is SIGNIFICANTLY SMALLER than random.")
            print("     The attribution does not identify the regions the")
            print("     prediction depends on. Report this plainly.")
        else:
            print("  => The guided drop is significantly LARGER than random.")
            print("     The fidelity metric is supported.")
    else:
        print("  => No significant difference. The fidelity figure provides")
        print("     no evidence either way.")

    # ---------------- IoU ----------------
    exp_chance = args.frac / (2 - args.frac)
    print()
    print("-" * 72)
    print("CONTROL 2: saliency--Grad-CAM IoU at matched resolution")
    print("-" * 72)
    print(f"  Grad-CAM native resolution : {cam_shape[0]}x{cam_shape[1]}")
    print(f"  IoU at 24x30               : {nat.mean():.3f} +/- {nat.std(ddof=1):.3f}")
    print(f"  IoU at {cam_shape[0]}x{cam_shape[1]} (matched)     : "
          f"{mat.mean():.3f} +/- {mat.std(ddof=1):.3f}")
    print(f"  chance expectation         : {exp_chance:.3f}")

    try:
        _, p_iou = wilcoxon(nat, mat, alternative="two-sided")
        print(f"  Wilcoxon, native vs matched: p = {p_iou:.3f}")
    except ValueError:
        pass

    # one-sample: is matched IoU above chance?
    try:
        _, p_chance = wilcoxon(mat - exp_chance, alternative="greater")
        print(f"  Wilcoxon, matched > chance : p = {p_chance:.3f}")
    except ValueError:
        pass

    # ---------------- LaTeX ----------------
    print("\n" + "=" * 72)
    print("LaTeX")
    print("=" * 72)
    print(f"Grad-CAM top decile & ${g.mean():.3f} \\pm {g.std(ddof=1):.3f}$ \\\\")
    print(f"Random 10\\% (control) & ${r.mean():.3f} \\pm {r.std(ddof=1):.3f}$ \\\\")
    print(f"IoU, $24\\times30$ & ${nat.mean():.3f} \\pm {nat.std(ddof=1):.3f}$ \\\\")
    print(f"IoU, ${cam_shape[0]}\\times{cam_shape[1]}$ matched & "
          f"${mat.mean():.3f} \\pm {mat.std(ddof=1):.3f}$ \\\\")
    if not np.isnan(w_p2):
        print(f"\nSentence: ``a Wilcoxon signed-rank test over {n_img} paired "
              f"measurements gives $p = {w_p2:.1e}$''")


if __name__ == "__main__":
    main()
