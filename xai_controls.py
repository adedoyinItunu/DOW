"""
xai_controls.py

Two controls on the per-sample attribution results.

CONTROL 1 -- random-perturbation fidelity baseline
    Perturbation fidelity is measured by occluding the top 10% of pixels
    an attribution method highlights, measure the drop in predicted class
    probability. Without a control that number is uninterpretable. On dense
    heat-maps where every cell carries traffic, perturbing ANY 10% of pixels
    may produce a comparable drop.

    This measures the drop from occluding a RANDOM 10%, averaged over
    several draws, so the attribution-guided figure has something to beat.

CONTROL 2 -- downsampled-saliency IoU
    the research report argues that a twofold resolution difference cannot account
    for saliency and Grad-CAM agreeing at chance level. That is asserted,
    not tested. Grad-CAM operates on block2's 12x15 feature map; saliency is
    per-pixel on 24x30.

    This average-pools saliency to 12x15, takes the top 10% at that
    resolution, and recomputes IoU against Grad-CAM at the same resolution.
    If agreement stays near chance, the assertion is proved. If it rises
    substantially, the resolution explanation was right and the reported claim does not hold.

Usage:
    python xai_controls.py --model dow_cnn_locked.pt --seed 42
    python xai_controls.py --n 60 --n-random 20
"""

import argparse
import numpy as np
import torch
import torch.nn.functional as F
from captum.attr import Saliency, LayerGradCam
from dow_data import load_and_split
from dow_model import DoWNetCNN


# ------------------------------------------------------------------
# attribution maps
# ------------------------------------------------------------------

def saliency_map(model, x, target):
    a = Saliency(model).attribute(x.unsqueeze(0), target=int(target))
    a = np.abs(a.squeeze().detach().numpy())
    return a / a.max() if a.max() > 0 else a


def gradcam_raw(model, x, target):
    """Grad-CAM at its native block2 resolution (no upsampling)."""
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
    flat = m.ravel()
    idx = np.argpartition(flat, -k)[-k:]
    out = np.zeros(m.size, dtype=bool)
    out[idx] = True
    return out.reshape(m.shape)


def iou(a, b):
    inter = (a & b).sum()
    union = (a | b).sum()
    return inter / union if union else 0.0


# ------------------------------------------------------------------
# CONTROL 1 -- fidelity with a random baseline
# ------------------------------------------------------------------

def prob_of(model, x, cls):
    with torch.no_grad():
        p = torch.softmax(model(x.unsqueeze(0)), dim=1)
    return float(p[0, cls])


def occlude(x, mask, fill=0.0):
    y = x.clone()
    y[0][torch.tensor(mask)] = fill
    return y


def fidelity_control(model, X, y, n, n_random, frac, seed):
    rng = np.random.default_rng(seed)
    guided, random_ = [], []

    for i in range(min(n, len(X))):
        x = torch.tensor(X[i])
        cls = int(y[i])
        p0 = prob_of(model, x, cls)

        # attribution-guided occlusion (Grad-CAM, upsampled)
        cam = gradcam_upsampled(model, x, cls)
        m = topk_mask(cam, frac)
        guided.append(p0 - prob_of(model, occlude(x, m), cls))

        # random occlusion, averaged over several draws
        drops = []
        k = int(round(m.size * frac))
        for _ in range(n_random):
            flat = np.zeros(m.size, dtype=bool)
            flat[rng.choice(m.size, size=k, replace=False)] = True
            drops.append(p0 - prob_of(model, occlude(x, flat.reshape(m.shape)), cls))
        random_.append(np.mean(drops))

    return np.array(guided), np.array(random_)


# ------------------------------------------------------------------
# CONTROL 2 -- IoU at matched resolution
# ------------------------------------------------------------------

def downsample(m, out_h, out_w):
    """Average-pool a 24x30 map down to the Grad-CAM resolution."""
    t = torch.tensor(m)[None, None]
    fh, fw = m.shape[0] // out_h, m.shape[1] // out_w
    pooled = F.avg_pool2d(t, kernel_size=(fh, fw))
    return pooled.squeeze().numpy()


def iou_control(model, X, y, n, frac):
    native, matched = [], []

    for i in range(min(n, len(X))):
        x = torch.tensor(X[i])
        cls = int(y[i])

        sal = saliency_map(model, x, cls)
        cam_up = gradcam_upsampled(model, x, cls)
        cam_raw = gradcam_raw(model, x, cls)

        # as currently reported: both at 24x30
        native.append(iou(topk_mask(sal, frac), topk_mask(cam_up, frac)))

        # matched resolution: saliency pooled down to block2's grid
        h, w = cam_raw.shape
        sal_ds = downsample(sal, h, w)
        matched.append(iou(topk_mask(sal_ds, frac), topk_mask(cam_raw, frac)))

    return np.array(native), np.array(matched), cam_raw.shape


# ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dow_cnn_locked.pt")
    ap.add_argument("--data", default="data.npz")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=50, help="test images to use")
    ap.add_argument("--n-random", type=int, default=20,
                    help="random occlusion draws per image")
    ap.add_argument("--frac", type=float, default=0.10, help="top fraction")
    args = ap.parse_args()

    (_, _, _), (_, _, _), (Xte, yte, _), names = load_and_split(
        args.data, args.seed)

    model = DoWNetCNN(n_classes=len(names))
    state = torch.load(args.model, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()

    print("=" * 70)
    print("XAI CONTROLS")
    print("=" * 70)
    print(f"model: {args.model}   images: {min(args.n, len(Xte))}   "
          f"top fraction: {args.frac:.0%}\n")

    # ---- Control 1 ----
    print("-" * 70)
    print("CONTROL 1: fidelity against a random-occlusion baseline")
    print("-" * 70)
    g, r = fidelity_control(model, Xte, yte, args.n, args.n_random,
                            args.frac, args.seed)
    print(f"  Grad-CAM guided occlusion : {g.mean():.3f} +/- {g.std():.3f}")
    print(f"  random occlusion          : {r.mean():.3f} +/- {r.std():.3f}")
    print(f"  difference                : {g.mean() - r.mean():+.3f}")
    ratio = g.mean() / r.mean() if r.mean() > 1e-6 else float("inf")
    print(f"  ratio                     : {ratio:.2f}x")

    if g.mean() > r.mean() + 2 * r.std() / np.sqrt(len(r)):
        print("\n  => The guided drop clearly exceeds the random baseline.")
        print("     The fidelity figure reflects the attribution, not merely")
        print("     the removal of information. Report both numbers.")
    else:
        print("\n  => The guided drop is NOT clearly above random. The fidelity")
        print("     figure is not evidence of a faithful explanation. Report")
        print("     both and state the limitation plainly.")

    # ---- Control 2 ----
    print()
    print("-" * 70)
    print("CONTROL 2: saliency-Grad-CAM IoU at matched resolution")
    print("-" * 70)
    nat, mat, cam_shape = iou_control(model, Xte, yte, args.n, args.frac)
    chance = args.frac   # expected IoU under independent selection ~ frac/(2-frac)
    exp_chance = args.frac / (2 - args.frac)

    print(f"  Grad-CAM native resolution : {cam_shape[0]}x{cam_shape[1]}")
    print(f"  IoU at 24x30 (as reported) : {nat.mean():.3f} +/- {nat.std():.3f}")
    print(f"  IoU at {cam_shape[0]}x{cam_shape[1]} (matched)      : "
          f"{mat.mean():.3f} +/- {mat.std():.3f}")
    print(f"  expected under chance      : {exp_chance:.3f}")

    if mat.mean() > nat.mean() + 0.10:
        print("\n  => Agreement rises substantially at matched resolution.")
        print("     The resolution difference DOES explain much of the")
        print("     disagreement. the reported claim does not hold: the claim")
        print("     that resolution cannot account for it is wrong.")
    else:
        print("\n  => Agreement does not rise at matched resolution.")
        print("     The resolution difference does NOT explain the")
        print("     disagreement, which supports the current the research report")
        print("     claim -- now tested rather than asserted.")

    # ---- LaTeX ----
    print("\n" + "=" * 70)
    print("LaTeX rows")
    print("=" * 70)
    print(f"Grad-CAM guided occlusion & ${g.mean():.3f} \\pm {g.std():.3f}$ \\\\")
    print(f"Random occlusion (control) & ${r.mean():.3f} \\pm {r.std():.3f}$ \\\\")
    print(f"IoU, $24\\times30$ & ${nat.mean():.3f} \\pm {nat.std():.3f}$ \\\\")
    print(f"IoU, ${cam_shape[0]}\\times{cam_shape[1]}$ matched & "
          f"${mat.mean():.3f} \\pm {mat.std():.3f}$ \\\\")


if __name__ == "__main__":
    main()
