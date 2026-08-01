"""
experiment6_xai_failures.py  --  explainability of the failure mode
==================================================================
Makki's "most valuable next step": connect the robustness failure to the
explainability objective. Applies saliency and Grad-CAM to the cross-configuration
FAILURE cases -- Configuration-B normal samples that the model classifies as
linear -- and compares them against reference groups:
    (a) correctly classified Configuration-A normal samples
    (b) genuine linear attacks
    (c) low-rate leech cases
    (d) high-rate flood cases

If the explanations show the model attending to altered baseline intensity /
diurnal regions rather than genuine ramp behaviour, this demonstrates that XAI can
DIAGNOSE a failure mode, not merely visualise correct predictions.

Prereqs (generate these first):
    python dow_data.py --per-class 300 --seed 42 --out data_configA.npz
    python dow_data.py --per-class 300 --seed 42 --peak-hour 10 --width 5.0 --amp 45 \
                       --out data_configB.npz

Run:
    python experiment6_xai_failures.py --model cnn_configA.pt \
        --config-a data_configA.npz --config-b data_configB.npz
"""
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from captum.attr import Saliency, LayerGradCam

from dow_data import load_all
from dow_model import DoWNetCNN

NORMAL, LINEAR = 0, 1


def saliency_map(model, x, target):
    sal = Saliency(model)
    a = sal.attribute(x.unsqueeze(0), target=target).squeeze().detach().numpy()
    a = np.abs(a)
    return a / a.max() if a.max() > 0 else a


def gradcam_map(model, x, target):
    gc = LayerGradCam(model, model.block2)   # block2 is the named Grad-CAM layer
    a = gc.attribute(x.unsqueeze(0), target=target)
    a = torch.nn.functional.interpolate(a, size=(24, 30), mode="bilinear",
                                        align_corners=False)
    a = a.squeeze().detach().numpy()
    a = np.abs(a)
    return a / a.max() if a.max() > 0 else a


def pick(Xn, yn, pred, true_cls, pred_cls, k=8):
    """Indices where true class == true_cls and predicted == pred_cls."""
    idx = np.where((yn == true_cls) & (pred == pred_cls))[0]
    return idx[:k]


def mean_gradcam(model, X, idx, target):
    if len(idx) == 0:
        return np.zeros((24, 30))
    maps = [gradcam_map(model, torch.tensor(X[i]), target) for i in idx]
    return np.mean(maps, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="cnn_configA.pt")
    ap.add_argument("--config-a", default="data_configA.npz")
    ap.add_argument("--config-b", default="data_configB.npz")
    ap.add_argument("--out", default="e6_xai_failures.png")
    args = ap.parse_args()

    Xa, ya, ia, names = load_all(args.config_a)
    Xb, yb, ib, _ = load_all(args.config_b)

    model = DoWNetCNN(n_classes=len(names))
    model.load_state_dict(torch.load(args.model, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        pa = model(torch.tensor(Xa)).argmax(1).numpy()
        pb = model(torch.tensor(Xb)).argmax(1).numpy()

    # the four comparison groups
    fail_idx = pick(Xb, yb, pb, NORMAL, LINEAR)          # B normals called linear (FAILURE)
    oknorm_idx = pick(Xa, ya, pa, NORMAL, NORMAL)        # A normals called normal (correct)
    linear_idx = pick(Xa, ya, pa, LINEAR, LINEAR)        # genuine linear attacks
    leech_idx = np.where((ia == 1) & (ya != 0))[0][:8]   # leech (any attack class)
    flood_idx = np.where((ia == 2) & (ya != 0))[0][:8]   # flood

    print("=============== EXPERIMENT 6: XAI of the failure mode ===============")
    print(f"Config-B normal->linear failures found: {int(((yb==NORMAL)&(pb==LINEAR)).sum())}")
    print(f"comparison groups (n shown): fail={len(fail_idx)} ok-normal={len(oknorm_idx)} "
          f"linear={len(linear_idx)} leech={len(leech_idx)} flood={len(flood_idx)}\n")

    # Compare where attention concentrates. For each group, mean Grad-CAM,
    # and a simple summary: how much attention is on the LAST week (ramp region,
    # cols 23-29) vs the whole map -- a ramp detector should attend to late columns.
    groups = [
        ("B normal->linear (FAIL)", Xb, fail_idx, LINEAR),   # target = predicted class
        ("A normal (correct)",      Xa, oknorm_idx, NORMAL),
        ("genuine linear",          Xa, linear_idx, LINEAR),
        ("leech",                   Xa, leech_idx, LINEAR),
        ("flood",                   Xa, flood_idx, LINEAR),
    ]
    print(f"{'group':>26} | {'late-cols attention share':>26}")
    print("-" * 58)
    heat = {}
    for label, X, idx, tgt in groups:
        m = mean_gradcam(model, X, idx, tgt)
        heat[label] = m
        late = m[:, 23:].sum() / (m.sum() + 1e-9)         # share on last week
        print(f"{label:>26} | {late:>26.3f}")

    print("\nReading: a genuine ramp/linear detector should place MORE attention on")
    print("the late columns (the ramp region). If the FAILURE group's attention looks")
    print("like the correct-normal group (spread / baseline) rather than the genuine-")
    print("linear group (late-column), the model is reacting to baseline intensity, not")
    print("ramp behaviour -- i.e. XAI diagnoses WHY the normal class fails.")

    # figure: mean Grad-CAM per group
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(groups), figsize=(2.4 * len(groups), 3))
        for ax, (label, *_ ) in zip(axes, groups):
            ax.imshow(heat[label], origin="lower", aspect="auto", cmap="jet")
            ax.set_title(label, fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle("Mean Grad-CAM by group (diagnosing the normal->linear failure)",
                     fontsize=11, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.9])
        fig.savefig(args.out, dpi=130, bbox_inches="tight"); plt.close(fig)
        print(f"\nsaved {args.out}")
    except Exception as e:
        print(f"(figure skipped: {e})")


if __name__ == "__main__":
    main()
