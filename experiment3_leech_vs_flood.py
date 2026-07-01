"""
experiment3_leech_vs_flood.py  --  EXPERIMENT 3 (leech vs flood contrast)
========================================================================
Compares how the trained CNN handles low-rate "leech" attacks vs high-rate
"flood" attacks, and contrasts their explanation maps.

    python experiment3_leech_vs_flood.py --data data.npz --model dow_cnn.pt
"""
import argparse
import numpy as np
import torch

from dow_data import load_and_split
from dow_model import DoWNetCNN, set_seed
from experiment2_xai import gradcam_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data.npz")
    ap.add_argument("--model", default="dow_cnn.pt")
    ap.add_argument("--out", default="e3_leech_vs_flood.png")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import matplotlib.pyplot as plt

    set_seed(args.seed)
    _, _, (Xte, yte, ite), names = load_and_split(args.data, args.seed)
    model = DoWNetCNN(n_classes=len(names))
    model.load_state_dict(torch.load(args.model)); model.eval()

    with torch.no_grad():
        pred = model(torch.tensor(Xte)).argmax(1).numpy()

    def report(mask, label):
        acc = (pred[mask] == yte[mask]).mean()
        as_normal = (pred[mask] == 0).mean()       # attack misread as normal
        print(f"{label:6s}  n={mask.sum():3d}  accuracy={acc:.3f}  "
              f"misread-as-normal={as_normal:.3f}")
        return acc, as_normal

    leech = ite == 1     # low-rate attacks
    flood = ite == 2     # high-rate attacks
    print("\n=============== EXPERIMENT 3: leech vs flood ===============")
    la, ln = report(leech, "leech")
    fa, fn = report(flood, "flood")
    print(f"\nGap: flood is {fa - la:+.3f} more accurate than leech; "
          f"leech is hidden in normal {ln - fn:+.3f} more often.")
    print("Interpretation: the low-rate leech is the harder case -- it looks like "
          "normal traffic, which is the whole motivation for the XAI analysis.")

    # ---- explanation contrast: a leech vs a flood example of the same class --
    cls = 1  # linear (the canonical ramp/leech)
    li = np.where((yte == cls) & (ite == 1))[0][0]
    fi = np.where((yte == cls) & (ite == 2))[0][0]
    fig, axes = plt.subplots(2, 2, figsize=(7, 6))
    for col, (idx, tag) in enumerate([(li, "leech (low-rate)"), (fi, "flood (high-rate)")]):
        x = torch.tensor(Xte[idx:idx + 1])
        t = int(model(x).argmax(1))
        axes[0, col].imshow(Xte[idx, 0], origin="lower", aspect="auto", cmap="magma")
        axes[0, col].set_title(f"{names[cls]} {tag}", fontsize=10, fontweight="bold")
        axes[1, col].imshow(gradcam_map(model, x, t), origin="lower", aspect="auto", cmap="jet")
        axes[1, col].set_title("Grad-CAM", fontsize=10)
        for r in range(2):
            axes[r, col].set_xticks([]); axes[r, col].set_yticks([])
    fig.suptitle("Experiment 3 - leech is diffuse/weak, flood is concentrated/strong",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(args.out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
