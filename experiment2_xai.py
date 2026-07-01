"""
experiment2_xai.py  --  EXPERIMENT 2 (XAI explanation analysis)
==============================================================
Runs the three XAI methods from the proposal on the trained CNN and computes the
three proposal metrics.

    Methods (Captum / shap):
        - gradient saliency  (captum.attr.Saliency)
        - Grad-CAM           (captum.attr.LayerGradCam on block2)
        - SHAP               (shap.DeepExplainer)
    Metrics:
        - explanation agreement : IoU of top-k pixels (saliency vs Grad-CAM)
        - SHAP consistency      : std of SHAP values for the same input over 5 runs
        - perturb-and-observe   : drop in predicted prob when top-k pixels are masked

    python experiment2_xai.py --data data.npz --model dow_cnn.pt
"""
import argparse
import numpy as np
import torch
import torch.nn.functional as F

from dow_data import load_and_split
from dow_model import DoWNetCNN, set_seed


def _norm01(a):
    a = np.abs(a)
    return a / a.max() if a.max() > 0 else a


def saliency_map(model, x, target):
    from captum.attr import Saliency
    xi = x.clone().requires_grad_(True)
    attr = Saliency(model).attribute(xi, target=int(target))
    return _norm01(attr.detach().numpy()[0, 0])


def gradcam_map(model, x, target):
    from captum.attr import LayerGradCam, LayerAttribution
    lgc = LayerGradCam(model, model.block2)
    attr = lgc.attribute(x, target=int(target), relu_attributions=True)
    attr = LayerAttribution.interpolate(attr, (24, 30))
    return _norm01(attr.detach().numpy()[0, 0])


def topk_mask(m, frac=0.10):
    k = max(1, int(frac * m.size))
    thr = np.sort(m.ravel())[::-1][k - 1]
    return m >= thr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data.npz")
    ap.add_argument("--model", default="dow_cnn.pt")
    ap.add_argument("--out", default="e2_explanations.png")
    ap.add_argument("--n-metric", type=int, default=120, help="samples for the metrics")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import shap
    import matplotlib.pyplot as plt

    set_seed(args.seed)
    (Xtr, ytr, _), _, (Xte, yte, _), names = load_and_split(args.data, args.seed)
    model = DoWNetCNN(n_classes=len(names))
    model.load_state_dict(torch.load(args.model)); model.eval()

    Xte_t = torch.tensor(Xte)
    bg = torch.tensor(Xtr[np.random.choice(len(Xtr), 80, replace=False)])

    # ---- per-class figure: input | saliency | Grad-CAM | SHAP ----------------
    explainer = shap.DeepExplainer(model, bg)
    fig, axes = plt.subplots(len(names), 4, figsize=(11, 2.5 * len(names)))
    titles = ["input", "saliency", "Grad-CAM", "SHAP |attr|"]
    for r, cname in enumerate(names):
        i = np.where(yte == r)[0][0]
        x = Xte_t[i:i + 1]
        pred = int(model(x).argmax(1))
        sal = saliency_map(model, x, pred)
        cam = gradcam_map(model, x, pred)
        sv = explainer.shap_values(x)
        sv = sv[pred] if isinstance(sv, list) else (sv[..., pred] if np.asarray(sv).ndim == 5 else sv)
        shp = _norm01(np.asarray(sv)[0, 0])
        for c, (panel, cmap) in enumerate(zip([x.numpy()[0, 0], sal, cam, shp],
                                              ["magma", "inferno", "jet", "viridis"])):
            ax = axes[r, c]
            ax.imshow(panel, origin="lower", aspect="auto", cmap=cmap)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(titles[c], fontsize=11, fontweight="bold")
            if c == 0:
                ax.set_ylabel(cname, fontsize=11, fontweight="bold")
    fig.suptitle("Experiment 2 - XAI maps per class", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(args.out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print("saved", args.out)

    # ---- METRIC 1: explanation agreement (IoU saliency vs Grad-CAM) ----------
    sel = np.random.choice(len(Xte), min(args.n_metric, len(Xte)), replace=False)
    ious = []
    for i in sel:
        x = Xte_t[i:i + 1]
        t = int(model(x).argmax(1))
        a, b = topk_mask(saliency_map(model, x, t)), topk_mask(gradcam_map(model, x, t))
        union = (a | b).sum()
        ious.append((a & b).sum() / union if union else 0.0)
    print(f"\nMETRIC 1  explanation agreement (IoU, top-10%): {np.mean(ious):.3f}")

    # ---- METRIC 2: SHAP consistency (std over 5 runs, different backgrounds) --
    stds = []
    for i in sel[:10]:
        x = Xte_t[i:i + 1]
        t = int(model(x).argmax(1))
        runs = []
        for _ in range(5):
            bg_r = torch.tensor(Xtr[np.random.choice(len(Xtr), 60, replace=False)])
            ex = shap.DeepExplainer(model, bg_r)
            sv = ex.shap_values(x)
            sv = sv[t] if isinstance(sv, list) else (sv[..., t] if np.asarray(sv).ndim == 5 else sv)
            runs.append(np.asarray(sv)[0, 0])
        stds.append(np.std(np.stack(runs), axis=0).mean())
    print(f"METRIC 2  SHAP consistency (mean std over 5 runs, lower=stabler): {np.mean(stds):.4f}")

    # ---- METRIC 3: perturb-and-observe fidelity ------------------------------
    drops = []
    with torch.no_grad():
        for i in sel:
            x = Xte_t[i:i + 1]
            t = int(model(x).argmax(1))
            p0 = F.softmax(model(x), 1)[0, t].item()
            mask = topk_mask(saliency_map(model, x, t))
            xp = x.clone(); xp[0, 0][torch.tensor(mask)] = 0.0
            p1 = F.softmax(model(xp), 1)[0, t].item()
            drops.append(p0 - p1)
    print(f"METRIC 3  perturb-and-observe fidelity (mean prob drop, higher=better): {np.mean(drops):.3f}")


if __name__ == "__main__":
    main()
