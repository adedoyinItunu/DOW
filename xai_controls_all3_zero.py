#!/usr/bin/env python3
"""
Q7 check. Extends the committed perturbation-fidelity control from Grad-CAM
alone to all three methods, and runs the whole thing under BOTH target
conventions: the true label (as committed) and the predicted class (as the
report describes).

Settings match xai_controls_stats.py: n=50 test images, top decile, mean fill,
20 random draws per image, locked checkpoint, seed 42.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import wilcoxon
from dow_data import load_and_split
from dow_model import DoWNetCNN
from xai_failure_stats import map_gradcam, map_saliency, make_shap_mapper

N_IMG, FRAC, N_RANDOM, SEED = 50, 0.10, 20, 42

(_, _, _), (_, _, _), (Xte, yte, _), names = load_and_split("data.npz", SEED)
model = DoWNetCNN(n_classes=len(names))
st = torch.load("dow_cnn_locked.pt", map_location="cpu")
if isinstance(st, dict) and "model_state_dict" in st:
    st = st["model_state_dict"]
model.load_state_dict(st)
model.eval()

# SHAP background: 50 samples from the reference configuration, as in the study
rng_bg = np.random.default_rng(0)
Xall = np.load("data.npz")["X"]
from dow_data import normalize
Xall = np.asarray(normalize(Xall))
if Xall.ndim == 3:
    Xall = Xall[:, None]
bg_idx = rng_bg.choice(len(Xall), size=50, replace=False)
background = torch.tensor(Xall[bg_idx], dtype=torch.float32)
shap_map = make_shap_mapper(model, background, len(names))

METHODS = {"saliency": map_saliency, "grad-cam": map_gradcam, "shap": shap_map}


def topk_mask(m, frac=FRAC):
    m = np.abs(m)
    k = max(1, int(round(m.size * frac)))
    idx = np.argpartition(m.ravel(), -k)[-k:]
    out = np.zeros(m.size, dtype=bool)
    out[idx] = True
    return out.reshape(m.shape)


def prob_of(x, cls):
    with torch.no_grad():
        return float(torch.softmax(model(x.unsqueeze(0)), 1)[0, cls])


def occlude(x, mask, fill):
    y = x.clone()
    y[0][torch.tensor(mask)] = fill
    return y


def iou(a, b):
    u = (a | b).sum()
    return (a & b).sum() / u if u else 0.0


with torch.no_grad():
    preds = model(torch.tensor(Xte[:N_IMG])).argmax(1).numpy()
truth = np.asarray(yte[:N_IMG])
print(f"predicted == true on {(preds == truth).sum()}/{N_IMG} control images "
      f"({(preds == truth).mean():.0%})\n")

for conv in ("true label (as committed)", "predicted class (as reported)"):
    targets = truth if conv.startswith("true") else preds
    print("=" * 68)
    print(f"TARGET CONVENTION: {conv}  [ZERO FILL]")
    print("=" * 68)

    rng = np.random.default_rng(SEED)
    guided = {k: [] for k in METHODS}
    random_ = []
    masks_store = {k: [] for k in METHODS}

    for i in range(N_IMG):
        x = torch.tensor(Xte[i])
        cls = int(targets[i])
        p0 = prob_of(x, cls)
        fill = 0.0

        for name, fn in METHODS.items():
            m = topk_mask(fn(model, x, cls))
            masks_store[name].append(m)
            guided[name].append(p0 - prob_of(occlude(x, m, fill), cls))

        k = int(round(720 * FRAC))
        drops = []
        for _ in range(N_RANDOM):
            flat = np.zeros(720, dtype=bool)
            flat[rng.choice(720, size=k, replace=False)] = True
            drops.append(p0 - prob_of(occlude(x, flat.reshape(24, 30), fill), cls))
        random_.append(np.mean(drops))

    r = np.array(random_)
    print(f"  random-occlusion control : {r.mean():.3f} +/- {r.std(ddof=1):.3f}\n")
    print(f"  {'method':<10} {'guided drop':>16} {'g>r':>8} {'Wilcoxon p':>12}  verdict")
    for name in METHODS:
        g = np.array(guided[name])
        d = g - r
        try:
            _, p = wilcoxon(g, r, alternative="two-sided")
        except ValueError:
            p = float("nan")
        verdict = ("beats random" if (p < 0.05 and d.mean() > 0)
                   else "WORSE than random" if (p < 0.05 and d.mean() < 0)
                   else "no evidence")
        print(f"  {name:<10} {g.mean():.3f} +/- {g.std(ddof=1):.3f} "
              f"{(d>0).sum():>5}/{N_IMG} {p:>12.3f}  {verdict}")

    print(f"\n  pairwise agreement (IoU, top decile, chance = 0.053)")
    keys = list(METHODS)
    for a in range(len(keys)):
        for b in range(a + 1, len(keys)):
            v = [iou(masks_store[keys[a]][i], masks_store[keys[b]][i])
                 for i in range(N_IMG)]
            v = np.array(v)
            print(f"    {keys[a]:>9} vs {keys[b]:<9} {v.mean():.3f} +/- {v.std(ddof=1):.3f}")
    print()
