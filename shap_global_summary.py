"""
shap_global_summary.py  --  ITEM 8: aggregate SHAP attribution
==============================================================
The project proposal specified aggregate SHAP summaries ranking hour-of-day and
day-of-month cells by global influence. Only the per-run consistency figure was
reported in the thesis. This script supplies the aggregate.

It also splits by intensity tag, which produces the leech-vs-flood comparison of
explanation maps that sub-question SQ2 asks for.

IMPORTANT: aggregate maps are reported at GROUP level. The thesis shows that
per-sample attributions do not survive their controls (the research report), while
group-level mean attention does (the research report). Do not present per-sample
signatures on the strength of this output.

    python shap_global_summary.py
    python shap_global_summary.py --n 180 --background 100
"""
import argparse
import numpy as np
import torch

from dow_data import load_and_split
from dow_model import DoWNetCNN

CLASS_NAMES = ["normal", "linear", "geometric", "random"]
INTENSITY = {0: "none", 1: "leech", 2: "flood"}


def extract_sv(sv, cls):
    """shap returns different shapes across versions; reduce to a (24,30) array."""
    if isinstance(sv, list):                       # list per class
        a = np.asarray(sv[cls])
    else:
        a = np.asarray(sv)
        if a.ndim == 5:                            # (N,C,H,W,classes)
            a = a[..., cls]
    while a.ndim > 2:                              # drop batch / channel dims
        a = a[0]
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dow_cnn_locked.pt")
    ap.add_argument("--data", default="data.npz")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=180, help="test images to aggregate")
    ap.add_argument("--background", type=int, default=100)
    ap.add_argument("--out", default="shap_global.npz")
    args = ap.parse_args()

    import shap

    model = DoWNetCNN()
    model.load_state_dict(torch.load(args.model, map_location="cpu"))
    model.eval()

    (Xtr, ytr, itr), _, (Xte, yte, ite), _ = load_and_split(args.data, seed=args.seed)
    bg = torch.tensor(Xtr[: args.background])
    explainer = shap.DeepExplainer(model, bg)

    n = min(args.n, len(Xte))
    acc_class = {c: [] for c in range(4)}
    acc_ci = {(c, i): [] for c in range(4) for i in range(3)}

    for k in range(n):
        x = torch.tensor(Xte[k: k + 1])
        cls = int(yte[k])
        sv = explainer.shap_values(x, check_additivity=False)
        m = np.abs(extract_sv(sv, cls))            # magnitude of attribution
        acc_class[cls].append(m)
        acc_ci[(cls, int(ite[k]))].append(m)
        if (k + 1) % 20 == 0:
            print(f"  ...{k + 1}/{n}")

    print("\n=== Mean |SHAP| by class: most influential hours and days ===")
    maps = {}
    for c in range(4):
        if not acc_class[c]:
            continue
        M = np.mean(acc_class[c], axis=0)          # (24,30)
        maps[CLASS_NAMES[c]] = M
        by_hour = M.mean(axis=1)                   # 24
        by_day = M.mean(axis=0)                    # 30
        top_h = np.argsort(by_hour)[::-1][:5]
        top_d = np.argsort(by_day)[::-1][:5]
        print(f"\n{CLASS_NAMES[c]}  (n={len(acc_class[c])})")
        print("  top hours: " + ", ".join(f"{h:02d}:00 ({by_hour[h]:.4f})" for h in top_h))
        print("  top days : " + ", ".join(f"day {d+1} ({by_day[d]:.4f})" for d in top_d))

    print("\n=== Leech vs flood, by class ===")
    for c in range(1, 4):
        for i in (1, 2):
            v = acc_ci[(c, i)]
            if not v:
                continue
            M = np.mean(v, axis=0)
            maps[f"{CLASS_NAMES[c]}_{INTENSITY[i]}"] = M
            by_day = M.mean(axis=0)
            late = by_day[23:].sum() / by_day.sum()
            print(f"  {CLASS_NAMES[c]:>10s}/{INTENSITY[i]:<5s} n={len(v):>3d}  "
                  f"late-month (days 24-30) attribution share = {late:.3f}")

    np.savez_compressed(args.out, **maps)
    print(f"\nsaved {args.out} ({len(maps)} maps). Plot these as 24x30 images for the")
    print("group-level signature figure. Report at group level only (see header).")


if __name__ == "__main__":
    main()
