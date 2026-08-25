#!/usr/bin/env python3
"""
Q8 check: recalibrated-per-seed (as committed) vs ONE genuinely independent
threshold calibrated on a held-out seed and applied unchanged everywhere.
"""
import numpy as np
from sklearn.metrics import roc_curve
from fixed_threshold_control import generate, mean_pixel_scores, youden_threshold

SCALES = [1.00, 0.70, 0.50, 0.30, 0.20, 0.10]
EVAL_SEEDS = [0, 1, 2, 3, 4]
CAL_SEEDS = [99, 123, 7, 999]

# ---- calibrate one threshold per candidate held-out seed -------------------
cal_thr = {}
for cs in CAL_SEEDS:
    X, y, tag = generate(1.00, cs)
    s = mean_pixel_scores(X)
    keep = (tag == 0) | (tag == 1)
    cal_thr[cs] = youden_threshold(s[keep], (tag[keep] == 1).astype(int))

# ---- also record the per-seed thresholds the committed script produces -----
own_thr = {}
for es in EVAL_SEEDS:
    X, y, tag = generate(1.00, es)
    s = mean_pixel_scores(X)
    keep = (tag == 0) | (tag == 1)
    own_thr[es] = youden_threshold(s[keep], (tag[keep] == 1).astype(int))

print("threshold values")
print("  per-seed (as committed):",
      ", ".join(f"{k}:{v:.4f}" for k, v in own_thr.items()))
print("  held-out calibration   :",
      ", ".join(f"{k}:{v:.4f}" for k, v in cal_thr.items()))
print(f"  spread across per-seed thresholds: "
      f"{np.std(list(own_thr.values())):.5f}\n")

# ---- cache scores for every eval seed x scale ------------------------------
scores = {}
for es in EVAL_SEEDS:
    for sc in SCALES:
        X, y, tag = generate(sc, es)
        scores[(es, sc)] = (mean_pixel_scores(X), tag)

def sweep(thr_for_seed, label):
    print(f"--- {label} ---")
    print(f"{'scale':>6}  {'detect':>17}")
    out = {}
    for sc in SCALES:
        vals = []
        for es in EVAL_SEEDS:
            s, tag = scores[(es, sc)]
            vals.append(float((s[tag == 1] > thr_for_seed(es)).mean()))
        a = np.asarray(vals)
        out[sc] = a
        print(f"{sc:>6.2f}  {a.mean():.3f} +/- {a.std():.3f}")
    fp = []
    for es in EVAL_SEEDS:
        s, tag = scores[(es, 1.00)]
        fp.append(float((s[tag == 0] > thr_for_seed(es)).mean()))
    print(f"  FPR on normal: {np.mean(fp):.4f} +/- {np.std(fp):.4f}\n")
    return out

a = sweep(lambda es: own_thr[es], "A. per-seed recalibration (as committed)")
b = {}
for cs in CAL_SEEDS:
    b[cs] = sweep(lambda es, t=cal_thr[cs]: t,
                  f"B. single independent threshold, calibrated on seed {cs}")

print("=" * 60)
print("difference (independent - committed), detection rate")
print(f"{'scale':>6}  " + "  ".join(f"seed{cs:>4}" for cs in CAL_SEEDS))
for sc in SCALES:
    row = "  ".join(f"{b[cs][sc].mean() - a[sc].mean():+8.3f}" for cs in CAL_SEEDS)
    print(f"{sc:>6.2f}  {row}")
