"""
experiment4_normalisation.py  --  normalisation as explanation-guided refinement
================================================================================
test_normalisation.py answers one question: does per-image normalisation recover
the legitimate-traffic class under a generator shift? It trains on all of
Configuration A and tests on all of Configuration B, so it reports nothing about
within-configuration performance and nothing about the leech.

That is not enough to call the change a refinement rather than a trade. A
per-image scheme removes absolute traffic intensity from the input, and absolute
intensity is exactly what separates a quiet leech from legitimate traffic
(the research report). The fix for the transfer failure may therefore cost
low-intensity detection, and the point of this experiment is to measure both.

Four axes are reported per scheme, with the model, the seeds, the splits and the
evaluation procedure held constant and only the scale transformation varying:

  1. within-configuration four-class macro-F1   (Config A held-out test split)
  2. cross-configuration normal false positives (Config B, true-normal samples
                                                 predicted as any attack class)
  3. low-intensity leech detection              (Config A test, intensity tag 1)
  4. high-rate flood detection                  (Config A test, intensity tag 2)

Splits are produced by replicating dow_data.load_and_split's calls on the RAW
counts with the same random_state, so the partitions are identical to every
other experiment in the study and only the normalisation differs.

    python experiment4_normalisation.py --seeds 0 1 2 3 4 | tee out_e4_norm.txt

Add --protocol converged to run under the corrected training protocol of the
architecture controls rather than the registered twenty-five epochs.
"""
import argparse

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import TensorDataset, DataLoader

from dow_data import CLASS_NAMES
from dow_model import DoWNetCNN, set_seed
from test_normalisation import SCHEMES


def raw_split(path, seed, val=0.15, test=0.15):
    """dow_data.load_and_split, but returning RAW counts so any scheme applies."""
    d = np.load(path, allow_pickle=True)
    X, y, inten = d["X"], d["y"], d["intensity"]
    Xtr, Xtmp, ytr, ytmp, itr, itmp = train_test_split(
        X, y, inten, test_size=val + test, random_state=seed, stratify=y)
    Xv, Xte, yv, yte, iv, ite = train_test_split(
        Xtmp, ytmp, itmp, test_size=test / (val + test),
        random_state=seed, stratify=ytmp)
    return (Xtr, ytr, itr), (Xv, yv, iv), (Xte, yte, ite)


def train(Xtr, ytr, n_classes, protocol, lr=1e-3, batch=32):
    dl = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(ytr)),
                    batch_size=batch, shuffle=True)
    model = DoWNetCNN(n_classes=n_classes)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    n_epochs = 200 if protocol == "converged" else 25
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
             if protocol == "converged" else None)
    for _ in range(n_epochs):
        model.train()
        for xb, yb in dl:
            opt.zero_grad(); crit(model(xb), yb).backward(); opt.step()
        if sched is not None:
            sched.step()
    model.eval()
    return model


def predict(model, X):
    with torch.no_grad():
        return model(torch.tensor(X)).argmax(1).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-a", default="data_configA.npz")
    ap.add_argument("--config-b", default="data_configB.npz")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--protocol", default="fixed25",
                    choices=["fixed25", "converged"])
    args = ap.parse_args()

    db = np.load(args.config_b, allow_pickle=True)
    Xb_raw, yb = db["X"], db["y"]
    n_classes = len(CLASS_NAMES)

    print("=" * 78)
    print("EXPERIMENT 4  --  normalisation as a targeted intervention")
    print(f"train {args.config_a} | cross-test {args.config_b} | "
          f"protocol {args.protocol} | seeds {args.seeds}")
    print("=" * 78)

    results = {}
    for scheme, fn in SCHEMES.items():
        within, leech, flood, normfp, crossnorm = [], [], [], [], []

        for s in args.seeds:
            set_seed(s)
            (Xtr, ytr, _), _, (Xte, yte, ite) = raw_split(args.config_a, s)
            model = train(fn(Xtr), ytr, n_classes, args.protocol)

            # --- axes 1, 3, 4: within configuration
            pred = predict(model, fn(Xte))
            _, _, f1, _ = precision_recall_fscore_support(
                yte, pred, labels=range(n_classes), zero_division=0)
            within.append(f1.mean())
            for tag, acc in ((1, leech), (2, flood)):
                m = ite == tag
                acc.append(float((pred[m] == yte[m]).mean()) if m.any() else np.nan)

            # --- axis 2: cross-configuration false positives on legitimate traffic
            predb = predict(model, fn(Xb_raw))
            is_normal = yb == 0
            normfp.append(float((predb[is_normal] != 0).mean()))
            _, _, f1b, _ = precision_recall_fscore_support(
                yb, predb, labels=range(n_classes), zero_division=0)
            crossnorm.append(f1b[0])

        results[scheme] = {k: (np.nanmean(v), np.nanstd(v)) for k, v in
                           dict(within=within, leech=leech, flood=flood,
                                normfp=normfp, crossnorm=crossnorm).items()}

        r = results[scheme]
        print(f"\nSCHEME: {scheme}")
        print(f"  1. within-config macro-F1       {r['within'][0]:.3f} +/- {r['within'][1]:.3f}")
        print(f"  2. cross-config normal FP rate  {r['normfp'][0]:.3f} +/- {r['normfp'][1]:.3f}"
              "   (1.000 = every legitimate sample flagged)")
        print(f"     cross-config normal-class F1 {r['crossnorm'][0]:.3f} +/- {r['crossnorm'][1]:.3f}")
        print(f"  3. leech detection (intensity 1){r['leech'][0]:>7.3f} +/- {r['leech'][1]:.3f}")
        print(f"  4. flood detection (intensity 2){r['flood'][0]:>7.3f} +/- {r['flood'][1]:.3f}")

    # ---- verdict
    base, alt = "fixed log1p(400)", "per-image robust"
    b, a = results[base], results[alt]
    print("\n" + "=" * 78)
    print(f"{'axis':<34}{base:>20}{alt:>22}")
    print("-" * 78)
    rows = [("within-config macro-F1", "within"),
            ("cross-config normal FP rate", "normfp"),
            ("leech detection", "leech"),
            ("flood detection", "flood")]
    for label, key in rows:
        print(f"{label:<34}{b[key][0]:>13.3f}{'':7}{a[key][0]:>15.3f}")
    print("-" * 78)

    fp_gain = b["normfp"][0] - a["normfp"][0]
    leech_cost = b["leech"][0] - a["leech"][0]
    print(f"cross-config false positives fall by {fp_gain:+.3f}")
    print(f"leech detection changes by           {-leech_cost:+.3f}")
    if fp_gain > 0.05 and leech_cost < 0.05:
        print("\nVERDICT: the alternative improves cross-configuration behaviour")
        print("without materially damaging attack detection. This supports the")
        print("intervention as a refinement rather than a trade.")
    elif fp_gain > 0.05:
        print("\nVERDICT: the alternative improves cross-configuration behaviour")
        print("at a measurable cost to low-intensity detection. Report as a")
        print("trade-off, not a fix, and give both numbers.")
    else:
        print("\nVERDICT: the alternative does not resolve the failure. This is a")
        print("negative result and still informative: the failure is not fixed")
        print("merely by changing the scale transformation.")


if __name__ == "__main__":
    main()
