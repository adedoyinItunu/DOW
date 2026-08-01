"""
audit_checkpoints.py

Loads each saved checkpoint WITHOUT retraining and reports the E1, E3
and E4 numbers it produces. This tells you which checkpoint generated
which figures in the thesis, and whether the old numbers were ever
self-consistent.

Usage:
    python audit_checkpoints.py
    python audit_checkpoints.py --data data.npz --seed 42
"""

import argparse
import os
import numpy as np
import torch
from sklearn.metrics import f1_score, confusion_matrix
from dow_data import load_and_split
from dow_model import DoWNetCNN

CHECKPOINTS = ["dow_cnn.pt", "cnn_configA.pt", "dow_cnn_locked.pt"]


def evaluate(ckpt, Xte, yte, ite, names, normal_idx):
    n = len(names)
    model = DoWNetCNN(n_classes=n)
    try:
        state = torch.load(ckpt, map_location="cpu")
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)
    except Exception as e:
        print(f"  !! could not load {ckpt}: {e}")
        return None
    model.eval()

    with torch.no_grad():
        pred = model(torch.tensor(Xte)).argmax(1).numpy()

    cm = confusion_matrix(yte, pred, labels=list(range(n)))
    macro = f1_score(yte, pred, average="macro")
    acc = (pred == yte).mean()

    attack_mask = yte != normal_idx
    attack_to_normal = int(((pred == normal_idx) & attack_mask).sum())

    leech_mask = ite == 1
    flood_mask = ite == 2
    leech_acc = (pred[leech_mask] == yte[leech_mask]).mean() if leech_mask.any() else float("nan")
    flood_acc = (pred[flood_mask] == yte[flood_mask]).mean() if flood_mask.any() else float("nan")
    leech_to_normal = int((pred[leech_mask] == normal_idx).sum())
    flood_to_normal = int((pred[flood_mask] == normal_idx).sum())

    return dict(cm=cm, macro=macro, acc=acc,
                attack_to_normal=attack_to_normal,
                leech_n=int(leech_mask.sum()), leech_acc=leech_acc,
                leech_to_normal=leech_to_normal,
                flood_n=int(flood_mask.sum()), flood_acc=flood_acc,
                flood_to_normal=flood_to_normal,
                n_correct=int((pred == yte).sum()), n_total=len(yte))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data.npz")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    (_, _, _), (_, _, _), (Xte, yte, ite), names = load_and_split(
        args.data, args.seed)
    normal_idx = names.index("normal") if "normal" in names else 0

    print(f"test split: n={len(yte)}  seed={args.seed}")
    print(f"classes: {names}\n")

    results = {}
    for ckpt in CHECKPOINTS:
        if not os.path.exists(ckpt):
            print(f"--- {ckpt}: NOT FOUND, skipping ---\n")
            continue
        print("=" * 60)
        print(f"CHECKPOINT: {ckpt}")
        print("=" * 60)
        r = evaluate(ckpt, Xte, yte, ite, names, normal_idx)
        if r is None:
            continue
        results[ckpt] = r

        print("confusion matrix (rows=true, cols=pred):")
        print(r["cm"])
        print(f"\nE1  accuracy: {r['acc']:.3f} ({r['n_correct']}/{r['n_total']})"
              f"   macro-F1: {r['macro']:.3f}")
        print(f"    attack -> normal: {r['attack_to_normal']}")
        print(f"E3  leech: n={r['leech_n']} acc={r['leech_acc']:.3f} "
              f"->normal={r['leech_to_normal']}")
        print(f"    flood: n={r['flood_n']} acc={r['flood_acc']:.3f} "
              f"->normal={r['flood_to_normal']}")
        print(f"E4  baseline overall accuracy: {r['acc']:.3f}")
        print()

    # ---- comparison against what the thesis currently reports ----
    print("=" * 60)
    print("COMPARISON WITH THESIS AS WRITTEN")
    print("=" * 60)
    print("Thesis reports: E1 macro-F1 0.955 | E3 leech 1.000, flood 0.985,")
    print("                both ->normal 0.000 | E4 baseline acc 0.994\n")

    hdr = f"{'checkpoint':22s} {'E1 macroF1':>11s} {'E3 leech':>9s} " \
          f"{'E3 flood':>9s} {'E4 acc':>8s} {'atk->norm':>10s}"
    print(hdr)
    print("-" * len(hdr))
    for ckpt, r in results.items():
        print(f"{ckpt:22s} {r['macro']:11.3f} {r['leech_acc']:9.3f} "
              f"{r['flood_acc']:9.3f} {r['acc']:8.3f} "
              f"{r['attack_to_normal']:10d}")

    print("\nHOW TO READ THIS:")
    print("  * If one checkpoint matches ALL FOUR thesis numbers, the old")
    print("    figures were self-consistent. Keep them; just state which")
    print("    checkpoint they came from in Section 3.8.")
    print("  * If no single checkpoint matches all four, the thesis numbers")
    print("    were drawn from different models. Use dow_cnn_locked.pt for")
    print("    everything and regenerate E1, E2, E3, E4, the transfer test")
    print("    and the Grad-CAM attention shares.")
    print("  * Either way, note which checkpoint each figure came from.")


if __name__ == "__main__":
    main()
