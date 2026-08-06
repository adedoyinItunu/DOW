"""
architecture_controls.py  --  the three unrun controls, one script
==================================================================
Crosses architecture with training protocol across the same seeds used in
perclass_across_seeds.py, so every number produced here is directly comparable
with the 0.756 +/- 0.149 and 0.966 +/- 0.030 already reported.

Two independent axes:

  --arch      minimal | nogap | downet          (dow_model_variants.py)
  --protocol  fixed25 | fixed25_sel | converged

    fixed25      25 epochs, final-epoch parameters. The protocol used by
                 every multi-seed analysis in the thesis. Reproduces
                 0.756 +/- 0.149 for arch=minimal on data_lramp.npz.
    fixed25_sel  25 epochs, best-validation epoch restored. The protocol
                 experiment1_baseline.py actually uses. Reproduces
                 0.966 +/- 0.030 for arch=minimal.
    converged    --epochs epochs with cosine annealing, final-epoch
                 parameters. The missing control: does the instability
                 attributed to the architecture survive a converged
                 training run?

  --vary      both | init

    both        seed drives weight initialisation AND the train/test
                partition (the thesis default).
    init        partition fixed at --split-seed, seed drives initialisation
                only. This measurement does not currently exist anywhere in
                the thesis, and it is what separates "the architecture is
                unstable" from "the partitions differ".

Start here, in this order:

    python dow_model_variants.py                     # sanity: param counts
    python architecture_controls.py --arch minimal --protocol fixed25
    python architecture_controls.py --arch minimal --protocol fixed25_sel
    python architecture_controls.py --arch minimal --protocol converged
    python architecture_controls.py --arch minimal --protocol fixed25 --vary init

The first two must reproduce the published figures. If they do not, stop and
find out why before running anything else.
"""
import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support

from dow_data import load_and_split
from dow_model import set_seed
from dow_model_variants import build


def train_one(Xtr, ytr, Xv, yv, n_classes, arch, protocol,
              epochs=25, lr=1e-3, batch=32, width=16):
    ds = TensorDataset(torch.tensor(Xtr), torch.tensor(ytr))
    dl = DataLoader(ds, batch_size=batch, shuffle=True)
    model = build(arch, n_classes=n_classes, width=width)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    n_epochs = epochs if protocol == "converged" else 25
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
             if protocol == "converged" else None)

    select = (protocol == "fixed25_sel")
    best_state, best_val = None, -1.0
    val_trace = []

    for _ in range(n_epochs):
        model.train()
        for xb, yb in dl:
            opt.zero_grad()
            crit(model(xb), yb).backward()
            opt.step()
        if sched is not None:
            sched.step()

        # validation accuracy is always recorded; it is only USED when
        # protocol == fixed25_sel. Recording it costs nothing and gives you
        # the epoch trace that section 4.2.1 currently reports as prose.
        model.eval()
        with torch.no_grad():
            pv = model(torch.tensor(Xv)).argmax(1).numpy()
        acc = float((pv == yv).mean())
        val_trace.append(acc)
        if select and acc > best_val:
            best_val = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if select and best_state is not None:
        model.load_state_dict(best_state)
    return model, val_trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_lramp.npz")
    ap.add_argument("--arch", default="minimal",
                    choices=["minimal", "nogap", "downet"])
    ap.add_argument("--protocol", default="fixed25",
                    choices=["fixed25", "fixed25_sel", "converged"])
    ap.add_argument("--vary", default="both", choices=["both", "init"])
    ap.add_argument("--split-seed", type=int, default=42,
                    help="partition seed held fixed when --vary init")
    ap.add_argument("--epochs", type=int, default=200,
                    help="epoch budget for protocol=converged")
    ap.add_argument("--width", type=int, default=16,
                    help="block2 channels for arch=nogap (16 = param-matched)")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--out", default=None, help="write results as JSON")
    args = ap.parse_args()

    tag = f"{args.arch}/{args.protocol}/vary={args.vary}"
    print(f"=== {tag} on {args.data} ===")
    if args.protocol == "converged":
        print(f"    {args.epochs} epochs, cosine annealing, final-epoch params")

    F, macros, traces = [], [], []
    names = None
    t0 = time.time()

    for s in args.seeds:
        set_seed(s)
        split_seed = args.split_seed if args.vary == "init" else s
        (Xtr, ytr, _), (Xv, yv, _), (Xte, yte, _), names = \
            load_and_split(args.data, split_seed)

        model, trace = train_one(Xtr, ytr, Xv, yv, len(names),
                                 args.arch, args.protocol,
                                 epochs=args.epochs, width=args.width)
        model.eval()
        with torch.no_grad():
            pred = model(torch.tensor(Xte)).argmax(1).numpy()
        _, _, f, _ = precision_recall_fscore_support(
            yte, pred, labels=range(len(names)), zero_division=0)

        F.append(f)
        macros.append(float(f.mean()))
        traces.append(trace)
        print(f"seed {s}: macro-F1 {f.mean():.3f}  " +
              " ".join(f"{n[:4]}={fi:.3f}" for n, fi in zip(names, f)))

    F = np.array(F)
    macros = np.array(macros)
    collapses = int((F == 0.0).sum())
    swings = [max(abs(t[i + 1] - t[i]) for i in range(len(t) - 1))
              for t in traces]

    print("\n" + "-" * 60)
    print(f"macro-F1        {macros.mean():.3f} +/- {macros.std():.3f}"
          f"   (range {macros.min():.3f}-{macros.max():.3f})")
    for i, n in enumerate(names):
        print(f"  {n:<10}    {F[:, i].mean():.3f} +/- {F[:, i].std():.3f}")
    print(f"per-class collapses to F1=0.000: {collapses} of {F.size}")
    print(f"largest adjacent-epoch validation swing: "
          f"{max(swings):.3f} (max over seeds), {np.mean(swings):.3f} (mean)")
    print(f"elapsed {time.time() - t0:.0f}s")
    print("-" * 60)
    print("Compare against, on data_lramp.npz, arch=minimal, vary=both:")
    print("  fixed25      0.756 +/- 0.149   (thesis 4.2)")
    print("  fixed25_sel  0.966 +/- 0.030   (thesis 4.1)")
    print("  logistic reg 0.972 +/- 0.008   (thesis 4.2.2)")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"tag": tag, "data": args.data, "seeds": args.seeds,
                       "macro_mean": float(macros.mean()),
                       "macro_std": float(macros.std()),
                       "per_class_f1_mean": F.mean(0).tolist(),
                       "per_class_f1_std": F.std(0).tolist(),
                       "collapses": collapses,
                       "val_traces": traces}, fh, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
