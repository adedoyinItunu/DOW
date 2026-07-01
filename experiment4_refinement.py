"""
experiment4_refinement.py  --  EXPERIMENT 4 (explanation-guided refinement)
==========================================================================
The "true experiment": change ONE thing, retrain, measure the effect.

E2/E3 show the model's weak spot is the low-rate leech. The intervention here is
data augmentation: add extra leech samples to TRAINING only (the test set stays
fixed), retrain a fresh CNN, and compare leech accuracy against the baseline.

    python experiment4_refinement.py --data data.npz --model dow_cnn.pt

This is the stretch goal in the proposal; a small or null improvement is still a
valid experimental result.
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from dow_data import load_and_split, normalize, _GEN
from dow_model import DoWNetCNN, set_seed


def make_extra_leech(n_per_class, seed):
    """Extra low-rate (intensity=1) attack samples for classes linear/geometric/random."""
    rng = np.random.default_rng(seed)
    X, y = [], []
    for c in [1, 2, 3]:
        for _ in range(n_per_class):
            X.append(_GEN[c](rng, 1)[None, ...]); y.append(c)
    return normalize(np.asarray(X, dtype="float32")), np.asarray(y, dtype="int64")


def train(Xtr, ytr, epochs=25, lr=1e-3, batch=32):
    ds = TensorDataset(torch.tensor(Xtr), torch.tensor(ytr))
    dl = DataLoader(ds, batch_size=batch, shuffle=True)
    model = DoWNetCNN(); opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        for xb, yb in dl:
            opt.zero_grad(); crit(model(xb), yb).backward(); opt.step()
    model.eval(); return model


def leech_acc(model, Xte, yte, ite):
    with torch.no_grad():
        pred = model(torch.tensor(Xte)).argmax(1).numpy()
    m = ite == 1
    return (pred[m] == yte[m]).mean(), (pred == yte).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data.npz")
    ap.add_argument("--model", default="dow_cnn.pt")
    ap.add_argument("--extra", type=int, default=200, help="extra leech samples per class")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    (Xtr, ytr, _), _, (Xte, yte, ite), names = load_and_split(args.data, args.seed)

    # baseline model (already trained in E1)
    base = DoWNetCNN(); base.load_state_dict(torch.load(args.model)); base.eval()
    b_leech, b_all = leech_acc(base, Xte, yte, ite)

    # intervention: augment training with extra leech, retrain fresh model
    Xl, yl = make_extra_leech(args.extra, seed=args.seed + 1)
    Xaug = np.concatenate([Xtr, Xl]); yaug = np.concatenate([ytr, yl])
    print(f"baseline train={len(ytr)}  ->  augmented train={len(yaug)} "
          f"(+{len(yl)} leech)")
    refined = train(Xaug, yaug)
    r_leech, r_all = leech_acc(refined, Xte, yte, ite)

    print("\n=============== EXPERIMENT 4: refinement ===============")
    print(f"                 leech-acc    overall-acc")
    print(f"baseline         {b_leech:.3f}        {b_all:.3f}")
    print(f"refined          {r_leech:.3f}        {r_all:.3f}")
    print(f"change           {r_leech - b_leech:+.3f}        {r_all - b_all:+.3f}")
    verdict = ("leech detection improved" if r_leech > b_leech else
               "no leech improvement" if r_leech == b_leech else "leech detection dropped")
    print(f"\nVerdict: {verdict} without hurting overall accuracy "
          f"({'ok' if r_all >= b_all - 0.01 else 'overall dropped'}).")


if __name__ == "__main__":
    main()
