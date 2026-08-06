"""
plot_val_traces.py  --  validation trajectories for Section 4.2.1
=================================================================
Section 4.2.1 quotes seven numbers in prose ("0.972, 0.883, 0.911, 0.861, 0.622,
1.000, 0.967") for what is now a row in Table 4.16 and one of the study's more
contested claims. This plots the trajectories instead, from the val_traces field
that architecture_controls.py writes into its JSON.

Panel (a) overlays every seed's trajectory for one condition, with the mean in
bold, so the reader can see that the oscillation is not specific to seed 42.
Panel (b) plots the distribution of largest adjacent-epoch swings per condition,
which is the quantity Table 4.16 reports.

Produce the inputs first:

    python architecture_controls.py --arch minimal --protocol fixed25 \
        --data data_lramp.npz  --out traces_lramp.json
    python architecture_controls.py --arch minimal --protocol fixed25 \
        --data data_lconst.npz --out traces_lconst.json

then:

    python plot_val_traces.py traces_lramp.json traces_lconst.json \
        --labels L-ramp L-const --out val_trajectories.png

Any number of JSON files can be passed; --labels must match in length, or the
tag recorded inside each file is used.
"""
import argparse
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def largest_swing(trace):
    return max(abs(trace[i + 1] - trace[i]) for i in range(len(trace) - 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="JSON files from architecture_controls.py")
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--out", default="val_trajectories.png")
    ap.add_argument("--highlight", type=int, default=None,
                    help="index into the seed list to draw in colour (e.g. seed 42's position)")
    args = ap.parse_args()

    runs = []
    for path in args.files:
        with open(path) as fh:
            runs.append(json.load(fh))

    labels = args.labels if args.labels else [r.get("tag", p)
                                              for r, p in zip(runs, args.files)]
    if len(labels) != len(runs):
        raise SystemExit("--labels must have one entry per file")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2),
                                   gridspec_kw={"width_ratios": [1.7, 1]})

    colours = ["C0", "C3", "C2", "C4"]

    # ---- panel (a): trajectories, first condition in full, others as mean only
    for k, (run, lab, col) in enumerate(zip(runs, labels, colours)):
        traces = np.array(run["val_traces"], dtype=float)
        epochs = np.arange(1, traces.shape[1] + 1)
        if k == 0:
            for i, t in enumerate(traces):
                ax1.plot(epochs, t, color=col, alpha=0.28, lw=0.9)
            if args.highlight is not None:
                ax1.plot(epochs, traces[args.highlight], color=col, lw=1.8,
                         ls="--", label=f"{lab}, seed index {args.highlight}")
        ax1.plot(epochs, traces.mean(0), color=col, lw=2.4,
                 label=f"{lab} (mean of {len(traces)} seeds)")

    ax1.set_xlabel("epoch")
    ax1.set_ylabel("validation accuracy")
    ax1.set_title("(a) Validation accuracy is not converged at epoch 25")
    ax1.set_ylim(0, 1.03)
    ax1.grid(alpha=0.25)
    ax1.legend(fontsize=8, loc="lower right", framealpha=0.9)

    # ---- panel (b): largest adjacent-epoch swing per seed
    data = [[largest_swing(t) for t in run["val_traces"]] for run in runs]
    # set tick labels separately: boxplot's `labels` kwarg was renamed
    # `tick_labels` in matplotlib 3.9, so neither name is portable
    bp = ax2.boxplot(data, widths=0.5, patch_artist=True,
                     medianprops={"color": "k"})
    ax2.set_xticks(range(1, len(labels) + 1))
    ax2.set_xticklabels(labels)
    for patch, col in zip(bp["boxes"], colours):
        patch.set_facecolor(col); patch.set_alpha(0.35)
    for i, d in enumerate(data, start=1):
        ax2.plot(np.full(len(d), i) + np.random.uniform(-0.09, 0.09, len(d)),
                 d, "o", color="k", ms=3.5, alpha=0.7)
        print(f"{labels[i-1]}: largest adjacent-epoch swing "
              f"mean {np.mean(d):.3f}, max {np.max(d):.3f}")

    ax2.set_ylabel("largest adjacent-epoch swing")
    ax2.set_title("(b) Oscillation is not seed-specific")
    ax2.set_ylim(bottom=0)
    ax2.grid(alpha=0.25, axis="y")

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
