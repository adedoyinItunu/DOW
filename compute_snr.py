"""
compute_snr.py

Tests whether the evasion threshold is INFORMATIONAL or REPRESENTATIONAL.

The objection this addresses: the leech is constructed by scaling injected load
onto a Poisson base. There must exist a scale at which the injected signal
falls below the noise, at which point NO model can detect it in ANY
representation. If detection collapses at that point, the threshold is a
property of the signal, not of the heat-map.

This script computes, for each intensity scale:

  * per-cell signal      S[h,d] = mean(attack) - mean(normal)
  * per-cell noise       sigma[h,d] = std(normal)
  * matched-filter SNR   d' = sum(S) / sqrt(sum(sigma^2))
                         -- the detectability of the whole image, which is
                            what a classifier seeing all 720 cells has access to
  * fraction of cells where S > sigma

Compare the scale at which d' crosses 1 against the scale at which the
detector collapses (between 0.5 and 0.3).

Usage:
    python compute_snr.py
    python compute_snr.py --n 2000 --classes linear geometric random
"""

import argparse
import numpy as np
from dow_data import GenParams, _GEN, CLASS_NAMES, _NORM_REF

LEECH, FLOOD = 1, 2


def sample_mean_std(cls, intensity, params, n, seed):
    """Generate n grids and return per-cell mean and std (raw counts)."""
    rng = np.random.default_rng(seed)
    grids = np.stack([_GEN[cls](rng, intensity, params) for _ in range(n)])
    return grids.mean(axis=0), grids.std(axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000,
                    help="samples per condition (default 1000)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scales", type=float, nargs="+",
                    default=[1.0, 0.7, 0.5, 0.3, 0.2, 0.1])
    ap.add_argument("--classes", nargs="+",
                    default=["linear", "geometric", "random"])
    args = ap.parse_args()

    base_params = GenParams()

    # ---- the noise floor: per-cell std of the normal class ----
    mu_norm, sigma_norm = sample_mean_std(0, 0, base_params, args.n, args.seed)
    total_noise = np.sqrt((sigma_norm ** 2).sum())

    print("=" * 70)
    print("SIGNAL-TO-NOISE ANALYSIS OF THE LEECH INTENSITY SWEEP")
    print("=" * 70)
    print(f"samples per condition: {args.n}   seed: {args.seed}")
    print(f"generator: amp={base_params.amp} peak={base_params.peak_hour} "
          f"width={base_params.width}\n")

    print("NOISE FLOOR (normal class, raw counts)")
    print(f"  mean per-cell count      : {mu_norm.mean():.2f}")
    print(f"  mean per-cell std        : {sigma_norm.mean():.2f}")
    print(f"  peak-hour cell mean      : {mu_norm.max():.2f}")
    print(f"  peak-hour cell std       : {sigma_norm.max():.2f}")
    print(f"  whole-image noise ||s||  : {total_noise:.2f}")
    print()

    name_to_idx = {n: i for i, n in enumerate(CLASS_NAMES)}

    for cls_name in args.classes:
        c = name_to_idx[cls_name]
        print("=" * 70)
        print(f"CLASS: {cls_name}  (leech, intensity=1)")
        print("=" * 70)
        print(f"{'scale':>6} | {'total signal':>12} | {'matched-filter':>14} | "
              f"{'cells with':>11} | {'peak cell':>10}")
        print(f"{'':>6} | {'(counts)':>12} | {'SNR (d-prime)':>14} | "
              f"{'S > sigma':>11} | {'SNR':>10}")
        print("-" * 70)

        for s in args.scales:
            p = GenParams(leech_scale=s)
            mu_atk, _ = sample_mean_std(c, LEECH, p, args.n, args.seed + 1)

            signal = mu_atk - mu_norm
            signal = np.clip(signal, 0, None)

            total_signal = signal.sum()
            dprime = total_signal / (total_noise + 1e-9)
            frac_above = (signal > sigma_norm).mean()
            peak_cell_snr = (signal / (sigma_norm + 1e-9)).max()

            flag = "  <-- d' crosses 1" if dprime < 1.0 else ""
            print(f"{s:6.2f} | {total_signal:12.1f} | {dprime:14.2f} | "
                  f"{frac_above:10.1%} | {peak_cell_snr:10.2f}{flag}")

        print()

    print("=" * 70)
    print("HOW TO READ THIS")
    print("=" * 70)
    print("""
The matched-filter SNR (d-prime) is the detectability of the whole 24x30
image, i.e. what a classifier with access to every cell could in principle
exploit. It is the relevant quantity, not the per-cell SNR.

  * If d' stays WELL ABOVE 1 at the scales where the detector collapses
    (between 0.5 and 0.3), then the information is present and the detector
    is failing to use it. The threshold is REPRESENTATIONAL or a property of
    the models tested, and the representational reading is supported.

  * If d' falls to around 1 at those same scales, the signal has dropped
    into the noise and no detector could recover it. The threshold is
    INFORMATIONAL -- a property of the detection problem. The finding is still real and better grounded, and the limit would not be imposed by the heat-map representation.

Note also that the model sees log1p-normalised values, which compress large
counts more than small ones. The raw-count SNR computed here is therefore an
UPPER BOUND on what the model has access to. If d' is already near 1 in raw
counts, it is worse after normalisation.
""")


if __name__ == "__main__":
    main()
