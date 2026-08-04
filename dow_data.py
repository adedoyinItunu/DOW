"""
dow_data.py  --  the DoWTS-compatible data generator
====================================================
DoWTS will not compile, so this recreates its four canonical traffic patterns as
24 x 30 heat-maps (rows = hour 0-23, cols = day 1-30, cell = invocation count):

    class 0  normal      diurnal baseline (busy midday, quiet night)
    class 1  linear      gradual ramp-up over the month
    class 2  geometric   exponential surge, mostly late month
    class 3  random      scattered speckle

Each ATTACK sample is generated at a low or high INTENSITY:
    intensity 1 = low  -> the slow "leech" (continual inconspicuous) case
    intensity 2 = high -> the obvious "flood" case
(normal = intensity 0). Experiment 3 (leech vs flood) uses this tag.

The low-rate leech samples use an adaptive pattern hidden inside the diurnal
envelope -- a more realistic leech than the primitive published ones.

--------------------------------------------------------------------------------
GENERATOR PARAMETERS (for the robustness experiment)
--------------------------------------------------------------------------------
The traffic-generation parameters are exposed as command-line flags so that a
second, DIFFERENT configuration can be produced for a train-on-A / test-on-B
robustness test. The DEFAULTS reproduce the original generator exactly, so all
existing results are unchanged when no flags are passed.

    peak_hour   diurnal peak (default 13.0)      --peak-hour
    width       diurnal spread (default 3.5)     --width
    amp         base amplitude (default 30.0)    --amp
    weekend     weekend factor (default 0.7)     --weekend
    leech_scale multiplier on leech rates (1.0)  --leech-scale
    flood_scale multiplier on flood rates (1.0)  --flood-scale

--------------------------------------------------------------------------------
LINEAR CLASS TEMPORAL SHAPE (--linear-const)
--------------------------------------------------------------------------------
Two definitions of the `linear` class are supported, selected by --linear-const.

    L-ramp   (default, linear_const=False)
        The injected hourly rate RAMPS across the month. This is the definition
        used for all originally reported results.

    L-const  (linear_const=True)
        The injected hourly rate is CONSTANT from a random onset day to the end
        of the month, so the CUMULATIVE total rises linearly while the hourly
        rate stays flat. This follows the DoWTS specification of the first
        attack pattern.

    Two implementation choices are made here and must be reported as such:
      (a) LOAD MATCHING. The constant-rate variant injects the same TOTAL load
          as the ramp variant it replaces, so that only the temporal SHAPE
          differs between the two conditions and any performance difference is
          attributable to shape rather than volume. DoWTS itself does not
          load-match (its rate follows from bot count, so the total depends on
          duration).
      (b) ONSET RANGE. Onset is drawn uniformly from the first half of the
          month. Under load matching, a very late onset would compress the whole
          month's load into a handful of days and produce a late-month spike --
          i.e. something resembling the `geometric` class -- which would
          reintroduce the confusion this condition exists to remove. Capping
          onset at day 15 guarantees at least 16 active days.

    `onset` is drawn unconditionally, in BOTH branches, so that the two
    conditions consume the RNG stream identically. This keeps the `normal`,
    `geometric` and `random` classes bit-for-bit identical between the two
    datasets, so the comparison isolates the linear class alone.

Run:
    python dow_data.py --per-class 300 --out data.npz              # Config A (default)
    python dow_data.py --per-class 300 --peak-hour 10 --width 5.0 \
                       --amp 45 --out data_configB.npz             # Config B (alternative)
    python dow_data.py --per-class 300 --linear-const \
                       --out data_lconst.npz                       # L-const condition
    python dow_data.py --preview                                   # save a preview PNG
"""
import argparse
from dataclasses import dataclass
import numpy as np

HOURS, DAYS = 24, 30
CLASS_NAMES = ["normal", "linear", "geometric", "random"]
_NORM_REF = np.log1p(400.0)


@dataclass
class GenParams:
    """Generator configuration. Defaults reproduce the original generator exactly."""
    peak_hour: float = 13.0
    width: float = 3.5
    amp: float = 30.0
    weekend: float = 0.7
    leech_scale: float = 1.0
    flood_scale: float = 1.0
    linear_const: bool = False   # False = L-ramp (as submitted); True = L-const (DoWTS)


def normalize(x):
    """log1p then fixed scaling (shared by every experiment script)."""
    return (np.log1p(np.clip(x, 0, None)) / _NORM_REF).astype("float32")


def _diurnal_base(rng, p: GenParams):
    hours = np.arange(HOURS)[:, None]
    days = np.arange(DAYS)[None, :]
    daily = 0.15 + np.exp(-((hours - p.peak_hour) ** 2) / (2 * p.width ** 2))
    weekly = np.where(np.isin(days % 7, [5, 6]), p.weekend, 1.0)
    return rng.poisson(p.amp * daily * weekly).astype("float32")


def _envelope(p: GenParams):
    hours = np.arange(HOURS)[:, None]
    return 0.15 + np.exp(-((hours - p.peak_hour) ** 2) / (2 * p.width ** 2))


def _gen_normal(rng, intensity, p: GenParams):
    return _diurnal_base(rng, p)


def _gen_linear(rng, intensity, p: GenParams):
    """
    Linear attack class, in one of two temporal shapes (see module docstring).

    L-ramp  (p.linear_const is False):  injected hourly rate ramps across month.
    L-const (p.linear_const is True):   injected hourly rate constant from onset,
                                        load-matched to the ramp it replaces.
    """
    g = _diurnal_base(rng, p)
    days = np.arange(DAYS)[None, :]
    ramp = days / (DAYS - 1)                     # 0 -> 1 across the month

    if intensity == 1:                           # leech: subtle, hidden in the daily shape
        slope = rng.uniform(8, 16) * p.leech_scale
        shape = _envelope(p)
    else:                                        # flood: uniform across hours
        slope = rng.uniform(35, 60) * p.flood_scale
        shape = 1.0

    # Drawn in BOTH branches so the two conditions consume the RNG identically.
    onset = int(rng.integers(0, DAYS // 2))

    if p.linear_const:                           # DoWTS: constant rate from onset
        active = DAYS - onset
        rate = ramp.sum() * slope / active       # load-matched to the ramp variant
        profile = np.zeros((1, DAYS))
        profile[0, onset:] = rate
        extra = shape * profile
    else:                                        # original: ramp in the hourly rate
        extra = shape * ramp * slope

    return g + rng.poisson(np.clip(extra, 0, None)).astype("float32")


def _gen_geometric(rng, intensity, p: GenParams):
    g = _diurnal_base(rng, p)
    days = np.arange(DAYS)[None, :]
    if intensity == 1:
        base = rng.uniform(0.3, 0.7) * p.leech_scale
    else:
        base = rng.uniform(1.2, 2.5) * p.flood_scale
    surge = np.clip(base * (1.18 ** days), 0, 130)
    return g + rng.poisson(np.broadcast_to(surge, g.shape)).astype("float32")


def _gen_random(rng, intensity, p: GenParams):
    g = _diurnal_base(rng, p)
    speckle = np.zeros_like(g)
    if intensity == 1:
        n = int(rng.integers(30, 60) * p.leech_scale); lo, hi = 20, 45
    else:
        n = int(rng.integers(80, 140) * p.flood_scale); lo, hi = 50, 100
    n = max(n, 1)
    hs = rng.integers(0, HOURS, n); ds = rng.integers(0, DAYS, n)
    speckle[hs, ds] += rng.integers(lo, hi, n)
    return g + speckle


_GEN = {0: _gen_normal, 1: _gen_linear, 2: _gen_geometric, 3: _gen_random}


def generate_dataset(per_class=300, seed=42, params: GenParams = None):
    """
    Returns X (N,1,24,30 raw counts), y (class 0-3), intensity (0 none/1 leech/2 flood).
    """
    if params is None:
        params = GenParams()
    rng = np.random.default_rng(seed)
    X, y, inten = [], [], []
    for c in range(4):
        for i in range(per_class):
            if c == 0:
                lvl = 0
            else:
                lvl = 1 if i < per_class // 2 else 2     # half leech, half flood
            grid = _GEN[c](rng, lvl, params)
            X.append(grid[None, ...])                    # channel-first (1,24,30)
            y.append(c); inten.append(lvl)
    X = np.asarray(X, dtype="float32")
    y = np.asarray(y, dtype="int64")
    inten = np.asarray(inten, dtype="int64")
    idx = rng.permutation(len(y))
    return X[idx], y[idx], inten[idx]


def load_and_split(path, seed=42, val=0.15, test=0.15):
    """Load the npz and return normalised, stratified train/val/test numpy splits."""
    from sklearn.model_selection import train_test_split
    d = np.load(path, allow_pickle=True)
    X, y, inten = normalize(d["X"]), d["y"], d["intensity"]
    Xtr, Xtmp, ytr, ytmp, itr, itmp = train_test_split(
        X, y, inten, test_size=val + test, random_state=seed, stratify=y)
    rel = test / (val + test)
    Xv, Xte, yv, yte, iv, ite = train_test_split(
        Xtmp, ytmp, itmp, test_size=rel, random_state=seed, stratify=ytmp)
    return (Xtr, ytr, itr), (Xv, yv, iv), (Xte, yte, ite), list(CLASS_NAMES)


def load_all(path):
    """Load the full (normalised) dataset without splitting -- used for cross-evaluation."""
    d = np.load(path, allow_pickle=True)
    return normalize(d["X"]), d["y"], d["intensity"], list(CLASS_NAMES)


def save_preview(path="data_preview.png", seed=1, params: GenParams = None):
    import matplotlib.pyplot as plt
    if params is None:
        params = GenParams()
    rng = np.random.default_rng(seed)
    rows = [("normal", 0, 0), ("linear leech", 1, 1), ("linear flood", 1, 2),
            ("geometric leech", 2, 1), ("geometric flood", 2, 2),
            ("random leech", 3, 1), ("random flood", 3, 2)]
    fig, axes = plt.subplots(1, len(rows), figsize=(2.1 * len(rows), 3))
    for ax, (title, c, lvl) in zip(axes, rows):
        ax.imshow(_GEN[c](rng, lvl, params), origin="lower", aspect="auto", cmap="magma")
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
    mode = "L-const (DoWTS constant-rate)" if params.linear_const else "L-ramp (as submitted)"
    fig.suptitle(f"Four-class DoW heat-maps (leech = low-rate, flood = high-rate)  --  {mode}",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)
    print("saved", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=300)
    ap.add_argument("--out", default="data.npz")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    # --- generator parameters (defaults reproduce the original generator) ---
    ap.add_argument("--peak-hour", type=float, default=13.0, help="diurnal peak hour")
    ap.add_argument("--width", type=float, default=3.5, help="diurnal spread")
    ap.add_argument("--amp", type=float, default=30.0, help="base amplitude")
    ap.add_argument("--weekend", type=float, default=0.7, help="weekend factor")
    ap.add_argument("--leech-scale", type=float, default=1.0, help="multiplier on leech rates")
    ap.add_argument("--flood-scale", type=float, default=1.0, help="multiplier on flood rates")
    ap.add_argument("--linear-const", action="store_true",
                    help="linear class as a load-matched constant-rate lift from a random "
                         "onset (DoWTS-faithful) instead of a ramp in the hourly rate")
    args = ap.parse_args()

    params = GenParams(peak_hour=args.peak_hour, width=args.width, amp=args.amp,
                       weekend=args.weekend, leech_scale=args.leech_scale,
                       flood_scale=args.flood_scale, linear_const=args.linear_const)

    if args.preview:
        save_preview(params=params); return

    X, y, inten = generate_dataset(args.per_class, args.seed, params)
    np.savez_compressed(args.out, X=X, y=y, intensity=inten,
                        class_names=np.array(CLASS_NAMES))
    print(f"saved {args.out}: X={X.shape} classes={CLASS_NAMES} "
          f"counts={np.bincount(y).tolist()} "
          f"intensity(0/1/2)={np.bincount(inten).tolist()}")
    print(f"  params: peak_hour={params.peak_hour} width={params.width} "
          f"amp={params.amp} weekend={params.weekend} "
          f"leech_scale={params.leech_scale} flood_scale={params.flood_scale} "
          f"linear_const={params.linear_const}")


if __name__ == "__main__":
    main()