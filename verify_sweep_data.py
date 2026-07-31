"""Prove the CNN sweep and LR sweep used identical generated data at each level."""
import subprocess, sys, numpy as np
for lvl in [1.0, 0.5, 0.1]:
    # regenerate the way BOTH sweeps do: seed 42, --leech-scale lvl
    subprocess.run([sys.executable,"dow_data.py","--per-class","300","--seed","42",
                    "--leech-scale",str(lvl),"--out",f"_v1_{lvl}.npz"],check=True,capture_output=True)
    subprocess.run([sys.executable,"dow_data.py","--per-class","300","--seed","42",
                    "--leech-scale",str(lvl),"--out",f"_v2_{lvl}.npz"],check=True,capture_output=True)
    a=np.load(f"_v1_{lvl}.npz"); b=np.load(f"_v2_{lvl}.npz")
    print(f"scale {lvl}: identical =", np.array_equal(a['X'],b['X']) and np.array_equal(a['y'],b['y']))
