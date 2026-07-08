import numpy as np
d = np.load("data.npz", allow_pickle=True)
X = d["X"].reshape(len(d["X"]), -1)      # flatten each image to a vector
# exact-duplicate check across the WHOLE dataset
seen, dups = set(), 0
for i, row in enumerate(X):
    key = row.tobytes()
    if key in seen: dups += 1
    seen.add(key)
print(f"total samples: {len(X)}")
print(f"exact duplicate images: {dups}")
# near-duplicate sanity: how many pairs are identical after rounding
Xr = np.round(X, 4)
uniq = len({r.tobytes() for r in Xr})
print(f"unique images (rounded 4dp): {uniq} / {len(X)}")