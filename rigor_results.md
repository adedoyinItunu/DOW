# Core-Rigor Results — 8 July 2026

## Step 1: Seed repetition (E1 baseline, 10 seeds)
Seed:     0      1      2      3      4      5      6      7      8      9
Macro-F1: 0.955  0.972  0.913  0.983  1.000  0.978  0.994  1.000  0.942  0.920

Mean ± std: 0.966 ± 0.030 (range 0.913–1.000)
Errors concentrated at linear/normal boundary (e.g. seed 0: linear recall 0.822).

## Step 2: Leakage / duplicate check
total samples: 1200
exact duplicate images: 0
unique images (rounded 4dp): 1200 / 1200
Split 840/180/180 fixed before training; no overlap.

## Step 3: Robustness across generator size (seed 0)
per-class 150 (600 imgs):  Macro-F1 0.871
per-class 300 (1200 imgs): Macro-F1 0.955
per-class 500 (2000 imgs): Macro-F1 0.990
Clean learning curve; 300/class retained as performance/cost balance.
