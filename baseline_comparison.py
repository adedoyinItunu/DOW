"""
baseline_comparison.py  --  is the CNN actually doing any work?
==============================================================
Trivial baselines to contextualise the CNN's performance, answering the obvious
examiner question. Reports macro-F1 for:
    1. Majority-class      -- always predict the most common class
    2. Logistic regression -- on the raw flattened 24x30 pixels (720 features)
    3. (reference) the CNN's own figures, printed for comparison

Uses the SAME split as the CNN (load_and_split), so the comparison is fair.

    python baseline_comparison.py --data data.npz --seed 0
"""
import argparse
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report

from dow_data import load_and_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data.npz")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    (Xtr, ytr, _), (_, _, _), (Xte, yte, _), names = load_and_split(args.data, args.seed)
    # flatten images to vectors for the classical baselines
    Xtr_f = Xtr.reshape(len(Xtr), -1)
    Xte_f = Xte.reshape(len(Xte), -1)

    print("=============== BASELINE COMPARISON ===============")
    print(f"data: {args.data}  seed: {args.seed}")
    print(f"train {len(ytr)}  test {len(yte)}  classes {names}\n")

    # 1. majority class
    maj = DummyClassifier(strategy="most_frequent").fit(Xtr_f, ytr)
    p_maj = maj.predict(Xte_f)
    f_maj = f1_score(yte, p_maj, average="macro")
    print(f"1. Majority-class baseline   macro-F1: {f_maj:.3f}")

    # 2. logistic regression on raw pixels
    lr = LogisticRegression(max_iter=2000, multi_class="multinomial")
    lr.fit(Xtr_f, ytr)
    p_lr = lr.predict(Xte_f)
    f_lr = f1_score(yte, p_lr, average="macro")
    print(f"2. Logistic reg (raw pixels) macro-F1: {f_lr:.3f}")
    print("\n   Logistic-regression per-class:")
    print(classification_report(yte, p_lr, target_names=names, digits=3, zero_division=0))

    print("Compare against the CNN ten-seed macro-F1 on the SAME dataset")
    print("(perclass_across_seeds.py): 0.759 +/- 0.169 L-ramp, 0.877 +/- 0.126 L-const. If the CNN")
    print("clearly exceeds both baselines, it is learning structure the trivial")
    print("models cannot; if logistic regression is close, the task is largely")
    print("linearly separable and the CNN's added value is modest.")


if __name__ == "__main__":
    main()
