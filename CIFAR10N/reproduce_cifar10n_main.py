#!/usr/bin/env python3
"""Compact reproduction of the main CIFAR-10N CLEAR audits.

Run the setup scripts first so ROOT contains data/CIFAR-10_human.pt,
data/alignment.json and features/resnet18_train_embeddings.npy.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
from numpy.random import default_rng
from scipy.special import softmax
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clear_core import bh_mask, sample_srs, clear_pvalues_from_cal, biased_calibration, evaluate_selection

LABEL_SETS = ["aggre_label", "worse_label", "random_label1", "random_label2", "random_label3"]
Q_LEVELS = [0.05, 0.10, 0.20]
AUDIT_SEEDS = list(range(1000, 1100))
VERIFY = 200


def l2_normalize(X):
    X = np.asarray(X, np.float32)
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


def centroid_probabilities(train_X, train_y, eval_X, temperature=0.10):
    train_X, eval_X = l2_normalize(train_X), l2_normalize(eval_X)
    cent = np.zeros((10, train_X.shape[1]), np.float32)
    present = np.zeros(10, bool)
    for c in range(10):
        idx = np.flatnonzero(train_y == c)
        if len(idx):
            cent[c] = train_X[idx].mean(axis=0); present[c] = True
    cent = l2_normalize(cent)
    sims = eval_X @ cent.T
    sims[:, ~present] = -1e6
    return softmax(sims / temperature, axis=1).astype(np.float32)


def oof_probs(X, y, seed=20260819):
    probs = np.zeros((len(y), 10), np.float32)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, va in cv.split(X, y):
        probs[va] = centroid_probabilities(X[tr], y[tr], X[va])
    return probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    root = Path(args.root)
    align = json.loads((root / "data" / "alignment.json").read_text())
    if not align.get("passed") or align.get("n") != 50000:
        raise RuntimeError("CIFAR-10N alignment check failed")
    X = np.load(root / "features" / "resnet18_train_embeddings.npy", mmap_mode="r")
    labels = torch.load(root / "data" / "CIFAR-10_human.pt", map_location="cpu", weights_only=False)
    clean = np.asarray(labels["clean_label"], int)
    rows = []
    for label_set in LABEL_SETS:
        noisy = np.asarray(labels[label_set], int)
        probs = oof_probs(X, noisy)
        score = 1.0 - probs[np.arange(len(noisy)), noisy]
        truth = (noisy != clean).astype(int)
        for seed in AUDIT_SEEDS:
            cal = sample_srs(len(noisy), VERIFY, default_rng(seed))
            test, p = clear_pvalues_from_cal(score, truth, cal)
            bcal = biased_calibration(score, VERIFY, default_rng(seed + 400000), strength=4.0)
            btest, bp = clear_pvalues_from_cal(score, truth, bcal)
            for q in Q_LEVELS:
                common = {"dataset": "CIFAR-10N", "label_set": label_set, "audit_seed": seed, "q": q, "noise_rate": float(truth.mean())}
                rows.append({**common, "method": "CLEAR-pooled", **evaluate_selection(test[bh_mask(p, q)], test, truth)})
                rows.append({**common, "method": "CLEAR-biased-gold", **evaluate_selection(btest[bh_mask(bp, q)], btest, truth)})
    out = pd.DataFrame(rows)
    dest = Path(args.out) if args.out else root / "outputs" / "cifar10n_reproduction.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest, index=False)
    summary = out.groupby(["label_set", "q", "method"], as_index=False).agg(FDR=("FDP", "mean"), power=("power", "mean"), mean_R=("R", "mean"), audits=("audit_seed", "nunique"), noise_rate=("noise_rate", "first"))
    summary.to_csv(dest.with_name(dest.stem + "_summary.csv"), index=False)
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()
