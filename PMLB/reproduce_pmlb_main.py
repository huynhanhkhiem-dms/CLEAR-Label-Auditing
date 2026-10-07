#!/usr/bin/env python3
"""Compact reproduction of the main PMLB CLEAR experiment.

This release script reproduces the principal CLEAR and score-biased-verification
rows used for the paper's headline PMLB comparison. The frozen 55-dataset list is
read from dataset_manifest.csv. Secondary/comparator analyses are documented by
archived compact summaries and the full internal release runner used for QA.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from numpy.random import default_rng
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from pmlb import fetch_data

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clear_core import bh_mask, sample_srs, clear_pvalues_from_cal, biased_calibration, evaluate_selection

NOISE_MECHANISMS = ["symmetric", "asymmetric", "instance_dependent"]
NOISE_LEVELS = [0.10, 0.20]
Q_LEVELS = [0.05, 0.10, 0.20]
SEEDS = [11, 23, 37, 53, 71]
VERIFY = 200


def base_seed(seed, mechanism, eta):
    return int(seed * 1000003 + int(round(eta * 1000)) * 101 + NOISE_MECHANISMS.index(mechanism) * 10007)


def audit_seed(seed, mechanism, eta):
    return int(seed * 99991 + int(round(eta * 1000)) * 1237 + NOISE_MECHANISMS.index(mechanism) * 18013)


def inject_noise(X, y, mechanism, eta, rng):
    y = np.asarray(y).copy()
    k = len(np.unique(y))
    noisy = y.copy()
    n = len(y)
    if mechanism == "symmetric":
        flip = rng.random(n) < eta
        for i in np.flatnonzero(flip):
            choices = [c for c in range(k) if c != y[i]]
            noisy[i] = int(rng.choice(choices))
    elif mechanism == "asymmetric":
        flip = rng.random(n) < eta
        noisy[flip] = (y[flip] + 1) % k
    elif mechanism == "instance_dependent":
        xn = StandardScaler().fit_transform(np.asarray(X, float))
        w = rng.normal(size=(xn.shape[1], k))
        logits = xn @ w
        logits[np.arange(n), y] = -np.inf
        alt = np.argmax(logits, axis=1)
        difficulty = np.max(np.where(np.isfinite(logits), logits, -1e9), axis=1)
        z = (difficulty - np.median(difficulty)) / (np.std(difficulty) + 1e-8)
        raw = expit(z)
        probs = np.clip(raw * (eta / (raw.mean() + 1e-12)), 0, 0.95)
        if probs.mean() > 0:
            probs = np.clip(probs * eta / probs.mean(), 0, 0.95)
        flip = rng.random(n) < probs
        noisy[flip] = alt[flip]
    else:
        raise ValueError(mechanism)
    return noisy, (noisy != y).astype(int)


def suspicion_scores(X, y_noisy, seed):
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, solver="lbfgs"))
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    probs = cross_val_predict(model, X, y_noisy, cv=cv, method="predict_proba", n_jobs=3)
    classes = np.unique(y_noisy)
    c2i = {c: i for i, c in enumerate(classes)}
    idx = np.array([c2i[v] for v in y_noisy], dtype=int)
    return 1.0 - probs[np.arange(len(y_noisy)), idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/pmlb_main.csv")
    args = ap.parse_args()
    manifest = pd.read_csv(Path(__file__).with_name("dataset_manifest.csv"))
    rows = []
    for dname in manifest["dataset"].astype(str):
        frame = fetch_data(dname, return_X_y=False)
        X = frame.drop(columns=["target"]).to_numpy(dtype=float)
        y = LabelEncoder().fit_transform(frame["target"].to_numpy())
        for seed in SEEDS:
            Xtr, _, ytrue, _ = train_test_split(X, y, test_size=0.20, stratify=y, random_state=seed)
            for mechanism in NOISE_MECHANISMS:
                for eta in NOISE_LEVELS:
                    noisy, truth = inject_noise(Xtr, ytrue, mechanism, eta, default_rng(base_seed(seed, mechanism, eta)))
                    score = suspicion_scores(Xtr, noisy, seed)
                    cal = sample_srs(len(noisy), VERIFY, default_rng(audit_seed(seed, mechanism, eta)))
                    test, p = clear_pvalues_from_cal(score, truth, cal)
                    bcal = biased_calibration(score, VERIFY, default_rng(audit_seed(seed, mechanism, eta) + 424242))
                    btest, bp = clear_pvalues_from_cal(score, truth, bcal)
                    for q in Q_LEVELS:
                        common = {"dataset": dname, "seed": seed, "mechanism": mechanism, "eta": eta, "q": q}
                        rows.append({**common, "method": "CLEAR-pooled", **evaluate_selection(test[bh_mask(p, q)], test, truth)})
                        rows.append({**common, "method": "CLEAR-biased-gold", **evaluate_selection(btest[bh_mask(bp, q)], btest, truth)})
    out = pd.DataFrame(rows)
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(p, index=False)
    summary = out.groupby(["method", "q"], as_index=False).agg(FDR=("FDP", "mean"), power=("power", "mean"), mean_R=("R", "mean"), evaluations=("FDP", "size"), datasets=("dataset", "nunique"))
    summary.to_csv(p.with_name(p.stem + "_summary.csv"), index=False)
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()
