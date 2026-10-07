#!/usr/bin/env python3
"""Fast consistency check for the headline values reported in the manuscript.

This script verifies the compact summary artifacts distributed with the repository.
It is not a substitute for rerunning the full PMLB/CIFAR-10N experiments.
"""
from __future__ import annotations
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TOL = 5e-7

def close(a, b, tol=TOL):
    return abs(float(a) - float(b)) <= tol

def get(df, method, q):
    r = df[(df["method"] == method) & (df["q"].round(8) == round(q, 8))]
    if len(r) != 1:
        raise AssertionError(f"Expected one row for {method=} {q=}, got {len(r)}")
    return r.iloc[0]

def main():
    p = pd.read_csv(ROOT / "PMLB" / "pmlb_summary.csv")
    c = pd.read_csv(ROOT / "CIFAR10N" / "primary_summary.csv")
    stats = json.loads((ROOT / "PMLB" / "main_stats.json").read_text())

    r = get(p, "CLEAR-pooled", 0.10)
    assert close(r.FDR, 0.0442261620269325)
    assert close(r.power, 0.2278115665839079)
    assert close(r.mean_R, 137.63030303030305)
    b = get(p, "CLEAR-biased-gold", 0.10)
    assert close(b.FDR, 0.3912531749841789)
    assert close(b.power, 0.6694738439071373)
    assert stats["main_rows"] == 29700 and stats["datasets"] == 55

    cr = get(c, "CLEAR-pooled", 0.10)
    assert close(cr.FDR, 0.0457927528486098)
    assert close(cr.power, 0.1983280644353076)
    wb = pd.read_csv(ROOT / "CIFAR10N" / "worse_summary.csv")
    worse = wb[(wb.label_set == "worse_label") & (wb.method == "CLEAR-pooled") & (wb.q.round(8) == 0.1)].iloc[0]
    assert close(worse.FDR, 0.087363445799484)
    assert close(worse.power, 0.3931288156041715)

    cb = pd.read_csv(ROOT / "CIFAR10N" / "budget_summary.csv")
    b50 = cb[cb.n_verify == 50].iloc[0]
    b800 = cb[cb.n_verify == 800].iloc[0]
    assert close(b50.power, 0.039683051680110874)
    assert close(b800.power, 0.2674548543064279)

    ds = pd.read_csv(ROOT / "PMLB" / "downstream_summary.csv")
    clear = ds[ds.method == "CLEAR-pooled"].iloc[0]
    noisy = ds[ds.method == "noisy"].iloc[0]
    assert close(noisy.accuracy, 0.7752437071133447)
    assert close(clear.accuracy, 0.781841561308736)
    assert close(noisy.balanced_accuracy, 0.692407483254766)
    assert close(clear.balanced_accuracy, 0.7029789957978341)
    assert close(clear.retained_fraction, 0.9651767370020243)

    print("PASS: compact artifacts reproduce the checked manuscript headline values.")

if __name__ == "__main__":
    main()
