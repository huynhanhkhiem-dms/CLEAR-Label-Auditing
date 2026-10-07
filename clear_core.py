"""Core CLEAR audit primitives used by the public reproduction scripts."""
from __future__ import annotations
import numpy as np


def bh_mask(pvalues, q):
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    if m == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p, kind="mergesort")
    ps = p[order]
    ok = np.flatnonzero(ps <= q * np.arange(1, m + 1) / m)
    if len(ok) == 0:
        return np.zeros(m, dtype=bool)
    k = int(ok.max()) + 1
    return p <= q * k / m


def normalize_score(scores):
    s = np.asarray(scores, dtype=float)
    lo, hi = float(s.min()), float(s.max())
    z = np.zeros_like(s) if hi <= lo else (s - lo) / (hi - lo)
    # Fixed deterministic tie-breaker, applied before the audit draw.
    return z + np.arange(len(z), dtype=float) * 1e-12


def sample_srs(n_records, n_verify, rng):
    n = min(int(n_verify), int(n_records) - 2)
    if n < 2:
        raise ValueError("verification budget too small")
    return np.sort(rng.choice(n_records, size=n, replace=False))


def clear_pvalues_from_cal(scores, error_truth, calibration):
    """Conformal p-values on the complement of a fixed verification sample."""
    s = normalize_score(scores)
    err = np.asarray(error_truth, dtype=int)
    cal = np.asarray(calibration, dtype=int)
    mask = np.ones(len(s), dtype=bool)
    mask[cal] = False
    test = np.flatnonzero(mask)
    # Nonconformity used in the reported experiments.
    v_cal = 2.0 * err[cal] - s[cal]
    v_test = -s[test]
    counts = np.searchsorted(np.sort(v_cal), v_test, side="left")
    p = (1.0 + counts) / (len(cal) + 1.0)
    return test, p


def biased_calibration(scores, n_verify, rng, strength=4.0):
    """Deliberately invalid stress test: preferentially verify low-suspicion items."""
    s = normalize_score(scores)
    w = np.exp(-float(strength) * s)
    w /= w.sum()
    return np.sort(rng.choice(len(s), size=int(n_verify), replace=False, p=w))


def evaluate_selection(selected, eligible, error_truth):
    selected = np.asarray(selected, dtype=int)
    eligible = np.asarray(eligible, dtype=int)
    err = np.asarray(error_truth, dtype=int)
    r = len(selected)
    false = int(np.sum(err[selected] == 0)) if r else 0
    true = int(np.sum(err[selected] == 1)) if r else 0
    n_errors = int(np.sum(err[eligible] == 1))
    return {
        "R": int(r),
        "false": false,
        "true": true,
        "FDP": float(false / max(1, r)),
        "power": float(true / max(1, n_errors)),
    }
