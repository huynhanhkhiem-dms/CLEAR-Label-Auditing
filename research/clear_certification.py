"""Research prototype: simultaneous finite-population certificates for frozen ranked audit lists.
This is not a novel FDR theorem or a replacement for the original CLEAR method.
"""
from dataclasses import dataclass
import numpy as np
from scipy.stats import hypergeom

def hg_upper_total(population, draws, observed, alpha):
    """Invert the exact hypergeometric lower tail for a one-sided upper bound."""
    N, n, x = int(population), int(draws), int(observed)
    if not (0 <= n <= N and 0 <= x <= n and 0 < alpha < 1):
        raise ValueError("Invalid N, n, x or alpha")
    if N == 0 or n == N:
        return x
    lo, hi = x, N-(n-x)
    while lo < hi:
        mid = (lo+hi+1)//2
        if hypergeom.cdf(x, N, mid, n) >= alpha:
            lo = mid
        else:
            hi = mid-1
    return lo

def fixed_rank_strata(scores, n_strata=8):
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("scores must be finite vector")
    if not 1 <= n_strata <= len(scores):
        raise ValueError("invalid strata count")
    order = np.argsort(-scores, kind="stable")
    strata = np.empty(len(scores), dtype=int)
    for h, block in enumerate(np.array_split(order, n_strata)):
        strata[block] = h
    return order, strata

def stratified_draw(strata, allocation, rng):
    groups = np.unique(strata)
    if len(groups) != len(allocation):
        raise ValueError("allocation length != number of strata")
    draws = []
    for h, n in zip(groups, allocation):
        idx = np.flatnonzero(strata == h)
        if not (0 <= int(n) <= len(idx)):
            raise ValueError("invalid allocation")
        draws.append(rng.choice(idx, size=int(n), replace=False))
    return np.sort(np.concatenate(draws))

@dataclass
class Certificate:
    cut: int
    returned_size: int
    upper_false: int
    upper_fdp: float
    observed_verified_false: int
    selected_unverified: np.ndarray

def simultaneous_certificates(scores, verified, observed_error, strata, cuts, delta=0.05):
    """Valid after label-dependent choice among the precommitted candidate cuts.
    Only verification outcomes, not withheld ground-truth labels, enter this API.
    Assumes frozen scores/strata/cuts/allocation and exact adjudication.
    """
    scores = np.asarray(scores, dtype=float)
    verified = np.asarray(verified, dtype=int)
    observed_error = np.asarray(observed_error, dtype=int)
    strata = np.asarray(strata, dtype=int)
    N = len(scores)
    if not (len(strata) == N and len(observed_error) == len(verified)):
        raise ValueError("inconsistent lengths")
    if any(int(k) != k or not 1 <= k <= N for k in cuts):
        raise ValueError("invalid cut")
    if len(set(cuts)) != len(cuts):
        raise ValueError("duplicate cuts")
    if not np.isin(observed_error, [0, 1]).all():
        raise ValueError("observed_error must be binary")
    if not 0 < delta < 1:
        raise ValueError("invalid delta")
    if not np.array_equal(np.unique(verified), np.sort(verified)):
        raise ValueError("verified must be unique")
    if np.any((verified < 0) | (verified >= N)):
        raise ValueError("verified outside repository")
    order = np.argsort(-scores, kind="stable")
    rank = np.empty(N, dtype=int)
    rank[order] = np.arange(N)
    groups = np.unique(strata)
    alpha = delta / (len(groups) * len(cuts))
    ver_mask = np.zeros(N, dtype=bool)
    ver_mask[verified] = True
    is_correct_verified = np.zeros(N, dtype=bool)
    is_correct_verified[verified] = observed_error == 0
    results = []
    for cut in cuts:
        prefix = rank < cut
        unknown = prefix & ~ver_mask
        bound = 0
        for h in groups:
            within = strata == h
            n = int(np.sum(within & ver_mask))
            y = int(np.sum(within & prefix & ver_mask & is_correct_verified))
            U = hg_upper_total(int(np.sum(within)), n, y, alpha)
            bound += min(int(np.sum(within & unknown)), max(0, U-y))
        size = int(unknown.sum())
        results.append(Certificate(int(cut), size, int(bound),
                    float(bound/size) if size else 0.0,
                    int(np.sum(prefix & ver_mask & is_correct_verified)),
                    np.flatnonzero(unknown)))
    return results

def largest_certified(certificates, q=0.10):
    good = [c for c in certificates if c.upper_fdp <= q]
    return max(good, key=lambda c: c.returned_size) if good else None
