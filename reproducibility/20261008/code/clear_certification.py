"""Finite-repository risk certificates for frozen score-ranked audit lists.

Research prototype, October 2026.  This is NOT a replacement for the
original CLEAR FDR guarantee; it controls realized false selection proportion
with high probability under precommitted stratified SRS sampling.
"""
from dataclasses import dataclass
import numpy as np
from scipy.stats import hypergeom


def hg_upper_total(population, draws, observed, alpha):
    """Exact one-sided (1-alpha) upper bound for binary population total.

    Under X~Hypergeom(N,K,n), return max candidate K whose lower tail
    P_K(X<=observed) >= alpha. For a census, the result equals observed.
    """
    N, n, x = int(population), int(draws), int(observed)
    if not (0 <= n <= N and 0 <= x <= n and 0 < alpha < 1):
        raise ValueError('Invalid N, n, x or alpha')
    if N == 0 or n == N:
        return x
    lo, hi = x, N - (n - x)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if hypergeom.cdf(x, N, mid, n) >= alpha:
            lo = mid
        else:
            hi = mid - 1
    return lo


def fixed_rank_strata(scores, n_strata=8):
    """Strata of adjacent frozen-score ranks; IDs 0=most suspicious."""
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError('scores must be finite vector')
    if not 1 <= n_strata <= len(scores):
        raise ValueError('invalid strata count')
    order = np.argsort(-scores, kind='stable')
    strata = np.empty(len(scores), dtype=int)
    for h, block in enumerate(np.array_split(order, n_strata)):
        strata[block] = h
    return order, strata


def stratified_draw(strata, allocation, rng):
    """Independent simple random samples without replacement within strata."""
    groups = np.unique(strata)
    if len(groups) != len(allocation):
        raise ValueError('allocation length != number of strata')
    draws = []
    for h, n in zip(groups, allocation):
        idx = np.flatnonzero(strata == h)
        if not (0 <= int(n) <= len(idx)):
            raise ValueError('invalid allocation')
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


def simultaneous_certificates(scores, verified, observed_error, strata, cuts,
                              delta=0.05):
    """Per-cut simultaneous finite-population FDP bounds.

    observed_error contains outcomes for verified records only, in verified
    order. Unverified correctness is not accepted as a function input.
    Each cut denotes a fixed prefix
    of the frozen ranking, including any verified records in that prefix.
    Bonferroni alpha=delta/(#strata * #cuts) permits data-dependent selection
    among these precommitted cuts after seeing verification outcomes.
    """
    scores = np.asarray(scores, dtype=float)
    verified = np.asarray(verified, dtype=int)
    observed_error = np.asarray(observed_error, dtype=int)
    strata = np.asarray(strata, dtype=int)
    N = len(scores)
    if not (len(strata) == N and len(observed_error) == len(verified)):
        raise ValueError('inconsistent lengths')
    if any(int(k) != k or not 1 <= k <= N for k in cuts):
        raise ValueError('invalid cut')
    if len(set(cuts)) != len(cuts):
        raise ValueError('duplicate cuts')
    if not np.isin(observed_error, [0, 1]).all():
        raise ValueError('truth_error must be binary')
    if not 0 < delta < 1:
        raise ValueError('invalid delta')
    if not np.array_equal(np.unique(verified), np.sort(verified)):
        raise ValueError('verified must be unique')
    if np.any((verified < 0) | (verified >= N)):
        raise ValueError('verified outside repository')
    order = np.argsort(-scores, kind='stable')
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
            # successes = *correct* labels in selected prefix
            y = int(np.sum(within & prefix & ver_mask & is_correct_verified))
            U = hg_upper_total(int(np.sum(within)), n, y, alpha)
            # subtract counted verified correct labels; they cannot be selected
            # also cap by the number of unverified prefix members
            bound += min(int(np.sum(within & unknown)), max(0, U - y))
        size = int(unknown.sum())
        results.append(Certificate(int(cut), size, int(bound),
                     float(bound / size) if size else 0.0,
                     int(np.sum(prefix & ver_mask & is_correct_verified)),
                     np.flatnonzero(unknown)))
    return results


def largest_certified(certificates, q=0.10):
    """Select largest number of unverified records with FDP upper bound <=q.

    The empty set is returned if nothing meets the criterion. Candidate
    cuts must have been fixed before viewing any verification outcomes.
    """
    good = [c for c in certificates if c.upper_fdp <= q]
    return max(good, key=lambda c: c.returned_size) if good else None
