"""
Initial design generators for manual optimization.

Supports:
- random: uniform sampling within each variable's domain
- lhs: Latin Hypercube Sampling for continuous variables; uniform for categoricals
- halton: low-discrepancy Halton sequence mapped to variable domains
- maximin_lhs: choose the LHS design maximizing the minimum pairwise distance
"""

from __future__ import annotations

from typing import List, Sequence, Any
import numpy as np


def _lhs_unit(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """Latin hypercube in [0,1]^d with n points."""
    H = np.empty((n, d), dtype=float)
    for j in range(d):
        perm = rng.permutation(n)
        H[:, j] = (perm + rng.random(n)) / n
    return H


def _transform_point(u: Sequence[float], variables) -> List[Any]:
    """Map a unit-cube point to the mixed space defined by variables."""
    out: List[Any] = []
    for ui, (name, v1, v2, _unit, vtype) in zip(u, variables):
        if vtype == "continuous":
            lo, hi = float(v1), float(v2)
            out.append(lo + ui * (hi - lo))
        else:
            cats = list(v1)
            k = len(cats)
            idx = min(int(np.floor(ui * k)), k - 1)
            out.append(cats[idx])
    return out


def random_design(variables, n: int, seed: int | None = None) -> List[List[Any]]:
    rng = np.random.default_rng(seed)
    d = len(variables)
    U = rng.random((n, d))
    return [_transform_point(U[i], variables) for i in range(n)]


def lhs_design(variables, n: int, seed: int | None = None) -> List[List[Any]]:
    rng = np.random.default_rng(seed)
    d = len(variables)
    U = _lhs_unit(n, d, rng)
    return [_transform_point(U[i], variables) for i in range(n)]


def _primes(k: int) -> List[int]:
    """Return first k primes (simple sieve)."""
    primes: List[int] = []
    num = 2
    while len(primes) < k:
        is_prime = True
        for p in primes:
            if p * p > num:
                break
            if num % p == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(num)
        num += 1
    return primes


def _radical_inverse(i: int, base: int) -> float:
    """Van der Corput radical inverse in given base (i >= 0)."""
    f = 1.0
    r = 0.0
    while i > 0:
        f /= base
        r += f * (i % base)
        i //= base
    return r


def halton_design(variables, n: int, seed: int | None = None) -> List[List[Any]]:
    """Halton sequence in [0,1]^d mapped to variable domains.
    `seed` acts as a sequence offset (skips `seed` terms).
    """
    d = len(variables)
    bases = _primes(d)
    offset = int(seed or 0)
    U = np.zeros((n, d), dtype=float)
    for t in range(n):
        i = offset + t + 1  # 1-indexed typical halton
        for j, b in enumerate(bases):
            U[t, j] = _radical_inverse(i, b)
    return [_transform_point(U[i], variables) for i in range(n)]


def _pairwise_min_distance(X: np.ndarray) -> float:
    """Compute minimum pairwise Euclidean distance among rows of X."""
    if X.shape[0] < 2:
        return np.inf
    m = X.shape[0]
    min_d = np.inf
    for i in range(m - 1):
        diffs = X[i + 1 :] - X[i]
        dists = np.sqrt(np.sum(diffs * diffs, axis=1))
        dmin = float(np.min(dists))
        if dmin < min_d:
            min_d = dmin
    return min_d


def maximin_lhs_design(variables, n: int, seed: int | None = None, n_candidates: int = 10) -> List[List[Any]]:
    """Generate several LHS candidates and keep the one with maximal min pairwise distance.

    Distance is computed only on continuous dimensions in unit space. If no continuous
    dimensions exist, falls back to standard LHS.
    """
    rng = np.random.default_rng(seed)
    d = len(variables)
    cont_idx = [k for k, (_n, v1, v2, _u, t) in enumerate(variables) if t == "continuous"]
    if len(cont_idx) == 0:
        return lhs_design(variables, n, seed)

    best_U = None
    best_score = -np.inf
    for _ in range(n_candidates):
        U = _lhs_unit(n, d, rng)
        Uc = U[:, cont_idx]
        score = _pairwise_min_distance(Uc)
        if score > best_score:
            best_score = score
            best_U = U
    assert best_U is not None
    return [_transform_point(best_U[i], variables) for i in range(n)]


def generate_initial_points(
    variables,
    n: int,
    method: str = "random",
    seed: int | None = 42,
) -> List[List[Any]]:
    method = (method or "random").lower().replace(" ", "_")
    if method == "lhs":
        return lhs_design(variables, n, seed)
    if method == "halton":
        return halton_design(variables, n, seed)
    if method in ("maximin_lhs", "maximin"):
        return maximin_lhs_design(variables, n, seed)
    # default: random
    return random_design(variables, n, seed)
