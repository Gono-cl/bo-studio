"""
Scalarization helpers for multiobjective (maximization).
"""

from __future__ import annotations

from typing import List, Sequence
import numpy as np


def sample_dirichlet_weights(m: int, k: int, seed: int | None = 42) -> np.ndarray:
    """Sample k weight vectors on the m-simplex (non-negative, sum=1)."""
    rng = np.random.default_rng(seed)
    W = rng.dirichlet(alpha=np.ones(m), size=k)
    return W


def simplex_grid_weights(m: int, k: int) -> np.ndarray:
    """Rough grid of weights by normalizing random points (deterministic grid is complex)."""
    return sample_dirichlet_weights(m, k, seed=0)


def weighted_sum(y: np.ndarray, w: np.ndarray) -> float:
    """Weighted sum scalarization (maximize)."""
    return float(np.dot(w, y))


def tchebycheff(y: np.ndarray, w: np.ndarray, z: np.ndarray) -> float:
    """Chebyshev scalarization (maximize): maximize -max_i w_i*(z_i - y_i)."""
    w = np.asarray(w)
    y = np.asarray(y)
    z = np.asarray(z)
    return float(-np.max(w * (z - y)))

