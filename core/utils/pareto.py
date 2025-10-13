"""
Pareto front utilities (maximization).
"""

from __future__ import annotations

from typing import Iterable, List, Sequence
import numpy as np


def is_nondominated(points: np.ndarray) -> np.ndarray:
    """
    Boolean mask for non-dominated points under maximization.
    points: (n, m) array of objective values to maximize.
    """
    n = points.shape[0]
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not mask[i]:
            continue
        # A point is dominated if any other point is >= in all objectives and > in at least one
        p = points[i]
        others = np.delete(points, i, axis=0)
        if others.size == 0:
            continue
        ge_all = (others >= p).all(axis=1)
        gt_any = (others > p).any(axis=1)
        dominated_by_any = (ge_all & gt_any).any()
        if dominated_by_any:
            mask[i] = False
    return mask


def pareto_front_indices(points: np.ndarray) -> List[int]:
    """Indices of non-dominated points, sorted by a simple tie-break (sum)."""
    mask = is_nondominated(points)
    idx = np.where(mask)[0].tolist()
    # optional: sort by sum of objectives descending for stable display
    idx.sort(key=lambda i: float(points[i].sum()), reverse=True)
    return idx

