from __future__ import annotations

import numpy as np


def knee_index_2d(points: np.ndarray) -> int | None:
    """
    Simple knee detection for a 2D Pareto curve (maximize both), sorted by x.
    Returns index of the point with maximum distance to the line connecting endpoints.
    """
    if points.shape[0] < 3:
        return None
    # Normalize
    P = points.astype(float)
    P[:, 0] = (P[:, 0] - P[:, 0].min()) / (P[:, 0].ptp() + 1e-12)
    P[:, 1] = (P[:, 1] - P[:, 1].min()) / (P[:, 1].ptp() + 1e-12)
    P = P[np.argsort(P[:, 0])]
    a = P[0]
    b = P[-1]
    ab = b - a
    ab_len = np.linalg.norm(ab) + 1e-12
    # Distance from point to line AB
    dists = np.abs(np.cross(P - a, ab) / ab_len)
    idx = int(np.argmax(dists))
    return idx

