from __future__ import annotations

import numpy as np


def hypervolume_2d(points: np.ndarray, ref: tuple[float, float]) -> float:
    """Compute 2D hypervolume (maximize) w.r.t. reference point.
    Assumes points are non-dominated and unique.
    """
    if points.size == 0:
        return 0.0
    P = points[np.argsort(points[:, 0])]  # sort by x
    hv = 0.0
    prev_x = ref[0]
    prev_y = ref[1]
    # integrate rectangles from ref to Pareto steps
    for x, y in P:
        hv += max(0.0, (x - prev_x)) * max(0.0, (y - ref[1]))
        prev_x = x
    return float(hv)

