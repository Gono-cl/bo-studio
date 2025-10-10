from __future__ import annotations

from typing import Callable, List, Tuple
import numpy as np


def forrester_1d(x: float, noise: float = 0.0, rng: np.random.Generator | None = None) -> float:
    """
    Classic 1D Forrester function on [0, 1]. Higher is better here.
    f(x) = ((6x-2)^2) * sin(12x-4)
    Optional Gaussian noise.
    """
    x = float(x)
    f = ((6 * x - 2) ** 2) * np.sin(12 * x - 4)
    if noise > 0:
        rng = rng or np.random.default_rng(42)
        f = float(f + rng.normal(0.0, noise))
    return float(f)


def so_demo_space(name: str) -> List[Tuple[str, float, float, str, str]]:
    """Return variable definitions for a named single‑objective demo."""
    name = name.lower()
    if name == "forrester":
        return [("x", 0.0, 1.0, "", "continuous")]
    # default fallback
    return [("x", 0.0, 1.0, "", "continuous")]


def so_demo_eval(name: str) -> Callable[[list], float]:
    name = name.lower()
    if name == "forrester":
        return lambda row: forrester_1d(row[0])
    return lambda row: forrester_1d(row[0])

