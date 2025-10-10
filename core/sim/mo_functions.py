from __future__ import annotations

from typing import Callable, List, Tuple
import numpy as np


def schaffer_n1(x: float) -> tuple[float, float]:
    """Schaffer N.1 (2 objectives to minimize traditionally). We flip to maximize by returning negatives.
    f1 = x^2, f2 = (x-2)^2  (minimize) => return (-f1, -f2) to treat as maximize.
    Domain: x in [-5, 5]
    """
    f1 = x * x
    f2 = (x - 2.0) ** 2
    return -float(f1), -float(f2)


def mo_demo_space(name: str) -> List[Tuple[str, float, float, str, str]]:
    name = name.lower()
    if name == "schaffer":
        return [("x", -5.0, 5.0, "", "continuous")]
    return [("x", -5.0, 5.0, "", "continuous")]


def mo_demo_eval(name: str) -> Callable[[list], tuple[float, float]]:
    name = name.lower()
    if name == "schaffer":
        return lambda row: schaffer_n1(row[0])
    return lambda row: schaffer_n1(row[0])

