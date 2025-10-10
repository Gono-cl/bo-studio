from __future__ import annotations

from typing import List, Tuple
import numpy as np


def chem_yield(temp: float, catalyst: float, noise: float = 0.0, rng: np.random.Generator | None = None) -> float:
    """
    Simple chemistry-inspired yield function of Temperature (°C) and Catalyst loading (fraction 0-1).
    Peak around T~80°C and Cat~0.6 with interaction; includes gentle ridge and asymmetry.
    """
    t = (temp - 80.0) / 15.0
    c = (catalyst - 0.6) / 0.15
    base = np.exp(-(t**2)) * np.exp(-0.5 * (c**2))
    ridge = 0.2 * np.exp(-((t - 0.8 * c) ** 2))
    y = 100.0 * (0.75 * base + ridge)  # scale to percentage-like
    if noise > 0:
        rng = rng or np.random.default_rng(42)
        y = float(y + rng.normal(0.0, noise))
    return float(y)


def chem_yield5(temp: float, catalyst: float, pressure: float, residence: float, polarity: float,
                noise: float = 0.0, rng: np.random.Generator | None = None) -> float:
    """
    Five-variable chemistry-inspired yield surface.
    Variables:
    - Temperature (°C): optimum ~85
    - Catalyst (0-1): optimum ~0.55
    - Pressure (bar): optimum ~10
    - Residence Time (h): optimum ~3.0
    - Solvent Polarity (0-1): optimum ~0.4
    Includes interactions: T–Cat synergy, Pressure–Residence balance, slight polarity ridge.
    Returns yield on roughly 0–100 scale.
    """
    # Normalize around optima and widths
    t = (temp - 85.0) / 18.0
    c = (catalyst - 0.55) / 0.18
    p = (pressure - 10.0) / 5.0
    r = (residence - 3.0) / 1.2
    s = (polarity - 0.4) / 0.25

    # Base Gaussian wells
    base = np.exp(-(t**2)) * np.exp(-(c**2)) * np.exp(-0.7 * (p**2)) * np.exp(-(r**2)) * np.exp(-0.5 * (s**2))

    # Interactions and ridges
    t_c_synergy = 0.25 * np.exp(-((t - 0.8 * c) ** 2))
    p_r_balance = 0.2 * np.exp(-((p + 0.8 * r) ** 2))
    polarity_ridge = 0.15 * np.exp(-((s - 0.5 * t) ** 2))

    y = 100.0 * (0.6 * base + t_c_synergy + p_r_balance + polarity_ridge)
    if noise > 0:
        rng = rng or np.random.default_rng(42)
        y = float(y + rng.normal(0.0, noise))
    return float(y)


def chem_demo_space(mode: str = "basic") -> List[Tuple[str, float, float, str, str]]:
    if mode == "extended":
        return [
            ("Temperature", 20.0, 120.0, "°C", "continuous"),
            ("Catalyst", 0.0, 1.0, "fraction", "continuous"),
            ("Pressure", 1.0, 20.0, "bar", "continuous"),
            ("Residence", 0.1, 10.0, "h", "continuous"),
            ("Polarity", 0.0, 1.0, "", "continuous"),
        ]
    return [
        ("Temperature", 20.0, 120.0, "°C", "continuous"),
        ("Catalyst", 0.0, 1.0, "fraction", "continuous"),
    ]


def chem_eval_row(row: list, mode: str = "basic") -> float:
    if mode == "extended":
        return chem_yield5(float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]))
    return chem_yield(float(row[0]), float(row[1]))
