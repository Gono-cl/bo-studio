from __future__ import annotations

from typing import List, Tuple, Dict, Any
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


def chem_yield_mixed(
    temperature: float,
    catalyst_loading: float,
    time_h: float,
    solvent: str,
    catalyst_metal: str,
    noise: float = 0.0,
    rng: np.random.Generator | None = None,
) -> float:
    """
    Mixed-variable synthetic chemistry yield with both continuous and categorical effects.

    Inputs:
      - temperature (C): continuous
      - catalyst_loading (0-1): continuous
      - time_h (hours): continuous
      - solvent: categorical {MeOH, THF, Toluene, DMF, DMSO}
      - catalyst_metal: categorical {Pd, Cu, Ni}

    Shape:
      - Continuous core: smooth Gaussian-like surface with metal-specific optimal temperature,
        optimal catalyst loading (~0.5), and optimal time (~3 h).
      - Interactions: temperature-loading ridge; diminishing returns at very high loading/time.
      - Categoricals: additive offsets and selective synergies (metal-solvent combinations).

    Returns an approximate percentage-like yield on [0, 100], clipped.
    """
    # Metal-specific optimal temperature (C) and widths
    t_opt_map: Dict[str, float] = {"Pd": 80.0, "Cu": 90.0, "Ni": 100.0}
    t_width_map: Dict[str, float] = {"Pd": 16.0, "Cu": 15.0, "Ni": 17.0}
    t_opt = t_opt_map.get(catalyst_metal, 90.0)
    t_w = t_width_map.get(catalyst_metal, 16.0)

    # Normalize continuous variables around optima
    t = (temperature - t_opt) / t_w
    c = (catalyst_loading - 0.5) / 0.2
    r = (time_h - 3.0) / 1.2

    # Base smooth surface (Gaussian wells)
    base = np.exp(-(t**2)) * np.exp(-(c**2)) * np.exp(-(r**2))

    # Interactions and diminishing returns
    t_c_ridge = 0.25 * np.exp(-((t - 0.7 * c) ** 2))
    high_load_penalty = -0.06 if catalyst_loading > 0.8 else 0.0
    long_time_penalty = -0.04 if time_h > 5.0 else 0.0

    # Categorical main effects (offsets)
    solvent_offset: Dict[str, float] = {
        "DMF": 8.0,
        "DMSO": 5.0,
        "THF": 3.0,
        "Toluene": 0.0,
        "MeOH": -5.0,
    }
    metal_offset: Dict[str, float] = {"Pd": 6.0, "Cu": 4.0, "Ni": 0.0}

    # Selected solvent-metal synergies (bonus/penalty)
    pair_bonus: Dict[Tuple[str, str], float] = {
        ("Pd", "DMF"): 6.0,
        ("Ni", "DMSO"): 4.0,
        ("Cu", "MeOH"): -6.0,
    }
    cat_bonus = pair_bonus.get((catalyst_metal, solvent), 0.0)

    # Extra viscosity penalty for very short residence in viscous solvents
    viscous = solvent in {"DMSO", "DMF"}
    short_time_penalty = -0.05 if viscous and time_h < 0.5 else 0.0

    # Combine components; scale to percentage-like
    y = 100.0 * (0.65 * base + t_c_ridge) / 1.4
    y += solvent_offset.get(solvent, 0.0) + metal_offset.get(catalyst_metal, 0.0) + cat_bonus
    y += 100.0 * (high_load_penalty + long_time_penalty + short_time_penalty)

    # Soft clipping then hard bounds
    y = float(np.clip(y, 0.0, 100.0))
    if noise > 0:
        rng = rng or np.random.default_rng(42)
        y = float(np.clip(y + rng.normal(0.0, noise), 0.0, 100.0))
    return y


def chem_demo_space_mixed() -> List[Tuple[str, Any, Any, str, str]]:
    """
    Mixed demonstration space with continuous and categorical variables.
    Format per variable: (name, v1, v2, unit, type)
      - For continuous: v1=min, v2=max
      - For categorical: v1=list of categories, v2=None
    """
    return [
        ("Temperature", 20.0, 120.0, "C", "continuous"),
        ("Catalyst_Loading", 0.0, 1.0, "fraction", "continuous"),
        ("Time", 0.1, 8.0, "h", "continuous"),
        ("Solvent", ["MeOH", "THF", "Toluene", "DMF", "DMSO"], None, "", "categorical"),
        ("Catalyst_Metal", ["Pd", "Cu", "Ni"], None, "", "categorical"),
    ]


def chem_eval_row_mixed(row: list) -> float:
    """Evaluate a row matching chem_demo_space_mixed order."""
    temperature = float(row[0])
    catalyst_loading = float(row[1])
    time_h = float(row[2])
    solvent = str(row[3])
    metal = str(row[4])
    return chem_yield_mixed(temperature, catalyst_loading, time_h, solvent, metal)


def chem_demo_space(mode: str = "basic") -> List[Tuple[str, float, float, str, str]]:
    # Existing implementation moved below; kept for backward compatibility
    if mode == "extended":
        return [
            ("Temperature", 20.0, 120.0, "��C", "continuous"),
            ("Catalyst", 0.0, 1.0, "fraction", "continuous"),
            ("Pressure", 1.0, 20.0, "bar", "continuous"),
            ("Residence", 0.1, 10.0, "h", "continuous"),
            ("Polarity", 0.0, 1.0, "", "continuous"),
        ]
    if mode == "mixed":
        # Provide the mixed-variable demonstration space
        return chem_demo_space_mixed()  # type: ignore[return-value]
    return [
        ("Temperature", 20.0, 120.0, "��C", "continuous"),
        ("Catalyst", 0.0, 1.0, "fraction", "continuous"),
    ]


def chem_eval_row(row: list, mode: str = "basic") -> float:
    if mode == "extended":
        return chem_yield5(float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]))
    if mode == "mixed":
        return chem_eval_row_mixed(row)
    return chem_yield(float(row[0]), float(row[1]))
