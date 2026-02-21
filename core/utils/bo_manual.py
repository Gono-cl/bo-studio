"""
Utilities to support the manual experiments flow (bound management, optimizers, suggestions).
Extracted from manual_experiments.py to keep the page lean.
"""

from __future__ import annotations

import os
import re
from typing import List, Tuple, Any

import pandas as pd
from skopt.space import Real, Categorical

from core.optimization.bayesian_optimization import StepBayesianOptimizer


def sanitize_name(name: str) -> str:
    name = (name or "").strip() or "manual_experiment"
    return re.sub(r'[^A-Za-z0-9_\- ]+', '_', name)


def list_valid_campaigns(base_dir: str) -> List[str]:
    if not os.path.exists(base_dir):
        return []
    valid: List[str] = []
    for d in sorted(os.listdir(base_dir)):
        p = os.path.join(base_dir, d)
        if os.path.isdir(p) and \
           os.path.exists(os.path.join(p, "manual_data.csv")) and \
           os.path.exists(os.path.join(p, "metadata.json")):
            valid.append(d)
    return valid


def _apply_optimizer_acq_settings(opt: StepBayesianOptimizer, acq_func: str, acq_xi: float, acq_kappa: float) -> None:
    """Best-effort configuration of acquisition hyperparameters on underlying skopt optimizer."""
    sk = getattr(opt, "skopt_optimizer", None) or getattr(opt, "_optimizer", None)
    if sk is None:
        return
    try:
        acq = str(acq_func).upper()
        kwargs = dict(getattr(sk, "acq_func_kwargs", {}) or {})
        if acq in {"EI", "PI"}:
            kwargs["xi"] = float(acq_xi)
        elif acq == "LCB":
            kwargs["kappa"] = float(acq_kappa)
        if hasattr(sk, "acq_func_kwargs"):
            setattr(sk, "acq_func_kwargs", kwargs)
        if hasattr(sk, "xi"):
            setattr(sk, "xi", float(acq_xi))
        if hasattr(sk, "kappa"):
            setattr(sk, "kappa", float(acq_kappa))
    except Exception:
        pass


def safe_build_optimizer(
    space,
    n_initial_points_remaining: int = 0,
    acq_func: str = "EI",
    acq_xi: float = 0.01,
    acq_kappa: float = 1.96,
    random_state: int = 42,
) -> StepBayesianOptimizer:
    """
    Build StepBayesianOptimizer and set underlying skopt.Optimizer knobs.

    This project wraps skopt.Optimizer inside StepBayesianOptimizer at attribute
    `_optimizer` (and property `skopt_optimizer`). We cannot pass
    `n_initial_points` to the wrapper constructor, so we set it directly on the
    underlying optimizer after construction.
    """
    opt = StepBayesianOptimizer(space, acq_func=acq_func, random_state=int(random_state))
    sk = getattr(opt, "skopt_optimizer", None) or getattr(opt, "_optimizer", None)
    try:
        if sk is not None:
            if hasattr(sk, "_n_initial_points"):
                setattr(sk, "_n_initial_points", n_initial_points_remaining)
            if hasattr(sk, "n_initial_points_"):
                setattr(sk, "n_initial_points_", n_initial_points_remaining)
            if hasattr(sk, "acq_func"):
                setattr(sk, "acq_func", acq_func)
    except Exception:
        pass
    _apply_optimizer_acq_settings(opt, acq_func=acq_func, acq_xi=acq_xi, acq_kappa=acq_kappa)
    return opt


def force_model_based(optimizer: StepBayesianOptimizer) -> None:
    """Force next suggest() to be acquisition-driven (no random initials)."""
    try:
        sk = getattr(optimizer, "skopt_optimizer", None) or getattr(optimizer, "_optimizer", None)
        if sk is not None:
            if hasattr(sk, "_n_initial_points"):
                setattr(sk, "_n_initial_points", 0)
            if hasattr(sk, "n_initial_points_"):
                setattr(sk, "n_initial_points_", 0)
    except Exception:
        pass


def unionize_bounds(curr_variables, seeds_df: pd.DataFrame | None):
    """Expand continuous bounds to include seed values; unify categories for categoricals."""
    if seeds_df is None or seeds_df.empty:
        return curr_variables
    new_vars = []
    for (name, v1, v2, unit, vtype) in curr_variables:
        if vtype == "continuous":
            col = seeds_df[name] if name in seeds_df.columns else pd.Series(dtype=float)
            col = pd.to_numeric(col, errors="coerce").dropna()
            lo = min([v1] + (col.tolist() if not col.empty else []))
            hi = max([v2] + (col.tolist() if not col.empty else []))
            new_vars.append((name, float(lo), float(hi), unit, "continuous"))
        else:
            col = seeds_df[name] if name in seeds_df.columns else pd.Series(dtype=object)
            cats = set(v1) | set(col.dropna().astype(str).unique().tolist())
            new_vars.append((name, sorted(list(cats)), None, unit, "categorical"))
    return new_vars


def _in_suggest_space(x, suggest_variables):
    for (val, (name, v1, v2, _unit, vtype)) in zip(x, suggest_variables):
        if vtype == "continuous":
            try:
                fv = float(val)
            except Exception:
                return False
            if not (v1 <= fv <= v2):
                return False
        else:
            if str(val) not in set(map(str, v1)):
                return False
    return True


def _project_to_suggest_space(x, suggest_variables):
    out = []
    for (val, (name, v1, v2, _unit, vtype)) in zip(x, suggest_variables):
        if vtype == "continuous":
            fv = float(val)
            out.append(min(max(fv, v1), v2))
        else:
            cats = list(v1)
            sval = str(val)
            out.append(sval if sval in set(map(str, cats)) else cats[0])
    return out


def _coerce_categorical_to_levels(value: Any, levels: list[Any]) -> Any | None:
    """Return the canonical level object from `levels` matching `value`."""
    if pd.isna(value):
        return None

    # Exact match first (keeps original level type).
    for level in levels:
        if value == level:
            return level

    # String-equivalent match.
    sval = str(value).strip()
    for level in levels:
        if str(level).strip() == sval:
            return level

    # Numeric-equivalent match (e.g., 7 <-> "7", 7.0 <-> "7").
    try:
        fval = float(sval)
        if pd.notna(fval):
            for level in levels:
                try:
                    if float(str(level).strip()) == fval:
                        return level
                except Exception:
                    continue
    except Exception:
        pass

    return None


def coerce_point_to_variables(point: list[Any], variables) -> list[Any] | None:
    """Coerce a point to match variable domain types exactly; return None if invalid."""
    if len(point) != len(variables):
        return None

    coerced: list[Any] = []
    for raw_value, (_name, v1, v2, _unit, vtype) in zip(point, variables):
        if str(vtype).lower() == "continuous":
            num = pd.to_numeric(pd.Series([raw_value]), errors="coerce").iloc[0]
            if pd.isna(num):
                return None
            fv = float(num)
            lo, hi = float(v1), float(v2)
            coerced.append(min(max(fv, lo), hi))
        else:
            levels = list(v1) if isinstance(v1, list) else [v1]
            val = _coerce_categorical_to_levels(raw_value, levels)
            if val is None:
                return None
            coerced.append(val)
    return coerced


def rebuild_optimizer_from_df(
    variables,
    df: pd.DataFrame,
    response_col: str,
    n_initial_points_remaining: int = 0,
    acq_func: str = "EI",
    direction: str = "Maximize",
    acq_xi: float = 0.01,
    acq_kappa: float = 1.96,
    random_state: int = 42,
) -> StepBayesianOptimizer:
    """Build StepBayesianOptimizer on 'variables' (ModelSpace), and observe seeds once."""
    space = []
    for name, v1, v2, _unit, vtype in variables:
        if vtype == "continuous":
            space.append(Real(v1, v2, name=name))
        else:
            space.append(Categorical(v1, name=name))

    opt = safe_build_optimizer(
        space,
        n_initial_points_remaining=n_initial_points_remaining,
        acq_func=acq_func,
        acq_xi=acq_xi,
        acq_kappa=acq_kappa,
        random_state=random_state,
    )

    df = df.copy()
    if response_col not in df.columns:
        raise ValueError(f"Response column '{response_col}' not found in reused data.")
    df[response_col] = pd.to_numeric(df[response_col], errors="coerce")
    df = df.dropna(subset=[response_col])

    # Batch observe to avoid repeated refits (much faster than per-row)
    X_batch = []
    y_batch = []
    maximize = str(direction).lower() != "minimize"
    for _, row in df.iterrows():
        try:
            y = float(row[response_col])
        except (ValueError, TypeError):
            continue
        if pd.notnull(y):
            raw_x = [row.get(name) for name, *_ in variables]
            x = coerce_point_to_variables(raw_x, variables)
            if x is None:
                continue
            X_batch.append(x)
            observed = -y if maximize else y
            y_batch.append(float(observed))

    try:
        if X_batch:
            sk = getattr(opt, "skopt_optimizer", None) or getattr(opt, "_optimizer", None)
            if sk is not None and hasattr(sk, "tell"):
                sk.tell(X_batch, y_batch)
                # keep wrapper history in sync (best-effort)
                try:
                    opt.x_iters.extend(X_batch)
                    opt.y_iters.extend(y_batch)
                except Exception:
                    pass
            else:
                # fallback: per-point observe
                for x, y in zip(X_batch, y_batch):
                    opt.observe(x, y)
    except Exception:
        # As a safety net, fall back to slow path if batch fails
        for x, y in zip(X_batch, y_batch):
            opt.observe(x, y)

    if n_initial_points_remaining == 0:
        force_model_based(opt)
    return opt


def _existing_points_set(manual_variables, manual_data):
    cols = [name for name, *_ in manual_variables]
    s = set()
    for row in manual_data:
        tup = tuple(row.get(c) for c in cols)
        s.add(tup)
    return s


def next_unique_suggestion(optimizer, manual_variables, manual_data, max_tries: int = 120):
    """Suggest a new point inside current bounds and not a duplicate."""
    suggest_variables = manual_variables
    seen = _existing_points_set(suggest_variables, manual_data)

    last_x = None
    for _ in range(max_tries):
        x = optimizer.suggest()
        last_x = x
        if not _in_suggest_space(x, suggest_variables):
            continue
        tup = tuple(xi for xi in x)
        if tup not in seen:
            return x

    if last_x is not None:
        x_proj = _project_to_suggest_space(last_x, suggest_variables)
        if tuple(x_proj) not in seen:
            return x_proj

    out: list[Any] = []
    for (name, v1, v2, unit, vtype) in suggest_variables:
        if vtype == "continuous":
            out.append(float(v1))
        else:
            out.append(v1[0])
    if tuple(out) in seen and len(suggest_variables) > 0:
        out2 = []
        for (val, (name, v1, v2, unit, vtype)) in zip(out, suggest_variables):
            if vtype == "continuous":
                eps = (v2 - v1) * 1e-6
                out2.append(min(max(val + eps, v1), v2))
            else:
                out2.append(val)
        return out2
    return out
