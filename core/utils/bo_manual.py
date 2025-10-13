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


def safe_build_optimizer(space, n_initial_points_remaining: int = 0, acq_func: str = "EI") -> StepBayesianOptimizer:
    """
    Build StepBayesianOptimizer and set underlying skopt.Optimizer knobs.

    This project wraps skopt.Optimizer inside StepBayesianOptimizer at attribute
    `_optimizer` (and property `skopt_optimizer`). We cannot pass
    `n_initial_points` to the wrapper constructor, so we set it directly on the
    underlying optimizer after construction.
    """
    opt = StepBayesianOptimizer(space, acq_func=acq_func)
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


def rebuild_optimizer_from_df(
    variables,
    df: pd.DataFrame,
    response_col: str,
    n_initial_points_remaining: int = 0,
    acq_func: str = "EI",
) -> StepBayesianOptimizer:
    """Build StepBayesianOptimizer on 'variables' (ModelSpace), and observe seeds once."""
    space = []
    for name, v1, v2, _unit, vtype in variables:
        if vtype == "continuous":
            space.append(Real(v1, v2, name=name))
        else:
            space.append(Categorical(v1, name=name))

    opt = safe_build_optimizer(space, n_initial_points_remaining, acq_func)

    df = df.copy()
    if response_col not in df.columns:
        raise ValueError(f"Response column '{response_col}' not found in reused data.")
    df[response_col] = pd.to_numeric(df[response_col], errors="coerce")
    df = df.dropna(subset=[response_col])

    # Batch observe to avoid repeated refits (much faster than per-row)
    X_batch = []
    y_batch = []
    for _, row in df.iterrows():
        try:
            y = float(row[response_col])
        except (ValueError, TypeError):
            continue
        if pd.notnull(y):
            X_batch.append([row.get(name) for name, *_ in variables])
            y_batch.append(float(-y))  # maximize -> minimize convention

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
