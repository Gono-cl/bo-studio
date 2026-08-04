from __future__ import annotations

from typing import Any

import pandas as pd


ADMIN_ANALYSIS_COLUMNS = {"Timestamp", "Experiment", "Iteration"}
ANALYSIS_ORDER_COLUMN = "__analysis_order__"


def variable_names(variables: list[Any] | tuple[Any, ...] | None) -> list[str]:
    names: list[str] = []
    for item in variables or []:
        if isinstance(item, (list, tuple)) and item:
            names.append(str(item[0]))
        elif isinstance(item, str):
            names.append(item)
    return names


def infer_objective_columns(
    df: pd.DataFrame,
    variables: list[Any] | tuple[Any, ...] | None,
    limit: int | None = None,
    exclude: set[str] | None = None,
) -> list[str]:
    excluded = set(exclude or ())
    excluded.update(ADMIN_ANALYSIS_COLUMNS)
    excluded.update(variable_names(variables))

    numeric_candidates: list[str] = []
    fallback_candidates: list[str] = []
    for col in df.columns:
        if col in excluded:
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().sum() > 0:
            numeric_candidates.append(col)
        else:
            fallback_candidates.append(col)

    selected = numeric_candidates or fallback_candidates
    return selected[:limit] if limit is not None else selected


def prepare_objective_progress_frame(df: pd.DataFrame, response: str | None) -> pd.DataFrame:
    if not response or response not in df.columns:
        return pd.DataFrame(columns=[ANALYSIS_ORDER_COLUMN, response or "Value"])

    df_view = df[[response]].copy().reset_index(drop=True)
    df_view[ANALYSIS_ORDER_COLUMN] = range(1, len(df_view) + 1)
    df_view[response] = pd.to_numeric(df_view[response], errors="coerce")
    return df_view.dropna(subset=[response]).copy()


def prepare_multiobjective_frame(
    df: pd.DataFrame,
    objectives: list[str] | tuple[str, ...] | None,
) -> tuple[pd.DataFrame, list[str]]:
    valid_objectives = [col for col in (objectives or []) if col in df.columns]
    if len(valid_objectives) < 2:
        return pd.DataFrame(), valid_objectives

    df_view = df.copy().reset_index(drop=True)
    df_view[ANALYSIS_ORDER_COLUMN] = range(1, len(df_view) + 1)
    for obj in valid_objectives:
        df_view[obj] = pd.to_numeric(df_view[obj], errors="coerce")
    df_view = df_view.dropna(subset=valid_objectives).copy()
    return df_view, valid_objectives


def _coerce_name_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    if isinstance(value, set):
        return [str(item) for item in sorted(value)]
    return []


def infer_db_analysis_context(
    exp: dict[str, Any],
    fallback_response: str | None = None,
    fallback_direction: str = "Maximize",
) -> dict[str, Any]:
    df = exp.get("df_results")
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df or [])
    variables = exp.get("variables", []) or []
    settings = exp.get("settings", {}) or {}

    objectives = _coerce_name_list(settings.get("objectives"))
    if not objectives:
        best_result = exp.get("best_result")
        if isinstance(best_result, list) and best_result and isinstance(best_result[0], dict):
            blocked = set(ADMIN_ANALYSIS_COLUMNS)
            blocked.update(variable_names(variables))
            objectives = [col for col in best_result[0].keys() if col in df.columns and col not in blocked]

    if len(objectives) >= 2:
        directions = (
            settings.get("objective_directions")
            or settings.get("mo_directions")
            or settings.get("directions")
            or {}
        )
        valid_objectives = [col for col in objectives if col in df.columns]
        if len(valid_objectives) >= 2:
            return {
                "df": df,
                "mode": "mo",
                "variables": variables,
                "mo_objectives": valid_objectives,
                "mo_directions": {obj: directions.get(obj, "Maximize") for obj in valid_objectives},
            }

    response = settings.get("objective") or fallback_response
    if response not in df.columns:
        inferred = infer_objective_columns(df, variables, limit=1)
        response = inferred[0] if inferred else response

    return {
        "df": df,
        "mode": "so",
        "variables": variables,
        "response": response,
        "response_direction": settings.get("response_direction", fallback_direction),
    }
