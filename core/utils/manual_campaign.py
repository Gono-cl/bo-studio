from __future__ import annotations

from typing import Any

import pandas as pd


def campaign_has_started(
    *,
    manual_initialized: bool = False,
    manual_data: list[dict[str, Any]] | None = None,
    suggestions: list[list[Any]] | None = None,
    iteration: int = 0,
    next_suggestion_cached: list[Any] | None = None,
) -> bool:
    """Return True once a SO campaign has started and campaign-defining settings should lock."""
    return bool(
        manual_initialized
        or bool(manual_data)
        or bool(suggestions)
        or int(iteration) > 0
        or next_suggestion_cached is not None
    )


def multiobjective_campaign_has_started(
    *,
    mo_initialized: bool = False,
    mo_data: list[dict[str, Any]] | None = None,
    mo_suggestions: list[list[Any]] | None = None,
    mo_iteration: int = 0,
    mo_pending_df: list[dict[str, Any]] | None = None,
) -> bool:
    """Return True once an MO campaign has started and campaign-defining settings should lock."""
    return bool(
        mo_initialized
        or bool(mo_data)
        or bool(mo_suggestions)
        or int(mo_iteration) > 0
        or bool(mo_pending_df)
    )


def parse_required_result_text(raw_value: Any, field_label: str = "Result") -> float:
    """Parse a required numeric input, accepting either dot or comma decimals."""
    text = "" if raw_value is None else str(raw_value).strip()
    if not text:
        raise ValueError(f"{field_label} is required.")

    normalized = text.replace(",", ".") if "," in text and "." not in text else text
    try:
        return float(normalized)
    except (TypeError, ValueError) as ex:
        raise ValueError(f"{field_label} must be a valid number.") from ex


def validate_initial_results(
    edited_df: pd.DataFrame,
    variables,
    response_col: str,
) -> list[tuple[list[Any], float, dict[str, Any]]]:
    """
    Validate the initial-results table before mutating optimizer state.

    Returns rows as `(x, y_value, row_dict)` when every result is present and numeric.
    """
    if edited_df is None or edited_df.empty:
        raise ValueError("Please fill in the initial-results table before submitting.")

    validated_rows: list[tuple[list[Any], float, dict[str, Any]]] = []
    missing_experiments: list[str] = []
    invalid_entries: list[str] = []

    for idx, row in edited_df.iterrows():
        experiment_id = row.get("Experiment", idx + 1)
        value = row.get(response_col)
        if value is None or str(value).strip() == "":
            missing_experiments.append(str(experiment_id))
            continue
        try:
            y_val = parse_required_result_text(value, field_label=f"{response_col} for experiment {experiment_id}")
        except ValueError:
            invalid_entries.append(f"{experiment_id}: {value}")
            continue

        x = [row[name] for name, *_ in variables]
        row_data = row.to_dict()
        row_data[response_col] = y_val
        validated_rows.append((x, y_val, row_data))

    if missing_experiments:
        raise ValueError(
            "Every initial experiment needs a result before BO can continue. "
            f"Missing values for experiment(s): {', '.join(missing_experiments)}."
        )
    if invalid_entries:
        raise ValueError(
            "Some initial results are not valid numbers: "
            f"{'; '.join(invalid_entries)}."
        )

    return validated_rows
