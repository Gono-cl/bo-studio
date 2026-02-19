from __future__ import annotations

from typing import Tuple


def format_n_init_range(low: int, high: int) -> str:
    low_i, high_i = int(low), int(high)
    return str(low_i) if low_i == high_i else f"{low_i}-{high_i}"


def recommend_n_init_range(
    n_variables: int,
    total_budget: int | None = None,
    *,
    noisy: bool = False,
    mixed: bool = False,
    multiobjective: bool = False,
) -> Tuple[int, int, str]:
    """
    Return a practical recommendation range for BO initial experiments.

    Heuristic basis (practical, not absolute):
    - baseline: 2d to 3d
    - increase toward 3d to 5d for noisy / mixed / multiobjective settings
    - if a total budget is known, keep the range within about 20% to 40% of budget
    """
    d = max(1, int(n_variables))

    low = max(4, 2 * d)
    high = max(low + 1, 3 * d)

    if noisy or mixed or multiobjective:
        low = max(low, 3 * d)
        high = max(high, 5 * d)

    budget_note = ""
    if total_budget is not None and total_budget > 0:
        b = int(total_budget)
        frac_low = max(1, int(round(0.20 * b)))
        frac_high = max(frac_low, int(round(0.40 * b)))
        # Only apply budget clipping when budget is not already below heuristic low.
        # If budget is too small, keep the heuristic recommendation and add a note.
        if b >= low:
            low = max(low, frac_low)
            high = min(high, frac_high)
            high = min(max(high, low), b)
        else:
            budget_note = (
                f" Current total budget ({b}) is below this rule-of-thumb scale; "
                "consider increasing total iterations."
            )

    scenario = []
    if noisy:
        scenario.append("noisy")
    if mixed:
        scenario.append("mixed-variable")
    if multiobjective:
        scenario.append("multiobjective")
    scenario_text = ", ".join(scenario) if scenario else "standard"

    range_text = format_n_init_range(low, high)
    explanation = f"Rule-of-thumb for {d} variable(s), {scenario_text} setting: start around {range_text} initial experiments."
    if budget_note:
        explanation += budget_note
    return int(low), int(high), explanation
