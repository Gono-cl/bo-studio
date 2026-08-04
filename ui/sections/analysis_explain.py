from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from core.utils.analysis_utils import variable_names
from ui.sections.analysis_cards import render_analysis_card


def _prepare_xy(
    df: pd.DataFrame,
    target: str,
    variables: list | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    cols = [c for c in variable_names(variables) if c in df.columns]
    if not cols:
        return np.empty((0, 0)), np.array([]), []

    X_df = df[cols].copy()
    for c in cols:
        numeric = pd.to_numeric(X_df[c], errors="coerce")
        # If most entries are numeric, keep numeric; otherwise encode as categorical codes.
        if numeric.notna().sum() >= max(1, int(0.7 * len(numeric))):
            X_df[c] = numeric
        else:
            X_df[c] = X_df[c].where(X_df[c].notna(), "missing").astype(str)
            X_df[c] = pd.factorize(X_df[c], sort=True)[0].astype(float)

    y = pd.to_numeric(df[target], errors="coerce")

    # Keep rows with complete X and y.
    valid_x = X_df.notna().all(axis=1)
    valid = valid_x & y.notna()
    if valid.sum() == 0:
        return np.empty((0, len(cols))), np.array([]), cols

    X = X_df.loc[valid, cols].to_numpy(dtype=float)
    yv = y.loc[valid].to_numpy(dtype=float)
    return X, yv, cols


def _render_bo_explainability_caution(n_points: int) -> None:
    render_analysis_card(
        "Important Interpretation Note",
        [
            "Bayesian Optimization is designed to find good experiments efficiently, not to provide causal explainability.",
            "The model shown here is a post-hoc surrogate fit on collected runs. Use these insights as guidance only, especially when the dataset is small or concentrated in a narrow region of the search space.",
            f"Current dataset size for this analysis: {n_points} experiment(s). Interpretation confidence generally improves with broader and larger datasets.",
        ],
        tone="orange",
    )


def _render_model_fit_check(X: np.ndarray, y: np.ndarray, target: str, n_estimators: int) -> None:
    render_analysis_card(
        "How to read Model Fit Check",
        [
            "This compares measured values against model predictions. Points near the dashed y=x line indicate better predictive agreement.",
            "Train metrics reflect in-sample fit. Test metrics (when available) better reflect generalization.",
        ],
        tone="blue",
    )
    st.markdown("#### Model Fit Check")

    if len(y) >= 6:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )
        rf_eval = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        rf_eval.fit(X_train, y_train)

        y_pred_train = rf_eval.predict(X_train)
        y_pred_test = rf_eval.predict(X_test)

        r2_train = r2_score(y_train, y_pred_train)
        mae_train = mean_absolute_error(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)
        mae_test = mean_absolute_error(y_test, y_pred_test)

        plot_df = pd.concat(
            [
                pd.DataFrame({"Measured": y_train, "Predicted": y_pred_train, "Split": "Train"}),
                pd.DataFrame({"Measured": y_test, "Predicted": y_pred_test, "Split": "Test"}),
            ],
            ignore_index=True,
        )
        title = "Measured vs Predicted (Train/Test Split)"
    else:
        rf_eval = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        rf_eval.fit(X, y)
        y_pred_train = rf_eval.predict(X)

        r2_train = r2_score(y, y_pred_train)
        mae_train = mean_absolute_error(y, y_pred_train)
        r2_test = None
        mae_test = None

        plot_df = pd.DataFrame({"Measured": y, "Predicted": y_pred_train, "Split": "Train"})
        title = "Measured vs Predicted"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Train R2", f"{r2_train:.3f}")
    c2.metric("Train MAE", f"{mae_train:.3g}")
    c3.metric("Test R2", f"{r2_test:.3f}" if r2_test is not None else "N/A")
    c4.metric("Test MAE", f"{mae_test:.3g}" if mae_test is not None else "N/A")

    fig = px.scatter(
        plot_df,
        x="Measured",
        y="Predicted",
        color="Split",
        color_discrete_map={"Train": "#2563eb", "Test": "#ef4444"},
        opacity=0.85,
        title=title,
    )

    mn = float(min(plot_df["Measured"].min(), plot_df["Predicted"].min()))
    mx = float(max(plot_df["Measured"].max(), plot_df["Predicted"].max()))
    fig.add_trace(
        go.Scatter(
            x=[mn, mx],
            y=[mn, mx],
            mode="lines",
            name="Ideal (y=x)",
            line=dict(color="black", dash="dash"),
        )
    )
    fig.update_layout(xaxis_title=f"Measured {target}", yaxis_title=f"Predicted {target}")
    st.plotly_chart(fig, use_container_width=True)


def render_analysis_explain(df: pd.DataFrame, target: str | None, variables: list | None = None) -> None:
    st.subheader("Model Fit and Explainability")
    if not target or target not in df.columns:
        st.info("Select or run an objective to analyze.")
        return

    X, y, cols = _prepare_xy(df, target, variables=variables)
    if X.size == 0 or y.size == 0 or len(cols) == 0:
        st.info("Not enough data to fit an analysis surrogate.")
        return

    _render_bo_explainability_caution(n_points=len(y))

    n_estimators = st.slider("RF trees", min_value=50, max_value=400, value=200, step=50)
    _render_model_fit_check(X, y, target, n_estimators)
    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    rf.fit(X, y)

    # Permutation importance
    render_analysis_card(
        "How to read Permutation Importance",
        [
            "Higher importance means model performance drops more when that variable is shuffled, so the model relied more on that variable in this dataset.",
        ],
        tone="green",
    )
    st.markdown("#### Permutation Importance")
    imp_df = None
    try:
        imp = permutation_importance(rf, X, y, n_repeats=10, random_state=42)
        imp_df = pd.DataFrame({"feature": cols, "importance": imp.importances_mean})
        imp_df = imp_df.sort_values(by="importance", ascending=False)
        fig = px.bar(imp_df, x="feature", y="importance")
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.info("Could not compute permutation importance for the current dataset.")

    # Partial dependence section removed by request.
