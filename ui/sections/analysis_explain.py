from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance, PartialDependenceDisplay


def _prepare_xy(df: pd.DataFrame, target: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    varnames = [n for n, *_ in st.session_state.get("manual_variables", [])]
    cols = [c for c in varnames if c in df.columns]
    X = df[cols].copy()
    X = X.apply(pd.to_numeric, errors="ignore")
    y = pd.to_numeric(df[target], errors="coerce")
    mask = y.notna()
    return X[mask].to_numpy(), y[mask].to_numpy(), cols


def render_analysis_explain(df: pd.DataFrame, target: str | None) -> None:
    st.subheader("Explainability")
    if not target or target not in df.columns:
        st.info("Select or run an objective to analyze.")
        return

    X, y, cols = _prepare_xy(df, target)
    if X.size == 0 or y.size == 0 or len(cols) == 0:
        st.info("Not enough data to fit an analysis surrogate.")
        return

    n_estimators = st.slider("RF trees", min_value=50, max_value=400, value=200, step=50)
    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    rf.fit(X, y)
    r2 = rf.score(X, y)
    st.markdown(f"Model fit (RF on collected data): R² = {r2:.3f}")

    # Permutation importance
    st.markdown("#### Permutation Importance")
    try:
        imp = permutation_importance(rf, X, y, n_repeats=10, random_state=42)
        imp_df = pd.DataFrame({"feature": cols, "importance": imp.importances_mean})
        imp_df = imp_df.sort_values(by="importance", ascending=False)
        fig = px.bar(imp_df, x="feature", y="importance")
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        pass

    # Partial Dependence for top 1–2 features
    st.markdown("#### Partial Dependence (Top features)")
    top = imp_df["feature"].tolist()[:2] if 'imp_df' in locals() else cols[:2]
    for feat in top:
        try:
            from sklearn.inspection import partial_dependence
            pdp = partial_dependence(rf, X, [cols.index(feat)])
            xs = pdp["values"][0]
            ys = pdp["average"][0]
            pdf = pd.DataFrame({feat: xs, target: ys})
            figp = px.line(pdf, x=feat, y=target)
            st.plotly_chart(figp, use_container_width=True)
        except Exception:
            continue

