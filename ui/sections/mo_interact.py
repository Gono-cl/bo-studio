from __future__ import annotations

import os
import json
from typing import List

import numpy as np
import pandas as pd
import streamlit as st
from skopt.space import Real, Categorical

from ui.components import data_editor, display_dataframe
from ui.charts import Charts
from core.utils.pareto import pareto_front_indices
from core.utils.scalarization import sample_dirichlet_weights, weighted_sum, tchebycheff
from core.utils.bo_manual import safe_build_optimizer


def _build_scalarized_optimizer(weights: np.ndarray, method: str = "weighted_sum"):
    # Build optimizer on current model variables
    mv = st.session_state.get("manual_variables", [])
    dims = []
    for name, v1, v2, _u, t in mv:
        if t == "continuous":
            dims.append(Real(v1, v2, name=name))
        else:
            dims.append(Categorical(v1, name=name))
    opt = safe_build_optimizer(dims, n_initial_points_remaining=0, acq_func=st.session_state.get("acq_func", "EI"))
    # Observe existing data using scalarized objective
    data = st.session_state.get("mo_data", [])
    objs = st.session_state.get("mo_objectives", [])
    if data and objs:
        df = pd.DataFrame(data)
        # Apply direction flips before scalarization
        dir_map = st.session_state.get("mo_directions", {})
        signs = np.array([1.0 if dir_map.get(o, "Maximize") == "Maximize" else -1.0 for o in objs], dtype=float)
        # ideal point for tchebycheff on transformed space
        z = (df[objs] * signs).max().values
        for _, row in df.iterrows():
            x = [row.get(name) for name, *_ in mv]
            y_vec = (row[objs].values.astype(float) * signs)
            if method == "tchebycheff":
                s = tchebycheff(y_vec, weights, z)
            else:
                s = weighted_sum(y_vec, weights)
            opt.observe(x, float(-s))  # maximize scalarization
    return opt


def render_mo_interact_and_pareto(user_save_dir: str):
    # Show current results and Pareto front
    data = st.session_state.get("mo_data", [])
    objs = st.session_state.get("mo_objectives", [])
    if data and objs:
        df = pd.DataFrame(data)
        st.markdown("### Multiobjective Results")
        display_dataframe(df, key="mo_results_df")
        # Pareto front indices
        # Apply per-objective direction: transform to maximization by flipping minimized ones
        dir_map = st.session_state.get("mo_directions", {})
        signs = np.array([1.0 if dir_map.get(o, "Maximize") == "Maximize" else -1.0 for o in objs], dtype=float)
        pts = df[objs].to_numpy(dtype=float) * signs
        idx_pf = pareto_front_indices(pts)
        st.markdown(f"Pareto front size: {len(idx_pf)}")
        # Simple plots
        if len(objs) == 2:
            import plotly.express as px
            import plotly.graph_objects as go
            fig = px.scatter(df, x=objs[0], y=objs[1], color=df.index.isin(idx_pf), labels={"color": "Pareto"})
            # Add red line connecting Pareto front (sorted by first objective)
            df_pf = df.iloc[idx_pf].sort_values(by=objs[0])
            fig.add_trace(go.Scatter(x=df_pf[objs[0]], y=df_pf[objs[1]], mode='lines+markers', line=dict(color='red', width=3), name='Pareto front'))
            st.plotly_chart(fig, use_container_width=True)
        elif len(objs) == 3:
            import plotly.express as px
            import plotly.graph_objects as go
            fig = px.scatter_3d(df, x=objs[0], y=objs[1], z=objs[2], color=df.index.isin(idx_pf))
            # Connect Pareto front in 3D (sorted by first objective)
            df_pf = df.iloc[idx_pf].sort_values(by=objs[0])
            fig.add_trace(go.Scatter3d(x=df_pf[objs[0]], y=df_pf[objs[1]], z=df_pf[objs[2]], mode='lines+markers', line=dict(color='red', width=6), name='Pareto front'))
            st.plotly_chart(fig, use_container_width=True)
        else:
            Charts.show_parallel_coordinates(data, objs[0])

    # Suggest next points via scalarization
    if st.session_state.get("manual_variables") and st.session_state.get("mo_objectives"):
        st.markdown("### Suggest Next MO Experiments")
        col1, col2, col3 = st.columns(3)
        with col1:
            k = st.number_input("# Weight Vectors", min_value=1, max_value=10, value=3)
        with col2:
            method = st.selectbox("Scalarization", ["Weighted Sum", "Tchebycheff"])
            method_key = "tchebycheff" if method.lower().startswith("tch") else "weighted_sum"
        with col3:
            seed = st.number_input("Seed", min_value=0, max_value=9999, value=42)

        if st.button("Propose K Scalarized Suggestions"):
            m = len(st.session_state.mo_objectives)
            W = sample_dirichlet_weights(m, int(k), seed=int(seed))
            suggestions = []
            for w in W:
                opt = _build_scalarized_optimizer(w, method=method_key)
                x = opt.suggest()
                suggestions.append(x)
            # Show suggestions and allow recording a batch
            cols = [name for name, *_ in st.session_state.manual_variables]
            df_sug = pd.DataFrame([dict(zip(cols, s)) for s in suggestions])
            st.markdown("#### Proposed Points")
            st.dataframe(df_sug, use_container_width=True)
            st.markdown("#### Enter results for each proposed point")
            df_res = df_sug.copy()
            for obj in st.session_state.mo_objectives:
                df_res[obj] = None
            edited = st.data_editor(df_res, key="mo_batch_results")
            if st.button("Submit MO Batch Results"):
                df2 = edited.copy()
                for obj in st.session_state.mo_objectives:
                    df2[obj] = pd.to_numeric(df2[obj], errors="coerce")
                if df2[st.session_state.mo_objectives].isna().any().any():
                    st.error("Please fill all objective values with numbers.")
                else:
                    st.session_state.mo_data.extend(df2.to_dict("records"))
                    st.success("Batch results recorded.")
