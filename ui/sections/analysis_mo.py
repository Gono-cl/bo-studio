from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from core.utils.pareto import pareto_front_indices
from core.utils.knee import knee_index_2d
from core.utils.hypervolume import hypervolume_2d
from ui.sections.analysis_cards import render_analysis_card


def render_analysis_mo(df: pd.DataFrame, objectives: list[str], directions: dict[str, str]) -> None:
    st.subheader("Multiobjective Diagnostics")
    if not objectives or len(objectives) < 2:
        st.info("Select at least two objectives.")
        return

    render_analysis_card(
        "How to read Multiobjective Diagnostics",
        [
            "Pareto points are non-dominated trade-offs: improving one objective usually worsens another.",
            "The knee point is an approximate balanced compromise, and hypervolume summarizes overall front quality.",
        ],
        tone="purple",
    )

    objs = objectives[:3]
    signs = np.array([1.0 if directions.get(o, "Maximize") == "Maximize" else -1.0 for o in objs], dtype=float)
    pts = df[objs].to_numpy(dtype=float) * signs
    idx_pf = pareto_front_indices(pts)
    st.markdown(f"Pareto front size: {len(idx_pf)}")

    if len(objs) == 2:
        render_analysis_card(
            "2D Pareto View",
            [
                f"Pareto view for {objs[0]} vs {objs[1]}.",
                "Red front line highlights non-dominated solutions.",
            ],
            tone="blue",
        )
        fig = px.scatter(df, x=objs[0], y=objs[1], color=df.index.isin(idx_pf), labels={"color": "Pareto"})
        df_pf = df.iloc[idx_pf].sort_values(by=objs[0])
        fig.add_trace(go.Scatter(x=df_pf[objs[0]], y=df_pf[objs[1]], mode='lines+markers', line=dict(color='red', width=3), name='Pareto front'))
        st.plotly_chart(fig, use_container_width=True)

        # Knee and HV
        P = (df_pf[objs].to_numpy(dtype=float) * signs[:2]) if len(df_pf) else np.empty((0, 2))
        if P.shape[0] >= 2:
            ki = knee_index_2d(P)
            if ki is not None:
                knee_pt = df_pf.iloc[ki]
                st.markdown(f"Knee point index (approx): {ki}")
            # simple HV w.r.t. min over data (in transformed space)
            ref = tuple((df[objs].to_numpy(dtype=float) * signs[:2]).min(axis=0))
            hv = hypervolume_2d(P, ref)
            st.markdown(f"Approx. 2D Hypervolume: {hv:.4g}")
    elif len(objs) == 3:
        render_analysis_card(
            "3D Pareto View",
            [
                f"Pareto view for {objs[0]}, {objs[1]}, and {objs[2]}.",
                "Non-dominated points define the trade-off surface.",
            ],
            tone="blue",
        )
        fig = px.scatter_3d(df, x=objs[0], y=objs[1], z=objs[2], color=df.index.isin(idx_pf))
        df_pf = df.iloc[idx_pf].sort_values(by=objs[0])
        fig.add_trace(go.Scatter3d(x=df_pf[objs[0]], y=df_pf[objs[1]], z=df_pf[objs[2]], mode='lines+markers', line=dict(color='red', width=6), name='Pareto front'))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("For >3 objectives, use the parallel coordinates on the main pages.")

