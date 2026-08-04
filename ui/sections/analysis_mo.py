from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from core.utils.analysis_utils import ANALYSIS_ORDER_COLUMN, prepare_multiobjective_frame
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

    df_plot, objs = prepare_multiobjective_frame(df, objectives)
    if df_plot.empty or len(objs) < 2:
        st.info("No complete numeric objective rows are available for multiobjective analysis.")
        return

    signs = np.array([1.0 if directions.get(o, "Maximize") == "Maximize" else -1.0 for o in objs], dtype=float)
    pts = df_plot[objs].to_numpy(dtype=float) * signs
    idx_pf = pareto_front_indices(pts)
    st.markdown(f"Pareto front size: {len(idx_pf)}")

    pareto_mask = np.zeros(len(df_plot), dtype=bool)
    if len(idx_pf) > 0:
        pareto_mask[np.asarray(idx_pf, dtype=int)] = True
    df_plot = df_plot.copy()
    df_plot["Pareto"] = np.where(pareto_mask, "Pareto", "Dominated")
    df_pf = df_plot.iloc[idx_pf].copy() if len(idx_pf) > 0 else df_plot.iloc[0:0].copy()

    if len(objs) > 3:
        st.info(
            "Pareto geometry plots are limited to 2 or 3 objectives. "
            "The Pareto front below was computed using all selected objectives."
        )
        show_cols = [ANALYSIS_ORDER_COLUMN] + objs
        st.dataframe(
            df_pf[show_cols].rename(columns={ANALYSIS_ORDER_COLUMN: "Experiment"}),
            use_container_width=True,
            hide_index=True,
        )
        return

    if len(objs) == 2:
        render_analysis_card(
            "2D Pareto View",
            [
                f"Pareto view for {objs[0]} vs {objs[1]}.",
                "Red front line highlights non-dominated solutions.",
            ],
            tone="blue",
        )
        fig = px.scatter(
            df_plot,
            x=objs[0],
            y=objs[1],
            color="Pareto",
            color_discrete_map={"Pareto": "#dc2626", "Dominated": "#94a3b8"},
        )
        df_pf_line = df_pf.sort_values(by=objs[0])
        fig.add_trace(
            go.Scatter(
                x=df_pf_line[objs[0]],
                y=df_pf_line[objs[1]],
                mode="lines+markers",
                line=dict(color="red", width=3),
                name="Pareto front",
            )
        )
        st.plotly_chart(fig, use_container_width=True)

        # Knee and HV
        P = (df_pf[objs].to_numpy(dtype=float) * signs[:2]) if len(df_pf) else np.empty((0, 2))
        if P.shape[0] >= 2:
            ki = knee_index_2d(P)
            if ki is not None:
                knee_pt = df_pf.iloc[ki]
                st.markdown(
                    f"Approximate knee point: Experiment {int(knee_pt[ANALYSIS_ORDER_COLUMN])} "
                    f"({objs[0]}={knee_pt[objs[0]]:.4g}, {objs[1]}={knee_pt[objs[1]]:.4g})"
                )
            # simple HV w.r.t. min over data (in transformed space)
            ref = tuple((df_plot[objs].to_numpy(dtype=float) * signs[:2]).min(axis=0))
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
        fig = px.scatter_3d(
            df_plot,
            x=objs[0],
            y=objs[1],
            z=objs[2],
            color="Pareto",
            color_discrete_map={"Pareto": "#dc2626", "Dominated": "#94a3b8"},
        )
        df_pf_line = df_pf.sort_values(by=objs[0])
        fig.add_trace(
            go.Scatter3d(
                x=df_pf_line[objs[0]],
                y=df_pf_line[objs[1]],
                z=df_pf_line[objs[2]],
                mode="lines+markers",
                line=dict(color="red", width=6),
                name="Pareto front",
            )
        )
        st.plotly_chart(fig, use_container_width=True)

