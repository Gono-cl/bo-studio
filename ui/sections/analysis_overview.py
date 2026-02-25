from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st
from ui.sections.analysis_cards import render_analysis_card


def _best_value(series: pd.Series, direction: str) -> float:
    if direction == "Minimize":
        return float(series.min())
    return float(series.max())


def _format_time_span(df: pd.DataFrame) -> str:
    if "Timestamp" not in df.columns:
        return "N/A"
    ts = pd.to_datetime(df["Timestamp"], errors="coerce").dropna()
    if ts.empty:
        return "N/A"
    return f"{ts.min().strftime('%Y-%m-%d %H:%M:%S')} to {ts.max().strftime('%Y-%m-%d %H:%M:%S')}"


def render_analysis_overview(
    df: pd.DataFrame,
    response: str | None,
    direction: str = "Maximize",
    extra_objectives: list[str] | None = None,
) -> None:
    st.subheader("Overview")

    # Summary
    render_analysis_card(
        "What this section shows",
        [
            "Quick campaign context: number of experiments, available columns, variables, and time coverage.",
        ],
        tone="blue",
    )
    with st.container(border=True):
        st.markdown("#### Dataset Summary")
        c1, c2, c3 = st.columns(3)
        c1.metric("Experiments", len(df))
        c2.metric("Columns", len(df.columns))
        c3.metric("Variables", len(st.session_state.get("manual_variables", [])))
        st.caption(f"Time span: {_format_time_span(df)}")
        st.caption("Columns: " + ", ".join(str(c) for c in df.columns))

    # Full campaign table in natural order
    if not df.empty:
        render_analysis_card(
            "How to read Experiment Data",
            [
                "Rows are shown in chronological campaign order (first to last). Use this table to inspect exact settings and measured outcomes for each experiment.",
            ],
            tone="green",
        )
        st.markdown("#### Experiment Data")
        df_view = df.copy().reset_index(drop=True)
        df_view["Experiment"] = range(1, len(df_view) + 1)
        ordered_cols = ["Experiment"] + [c for c in df_view.columns if c != "Experiment"]
        df_view = df_view[ordered_cols]
        st.dataframe(df_view, use_container_width=True, hide_index=True)

    # Convergence plot (objective over experiments in natural order)
    if response and response in df.columns:
        df_conv = df[[response]].copy().reset_index(drop=True)
        df_conv[response] = pd.to_numeric(df_conv[response], errors="coerce")
        df_conv = df_conv.dropna(subset=[response]).copy()
        if not df_conv.empty:
            render_analysis_card(
                "How to read Objective Progress",
                [
                    "Tracks the measured objective over experiment number. Look for trends, plateaus, and jumps after strategy or parameter changes.",
                ],
                tone="orange",
            )
            df_conv["Experiment"] = range(1, len(df_conv) + 1)
            df_conv = df_conv[["Experiment", response]]
            st.markdown("#### Objective Progress")
            fig_conv = px.line(df_conv, x="Experiment", y=response, markers=True)
            fig_conv.update_layout(
                xaxis_title="Experiment",
                yaxis_title=response,
            )
            st.plotly_chart(fig_conv, use_container_width=True)

    # Pairwise scatter/regression/correlation grid
    render_analysis_card(
        "How to read Pairwise Relationships",
        [
            "Diagonal panels show value distributions, lower panels show scatter + trend line, and upper panels show correlation strength.",
            "Use this view to spot interactions and collinearity.",
        ],
        tone="purple",
    )
    st.markdown("#### Pairwise Relationships")
    # choose up to 6 columns: variables + response(s)
    varnames = [n for n, *_ in st.session_state.get("manual_variables", [])]
    cols = varnames[:4]
    if response and response not in cols:
        cols.append(response)
    if extra_objectives:
        for c in extra_objectives:
            if c not in cols:
                cols.append(c)
            if len(cols) >= 6:
                break
    cols = [c for c in cols if c in df.columns]
    if len(cols) >= 2:
        try:
            # Seaborn PairGrid with:
            # - lower: scatter + regression line
            # - diag: histogram
            # - upper: correlation coefficient
            data = df[cols].copy()
            # ensure numeric for plotting (coerce where possible)
            for c in data.columns:
                data[c] = pd.to_numeric(data[c], errors="coerce")
            data = data.dropna()
            if data.shape[0] > 1:
                g = sns.PairGrid(data, diag_sharey=False, height=1.7, aspect=1.0)
                g.map_lower(sns.regplot, scatter_kws={"s": 15, "alpha": 0.6}, line_kws={"color": "black"})
                g.map_diag(sns.histplot, bins=20, color="#6c757d")

                def _corrcoef(x, y, **kws):
                    ax = plt.gca()
                    r = np.corrcoef(x, y)[0, 1]
                    ax.annotate(
                        f"{r:.3f}",
                        xy=(0.5, 0.5),
                        xycoords=ax.transAxes,
                        ha="center",
                        va="center",
                        fontsize=11,
                    )
                    # add thin trend line for context
                    try:
                        sns.regplot(x=x, y=y, scatter=False, ax=ax, color="black", truncate=True)
                    except Exception:
                        pass

                g.map_upper(_corrcoef)
                for ax in g.axes.flatten():
                    if ax is not None:
                        ax.tick_params(labelsize=8)
                plt.tight_layout()
                st.pyplot(g.fig, clear_figure=True, use_container_width=False)
            else:
                st.info("Not enough numeric data for pairwise relationships.")
        except Exception:
            # Fallback to Plotly scatter matrix if seaborn unavailable
            try:
                fig2 = px.scatter_matrix(df[cols])
                st.plotly_chart(fig2, use_container_width=True)
            except Exception:
                pass
