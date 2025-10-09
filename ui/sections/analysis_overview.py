from __future__ import annotations

import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def _best_value(series: pd.Series, direction: str) -> float:
    if direction == "Minimize":
        return float(series.min())
    return float(series.max())


def render_analysis_overview(df: pd.DataFrame, response: str | None, direction: str = "Maximize", extra_objectives: list[str] | None = None) -> None:
    st.subheader("Overview")

    # Summary
    st.markdown("#### Dataset Summary")
    st.write({
        "rows": len(df),
        "columns": list(df.columns),
        "time_span": f"{df.get('Timestamp', pd.Series()).min()} → {df.get('Timestamp', pd.Series()).max()}" if 'Timestamp' in df.columns else "N/A",
    })

    # Best runs table
    if response and response in df.columns:
        st.markdown("#### Best Runs")
        best_dir = False if direction == "Maximize" else True
        df_sorted = df.copy()
        df_sorted[response] = pd.to_numeric(df_sorted[response], errors="coerce")
        df_sorted = df_sorted.dropna(subset=[response])
        df_best = df_sorted.sort_values(by=response, ascending=best_dir).head(10)
        st.dataframe(df_best, use_container_width=True)

        # Trend of best‑so‑far
        st.markdown("#### Best‑so‑far Trend")
        vals = df_sorted[response].tolist()
        best_so_far = []
        cur = None
        for v in vals:
            if cur is None:
                cur = v
            else:
                cur = max(cur, v) if direction == "Maximize" else min(cur, v)
            best_so_far.append(cur)
        tdf = pd.DataFrame({"iteration": range(1, len(best_so_far) + 1), "best": best_so_far})
        fig = px.line(tdf, x="iteration", y="best", markers=True)
        st.plotly_chart(fig, use_container_width=True)

    # Pairwise scatter/regression/correlation grid
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
                g = sns.PairGrid(data, diag_sharey=False)
                g.map_lower(sns.regplot, scatter_kws={"s": 15, "alpha": 0.6}, line_kws={"color": "black"})
                g.map_diag(sns.histplot, bins=20, color="#6c757d")

                def _corrcoef(x, y, **kws):
                    ax = plt.gca()
                    r = np.corrcoef(x, y)[0, 1]
                    ax.annotate(f"{r:.3f}", xy=(0.5, 0.5), xycoords=ax.transAxes,
                                ha="center", va="center", fontsize=11)
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
                st.pyplot(g.fig, clear_figure=True, use_container_width=True)
            else:
                st.info("Not enough numeric data for pairwise relationships.")
        except Exception:
            # Fallback to Plotly scatter matrix if seaborn unavailable
            try:
                fig2 = px.scatter_matrix(df[cols])
                st.plotly_chart(fig2, use_container_width=True)
            except Exception:
                pass
