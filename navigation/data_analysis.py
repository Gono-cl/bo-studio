import os
import streamlit as st
import pandas as pd

from ui.sections.analysis_overview import render_analysis_overview
from ui.sections.analysis_explain import render_analysis_explain
from ui.sections.analysis_mo import render_analysis_mo


st.title("Data Analysis")
st.success(
    """
    Explore, explain, and compare results:
    - Overview: summary stats, best runs, trends, and pairwise plots.
    - Explain: fit a lightweight surrogate to show importance and partial dependence.
    - Multiobjective: Pareto diagnostics, knee point, and hypervolume (2D).
    """
)


def _get_current_data():
    so = st.session_state.get("manual_data", [])
    mo = st.session_state.get("mo_data", [])
    return so, mo


so_data, mo_data = _get_current_data()

mode = st.radio("Select dataset", ["Single Objective", "Multiobjective"], index=0 if len(so_data) else 1 if len(mo_data) else 0)

if mode == "Single Objective":
    df = pd.DataFrame(so_data)
    if df.empty:
        st.info("No single‑objective data available in this session. Run or load a campaign first.")
        st.stop()
    response = st.session_state.get("response", None)
    direction = st.session_state.get("response_direction", "Maximize")
    render_analysis_overview(df, response=response, direction=direction)
    render_analysis_explain(df, target=response)
else:
    df = pd.DataFrame(mo_data)
    if df.empty:
        st.info("No multiobjective data available in this session. Run or load a campaign first.")
        st.stop()
    # choose two objectives to analyze
    mo_objs = st.session_state.get("mo_objectives", [])
    mo_dirs = st.session_state.get("mo_directions", {})
    if not mo_objs:
        # heuristic: any non‑variable numeric columns
        varnames = [n for n, *_ in st.session_state.get("manual_variables", [])]
        mo_objs = [c for c in df.columns if c not in varnames][:2]
    sel = st.multiselect("Objectives for analysis", mo_objs, default=mo_objs[:2], key="analysis_mo_select")
    if len(sel) < 2:
        st.warning("Select at least two objectives for Pareto analysis.")
    render_analysis_overview(df, response=sel[0] if sel else None, direction=mo_dirs.get(sel[0], "Maximize") if sel else "Maximize", extra_objectives=sel)
    render_analysis_mo(df, objectives=sel, directions={o: mo_dirs.get(o, "Maximize") for o in sel})

