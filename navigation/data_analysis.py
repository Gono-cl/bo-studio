import os
import streamlit as st
import pandas as pd
from core.utils import db_handler

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

source = st.radio("Data source", ["Current Session", "Saved Campaign", "Database"], index=0)

loaded_df = None
loaded_mode = None  # 'so' or 'mo'
loaded_meta = {}

if source == "Saved Campaign":
    user_email = st.session_state.get("user_email", "default_user")
    SAVE_DIR = "/mnt/data/resumable_manual_runs" if os.getenv("RENDER") == "true" else os.path.join(os.getcwd(), "resumable_manual_runs")
    user_save_dir = os.path.join(SAVE_DIR, user_email)
    options = ["None"]
    metas = {}
    if os.path.isdir(user_save_dir):
        for d in sorted(os.listdir(user_save_dir)):
            p = os.path.join(user_save_dir, d)
            meta_p = os.path.join(p, "metadata.json")
            data_p = os.path.join(p, "manual_data.csv")
            if os.path.isdir(p) and os.path.exists(meta_p) and os.path.exists(data_p):
                options.append(d)
                metas[d] = (meta_p, data_p)
    sel = st.selectbox("Select campaign", options)
    if sel != "None":
        meta_p, data_p = metas[sel]
        import json
        with open(meta_p, "r") as f:
            loaded_meta = json.load(f)
        loaded_df = pd.read_csv(data_p)
        # set session vars to align analysis helpers
        st.session_state.manual_variables = loaded_meta.get("variables", [])
        if loaded_meta.get("mode") == "multiobjective":
            loaded_mode = 'mo'
            st.session_state.mo_objectives = loaded_meta.get("mo_objectives", [])
            st.session_state.mo_directions = loaded_meta.get("mo_directions", {})
        else:
            loaded_mode = 'so'
            st.session_state.response = loaded_meta.get("response", st.session_state.get("response"))
            st.session_state.response_direction = loaded_meta.get("response_direction", st.session_state.get("response_direction", "Maximize"))

elif source == "Database":
    user_email = st.session_state.get("user_email", "default_user")
    rows = db_handler.list_experiments(user_email)
    if not rows:
        st.info("No experiments found in database for this user.")
    else:
        options = {f"{rid} — {name} ({ts})": rid for rid, name, ts in rows}
        label = st.selectbox("Select experiment", list(options.keys()))
        exp = db_handler.load_experiment(options[label])
        if exp:
            loaded_df = exp["df_results"]
            st.session_state.manual_variables = exp["variables"]
            settings = exp.get("settings", {}) or {}
            st.session_state.response = settings.get("objective", st.session_state.get("response"))
            st.session_state.response_direction = st.session_state.get("response_direction", "Maximize")
            loaded_mode = 'so'

# Resolve mode and df based on selection, else fall back to current session
if loaded_df is not None:
    if loaded_mode == 'so':
        df = loaded_df
        response = st.session_state.get("response", None)
        direction = st.session_state.get("response_direction", "Maximize")
        render_analysis_overview(df, response=response, direction=direction)
        render_analysis_explain(df, target=response)
    elif loaded_mode == 'mo':
        df = loaded_df
        mo_objs = st.session_state.get("mo_objectives", [])
        mo_dirs = st.session_state.get("mo_directions", {})
        sel = st.multiselect("Objectives for analysis", mo_objs, default=mo_objs[:2], key="analysis_mo_select")
        if len(sel) < 2:
            st.warning("Select at least two objectives for Pareto analysis.")
        render_analysis_overview(df, response=sel[0] if sel else None, direction=mo_dirs.get(sel[0], "Maximize") if sel else "Maximize", extra_objectives=sel)
        render_analysis_mo(df, objectives=sel, directions={o: mo_dirs.get(o, "Maximize") for o in sel})
else:
    # Use current session
    mode = st.radio("Select dataset", ["Single Objective", "Multiobjective"], index=0 if len(so_data) else 1 if len(mo_data) else 0)
    if mode == "Single Objective":
        df = pd.DataFrame(so_data)
        if df.empty:
            st.info("No single‑objective data available in this session. Run, load a campaign, or choose 'Database'.")
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
        mo_objs = st.session_state.get("mo_objectives", [])
        mo_dirs = st.session_state.get("mo_directions", {})
        if not mo_objs:
            varnames = [n for n, *_ in st.session_state.get("manual_variables", [])]
            mo_objs = [c for c in df.columns if c not in varnames][:2]
        sel = st.multiselect("Objectives for analysis", mo_objs, default=mo_objs[:2], key="analysis_mo_select")
        if len(sel) < 2:
            st.warning("Select at least two objectives for Pareto analysis.")
        render_analysis_overview(df, response=sel[0] if sel else None, direction=mo_dirs.get(sel[0], "Maximize") if sel else "Maximize", extra_objectives=sel)
        render_analysis_mo(df, objectives=sel, directions={o: mo_dirs.get(o, "Maximize") for o in sel})
