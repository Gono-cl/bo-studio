"""
Sidebar resume of an exact previous run (load campaign state).
"""

from __future__ import annotations

import os
import json
import pandas as pd
import streamlit as st
from skopt.space import Real, Categorical
import dill as pickle

from ui.components import resume_campaign_selector, load_campaign_button
from core.utils.bo_manual import safe_build_optimizer, force_model_based


def render_resume_exact(user_save_dir: str, target=None, show_divider: bool = True) -> None:
    target = target or st.sidebar
    if show_divider:
        target.markdown("---")
    resume_file = resume_campaign_selector(
        user_save_dir, key="resume_campaign", target=target, show_divider=False
    )

    if resume_file == "None":
        return
    if not load_campaign_button(target=target):
        return

    run_path = os.path.join(user_save_dir, resume_file)
    try:
        with open(os.path.join(run_path, "optimizer.pkl"), "rb") as f:
            st.session_state.manual_optimizer = pickle.load(f)  # type: ignore[name-defined]
    except Exception:
        st.warning("optimizer.pkl not found or unreadable. The optimizer will be rebuilt from data when needed.")
        st.session_state.manual_optimizer = None

    try:
        df_loaded = pd.read_csv(os.path.join(run_path, "manual_data.csv"))
    except Exception:
        df_loaded = pd.DataFrame()
        st.warning("manual_data.csv missing or empty. Starting with an empty dataset.")

    with open(os.path.join(run_path, "metadata.json"), "r") as f:
        metadata = json.load(f)

    st.session_state.manual_data = df_loaded.to_dict("records")
    st.session_state.manual_variables = metadata.get("variables", [])
    st.session_state.model_variables = metadata.get("model_variables", st.session_state.manual_variables)
    st.session_state.iteration = metadata.get("iteration", len(df_loaded))
    st.session_state.campaign_name = resume_file
    st.session_state.n_init = metadata.get("n_init", 1)
    st.session_state.total_iters = metadata.get("total_iters", 1)
    st.session_state.response = metadata.get("response", st.session_state.get("response", "Yield"))
    st.session_state.manual_initialized = True
    st.session_state.initial_results_submitted = metadata.get("initialization_complete", False)
    st.session_state.experiment_name = metadata.get("experiment_name", "")
    st.session_state.experiment_notes = metadata.get("experiment_notes", "")

    # Rebuild optimizer from data to ensure space is correct
    if st.session_state.manual_variables and len(st.session_state.manual_data) > 0:
        model_vars = st.session_state.model_variables or st.session_state.manual_variables
        opt_vars = []
        for name, v1, v2, _, vtype in model_vars:
            if vtype == "continuous":
                opt_vars.append(Real(v1, v2, name=name))
            else:
                opt_vars.append(Categorical(v1, name=name))

        optimizer = safe_build_optimizer(opt_vars, n_initial_points_remaining=0, acq_func="EI")
        df_tmp = pd.DataFrame(st.session_state.manual_data)
        resp = st.session_state.response
        if resp in df_tmp.columns:
            df_tmp[resp] = pd.to_numeric(df_tmp[resp], errors="coerce")
            for _, row in df_tmp.iterrows():
                try:
                    y_val = float(row.get(resp, float("nan")))
                    if pd.notnull(y_val):
                        x = [row.get(name) for name, *_ in model_vars]
                        optimizer.observe(x, -y_val)
                except (ValueError, TypeError):
                    continue
        force_model_based(optimizer)
        st.session_state.manual_optimizer = optimizer

    st.success(f"Loaded campaign: {resume_file}")
