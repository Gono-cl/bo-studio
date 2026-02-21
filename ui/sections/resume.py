"""
Sidebar resume of an exact previous run (load campaign state).
"""

from __future__ import annotations

import os
import json
import pandas as pd
import streamlit as st
import dill as pickle

from ui.components import resume_campaign_selector, load_campaign_button
from core.utils.bo_manual import rebuild_optimizer_from_df, coerce_point_to_variables


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

    st.session_state.manual_variables = metadata.get("variables", [])
    st.session_state.model_variables = metadata.get("model_variables", st.session_state.manual_variables)
    if not df_loaded.empty and st.session_state.manual_variables:
        var_names = [name for name, *_ in st.session_state.manual_variables]
        normalized_rows = []
        for _, row in df_loaded.iterrows():
            row_dict = row.to_dict()
            raw_x = [row_dict.get(name) for name in var_names]
            coerced_x = coerce_point_to_variables(raw_x, st.session_state.manual_variables)
            if coerced_x is not None:
                for name, val in zip(var_names, coerced_x):
                    row_dict[name] = val
            normalized_rows.append(row_dict)
        df_loaded = pd.DataFrame(normalized_rows, columns=df_loaded.columns)
    st.session_state.manual_data = df_loaded.to_dict("records")
    st.session_state.iteration = metadata.get("iteration", len(df_loaded))
    st.session_state.campaign_name = resume_file
    st.session_state.n_init = metadata.get("n_init", 1)
    st.session_state.total_iters = metadata.get("total_iters", 1)
    st.session_state.response = metadata.get("response", st.session_state.get("response", "Yield"))
    st.session_state.response_direction = metadata.get("response_direction", st.session_state.get("response_direction", "Maximize"))
    st.session_state.acq_func = metadata.get("acq_func", st.session_state.get("acq_func", "EI"))
    st.session_state.acq_xi = metadata.get("acq_xi", st.session_state.get("acq_xi", 0.01))
    st.session_state.acq_kappa = metadata.get("acq_kappa", st.session_state.get("acq_kappa", 1.96))
    st.session_state.bo_seed = metadata.get("bo_seed", st.session_state.get("bo_seed", 42))
    st.session_state.init_method = metadata.get("init_method", st.session_state.get("init_method", "random"))
    st.session_state.manual_initialized = True
    st.session_state.initial_results_submitted = metadata.get("initialization_complete", False)
    st.session_state.experiment_name = metadata.get("experiment_name", "")
    st.session_state.experiment_notes = metadata.get("experiment_notes", "")
    loaded_custom = metadata.get("custom_objectives", [])
    if isinstance(loaded_custom, dict):
        loaded_custom = list(loaded_custom.keys())
    elif isinstance(loaded_custom, set):
        loaded_custom = sorted(loaded_custom)
    elif not isinstance(loaded_custom, list):
        loaded_custom = []
    st.session_state.custom_objectives = loaded_custom

    # Rebuild optimizer from data to ensure space is correct
    if st.session_state.manual_variables and len(st.session_state.manual_data) > 0:
        model_vars = st.session_state.model_variables or st.session_state.manual_variables
        df_tmp = pd.DataFrame(st.session_state.manual_data)
        resp = st.session_state.response
        if resp in df_tmp.columns:
            st.session_state.manual_optimizer = rebuild_optimizer_from_df(
                model_vars,
                df_tmp,
                resp,
                n_initial_points_remaining=0,
                acq_func=st.session_state.get("acq_func", "EI"),
                direction=st.session_state.get("response_direction", "Maximize"),
                acq_xi=float(st.session_state.get("acq_xi", 0.01)),
                acq_kappa=float(st.session_state.get("acq_kappa", 1.96)),
                random_state=int(st.session_state.get("bo_seed", 42)),
            )
        else:
            st.warning("Response column not found in loaded data. Optimizer was not rebuilt.")

    st.success(f"Loaded campaign: {resume_file}")
