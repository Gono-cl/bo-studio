from __future__ import annotations

import os
import json
import pandas as pd
import streamlit as st


def render_mo_resume(user_save_dir: str) -> None:
    """Sidebar control to resume a saved MO campaign (filters by mode)."""
    st.sidebar.markdown("---")
    # Collect campaigns that have metadata.json with mode == 'multiobjective'
    options = ["None"]
    for d in sorted(os.listdir(user_save_dir)):
        p = os.path.join(user_save_dir, d)
        meta_p = os.path.join(p, "metadata.json")
        if os.path.isdir(p) and os.path.exists(meta_p):
            try:
                with open(meta_p, "r") as f:
                    meta = json.load(f)
                if meta.get("mode") == "multiobjective":
                    options.append(d)
            except Exception:
                pass

    resume_file = st.sidebar.selectbox("🔄 Resume Multiobjective Campaign", options=options, key="mo_resume_campaign")
    if resume_file == "None":
        return

    if st.sidebar.button("Load MO Campaign"):
        run_path = os.path.join(user_save_dir, resume_file)
        try:
            df_loaded = pd.read_csv(os.path.join(run_path, "manual_data.csv"))
        except Exception:
            df_loaded = pd.DataFrame()
            st.warning("manual_data.csv missing or empty. Starting with an empty dataset.")
        with open(os.path.join(run_path, "metadata.json"), "r") as f:
            metadata = json.load(f)

        st.session_state.mo_data = df_loaded.to_dict("records")
        st.session_state.manual_variables = metadata.get("variables", [])
        st.session_state.mo_objectives = metadata.get("mo_objectives", [])
        st.session_state.mo_directions = metadata.get("mo_directions", {})
        st.session_state.mo_n_init = metadata.get("mo_n_init", 0)
        st.session_state.mo_total_iters = metadata.get("mo_total_iters", 0)
        st.session_state.mo_init_method = metadata.get("mo_init_method", "lhs")
        st.session_state.acq_func = metadata.get("acq_func", "EI")
        st.session_state.experiment_name = metadata.get("experiment_name", "")
        st.session_state.experiment_notes = metadata.get("experiment_notes", "")
        st.session_state.custom_objectives = metadata.get("custom_objectives", {})

        st.session_state.mo_initialized = True
        st.session_state.mo_suggestions = []
        st.session_state.mo_iteration = len(st.session_state.mo_data)
        st.success(f"Loaded MO campaign: {resume_file}")

