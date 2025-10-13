from __future__ import annotations

import os
import json
import pandas as pd
import streamlit as st

from core.utils.bo_manual import sanitize_name


def render_mo_experiment_header(user_save_dir: str):
    """Render MO experiment name/notes and return (name, notes, run_name, run_path)."""
    experiment_name = st.text_input("Experiment Name", value=st.session_state.get("experiment_name", ""))
    experiment_notes = st.text_area("Notes (optional)", value=st.session_state.get("experiment_notes", ""))
    _ = st.date_input("Experiment date")

    run_name = sanitize_name(experiment_name)
    st.session_state["campaign_name"] = run_name
    run_path = os.path.join(user_save_dir, run_name)
    os.makedirs(run_path, exist_ok=True)

    st.session_state.experiment_name = experiment_name
    st.session_state.experiment_notes = experiment_notes
    return experiment_name, experiment_notes, run_name, run_path


def render_mo_save_campaign(run_path: str) -> None:
    """Sidebar Save button for MO campaigns: writes data and metadata."""
    if st.sidebar.button("💾 Save Campaign (MO)"):
        mo_data = st.session_state.get("mo_data", [])
        if mo_data:
            df_save = pd.DataFrame(mo_data)
        else:
            base_cols = [name for name, *_ in st.session_state.get("manual_variables", [])]
            objs = st.session_state.get("mo_objectives", [])
            df_save = pd.DataFrame(columns=base_cols + list(objs))
        df_save.to_csv(os.path.join(run_path, "manual_data.csv"), index=False)

        metadata = {
            "mode": "multiobjective",
            "variables": st.session_state.get("manual_variables", []),
            "iteration": len(mo_data),
            "mo_objectives": st.session_state.get("mo_objectives", []),
            "mo_directions": st.session_state.get("mo_directions", {}),
            "mo_n_init": st.session_state.get("mo_n_init", 0),
            "mo_total_iters": st.session_state.get("mo_total_iters", 0),
            "mo_init_method": st.session_state.get("mo_init_method", "lhs"),
            "acq_func": st.session_state.get("acq_func", "EI"),
            "experiment_name": st.session_state.get("experiment_name", ""),
            "experiment_notes": st.session_state.get("experiment_notes", ""),
            "custom_objectives": st.session_state.get("custom_objectives", {}),
        }
        with open(os.path.join(run_path, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4)
        st.sidebar.success(f"Campaign '{os.path.basename(run_path)}' saved (MO)")

