"""
Header and persistence controls for the manual experiments page.
"""

from __future__ import annotations

import os
import json
from datetime import datetime

import pandas as pd
import streamlit as st
import dill as pickle

from ui.components import save_button
from core.utils.bo_manual import sanitize_name


def render_title_and_reset(defaults: dict) -> None:
    st.title("Single Objective Optimization Campaign")
    if st.button("Reset Campaign"):
        for key in list(st.session_state.keys()):
            if key not in ("user_email",):
                del st.session_state[key]
        for k, v in defaults.items():
            st.session_state[k] = v
        st.rerun()


def render_experiment_header(user_save_dir: str):
    """Renders experiment metadata inputs and returns tuple(name, notes, run_name, run_path)."""
    experiment_name = st.text_input("Experiment Name", value=st.session_state.get("experiment_name", ""))
    experiment_notes = st.text_area("Notes (optional)", value=st.session_state.get("experiment_notes", ""))
    _ = st.date_input("Experiment date")

    run_name = sanitize_name(experiment_name)
    st.session_state["campaign_name"] = run_name
    run_path = os.path.join(user_save_dir, run_name)
    os.makedirs(run_path, exist_ok=True)

    # keep for later use in completed state
    st.session_state.experiment_name = experiment_name
    st.session_state.experiment_notes = experiment_notes

    return experiment_name, experiment_notes, run_name, run_path


def render_save_campaign(run_path: str, target=None) -> None:
    """Save button: writes optimizer (best-effort), data CSV and metadata JSON."""
    target = target or st.sidebar
    if save_button("Save Campaign", target=target):
        try:
            with open(os.path.join(run_path, "optimizer.pkl"), "wb") as f:
                pickle.dump(st.session_state.manual_optimizer, f)
        except Exception:
            target.warning("Could not save optimizer.pkl (optimizer may be None). Proceeding with data and metadata.")

        if st.session_state.manual_data:
            df_save = pd.DataFrame(st.session_state.manual_data)
        else:
            base_cols = [name for name, *_ in st.session_state.manual_variables]
            if st.session_state.get("response"):
                base_cols.append(st.session_state["response"])
            df_save = pd.DataFrame(columns=base_cols)
        df_save.to_csv(os.path.join(run_path, "manual_data.csv"), index=False)

        metadata = {
            "variables": st.session_state.manual_variables,
            "model_variables": st.session_state.get("model_variables", st.session_state.manual_variables),
            "iteration": st.session_state.get("iteration", len(df_save)),
            "n_init": st.session_state.n_init,
            "total_iters": st.session_state.total_iters,
            "response": st.session_state.get("response", "Yield"),
            "experiment_name": st.session_state.get("experiment_name", ""),
            "experiment_notes": st.session_state.get("experiment_notes", ""),
            "initialization_complete": st.session_state.get("initial_results_submitted", False),
            "response_direction": st.session_state.get("response_direction", "Maximize"),
            "custom_objectives": st.session_state.get("custom_objectives", {}),
        }
        with open(os.path.join(run_path, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4)
        target.success(f"Campaign '{os.path.basename(run_path)}' saved successfully!")
