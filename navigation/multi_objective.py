import os
import streamlit as st
from core.utils.app_paths import get_campaigns_dir

from ui.sections.mo_setup import render_mo_setup_and_initials
from ui.sections.mo_interact import render_mo_interact_and_pareto
from ui.sections.mo_header import render_mo_experiment_header, render_mo_save_campaign
from ui.sections.mo_resume import render_mo_resume
from ui.sections.mo_reuse import render_mo_reuse_seeds


def _ensure_defaults():
    defaults = {
        "manual_variables": [],
        "mo_objectives": ["Yield", "Conversion"],
        "mo_initialized": False,
        "mo_suggestions": [],
        "mo_data": [],
        "mo_custom_objectives": [],
        "mo_n_init": 6,
        "mo_total_iters": 20,
        "mo_iteration": 0,
        "mo_pending_df": [],
        "mo_init_method": "lhs",
        "acq_func": "EI",
        "acq_xi": 0.01,
        "acq_kappa": 1.96,
        "bo_seed": 42,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)
    return defaults


defaults = _ensure_defaults()
st.title("Multiobjective Optimization Campaign")
if st.button("Reset Campaign"):
    keep_keys = {"user_email", "main_nav_selection"}
    for key in list(st.session_state.keys()):
        if key not in keep_keys:
            del st.session_state[key]
    for k, v in defaults.items():
        st.session_state[k] = v
    st.rerun()

user_email = st.session_state.get("user_email", "default_user")
SAVE_DIR = str(get_campaigns_dir())
user_save_dir = os.path.join(SAVE_DIR, user_email)
os.makedirs(user_save_dir, exist_ok=True)

# Sidebar resume (MO only)
render_mo_resume(user_save_dir)

# Header + Save
experiment_name, experiment_notes, run_name, run_path = render_mo_experiment_header(user_save_dir)
render_mo_save_campaign(run_path)

render_mo_setup_and_initials()
render_mo_reuse_seeds(user_save_dir)
render_mo_interact_and_pareto(user_save_dir)
