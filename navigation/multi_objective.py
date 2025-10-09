import os
import streamlit as st

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
        "mo_n_init": 6,
        "mo_total_iters": 20,
        "mo_init_method": "lhs",
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


_ensure_defaults()
st.title("Multiobjective Optimization Campaign")
st.success(
    """
    This session lets you:
    - Define variables and select multiple objectives with per-objective directions (maximize/minimize).
    - Create custom objectives from expressions (e.g., 0.7*Yield + 0.3*Purity).
    - Generate initial designs (Random, LHS, Halton, Maximin LHS), input results, and view the Pareto front (red line connects non-dominated points).
    - Get scalarization-based suggestions (Weighted Sum, Tchebycheff) and record batch results.
    - Save and resume multiobjective campaigns (local and Render environments).
    """
)

user_email = st.session_state.get("user_email", "default_user")
SAVE_DIR = "resumable_manual_runs" if os.getenv("RENDER") != "true" else "/mnt/data/resumable_manual_runs"
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
