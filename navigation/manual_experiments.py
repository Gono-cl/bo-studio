import os
import streamlit as st
import pandas as pd
import dill as pickle  # persistence
from core.utils.app_paths import get_campaigns_dir

from ui.sections.resume import render_resume_exact
from ui.sections.header import render_title_and_reset, render_experiment_header, render_save_campaign
from ui.sections.variables import render_variables_section
from ui.sections.setup import render_setup_and_initials
from ui.sections.reuse import render_reuse_seeds
from ui.sections.interact import render_interact_and_complete


# =========================================================
# Config & Session Defaults
# =========================================================
SAVE_DIR = str(get_campaigns_dir())

defaults = {
    "manual_variables": [],       # SuggestSpace
    "model_variables": None,      # ModelSpace (union when reusing; else same as manual_variables)
    "manual_data": [],
    "manual_optimizer": None,
    "manual_initialized": False,
    "suggestions": [],
    "iteration": 0,
    "initial_results_submitted": False,
    "next_suggestion_cached": None,
    "submitted_initial": False,
    "edited_initial_df": None,
    "n_init": 5,
    "total_iters": 10,
    "edit_mode": False,
    "recalc_needed": False,
    "response": "Yield",
    "custom_objectives": [],
    "var_type": "Continuous",
    "acq_xi": 0.01,
    "acq_kappa": 1.96,
    "bo_seed": 42,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v



# =========================================================
# Title, Header & Save
# =========================================================
user_email = st.session_state.get("user_email", "default_user")
user_save_dir = os.path.join(SAVE_DIR, user_email)
os.makedirs(user_save_dir, exist_ok=True)
render_title_and_reset(defaults)

# Sidebar controls (below navigation): resume/reuse follow MO structure.
render_resume_exact(user_save_dir, target=st.sidebar, show_divider=True)

experiment_name, experiment_notes, run_name, run_path = render_experiment_header(user_save_dir)
render_save_campaign(run_path, target=st.sidebar)


# =========================================================
# Variables, Setup, Reuse, Interaction
# =========================================================
render_variables_section()
render_setup_and_initials()
render_reuse_seeds(user_save_dir, target=st.sidebar, show_divider=True)
render_interact_and_complete(user_save_dir, experiment_name, experiment_notes, run_name)

