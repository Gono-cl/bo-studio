import importlib.util

import streamlit as st

from core.utils import db_handler
from core.utils.path_utils import resource_path


st.set_page_config(
    page_title="BO Studio - Bayesian Optimization Made Simple",
    page_icon="BO",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

db_handler.init_db()


def ensure_local_user() -> None:
    st.session_state.setdefault("user_email", "local_user@example.com")
    st.session_state.setdefault("user_name", "LocalUser")
    st.session_state.setdefault("token", "local_token")


def render_page(page_path: str) -> None:
    spec = importlib.util.spec_from_file_location("page_module", resource_path(page_path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)


ensure_local_user()


PAGES = {
    "Home": "navigation/Home.py",
    "Single Objective Optimization": "navigation/manual_experiments.py",
    "Multi Objective Optimization": "navigation/multi_objective.py",
    "Data Analysis": "navigation/data_analysis.py",
    "Bayesian Optimization Classroom": "navigation/bo_classroom_v2.py",
    "Experiment Database": "navigation/experiment_database.py",
}

with st.sidebar:
    st.write(f"User: {st.session_state.get('user_name', '')}")
    st.write(f"Email: {st.session_state.get('user_email', '')}")
    st.caption("Running in local mode (authentication disabled)")
    st.caption("Build: manual-router-v1")
    st.image(str(resource_path("images/image.png")), width=300)
    st.title("Navigation")
    selection = st.radio("Go to", list(PAGES.keys()), key="main_nav_selection")

render_page(PAGES[selection])
