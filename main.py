import importlib.util

import streamlit as st

from core.utils import db_handler
from core.utils.path_utils import resource_path


st.set_page_config(
    page_title="BO Studio - Bayesian Optimization Made Simple",
    page_icon="📊",
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


ensure_local_user()

# Dedicate a placeholder for the main body so we can clear it each rerun.
page_placeholder = st.session_state.get("_page_placeholder")
if page_placeholder is None:
    page_placeholder = st.empty()
st.session_state["_page_placeholder"] = page_placeholder


def render_page(page_path: str, sidebar_ctx) -> None:
    """Import and execute a page module inside the provided containers."""
    spec = importlib.util.spec_from_file_location("page", resource_path(page_path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    original_sidebar = st.sidebar
    st.sidebar = sidebar_ctx
    try:
        body_container = page_placeholder.container()
        body_container.empty()
        with body_container:
            module.sidebar = sidebar_ctx  # optional access inside module
            spec.loader.exec_module(module)
    finally:
        st.sidebar = original_sidebar


PAGES = {
    "🏠 Home": "navigation/Home.py",
    "🎯 Single Objective Optimization": "navigation/manual_experiments.py",
    "🔁 Multi Objective Optimization": "navigation/multi_objective.py",
    "📊 Data Analysis": "navigation/data_analysis.py",
    "🎓 Bayesian Optimization Classroom": "navigation/BO_classroom.py",
    "🧪 Simulation Case 1": "navigation/BO_classroom2.py",
    "📚 Experiment DataBase": "navigation/experiment_database.py",
}

# Build the entire sidebar inside a single container so it resets every rerun.
sidebar_root = st.sidebar.empty()
sidebar_ctx = sidebar_root.container()
sidebar_ctx.write(f"👤 {st.session_state.get('user_name', '')}")
sidebar_ctx.write(f"✉️ {st.session_state.get('user_email', '')}")
sidebar_ctx.caption("Running in local mode (authentication disabled)")
sidebar_ctx.image(str(resource_path("images/image.png")), width=300)
sidebar_ctx.title("📁 Navigation")
selection = sidebar_ctx.radio("Go to", list(PAGES.keys()))

# Dynamic controls for the current page render below the navigation.
dynamic_sidebar = sidebar_ctx.container()

render_page(PAGES[selection], dynamic_sidebar)
