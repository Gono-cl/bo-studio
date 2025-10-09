import os
import importlib.util
from pathlib import Path
import streamlit as st

from core.utils import db_handler
from auth import get_login_url, get_token, get_user_info


st.set_page_config(
    page_title="BO Studio — Bayesian Optimization Made Simple",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

hide_streamlit_style = """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

db_handler.init_db()

# Handle logout
query_params = st.query_params
if "logout" in query_params:
    for key in ["user_email", "user_name", "token"]:
        st.session_state.pop(key, None)
    st.query_params.clear()
    st.rerun()

# Auth: local vs server
if os.getenv("RENDER") != "true":
    st.session_state.setdefault("user_email", "local_user@example.com")
    st.session_state.setdefault("user_name", "LocalUser")
    st.session_state.setdefault("token", "local_token")
else:
    if "user_email" not in st.session_state:
        if "code" in query_params:
            code = query_params["code"]
            token_data = get_token(code)
            access_token = token_data.get("access_token")
            if access_token:
                user_info = get_user_info(access_token)
                st.session_state["user_email"] = user_info.get("email")
                st.session_state["user_name"] = user_info.get("name", "")
                st.session_state["token"] = access_token
                st.query_params.clear()
                st.rerun()
            else:
                st.error("Failed to get access token.")
                st.stop()
        else:
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                st.image("images/image.png", width=500)
            
            st.markdown(
                f"""
                <div style="display: flex; align-items: center; justify-content: center; flex-direction: column;">
                    <p style=\"max-width: 600px; color: #555;\">
                        Welcome to <b>BO Studio</b>! Run, track, and analyze your optimization experiments with ease.<br>
                        Log in with Google to get started and access your personal experiment database.
                    </p>
                </div>
                <div style=\"display: flex; justify-content: center; margin-top: 30px;\">
                    <a href=\"{get_login_url()}\" target=\"_self\"> 
                        <button style=\"font-size: 18px; padding: 8px 24px;\">Log in with Google</button>
                    </a>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.stop()

# Sidebar header
st.sidebar.write(f"👤 {st.session_state.get('user_name', '')}")
st.sidebar.write(f"✉️ {st.session_state.get('user_email', '')}")
if os.getenv("RENDER") == "true":
    if st.sidebar.button("🚪 Log out"):
        st.experimental_set_query_params(logout="1")
        st.rerun()
else:
    st.sidebar.caption("Running in local mode")

# Pages
PAGES = {
    "🏠 Home": "navigation/Home.py",
    "🧪 Manual Optimization": "navigation/manual_experiments.py",
    "🎯 Multiobjective (Manual)": "navigation/multi_objective.py",
    "🗄️ Experiment DataBase": "navigation/experiment_database.py",
    "📝 Preview Saved Run": "navigation/preview_run.py",
    "📚 Bayesian Optimization Classroom": "navigation/BO_classroom.py",
    "🧪 Simulation Case 1": "navigation/BO_classroom2.py",
    "📈 Acquisition Explorer": "navigation/af_explorer.py",
    "📊 Data Analysis": "navigation/data_analysis.py",
    "❓ FAQ — Help & Guidance": "navigation/faq.py",
}

st.sidebar.image("images/image.png", width=300)
st.sidebar.title("📁 Navigation")

if "selected_page" in st.session_state:
    selection = st.session_state.selected_page
    del st.session_state.selected_page
else:
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))


def load_page(page_path: str) -> None:
    spec = importlib.util.spec_from_file_location("page", Path(page_path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)


load_page(PAGES[selection])
