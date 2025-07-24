from streamlit_oauth import OAuth2Component
import streamlit as st
import os
from dotenv import load_dotenv
import urllib.parse
from core.utils import db_handler
import requests
import importlib.util
from pathlib import Path
from auth import get_login_url, get_token, get_user_info



# ===== Streamlit page configuration =====
st.set_page_config(
    page_title="BO Studio – Bayesian Optimization Made Simple",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== Hide default Streamlit UI =====
hide_streamlit_style = """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# ===== Initialize database =====
db_handler.init_db()

# ===== Initialize Google Login =====


# --- Main logic ---
query_params = st.query_params
if "logout" in query_params:
    for key in ["user_email", "user_name", "token"]:
        st.session_state.pop(key, None)
    st.query_params.clear()
    st.rerun()

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
        st.markdown(f'<a href="{get_login_url()}" target="_self"><button>🔐 Log in with Google</button></a>', unsafe_allow_html=True)
        st.stop()

# --- User is logged in ---
st.sidebar.write(f"👤 {st.session_state['user_name']}")
st.sidebar.write(f"✉️ {st.session_state['user_email']}")
if st.sidebar.button("🚪 Log out"):
    st.experimental_set_query_params(logout="1")
    st.rerun()

# ===== Define app pages =====
PAGES = {
    "🏠 Home": "Home.py",
    "🧰 Manual Optimization": "manual_experiments.py",
    "📚 Experiment DataBase": "experiment_database.py",
    "🔍 Preview Saved Run": "preview_run.py",
    "🎓 Bayesian Optimization Classroom": "BO_classroom.py",
    "❓ FAQ – Help & Guidance": "faq.py"
}

# ===== Sidebar navigation =====
st.sidebar.image("assets/image.png", width=300)
st.sidebar.title("📁 Navigation")

if "selected_page" in st.session_state:
    selection = st.session_state.selected_page
    del st.session_state.selected_page
else:
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))

# ===== Load selected page =====
def load_page(page_path):
    spec = importlib.util.spec_from_file_location("page", Path(page_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

load_page(PAGES[selection])




