from streamlit_oauth import OAuth2Component
import streamlit as st
import os
from dotenv import load_dotenv
from core.utils import db_handler



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

load_dotenv('/etc/secrets/google_auth_secrets.env')

client_id = os.getenv("CLIENT_ID")
client_secret = os.getenv("CLIENT_SECRET")
redirect_uri = os.getenv("REDIRECT_URI")

oauth2 = OAuth2Component(
    client_id=client_id,
    client_secret=client_secret,
    scope="https://www.googleapis.com/auth/userinfo.email https://www.googleapis.com/auth/userinfo.profile"
)

token = oauth2.authorize_button("🔐 Log in with Google", key="google_login", redirect_uri=redirect_uri)

if token:
    user_info = oauth2.get_user_info(token, "https://www.googleapis.com/oauth2/v2/userinfo")
    st.session_state["user_email"] = user_info["email"]
    st.session_state["user_name"] = user_info.get("name", "")
else:
    st.stop()

# ===== Sidebar: logout + user info =====
if st.sidebar.button("🚪 Log out"):
    for key in ["user_name", "user_email", "token"]:
        st.session_state.pop(key, None)
    st.rerun()
st.sidebar.write(f"👤 {st.session_state['user_name']}")
st.sidebar.write(f"✉️ {st.session_state['user_email']}")

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



