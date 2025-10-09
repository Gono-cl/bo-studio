import os
import pandas as pd
import streamlit as st

from core.utils.acquisition_viz import plot_gp_and_acq_1d
from core.utils.bo_manual import rebuild_optimizer_from_df
from core.utils import db_handler


st.title("Acquisition Explorer (1D)")
st.success("Visualize the GP posterior and acquisition function along one variable, holding others fixed.")

# ------------------------- Data/optimizer source -------------------------
source = st.radio("Data source", ["Current Session", "Saved Campaign", "Database"], index=0)

opt = None
vars_ms = []
data = []

if source == "Current Session":
    opt = st.session_state.get("manual_optimizer")
    vars_ms = st.session_state.get("manual_variables", [])
    data = st.session_state.get("manual_data", [])
elif source == "Saved Campaign":
    user_email = st.session_state.get("user_email", "default_user")
    SAVE_DIR = "/mnt/data/resumable_manual_runs" if os.getenv("RENDER") == "true" else os.path.join(os.getcwd(), "resumable_manual_runs")
    user_save_dir = os.path.join(SAVE_DIR, user_email)
    options = ["None"]
    metas = {}
    if os.path.isdir(user_save_dir):
        for d in sorted(os.listdir(user_save_dir)):
            p = os.path.join(user_save_dir, d)
            meta_p = os.path.join(p, "metadata.json")
            data_p = os.path.join(p, "manual_data.csv")
            if os.path.isdir(p) and os.path.exists(meta_p) and os.path.exists(data_p):
                try:
                    import json
                    with open(meta_p, "r") as f:
                        meta = json.load(f)
                    if meta.get("mode") not in ("multiobjective",):
                        options.append(d)
                        metas[d] = (meta, data_p)
                except Exception:
                    continue
    sel = st.selectbox("Select campaign", options)
    if sel != "None":
        meta, data_p = metas[sel]
        df_loaded = pd.read_csv(data_p)
        vars_ms = meta.get("variables", [])
        resp = meta.get("response", st.session_state.get("response", None))
        st.session_state.response = resp
        st.session_state.response_direction = meta.get("response_direction", st.session_state.get("response_direction", "Maximize"))
        st.session_state.manual_variables = vars_ms
        data = df_loaded.to_dict("records")
        try:
            opt = rebuild_optimizer_from_df(vars_ms, df_loaded, resp, n_initial_points_remaining=0, acq_func=meta.get("acq_func", "EI"))
        except Exception:
            opt = None
elif source == "Database":
    user_email = st.session_state.get("user_email", "default_user")
    rows = db_handler.list_experiments(user_email)
    if not rows:
        st.info("No experiments found in database for this user.")
    else:
        options = {f"{rid} — {name} ({ts})": rid for rid, name, ts in rows}
        label = st.selectbox("Select experiment", list(options.keys()))
        exp = db_handler.load_experiment(options[label])
        if exp:
            df_loaded = exp["df_results"]
            vars_ms = exp["variables"]
            st.session_state.manual_variables = vars_ms
            settings = exp.get("settings", {}) or {}
            resp = settings.get("objective", st.session_state.get("response", None))
            st.session_state.response = resp
            st.session_state.response_direction = st.session_state.get("response_direction", "Maximize")
            data = df_loaded.to_dict("records")
            try:
                opt = rebuild_optimizer_from_df(vars_ms, df_loaded, resp, n_initial_points_remaining=0, acq_func=settings.get("acq_func", "EI"))
            except Exception:
                opt = None

if not opt or not vars_ms:
    st.info("Run or load a single-objective campaign first.")
    st.stop()

varnames = [n for n, *_ in vars_ms]
vary = st.selectbox("Variable to explore", varnames)

# Build fixed values UI
st.markdown("#### Fix remaining variables")
fixed = {}
df = pd.DataFrame(data) if data else pd.DataFrame(columns=varnames)
cols = st.columns(2)
for i, (name, v1, v2, _u, vtype) in enumerate(vars_ms):
    if name == vary:
        continue
    with cols[i % 2]:
        if vtype == "continuous":
            default = None
            if name in df.columns and not df.empty:
                try:
                    default = float(pd.to_numeric(df[name], errors="coerce").median())
                except Exception:
                    default = None
            if default is None:
                default = 0.5 * (float(v1) + float(v2))
            fixed[name] = st.number_input(f"{name}", value=float(default))
        else:
            cats = list(v1)
            default = df[name].mode().iloc[0] if name in df.columns and not df.empty else cats[0]
            fixed[name] = st.selectbox(f"{name}", options=cats, index=cats.index(default) if default in cats else 0)

xi = st.number_input("Exploration (xi)", min_value=0.0, max_value=1.0, value=0.01, step=0.01)

if st.button("Plot GP and Acquisition"):
    direction = st.session_state.get("response_direction", "Maximize")
    response = st.session_state.get("response", None)
    plot_gp_and_acq_1d(opt, vary, fixed, data_df=df if not df.empty else None, response_name=response, direction=direction, xi=xi)

