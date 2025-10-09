import streamlit as st
import pandas as pd
import os

from core.utils.acquisition_viz import plot_gp_and_acq_1d


st.title("Acquisition Explorer (1D)")
st.success("Visualize the GP posterior and acquisition function along one variable, holding others fixed.")

opt = st.session_state.get("manual_optimizer")
vars_ms = st.session_state.get("manual_variables", [])
data = st.session_state.get("manual_data", [])

if not opt or not vars_ms:
    st.info("Run or load a single‑objective campaign first.")
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
            # baseline: median from data, else midpoint
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

