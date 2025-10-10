import os
import numpy as np
import pandas as pd
import streamlit as st

from ui.charts import Charts
from core.utils.init_designs import generate_initial_points
from core.utils.acquisition_viz import plot_gp_and_acq_1d
from core.utils.bo_manual import rebuild_optimizer_from_df
from ui.sections.analysis_mo import render_analysis_mo
from ui.sections.mo_interact import _build_scalarized_optimizer  # reuse helper


st.title("BO Classroom 2.0")
st.success(
    """
    Explore Bayesian Optimization concepts interactively:
    - Foundations: initial designs, GP posterior and acquisition (1D slices).
    - Practice Lab: quick best‑so‑far tracker on your session data.
    - Multiobjective: scalarization trainer and Pareto diagnostics.
    """
)


tab1, tab2, tab3 = st.tabs(["Foundations", "Practice Lab", "Multiobjective"])


with tab1:
    st.subheader("Initial Designs")
    vars_ms = st.session_state.get("manual_variables", [])
    if not vars_ms:
        st.info("Define variables in the Manual Optimization page to see demos here.")
    else:
        colA, colB = st.columns(2)
        with colA:
            n_init = st.slider("n_init", 3, 50, 8)
        with colB:
            method = st.selectbox("Method", ["Random", "LHS", "Halton", "Maximin LHS"]) 
        pts = generate_initial_points(vars_ms, n_init, method=method.lower().replace(" ", "_"), seed=42)
        with st.expander("Preview Initial Design", expanded=False):
            Charts.show_initial_design(pts, vars_ms)

    st.subheader("GP + Acquisition (1D slice)")
    opt = st.session_state.get("manual_optimizer")
    data = st.session_state.get("manual_data", [])
    if not vars_ms or not opt:
        st.info("Run or load a single‑objective campaign to visualize GP and acquisition.")
    else:
        varnames = [n for n, *_ in vars_ms]
        vary = st.selectbox("Variable to explore", varnames, key="classroom_vary")
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
                    fixed[name] = st.number_input(f"{name}", value=float(default), key=f"fix_{name}")
                else:
                    cats = list(v1)
                    default = df[name].mode().iloc[0] if name in df.columns and not df.empty else cats[0]
                    fixed[name] = st.selectbox(f"{name}", options=cats, index=cats.index(default) if default in cats else 0, key=f"fix_{name}")
        xi = st.number_input("Exploration (xi)", min_value=0.0, max_value=1.0, value=0.01, step=0.01, key="classroom_xi")
        res = st.slider("Resolution (points)", 50, 400, 150, 25, key="classroom_res")
        if st.button("Plot 1D GP + AF", key="plot_gp_af"):
            direction = st.session_state.get("response_direction", "Maximize")
            response = st.session_state.get("response", None)
            plot_gp_and_acq_1d(opt, vary, fixed, data_df=df if not df.empty else None, response_name=response, direction=direction, xi=xi, resolution=res)


with tab2:
    st.subheader("Practice Lab (Best‑so‑far)")
    df = pd.DataFrame(st.session_state.get("manual_data", []))
    response = st.session_state.get("response", None)
    if df.empty or not response or response not in df.columns:
        st.info("Collect some results in Manual Optimization to see progress here.")
    else:
        df[response] = pd.to_numeric(df[response], errors="coerce")
        vals = df[response].dropna().tolist()
        direction = st.session_state.get("response_direction", "Maximize")
        best_so_far = []
        cur = None
        for v in vals:
            if cur is None:
                cur = v
            else:
                cur = max(cur, v) if direction == "Maximize" else min(cur, v)
            best_so_far.append(cur)
        import plotly.express as px
        tdf = pd.DataFrame({"iteration": range(1, len(best_so_far) + 1), "best": best_so_far})
        fig = px.line(tdf, x="iteration", y="best", markers=True)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Tip: use the Acquisition Explorer and Initial Designs in the Foundations tab to reason about next points.")


with tab3:
    st.subheader("Scalarization Trainer & Pareto")
    mo_df = pd.DataFrame(st.session_state.get("mo_data", []))
    vars_ms = st.session_state.get("manual_variables", [])
    mo_objs = st.session_state.get("mo_objectives", [])
    mo_dirs = st.session_state.get("mo_directions", {})
    if mo_df.empty or not mo_objs:
        st.info("Run or load a multiobjective session to explore Pareto and scalarization.")
    else:
        # Pareto diagnostics
        render_analysis_mo(mo_df, objectives=mo_objs[: min(3, len(mo_objs))], directions=mo_dirs)

        st.markdown("#### Propose a scalarized suggestion")
        method = st.selectbox("Scalarization", ["Weighted Sum", "Tchebycheff"], key="classroom_scalar_method")
        w_raw = []
        cols = st.columns(min(3, len(mo_objs)))
        for i, o in enumerate(mo_objs):
            with cols[i % len(cols)]:
                w_raw.append(st.number_input(f"w[{o}]", min_value=0.0, max_value=1.0, value=1.0 / len(mo_objs), step=0.05, key=f"w_{o}"))
        w = np.array(w_raw, dtype=float)
        if w.sum() == 0:
            w = np.ones_like(w) / len(w)
        else:
            w = w / w.sum()
        if st.button("Suggest via scalarization", key="classroom_suggest_mo"):
            try:
                opt = _build_scalarized_optimizer(w, method="tchebycheff" if method.lower().startswith("tch") else "weighted_sum")
                x = opt.suggest()
                st.success("Proposed point (per scalarization):")
                st.write({name: val for (name, *_), val in zip(vars_ms, x)})
            except Exception as ex:
                st.error(f"Could not propose suggestion: {ex}")

