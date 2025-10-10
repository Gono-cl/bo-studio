import numpy as np
import pandas as pd
import streamlit as st

from ui.charts import Charts
from core.utils.init_designs import generate_initial_points
from core.utils.acquisition_viz import plot_gp_and_acq_1d
from core.utils.bo_manual import rebuild_optimizer_from_df, safe_build_optimizer
from ui.sections.analysis_mo import render_analysis_mo
from core.sim.so_functions import so_demo_space, so_demo_eval
from core.sim.mo_functions import mo_demo_space, mo_demo_eval
from core.sim.chem_functions import chem_demo_space, chem_eval_row


st.title("BO Classroom 2.0")
st.success(
    """
    A self‑contained classroom for learning Bayesian Optimization without prior data.
    - Theory: key ideas (surrogate, acquisition, exploration–exploitation, scalarization, Pareto).
    - Foundations: initial designs and GP + acquisition visualization on demo functions.
    - Practice Lab: run and compare simulated campaigns.
    - Multiobjective: Pareto and scalarization on a chemistry‑like demo.
    """
)

with st.expander("Theory: key concepts", expanded=False):
    st.markdown(
        """
        - Surrogate (Gaussian Process): fits a probability model to observed experiments; returns mean and uncertainty.
        - Acquisition (EI/PI/LCB): uses the surrogate to score candidates for the next experiment.
          - EI (Expected Improvement): favors points likely to beat the current best by a margin (xi tunes exploration).
          - PI (Probability of Improvement): maximizes chance of improvement (more greedy).
          - LCB (Lower Confidence Bound): explores by preferring high uncertainty (kappa tunes exploration).
        - Initialization (DoE): Random, LHS, Halton, Maximin LHS provide different space coverage.
        - Multiobjective: convert multiple goals to one via scalarization (Weighted Sum, Tchebycheff) or analyze the Pareto front directly.
        - Pareto front: set of non‑dominated trade‑offs; knee points are high‑leverage choices; hypervolume measures front quality.
        """
    )


tab1, tab2, tab3 = st.tabs(["Foundations", "Practice Lab", "Multiobjective"])


with tab1:
    st.subheader("Initial Designs")
    demo_fn = st.selectbox("Demo function", [
        "Forrester (1D)",
        "Chemistry: Yield vs Temperature & Catalyst (2 vars)",
        "Chemistry: Yield vs Temp/Cat/Pressure/Residence/Polarity (5 vars)",
    ], index=0)
    if demo_fn.startswith("Forrester"):
        vars_ms = so_demo_space("forrester")
        eval_fn = so_demo_eval("forrester")
    else:
        if "5 vars" in demo_fn:
            vars_ms = chem_demo_space("extended")
            eval_fn = lambda row: chem_eval_row(row, mode="extended")
        else:
            vars_ms = chem_demo_space("basic")
            eval_fn = lambda row: chem_eval_row(row, mode="basic")

    colA, colB = st.columns(2)
    with colA:
        n_init = st.slider("n_init", 3, 50, 8)
    with colB:
        method = st.selectbox("Method", ["Random", "LHS", "Halton", "Maximin LHS"]) 
    pts = generate_initial_points(vars_ms, n_init, method=method.lower().replace(" ", "_"), seed=42)
    with st.expander("Preview Initial Design", expanded=False):
        Charts.show_initial_design(pts, vars_ms)

    st.subheader("GP + Acquisition (1D slice)")
    # Create synthetic observations from the selected demo
    varnames = [n for n, *_ in vars_ms]
    # Use the current n_init setting for synthetic observations as well
    demo_pts = generate_initial_points(vars_ms, int(n_init), method="lhs", seed=0)
    rows = []
    for row in demo_pts:
        y = eval_fn(row)
        rows.append({varnames[i]: row[i] for i in range(len(varnames))} | {"y": y})
    df_demo = pd.DataFrame(rows)
    opt_demo = rebuild_optimizer_from_df(vars_ms, df_demo, response_col="y", n_initial_points_remaining=0, acq_func="EI")

    vary = varnames[0] if len(varnames) == 1 else st.selectbox("Variable to explore (1D slice)", varnames, key="classroom_vary")
    fixed = {}
    cols = st.columns(2)
    for i, (name, v1, v2, _u, vtype) in enumerate(vars_ms):
        if name == vary:
            continue
        with cols[i % 2]:
            default = 0.5 * (float(v1) + float(v2))
            fixed[name] = st.number_input(f"{name}", value=float(default), key=f"fix_{name}")
    xi = st.number_input("Exploration (xi)", min_value=0.0, max_value=1.0, value=0.01, step=0.01, key="classroom_xi")
    res = st.slider("Resolution (points)", 50, 400, 150, 25, key="classroom_res")
    if st.button("Plot 1D GP + AF", key="plot_gp_af"):
        plot_gp_and_acq_1d(opt_demo, vary, fixed, data_df=df_demo, response_name="y", direction="Maximize", xi=xi, resolution=res)


with tab2:
    st.subheader("Practice Lab (Simulated runs)")
    demo_choice = st.selectbox("Demo function", [
        "Forrester (1D)",
        "Chemistry: Yield vs Temperature & Catalyst (2 vars)",
        "Chemistry: Yield vs Temp/Cat/Pressure/Residence/Polarity (5 vars)",
    ], index=1)
    col1, col2, col3 = st.columns(3)
    with col1:
        budget = st.number_input("Total iterations", min_value=5, max_value=100, value=20)
    with col2:
        n_init = st.number_input("Initial points", min_value=2, max_value=20, value=6)
    with col3:
        acq = st.selectbox("Acquisition", ["EI", "PI", "LCB"], index=0)
    method = st.selectbox("Init method", ["Random", "LHS", "Halton", "Maximin LHS"], index=1)
    run_name = st.text_input("Run name", value="demo_run_1")
    if st.button("Run Demo BO"):
        if demo_choice.startswith("Forrester"):
            space = so_demo_space("forrester")
            fn = so_demo_eval("forrester")
        else:
            if "5 vars" in demo_choice:
                space = chem_demo_space("extended")
                fn = lambda row: chem_eval_row(row, mode="extended")
            else:
                space = chem_demo_space("basic")
                fn = lambda row: chem_eval_row(row, mode="basic")
        # Build optimizer
        from skopt.space import Real, Categorical
        dims = [Real(v1, v2, name=name) if t == "continuous" else Categorical(v1, name=name) for name, v1, v2, _u, t in space]
        opt = safe_build_optimizer(dims, n_initial_points_remaining=0, acq_func=acq)
        # Initial design
        X = generate_initial_points(space, int(n_init), method=method.lower().replace(" ", "_"), seed=0)
        Y = []
        for row in X:
            y = fn(row)
            opt.observe(row, -y)
            Y.append(y)
        # BO loop
        for _ in range(int(budget) - int(n_init)):
            x = opt.suggest()
            y = fn(x)
            opt.observe(x, -y)
            X.append(x)
            Y.append(y)
        df_run = pd.DataFrame([{space[i][0]: X[k][i] for i in range(len(space))} | {"y": Y[k]} for k in range(len(X))])
        st.session_state.setdefault("classroom_runs", {})[run_name] = {"df": df_run, "settings": {"budget": int(budget), "n_init": int(n_init), "acq": acq, "method": method, "demo": demo_choice}}
        st.success(f"Run '{run_name}' completed.")

    # Compare runs
    runs = st.session_state.get("classroom_runs", {})
    if runs:
        import plotly.express as px
        fig = None
        for name, payload in runs.items():
            df_run = payload["df"].copy()
            df_run["y"] = pd.to_numeric(df_run["y"], errors="coerce")
            vals = df_run["y"].tolist()
            best = []
            cur = None
            for v in vals:
                cur = v if cur is None else max(cur, v)
                best.append(cur)
            tdf = pd.DataFrame({"iteration": range(1, len(best) + 1), "best": best, "run": name})
            fig = px.line(tdf, x="iteration", y="best", color="run", markers=True) if fig is None else fig.add_traces(px.line(tdf, x="iteration", y="best", color="run").data)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Tip: compare acquisitions and init methods; the chemistry demo mimics Temperature/Catalyst trade‑offs.")


with tab3:
    st.subheader("Scalarization Trainer & Pareto")
    # Demo MO dataset (Schaffer N.1 transformed for maximization)
    space = mo_demo_space("schaffer")
    x_vals = np.linspace(space[0][1], space[0][2], 200)
    f1 = []
    f2 = []
    for x in x_vals:
        a, b = mo_demo_eval("schaffer")([x])
        f1.append(a)
        f2.append(b)
    mo_df = pd.DataFrame({space[0][0]: x_vals, "f1": f1, "f2": f2})
    mo_dirs = {"f1": "Maximize", "f2": "Maximize"}
    render_analysis_mo(mo_df, objectives=["f1", "f2"], directions=mo_dirs)

    st.markdown("#### Scalarization Playground (grid-based)")
    w1 = st.slider("w[f1]", 0.0, 1.0, 0.5, 0.05)
    w2 = 1.0 - w1
    score = w1 * mo_df["f1"].to_numpy() + w2 * mo_df["f2"].to_numpy()
    idx = int(np.argmax(score))
    st.write("Suggested x (grid argmax):", float(mo_df.iloc[idx][space[0][0]]))
    st.write({"f1": float(mo_df.iloc[idx]["f1"]), "f2": float(mo_df.iloc[idx]["f2"])})
