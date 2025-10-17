import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from skopt import gp_minimize
from skopt.space import Real
import pandas as pd
import plotly.express as px


st.title("BO Classroom: Simulation Case 1")

# Add an introductory section with the saved reaction image
st.markdown("### Reaction Overview")
st.image("images/image_reaction1.png", use_container_width=True)

st.markdown(
    """
This example simulates the optimization of a copper-mediated radiofluorination reaction:
**[18F]pFBnOH synthesis**, as studied in Bowden et al. (2019).

We optimize the radiochemical conversion (%RCC) using the following parameters:

- Catalyst loading (Cu(OTf)2 equivalents)
- Pyridine (ligand) equivalents
- Precursor (substrate) µmol

How to run the optimization
1. Use the sidebar to configure Bayesian Optimization (BO): iterations, initial random points, and acquisition function.
2. Adjust the parameter ranges for catalyst loading, pyridine equivalents, and substrate µmol.
3. Click "Run Optimization" to start the optimization.
4. View the best radiochemical conversion (%RCC) and the explored points.
"""
)


# Function to scale variables to [-1, 1]
def code(x, min_val, max_val):
    return 2 * (x - min_val) / (max_val - min_val) - 1


# RCC model with coded (normalized) variables from the RSM in the paper
def pfbnoh_model(params):
    Cu, Pyridine, Substrate = params
    Cu_c = code(Cu, 1.0, 4.0)
    Pyr_c = code(Pyridine, 5.0, 30.0)
    Sub_c = code(Substrate, 5.0, 25.0)

    RCC = (
        52.97
        + 13.35 * Cu_c
        - 13.43 * Pyr_c
        - 1.08 * Sub_c
        + 2.53 * Cu_c * Pyr_c
        - 2.78 * Cu_c * Sub_c
        - 11.48 * Pyr_c * Sub_c
        - 2.35 * Cu_c**2
        - 6.30 * Pyr_c**2
        - 6.10 * Sub_c**2
    )
    # Return negative for minimization; clip to physical [0,100]
    return -np.clip(RCC, 0, 100)


# --- User settings ---
st.sidebar.header("BO Settings")
n_calls = st.sidebar.slider(
    "Number of BO Iterations",
    5,
    30,
    15,
    help="Total number of optimization steps (including initial random points)."
)
n_initial_points = st.sidebar.slider(
    "Initial Random Points",
    1,
    10,
    3,
    help="How many random experiments to run before BO starts."
)
acq_func = st.sidebar.selectbox(
    "Acquisition Function",
    ["EI", "PI", "LCB"],
    help="Strategy for selecting the next experiment."
)

# --- Parameter ranges ---
st.sidebar.subheader("Parameter Ranges")
cu_min, cu_max = st.sidebar.slider(
    "Cu(OTf)2 equivalents range",
    1.0,
    4.0,
    (1.0, 4.0),
    step=0.1,
    help="Valid range for the RSM model: 1–4 eq."
)
pyr_min, pyr_max = st.sidebar.slider(
    "Pyridine equivalents range",
    5.0,
    30.0,
    (5.0, 30.0),
    step=0.5,
    help="Valid range for the RSM model: 5–30 eq."
)
sub_min, sub_max = st.sidebar.slider(
    "Substrate (µmol) range",
    5.0,
    25.0,
    (5.0, 25.0),
    step=0.5,
    help="Valid range for the RSM model: 5–25 µmol."
)

# Additional safety: clamp to the RSM-coded region
RSM_BOUNDS = {"Cu": (1.0, 4.0), "Pyridine": (5.0, 30.0), "Substrate": (5.0, 25.0)}
_adjusted = []
_cu_new = (max(cu_min, RSM_BOUNDS["Cu"][0]), min(cu_max, RSM_BOUNDS["Cu"][1]))
if _cu_new != (cu_min, cu_max):
    cu_min, cu_max = _cu_new
    _adjusted.append("Cu")
_pyr_new = (max(pyr_min, RSM_BOUNDS["Pyridine"][0]), min(pyr_max, RSM_BOUNDS["Pyridine"][1]))
if _pyr_new != (pyr_min, pyr_max):
    pyr_min, pyr_max = _pyr_new
    _adjusted.append("Pyridine")
_sub_new = (max(sub_min, RSM_BOUNDS["Substrate"][0]), min(sub_max, RSM_BOUNDS["Substrate"][1]))
if _sub_new != (sub_min, sub_max):
    sub_min, sub_max = _sub_new
    _adjusted.append("Substrate")
if _adjusted:
    st.info("Ranges adjusted to RSM model validity for: " + ", ".join(_adjusted))


# --- Run BO ---
if st.button("Run Optimization", help="Run Bayesian Optimization with the selected settings and parameter ranges."):
    space = [
        Real(cu_min, cu_max, name="Cu_eq"),
        Real(pyr_min, pyr_max, name="Pyridine_eq"),
        Real(sub_min, sub_max, name="Substrate_µmol"),
    ]

    result = gp_minimize(
        pfbnoh_model,
        dimensions=space,
        n_calls=n_calls,
        n_initial_points=n_initial_points,
        acq_func=acq_func,
        random_state=0,
    )

    # Store the result in session state
    st.session_state["last_result"] = result

    # Save explored points and RCC values for comparison
    explored_points = result.x_iters
    explored_rcc = [-y for y in result.func_vals]
    st.session_state.setdefault("comparison_data", [])
    st.session_state["comparison_data"].append(
        {
            "Best RCC": -result.fun,
            "Cu": result.x[0],
            "Pyridine": result.x[1],
            "Substrate": result.x[2],
            "Settings": {
                "n_calls": n_calls,
                "n_initial_points": n_initial_points,
                "acq_func": acq_func,
                "cu_range": (cu_min, cu_max),
                "pyr_range": (pyr_min, pyr_max),
                "sub_range": (sub_min, sub_max),
                "explored_points": explored_points,
                "explored_rcc": explored_rcc,
            },
        }
    )

    st.success(
        f"Best RCC: {-result.fun:.2f}% at Cu = {result.x[0]:.2f} eq, Pyridine = {result.x[1]:.2f} eq, Substrate = {result.x[2]:.2f} µmol"
    )
    st.caption(
        "The best result found by BO in this run. Try changing the settings or parameter ranges to see how the outcome changes!"
    )

    # Table of explored points
    df = pd.DataFrame(explored_points, columns=["Cu_eq", "Pyridine_eq", "Substrate_µmol"])
    df["% RCC"] = explored_rcc
    st.markdown("### Explored Points and RCC Values")
    st.dataframe(
        df.style.format(
            {"Cu_eq": "{:.2f}", "Pyridine_eq": "{:.2f}", "Substrate_µmol": "{:.2f}", "% RCC": "{:.2f}"}
        ),
        use_container_width=True,
    )
    st.caption(
        "Each row shows a set of conditions tested by BO and the resulting radiochemical conversion (%RCC)."
    )

    # Plot convergence + benchmarks
    def _rsm_max_estimate():
        cu_vals = np.linspace(1.0, 4.0, 41)
        pyr_vals = np.linspace(5.0, 30.0, 41)
        sub_vals = np.linspace(5.0, 25.0, 41)
        best = -1.0
        for cu in cu_vals:
            for py in pyr_vals:
                # slight speed-up: vectorize last loop
                subs = sub_vals
                vals = np.array([-pfbnoh_model([cu, py, s]) for s in subs])
                m = float(vals.max())
                if m > best:
                    best = m
        return best

    rsm_best = _rsm_max_estimate()
    article_best = 65.0

    def first_geq(values, thr):
        for i, v in enumerate(values, start=1):
            if v >= thr:
                return i
        return None

    it_to_65 = first_geq(explored_rcc, article_best)
    it_to_70 = first_geq(explored_rcc, 70.0)

    fig1, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(range(1, len(result.func_vals) + 1), explored_rcc, marker="o", label="BO")
    ax1.axhline(article_best, color="tab:red", linestyle="--", label="Article best (65%)")
    ax1.axhline(rsm_best, color="tab:green", linestyle=":", label=f"RSM optimum (~{rsm_best:.2f}%)")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("% RCC")
    ax1.set_title("Convergence Over Iterations")
    ax1.grid(True)
    ax1.legend(loc="lower right")
    st.pyplot(fig1)

    # Report how many evaluations were needed to reach targets
    col1, col2, col3 = st.columns(3)
    col1.metric("Article best (65%) reached at", it_to_65 or "not reached")
    col2.metric("70% reached at", it_to_70 or "not reached")
    col3.metric("RSM optimum (est.)", f"{rsm_best:.2f}%")

    # Parallel coordinates plot of explored points
    from sklearn.preprocessing import LabelEncoder

    def show_parallel_coordinates(data: list, response_name: str):
        if len(data) == 0:
            return
        df_local = pd.DataFrame(data).copy()
        df_local[response_name] = pd.to_numeric(df_local[response_name], errors="coerce")
        cols_to_plot = ["Cu_eq", "Pyridine_eq", "Substrate_µmol", response_name]
        df_local = df_local[cols_to_plot]
        st.markdown("### Parallel Coordinates Plot")
        # Encode object columns numerically
        for col in df_local.columns:
            if df_local[col].dtype == object:
                le = LabelEncoder()
                df_local[col] = le.fit_transform(df_local[col])
        labels = {col: col for col in df_local.columns}
        fig = px.parallel_coordinates(
            df_local,
            color=response_name,
            color_continuous_scale=px.colors.sequential.Viridis[::-1],
            labels=labels,
        )
        fig.update_layout(
            font=dict(size=20, color="black"),
            height=500,
            margin=dict(l=50, r=50, t=50, b=40),
            coloraxis_colorbar=dict(
                title=dict(text=response_name, font=dict(size=20, color="black")),
                tickfont=dict(size=20, color="black"),
                len=0.8,
                thickness=40,
                tickprefix=" ",
                xpad=5,
            ),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Call the parallel coordinates plot for BO results
    show_parallel_coordinates(df, "% RCC")


# --- Comparison Mode ---
if "comparison_data" not in st.session_state:
    st.session_state["comparison_data"] = []

if st.session_state["comparison_data"]:
    st.markdown("### Compare Saved Runs")
    comparison_df = pd.DataFrame(st.session_state["comparison_data"])
    st.dataframe(
        comparison_df.style.format(
            {"Best RCC": "{:.2f}", "Cu": "{:.2f}", "Pyridine": "{:.2f}", "Substrate": "{:.2f}"}
        ),
        use_container_width=True,
    )

    # Download comparison data
    csv = comparison_df.to_csv(index=False)
    st.download_button(
        label="Download Comparison Data",
        data=csv,
        file_name="comparison_data.csv",
        mime="text/csv",
    )

    # Visualization of comparison
    st.markdown("### Comparison Visualization")
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, run in enumerate(st.session_state["comparison_data"]):
        ax.bar(i, run["Best RCC"], label=f"Run {i+1}")
    ax.set_ylabel("Best RCC (%)")
    ax.set_title("Comparison of Best RCC Across Runs")
    ax.legend()
    st.pyplot(fig)

    st.markdown("### Full Comparison of All Experiments")

    # Combine all experiments into a single DataFrame
    all_experiments = []
    for i, run in enumerate(st.session_state["comparison_data"]):
        run_df = pd.DataFrame(
            run["Settings"].get("explored_points", []),
            columns=["Cu_eq", "Pyridine_eq", "Substrate_µmol"],
        )
        run_df["% RCC"] = run["Settings"].get("explored_rcc", [])
        run_df["Run"] = f"Run {i+1}"
        run_df["Experiment"] = range(1, len(run_df) + 1)
        all_experiments.append(run_df)

    if all_experiments:
        combined_df = pd.concat(all_experiments, ignore_index=True)

        # Scatter and line plot of all experiments
        fig2 = px.scatter(
            combined_df,
            x="Experiment",
            y="% RCC",
            color="Run",
            hover_data=["Cu_eq", "Pyridine_eq", "Substrate_µmol"],
            title="Comparison of All Experiments Across Runs",
            labels={"Experiment": "Experiment Number", "% RCC": "Radiochemical Conversion (%)"},
        )

        # Add line traces for each run with the same color as the scatter points
        for run_name, run_data in combined_df.groupby("Run"):
            fig2.add_scatter(
                x=run_data["Experiment"],
                y=run_data["% RCC"],
                mode="lines",
                line=dict(
                    color=fig2.data[[trace.name for trace in fig2.data].index(run_name)].marker.color
                ),
                name=f"{run_name} (Line)",
            )

        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No detailed experiment data available for comparison.")

# --- Clear Campaign ---
if st.button("Clear Campaign", help="Reset all saved runs and start fresh."):
    st.session_state["comparison_data"] = []
    if "last_result" in st.session_state:
        del st.session_state["last_result"]
    st.success("Campaign cleared! All saved runs have been removed.")

