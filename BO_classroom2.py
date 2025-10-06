import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from skopt import gp_minimize
from skopt.space import Real
import pandas as pd  # Ensure pandas is available globally
import plotly.express as px  # Ensure plotly.express is available globally


st.title("🎯 BO Classroom: Simulation Case 1")

# Add an introductory section with the saved reaction image
st.markdown("### Reaction Overview")
st.image("image_reaction1.png", use_container_width=True)

st.markdown("""
This example simulates the optimization of a copper-mediated radiofluorination reaction:
**[¹⁸F]pFBnOH synthesis**, as studied in [Bowden et al. (2019)](https://www.nature.com/articles/s41598-019-47846-6).

We optimize the radiochemical conversion (%RCC) using the following parameters:

- **Catalyst loading (Cu(OTf)₂ equivalents)**: The amount of catalyst used in the reaction, typically ranging from 1 to 4 equivalents.
- **Pyridine (ligand) equivalents**: The amount of ligand (Pyridine) used, usually between 5 and 30 equivalents.
- **Precursor (substrate) µmol**: The amount of precursor (substrate) in micromoles, typically between 5 and 25 µmol.

### How to Run the Optimization
1. Use the sidebar to configure the Bayesian Optimization (BO) settings, such as the number of iterations, initial random points, and acquisition function.
2. Adjust the parameter ranges for catalyst loading, pyridine equivalents, and substrate µmol to explore different conditions.
3. Click the **"🚀 Run Optimization"** button to start the optimization process.
4. View the results, including the best radiochemical conversion (%RCC) and the explored points, in the main interface.

### Why This Reaction Was Selected
The [¹⁸F]pFBnOH synthesis reaction is a well-studied example in the field of radiochemistry. It serves as an excellent case study for optimization because:
- It involves multiple parameters that influence the outcome, making it ideal for demonstrating the power of Bayesian Optimization.
- The goal is to determine the minimum number of experiments needed to achieve the maximum radiochemical conversion (%RCC), showcasing the efficiency of BO in reducing experimental effort.
""")

# --- Synthetic surrogate model (mock coefficients based on DoE trends) ---
def pfbnoh_model(params):
    Cu, Pyridine, Substrate = params
    RCC = (
        -6.59
        + 14.53 * Cu
        + 1.38 * Pyridine
        + 3.79 * Substrate
        + 0.13 * Cu * Pyridine
        - 0.19 * Cu * Substrate
        - 0.09 * Pyridine * Substrate
        - 1.04 * Cu**2
        - 0.04 * Pyridine**2
        - 0.06 * Substrate**2
    )
    RCC = np.clip(RCC, 0, 100)
    return -RCC  # for maximization


# --- User settings ---
st.sidebar.header("⚙️ BO Settings")
n_calls = st.sidebar.slider(
    "Number of BO Iterations",
    5, 30, 15,
    help="Total number of optimization steps (including initial random points). More steps = more exploration."
)
n_initial_points = st.sidebar.slider(
    "Initial Random Points",
    1, 10, 3,
    help="How many random experiments to run before BO starts."
)
acq_func = st.sidebar.selectbox(
    "Acquisition Function",
    ["EI", "PI", "LCB"],
    help="Strategy for selecting the next experiment: EI (Expected Improvement), PI (Probability of Improvement), LCB (Lower Confidence Bound)."
)

# --- Parameter ranges ---
st.sidebar.subheader("Parameter Ranges")
cu_min, cu_max = st.sidebar.slider(
    "Cu(OTf)₂ equivalents range",
    0.5, 10.0, (3.0, 7.0), step=0.1,
    help="Range of catalyst loading (Cu). Typical: 1–4 eq."
)
pyr_min, pyr_max = st.sidebar.slider(
    "Pyridine equivalents range",
    1.0, 50.0, (10.0, 30.0), step=0.5,
    help="Range of ligand (Pyridine) equivalents. Typical: 5–30 eq."
)
sub_min, sub_max = st.sidebar.slider(
    "Substrate (µmol) range",
    1.0, 50.0, (10.0, 30.0), step=0.5,
    help="Range of precursor (substrate) in µmol. Typical: 5–25 µmol."
)

# --- Run BO ---

if st.button("🚀 Run Optimization", help="Run Bayesian Optimization with the selected settings and parameter ranges."):
    space = [
        Real(cu_min, cu_max, name="Cu_eq"),
        Real(pyr_min, pyr_max, name="Pyridine_eq"),
        Real(sub_min, sub_max, name="Substrate_µmol")
    ]

    result = gp_minimize(
        pfbnoh_model,
        dimensions=space,
        n_calls=n_calls,
        n_initial_points=n_initial_points,
        acq_func=acq_func,
        random_state=0
    )

    # Store the result in session state
    st.session_state["last_result"] = result

    # Save explored points and RCC values for comparison
    explored_points = result.x_iters
    explored_rcc = [-y for y in result.func_vals]
    st.session_state["comparison_data"].append({
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
            "explored_rcc": explored_rcc
        }
    })

    st.success(f"Best RCC: {-result.fun:.2f}% at Cu = {result.x[0]:.2f} eq, Pyridine = {result.x[1]:.2f} eq, Substrate = {result.x[2]:.2f} µmol")
    st.caption("The best result found by BO in this run. Try changing the settings or parameter ranges to see how the outcome changes!")

    # Table of explored points
    df = pd.DataFrame(explored_points, columns=["Cu_eq", "Pyridine_eq", "Substrate_µmol"])
    df["% RCC"] = explored_rcc
    st.markdown("### Explored Points and RCC Values")
    st.dataframe(df.style.format({"Cu_eq": "{:.2f}", "Pyridine_eq": "{:.2f}", "Substrate_µmol": "{:.2f}", "% RCC": "{:.2f}"}), use_container_width=True)
    st.caption("Each row shows a set of conditions tested by BO and the resulting radiochemical conversion (%RCC).")

    # Plot convergence
    fig1, ax1 = plt.subplots(figsize=(8, 4))  # Adjusted size
    ax1.plot(range(1, len(result.func_vals)+1), explored_rcc, marker='o')
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("% RCC")
    ax1.set_title("Convergence Over Iterations")
    ax1.grid(True)
    st.pyplot(fig1)

    # Parallel coordinates plot of explored points
    from sklearn.preprocessing import LabelEncoder

    def show_parallel_coordinates(data: list, response_name: str):
        if len(data) == 0:
            return
        df = pd.DataFrame(data).copy()
        df[response_name] = pd.to_numeric(df[response_name], errors="coerce")
        cols_to_plot = ["Cu_eq", "Pyridine_eq", "Substrate_µmol", response_name]
        df = df[cols_to_plot]
        st.markdown("### 🔀 Parallel Coordinates Plot")
        # Encode object columns numerically and prepare label legend
        for col in df.columns:
            if df[col].dtype == object:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
        labels = {col: col for col in df.columns}
        fig = px.parallel_coordinates(
            df,
            color=response_name,
            color_continuous_scale=px.colors.sequential.Viridis[::-1],
            labels=labels
        )
        fig.update_layout(
            font=dict(size=20, color='black'),
            height=500,
            margin=dict(l=50, r=50, t=50, b=40),
            coloraxis_colorbar=dict(
                title=dict(
                    text=response_name,
                    font=dict(size=20, color='black'),
                ),
                tickfont=dict(size=20, color='black'),
                len=0.8,
                thickness=40,
                tickprefix=" ",
                xpad=5
            )
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
    st.dataframe(comparison_df.style.format({"Best RCC": "{:.2f}", "Cu": "{:.2f}", "Pyridine": "{:.2f}", "Substrate": "{:.2f}"}), use_container_width=True)

    # Download comparison data
    csv = comparison_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Comparison Data",
        data=csv,
        file_name="comparison_data.csv",
        mime="text/csv"
    )

    # Visualization of comparison
    st.markdown("### Comparison Visualization")
    fig, ax = plt.subplots(figsize=(8, 4))  # Adjusted size
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
        run_df = pd.DataFrame(run["Settings"].get("explored_points", []), columns=["Cu_eq", "Pyridine_eq", "Substrate_µmol"])
        run_df["% RCC"] = run["Settings"].get("explored_rcc", [])
        run_df["Run"] = f"Run {i+1}"
        run_df["Experiment"] = range(1, len(run_df) + 1)  # Add experiment number
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
            labels={"Experiment": "Experiment Number", "% RCC": "Radiochemical Conversion (%)"}
        )

        # Add line traces for each run with the same color as the scatter points
        for run_name, run_data in combined_df.groupby("Run"):
            fig2.add_scatter(
                x=run_data["Experiment"],
                y=run_data["% RCC"],
                mode="lines",
                line=dict(color=fig2.data[[trace.name for trace in fig2.data].index(run_name)].marker.color),
                name=f"{run_name} (Line)"
            )

        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No detailed experiment data available for comparison.")

# --- Clear Campaign ---
if st.button("🗑️ Clear Campaign", help="Reset all saved runs and start fresh."):
    st.session_state["comparison_data"] = []
    if "last_result" in st.session_state:
        del st.session_state["last_result"]
    st.success("Campaign cleared! All saved runs have been removed.")


