import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.preprocessing import LabelEncoder

from core.utils import db_handler
from core.utils.analysis_utils import (
    ANALYSIS_ORDER_COLUMN,
    infer_db_analysis_context,
    infer_objective_columns,
    prepare_multiobjective_frame,
    prepare_objective_progress_frame,
    variable_names,
)
from core.utils.pareto import pareto_front_indices


def _display_setting(label: str, value) -> None:
    if isinstance(value, list):
        value = ", ".join(str(item) for item in value) if value else "N/A"
    elif isinstance(value, dict):
        value = ", ".join(f"{key}: {val}" for key, val in value.items()) if value else "N/A"
    elif value in (None, ""):
        value = "N/A"
    st.write(f"**{label}:** {value}")


def _get_objective_options(context: dict, df_results: pd.DataFrame, variables: list) -> list[str]:
    if context.get("mode") == "mo":
        options = [obj for obj in context.get("mo_objectives", []) if obj in df_results.columns]
        if len(options) >= 2:
            return options
    response = context.get("response")
    if response and response in df_results.columns:
        return [response]
    return infer_objective_columns(df_results, variables)


def _render_metadata(exp_id: int, exp_data: dict, context: dict, df_results: pd.DataFrame) -> None:
    st.subheader("Metadata")
    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        c1.metric("Record ID", exp_id)
        c2.metric("Mode", "Multiobjective" if context.get("mode") == "mo" else "Single Objective")
        c3.metric("Rows", len(df_results))
        st.write(f"**Name:** {exp_data.get('name') or 'Untitled experiment'}")
        st.write(f"**Timestamp:** {exp_data.get('timestamp', 'N/A')}")
        st.write(f"**Notes:** {exp_data.get('notes') or 'N/A'}")


def _render_settings(settings: dict, context: dict) -> None:
    st.subheader("Optimization Settings")
    with st.container(border=True):
        _display_setting(
            "Initial Experiments",
            settings.get("initial_experiments", settings.get("mo_n_init")),
        )
        _display_setting(
            "Total Iterations",
            settings.get("total_iterations", settings.get("mo_total_iters")),
        )
        if context.get("mode") == "mo":
            objectives = context.get("mo_objectives", [])
            directions = context.get("mo_directions", {})
            _display_setting("Objectives", objectives)
            _display_setting("Objective Directions", directions)
        else:
            _display_setting("Objective", context.get("response"))
            _display_setting(
                "Objective Direction",
                settings.get("response_direction", context.get("response_direction")),
            )
        _display_setting("Method", settings.get("method"))
        _display_setting("Acquisition", settings.get("acq_func"))
        _display_setting("Init Method", settings.get("init_method"))
        _display_setting("Seed", settings.get("bo_seed"))


def _render_results_table(df_results: pd.DataFrame) -> None:
    st.subheader("Results")
    df_view = df_results.copy()
    for col in df_view.columns:
        if df_view[col].dtype == "object":
            df_view[col] = df_view[col].astype(str)
    st.dataframe(df_view, use_container_width=True, hide_index=True)


def _render_best_result(exp_data: dict, context: dict, df_results: pd.DataFrame) -> None:
    st.subheader("Best Result")
    best_result = exp_data.get("best_result")
    if isinstance(best_result, list) and best_result and isinstance(best_result[0], dict):
        pareto_df = pd.DataFrame(best_result)
        st.dataframe(pareto_df, use_container_width=True, hide_index=True)
    elif isinstance(best_result, dict):
        st.dataframe(pd.DataFrame([best_result]), use_container_width=True, hide_index=True)
    elif best_result:
        st.write(best_result)
    else:
        st.write("No best-result summary saved for this record.")

    objectives = context.get("mo_objectives", []) if context.get("mode") == "mo" else []
    if len(objectives) < 2:
        return

    df_plot, valid_objectives = prepare_multiobjective_frame(df_results, objectives)
    if df_plot.empty or len(valid_objectives) < 2:
        st.info("Not enough complete numeric objective rows to render a Pareto view.")
        return

    st.markdown("### Pareto Front View")
    if len(valid_objectives) > 2:
        st.caption("Projection shown on the selected objective pair. Pareto membership is computed using all saved objectives.")
        c1, c2 = st.columns(2)
        with c1:
            x_obj = st.selectbox("Pareto x-axis objective", valid_objectives, key="db_pareto_x")
        y_candidates = [obj for obj in valid_objectives if obj != x_obj] or valid_objectives
        with c2:
            y_obj = st.selectbox("Pareto y-axis objective", y_candidates, key="db_pareto_y")
    else:
        x_obj, y_obj = valid_objectives[:2]

    directions = context.get("mo_directions", {})
    signs = np.array(
        [1.0 if directions.get(obj, "Maximize") == "Maximize" else -1.0 for obj in valid_objectives],
        dtype=float,
    )
    idx_pf = pareto_front_indices(df_plot[valid_objectives].to_numpy(dtype=float) * signs)
    is_pareto = np.zeros(len(df_plot), dtype=bool)
    if len(idx_pf) > 0:
        is_pareto[np.asarray(idx_pf, dtype=int)] = True
    df_plot = df_plot.copy()
    df_plot["Pareto"] = np.where(is_pareto, "Pareto", "Dominated")
    st.caption(f"Pareto front size: {len(idx_pf)}")

    fig = px.scatter(
        df_plot,
        x=x_obj,
        y=y_obj,
        color="Pareto",
        color_discrete_map={"Pareto": "#dc2626", "Dominated": "#94a3b8"},
        opacity=0.9,
    )
    if len(idx_pf) > 0:
        df_pf = df_plot.iloc[idx_pf].sort_values(by=x_obj)
        fig.add_trace(
            go.Scatter(
                x=df_pf[x_obj],
                y=df_pf[y_obj],
                mode="lines+markers",
                line=dict(color="#dc2626", width=3),
                marker=dict(size=8),
                name="Pareto front",
            )
        )
    fig.update_layout(height=460, legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)


def _render_progress_chart(df_results: pd.DataFrame, objective_options: list[str], context: dict) -> None:
    if not objective_options:
        st.info("No objective columns were identified for progress visualization.")
        return

    st.subheader("Objective Progress")
    default_objective = objective_options[0]
    selected_objective = (
        st.selectbox("Objective to view", objective_options, key="db_progress_objective")
        if len(objective_options) > 1
        else default_objective
    )

    progress_df = prepare_objective_progress_frame(df_results, selected_objective)
    if progress_df.empty:
        st.info("No numeric values are available for the selected objective.")
        return

    fig = px.line(progress_df, x=ANALYSIS_ORDER_COLUMN, y=selected_objective, markers=True)
    fig.update_layout(xaxis_title="Experiment", yaxis_title=selected_objective, height=360)
    st.plotly_chart(fig, use_container_width=True)

    if context.get("mode") == "mo":
        direction = context.get("mo_directions", {}).get(selected_objective, "Maximize")
    else:
        direction = context.get("response_direction", "Maximize")
    best_value = (
        progress_df[selected_objective].min() if direction == "Minimize" else progress_df[selected_objective].max()
    )
    label = "Current lowest" if direction == "Minimize" else "Current best"
    st.caption(f"{label} {selected_objective}: {best_value:.4g}")


def _render_parallel_coordinates(
    df_results: pd.DataFrame,
    variables: list,
    objective_options: list[str],
) -> None:
    st.subheader("Parallel Coordinates Plot")
    variable_cols = variable_names(variables)
    cols_to_plot: list[str] = []
    for col in variable_cols + objective_options:
        if col in df_results.columns and col not in cols_to_plot:
            cols_to_plot.append(col)

    if len(cols_to_plot) < 2 or not objective_options:
        st.info("Not enough saved variable/objective columns are available for a parallel-coordinates view.")
        return

    color_obj = (
        st.selectbox("Color lines by objective", objective_options, key="db_parallel_color")
        if len(objective_options) > 1
        else objective_options[0]
    )

    df_plot = df_results[cols_to_plot].copy()
    for obj in objective_options:
        if obj in df_plot.columns:
            df_plot[obj] = pd.to_numeric(df_plot[obj], errors="coerce")
    df_plot = df_plot.dropna(subset=[color_obj]).copy()
    if df_plot.empty:
        st.info("No numeric values are available for the selected color objective.")
        return

    categorical_cols: list[str] = []
    legend_entries: list[tuple[str, dict[int, str]]] = []
    var_meta = {}
    for item in variables:
        if isinstance(item, (list, tuple)) and len(item) >= 5:
            var_meta[str(item[0])] = item[4]
    for col in cols_to_plot:
        if col in objective_options:
            continue
        if var_meta.get(col) == "categorical" or df_plot[col].dtype == object:
            le = LabelEncoder()
            try:
                df_plot[col] = le.fit_transform(df_plot[col].astype(str))
                categorical_cols.append(col)
                legend_entries.append((col, dict(enumerate(le.classes_))))
            except Exception:
                continue

    fig = px.parallel_coordinates(
        df_plot,
        dimensions=cols_to_plot,
        color=color_obj,
        color_continuous_scale=px.colors.sequential.Viridis_r,
        labels={col: col for col in cols_to_plot},
    )
    try:
        for dim in fig.data[0].dimensions:
            if getattr(dim, "label", None) not in categorical_cols:
                dim.tickformat = ".2f"
    except Exception:
        pass
    fig.update_layout(
        font=dict(size=18, color="black"),
        height=520,
        margin=dict(l=70, r=50, t=40, b=40),
        coloraxis_colorbar=dict(
            title=dict(text=color_obj, font=dict(size=18, color="black")),
            tickfont=dict(size=16, color="black"),
            len=0.8,
            thickness=32,
        ),
    )
    st.plotly_chart(fig, use_container_width=True)

    if legend_entries:
        st.markdown("### Categorical Legends")
        for col, mapping in legend_entries:
            st.markdown(f"**{col}**")
            for code, label in mapping.items():
                st.markdown(f"- `{code}` -> `{label}`")


st.title("Experiment Database")
st.markdown("### Experiment History")

user_email = st.session_state.get("user_email", "default_user")
experiments = db_handler.list_experiments(user_email)

if not experiments:
    st.info("No experiments saved yet.")
else:
    with st.expander("Show/Hide Experiment List and Delete", expanded=False):
        st.markdown("### Select experiments to delete")
        exp_df = pd.DataFrame(experiments, columns=["ID", "Name", "Timestamp"])
        exp_df["Delete?"] = False

        selected = st.data_editor(
            exp_df,
            column_config={"Delete?": st.column_config.CheckboxColumn("Delete?")},
            disabled=["ID", "Name", "Timestamp"],
            num_rows="fixed",
            use_container_width=True,
            key="exp_editor",
        )

        to_delete = selected[selected["Delete?"] == True]["ID"].tolist()
        delete_btn = st.button("Delete Selected Experiments", disabled=not to_delete)
        if delete_btn and to_delete:
            db_handler.delete_experiments(to_delete)
            st.success(f"Deleted {len(to_delete)} experiment(s).")
            st.rerun()

    selected_id = st.selectbox(
        "Select an experiment",
        experiments,
        format_func=lambda x: f"{x[1]} ({x[2]})",
    )
    if selected_id:
        exp_data = db_handler.load_experiment(selected_id[0])
        if not exp_data:
            st.error("Could not load the selected experiment.")
            st.stop()

        df_results = exp_data.get("df_results", pd.DataFrame()).copy()
        variables = exp_data.get("variables", []) or []
        context = infer_db_analysis_context(exp_data)
        settings = exp_data.get("settings", {}) or {}
        objective_options = _get_objective_options(context, df_results, variables)

        _render_metadata(selected_id[0], exp_data, context, df_results)
        _render_settings(settings, context)
        _render_results_table(df_results)
        _render_best_result(exp_data, context, df_results)
        _render_progress_chart(df_results, objective_options, context)
        _render_parallel_coordinates(df_results, variables, objective_options)
