from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
from skopt.space import Real, Categorical
from sklearn.preprocessing import LabelEncoder

from ui.components import data_editor
from core.utils import db_handler
from core.utils.pareto import pareto_front_indices
from core.utils.scalarization import sample_dirichlet_weights, weighted_sum, tchebycheff
from core.utils.bo_manual import safe_build_optimizer, coerce_point_to_variables, next_unique_suggestion


def _build_scalarized_optimizer(weights: np.ndarray, method: str = "weighted_sum"):
    # Build optimizer on current model variables
    mv = st.session_state.get("manual_variables", [])
    dims = []
    for name, v1, v2, _u, t in mv:
        if t == "continuous":
            dims.append(Real(v1, v2, name=name))
        else:
            dims.append(Categorical(v1, name=name))
    opt = safe_build_optimizer(
        dims,
        n_initial_points_remaining=0,
        acq_func=st.session_state.get("acq_func", "EI"),
        acq_xi=float(st.session_state.get("acq_xi", 0.01)),
        acq_kappa=float(st.session_state.get("acq_kappa", 1.96)),
        random_state=int(st.session_state.get("bo_seed", 42)),
    )
    # Observe existing data using scalarized objective
    data = st.session_state.get("mo_data", [])
    objs = st.session_state.get("mo_objectives", [])
    if data and objs:
        df = pd.DataFrame(data)
        if not set(objs).issubset(df.columns):
            return opt
        for obj in objs:
            df[obj] = pd.to_numeric(df[obj], errors="coerce")
        df = df.dropna(subset=objs).copy()
        if df.empty:
            return opt
        # Apply direction flips before scalarization
        dir_map = st.session_state.get("mo_directions", {})
        signs = np.array([1.0 if dir_map.get(o, "Maximize") == "Maximize" else -1.0 for o in objs], dtype=float)
        # ideal point for tchebycheff on transformed space
        z = (df[objs] * signs).max().values
        for _, row in df.iterrows():
            raw_x = [row.get(name) for name, *_ in mv]
            x = coerce_point_to_variables(raw_x, mv)
            if x is None:
                continue
            y_vec = (row[objs].values.astype(float) * signs)
            if method == "tchebycheff":
                s = tchebycheff(y_vec, weights, z)
            else:
                s = weighted_sum(y_vec, weights)
            opt.observe(x, float(-s))  # maximize scalarization
    return opt


def _compute_pareto_front_records(
    df: pd.DataFrame,
    objectives: list[str],
    directions: dict[str, str],
) -> list[dict]:
    valid_objs = [obj for obj in objectives if obj in df.columns]
    if len(valid_objs) < 2:
        return []

    df_pf = df.copy()
    for obj in valid_objs:
        df_pf[obj] = pd.to_numeric(df_pf[obj], errors="coerce")
    df_pf = df_pf.dropna(subset=valid_objs).copy()
    if df_pf.empty:
        return []

    signs = np.array(
        [1.0 if directions.get(obj, "Maximize") == "Maximize" else -1.0 for obj in valid_objs],
        dtype=float,
    )
    idx_pf = pareto_front_indices(df_pf[valid_objs].to_numpy(dtype=float) * signs)
    return df_pf.iloc[idx_pf].to_dict("records")


def _render_mo_completion(df: pd.DataFrame) -> None:
    total_iters = int(st.session_state.get("mo_total_iters", 0) or 0)
    current_iter = len(st.session_state.get("mo_data", []))
    if total_iters <= 0 or current_iter < total_iters:
        return

    with st.container(border=True):
        st.markdown("### Multiobjective Optimization Completed")
        st.caption("Review final results, download CSV if needed, and save the campaign to the database.")
        st.success("All configured MO iterations are completed.")
        st.dataframe(df, use_container_width=True, hide_index=True)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download MO Results as CSV",
            data=csv,
            file_name="multiobjective_optimization_results.csv",
            mime="text/csv",
        )

        if st.button("Save to Database (MO)", key="mo_save_to_database"):
            directions = st.session_state.get("mo_directions", {})
            objectives = st.session_state.get("mo_objectives", [])
            pareto_front = _compute_pareto_front_records(df, objectives, directions)
            optimization_settings = {
                "mode": "multiobjective",
                "initial_experiments": int(st.session_state.get("mo_n_init", 0)),
                "total_iterations": int(st.session_state.get("mo_total_iters", 0)),
                "objectives": list(objectives),
                "objective_directions": dict(directions),
                "acq_func": st.session_state.get("acq_func", "EI"),
                "acq_xi": float(st.session_state.get("acq_xi", 0.01)),
                "acq_kappa": float(st.session_state.get("acq_kappa", 1.96)),
                "init_method": st.session_state.get("mo_init_method", "lhs"),
                "bo_seed": int(st.session_state.get("bo_seed", 42)),
                "method": "Manual Multiobjective Bayesian Optimization",
            }
            db_handler.save_experiment(
                user_email=st.session_state.get("user_email", "default_user"),
                name=st.session_state.get("experiment_name", ""),
                notes=st.session_state.get("experiment_notes", ""),
                variables=st.session_state.get("manual_variables", []),
                df_results=df,
                best_result=pareto_front,
                settings=optimization_settings,
            )
            st.success("MO experiment saved to database.")


def _show_mo_progress_chart(df: pd.DataFrame, objectives: list[str]) -> None:
    valid_objs = [o for o in objectives if o in df.columns]
    if not valid_objs:
        return

    default_obj = st.session_state.get("mo_progress_objective", valid_objs[0])
    if default_obj not in valid_objs:
        default_obj = valid_objs[0]

    with st.container(border=True):
        st.markdown("### Optimization Progress")
        progress_obj = st.selectbox(
            "Objective for progress chart",
            options=valid_objs,
            index=valid_objs.index(default_obj),
            key="mo_progress_objective",
        )

        df_plot = df.copy()
        df_plot[progress_obj] = pd.to_numeric(df_plot[progress_obj], errors="coerce")
        df_plot = df_plot.dropna(subset=[progress_obj]).copy()
        if df_plot.empty:
            st.info("No numeric values available yet for the selected objective.")
            return

        df_plot["Iteration"] = range(1, len(df_plot) + 1)
        chart = alt.Chart(df_plot).mark_line(
            point=alt.OverlayMarkDef(size=110, filled=True)
        ).encode(
            x=alt.X(
                "Iteration:Q",
                title="Experiment Number",
                axis=alt.Axis(
                    format="d",
                    tickMinStep=1,
                    labelColor="black",
                    titleColor="black",
                    labelFontSize=14,
                    titleFontSize=16,
                ),
            ),
            y=alt.Y(
                f"{progress_obj}:Q",
                title=progress_obj,
                axis=alt.Axis(
                    labelColor="black",
                    titleColor="black",
                    labelFontSize=14,
                    titleFontSize=16,
                ),
            ),
            tooltip=["Iteration", progress_obj],
        ).properties(height=400)
        st.altair_chart(chart, use_container_width=True)

        direction = st.session_state.get("mo_directions", {}).get(progress_obj, "Maximize")
        if direction == "Minimize":
            best_val = df_plot[progress_obj].min()
            st.markdown(f"**Current lowest {progress_obj}:** {best_val:.4g}")
        else:
            best_val = df_plot[progress_obj].max()
            st.markdown(f"**Current best {progress_obj}:** {best_val:.4g}")


def _show_mo_parallel_coordinates(df: pd.DataFrame, objectives: list[str]) -> None:
    valid_objs = [o for o in objectives if o in df.columns]
    if not valid_objs:
        return

    input_vars = [name for name, *_ in st.session_state.get("manual_variables", [])]
    cols_to_plot = [c for c in (input_vars + valid_objs) if c in df.columns]
    if len(cols_to_plot) < 2:
        return

    default_color = st.session_state.get("mo_parallel_color_objective", valid_objs[0])
    if default_color not in valid_objs:
        default_color = valid_objs[0]

    with st.container(border=True):
        st.markdown("### Parallel Coordinates Plot")
        color_obj = st.selectbox(
            "Color lines by objective",
            options=valid_objs,
            index=valid_objs.index(default_color),
            key="mo_parallel_color_objective",
        )

        df_plot = df[cols_to_plot].copy()
        for o in valid_objs:
            df_plot[o] = pd.to_numeric(df_plot[o], errors="coerce")
        df_plot = df_plot.dropna(subset=[color_obj]).copy()
        if df_plot.empty:
            st.info("No numeric values available yet for the selected color objective.")
            return

        # Encode categorical variables for plotly parallel coordinates and keep a legend.
        categorical_cols = []
        legend_entries: list[tuple[str, dict[int, str]]] = []
        var_meta = {name: vtype for name, *_u, vtype in st.session_state.get("manual_variables", [])}
        for col in cols_to_plot:
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
            labels={c: c for c in cols_to_plot},
        )
        try:
            for dim in fig.data[0].dimensions:
                if getattr(dim, "label", None) not in categorical_cols:
                    dim.tickformat = ".2f"
        except Exception:
            pass
        fig.update_layout(
            font=dict(size=20, color="black"),
            height=500,
            margin=dict(l=70, r=50, t=50, b=40),
            coloraxis_colorbar=dict(
                title=dict(text=color_obj, font=dict(size=20, color="black")),
                tickfont=dict(size=20, color="black"),
                len=0.8,
                thickness=40,
                tickprefix=" ",
                xpad=5,
            ),
        )
        st.plotly_chart(fig, use_container_width=True)

        if legend_entries:
            st.markdown("#### Categorical Legends")
            for col, mapping in legend_entries:
                st.markdown(f"**{col}**")
                for code, label in mapping.items():
                    st.markdown(f"- `{code}` -> `{label}`")


def _show_mo_pareto_plot(df: pd.DataFrame, objectives: list[str]) -> None:
    valid_objs = [o for o in objectives if o in df.columns]
    if len(valid_objs) < 2:
        return

    with st.container(border=True):
        st.markdown("### Pareto View")
        if len(valid_objs) == 2:
            x_obj, y_obj = valid_objs[0], valid_objs[1]
        else:
            c1, c2 = st.columns(2)
            with c1:
                x_obj = st.selectbox("Pareto x-axis objective", valid_objs, key="mo_pareto_x")
            y_candidates = [o for o in valid_objs if o != x_obj] or valid_objs
            default_y = st.session_state.get("mo_pareto_y", y_candidates[0])
            if default_y not in y_candidates:
                default_y = y_candidates[0]
            with c2:
                y_obj = st.selectbox(
                    "Pareto y-axis objective",
                    y_candidates,
                    index=y_candidates.index(default_y),
                    key="mo_pareto_y",
                )

        df_plot = df.copy()
        for o in valid_objs:
            df_plot[o] = pd.to_numeric(df_plot[o], errors="coerce")
        df_plot = df_plot.dropna(subset=valid_objs).copy()
        if df_plot.empty:
            st.info("No complete numeric objective rows available yet for Pareto analysis.")
            return

        dir_map = st.session_state.get("mo_directions", {})
        signs = np.array([1.0 if dir_map.get(o, "Maximize") == "Maximize" else -1.0 for o in valid_objs], dtype=float)
        pts = df_plot[valid_objs].to_numpy(dtype=float) * signs
        idx_pf = pareto_front_indices(pts)
        st.caption(f"Pareto front size: {len(idx_pf)}")

        is_pareto = np.zeros(len(df_plot), dtype=bool)
        if len(idx_pf) > 0:
            is_pareto[np.asarray(idx_pf, dtype=int)] = True
        df_plot = df_plot.copy()
        df_plot["Pareto"] = np.where(is_pareto, "Pareto", "Dominated")

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


def render_mo_interact_and_pareto(user_save_dir: str):
    # Show MO charts and controls once there is data.
    data = st.session_state.get("mo_data", [])
    objs = st.session_state.get("mo_objectives", [])
    current_iter = len(data)
    total_iters = int(st.session_state.get("mo_total_iters", 0) or 0)
    if data and objs:
        df = pd.DataFrame(data)
        _show_mo_progress_chart(df, objs)
        _show_mo_parallel_coordinates(df, objs)
        _show_mo_pareto_plot(df, objs)

        # ---- Edit results (MO) ----
        with st.container(border=True):
            st.markdown("### Edit Previous Results (MO)")
            st.caption("Use this only to fix recorded values.")
            if st.button("Enable Edit Mode (MO)"):
                st.session_state.edit_mode_mo = True
            if st.session_state.get("edit_mode_mo"):
                edited_df = data_editor(df, key="mo_edit_results_editor")
                if st.button("Save Edits (MO)"):
                    try:
                        st.session_state.mo_data = edited_df.to_dict("records") if hasattr(edited_df, "to_dict") else list(edited_df)
                    except Exception:
                        st.session_state.mo_data = edited_df
                    st.session_state.edit_mode_mo = False
                    st.success("Edits saved.")
                    st.rerun()

        # ---- Truncate to previous experiment (MO) ----
        with st.container(border=True):
            st.markdown("### Return to a Previous Experiment (MO)")
            st.caption("Keep rows up to the selected experiment and discard the rest.")
            max_idx = len(st.session_state.mo_data)
            trunc_idx = st.number_input(
                "Keep experiments up to (inclusive):",
                min_value=1,
                max_value=max_idx,
                value=max_idx,
                step=1,
                key="mo_trunc_idx",
            )
            if st.button("Return and Restart From Here (MO)"):
                st.session_state.mo_data = st.session_state.mo_data[:trunc_idx]
                st.session_state.mo_iteration = trunc_idx
                st.session_state.mo_suggestions = []
                st.session_state.mo_pending_df = []
                st.success("Truncated and ready to continue (MO).")
                st.rerun()

    # Suggest next points via scalarization
    if st.session_state.get("manual_variables") and st.session_state.get("mo_objectives"):
        with st.container(border=True):
            st.markdown("### Suggest Next MO Experiments")
            method_key = "weighted_sum"
            if len(st.session_state.get("mo_objectives", [])) < 2:
                st.info("Select at least two objectives to continue the MO campaign.")
                return
            budget_reached = total_iters > 0 and current_iter >= total_iters
            if budget_reached:
                st.session_state.mo_pending_df = []
                st.success(
                    f"MO campaign reached its configured budget of {total_iters} experiment(s). "
                    "Increase total iterations or reset the campaign to continue."
                )
            else:
                st.caption(f"Remaining MO budget: {max(0, total_iters - current_iter)} experiment(s).")

            if not budget_reached:
                # Build or render one pending suggestion persistently in session to survive reruns while editing
                if st.button("Get Next MO Suggestion"):
                    m = len(st.session_state.mo_objectives)
                    seed = int(st.session_state.get("bo_seed", 42)) + int(len(st.session_state.get("mo_data", [])))
                    w = sample_dirichlet_weights(m, 1, seed=seed)[0]
                    opt = _build_scalarized_optimizer(w, method=method_key)
                    x = next_unique_suggestion(
                        opt,
                        st.session_state.manual_variables,
                        st.session_state.get("mo_data", []),
                        max_tries=120,
                    )
                    cols = [name for name, *_ in st.session_state.manual_variables]
                    df_sug = pd.DataFrame([dict(zip(cols, x))])
                    df_sug.insert(0, "Experiment", len(st.session_state.get("mo_data", [])) + 1)
                    df_res = df_sug.copy()
                    for obj in st.session_state.mo_objectives:
                        df_res[obj] = None
                    st.session_state.mo_pending_df = df_res.to_dict("records")
                    st.session_state.pop("mo_single_result_editor", None)

                # Render pending editor if exists
                pending = st.session_state.get("mo_pending_df")
                if pending:
                    df_pending = pd.DataFrame(pending)
                    st.markdown("#### Next Proposed Experiment")
                    variable_cols = [name for name, *_ in st.session_state.manual_variables if name in df_pending.columns]
                    disabled_cols = variable_cols + ["Experiment"]
                    with st.form("mo_single_result_form", clear_on_submit=False):
                        edited = data_editor(
                            df_pending,
                            key="mo_single_result_editor",
                            hide_index=True,
                            num_rows="fixed",
                            disabled=disabled_cols,
                        )
                        submit_mo_result = st.form_submit_button("Submit MO Result")

                    if submit_mo_result:
                        df2 = edited.copy()
                        for obj in st.session_state.mo_objectives:
                            series = df2[obj]
                            if getattr(series, "dtype", None) == object:
                                series = series.astype(str).str.replace(",", ".", regex=False)
                            df2[obj] = pd.to_numeric(series, errors="coerce")

                        missing_objs = [obj for obj in st.session_state.mo_objectives if df2[obj].isna().any()]
                        if missing_objs:
                            st.error(
                                "Please fill all objective values with numbers. "
                                f"Missing/invalid: {', '.join(missing_objs)}."
                            )
                        else:
                            if "Experiment" in df2.columns:
                                df2 = df2.drop(columns=["Experiment"])
                            st.session_state.mo_data.extend(df2.to_dict("records"))
                            st.session_state.mo_pending_df = []
                            st.session_state.mo_iteration = len(st.session_state.mo_data)
                            st.success("MO result submitted. Press 'Get Next MO Suggestion' for the next point.")

    if st.session_state.get("mo_data"):
        _render_mo_completion(pd.DataFrame(st.session_state.get("mo_data", [])))
