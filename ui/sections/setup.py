"""
Experiment setup and initial suggestions/results capture.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
from skopt.space import Real, Categorical

from ui.components import data_editor
from core.utils.bo_manual import safe_build_optimizer, force_model_based
from core.utils.manual_campaign import campaign_has_started, validate_initial_results
from core.utils.init_designs import generate_initial_points
from core.utils.n_init_guidance import recommend_n_init_range, format_n_init_range
from ui.charts import Charts


def render_setup_and_initials() -> None:
    settings_locked = campaign_has_started(
        manual_initialized=bool(st.session_state.get("manual_initialized")),
        manual_data=st.session_state.get("manual_data", []),
        suggestions=st.session_state.get("suggestions", []),
        iteration=int(st.session_state.get("iteration", 0)),
        next_suggestion_cached=st.session_state.get("next_suggestion_cached"),
    )

    with st.container(border=True):
        st.markdown("### Experiment Setup")
        if settings_locked:
            st.info(
                "Campaign-defining settings are locked after initialization. "
                "Reset the campaign to change variables, objective, BO policy, or seed. "
                "You may still change total iterations."
            )
        col5, col6 = st.columns(2)
        with col5:
            # Build dynamic list of available responses: defaults + any existing numeric columns in data
            base_responses = ["Yield", "Conversion", "Transformation", "Productivity", "Byproduct", "Cost", "Time", "E-factor", "Space-Time Yield", "Selectivity", "Purity", "Mass Yield", "Atom Economy", "Carbon Efficiency", "Energy Efficiency", "Process Mass Intensity"]
            existing_cols = []
            if st.session_state.get("manual_data"):
                try:
                    df_tmp = pd.DataFrame(st.session_state.manual_data)
                    variable_names = [name for name, *_ in st.session_state.manual_variables]
                    existing_cols = [
                        c
                        for c in df_tmp.columns
                        if c not in variable_names
                        and pd.to_numeric(df_tmp[c], errors="coerce").notna().any()
                    ]
                except Exception:
                    existing_cols = []
            # Include any previously defined custom objectives
            custom_objs = st.session_state.get("custom_objectives", [])
            if isinstance(custom_objs, dict):
                custom_objs = list(custom_objs.keys())
            elif isinstance(custom_objs, set):
                custom_objs = sorted(custom_objs)
            elif not isinstance(custom_objs, list):
                custom_objs = []
            st.session_state.custom_objectives = custom_objs

            options = list(dict.fromkeys(base_responses + custom_objs + existing_cols))
            default_resp = st.session_state.get("response", "Yield")
            if default_resp not in options:
                options.insert(0, default_resp)
            response = st.selectbox(
                "Response to Optimize",
                options,
                index=options.index(default_resp),
                disabled=settings_locked,
            )
            st.session_state.response = response
            st.caption("If the objective is not in the list, create your own objective below.")
            col_obj1, col_obj2 = st.columns([3, 1])
            with col_obj1:
                new_obj = st.text_input(
                    "New objective name",
                    key="so_custom_obj_name",
                    disabled=settings_locked,
                )
            with col_obj2:
                st.markdown("&nbsp;", unsafe_allow_html=True)
                add_obj = st.button("Add Objective", key="so_add_custom_obj", disabled=settings_locked)
            if add_obj:
                obj_name = (new_obj or "").strip()
                if not obj_name:
                    st.warning("Please enter an objective name.")
                elif obj_name in options:
                    st.info(f"'{obj_name}' is already available.")
                else:
                    st.session_state.custom_objectives = custom_objs + [obj_name]
                    st.session_state.response = obj_name
                    st.success(f"Custom objective '{obj_name}' added.")
                    st.rerun()
        with col6:
            st.number_input(
                "# Initial Experiments",
                min_value=1,
                max_value=50,
                key="n_init",
                help="Number of initial design points collected before BO suggestions dominate.",
                disabled=settings_locked,
            )
            st.number_input("Total Iterations", min_value=1, max_value=100, key="total_iters")

        n_vars = max(1, len(st.session_state.get("manual_variables", [])))
        mixed = any((len(v) >= 5 and str(v[4]).lower() == "categorical") for v in st.session_state.get("manual_variables", []))
        rec_low, rec_high, rec_text = recommend_n_init_range(
            n_vars,
            total_budget=int(st.session_state.total_iters),
            mixed=mixed,
            multiobjective=False,
        )
        rec_range_text = format_n_init_range(rec_low, rec_high)
        st.caption(f"n_init guidance: {rec_text}")
        if int(st.session_state.n_init) < rec_low or int(st.session_state.n_init) > rec_high:
            st.info(
                f"Current n_init={int(st.session_state.n_init)} is outside the suggested range ({rec_range_text}). "
                "You can still use it, but this may affect BO efficiency."
            )

        col7, col8, col9 = st.columns(3)
        with col7:
            init_options = ["Random", "LHS", "Halton", "Maximin LHS"]
            init_keys = ["random", "lhs", "halton", "maximin_lhs"]
            default_init = st.session_state.get("init_method", "random").lower().replace(" ", "_")
            try:
                init_index = init_keys.index(default_init)
            except ValueError:
                init_index = 0
            init_choice = st.selectbox(
                "Initialization Method",
                init_options,
                index=init_index,
                disabled=settings_locked,
            )
            st.session_state.init_method = init_keys[init_options.index(init_choice)]
        with col8:
            acq_options = ["EI", "PI", "LCB"]
            default_acq = st.session_state.get("acq_func", "EI")
            st.session_state.acq_func = st.selectbox(
                "Acquisition Function",
                acq_options,
                index=acq_options.index(default_acq),
                disabled=settings_locked,
            )
        with col9:
            st.number_input(
                "Optimizer Seed",
                min_value=0,
                max_value=9999,
                key="bo_seed",
                help="Controls reproducibility of initialization design and BO suggestion sequence.",
                disabled=settings_locked,
            )

        col10, col11 = st.columns(2)
        with col10:
            st.number_input(
                "Exploration xi (EI/PI)",
                min_value=0.0,
                max_value=0.5,
                step=0.01,
                key="acq_xi",
                help="Higher xi increases exploration pressure for EI/PI.",
                disabled=settings_locked,
            )
        with col11:
            st.number_input(
                "Exploration kappa (LCB)",
                min_value=0.1,
                max_value=10.0,
                step=0.1,
                key="acq_kappa",
                help="Higher kappa increases exploration pressure for LCB.",
                disabled=settings_locked,
            )

        # Direction of optimization (single objective)
        st.session_state.response_direction = st.selectbox(
            "Direction",
            ["Maximize", "Minimize"],
            index=["Maximize", "Minimize"].index(st.session_state.get("response_direction", "Maximize")),
            disabled=settings_locked,
        )

        if st.button("Suggest Initial Experiments", disabled=settings_locked):
            if st.session_state.manual_initialized and st.session_state.manual_data:
                st.info("Already initialized (possibly via reuse). Use Get Next Suggestion to continue.")
            elif not st.session_state.manual_variables:
                st.warning("Please define at least one variable first.")
            else:
                st.session_state.model_variables = st.session_state.manual_variables
                opt_vars = []
                for name, val1, val2, _, vtype in st.session_state.model_variables:
                    if vtype == "continuous":
                        opt_vars.append(Real(val1, val2, name=name))
                    else:
                        opt_vars.append(Categorical(val1, name=name))
                optimizer = safe_build_optimizer(
                    opt_vars,
                    n_initial_points_remaining=0,
                    acq_func=st.session_state.acq_func,
                    acq_xi=float(st.session_state.get("acq_xi", 0.01)),
                    acq_kappa=float(st.session_state.get("acq_kappa", 1.96)),
                    random_state=int(st.session_state.get("bo_seed", 42)),
                )
                st.session_state.manual_optimizer = optimizer
                st.session_state.manual_data = []
                st.session_state.manual_initialized = True
                st.session_state.iteration = 0
                st.session_state.initial_results_submitted = False
                st.session_state.next_suggestion_cached = None
                st.session_state.suggestions = generate_initial_points(
                    st.session_state.model_variables,
                    st.session_state.n_init,
                    method=st.session_state.get("init_method", "random"),
                    seed=int(st.session_state.get("bo_seed", 42)),
                )
                st.success("Initial experiments suggested successfully!")

    if not st.session_state.initial_results_submitted and st.session_state.suggestions:
        st.markdown("### Initial Experiments (User Input Required)")
        st.caption("Press Enter or click outside each cell to confirm your entry before submitting.")

        default_data = []
        for idx, vals in enumerate(st.session_state.suggestions, start=1):
            row = {"Experiment": idx}
            row.update({name: val for (name, *_), val in zip(st.session_state.manual_variables, vals)})
            row[st.session_state.response] = None
            default_data.append(row)

        with st.expander("Preview Initial Design", expanded=False):
            try:
                Charts.show_initial_design(st.session_state.suggestions, st.session_state.manual_variables)
            except Exception:
                pass

        edited_df_init = data_editor(
            default_data,
            key="initial_results_editor",
            editable=True,
            num_rows="fixed",
            hide_index=True,
            disabled=["Experiment"],
        )

        if st.button("Submit Initial Results"):
            if edited_df_init is not None:
                try:
                    validated_rows = validate_initial_results(
                        edited_df_init.copy(),
                        st.session_state.manual_variables,
                        st.session_state.response,
                    )
                except ValueError as ex:
                    st.error(str(ex))
                else:
                    resp = st.session_state.response
                    for x, y_val, row_data in validated_rows:
                        observed = -y_val if st.session_state.get("response_direction", "Maximize") == "Maximize" else y_val
                        st.session_state.manual_optimizer.observe(x, observed)
                        row_data["Timestamp"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                        st.session_state.manual_data.append(row_data)

                    st.session_state.iteration += len(validated_rows)
                    st.session_state.suggestions = []
                    st.session_state.initial_results_submitted = True
                    st.session_state.submitted_initial = False
                    st.session_state.edited_initial_df = None
                    force_model_based(st.session_state.manual_optimizer)
                    st.success("Initial results submitted successfully.")
            else:
                st.error("Please fill in the table before submitting.")
