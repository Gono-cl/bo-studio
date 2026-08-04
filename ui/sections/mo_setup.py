from __future__ import annotations
import pandas as pd
import streamlit as st
from ui.sections.variables import render_variables_section
from ui.components import data_editor
from core.utils.init_designs import generate_initial_points
from core.utils.n_init_guidance import recommend_n_init_range, format_n_init_range
from core.utils.manual_campaign import multiobjective_campaign_has_started
from ui.charts import Charts


def render_mo_setup_and_initials() -> None:
    # Variables reuse the same section UI
    render_variables_section(mode="mo")

    settings_locked = multiobjective_campaign_has_started(
        mo_initialized=bool(st.session_state.get("mo_initialized")),
        mo_data=st.session_state.get("mo_data", []),
        mo_suggestions=st.session_state.get("mo_suggestions", []),
        mo_iteration=int(st.session_state.get("mo_iteration", 0)),
        mo_pending_df=st.session_state.get("mo_pending_df", []),
    )

    with st.container(border=True):
        st.markdown("### Experiment Setup")
        if settings_locked:
            st.info(
                "MO campaign-defining settings are locked after initialization. "
                "Reset the campaign to change variables, objectives, BO policy, or seed. "
                "You may still change total iterations."
            )
        # Build available objectives: defaults + any numeric columns present in mo_data
        defaults = ["Yield", "Conversion", "Transformation", "Productivity", "Byproduct", "Cost", "Time", "E-factor", "Space-Time Yield", "Selectivity", "Purity", "Mass Yield", "Atom Economy", "Carbon Efficiency", "Energy Efficiency", "Process Mass Intensity"]
        extra = []
        if st.session_state.get("mo_data"):
            try:
                df_existing = pd.DataFrame(st.session_state.mo_data)
                variable_names = [n for n, *_ in st.session_state.manual_variables]
                extra = [
                    c
                    for c in df_existing.columns
                    if c not in variable_names
                    and pd.to_numeric(df_existing[c], errors="coerce").notna().any()
                ]
            except Exception:
                extra = []
        custom_names = st.session_state.get("mo_custom_objectives", [])
        if isinstance(custom_names, set):
            custom_names = sorted(custom_names)
        elif isinstance(custom_names, dict):
            custom_names = list(custom_names.keys())
        elif not isinstance(custom_names, list):
            custom_names = []
        st.session_state.mo_custom_objectives = custom_names

        available_objs = list(dict.fromkeys(defaults + custom_names + extra))
        prior_selected = st.session_state.get("mo_objectives", ["Yield", "Conversion"])
        valid_prior = [obj for obj in prior_selected if obj in available_objs]
        if not valid_prior:
            valid_prior = available_objs[:2]

        if "mo_objectives_select" not in st.session_state:
            st.session_state.mo_objectives_select = valid_prior
        else:
            current_widget = [obj for obj in st.session_state.mo_objectives_select if obj in available_objs]
            if not current_widget:
                st.session_state.mo_objectives_select = valid_prior
            else:
                st.session_state.mo_objectives_select = current_widget

        selected = st.multiselect(
            "Select Objectives",
            available_objs,
            key="mo_objectives_select",
            disabled=settings_locked,
        )
        if selected:
            st.session_state.mo_objectives = selected
        else:
            st.session_state.mo_objectives = available_objs[:2]
        st.caption("If an objective is not in the list, create your own objective below.")
        col_obj1, col_obj2 = st.columns([3, 1])
        with col_obj1:
            new_obj = st.text_input("New objective name", key="mo_custom_obj_name", disabled=settings_locked)
        with col_obj2:
            st.markdown("&nbsp;", unsafe_allow_html=True)
            add_obj = st.button("Add Objective", key="mo_add_custom_obj", disabled=settings_locked)
        if add_obj:
            obj_name = (new_obj or "").strip()
            if not obj_name:
                st.warning("Please enter an objective name.")
            elif obj_name in available_objs:
                st.info(f"'{obj_name}' is already available.")
            else:
                st.session_state.mo_custom_objectives = custom_names + [obj_name]
                current = st.session_state.get("mo_objectives", [])
                updated = list(dict.fromkeys(current + [obj_name]))
                st.session_state.mo_objectives = updated
                st.session_state.mo_objectives_select = updated
                st.success(f"Custom objective '{obj_name}' added.")
                st.rerun()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.number_input(
                "# Initial Experiments",
                min_value=2,
                max_value=100,
                value=st.session_state.get("mo_n_init", 6),
                key="mo_n_init",
                help="Number of initial design points before scalarization-based BO suggestions.",
                disabled=settings_locked,
            )
        with col2:
            init_options = ["Random", "LHS", "Halton", "Maximin LHS"]
            init_keys = ["random", "lhs", "halton", "maximin_lhs"]
            default_init = st.session_state.get("mo_init_method", "lhs")
            init_choice = st.selectbox(
                "Initialization Method",
                init_options,
                index=init_keys.index(default_init) if default_init in init_keys else 1,
                disabled=settings_locked,
            )
            st.session_state.mo_init_method = init_keys[init_options.index(init_choice)]
        with col3:
            st.number_input("Total Iterations", min_value=1, max_value=200, value=st.session_state.get("mo_total_iters", 20), key="mo_total_iters")

        col4, col5, col6 = st.columns(3)
        with col4:
            acq_options = ["EI", "PI", "LCB"]
            default_acq = st.session_state.get("acq_func", "EI")
            st.session_state.acq_func = st.selectbox(
                "Acquisition Function",
                acq_options,
                index=acq_options.index(default_acq) if default_acq in acq_options else 0,
                key="mo_acq_func_select",
                disabled=settings_locked,
            )
        with col5:
            st.number_input(
                "Optimizer Seed",
                min_value=0,
                max_value=9999,
                value=int(st.session_state.get("bo_seed", 42)),
                key="bo_seed",
                help="Controls reproducibility of initialization design and BO suggestion sequence.",
                disabled=settings_locked,
            )
        with col6:
            st.number_input(
                "Exploration xi (EI/PI)",
                min_value=0.0,
                max_value=0.5,
                value=float(st.session_state.get("acq_xi", 0.01)),
                step=0.01,
                key="acq_xi",
                help="Higher xi increases exploration pressure for EI/PI.",
                disabled=settings_locked,
            )
        st.number_input(
            "Exploration kappa (LCB)",
            min_value=0.1,
            max_value=10.0,
            value=float(st.session_state.get("acq_kappa", 1.96)),
            step=0.1,
            key="acq_kappa",
            help="Higher kappa increases exploration pressure for LCB.",
            disabled=settings_locked,
        )

        n_vars = max(1, len(st.session_state.get("manual_variables", [])))
        mixed = any((len(v) >= 5 and str(v[4]).lower() == "categorical") for v in st.session_state.get("manual_variables", []))
        rec_low, rec_high, rec_text = recommend_n_init_range(
            n_vars,
            total_budget=int(st.session_state.get("mo_total_iters", 20)),
            mixed=mixed,
            multiobjective=True,
        )
        rec_range_text = format_n_init_range(rec_low, rec_high)
        st.caption(f"n_init guidance: {rec_text}")
        if int(st.session_state.get("mo_n_init", 0)) < rec_low or int(st.session_state.get("mo_n_init", 0)) > rec_high:
            st.info(
                f"Current n_init={int(st.session_state.get('mo_n_init', 0))} is outside the suggested range ({rec_range_text}). "
                "This can be valid, but often changes convergence quality."
            )

        # Direction selection per objective (explicit selectboxes for compatibility)
        if st.session_state.get("mo_objectives"):
            st.markdown("#### Objective Directions")
            current = st.session_state.get("mo_directions", {})
            new_dirs = {}
            for obj in st.session_state.mo_objectives:
                curr = current.get(obj, "Maximize")
                choice = st.selectbox(
                    f"{obj} direction",
                    ["Maximize", "Minimize"],
                    index=["Maximize", "Minimize"].index(curr) if curr in ["Maximize", "Minimize"] else 0,
                    key=f"mo_dir_{obj}",
                    disabled=settings_locked,
                )
                new_dirs[obj] = choice
            st.session_state.mo_directions = new_dirs

        if st.button("Suggest Initial Experiments (MO)", disabled=settings_locked):
            if not st.session_state.manual_variables:
                st.warning("Please define at least one variable first.")
            elif len(st.session_state.get("mo_objectives", [])) < 2:
                st.warning("Select at least two objectives before starting a multiobjective campaign.")
            else:
                st.session_state.mo_suggestions = generate_initial_points(
                    st.session_state.manual_variables,
                    int(st.session_state.mo_n_init),
                    method=st.session_state.mo_init_method,
                    seed=int(st.session_state.get("bo_seed", 42)),
                )
                st.session_state.mo_initialized = True
                st.session_state.mo_data = []
                st.success("Initial MO experiments suggested.")

    if st.session_state.get("mo_initialized") and st.session_state.get("mo_suggestions"):
        st.markdown("### Initial MO Experiments (Enter results for each objective)")
        with st.expander("Preview Initial Design", expanded=False):
            try:
                Charts.show_initial_design(st.session_state.mo_suggestions, st.session_state.manual_variables)
            except Exception:
                pass
        default_rows = []
        for idx, vals in enumerate(st.session_state.mo_suggestions, start=1):
            row = {"Experiment": idx}
            row.update({name: val for (name, *_), val in zip(st.session_state.manual_variables, vals)})
            for obj in st.session_state.mo_objectives:
                row[obj] = None
            default_rows.append(row)

        edited_df = data_editor(
            pd.DataFrame(default_rows),
            key="mo_initial_editor",
            hide_index=True,
            disabled=["Experiment"],
        )
        if st.button("Submit MO Initial Results"):
            df = edited_df.copy()
            for obj in st.session_state.mo_objectives:
                df[obj] = pd.to_numeric(df[obj], errors="coerce")
            if df[st.session_state.mo_objectives].isna().any().any():
                st.error("Please fill all objective values with numbers.")
            else:
                if "Experiment" in df.columns:
                    df = df.drop(columns=["Experiment"])
                st.session_state.mo_data.extend(df.to_dict("records"))
                st.session_state.mo_iteration = len(st.session_state.mo_data)
                st.session_state.mo_suggestions = []
                st.success("MO initial results recorded.")
