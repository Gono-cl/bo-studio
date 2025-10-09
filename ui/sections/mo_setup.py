from __future__ import annotations

import pandas as pd
import streamlit as st
from skopt.space import Real, Categorical

from ui.sections.variables import render_variables_section
from core.utils.init_designs import generate_initial_points


def render_mo_setup_and_initials() -> None:
    # Variables reuse the same section UI
    render_variables_section()

    st.subheader("Multiobjective Setup")
    # Build available objectives: defaults + any numeric columns present in mo_data
    defaults = ["Yield", "Conversion", "Transformation", "Productivity"]
    extra = []
    if st.session_state.get("mo_data"):
        try:
            df_existing = pd.DataFrame(st.session_state.mo_data)
            extra = [c for c in df_existing.columns if c not in [n for n, *_ in st.session_state.manual_variables]]
        except Exception:
            extra = []
    available_objs = list(dict.fromkeys(defaults + extra))
    selected = st.multiselect(
        "Select Objectives",
        available_objs,
        default=["Yield", "Conversion"],
        key="mo_objectives_select",
    )
    if selected:
        st.session_state.mo_objectives = selected
    else:
        st.session_state.mo_objectives = available_objs[:2]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.number_input("# Initial Experiments", min_value=2, max_value=100, value=st.session_state.get("mo_n_init", 6), key="mo_n_init")
    with col2:
        init_options = ["Random", "LHS", "Halton", "Maximin LHS"]
        init_keys = ["random", "lhs", "halton", "maximin_lhs"]
        default_init = st.session_state.get("mo_init_method", "lhs")
        init_choice = st.selectbox("Initialization Method", init_options, index=init_keys.index(default_init) if default_init in init_keys else 1)
        st.session_state.mo_init_method = init_keys[init_options.index(init_choice)]
    with col3:
        st.number_input("Total Iterations", min_value=1, max_value=200, value=st.session_state.get("mo_total_iters", 20), key="mo_total_iters")

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
            )
            new_dirs[obj] = choice
        st.session_state.mo_directions = new_dirs

    # Custom objective creation
    with st.expander("Create Custom Objective", expanded=False):
        st.caption("Define a new objective as an expression of existing columns, e.g., 'Yield / Cost' or '0.7*Yield + 0.3*Purity'")
        new_name = st.text_input("Objective name", key="mo_custom_name")
        expr = st.text_input("Expression (pandas eval)", key="mo_custom_expr", placeholder="0.7*Yield + 0.3*Conversion")
        if st.button("Add Custom Objective"):
            if not new_name or not expr:
                st.warning("Provide both a name and an expression.")
            else:
                # Build df context from current mo_data if any, else from variables only
                source_rows = st.session_state.get("mo_data", [])
                if not source_rows:
                    st.info("No data yet; the custom objective will appear once results exist.")
                    st.session_state.setdefault("mo_custom_objectives", set()).add(new_name)
                else:
                    dfc = pd.DataFrame(source_rows)
                    try:
                        dfc[new_name] = dfc.eval(expr)
                        st.session_state.mo_data = dfc.to_dict("records")
                        st.session_state.setdefault("mo_custom_objectives", set()).add(new_name)
                        st.success(f"Custom objective '{new_name}' added.")
                    except Exception as ex:
                        st.error(f"Could not compute expression: {ex}")

    if st.button("Suggest Initial Experiments (MO)"):
        if not st.session_state.manual_variables:
            st.warning("Please define at least one variable first.")
        else:
            st.session_state.mo_suggestions = generate_initial_points(
                st.session_state.manual_variables,
                int(st.session_state.mo_n_init),
                method=st.session_state.mo_init_method,
                seed=42,
            )
            st.session_state.mo_initialized = True
            st.session_state.mo_data = []
            st.success("Initial MO experiments suggested.")

    if st.session_state.get("mo_initialized") and st.session_state.get("mo_suggestions"):
        st.markdown("### Initial MO Experiments (Enter results for each objective)")
        default_rows = []
        for vals in st.session_state.mo_suggestions:
            row = {name: val for (name, *_), val in zip(st.session_state.manual_variables, vals)}
            for obj in st.session_state.mo_objectives:
                row[obj] = None
            default_rows.append(row)
        edited_df = st.data_editor(pd.DataFrame(default_rows), key="mo_initial_editor")
        if st.button("Submit MO Initial Results"):
            df = edited_df.copy()
            for obj in st.session_state.mo_objectives:
                df[obj] = pd.to_numeric(df[obj], errors="coerce")
            if df[st.session_state.mo_objectives].isna().any().any():
                st.error("Please fill all objective values with numbers.")
            else:
                st.session_state.mo_data.extend(df.to_dict("records"))
                st.session_state.mo_iteration = len(st.session_state.mo_data)
                st.success("MO initial results recorded.")
