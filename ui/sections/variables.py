"""
Variable definition and editing section.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ui.components import data_editor
from core.utils.manual_campaign import campaign_has_started


def render_variables_section() -> None:
    with st.container(border=True):
        st.markdown("### Define Variables")
        _render_variables_section_content()


def _render_variables_section_content() -> None:
    variables_locked = campaign_has_started(
        manual_initialized=bool(st.session_state.get("manual_initialized")),
        manual_data=st.session_state.get("manual_data", []),
        suggestions=st.session_state.get("suggestions", []),
        iteration=int(st.session_state.get("iteration", 0)),
        next_suggestion_cached=st.session_state.get("next_suggestion_cached"),
    )

    if variables_locked:
        st.info("Variables are locked after the campaign starts. Reset the campaign to change the search space.")

    st.session_state.var_type = st.selectbox(
        "Variable Type",
        ["Continuous", "Categorical"],
        key="var_type_select",
        disabled=variables_locked,
    )

    with st.form("manual_var_form"):
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            var_name = st.text_input("Variable Name", disabled=variables_locked)
        with col2:
            if st.session_state.var_type == "Continuous":
                lower = st.number_input("Lower Bound", value=0.0, format="%.4f", disabled=variables_locked)
                upper = st.number_input("Upper Bound", value=1.0, format="%.4f", disabled=variables_locked)
            else:
                categories = st.text_input("Categories (comma-separated)", value="Type", disabled=variables_locked)
        with col3:
            unit = st.text_input("Unit", disabled=variables_locked)

        add_var = st.form_submit_button("Add Variable", disabled=variables_locked)
        if add_var and var_name:
            if st.session_state.var_type == "Continuous" and lower < upper:
                st.session_state.manual_variables.append((var_name, lower, upper, unit, "continuous"))
            elif st.session_state.var_type == "Categorical" and categories:
                values = [x.strip() for x in categories.split(",") if x.strip()]
                st.session_state.manual_variables.append((var_name, values, None, unit, "categorical"))
            st.session_state["refresh_edit_variables_editor"] = True

    if not st.session_state.manual_variables:
        return

    st.markdown("### Edit Variables")
    if st.session_state.pop("refresh_edit_variables_editor", False):
        st.session_state.pop("edit_variables_cont_editor", None)
        st.session_state.pop("edit_variables_cat_editor", None)

    continuous_rows = []
    categorical_rows = []
    for name, val1, val2, unit, vtype in st.session_state.manual_variables:
        if str(vtype).lower() == "continuous":
            continuous_rows.append(
                {
                    "Name": name,
                    "Lower Bound": val1,
                    "Upper Bound": val2,
                    "Unit": unit,
                }
            )
        else:
            categorical_rows.append(
                {
                    "Name": name,
                    "Categories": ", ".join(map(str, val1)) if isinstance(val1, list) else str(val1),
                    "Unit": unit,
                }
            )

    with st.form("edit_variables_form"):
        edited_cont_df = None
        edited_cat_df = None
        if continuous_rows:
            st.markdown("**Continuous variables**")
            edited_cont_df = data_editor(
                pd.DataFrame(continuous_rows),
                key="edit_variables_cont_editor",
                editable=not variables_locked,
                hide_index=True,
                num_rows="fixed",
            )
        if categorical_rows:
            st.markdown("**Categorical variables**")
            edited_cat_df = data_editor(
                pd.DataFrame(categorical_rows),
                key="edit_variables_cat_editor",
                editable=not variables_locked,
                hide_index=True,
                num_rows="fixed",
            )
        st.markdown("**Delete variable**")
        delete_var_in_form = st.selectbox(
            "Select a Variable to Delete",
            options=["None"] + [v[0] for v in st.session_state.manual_variables],
            key="delete_var_in_edit_form",
            disabled=variables_locked,
        )
        action_col1, action_col2 = st.columns(2)
        with action_col1:
            save_variable_changes = st.form_submit_button("Save Variable Changes", disabled=variables_locked)
        with action_col2:
            delete_variable_changes = st.form_submit_button("Delete Variable", disabled=variables_locked)

    if delete_variable_changes:
        if delete_var_in_form != "None":
            st.session_state.manual_variables = [v for v in st.session_state.manual_variables if v[0] != delete_var_in_form]
            if st.session_state.get("model_variables") is None:
                st.session_state.model_variables = st.session_state.manual_variables
            st.session_state["refresh_edit_variables_editor"] = True
            st.success(f"Variable '{delete_var_in_form}' deleted successfully!")
        else:
            st.info("Select a variable to delete.")
    elif save_variable_changes:
        updated_variables = []
        if edited_cont_df is not None:
            for _, row in edited_cont_df.iterrows():
                try:
                    low = float(row["Lower Bound"])
                    high = float(row["Upper Bound"])
                except (TypeError, ValueError):
                    continue
                if low < high:
                    updated_variables.append((str(row["Name"]), low, high, row["Unit"], "continuous"))
        if edited_cat_df is not None:
            for _, row in edited_cat_df.iterrows():
                cats = row.get("Categories", "")
                if isinstance(cats, list):
                    values = [str(x).strip() for x in cats if str(x).strip()]
                else:
                    values = [x.strip() for x in str(cats).split(",") if x.strip()]
                if values:
                    updated_variables.append((str(row["Name"]), values, None, row["Unit"], "categorical"))
        st.session_state.manual_variables = updated_variables
        if st.session_state.get("model_variables") is None:
            st.session_state.model_variables = st.session_state.manual_variables
        st.session_state["refresh_edit_variables_editor"] = True
        st.success("Variables updated successfully!")

