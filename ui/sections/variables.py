"""
Variable definition and editing section.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ui.components import data_editor


def render_variables_section() -> None:
    st.subheader("Define and Edit Variables")
    st.session_state.var_type = st.selectbox("Variable Type", ["Continuous", "Categorical"], key="var_type_select")

    with st.form("manual_var_form"):
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            var_name = st.text_input("Variable Name")
        with col2:
            if st.session_state.var_type == "Continuous":
                lower = st.number_input("Lower Bound", value=0.0, format="%.4f")
                upper = st.number_input("Upper Bound", value=1.0, format="%.4f")
            else:
                categories = st.text_input("Categories (comma-separated)", value="Type")
        with col3:
            unit = st.text_input("Unit")

        add_var = st.form_submit_button("Add Variable")
        if add_var and var_name:
            if st.session_state.var_type == "Continuous" and lower < upper:
                st.session_state.manual_variables.append((var_name, lower, upper, unit, "continuous"))
            elif st.session_state.var_type == "Categorical" and categories:
                values = [x.strip() for x in categories.split(",") if x.strip()]
                st.session_state.manual_variables.append((var_name, values, None, unit, "categorical"))

    if not st.session_state.manual_variables:
        return

    st.markdown("### Edit Variables")
    variables_df = pd.DataFrame(
        [
            {
                "Name": name,
                "Type": vtype,
                "Value 1": val1,
                "Value 2": val2 if vtype == "continuous" else None,
                "Unit": unit,
            }
            for name, val1, val2, unit, vtype in st.session_state.manual_variables
        ]
    )
    edited_df = data_editor(variables_df, key="edit_variables_editor")
    if st.button("Save Variable Changes"):
        updated_variables = []
        for _, row in edited_df.iterrows():
            if row["Type"] == "continuous" and pd.notnull(row["Value 1"]) and pd.notnull(row["Value 2"]) and row["Value 1"] < row["Value 2"]:
                updated_variables.append((row["Name"], float(row["Value 1"]), float(row["Value 2"]), row["Unit"], "continuous"))
            elif row["Type"] == "categorical":
                v1 = row["Value 1"]
                if isinstance(v1, list):
                    values = v1
                elif isinstance(v1, str):
                    values = [x.strip() for x in v1.split(",") if x.strip()]
                else:
                    values = []
                if values:
                    updated_variables.append((row["Name"], values, None, row["Unit"], "categorical"))
        st.session_state.manual_variables = updated_variables
        if st.session_state.get("model_variables") is None:
            st.session_state.model_variables = st.session_state.manual_variables
        st.success("Variables updated successfully!")

    delete_var = st.selectbox("Select a Variable to Delete", options=["None"] + [v[0] for v in st.session_state.manual_variables])
    if delete_var != "None" and st.button("Delete Variable"):
        st.session_state.manual_variables = [v for v in st.session_state.manual_variables if v[0] != delete_var]
        if st.session_state.get("model_variables") is None:
            st.session_state.model_variables = st.session_state.manual_variables
        st.success(f"Variable '{delete_var}' deleted successfully!")

