"""
Experiment setup and initial suggestions/results capture.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
from skopt.space import Real, Categorical

from ui.components import data_editor
from core.utils.bo_manual import safe_build_optimizer, force_model_based


def render_setup_and_initials() -> None:
    st.subheader("Experiment Setup")
    col5, col6 = st.columns(2)
    with col5:
        response = st.selectbox(
            "Response to Optimize",
            ["Yield", "Conversion", "Transformation", "Productivity"],
            index=["Yield", "Conversion", "Transformation", "Productivity"].index(st.session_state.get("response", "Yield")),
        )
        st.session_state.response = response
    with col6:
        st.number_input("# Initial Experiments", min_value=1, max_value=50, value=st.session_state.n_init, key="n_init")
        st.number_input("Total Iterations", min_value=1, max_value=100, value=st.session_state.total_iters, key="total_iters")

    if st.button("Suggest Initial Experiments"):
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
            optimizer = safe_build_optimizer(opt_vars, n_initial_points_remaining=st.session_state.n_init, acq_func="EI")
            st.session_state.manual_optimizer = optimizer
            st.session_state.manual_data = []
            st.session_state.manual_initialized = True
            st.session_state.iteration = 0
            st.session_state.initial_results_submitted = False
            st.session_state.next_suggestion_cached = None
            st.session_state.suggestions = [optimizer.suggest() for _ in range(st.session_state.n_init)]
            st.success("Initial experiments suggested successfully!")

    if not st.session_state.initial_results_submitted and st.session_state.suggestions:
        st.markdown("### Initial Experiments (User Input Required)")
        st.caption("Press Enter or click outside each cell to confirm your entry before submitting.")

        default_data = []
        for vals in st.session_state.suggestions:
            row = {name: val for (name, *_), val in zip(st.session_state.manual_variables, vals)}
            row[st.session_state.response] = None
            default_data.append(row)

        edited_df_init = data_editor(default_data, key="initial_results_editor", editable=True, num_rows="fixed")

        if st.button("Submit Initial Results"):
            if edited_df_init is not None:
                st.session_state.edited_initial_df = edited_df_init.copy()
                st.session_state.submitted_initial = True
            else:
                st.error("Please fill in the table before submitting.")

    if st.session_state.get("submitted_initial") and st.session_state.get("edited_initial_df") is not None:
        valid_rows = 0
        resp = st.session_state.response
        for _, row in st.session_state.edited_initial_df.iterrows():
            value = row.get(resp)
            if value is None or str(value).strip() == "":
                continue
            try:
                y_val = float(value)
                x = [row[name] for name, *_ in st.session_state.manual_variables]
                st.session_state.manual_optimizer.observe(x, -y_val)
                row_data = row.to_dict()
                row_data[resp] = y_val
                row_data["Timestamp"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state.manual_data.append(row_data)
                valid_rows += 1
            except ValueError:
                st.error(f"Invalid number format: {value}")
                st.stop()

        if valid_rows < len(st.session_state.edited_initial_df):
            st.warning(f"Only {valid_rows} of {len(st.session_state.edited_initial_df)} experiments had valid results.")
            st.stop()

        st.session_state.iteration += valid_rows
        st.session_state.suggestions = []
        st.session_state.initial_results_submitted = True
        st.session_state.submitted_initial = False
        force_model_based(st.session_state.manual_optimizer)

