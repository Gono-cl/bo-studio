"""
Charts, editing previous results, truncation, next suggestion, and completed state.
"""

from __future__ import annotations

import os
import json
from datetime import datetime

import pandas as pd
import streamlit as st

from ui.components import data_editor, display_dataframe
from ui.charts import Charts
from core.utils.bo_manual import next_unique_suggestion, rebuild_optimizer_from_df
from core.utils import db_handler


def render_interact_and_complete(user_save_dir: str, experiment_name: str, experiment_notes: str, run_name: str) -> None:
    # Recalculate optimizer if flagged by edits/truncation
    if st.session_state.get("recalc_needed"):
        df = pd.DataFrame(st.session_state.manual_data)
        resp = st.session_state.get("response", "Yield")
        model_vars = st.session_state.get("model_variables") or st.session_state.get("manual_variables", [])
        if df.empty or resp not in df.columns or not model_vars:
            st.warning("No valid data or variables to rebuild optimizer yet.")
        else:
            try:
                optimizer = rebuild_optimizer_from_df(
                    model_vars,
                    df,
                    resp,
                    n_initial_points_remaining=0,
                    acq_func=st.session_state.get("acq_func", "EI"),
                )
                st.session_state.manual_optimizer = optimizer
                st.session_state.iteration = len(df)
                st.session_state.initial_results_submitted = True
                st.session_state.next_suggestion_cached = None
                st.session_state.suggestions = []
                st.session_state.recalc_needed = False
                st.info("Optimizer recalculated from current results.")
            except Exception as ex:
                st.error(f"Could not recalculate optimizer: {ex}")
    if len(st.session_state.manual_data) > 0:
        Charts.show_progress_chart(st.session_state.manual_data, st.session_state.response)
        Charts.show_parallel_coordinates(st.session_state.manual_data, st.session_state.response)

    if len(st.session_state.manual_data) > 0:
        st.markdown("### Edit Previous Results")
        if st.button("Enable Edit Mode"):
            st.session_state.edit_mode = True
        if st.session_state.edit_mode:
            edited_df = data_editor(st.session_state.manual_data, key="edit_results_editor")
            if st.button("Save Edits"):
                st.session_state.manual_data = edited_df.to_dict("records") if hasattr(edited_df, "to_dict") else list(edited_df)
                st.session_state.edit_mode = False
                st.session_state.recalc_needed = True
                st.success("Edits saved! The optimizer will be recalculated.")
                st.rerun()

    if len(st.session_state.manual_data) > 0:
        st.markdown("### Return to a Previous Experiment")
        max_idx = len(st.session_state.manual_data)
        trunc_idx = st.number_input("Keep experiments up to (inclusive):", min_value=1, max_value=max_idx, value=max_idx, step=1)
        if st.button("Return and Restart From Here"):
            st.session_state.manual_data = st.session_state.manual_data[:trunc_idx]
            st.session_state.iteration = trunc_idx
            st.session_state.initial_results_submitted = True
            st.session_state.next_suggestion_cached = None
            st.session_state.suggestions = []
            st.session_state.recalc_needed = True
            st.success("Truncated and ready to continue.")
            st.rerun()

    if (
        st.session_state.manual_initialized
        and st.session_state.manual_optimizer is not None
        and st.session_state.iteration < st.session_state.total_iters
        and st.session_state.initial_results_submitted
    ):
        if st.button("Get Next Suggestion"):
            st.session_state.next_suggestion_cached = next_unique_suggestion(
                st.session_state.manual_optimizer,
                st.session_state.manual_variables,
                st.session_state.manual_data,
                max_tries=120,
            )

    if st.session_state.get("next_suggestion_cached") is not None:
        st.markdown("### Next Experiment Suggestion")
        next_row = {name: val for (name, *_), val in zip(st.session_state.manual_variables, st.session_state.next_suggestion_cached)}
        display_dataframe(pd.DataFrame([next_row]), key="next_row_df")
        result = st.number_input(
            f"Result for {st.session_state.response} (Experiment {st.session_state.iteration + 1})",
            key=f"next_result_{st.session_state.iteration}",
        )
        if st.button("Submit Result"):
            st.success("Result submitted. Press 'Get Next Suggestion' for the next point. Charts update automatically.")
            if pd.notnull(result):
                x = [next_row[name] for name, *_ in st.session_state.manual_variables]
                y_val = float(result)
                st.session_state.manual_optimizer.observe(x, -y_val)
                row_data = {**next_row}
                row_data[st.session_state.response] = y_val
                row_data["Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state.manual_data.append(row_data)
                st.session_state.iteration += 1
                st.session_state.next_suggestion_cached = None

    if st.session_state.iteration >= st.session_state.total_iters and st.session_state.total_iters > 0:
        st.markdown("### Optimization Completed")
        st.success("All iterations are completed! You can export the data or review the results.")
        df_results = pd.DataFrame(st.session_state.manual_data)
        display_dataframe(df_results, key="results_df")
        csv = df_results.to_csv(index=False).encode("utf-8")
        st.download_button("Download Results as CSV", data=csv, file_name="manual_optimization_results.csv", mime="text/csv")

        if st.button("Save to Database"):
            if st.session_state.response in df_results.columns and df_results[st.session_state.response].notna().any():
                best_row = df_results.loc[df_results[st.session_state.response].idxmax()].to_dict()
            else:
                best_row = {}
            optimization_settings = {
                "initial_experiments": st.session_state.n_init,
                "total_iterations": st.session_state.total_iters,
                "objective": st.session_state.response,
                "method": "Manual Bayesian Optimization",
            }
            db_handler.save_experiment(
                user_email=st.session_state.get("user_email", "default_user"),
                name=experiment_name,
                notes=experiment_notes,
                variables=st.session_state.manual_variables,
                df_results=df_results,
                best_result=best_row,
                settings=optimization_settings,
            )

            run_path = os.path.join(user_save_dir, run_name)
            os.makedirs(run_path, exist_ok=True)
            df_results.to_csv(os.path.join(run_path, "manual_data.csv"), index=False)
            metadata = {
                "variables": st.session_state.manual_variables,
                "model_variables": st.session_state.get("model_variables", st.session_state.manual_variables),
                "iteration": st.session_state.get("iteration", len(df_results)),
                "n_init": st.session_state.n_init,
                "total_iters": st.session_state.total_iters,
                "response": st.session_state.response,
                "experiment_name": experiment_name,
                "experiment_notes": experiment_notes,
                "initialization_complete": st.session_state.get("initial_results_submitted", False),
            }
            with open(os.path.join(run_path, "metadata.json"), "w") as f:
                json.dump(metadata, f, indent=4)
            st.success("Experiment saved successfully! All campaign files have been generated.")
