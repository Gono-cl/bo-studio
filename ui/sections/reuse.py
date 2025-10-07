"""
Sidebar flow to reuse a previous campaign as seeds (union model space, then continue BO).
"""

from __future__ import annotations

import os
import json

import pandas as pd
import streamlit as st

from ui.components import data_editor
from core.utils.bo_manual import unionize_bounds, rebuild_optimizer_from_df


def render_reuse_seeds(user_save_dir: str) -> None:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Reuse Previous Campaign as Seeds")

    _reuse_options = ["None"] + [d for d in os.listdir(user_save_dir) if os.path.isdir(os.path.join(user_save_dir, d))]
    reuse_campaign = st.sidebar.selectbox("Select a Previous Campaign to Reuse", options=_reuse_options)
    if reuse_campaign == "None":
        return

    reuse_path = os.path.join(user_save_dir, reuse_campaign)
    try:
        prev_df_raw = pd.read_csv(os.path.join(reuse_path, "manual_data.csv"))
        with open(os.path.join(reuse_path, "metadata.json"), "r") as f:
            prev_meta = json.load(f)
    except Exception as ex:
        st.error(f"Could not load previous campaign: {ex}")
        return

    prev_variables = prev_meta.get("variables", [])
    curr_variables = st.session_state.manual_variables

    if len(prev_variables) != len(curr_variables):
        st.error("Variable count mismatch between previous and current campaign.")
        return
    if not all(p[0] == c[0] for p, c in zip(prev_variables, curr_variables)):
        st.error("Variable names do not match between campaigns.")
        return

    resp = st.session_state.get("response", "Yield")
    if resp not in prev_df_raw.columns:
        candidates = ["Yield", "Conversion", "Transformation", "Productivity"]
        fallback = next((c for c in candidates if c in prev_df_raw.columns), None)
        if fallback is None:
            st.error("No valid response column found in previous data.")
            return
        resp = fallback
        st.session_state.response = resp
        st.info(f"Using '{resp}' as response column from previous data.")

    required_cols = [name for name, *_ in curr_variables]
    missing = [c for c in required_cols if c not in prev_df_raw.columns]
    if missing:
        st.error(f"Missing variable columns in previous data: {missing}")
        return

    if "prev_df_editor_cache" not in st.session_state or st.session_state.get("prev_df_source_campaign") != reuse_campaign:
        df_for_editor = prev_df_raw.copy()
        df_for_editor.insert(0, "Use", True)
        st.session_state.prev_df_editor_cache = df_for_editor
        st.session_state.prev_df_source_campaign = reuse_campaign

    st.markdown("### Previous Experiments (edit + select)")
    default_cols = ["Use"] + required_cols + [resp]
    extra_cols = [c for c in st.session_state.prev_df_editor_cache.columns if c not in default_cols]
    show_cols = st.multiselect(
        "Columns to display",
        options=list(st.session_state.prev_df_editor_cache.columns),
        default=default_cols + ([c for c in extra_cols if c.lower() in ["timestamp"]]),
        key="reuse_cols_multiselect",
    )
    if not show_cols:
        show_cols = list(st.session_state.prev_df_editor_cache.columns)

    edited_prev_df = data_editor(
        st.session_state.prev_df_editor_cache[show_cols],
        key=f"reuse_editor_{reuse_campaign}",
        editable=True,
        use_container_width=True,
        column_config={
            "Use": st.column_config.CheckboxColumn("Use", help="Tick rows you want to include", default=True)
        },
    )
    st.session_state.prev_df_editor_cache.loc[:, show_cols] = edited_prev_df

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        if st.button("Select all"):
            st.session_state.prev_df_editor_cache["Use"] = True
            st.rerun()
    with c2:
        if st.button("Clear all"):
            st.session_state.prev_df_editor_cache["Use"] = False
            st.rerun()
    with c3:
        skip_random = st.checkbox(
            "Skip additional random initial points (start BO suggestions immediately)",
            value=True,
            key="reuse_skip_random",
        )

    if st.button("Use selected experiments"):
        selected_df = st.session_state.prev_df_editor_cache.copy()
        selected_df[resp] = pd.to_numeric(selected_df[resp], errors="coerce")
        selected_df = selected_df[selected_df["Use"] & selected_df[resp].notna()]

        keep_cols = required_cols + [resp]
        extra_keep = [c for c in ["Timestamp"] if c in selected_df.columns]
        selected_df = selected_df[keep_cols + extra_keep].copy()

        if selected_df.empty:
            st.error("Select at least one valid row (with numeric response).")
            return

        model_variables = unionize_bounds(st.session_state.manual_variables, selected_df)
        seed_count = len(selected_df)
        remaining_init = 0 if skip_random else max(0, int(st.session_state.n_init) - seed_count)

        optimizer = rebuild_optimizer_from_df(
            model_variables,
            selected_df,
            resp,
            n_initial_points_remaining=remaining_init,
            acq_func="EI",
        )

        st.session_state.model_variables = model_variables
        st.session_state.manual_optimizer = optimizer
        st.session_state.manual_initialized = True
        st.session_state.manual_data = selected_df.to_dict("records")
        st.session_state.iteration = seed_count
        st.session_state.initial_results_submitted = True
        st.session_state.submitted_initial = False
        st.session_state.suggestions = []
        st.session_state.next_suggestion_cached = None

        msg = f"Reused {seed_count} experiment(s) from '{reuse_campaign}'. "
        msg += "Starting BO now." if remaining_init == 0 else f"{remaining_init} initial random(s) remain."
        st.success(msg)

