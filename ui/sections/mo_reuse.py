from __future__ import annotations

import os
import json
import pandas as pd
import streamlit as st

from core.utils.bo_manual import unionize_bounds
from ui.components import data_editor


def render_mo_reuse_seeds(user_save_dir: str) -> None:
    """
    Reuse a previous multiobjective campaign as seeds for the current session.
    - Loads a saved MO campaign's data and lets the user select rows to include.
    - Ensures variables match; allows choosing which objectives to bring if needed.
    - Extends/aligns current variable bounds via unionize_bounds.
    - Populates st.session_state.mo_data and related state.
    """
    st.sidebar.markdown("---")
    st.sidebar.subheader("Reuse Previous MO Campaign as Seeds")

    # Collect only campaigns marked as multiobjective
    options = ["None"]
    meta_map = {}
    for d in sorted(os.listdir(user_save_dir)):
        p = os.path.join(user_save_dir, d)
        meta_p = os.path.join(p, "metadata.json")
        if os.path.isdir(p) and os.path.exists(meta_p):
            try:
                with open(meta_p, "r") as f:
                    meta = json.load(f)
                if meta.get("mode") == "multiobjective":
                    options.append(d)
                    meta_map[d] = meta
            except Exception:
                continue

    reuse_campaign = st.sidebar.selectbox("Select a Previous MO Campaign to Reuse", options=options, key="mo_reuse_campaign")
    if reuse_campaign == "None":
        return

    reuse_path = os.path.join(user_save_dir, reuse_campaign)
    try:
        prev_df_raw = pd.read_csv(os.path.join(reuse_path, "manual_data.csv"))
        prev_meta = meta_map.get(reuse_campaign, {})
    except Exception as ex:
        st.sidebar.error(f"Could not load campaign '{reuse_campaign}': {ex}")
        return

    # Variable compatibility
    curr_variables = st.session_state.get("manual_variables", [])
    prev_variables = prev_meta.get("variables", [])
    if len(prev_variables) != len(curr_variables) or not all(p[0] == c[0] for p, c in zip(prev_variables, curr_variables)):
        st.sidebar.error("Variable mismatch between previous and current campaign.")
        return

    # Determine objectives to bring
    proposed_objs = st.session_state.get("mo_objectives") or prev_meta.get("mo_objectives", [])
    if not proposed_objs:
        # Heuristic: all non-variable numeric columns
        variable_names = [n for n, *_ in curr_variables]
        numeric_cols = [c for c in prev_df_raw.columns if c not in variable_names]
        proposed_objs = numeric_cols[:2]

    # Let user choose objectives among columns present
    available_obj_cols = [c for c in prev_df_raw.columns if c not in [n for n, *_ in curr_variables]]
    chosen_objs = st.sidebar.multiselect("Objectives to import", options=available_obj_cols, default=[c for c in proposed_objs if c in available_obj_cols], key="mo_reuse_objs")
    if not chosen_objs:
        st.sidebar.warning("Select at least one objective column to import.")
        return

    # Ensure numeric
    df_view = prev_df_raw.copy()
    for c in chosen_objs:
        df_view[c] = pd.to_numeric(df_view[c], errors="coerce")

    # Prepare editor with Use column
    required_cols = [n for n, *_ in curr_variables]
    missing = [c for c in required_cols if c not in df_view.columns]
    if missing:
        st.sidebar.error(f"Missing variable columns in previous data: {missing}")
        return

    # Build editable preview in main area to allow comfortable selection
    st.markdown("### Previous MO Experiments (edit + select)")
    if "mo_prev_df_editor_cache" not in st.session_state or st.session_state.get("mo_prev_df_source_campaign") != reuse_campaign:
        df_for_editor = df_view.copy()
        df_for_editor.insert(0, "Use", True)
        st.session_state.mo_prev_df_editor_cache = df_for_editor
        st.session_state.mo_prev_df_source_campaign = reuse_campaign

    default_cols = ["Use"] + required_cols + chosen_objs
    extra_cols = [c for c in st.session_state.mo_prev_df_editor_cache.columns if c not in default_cols]
    show_cols = st.multiselect(
        "Columns to display",
        options=list(st.session_state.mo_prev_df_editor_cache.columns),
        default=default_cols + ([c for c in extra_cols if c.lower() in ["timestamp"]]),
        key="mo_reuse_cols_multiselect",
    )
    if not show_cols:
        show_cols = list(st.session_state.mo_prev_df_editor_cache.columns)

    edited_prev_df = data_editor(
        st.session_state.mo_prev_df_editor_cache[show_cols],
        key=f"mo_reuse_editor_{reuse_campaign}",
        editable=True,
        use_container_width=True,
        column_config={
            "Use": st.column_config.CheckboxColumn("Use", help="Tick rows you want to include", default=True)
        },
    )
    st.session_state.mo_prev_df_editor_cache.loc[:, show_cols] = edited_prev_df

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Select all (MO reuse)"):
            st.session_state.mo_prev_df_editor_cache["Use"] = True
            st.rerun()
    with c2:
        if st.button("Clear all (MO reuse)"):
            st.session_state.mo_prev_df_editor_cache["Use"] = False
            st.rerun()

    if st.button("Use selected experiments (MO)"):
        selected_df = st.session_state.mo_prev_df_editor_cache.copy()
        # keep only selected and with all chosen objectives present
        selected_df = selected_df[selected_df["Use"]].copy()
        selected_df = selected_df[[c for c in required_cols if c in selected_df.columns] + chosen_objs + (["Timestamp"] if "Timestamp" in selected_df.columns else [])]
        if selected_df.empty:
            st.error("Select at least one row to import.")
            return

        # Align bounds (union with current)
        model_variables = unionize_bounds(curr_variables, selected_df)
        st.session_state.manual_variables = model_variables

        # Update MO state
        st.session_state.mo_objectives = chosen_objs
        # Directions: keep existing if valid; else default to Maximize
        dirs = st.session_state.get("mo_directions", {})
        st.session_state.mo_directions = {o: (dirs.get(o, "Maximize")) for o in chosen_objs}

        st.session_state.mo_data = selected_df.to_dict("records")
        st.session_state.mo_iteration = len(st.session_state.mo_data)
        st.session_state.mo_initialized = True
        st.session_state.mo_suggestions = []
        st.success(f"Imported {len(selected_df)} experiment(s) from '{reuse_campaign}'. You can continue from here.")

