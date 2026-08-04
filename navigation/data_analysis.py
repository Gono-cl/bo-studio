import os
import streamlit as st
import pandas as pd
from core.utils import db_handler
from core.utils.analysis_utils import infer_db_analysis_context, infer_objective_columns
from core.utils.app_paths import get_campaigns_dir

from ui.sections.analysis_overview import render_analysis_overview
from ui.sections.analysis_explain import render_analysis_explain
from ui.sections.analysis_mo import render_analysis_mo
from ui.sections.analysis_cards import render_analysis_card


st.title("Data Analysis")

render_analysis_card(
    "How to use this page",
    [
        "1) Choose the data source.",
        "2) Review Overview for data quality and objective behavior.",
        "3) Check Model Fit before interpreting feature effects.",
        "4) For multiobjective campaigns, inspect Pareto diagnostics.",
    ],
    tone="green",
)


def _get_current_data():
    so = st.session_state.get("manual_data", [])
    mo = st.session_state.get("mo_data", [])
    return so, mo


def _render_so_analysis(
    df: pd.DataFrame,
    variables: list,
    response: str | None,
    direction: str,
) -> None:
    with st.container(border=True):
        render_analysis_overview(df, response=response, direction=direction, variables=variables)
    with st.container(border=True):
        render_analysis_explain(df, target=response, variables=variables)


def _render_mo_analysis(
    df: pd.DataFrame,
    variables: list,
    objectives: list[str],
    directions: dict[str, str],
    selector_key: str,
) -> None:
    available_objectives = [obj for obj in objectives if obj in df.columns]
    if len(available_objectives) < 2:
        inferred = infer_objective_columns(df, variables)
        available_objectives = [obj for obj in inferred if obj in df.columns]

    if not available_objectives:
        st.info("No objective columns could be identified for multiobjective analysis.")
        return

    sel = st.multiselect(
        "Objectives for analysis",
        available_objectives,
        default=available_objectives[:2],
        key=selector_key,
    )
    if len(sel) < 2:
        st.warning("Select at least two objectives for Pareto analysis.")
    with st.container(border=True):
        render_analysis_overview(
            df,
            response=sel[0] if sel else None,
            direction=directions.get(sel[0], "Maximize") if sel else "Maximize",
            extra_objectives=sel,
            variables=variables,
        )
    with st.container(border=True):
        render_analysis_mo(df, objectives=sel, directions={o: directions.get(o, "Maximize") for o in sel})


so_data, mo_data = _get_current_data()

source = st.radio("Data source", ["Current Session", "Saved Campaign", "Database"], index=0)

loaded_df = None
loaded_mode = None  # 'so' or 'mo'
loaded_variables = st.session_state.get("manual_variables", [])
loaded_response = st.session_state.get("response")
loaded_direction = st.session_state.get("response_direction", "Maximize")
loaded_mo_objectives = st.session_state.get("mo_objectives", [])
loaded_mo_directions = st.session_state.get("mo_directions", {})

if source == "Saved Campaign":
    user_email = st.session_state.get("user_email", "default_user")
    save_dir = str(get_campaigns_dir())
    user_save_dir = os.path.join(save_dir, user_email)
    options = ["None"]
    metas = {}
    if os.path.isdir(user_save_dir):
        for d in sorted(os.listdir(user_save_dir)):
            p = os.path.join(user_save_dir, d)
            meta_p = os.path.join(p, "metadata.json")
            data_p = os.path.join(p, "manual_data.csv")
            if os.path.isdir(p) and os.path.exists(meta_p) and os.path.exists(data_p):
                options.append(d)
                metas[d] = (meta_p, data_p)
    sel = st.selectbox("Select campaign", options)
    if sel != "None":
        meta_p, data_p = metas[sel]
        import json

        with open(meta_p, "r") as f:
            loaded_meta = json.load(f)
        loaded_df = pd.read_csv(data_p)
        loaded_variables = loaded_meta.get("variables", [])
        if loaded_meta.get("mode") == "multiobjective":
            loaded_mode = "mo"
            loaded_mo_objectives = loaded_meta.get("mo_objectives", [])
            loaded_mo_directions = loaded_meta.get("mo_directions", {})
        else:
            loaded_mode = "so"
            loaded_response = loaded_meta.get("response", loaded_response)
            loaded_direction = loaded_meta.get("response_direction", loaded_direction)

elif source == "Database":
    user_email = st.session_state.get("user_email", "default_user")
    rows = db_handler.list_experiments(user_email)
    if not rows:
        st.info("No experiments found in database for this user.")
    else:
        options = {f"{rid} - {name} ({ts})": rid for rid, name, ts in rows}
        label = st.selectbox("Select experiment", list(options.keys()))
        exp = db_handler.load_experiment(options[label])
        if exp:
            context = infer_db_analysis_context(
                exp,
                fallback_response=st.session_state.get("response"),
                fallback_direction=st.session_state.get("response_direction", "Maximize"),
            )
            loaded_df = context["df"]
            loaded_mode = context["mode"]
            loaded_variables = context.get("variables", [])
            loaded_response = context.get("response")
            loaded_direction = context.get("response_direction", "Maximize")
            loaded_mo_objectives = context.get("mo_objectives", [])
            loaded_mo_directions = context.get("mo_directions", {})

if loaded_df is not None:
    if loaded_mode == "so":
        _render_so_analysis(
            loaded_df,
            variables=loaded_variables,
            response=loaded_response,
            direction=loaded_direction,
        )
    elif loaded_mode == "mo":
        _render_mo_analysis(
            loaded_df,
            variables=loaded_variables,
            objectives=loaded_mo_objectives,
            directions=loaded_mo_directions,
            selector_key="analysis_mo_select_loaded",
        )
else:
    mode = st.radio("Select dataset", ["Single Objective", "Multiobjective"], index=0 if len(so_data) else 1 if len(mo_data) else 0)
    if mode == "Single Objective":
        df = pd.DataFrame(so_data)
        if df.empty:
            st.info("No single-objective data available in this session. Run, load a campaign, or choose 'Database'.")
            st.stop()
        _render_so_analysis(
            df,
            variables=st.session_state.get("manual_variables", []),
            response=st.session_state.get("response", None),
            direction=st.session_state.get("response_direction", "Maximize"),
        )
    else:
        df = pd.DataFrame(mo_data)
        if df.empty:
            st.info("No multiobjective data available in this session. Run or load a campaign first.")
            st.stop()
        _render_mo_analysis(
            df,
            variables=st.session_state.get("manual_variables", []),
            objectives=st.session_state.get("mo_objectives", []),
            directions=st.session_state.get("mo_directions", {}),
            selector_key="analysis_mo_select_session",
        )
