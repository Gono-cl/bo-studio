"""
Reusable Streamlit UI components for BO Studio.

Keep the main app code clean by moving repeated UI bits here.
"""

from __future__ import annotations

import os
import pandas as pd
import streamlit as st


# ----------------------------- Basic widgets -----------------------------
def user_info() -> None:
    st.sidebar.markdown("### User Info")
    st.sidebar.write(f"User: {st.session_state.get('user_name', 'Guest')}")


def campaign_selector(campaigns: list[str] | tuple[str, ...], key: str = "campaign_selector") -> str:
    return st.sidebar.selectbox("Select Campaign", campaigns, key=key)


def save_button(label: str = "Save Campaign", target=None) -> bool:
    target = target or st.sidebar
    return target.button(label)


def data_editor(data, key: str, editable: bool = True, **kwargs) -> pd.DataFrame:
    return st.data_editor(pd.DataFrame(data), key=key, disabled=not editable, **kwargs)


def display_dataframe(data, key: str = "dataframe") -> None:
    st.dataframe(pd.DataFrame(data), use_container_width=True, key=key)


# ----------------------------- Campaign utilities -----------------------------
def list_valid_campaigns(base_dir: str) -> list[str]:
    """
    List subfolders that look like valid campaigns (have manual_data.csv and metadata.json).
    """
    if not os.path.exists(base_dir):
        return []
    valid: list[str] = []
    for d in sorted(os.listdir(base_dir)):
        p = os.path.join(base_dir, d)
        if os.path.isdir(p) and \
           os.path.exists(os.path.join(p, "manual_data.csv")) and \
           os.path.exists(os.path.join(p, "metadata.json")):
            valid.append(d)
    return valid


def resume_campaign_selector(
    user_save_dir: str,
    key: str = "resume_campaign",
    target=None,
    show_divider: bool = True,
) -> str:
    """
    Selectbox to choose a previous campaign to resume.
    Returns the selected campaign name or "None".
    """
    target = target or st.sidebar
    if show_divider:
        target.markdown("---")
    options = ["None"] + list_valid_campaigns(user_save_dir)
    return target.selectbox("Resume from Previous Manual Campaign", options=options, key=key)

# Backcompat alias (internal callers in older code)
_list_valid_campaigns = list_valid_campaigns


def load_campaign_button(label: str = "Load Previous Manual Campaign", target=None) -> bool:
    """Sidebar button to load the selected campaign."""
    target = target or st.sidebar
    return target.button(label)


# ----------------------------- Messaging helpers -----------------------------
def show_warning(message: str) -> None:
    st.warning(message)


def show_success(message: str) -> None:
    st.success(message)


# ----------------------------- Inputs -----------------------------
def experiment_name_input(default_value: str = "") -> str:
    return st.text_input("Experiment Name", value=default_value)


# Add more reusable components as needed.
