from __future__ import annotations

import html
import streamlit as st


CARD_THEMES = {
    "blue": {"bg": "#eaf4ff", "border": "#bfd8f6", "accent": "#1f77b4"},
    "green": {"bg": "#ecfdf5", "border": "#b7ebd0", "accent": "#0f9d58"},
    "orange": {"bg": "#fff7ed", "border": "#fed7aa", "accent": "#c2410c"},
    "purple": {"bg": "#f5f3ff", "border": "#ddd6fe", "accent": "#6d28d9"},
    "gray": {"bg": "#f7f9fc", "border": "#d8e0ea", "accent": "#334155"},
}


def render_analysis_card(
    title: str,
    paragraphs: list[str] | None = None,
    bullets: list[str] | None = None,
    tone: str = "blue",
) -> None:
    theme = CARD_THEMES.get(tone, CARD_THEMES["blue"])
    parts: list[str] = [
        f"<div style='font-weight:700; font-size:1.02em; color:{theme['accent']}; margin-bottom:6px;'>{html.escape(title)}</div>"
    ]

    for p in paragraphs or []:
        parts.append(f"<div style='margin-bottom:6px; color:#111827;'>{html.escape(p)}</div>")

    if bullets:
        parts.append("<ul style='margin:4px 0 0 18px; color:#111827;'>")
        for b in bullets:
            parts.append(f"<li style='margin-bottom:3px;'>{html.escape(b)}</li>")
        parts.append("</ul>")

    style = (
        f"background:{theme['bg']}; border:1px solid {theme['border']}; border-left:6px solid {theme['accent']}; "
        "border-radius:10px; padding:12px 14px; margin:6px 0 12px 0; line-height:1.55;"
    )
    st.markdown(f"<div style='{style}'>{''.join(parts)}</div>", unsafe_allow_html=True)

