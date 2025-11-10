"""Helpers for resolving resource paths in both dev and frozen builds."""
from __future__ import annotations

from pathlib import Path
import sys


def _detect_base_path() -> Path:
    """Return the directory where bundled resources live."""
    if hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    # File lives in core/utils, so two parents up is the project root
    return Path(__file__).resolve().parents[2]


BASE_PATH = _detect_base_path()


def resource_path(relative: str | Path) -> Path:
    """Return an absolute path for a file shipped with the app."""
    return BASE_PATH / Path(relative)
