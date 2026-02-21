from __future__ import annotations

import os
import sys
from pathlib import Path


def get_storage_root() -> Path:
    """Return the base directory for persistent app data."""
    if os.getenv("RENDER") == "true":
        return Path("/mnt/data")

    env_root = os.getenv("BOSTUDIO_STORAGE_ROOT")
    if env_root:
        return Path(env_root)

    # Portable fallback for frozen desktop builds.
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent

    # Source/dev fallback keeps prior behavior.
    return Path(os.getcwd())


def get_campaigns_dir() -> Path:
    path = get_storage_root() / "resumable_manual_runs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_data_dir() -> Path:
    path = get_storage_root() / "data"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_db_path() -> Path:
    return get_data_dir() / "experiments.db"

