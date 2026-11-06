"""Where the app keeps things. Nothing here goes in /tmp: runs and captures take hours to make."""

import os
from pathlib import Path

HOME = Path.home()
COLIBRI = Path(os.environ.get("COLIBRI_DIR") or HOME / "colibri")
DEFAULT_MODEL = HOME / "colibri-models/kimi-k3-abliterated"


def _xdg(var: str, fallback: str) -> Path:
    path = Path(os.environ.get(var) or HOME / fallback) / "k3-abliterator"
    path.mkdir(parents=True, exist_ok=True)
    return path


def data_dir() -> Path:
    """Runs, captures, directions and the built engine."""
    return _xdg("XDG_DATA_HOME", ".local/share")


def state_dir() -> Path:
    """Logs, update checks, problem-report queue."""
    return _xdg("XDG_STATE_HOME", ".local/state")


def log_file() -> Path:
    return state_dir() / "app.log"


def engine_log() -> Path:
    return state_dir() / "engine.log"
