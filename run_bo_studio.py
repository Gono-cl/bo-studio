import os
import importlib
import socket
import sys
import tempfile
import traceback
import webbrowser
import json
from pathlib import Path

STARTUP_LOG_ENABLED = os.getenv("BENCHBO_STARTUP_LOG", "").strip().lower() in {"1", "true", "yes", "on"}
STARTUP_LOG_FILE = Path(tempfile.gettempdir()) / "benchbo_startup.log"


def _startup_log(message: str) -> None:
    if not STARTUP_LOG_ENABLED:
        return
    try:
        with STARTUP_LOG_FILE.open("a", encoding="utf-8") as handle:
            handle.write(f"{message}\n")
    except Exception:
        pass


_startup_log("startup: module import begin")
try:
    from streamlit import config as _config
    from streamlit.web import bootstrap
    _startup_log("startup: streamlit imports ok")
except Exception:
    _startup_log("startup: streamlit imports failed")
    _startup_log(traceback.format_exc())
    raise

PRELOAD_MODULES = [
    "skopt",
    "skopt.space",
    "seaborn",
    "sklearn",
    "sklearn.preprocessing",
    "sklearn.inspection",
]

for _module in PRELOAD_MODULES:
    try:
        importlib.import_module(_module)
    except ImportError:
        # During development builds some optional deps might be missing; the
        # PyInstaller command adds them explicitly via --hidden-import.
        pass
    except Exception:
        _startup_log(f"startup: preload failed for {_module}")
        _startup_log(traceback.format_exc())
        raise
_startup_log("startup: preload imports ok")

DESIRED_PORT = 8501
def _resolve_bundle_root() -> Path:
    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            return Path(meipass).resolve()
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


BUNDLE_ROOT = _resolve_bundle_root()
APP_PATH = BUNDLE_ROOT / "main.py"
APP_STATE_DIR = Path(os.getenv("LOCALAPPDATA", str(Path.home()))) / "BenchBO"
LAST_PORT_FILE = APP_STATE_DIR / "last_port.txt"
LAST_SESSION_FILE = APP_STATE_DIR / "last_session.json"

_startup_log(f"startup: bundle root {BUNDLE_ROOT}")
_startup_log(f"startup: app path {APP_PATH} exists={APP_PATH.exists()}")


def _is_port_in_use(host: str, port: int) -> bool:
    """Return True when a TCP server is already listening on host:port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex((host, port)) == 0


def _find_available_port(start_port: int, max_tries: int = 50) -> int:
    """Find the first bindable port starting at start_port."""
    for port in range(start_port, start_port + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No available port found in range {start_port}-{start_port + max_tries - 1}.")


def _open_browser(port: int) -> None:
    webbrowser.open_new_tab(f"http://127.0.0.1:{port}")


def _load_last_port() -> int | None:
    try:
        if LAST_PORT_FILE.exists():
            value = int(LAST_PORT_FILE.read_text(encoding="utf-8").strip())
            if 1 <= value <= 65535:
                return value
    except Exception:
        return None
    return None


def _save_last_port(port: int) -> None:
    try:
        APP_STATE_DIR.mkdir(parents=True, exist_ok=True)
        LAST_PORT_FILE.write_text(str(port), encoding="utf-8")
    except Exception:
        # Non-critical: startup should continue even if we cannot persist state.
        pass


def _normalize_path_text(path_text: str | None) -> str:
    if not path_text:
        return ""
    try:
        return str(Path(path_text).resolve()).lower()
    except Exception:
        return str(path_text).strip().lower()


def _load_last_session() -> dict:
    # Preferred format.
    try:
        if LAST_SESSION_FILE.exists():
            payload = json.loads(LAST_SESSION_FILE.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                port = int(payload.get("port", 0))
                if 1 <= port <= 65535:
                    return {
                        "port": port,
                        "mode": str(payload.get("mode", "")),
                        "storage_root": str(payload.get("storage_root", "")),
                    }
    except Exception:
        pass

    # Backward-compat fallback (old last_port.txt only).
    old_port = _load_last_port()
    if old_port is not None:
        return {"port": old_port, "mode": "", "storage_root": ""}
    return {}


def _save_last_session(port: int, mode: str, storage_root: str) -> None:
    try:
        APP_STATE_DIR.mkdir(parents=True, exist_ok=True)
        LAST_SESSION_FILE.write_text(
            json.dumps(
                {
                    "port": int(port),
                    "mode": str(mode),
                    "storage_root": str(storage_root),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    except Exception:
        pass
    _save_last_port(port)


def configure_streamlit(port: int) -> None:
    """Apply host/port overrides before the runtime initializes."""
    os.environ.setdefault("STREAMLIT_DEVELOPMENT_MODE", "false")
    os.environ.setdefault("STREAMLIT_SERVER_PORT", str(port))
    os.environ.setdefault("STREAMLIT_SERVER_ADDRESS", "127.0.0.1")
    os.environ.setdefault("STREAMLIT_SERVER_HEADLESS", "false")
    os.environ.setdefault("STREAMLIT_BROWSER_SERVER_PORT", str(port))
    os.environ.setdefault("STREAMLIT_BROWSER_SERVER_ADDRESS", "127.0.0.1")

    _config.set_option("global.developmentMode", False)
    _config.set_option("server.headless", False)
    _config.set_option("server.address", "127.0.0.1")
    _config.set_option("server.port", port)
    _config.set_option("browser.serverAddress", "127.0.0.1")
    _config.set_option("browser.serverPort", port)


def main() -> None:
    _startup_log("startup: main entered")
    is_frozen = bool(getattr(sys, "frozen", False))
    mode = "frozen" if is_frozen else "source"
    storage_root = ""

    # In portable frozen mode, persist data alongside the executable folder.
    if is_frozen:
        exe_dir = Path(sys.executable).resolve().parent
        storage_root = str(exe_dir)
        os.environ.setdefault("BENCHBO_STORAGE_ROOT", storage_root)
        os.environ.setdefault("BOSTUDIO_STORAGE_ROOT", storage_root)
    else:
        storage_root = str(Path.cwd().resolve())

    # In packaged EXE mode, reopen existing BenchBO session if already running.
    # In source/dev mode, always start a fresh Streamlit server so code edits are
    # reflected immediately (avoids attaching to stale previous sessions).
    if is_frozen:
        session = _load_last_session()
        last_port = int(session.get("port", 0) or 0)
        same_mode = str(session.get("mode", "")).lower() == "frozen"
        same_root = _normalize_path_text(session.get("storage_root")) == _normalize_path_text(storage_root)
        if last_port and same_mode and same_root and _is_port_in_use("127.0.0.1", last_port):
            _open_browser(last_port)
            return

    selected_port = _find_available_port(DESIRED_PORT)
    _startup_log(f"startup: selected port {selected_port}")
    _save_last_session(selected_port, mode=mode, storage_root=storage_root)
    _startup_log("startup: session saved")
    configure_streamlit(selected_port)
    _startup_log("startup: streamlit configured")
    # streamlit.web.bootstrap.run expects is_hello as a bool (2nd arg).
    _startup_log("startup: bootstrap.run begin")
    bootstrap.run(str(APP_PATH), False, [], {})


if __name__ == "__main__":
    main()
