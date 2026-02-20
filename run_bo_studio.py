import os
import importlib
import socket
import sys
import webbrowser
from pathlib import Path

from streamlit import config as _config
from streamlit.web import bootstrap

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

DESIRED_PORT = 8501
APP_PATH = Path(__file__).parent / "main.py"
APP_STATE_DIR = Path(os.getenv("LOCALAPPDATA", str(Path.home()))) / "BOStudio"
LAST_PORT_FILE = APP_STATE_DIR / "last_port.txt"


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
    is_frozen = bool(getattr(sys, "frozen", False))

    # In packaged EXE mode, reopen existing BO Studio session if already running.
    # In source/dev mode, always start a fresh Streamlit server so code edits are
    # reflected immediately (avoids attaching to stale previous sessions).
    if is_frozen:
        last_port = _load_last_port()
        if last_port is not None and _is_port_in_use("127.0.0.1", last_port):
            _open_browser(last_port)
            return

    selected_port = _find_available_port(DESIRED_PORT)
    _save_last_port(selected_port)
    configure_streamlit(selected_port)
    # streamlit.web.bootstrap.run expects is_hello as a bool (2nd arg).
    bootstrap.run(str(APP_PATH), False, [], {})


if __name__ == "__main__":
    main()
