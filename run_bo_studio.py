import os
import importlib
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


def configure_streamlit() -> None:
    """Apply host/port overrides before the runtime initializes."""
    os.environ.setdefault("STREAMLIT_DEVELOPMENT_MODE", "false")
    os.environ.setdefault("STREAMLIT_SERVER_PORT", str(DESIRED_PORT))
    os.environ.setdefault("STREAMLIT_SERVER_ADDRESS", "127.0.0.1")
    os.environ.setdefault("STREAMLIT_SERVER_HEADLESS", "false")
    os.environ.setdefault("STREAMLIT_BROWSER_SERVER_PORT", str(DESIRED_PORT))
    os.environ.setdefault("STREAMLIT_BROWSER_SERVER_ADDRESS", "127.0.0.1")

    _config.set_option("global.developmentMode", False)
    _config.set_option("server.headless", False)
    _config.set_option("server.address", "127.0.0.1")
    _config.set_option("server.port", DESIRED_PORT)
    _config.set_option("browser.serverAddress", "127.0.0.1")
    _config.set_option("browser.serverPort", DESIRED_PORT)


def main() -> None:
    configure_streamlit()
    bootstrap.run(str(APP_PATH), "", [], {})


if __name__ == "__main__":
    main()
