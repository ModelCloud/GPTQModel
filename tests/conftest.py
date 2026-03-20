import os
import sys
from pathlib import Path


# Reduce logbar progress noise for pytest runs unless a caller explicitly
# overrides the environment. Keep the library default unchanged in LogBar
# itself; this is only a test harness preference.
os.environ.setdefault("LOGBAR_ANIMATION", "0")
os.environ.setdefault("LOGBAR_PROGRESS_OUTPUT_INTERVAL", "10")

_MODELS_TESTS_DIR = Path(__file__).resolve().parent / "models"
if str(_MODELS_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_MODELS_TESTS_DIR))
