import os
from pathlib import Path
import sys


# Reduce logbar progress animation noise for pytest runs unless a caller
# explicitly opts back in via the environment.
os.environ.setdefault("LOGBAR_ANIMATION", "0")

_MODELS_TESTS_DIR = Path(__file__).resolve().parent / "models"
if str(_MODELS_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_MODELS_TESTS_DIR))
