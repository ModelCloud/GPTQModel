from pathlib import Path
import sys


_MODELS_TESTS_DIR = Path(__file__).resolve().parent / "models"
if str(_MODELS_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_MODELS_TESTS_DIR))
