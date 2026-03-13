import importlib.util
from pathlib import Path


_MODEL_TEST_PATH = Path(__file__).resolve().parent / "tests" / "models" / "model_test.py"
_SPEC = importlib.util.spec_from_file_location("_model_test_impl", _MODEL_TEST_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Unable to load ModelTest from {_MODEL_TEST_PATH}")

_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

globals().update({name: getattr(_MODULE, name) for name in dir(_MODULE) if not name.startswith("_")})
