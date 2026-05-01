import importlib.util
import sys
import types
from pathlib import Path


def _load_komodo_cann_module(monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "gptqmodel" / "utils" / "komodo_cann.py"

    package = types.ModuleType("gptqmodel")
    package.__path__ = [str(repo_root / "gptqmodel")]
    utils_package = types.ModuleType("gptqmodel.utils")
    utils_package.__path__ = [str(repo_root / "gptqmodel" / "utils")]
    cpp_module = types.ModuleType("gptqmodel.utils.cpp")

    class DummyTorchOpsJitExtension:
        def __init__(self, *args, **kwargs):
            self._last_error = ""

        def load(self):
            return True

        def last_error_message(self):
            return self._last_error

        def clear_cache(self):
            pass

    cpp_module.TorchOpsJitExtension = DummyTorchOpsJitExtension
    cpp_module.default_jit_cflags = lambda: []
    cpp_module.default_torch_ops_build_root = lambda name: Path("/tmp") / name

    monkeypatch.setitem(sys.modules, "gptqmodel", package)
    monkeypatch.setitem(sys.modules, "gptqmodel.utils", utils_package)
    monkeypatch.setitem(sys.modules, "gptqmodel.utils.cpp", cpp_module)

    spec = importlib.util.spec_from_file_location("gptqmodel.utils.komodo_cann", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "gptqmodel.utils.komodo_cann", module)
    spec.loader.exec_module(module)
    return module


def test_cann_root_prefers_active_ascend_home(monkeypatch):
    komodo_cann = _load_komodo_cann_module(monkeypatch)
    monkeypatch.setenv("ASCEND_HOME_PATH", "/custom/ascend/home")
    monkeypatch.setenv("ASCEND_TOOLKIT_HOME", "/custom/ascend/toolkit")

    assert komodo_cann._cann_root() == Path("/custom/ascend/home")


def test_cann_root_falls_back_to_latest_toolkit(monkeypatch):
    komodo_cann = _load_komodo_cann_module(monkeypatch)
    monkeypatch.delenv("ASCEND_HOME_PATH", raising=False)
    monkeypatch.delenv("ASCEND_TOOLKIT_HOME", raising=False)

    def fake_exists(path):
        return str(path) == "/usr/local/Ascend/ascend-toolkit/latest"

    monkeypatch.setattr(Path, "exists", fake_exists)

    assert komodo_cann._cann_root() == Path("/usr/local/Ascend/ascend-toolkit/latest")
