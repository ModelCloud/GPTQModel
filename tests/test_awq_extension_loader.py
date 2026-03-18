import sys
from pathlib import Path

import pytest

from gptqmodel.quantization.awq.utils import module as awq_module
from gptqmodel.utils import _extension_loader


def test_awq_try_import_uses_shared_extension_loader(monkeypatch):
    sentinel = object()
    seen = {}

    def fake_load_extension_module(module_name, package="gptqmodel"):
        seen["module_name"] = module_name
        seen["package"] = package
        return sentinel

    monkeypatch.setattr(awq_module, "load_extension_module", fake_load_extension_module)

    module, msg = awq_module.try_import("gptqmodel_awq_kernels")

    assert module is sentinel
    assert msg == ""
    assert seen == {"module_name": "gptqmodel_awq_kernels", "package": "gptqmodel"}


def test_awq_try_import_finds_repo_root_extension_from_tests_dir(monkeypatch):
    ext_path = _extension_loader._resolve_extension_path("gptqmodel_awq_kernels", "gptqmodel")
    if ext_path is None:
        pytest.skip("AWQ extension is not built in this environment.")

    original_import_module = _extension_loader.importlib.import_module

    def fake_import_module(name, *args, **kwargs):
        if name == "gptqmodel_awq_kernels":
            raise ModuleNotFoundError("No module named 'gptqmodel_awq_kernels'")
        return original_import_module(name, *args, **kwargs)

    monkeypatch.chdir(Path(__file__).resolve().parent)
    monkeypatch.setattr(_extension_loader.importlib, "import_module", fake_import_module)
    sys.modules.pop("gptqmodel_awq_kernels", None)

    module, msg = awq_module.try_import("gptqmodel_awq_kernels")

    assert module is not None, msg
    assert getattr(module, "__file__", "").endswith(ext_path.name)
