# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import builtins

import pytest

from gptqmodel.nn_modules.qlinear import gemm_hf_kernel as hf_kernel_module
from gptqmodel.nn_modules.qlinear.gemm_hf_kernel import HFKernelLinear
from gptqmodel.nn_modules.qlinear.gemm_hf_kernel_awq import HFKernelAwqLinear
from gptqmodel.utils import python as python_utils


def test_free_threading_build_helper_uses_py_gil_disabled(monkeypatch):
    monkeypatch.setattr(
        python_utils.sysconfig,
        "get_config_var",
        lambda key: 1 if key == "Py_GIL_DISABLED" else None,
    )
    assert python_utils.is_free_threading_build()

    monkeypatch.setattr(
        python_utils.sysconfig,
        "get_config_var",
        lambda key: 0 if key == "Py_GIL_DISABLED" else None,
    )
    assert not python_utils.is_free_threading_build()


@pytest.mark.parametrize("kernel_cls", [HFKernelLinear, HFKernelAwqLinear])
def test_hf_kernel_validate_once_blocks_free_threading_build(monkeypatch, kernel_cls):
    monkeypatch.setattr(hf_kernel_module, "is_free_threading_build", lambda: True)

    attempted = {"value": False}
    original_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "kernels" or name.startswith("kernels."):
            attempted["value"] = True
            raise AssertionError(f"unexpected import of {name}")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    ok, err = kernel_cls.validate_once()

    assert not attempted["value"]
    assert not ok
    assert isinstance(err, RuntimeError)
    assert kernel_cls.__name__ in str(err)
    assert "free-threaded Python builds" in str(err)
    assert "Py_GIL_DISABLED=1" in str(err)
