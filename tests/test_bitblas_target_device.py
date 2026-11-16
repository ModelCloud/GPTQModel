# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""BitBLAS target detection must use a logical CUDA index, not the env entry."""

import sys
import types

import pytest
import torch

import gptqmodel.nn_modules.qlinear.bitblas as bitblas_module


DETECTOR_MODULE = "gptqmodel.nn_modules.qlinear.bitblas_target_detector"


def _install_fake_bitblas(monkeypatch):
    fake_bitblas = types.ModuleType("bitblas")
    fake_bitblas.__version__ = bitblas_module.MINIMUM_BITBLAS_VERSION
    fake_bitblas.set_log_level = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, "bitblas", fake_bitblas)

    detected = []

    def fake_detect(gpu_id=0):
        detected.append(gpu_id)
        return "cuda -arch=sm_89"

    fake_detector = types.ModuleType(DETECTOR_MODULE)
    fake_detector.patched_auto_detect_nvidia_target = fake_detect
    monkeypatch.setitem(sys.modules, DETECTOR_MODULE, fake_detector)

    monkeypatch.setattr(bitblas_module, "BITBLAS_TARGET", None)
    monkeypatch.setattr(bitblas_module, "BITBLAS_DATABASE_PATH", "/tmp/bitblas-test")
    monkeypatch.setenv("TVM_TARGET", "cuda")

    return detected


@pytest.mark.parametrize(
    "visible_devices",
    [
        "1,3",
        "GPU-2f1c4a0e-0d5b-4c4a-9a7e-6f0f0d4b8c11",
        "MIG-GPU-2f1c4a0e-0d5b-4c4a-9a7e-6f0f0d4b8c11/1/0",
    ],
)
def test_import_bitblas_probes_logical_device_zero(monkeypatch, visible_devices):
    detected = _install_fake_bitblas(monkeypatch)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible_devices)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)

    bitblas_module.import_bitblas()

    assert detected == [0]
    assert bitblas_module.BITBLAS_TARGET == "cuda -arch=sm_89"


def test_import_bitblas_probes_the_selected_logical_device(monkeypatch):
    detected = _install_fake_bitblas(monkeypatch)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,1")
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)

    bitblas_module.import_bitblas()

    assert detected == [1]
