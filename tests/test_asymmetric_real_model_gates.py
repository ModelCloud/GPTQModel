# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from importlib import import_module
from types import SimpleNamespace

import pytest
import torch

from gptqmodel import extension
from tests.kernels import test_asymmetric_real_models as real_models


_CASES = real_models._AWQ_REAL_CASES + real_models._GPTQ_REAL_CASES


def _mock_device(monkeypatch, capability=(9, 0), shared_memory=227328, rocm=False):
    def get_capability(device=None):
        assert device == real_models._DEVICE
        return capability

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", get_capability)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device=None: SimpleNamespace(
            name="test GPU",
            shared_memory_per_block=49152,
            shared_memory_per_block_optin=shared_memory,
        ),
    )
    for module_name in (
        "gptqmodel.utils.machete",
        "gptqmodel.utils.marlin",
        *{case.candidate_cls.__module__ for case in _CASES},
    ):
        monkeypatch.setattr(import_module(module_name), "IS_ROCM", rocm)


@pytest.mark.parametrize("case", _CASES, ids=lambda case: case.name)
@pytest.mark.parametrize(
    "capability,shared_memory,rocm,supported",
    [
        ((9, 0), 227328, False, {case.name for case in _CASES}),
        ((8, 9), 102400, False, {"awq_marlin"}),
        ((10, 0), 227328, False, {"awq_marlin"}),
        ((7, 5), 65536, False, set()),
        ((9, 0), 102400, False, {"awq_marlin"}),
        ((9, 0), 227328, True, set()),
    ],
    ids=["hopper", "ada", "blackwell", "turing", "low_shared_memory", "rocm"],
)
def test_gate_checks_hardware_without_loading_extensions(
    monkeypatch,
    case,
    capability,
    shared_memory,
    rocm,
    supported,
):
    _mock_device(monkeypatch, capability, shared_memory, rocm)

    def unexpected_load(*args, **kwargs):
        pytest.fail("Hardware gating must not load or compile an extension")

    monkeypatch.setattr(extension, "_load_one", unexpected_load)
    if case.name in supported:
        real_models._require_candidate_backend(case)
    else:
        with pytest.raises(pytest.skip.Exception):
            real_models._require_candidate_backend(case)


@pytest.mark.parametrize("case", _CASES, ids=lambda case: case.name)
def test_generation_propagates_extension_build_failure(monkeypatch, case):
    _mock_device(monkeypatch)
    message = "CUDA source compilation regression"

    def failed_load(*args, **kwargs):
        raise RuntimeError(message)

    monkeypatch.setattr(extension, "_load_one", failed_load)
    monkeypatch.setattr(extension, "error", lambda name: message)

    def load_model(model_id, backend):
        if backend == case.baseline_backend:
            return object()
        extension_name = "marlin_fp16" if case.name == "awq_marlin" else "machete"
        return extension.op(extension_name, "unused")

    monkeypatch.setattr(real_models, "_load_model", load_model)
    monkeypatch.setattr(real_models, "_generate_completion", lambda model: "baseline")
    monkeypatch.setattr(real_models, "_release_model", lambda model: None)

    # Exercise the actual test entry point: a supported host must reach model
    # loading and report the build error, rather than skip before loading.
    with pytest.raises(RuntimeError, match=message):
        try:
            real_models.test_asymmetric_real_generation_matches_torch_and_is_sane(case)
        except pytest.skip.Exception as exc:
            pytest.fail(f"Build failure was hidden by a skip: {exc}")
