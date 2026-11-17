# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Staged GSQ GPTQ must run inside the run-level deterministic BLAS pin.

`quantize()` used to call `quantize_llama_gsq_public` and return before the
shared `quant_blas_backend(deterministic=...)` scope, so on ROCm its
calibration forwards ran on hipBLASLt and the checkpoint depended on the
process (issue #4232). These tests assert the call happens inside the
deterministic pin without needing a GPU.
"""

from contextlib import contextmanager

import gptqmodel.looper.gsq_training_model as gsq_model
import gptqmodel.models.base as base_mod
from gptqmodel import CalibrationConfig
from gptqmodel.models.base import BaseQModel
from gptqmodel.quantization import GPTQConfig
from gptqmodel.utils.backend import BACKEND


def _bare_model(quantize_config):
    model = BaseQModel.__new__(BaseQModel)
    model.quantize_config = quantize_config
    model.quantized = False
    model._auto_fusion = False
    model.quant_region_timer = None
    model.after_quantize = lambda: None
    return model


def _staged_gsq_config():
    return GPTQConfig(
        bits=4,
        group_size=32,
        gsq_training={
            "enabled": True,
            "epochs": 1,
            "qk_steps": 2,
            "batch_size": 1,
            "microbatch_size": 1,
        },
    )


def test_staged_gsq_gptq_runs_inside_deterministic_blas_pin(monkeypatch):
    entered = []
    active = {"depth": 0}

    @contextmanager
    def fake_blas_backend(device=None, *, deterministic=None):
        entered.append({"device": device, "deterministic": deterministic})
        active["depth"] += 1
        try:
            yield
        finally:
            active["depth"] -= 1

    monkeypatch.setattr(base_mod, "quant_blas_backend", fake_blas_backend)

    seen = {}

    def fake_gsq_public(model, **kwargs):
        seen["depth_at_call"] = active["depth"]
        seen["calibration"] = kwargs["calibration"]
        return {}

    monkeypatch.setattr(gsq_model, "quantize_llama_gsq_public", fake_gsq_public)
    monkeypatch.setattr(base_mod, "_log_quantize_summary", lambda *args, **kwargs: None)

    model = _bare_model(_staged_gsq_config())
    calibration = [{"input_ids": [1, 2, 3]}]

    result = model.quantize(
        CalibrationConfig(
            dataset=calibration,
            min_length=1,
        ),
        backend=BACKEND.TORCH,
    )

    assert result == {}
    assert seen["calibration"].dataset is calibration
    # The staged run must start inside the pin, not before or after it.
    assert seen["depth_at_call"] == 1
    assert [call["deterministic"] for call in entered] == [True]
    assert active["depth"] == 0


def test_staged_gsq_pin_wraps_only_the_staged_run(monkeypatch):
    order = []

    @contextmanager
    def fake_blas_backend(device=None, *, deterministic=None):
        order.append("enter")
        yield
        order.append("exit")

    monkeypatch.setattr(base_mod, "quant_blas_backend", fake_blas_backend)
    monkeypatch.setattr(
        gsq_model,
        "quantize_llama_gsq_public",
        lambda model, **kwargs: order.append("quantize") or {},
    )
    monkeypatch.setattr(base_mod, "_log_quantize_summary", lambda *args, **kwargs: None)

    model = _bare_model(_staged_gsq_config())
    model.quantize(CalibrationConfig(dataset=[{"input_ids": [1, 2, 3]}], min_length=1), backend=BACKEND.TORCH)

    assert order == ["enter", "quantize", "exit"]

