# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for #4809.

`GPTQModel.export(..., "mlx")` loads the checkpoint with a per-method backend
and no device. For AWQ it picks `BACKEND.AWQ_TORCH`, which resolves the GEMV
layouts to CPU-only readers (`AwqGEMVTorchLinear` / `AwqGEMVFastTorchLinear`),
so on any host with a visible GPU the export failed with
`does not support device: cuda:0` before the conversion started.

The MLX converter only reads packed host tensors, so the export now loads on
CPU. Paro keeps the default device because its export backend is CUDA-only.
"""

import sys
import types
from types import SimpleNamespace

import pytest

import gptqmodel.models.auto as auto
from gptqmodel.models.auto import GPTQModel
from gptqmodel.utils.backend import BACKEND


def _install_fake_mlx(monkeypatch, saved):
    utils = types.ModuleType("mlx_lm.utils")
    utils.save_model = lambda path, weights, donate_model=False: saved.append(("model", path))
    utils.save_config = lambda config, config_path=None: saved.append(("config", config_path))
    package = types.ModuleType("mlx_lm")
    package.utils = utils
    monkeypatch.setitem(sys.modules, "mlx_lm", package)
    monkeypatch.setitem(sys.modules, "mlx_lm.utils", utils)

    converter = types.ModuleType("gptqmodel.utils.mlx")
    converter.convert_gptq_to_mlx_weights = lambda *args, **kwargs: ({}, {})
    monkeypatch.setitem(sys.modules, "gptqmodel.utils.mlx", converter)


def _run_mlx_export(monkeypatch, tmp_path, method, *, format_=None):
    recorded = {}
    saved = []

    def fake_load(model_id_or_path, **kwargs):
        recorded.update(kwargs)
        tokenizer = SimpleNamespace(
            save_pretrained=lambda path: saved.append(("tokenizer", path))
        )
        return SimpleNamespace(
            model=SimpleNamespace(),
            model_local_path=model_id_or_path,
            lm_head="lm_head.weight",
            tokenizer=tokenizer,
        )

    quantization_config = {"method": method}
    if format_ is not None:
        quantization_config["format"] = format_

    monkeypatch.setattr(auto, "resolve_trust_remote_code", lambda *args, **kwargs: False)
    monkeypatch.setattr(
        auto.AutoConfig,
        "from_pretrained",
        staticmethod(lambda *args, **kwargs: SimpleNamespace(quantization_config=quantization_config)),
    )
    monkeypatch.setattr(GPTQModel, "load", staticmethod(fake_load))
    _install_fake_mlx(monkeypatch, saved)

    source = tmp_path / "source"
    source.mkdir()
    GPTQModel.export(str(source), str(tmp_path / "out"), "mlx")

    assert saved, "the export wrote nothing"
    return recorded


@pytest.mark.parametrize("format_", ["gemv", "gemv_fast", "llm-awq", "gemm"])
def test_awq_mlx_export_loads_on_cpu(monkeypatch, tmp_path, format_):
    recorded = _run_mlx_export(monkeypatch, tmp_path, "awq", format_=format_)

    assert recorded["backend"] == BACKEND.AWQ_TORCH
    # A visible accelerator must not select the CPU-only GEMV readers.
    assert recorded["device"] == "cpu"


@pytest.mark.parametrize("method", ["gptq", "fp8", "nf12", "macaq", "gguf", "bitsandbytes"])
def test_other_host_side_exporters_load_on_cpu(monkeypatch, tmp_path, method):
    recorded = _run_mlx_export(monkeypatch, tmp_path, method)

    assert recorded["device"] == "cpu"


def test_paro_keeps_the_default_device(monkeypatch, tmp_path):
    """Paro's export backend is CUDA-only, so it must not be forced onto CPU."""

    recorded = _run_mlx_export(monkeypatch, tmp_path, "paro")

    assert recorded["backend"] == BACKEND.PARO_CUDA
    assert "device" not in recorded
