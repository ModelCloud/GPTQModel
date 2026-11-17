# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""CPU regressions for the ``GPTQModel.export`` contract (#4386).

Covers the four review findings: unknown formats must fail closed before a
model load, ``trust_remote_code`` must reach the loader, the dense ``hf``
output must not keep quantization metadata, and ``BaseQModel.to`` must raise a
real exception when the wrapped model has no ``to``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from gptqmodel import GPTQModel
from gptqmodel.models import auto as auto_module
from gptqmodel.models.base import BaseQModel
from gptqmodel.utils.model_dequant import copy_aux_files
from tests.test_export import _tiny_llama_checkpoint


_QUANT_METADATA_FILES = ("quantize_config.json", "quant_config.json", "quant_log.csv")


@pytest.mark.parametrize("bad_format", ["gguf", "safetensors", "HF2", "mlx2", ""])
def test_export_rejects_unknown_format_before_loading(tmp_path, monkeypatch, bad_format):
    load_calls = []

    def fake_load(*args, **kwargs):
        load_calls.append((args, kwargs))
        raise AssertionError("export must reject the format before loading the model")

    monkeypatch.setattr(GPTQModel, "load", staticmethod(fake_load))

    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported export format"):
        GPTQModel.export(str(source), str(tmp_path / "out"), bad_format)

    assert load_calls == []


@pytest.mark.parametrize("export_format", ["hf", "HF", "Hf"])
def test_export_normalizes_hf_format_case_insensitively(tmp_path, monkeypatch, export_format):
    from gptqmodel.utils import model_dequant

    source = tmp_path / "source"
    source.mkdir()

    calls = []
    monkeypatch.setattr(
        model_dequant,
        "dequantize_model",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    GPTQModel.export(str(source), str(tmp_path / "out"), export_format)

    assert len(calls) == 1


def test_export_mlx_forwards_trust_remote_code(tmp_path, monkeypatch):
    captured = {}

    monkeypatch.setattr(auto_module, "resolve_trust_remote_code", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        auto_module.AutoConfig,
        "from_pretrained",
        classmethod(lambda cls, *args, **kwargs: SimpleNamespace(quantization_config={"quant_method": "gptq"})),
    )

    def fake_load(model_id_or_path, **kwargs):
        captured["model_id_or_path"] = model_id_or_path
        captured["kwargs"] = kwargs
        raise RuntimeError("stop after load")

    monkeypatch.setattr(GPTQModel, "load", staticmethod(fake_load))

    with pytest.raises(RuntimeError, match="stop after load"):
        GPTQModel.export(
            "remote-code-checkpoint",
            str(tmp_path / "out"),
            "mlx",
            trust_remote_code=True,
        )

    assert captured["model_id_or_path"] == "remote-code-checkpoint"
    assert captured["kwargs"].get("trust_remote_code") is True


@pytest.mark.parametrize(
    ("quant_method", "expected_backend"),
    [
        ("gptq", "GPTQ_TORCH"),
        ("awq", "AWQ_TORCH"),
        ("paro", "PARO_CUDA"),
        ("gguf", "GGUF_TORCH"),
    ],
)
def test_export_mlx_loads_the_checkpoint_on_cpu(
    tmp_path, monkeypatch, quant_method, expected_backend
):
    """``export(..., "mlx")`` only reads packed tensors on the host (#4809).

    Leaving the device implicit made the loader pick the accelerator whenever
    one was visible. The AWQ GEMV/GEMV_FAST Torch readers are CPU-only, so a
    GPU host failed with ``does not support device: cuda:0`` before any
    conversion, even though the converter supports those formats.
    """

    from gptqmodel.utils.backend import BACKEND

    captured = {}
    monkeypatch.setattr(auto_module, "resolve_trust_remote_code", lambda *args, **kwargs: False)
    monkeypatch.setattr(
        auto_module.AutoConfig,
        "from_pretrained",
        classmethod(
            lambda cls, *args, **kwargs: SimpleNamespace(
                quantization_config={"quant_method": quant_method}
            )
        ),
    )

    def fake_load(model_id_or_path, **kwargs):
        captured["kwargs"] = kwargs
        raise RuntimeError("stop after load")

    monkeypatch.setattr(GPTQModel, "load", staticmethod(fake_load))

    with pytest.raises(RuntimeError, match="stop after load"):
        GPTQModel.export("checkpoint", str(tmp_path / "out"), "mlx")

    assert captured["kwargs"].get("device") == "cpu"
    assert captured["kwargs"].get("backend") is getattr(BACKEND, expected_backend)


def test_export_hf_drops_quantization_metadata(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    _tiny_llama_checkpoint(source, bits=4, lm_head_bits=4, legacy_config=True)
    (source / "quant_config.json").write_text("{}", encoding="utf-8")
    (source / "quant_log.csv").write_text("layer,bits\n", encoding="utf-8")

    out = tmp_path / "out"
    GPTQModel.export(str(source), str(out), "hf")

    assert (out / "model.safetensors").is_file()
    for name in _QUANT_METADATA_FILES:
        assert not (out / name).exists(), f"dense export kept quantization metadata: {name}"
        assert (source / name).exists(), f"export must not delete the source file: {name}"


def test_copy_aux_files_still_copies_quantization_metadata(tmp_path):
    """Quantized checkpoints keep their metadata; only the dense export drops it."""

    source = tmp_path / "source"
    source.mkdir()
    for name in _QUANT_METADATA_FILES:
        (source / name).write_text("{}", encoding="utf-8")

    out = tmp_path / "out"
    out.mkdir()
    copy_aux_files(source, out, skip=set())

    for name in _QUANT_METADATA_FILES:
        assert (out / name).is_file()


def test_base_qmodel_to_raises_not_implemented_without_to():
    class ModelWithoutTo:
        pass

    holder = object.__new__(BaseQModel)
    holder.model = ModelWithoutTo()

    with pytest.raises(NotImplementedError, match=r"does not support the to\(\) method"):
        BaseQModel.to(holder, "cpu")
