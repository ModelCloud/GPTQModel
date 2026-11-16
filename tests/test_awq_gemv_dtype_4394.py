# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""#4394: reject non-fp16 AWQ GEMV configs before calibration.

``AwqGEMVLinear``/``AwqGEMVTritonLinear`` and the GEMV_FAST/LLM_AWQ kernels
declare ``SUPPORTS_DTYPES = [torch.float16]``, but the dtype was only checked
when layer 0 was built in finalize. The AWQ search had already run by then, so
a bf16 model paid for calibration and only then crashed.
"""

import json

import pytest
import torch

from gptqmodel import GPTQModel
from gptqmodel.models import auto
from gptqmodel.quantization.config import FORMAT, AWQConfig


FP16_ONLY_FORMATS = (FORMAT.GEMV, FORMAT.GEMV_FAST, FORMAT.LLM_AWQ)


class _ModelLoadReached(Exception):
    """Sentinel: the load-time dtype guard let the request through."""


def _write_tiny_model(path, *, dtype="bfloat16"):
    (path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "llama",
                "architectures": ["LlamaForCausalLM"],
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_attention_heads": 4,
                "num_key_value_heads": 4,
                "num_hidden_layers": 1,
                "vocab_size": 128,
                "max_position_embeddings": 128,
                "rms_norm_eps": 1e-6,
                "torch_dtype": dtype,
            }
        )
    )


def _fake_definition(*, require_dtype=None):
    class FakeDefinition:
        @staticmethod
        def from_pretrained(**_kwargs):
            raise _ModelLoadReached

    FakeDefinition.require_dtype = require_dtype
    return FakeDefinition


@pytest.mark.parametrize("format_value", FP16_ONLY_FORMATS, ids=lambda fmt: fmt.value)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=str)
def test_fp16_only_awq_formats_reject_non_fp16_dtype(format_value, dtype):
    config = AWQConfig(bits=4, group_size=128, format=format_value)

    with pytest.raises(ValueError, match="only supports `dtype=torch.float16`") as exc:
        config.validate_dtype(dtype)

    message = str(exc.value)
    assert "dtype=torch.float16" in message
    assert format_value.value in message
    assert "FORMAT.GEMM" in message


@pytest.mark.parametrize("format_value", FP16_ONLY_FORMATS, ids=lambda fmt: fmt.value)
def test_fp16_only_awq_formats_accept_fp16(format_value):
    AWQConfig(bits=4, group_size=128, format=format_value).validate_dtype(torch.float16)


@pytest.mark.parametrize(
    "format_value",
    [FORMAT.GEMM, FORMAT.MARLIN, FORMAT.BITBLAS],
    ids=lambda fmt: fmt.value,
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=str)
def test_gemm_marlin_and_bitblas_dtypes_are_unaffected(format_value, dtype):
    AWQConfig(bits=4, group_size=128, format=format_value).validate_dtype(dtype)


def test_load_rejects_bf16_gemv_before_model_load(tmp_path, monkeypatch):
    _write_tiny_model(tmp_path)
    monkeypatch.setattr(
        auto, "check_and_get_model_definition", lambda *_args, **_kwargs: _fake_definition()
    )

    with pytest.raises(ValueError, match="only supports `dtype=torch.float16`") as exc:
        GPTQModel.load(
            str(tmp_path),
            quantize_config=AWQConfig(bits=4, group_size=128, format=FORMAT.GEMV),
        )

    assert "dtype=torch.float16" in str(exc.value)


def test_load_accepts_explicit_fp16_gemv(tmp_path, monkeypatch):
    _write_tiny_model(tmp_path)
    monkeypatch.setattr(
        auto, "check_and_get_model_definition", lambda *_args, **_kwargs: _fake_definition()
    )

    with pytest.raises(_ModelLoadReached):
        GPTQModel.load(
            str(tmp_path),
            dtype=torch.float16,
            quantize_config=AWQConfig(bits=4, group_size=128, format=FORMAT.GEMV),
        )


def test_load_keeps_bf16_gemm_working(tmp_path, monkeypatch):
    _write_tiny_model(tmp_path)
    monkeypatch.setattr(
        auto, "check_and_get_model_definition", lambda *_args, **_kwargs: _fake_definition()
    )

    with pytest.raises(_ModelLoadReached):
        GPTQModel.load(
            str(tmp_path),
            quantize_config=AWQConfig(bits=4, group_size=128, format=FORMAT.GEMM),
        )


def test_model_required_dtype_still_satisfies_gemv(tmp_path, monkeypatch):
    _write_tiny_model(tmp_path)
    monkeypatch.setattr(
        auto,
        "check_and_get_model_definition",
        lambda *_args, **_kwargs: _fake_definition(require_dtype=torch.float16),
    )

    with pytest.raises(_ModelLoadReached):
        GPTQModel.load(
            str(tmp_path),
            quantize_config=AWQConfig(bits=4, group_size=128, format=FORMAT.GEMV),
        )
