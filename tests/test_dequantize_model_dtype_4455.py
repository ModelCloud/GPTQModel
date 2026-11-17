# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Issue #4455: ``dequantize_model`` keeps the checkpoint dtype.

``dequantize_model`` used to default to bfloat16, so an fp16 GPTQ checkpoint
was written with ``dtype=bfloat16`` and BF16 tensors. The dense dtype now
defaults to the source checkpoint's own float dtype; an explicit
``target_dtype`` still wins and a checkpoint without one falls back to bf16.
"""

from __future__ import annotations

import json

import pytest
import torch
from safetensors import safe_open

from gptqmodel.utils.model_dequant import dequantize_model, resolve_dense_target_dtype
from tests.test_model_dequant_dense import _tiny_llama_checkpoint


pytestmark = [pytest.mark.cpu]


def _source(tmp_path, **dtype_fields):
    source = tmp_path / "source"
    source.mkdir()
    _tiny_llama_checkpoint(source, bits=4, lm_head_bits=4, legacy_config=False)
    config_path = source / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.pop("dtype", None)
    config.update(dtype_fields)
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return source


def _output_dtypes(out):
    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    with safe_open(str(out / "model.safetensors"), framework="pt") as reader:
        dtypes = {reader.get_tensor(key).dtype for key in reader.keys()}
    return config["dtype"], dtypes


@pytest.mark.parametrize(
    ("dtype_fields", "expected"),
    [
        ({"dtype": "float16"}, torch.float16),
        ({"torch_dtype": "float16"}, torch.float16),
        ({"dtype": "bfloat16"}, torch.bfloat16),
        ({"dtype": "float32"}, torch.float32),
        ({}, torch.bfloat16),
    ],
    ids=["dtype-fp16", "legacy-torch_dtype-fp16", "bf16", "fp32", "undeclared-falls-back-bf16"],
)
def test_dequantize_model_keeps_checkpoint_dtype(tmp_path, dtype_fields, expected):
    source = _source(tmp_path, **dtype_fields)
    out = tmp_path / "out"

    dequantize_model(source, out)

    config_dtype, tensor_dtypes = _output_dtypes(out)
    assert config_dtype == str(expected).removeprefix("torch.")
    assert tensor_dtypes == {expected}


def test_dequantize_model_explicit_target_dtype_overrides_checkpoint(tmp_path):
    source = _source(tmp_path, dtype="float16")
    out = tmp_path / "out"

    dequantize_model(source, out, target_dtype=torch.float32)

    config_dtype, tensor_dtypes = _output_dtypes(out)
    assert config_dtype == "float32"
    assert tensor_dtypes == {torch.float32}


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ({"dtype": "torch.float16"}, torch.float16),
        ({"text_config": {"dtype": "float16"}}, torch.float16),
        ({"dtype": "float8_e4m3fn"}, torch.bfloat16),
        ({"dtype": None, "torch_dtype": "float32"}, torch.float32),
    ],
)
def test_resolve_dense_target_dtype(config, expected):
    assert resolve_dense_target_dtype(config) is expected
