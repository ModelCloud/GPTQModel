# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint reload must preserve FP8 inverse-scale precision under BF16."""

import json

import pytest
import torch
from safetensors.torch import save_file

from gptqmodel.nn_modules.qlinear.fp8 import TorchFP8Linear
from gptqmodel.utils.model import load_checkpoint_in_model_then_tie_weights


class _FP8Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = TorchFP8Linear(
            bits=8,
            group_size=-1,
            sym=True,
            desc_act=False,
            in_features=1,
            out_features=1,
            bias=False,
            register_buffers=True,
            format="float8_e5m2",
        )

    def tie_weights(self, **kwargs):
        del kwargs

    def forward(self, x):
        return self.linear(x)


def _checkpoint(tmp_path, *, sharded=False, prefix=""):
    weights = {
        f"{prefix}linear.weight": torch.tensor([[57344.0]], dtype=torch.float8_e5m2),
        f"{prefix}linear.weight_scale_inv": torch.tensor([65792.0], dtype=torch.float32),
    }
    if not sharded:
        path = tmp_path / "model.safetensors"
        save_file(weights, path, metadata={"format": "pt"})
        return path

    names = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")
    for (key, tensor), filename in zip(weights.items(), names):
        save_file({key: tensor}, tmp_path / filename, metadata={"format": "pt"})
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": dict(zip(weights, names))}), encoding="utf-8")
    return index


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("sharded", [False, True])
def test_fp8_e5m2_scale_reload_matches_independent_torch_oracle(tmp_path, dtype, sharded):
    checkpoint = _checkpoint(tmp_path, sharded=sharded)
    model = _FP8Model()
    load_checkpoint_in_model_then_tie_weights(
        model,
        checkpoint=str(checkpoint),
        dtype=dtype,
        device_map={"": "cpu"},
        keep_in_fp32_modules=["linear.weight_scale_inv"],
        fp32_bf16_buffer_names=["linear.weight_scale_inv"],
    )

    # The reference owns its arithmetic and source values; no loaded tensor or
    # production dequantizer supplies its expected result.
    x = torch.ones((1, 1), dtype=dtype)
    decoded = torch.tensor([[57344.0]], dtype=torch.float32) / torch.tensor([65792.0], dtype=torch.float32)
    oracle = (x.float() @ decoded.T).to(dtype)
    with torch.inference_mode():
        output = model(x)

    assert model.linear.weight_scale_inv.dtype == torch.float32
    torch.testing.assert_close(model.linear.weight_scale_inv, torch.tensor([65792.0]), rtol=0, atol=0)
    assert output.dtype == dtype
    torch.testing.assert_close(output, oracle, rtol=2e-3, atol=2e-3)


def test_fp8_scale_reload_handles_single_file_key_conversion(tmp_path):
    checkpoint = _checkpoint(tmp_path, prefix="hf_")
    model = _FP8Model()
    load_checkpoint_in_model_then_tie_weights(
        model,
        checkpoint=str(checkpoint),
        dtype=torch.bfloat16,
        device_map={"": "cpu"},
        keep_in_fp32_modules=["linear.weight_scale_inv"],
        fp32_bf16_buffer_names=["linear.weight_scale_inv"],
        key_converter=lambda name: name.removeprefix("hf_"),
    )

    torch.testing.assert_close(model.linear.weight_scale_inv, torch.tensor([65792.0]), rtol=0, atol=0)


def test_fp8_scale_reload_rejects_missing_checkpoint_scale(tmp_path):
    checkpoint = tmp_path / "model.safetensors"
    save_file(
        {"linear.weight": torch.tensor([[57344.0]], dtype=torch.float8_e5m2)},
        checkpoint,
        metadata={"format": "pt"},
    )

    with pytest.raises(RuntimeError, match="linear.weight_scale_inv"):
        load_checkpoint_in_model_then_tie_weights(
            _FP8Model(),
            checkpoint=str(checkpoint),
            dtype=torch.bfloat16,
            device_map={"": "cpu"},
            keep_in_fp32_modules=["linear.weight_scale_inv"],
            fp32_bf16_buffer_names=["linear.weight_scale_inv"],
        )
