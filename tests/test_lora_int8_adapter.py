# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from safetensors.torch import save_file

from gptqmodel.adapter.adapter import HF_ADAPTER_CONFIG_FILE_NAME, HF_ADAPTER_WEIGHT_KEY_PREFIX, Lora
from gptqmodel.adapter.peft import LoraConfig
from gptqmodel.adapter.quant import (
    compressed_weight_keys,
    dequantize_tensor_groupwise_int,
    dequantize_tensor_groupwise_int8,
    lora_grouped_format_from_bits,
    quantize_tensor_groupwise_int,
    quantize_tensor_groupwise_int8,
)


def _compressed_entries(weight_key: str, tensor: torch.Tensor, group_size: int, bits: int = 8) -> dict[str, torch.Tensor]:
    q_key, scales_key, shape_key = compressed_weight_keys(weight_key)
    qweight, scales, shape = quantize_tensor_groupwise_int(tensor, bits=bits, group_size=group_size)
    return {
        q_key: qweight,
        scales_key: scales,
        shape_key: shape,
    }


def test_groupwise_int8_lora_roundtrip_error_is_small():
    torch.manual_seed(0)
    tensor = torch.randn(257, 193, dtype=torch.bfloat16) * 0.02
    qweight, scales, shape = quantize_tensor_groupwise_int8(tensor, group_size=128)

    actual = dequantize_tensor_groupwise_int8(
        qweight=qweight,
        scales=scales,
        shape=shape,
        group_size=128,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    expected = tensor.float()
    rel_l2 = torch.linalg.vector_norm(actual - expected) / torch.linalg.vector_norm(expected)

    assert qweight.dtype == torch.int8
    assert scales.dtype == torch.bfloat16
    assert rel_l2.item() < 0.01


@pytest.mark.parametrize(
    ("bits", "rel_l2_limit"),
    [
        (4, 0.13),
        (6, 0.035),
        (8, 0.01),
    ],
)
def test_groupwise_low_bit_lora_roundtrip_error_is_bounded(bits, rel_l2_limit):
    torch.manual_seed(10 + bits)
    tensor = torch.randn(257, 193, dtype=torch.bfloat16) * 0.02
    qweight, scales, shape = quantize_tensor_groupwise_int(tensor, bits=bits, group_size=128)

    actual = dequantize_tensor_groupwise_int(
        qweight=qweight,
        scales=scales,
        shape=shape,
        bits=bits,
        group_size=128,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    expected = tensor.float()
    rel_l2 = torch.linalg.vector_norm(actual - expected) / torch.linalg.vector_norm(expected)

    assert qweight.dtype == (torch.int8 if bits == 8 else torch.uint8)
    assert scales.dtype == torch.bfloat16
    assert rel_l2.item() < rel_l2_limit


@pytest.mark.parametrize("bits", [4, 6, 8])
def test_lora_loads_groupwise_low_bit_adapter_and_applies_dequantized_weights(tmp_path, bits):
    torch.manual_seed(1)
    in_features, rank, out_features = 17, 8, 19
    group_size = 16
    weight_key = "model.layers.0.self_attn.q_proj"
    save_key = f"{HF_ADAPTER_WEIGHT_KEY_PREFIX}{weight_key}"
    lora_a_runtime = torch.randn(in_features, rank, dtype=torch.bfloat16) * 0.03
    lora_b_runtime = torch.randn(rank, out_features, dtype=torch.bfloat16) * 0.03
    lora_a_saved = lora_a_runtime.T.contiguous()
    lora_b_saved = lora_b_runtime.T.contiguous()

    weights = {}
    weights.update(_compressed_entries(f"{save_key}.lora_A.weight", lora_a_saved, group_size, bits))
    weights.update(_compressed_entries(f"{save_key}.lora_B.weight", lora_b_saved, group_size, bits))
    save_file(weights, tmp_path / "adapter_model.safetensors", metadata={"format": "pt"})
    LoraConfig(
        r=rank,
        lora_alpha=rank,
        target_modules=["q_proj"],
        gptqmodel_lora_weight_format=lora_grouped_format_from_bits(bits),
        gptqmodel_lora_weight_bits=bits,
        gptqmodel_lora_group_size=group_size,
        gptqmodel_lora_scale_dtype="bfloat16",
        gptqmodel_lora_dequant_mode="forward",
    ).save_pretrained(str(tmp_path))

    adapter = Lora(rank=rank, path=str(tmp_path))
    adapter.post_init(weight_key=weight_key, device=torch.device("cpu"))

    x = torch.randn(3, in_features, dtype=torch.float32)
    out = torch.randn(3, out_features, dtype=torch.float32)
    q_a, s_a, sh_a = (
        weights[compressed_weight_keys(f"{save_key}.lora_A.weight")[0]],
        weights[compressed_weight_keys(f"{save_key}.lora_A.weight")[1]],
        weights[compressed_weight_keys(f"{save_key}.lora_A.weight")[2]],
    )
    q_b, s_b, sh_b = (
        weights[compressed_weight_keys(f"{save_key}.lora_B.weight")[0]],
        weights[compressed_weight_keys(f"{save_key}.lora_B.weight")[1]],
        weights[compressed_weight_keys(f"{save_key}.lora_B.weight")[2]],
    )
    expected_a = dequantize_tensor_groupwise_int(
        qweight=q_a,
        scales=s_a,
        shape=sh_a,
        bits=bits,
        group_size=group_size,
        device=torch.device("cpu"),
        dtype=torch.float32,
    ).T.contiguous()
    expected_b = dequantize_tensor_groupwise_int(
        qweight=q_b,
        scales=s_b,
        shape=sh_b,
        bits=bits,
        group_size=group_size,
        device=torch.device("cpu"),
        dtype=torch.float32,
    ).T.contiguous()
    expected = out + (x @ expected_a) @ expected_b
    actual = adapter.apply(x=x, out=out.clone())

    assert adapter.lora_A is None
    assert adapter.lora_B is None
    assert adapter.lora_weight_bits == bits
    assert adapter.lora_A_qweight.dtype == (torch.int8 if bits == 8 else torch.uint8)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("bits", [4, 6, 8])
def test_lora_groupwise_low_bit_load_mode_materializes_bf16(tmp_path, monkeypatch, bits):
    torch.manual_seed(2)
    rank = 4
    weight_key = "model.layers.0.mlp.up_proj"
    save_key = f"{HF_ADAPTER_WEIGHT_KEY_PREFIX}{weight_key}"
    weights = {}
    weights.update(_compressed_entries(f"{save_key}.lora_A.weight", torch.randn(rank, 16, dtype=torch.bfloat16), 8, bits))
    weights.update(_compressed_entries(f"{save_key}.lora_B.weight", torch.randn(32, rank, dtype=torch.bfloat16), 8, bits))
    save_file(weights, tmp_path / "adapter_model.safetensors", metadata={"format": "pt"})
    LoraConfig(
        r=rank,
        lora_alpha=rank,
        target_modules=["up_proj"],
        gptqmodel_lora_weight_format=lora_grouped_format_from_bits(bits),
        gptqmodel_lora_weight_bits=bits,
        gptqmodel_lora_group_size=8,
        gptqmodel_lora_dequant_mode="forward",
    ).save_pretrained(str(tmp_path))
    monkeypatch.setenv("GPTQMODEL_LORA_DEQUANT_MODE", "load")

    adapter = Lora(rank=rank, path=str(tmp_path))
    adapter.post_init(weight_key=weight_key, device=torch.device("cpu"))

    assert adapter.lora_A is not None
    assert adapter.lora_B is not None
    assert adapter.lora_A.dtype == torch.bfloat16
    assert adapter.lora_B.dtype == torch.bfloat16
    assert adapter.lora_A_qweight is None
    assert (tmp_path / HF_ADAPTER_CONFIG_FILE_NAME).exists()
