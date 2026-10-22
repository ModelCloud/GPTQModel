# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GGUF format: ggml-org/llama.cpp, MIT, https://github.com/ggml-org/llama.cpp

from __future__ import annotations

import numpy as np
import pytest
import torch

from gptqmodel.nn_modules.qlinear.gguf import GGUFTorchLinear


def _oracle_dequantize(qweight: torch.Tensor, tensor_qtype: str) -> np.ndarray:
    raw = qweight.detach().cpu().numpy()
    rows = raw.shape[0]
    type_size = 18 if tensor_qtype == "Q4_0" else 34
    block_size = 32
    blocks = raw.reshape(rows, -1, type_size)
    output = np.empty((rows, blocks.shape[1] * block_size), dtype=np.float32)

    for row in range(rows):
        for block_index in range(blocks.shape[1]):
            block = blocks[row, block_index]
            scale = np.frombuffer(block[:2].tobytes(), dtype=np.float16)[0].astype(np.float32)
            if tensor_qtype == "Q4_0":
                packed = block[2:]
                values = np.concatenate((packed & 0x0F, packed >> 4)).astype(np.int16) - 8
            else:
                values = np.frombuffer(block[2:].tobytes(), dtype=np.int8)
            start = block_index * block_size
            output[row, start : start + block_size] = scale * values

    return output


def _build_module(
    bits: str,
    *,
    in_features: int,
    out_features: int = 5,
    seed: int = 41,
) -> tuple[torch.nn.Linear, GGUFTorchLinear]:
    torch.manual_seed(seed)
    linear = torch.nn.Linear(in_features, out_features, bias=True, dtype=torch.float32)
    with torch.no_grad():
        linear.weight.normal_(mean=0.0, std=0.03)
        linear.bias.normal_(mean=0.0, std=0.01)

    module = GGUFTorchLinear(
        bits=bits,
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=True,
        register_buffers=True,
    ).eval()
    module.pack_original(linear, scales=None, zeros=None)
    return linear, module


@pytest.mark.parametrize(
    ("bits", "tensor_qtype", "in_features"),
    [
        ("q4_0", "Q4_0", 32),
        ("q4_0", "Q4_0", 37),
        ("q8_0", "Q8_0", 32),
        ("q8_0", "Q8_0", 37),
    ],
)
def test_gguf_torch_dequantize_matches_independent_block_oracle(
    bits: str,
    tensor_qtype: str,
    in_features: int,
):
    _, module = _build_module(bits, in_features=in_features)
    expected = _oracle_dequantize(module.qweight, tensor_qtype)[:, :in_features]
    actual = module.dequantize_weight(dtype=torch.float32).T.detach().cpu().numpy()

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)
    assert module.padded_in_features % module.gguf_block_size == 0
    if module.padded_in_features > in_features:
        np.testing.assert_array_equal(
            _oracle_dequantize(module.qweight, tensor_qtype)[:, in_features:],
            np.zeros((module.out_features, module.padded_in_features - in_features), dtype=np.float32),
        )


def test_gguf_q4_0_uses_fp16_scale_and_low_nibble_first_layout():
    module = GGUFTorchLinear(
        bits="q4_0",
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=32,
        out_features=1,
        bias=False,
        register_buffers=True,
    ).eval()
    scale_bytes = np.array([1.5], dtype=np.float16).view(np.uint8)
    qweight = np.zeros((1, 18), dtype=np.uint8)
    qweight[0, :2] = scale_bytes
    qweight[0, 2:] = 0x10
    module.qweight.copy_(torch.from_numpy(qweight))

    actual = module.dequantize_weight(dtype=torch.float32).reshape(-1)
    expected = torch.cat(
        (
            torch.full((16,), -12.0, dtype=torch.float32),
            torch.full((16,), -10.5, dtype=torch.float32),
        )
    )

    assert module.qweight[0, :2].numpy().tobytes() == np.float16(1.5).tobytes()
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_gguf_q8_0_uses_fp16_scale_and_signed_int8_layout():
    module = GGUFTorchLinear(
        bits="q8_0",
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=32,
        out_features=1,
        bias=False,
        register_buffers=True,
    ).eval()
    values = np.arange(-16, 16, dtype=np.int8)
    scale_bytes = np.array([0.25], dtype=np.float16).view(np.uint8)
    qweight = np.zeros((1, 34), dtype=np.uint8)
    qweight[0, :2] = scale_bytes
    qweight[0, 2:] = values.view(np.uint8)
    module.qweight.copy_(torch.from_numpy(qweight))

    actual = module.dequantize_weight(dtype=torch.float32).reshape(-1)
    expected = torch.from_numpy(values.astype(np.float32) * np.float32(0.25))

    assert module.qweight[0, :2].numpy().tobytes() == np.float16(0.25).tobytes()
    torch.testing.assert_close(module.qweight[0, 2:].view(torch.int8), torch.from_numpy(values), rtol=0, atol=0)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize(("bits", "tensor_qtype"), [("q4_0", "Q4_0"), ("q8_0", "Q8_0")])
def test_gguf_q4_q8_forward_matches_dense_oracle_and_fused_path(
    dtype: torch.dtype,
    bits: str,
    tensor_qtype: str,
):
    linear, module = _build_module(bits, in_features=37, out_features=5, seed=73)
    module.gguf_fused_cpu_max_rows = 16
    module.gguf_fused_cpu_min_matrix_elements = 0
    module.autotune_enabled = False
    module.clear_autotune()
    module.clear_weight_cache()

    torch.manual_seed(91)
    x = torch.randn(7, linear.in_features, dtype=dtype)
    expected_weight = torch.from_numpy(_oracle_dequantize(module.qweight, tensor_qtype)[:, : linear.in_features])
    expected_linear = torch.matmul(x.to(torch.float32), expected_weight.T)
    expected = expected_linear + module.bias.to(torch.float32)

    with torch.inference_mode():
        dense = module._forward_dequant_matmul(x)
        fused = module._forward_fused_k(x)
        actual = module(x)

    tolerance = {"rtol": 2e-3, "atol": 2e-3} if dtype == torch.bfloat16 else {"rtol": 1e-5, "atol": 1e-5}
    torch.testing.assert_close(dense.to(torch.float32), expected_linear, **tolerance)
    torch.testing.assert_close(fused.to(torch.float32), dense.to(torch.float32), **tolerance)
    torch.testing.assert_close(actual.to(torch.float32), expected, **tolerance)
    assert module._should_use_fused_k_forward(x)


def test_gguf_fused_autotune_is_cached_per_shape(monkeypatch):
    _, module = _build_module("q4_0", in_features=37, out_features=5)
    module.gguf_fused_cpu_max_rows = 16
    module.gguf_fused_cpu_min_matrix_elements = 0
    module.autotune_enabled = True
    module.clear_autotune()

    calls = []

    def fake_autotune(self, x_flat):
        calls.append(tuple(x_flat.shape))
        return x_flat.shape[0] > 1

    monkeypatch.setattr(type(module), "_autotune", fake_autotune)

    single_token = torch.randn(1, module.in_features, dtype=torch.bfloat16)
    many_tokens = torch.randn(7, module.in_features, dtype=torch.bfloat16)

    assert module._should_use_fused_k_forward(single_token) is False
    assert module._should_use_fused_k_forward(many_tokens) is True
    assert module._should_use_fused_k_forward(single_token) is False
    assert module._should_use_fused_k_forward(many_tokens) is True
    assert calls == [(1, module.in_features), (7, module.in_features)]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("bits", ["q4_0", "q8_0", "q4_k", "q5_k", "q6_k"])
def test_gguf_cpu_dense_weight_cache_matches_uncached(dtype: torch.dtype, bits: str):
    linear, module = _build_module(bits, in_features=256, out_features=32, seed=117)
    x = torch.randn(7, linear.in_features, dtype=dtype)

    with torch.inference_mode():
        expected = module.dequantize_weight(device="cpu", dtype=dtype)
        first = module._dense_weight_for_inference(x)
        second = module._dense_weight_for_inference(x)
        actual = module(x)

    assert first is second
    torch.testing.assert_close(first, expected, rtol=0.0, atol=0.0)
    assert actual.dtype == dtype

    module.clear_weight_cache()
    with torch.inference_mode():
        refreshed = module._dense_weight_for_inference(x)
    assert refreshed is not first
    torch.testing.assert_close(refreshed, expected, rtol=0.0, atol=0.0)


def test_gguf_cpu_dense_weight_cache_can_be_disabled(monkeypatch):
    _, module = _build_module("q4_0", in_features=128, out_features=16, seed=211)
    x = torch.randn(7, module.in_features, dtype=torch.bfloat16)
    monkeypatch.setattr("gptqmodel.nn_modules.qlinear.gguf._GGUF_CACHE_WEIGHTS", False)

    with torch.inference_mode():
        first = module._dense_weight_for_inference(x)
        second = module._dense_weight_for_inference(x)

    assert first is not second
