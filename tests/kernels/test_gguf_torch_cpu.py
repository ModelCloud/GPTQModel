# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GGUF format: ggml-org/llama.cpp, MIT, https://github.com/ggml-org/llama.cpp

from __future__ import annotations

import numpy as np
import pytest
import torch

from gptqmodel.nn_modules.qlinear.gguf import (
    GGUFTorchLinear,
    _dequantize_gguf_tensor_numpy,
)


def _oracle_dequantize(qweight: torch.Tensor, tensor_qtype: str) -> np.ndarray:
    raw = qweight.detach().cpu().numpy()
    if tensor_qtype in {"Q4_K", "Q5_K", "Q6_K"}:
        return _dequantize_gguf_tensor_numpy(raw, tensor_qtype)

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


@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.bfloat16, torch.float16],
    ids=["fp32", "bf16", "fp16"],
)
@pytest.mark.parametrize(
    ("bits", "tensor_qtype"),
    [
        ("q4_0", "Q4_0"),
        ("q8_0", "Q8_0"),
        ("q4_k", "Q4_K"),
        ("q5_k", "Q5_K"),
        ("q6_k", "Q6_K"),
    ],
)
def test_gguf_cpu_fused_forward_matches_independent_oracle(
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
        fused = module._forward_fused_k(x, bias=module.bias)
        actual = module(x)

    raw_error = (fused.float() - expected).norm() / expected.norm().clamp_min(1e-12)
    visible_error = (fused.float() - expected.to(dtype).float()).norm() / expected.to(dtype).float().norm().clamp_min(1e-12)
    assert raw_error < 2e-3
    assert visible_error < 2e-3
    assert fused.dtype == dtype
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, expected.to(dtype), rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(actual, fused, rtol=0.0, atol=0.0)
    assert module._should_use_fused_k_forward(x)


def test_gguf_cpu_fused_forward_retains_only_packed_weight():
    _, module = _build_module("q4_k", in_features=37, out_features=5, seed=74)
    module.gguf_fused_cpu_max_rows = 16
    module.gguf_fused_cpu_min_matrix_elements = 0
    module.autotune_enabled = False
    module.clear_autotune()
    module.clear_weight_cache()

    x = torch.randn(7, module.in_features, dtype=torch.bfloat16)
    with torch.inference_mode():
        first = module(x)
        second = module(x)

    assert torch.equal(first, second)
    assert module._should_use_fused_k_forward(x)
    assert set(dict(module.named_buffers())) == {"qweight", "bias"}
    dense_shape = (module.out_features, module.in_features)
    assert not any(
        tuple(value.shape) == dense_shape
        for value in vars(module).values()
        if torch.is_tensor(value)
    )
    assert not any(
        name in vars(module)
        for name in ("_dequantized_weight", "_dense_weight", "_weight_cache")
    )


def test_gguf_fused_decision_is_a_pure_function_of_the_static_gates():
    """The fused/dense choice must not depend on timing, load or call order (#4590)."""

    _, module = _build_module("q4_k_m", in_features=256, out_features=256, seed=17)
    module.gguf_fused_cpu_max_rows = 32
    module.gguf_fused_cpu_min_matrix_elements = 0

    def timing_forbidden(*args, **kwargs):
        raise AssertionError("the fused/dense choice must not time either path")

    original_runner = getattr(module, "_benchmark_forward_runner", None)
    original_autotune = getattr(module, "_autotune", None)
    if original_runner is not None:
        module._benchmark_forward_runner = timing_forbidden
    if original_autotune is not None:
        module._autotune = timing_forbidden
    try:
        torch.manual_seed(3)
        small = torch.randn(7, module.in_features, dtype=torch.bfloat16)
        large = torch.randn(33, module.in_features, dtype=torch.bfloat16)
        decisions = [
            [module._should_use_fused_k_forward(small), module._should_use_fused_k_forward(large)]
            for _ in range(5)
        ]
    finally:
        if original_runner is not None:
            module._benchmark_forward_runner = original_runner
        if original_autotune is not None:
            module._autotune = original_autotune

    # Same shape -> same answer on every call; the row ceiling still gates the
    # larger batch out of the fused path.
    assert decisions == [[True, False]] * 5
    assert module.get_autotune_result() is None


def test_gguf_fused_path_ignores_the_autotune_flag():
    """`autotune_enabled` no longer selects the path, so it cannot flip the numerics."""

    _, module = _build_module("q4_k_m", in_features=256, out_features=256, seed=19)
    module.gguf_fused_cpu_max_rows = 32
    module.gguf_fused_cpu_min_matrix_elements = 0
    torch.manual_seed(23)
    x = torch.randn(7, module.in_features, dtype=torch.bfloat16)

    with torch.inference_mode():
        module.autotune_enabled = True
        enabled = module(x)
        module.autotune_enabled = False
        disabled = module(x)

    assert torch.equal(enabled, disabled)


@pytest.mark.parametrize("bits", ["q4_0", "q4_k", "q8_0"])
def test_gguf_forward_uses_dense_when_the_row_gate_excludes_fused(bits: str):
    linear, module = _build_module(bits, in_features=64, out_features=8)
    module.gguf_fused_cpu_max_rows = 4
    module.gguf_fused_cpu_min_matrix_elements = 0
    torch.manual_seed(29)
    x = torch.randn(7, linear.in_features, dtype=torch.bfloat16)
    calls = []

    original_dense = module._forward_dequant_matmul

    def _dense(x_flat, bias=None):
        calls.append(("dense", bias is not None))
        return original_dense(x_flat, bias=bias)

    def _fused(*args, **kwargs):
        raise AssertionError("gate-excluded fused path must not run")

    module._forward_dequant_matmul = _dense
    module._forward_fused_k = _fused
    try:
        with torch.inference_mode():
            output = module(x)
    finally:
        module._forward_dequant_matmul = original_dense

    assert calls == [("dense", True)]
    assert output.shape == (x.shape[0], module.out_features)


def test_gguf_static_gates_match_the_measured_crossover():
    """Pin the shipped CPU/CUDA gate defaults against silent regeneration drift."""

    from gptqmodel.utils.kernel import KernelConfig

    defaults = KernelConfig()
    assert defaults.gguf_fused_cpu_max_rows == 64
    assert defaults.gguf_fused_cpu_min_matrix_elements == 0
    assert defaults.gguf_fused_cuda_max_rows == 32
    assert defaults.gguf_fused_cuda_min_matrix_elements == 8_388_608
    # `GGUFTorchLinear` must not re-enable the timing race.
    assert GGUFTorchLinear.AUTOTUNE is False
