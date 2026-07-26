# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gc
import math
import threading
import time

import pytest
import torch

from gptqmodel.nn_modules.qlinear.amplin import AmplinLinear
from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.utils import amplin
from gptqmodel.utils.backend import BACKEND


BITS = 4
GROUP_SIZE = 128
SIZE_K = 4096
SIZE_N = 64
ZERO = 8

QWEN3_8B_SHAPES = (
    (4096, 1024),
    (4096, 4096),
    (4096, 12288),
    (12288, 4096),
)
LAGUNA_S_2_1_SHAPES = (
    (1024, 3072),
    (3072, 48),
    (3072, 72),
    (3072, 1024),
    (3072, 6144),
    (3072, 9216),
    (3072, 12288),
    (6144, 3072),
    (9216, 3072),
    (12288, 3072),
)
REAL_MODEL_SHAPES = tuple(
    pytest.param(size_k, size_n, id=f"k{size_k}-n{size_n}")
    for size_k, size_n in QWEN3_8B_SHAPES + LAGUNA_S_2_1_SHAPES
)
BATCHED_ROWS = (1, 2, 4, 8, 16)
LARGE_MLP_SHAPES = (
    pytest.param(4096, 12288, id="qwen-mlp-up"),
    pytest.param(12288, 4096, id="qwen-mlp-down"),
    pytest.param(3072, 12288, id="laguna-dense-up"),
    pytest.param(12288, 3072, id="laguna-dense-down"),
)
MULTIROW_BATCHED_ROWS = (2, 4, 8, 16)
HMMA_BATCHED_SHAPES = (
    pytest.param(16, 4096, 12288, id="qwen-mlp-up-m16"),
    pytest.param(32, 4096, 12288, id="qwen-mlp-up-m32"),
    pytest.param(64, 4096, 12288, id="qwen-mlp-up-m64"),
    pytest.param(256, 4096, 12288, id="qwen-mlp-up-m256"),
    pytest.param(16, 1024, 3072, id="laguna-expert-down-m16"),
    pytest.param(32, 1024, 3072, id="laguna-expert-down-m32"),
    pytest.param(64, 1024, 3072, id="laguna-expert-down-m64"),
    pytest.param(256, 1024, 3072, id="laguna-expert-down-m256"),
)
MMA_LANE_N32_TAIL_SHAPES = tuple(
    pytest.param(size_m, size_n, id=f"laguna-m{size_m}-n{size_n}")
    for size_n in (48, 72)
    for size_m in (32, 64, 256)
)


def _unpack_gptq_w4(qweight: torch.Tensor, size_k: int) -> torch.Tensor:
    packed_unsigned = qweight.to(torch.int64) & 0xFFFFFFFF
    shifts = torch.arange(0, 32, BITS, device=qweight.device, dtype=torch.int64).view(1, -1, 1)
    return ((packed_unsigned.unsqueeze(1) >> shifts) & 0xF).reshape(size_k, qweight.size(1))


@pytest.fixture(scope="module")
def packed_case() -> dict[str, torch.Tensor]:
    rows = torch.arange(SIZE_K, dtype=torch.int64).view(-1, 1)
    columns = torch.arange(SIZE_N, dtype=torch.int64).view(1, -1)
    codes = ((rows * 5 + columns * 3 + rows // 17) & 0xF).to(torch.int32)
    group_indices = torch.arange(SIZE_K, dtype=torch.int32) // GROUP_SIZE
    scale_steps = (
        torch.arange(SIZE_K // GROUP_SIZE, dtype=torch.int32).view(-1, 1) * 3
        + torch.arange(SIZE_N, dtype=torch.int32).view(1, -1)
    ) % 8 + 1
    scales = scale_steps.to(torch.float32) / 1024.0
    zeros = torch.full_like(scales, ZERO, dtype=torch.int32)
    dense_weight = scales[group_indices.long()] * (codes.to(torch.float32) - ZERO)

    linear = torch.nn.Linear(SIZE_K, SIZE_N, bias=False, dtype=torch.float32)
    with torch.no_grad():
        linear.weight.copy_(dense_weight.t())

    module = TorchLinear(
        bits=BITS,
        group_size=GROUP_SIZE,
        sym=True,
        desc_act=False,
        in_features=SIZE_K,
        out_features=SIZE_N,
        bias=False,
        pack_dtype=torch.int32,
        register_buffers=False,
    )
    module.pack_original(
        linear=linear,
        scales=scales.t().contiguous(),
        zeros=zeros.t().contiguous(),
        g_idx=group_indices,
    )
    module.post_init()

    unpacked = _unpack_gptq_w4(module.qweight, SIZE_K)
    torch.testing.assert_close(unpacked, codes.to(torch.int64), rtol=0, atol=0)
    torch.testing.assert_close(module.scales, scales.to(torch.float16), rtol=0, atol=0)
    torch.testing.assert_close(module.dequantize_weight(), dense_weight.to(torch.float16), rtol=0, atol=0)
    return {
        "codes": unpacked,
        "qweight": module.qweight,
        "scales": module.scales,
    }


@pytest.fixture(scope="module")
def packed_case_n256() -> dict[str, torch.Tensor]:
    size_n = 512
    rows = torch.arange(SIZE_K, dtype=torch.int64).view(-1, 1)
    columns = torch.arange(size_n, dtype=torch.int64).view(1, -1)
    codes = ((rows * 5 + columns * 3 + rows // 17) & 0xF).to(torch.int32)
    group_indices = torch.arange(SIZE_K, dtype=torch.int32) // GROUP_SIZE
    scale_steps = (
        torch.arange(SIZE_K // GROUP_SIZE, dtype=torch.int32).view(-1, 1) * 3
        + torch.arange(size_n, dtype=torch.int32).view(1, -1)
    ) % 8 + 1
    scales = scale_steps.to(torch.float32) / 1024.0
    zeros = torch.full_like(scales, ZERO, dtype=torch.int32)
    dense_weight = scales[group_indices.long()] * (codes.to(torch.float32) - ZERO)

    linear = torch.nn.Linear(SIZE_K, size_n, bias=False, dtype=torch.float32)
    with torch.no_grad():
        linear.weight.copy_(dense_weight.t())

    module = TorchLinear(
        bits=BITS,
        group_size=GROUP_SIZE,
        sym=True,
        desc_act=False,
        in_features=SIZE_K,
        out_features=size_n,
        bias=False,
        pack_dtype=torch.int32,
        register_buffers=False,
    )
    module.pack_original(
        linear=linear,
        scales=scales.t().contiguous(),
        zeros=zeros.t().contiguous(),
        g_idx=group_indices,
    )
    module.post_init()

    unpacked = _unpack_gptq_w4(module.qweight, SIZE_K)
    torch.testing.assert_close(unpacked, codes.to(torch.int64), rtol=0, atol=0)
    torch.testing.assert_close(module.scales, scales.to(torch.float16), rtol=0, atol=0)
    torch.testing.assert_close(module.dequantize_weight(), dense_weight.to(torch.float16), rtol=0, atol=0)
    return {
        "codes": unpacked,
        "qweight": module.qweight,
        "scales": module.scales,
    }


@pytest.fixture(scope="module")
def sm80_device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("Amplin kernel tests require CUDA")
    device = torch.device("cuda", torch.cuda.current_device())
    capability = torch.cuda.get_device_capability(device)
    if capability != (8, 0):
        pytest.skip(f"Amplin requires compute capability 8.0, got {capability[0]}.{capability[1]}")
    if not amplin.amplin_runtime_available():
        pytest.fail(amplin.amplin_runtime_error())
    return device


def _device_case(
    packed_case: dict[str, torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(20260724)
    input = torch.randn((1, SIZE_K), device=device, dtype=torch.float32, generator=generator).mul_(0.25).to(dtype)
    qweight = packed_case["qweight"].to(device=device).contiguous()
    scales = packed_case["scales"].to(device=device, dtype=dtype).contiguous()
    codes = packed_case["codes"].to(device=device)
    group_indices = torch.arange(SIZE_K, device=device, dtype=torch.int64) // GROUP_SIZE
    dense_weight = (codes.to(torch.float32) - ZERO) * scales.to(torch.float32)[group_indices]
    expected = input.to(torch.float32) @ dense_weight
    return input, qweight, scales, expected


def test_amplin_real_gptq_packer_round_trip(packed_case: dict[str, torch.Tensor]):
    assert packed_case["qweight"].shape == (SIZE_K // 8, SIZE_N)
    assert packed_case["qweight"].dtype == torch.int32
    assert packed_case["scales"].shape == (SIZE_K // GROUP_SIZE, SIZE_N)
    assert packed_case["scales"].dtype == torch.float16


@pytest.mark.cuda
@pytest.mark.parametrize(("size_k", "size_n"), REAL_MODEL_SHAPES)
def test_amplin_hmma_execution_layout_round_trip_real_model_shapes(
    sm80_device: torch.device,
    size_k: int,
    size_n: int,
):
    generator = torch.Generator(device=sm80_device)
    generator.manual_seed(20260724 + size_k * 7 + size_n)
    qweight = torch.randint(
        -(2**31),
        2**31 - 1,
        (size_k // 8, size_n),
        device=sm80_device,
        dtype=torch.int32,
        generator=generator,
    )
    scales = torch.rand(
        (size_k // GROUP_SIZE, size_n),
        device=sm80_device,
        dtype=torch.float32,
        generator=generator,
    ).to(torch.float16)

    packed_qweight, packed_scales = amplin.pack_hmma_weights(qweight, scales)
    packed_mma_lane_qweight = amplin.pack_mma_lane_qweight(qweight)
    packed_mma_lane_n32_qweight = amplin.pack_mma_lane_n32_qweight(qweight)
    packed_mma_lane_n64_qweight = amplin.pack_mma_lane_n64_qweight(qweight)
    num_n_tiles = (size_n + amplin.HMMA_N_TILE - 1) // amplin.HMMA_N_TILE
    num_groups = size_k // amplin.HMMA_K_TILE
    assert packed_qweight.shape == (
        num_n_tiles,
        num_groups,
        amplin.HMMA_N_TILE,
        amplin.HMMA_PACKED_K_WORDS,
    )
    assert packed_scales.shape == (num_n_tiles, num_groups, amplin.HMMA_N_TILE)
    assert packed_mma_lane_qweight.shape == (
        num_n_tiles,
        num_groups,
        amplin.MMA_LANE_K_STEPS,
        amplin.MMA_LANE_N_WARPS,
        amplin.MMA_LANES,
    )
    assert packed_mma_lane_n32_qweight.shape == (
        num_n_tiles * 2,
        num_groups,
        amplin.MMA_LANE_K_STEPS,
        amplin.MMA_LANES,
        2,
    )
    assert packed_mma_lane_n64_qweight.shape == (
        num_n_tiles,
        num_groups,
        amplin.MMA_LANE_K_STEPS,
        amplin.MMA_LANES,
        amplin.MMA_LANE_N_WARPS,
    )
    assert packed_qweight.is_contiguous()
    assert packed_scales.is_contiguous()
    assert packed_mma_lane_qweight.is_contiguous()
    assert packed_mma_lane_n32_qweight.is_contiguous()
    assert packed_mma_lane_n64_qweight.is_contiguous()
    assert packed_qweight.dtype == torch.int32
    assert packed_scales.dtype == torch.float16
    assert packed_mma_lane_qweight.dtype == torch.int32
    assert packed_mma_lane_n32_qweight.dtype == torch.int32
    assert packed_mma_lane_n64_qweight.dtype == torch.int32

    restored_qweight = amplin.unpack_hmma_qweight(packed_qweight, size_n=size_n)
    restored_scales = amplin.unpack_hmma_scales(packed_scales, size_n=size_n)
    restored_mma_lane_qweight = amplin.unpack_mma_lane_qweight(
        packed_mma_lane_qweight,
        size_n=size_n,
    )
    restored_mma_lane_n32_qweight = amplin.unpack_mma_lane_n32_qweight(
        packed_mma_lane_n32_qweight,
        size_n=size_n,
    )
    restored_mma_lane_n64_qweight = amplin.unpack_mma_lane_n64_qweight(
        packed_mma_lane_n64_qweight,
        size_n=size_n,
    )
    torch.testing.assert_close(restored_qweight, qweight, rtol=0, atol=0)
    torch.testing.assert_close(restored_scales, scales, rtol=0, atol=0)
    torch.testing.assert_close(restored_mma_lane_qweight, qweight, rtol=0, atol=0)
    torch.testing.assert_close(restored_mma_lane_n32_qweight, qweight, rtol=0, atol=0)
    torch.testing.assert_close(restored_mma_lane_n64_qweight, qweight, rtol=0, atol=0)
    if size_n % amplin.HMMA_N_TILE == 0:
        assert packed_mma_lane_qweight.nbytes == qweight.nbytes
        assert packed_mma_lane_n32_qweight.nbytes == qweight.nbytes
        assert packed_mma_lane_n64_qweight.nbytes == qweight.nbytes

    for group, packed_k, column in (
        (0, 0, 0),
        (num_groups - 1, amplin.HMMA_PACKED_K_WORDS - 1, size_n - 1),
    ):
        tile_n, tile_column = divmod(column, amplin.HMMA_N_TILE)
        assert packed_qweight[tile_n, group, tile_column, packed_k] == qweight[
            group * amplin.HMMA_PACKED_K_WORDS + packed_k,
            column,
        ]

    padded_n = num_n_tiles * amplin.HMMA_N_TILE
    if padded_n != size_n:
        padded_qweight = amplin.unpack_hmma_qweight(packed_qweight, size_n=padded_n)
        padded_scales = amplin.unpack_hmma_scales(packed_scales, size_n=padded_n)
        expected_zero_word = torch.tensor(-2004318072, device=sm80_device, dtype=torch.int32)
        assert torch.all(padded_qweight[:, size_n:] == expected_zero_word)
        assert torch.count_nonzero(padded_scales[:, size_n:]) == 0


def test_amplin_hmma_execution_layout_rejects_invalid_tensors():
    qweight = torch.zeros((GROUP_SIZE // 8, SIZE_N), dtype=torch.int32)
    scales = torch.ones((1, SIZE_N), dtype=torch.float16)

    with pytest.raises(ValueError, match="must use torch.int32"):
        amplin.pack_hmma_qweight(qweight.to(torch.int64))
    with pytest.raises(ValueError, match="two-dimensional"):
        amplin.pack_hmma_qweight(qweight.unsqueeze(0))
    with pytest.raises(ValueError, match="K must be positive and divisible by 128"):
        amplin.pack_hmma_qweight(qweight[:-1])
    with pytest.raises(ValueError, match="N must be positive"):
        amplin.pack_hmma_qweight(qweight[:, :0])
    with pytest.raises(ValueError, match=r"canonical shape \[K/128, N\]"):
        amplin.pack_hmma_weights(qweight, scales[:, :-1])
    with pytest.raises(ValueError, match="floating-point"):
        amplin.pack_hmma_scales(scales.to(torch.int32))
    with pytest.raises(ValueError, match="no larger than the packed N"):
        amplin.unpack_hmma_qweight(amplin.pack_hmma_qweight(qweight), size_n=SIZE_N + 1)
    with pytest.raises(ValueError, match="no larger than the packed N"):
        amplin.unpack_hmma_scales(amplin.pack_hmma_scales(scales), size_n=SIZE_N + 1)
    with pytest.raises(ValueError, match="no larger than the packed N"):
        amplin.unpack_mma_lane_qweight(amplin.pack_mma_lane_qweight(qweight), size_n=SIZE_N + 1)
    with pytest.raises(ValueError, match="no larger than the packed N"):
        amplin.unpack_mma_lane_n32_qweight(amplin.pack_mma_lane_n32_qweight(qweight), size_n=SIZE_N + 1)
    with pytest.raises(ValueError, match=r"shape \[N32, K128, K16, lane, 2\]"):
        amplin.unpack_mma_lane_n32_qweight(torch.zeros((2, 1, 8, 32), dtype=torch.int32), size_n=SIZE_N)
    with pytest.raises(ValueError, match="no larger than the packed N"):
        amplin.unpack_mma_lane_n64_qweight(amplin.pack_mma_lane_n64_qweight(qweight), size_n=SIZE_N + 1)
    with pytest.raises(ValueError, match=r"shape \[N64, K128, K16, lane, 4\]"):
        amplin.unpack_mma_lane_n64_qweight(torch.zeros((1, 1, 8, 32), dtype=torch.int32), size_n=SIZE_N)


def test_amplin_mma_lane_layout_matches_ampere_b_fragment_ownership(
    packed_case: dict[str, torch.Tensor],
):
    packed = amplin.pack_mma_lane_qweight(packed_case["qweight"])
    interleaved = amplin.pack_mma_lane_n32_qweight(packed_case["qweight"])
    interleaved_n64 = amplin.pack_mma_lane_n64_qweight(packed_case["qweight"])
    assert packed.shape == (
        1,
        SIZE_K // amplin.HMMA_K_TILE,
        amplin.MMA_LANE_K_STEPS,
        amplin.MMA_LANE_N_WARPS,
        amplin.MMA_LANES,
    )
    assert packed.dtype == torch.int32
    assert packed.is_contiguous()
    assert interleaved.shape == (
        2,
        SIZE_K // amplin.HMMA_K_TILE,
        amplin.MMA_LANE_K_STEPS,
        amplin.MMA_LANES,
        2,
    )
    assert interleaved.is_contiguous()
    assert interleaved_n64.shape == (
        1,
        SIZE_K // amplin.HMMA_K_TILE,
        amplin.MMA_LANE_K_STEPS,
        amplin.MMA_LANES,
        amplin.MMA_LANE_N_WARPS,
    )
    assert interleaved_n64.is_contiguous()
    codes = packed_case["codes"]

    for k_step, n_warp, lane in ((0, 0, 0), (3, 2, 17), (7, 3, 31)):
        quad, thread_in_quad = divmod(lane, 4)
        base_k = k_step * 16
        base_n = n_warp * 16
        expected_codes = (
            codes[base_k + thread_in_quad * 2, base_n + quad],
            codes[base_k + 8 + thread_in_quad * 2, base_n + quad],
            codes[base_k + thread_in_quad * 2, base_n + quad + 8],
            codes[base_k + 8 + thread_in_quad * 2, base_n + quad + 8],
            codes[base_k + thread_in_quad * 2 + 1, base_n + quad],
            codes[base_k + 8 + thread_in_quad * 2 + 1, base_n + quad],
            codes[base_k + thread_in_quad * 2 + 1, base_n + quad + 8],
            codes[base_k + 8 + thread_in_quad * 2 + 1, base_n + quad + 8],
        )
        packed_unsigned = int(packed[0, 0, k_step, n_warp, lane].item()) & 0xFFFFFFFF
        actual_codes = tuple((packed_unsigned >> (index * 4)) & 0xF for index in range(8))
        assert actual_codes == tuple(int(code.item()) for code in expected_codes)

    for n32_tile, k_step, lane in ((0, 0, 0), (0, 3, 17), (1, 7, 31)):
        base_n_warp = n32_tile * 2
        assert interleaved[n32_tile, 0, k_step, lane, 0] == packed[0, 0, k_step, base_n_warp, lane]
        assert interleaved[n32_tile, 0, k_step, lane, 1] == packed[0, 0, k_step, base_n_warp + 1, lane]

    for k_step, lane in ((0, 0), (3, 17), (7, 31)):
        for n16_tile in range(amplin.MMA_LANE_N_WARPS):
            assert interleaved_n64[0, 0, k_step, lane, n16_tile] == packed[0, 0, k_step, n16_tile, lane]


@pytest.mark.cuda
@pytest.mark.parametrize(
    ("dtype", "atol"),
    [
        (torch.float16, 2e-3),
        (torch.bfloat16, 2e-2),
    ],
)
def test_amplin_mma_lane_tile_matches_fp32_reference(
    packed_case: dict[str, torch.Tensor],
    sm80_device: torch.device,
    dtype: torch.dtype,
    atol: float,
):
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("selected CUDA device does not support BF16")
    generator = torch.Generator(device=sm80_device)
    generator.manual_seed(20260724)
    input = torch.randn(
        (16, 16),
        device=sm80_device,
        dtype=torch.float32,
        generator=generator,
    ).mul_(0.25).to(dtype)
    packed = amplin.pack_mma_lane_qweight(packed_case["qweight"])
    packed_tile = packed[0, 0, 0, 0].to(device=sm80_device)
    scales = packed_case["scales"][0, :16].to(device=sm80_device, dtype=dtype)
    dense_weight = (
        packed_case["codes"][:16, :16].to(device=sm80_device, dtype=torch.float32) - ZERO
    ) * scales.to(torch.float32)
    expected = input.to(torch.float32) @ dense_weight

    actual = amplin.mma_lane_tile(input, packed_tile, scales)
    global_a = amplin.mma_lane_tile_global_a(input, packed_tile, scales)
    repeated = amplin.mma_lane_tile(input, packed_tile, scales)
    global_a_repeated = amplin.mma_lane_tile_global_a(input, packed_tile, scales)
    torch.cuda.synchronize(sm80_device)

    assert actual.shape == (16, 16)
    assert actual.dtype == dtype
    assert actual.device == sm80_device
    assert global_a.shape == actual.shape
    assert global_a.dtype == actual.dtype
    assert global_a.device == actual.device
    torch.testing.assert_close(actual.to(torch.float32), expected, rtol=0, atol=atol)
    torch.testing.assert_close(global_a, actual, rtol=0, atol=0)
    torch.testing.assert_close(repeated, actual, rtol=0, atol=0)
    torch.testing.assert_close(global_a_repeated, global_a, rtol=0, atol=0)


@pytest.mark.cuda
def test_amplin_mma_lane_tile_rejects_inputs_outside_contract(
    sm80_device: torch.device,
):
    input = torch.zeros((16, 16), device=sm80_device, dtype=torch.float16)
    packed_qweight = torch.zeros((32,), device=sm80_device, dtype=torch.int32)
    scales = torch.ones((16,), device=sm80_device, dtype=torch.float16)

    with pytest.raises(RuntimeError, match=r"input must have shape \[16, 16\]"):
        amplin.mma_lane_tile(input[:-1], packed_qweight, scales)
    with pytest.raises(RuntimeError, match=r"input must have shape \[16, 16\]"):
        amplin.mma_lane_tile_global_a(input[:-1], packed_qweight, scales)
    with pytest.raises(RuntimeError, match=r"qweight must have shape \[32\]"):
        amplin.mma_lane_tile(input, packed_qweight[:-1], scales)
    with pytest.raises(RuntimeError, match=r"scales must have shape \[16\]"):
        amplin.mma_lane_tile(input, packed_qweight, scales[:-1])
    with pytest.raises(RuntimeError, match="scales dtype must match"):
        amplin.mma_lane_tile(input, packed_qweight, scales.to(torch.bfloat16))


@pytest.mark.cuda
@pytest.mark.parametrize(("size_m", "size_k", "size_n"), HMMA_BATCHED_SHAPES)
def test_amplin_hmma_gemm_matches_fp32_dequant_reference(
    sm80_device: torch.device,
    size_m: int,
    size_k: int,
    size_n: int,
):
    generator = torch.Generator(device=sm80_device)
    generator.manual_seed(20260724 + size_m * 11 + size_k * 3 + size_n)
    qweight = torch.randint(
        -(2**31),
        2**31 - 1,
        (size_k // 8, size_n),
        device=sm80_device,
        dtype=torch.int32,
        generator=generator,
    )
    scale_values = (
        torch.rand(
            (size_k // GROUP_SIZE, size_n),
            device=sm80_device,
            dtype=torch.float32,
            generator=generator,
        )
        * 0.007
        + 0.001
    )
    codes = _unpack_gptq_w4(qweight, size_k)
    group_indices = torch.arange(size_k, device=sm80_device, dtype=torch.int64) // GROUP_SIZE
    packed_qweight = amplin.pack_hmma_qweight(qweight)
    packed_mma_lane_qweight = amplin.pack_mma_lane_qweight(qweight)

    for dtype, atol in ((torch.float16, 2e-3), (torch.bfloat16, 2e-2)):
        if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            continue
        scales = scale_values.to(dtype)
        packed_scales = amplin.pack_hmma_scales(scales)
        input = (
            torch.randn(
                (size_m, size_k),
                device=sm80_device,
                dtype=torch.float32,
                generator=generator,
            )
            .mul_(0.25)
            .to(dtype)
        )
        dense_weight = (codes.to(torch.float32) - ZERO) * scales.to(torch.float32)[group_indices]
        expected = input.to(torch.float32) @ dense_weight
        actual = amplin.gemm_hmma(
            input,
            packed_qweight,
            packed_scales,
            logical_n=size_n,
        )
        v0_control = amplin.gemm_hmma_v0(
            input,
            packed_qweight,
            packed_scales,
            logical_n=size_n,
        )
        m64_v1_control = (
            amplin.gemm_hmma_m64_v1(
                input,
                packed_qweight,
                packed_scales,
                logical_n=size_n,
            )
            if size_m % 64 == 0
            else None
        )
        m64_v2_control = (
            amplin.gemm_hmma_m64_v2(
                input,
                packed_qweight,
                packed_scales,
                logical_n=size_n,
            )
            if size_m % 64 == 0
            else None
        )
        m64_v2_sync_a128_control = (
            amplin.gemm_hmma_m64_v2_sync_a128(
                input,
                packed_qweight,
                packed_scales,
                logical_n=size_n,
            )
            if size_m % 64 == 0
            else None
        )
        m64_v3_control = (
            amplin.gemm_hmma_m64_v3(
                input,
                packed_qweight,
                packed_scales,
                logical_n=size_n,
            )
            if size_m % 64 == 0
            else None
        )
        mma_lane_m64 = (
            amplin.mma_lane_m64(
                input,
                packed_mma_lane_qweight,
                packed_scales,
                logical_n=size_n,
            )
            if size_m % 64 == 0
            else None
        )
        mma_lane_m64_global_a = (
            amplin.mma_lane_m64_global_a(
                input,
                packed_mma_lane_qweight,
                packed_scales,
                logical_n=size_n,
            )
            if size_m % 64 == 0
            else None
        )
        mma_lane_m32_global_a = (
            amplin.mma_lane_m32_global_a(
                input,
                packed_mma_lane_qweight,
                packed_scales,
                logical_n=size_n,
            )
            if size_m % 32 == 0
            else None
        )
        mma_lane_m32_n32_global_a = (
            amplin.mma_lane_m32_n32_global_a(
                input,
                packed_mma_lane_qweight,
                packed_scales,
                logical_n=size_n,
            )
            if size_m % 32 == 0
            else None
        )
        repeated = amplin.gemm_hmma(
            input,
            packed_qweight,
            packed_scales,
            logical_n=size_n,
        )
        torch.cuda.synchronize(sm80_device)

        assert actual.shape == (size_m, size_n)
        assert actual.dtype == dtype
        assert actual.device == sm80_device
        assert torch.isfinite(actual).all()
        schedule_outputs = [("selected", actual), ("v0-control", v0_control)]
        if m64_v1_control is not None:
            schedule_outputs.append(("m64-v1-control", m64_v1_control))
        if m64_v2_control is not None:
            schedule_outputs.append(("m64-v2-control", m64_v2_control))
        if m64_v2_sync_a128_control is not None:
            schedule_outputs.append(("m64-v2-sync-a128-control", m64_v2_sync_a128_control))
        if m64_v3_control is not None:
            schedule_outputs.append(("m64-v3-control", m64_v3_control))
        if mma_lane_m64 is not None:
            schedule_outputs.append(("mma-lane-m64", mma_lane_m64))
        if mma_lane_m64_global_a is not None:
            schedule_outputs.append(("mma-lane-m64-global-a", mma_lane_m64_global_a))
        if mma_lane_m32_global_a is not None:
            schedule_outputs.append(("mma-lane-m32-global-a", mma_lane_m32_global_a))
        if mma_lane_m32_n32_global_a is not None:
            schedule_outputs.append(("mma-lane-m32-n32-global-a", mma_lane_m32_n32_global_a))
        for schedule, output in schedule_outputs:
            torch.testing.assert_close(
                output.to(torch.float32),
                expected,
                rtol=0,
                atol=atol,
                msg=lambda message: (
                    f"Amplin HMMA {schedule} mismatch for M={size_m}, K={size_k}, "
                    f"N={size_n}, dtype={dtype}: {message}"
                ),
            )
        torch.testing.assert_close(repeated, actual, rtol=0, atol=0)


@pytest.mark.cuda
@pytest.mark.parametrize(("size_m", "size_n"), MMA_LANE_N32_TAIL_SHAPES)
def test_amplin_mma_lane_n32_matches_laguna_tail_reference(
    sm80_device: torch.device,
    size_m: int,
    size_n: int,
):
    size_k = 3072
    generator = torch.Generator(device=sm80_device)
    generator.manual_seed(20260724 + size_m * 13 + size_n)
    qweight = torch.randint(
        -(2**31),
        2**31 - 1,
        (size_k // 8, size_n),
        device=sm80_device,
        dtype=torch.int32,
        generator=generator,
    )
    scale_values = (
        torch.rand(
            (size_k // GROUP_SIZE, size_n),
            device=sm80_device,
            dtype=torch.float32,
            generator=generator,
        )
        * 0.007
        + 0.001
    )
    codes = _unpack_gptq_w4(qweight, size_k)
    group_indices = torch.arange(size_k, device=sm80_device, dtype=torch.int64) // GROUP_SIZE
    packed_qweight = amplin.pack_mma_lane_qweight(qweight)

    for dtype, atol in ((torch.float16, 2e-3), (torch.bfloat16, 2e-2)):
        if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            continue
        scales = scale_values.to(dtype)
        packed_scales = amplin.pack_hmma_scales(scales)
        input = (
            torch.randn(
                (size_m, size_k),
                device=sm80_device,
                dtype=torch.float32,
                generator=generator,
            )
            .mul_(0.25)
            .to(dtype)
        )
        dense_weight = (codes.to(torch.float32) - ZERO) * scales.to(torch.float32)[group_indices]
        expected = input.to(torch.float32) @ dense_weight
        actual = amplin.mma_lane_m32_n32_global_a(
            input,
            packed_qweight,
            packed_scales,
            logical_n=size_n,
        )
        repeated = amplin.mma_lane_m32_n32_global_a(
            input,
            packed_qweight,
            packed_scales,
            logical_n=size_n,
        )
        torch.cuda.synchronize(sm80_device)

        assert actual.shape == (size_m, size_n)
        assert actual.dtype == dtype
        assert actual.device == sm80_device
        torch.testing.assert_close(actual.to(torch.float32), expected, rtol=0, atol=atol)
        torch.testing.assert_close(repeated, actual, rtol=0, atol=0)


@pytest.mark.cuda
def test_amplin_hmma_gemm_rejects_inputs_outside_contract(sm80_device: torch.device):
    qweight = torch.zeros((GROUP_SIZE // 8, SIZE_N), device=sm80_device, dtype=torch.int32)
    scales = torch.ones((1, SIZE_N), device=sm80_device, dtype=torch.float16)
    packed_qweight, packed_scales = amplin.pack_hmma_weights(qweight, scales)
    packed_mma_lane_qweight = amplin.pack_mma_lane_qweight(qweight)
    packed_mma_lane_n32_qweight = amplin.pack_mma_lane_n32_qweight(qweight)
    packed_mma_lane_n64_qweight = amplin.pack_mma_lane_n64_qweight(qweight)
    input = torch.ones((16, GROUP_SIZE), device=sm80_device, dtype=torch.float16)

    with pytest.raises(RuntimeError, match="flattened M must be positive and divisible by 16"):
        amplin.gemm_hmma(input[:15].contiguous(), packed_qweight, packed_scales, logical_n=SIZE_N)
    with pytest.raises(RuntimeError, match="logical N must be positive, divisible by 64"):
        amplin.gemm_hmma(input, packed_qweight, packed_scales, logical_n=SIZE_N - 1)
    with pytest.raises(RuntimeError, match=r"qweight must have shape \[N/64, K/128, 64, 16\]"):
        amplin.gemm_hmma(input, packed_qweight[:, :, :, :-1].contiguous(), packed_scales, logical_n=SIZE_N)
    with pytest.raises(RuntimeError, match=r"scales must have shape \[N/64, K/128, 64\]"):
        amplin.gemm_hmma(input, packed_qweight, packed_scales[:, :, :-1].contiguous(), logical_n=SIZE_N)
    with pytest.raises(RuntimeError, match="scales dtype must match"):
        amplin.gemm_hmma(input, packed_qweight, packed_scales.to(torch.bfloat16), logical_n=SIZE_N)
    with pytest.raises(RuntimeError, match="M64 control requires flattened M divisible by 64"):
        amplin.gemm_hmma_m64_v1(input, packed_qweight, packed_scales, logical_n=SIZE_N)
    with pytest.raises(RuntimeError, match="M64 control requires flattened M divisible by 64"):
        amplin.gemm_hmma_m64_v2(input, packed_qweight, packed_scales, logical_n=SIZE_N)
    with pytest.raises(RuntimeError, match="M64 control requires flattened M divisible by 64"):
        amplin.gemm_hmma_m64_v2_sync_a128(input, packed_qweight, packed_scales, logical_n=SIZE_N)
    with pytest.raises(RuntimeError, match="M64 control requires flattened M divisible by 64"):
        amplin.gemm_hmma_m64_v3(input, packed_qweight, packed_scales, logical_n=SIZE_N)
    with pytest.raises(RuntimeError, match="flattened M must be positive and divisible by 64"):
        amplin.mma_lane_m64(
            input,
            packed_mma_lane_qweight,
            packed_scales,
            logical_n=SIZE_N,
        )
    with pytest.raises(RuntimeError, match="flattened M must be positive and divisible by 64"):
        amplin.mma_lane_m64_global_a(
            input,
            packed_mma_lane_qweight,
            packed_scales,
            logical_n=SIZE_N,
        )
    with pytest.raises(RuntimeError, match="flattened M must be positive and divisible by 32"):
        amplin.mma_lane_m32_global_a(
            input,
            packed_mma_lane_qweight,
            packed_scales,
            logical_n=SIZE_N,
        )
    with pytest.raises(RuntimeError, match="flattened M must be positive and divisible by 32"):
        amplin.mma_lane_m32_n32_global_a(
            input,
            packed_mma_lane_qweight,
            packed_scales,
            logical_n=SIZE_N,
        )
    with pytest.raises(RuntimeError, match="N32 logical N must be positive, divisible by 8"):
        amplin.mma_lane_m32_n32_global_a(
            torch.ones((32, GROUP_SIZE), device=sm80_device, dtype=torch.float16),
            packed_mma_lane_qweight,
            packed_scales,
            logical_n=SIZE_N - 1,
        )
    with pytest.raises(RuntimeError, match="flattened M must be between 1 and 16"):
        amplin.mma_lane_m16_n16_padded(
            torch.ones((17, GROUP_SIZE), device=sm80_device, dtype=torch.float16),
            packed_mma_lane_qweight,
            packed_scales,
            logical_n=SIZE_N,
        )
    with pytest.raises(RuntimeError, match="logical N must be positive, divisible by 16"):
        amplin.mma_lane_m16_n16_padded(
            input,
            packed_mma_lane_qweight,
            packed_scales,
            logical_n=SIZE_N - 1,
        )
    with pytest.raises(RuntimeError, match="evenly divisible"):
        amplin.mma_lane_m16_n16_splitk4(
            input,
            packed_mma_lane_qweight,
            packed_scales,
            logical_n=SIZE_N,
        )
    with pytest.raises(RuntimeError, match="evenly divisible"):
        amplin.mma_lane_m16_n16_splitk8(
            input,
            packed_mma_lane_qweight,
            packed_scales,
            logical_n=SIZE_N,
        )
    with pytest.raises(RuntimeError, match="evenly divisible"):
        amplin.mma_lane_m16_n16_splitk12(
            input,
            packed_mma_lane_qweight,
            packed_scales,
            logical_n=SIZE_N,
        )
    with pytest.raises(RuntimeError, match="evenly divisible"):
        amplin.mma_lane_m16_n32_splitk16(
            input,
            packed_mma_lane_qweight,
            packed_scales,
            logical_n=SIZE_N,
        )
    with pytest.raises(RuntimeError, match="evenly divisible"):
        amplin.mma_lane_m16_n32_splitk8(
            input,
            packed_mma_lane_qweight,
            packed_scales,
            logical_n=SIZE_N,
        )
    with pytest.raises(RuntimeError, match="evenly divisible"):
        amplin.mma_lane_m16_n32_splitk8_pipe2(
            input,
            packed_mma_lane_qweight,
            packed_scales,
            logical_n=SIZE_N,
        )
    with pytest.raises(RuntimeError, match="evenly divisible"):
        amplin.mma_lane_m16_n32_splitk12(
            input,
            packed_mma_lane_qweight,
            packed_scales,
            logical_n=SIZE_N,
        )
    with pytest.raises(RuntimeError, match="evenly divisible"):
        amplin.mma_lane_m16_n32_splitk12_pipe2_interleaved(
            input,
            packed_mma_lane_n32_qweight,
            packed_scales,
            logical_n=SIZE_N,
        )
    # Split-K24 N64 is now legal for any positive group count; verify it runs on
    # a non-multiple-of-24 group count (1 group) and matches the deterministic
    # dequantized result for zero qweight and unit scales.
    output = amplin.mma_lane_m16_n64_splitk24_pipe2_interleaved(
        input,
        packed_mma_lane_n64_qweight,
        packed_scales,
        logical_n=SIZE_N,
    )
    expected = torch.full(
        (16, SIZE_N), -1024.0, device=sm80_device, dtype=torch.float16
    )
    assert torch.allclose(output, expected, atol=0.0, rtol=0.0)
    # Split-K12x2 N64 cooperative should now be legal for any positive group
    # count (one group here) and match the deterministic reference.
    output = amplin.mma_lane_m16_n64_splitk12x2_coop_interleaved(
        input,
        packed_mma_lane_n64_qweight,
        packed_scales,
        logical_n=SIZE_N,
    )
    assert torch.allclose(output, expected, atol=0.0, rtol=0.0)
    with pytest.raises(RuntimeError, match="evenly divisible"):
        amplin.mma_lane_m16_n16_splitk16(
            input,
            packed_mma_lane_qweight,
            packed_scales,
            logical_n=SIZE_N,
        )


@pytest.mark.cuda
@pytest.mark.parametrize(
    ("dtype", "atol"),
    [
        (torch.float16, 2e-4),
        (torch.bfloat16, 2e-3),
    ],
)
def test_amplin_gemv_matches_fp32_dequant_reference(
    packed_case: dict[str, torch.Tensor],
    sm80_device: torch.device,
    dtype: torch.dtype,
    atol: float,
):
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA BF16 support required")

    input, qweight, scales, expected = _device_case(packed_case, sm80_device, dtype)
    actual = amplin.gemv(input, qweight, scales)
    repeated = amplin.gemv(input, qweight, scales)
    torch.cuda.synchronize(sm80_device)

    assert actual.shape == (1, SIZE_N)
    assert actual.dtype == dtype
    assert actual.device == sm80_device
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.to(torch.float32), expected, rtol=0, atol=atol)
    torch.testing.assert_close(repeated, actual, rtol=0, atol=0)


@pytest.mark.cuda
@pytest.mark.parametrize(("size_k", "size_n"), REAL_MODEL_SHAPES)
def test_amplin_real_model_shapes_and_batches_match_fp32_dequant_reference(
    sm80_device: torch.device,
    size_k: int,
    size_n: int,
):
    generator = torch.Generator(device=sm80_device)
    generator.manual_seed(20260724 + size_k * 3 + size_n)
    qweight = torch.randint(
        -(2**31),
        2**31 - 1,
        (size_k // 8, size_n),
        device=sm80_device,
        dtype=torch.int32,
        generator=generator,
    )
    scale_values = (
        torch.rand(
            (size_k // GROUP_SIZE, size_n),
            device=sm80_device,
            dtype=torch.float32,
            generator=generator,
        )
        * 0.007
        + 0.001
    )
    codes = _unpack_gptq_w4(qweight, size_k)
    group_indices = torch.arange(size_k, device=sm80_device, dtype=torch.int64) // GROUP_SIZE

    for dtype, atol in ((torch.float16, 2e-3), (torch.bfloat16, 2e-2)):
        if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            continue
        scales = scale_values.to(dtype)
        dense_weight = (codes.to(torch.float32) - ZERO) * scales.to(torch.float32)[group_indices]
        for size_m in BATCHED_ROWS:
            input = (
                torch.randn(
                    (size_m, size_k),
                    device=sm80_device,
                    dtype=torch.float32,
                    generator=generator,
                )
                .mul_(0.25)
                .to(dtype)
            )
            expected = input.to(torch.float32) @ dense_weight
            actual = amplin.gemv(input, qweight, scales)
            torch.cuda.synchronize(sm80_device)

            assert actual.shape == (size_m, size_n)
            assert actual.dtype == dtype
            assert actual.device == sm80_device
            assert torch.isfinite(actual).all()
            torch.testing.assert_close(
                actual.to(torch.float32),
                expected,
                rtol=0,
                atol=atol,
                msg=lambda message: (
                    f"Amplin mismatch for M={size_m}, K={size_k}, N={size_n}, dtype={dtype}: {message}"
                ),
            )


@pytest.mark.cuda
@pytest.mark.parametrize("size_n", (3072, 4096))
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_amplin_k12288_wide_large_down_projection_matches_fp32_dequant_reference(
    sm80_device: torch.device,
    size_n: int,
    dtype: torch.dtype,
):
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA BF16 support required")

    size_k = 12288
    generator = torch.Generator(device=sm80_device)
    generator.manual_seed(20260724 + size_n)
    qweight = torch.randint(
        -(2**31),
        2**31 - 1,
        (size_k // 8, size_n),
        device=sm80_device,
        dtype=torch.int32,
        generator=generator,
    )
    scales = (
        torch.rand(
            (size_k // GROUP_SIZE, size_n),
            device=sm80_device,
            dtype=torch.float32,
            generator=generator,
        )
        * 0.007
        + 0.001
    ).to(dtype)
    input = (
        torch.randn(
            (1, size_k),
            device=sm80_device,
            dtype=torch.float32,
            generator=generator,
        )
        .mul_(0.25)
        .to(dtype)
    )
    codes = _unpack_gptq_w4(qweight, size_k)
    group_indices = torch.arange(size_k, device=sm80_device, dtype=torch.int64) // GROUP_SIZE
    dense_weight = (codes.to(torch.float32) - ZERO) * scales.to(torch.float32)[group_indices]
    expected = input.to(torch.float32) @ dense_weight

    actual = amplin.gemv_k12288_wide(input, qweight, scales)
    repeated = amplin.gemv_k12288_wide(input, qweight, scales)
    stream = torch.cuda.Stream(device=sm80_device)
    with torch.cuda.stream(stream):
        selected = amplin.gemv(input, qweight, scales)
    stream.synchronize()

    atol = 2e-3 if dtype == torch.float16 else 2e-2
    assert actual.shape == (1, size_n)
    assert actual.dtype == dtype
    assert actual.device == sm80_device
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.to(torch.float32), expected, rtol=0, atol=atol)
    torch.testing.assert_close(repeated, actual, rtol=0, atol=0)
    torch.testing.assert_close(selected, actual, rtol=0, atol=0)


@pytest.mark.cuda
@pytest.mark.parametrize(("size_k", "size_n"), LARGE_MLP_SHAPES)
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_amplin_multirow_large_mlp_projection_matches_fp32_dequant_reference(
    sm80_device: torch.device,
    size_k: int,
    size_n: int,
    dtype: torch.dtype,
):
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA BF16 support required")

    generator = torch.Generator(device=sm80_device)
    generator.manual_seed(20260724 + size_k + size_n)
    qweight = torch.randint(
        -(2**31),
        2**31 - 1,
        (size_k // 8, size_n),
        device=sm80_device,
        dtype=torch.int32,
        generator=generator,
    )
    scales = (
        torch.rand(
            (size_k // GROUP_SIZE, size_n),
            device=sm80_device,
            dtype=torch.float32,
            generator=generator,
        )
        * 0.007
        + 0.001
    ).to(dtype)
    full_input = (
        torch.randn(
            (max(MULTIROW_BATCHED_ROWS), size_k),
            device=sm80_device,
            dtype=torch.float32,
            generator=generator,
        )
        .mul_(0.25)
        .to(dtype)
    )
    codes = _unpack_gptq_w4(qweight, size_k)
    group_indices = torch.arange(size_k, device=sm80_device, dtype=torch.int64) // GROUP_SIZE
    dense_weight = (codes.to(torch.float32) - ZERO) * scales.to(torch.float32)[group_indices]
    stream = torch.cuda.Stream(device=sm80_device)
    stream.wait_stream(torch.cuda.current_stream(sm80_device))

    for size_m in MULTIROW_BATCHED_ROWS:
        input = full_input[:size_m].contiguous()
        expected = input.to(torch.float32) @ dense_weight
        with torch.cuda.stream(stream):
            actual = amplin.gemv_multirow(input, qweight, scales)
        stream.synchronize()

        atol = 2e-3 if dtype == torch.float16 else 2e-2
        assert actual.shape == (size_m, size_n)
        assert actual.dtype == dtype
        assert actual.device == sm80_device
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(
            actual.to(torch.float32),
            expected,
            rtol=0,
            atol=atol,
            msg=lambda message: (
                f"Amplin multi-row mismatch for M={size_m}, K={size_k}, "
                f"N={size_n}, dtype={dtype}: {message}"
            ),
        )


@pytest.mark.cuda
@pytest.mark.parametrize(("size_k", "size_n"), LARGE_MLP_SHAPES)
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_amplin_padded_m16_large_mlp_projection_matches_fp32_dequant_reference(
    sm80_device: torch.device,
    size_k: int,
    size_n: int,
    dtype: torch.dtype,
):
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA BF16 support required")

    generator = torch.Generator(device=sm80_device)
    generator.manual_seed(20260724 + size_k * 5 + size_n)
    qweight = torch.randint(
        -(2**31),
        2**31 - 1,
        (size_k // 8, size_n),
        device=sm80_device,
        dtype=torch.int32,
        generator=generator,
    )
    scales = (
        torch.rand(
            (size_k // GROUP_SIZE, size_n),
            device=sm80_device,
            dtype=torch.float32,
            generator=generator,
        )
        * 0.007
        + 0.001
    ).to(dtype)
    packed_qweight = amplin.pack_mma_lane_qweight(qweight)
    packed_n32_qweight = (
        amplin.pack_mma_lane_n32_qweight(qweight)
        if size_k == 12288
        else None
    )
    packed_n64_qweight = amplin.pack_mma_lane_n64_qweight(qweight)
    packed_scales = amplin.pack_hmma_scales(scales)
    full_input = (
        torch.randn(
            (max(MULTIROW_BATCHED_ROWS), size_k),
            device=sm80_device,
            dtype=torch.float32,
            generator=generator,
        )
        .mul_(0.25)
        .to(dtype)
    )
    codes = _unpack_gptq_w4(qweight, size_k)
    group_indices = torch.arange(size_k, device=sm80_device, dtype=torch.int64) // GROUP_SIZE
    dense_weight = (codes.to(torch.float32) - ZERO) * scales.to(torch.float32)[group_indices]
    stream = torch.cuda.Stream(device=sm80_device)
    stream.wait_stream(torch.cuda.current_stream(sm80_device))

    for size_m in MULTIROW_BATCHED_ROWS:
        input = full_input[:size_m].contiguous()
        expected = input.to(torch.float32) @ dense_weight
        with torch.cuda.stream(stream):
            outputs = {
                "padded": amplin.mma_lane_m16_n16_padded(
                    input,
                    packed_qweight,
                    packed_scales,
                    logical_n=size_n,
                )
            }
            if size_k == 12288:
                outputs["split-k4"] = amplin.mma_lane_m16_n16_splitk4(
                    input,
                    packed_qweight,
                    packed_scales,
                    logical_n=size_n,
                )
                outputs["split-k8"] = amplin.mma_lane_m16_n16_splitk8(
                    input,
                    packed_qweight,
                    packed_scales,
                    logical_n=size_n,
                )
                outputs["split-k12"] = amplin.mma_lane_m16_n16_splitk12(
                    input,
                    packed_qweight,
                    packed_scales,
                    logical_n=size_n,
                )
                outputs["split-k8-n32"] = amplin.mma_lane_m16_n32_splitk8(
                    input,
                    packed_qweight,
                    packed_scales,
                    logical_n=size_n,
                )
                outputs["split-k12-n32"] = amplin.mma_lane_m16_n32_splitk12(
                    input,
                    packed_qweight,
                    packed_scales,
                    logical_n=size_n,
                )
                outputs["split-k16-n32"] = amplin.mma_lane_m16_n32_splitk16(
                    input,
                    packed_qweight,
                    packed_scales,
                    logical_n=size_n,
                )
                outputs["split-k12-n32-pipe2"] = amplin.mma_lane_m16_n32_splitk12_pipe2(
                    input,
                    packed_qweight,
                    packed_scales,
                    logical_n=size_n,
                )
                outputs["split-k8-n32-pipe2"] = amplin.mma_lane_m16_n32_splitk8_pipe2(
                    input,
                    packed_qweight,
                    packed_scales,
                    logical_n=size_n,
                )
                outputs["split-k12-n32-pipe2-interleaved"] = (
                    amplin.mma_lane_m16_n32_splitk12_pipe2_interleaved(
                        input,
                        packed_n32_qweight,
                        packed_scales,
                        logical_n=size_n,
                    )
                )
                outputs["split-k24-n64-pipe2-interleaved"] = (
                    amplin.mma_lane_m16_n64_splitk24_pipe2_interleaved(
                        input,
                        packed_n64_qweight,
                        packed_scales,
                        logical_n=size_n,
                    )
                )
                outputs["split-k12x2-n64-coop-interleaved"] = (
                    amplin.mma_lane_m16_n64_splitk12x2_coop_interleaved(
                        input,
                        packed_n64_qweight,
                        packed_scales,
                        logical_n=size_n,
                    )
                )
                if size_n % 256 == 0:
                    outputs["tile4-n64-shared-a"] = (
                        amplin.mma_lane_m16_n64_tile4_shared_a(
                            input,
                            packed_n64_qweight,
                            packed_scales,
                            logical_n=size_n,
                        )
                    )
                    if size_n % 512 == 0:
                        outputs["tile8-n64-shared-a"] = (
                            amplin.mma_lane_m16_n64_tile8_shared_a(
                                input,
                                packed_n64_qweight,
                                packed_scales,
                                logical_n=size_n,
                            )
                        )
                outputs["split-k16-n32-pipe2"] = amplin.mma_lane_m16_n32_splitk16_pipe2(
                    input,
                    packed_qweight,
                    packed_scales,
                    logical_n=size_n,
                )
                outputs["split-k16"] = amplin.mma_lane_m16_n16_splitk16(
                    input,
                    packed_qweight,
                    packed_scales,
                    logical_n=size_n,
                )
        stream.synchronize()

        atol = 2e-3 if dtype == torch.float16 else 2e-2
        for schedule, actual in outputs.items():
            assert actual.shape == (size_m, size_n)
            assert actual.dtype == dtype
            assert actual.device == sm80_device
            assert torch.isfinite(actual).all()
            torch.testing.assert_close(
                actual.to(torch.float32),
                expected,
                rtol=0,
                atol=atol,
                msg=lambda message: (
                    f"Amplin padded-M16 {schedule} mismatch for M={size_m}, "
                    f"K={size_k}, N={size_n}, dtype={dtype}: {message}"
                ),
            )


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_amplin_mma_lane_m32_n64_splitk24_matches_fp32_reference(
    packed_case: dict[str, torch.Tensor],
    sm80_device: torch.device,
    dtype: torch.dtype,
):
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported(sm80_device):
        pytest.skip("CUDA BF16 support required")

    generator = torch.Generator(device=sm80_device)
    generator.manual_seed(20260724)
    input = torch.randn((32, SIZE_K), device=sm80_device, dtype=torch.float32, generator=generator).mul_(0.25).to(dtype)
    qweight = packed_case["qweight"].to(device=sm80_device).contiguous()
    scales = packed_case["scales"].to(device=sm80_device, dtype=dtype).contiguous()
    codes = packed_case["codes"].to(device=sm80_device)
    group_indices = torch.arange(SIZE_K, device=sm80_device, dtype=torch.int64) // GROUP_SIZE
    dense_weight = (codes.to(torch.float32) - ZERO) * scales.to(torch.float32)[group_indices]
    expected = input.to(torch.float32) @ dense_weight

    packed_n64_qweight = amplin.pack_mma_lane_n64_qweight(qweight)
    _, packed_scales = amplin.pack_hmma_weights(qweight, scales)
    for op in (
        amplin.mma_lane_m32_n64_splitk24_pipe2_interleaved,
        amplin.mma_lane_m32_n64_splitk20_pipe2_interleaved,
        amplin.mma_lane_m32_n64_splitk16_pipe2_interleaved,
        amplin.mma_lane_m32_n64_splitk12_pipe2_interleaved,
    ):
        actual = op(
            input,
            packed_n64_qweight,
            packed_scales,
            logical_n=SIZE_N,
        )
        torch.cuda.synchronize(sm80_device)

        assert actual.shape == (32, SIZE_N)
        assert actual.dtype == dtype
        assert actual.device == sm80_device
        assert torch.isfinite(actual).all()
        atol = 2e-3 if dtype == torch.float16 else 2e-2
        torch.testing.assert_close(
            actual.to(torch.float32),
            expected,
            rtol=0,
            atol=atol,
        )


@pytest.mark.cuda
@pytest.mark.parametrize(
    ("dtype", "atol"),
    [
        (torch.float16, 2e-3),
        (torch.bfloat16, 2e-2),
    ],
)
def test_amplin_mma_lane_m32_n64_tile4_matches_fp32_reference(
    packed_case_n256: dict[str, torch.Tensor],
    sm80_device: torch.device,
    dtype: torch.dtype,
    atol: float,
):
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported(sm80_device):
        pytest.skip("CUDA BF16 support required")

    generator = torch.Generator(device=sm80_device)
    generator.manual_seed(20260724)
    input = torch.randn((32, SIZE_K), device=sm80_device, dtype=torch.float32, generator=generator).mul_(0.25).to(dtype)
    qweight = packed_case_n256["qweight"].to(device=sm80_device).contiguous()
    scales = packed_case_n256["scales"].to(device=sm80_device, dtype=dtype).contiguous()
    codes = packed_case_n256["codes"].to(device=sm80_device)
    group_indices = torch.arange(SIZE_K, device=sm80_device, dtype=torch.int64) // GROUP_SIZE
    dense_weight = (codes.to(torch.float32) - ZERO) * scales.to(torch.float32)[group_indices]
    expected = input.to(torch.float32) @ dense_weight

    packed_n64_qweight = amplin.pack_mma_lane_n64_qweight(qweight)
    _, packed_scales = amplin.pack_hmma_weights(qweight, scales)

    for name, op in (
        ("tile2", amplin.mma_lane_m32_n64_tile2_shared_a),
        ("tile2_interleaved_dequant", amplin.mma_lane_m32_n64_tile2_interleaved_dequant),
        ("tile2_splitk2", amplin.mma_lane_m32_n64_tile2_splitk2),
        ("tile2_splitk4", amplin.mma_lane_m32_n64_tile2_splitk4),
        ("tile1_splitk4", amplin.mma_lane_m32_n64_tile1_splitk4),
        ("tile1_splitk8", amplin.mma_lane_m32_n64_tile1_splitk8),
        ("tile4", amplin.mma_lane_m32_n64_tile4_shared_a),
        ("tile8", amplin.mma_lane_m32_n64_tile8_shared_a),
    ):
        actual = op(
            input,
            packed_n64_qweight,
            packed_scales,
            logical_n=512,
        )
        torch.cuda.synchronize(sm80_device)

        assert actual.shape == (32, 512), name
        assert actual.dtype == dtype, name
        assert actual.device == sm80_device, name
        assert torch.isfinite(actual).all(), name
        torch.testing.assert_close(
            actual.to(torch.float32),
            expected,
            rtol=0,
            atol=atol,
        )


@pytest.mark.cuda
@pytest.mark.parametrize(
    "batch_shape",
    (
        pytest.param((1,), id="batch-1"),
        pytest.param((2,), id="batch-2"),
        pytest.param((2, 2), id="batch-4"),
        pytest.param((2, 2, 2), id="batch-8"),
    ),
)
def test_amplin_preserves_batched_input_prefix_shape(
    packed_case: dict[str, torch.Tensor],
    sm80_device: torch.device,
    batch_shape: tuple[int, ...],
):
    input, qweight, scales, _ = _device_case(packed_case, sm80_device, torch.float16)
    size_m = math.prod(batch_shape)
    batched_input = input.expand(size_m, -1).clone().reshape(*batch_shape, SIZE_K)
    actual = amplin.gemv(batched_input, qweight, scales)
    flat_actual = amplin.gemv(batched_input.reshape(size_m, SIZE_K), qweight, scales)
    torch.cuda.synchronize(sm80_device)

    assert actual.shape == (*batch_shape, SIZE_N)
    torch.testing.assert_close(actual.reshape(size_m, SIZE_N), flat_actual, rtol=0, atol=0)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_amplin_gemv_uses_current_stream(
    packed_case: dict[str, torch.Tensor],
    sm80_device: torch.device,
    dtype: torch.dtype,
):
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA BF16 support required")

    input, qweight, scales, expected = _device_case(packed_case, sm80_device, dtype)
    stream = torch.cuda.Stream(device=sm80_device)
    with torch.cuda.stream(stream):
        actual = amplin.gemv(input, qweight, scales)
    stream.synchronize()

    atol = 2e-4 if dtype == torch.float16 else 2e-3
    torch.testing.assert_close(actual.to(torch.float32), expected, rtol=0, atol=atol)


@pytest.mark.cuda
def test_amplin_gemv_rejects_inputs_outside_contract(
    packed_case: dict[str, torch.Tensor],
    sm80_device: torch.device,
):
    input, qweight, scales, _ = _device_case(packed_case, sm80_device, torch.float16)

    non_contiguous_input = torch.empty((1, SIZE_K * 2), device=sm80_device, dtype=torch.float16)[:, ::2]
    with pytest.raises(RuntimeError, match="must be contiguous"):
        amplin.gemv(non_contiguous_input, qweight, scales)

    with pytest.raises(RuntimeError, match="K must be positive and divisible by group size 128"):
        amplin.gemv(input[:, :4032].contiguous(), qweight, scales)

    with pytest.raises(RuntimeError, match="qweight must be int32"):
        amplin.gemv(input, qweight.to(torch.int64), scales)

    with pytest.raises(RuntimeError, match="scales dtype must match"):
        amplin.gemv(input, qweight, scales.to(torch.bfloat16))

    with pytest.raises(RuntimeError, match="N must be positive"):
        amplin.gemv(input, qweight[:, :0].contiguous(), scales[:, :0].contiguous())

    with pytest.raises(RuntimeError, match=r"qweight must have canonical shape \[K/8, N\]"):
        amplin.gemv(input, qweight[:-1].contiguous(), scales)

    with pytest.raises(RuntimeError, match=r"scales must have shape \[K/128, N\]"):
        amplin.gemv(input, qweight, scales[:-1].contiguous())

    with pytest.raises(RuntimeError, match="at least one activation row"):
        amplin.gemv(input[:0].contiguous(), qweight, scales)

    with pytest.raises(RuntimeError, match="weight tensors must be CUDA"):
        amplin.gemv(input, qweight.cpu(), scales)


@pytest.mark.cuda
@pytest.mark.parametrize("size_m", (1, 16, 32))
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_amplin_dynamic_routes_and_matches_fp32_reference(
    packed_case: dict[str, torch.Tensor],
    sm80_device: torch.device,
    size_m: int,
    dtype: torch.dtype,
):
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA BF16 support required")

    amplin.clear_dynamic_routing_table()
    qweight = packed_case["qweight"].to(sm80_device).contiguous()
    scales = packed_case["scales"].to(sm80_device, dtype=dtype).contiguous()
    codes = packed_case["codes"].to(sm80_device)
    group_indices = torch.arange(SIZE_K, device=sm80_device, dtype=torch.int64) // GROUP_SIZE
    dense_weight = (codes.to(torch.float32) - ZERO) * scales.to(torch.float32)[group_indices]

    generator = torch.Generator(device=sm80_device)
    generator.manual_seed(20260724 + size_m)
    input = (
        torch.randn((size_m, SIZE_K), device=sm80_device, dtype=torch.float32, generator=generator)
        .mul_(0.25)
        .to(dtype)
    )
    expected = input.to(torch.float32) @ dense_weight

    actual = amplin.dynamic(input, qweight, scales, warmup=1, iters=3)
    torch.cuda.synchronize(sm80_device)

    assert actual.shape == (size_m, SIZE_N)
    assert actual.dtype == dtype
    assert actual.device == sm80_device
    assert torch.isfinite(actual).all()

    key = (size_m, SIZE_K, SIZE_N, "fp16" if dtype == torch.float16 else "bf16")
    assert key in amplin.get_dynamic_routing_table()

    atol = 2e-3 if dtype == torch.float16 else 2e-2
    torch.testing.assert_close(actual.to(torch.float32), expected, rtol=0, atol=atol)

    # Cached path should reuse the same kernel and produce identical output.
    cached = amplin.dynamic(input, qweight, scales)
    torch.testing.assert_close(cached, actual, rtol=0, atol=0)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_amplin_marlin_style_matches_fp32_reference(
    packed_case: dict[str, torch.Tensor],
    sm80_device: torch.device,
    dtype: torch.dtype,
):
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA BF16 support required")

    size_m = 32
    qweight = packed_case["qweight"].to(sm80_device).contiguous()
    scales = packed_case["scales"].to(sm80_device, dtype=dtype).contiguous()
    codes = packed_case["codes"].to(sm80_device)
    group_indices = torch.arange(SIZE_K, device=sm80_device, dtype=torch.int64) // GROUP_SIZE
    dense_weight = (codes.to(torch.float32) - ZERO) * scales.to(torch.float32)[group_indices]

    generator = torch.Generator(device=sm80_device)
    generator.manual_seed(20260724 + size_m)
    input = (
        torch.randn((size_m, SIZE_K), device=sm80_device, dtype=torch.float32, generator=generator)
        .mul_(0.25)
        .to(dtype)
    )
    expected = input.to(torch.float32) @ dense_weight

    # Force the marlin_style path and verify correctness against the dense reference.
    key = (size_m, SIZE_K, SIZE_N, "fp16" if dtype == torch.float16 else "bf16")
    previous_static = amplin.get_static_routing_table()
    try:
        amplin.set_routing_table({**previous_static, key: "marlin_style"})
        amplin.clear_dynamic_routing_table()
        actual = amplin.dynamic(input, qweight, scales, warmup=1, iters=3)
    finally:
        amplin.set_routing_table(previous_static)
        amplin.clear_dynamic_routing_table()

    torch.cuda.synchronize(sm80_device)
    assert actual.shape == (size_m, SIZE_N)
    assert actual.dtype == dtype
    assert actual.device == sm80_device
    assert torch.isfinite(actual).all()

    atol = 2e-3 if dtype == torch.float16 else 2e-2
    torch.testing.assert_close(actual.to(torch.float32), expected, rtol=0, atol=atol)

    # Second call should use the fast-dispatch cache and be identical.
    cached = amplin.dynamic(input, qweight, scales)
    torch.testing.assert_close(cached, actual, rtol=0, atol=0)

    # 3-D batched inputs must flatten to the same M and return the original prefix shape.
    batched_input = input.reshape(2, 16, SIZE_K)
    batched_expected = batched_input.to(torch.float32) @ dense_weight
    batched = amplin.dynamic(batched_input, qweight, scales)
    torch.cuda.synchronize(sm80_device)
    assert batched.shape == (2, 16, SIZE_N)
    torch.testing.assert_close(batched.to(torch.float32), batched_expected, rtol=0, atol=atol)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_amplin_marlin_style_run_torch_compile_fullgraph(
    packed_case: dict[str, torch.Tensor],
    sm80_device: torch.device,
    dtype: torch.dtype,
):
    """The C++ marlin_style_run op must be traceable by torch.compile(fullgraph=True)."""
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA BF16 support required")
    if not amplin._marlin_available_cached(dtype):
        pytest.skip("Marlin runtime not available for this dtype")

    size_m = 32
    qweight = packed_case["qweight"].to(sm80_device).contiguous()
    scales = packed_case["scales"].to(sm80_device, dtype=dtype).contiguous()

    generator = torch.Generator(device=sm80_device)
    generator.manual_seed(20260724 + size_m)
    input_t = (
        torch.randn((size_m, SIZE_K), device=sm80_device, dtype=torch.float32, generator=generator)
        .mul_(0.25)
        .to(dtype)
    )

    key = (size_m, SIZE_K, SIZE_N, "fp16" if dtype == torch.float16 else "bf16")
    previous_static = amplin.get_static_routing_table()
    try:
        amplin.set_routing_table({**previous_static, key: "marlin_style"})
        amplin.clear_dynamic_routing_table()
        reference = amplin.dynamic(input_t, qweight, scales)

        # Compile the raw C++ marlin_style_run op, not the Python fast-runner
        # closure, so the test isolates the Meta/Composite implementation.
        # The eager reference call moved the canonical weights to CPU; bring fresh
        # GPU copies for the compile path so dispatch sees all tensors on cuda:0.
        qweight = qweight.to(sm80_device).contiguous()
        scales = scales.to(sm80_device, dtype=dtype).contiguous()
        marlin_qweight, marlin_scales, workspace, b_q_type = amplin._get_marlin_packed(
            qweight, scales, SIZE_N, dtype, sm80_device
        )
        cpp_op = amplin._get_amplin_op("marlin_style_run")
        size_k = qweight.size(0) * 8

        def fn(x: torch.Tensor) -> torch.Tensor:
            return cpp_op(x, marlin_qweight, marlin_scales, workspace, b_q_type.id, SIZE_N, size_k)

        compiled = torch.compile(fn, fullgraph=True)
        output = compiled(input_t)
    finally:
        amplin.set_routing_table(previous_static)
        amplin.clear_dynamic_routing_table()

    assert output.shape == reference.shape
    assert output.dtype == reference.dtype
    torch.testing.assert_close(output, reference, rtol=0, atol=0)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_amplin_dynamic_torch_compile_fullgraph(
    packed_case: dict[str, torch.Tensor],
    sm80_device: torch.device,
    dtype: torch.dtype,
):
    """The Python ``amplin.dynamic`` router must be opaque to ``torch.compile(fullgraph=True)``."""
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA BF16 support required")
    if not amplin._marlin_available_cached(dtype):
        pytest.skip("Marlin runtime not available for this dtype")

    size_m = 32
    qweight = packed_case["qweight"].to(sm80_device).contiguous()
    scales = packed_case["scales"].to(sm80_device, dtype=dtype).contiguous()

    generator = torch.Generator(device=sm80_device)
    generator.manual_seed(20260724 + size_m)
    input_t = (
        torch.randn((size_m, SIZE_K), device=sm80_device, dtype=torch.float32, generator=generator)
        .mul_(0.25)
        .to(dtype)
    )

    key = (size_m, SIZE_K, SIZE_N, "fp16" if dtype == torch.float16 else "bf16")
    previous_static = amplin.get_static_routing_table()
    try:
        amplin.set_routing_table({**previous_static, key: "marlin_style"})
        amplin.clear_dynamic_routing_table()
        reference = amplin.dynamic(input_t, qweight, scales)

        # The eager reference call canonicalized the weights to CPU.  Use fresh
        # GPU copies for compilation so the custom-op dispatch sees one device.
        qweight = qweight.to(sm80_device).contiguous()
        scales = scales.to(sm80_device, dtype=dtype).contiguous()
        compiled = torch.compile(amplin.dynamic, fullgraph=True)
        output = compiled(input_t, qweight, scales)
    finally:
        amplin.set_routing_table(previous_static)
        amplin.clear_dynamic_routing_table()

    assert output.shape == reference.shape
    assert output.dtype == reference.dtype
    torch.testing.assert_close(output, reference, rtol=0, atol=0)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_amplin_dynamic_thread_safety(
    packed_case: dict[str, torch.Tensor],
    sm80_device: torch.device,
    dtype: torch.dtype,
):
    """Concurrent ``amplin.dynamic`` calls from many threads must be race-free."""
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA BF16 support required")

    qweight = packed_case["qweight"].to(sm80_device).contiguous()
    scales = packed_case["scales"].to(sm80_device, dtype=dtype).contiguous()
    codes = packed_case["codes"].to(sm80_device)
    group_indices = torch.arange(SIZE_K, device=sm80_device, dtype=torch.int64) // GROUP_SIZE
    dense_weight = (codes.to(torch.float32) - ZERO) * scales.to(torch.float32)[group_indices]

    # Reference inputs/outputs for the two batch sizes that exercise different kernels.
    inputs: dict[int, torch.Tensor] = {}
    expected: dict[int, torch.Tensor] = {}
    for size_m in (1, 16):
        generator = torch.Generator(device=sm80_device)
        generator.manual_seed(20260724 + size_m)
        input_t = (
            torch.randn((size_m, SIZE_K), device=sm80_device, dtype=torch.float32, generator=generator)
            .mul_(0.25)
            .to(dtype)
        )
        inputs[size_m] = input_t
        expected[size_m] = input_t.to(torch.float32) @ dense_weight

    amplin.clear_dynamic_routing_table()
    errors: list[BaseException] = []
    barrier = threading.Barrier(8)

    def worker() -> None:
        try:
            barrier.wait()
            for i in range(20):
                size_m = (1, 16)[i % 2]
                actual = amplin.dynamic(
                    inputs[size_m], qweight, scales, warmup=1, iters=2
                )
                torch.cuda.synchronize(sm80_device)
                atol = 2e-3 if dtype == torch.float16 else 2e-2
                torch.testing.assert_close(
                    actual.to(torch.float32), expected[size_m], rtol=0, atol=atol
                )
        except BaseException as e:
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors[0]


@pytest.mark.cuda
def test_amplin_thread_local_cache_isolation(sm80_device: torch.device) -> None:
    """Each Python thread must get its own _ThreadLocalAmplinCaches instance."""
    ids: dict[int, int] = {}

    def worker(tid: int) -> None:
        ids[tid] = id(amplin._thread_caches())

    t1 = threading.Thread(target=worker, args=(1,))
    t2 = threading.Thread(target=worker, args=(2,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert len(ids) == 2
    assert ids[1] != ids[2]
    assert id(amplin._thread_caches()) not in (ids[1], ids[2])


@pytest.mark.cuda
def test_amplin_thread_local_cache_released_on_thread_death(
    packed_case: dict[str, torch.Tensor],
    sm80_device: torch.device,
) -> None:
    """Per-thread GPU caches are cleared by the weakref finalizer when the thread dies."""
    qweight = packed_case["qweight"].to(sm80_device).contiguous()
    scales = packed_case["scales"].to(sm80_device, dtype=torch.float16).contiguous()

    generator = torch.Generator(device=sm80_device)
    generator.manual_seed(20260726)
    input_t = (
        torch.randn((16, SIZE_K), device=sm80_device, dtype=torch.float32, generator=generator)
        .mul_(0.25)
        .to(torch.float16)
    )

    caches_holder: dict[str, object] = {}

    def worker() -> None:
        amplin.clear_thread_caches()
        _ = amplin.dynamic(input_t, qweight, scales, warmup=1, iters=2)
        caches_holder["caches"] = amplin._thread_caches()

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    del t

    # The Thread object is weakly referenced; collect it so the finalizer fires.
    for _ in range(20):
        gc.collect()
        caches = caches_holder.get("caches")
        if caches is not None and not caches.weight_cache:  # type: ignore[attr-defined]
            break
        time.sleep(0.05)
    else:
        raise AssertionError("thread cache finalizer did not clear the per-thread caches")

    caches = caches_holder["caches"]
    assert not caches.weight_cache  # type: ignore[attr-defined]
    assert not caches.weight_cache_pending  # type: ignore[attr-defined]
    assert not caches.weight_cache_resident  # type: ignore[attr-defined]
    assert not caches.fast_dispatch_cache  # type: ignore[attr-defined]
    assert not caches.marlin_pack_cache  # type: ignore[attr-defined]
    assert not caches.marlin_pack_cache_pending  # type: ignore[attr-defined]
    assert not caches.marlin_pack_cache_resident  # type: ignore[attr-defined]
    assert not caches.weight_copy_streams  # type: ignore[attr-defined]
    assert not caches.marlin_copy_streams  # type: ignore[attr-defined]


@pytest.mark.cuda
def test_amplin_packed_layout_moves_original_weights_to_cpu(
    packed_case: dict[str, torch.Tensor],
    sm80_device: torch.device,
) -> None:
    """A non-``none`` packed layout must leave the canonical weights in CPU RAM."""
    qweight = packed_case["qweight"].to(sm80_device).contiguous()
    scales = packed_case["scales"].to(sm80_device, dtype=torch.float16).contiguous()

    generator = torch.Generator(device=sm80_device)
    generator.manual_seed(20260724 + 16)
    input_t = (
        torch.randn((16, SIZE_K), device=sm80_device, dtype=torch.float32, generator=generator)
        .mul_(0.25)
        .to(torch.float16)
    )

    key = (16, SIZE_K, SIZE_N, "fp16")
    previous_static = amplin.get_static_routing_table()
    try:
        amplin.set_routing_table({**previous_static, key: "gemm_hmma"})
        amplin.clear_dynamic_routing_table()
        _ = amplin.dynamic(input_t, qweight, scales, warmup=1, iters=2)
        assert qweight.device.type == "cpu"
        assert scales.device.type == "cpu"
    finally:
        amplin.set_routing_table(previous_static)
        amplin.clear_dynamic_routing_table()


@pytest.mark.cuda
def test_amplin_gemv_keeps_original_weights_on_gpu(
    packed_case: dict[str, torch.Tensor],
    sm80_device: torch.device,
) -> None:
    """The ``none``/gemv layout must keep the canonical weights GPU-resident."""
    qweight = packed_case["qweight"].to(sm80_device).contiguous()
    scales = packed_case["scales"].to(sm80_device, dtype=torch.float16).contiguous()

    generator = torch.Generator(device=sm80_device)
    generator.manual_seed(20260724 + 1)
    input_t = (
        torch.randn((1, SIZE_K), device=sm80_device, dtype=torch.float32, generator=generator)
        .mul_(0.25)
        .to(torch.float16)
    )

    key = (1, SIZE_K, SIZE_N, "fp16")
    previous_static = amplin.get_static_routing_table()
    try:
        amplin.set_routing_table({**previous_static, key: "gemv"})
        amplin.clear_dynamic_routing_table()
        _ = amplin.dynamic(input_t, qweight, scales, warmup=1, iters=2)
        assert qweight.device.type == "cuda"
        assert scales.device.type == "cuda"
    finally:
        amplin.set_routing_table(previous_static)
        amplin.clear_dynamic_routing_table()


@pytest.mark.cuda
def test_amplin_round_trip_gemv_and_packed_layout(
    packed_case: dict[str, torch.Tensor],
    sm80_device: torch.device,
) -> None:
    """Switching between gemv and a packed layout must restore weights and stay correct."""
    qweight = packed_case["qweight"].to(sm80_device).contiguous()
    scales = packed_case["scales"].to(sm80_device, dtype=torch.float16).contiguous()
    codes = packed_case["codes"].to(sm80_device)
    group_indices = torch.arange(SIZE_K, device=sm80_device, dtype=torch.int64) // GROUP_SIZE
    dense_weight = (codes.to(torch.float32) - ZERO) * scales.to(torch.float32)[group_indices]

    key1 = (1, SIZE_K, SIZE_N, "fp16")
    key16 = (16, SIZE_K, SIZE_N, "fp16")
    previous_static = amplin.get_static_routing_table()
    try:
        amplin.set_routing_table({**previous_static, key1: "gemv", key16: "gemm_hmma"})
        amplin.clear_dynamic_routing_table()

        generator = torch.Generator(device=sm80_device)
        generator.manual_seed(20260724 + 1)
        input1 = (
            torch.randn((1, SIZE_K), device=sm80_device, dtype=torch.float32, generator=generator)
            .mul_(0.25)
            .to(torch.float16)
        )
        generator.manual_seed(20260724 + 16)
        input16 = (
            torch.randn((16, SIZE_K), device=sm80_device, dtype=torch.float32, generator=generator)
            .mul_(0.25)
            .to(torch.float16)
        )

        # First packed call is correct.
        out16 = amplin.dynamic(input16, qweight, scales, warmup=1, iters=2)
        torch.testing.assert_close(
            out16.to(torch.float32), input16.to(torch.float32) @ dense_weight, rtol=0, atol=2e-3
        )

        # gemv runs correctly from the same canonical weights.
        out1 = amplin.dynamic(input1, qweight, scales, warmup=1, iters=2)
        torch.testing.assert_close(
            out1.to(torch.float32), input1.to(torch.float32) @ dense_weight, rtol=0, atol=2e-3
        )

        # Packed call again still produces correct output.
        out16_2 = amplin.dynamic(input16, qweight, scales, warmup=1, iters=2)
        torch.testing.assert_close(
            out16_2.to(torch.float32), input16.to(torch.float32) @ dense_weight, rtol=0, atol=2e-3
        )
    finally:
        amplin.set_routing_table(previous_static)
        amplin.clear_dynamic_routing_table()


@pytest.mark.cuda
def test_amplin_original_weight_residency_thread_safety(
    packed_case: dict[str, torch.Tensor],
    sm80_device: torch.device,
) -> None:
    """Shared qweight/scales can be concurrently routed to gemv and packed layouts."""
    qweight = packed_case["qweight"].to(sm80_device).contiguous()
    scales = packed_case["scales"].to(sm80_device, dtype=torch.float16).contiguous()
    codes = packed_case["codes"].to(sm80_device)
    group_indices = torch.arange(SIZE_K, device=sm80_device, dtype=torch.int64) // GROUP_SIZE
    dense_weight = (codes.to(torch.float32) - ZERO) * scales.to(torch.float32)[group_indices]

    key1 = (1, SIZE_K, SIZE_N, "fp16")
    key16 = (16, SIZE_K, SIZE_N, "fp16")
    previous_static = amplin.get_static_routing_table()
    try:
        amplin.set_routing_table({**previous_static, key1: "gemv", key16: "gemm_hmma"})
        amplin.clear_dynamic_routing_table()

        generators = {
            1: torch.Generator(device=sm80_device).manual_seed(20260724 + 1),
            16: torch.Generator(device=sm80_device).manual_seed(20260724 + 16),
        }
        inputs = {
            m: torch.randn((m, SIZE_K), device=sm80_device, dtype=torch.float32, generator=generators[m])
            .mul_(0.25)
            .to(torch.float16)
            for m in (1, 16)
        }
        expected = {m: inputs[m].to(torch.float32) @ dense_weight for m in (1, 16)}

        errors: list[BaseException] = []
        barrier = threading.Barrier(8)

        def worker() -> None:
            try:
                barrier.wait()
                for i in range(20):
                    size_m = (1, 16)[i % 2]
                    actual = amplin.dynamic(inputs[size_m], qweight, scales, warmup=1, iters=2)
                    torch.cuda.synchronize(sm80_device)
                    torch.testing.assert_close(
                        actual.to(torch.float32), expected[size_m], rtol=0, atol=2e-3
                    )
            except BaseException as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, errors[0]
    finally:
        amplin.set_routing_table(previous_static)
        amplin.clear_dynamic_routing_table()


@pytest.mark.cuda
@pytest.mark.parametrize(
    "size_k,size_n",
    [
        pytest.param(4096, 4096, id="k4096-n4096"),
        pytest.param(4096, 12288, id="k4096-n12288"),
        pytest.param(12288, 4096, id="k12288-n4096"),
    ],
)
def test_amplin_linear_prefill_selects_largest_m_family_member(
    monkeypatch,
    sm80_device,
    size_k,
    size_n,
) -> None:
    """AmplinLinear.forward switches to the largest-M family member at prefill time."""
    monkeypatch.setenv("KERNEL_BATCH_HINT", "16")

    qweight = torch.randint(
        -2**31, 2**31 - 1, (size_k // 8, size_n), device=sm80_device, dtype=torch.int32
    )
    scales = (
        torch.rand((size_k // 128, size_n), device=sm80_device, dtype=torch.float32) * 0.007
        + 0.001
    ).to(torch.float16)

    codes = _unpack_gptq_w4(qweight, size_k)
    group_indices = torch.arange(size_k, device=sm80_device, dtype=torch.int64) // 128
    dense_weight = (codes.to(torch.float32) - 8) * scales.to(torch.float32)[group_indices]

    layer = AmplinLinear(
        bits=4,
        group_size=128,
        sym=True,
        desc_act=False,
        in_features=size_k,
        out_features=size_n,
        bias=False,
        pack_dtype=torch.int32,
        adapter=None,
        register_buffers=True,
        backend=BACKEND.GPTQ_AMPLIN,
        name=f"test_{size_k}_{size_n}",
    ).to(sm80_device)
    layer.qweight.copy_(qweight)
    layer.scales.copy_(scales)

    layer.post_init()

    # decode-time M=16 should use the HINT-selected dispatch.
    input_16 = (
        torch.randn((16, size_k), device=sm80_device, dtype=torch.float16).mul_(0.25).contiguous()
    )
    expected_16 = input_16.to(torch.float32) @ dense_weight
    out_16 = layer(input_16)
    torch.testing.assert_close(out_16.to(torch.float32), expected_16, rtol=0, atol=2e-3)

    # prefill-time M=512 should pick a family member that handles the full batch.
    member = amplin._select_family_member(layer._amplin_family, 512)
    assert member is not None, f"No large-M family member for {size_k}x{size_n}"
    assert member.max_m is None, f"Prefill member {member.name} still chunks at max_m={member.max_m}"

    input_512 = (
        torch.randn((512, size_k), device=sm80_device, dtype=torch.float16).mul_(0.25).contiguous()
    )
    expected_512 = input_512.to(torch.float32) @ dense_weight
    out_512 = layer(input_512)
    assert out_512.shape == (512, size_n)
    torch.testing.assert_close(out_512.to(torch.float32), expected_512, rtol=0, atol=2e-3)

    del layer
