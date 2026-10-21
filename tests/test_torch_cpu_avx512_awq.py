# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

from gptqmodel.models._const import DEVICE
from gptqmodel.nn_modules.qlinear.torch_aten_kernel_awq import TorchAtenAwqLinear
from gptqmodel.nn_modules.qlinear.torch_cpu_avx512 import (
    TorchCpuAvx512Linear,
    pack_hf_int4,
)
from gptqmodel.nn_modules.qlinear.torch_cpu_avx512_awq import (
    TorchCpuAvx512AwqLinear,
)
from gptqmodel.quantization import FORMAT, METHOD
from gptqmodel.quantization.awq.utils.packing_utils import (
    dequantize_gemm,
    pack_awq,
    reverse_awq_order,
    unpack_awq,
)
from gptqmodel.utils.backend import BACKEND
from gptqmodel.utils.cpu_features import avx512_bf16_error, has_avx512_bf16
from gptqmodel.utils.importer import clear_validation_cache, select_quant_linear


pytestmark = [pytest.mark.cpu]


def _skip_unless_avx512_bf16() -> None:
    if not has_avx512_bf16():
        pytest.skip(avx512_bf16_error())
    ok, err = TorchCpuAvx512AwqLinear.cached_validate_once()
    if not ok:
        pytest.skip(str(err))


def _awq_layout_oracle(
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    out_features: int,
    in_features: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert AWQ words with the upstream unpack/reorder reference."""

    codes, zeros = unpack_awq(qweight, qzeros, 4, count=out_features)
    codes, zeros = reverse_awq_order(codes, zeros, 4)
    codes = (codes & 0xF).to(torch.uint8)
    zeros = (zeros & 0xF).to(torch.uint8).reshape(-1, out_features)
    return pack_hf_int4(codes.t().contiguous()), zeros.contiguous()


def _make_awq_buffers(
    in_features: int,
    out_features: int,
    group_size: int,
    *,
    kind: str = "random",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    groups = in_features // group_size
    if kind == "random":
        codes = torch.randint(0, 16, (in_features, out_features), dtype=torch.uint8)
        zeros = torch.randint(0, 16, (groups, out_features), dtype=torch.uint8)
    elif kind == "zero":
        codes = torch.zeros((in_features, out_features), dtype=torch.uint8)
        zeros = torch.zeros((groups, out_features), dtype=torch.uint8)
    elif kind == "max":
        codes = torch.full((in_features, out_features), 15, dtype=torch.uint8)
        zeros = torch.full((groups, out_features), 15, dtype=torch.uint8)
    elif kind == "alternating":
        rows = torch.arange(in_features, dtype=torch.int64)[:, None]
        cols = torch.arange(out_features, dtype=torch.int64)[None, :]
        codes = (((rows + cols) & 1) * 15).to(torch.uint8)
        groups_idx = torch.arange(groups, dtype=torch.int64)[:, None]
        zeros = (((groups_idx + cols) & 1) * 15).to(torch.uint8)
    else:
        raise ValueError(f"unknown fixture kind: {kind}")

    qweight, qzeros = pack_awq(codes, zeros, 4)
    scales = torch.rand((groups, out_features), dtype=torch.float16) * 0.04 + 0.01
    return qweight, qzeros, scales


def _make_layer(
    *,
    in_features: int,
    out_features: int,
    group_size: int,
    desc_act: bool = False,
    bias: bool = False,
) -> TorchCpuAvx512AwqLinear:
    layer = TorchCpuAvx512AwqLinear(
        bits=4,
        group_size=group_size,
        sym=True,
        desc_act=desc_act,
        in_features=in_features,
        out_features=out_features,
        bias=bias,
        pack_dtype=torch.int32,
        register_buffers=True,
    )
    layer.eval()
    return layer


def _fill_layer(
    layer: TorchCpuAvx512AwqLinear,
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
) -> None:
    layer.qweight.copy_(qweight)
    layer.qzeros.copy_(qzeros)
    layer.scales.copy_(scales)


@pytest.mark.parametrize(
    ("out_features", "in_features", "group_size"),
    ((32, 64, 16), (64, 128, 32), (96, 256, 64), (128, 512, 128)),
)
@pytest.mark.parametrize("kind", ("random", "zero", "max", "alternating"))
def test_awq_packer_matches_independent_layout_oracle(
    out_features,
    in_features,
    group_size,
    kind,
):
    _skip_unless_avx512_bf16()
    torch.manual_seed(1000 + out_features + in_features + group_size)
    qweight, qzeros, _ = _make_awq_buffers(
        in_features,
        out_features,
        group_size,
        kind=kind,
    )

    packed, zero_codes = TorchCpuAvx512AwqLinear.pack_awq_int4_kernel(
        qweight,
        qzeros,
        out_features,
        in_features,
    )
    expected_packed, expected_zeros = _awq_layout_oracle(
        qweight,
        qzeros,
        out_features,
        in_features,
    )

    assert torch.equal(packed, expected_packed)
    assert torch.equal(zero_codes, expected_zeros)


def test_awq_packer_rejects_invalid_inputs():
    _skip_unless_avx512_bf16()
    kernel = TorchCpuAvx512AwqLinear.pack_awq_int4_kernel
    out_features, in_features, group_size = 64, 256, 64
    groups = in_features // group_size
    qweight = torch.zeros((in_features, out_features // 8), dtype=torch.int32)
    qzeros = torch.zeros((groups, out_features // 8), dtype=torch.int32)

    with pytest.raises(RuntimeError, match="qweight must be int32"):
        kernel(qweight.float(), qzeros, out_features, in_features)
    with pytest.raises(RuntimeError, match="qzeros must be int32"):
        kernel(qweight, qzeros.long(), out_features, in_features)
    with pytest.raises(RuntimeError, match="qweight must be 2D"):
        kernel(qweight.unsqueeze(0), qzeros, out_features, in_features)
    with pytest.raises(RuntimeError, match="qzeros must be 2D"):
        kernel(qweight, qzeros.flatten(), out_features, in_features)
    with pytest.raises(RuntimeError, match="qweight must be contiguous"):
        kernel(
            qweight.repeat_interleave(2, dim=1)[:, ::2],
            qzeros,
            out_features,
            in_features,
        )
    with pytest.raises(RuntimeError, match="qzeros must be contiguous"):
        kernel(
            qweight,
            qzeros.repeat_interleave(2, dim=1)[:, ::2],
            out_features,
            in_features,
        )
    with pytest.raises(RuntimeError, match="divisible by 32"):
        kernel(qweight, qzeros, 33, in_features)
    with pytest.raises(
        RuntimeError,
        match="in_features must be even",
    ):
        kernel(
            torch.zeros((in_features - 1, out_features // 8), dtype=torch.int32),
            qzeros,
            out_features,
            in_features - 1,
        )
    with pytest.raises(
        RuntimeError,
        match="qweight feature dimension does not match in_features",
    ):
        kernel(qweight[:-1], qzeros, out_features, in_features)
    with pytest.raises(
        RuntimeError,
        match="qweight output dimension does not match out_features",
    ):
        kernel(
            torch.zeros((in_features, out_features // 8 + 1), dtype=torch.int32),
            qzeros,
            out_features,
            in_features,
        )
    with pytest.raises(
        RuntimeError,
        match="qzeros output dimension does not match out_features",
    ):
        kernel(
            qweight,
            torch.zeros((groups, out_features // 8 + 1), dtype=torch.int32),
            out_features,
            in_features,
        )
    with pytest.raises(
        RuntimeError,
        match="qzeros group count must be > 0",
    ):
        kernel(
            qweight,
            torch.zeros((0, out_features // 8), dtype=torch.int32),
            out_features,
            in_features,
        )
    with pytest.raises(
        RuntimeError,
        match="qzeros group count must divide in_features",
    ):
        kernel(
            qweight,
            torch.zeros((3, out_features // 8), dtype=torch.int32),
            out_features,
            in_features,
        )


@pytest.mark.parametrize("desc_act", (False, True))
def test_awq_layer_transform_and_dequantize_match_oracle(desc_act):
    _skip_unless_avx512_bf16()
    torch.manual_seed(2026 if desc_act else 2025)
    in_features, out_features, group_size = 256, 64, 64
    qweight, qzeros, scales = _make_awq_buffers(
        in_features,
        out_features,
        group_size,
    )
    layer = _make_layer(
        in_features=in_features,
        out_features=out_features,
        group_size=group_size,
        desc_act=desc_act,
    )
    _fill_layer(layer, qweight, qzeros, scales)

    expected_weight = dequantize_gemm(
        qweight=qweight,
        qzeros=qzeros,
        scales=scales,
        bits=4,
        group_size=group_size,
    )
    torch.testing.assert_close(
        layer.dequantize_weight(),
        expected_weight,
        atol=2e-3,
        rtol=2e-3,
    )

    expected_packed, expected_zeros = _awq_layout_oracle(
        qweight,
        qzeros,
        out_features,
        in_features,
    )
    layer.transform_cpu()
    assert layer.scales.dtype == torch.bfloat16
    assert torch.equal(layer.qweight, expected_packed)
    assert torch.equal(layer.qzeros, expected_zeros)

    packed = layer.qweight.clone()
    zero_codes = layer.qzeros.clone()
    layer.transform_cpu()
    assert torch.equal(layer.qweight, packed)
    assert torch.equal(layer.qzeros, zero_codes)

    restored = layer.dequantize_weight()
    torch.testing.assert_close(restored, expected_weight, atol=2e-3, rtol=2e-3)

    with pytest.raises(NotImplementedError, match="inference-only"):
        layer.train(True)


@pytest.mark.parametrize(
    ("tokens", "in_features", "out_features", "group_size"),
    (
        (1, 256, 64, 64),
        (7, 512, 128, 128),
        (31, 1024, 256, 128),
        (64, 2048, 512, 64),
    ),
)
def test_awq_layer_matches_dequantize_gemm_oracle(
    tokens,
    in_features,
    out_features,
    group_size,
):
    _skip_unless_avx512_bf16()
    torch.manual_seed(3000 + tokens + in_features + out_features)
    qweight, qzeros, scales = _make_awq_buffers(
        in_features,
        out_features,
        group_size,
    )
    layer = _make_layer(
        in_features=in_features,
        out_features=out_features,
        group_size=group_size,
    )
    _fill_layer(layer, qweight, qzeros, scales)

    x = torch.randn(tokens, in_features, dtype=torch.bfloat16)
    reference_weight = dequantize_gemm(
        qweight=qweight,
        qzeros=qzeros,
        scales=scales.to(torch.bfloat16),
        bits=4,
        group_size=group_size,
    )
    reference = torch.matmul(x.float(), reference_weight.float()).to(torch.bfloat16)
    output = layer(x)

    assert output.dtype == torch.bfloat16
    torch.testing.assert_close(output, reference, atol=0.2, rtol=0.01)
    relative_error = torch.linalg.vector_norm(
        (output.float() - reference.float()).flatten()
    ) / torch.linalg.vector_norm(reference.float().flatten())
    assert relative_error < 3e-3


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_awq_layer_preserves_activation_dtype_and_rank(dtype):
    _skip_unless_avx512_bf16()
    torch.manual_seed(4000 + (1 if dtype == torch.float16 else 0))
    in_features, out_features, group_size = 256, 64, 64
    qweight, qzeros, scales = _make_awq_buffers(
        in_features,
        out_features,
        group_size,
    )
    layer = _make_layer(
        in_features=in_features,
        out_features=out_features,
        group_size=group_size,
    )
    _fill_layer(layer, qweight, qzeros, scales)

    x = torch.randn(2, 3, 5, in_features, dtype=dtype)
    reference_weight = dequantize_gemm(
        qweight=qweight,
        qzeros=qzeros,
        scales=scales,
        bits=4,
        group_size=group_size,
    )
    reference = torch.matmul(x, reference_weight.to(dtype))
    output = layer(x)

    assert output.shape == (2, 3, 5, out_features)
    assert output.dtype == dtype
    torch.testing.assert_close(output, reference, atol=0.25, rtol=0.02)

    empty = torch.empty((0, in_features), dtype=dtype)
    empty_output = layer(empty)
    assert empty_output.shape == (0, out_features)
    assert empty_output.dtype == dtype


def test_awq_layer_adds_bias_after_shared_kernel():
    _skip_unless_avx512_bf16()
    torch.manual_seed(5000)
    in_features, out_features, group_size = 256, 64, 64
    qweight, qzeros, scales = _make_awq_buffers(
        in_features,
        out_features,
        group_size,
    )
    layer = _make_layer(
        in_features=in_features,
        out_features=out_features,
        group_size=group_size,
        bias=True,
    )
    _fill_layer(layer, qweight, qzeros, scales)
    bias = torch.randn(out_features, dtype=torch.float16)
    layer.bias.copy_(bias)

    x = torch.randn(7, in_features, dtype=torch.bfloat16)
    reference_weight = dequantize_gemm(
        qweight=qweight,
        qzeros=qzeros,
        scales=scales.to(torch.bfloat16),
        bits=4,
        group_size=group_size,
    )
    reference = (torch.matmul(x.float(), reference_weight.float()) + bias.float()).to(
        torch.bfloat16
    )
    output = layer(x)

    torch.testing.assert_close(output, reference, atol=0.2, rtol=0.01)


def test_awq_reuses_the_gptq_compute_kernel():
    _skip_unless_avx512_bf16()
    ok, err = TorchCpuAvx512Linear.cached_validate_once()
    if not ok:
        pytest.skip(str(err))

    assert (
        TorchCpuAvx512AwqLinear.gemm_int4_qzero_group_scaled_cpu_kernel
        is TorchCpuAvx512Linear.gemm_int4_qzero_group_scaled_cpu_kernel
    )


def test_awq_auto_selection_prefers_shared_avx512_kernel():
    if not has_avx512_bf16():
        pytest.skip(avx512_bf16_error())
    ok, err = TorchCpuAvx512AwqLinear.cached_validate_once()
    if not ok:
        pytest.skip(str(err))

    clear_validation_cache()
    selected = select_quant_linear(
        bits=4,
        group_size=128,
        desc_act=False,
        sym=True,
        device=DEVICE.CPU,
        backend=BACKEND.AUTO,
        format=FORMAT.GEMM,
        quant_method=METHOD.AWQ,
        pack_dtype=torch.int32,
    )
    assert selected is TorchCpuAvx512AwqLinear

    if TorchAtenAwqLinear.cached_validate_once()[0]:
        candidates = select_quant_linear(
            bits=4,
            group_size=128,
            desc_act=False,
            sym=True,
            device=DEVICE.CPU,
            backend=BACKEND.AUTO,
            format=FORMAT.GEMM,
            quant_method=METHOD.AWQ,
            pack_dtype=torch.int32,
            multi_select=True,
        )
        assert candidates.index(TorchCpuAvx512AwqLinear) < candidates.index(
            TorchAtenAwqLinear
        )
