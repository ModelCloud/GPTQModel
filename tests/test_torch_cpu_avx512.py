# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from gptqmodel.models._const import DEVICE
from gptqmodel.nn_modules.qlinear import torch_cpu_avx512 as avx512_module
from gptqmodel.nn_modules.qlinear.torch_aten_kernel import (
    TorchAtenLinear,
    _has_local_int4pack_cpu_ops,
)
from gptqmodel.nn_modules.qlinear.torch_cpu_avx512 import (
    TorchCpuAvx512Linear,
    pack_hf_int4,
    unpack_hf_int4,
)
from gptqmodel.quantization import FORMAT, METHOD
from gptqmodel.utils import cpu_features
from gptqmodel.utils.backend import BACKEND
from gptqmodel.utils.cpu_features import avx512_bf16_error, has_avx512_bf16
from gptqmodel.utils.importer import clear_validation_cache, select_quant_linear


pytestmark = [pytest.mark.cpu]


def _hf_reference_unpack(packed: torch.Tensor, out_features: int, in_features: int) -> torch.Tensor:
    """Independent implementation of the upstream HF unpacking formula."""

    block_n = 32
    qw = packed.reshape(-1, block_n)
    low = qw & 0x0F
    high = (qw >> 4) & 0x0F
    restored = torch.cat((low, high), dim=1)
    restored = restored.reshape(out_features // block_n, in_features // 2, block_n, 2)
    restored = restored.transpose(-3, -2)
    return restored.reshape(out_features, in_features).to(torch.uint8)


def _skip_unless_avx512_bf16() -> None:
    if not has_avx512_bf16():
        pytest.skip(avx512_bf16_error())
    ok, err = TorchCpuAvx512Linear.cached_validate_once()
    if not ok:
        pytest.skip(str(err))


def _packer_kernel():
    _skip_unless_avx512_bf16()
    return TorchCpuAvx512Linear.pack_gptq_int4_kernel


def _make_layer(
    *,
    in_features: int,
    out_features: int,
    group_size: int,
    desc_act: bool,
    bias: bool = False,
) -> TorchCpuAvx512Linear:
    layer = TorchCpuAvx512Linear(
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
    layer.post_init()
    return layer


def _make_aten_layer(
    *,
    in_features: int,
    out_features: int,
    group_size: int,
    desc_act: bool,
    bias: bool = False,
) -> TorchAtenLinear:
    layer = TorchAtenLinear(
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
    layer.post_init()
    return layer


def _skip_unless_aten_cpu() -> None:
    if not _has_local_int4pack_cpu_ops():
        pytest.skip("ATen CPU int4pack ops are unavailable")
    ok, err = TorchAtenLinear.cached_validate_once()
    if not ok:
        pytest.skip(str(err))


def _fill_gptq_buffers(
    layer: TorchCpuAvx512Linear,
    *,
    codes: torch.Tensor,
    zeros: torch.Tensor,
    scales: torch.Tensor,
    g_idx: torch.Tensor,
) -> None:
    """Fill GPTQ packed buffers from logical ``[in_features, out_features]`` codes."""

    pack_factor = layer.pack_factor
    in_features, out_features = codes.shape
    qweight = torch.zeros(
        (in_features // pack_factor, out_features),
        dtype=torch.int32,
    )
    for nibble in range(pack_factor):
        qweight |= (codes[nibble::pack_factor] & 0xF) << (4 * nibble)
    layer.qweight.copy_(qweight)

    groups, _ = zeros.shape
    qzeros = torch.zeros(
        (groups, out_features // pack_factor),
        dtype=torch.int32,
    )
    for nibble in range(pack_factor):
        qzeros |= (zeros[:, nibble::pack_factor] & 0xF) << (4 * nibble)
    layer.qzeros.copy_(qzeros)
    layer.scales = scales.contiguous()
    layer.g_idx.copy_(g_idx)


def _dequantize_oracle(
    codes: torch.Tensor,
    zeros: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Dequantize in FP32 so the oracle is independent of the kernel's BF16 path."""

    codes_kn = codes.t().to(torch.float32)
    zeros_kn = zeros.to(torch.float32).repeat_interleave(group_size, dim=0)
    scales_kn = scales.to(torch.float32).repeat_interleave(group_size, dim=0)
    return (codes_kn - zeros_kn) * scales_kn


def _layer_dequantize_oracle(
    codes: torch.Tensor,
    zeros: torch.Tensor,
    scales: torch.Tensor,
    g_idx: torch.Tensor,
) -> torch.Tensor:
    """Reference GPTQ dequantization in the checkpoint's channel order."""

    groups = g_idx.to(torch.long)
    return (
        (codes.float() - zeros.index_select(0, groups).float())
        * scales.index_select(0, groups).float()
    )


def _aten_affine_oracle(
    codes: torch.Tensor,
    zeros: torch.Tensor,
    scales: torch.Tensor,
    g_idx: torch.Tensor,
) -> torch.Tensor:
    """Reference for ATen int4pack's signed-code plus zero-offset contract.

    ``_cpu_int4pack_zero_offsets`` materializes ``(8 - zero_code) * scale`` in
    the scale dtype. The ATen GEMM then evaluates ``signed_code * scale +
    zero_offset``. Keep that rounded zero-offset term in the oracle instead of
    folding it into an exact ``(code - zero) * scale`` expression.
    """

    groups = g_idx.to(torch.long)
    scale = scales.index_select(0, groups).to(torch.float32)
    zero_offset = (
        (8 - zeros).to(dtype=scales.dtype) * scales
    ).index_select(0, groups).to(torch.float32)
    return (codes.to(torch.float32) - 8.0) * scale + zero_offset


def _packed_transform_oracle(layer: TorchCpuAvx512Linear):
    """Reference checkpoint-to-HF conversion using only Torch operations."""

    weight = _unpack_qweight_codes(layer)
    weight = (
        weight.index_select(0, layer._build_ret_idx())
        .t()
        .contiguous()
    )
    zero_codes = torch.bitwise_right_shift(
        torch.unsqueeze(layer.qzeros, 2).expand(-1, -1, layer.pack_factor),
        layer.wf_unsqueeze_zero,
    ).to(torch.uint8)
    zero_codes = (
        torch.bitwise_and(zero_codes, layer.maxq)
        .reshape(layer.scales.shape)
        .contiguous()
    )
    return pack_hf_int4(weight), zero_codes


def _unpack_qweight_codes(layer: TorchCpuAvx512Linear) -> torch.Tensor:
    weight = torch.bitwise_and(
        torch.bitwise_right_shift(
            torch.unsqueeze(layer.qweight, 1).expand(-1, layer.pack_factor, -1),
            layer.wf_unsqueeze_neg_one,
        ).to(torch.uint8),
        layer.maxq,
    )
    return weight.reshape(
        weight.shape[0] * weight.shape[1],
        weight.shape[2],
    )


def test_cpu_feature_detection_requires_vbmi(monkeypatch):
    flags = (
        "flags : xsave avx512f avx512bw avx512dq avx512vl "
        "avx512_bf16 avx512_vnni"
    )
    monkeypatch.setattr(cpu_features.platform, "system", lambda: "Linux")
    monkeypatch.setattr(cpu_features.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(Path, "read_text", lambda self, *args, **kwargs: flags)

    has_avx512_bf16.cache_clear()
    try:
        assert not has_avx512_bf16()
        monkeypatch.setattr(
            Path,
            "read_text",
            lambda self, *args, **kwargs: flags + " avx512vbmi",
        )
        has_avx512_bf16.cache_clear()
        assert has_avx512_bf16()
    finally:
        has_avx512_bf16.cache_clear()


def test_cpu_feature_detection_requires_xsave(monkeypatch):
    flags = (
        "flags : avx512f avx512bw avx512dq avx512vl "
        "avx512_bf16 avx512_vnni avx512vbmi"
    )
    monkeypatch.setattr(cpu_features.platform, "system", lambda: "Linux")
    monkeypatch.setattr(cpu_features.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(Path, "read_text", lambda self, *args, **kwargs: flags)

    has_avx512_bf16.cache_clear()
    try:
        assert not has_avx512_bf16()
    finally:
        has_avx512_bf16.cache_clear()


@pytest.mark.parametrize(
    ("out_features", "in_features"),
    ((32, 64), (64, 128), (96, 256), (128, 512)),
)
def test_hf_pack_layout_matches_upstream(out_features, in_features):
    torch.manual_seed(1234 + out_features + in_features)
    codes = torch.randint(0, 16, (out_features, in_features), dtype=torch.uint8)
    packed = pack_hf_int4(codes)

    assert packed.shape == (out_features, in_features // 2)
    assert torch.equal(_hf_reference_unpack(packed, out_features, in_features), codes)
    assert torch.equal(unpack_hf_int4(packed, out_features, in_features), codes)

    random_packed = torch.randint(
        0,
        256,
        (out_features, in_features // 2),
        dtype=torch.uint8,
    )
    assert torch.equal(
        unpack_hf_int4(random_packed, out_features, in_features),
        _hf_reference_unpack(random_packed, out_features, in_features),
    )


def test_hf_pack_rejects_invalid_shapes_and_dtypes():
    with pytest.raises(ValueError, match="divisible by 32"):
        pack_hf_int4(torch.zeros((33, 64), dtype=torch.uint8))
    with pytest.raises(ValueError, match="must be even"):
        pack_hf_int4(torch.zeros((32, 63), dtype=torch.uint8))
    with pytest.raises(TypeError, match="uint8"):
        pack_hf_int4(torch.zeros((32, 64), dtype=torch.int32))
    with pytest.raises(ValueError, match="packed shape"):
        unpack_hf_int4(torch.zeros((32, 31), dtype=torch.uint8), 32, 64)


def test_fused_packer_handles_identity_and_reordered_inputs():
    kernel = _packer_kernel()
    out_features, in_features, group_size = 64, 256, 64
    groups = in_features // group_size
    torch.manual_seed(4321)
    qweight = torch.randint(
        -(2**31),
        2**31,
        (in_features // 8, out_features),
        dtype=torch.int32,
    )
    qzeros = torch.randint(
        0,
        2**31,
        (groups, out_features // 8),
        dtype=torch.int32,
    )

    layer = _make_layer(
        in_features=in_features,
        out_features=out_features,
        group_size=group_size,
        desc_act=True,
    )
    layer.qweight.copy_(qweight)
    layer.qzeros.copy_(qzeros)
    reordered = torch.randperm(in_features, dtype=torch.int32)
    _, expected_zeros = _packed_transform_oracle(layer)
    expected_packed = pack_hf_int4(
        _unpack_qweight_codes(layer)
        .index_select(0, reordered)
        .t()
        .contiguous()
    )

    packed, zeros = kernel(
        qweight,
        qzeros,
        reordered,
        out_features,
        in_features,
    )
    assert torch.equal(packed, expected_packed)
    assert torch.equal(zeros, expected_zeros)

    packed_identity, zeros_identity = kernel(
        qweight,
        qzeros,
        torch.empty(0, dtype=torch.int32),
        out_features,
        in_features,
    )
    packed_explicit, zeros_explicit = kernel(
        qweight,
        qzeros,
        torch.arange(in_features, dtype=torch.int32),
        out_features,
        in_features,
    )
    assert torch.equal(packed_identity, packed_explicit)
    assert torch.equal(zeros_identity, zeros_explicit)


def test_fused_packer_rejects_invalid_inputs():
    kernel = _packer_kernel()
    out_features, in_features, group_size = 64, 256, 64
    groups = in_features // group_size
    qweight = torch.zeros((in_features // 8, out_features), dtype=torch.int32)
    qzeros = torch.zeros((groups, out_features // 8), dtype=torch.int32)
    ret_idx = torch.arange(in_features, dtype=torch.int32)

    with pytest.raises(RuntimeError, match="qweight must be int32"):
        kernel(qweight.float(), qzeros, ret_idx, out_features, in_features)
    with pytest.raises(RuntimeError, match="qzeros must be int32"):
        kernel(qweight, qzeros.long(), ret_idx, out_features, in_features)
    with pytest.raises(RuntimeError, match="qweight must be 2D"):
        kernel(qweight.unsqueeze(0), qzeros, ret_idx, out_features, in_features)
    with pytest.raises(RuntimeError, match="qweight must be contiguous"):
        kernel(
            qweight.repeat_interleave(2, dim=1)[:, ::2],
            qzeros,
            ret_idx,
            out_features,
            in_features,
        )
    with pytest.raises(RuntimeError, match="divisible by 32"):
        kernel(qweight, qzeros, ret_idx, 33, in_features)
    with pytest.raises(RuntimeError, match="divisible by 8"):
        kernel(qweight, qzeros, ret_idx, out_features, 252)
    with pytest.raises(RuntimeError, match="qweight feature dimension"):
        kernel(qweight[:-1], qzeros, ret_idx, out_features, in_features)
    with pytest.raises(
        RuntimeError,
        match="qzeros output dimension does not match out_features",
    ):
        kernel(
            qweight,
            torch.zeros((groups, out_features // 8 + 1), dtype=torch.int32),
            ret_idx,
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
            ret_idx,
            out_features,
            in_features,
        )
    with pytest.raises(RuntimeError, match="qzeros group count must be > 0"):
        kernel(
            qweight,
            torch.zeros((0, out_features // 8), dtype=torch.int32),
            ret_idx,
            out_features,
            in_features,
        )
    with pytest.raises(
        RuntimeError,
        match="qzeros group count must divide in_features",
    ):
        kernel(
            qweight,
            torch.zeros((in_features + group_size, out_features // 8), dtype=torch.int32),
            ret_idx,
            out_features,
            in_features,
        )
    with pytest.raises(RuntimeError, match="qweight output dimension"):
        kernel(
            torch.zeros((in_features // 8, out_features + 1), dtype=torch.int32),
            qzeros,
            ret_idx,
            out_features,
            in_features,
        )
    with pytest.raises(RuntimeError, match="qzeros must be 2D"):
        kernel(qweight, qzeros.flatten(), ret_idx, out_features, in_features)
    with pytest.raises(RuntimeError, match="qzeros must be contiguous"):
        kernel(
            qweight,
            qzeros.repeat_interleave(2, dim=1)[:, ::2],
            ret_idx,
            out_features,
            in_features,
        )
    with pytest.raises(RuntimeError, match="ret_idx must be int32"):
        kernel(qweight, qzeros, ret_idx.long(), out_features, in_features)
    with pytest.raises(RuntimeError, match="ret_idx must be 1D"):
        kernel(
            qweight,
            qzeros,
            ret_idx.unsqueeze(0),
            out_features,
            in_features,
        )
    with pytest.raises(RuntimeError, match="ret_idx size must match in_features"):
        kernel(qweight, qzeros, ret_idx[:-1], out_features, in_features)
    with pytest.raises(RuntimeError, match="ret_idx must be contiguous"):
        kernel(
            qweight,
            qzeros,
            torch.arange(in_features * 2, dtype=torch.int32)[::2],
            out_features,
            in_features,
        )
    with pytest.raises(
        RuntimeError,
        match=r"ret_idx values must be in \[0, in_features\)",
    ):
        kernel(
            qweight,
            qzeros,
            torch.full((in_features,), -1, dtype=torch.int32),
            out_features,
            in_features,
        )
    with pytest.raises(
        RuntimeError,
        match=r"ret_idx values must be in \[0, in_features\)",
    ):
        kernel(
            qweight,
            qzeros,
            torch.full((in_features,), in_features, dtype=torch.int32),
            out_features,
            in_features,
        )


def test_parallel_thread_count_follows_torch_intraop_budget():
    _skip_unless_avx512_bf16()
    thread_count = torch.ops.gptqmodel_int4_qzero_group_scaled_cpu._parallel_thread_count
    probe = torch.empty(0)
    previous = torch.get_num_threads()
    try:
        for configured in (1, 2, 3, 8):
            torch.set_num_threads(configured)
            assert thread_count(probe, 1) == configured
            assert thread_count(probe, 32) == max(1, (configured >> 1) * 2)
    finally:
        torch.set_num_threads(previous)

    with pytest.raises(RuntimeError, match="rows must be >= 0"):
        thread_count(probe, -1)


def test_unsupported_host_disables_backend_and_falls_back(monkeypatch):
    monkeypatch.setattr(avx512_module, "has_avx512_bf16", lambda: False)
    monkeypatch.setattr(
        avx512_module,
        "avx512_bf16_error",
        lambda: "AVX512-BF16 is unavailable in this test",
    )
    TorchCpuAvx512Linear.cached_validate_once.cache_clear()
    TorchCpuAvx512Linear.gemm_int4_qzero_group_scaled_cpu_kernel = None
    TorchCpuAvx512Linear.pack_gptq_int4_kernel = None
    clear_validation_cache()
    try:
        ok, error = TorchCpuAvx512Linear.validate(
            bits=4,
            group_size=128,
            desc_act=False,
            sym=True,
            in_features=256,
            out_features=64,
            pack_dtype=torch.int32,
            device=DEVICE.CPU,
        )
        assert not ok
        assert "unavailable" in str(error)

        if TorchAtenLinear.cached_validate_once()[0]:
            selected = select_quant_linear(
                bits=4,
                group_size=128,
                desc_act=False,
                sym=True,
                device=DEVICE.CPU,
                backend=BACKEND.AUTO,
                format=FORMAT.GPTQ_V2,
                quant_method=METHOD.GPTQ,
                pack_dtype=torch.int32,
            )
            assert selected is TorchAtenLinear
    finally:
        TorchCpuAvx512Linear.cached_validate_once.cache_clear()
        TorchCpuAvx512Linear.gemm_int4_qzero_group_scaled_cpu_kernel = None
        TorchCpuAvx512Linear.pack_gptq_int4_kernel = None
        clear_validation_cache()


@pytest.mark.parametrize(
    ("tokens", "in_features", "out_features", "group_size"),
    (
        (1, 512, 64, 16),
        (1, 256, 64, 32),
        (7, 512, 128, 32),
        (7, 1024, 128, 256),
        (7, 1024, 128, 512),
        (1, 256, 64, 64),
        (7, 512, 128, 128),
        (25, 1024, 128, 256),
        (25, 1024, 128, 512),
        (25, 512, 128, 128),
        (64, 1024, 256, 128),
        (128, 2048, 512, 64),
    ),
)
def test_kernel_matches_independent_bf16_oracle(
    tokens,
    in_features,
    out_features,
    group_size,
):
    _skip_unless_avx512_bf16()
    torch.manual_seed(1000 + tokens + in_features + out_features)

    groups = in_features // group_size
    codes = torch.randint(0, 16, (out_features, in_features), dtype=torch.uint8)
    zeros = torch.randint(0, 16, (groups, out_features), dtype=torch.uint8)
    scales = torch.rand((groups, out_features), dtype=torch.float32) * 0.04 + 0.01
    x = torch.randn(tokens, in_features, dtype=torch.bfloat16)

    weight = _dequantize_oracle(codes, zeros, scales, group_size)
    reference_fp32 = torch.matmul(x.float(), weight)
    reference = reference_fp32.to(torch.bfloat16)
    output = TorchCpuAvx512Linear.gemm_int4_qzero_group_scaled_cpu_kernel(
        x,
        pack_hf_int4(codes),
        zeros,
        scales,
        group_size,
    )

    assert output.dtype == torch.bfloat16
    torch.testing.assert_close(output, reference, atol=0.2, rtol=0.01)
    output_error = (output.float() - reference.float()).norm() / reference.float().norm()
    arithmetic_error = (output.float() - reference_fp32).norm() / reference_fp32.norm()
    assert output_error.item() < 2e-3
    assert arithmetic_error.item() < 2e-3


def test_layer_fp16_dense_fallback_preserves_input_dtype():
    _skip_unless_avx512_bf16()
    torch.manual_seed(77)
    tokens, in_features, out_features, group_size = 7, 512, 128, 128
    groups = in_features // group_size
    layer = _make_layer(
        in_features=in_features,
        out_features=out_features,
        group_size=group_size,
        desc_act=False,
    )
    codes = torch.randint(0, 16, (in_features, out_features), dtype=torch.int32)
    zeros = torch.randint(0, 16, (groups, out_features), dtype=torch.int32)
    scales = torch.rand((groups, out_features), dtype=torch.float16) * 0.04 + 0.01
    g_idx = torch.arange(in_features, dtype=torch.int32) // group_size
    _fill_gptq_buffers(
        layer,
        codes=codes,
        zeros=zeros,
        scales=scales,
        g_idx=g_idx,
    )
    weight = layer.dequantize_weight()
    x = torch.randn(tokens, in_features, dtype=torch.float16)

    layer.eval()
    reference = torch.matmul(x.float(), weight.float()).to(torch.float16)
    output = layer(x)

    assert output.dtype == torch.float16
    assert layer.linear_mode == "train"
    assert layer._cpu_transformed is False
    torch.testing.assert_close(output, reference, atol=0.25, rtol=0.02)


@pytest.mark.parametrize(
    ("dtype", "checkpoint_scale_dtype"),
    (
        (torch.float16, torch.float16),
        (torch.bfloat16, torch.float16),
    ),
)
def test_aten_cpu_fused_forward_matches_fp32_oracle(dtype, checkpoint_scale_dtype):
    _skip_unless_aten_cpu()
    torch.manual_seed(79 if dtype == torch.float16 else 80)
    tokens, in_features, out_features, group_size = 7, 512, 128, 128
    groups = in_features // group_size
    layer = _make_aten_layer(
        in_features=in_features,
        out_features=out_features,
        group_size=group_size,
        desc_act=False,
    )
    codes = torch.randint(0, 16, (in_features, out_features), dtype=torch.int32)
    zeros = torch.randint(0, 16, (groups, out_features), dtype=torch.int32)
    scales = torch.rand((groups, out_features), dtype=checkpoint_scale_dtype) * 0.04 + 0.01
    g_idx = torch.arange(in_features, dtype=torch.int32) // group_size
    _fill_gptq_buffers(
        layer,
        codes=codes,
        zeros=zeros,
        scales=scales,
        g_idx=g_idx,
    )

    x = torch.randn(tokens, in_features, dtype=dtype)
    # ATen int4pack requires scales/zero-points to share the activation dtype.
    # The FP16 checkpoint scales are therefore rounded to BF16 for a BF16
    # activation. Use that same representation in the arithmetic oracle so
    # this test isolates kernel error from the intentional input conversion.
    arithmetic_scales = scales.to(dtype)
    weight = _aten_affine_oracle(codes, zeros, arithmetic_scales, g_idx)
    reference_fp32 = torch.matmul(x.float(), weight)

    layer.eval()
    with torch.inference_mode():
        output = layer(x)

    assert output.dtype == dtype
    assert layer.linear_mode == "inference"
    assert layer._cpu_transformed is True
    assert layer._cpu_transform_dtype == dtype
    assert layer.scales.dtype == dtype
    torch.testing.assert_close(output, reference_fp32.to(dtype), atol=0.2, rtol=0.01)
    normalized_error = (output.float() - reference_fp32).norm() / reference_fp32.norm()
    assert normalized_error.item() < 2e-3


@pytest.mark.parametrize(
    ("first_dtype", "second_dtype"),
    (
        (torch.float16, torch.bfloat16),
        (torch.bfloat16, torch.float16),
    ),
)
def test_aten_cpu_rejects_mixed_activation_dtype_after_transform(first_dtype, second_dtype):
    _skip_unless_aten_cpu()
    torch.manual_seed(81)
    in_features, out_features, group_size = 256, 64, 64
    groups = in_features // group_size
    layer = _make_aten_layer(
        in_features=in_features,
        out_features=out_features,
        group_size=group_size,
        desc_act=False,
    )
    codes = torch.randint(0, 16, (in_features, out_features), dtype=torch.int32)
    zeros = torch.randint(0, 16, (groups, out_features), dtype=torch.int32)
    scales = torch.rand((groups, out_features), dtype=torch.float16) * 0.04 + 0.01
    g_idx = torch.arange(in_features, dtype=torch.int32) // group_size
    _fill_gptq_buffers(
        layer,
        codes=codes,
        zeros=zeros,
        scales=scales,
        g_idx=g_idx,
    )
    layer.eval()

    with torch.inference_mode():
        layer(torch.randn(1, in_features, dtype=first_dtype))
        with pytest.raises(RuntimeError, match="activation dtype|received"):
            layer(torch.randn(1, in_features, dtype=second_dtype))


def test_layer_fused_bias_matches_independent_fp32_oracle():
    _skip_unless_avx512_bf16()
    torch.manual_seed(78)
    tokens, in_features, out_features, group_size = 7, 512, 128, 128
    groups = in_features // group_size
    layer = _make_layer(
        in_features=in_features,
        out_features=out_features,
        group_size=group_size,
        desc_act=False,
        bias=True,
    )
    codes = torch.randint(0, 16, (in_features, out_features), dtype=torch.int32)
    zeros = torch.randint(0, 16, (groups, out_features), dtype=torch.int32)
    scales = torch.rand((groups, out_features), dtype=torch.float32) * 0.04 + 0.01
    bias = torch.randn(out_features, dtype=torch.float16)
    g_idx = torch.arange(in_features, dtype=torch.int32) // group_size
    _fill_gptq_buffers(
        layer,
        codes=codes,
        zeros=zeros,
        scales=scales,
        g_idx=g_idx,
    )
    layer.bias.copy_(bias)

    x = torch.randn(tokens, in_features, dtype=torch.bfloat16)
    weight = _layer_dequantize_oracle(codes, zeros, scales, g_idx)
    reference_fp32 = torch.matmul(x.float(), weight) + bias.float()
    reference = reference_fp32.to(torch.bfloat16)

    layer.eval()
    with torch.inference_mode():
        output = layer(x)

    assert output.dtype == torch.bfloat16
    assert layer.linear_mode == "inference"
    assert layer._cpu_transformed is True
    assert layer.bias.dtype == torch.float32
    torch.testing.assert_close(output, reference, atol=0.2, rtol=0.01)
    normalized_error = (output.float() - reference_fp32).norm() / reference_fp32.norm()
    assert normalized_error.item() < 2e-3


def test_fused_bias_validation():
    _skip_unless_avx512_bf16()
    out_features, in_features, group_size = 64, 256, 128
    groups = in_features // group_size
    codes = torch.zeros((out_features, in_features), dtype=torch.uint8)
    zeros = torch.zeros((groups, out_features), dtype=torch.uint8)
    scales = torch.ones((groups, out_features), dtype=torch.float32)
    x = torch.zeros((1, in_features), dtype=torch.bfloat16)
    packed = pack_hf_int4(codes)
    kernel = TorchCpuAvx512Linear.gemm_int4_qzero_group_scaled_cpu_kernel

    with pytest.raises(RuntimeError, match="bias must be float32"):
        kernel(x, packed, zeros, scales, group_size, torch.zeros(out_features, dtype=torch.float16))
    with pytest.raises(RuntimeError, match="bias must be 1D"):
        kernel(x, packed, zeros, scales, group_size, torch.zeros((1, out_features)))
    with pytest.raises(RuntimeError, match="bias must be contiguous"):
        kernel(
            x,
            packed,
            zeros,
            scales,
            group_size,
            torch.zeros((out_features * 2,))[::2],
        )
    with pytest.raises(RuntimeError, match="bias out_features"):
        kernel(x, packed, zeros, scales, group_size, torch.zeros(out_features + 1))
    with pytest.raises(RuntimeError, match="Meta tensors"):
        kernel(
            x,
            packed,
            zeros,
            scales,
            group_size,
            torch.empty(out_features, device="meta"),
        )


def test_kernel_empty_batch_and_validation():
    _skip_unless_avx512_bf16()
    out_features, in_features, group_size = 64, 256, 128
    groups = in_features // group_size
    codes = torch.randint(0, 16, (out_features, in_features), dtype=torch.uint16).to(torch.uint8)
    zeros = torch.zeros((groups, out_features), dtype=torch.uint16).to(torch.uint8)
    scales = torch.ones((groups, out_features), dtype=torch.float32)
    packed = pack_hf_int4(codes)
    kernel = TorchCpuAvx512Linear.gemm_int4_qzero_group_scaled_cpu_kernel

    output = kernel(
        torch.empty((0, in_features), dtype=torch.bfloat16),
        packed,
        zeros,
        scales,
        group_size,
    )
    assert output.shape == (0, out_features)
    assert output.dtype == torch.bfloat16

    with pytest.raises(RuntimeError, match="input must be bfloat16"):
        kernel(
            torch.empty((1, in_features), dtype=torch.float32),
            packed,
            zeros,
            scales,
            group_size,
        )
    with pytest.raises(RuntimeError, match="divisible by blocksize"):
        kernel(
            torch.empty((1, in_features), dtype=torch.bfloat16),
            packed,
            zeros,
            scales,
            group_size + 1,
        )


@pytest.mark.parametrize("desc_act", (False, True))
def test_layer_transform_dequantize_and_forward(desc_act):
    _skip_unless_avx512_bf16()
    torch.manual_seed(9 if desc_act else 8)
    in_features, out_features, group_size = 256, 64, 64
    groups = in_features // group_size

    layer = _make_layer(
        in_features=in_features,
        out_features=out_features,
        group_size=group_size,
        desc_act=desc_act,
    )
    codes = torch.randint(0, 16, (in_features, out_features), dtype=torch.int32)
    zeros = torch.randint(0, 15, (groups, out_features), dtype=torch.int32)
    scales = torch.rand((groups, out_features), dtype=torch.float16) * 0.04 + 0.01
    if desc_act:
        g_idx = (torch.arange(in_features, dtype=torch.int32) * 7) % groups
    else:
        g_idx = torch.arange(in_features, dtype=torch.int32) // group_size

    _fill_gptq_buffers(
        layer,
        codes=codes,
        zeros=zeros,
        scales=scales,
        g_idx=g_idx,
    )
    assert layer._ret_idx_is_identity() is (not desc_act)
    if not desc_act:
        assert torch.equal(
            layer.ret_idx,
            torch.arange(in_features, dtype=torch.int32),
        )
    expected_qweight, expected_qzeros = _packed_transform_oracle(layer)
    reference_weight = layer.dequantize_weight()

    layer.transform_cpu()
    assert torch.equal(layer.qweight, expected_qweight)
    assert torch.equal(layer.qzeros, expected_qzeros)
    packed_weight = layer.qweight.clone()
    layer.transform_cpu()
    assert torch.equal(packed_weight, layer.qweight)

    restored_weight = layer.dequantize_weight()
    torch.testing.assert_close(restored_weight, reference_weight, atol=2e-3, rtol=2e-3)

    layer.eval()
    x = torch.randn(5, in_features, dtype=torch.bfloat16)
    output = layer(x)
    reference = torch.matmul(x.float(), reference_weight.float()).to(torch.bfloat16)
    torch.testing.assert_close(output, reference, atol=0.2, rtol=0.01)

    with pytest.raises(NotImplementedError, match="inference-only"):
        layer.train(True)


@pytest.mark.parametrize("desc_act", (False, True))
def test_layer_dequantize_matches_independent_checkpoint_oracle(desc_act):
    _skip_unless_avx512_bf16()
    torch.manual_seed(91 if desc_act else 90)
    in_features, out_features, group_size = 256, 64, 64
    groups = in_features // group_size

    layer = _make_layer(
        in_features=in_features,
        out_features=out_features,
        group_size=group_size,
        desc_act=desc_act,
    )
    codes = torch.randint(0, 16, (in_features, out_features), dtype=torch.int32)
    zeros = torch.randint(0, 15, (groups, out_features), dtype=torch.int32)
    scales = torch.rand((groups, out_features), dtype=torch.bfloat16) * 0.04 + 0.01
    if desc_act:
        g_idx = (torch.arange(in_features, dtype=torch.int32) * 7) % groups
    else:
        g_idx = torch.arange(in_features, dtype=torch.int32) // group_size

    _fill_gptq_buffers(
        layer,
        codes=codes,
        zeros=zeros,
        scales=scales,
        g_idx=g_idx,
    )
    oracle = _layer_dequantize_oracle(codes, zeros, scales, g_idx)

    before = layer.dequantize_weight().float()
    torch.testing.assert_close(before, oracle, atol=2e-3, rtol=2e-3)

    layer.transform_cpu()
    after_raw = layer.dequantize_weight()
    assert after_raw.dtype == torch.bfloat16
    after = after_raw.float()
    torch.testing.assert_close(after, oracle, atol=2e-3, rtol=2e-3)


def test_auto_selection_and_shape_fallback():
    if not has_avx512_bf16():
        pytest.skip(avx512_bf16_error())
    selected = select_quant_linear(
        bits=4,
        group_size=128,
        desc_act=False,
        sym=True,
        device=DEVICE.CPU,
        backend=BACKEND.AUTO,
        format=FORMAT.GPTQ_V2,
        quant_method=METHOD.GPTQ,
        pack_dtype=torch.int32,
    )
    assert selected is TorchCpuAvx512Linear

    ok, _ = TorchCpuAvx512Linear.validate(
        bits=4,
        group_size=128,
        desc_act=False,
        sym=True,
        in_features=256,
        out_features=33,
        pack_dtype=torch.int32,
        device=DEVICE.CPU,
    )
    assert not ok

    ok, _ = TorchCpuAvx512Linear.validate(
        bits=4,
        group_size=128,
        desc_act=False,
        sym=True,
        in_features=250,
        out_features=64,
        pack_dtype=torch.int32,
        device=DEVICE.CPU,
    )
    assert not ok

    if TorchAtenLinear.cached_validate_once()[0]:
        aten = select_quant_linear(
            bits=4,
            group_size=128,
            desc_act=False,
            sym=True,
            device=DEVICE.CPU,
            backend=BACKEND.GPTQ_TORCH_ATEN,
            format=FORMAT.GPTQ_V2,
            quant_method=METHOD.GPTQ,
            pack_dtype=torch.int32,
        )
        assert aten is TorchAtenLinear


@pytest.mark.parametrize("group_size", (256, 512))
def test_auto_selection_supports_large_groups(group_size):
    if not has_avx512_bf16():
        pytest.skip(avx512_bf16_error())

    selected = select_quant_linear(
        bits=4,
        group_size=group_size,
        desc_act=False,
        sym=True,
        device=DEVICE.CPU,
        backend=BACKEND.AUTO,
        format=FORMAT.GPTQ_V2,
        quant_method=METHOD.GPTQ,
        pack_dtype=torch.int32,
    )
    assert selected is TorchCpuAvx512Linear


def test_aten_cpu_group_size_contract_matches_the_op():
    assert TorchAtenLinear.SUPPORTS_GROUP_SIZE == [32, 64, 128, 256]

    if not _has_local_int4pack_cpu_ops():
        pytest.skip("aten CPU int4pack ops are unavailable")

    ok, err = TorchAtenLinear.validate(
        bits=4,
        group_size=16,
        desc_act=False,
        sym=True,
        in_features=512,
        out_features=64,
        pack_dtype=torch.int32,
        device=DEVICE.CPU,
    )
    assert not ok
    assert "group_size" in str(err)

    ok, err = TorchAtenLinear.validate(
        bits=4,
        group_size=256,
        desc_act=False,
        sym=True,
        in_features=512,
        out_features=64,
        pack_dtype=torch.int32,
        device=DEVICE.CPU,
    )
    assert ok, err
