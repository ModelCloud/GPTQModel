# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Kernel-memory regressions for issue #4387 items 1 and 4.

Item 1: QQQ's INT32 global-reduce scratch is one lazily grown buffer per
device, not one per module (5.25 GiB on Llama-3-8B).  Item 4: the portable
Torch QQQ kernel unpacks INT4 codes without the old INT64/FP32 per-forward
temporaries.  Neither test needs the QQQ CUDA extension.
"""

import pytest
import torch

from gptqmodel.nn_modules.qlinear import qqq as qqq_module
from gptqmodel.nn_modules.qlinear.qqq import QQQLinear, QQQTorchLinear
from gptqmodel.utils.rocm import IS_ROCM


_OLD_DIVISORS = (1, 16, 256, 4096, 65536, 1048576, 16777216, 268435456)


def _build_torch_module(
    *,
    in_features: int = 256,
    out_features: int = 128,
    group_size: int = 128,
    dtype: torch.dtype = torch.float16,
) -> QQQTorchLinear:
    torch.manual_seed(730 + in_features + group_size)
    linear = torch.nn.Linear(in_features, out_features, bias=False, dtype=torch.float16)
    linear.weight.data.normal_(0, 0.08)
    groups = in_features // group_size if group_size != -1 else 1
    scales = torch.rand((out_features, groups), dtype=torch.float16) * 0.02 + 0.01
    s_extra = (
        torch.rand(out_features, dtype=torch.float32) * 0.001 + 0.001
        if group_size != -1
        else None
    )
    module = QQQTorchLinear(
        bits=4,
        group_size=group_size,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=False,
        register_buffers=True,
    )
    module.pack(linear, scales, s_extra)
    module.post_init()
    return module.eval().to(dtype=dtype)


def _oracle_int8_weight(module: QQQTorchLinear) -> torch.Tensor:
    """Reproduce the pre-#4387 INT64 unpack and dequantization independently."""

    packed = module.B
    packed_i64 = packed.to(torch.int64)
    packed_u32 = torch.where(packed_i64 < 0, packed_i64 + (1 << 32), packed_i64)
    divisors = torch.tensor(_OLD_DIVISORS, dtype=torch.int64, device=packed.device)
    quotients = torch.floor_divide(packed_u32.unsqueeze(-1), divisors)
    lanes = (quotients - torch.floor_divide(quotients, 16) * 16).reshape(
        packed.shape[0], packed.shape[1] * 8
    )
    perm_i = module._perm_i
    codes = lanes.reshape(-1, perm_i.numel())[:, perm_i].reshape(
        module.in_features // module.tile, module.out_features * module.tile
    )
    codes = (
        codes.reshape(
            module.in_features // module.tile,
            module.out_features // module.tile,
            module.tile,
            module.tile,
        )
        .permute(0, 2, 1, 3)
        .reshape(module.in_features, module.out_features)
    )
    if module.group_size != module.in_features:
        s_group = module._unpermute_scales(module.s_group, module._scale_perm_i).float()
        group_idx = torch.floor_divide(
            torch.arange(module.in_features, dtype=torch.int64, device=codes.device),
            module.group_size,
        )
        weight = ((codes.float() - 8.0) * s_group[group_idx]).round().clamp(-128, 127)
        return weight.to(torch.int8)
    signed = torch.where(codes >= 8, codes - 16, codes)
    return (signed * 16).to(torch.int8)


def test_qqq_reduce_scratch_is_one_shared_per_device_buffer():
    qqq_module._clear_reduce_scratch()
    device = torch.device("cpu")

    narrow = qqq_module._shared_reduce_scratch(device, 128)
    narrower = qqq_module._shared_reduce_scratch(device, 64)
    assert narrow.shape == (1024, 128)
    assert narrower.shape == (1024, 64)
    assert narrow.is_contiguous() and narrower.is_contiguous()
    assert narrow.untyped_storage().data_ptr() == narrower.untyped_storage().data_ptr()

    wide = qqq_module._shared_reduce_scratch(device, 256)
    assert wide.shape == (1024, 256)
    assert wide.numel() == 1024 * 256
    # Growing replaces the allocation, and later callers share the new one.
    after_growth = qqq_module._shared_reduce_scratch(device, 128)
    assert after_growth.untyped_storage().data_ptr() == wide.untyped_storage().data_ptr()


def test_qqq_linear_modules_share_the_scratch_and_register_no_reduce_buffer(monkeypatch):
    qqq_module._clear_reduce_scratch()
    monkeypatch.setattr(
        QQQLinear, "cached_validate_once", classmethod(lambda cls: (True, None))
    )
    modules = [
        QQQLinear(
            bits=4,
            group_size=128,
            sym=True,
            desc_act=False,
            in_features=128,
            out_features=128,
            register_buffers=False,
        )
        for _ in range(2)
    ]
    for module in modules:
        assert "reduce_buffer" not in module._buffers
        assert all("reduce_buffer" not in name for name, _ in module.named_buffers())
    scratch = [module._reduce_scratch(torch.device("cpu")) for module in modules]
    assert scratch[0].untyped_storage().data_ptr() == scratch[1].untyped_storage().data_ptr()
    assert scratch[0].shape == (modules[0].max_par * 64, modules[0].out_features)


def test_qqq_native_forward_passes_the_shared_scratch_as_c(monkeypatch):
    qqq_module._clear_reduce_scratch()
    monkeypatch.setattr(
        QQQLinear, "cached_validate_once", classmethod(lambda cls: (True, None))
    )
    monkeypatch.setattr(qqq_module, "qqq_runtime_available", lambda: True)
    calls = []

    def fake_gemm(A, B, C, D, s1, s2, s3, workspace, thread_k=-1, thread_n=-1, sms=-1, max_par=16):
        calls.append((tuple(C.shape), C.untyped_storage().data_ptr()))
        D.zero_()

    monkeypatch.setattr(qqq_module, "qqq_gemm", fake_gemm)
    wide = QQQLinear(
        bits=4, group_size=128, sym=True, desc_act=False,
        in_features=128, out_features=256, register_buffers=True,
    )
    narrow = QQQLinear(
        bits=4, group_size=128, sym=True, desc_act=False,
        in_features=128, out_features=128, register_buffers=True,
    )
    wide(torch.randn(2, 128, dtype=torch.float16))
    narrow(torch.randn(2, 128, dtype=torch.float16))

    assert calls == [
        ((1024, 256), calls[0][1]),
        ((1024, 128), calls[0][1]),
    ]


def test_qqq_reduce_scratch_is_exclusive_per_stream():
    """Two live streams must not share the writable C reduction scratch.

    The native kernel reads and writes ``C`` for the whole call, so an
    allocation lock is not enough: the buffer must belong to the stream whose
    work will use it.  ``stream`` lets this run without a CUDA device; the
    CUDA test below exercises the real ``torch.cuda.current_stream`` lookup.
    """

    qqq_module._clear_reduce_scratch()
    device = torch.device("cpu")

    first = qqq_module._shared_reduce_scratch(device, 64, stream="stream-a")
    second = qqq_module._shared_reduce_scratch(device, 64, stream="stream-b")
    assert first.untyped_storage().data_ptr() != second.untyped_storage().data_ptr()

    # Calls on one stream are ordered by that stream and may reuse its buffer.
    first_again = qqq_module._shared_reduce_scratch(device, 64, stream="stream-a")
    assert first_again.untyped_storage().data_ptr() == first.untyped_storage().data_ptr()

    # A device with no stream keeps one shared buffer, as before.
    qqq_module._clear_reduce_scratch()
    plain = qqq_module._shared_reduce_scratch(device, 64)
    plain_again = qqq_module._shared_reduce_scratch(device, 64)
    assert plain.untyped_storage().data_ptr() == plain_again.untyped_storage().data_ptr()
    qqq_module._clear_reduce_scratch()


@pytest.mark.cuda
@pytest.mark.skipif(
    not torch.cuda.is_available() or IS_ROCM,
    reason="QQQ's native kernel is CUDA-only; HIP exposes CUDA streams too",
)
def test_qqq_reduce_scratch_is_exclusive_per_cuda_stream():
    """Concurrent CUDA streams get independent reduce scratch, not one alias."""

    qqq_module._clear_reduce_scratch()
    device = torch.device("cuda", torch.cuda.current_device())
    first_stream = torch.cuda.Stream(device=device)
    second_stream = torch.cuda.Stream(device=device)
    try:
        with torch.cuda.stream(first_stream):
            first = qqq_module._shared_reduce_scratch(device, 64)
        with torch.cuda.stream(second_stream):
            second = qqq_module._shared_reduce_scratch(device, 64)
        with torch.cuda.stream(first_stream):
            first_again = qqq_module._shared_reduce_scratch(device, 64)

        assert first.data_ptr() != second.data_ptr()
        assert first_again.data_ptr() == first.data_ptr()
    finally:
        first_stream.synchronize()
        second_stream.synchronize()
        qqq_module._clear_reduce_scratch()


@pytest.mark.cuda
@pytest.mark.skipif(
    not torch.cuda.is_available() or IS_ROCM,
    reason="QQQ's native kernel is CUDA-only; HIP exposes CUDA streams too",
)
def test_qqq_native_forward_separates_scratch_across_streams(monkeypatch):
    """Two eager modules on different streams must not pass the same C range."""

    qqq_module._clear_reduce_scratch()
    monkeypatch.setattr(
        QQQLinear, "cached_validate_once", classmethod(lambda cls: (True, None))
    )
    monkeypatch.setattr(qqq_module, "qqq_runtime_available", lambda: True)
    pointers = []

    def fake_gemm(A, B, C, D, s1, s2, s3, workspace, thread_k=-1, thread_n=-1, sms=-1, max_par=16):
        pointers.append(C.data_ptr())
        D.zero_()

    monkeypatch.setattr(qqq_module, "qqq_gemm", fake_gemm)
    device = torch.device("cuda", torch.cuda.current_device())
    modules = [
        QQQLinear(
            bits=4,
            group_size=128,
            sym=True,
            desc_act=False,
            in_features=128,
            out_features=128,
            register_buffers=True,
        ).to(device)
        for _ in range(2)
    ]
    x = torch.randn(2, 128, dtype=torch.float16, device=device)
    streams = [torch.cuda.Stream(device=device), torch.cuda.Stream(device=device)]
    try:
        for module, stream in zip(modules, streams):
            with torch.cuda.stream(stream):
                module(x)
    finally:
        for stream in streams:
            stream.synchronize()
        qqq_module._clear_reduce_scratch()

    assert len(pointers) == 2
    assert pointers[0] != pointers[1]


def test_qqq_torch_unpack_uint4_matches_int64_oracle_bitwise():
    torch.manual_seed(4387)
    packed = torch.randint(-(2**31), 2**31 - 1, (11, 7), dtype=torch.int32)
    packed_i64 = packed.to(torch.int64)
    packed_u32 = torch.where(packed_i64 < 0, packed_i64 + (1 << 32), packed_i64)
    divisors = torch.tensor(_OLD_DIVISORS, dtype=torch.int64)
    quotients = torch.floor_divide(packed_u32.unsqueeze(-1), divisors)
    expected = (quotients - torch.floor_divide(quotients, 16) * 16).reshape(11, 56)

    actual = qqq_module._unpack_uint4(packed)
    assert actual.dtype == torch.uint8
    assert actual.shape == expected.shape
    assert torch.equal(actual.long(), expected)


@pytest.mark.parametrize("group_size", [-1, 128])
def test_qqq_torch_forward_is_bitwise_unchanged(group_size):
    module = _build_torch_module(group_size=group_size)
    x = torch.randn(7, module.in_features, dtype=torch.float16)

    with torch.inference_mode():
        actual = module(x)

    quant_a, s1 = module.dynamic_quant(x.reshape(-1, module.in_features))
    weight = _oracle_int8_weight(module)
    s_channel = module._unpermute_scales(
        module.s_channel, module._scale_perm_single_i
    ).float()
    accum = quant_a.float() @ weight.float()
    expected = (accum * s1 * s_channel).to(torch.float16).reshape(7, module.out_features)

    assert torch.equal(actual, expected)
    assert "reduce_buffer" not in module._buffers


def _gemm_library_transient_bytes(rows: int, in_features: int, out_features: int) -> int:
    """Per-call transient of a bare FP32 GEMM with the forward's shape.

    The bound below covers QQQ's own temporaries. The BLAS library's per-call
    workspace is outside the kernel's control and differs by platform: about
    8 MiB with cuBLAS, but about 80 MiB with hipBLASLt on ROCm for this
    M=1 shape (#4462), which alone exceeds the bound.
    """

    a = torch.randn(rows, in_features, dtype=torch.float32, device="cuda")
    b = torch.randn(in_features, out_features, dtype=torch.float32, device="cuda")
    with torch.inference_mode():
        torch.matmul(a, b)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    with torch.inference_mode():
        out = torch.matmul(a, b)
    torch.cuda.synchronize()
    transient = torch.cuda.max_memory_allocated() - base
    return max(0, transient - out.numel() * out.element_size())


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_qqq_torch_cuda_forward_peak_transient_is_bounded():
    """A 896x4864 layer used to allocate ~108 MiB of transient per forward."""

    module = _build_torch_module(
        in_features=896, out_features=4864, group_size=128, dtype=torch.bfloat16
    ).to("cuda")
    x = torch.randn(1, 896, dtype=torch.bfloat16, device="cuda")
    gemm_workspace = _gemm_library_transient_bytes(1, module.in_features, module.out_features)

    with torch.inference_mode():
        module(x)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    with torch.inference_mode():
        output = module(x)
    torch.cuda.synchronize()
    transient = torch.cuda.max_memory_allocated() - base - gemm_workspace

    weight_bytes = module.in_features * module.out_features
    # Pre-fix measured 108.06 MiB; the INT32/INT8 unpack measured 52.82 MiB
    # (both on CUDA, including the ~8 MiB cuBLAS workspace now excluded).
    assert transient < 70 * 2**20, (
        f"transient peak {transient / 2**20:.2f} MiB "
        f"(excluding {gemm_workspace / 2**20:.2f} MiB GEMM workspace)"
    )
    assert transient < 16 * weight_bytes
    assert output.dtype == torch.bfloat16
    assert torch.isfinite(output).all()
    assert not hasattr(module, "_torch_weight_cache")
    assert module._torch_packed_cache is None

