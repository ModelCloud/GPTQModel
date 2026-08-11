# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import gc
import threading
import weakref

import pytest
import torch

import gptqmodel.utils.pangolin_mps as pangolin_mps_module
from gptqmodel.models._const import DEVICE
from gptqmodel.nn_modules.qlinear.pangolin import PangolinQuantLinear
from gptqmodel.quantization import FORMAT
from gptqmodel.utils.pangolin_mps import (
    PANGOLIN_MPS_BITS,
    pangolin_mps_gemv,
    pangolin_mps_supported,
)
from gptqmodel.utils.planar_packing import planar_pack_cols, planar_pack_rows

pytestmark = [
    pytest.mark.mps,
    pytest.mark.skipif(
        not pangolin_mps_supported(), reason="requires runtime Metal shaders"
    ),
]


def _case(
    bits,
    *,
    m=3,
    k=64,
    n=64,
    groups=2,
    seed=0,
    negative_idx=False,
    block_uniform=False,
    group_size=32,
):
    generator = torch.Generator().manual_seed(seed)
    codes = torch.randint(0, 1 << bits, (k, n), generator=generator, dtype=torch.int32)
    zeros = torch.randint(
        0, 1 << bits, (groups, n), generator=generator, dtype=torch.int32
    )
    scales = (torch.rand((groups, n), generator=generator) * 0.2 + 0.01).to(
        torch.float16
    )
    x = torch.randn((m, k), generator=generator).to(torch.float16)
    g_idx = (
        torch.arange(k, dtype=torch.int32) // group_size
        if block_uniform
        else torch.arange(k, dtype=torch.int32) % groups
    )
    if negative_idx:
        g_idx = g_idx - groups
    qweight = planar_pack_rows(codes, bits)
    qzeros = planar_pack_cols(zeros, bits)
    normalized = torch.where(g_idx < 0, g_idx + groups, g_idx).long()
    reference = x.float() @ (
        (codes - zeros[normalized]).float() * scales[normalized].float()
    )
    return tuple(
        t.to("mps") for t in (x, qweight, scales, qzeros, g_idx)
    ), reference.half()


def _run_m8_n8(operands, bits):
    x, qweight, scales, qzeros, g_idx = operands
    m, k = x.shape
    groups, n = scales.shape
    output = torch.empty((m, n), dtype=x.dtype, device=x.device)
    pangolin_mps_module._library().pangolin_fp16_m8_n8(
        x,
        qweight,
        scales,
        qzeros,
        g_idx,
        output,
        m,
        k,
        n,
        groups,
        bits,
        0,
        1,
        threads=(n // 8) * 256,
        group_size=256,
    )
    return output


def _run_m16_n8(operands, bits):
    x, qweight, scales, qzeros, g_idx = operands
    m, k = x.shape
    groups, n = scales.shape
    output = torch.empty((m, n), dtype=x.dtype, device=x.device)
    pangolin_mps_module._library().pangolin_fp16_m16_n8(
        x,
        qweight,
        scales,
        qzeros,
        g_idx,
        output,
        m,
        k,
        n,
        groups,
        bits,
        int(bits in (3, 5, 6, 7)),
        1,
        threads=(n // 8) * 256,
        group_size=256,
    )
    return output


@pytest.mark.parametrize("bits", PANGOLIN_MPS_BITS)
@pytest.mark.parametrize("m", (1, 2, 3, 4, 8, 9, 16, 23, 24, 31, 32))
@pytest.mark.parametrize("negative_idx", (False, True))
def test_all_bits_shapes_and_signed_group_indices(bits, m, negative_idx):
    operands, reference = _case(
        bits, m=m, seed=bits * 100 + m, negative_idx=negative_idx
    )
    result = pangolin_mps_gemv(*operands, bits, planar=bits in (3, 5, 6, 7)).cpu()
    # Both implementations accumulate in float32 and round once to float16.
    torch.testing.assert_close(result, reference, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("bits", PANGOLIN_MPS_BITS)
@pytest.mark.parametrize("m", (5, 16, 32))
def test_deterministic_across_repeated_launches(bits, m):
    operands, _ = _case(bits, m=m, seed=9000 + bits + m)
    outputs = [
        pangolin_mps_gemv(*operands, bits, planar=bits in (3, 5, 6, 7)).cpu()
        for _ in range(3)
    ]
    assert torch.equal(outputs[0], outputs[1])
    assert torch.equal(outputs[1], outputs[2])


@pytest.mark.parametrize("bits", PANGOLIN_MPS_BITS)
@pytest.mark.parametrize("m", (1, 4, 16, 32))
def test_block_uniform_fast_path_matches_reference(bits, m):
    operands, reference = _case(bits, m=m, seed=3000 + bits + m, block_uniform=True)
    result = pangolin_mps_gemv(
        *operands,
        bits,
        planar=bits in (3, 5, 6, 7),
        _g_idx_validated=True,
        _g_idx_block_uniform=True,
    ).cpu()
    torch.testing.assert_close(result, reference, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("bits", (2, 4, 8))
@pytest.mark.parametrize("m", (1, 2, 3))
def test_small_m_continuous_uniform_path_is_exact_and_deterministic(bits, m):
    for seed in range(3):
        operands, reference = _case(
            bits,
            m=m,
            k=256,
            n=64,
            groups=2,
            seed=41000 + bits * 100 + m * 10 + seed,
            negative_idx=seed == 2,
            block_uniform=True,
            group_size=128,
        )
        for _ in range(10):
            result = pangolin_mps_gemv(
                *operands,
                bits,
                planar=False,
                _g_idx_validated=True,
                _g_idx_block_uniform=True,
            ).cpu()
            torch.testing.assert_close(result, reference, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("bits", (2, 4, 8))
@pytest.mark.parametrize("m", (4, 8))
def test_m8_n4_path_is_exact_and_deterministic(bits, m):
    for seed in range(3):
        operands, reference = _case(
            bits,
            m=m,
            k=256,
            n=64,
            groups=2,
            seed=43000 + bits * 100 + m * 10 + seed,
            negative_idx=seed == 2,
            block_uniform=True,
            group_size=128,
        )
        for _ in range(10):
            result = pangolin_mps_gemv(
                *operands,
                bits,
                planar=False,
                _g_idx_validated=True,
                _g_idx_block_uniform=True,
            ).cpu()
            torch.testing.assert_close(result, reference, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("bits", (2, 4, 8))
@pytest.mark.parametrize("m", (4, 8))
def test_m8_n8_kernel_is_exact_and_deterministic(bits, m):
    for seed in range(3):
        operands, reference = _case(
            bits,
            m=m,
            k=256,
            n=64,
            groups=2,
            seed=47000 + bits * 100 + m * 10 + seed,
            negative_idx=seed == 2,
            block_uniform=True,
            group_size=128,
        )
        for _ in range(10):
            result = _run_m8_n8(operands, bits).cpu()
            torch.testing.assert_close(result, reference, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("bits", PANGOLIN_MPS_BITS)
@pytest.mark.parametrize("m", (9, 16))
def test_m16_n4_path_is_exact_and_deterministic(bits, m):
    for seed in range(3):
        operands, reference = _case(
            bits,
            m=m,
            k=256,
            n=64,
            groups=2,
            seed=45000 + bits * 100 + m * 10 + seed,
            negative_idx=seed == 2,
            block_uniform=True,
            group_size=128,
        )
        for _ in range(10):
            result = pangolin_mps_gemv(
                *operands,
                bits,
                planar=bits in (3, 5, 6, 7),
                _g_idx_validated=True,
                _g_idx_block_uniform=True,
            ).cpu()
            torch.testing.assert_close(result, reference, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("bits", PANGOLIN_MPS_BITS)
@pytest.mark.parametrize("m", (9, 16))
def test_m16_n8_kernel_is_exact_and_deterministic(bits, m):
    for seed in range(3):
        operands, reference = _case(
            bits,
            m=m,
            k=256,
            n=64,
            groups=2,
            seed=55000 + bits * 100 + m * 10 + seed,
            negative_idx=seed == 2,
            block_uniform=True,
            group_size=128,
        )
        for _ in range(10):
            result = _run_m16_n8(operands, bits).cpu()
            torch.testing.assert_close(result, reference, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("m", (5, 9, 16, 32))
def test_device_guard_prevents_oob_when_prevalidated_g_idx_is_mutated(m):
    operands, _ = _case(4, m=m, block_uniform=True)
    operands[-1].fill_(99)
    result = pangolin_mps_gemv(
        *operands,
        4,
        planar=False,
        _g_idx_validated=True,
        _g_idx_block_uniform=True,
    ).cpu()
    assert torch.isnan(result).all()


@pytest.mark.parametrize("m", (1, 2, 3))
def test_small_m_vector_path_2bit_large_k_matches_reference(m):
    operands, reference = _case(2, m=m, k=2048, n=32, groups=64, seed=22048 + m)
    result = pangolin_mps_gemv(*operands, 2, planar=False).cpu()
    torch.testing.assert_close(result, reference, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("m", (1, 2, 3))
def test_small_m_vector_path_device_guard_prevents_oob(m):
    operands, _ = _case(4, m=m)
    operands[-1].fill_(99)
    result = pangolin_mps_gemv(
        *operands,
        4,
        planar=False,
        _g_idx_validated=True,
        _g_idx_block_uniform=False,
    ).cpu()
    assert torch.isnan(result).all()


@pytest.mark.parametrize("m", (1, 2, 3))
def test_small_m_planar_uniform_device_guard_prevents_oob(m):
    operands, _ = _case(3, m=m, block_uniform=True)
    operands[-1].fill_(99)
    result = pangolin_mps_gemv(
        *operands,
        3,
        planar=True,
        _g_idx_validated=True,
        _g_idx_block_uniform=True,
    ).cpu()
    assert torch.isnan(result).all()


@pytest.mark.parametrize("m", (1, 2, 3))
def test_small_m_continuous_uniform_device_guard_prevents_oob(m):
    operands, _ = _case(4, m=m, block_uniform=True)
    operands[-1].fill_(99)
    result = pangolin_mps_gemv(
        *operands,
        4,
        planar=False,
        _g_idx_validated=True,
        _g_idx_block_uniform=True,
    ).cpu()
    assert torch.isnan(result).all()


def test_m8_n8_device_guard_prevents_oob():
    operands, _ = _case(
        4, m=8, k=256, n=32, groups=2, block_uniform=True, group_size=128
    )
    operands[-1].fill_(99)
    result = _run_m8_n8(operands, 4).cpu()
    assert torch.isnan(result).all()


def test_m16_n8_device_guard_prevents_oob():
    operands, _ = _case(
        5, m=16, k=256, n=32, groups=2, block_uniform=True, group_size=128
    )
    operands[-1].fill_(99)
    result = _run_m16_n8(operands, 5).cpu()
    assert torch.isnan(result).all()


@pytest.mark.parametrize("m", (3, 4, 8, 9, 16))
def test_misaligned_contiguous_views_use_safe_scalar_loads(m):
    operands, reference = _case(4, m=m, seed=40404 + m)
    x, qweight, scales, qzeros, g_idx = operands
    qweight_storage = torch.empty(
        qweight.numel() + 1, dtype=qweight.dtype, device=qweight.device
    )
    misaligned_qweight = qweight_storage[1:].view_as(qweight)
    misaligned_qweight.copy_(qweight)
    scale_storage = torch.empty(
        scales.numel() + 1, dtype=scales.dtype, device=scales.device
    )
    misaligned_scales = scale_storage[1:].view_as(scales)
    misaligned_scales.copy_(scales)
    qzero_storage = torch.empty(
        qzeros.numel() + 1, dtype=qzeros.dtype, device=qzeros.device
    )
    misaligned_qzeros = qzero_storage[1:].view_as(qzeros)
    misaligned_qzeros.copy_(qzeros)
    assert all(
        tensor.is_contiguous()
        for tensor in (misaligned_qweight, misaligned_scales, misaligned_qzeros)
    )
    assert misaligned_qweight.storage_offset() == 1
    assert misaligned_scales.storage_offset() == 1
    assert misaligned_qzeros.storage_offset() == 1

    result = pangolin_mps_gemv(
        x,
        misaligned_qweight,
        misaligned_scales,
        misaligned_qzeros,
        g_idx,
        4,
        planar=False,
    ).cpu()
    torch.testing.assert_close(result, reference, rtol=1e-3, atol=1e-3)


def test_launches_are_thread_safe():
    cases = [
        _case(bits, m=m, seed=700 + bits)
        for bits, m in zip(PANGOLIN_MPS_BITS, (1, 4, 9, 16, 24, 32, 5))
    ]
    errors = []

    def run(bits, operands, reference):
        try:
            result = pangolin_mps_gemv(
                *operands, bits, planar=bits in (3, 5, 6, 7)
            ).cpu()
            torch.testing.assert_close(result, reference, rtol=1e-3, atol=1e-3)
        except (
            AssertionError,
            RuntimeError,
            ValueError,
        ) as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    threads = [
        threading.Thread(target=run, args=(bits, *case))
        for bits, case in zip(PANGOLIN_MPS_BITS, cases)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []


def test_m8_n8_launches_are_thread_safe():
    cases = [
        (
            bits,
            *_case(
                bits,
                m=8,
                k=256,
                n=64,
                groups=2,
                seed=51000 + bits,
                block_uniform=True,
                group_size=128,
            ),
        )
        for bits in (2, 8)
    ]
    errors = []

    def run(bits, operands, reference):
        try:
            result = _run_m8_n8(operands, bits).cpu()
            torch.testing.assert_close(result, reference, rtol=1e-3, atol=1e-3)
        except (AssertionError, RuntimeError, ValueError) as exc:
            errors.append(exc)

    threads = [threading.Thread(target=run, args=case) for case in cases]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []


def test_m16_n8_launches_are_thread_safe():
    cases = [
        (
            bits,
            *_case(
                bits,
                m=16,
                k=256,
                n=64,
                groups=2,
                seed=56000 + bits,
                block_uniform=True,
                group_size=128,
            ),
        )
        for bits in (4, 5)
    ]
    errors = []

    def run(bits, operands, reference):
        try:
            result = _run_m16_n8(operands, bits).cpu()
            torch.testing.assert_close(result, reference, rtol=1e-3, atol=1e-3)
        except (AssertionError, RuntimeError, ValueError) as exc:
            errors.append(exc)

    threads = [threading.Thread(target=run, args=case) for case in cases]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []


def test_call_does_not_retain_operand_tensors():
    operands, _ = _case(5)
    refs = [weakref.ref(tensor) for tensor in operands]
    result = pangolin_mps_gemv(*operands, 5, planar=True)
    torch.mps.synchronize()
    del result, operands
    gc.collect()
    assert all(ref() is None for ref in refs)


def test_m8_n4_call_does_not_retain_operand_tensors():
    operands, _ = _case(4, m=8, block_uniform=True)
    refs = [weakref.ref(tensor) for tensor in operands]
    result = pangolin_mps_gemv(
        *operands,
        4,
        planar=False,
        _g_idx_validated=True,
        _g_idx_block_uniform=True,
    )
    torch.mps.synchronize()
    del result, operands
    gc.collect()
    assert all(ref() is None for ref in refs)


def test_m8_n8_call_does_not_retain_operand_tensors():
    operands, _ = _case(
        4, m=8, k=256, n=64, groups=2, block_uniform=True, group_size=128
    )
    refs = [weakref.ref(tensor) for tensor in operands]
    result = _run_m8_n8(operands, 4)
    torch.mps.synchronize()
    del result, operands
    gc.collect()
    assert all(ref() is None for ref in refs)


def test_m16_n4_call_does_not_retain_operand_tensors():
    operands, _ = _case(5, m=16, block_uniform=True)
    refs = [weakref.ref(tensor) for tensor in operands]
    result = pangolin_mps_gemv(
        *operands,
        5,
        planar=True,
        _g_idx_validated=True,
        _g_idx_block_uniform=True,
    )
    torch.mps.synchronize()
    del result, operands
    gc.collect()
    assert all(ref() is None for ref in refs)


def test_m16_n8_call_does_not_retain_operand_tensors():
    operands, _ = _case(
        5, m=16, k=256, n=64, groups=2, block_uniform=True, group_size=128
    )
    refs = [weakref.ref(tensor) for tensor in operands]
    result = _run_m16_n8(operands, 5)
    torch.mps.synchronize()
    del result, operands
    gc.collect()
    assert all(ref() is None for ref in refs)


@pytest.mark.parametrize(
    ("mutator", "error", "match"),
    [
        (lambda values: (values[0].unsqueeze(0), *values[1:]), ValueError, "2D"),
        (
            lambda values: (*values[:4], torch.full_like(values[4], 99)),
            ValueError,
            "outside",
        ),
        (lambda values: (values[0].float(), *values[1:]), TypeError, "float16"),
    ],
)
def test_invalid_inputs_fail_before_unchecked_shader_access(mutator, error, match):
    operands, _ = _case(4)
    with pytest.raises(error, match=match):
        pangolin_mps_gemv(*mutator(operands), 4, planar=False)


def test_layout_mismatch_is_rejected():
    operands, _ = _case(3)
    with pytest.raises(ValueError, match="invalid planar"):
        pangolin_mps_gemv(*operands, 3, planar=False)


def test_empty_batch_is_validated_and_returns_empty_output():
    operands, _ = _case(6, m=0)
    result = pangolin_mps_gemv(*operands, 6, planar=True)
    assert result.shape == (0, 64)
    assert result.dtype == torch.float16

    with pytest.raises(ValueError, match="packed shapes"):
        pangolin_mps_gemv(
            operands[0],
            operands[1][:-1],
            *operands[2:],
            6,
            planar=True,
        )


@pytest.mark.parametrize("bits", PANGOLIN_MPS_BITS)
def test_quant_linear_constructor_and_forward_integration(bits):
    operands, reference = _case(bits, m=3, block_uniform=True, seed=12000 + bits)
    x, qweight, scales, qzeros, g_idx = operands
    layer = PangolinQuantLinear(
        bits=bits,
        group_size=32,
        sym=False,
        desc_act=False,
        in_features=64,
        out_features=64,
        bias=False,
        format=FORMAT.GPTQ_P,
        dtype=torch.float16,
        trainable=False,
    ).to("mps")
    layer.qweight.copy_(qweight)
    layer.scales.copy_(scales)
    layer.qzeros.copy_(qzeros)
    layer.g_idx.copy_(g_idx)
    layer.post_init()
    layer.eval()
    torch.testing.assert_close(layer(x).cpu(), reference, rtol=1e-3, atol=1e-3)


def test_asymmetric_3bit_contract_is_apple_only():
    args = {
        "bits": 3,
        "group_size": 32,
        "sym": False,
        "desc_act": False,
        "in_features": 64,
        "out_features": 64,
        "pack_dtype": torch.int32,
        "format": FORMAT.GPTQ_P,
        "dtype": torch.float16,
        "trainable": False,
    }
    valid, error = PangolinQuantLinear.validate(**args, device=DEVICE.MPS)
    assert valid and error is None

    valid, error = PangolinQuantLinear.validate(**args, device=DEVICE.CUDA)
    assert not valid
    assert "sym=True" in str(error)


@pytest.mark.parametrize("bits", (2, 3, 4, 5, 6))
@pytest.mark.parametrize("m", (1, 3))
def test_asymmetric_group128_quant_linear_matches_reference(bits, m):
    operands, reference = _case(
        bits,
        m=m,
        k=256,
        n=64,
        groups=2,
        seed=12800 + bits * 10 + m,
        block_uniform=True,
        group_size=128,
    )
    x, qweight, scales, qzeros, g_idx = operands
    layer = PangolinQuantLinear(
        bits=bits,
        group_size=128,
        sym=False,
        desc_act=False,
        in_features=256,
        out_features=64,
        bias=False,
        format=FORMAT.GPTQ_P,
        dtype=torch.float16,
        trainable=False,
    ).to("mps")
    layer.qweight.copy_(qweight)
    layer.scales.copy_(scales)
    layer.qzeros.copy_(qzeros)
    layer.g_idx.copy_(g_idx)
    layer.post_init()
    layer.eval()

    torch.testing.assert_close(layer(x).cpu(), reference, rtol=1e-3, atol=1e-3)


def test_replaced_g_idx_does_not_reuse_stale_block_uniform_classification():
    operands, _ = _case(4, m=3, block_uniform=True, seed=13004)
    x, qweight, scales, qzeros, g_idx = operands
    layer = PangolinQuantLinear(
        bits=4,
        group_size=32,
        sym=True,
        desc_act=False,
        in_features=64,
        out_features=64,
        bias=False,
        format=FORMAT.GPTQ_P,
        dtype=torch.float16,
        trainable=False,
    ).to("mps")
    layer.qweight.copy_(qweight)
    layer.scales.copy_(scales)
    layer.qzeros.copy_(qzeros)
    layer.g_idx.copy_(g_idx)
    layer.post_init()
    layer.eval()

    replacement = (torch.arange(64, dtype=torch.int32) % 2).to("mps")
    expected = pangolin_mps_gemv(
        x, qweight, scales, qzeros, replacement, 4, planar=False
    )
    layer.g_idx = replacement
    torch.testing.assert_close(layer(x), expected, rtol=1e-3, atol=1e-3)
