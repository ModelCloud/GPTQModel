# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import gc
import weakref
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import torch

mx = pytest.importorskip("mlx.core")

import gptqmodel.utils.pangolin_mlx as pangolin_mlx_module
from gptqmodel.utils.pangolin_mlx import PANGOLIN_MLX_BITS, pangolin_mlx_gemv
from gptqmodel.utils.planar_packing import planar_pack_cols, planar_pack_rows


def _mlx(tensor):
    return mx.array(tensor.numpy())


def _run_m8_n8(operands, bits):
    x, qweight, scales, qzeros, g_idx = operands
    m, k = x.shape
    groups, n = scales.shape
    dims = pangolin_mlx_module._dims_array(
        m, k, n, groups, bits, bits in (3, 5, 6, 7), 4
    )
    return pangolin_mlx_module._kernel(mode=11)(
        inputs=[x, qweight, scales, qzeros, g_idx, dims],
        template=[("T", mx.float16)],
        grid=((n // 8) * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(m, n)],
        output_dtypes=[mx.float16],
    )[0]


def _run_m16_n8(operands, bits):
    x, qweight, scales, qzeros, g_idx = operands
    m, k = x.shape
    groups, n = scales.shape
    dims = pangolin_mlx_module._dims_array(
        m, k, n, groups, bits, bits in (3, 5, 6, 7), 4
    )
    return pangolin_mlx_module._kernel(mode=12)(
        inputs=[x, qweight, scales, qzeros, g_idx, dims],
        template=[("T", mx.float16)],
        grid=((n // 8) * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(m, n)],
        output_dtypes=[mx.float16],
    )[0]


def _uniform_case(bits, m, seed, *, negative_idx=False, k=256):
    generator = torch.Generator().manual_seed(seed)
    n, groups = 64, k // 128
    codes = torch.randint(0, 1 << bits, (k, n), generator=generator, dtype=torch.int32)
    zeros = torch.randint(
        0, 1 << bits, (groups, n), generator=generator, dtype=torch.int32
    )
    scales = (torch.rand((groups, n), generator=generator) * 0.2 + 0.01).half()
    x = torch.randn((m, k), generator=generator).half()
    g_idx = torch.arange(k, dtype=torch.int32) // 128
    if negative_idx:
        g_idx -= groups
    normalized = torch.where(g_idx < 0, g_idx + groups, g_idx).long()
    reference = (
        x.float() @ ((codes - zeros[normalized]).float() * scales[normalized].float())
    ).half()
    operands = tuple(
        map(
            _mlx,
            (
                x,
                planar_pack_rows(codes, bits),
                scales,
                planar_pack_cols(zeros, bits),
                g_idx,
            ),
        )
    )
    return operands, reference


@pytest.mark.parametrize("bits", PANGOLIN_MLX_BITS)
@pytest.mark.parametrize("m", (1, 3, 4, 5, 6, 8, 9, 16, 17, 24, 31, 32))
@pytest.mark.parametrize("negative_idx", (False, True))
def test_pangolin_mlx_matches_reference(bits, m, negative_idx):
    generator = torch.Generator().manual_seed(1700 + bits * 10 + m)
    k, n, groups = 64, 64, 2
    codes = torch.randint(0, 1 << bits, (k, n), generator=generator, dtype=torch.int32)
    zeros = torch.randint(
        0, 1 << bits, (groups, n), generator=generator, dtype=torch.int32
    )
    scales = (torch.rand((groups, n), generator=generator) * 0.2 + 0.01).half()
    x = torch.randn((m, k), generator=generator).half()
    g_idx = torch.arange(k, dtype=torch.int32) % groups
    if negative_idx:
        g_idx -= groups
    normalized = torch.where(g_idx < 0, g_idx + groups, g_idx).long()
    reference = (
        x.float() @ ((codes - zeros[normalized]).float() * scales[normalized].float())
    ).half()
    result = pangolin_mlx_gemv(
        *map(
            _mlx,
            (
                x,
                planar_pack_rows(codes, bits),
                scales,
                planar_pack_cols(zeros, bits),
                g_idx,
            ),
        ),
        bits,
        planar=bits in (3, 5, 6, 7),
    )
    np.testing.assert_allclose(
        np.asarray(result), reference.numpy(), rtol=1e-3, atol=1e-3
    )


def test_pangolin_mlx_rejects_out_of_bounds_group_index():
    bits = 4
    x = mx.ones((1, 32), dtype=mx.float16)
    qweight = mx.zeros((4, 32), dtype=mx.int32)
    scales = mx.ones((1, 32), dtype=mx.float16)
    qzeros = mx.zeros((1, 4), dtype=mx.int32)
    g_idx = mx.full((32,), 1, dtype=mx.int32)
    with pytest.raises(ValueError, match="out-of-bounds"):
        pangolin_mlx_gemv(x, qweight, scales, qzeros, g_idx, bits, planar=False)


@pytest.mark.parametrize("bits", (2, 3, 4, 5, 6))
@pytest.mark.parametrize("m", (1, 3))
def test_pangolin_mlx_asymmetric_group128_matches_reference(bits, m):
    generator = torch.Generator().manual_seed(12800 + bits * 10 + m)
    k, n, groups = 256, 64, 2
    codes = torch.randint(0, 1 << bits, (k, n), generator=generator, dtype=torch.int32)
    zeros = torch.randint(
        0, 1 << bits, (groups, n), generator=generator, dtype=torch.int32
    )
    scales = (torch.rand((groups, n), generator=generator) * 0.2 + 0.01).half()
    x = torch.randn((m, k), generator=generator).half()
    g_idx = torch.arange(k, dtype=torch.int32) // 128
    reference = (
        x.float()
        @ ((codes - zeros[g_idx.long()]).float() * scales[g_idx.long()].float())
    ).half()

    result = pangolin_mlx_gemv(
        *map(
            _mlx,
            (
                x,
                planar_pack_rows(codes, bits),
                scales,
                planar_pack_cols(zeros, bits),
                g_idx,
            ),
        ),
        bits,
        planar=bits in (3, 5, 6),
        _g_idx_validated=True,
    )
    np.testing.assert_allclose(
        np.asarray(result), reference.numpy(), rtol=1e-3, atol=1e-3
    )


@pytest.mark.parametrize("bits", (3, 5, 6, 7))
@pytest.mark.parametrize("m", (1, 2, 3))
def test_pangolin_mlx_small_m_planar_uniform_is_exact_and_deterministic(bits, m):
    for seed in range(3):
        generator = torch.Generator().manual_seed(42000 + bits * 100 + m * 10 + seed)
        k, n, groups = 256, 64, 2
        codes = torch.randint(
            0, 1 << bits, (k, n), generator=generator, dtype=torch.int32
        )
        zeros = torch.randint(
            0, 1 << bits, (groups, n), generator=generator, dtype=torch.int32
        )
        scales = (torch.rand((groups, n), generator=generator) * 0.2 + 0.01).half()
        x = torch.randn((m, k), generator=generator).half()
        g_idx = torch.arange(k, dtype=torch.int32) // 128
        if seed == 2:
            g_idx -= groups
        normalized = torch.where(g_idx < 0, g_idx + groups, g_idx).long()
        reference = (
            x.float()
            @ ((codes - zeros[normalized]).float() * scales[normalized].float())
        ).half()
        operands = tuple(
            map(
                _mlx,
                (
                    x,
                    planar_pack_rows(codes, bits),
                    scales,
                    planar_pack_cols(zeros, bits),
                    g_idx,
                ),
            )
        )

        for _ in range(10):
            hints = (
                {}
                if seed == 0
                else {
                    "_g_idx_validated": True,
                    "_g_idx_block_uniform": True,
                }
            )
            result = pangolin_mlx_gemv(
                *operands,
                bits,
                planar=True,
                **hints,
            )
            np.testing.assert_allclose(
                np.asarray(result), reference.numpy(), rtol=1e-3, atol=1e-3
            )


@pytest.mark.parametrize("bits", PANGOLIN_MLX_BITS)
@pytest.mark.parametrize("m", (4, 8))
def test_pangolin_mlx_m8_n4_is_exact_and_deterministic(bits, m):
    for seed in range(3):
        operands, reference = _uniform_case(
            bits,
            m,
            44000 + bits * 100 + m * 10 + seed,
            negative_idx=seed == 2,
        )
        for _ in range(10):
            result = pangolin_mlx_gemv(
                *operands,
                bits,
                planar=bits in (3, 5, 6, 7),
                _g_idx_validated=True,
            )
            np.testing.assert_allclose(
                np.asarray(result), reference.numpy(), rtol=1e-3, atol=1e-3
            )


@pytest.mark.parametrize("bits", PANGOLIN_MLX_BITS)
@pytest.mark.parametrize("m", (4, 6))
def test_pangolin_mlx_m8_n8_kernel_is_exact_and_deterministic(bits, m):
    for seed in range(3):
        operands, reference = _uniform_case(
            bits,
            m,
            48000 + bits * 100 + m * 10 + seed,
            negative_idx=seed == 2,
            k=256,
        )
        for _ in range(10):
            result = _run_m8_n8(operands, bits)
            np.testing.assert_allclose(
                np.asarray(result), reference.numpy(), rtol=1e-3, atol=1e-3
            )


@pytest.mark.parametrize("bits", (2, 4, 8))
def test_pangolin_mlx_m8_n8_continuous_m8_is_exact_and_deterministic(bits):
    for seed in range(3):
        operands, reference = _uniform_case(
            bits,
            8,
            49000 + bits * 100 + seed,
            negative_idx=seed == 2,
            k=256,
        )
        for _ in range(10):
            result = _run_m8_n8(operands, bits)
            np.testing.assert_allclose(
                np.asarray(result), reference.numpy(), rtol=1e-3, atol=1e-3
            )


@pytest.mark.parametrize("bits", PANGOLIN_MLX_BITS)
@pytest.mark.parametrize("m", (9, 16))
def test_pangolin_mlx_m16_n4_is_exact_and_deterministic(bits, m):
    for seed in range(3):
        operands, reference = _uniform_case(
            bits,
            m,
            46000 + bits * 100 + m * 10 + seed,
            negative_idx=seed == 2,
        )
        for _ in range(10):
            result = pangolin_mlx_gemv(
                *operands,
                bits,
                planar=bits in (3, 5, 6, 7),
                _g_idx_validated=True,
            )
            np.testing.assert_allclose(
                np.asarray(result), reference.numpy(), rtol=1e-3, atol=1e-3
            )


@pytest.mark.parametrize("bits", PANGOLIN_MLX_BITS)
@pytest.mark.parametrize("m", (9, 16))
def test_pangolin_mlx_m16_n8_kernel_is_exact_and_deterministic(bits, m):
    for seed in range(3):
        operands, reference = _uniform_case(
            bits,
            m,
            57000 + bits * 100 + m * 10 + seed,
            negative_idx=seed == 2,
        )
        for _ in range(10):
            result = _run_m16_n8(operands, bits)
            np.testing.assert_allclose(
                np.asarray(result), reference.numpy(), rtol=1e-3, atol=1e-3
            )


@pytest.mark.parametrize("m", (1, 2, 3))
def test_pangolin_mlx_small_m_uniform_device_guard_prevents_oob(m):
    bits = 3
    x = mx.ones((m, 32), dtype=mx.float16)
    qweight = mx.zeros((bits, 32), dtype=mx.int32)
    scales = mx.ones((1, 32), dtype=mx.float16)
    qzeros = mx.zeros((1, bits), dtype=mx.int32)
    g_idx = mx.full((32,), 1, dtype=mx.int32)
    result = pangolin_mlx_gemv(
        x,
        qweight,
        scales,
        qzeros,
        g_idx,
        bits,
        planar=True,
        _g_idx_validated=True,
        _g_idx_block_uniform=True,
    )
    assert np.isnan(np.asarray(result)).all()


@pytest.mark.parametrize("m", (9, 16, 17, 32))
def test_pangolin_mlx_multirow_device_guard_prevents_oob(m):
    bits = 4
    x = mx.ones((m, 32), dtype=mx.float16)
    qweight = mx.zeros((4, 32), dtype=mx.int32)
    scales = mx.ones((1, 32), dtype=mx.float16)
    qzeros = mx.zeros((1, 4), dtype=mx.int32)
    g_idx = mx.full((32,), 1, dtype=mx.int32)
    result = pangolin_mlx_gemv(
        x,
        qweight,
        scales,
        qzeros,
        g_idx,
        bits,
        planar=False,
        _g_idx_validated=True,
    )
    assert np.isnan(np.asarray(result)).all()


@pytest.mark.parametrize("m", (4, 8))
def test_pangolin_mlx_m8_n4_device_guard_prevents_oob(m):
    bits = 5
    x = mx.ones((m, 32), dtype=mx.float16)
    qweight = mx.zeros((bits, 32), dtype=mx.int32)
    scales = mx.ones((1, 32), dtype=mx.float16)
    qzeros = mx.zeros((1, bits), dtype=mx.int32)
    g_idx = mx.full((32,), 1, dtype=mx.int32)
    result = pangolin_mlx_gemv(
        x,
        qweight,
        scales,
        qzeros,
        g_idx,
        bits,
        planar=True,
        _g_idx_validated=True,
    )
    assert np.isnan(np.asarray(result)).all()


def test_pangolin_mlx_m8_n8_device_guard_prevents_oob():
    bits, m, k, n, groups = 4, 8, 256, 32, 2
    result = _run_m8_n8(
        (
            mx.ones((m, k), dtype=mx.float16),
            mx.zeros((k // (32 // bits), n), dtype=mx.int32),
            mx.ones((groups, n), dtype=mx.float16),
            mx.zeros((groups, n // (32 // bits)), dtype=mx.int32),
            mx.full((k,), 99, dtype=mx.int32),
        ),
        bits,
    )
    assert np.isnan(np.asarray(result)).all()


def test_pangolin_mlx_m16_n8_device_guard_prevents_oob():
    bits, m, k, n, groups = 5, 16, 256, 32, 2
    result = _run_m16_n8(
        (
            mx.ones((m, k), dtype=mx.float16),
            mx.zeros(((k // 32) * bits, n), dtype=mx.int32),
            mx.ones((groups, n), dtype=mx.float16),
            mx.zeros((groups, (n // 32) * bits), dtype=mx.int32),
            mx.full((k,), 99, dtype=mx.int32),
        ),
        bits,
    )
    assert np.isnan(np.asarray(result)).all()


def test_pangolin_mlx_multirow_launches_are_thread_safe():
    cases = []
    for bits, m in zip(PANGOLIN_MLX_BITS, (8, 16, 17, 24, 31, 32, 9)):
        generator = torch.Generator().manual_seed(8100 + bits * 10 + m)
        k, n, groups = 64, 64, 2
        codes = torch.randint(
            0, 1 << bits, (k, n), generator=generator, dtype=torch.int32
        )
        zeros = torch.randint(
            0, 1 << bits, (groups, n), generator=generator, dtype=torch.int32
        )
        scales = (torch.rand((groups, n), generator=generator) * 0.2 + 0.01).half()
        x = torch.randn((m, k), generator=generator).half()
        g_idx = torch.arange(k, dtype=torch.int32) % groups
        reference = (
            x.float()
            @ ((codes - zeros[g_idx.long()]).float() * scales[g_idx.long()].float())
        ).half()
        operands = tuple(
            map(
                _mlx,
                (
                    x,
                    planar_pack_rows(codes, bits),
                    scales,
                    planar_pack_cols(zeros, bits),
                    g_idx,
                ),
            )
        )
        cases.append((bits, operands, reference))

    def run(case):
        bits, operands, reference = case
        result = pangolin_mlx_gemv(*operands, bits, planar=bits in (3, 5, 6, 7))
        np.testing.assert_allclose(
            np.asarray(result), reference.numpy(), rtol=1e-3, atol=1e-3
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(run, cases))


def test_pangolin_mlx_m8_n8_launches_are_thread_safe():
    cases = [(bits, *_uniform_case(bits, 8, 52000 + bits)) for bits in (4, 5)]

    def run(case):
        bits, operands, reference = case
        result = _run_m8_n8(operands, bits)
        np.testing.assert_allclose(
            np.asarray(result), reference.numpy(), rtol=1e-3, atol=1e-3
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(run, cases))


def test_pangolin_mlx_m16_n8_launches_are_thread_safe():
    cases = [(bits, *_uniform_case(bits, 16, 58000 + bits)) for bits in (4, 5)]

    def run(case):
        bits, operands, reference = case
        result = _run_m16_n8(operands, bits)
        np.testing.assert_allclose(
            np.asarray(result), reference.numpy(), rtol=1e-3, atol=1e-3
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(run, cases))


def test_pangolin_mlx_multirow_call_does_not_retain_operands():
    bits, m, k, n, groups = 5, 16, 64, 64, 2
    x = mx.ones((m, k), dtype=mx.float16)
    qweight = mx.zeros(((k // 32) * bits, n), dtype=mx.int32)
    scales = mx.ones((groups, n), dtype=mx.float16)
    qzeros = mx.zeros((groups, (n // 32) * bits), dtype=mx.int32)
    g_idx = mx.zeros((k,), dtype=mx.int32)
    operands = (x, qweight, scales, qzeros, g_idx)
    refs = [weakref.ref(value) for value in operands]

    result = pangolin_mlx_gemv(*operands, bits, planar=True)
    mx.eval(result)
    del result, operands, x, qweight, scales, qzeros, g_idx
    gc.collect()

    assert all(ref() is None for ref in refs)


def test_pangolin_mlx_m8_n4_call_does_not_retain_operands():
    bits, m, k, n, groups = 5, 8, 64, 64, 2
    x = mx.ones((m, k), dtype=mx.float16)
    qweight = mx.zeros(((k // 32) * bits, n), dtype=mx.int32)
    scales = mx.ones((groups, n), dtype=mx.float16)
    qzeros = mx.zeros((groups, (n // 32) * bits), dtype=mx.int32)
    g_idx = mx.zeros((k,), dtype=mx.int32)
    operands = (x, qweight, scales, qzeros, g_idx)
    refs = [weakref.ref(value) for value in operands]

    result = pangolin_mlx_gemv(*operands, bits, planar=True, _g_idx_validated=True)
    mx.eval(result)
    del result, operands, x, qweight, scales, qzeros, g_idx
    gc.collect()

    assert all(ref() is None for ref in refs)


def test_pangolin_mlx_m8_n8_call_does_not_retain_operands():
    bits, m, k, n, groups = 4, 8, 256, 64, 2
    x = mx.ones((m, k), dtype=mx.float16)
    qweight = mx.zeros((k // (32 // bits), n), dtype=mx.int32)
    scales = mx.ones((groups, n), dtype=mx.float16)
    qzeros = mx.zeros((groups, n // (32 // bits)), dtype=mx.int32)
    g_idx = mx.zeros((k,), dtype=mx.int32)
    operands = (x, qweight, scales, qzeros, g_idx)
    refs = [weakref.ref(value) for value in operands]

    result = _run_m8_n8(operands, bits)
    mx.eval(result)
    del result, operands, x, qweight, scales, qzeros, g_idx
    gc.collect()

    assert all(ref() is None for ref in refs)


def test_pangolin_mlx_m16_n8_call_does_not_retain_operands():
    bits, m, k, n, groups = 5, 16, 256, 64, 2
    x = mx.ones((m, k), dtype=mx.float16)
    qweight = mx.zeros(((k // 32) * bits, n), dtype=mx.int32)
    scales = mx.ones((groups, n), dtype=mx.float16)
    qzeros = mx.zeros((groups, (n // 32) * bits), dtype=mx.int32)
    g_idx = mx.zeros((k,), dtype=mx.int32)
    operands = (x, qweight, scales, qzeros, g_idx)
    refs = [weakref.ref(value) for value in operands]

    result = _run_m16_n8(operands, bits)
    mx.eval(result)
    del result, operands, x, qweight, scales, qzeros, g_idx
    gc.collect()

    assert all(ref() is None for ref in refs)


@pytest.mark.parametrize("bits", (2, 3, 4, 5, 6))
def test_pangolin_mlx_n4_call_does_not_retain_operands(bits):
    m, k, n, groups = 3, 64, 64, 2
    x = mx.ones((m, k), dtype=mx.float16)
    qweight_rows = (k // 32) * bits if bits in (3, 5, 6) else k // (32 // bits)
    qzero_cols = (n // 32) * bits if bits in (3, 5, 6) else n // (32 // bits)
    qweight = mx.zeros((qweight_rows, n), dtype=mx.int32)
    scales = mx.ones((groups, n), dtype=mx.float16)
    qzeros = mx.zeros((groups, qzero_cols), dtype=mx.int32)
    g_idx = mx.zeros((k,), dtype=mx.int32)
    operands = (x, qweight, scales, qzeros, g_idx)
    refs = [weakref.ref(value) for value in operands]

    result = pangolin_mlx_gemv(
        *operands,
        bits,
        planar=bits in (3, 5, 6),
        _g_idx_validated=True,
        _g_idx_block_uniform=bits in (3, 5, 6),
    )
    mx.eval(result)
    del result, operands, x, qweight, scales, qzeros, g_idx
    gc.collect()

    assert all(ref() is None for ref in refs)


def test_pangolin_mlx_rejects_layout_mismatch():
    with pytest.raises(ValueError, match="invalid planar"):
        pangolin_mlx_gemv(None, None, None, None, None, 5, planar=False)
