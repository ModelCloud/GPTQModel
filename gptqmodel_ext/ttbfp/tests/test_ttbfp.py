# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import ttbfp
from ttbfp import reference as ref

ROOT = Path(__file__).resolve().parents[1]


def bits(a):
    return np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)


def make_input(rng, shape):
    """Values with a wide range of exponents, plus zeros and denormals."""
    mode = int(rng.integers(0, 6))
    if mode == 0:
        x = rng.standard_normal(shape).astype(np.float32)
    elif mode == 1:
        x = (rng.standard_normal(shape) *
             np.float32(10.0) ** rng.uniform(-25, 25, shape)).astype(np.float32)
    elif mode == 2:
        x = rng.standard_normal(shape).astype(np.float32) * 2.0
    elif mode == 3:
        x = np.zeros(shape, dtype=np.float32)
    elif mode == 4:
        # Exact powers of two and sums that exercise guard-bit rounding.
        e = rng.integers(-20, 20, shape)
        x = np.ldexp(np.ones(shape, dtype=np.float32), e.astype(np.int32))
        x = x + np.ldexp(np.ones(shape, dtype=np.float32),
                         (e - rng.integers(1, 9, shape)).astype(np.int32))
    else:
        x = rng.standard_normal(shape).astype(np.float32) * np.float32(2.0) ** -130
    x = x.astype(np.float32)
    if rng.random() < 0.3:
        x.flat[::7] = 0.0
    if rng.random() < 0.3:
        x.flat[::11] = -0.0
    return x


SHAPES = [(1, 16), (2, 1), (1, 15), (3, 16), (5, 17), (7, 31), (4, 32),
          (9, 48), (32, 32), (33, 80), (6, 33)]


@pytest.mark.parametrize("fmt", [ttbfp.BFP8, ttbfp.BFP4, ttbfp.BFP2])
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_pack_matches_reference(fmt, shape, seed):
    rng = np.random.default_rng(seed)
    x = make_input(rng, shape)
    e_ref, d_ref = ref.pack(x, fmt)
    e_c, d_c = ttbfp.pack(x, fmt)
    assert e_c.shape == e_ref.shape
    assert d_c.shape == d_ref.shape
    np.testing.assert_array_equal(e_c, e_ref)
    np.testing.assert_array_equal(d_c, d_ref)


@pytest.mark.parametrize("fmt", [ttbfp.BFP8, ttbfp.BFP4, ttbfp.BFP2])
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("ftz", [True, False])
def test_unpack_matches_reference(fmt, shape, ftz):
    rng = np.random.default_rng(1234 + shape[1])
    x = make_input(rng, shape)
    e, d = ttbfp.pack(x, fmt)
    y_ref = ref.dequantize(e, d, fmt, cols=shape[1], ftz=ftz)
    y_c = ttbfp.unpack(e, d, fmt, cols=shape[1], ftz=ftz)
    np.testing.assert_array_equal(bits(y_c), bits(y_ref))


@pytest.mark.parametrize("fmt", [ttbfp.BFP8, ttbfp.BFP4, ttbfp.BFP2])
def test_pack_is_stable_on_zeros(fmt):
    x = np.zeros((4, 64), dtype=np.float32)
    e, d = ttbfp.pack(x, fmt)
    assert np.all(e == 0)
    # Reference keeps the implicit-one mantissa for zero blocks.
    assert np.all(ttbfp.unpack(e, d, fmt) == 0.0)


@pytest.mark.parametrize("fmt", [8, 4, 2])
def test_sizes(fmt):
    rows, cols = 5, 33
    nblk = 3
    assert ttbfp.blocks_per_row(cols) == nblk
    db = {8: 16, 4: 8, 2: 4}[fmt]
    assert ttbfp.packed_bytes(fmt, rows, cols) == rows * nblk * (1 + db)
    assert ttbfp.tile_bytes(fmt) == {8: 1088, 4: 576, 2: 320}[fmt]


def _rand_matmul_case(rng, fmt, M, K, N):
    a = (rng.standard_normal((M, K)) * 0.5).astype(np.float32)
    w = (rng.standard_normal((K, N)) * 0.5).astype(np.float32)
    e, d = ttbfp.pack(w, fmt)
    c = ttbfp.matmul(a, e, d, fmt, cols=N)
    c_ref = ref.matmul(a, e, d, fmt, cols=N)
    return c, c_ref


@pytest.mark.parametrize(
    "fmt,M,K,N",
    [
        (8, 1, 64, 64),
        (4, 1, 64, 64),
        (8, 3, 33, 48),
        (4, 5, 17, 33),
        (8, 4, 128, 256),
        (4, 7, 96, 80),
        (8, 33, 64, 16),
        (4, 2, 1, 31),
    ],
)
def test_matmul_matches_float64_reference(fmt, M, K, N):
    rng = np.random.default_rng(M * 1000 + K + N)
    c, c_ref = _rand_matmul_case(rng, fmt, M, K, N)
    denom = max(np.max(np.abs(c_ref)), 1e-30)
    err = np.max(np.abs(c.astype(np.float64) - c_ref)) / denom
    assert err < 5e-6, f"relative max error {err}"


@pytest.mark.parametrize(
    "fmt,M,K,N",
    [
        (8, 1, 64, 64),
        (4, 1, 64, 64),
        (8, 3, 33, 48),
        (4, 5, 17, 33),
        (8, 8, 128, 256),
        (4, 7, 96, 80),
        (8, 33, 64, 16),
        (4, 2, 1, 31),
    ],
)
def test_matmul_scalar_and_threads_bitwise(fmt, M, K, N):
    """AVX2, scalar and any thread count must agree bit-for-bit."""
    rng = np.random.default_rng(7 * M + K + N)
    a = (rng.standard_normal((M, K)) * 0.5).astype(np.float32)
    w = (rng.standard_normal((K, N)) * 0.5).astype(np.float32)
    e, d = ttbfp.pack(w, fmt)

    tmp = Path(tempfile_dir())
    np.savez(tmp / "case.npz", a=a, e=e, d=d)

    script = """
import numpy as np, sys, ttbfp
z = np.load(sys.argv[1])
c = ttbfp.matmul(z['a'], z['e'], z['d'], int(sys.argv[2]), cols=int(sys.argv[3]))
np.save(sys.argv[4], c)
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "python")
    env["TTBFP_LIB"] = str(ROOT / "libttbfp.so")

    outs = []
    for name, extra in [("default", {}),
                        ("avx2", {"TTBFP_FORCE_AVX2": "1"}),
                        ("scalar", {"TTBFP_FORCE_SCALAR": "1"}),
                        ("threads1", {"OMP_NUM_THREADS": "1"}),
                        ("threads7", {"OMP_NUM_THREADS": "7"})]:
        out = tmp / f"{name}.npy"
        env2 = dict(env)
        env2.update(extra)
        subprocess.run([sys.executable, "-c", script, str(tmp / "case.npz"),
                        str(fmt), str(N), str(out)],
                       check=True, env=env2, cwd=str(ROOT))
        outs.append(np.load(out))
    for o in outs[1:]:
        np.testing.assert_array_equal(bits(o), bits(outs[0]))


def test_matmul_bf16_matches_f32_path():
    rng = np.random.default_rng(99)
    M, K, N = 4, 64, 48
    a = rng.standard_normal((M, K)).astype(np.float32)
    w = rng.standard_normal((K, N)).astype(np.float32)
    e, d = ttbfp.pack(w, 8)
    a_bf = (a.view(np.uint32) >> 16).astype(np.uint16)
    a_from_bf = (a_bf.astype(np.uint32) << 16).view(np.float32)
    c1 = ttbfp.matmul_bf16(a_bf, e, d, 8, cols=N)
    c2 = ttbfp.matmul(a_from_bf, e, d, 8, cols=N)
    np.testing.assert_array_equal(bits(c1), bits(c2))


@pytest.mark.parametrize(
    "fmt,tile_bytes",
    [(8, 1088), (4, 576), (2, 320)],
)
def test_tile_layout(fmt, tile_bytes):
    rng = np.random.default_rng(5)
    tile = rng.standard_normal((32, 32)).astype(np.float32)
    blob_c = ttbfp.tile_pack(tile, fmt)
    blob_ref = ref.tile_pack(tile, fmt)
    assert blob_c.size == tile_bytes
    np.testing.assert_array_equal(blob_c, blob_ref)
    back_c = ttbfp.tile_unpack(blob_c, fmt)
    back_ref = ref.tile_unpack(blob_ref, fmt)
    np.testing.assert_array_equal(bits(back_c), bits(back_ref))


@pytest.mark.parametrize(
    "fmt,tile_bytes",
    [(8, 1088), (4, 576), (2, 320)],
)
def test_tile_face_order(fmt, tile_bytes):
    # Each face is a distinct constant; exponents are one per face row and
    # stored face-major (TL, TR, BL, BR).
    tile = np.zeros((32, 32), dtype=np.float32)
    vals = [1.0, 2.0, 4.0, 8.0]
    for f in range(4):
        fr, fc = divmod(f, 2)
        tile[fr * 16:(fr + 1) * 16, fc * 16:(fc + 1) * 16] = vals[f]
    blob = ttbfp.tile_pack(tile, fmt)
    exps = blob[:64].reshape(4, 16)
    assert np.all(exps == exps[:, :1])
    assert list(exps[:, 0]) == [127, 128, 129, 130]
    back = ttbfp.tile_unpack(blob, fmt)
    np.testing.assert_allclose(back, tile, rtol=0.0, atol=0.0)


def test_ftz_behaviour():
    # A block whose shared exponent is tiny: magnitudes below 2^-126 are
    # flushed by FTZ.
    x = np.array([[np.float32(2.0) ** -126, np.float32(2.0) ** -130] +
                  [0.0] * 14], dtype=np.float32)
    e, d = ttbfp.pack(x, 8)
    y_on = ttbfp.unpack(e, d, 8, cols=16, ftz=True)
    y_off = ttbfp.unpack(e, d, 8, cols=16, ftz=False)
    assert y_on[0, 1] == 0.0
    assert y_off[0, 1] != 0.0


def tempfile_dir():
    import tempfile
    return tempfile.mkdtemp(prefix="ttbfp-test-")


@pytest.mark.parametrize("fmt", [ttbfp.TTBF8, ttbfp.TTBF4, ttbfp.TTBF2])
@pytest.mark.parametrize("shape", [(4, 64, 48), (1, 32, 16), (7, 33, 80),
                                   (16, 96, 128), (5, 17, 31)])
def test_prepacked_matches_rowmajor(fmt, shape):
    """The block-major layout must produce bitwise identical results."""
    M, K, N = shape
    rng = np.random.default_rng(M * 131 + K + N)
    a = (rng.standard_normal((M, K)) * 0.5).astype(np.float32)
    w = (rng.standard_normal((K, N)) * 0.5).astype(np.float32)
    e, d = ttbfp.pack(w, fmt)
    et, dt = ttbfp.transpose_packed(e, d, fmt)
    c0 = ttbfp.matmul(a, e, d, fmt, cols=N)
    c1 = ttbfp.matmul_prepacked(a, et, dt, fmt, cols=N)
    np.testing.assert_array_equal(bits(c1), bits(c0))


@pytest.mark.parametrize("fmt", [ttbfp.TTBF8, ttbfp.TTBF4, ttbfp.TTBF2])
def test_transpose_packed_roundtrip(fmt):
    rng = np.random.default_rng(11)
    w = rng.standard_normal((37, 53)).astype(np.float32)
    e, d = ttbfp.pack(w, fmt)
    et, dt = ttbfp.transpose_packed(e, d, fmt)
    # Rebuild the row-major layout by hand and compare.
    nblk = e.shape[1]
    db = {8: 16, 4: 8, 2: 4}[fmt]
    e2 = np.empty_like(e)
    d2 = np.empty_like(d)
    for j in range(nblk):
        e2[:, j] = et[j]
        d2[:, j * db:(j + 1) * db] = dt[j].reshape(37, db)
    np.testing.assert_array_equal(e2, e)
    np.testing.assert_array_equal(d2, d)


@pytest.mark.parametrize("fmt", [ttbfp.TTBF8, ttbfp.TTBF4, ttbfp.TTBF2])
def test_pack_transposed_matches_reference(fmt):
    rng = np.random.default_rng(12)
    w = rng.standard_normal((9, 40)).astype(np.float32)
    e, d = ttbfp.pack(w, fmt)
    et, dt = ttbfp.pack_transposed(w, fmt)
    np.testing.assert_array_equal(et, e.T.copy())
    db = {8: 16, 4: 8, 2: 4}[fmt]
    np.testing.assert_array_equal(
        dt, d.reshape(9, -1, db).transpose(1, 0, 2).reshape(-1, 9 * db))


@pytest.mark.parametrize("fmt", [ttbfp.TTBF8, ttbfp.TTBF4, ttbfp.TTBF2])
@pytest.mark.parametrize("shape", [(1, 64, 64), (2, 33, 48), (4, 128, 256),
                                   (7, 17, 31), (16, 96, 80), (5, 1, 16),
                                   (33, 48, 32)])
def test_prepacked_s8_matches_rowmajor(fmt, shape):
    """The int8 kernel-native layout must be bitwise identical."""
    M, K, N = shape
    rng = np.random.default_rng(M * 977 + K + N)
    a = (rng.standard_normal((M, K)) * 0.5).astype(np.float32)
    w = (rng.standard_normal((K, N)) * 0.5).astype(np.float32)
    e, d = ttbfp.pack(w, fmt)
    et, dt = ttbfp.transpose_packed(e, d, fmt)
    est, s8 = ttbfp.prepack_s8(w, fmt)
    c0 = ttbfp.matmul(a, e, d, fmt, cols=N)
    c1 = ttbfp.matmul_prepacked(a, et, dt, fmt, cols=N)
    c2 = ttbfp.matmul_prepacked_s8(a, est, s8, fmt, cols=N)
    np.testing.assert_array_equal(bits(c1), bits(c0))
    np.testing.assert_array_equal(bits(c2), bits(c0))


@pytest.mark.parametrize("fmt", [ttbfp.TTBF8, ttbfp.TTBF4, ttbfp.TTBF2])
def test_convert_s8_preserves_values(fmt):
    """int8 datums must decode to exactly the same floats."""
    rng = np.random.default_rng(3)
    w = rng.standard_normal((23, 48)).astype(np.float32)
    e, d = ttbfp.pack(w, fmt)
    et, dt = ttbfp.transpose_packed(e, d, fmt)
    est, s8 = ttbfp.convert_s8(e, d, fmt)
    np.testing.assert_array_equal(est, et)
    # Decode both layouts to float and compare bit for bit.
    db = {8: 16, 4: 8, 2: 4}[fmt]
    nblk = e.shape[1]
    d2 = np.empty_like(d)
    for j in range(nblk):
        d2[:, j * db:(j + 1) * db] = dt[j].reshape(23, db)
    y0 = ttbfp.unpack(e, d, fmt, cols=48, ftz=False)
    y1 = ttbfp.unpack(est.T.copy(), d2, fmt, cols=48, ftz=False)
    np.testing.assert_array_equal(bits(y0), bits(y1))
    # A single int8 lane per datum: 16 bytes per block for both formats.
    assert s8.shape == (nblk, 23 * 16)
