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


@pytest.mark.parametrize("fmt", [ttbfp.BF8_B, ttbfp.BF4_B, ttbfp.BF2_B])
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


@pytest.mark.parametrize("fmt", [ttbfp.BF8_B, ttbfp.BF4_B, ttbfp.BF2_B])
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("ftz", [True, False])
def test_unpack_matches_reference(fmt, shape, ftz):
    rng = np.random.default_rng(1234 + shape[1])
    x = make_input(rng, shape)
    e, d = ttbfp.pack(x, fmt)
    y_ref = ref.dequantize(e, d, fmt, cols=shape[1], ftz=ftz)
    y_c = ttbfp.unpack(e, d, fmt, cols=shape[1], ftz=ftz)
    np.testing.assert_array_equal(bits(y_c), bits(y_ref))


@pytest.mark.parametrize("fmt", [ttbfp.BF8_B, ttbfp.BF4_B, ttbfp.BF2_B])
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


@pytest.mark.parametrize("fmt", [ttbfp.BF8_B, ttbfp.BF4_B, ttbfp.BF2_B])
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


@pytest.mark.parametrize("fmt", [ttbfp.BF8_B, ttbfp.BF4_B, ttbfp.BF2_B])
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


@pytest.mark.parametrize("fmt", [ttbfp.BF8_B, ttbfp.BF4_B, ttbfp.BF2_B])
def test_pack_transposed_matches_reference(fmt):
    rng = np.random.default_rng(12)
    w = rng.standard_normal((9, 40)).astype(np.float32)
    e, d = ttbfp.pack(w, fmt)
    et, dt = ttbfp.pack_transposed(w, fmt)
    np.testing.assert_array_equal(et, e.T.copy())
    db = {8: 16, 4: 8, 2: 4}[fmt]
    np.testing.assert_array_equal(
        dt, d.reshape(9, -1, db).transpose(1, 0, 2).reshape(-1, 9 * db))


@pytest.mark.parametrize("fmt", [ttbfp.BF8_B, ttbfp.BF4_B, ttbfp.BF2_B])
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


@pytest.mark.parametrize("fmt", [ttbfp.BF8_B, ttbfp.BF4_B, ttbfp.BF2_B])
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


# ---------------------------------------------------------------------------
# Regression tests for the native-kernel review of PR #3808.
# ---------------------------------------------------------------------------

def _matmul_into(out, a, exps, data, fmt, cols, *, bf16=False, ftz=True):
    """Call the C entry point with a caller-provided output buffer.

    This lets a test prefill ``out`` with a sentinel, so a kernel that skips
    a row tile leaves an unmistakable value instead of whatever ``np.empty``
    happened to contain.
    """
    a = np.ascontiguousarray(a, dtype=np.uint16 if bf16 else np.float32)
    exps = np.ascontiguousarray(exps, dtype=np.uint8)
    data = np.ascontiguousarray(data, dtype=np.uint8)
    M, K = a.shape
    fn = ttbfp.lib.ttbfp_matmul_bf16 if bf16 else ttbfp.lib.ttbfp_matmul
    ap = a.ctypes.data_as(ttbfp._u16p if bf16 else ttbfp._f32p)
    fn(int(fmt), ap, K, exps.ctypes.data_as(ttbfp._u8p),
       data.ctypes.data_as(ttbfp._u8p), M, K, int(cols),
       out.ctypes.data_as(ttbfp._f32p), int(cols), int(bool(ftz)))


@pytest.mark.parametrize("fmt", [ttbfp.BF8_B, ttbfp.BF4_B, ttbfp.BF2_B])
@pytest.mark.parametrize("entry", ["f32", "bf16"])
@pytest.mark.parametrize(
    "N", [16, 17, 33], ids=["full", "partial", "two-full-plus-partial"])
def test_matmul_all_row_remainders(fmt, entry, N, monkeypatch):
    """Every M remainder must be written, in both column-block switches.

    The sign-magnitude AVX-512 path selects a 16-row tile, so M = 16 + r for
    r in 1..15 used to leave the r-row remainder untouched.
    """
    rng = np.random.default_rng(0xB3808 + fmt * 100 + N)
    K = 8
    w = (rng.standard_normal((K, N)) * 0.5).astype(np.float32)
    e, d = ttbfp.pack(w, fmt)
    for rem in range(1, 16):
        M = 16 + rem
        a = (rng.standard_normal((M, K)) * 0.5).astype(np.float32)
        a_in = ((a.view(np.uint32) >> 16).astype(np.uint16)
                if entry == "bf16" else a)
        bf16 = entry == "bf16"

        monkeypatch.setenv("TTBFP_MRT", "16")
        got = np.full((M, N), np.float32(1e30), dtype=np.float32)
        _matmul_into(got, a_in, e, d, fmt, N, bf16=bf16)

        monkeypatch.setenv("TTBFP_MRT", "4")
        want = np.full((M, N), np.float32(-1e30), dtype=np.float32)
        _matmul_into(want, a_in, e, d, fmt, N, bf16=bf16)

        np.testing.assert_array_equal(bits(got), bits(want))


def test_scalar_bf2_decode_reads_two_bit_fields(tmp_path):
    """The scalar fallback must decode bf2_b as four 2-bit fields per byte.

    Block 0 is 0x55 in all four bytes (sixteen +1 datums).  A nibble decoder
    reads eight bytes and pulls block 1's zero bytes into lanes 8..15, so the
    first block decodes as eight ones followed by eight zeros.
    """
    e = np.full((1, 2), 127, dtype=np.uint8)
    d = np.zeros((1, 8), dtype=np.uint8)
    d[0, :4] = 0x55
    a = np.ones((1, 1), dtype=np.float32)
    np.savez(tmp_path / "case.npz", a=a, e=e, d=d)
    script = """
import numpy as np, sys, ttbfp
z = np.load(sys.argv[1])
c = ttbfp.matmul(z['a'], z['e'], z['d'], 2, cols=32)
np.save(sys.argv[2], c)
"""
    out = tmp_path / "out.npy"
    env = dict(os.environ)
    env.update(PYTHONPATH=str(ROOT / "python"),
               TTBFP_LIB=str(ROOT / "libttbfp.so"),
               TTBFP_FORCE_SCALAR="1")
    subprocess.run([sys.executable, "-c", script, str(tmp_path / "case.npz"),
                    str(out)], check=True, env=env, cwd=str(ROOT))
    c = np.load(out)
    np.testing.assert_array_equal(c[0, :16], np.ones(16, dtype=np.float32))
    np.testing.assert_array_equal(c[0, 16:], np.zeros(16, dtype=np.float32))


@pytest.mark.parametrize(
    "fmt,value", [(ttbfp.BF8_B, 2.0 ** -126), (ttbfp.BF4_B, 2.0 ** -126)],
    ids=["bf8_b", "bf4_b"])
def test_normal_decode_survives_ambient_ftz(fmt, value, tmp_path):
    """A normal decoded weight must not be flushed via a subnormal scale.

    The scale table is initialized while FTZ is on, then FTZ is left on for
    one matmul and turned off for another.  2^126 * 2^-126 must be exactly 1.
    """
    pytest.importorskip("torch")
    np.savez(tmp_path / "case.npz",
             x=np.full((1, 16), np.float32(value), dtype=np.float32))
    script = """
import numpy as np, sys, torch, ttbfp
z = np.load(sys.argv[1])
fmt = int(sys.argv[2])
torch.set_flush_denormal(True)
e, d = ttbfp.pack(z['x'], fmt)
ttbfp.unpack(e, d, fmt, cols=16)          # initialize the static table
a = np.array([[np.float32(2.0 ** 126)]], dtype=np.float32)
c_on = ttbfp.matmul(a, e, d, fmt, cols=16)
torch.set_flush_denormal(False)
c_off = ttbfp.matmul(a, e, d, fmt, cols=16)
np.savez(sys.argv[3], on=c_on, off=c_off)
"""
    out = tmp_path / "out.npz"
    env = dict(os.environ)
    env.update(PYTHONPATH=str(ROOT / "python"),
               TTBFP_LIB=str(ROOT / "libttbfp.so"))
    subprocess.run([sys.executable, "-c", script, str(tmp_path / "case.npz"),
                    str(fmt), str(out)], check=True, env=env, cwd=str(ROOT))
    z = np.load(out)
    np.testing.assert_array_equal(z["on"], np.ones((1, 16), dtype=np.float32))
    np.testing.assert_array_equal(z["off"], np.ones((1, 16), dtype=np.float32))


@pytest.mark.parametrize(
    "fmt,db", [(ttbfp.BF8_B, 16), (ttbfp.BF4_B, 8), (ttbfp.BF2_B, 4)])
def test_rejects_malformed_packed_buffers(fmt, db):
    """Short or mis-shaped packed buffers must raise, not reach ctypes."""
    rng = np.random.default_rng(404)
    w = rng.standard_normal((5, 32)).astype(np.float32)
    e, d = ttbfp.pack(w, fmt)
    assert d.shape == (5, 2 * db)
    a = np.ones((2, 5), dtype=np.float32)
    a_bf = np.ones((2, 5), dtype=np.uint16)

    with pytest.raises(ValueError):
        ttbfp.unpack(e, d[:, :-1], fmt)
    with pytest.raises(ValueError):
        ttbfp.unpack(e[0], d, fmt)
    with pytest.raises(ValueError):
        ttbfp.matmul(a, e, d[:, :-1], fmt)
    with pytest.raises(ValueError):
        ttbfp.matmul_bf16(a_bf, e, d[:, :-1], fmt)

    et, dt = ttbfp.transpose_packed(e, d, fmt)
    assert dt.shape == (2, 5 * db)
    with pytest.raises(ValueError):
        ttbfp.matmul_prepacked(a, et, dt[:, :-1], fmt)
    with pytest.raises(ValueError):
        ttbfp.matmul_prepacked_bf16(a_bf, et, dt[:, :-1], fmt)

    est, s8 = ttbfp.prepack_s8(w, fmt)
    with pytest.raises(ValueError):
        ttbfp.matmul_prepacked_s8(a, est, s8[:, :-1], fmt)
    with pytest.raises(ValueError):
        ttbfp.matmul_prepacked_s8_bf16(a_bf, est, s8[:, :-1], fmt)

    blob = ttbfp.tile_pack(np.zeros((32, 32), dtype=np.float32), fmt)
    with pytest.raises(ValueError):
        ttbfp.tile_unpack(blob[:-1], fmt)


def test_flat_cols_must_preserve_stored_stride():
    """Cropping whole blocks changes the flat row stride; reject it."""
    w = np.array([[1.0] * 16 + [11.0] * 16,
                  [2.0] * 16 + [12.0] * 16,
                  [3.0] * 16 + [13.0] * 16], dtype=np.float32)
    e, d = ttbfp.pack(w, ttbfp.BF8_B)
    with pytest.raises(ValueError):
        ttbfp.unpack(e, d, ttbfp.BF8_B, cols=16)
    y = ttbfp.unpack(e, d, ttbfp.BF8_B, cols=17)
    np.testing.assert_allclose(y[:, 0], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(y[:, 16], [11.0, 12.0, 13.0])

    a = np.eye(3, dtype=np.float32)
    with pytest.raises(ValueError):
        ttbfp.matmul(a, e, d, ttbfp.BF8_B, cols=16)
    c = ttbfp.matmul(a, e, d, ttbfp.BF8_B, cols=17)
    np.testing.assert_allclose(c[:, 0], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(c[:, 16], [11.0, 12.0, 13.0])


@pytest.mark.parametrize(
    "fmt,db,datum", [(ttbfp.BF8_B, 16, 0x20), (ttbfp.BF4_B, 8, 0x02)],
    ids=["bf8_b", "bf4_b"])
@pytest.mark.parametrize(
    "force", ["", "TTBFP_FORCE_SCALAR", "TTBFP_FORCE_AVX2"],
    ids=["avx512", "scalar", "avx2"])
def test_ftz_false_preserves_subnormal_weight(fmt, db, datum, force, tmp_path):
    """ftz=False must survive ambient FTZ/DAZ, including DAZ on FMA inputs.

    bf8_b e=1/mag=32 and bf4_b e=1/mag=2 both decode to 2^-127, a subnormal.
    With A=2^126 the product is exactly 0.5.  ftz=True must still flush it.
    """
    pytest.importorskip("torch")
    script = """
import numpy as np, sys, torch, ttbfp
fmt, db, datum = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
torch.set_flush_denormal(True)
e = np.array([[1]], np.uint8)
d = np.zeros((1, db), np.uint8)
d[0, 0] = datum
a = np.array([[2.0 ** 126]], np.float32)
off = ttbfp.matmul(a, e, d, fmt, cols=16, ftz=False)
on = ttbfp.matmul(a, e, d, fmt, cols=16, ftz=True)
y = ttbfp.unpack(e, d, fmt, cols=16, ftz=False)
s8 = np.zeros((1, 16), np.uint8)
s8[0, 0] = datum
s8_off = ttbfp.matmul_prepacked_s8(a, e.T.copy(), s8, fmt, cols=16, ftz=False)
np.savez(sys.argv[4], off=off, on=on, y=y, s8=s8_off)
"""
    out = tmp_path / "out.npz"
    env = dict(os.environ)
    env.update(PYTHONPATH=str(ROOT / "python"),
               TTBFP_LIB=str(ROOT / "libttbfp.so"))
    if force:
        env[force] = "1"
    subprocess.run([sys.executable, "-c", script, str(fmt), str(db),
                    str(datum), str(out)], check=True, env=env, cwd=str(ROOT))
    z = np.load(out)
    assert z["off"][0, 0] == np.float32(0.5)
    assert np.all(z["off"][0, 1:] == 0.0)
    assert np.all(z["on"] == 0.0)
    assert z["s8"][0, 0] == np.float32(0.5)
    assert np.all(z["s8"][0, 1:] == 0.0)
    assert z["y"].view(np.uint32)[0, 0] == 0x00400000
