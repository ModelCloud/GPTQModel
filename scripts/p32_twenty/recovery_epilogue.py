"""Experimental sm80 correction expansion/add; native INT4 remains separate."""

import triton
import triton.language as tl


@triton.jit
def expansion_add(H, B, BASE, OUT, M: tl.constexpr, N: tl.constexpr, R: tl.constexpr):
    rows = tl.program_id(0) * 16 + tl.arange(0, 16)
    cols = tl.program_id(1) * 32 + tl.arange(0, 32)
    rank = tl.arange(0, 16)
    h = tl.load(
        H + rows[:, None] * R + rank[None, :],
        (rows[:, None] < M) & (rank[None, :] < R),
        0,
    )
    b = tl.load(
        B + rank[:, None] * N + cols[None, :],
        (rank[:, None] < R) & (cols[None, :] < N),
        0,
    )
    correction = tl.dot(h, b, out_dtype=tl.float32).to(tl.float16).to(tl.float32)
    base = tl.load(
        BASE + rows[:, None] * N + cols[None, :],
        (rows[:, None] < M) & (cols[None, :] < N),
        0,
    ).to(tl.float32)
    tl.store(
        OUT + rows[:, None] * N + cols[None, :],
        base + correction,
        (rows[:, None] < M) & (cols[None, :] < N),
    )
