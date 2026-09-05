"""Experimental double-buffered correction GEMM; never imported by production.

Pipeline adapted from ROCm/gfx950-gluon-tutorials v4_global_prefetch at
4d7d632a320b25a789bbdb7a9ba8a8683dce2142. Layouts, correction, tails and
buffer-reuse synchronization are specialized here for QVQ experiments.

MIT License
Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd.cdna4 import async_copy, mfma
from triton.language import range as loop_range


@gluon.jit
def _consume(x_shared, high_shared, low_shared, index, primary, correction,
             mma: gl.constexpr, interleave: gl.constexpr):
    a = x_shared.index(index).load(gl.DotOperandLayout(0, mma, 8))
    b = high_shared.index(index).load(gl.DotOperandLayout(1, mma, 8))
    primary = mfma(a, b, primary)
    c = low_shared.index(index).load(gl.DotOperandLayout(1, mma, 8))
    if interleave:
        primary = mfma(a, c, primary)
    else:
        correction = mfma(a, c, correction)
    return primary, correction


@gluon.jit
def _load_tile(xs, hs, ls, index, mma: gl.constexpr):
    a = xs.index(index).load(gl.DotOperandLayout(0, mma, 8))
    b = hs.index(index).load(gl.DotOperandLayout(1, mma, 8))
    c = ls.index(index).load(gl.DotOperandLayout(1, mma, 8))
    return a, b, c


@gluon.jit
def _accumulate(a, b, c, primary, correction, interleave: gl.constexpr):
    primary = mfma(a, b, primary)
    if interleave:
        primary = mfma(a, c, primary)
    else:
        correction = mfma(a, c, correction)
    return primary, correction


@gluon.jit
def folded_residual_prefetch_kernel(
    x_ptr, high_ptr, low_ptr, output_ptr,
    size_m: gl.constexpr, size_n: gl.constexpr, size_k: gl.constexpr,
    block_m: gl.constexpr, block_n: gl.constexpr, block_k: gl.constexpr,
    interleave: gl.constexpr = False, prune_masks: gl.constexpr = True,
    register_prefetch: gl.constexpr = False,
):
    gl.static_assert(size_k >= block_k and size_k % block_k == 0)
    gl.static_assert((block_k == 32 or block_k == 64) and (block_m == 64 or block_m == 128)
                     and (block_n == 64 or block_n == 128))
    # Match the physical vector/lane/wave order of global-to-LDS transfers.
    aw: gl.constexpr = [[1, 0], [2, 0]]
    bw: gl.constexpr = [[0, 1], [0, 2]]
    if block_k == 32:
        ar: gl.constexpr = [[0, 1], [0, 2], [0, 4]] + ([[4, 0]] if block_m == 128 else [])
        al: gl.constexpr = [[0, 8], [0, 16]] + (
            [[8, 0], [16, 0], [32, 0], [64, 0]] if block_m == 128 else [[4, 0], [8, 0], [16, 0], [32, 0]])
        br: gl.constexpr = [[1, 0], [2, 0], [4, 0]] + ([[0, 4]] if block_n == 128 else [])
        bl: gl.constexpr = [[8, 0], [16, 0]] + (
            [[0, 8], [0, 16], [0, 32], [0, 64]] if block_n == 128 else [[0, 4], [0, 8], [0, 16], [0, 32]])
    else:
        ar: gl.constexpr = [[0, 1], [0, 2], [0, 4], [4, 0]] + ([[8, 0]] if block_m == 128 else [])
        al: gl.constexpr = [[0, 8], [0, 16], [0, 32]] + (
            [[16, 0], [32, 0], [64, 0]] if block_m == 128 else [[8, 0], [16, 0], [32, 0]])
        br: gl.constexpr = [[1, 0], [2, 0], [4, 0], [0, 4]] + ([[0, 8]] if block_n == 128 else [])
        bl: gl.constexpr = [[8, 0], [16, 0], [32, 0]] + (
            [[0, 16], [0, 32], [0, 64]] if block_n == 128 else [[0, 8], [0, 16], [0, 32]])
    a_layout: gl.constexpr = gl.DistributedLinearLayout(ar, al, aw, [], [block_m, block_k])
    b_layout: gl.constexpr = gl.DistributedLinearLayout(br, bl, bw, [], [block_k, block_n])
    a_shared_layout: gl.constexpr = gl.PaddedSharedLayout(
        [[512, 16]], ar[:3] + al + aw + ar[3:], [], [block_m, block_k])
    b_shared_layout: gl.constexpr = gl.PaddedSharedLayout(
        [[512, 16]], br[:3] + bl + bw + br[3:], [], [block_k, block_n])
    # Match MFMA operand ownership to the vector/lane order of the padded LDS tiles.
    mma: gl.constexpr = gl.amd.AMDMFMALayout(4, [16, 16, 32], True, [2, 2])
    xs = gl.allocate_shared_memory(gl.float16, [2, block_m, block_k], a_shared_layout)
    hs = gl.allocate_shared_memory(gl.float16, [2, block_k, block_n], b_shared_layout)
    ls = gl.allocate_shared_memory(gl.float16, [2, block_k, block_n], b_shared_layout)
    rows = gl.program_id(0) * block_m + gl.arange(0, block_m, layout=gl.SliceLayout(1, a_layout))
    ak = gl.arange(0, block_k, layout=gl.SliceLayout(0, a_layout))
    columns = gl.program_id(1) * block_n + gl.arange(0, block_n, layout=gl.SliceLayout(0, b_layout))
    bk = gl.arange(0, block_k, layout=gl.SliceLayout(1, b_layout))
    ao = rows[:, None] * size_k + ak[None, :]
    bo = bk[:, None] + columns[None, :] * size_k
    amask = gl.full((block_m, 1), True, gl.int1, a_layout)
    bmask = gl.full((1, block_n), True, gl.int1, b_layout)
    if not prune_masks or size_m % block_m != 0:
        amask = rows[:, None] < size_m
    if not prune_masks or size_n % block_n != 0:
        bmask = columns[None, :] < size_n
    async_copy.buffer_load_to_shared(xs.index(0), x_ptr, ao, amask, 0)
    async_copy.buffer_load_to_shared(hs.index(0), high_ptr, bo, bmask, 0)
    async_copy.buffer_load_to_shared(ls.index(0), low_ptr, bo, bmask, 0)
    async_copy.commit_group()
    primary = gl.full((block_m, block_n), 0, gl.float32, mma)
    correction = gl.full((block_m, block_n), 0, gl.float32, mma)
    if register_prefetch and size_k // block_k > 1:
        # Prime two LDS tiles and carry the first tile's operands in registers.
        async_copy.buffer_load_to_shared(xs.index(1), x_ptr + block_k, ao, amask, 0)
        async_copy.buffer_load_to_shared(hs.index(1), high_ptr + block_k, bo, bmask, 0)
        async_copy.buffer_load_to_shared(ls.index(1), low_ptr + block_k, bo, bmask, 0)
        async_copy.commit_group()
        async_copy.wait_group(1)
        a, b, c = _load_tile(xs, hs, ls, 0, mma)
        gl.barrier()
        for tile in loop_range(size_k // block_k - 2, loop_unroll_factor=2):
            write_index = tile % 2
            next_index = 1 - write_index
            offset = (tile + 2) * block_k
            async_copy.buffer_load_to_shared(xs.index(write_index), x_ptr + offset, ao, amask, 0)
            async_copy.buffer_load_to_shared(hs.index(write_index), high_ptr + offset, bo, bmask, 0)
            async_copy.buffer_load_to_shared(ls.index(write_index), low_ptr + offset, bo, bmask, 0)
            async_copy.commit_group()
            async_copy.wait_group(1)
            next_a, next_b, next_c = _load_tile(xs, hs, ls, next_index, mma)
            primary, correction = _accumulate(a, b, c, primary, correction, interleave)
            a, b, c = next_a, next_b, next_c
            # All waves have read next_index before it can be reused next iteration.
            gl.barrier()
        async_copy.wait_group(0)
        last_a, last_b, last_c = _load_tile(xs, hs, ls, (size_k // block_k - 1) % 2, mma)
        primary, correction = _accumulate(a, b, c, primary, correction, interleave)
        primary, correction = _accumulate(last_a, last_b, last_c, primary, correction, interleave)
    else:
        for tile in range(size_k // block_k - 1):
            read_index = tile % 2
            write_index = 1 - read_index
            offset = (tile + 1) * block_k
            async_copy.buffer_load_to_shared(xs.index(write_index), x_ptr + offset, ao, amask, 0)
            async_copy.buffer_load_to_shared(hs.index(write_index), high_ptr + offset, bo, bmask, 0)
            async_copy.buffer_load_to_shared(ls.index(write_index), low_ptr + offset, bo, bmask, 0)
            async_copy.commit_group()
            async_copy.wait_group(1)
            primary, correction = _consume(xs, hs, ls, read_index, primary, correction, mma, interleave)
            # Every wave must finish reading this slot before the next iteration reuses it.
            gl.barrier()
        async_copy.wait_group(0)
        primary, correction = _consume(xs, hs, ls, (size_k // block_k - 1) % 2,
                                       primary, correction, mma, interleave)
    result = primary if interleave else primary + correction
    om = gl.program_id(0) * block_m + gl.arange(0, block_m, layout=gl.SliceLayout(1, mma))
    on = gl.program_id(1) * block_n + gl.arange(0, block_n, layout=gl.SliceLayout(0, mma))
    if prune_masks and size_m % block_m == 0 and size_n % block_n == 0:
        gl.store(output_ptr + om[:, None] * size_n + on[None, :], result)
    else:
        gl.store(output_ptr + om[:, None] * size_n + on[None, :], result,
                 (om[:, None] < size_m) & (on[None, :] < size_n))
