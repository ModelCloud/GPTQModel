# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import re

import pytest
import torch

from gptqmodel.utils import macaq_triton


@pytest.mark.parametrize("sm", [100, 103, 110, 120, 121])
@pytest.mark.parametrize("rows", [1, 129])
@pytest.mark.parametrize("fp4", [False, True])
def test_v41_blackwell_mx_emits_block_scaled_mma(sm, rows, fp4):
    triton = pytest.importorskip("triton")
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    kernel = macaq_triton._dense_mx_gemm_kernel
    tiles = macaq_triton.native_mx_launch_config(divmod(sm, 10), rows, fp4)
    options = {key: tiles.pop(key) for key in ("num_warps", "num_stages")}
    constants = dict(
        OUT_SCALE=1.0, HAS_BIAS=False, A_FORMAT="e4m3",
        B_FORMAT="e2m1" if fp4 else "e4m3", SCALE_ROWS=1 if fp4 else 32, **tiles,
    )
    signature = dict.fromkeys(kernel.arg_names, "i32")
    signature.update(dict.fromkeys(constants, "constexpr"))
    signature.update(a_ptr="*u8", b_ptr="*u8", asc_ptr="*u8", bsc_ptr="*u8", c_ptr="*bf16", bias_ptr="*u8")
    compiled = triton.compile(ASTSource(kernel, signature, constants), target=GPUTarget("cuda", sm, 32), options=options)
    op = r"tcgen05\.mma" if sm < 120 else r"mma\.sync"
    assert re.search(op + r"[\w:.]*\.block_scale", compiled.asm["ptx"])


@pytest.mark.cuda
def test_v41_fp4_to_fp8_tile_preserves_all_codes():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the Triton bit conversion")
    triton = pytest.importorskip("triton")
    import triton.language as tl

    from gptqmodel.utils.macaq_triton import _fp4_to_fp8_codes

    @triton.jit
    def expand(source, output):
        packed = tl.load(source + tl.arange(0, 256))
        tl.store(output + tl.arange(0, 512), _fp4_to_fp8_codes(packed))

    raw = torch.arange(256, dtype=torch.int32)
    values = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6])
    codes = torch.stack((raw & 15, raw >> 4), dim=-1).flatten()
    expected = values[codes].to(torch.float8_e4m3fn).view(torch.uint8)
    result = torch.empty_like(expected, device="cuda")
    expand[(1,)](raw.to(device="cuda", dtype=torch.uint8), result)
    assert torch.equal(result.cpu(), expected)
