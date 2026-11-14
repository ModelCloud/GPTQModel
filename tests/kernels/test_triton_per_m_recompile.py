# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Per-token-count Triton kernels must not recompile for every new M.

The GGUF wide-output restore kernel and the W4A/NVFP4 activation packers used to
declare the token count (and sizes derived from it) as ``tl.constexpr``. Every
new batch size then paid a full Triton compile (tens to hundreds of
milliseconds) on the first forward.
"""

import pytest
import torch


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA or ROCm GPU required"
)


def _compiled_variants(kernel, device):
    caches = getattr(kernel, "device_caches", None)
    if caches is None:
        pytest.skip("this Triton version does not expose device_caches")
    entry = caches[device]
    return len(entry[0])


@requires_cuda
@pytest.mark.cuda
def test_gguf_wide_output_restore_reuses_one_compilation_across_m():
    from gptqmodel.nn_modules.qlinear import gguf_triton
    from gptqmodel.nn_modules.qlinear.gguf import GGUFTorchLinear
    from gptqmodel.nn_modules.qlinear.gguf_triton import GGUFTritonKernel

    in_features, out_features = 256, 64
    linear = torch.nn.Linear(in_features, out_features, bias=False, dtype=torch.float32)
    with torch.no_grad():
        linear.weight.fill_(1.0)
    source = GGUFTorchLinear(
        bits="q4_k", group_size=-1, sym=True, desc_act=False,
        in_features=in_features, out_features=out_features,
        bias=False, register_buffers=False,
    )
    source.pack_original(linear, scales=None, zeros=None)
    source.post_init()
    module = GGUFTritonKernel(
        bits="q4_k", group_size=-1, sym=True, desc_act=False,
        in_features=in_features, out_features=out_features,
        bias=False, register_buffers=True,
    )
    module.load_state_dict(source.state_dict(), strict=True)
    module = module.to("cuda").eval()

    kernel = gguf_triton._gguf_restore_wide_output_kernel
    device = torch.cuda.current_device()
    before = _compiled_variants(kernel, device) if kernel.device_caches else 0

    for rows in (16, 17, 33, 64):
        # BF16 without a bias is the wide-input path that runs the restore kernel.
        module(torch.full((rows, in_features), 300.0, device="cuda", dtype=torch.bfloat16))

    assert _compiled_variants(kernel, device) - before <= 1


@requires_cuda
@pytest.mark.cuda
def test_w4a_grouped_fp8_gemm_reuses_one_compilation_across_m():
    from gptqmodel.nn_modules.qlinear import w4a_triton

    if w4a_triton.native_fp8_gemm_error() is not None:
        pytest.skip(f"native FP8 GEMM unavailable: {w4a_triton.native_fp8_gemm_error()}")

    k, n = 256, 128
    weight = (torch.randn(k, n, device="cuda") / 4).to(torch.float8_e4m3fn)
    scales = torch.rand(k // 128, n, device="cuda", dtype=torch.float32) + 0.5
    kernel = w4a_triton._grouped_fp8_gemm
    device = torch.cuda.current_device()
    before = _compiled_variants(kernel, device) if kernel.device_caches else 0

    for rows in (16, 17, 33, 64):
        w4a_triton.fp8_linear(
            torch.randn(rows, k, device="cuda", dtype=torch.float16), weight, scales, None
        )

    assert _compiled_variants(kernel, device) - before <= 1


@requires_cuda
@pytest.mark.cuda
def test_nvfp4_pack_reuses_one_compilation_across_m():
    from gptqmodel.nn_modules.qlinear import w4a_nvfp4_triton
    from gptqmodel.utils.rocm import IS_ROCM

    if IS_ROCM:
        pytest.skip("NVFP4 activation packing is CUDA-only")

    kernel = w4a_nvfp4_triton._quantize_nvfp4_input_nvidia
    device = torch.cuda.current_device()
    before = _compiled_variants(kernel, device) if kernel.device_caches else 0
    global_scale = torch.tensor([1.0], device="cuda")

    for rows in (16, 17, 33, 64):
        w4a_nvfp4_triton.nvfp4_pack_and_swizzle(
            torch.randn(rows, 256, device="cuda", dtype=torch.float16),
            global_scale,
            recipe="nvidia",
        )

    assert _compiled_variants(kernel, device) - before <= 1
