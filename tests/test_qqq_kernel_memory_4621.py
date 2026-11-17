# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""QQQ reduce scratch must not outlive the model (issue #4621).

#4387 gave QQQ one lazily grown reduce-scratch buffer per device, and #4524
keyed that cache by ``(device, stream)`` to fix a cross-stream race.  Neither
design released the cache: every stream that ever ran a QQQ forward kept a
max-width buffer for the life of the process, so 32 pooled streams pinned
1.75 GiB on an 8B model and ``del model`` returned none of it.  ``C`` is
call-local state, so it is now allocated per call and returned to the caching
allocator.

These tests pin the two facts that make the per-call allocation correct and
cheap: the kernel never reads ``C``'s initial contents, and the steady-state
footprint does not grow with the number of streams a process has seen.
"""

import gc

import pytest
import torch

from gptqmodel.nn_modules.qlinear.qqq import QQQLinear
from gptqmodel.utils.qqq import qqq_gemm, qqq_runtime_available
from gptqmodel.utils.rocm import IS_ROCM


NATIVE_QQQ = (
    torch.cuda.is_available()
    and not IS_ROCM
    and qqq_runtime_available()
)


def _build_native_module(in_features: int, out_features: int) -> QQQLinear:
    torch.manual_seed(4621 + in_features + out_features)
    linear = torch.nn.Linear(in_features, out_features, bias=False, dtype=torch.float16)
    linear.weight.data.normal_(0, 0.02)
    module = QQQLinear(
        bits=4,
        group_size=128,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=False,
        register_buffers=True,
    )
    module.pack(
        linear,
        torch.rand((out_features, in_features // 128), dtype=torch.float16) * 0.02 + 0.01,
        torch.rand(out_features, dtype=torch.float32) * 0.001 + 0.001,
    )
    module.post_init()
    return module.eval().cuda()


@pytest.mark.cuda
@pytest.mark.skipif(not NATIVE_QQQ, reason="QQQ's native kernel is CUDA-only")
@pytest.mark.parametrize(
    ("in_features", "out_features", "rows"),
    [
        (896, 4864, 1),     # one slice: C is never read back
        (1024, 4096, 16),
        (4096, 14336, 16),  # several slices: the global reduce path
        (4096, 14336, 64),  # max_par parallel problems
    ],
)
def test_qqq_gemm_ignores_the_initial_contents_of_c(in_features, out_features, rows):
    """``C`` is write-before-read, which is why it may be uninitialized.

    A kernel that consumed ``C`` from a previous call would make the per-call
    allocation invalid, so this pins the contract the fix relies on.
    """

    module = _build_native_module(in_features, out_features)
    x = torch.randn(rows, in_features, device="cuda", dtype=torch.float16)
    with torch.inference_mode():
        quant_a, s1 = module.dynamic_quant(x)
        outputs = []
        for fill in (0, -1, 0x7FFFFFFF):
            out = torch.empty(rows, out_features, dtype=torch.float16, device="cuda")
            scratch = torch.full(
                (module.max_par * 64, out_features),
                fill,
                dtype=torch.int32,
                device="cuda",
            )
            qqq_gemm(
                quant_a, module.B, scratch, out, s1,
                module.s_channel, module.s_group, module.workspace,
                max_par=module.max_par,
            )
            outputs.append(out)
        torch.cuda.synchronize()

    assert torch.equal(outputs[0], outputs[1])
    assert torch.equal(outputs[0], outputs[2])
    assert torch.isfinite(outputs[0].float()).all()


@pytest.mark.cuda
@pytest.mark.skipif(not NATIVE_QQQ, reason="QQQ's native kernel is CUDA-only")
def test_qqq_reduce_scratch_is_freed_when_the_model_is_dropped():
    """Streams must not pin scratch after the model goes away (#4621).

    The pre-fix cache held ``32 x 56 MiB`` on an 8B-shaped layer once 32
    streams had run a forward, and stayed allocated after ``del model``.
    """

    streams = 8
    module = _build_native_module(1024, 4096)
    x = torch.randn(16, 1024, device="cuda", dtype=torch.float16)

    with torch.inference_mode():
        module(x)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated()

    workers = [torch.cuda.Stream() for _ in range(streams)]
    try:
        with torch.inference_mode():
            for stream in workers:
                with torch.cuda.stream(stream):
                    module(x)
            for stream in workers:
                stream.synchronize()
        torch.cuda.synchronize()
        live = torch.cuda.memory_allocated() - baseline
        # One max-width buffer for this layer is 64 * 16 * 4096 * 4 = 16 MiB.
        # A cache would retain one per stream; a per-call allocation retains
        # none once the forwards return.
        assert live < 2 * 2**20, f"retained {live / 2**20:.1f} MiB of scratch"
    finally:
        for stream in workers:
            stream.synchronize()

    del module, x
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    assert torch.cuda.memory_reserved() / 2**20 < baseline / 2**20 + 256
