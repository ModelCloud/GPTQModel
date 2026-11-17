# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""The #4811 packed-weight re-layout must be published only after its copy ran.

``_permute_packed_weight`` transposes the packed storage once per module and
then exposes ``qweight`` as a view of it.  ``copy_`` only *enqueues* that
transpose on the calling stream, so publishing ``self.qweight`` straight away
let a second cold build on another stream read bytes the copy had not written
yet.  Its derived cache entry (``scale``/``d``/...) was then built from
uninitialized memory and kept for the rest of the session, returning garbage
(including NaN) even on later single-threaded forwards (#4879).

Every read below happens on a non-default stream: the legacy default stream is
implicitly ordered against other blocking streams, which hides the race.
"""

from __future__ import annotations

import threading

import pytest
import torch

from gptqmodel.nn_modules.qlinear.gguf import GGUFTorchLinear
from gptqmodel.nn_modules.qlinear.gguf_triton import GGUFTritonKernel


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA or ROCm GPU required"
)

# Long enough that the transpose queued behind it cannot have run by the time
# ``_permute_packed_weight`` returns on a tree that does not wait (#4879).
_BUSY_CYCLES = 200_000_000


def _triton_modules(count: int, in_features: int = 2048, out_features: int = 1024):
    """``count`` Triton modules that share one checkpoint's packed weights."""

    torch.manual_seed(4879)
    linear = torch.nn.Linear(in_features, out_features, bias=False, dtype=torch.float32)
    with torch.no_grad():
        linear.weight.normal_(mean=0.0, std=0.03)

    torch_module = GGUFTorchLinear(
        bits="q6_k",
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=False,
        register_buffers=False,
    )
    torch_module.pack_original(linear, scales=None, zeros=None)
    torch_module.post_init()
    state = torch_module.state_dict()

    modules = []
    for _ in range(count):
        module = GGUFTritonKernel(
            bits="q6_k",
            group_size=-1,
            sym=True,
            desc_act=False,
            in_features=in_features,
            out_features=out_features,
            bias=False,
            register_buffers=True,
        )
        module.load_state_dict(state, strict=True)
        modules.append(module.to("cuda").eval())
    return modules


@requires_cuda
@pytest.mark.cuda
def test_build_started_after_the_publish_reads_finished_bytes():
    """The issue's mechanism: a second cold build while the copy is pending."""

    reference_module, module = _triton_modules(2)
    x = torch.randn(8, 2048, device="cuda", dtype=torch.float16)
    torch.cuda.synchronize()
    with torch.inference_mode():
        reference = reference_module(x).clone()

    side = torch.cuda.Stream()
    reader = torch.cuda.Stream()

    def permute() -> None:
        with torch.cuda.stream(side):
            torch.cuda._sleep(_BUSY_CYCLES)
            module._permute_packed_weight()

    worker = threading.Thread(target=permute)
    worker.start()
    # Wait for the module to advertise the permuted layout: the copy that
    # fills it must already have run when that flag flips.
    while not module._packed_weight_is_permuted():
        pass
    worker.join()

    with torch.inference_mode(), torch.cuda.stream(reader):
        output = module(x).clone()
    reader.synchronize()

    assert torch.equal(output, reference), (
        "the cache was built from bytes the re-layout copy had not written"
    )


@requires_cuda
@pytest.mark.cuda
def test_concurrent_cold_builds_match_the_serial_reference():
    """Concurrent cold first forwards must not bake garbage into the cache."""

    reference_module, module = _triton_modules(2)
    x = torch.randn(8, 2048, device="cuda", dtype=torch.float16)
    torch.cuda.synchronize()
    with torch.inference_mode():
        reference = reference_module(x).clone()

    threads_n = 4
    streams = [torch.cuda.Stream() for _ in range(threads_n)]
    outputs: list[torch.Tensor | None] = [None] * threads_n
    barrier = threading.Barrier(threads_n)
    errors: list[BaseException] = []

    def work(index: int) -> None:
        try:
            with torch.inference_mode(), torch.cuda.stream(streams[index]):
                barrier.wait()
                outputs[index] = module(x).clone()
                streams[index].synchronize()
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    workers = [threading.Thread(target=work, args=(i,)) for i in range(threads_n)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()

    assert not errors, errors
    for index, output in enumerate(outputs):
        assert torch.equal(output, reference), f"thread {index} diverged"

    # The published cache entry must stay correct for later serial forwards.
    with torch.inference_mode():
        assert torch.equal(module(x), reference)
