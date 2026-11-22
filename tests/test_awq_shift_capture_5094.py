# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""The AWQ extraction-offset table must never be published from a capture (#5094).

#5053 replaced the per-call host-built shift tensor in ``dequantize_gemm()``
with the process-global ``_AWQ_SHIFTS_CACHE``, keyed by
``(device.type, device.index, bits)`` and built with device arithmetic so a
CUDA-graph capture no longer trips over a host-to-device copy.

That fixed capture, but device work recorded inside a capture does not run: a
table first built inside ``torch.cuda.graph(...)`` holds uninitialized memory
(zeros in practice) and is still stored in the global cache. If the capture is
abandoned, every later eager AWQ dense dequant on that device decodes with
those zeros. A capture that succeeds but has not been replayed has the same
problem for an eager call in between.

These tests force a cold table (the state of a fresh process), attempt a
capture first, and then compare eager output against an independent CPU
reference. They cover the shared ``dequantize_gemm()`` helper, its
``AwqTorchLinear`` and dense ``ParoLinear`` consumers, and the CPU path.

#5191 added the other half of the publication rule: device arithmetic is only
*enqueued* on the caller's stream, so a table published before that stream has
run is uninitialized for every other stream. A second test makes the first
call on a stream that is still busy and the next call on another stream, then
compares both with the serial output.
"""

import threading

import pytest
import torch

from gptqmodel.nn_modules.qlinear.paro import ParoLinear
from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
from gptqmodel.quantization.awq.utils.packing_utils import (
    _AWQ_SHIFTS_CACHE,
    AWQ_REVERSE_ORDER,
    _awq_shifts,
    dequantize_gemm,
    pack_awq,
)
from gptqmodel.utils.paro import build_identity_rotation_buffers


CUDA_AVAILABLE = torch.cuda.is_available()
CUDA_DEVICE = torch.device("cuda", 0)
CUDA_KEY = ("cuda", 0, 4)
EXPECTED_SHIFTS = [lane * 4 for lane in AWQ_REVERSE_ORDER]


@pytest.fixture
def cold_shift_cache():
    """Run with an empty cache, then restore the process cache.

    Clearing reproduces the first dense AWQ dequant of a fresh process, which
    is the only state in which the defect is reachable. The saved entries are
    restored so a test cannot change a later test's warm-cache behaviour.
    """

    saved = dict(_AWQ_SHIFTS_CACHE)
    _AWQ_SHIFTS_CACHE.clear()
    try:
        yield
    finally:
        _AWQ_SHIFTS_CACHE.clear()
        _AWQ_SHIFTS_CACHE.update(saved)


def _packed_layer(in_features: int, out_features: int, group_size: int, seed: int = 0):
    """Deterministic AWQ layer; the CPU dequant is the independent reference."""

    torch.manual_seed(seed)
    groups = in_features // group_size
    int_weight = torch.randint(0, 16, (in_features, out_features), dtype=torch.int32)
    zero_points = torch.randint(0, 16, (groups, out_features), dtype=torch.int32)
    qweight, qzeros = pack_awq(int_weight, zero_points, 4)
    scales = (torch.rand(groups, out_features, dtype=torch.float32) * 0.01).to(
        torch.float16
    )
    return qweight, qzeros, scales


def _abandon_capture(body) -> BaseException:
    """Run ``body`` inside a capture, then fail on a later, uncapturable op."""

    graph = torch.cuda.CUDAGraph()
    with pytest.raises(RuntimeError) as info:
        with torch.cuda.graph(graph):
            body()
            # A host-to-device copy is rejected inside a capture. It stands in
            # for any later op in the caller's forward that cannot be captured.
            torch.ones(1).cuda()
    del graph
    return info.value


def _awq_module(device: torch.device, dtype: torch.dtype) -> AwqTorchLinear:
    in_features, out_features, group_size = 256, 128, 64
    qweight, qzeros, scales = _packed_layer(in_features, out_features, group_size)
    bias = torch.randn(out_features, dtype=dtype)

    module = AwqTorchLinear(
        bits=4,
        group_size=group_size,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=True,
        dtype=dtype,
        register_buffers=True,
    )
    module.qweight.copy_(qweight)
    module.qzeros.copy_(qzeros)
    module.scales.copy_(scales)
    module.bias.copy_(bias)
    module.post_init()
    module.eval()
    return module.to(device=device)


def _paro_module(device: torch.device, dtype: torch.dtype) -> ParoLinear:
    in_features, out_features, group_size = 256, 128, 128
    qweight, qzeros, scales = _packed_layer(in_features, out_features, group_size)
    bias = torch.randn(out_features, dtype=dtype)

    module = ParoLinear(
        bits=4,
        group_size=group_size,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=True,
        register_buffers=True,
        krot=8,
    )
    module.qweight.copy_(qweight)
    module.qzeros.copy_(qzeros)
    module.scales.copy_(scales)
    module.bias.copy_(bias)
    pairs, theta, channel_scales = build_identity_rotation_buffers(
        in_features=in_features,
        group_size=group_size,
        krot=8,
        dtype=dtype,
    )
    module.pairs.copy_(pairs)
    module.theta.copy_(theta)
    module.channel_scales.copy_(channel_scales)
    module.post_init()
    module.eval()
    return module.to(device=device, dtype=dtype)


def _dense_reference(x: torch.Tensor, module) -> torch.Tensor:
    """FP64 oracle from the CPU-dequantized weight, independent of the cache."""

    weight = dequantize_gemm(
        module.qweight.cpu(),
        module.qzeros.cpu(),
        module.scales.cpu(),
        module.bits,
        module.group_size,
    )
    bias = module.bias.cpu().double() if module.bias is not None else 0.0
    return (x.cpu().double() @ weight.double() + bias).to(x.dtype)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="needs a CUDA/ROCm GPU")
def test_abandoned_capture_does_not_poison_dequantize_gemm(cold_shift_cache):
    qweight, qzeros, scales = _packed_layer(1024, 512, 128)
    reference = dequantize_gemm(qweight, qzeros, scales, 4, 128)
    qweight = qweight.cuda()
    qzeros = qzeros.cuda()
    scales = scales.cuda()

    error = _abandon_capture(lambda: dequantize_gemm(qweight, qzeros, scales, 4, 128))
    assert "CUDA graph capture" in str(error), (
        "the fixture must actually attempt a capture and fail on the later op: "
        f"{error!r}"
    )

    # The abandoned capture must not have published a table at all.
    assert CUDA_KEY not in _AWQ_SHIFTS_CACHE, (
        "a capture published an extraction table; an abandoned capture leaves "
        "it holding never-computed values"
    )

    actual = dequantize_gemm(qweight, qzeros, scales, 4, 128)
    assert torch.equal(actual.cpu(), reference), (
        "eager dequant after an abandoned capture decoded different weights: "
        f"max |diff| = {(actual.float().cpu() - reference.float()).abs().max().item()}"
    )
    assert _AWQ_SHIFTS_CACHE[CUDA_KEY].tolist() == EXPECTED_SHIFTS


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="needs a CUDA/ROCm GPU")
def test_eager_call_before_replay_uses_a_computed_table(cold_shift_cache):
    qweight, qzeros, scales = _packed_layer(256, 128, 64)
    reference = dequantize_gemm(qweight, qzeros, scales, 4, 64)
    qweight = qweight.cuda()
    qzeros = qzeros.cuda()
    scales = scales.cuda()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = dequantize_gemm(qweight, qzeros, scales, 4, 64)

    # The capture has not been replayed, so its table is still uncomputed.
    eager = dequantize_gemm(qweight, qzeros, scales, 4, 64)
    assert torch.equal(eager.cpu(), reference), (
        "an eager dequant before the first replay consumed the capture's "
        f"never-computed table: max |diff| = "
        f"{(eager.float().cpu() - reference.float()).abs().max().item()}"
    )

    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(captured.cpu(), reference), "the replayed capture must still be correct"
    del graph


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="needs a CUDA/ROCm GPU")
def test_successful_capture_does_not_publish_shift_table(cold_shift_cache):
    qweight, qzeros, scales = _packed_layer(256, 128, 64)
    qweight = qweight.cuda()
    qzeros = qzeros.cuda()
    scales = scales.cuda()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        dequantize_gemm(qweight, qzeros, scales, 4, 64)
    assert CUDA_KEY not in _AWQ_SHIFTS_CACHE, (
        "a successful capture must not publish a table either: until the graph "
        "is replayed the recorded arithmetic has not run"
    )
    graph.replay()
    torch.cuda.synchronize()
    del graph


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="needs a CUDA/ROCm GPU")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_awq_torch_linear_after_abandoned_capture(cold_shift_cache, dtype):
    module = _awq_module(CUDA_DEVICE, dtype)
    x = torch.randn(4, module.in_features, device=CUDA_DEVICE, dtype=dtype)
    expected = _dense_reference(x, module)

    with torch.no_grad():
        error = _abandon_capture(
            lambda: module._dequantize_weight_for_inference(
                x.reshape(-1, x.shape[-1]), dtype=dtype
            )
        )
        assert "CUDA graph capture" in str(error)
        actual = module(x)

    torch.testing.assert_close(actual.cpu(), expected, atol=5e-3, rtol=5e-3)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="needs a CUDA/ROCm GPU")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_awq_torch_linear_first_dequant_capture_replays(cold_shift_cache, dtype):
    """A real first-call capture records its own table and replays correctly."""

    module = _awq_module(CUDA_DEVICE, dtype)
    x = torch.randn(2, module.in_features, device=CUDA_DEVICE, dtype=dtype)
    expected = _dense_reference(x, module)

    with torch.no_grad():
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = module(x)
        assert CUDA_KEY not in _AWQ_SHIFTS_CACHE
        graph.replay()
        torch.cuda.synchronize()

    torch.testing.assert_close(captured.cpu(), expected, atol=5e-3, rtol=5e-3)
    del graph


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="needs a CUDA/ROCm GPU")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_paro_linear_dense_after_abandoned_capture(cold_shift_cache, monkeypatch, dtype):
    monkeypatch.setattr(
        ParoLinear, "_cuda_awq_kernel_supported", lambda self, compute_dtype: False
    )
    module = _paro_module(CUDA_DEVICE, dtype)
    x = torch.randn(2, module.in_features, device=CUDA_DEVICE, dtype=dtype)

    with torch.no_grad():
        rotated = module._rotate_inputs(x.reshape(-1, x.shape[-1]).contiguous())
        expected = _dense_reference(rotated, module)

        error = _abandon_capture(lambda: module._forward_dense(rotated))
        assert "CUDA graph capture" in str(error)
        actual = module._forward_dense(rotated)

    torch.testing.assert_close(actual.cpu(), expected, atol=2e-2, rtol=2e-2)


def test_cpu_awq_torch_linear_matches_oracle(cold_shift_cache):
    """The CPU kernel that shares ``dequantize_gemm`` keeps decoding correctly."""

    module = _awq_module(torch.device("cpu"), torch.float16)
    x = torch.randn(3, module.in_features, dtype=torch.float16)
    expected = _dense_reference(x, module)

    with torch.no_grad():
        actual = module(x)

    torch.testing.assert_close(actual, expected, atol=5e-3, rtol=5e-3)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="needs a CUDA/ROCm GPU")
def test_eager_shift_table_is_cached_once(cold_shift_cache):
    """No per-call table allocation: eager calls reuse one published tensor."""

    first = _awq_shifts(CUDA_DEVICE, 4)
    assert first.tolist() == EXPECTED_SHIFTS
    assert _awq_shifts(CUDA_DEVICE, 4) is first
    assert _AWQ_SHIFTS_CACHE[CUDA_KEY] is first

    qweight, qzeros, scales = _packed_layer(256, 128, 64)
    qweight = qweight.cuda()
    qzeros = qzeros.cuda()
    scales = scales.cuda()
    dequantize_gemm(qweight, qzeros, scales, 4, 64)
    dequantize_gemm(qweight, qzeros, scales, 4, 64)
    assert _AWQ_SHIFTS_CACHE[CUDA_KEY] is first


def _two_stream_first_call(call):
    """Run ``call`` on a busy stream A, then again on stream B after A publishes."""

    published = threading.Event()
    outputs, errors = {}, {}

    def first():
        torch.cuda.set_device(CUDA_DEVICE)
        stream = torch.cuda.Stream(device=CUDA_DEVICE)
        try:
            with torch.inference_mode(), torch.cuda.stream(stream):
                # Stream A is still busy with earlier work when the process's
                # first AWQ dequant publishes the table.
                torch.cuda._sleep(100_000_000)
                outputs["a"] = call()
        except BaseException as exc:  # noqa: BLE001 - re-raised by the caller
            errors["a"] = exc
        finally:
            published.set()
            stream.synchronize()

    def second():
        torch.cuda.set_device(CUDA_DEVICE)
        if not published.wait(timeout=60):
            errors["b"] = AssertionError("stream A never published the table")
            return
        stream = torch.cuda.Stream(device=CUDA_DEVICE)
        try:
            with torch.inference_mode(), torch.cuda.stream(stream):
                outputs["b"] = call()
        except BaseException as exc:  # noqa: BLE001
            errors["b"] = exc
        finally:
            stream.synchronize()

    threads = [threading.Thread(target=first), threading.Thread(target=second)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)
    assert not any(thread.is_alive() for thread in threads), "two-stream AWQ first call hung"
    assert not errors, errors
    return outputs["a"], outputs["b"]


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="needs a CUDA/ROCm GPU")
def test_two_stream_first_call_uses_a_complete_shift_table(cold_shift_cache):
    """A second stream must not consume the table before its build has run (#5191).

    Stream A is still busy when it makes the process's first four-bit dequant,
    so the table's device arithmetic is only enqueued. Stream B calls right
    after A publishes and used to decode with uninitialized offsets. Repeat
    because the first iteration can reuse memory that still holds a correct
    table; the defect then shows on the following iterations.
    """

    qweight, qzeros, scales = _packed_layer(512, 512, 128)
    qweight = qweight.cuda()
    qzeros = qzeros.cuda()
    scales = scales.cuda()
    serial = dequantize_gemm(qweight, qzeros, scales, 4, 128).cpu()

    for iteration in range(5):
        _AWQ_SHIFTS_CACHE.clear()
        torch.cuda.synchronize()
        first, second = _two_stream_first_call(
            lambda: dequantize_gemm(qweight, qzeros, scales, 4, 128)
        )
        assert torch.equal(first.cpu(), serial), (
            f"iteration {iteration}: the stream that built the table decoded wrong weights"
        )
        assert torch.equal(second.cpu(), serial), (
            f"iteration {iteration}: a second stream decoded with an incomplete table; "
            f"max |diff| = {(second.float().cpu() - serial.float()).abs().max().item()}"
        )
        assert _AWQ_SHIFTS_CACHE[CUDA_KEY].tolist() == EXPECTED_SHIFTS


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="needs a CUDA/ROCm GPU")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_two_stream_first_awq_torch_linear_call(cold_shift_cache, dtype):
    """The consumer-level first call must not hand another stream a partial table."""

    module = _awq_module(CUDA_DEVICE, dtype)
    x = torch.randn(4, module.in_features, device=CUDA_DEVICE, dtype=dtype)
    with torch.no_grad():
        serial = module(x).clone()

    for iteration in range(3):
        _AWQ_SHIFTS_CACHE.clear()
        torch.cuda.synchronize()
        with torch.no_grad():
            first, second = _two_stream_first_call(lambda: module(x))
        assert torch.equal(first, serial), (
            f"iteration {iteration}: the stream that built the table decoded wrong weights"
        )
        assert torch.equal(second, serial), (
            f"iteration {iteration}: a second stream decoded with an incomplete table; "
            f"max |diff| = {(second.float() - serial.float()).abs().max().item()}"
        )
