from __future__ import annotations

import copy
import math
import time
from typing import Callable

import pytest
import torch
from tabulate import tabulate
from torch.nn.parallel import replicate as torch_replicate

from gptqmodel.quantization.config import QuantizeConfig
from gptqmodel.quantization.gptq import GPTQ
from gptqmodel.utils.torch import timed_gc_collect


######### test_gptq_queue.py ##########
@torch.no_grad()
def test_out_of_order_batches_finalize_matches_reference():
    torch.manual_seed(0)

    module = torch.nn.Linear(4, 4)
    reference_module = copy.deepcopy(module)

    cfg = QuantizeConfig()
    gptq = GPTQ(module, cfg)
    reference = GPTQ(reference_module, copy.deepcopy(cfg))

    x0 = torch.randn(1, 1, 4)
    x1 = torch.randn(1, 1, 4)

    y0 = module(x0)
    y1 = module(x1)

    # Add batches out of order to ensure accumulation is order agnostic.
    gptq.add_batch(x1, y1, batch_index=1)
    gptq.add_batch(x0, y0, batch_index=0)

    gptq.finalize_hessian()

    reference.add_batch(x0, y0, batch_index=0)
    reference.add_batch(x1, y1, batch_index=1)
    reference.finalize_hessian()

    assert gptq.H is not None
    torch.testing.assert_close(gptq.H, reference.H)
    assert gptq.nsamples == reference.nsamples
    assert not gptq._device_hessian_partials


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_finalize_hessian_preserves_device(monkeypatch):
    module = torch.nn.Linear(4, 4).cuda()
    cfg = QuantizeConfig()
    gptq = GPTQ(module, cfg)

    module_device = module.weight.device

    def fake_process_batch(self, inp):
        xtx = torch.eye(self.columns, dtype=torch.float32, device=module_device)
        return 1, xtx.clone(), module_device

    monkeypatch.setattr(GPTQ, "process_batch", fake_process_batch, raising=False)

    inp = torch.zeros(1, device=module_device)

    gptq.add_batch(inp, inp, batch_index=1)
    gptq.add_batch(inp, inp, batch_index=0)

    # No Hessian materialized until finalize is invoked.
    assert gptq.H is None
    assert module_device in gptq._device_hessian_partials

    gptq.finalize_hessian()

    assert gptq.H is not None
    assert gptq.H.device == module_device
    assert not gptq._device_hessian_partials

    torch.cuda.synchronize()


########## test_torch_replicate.py ##########

TIMED_TRIALS = 5
WARMUP_TRIALS = 1


def _build_template_module() -> torch.nn.Module:
    torch.manual_seed(0)
    return torch.nn.Sequential(
        torch.nn.Linear(4096, 4096, bias=False),
        torch.nn.GELU(),
        torch.nn.Linear(4096, 4096, bias=False),
    )


def _replicate_strategy(module: torch.nn.Module, devices: list[torch.device]) -> list[torch.nn.Module]:
    return torch_replicate(module, devices)


def _deepcopy_strategy(module: torch.nn.Module, devices: list[torch.device]) -> list[torch.nn.Module]:
    clones: list[torch.nn.Module] = []
    for dev in devices:
        replica = copy.deepcopy(module)
        clones.append(replica.to(dev))
    return clones


def _benchmark(
        strategy: Callable[[torch.nn.Module, list[torch.device]], list[torch.nn.Module]],
        devices: list[torch.device],
        template: torch.nn.Module,
        *,
        trials: int = TIMED_TRIALS,
        warmup: int = WARMUP_TRIALS,
) -> tuple[list[float], list[int]]:
    times: list[float] = []
    mems: list[int] = []

    def _run(record: bool) -> None:
        module = copy.deepcopy(template).to(devices[0]).eval()
        torch.cuda.synchronize()

        baselines = {}
        for dev in devices:
            baselines[dev] = torch.cuda.memory_allocated(dev)
            torch.cuda.reset_peak_memory_stats(dev)

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        clones = strategy(module, devices)
        end_event.record()
        torch.cuda.synchronize()

        if record:
            duration = start_event.elapsed_time(end_event) / 1000.0
            extra_mem = 0
            for dev in devices:
                peak = torch.cuda.max_memory_allocated(dev)
                extra_mem += max(0, peak - baselines[dev])

            times.append(duration)
            mems.append(extra_mem)

        del clones
        del module
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    for _ in range(warmup):
        _run(record=False)
    for _ in range(trials):
        _run(record=True)

    return times, mems


def _summarise_metrics(times: list[float], mems: list[int]):
    avg_time = sum(times) / len(times)
    avg_mem = sum(mems) / len(mems)
    return {
        "time_avg": avg_time,
        "time_min": min(times),
        "time_max": max(times),
        "mem_avg": avg_mem,
        "mem_min": min(mems),
        "mem_max": max(mems),
    }


def _random_linear(in_features: int = 8, out_features: int = 4) -> torch.nn.Linear:
    torch.manual_seed(0)
    layer = torch.nn.Linear(in_features, out_features, bias=True)
    layer.eval()
    return layer


def _assert_replicas_match(
        replicas: list[torch.nn.Module],
        devices: list[torch.device],
        reference_output: torch.Tensor,
        input_tensor: torch.Tensor,
) -> None:
    for replica, device in zip(replicas, devices):
        assert replica is not None
        replica.eval()
        for param in replica.parameters():
            assert param.device == device
        result = replica(input_tensor.to(device)).to("cpu")
        torch.testing.assert_close(result, reference_output, atol=1e-6, rtol=1e-5)


@pytest.mark.cuda
@pytest.mark.inference
@pytest.mark.xfail(reason="torch.nn.parallel.replicate requires GPU resident tensors", strict=True)
def test_replicate_from_cpu_to_multiple_gpu():
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("Requires at least two CUDA devices")

    devices = [torch.device(f"cuda:{idx}") for idx in range(2)]
    module = _random_linear()
    torch.randn(2, module.in_features)

    torch_replicate(module, devices)


@pytest.mark.cuda
@pytest.mark.inference
def test_replicate_from_gpu_to_multiple_gpu():
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("Requires at least two CUDA devices")

    devices = [torch.device(f"cuda:{idx}") for idx in range(2)]
    module = _random_linear().to(devices[0])
    input_tensor = torch.randn(2, module.in_features, device=devices[0])
    reference = module(input_tensor).to("cpu")

    replicas = torch_replicate(module, devices)
    _assert_replicas_match(replicas, devices, reference, input_tensor.to("cpu"))

    del replicas
    timed_gc_collect()
    torch.cuda.empty_cache()


@pytest.mark.cuda
def test_torch_replicate_benchmark():
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("torch.nn.parallel.replicate comparison requires at least two CUDA devices")

    devices = [torch.device(f"cuda:{idx}") for idx in range(2)]
    template = _build_template_module()

    replicate_times, replicate_mems = _benchmark(_replicate_strategy, devices, template)
    deepcopy_times, deepcopy_mems = _benchmark(_deepcopy_strategy, devices, template)

    replicate_summary = _summarise_metrics(replicate_times, replicate_mems)
    deepcopy_summary = _summarise_metrics(deepcopy_times, deepcopy_mems)

    table = [
        [
            "replicate",
            replicate_summary["time_avg"],
            replicate_summary["time_min"],
            replicate_summary["time_max"],
            replicate_summary["mem_avg"] / (1024**2),
            replicate_summary["mem_min"] / (1024**2),
            replicate_summary["mem_max"] / (1024**2),
            ],
        [
            "deepcopy",
            deepcopy_summary["time_avg"],
            deepcopy_summary["time_min"],
            deepcopy_summary["time_max"],
            deepcopy_summary["mem_avg"] / (1024**2),
            deepcopy_summary["mem_min"] / (1024**2),
            deepcopy_summary["mem_max"] / (1024**2),
            ],
    ]

    headers = [
        "strategy",
        "time_avg_s",
        "time_min_s",
        "time_max_s",
        "mem_avg_MB",
        "mem_min_MB",
        "mem_max_MB",
    ]

    print(tabulate(table, headers=headers, floatfmt=".4f"))

    assert replicate_summary["time_avg"] <= deepcopy_summary["time_avg"], (
        "replicate slower than deepcopy: "
        f"replicate={replicate_summary['time_avg']:.4f}s, deepcopy={deepcopy_summary['time_avg']:.4f}s"
    )
    assert replicate_summary["mem_avg"] <= deepcopy_summary["mem_avg"], (
        "replicate used more memory: "
        f"replicate={replicate_summary['mem_avg'] / (1024**2):.1f}MB, "
        f"deepcopy={deepcopy_summary['mem_avg'] / (1024**2):.1f}MB"
    )

########### test_cuda_stream.py ###########

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for this test"
)

def _mb(nbytes): return nbytes / (1024**2)

def _banner(msg):
    print("\n" + "=" * 80)
    print(msg)
    print("=" * 80)

def test_three_d2h_transfers_concurrency_vs_serial():
    dev = torch.device("cuda", 0)
    props = torch.cuda.get_device_properties(dev)
    _banner(
        f"GPU: {props.name} | asyncEngineCount={getattr(props, 'asyncEngineCount', 'n/a')} | "
        f"PCIe/Link: unknown (PyTorch doesn't expose)\n"
        "Expectation: multiple D2H on a single GPU serialize onto one D2H engine."
    )

    torch.cuda.set_device(dev)

    # Use a size large enough to dominate overhead but not stress CI.
    # ~256 MiB each => 3 * 256 MiB = 768 MiB total device RAM + pinned host buffers.
    elements = (256 * 1024 * 1024) // 2  # fp16 => 2 bytes/elt
    dtype = torch.float16

    # Device tensors
    d0 = torch.empty(elements, dtype=dtype, device=dev)
    d1 = torch.empty_like(d0)
    d2 = torch.empty_like(d0)

    # Pinned host buffers (required for async copies)
    h0 = torch.empty_like(d0, device="cpu", pin_memory=True)
    h1 = torch.empty_like(d1, device="cpu", pin_memory=True)
    h2 = torch.empty_like(d2, device="cpu", pin_memory=True)

    # Warmup: one D2H copy to touch paths
    h0.copy_(d0, non_blocking=True)
    torch.cuda.synchronize()

    # --- Serialized on a single stream ---
    s_serial = torch.cuda.Stream()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.cuda.stream(s_serial):
        h0.copy_(d0, non_blocking=True)
        h1.copy_(d1, non_blocking=True)
        h2.copy_(d2, non_blocking=True)
    torch.cuda.synchronize()
    serial_time = time.perf_counter() - t0

    # --- Launched concurrently on three streams ---
    s0, s1, s2 = torch.cuda.Stream(), torch.cuda.Stream(), torch.cuda.Stream()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    with torch.cuda.stream(s0):
        h0.copy_(d0, non_blocking=True)
    with torch.cuda.stream(s1):
        h1.copy_(d1, non_blocking=True)
    with torch.cuda.stream(s2):
        h2.copy_(d2, non_blocking=True)
    torch.cuda.synchronize()
    concurrent_time = time.perf_counter() - t1

    total_mb = 3 * _mb(d0.numel() * d0.element_size())
    print(f"\nTransferred total ~{total_mb:.1f} MiB (3 x ~{total_mb/3:.1f} MiB) D2H")
    print(f"[SERIAL]     {serial_time:.4f} s  | ~{total_mb/serial_time:.1f} MiB/s effective")
    print(f"[CONCURRENT] {concurrent_time:.4f} s  | ~{total_mb/concurrent_time:.1f} MiB/s effective")

    # We expect little to no speedup when "concurrent" (same-direction copies share the D2H engine).
    # Allow some tolerance either way depending on driver/runtime details.
    assert concurrent_time >= 0.8 * serial_time, (
        "Unexpected large speedup from concurrent D2H; "
        "this would contradict single-engine D2H behavior."
    )
    assert concurrent_time <= 1.3 * serial_time, (
        "Concurrent D2H took much longer than serialized; "
        "this suggests overheads far above expectation."
    )

def test_h2d_d2h_bidirectional_overlap_possible():
    """Optional: demonstrate one H2D can overlap one D2H if GPU has ≥2 copy engines."""
    dev = torch.device("cuda", 0)
    props = torch.cuda.get_device_properties(dev)
    if getattr(props, "asyncEngineCount", 0) < 2:
        pytest.skip("GPU reports <2 copy engines; bidirectional overlap unlikely.")

    torch.cuda.set_device(dev)

    elements = (128 * 1024 * 1024) // 1  # 128 MiB in bytes (uint8)
    dtype = torch.uint8

    # Host buffers (pinned) and device tensors
    h_src = torch.empty(elements, dtype=dtype, device="cpu", pin_memory=True)
    h_dst = torch.empty(elements, dtype=dtype, device="cpu", pin_memory=True)
    d_buf = torch.empty(elements, dtype=dtype, device=dev)

    # Warmup
    d_buf.copy_(h_src, non_blocking=True)
    h_dst.copy_(d_buf, non_blocking=True)
    torch.cuda.synchronize()

    # Baseline: serialize H2D then D2H on one stream
    s = torch.cuda.Stream()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.cuda.stream(s):
        d_buf.copy_(h_src, non_blocking=True)  # H2D
        h_dst.copy_(d_buf, non_blocking=True)  # D2H
    torch.cuda.synchronize()
    serial = time.perf_counter() - t0

    # Overlap: H2D on one stream, D2H on another (should overlap on separate engines)
    sh2d, sd2h = torch.cuda.Stream(), torch.cuda.Stream()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    with torch.cuda.stream(sh2d):
        d_buf.copy_(h_src, non_blocking=True)  # H2D
    with torch.cuda.stream(sd2h):
        h_dst.copy_(d_buf, non_blocking=True)  # D2H
    torch.cuda.synchronize()
    overlapped = time.perf_counter() - t1

    print(f"\n[H2D->D2H] SERIAL   {serial:.4f} s")
    print(f"[H2D||D2H] OVERLAP  {overlapped:.4f} s  (expect <= ~serial)")

    # Expect some overlap benefit (not necessarily 2x).
    assert overlapped <= 0.9 * serial or math.isclose(overlapped, serial, rel_tol=0.05)
