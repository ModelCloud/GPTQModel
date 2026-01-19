# SPDX-FileCopyrightText: 2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import contextlib
import copy
import math
import statistics
import time
import tracemalloc
import types
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Tuple

import pytest
import torch
import torch.nn as nn
from tabulate import tabulate

from gptqmodel.quantization import gptq as gptq_impl
from gptqmodel.quantization.config import HessianConfig, QuantizeConfig
from gptqmodel.quantization.gptq import GPTQ
from gptqmodel.utils.attn_mask import apply_keep_mask_bt, normalize_seq_mask
from gptqmodel.utils.safe import THREADPOOLCTL


######### test_hessian_accumulation_cpu.py ##########
pytestmark = pytest.mark.skipif(
    (not torch.cuda.is_available()) or torch.cuda.device_count() <= 6,
    reason="CUDA device 6 is required for this benchmark test",
    )


@dataclass
class BenchmarkResult:
    per_batch_seconds: float
    total_seconds: float
    batches_measured: int


def _make_module(hidden_dim: int, device: torch.device) -> nn.Linear:
    module = nn.Linear(hidden_dim, hidden_dim, bias=False, dtype=torch.float16).to(device)
    module.eval()
    return module


def _generate_samples(
        batches: int,
        batch_size: int,
        seq_len: int,
        hidden_dim: int,
        device: torch.device,
) -> List[torch.Tensor]:
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    samples = []
    for _ in range(batches):
        sample = torch.randn(batch_size, seq_len, hidden_dim, device=device, dtype=torch.float16)
        samples.append(sample.contiguous())
    return samples


def _to_pinned_cpu(samples: List[torch.Tensor]) -> List[torch.Tensor]:
    pinned = []
    for tensor in samples:
        host = tensor.to(device="cpu", non_blocking=False).contiguous()
        pinned.append(host.pin_memory())
    return pinned


def _benchmark_add_batch(
        module: nn.Module,
        samples: List[torch.Tensor],
        warmup_batches: int,
        device: torch.device,
) -> BenchmarkResult:
    gptq = GPTQ(module)
    dummy_outputs = torch.empty(0, device=samples[0].device)

    # Warmup to populate internal workspaces and caches
    for idx in range(warmup_batches):
        gptq.add_batch(samples[idx], dummy_outputs, batch_index=idx)

    if samples[0].device.type == "cuda":
        torch.cuda.synchronize(device)

    measured = 0
    start = time.perf_counter()

    for idx in range(warmup_batches, len(samples)):
        gptq.add_batch(samples[idx], dummy_outputs, batch_index=idx)
        measured += 1

    if samples[0].device.type == "cuda":
        torch.cuda.synchronize(device)

    total = time.perf_counter() - start
    per_batch = total / measured if measured else 0.0
    return BenchmarkResult(per_batch_seconds=per_batch, total_seconds=total, batches_measured=measured)


def test_hessian_accumulation_cpu_vs_gpu():
    device = torch.device("cuda", 6)
    torch.cuda.set_device(device)

    configs: List[Tuple[str, int]] = [
        ("llama3", 4096),
        ("qwen3", 3584),
    ]

    total_batches = 6
    warmup_batches = 2
    batch_size = 4
    seq_len = 256

    for name, hidden_dim in configs:
        module_gpu = _make_module(hidden_dim, device=device)
        gpu_samples = _generate_samples(total_batches, batch_size, seq_len, hidden_dim, device)
        cpu_samples = _to_pinned_cpu(gpu_samples)

        gpu_result = _benchmark_add_batch(module_gpu, gpu_samples, warmup_batches, device)

        module_cpu = _make_module(hidden_dim, device=device)
        with THREADPOOLCTL.threadpool_limits(limits=16):
            cpu_result = _benchmark_add_batch(module_cpu, cpu_samples, warmup_batches, device)

        assert gpu_result.batches_measured == cpu_result.batches_measured == total_batches - warmup_batches

        print(
            f"[{name.upper()}] GPU add_batch: {gpu_result.per_batch_seconds*1e3:.3f} ms/batch "
            f"(total {gpu_result.total_seconds:.3f} s) | "
            f"CPU add_batch (threads=16): {cpu_result.per_batch_seconds*1e3:.3f} ms/batch "
            f"(total {cpu_result.total_seconds:.3f} s)"
        )

        assert cpu_result.per_batch_seconds >= gpu_result.per_batch_seconds

######### test_hessian_chunk.py #########

@pytest.fixture(autouse=True)
def reset_workspace_caches():
    gptq_impl._WORKSPACE_CACHE.clear()
    gptq_impl._WORKSPACE_LOCKS.clear()
    gptq_impl._BF16_SUPPORT_CACHE.clear()
    yield
    gptq_impl._WORKSPACE_CACHE.clear()
    gptq_impl._WORKSPACE_LOCKS.clear()
    gptq_impl._BF16_SUPPORT_CACHE.clear()


def _clone_module(module: torch.nn.Module) -> torch.nn.Module:
    replica = type(module)(module.in_features, module.out_features, bias=False)
    replica.load_state_dict(module.state_dict())
    replica.eval()
    return replica


def _instrument_chunks(gptq: GPTQ) -> None:
    original = gptq.borrow_materialized_chunk_fp32

    @contextlib.contextmanager
    def wrapped(self, chunk, rows):
        self._chunk_invocations += 1
        with original(chunk, rows) as materialized:
            yield materialized

    gptq._chunk_invocations = 0
    gptq.borrow_materialized_chunk_fp32 = types.MethodType(wrapped, gptq)


def test_hessian_chunk_consistency_matches_full_precision():
    torch.manual_seed(0)

    base = torch.nn.Linear(32, 16, bias=False).eval()
    module_full = _clone_module(base)
    module_chunked = _clone_module(base)

    qcfg_full = QuantizeConfig(
        hessian=HessianConfig(
            chunk_size=None,
            chunk_bytes=1_000_000_000,
            staging_dtype=torch.float32,
        ),
    )
    qcfg_chunked = QuantizeConfig(
        hessian=HessianConfig(
            chunk_size=16,
            staging_dtype=torch.float32,
        ),
    )

    gptq_full = GPTQ(module_full, qcfg_full)
    gptq_chunked = GPTQ(module_chunked, qcfg_chunked)

    calib = torch.randn(128, 32, dtype=torch.float16)

    _, full_xtx, full_device = gptq_full.process_batch(calib.clone())
    _, chunked_xtx, chunked_device = gptq_chunked.process_batch(calib.clone())

    assert full_device == chunked_device
    assert full_xtx is not None and chunked_xtx is not None
    assert torch.allclose(full_xtx, chunked_xtx, atol=3e-6, rtol=3e-6)


@pytest.mark.cuda
def test_hessian_staging_dtype_accuracy_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for staging dtype accuracy test")

    torch.manual_seed(4)

    device = torch.device("cuda", 0)
    base = torch.nn.Linear(64, 32, bias=False, dtype=torch.float16).to(device).eval()

    calib = torch.randn(256, 64, device=device, dtype=torch.float16)

    qcfg_default = QuantizeConfig(hessian=HessianConfig(staging_dtype=torch.float32))
    gptq_default = GPTQ(_clone_module(base).to(device), qcfg_default)
    _, baseline_xtx, _ = gptq_default.process_batch(calib.clone())

    for staging_dtype in (torch.bfloat16, torch.float16):
        qcfg = QuantizeConfig(hessian=HessianConfig(staging_dtype=staging_dtype))
        gptq = GPTQ(_clone_module(base).to(device), qcfg)
        _, staged_xtx, _ = gptq.process_batch(calib.clone())

        assert baseline_xtx is not None and staged_xtx is not None
        assert torch.allclose(baseline_xtx, staged_xtx, atol=1e-2, rtol=1e-2)


@pytest.mark.cuda
def test_hessian_bf16_vs_fp32_staging_closeness():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for bf16 staging accuracy test")

    torch.manual_seed(5)

    device = torch.device("cuda", 0)
    base = torch.nn.Linear(96, 48, bias=False, dtype=torch.float16).to(device).eval()
    calib = torch.randn(192, 96, device=device, dtype=torch.float16)

    qcfg_default = QuantizeConfig()
    gptq_default = GPTQ(_clone_module(base).to(device), qcfg_default)
    _, default_xtx, _ = gptq_default.process_batch(calib.clone())

    qcfg_fp32 = QuantizeConfig(hessian=HessianConfig(staging_dtype=torch.float32))
    gptq_fp32 = GPTQ(_clone_module(base).to(device), qcfg_fp32)
    _, fp32_xtx, _ = gptq_fp32.process_batch(calib.clone())

    qcfg_bf16 = QuantizeConfig(hessian=HessianConfig(staging_dtype=torch.bfloat16))
    gptq_bf16 = GPTQ(_clone_module(base).to(device), qcfg_bf16)
    _, bf16_xtx, _ = gptq_bf16.process_batch(calib.clone())

    assert default_xtx is not None and fp32_xtx is not None and bf16_xtx is not None
    assert torch.allclose(default_xtx, fp32_xtx, atol=1e-4, rtol=1e-4)
    assert torch.allclose(fp32_xtx, bf16_xtx, atol=1e-2, rtol=1e-2)


def test_hessian_chunk_invocations_and_workspace_shape():
    torch.manual_seed(1)

    base = torch.nn.Linear(64, 32, bias=False).eval()

    large_cfg = QuantizeConfig(hessian=HessianConfig(chunk_size=256))
    large_gptq = GPTQ(_clone_module(base), large_cfg)
    _instrument_chunks(large_gptq)

    small_cfg = QuantizeConfig(hessian=HessianConfig(chunk_size=16))
    small_gptq = GPTQ(_clone_module(base), small_cfg)
    _instrument_chunks(small_gptq)

    calib = torch.randn(120, 64, dtype=torch.float16)

    large_gptq.process_batch(calib.clone())
    assert large_gptq._chunk_invocations == 1

    large_summary = getattr(large_gptq, "_borrow_workspace_last_summary", None)
    assert large_summary is not None
    assert large_summary["requests"] == 1
    assert large_summary["materialized_hits"] == 0
    assert large_summary["materialized_misses"] == 1
    assert large_summary["staging_misses"] == 1
    large_totals = getattr(large_gptq, "_borrow_workspace_totals", {})
    assert large_totals.get("requests") == 1
    assert large_totals.get("materialized_misses") == 1
    large_gptq.log_workspace_stats(context="test_hessian_chunk", reset=True)
    assert getattr(large_gptq, "_borrow_workspace_totals", {}).get("requests") == 0

    small_gptq.process_batch(calib.clone())
    expected_chunks = math.ceil(calib.shape[0] / small_cfg.hessian.chunk_size)
    assert small_gptq._chunk_invocations == expected_chunks

    small_summary = getattr(small_gptq, "_borrow_workspace_last_summary", None)
    assert small_summary is not None
    assert small_summary["requests"] == expected_chunks
    assert small_summary["materialized_hits"] + small_summary["materialized_misses"] == expected_chunks
    assert small_summary["materialized_hits"] >= expected_chunks - 1
    assert pytest.approx(
        small_summary["hit_rate"],
        rel=1e-6,
    ) == small_summary["materialized_hits"] / expected_chunks
    small_totals = getattr(small_gptq, "_borrow_workspace_totals", {})
    assert small_totals.get("requests") == expected_chunks
    assert small_totals.get("materialized_hits") >= expected_chunks - 1
    small_gptq.log_workspace_stats(context="test_hessian_chunk", reset=True)
    assert getattr(small_gptq, "_borrow_workspace_totals", {}).get("requests") == 0

    device = torch.device(base.weight.device)
    cache_key = gptq_impl._workspace_cache_key(device)

    assert cache_key in gptq_impl._WORKSPACE_CACHE
    large_workspace = gptq_impl._WORKSPACE_CACHE[cache_key]
    assert large_workspace.shape[0] >= calib.shape[0]
    assert large_workspace.shape[1] == large_gptq.columns
    assert large_workspace.dtype == torch.float32

    small_workspace = gptq_impl._WORKSPACE_CACHE[cache_key]
    assert small_workspace is large_workspace

    staging_dtype = small_gptq.preferred_staging_dtype(calib.dtype, device)
    if staging_dtype == torch.bfloat16:
        staged_workspace = gptq_impl._WORKSPACE_CACHE[cache_key]
        assert staged_workspace.dtype == torch.bfloat16


def test_hessian_chunk_bytes_budget():
    torch.manual_seed(2)

    base = torch.nn.Linear(48, 24, bias=False).eval()
    module = _clone_module(base)

    bytes_budget = 16 * 48 * 4
    qcfg = QuantizeConfig(hessian=HessianConfig(chunk_size=None, chunk_bytes=bytes_budget))
    gptq = GPTQ(module, qcfg)
    _instrument_chunks(gptq)

    calib = torch.randn(64, 48, dtype=torch.float16)
    gptq.process_batch(calib)

    assert gptq._chunk_invocations == math.ceil(calib.shape[0] / 16)

    device = torch.device(module.weight.device)
    cache_key = gptq_impl._workspace_cache_key(device)
    workspace = gptq_impl._WORKSPACE_CACHE[cache_key]
    assert workspace.shape[0] == 16
    assert workspace.shape[1] == gptq.columns


@pytest.mark.cuda
def test_hessian_workspace_thread_safety_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for workspace stress test")

    device = torch.device("cuda", 0)

    base = torch.nn.Linear(128, 64, bias=False).to(device)
    cfg = QuantizeConfig(
        hessian=HessianConfig(
            chunk_size=128,
            staging_dtype=torch.bfloat16,
        ),
    )

    gptq_workers = [GPTQ(_clone_module(base).to(device), cfg) for _ in range(3)]
    rows = 512
    iters_per_worker = 6

    def worker(task_id: int) -> None:
        gptq = gptq_workers[task_id % len(gptq_workers)]
        torch.cuda.set_device(device.index or 0)
        for i in range(iters_per_worker):
            calib = torch.randn(rows, base.in_features, device=device, dtype=torch.float16)
            batch_size, xtx, canonical_device = gptq.process_batch(calib)
            assert batch_size == rows
            assert xtx is not None
            assert canonical_device == device

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(worker, idx) for idx in range(16)]
        for fut in futures:
            fut.result()

    for gptq in gptq_workers:
        assert getattr(gptq, "_final_hessian_device_hint", None) == device

    cols = base.in_features
    cache_key = gptq_impl._workspace_cache_key(device)
    assert cache_key in gptq_impl._WORKSPACE_CACHE
    cached_workspace = gptq_impl._WORKSPACE_CACHE[cache_key]
    expected_rows = cfg.hessian.chunk_size or rows
    assert cached_workspace.shape[0] >= expected_rows
    assert cached_workspace.shape[1] == cols

    stage_dtype = gptq_workers[0].preferred_staging_dtype(torch.float16, device)
    if stage_dtype == torch.bfloat16:
        assert cached_workspace.dtype == torch.bfloat16
    else:
        assert cached_workspace.dtype == torch.float32


def _benchmark_case(
        base_module: torch.nn.Module,
        cfg_factory: Callable[[], QuantizeConfig],
        calib: torch.Tensor,
        num_iterations: int = 10,
) -> Dict[str, float]:
    device = base_module.weight.device
    use_cuda = device.type == "cuda" and torch.cuda.is_available()

    warmup_module = _clone_module(base_module)
    warmup = GPTQ(warmup_module, cfg_factory())
    warmup.process_batch(calib.clone())

    def measure_once() -> Tuple[float, float]:
        module = _clone_module(base_module)
        gptq = GPTQ(module, cfg_factory())

        if use_cuda:
            torch.cuda.reset_peak_memory_stats(device)
        else:
            tracemalloc.start()

        start = time.perf_counter()
        gptq.process_batch(calib.clone())
        if use_cuda:
            torch.cuda.synchronize(device)
        elapsed = (time.perf_counter() - start) * 1000.0

        if use_cuda:
            peak_mem = torch.cuda.max_memory_allocated(device)
        else:
            _, peak_mem = tracemalloc.get_traced_memory()
            tracemalloc.stop()

        return elapsed, peak_mem / (1024 * 1024)

    timings: List[float] = []
    memories: List[float] = []
    for _ in range(num_iterations):
        elapsed, mem_mb = measure_once()
        timings.append(elapsed)
        memories.append(mem_mb)

    mean_ms = statistics.fmean(timings)
    stdev_ms = statistics.pstdev(timings)
    mean_mem = statistics.fmean(memories)

    config_sample = cfg_factory()

    return {
        "chunk_size": config_sample.hessian.chunk_size,
        "chunk_bytes": config_sample.hessian.chunk_bytes,
        "bf16": config_sample.hessian.staging_dtype == torch.bfloat16,
        "mean_ms": mean_ms,
        "stdev_ms": stdev_ms,
        "mean_mem_mb": mean_mem,
    }


def _print_benchmark_table(rows: Iterable[Dict[str, float]]) -> None:
    table_rows = []
    for row in rows:
        table_rows.append(
            [
                "None" if row["chunk_size"] is None else row["chunk_size"],
                "None" if row["chunk_bytes"] is None else row["chunk_bytes"],
                row["bf16"],
                f"{row['mean_ms']:.3f}",
                f"{row['stdev_ms']:.3f}",
                f"{row['mean_mem_mb']:.3f}",
            ]
        )

    headers = [
        "chunk_size",
        "chunk_bytes",
        "bf16",
        "mean_ms",
        "stdev_ms",
        "mean_mem_mb",
    ]

    print(tabulate(table_rows, headers=headers, tablefmt="github"))


def test_hessian_chunk_benchmark_table():
    torch.manual_seed(3)

    base = torch.nn.Linear(96, 48, bias=False).eval()
    calib = torch.randn(256, 96, dtype=torch.float16)

    configs: List[Callable[[], QuantizeConfig]] = [
        lambda: QuantizeConfig(
            hessian=HessianConfig(
                chunk_size=None,
                chunk_bytes=512 * 1024 * 1024,
                staging_dtype=torch.float32,
            ),
        ),
        lambda: QuantizeConfig(
            hessian=HessianConfig(
                chunk_size=64,
                staging_dtype=torch.float32,
            ),
        ),
        lambda: QuantizeConfig(
            hessian=HessianConfig(
                chunk_size=32,
                staging_dtype=torch.bfloat16,
            ),
        ),
        lambda: QuantizeConfig(
            hessian=HessianConfig(
                chunk_size=None,
                chunk_bytes=64 * 1024 * 1024,
                staging_dtype=torch.bfloat16,
            ),
        ),
    ]

    results = []
    for cfg_factory in configs:
        result = _benchmark_case(base, cfg_factory, calib, num_iterations=10)
        results.append(result)

    _print_benchmark_table(results)

    assert len(results) == len(configs)
    assert all(result["mean_ms"] > 0.0 for result in results)
    assert all(result["mean_mem_mb"] > 0.0 for result in results)

######### test_hessian_inverse.py #########

def _build_gptq(damp_percent: float, damp_auto_increment: float) -> GPTQ:
    module = nn.Linear(2, 2, bias=False)
    qcfg = QuantizeConfig(damp_percent=damp_percent, damp_auto_increment=damp_auto_increment)
    return GPTQ(module, qcfg=qcfg)


def _damped_hessian(base: torch.Tensor, used_damp: float) -> torch.Tensor:
    """Reconstruct the damped matrix the solver actually inverted."""
    damped = base.clone()
    diag_view = damped.diagonal()
    mean = torch.mean(diag_view)
    diag_view.add_(used_damp * mean)
    return damped


def test_hessian_inverse_handles_rank_deficiency():
    gptq = _build_gptq(damp_percent=0.05, damp_auto_increment=0.05)
    device = gptq.module.target_device
    hessian = torch.tensor([[1.0, 1.0], [1.0, 1.0]], dtype=torch.float32, device=device)

    hessian_inv, damp = gptq.hessian_inverse(hessian)

    assert hessian_inv is not None
    assert hessian_inv.shape == hessian.shape
    assert 0 < damp < 1
    assert torch.allclose(hessian_inv, torch.triu(hessian_inv))
    # Accuracy sanity check: recovered triangular factor should match the inverse of the damped matrix.
    reconstructed = hessian_inv.transpose(-1, -2) @ hessian_inv
    expected_inverse = torch.linalg.inv(_damped_hessian(hessian, damp))
    assert torch.allclose(reconstructed, expected_inverse, atol=1e-5, rtol=1e-4)


def test_hessian_inverse_returns_none_for_indefinite_matrix():
    gptq = _build_gptq(damp_percent=0.05, damp_auto_increment=0.25)
    device = gptq.module.target_device
    hessian = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32, device=device)
    original = hessian.clone()

    hessian_inv, damp = gptq.hessian_inverse(hessian)

    assert hessian_inv is None
    assert damp == 1.0
    # The diagonal should reflect the final floor attempt.
    assert torch.allclose(hessian.diagonal(), torch.full((2,), 0.1, device=device))
    # Off-diagonals must remain untouched.
    assert torch.allclose(hessian - torch.diag(hessian.diagonal()), original - torch.diag(original.diagonal()))


def test_hessian_inverse_matches_reference_for_positive_definite_matrix():
    gptq = _build_gptq(damp_percent=0.05, damp_auto_increment=0.05)
    device = gptq.module.target_device
    original = torch.tensor(
        [[4.0, 1.0, 0.5], [1.0, 3.0, 0.2], [0.5, 0.2, 2.5]],
        dtype=torch.float32,
        device=device,
    )
    hessian = original.clone()

    hessian_inv, used_damp = gptq.hessian_inverse(hessian)

    assert hessian_inv is not None
    # Ensure the solver does not mutate a healthy block.
    assert torch.allclose(hessian, original)

    damped = _damped_hessian(hessian, used_damp)
    reconstructed = hessian_inv.transpose(-1, -2) @ hessian_inv
    expected_inverse = torch.linalg.inv(damped)
    assert torch.allclose(reconstructed, expected_inverse, atol=1e-6, rtol=1e-5)


def test_hessian_inverse_applies_diagonal_floor_for_semi_definite_input():
    gptq = _build_gptq(damp_percent=0.05, damp_auto_increment=0.0)
    device = gptq.module.target_device
    hessian = torch.tensor([[0.0, 0.01], [0.01, 0.0]], dtype=torch.float32, device=device)

    hessian_inv, used_damp = gptq.hessian_inverse(hessian)

    assert hessian_inv is not None
    assert used_damp == pytest.approx(gptq.qcfg.damp_percent)
    # Diagonal should be floored to a positive value so later steps see a PD matrix.
    assert torch.all(hessian.diagonal() > 0)
    assert torch.allclose(hessian.diagonal(), torch.full((2,), 0.01, device=device), atol=1e-7, rtol=0.0)

    damped = _damped_hessian(hessian, used_damp)
    # Should be positive definite after flooring, so Cholesky succeeds.
    torch.linalg.cholesky(damped)
    reconstructed = hessian_inv.transpose(-1, -2) @ hessian_inv
    expected_inverse = torch.linalg.inv(damped)
    assert torch.allclose(reconstructed, expected_inverse, atol=1e-5, rtol=1e-4)


def test_hessian_inverse_handles_singleton_flooring():
    gptq = _build_gptq(damp_percent=0.05, damp_auto_increment=0.0)
    device = gptq.module.target_device
    hessian = torch.tensor([[0.0]], dtype=torch.float32, device=device)

    hessian_inv, used_damp = gptq.hessian_inverse(hessian)

    assert hessian_inv is not None
    assert hessian_inv.shape == hessian.shape
    assert torch.allclose(hessian.diagonal(), torch.tensor([1e-6], dtype=torch.float32, device=device))

    damped = _damped_hessian(hessian, used_damp)
    reconstructed = hessian_inv.transpose(-1, -2) @ hessian_inv
    expected_inverse = torch.linalg.inv(damped)
    assert torch.allclose(reconstructed, expected_inverse, atol=1e-6, rtol=1e-4)

######### test_hessian_merge.py #########

@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="requires at least two CUDA devices",
    )
@torch.no_grad()
def test_hessian_merge_multi_gpu_matches_serial():
    torch.manual_seed(0)

    in_features = 16
    out_features = 8
    batch_count = 64
    device_count = 2
    per_device = batch_count // device_count
    devices = [torch.device(f"cuda:{idx}") for idx in range(device_count)]

    base = torch.nn.Linear(in_features, out_features, bias=False).eval()
    cfg_serial = QuantizeConfig()
    cfg_multi = copy.deepcopy(cfg_serial)

    serial_module = copy.deepcopy(base).to(devices[0])
    multi_module = copy.deepcopy(base).to(devices[0])

    gptq_serial = GPTQ(serial_module, cfg_serial)
    gptq_multi = GPTQ(multi_module, cfg_multi)

    samples = [torch.randn(1, 1, in_features) for _ in range(batch_count)]

    for idx, sample in enumerate(samples):
        sample_gpu = sample.to(devices[0])
        gptq_serial.add_batch(sample_gpu, torch.empty(0, device=devices[0]), batch_index=idx)
        del sample_gpu
    torch.cuda.synchronize(device=devices[0])

    gptq_serial.finalize_hessian()
    serial_hessian = gptq_serial.H.detach().cpu()
    assert gptq_serial.nsamples == batch_count

    for device_idx, device in enumerate(devices):
        start = device_idx * per_device
        end = start + per_device
        for idx in range(start, end):
            sample_gpu = samples[idx].to(device)
            gptq_multi.add_batch(sample_gpu, torch.empty(0, device=device), batch_index=idx)
            del sample_gpu
        torch.cuda.synchronize(device=device)

    partials_snapshot = {
        dev: tensor.clone()
        for dev, tensor in gptq_multi._device_hessian_partials.items()
    }
    sample_counts_snapshot = dict(gptq_multi._device_sample_counts)

    gptq_multi.finalize_hessian()
    merged_hessian = gptq_multi.H.detach().cpu()
    assert gptq_multi.nsamples == batch_count

    max_abs_diff = (merged_hessian - serial_hessian).abs().max().item()
    print(
        "[hessian-no-mask] "
        f"serial_nsamples={gptq_serial.nsamples} "
        f"multi_nsamples={gptq_multi.nsamples} "
        f"max_abs_diff={max_abs_diff:.6e}"
    )

    total_samples = sum(sample_counts_snapshot.values())
    assert total_samples == batch_count

    manual_device = gptq_multi.H.device
    manual_accum = torch.zeros(
        (gptq_multi.columns, gptq_multi.columns),
        dtype=torch.float64,
        device=manual_device,
    )
    for dev, tensor in partials_snapshot.items():
        manual_accum.add_(tensor.to(device=manual_device, dtype=torch.float64))
    manual_accum.mul_(2.0 / float(total_samples))
    manual_result = manual_accum.to(dtype=torch.float32).cpu()

    # The materialized Hessian should match the explicit fp64 reduction exactly.
    assert torch.equal(merged_hessian, manual_result)

    # And the merged Hessian should agree with the serial reference to float32 resolution.
    torch.testing.assert_close(merged_hessian, serial_hessian, atol=5e-7, rtol=5e-7)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 8,
    reason="requires CUDA devices >= 8 to exercise GPUs 6 and 7",
    )
@torch.no_grad()
def test_hessian_merge_multi_gpu_with_attention_mask():
    torch.manual_seed(123)

    in_features = 32
    out_features = 16
    batch_size = 3
    seq_len = 21
    batch_count = 10

    device_serial = torch.device("cuda:6")
    devices = [torch.device("cuda:6"), torch.device("cuda:7")]

    base = torch.nn.Linear(in_features, out_features, bias=False).eval()
    cfg_serial = QuantizeConfig(mock_quantization=True, desc_act=False)
    cfg_multi = copy.deepcopy(cfg_serial)

    serial_module = copy.deepcopy(base).to(device_serial)
    multi_module = copy.deepcopy(base).to(device_serial)

    gptq_serial = GPTQ(serial_module, cfg_serial)
    gptq_multi = GPTQ(multi_module, cfg_multi)

    samples = []
    for _ in range(batch_count):
        hidden = torch.randn(batch_size, seq_len, in_features, dtype=torch.float32)
        mask = torch.ones(batch_size, seq_len, dtype=torch.int32)
        for row in range(batch_size):
            # ensure at least one valid token per row, trim a random tail portion
            cutoff = torch.randint(1, seq_len + 1, ()).item()
            if cutoff < seq_len:
                mask[row, cutoff:] = 0
        samples.append((hidden, mask))

    total_kept_tokens = 0
    for idx, (hidden, mask) in enumerate(samples):
        hidden_gpu = hidden.to(device_serial)
        mask_gpu = mask.to(device_serial)
        keep = normalize_seq_mask(mask_gpu)
        trimmed = apply_keep_mask_bt(hidden_gpu, keep)
        total_kept_tokens += trimmed.shape[0]
        gptq_serial.add_batch(trimmed, torch.empty(0, device=device_serial), batch_index=idx)
    torch.cuda.synchronize(device=device_serial)

    gptq_serial.finalize_hessian()
    serial_hessian = gptq_serial.H.detach().cpu()
    assert gptq_serial.nsamples == total_kept_tokens

    per_device = batch_count // len(devices)
    remainder = batch_count % len(devices)
    start = 0

    device_token_counts = {}
    for device_idx, device in enumerate(devices):
        extra = 1 if device_idx < remainder else 0
        end = start + per_device + extra
        for idx in range(start, end):
            hidden, mask = samples[idx]
            hidden_gpu = hidden.to(device)
            mask_gpu = mask.to(device)
            keep = normalize_seq_mask(mask_gpu)
            trimmed = apply_keep_mask_bt(hidden_gpu, keep)
            device_token_counts[device] = device_token_counts.get(device, 0) + trimmed.shape[0]
            gptq_multi.add_batch(trimmed, torch.empty(0, device=device), batch_index=idx)
        torch.cuda.synchronize(device=device)
        start = end

    assert sum(device_token_counts.values()) == total_kept_tokens

    partial_counts_snapshot = dict(gptq_multi._device_sample_counts)
    assert partial_counts_snapshot == device_token_counts

    gptq_multi.finalize_hessian()
    merged_hessian = gptq_multi.H.detach().cpu()

    max_abs_diff = (merged_hessian - serial_hessian).abs().max().item()
    print(
        "[hessian-mask] "
        f"serial_tokens={total_kept_tokens} "
        f"multi_tokens={gptq_multi.nsamples} "
        f"per_device={{{', '.join(f'{str(dev)}:{count}' for dev, count in device_token_counts.items())}}} "
        f"max_abs_diff={max_abs_diff:.6e}"
    )

    assert gptq_multi.nsamples == total_kept_tokens
    torch.testing.assert_close(merged_hessian, serial_hessian, atol=5e-7, rtol=5e-7)

