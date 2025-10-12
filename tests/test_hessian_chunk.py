# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import math
import statistics
import time
import tracemalloc
import types
from typing import Callable, Dict, Iterable, List, Tuple

import pytest
import torch
from tabulate import tabulate

from gptqmodel.quantization import gptq as gptq_impl
from gptqmodel.quantization.config import QuantizeConfig
from gptqmodel.quantization.gptq import GPTQ


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
    original = gptq._borrow_materialized_chunk_fp32

    def wrapped(self, chunk, rows):
        self._chunk_invocations += 1
        return original(chunk, rows)

    gptq._chunk_invocations = 0
    gptq._borrow_materialized_chunk_fp32 = types.MethodType(wrapped, gptq)


def test_hessian_chunk_consistency_matches_full_precision():
    torch.manual_seed(0)

    base = torch.nn.Linear(32, 16, bias=False).eval()
    module_full = _clone_module(base)
    module_chunked = _clone_module(base)

    qcfg_full = QuantizeConfig(
        hessian_chunk_size=None,
        hessian_chunk_bytes=1_000_000_000,
        hessian_use_bfloat16_staging=False,
    )
    qcfg_chunked = QuantizeConfig(
        hessian_chunk_size=16,
        hessian_use_bfloat16_staging=False,
    )

    gptq_full = GPTQ(module_full, qcfg_full)
    gptq_chunked = GPTQ(module_chunked, qcfg_chunked)

    calib = torch.randn(128, 32, dtype=torch.float16)

    gptq_full.process_batch(calib.clone())
    gptq_chunked.process_batch(calib.clone())

    assert torch.allclose(gptq_full.H, gptq_chunked.H, atol=1e-5, rtol=1e-5)


def test_hessian_chunk_invocations_and_workspace_shape():
    torch.manual_seed(1)

    base = torch.nn.Linear(64, 32, bias=False).eval()

    large_cfg = QuantizeConfig(hessian_chunk_size=256)
    large_gptq = GPTQ(_clone_module(base), large_cfg)
    _instrument_chunks(large_gptq)

    small_cfg = QuantizeConfig(hessian_chunk_size=16)
    small_gptq = GPTQ(_clone_module(base), small_cfg)
    _instrument_chunks(small_gptq)

    calib = torch.randn(120, 64, dtype=torch.float16)

    large_gptq.process_batch(calib.clone())
    assert large_gptq._chunk_invocations == 1

    small_gptq.process_batch(calib.clone())
    expected_chunks = math.ceil(calib.shape[0] / small_cfg.hessian_chunk_size)
    assert small_gptq._chunk_invocations == expected_chunks

    device = torch.device(base.weight.device)
    cols = base.in_features
    fp32_key = gptq_impl._workspace_cache_key(device, torch.float32, cols)

    assert fp32_key in gptq_impl._WORKSPACE_CACHE
    large_workspace = gptq_impl._WORKSPACE_CACHE[fp32_key]
    assert large_workspace.shape[0] >= calib.shape[0]
    assert large_workspace.shape[1] == large_gptq.columns

    small_workspace = gptq_impl._WORKSPACE_CACHE[fp32_key]
    assert small_workspace is large_workspace

    staging_dtype = small_gptq._preferred_staging_dtype(calib.dtype, device)
    if staging_dtype == torch.bfloat16:
        staging_key = gptq_impl._workspace_cache_key(device, staging_dtype, cols)
        assert staging_key in gptq_impl._WORKSPACE_CACHE


def test_hessian_chunk_bytes_budget():
    torch.manual_seed(2)

    base = torch.nn.Linear(48, 24, bias=False).eval()
    module = _clone_module(base)

    bytes_budget = 16 * 48 * 4
    qcfg = QuantizeConfig(hessian_chunk_size=None, hessian_chunk_bytes=bytes_budget)
    gptq = GPTQ(module, qcfg)
    _instrument_chunks(gptq)

    calib = torch.randn(64, 48, dtype=torch.float16)
    gptq.process_batch(calib)

    assert gptq._chunk_invocations == math.ceil(calib.shape[0] / 16)

    device = torch.device(module.weight.device)
    cols = base.in_features
    fp32_key = gptq_impl._workspace_cache_key(device, torch.float32, cols)
    workspace = gptq_impl._WORKSPACE_CACHE[fp32_key]
    assert workspace.shape[0] == 16
    assert workspace.shape[1] == gptq.columns


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
        "chunk_size": config_sample.hessian_chunk_size,
        "chunk_bytes": config_sample.hessian_chunk_bytes,
        "bf16": config_sample.hessian_use_bfloat16_staging,
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
            hessian_chunk_size=None,
            hessian_chunk_bytes=512 * 1024 * 1024,
            hessian_use_bfloat16_staging=False,
        ),
        lambda: QuantizeConfig(
            hessian_chunk_size=64,
            hessian_use_bfloat16_staging=False,
        ),
        lambda: QuantizeConfig(
            hessian_chunk_size=32,
            hessian_use_bfloat16_staging=True,
        ),
        lambda: QuantizeConfig(
            hessian_chunk_size=None,
            hessian_chunk_bytes=64 * 1024 * 1024,
            hessian_use_bfloat16_staging=True,
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
