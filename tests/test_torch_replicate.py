# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import copy
from typing import Callable

import pytest
import torch
from tabulate import tabulate
from torch.nn.parallel import replicate as torch_replicate

from gptqmodel.utils.torch import timed_gc_collect


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
