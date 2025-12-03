import os


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:256,garbage_collection_threshold:0.7" #"expandable_segments:True"

import time

import pytest
import torch
from parameterized import parameterized
from pytest import MonkeyPatch
from torch import nn

from gptqmodel.looper.awq_processor import AWQProcessor
from gptqmodel.quantization.config import QuantizeConfig, METHOD, FORMAT


QWEN3_HIDDEN_SIZE = 3584


def _compute_legacy_w_mean(layers, group_size):
    weights = [layer.weight.detach().to(torch.float32).cpu() for layer in layers]
    weight = torch.cat(weights, dim=0)
    org_shape = weight.shape
    weight = weight.view(-1, group_size)
    w_scale = weight.abs() / (weight.abs().amax(dim=1, keepdim=True) + 1e-6)
    w_scale = w_scale.view(org_shape)
    return w_scale.mean(0)

def _compute_fast_w_mean(layers, group_size):
    first_weight = layers[0].weight
    num_channels = first_weight.shape[1]
    device = first_weight.device
    dtype = first_weight.dtype
    w_sum = torch.zeros(num_channels, dtype=torch.float32, device=device)
    row_count = 0

    for layer in layers:
        weight = layer.weight
        org_shape = weight.shape
        weight_abs = weight.abs()
        weight_group = weight_abs.view(-1, group_size)
        group_scale = weight_group.amax(dim=1, keepdim=True) + 1e-6
        normalized = (weight_group / group_scale).view(org_shape)
        w_sum += normalized.sum(dim=0, dtype=torch.float32)
        row_count += org_shape[0]

    if row_count == 0:
        return torch.zeros(num_channels, dtype=dtype, device=device)
    return (w_sum / row_count).to(dtype)


def _compute_fast_w_mean_multi(layer_groups, group_size):
    total_sum = None
    total_rows = 0
    for layers in layer_groups:
        first_weight = layers[0].weight
        device = first_weight.device
        num_channels = first_weight.shape[1]
        w_sum = torch.zeros(num_channels, dtype=torch.float32, device=device)
        rows = 0
        for layer in layers:
            weight = layer.weight
            org_shape = weight.shape
            weight_abs = weight.abs()
            weight_group = weight_abs.view(-1, group_size)
            group_scale = weight_group.amax(dim=1, keepdim=True) + 1e-6
            normalized = (weight_group / group_scale).view(org_shape)
            w_sum += normalized.sum(dim=0, dtype=torch.float32)
            rows += org_shape[0]
        if total_sum is None:
            total_sum = w_sum.cpu()
        else:
            total_sum += w_sum.cpu()
        total_rows += rows
    return (total_sum / total_rows)


class _DummyQwen3SelfAttention(nn.Module):
    def __init__(self, hidden_size: int, device: str, dtype: torch.dtype) -> None:
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False, device=device, dtype=dtype)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False, device=device, dtype=dtype)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False, device=device, dtype=dtype)


class _TestAWQProcessor(AWQProcessor):
    def __init__(self, qcfg: QuantizeConfig):
        super().__init__(
            tokenizer=None,
            qcfg=qcfg,
            calibration=None,
            prepare_dataset_func=None,
            calibration_concat_size=None,
            calibration_sort=None,
            batch_size=1,
            gptq_model=None,
            model=None,
            require_fwd=True,
            calculate_w_wq_diff=False,
            calibration_concat_separator=None,
        )

    def _module_forward(self, x: torch.Tensor, module: torch.nn.Module, module_kwargs):
        return module(x)


@parameterized.expand([
    ("cpu_gs32", "cpu", 32),
    ("cpu_gs64", "cpu", 64),
    ("cpu_gs128", "cpu", 128),
    ("cuda0_gs32", "cuda:0", 32),
    ("cuda0_gs64", "cuda:0", 64),
    ("cuda0_gs128", "cuda:0", 128),
    ("cuda0_cuda1_gs128", ("cuda:0", "cuda:1"), 128),
])
def test_awq_weight_mean_matches_legacy_impl(param_name, device, group_size):
    if isinstance(device, (list, tuple)):
        devices = list(device)
        for dev in devices:
            if not torch.cuda.is_available() or torch.device(dev).index >= torch.cuda.device_count():
                pytest.skip(f"{dev} is not available")
    elif isinstance(device, str) and device.startswith("cuda"):
        if not torch.cuda.is_available():
            pytest.skip("CUDA is not available for this test run.")

    torch.manual_seed(0)
    if isinstance(device, (list, tuple)):
        dtype = torch.float16
        layer_groups = []
        for dev in device:
            layer_groups.append([
                nn.Linear(QWEN3_HIDDEN_SIZE, QWEN3_HIDDEN_SIZE, bias=False, device=dev, dtype=dtype)
                for _ in range(3)
            ])

        baseline_layers = [layer for group in layer_groups for layer in group]
        baseline = _compute_legacy_w_mean(baseline_layers, group_size)
        fast = _compute_fast_w_mean_multi(layer_groups, group_size)
        fast = fast.to(baseline.dtype)

        # Accuracy table
        abs_diff = (fast - baseline).abs()
        with torch.no_grad():
            safe_baseline = torch.where(baseline == 0, torch.ones_like(baseline), baseline)
            rel_diff = abs_diff / safe_baseline.abs()
        max_abs_diff = abs_diff.max().item()
        max_rel_diff = rel_diff.max().item()

        header = f"{'Metric':<20}{'Measured':<20}{'Tolerance':<20}"
        separator = "-" * len(header)
        print(f"AWQ weight mean comparison (fast vs baseline) [{param_name}]")
        print(separator)
        print(header)
        print(separator)
        atol = 5e-4
        rtol = 1e-3
        print(f"{'max_abs_diff':<20}{max_abs_diff:<20.6e}{atol:<20.6e}")
        print(f"{'max_rel_diff':<20}{max_rel_diff:<20.6e}{rtol:<20.6e}")
        print(separator)
        assert torch.allclose(fast, baseline, rtol=rtol, atol=atol)

        # Timing comparison
        def _time_it(fn, runs=5, warmup=2):
            for _ in range(warmup):
                fn()
            torch.cuda.synchronize(torch.device(device[0]).index)
            start = time.perf_counter()
            for _ in range(runs):
                fn()
            torch.cuda.synchronize(torch.device(device[0]).index)
            return (time.perf_counter() - start) / runs

        def fast_fn():
            _ = _compute_fast_w_mean_multi(layer_groups, group_size)

        def legacy_fn():
            _ = _compute_legacy_w_mean(baseline_layers, group_size)

        fast_time = _time_it(fast_fn)
        legacy_time = _time_it(legacy_fn)

        GREEN = "\033[32m"
        RED = "\033[31m"
        YELLOW = "\033[33m"
        RESET = "\033[0m"

        delta_ms = (fast_time - legacy_time) * 1e3
        rel = (fast_time / legacy_time) if legacy_time > 0 else float("inf")
        if rel <= 1.0:
            color = GREEN
            verdict = "faster"
        elif rel <= 1.05:
            color = YELLOW
            verdict = "≈ parity"
        else:
            color = RED
            verdict = "slower"

        print(f"AWQ weight mean timing [{param_name}]")
        print("+----------------+------------+--------------+-------------+---------------+")
        print("| Metric         | Fast (ms)  | Legacy (ms)  | Delta (ms)  | Relative      |")
        print("+----------------+------------+--------------+-------------+---------------+")
        print(
            f"| runtime        | {fast_time*1e3:10.3f} | {legacy_time*1e3:12.3f} | "
            f"{delta_ms:11.3f} | {color}{rel:>11.3%} {verdict:<7}{RESET}|"
        )
        print("+----------------+------------+--------------+-------------+---------------+")

        assert fast_time <= legacy_time * 1.05, (
            f"Streaming mean slower than legacy for {param_name}: "
            f"{fast_time*1e3:.3f} ms vs {legacy_time*1e3:.3f} ms"
        )
        return

    device_str = device
    dtype = torch.float16 if device_str.startswith("cuda") else torch.float32

    attn = _DummyQwen3SelfAttention(QWEN3_HIDDEN_SIZE, device_str, dtype)
    layers = [attn.q_proj, attn.k_proj, attn.v_proj]

    batch_size = 4
    inp = torch.randn(batch_size, QWEN3_HIDDEN_SIZE, device=device_str, dtype=dtype)

    processor = _TestAWQProcessor(QuantizeConfig(quant_method=METHOD.AWQ, format=FORMAT.GEMM, group_size=group_size))

    captured = {}

    def fake_compute_best_scale(
        self,
        _inp,
        w_mean,
        x_mean,
        module2inspect,
        layers_arg,
        fp16_output,
        module_kwargs,
    ):
        captured["fast"] = w_mean.detach().to(torch.float32).cpu()
        captured["baseline"] = (
            _compute_legacy_w_mean(layers_arg, self.qcfg.group_size).detach().to(torch.float32).cpu()
        )
        return torch.ones_like(w_mean, dtype=w_mean.dtype).detach().cpu(), 0.0

    monkey_patcher = MonkeyPatch()
    monkey_patcher.setattr(AWQProcessor, "_compute_best_scale", fake_compute_best_scale)

    try:
        processor._search_best_scale(
            attn,
            layers[0],
            layers,
            inp,
            module2inspect=layers[0],
            kwargs={},
        )
    finally:
        monkey_patcher.undo()

    assert "fast" in captured and "baseline" in captured
    if dtype == torch.float32:
        atol = 2e-7
        rtol = 2e-7
    else:
        atol = 5e-4
        rtol = 1e-3
    fast = captured["fast"]
    baseline = captured["baseline"]

    abs_diff = (fast - baseline).abs()
    with torch.no_grad():
        safe_baseline = torch.where(baseline == 0, torch.ones_like(baseline), baseline)
        rel_diff = abs_diff / safe_baseline.abs()

    max_abs_diff = abs_diff.max().item()
    max_rel_diff = rel_diff.max().item()

    header = f"{'Metric':<20}{'Measured':<20}{'Tolerance':<20}"
    separator = "-" * len(header)
    print(f"AWQ weight mean comparison (fast vs baseline) [{param_name}]")
    print(separator)
    print(header)
    print(separator)
    print(f"{'max_abs_diff':<20}{max_abs_diff:<20.6e}{atol:<20.6e}")
    print(f"{'max_rel_diff':<20}{max_rel_diff:<20.6e}{rtol:<20.6e}")
    print(separator)

    assert torch.allclose(fast, baseline, rtol=rtol, atol=atol)

    def _time_it(fn, runs=5, warmup=2):
        for _ in range(warmup):
            fn()
        if device == "cuda":
            torch.cuda.synchronize(device_str)
        start = time.perf_counter()
        for _ in range(runs):
            fn()
        if device == "cuda":
            torch.cuda.synchronize(device_str)
        return (time.perf_counter() - start) / runs

    fast_time = _time_it(lambda: _compute_fast_w_mean(layers, group_size))
    legacy_time = _time_it(lambda: _compute_legacy_w_mean(layers, group_size))

    GREEN = "\033[32m"
    RED = "\033[31m"
    YELLOW = "\033[33m"
    RESET = "\033[0m"

    delta_ms = (fast_time - legacy_time) * 1e3
    rel = (fast_time / legacy_time) if legacy_time > 0 else float("inf")
    if rel <= 1.0:
        color = GREEN
        verdict = "faster"
    elif rel <= 1.05:
        color = YELLOW
        verdict = "≈ parity"
    else:
        color = RED
        verdict = "slower"

    print(f"AWQ weight mean timing [{param_name}]")
    print("+----------------+------------+--------------+-------------+---------------+")
    print("| Metric         | Fast (ms)  | Legacy (ms)  | Delta (ms)  | Relative      |")
    print("+----------------+------------+--------------+-------------+---------------+")
    print(
        f"| runtime        | {fast_time*1e3:10.3f} | {legacy_time*1e3:12.3f} | "
        f"{delta_ms:11.3f} | {color}{rel:>11.3%} {verdict:<7}{RESET}|"
    )
    print("+----------------+------------+--------------+-------------+---------------+")

    assert fast_time <= legacy_time * 1.05, (
        f"Streaming mean slower than legacy for {param_name}: "
        f"{fast_time*1e3:.3f} ms vs {legacy_time*1e3:.3f} ms"
    )
