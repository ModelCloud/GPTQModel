#!/usr/bin/env python3
"""Benchmark GPTQ.quantize() for the largest GLM-4.6 MoE expert Linear layer."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from accelerate import init_empty_weights
from safetensors import safe_open
from transformers import AutoConfig, Glm4MoeForCausalLM
from tabulate import tabulate

from gptqmodel.quantization.config import FORMAT, METHOD, QuantizeConfig
from gptqmodel.quantization.gptq import GPTQ

MODEL_PATH = Path("/monster/data/model/GLM-4.6")
BATCH_SIZE = 4
SEQ_LEN = 2048
BLOCKSIZE = 128
DTYPE = torch.float32
MATMUL_DTYPE = torch.bfloat16
# Token-wise forward shapes to test (batch, hidden) x (hidden, intermediate)
FORWARD_BATCH_SIZES = (1, 2, 3, 4)
# Honor CUDA_VISIBLE_DEVICES masking by letting torch resolve the active device.
GPU_DEVICE = torch.device("cuda")
TIME_PRECISION = 5
ANSI_RED = "\033[31m"
ANSI_GREEN = "\033[32m"
ANSI_RESET = "\033[0m"


def has_cpu_autocast() -> bool:
    new_api = hasattr(torch, "amp") and hasattr(torch.amp, "autocast")
    legacy_api = hasattr(torch, "cpu") and hasattr(torch.cpu, "amp") and hasattr(torch.cpu.amp, "autocast")
    return new_api or legacy_api


def cpu_autocast_context(dtype: torch.dtype):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type="cpu", dtype=dtype)
    if hasattr(torch, "cpu") and hasattr(torch.cpu, "amp") and hasattr(torch.cpu.amp, "autocast"):
        return torch.cpu.amp.autocast(dtype=dtype)
    raise RuntimeError("torch.amp.autocast or torch.cpu.amp.autocast is not available for CPU.")


CPU_AUTOCAST_AVAILABLE = has_cpu_autocast()


def fmt_value(value: float, precision: int = 2) -> str:
    if not torch.isfinite(torch.tensor(value)):
        return str(value)
    return f"{value:.{precision}f}"


def fmt_time(value: float) -> str:
    return fmt_value(value, precision=TIME_PRECISION)


def to_channels_last_2d(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() != 2:
        raise ValueError("channels_last conversion expects a rank-2 tensor.")
    leading = tensor.size(0)
    reshaped = tensor.contiguous().view(leading, tensor.size(1), 1, 1)
    cl = reshaped.to(memory_format=torch.channels_last)
    return cl.contiguous().view_as(tensor)


@dataclass
class BenchRun:
    device_label: str
    elapsed_s: float
    nsamples: int

    @property
    def samples_per_s(self) -> float:
        return float("inf") if self.elapsed_s == 0 else self.nsamples / self.elapsed_s


def locate_largest_expert_linear(model_path: Path) -> Tuple[str, Tuple[int, int]]:
    config = AutoConfig.from_pretrained(str(model_path))
    with init_empty_weights(include_buffers=True):
        model = Glm4MoeForCausalLM(config)

    largest_name = None
    largest_shape = None
    largest_numel = -1

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and "experts" in name:
            numel = module.weight.numel()
            if numel > largest_numel:
                largest_name = name
                largest_shape = tuple(module.weight.shape)
                largest_numel = numel

    if largest_shape is None:
        raise RuntimeError("Unable to find any MoE expert nn.Linear modules.")
    return largest_name, largest_shape


def load_weight_map(model_path: Path) -> Dict[str, str]:
    index_file = model_path / "model.safetensors.index.json"
    if not index_file.exists():
        return {}
    data = json.loads(index_file.read_text())
    return data.get("weight_map", {})


def resolve_tensor_shard(model_path: Path, tensor_name: str, weight_map: Dict[str, str]) -> Path:
    filename = weight_map.get(tensor_name)
    if filename:
        return model_path / filename
    # Fall back to scanning shards if no explicit mapping is available.
    for shard_path in sorted(model_path.glob("model-*.safetensors")):
        with safe_open(shard_path, framework="pt", device="cpu") as shard:
            if tensor_name in shard.keys():
                return shard_path
    single_file = model_path / "model.safetensors"
    if single_file.exists():
        with safe_open(single_file, framework="pt", device="cpu") as shard:
            if tensor_name in shard.keys():
                return single_file
    raise FileNotFoundError(f"Unable to locate tensor {tensor_name} within {model_path}")


def load_linear_weight(model_path: Path, module_name: str) -> torch.Tensor:
    tensor_name = f"{module_name}.weight"
    weight_map = load_weight_map(model_path)
    shard_path = resolve_tensor_shard(model_path, tensor_name, weight_map)
    with safe_open(shard_path, framework="pt", device="cpu") as shard:
        return shard.get_tensor(tensor_name).to(dtype=DTYPE).contiguous()


def build_quant_config(device: torch.device | None = None) -> QuantizeConfig:
    qcfg = QuantizeConfig(
        quant_method=METHOD.GPTQ,
        format=FORMAT.GPTQ,
        bits=4,
        group_size=32,
        desc_act=False,
        act_group_aware=True,
        sym=True,
        fail_safe=True,
        damp_percent=0.05,
        offload_to_disk=False,
    )
    qcfg.device = device
    return qcfg


def clone_linear(shape: Tuple[int, int], weights: torch.Tensor, device: torch.device) -> nn.Linear:
    out_features, in_features = shape
    layer = nn.Linear(in_features, out_features, bias=False, device=device, dtype=DTYPE)
    with torch.no_grad():
        layer.weight.copy_(weights.to(device))
    return layer


def run_quantize_benchmark(layer: nn.Module, inputs: torch.Tensor, device: torch.device) -> BenchRun:
    qcfg = build_quant_config(device=device)
    gptq = GPTQ(layer, qcfg=qcfg)
    gptq.fail_safe = bool(qcfg.fail_safe)
    gptq.quantizer.configure(perchannel=True)

    gptq.add_batch(inputs, None)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    gptq.quantize(blocksize=BLOCKSIZE)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    return BenchRun(device_label=str(device), elapsed_s=time.perf_counter() - start, nsamples=gptq.nsamples)


def run_forward_matmul(
    weights: torch.Tensor,
    inputs: torch.Tensor,
    device: torch.device,
    batch_size: int,
    enable_cpu_amp: bool = False,
    use_channels_last: bool = False,
) -> BenchRun:
    if enable_cpu_amp and device.type != "cpu":
        raise ValueError("CPU autocast is only applicable to CPU devices.")
    if enable_cpu_amp and not CPU_AUTOCAST_AVAILABLE:
        raise RuntimeError("torch.cpu.amp.autocast is not available in this environment.")
    if use_channels_last and device.type != "cpu":
        raise ValueError("channels_last memory format test only applies to CPU tensors.")

    matmul_dtype = MATMUL_DTYPE
    weight = weights.to(device=device, dtype=matmul_dtype, copy=True)
    inp = inputs.to(device=device, dtype=matmul_dtype, copy=True)

    if device.type == "cpu" and use_channels_last:
        weight = to_channels_last_2d(weight)
        inp = to_channels_last_2d(inp)

    if device.type == "cuda":
        sync = torch.cuda.synchronize
    else:
        sync = lambda *args, **kwargs: None

    ctx = nullcontext()
    if device.type == "cpu" and enable_cpu_amp:
        ctx = cpu_autocast_context(dtype=MATMUL_DTYPE)

    sync(device)
    start = time.perf_counter()
    with ctx:
        _ = inp @ weight.t()
    sync(device)

    return BenchRun(device_label=str(device), elapsed_s=time.perf_counter() - start, nsamples=batch_size)


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"{MODEL_PATH} is not available.")

    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    else:
        raise RuntimeError("CUDA must be available to run the GPU benchmark.")

    expert_name, expert_shape = locate_largest_expert_linear(MODEL_PATH)
    expert_weights = load_linear_weight(MODEL_PATH, expert_name)
    base_inputs = torch.randn(BATCH_SIZE, SEQ_LEN, expert_shape[1], dtype=DTYPE)
    forward_inputs = {
        bsz: torch.randn(bsz, expert_shape[1], dtype=DTYPE)
        for bsz in FORWARD_BATCH_SIZES
    }

    cpu_device = torch.device("cpu")
    cpu_layer = clone_linear(expert_shape, expert_weights, cpu_device)
    cpu_inputs = base_inputs.to(cpu_device)
    cpu_run = run_quantize_benchmark(cpu_layer, cpu_inputs, cpu_device)

    gpu_layer = clone_linear(expert_shape, expert_weights, GPU_DEVICE)
    gpu_inputs = base_inputs.to(GPU_DEVICE)
    gpu_run = run_quantize_benchmark(gpu_layer, gpu_inputs, GPU_DEVICE)

    forward_runs = []
    for bsz in FORWARD_BATCH_SIZES:
        cpu_forward = run_forward_matmul(expert_weights, forward_inputs[bsz], cpu_device, bsz)
        cpu_forward_cl = run_forward_matmul(
            expert_weights,
            forward_inputs[bsz],
            cpu_device,
            bsz,
            use_channels_last=True,
        )
        cpu_forward_amp: Optional[BenchRun] = None
        cpu_forward_amp_cl: Optional[BenchRun] = None
        if CPU_AUTOCAST_AVAILABLE:
            cpu_forward_amp = run_forward_matmul(
                expert_weights,
                forward_inputs[bsz],
                cpu_device,
                bsz,
                enable_cpu_amp=True,
            )
            cpu_forward_amp_cl = run_forward_matmul(
                expert_weights,
                forward_inputs[bsz],
                cpu_device,
                bsz,
                enable_cpu_amp=True,
                use_channels_last=True,
            )
        gpu_forward = run_forward_matmul(expert_weights, forward_inputs[bsz], GPU_DEVICE, bsz)
        forward_runs.append(
            (
                bsz,
                cpu_forward,
                cpu_forward_cl,
                cpu_forward_amp,
                cpu_forward_amp_cl,
                gpu_forward,
            )
        )

    print(f"Largest expert linear: {expert_name} with weight shape {expert_shape}")
    print(f"Calibration tokens: batch={BATCH_SIZE}, seq={SEQ_LEN}, total_tokens={BATCH_SIZE * SEQ_LEN}")

    quant_rows = [
        [
            "CPU",
            fmt_time(cpu_run.elapsed_s),
            cpu_run.nsamples,
            f"{cpu_run.samples_per_s:.2f}",
            cpu_run.device_label,
        ],
        [
            "GPU",
            fmt_time(gpu_run.elapsed_s),
            gpu_run.nsamples,
            f"{gpu_run.samples_per_s:.2f}",
            gpu_run.device_label,
        ],
    ]
    print("\n--- GPTQ quantize timing ---")
    print(tabulate(quant_rows, headers=["Device", "time (s)", "nsamples", "samples/s", "label"], tablefmt="github"))
    print(f"GPU speedup vs CPU: {cpu_run.elapsed_s / gpu_run.elapsed_s:.2f}x faster\n")

    time_headers = [
        "batch",
        "CPU bf16 (s)",
        "CPU bf16 CL (s)",
        "CPU autocast (s)",
        "CPU autocast CL (s)",
        "GPU (s)",
    ]
    throughput_headers = [
        "batch",
        "CPU bf16 samp/s",
        "vs GPU",
        "CPU bf16 CL samp/s",
        "vs GPU",
        "CPU autocast samp/s",
        "vs GPU",
        "CPU autocast CL samp/s",
        "vs GPU",
        "GPU samp/s",
    ]
    time_rows = []
    throughput_rows = []
    for (
        bsz,
        cpu_forward,
        cpu_forward_cl,
        cpu_forward_amp,
        cpu_forward_amp_cl,
        gpu_forward,
    ) in forward_runs:
        def fmt(run: Optional[BenchRun]) -> Tuple[str, str]:
            if run is None:
                return "-", "-"
            return fmt_time(run.elapsed_s), fmt_value(run.samples_per_s)

        cpu_bf16_time, cpu_bf16_samples = fmt(cpu_forward)
        cpu_bf16_cl_time, cpu_bf16_cl_samples = fmt(cpu_forward_cl)
        cpu_amp_time, cpu_amp_samples = fmt(cpu_forward_amp)
        cpu_amp_cl_time, cpu_amp_cl_samples = fmt(cpu_forward_amp_cl)
        gpu_time, gpu_samples = fmt(gpu_forward)

        def speedup(ref: Optional[BenchRun]) -> str:
            if ref is None:
                return "-"
            gpu_time = gpu_forward.elapsed_s
            ref_time = ref.elapsed_s
            if gpu_time == 0 or ref_time == 0:
                return "-"
            if math.isclose(ref_time, gpu_time, rel_tol=1e-9):
                return f"{fmt_value(1.0, precision=2)}x same"
            cpu_faster = ref_time < gpu_time
            if cpu_faster:
                ratio = gpu_time / ref_time
                prefix = f"{ANSI_GREEN}+"
                qualifier = "faster"
            else:
                ratio = ref_time / gpu_time
                prefix = f"{ANSI_RED}-"
                qualifier = "slower"
            return f"{prefix}{fmt_value(ratio, precision=2)}x {qualifier}{ANSI_RESET}"

        speedup_bf16 = speedup(cpu_forward)
        speedup_bf16_cl = speedup(cpu_forward_cl)
        speedup_amp = speedup(cpu_forward_amp)
        speedup_amp_cl = speedup(cpu_forward_amp_cl)

        time_rows.append(
            [
                bsz,
                cpu_bf16_time,
                cpu_bf16_cl_time,
                cpu_amp_time,
                cpu_amp_cl_time,
                gpu_time,
            ]
        )
        throughput_rows.append(
            [
                bsz,
                cpu_bf16_samples,
                speedup_bf16,
                cpu_bf16_cl_samples,
                speedup_bf16_cl,
                cpu_amp_samples,
                speedup_amp,
                cpu_amp_cl_samples,
                speedup_amp_cl,
                gpu_samples,
            ]
        )

    print("--- bf16 token-level matmul (batch × hidden × intermediate) ---")
    if not CPU_AUTOCAST_AVAILABLE:
        print("Note: torch.cpu.amp.autocast is unavailable; CPU autocast columns contain '-'.")
    print("Time per batch:")
    print(tabulate(time_rows, headers=time_headers, tablefmt="github", disable_numparse=True))
    print("\nThroughput and GPU speedups:")
    print(tabulate(throughput_rows, headers=throughput_headers, tablefmt="github", disable_numparse=True))


if __name__ == "__main__":
    main()
