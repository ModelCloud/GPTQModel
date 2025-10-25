#!/usr/bin/env python3
"""Benchmark GPTQ.quantize() for the largest GLM-4.6 MoE expert Linear layer."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from accelerate import init_empty_weights
from safetensors import safe_open
from transformers import AutoConfig, Glm4MoeForCausalLM

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


def run_forward_matmul(weights: torch.Tensor, inputs: torch.Tensor, device: torch.device, batch_size: int) -> BenchRun:
    weight = weights.to(device=device, dtype=MATMUL_DTYPE, copy=True)
    inp = inputs.to(device=device, dtype=MATMUL_DTYPE, copy=True)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    _ = inp @ weight.t()
    if device.type == "cuda":
        torch.cuda.synchronize(device)

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
        gpu_forward = run_forward_matmul(expert_weights, forward_inputs[bsz], GPU_DEVICE, bsz)
        forward_runs.append((bsz, cpu_forward, gpu_forward))

    print(f"Largest expert linear: {expert_name} with weight shape {expert_shape}")
    print(f"Calibration tokens: batch={BATCH_SIZE}, seq={SEQ_LEN}, total_tokens={BATCH_SIZE * SEQ_LEN}")
    print(f"CPU quantize time:  {cpu_run.elapsed_s:.3f}s (nsamples={cpu_run.nsamples}) on {cpu_run.device_label}")
    print(f"GPU quantize time:  {gpu_run.elapsed_s:.3f}s (nsamples={gpu_run.nsamples}) on {gpu_run.device_label}")
    print(f"GPU speedup vs CPU: {cpu_run.elapsed_s / gpu_run.elapsed_s:.2f}x faster")
    print("--- bf16 token-level matmul (batch × hidden × intermediate) ---")
    for bsz, cpu_forward, gpu_forward in forward_runs:
        speedup = cpu_forward.elapsed_s / gpu_forward.elapsed_s
        print(
            f"batch={bsz}: CPU {cpu_forward.elapsed_s:.5f}s ({cpu_forward.samples_per_s:.2f} samples/s) "
            f"vs GPU {gpu_forward.elapsed_s:.5f}s ({gpu_forward.samples_per_s:.2f} samples/s) "
            f"-> GPU {speedup:.2f}x faster"
        )


if __name__ == "__main__":
    main()
