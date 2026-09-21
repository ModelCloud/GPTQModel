"""Measure complete ExLlamaV2 GPTQ fused/dense forwards.

Run with the GPU-visible environment, for example:

  PATH=/root/miniconda3/envs/gp_314t/bin:$PATH \
  python tests/benchmark/exllamav2_dispatch.py --device 0 --repeats 30

The output is JSONL so four independent GPU workers can be run and merged
without changing the timing code.  This script intentionally initializes the
JIT extension and q_handle before timing.  Each timed call includes the normal
input conversion, output allocation, dispatch, reconstruct (for dense), bias,
and adapter hooks used by ExllamaV2Linear.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

from gptqmodel import BACKEND
from gptqmodel.nn_modules.qlinear.exllamav2 import ExllamaV2Linear
from gptqmodel.utils.exllamav2 import ScratchSpace


MS = [1, 8, 16, 32, 40, 48, 49, 50, 51, 52, 64, 96, 128, 256, 512, 2048]
SHAPES = [(2048, 512), (2048, 2048), (2048, 8192), (8192, 2048), (4096, 4096)]
GROUP_SIZES = [32, 64, 128, -1]


def _module(k, n, group_size, desc_act, device):
    module = ExllamaV2Linear(
        bits=4,
        group_size=group_size,
        desc_act=desc_act,
        sym=True,
        in_features=k,
        out_features=n,
        bias=True,
        pack_dtype=torch.int32,
        backend=BACKEND.GPTQ_EXLLAMA_V2,
    ).to(device).eval()

    generator = torch.Generator(device=device).manual_seed(1729 + k + n + group_size)
    module.qweight.copy_(torch.randint(-100, 100, module.qweight.shape, generator=generator, device=device, dtype=torch.int32))
    module.qzeros.copy_(torch.randint(0, 2**32, module.qzeros.shape, generator=generator, device=device, dtype=torch.int64).to(torch.int32))
    module.scales.copy_(torch.rand(module.scales.shape, generator=generator, device=device, dtype=torch.float16) + 0.05)
    module.bias.copy_(torch.rand(module.bias.shape, generator=generator, device=device, dtype=torch.float16))

    if desc_act:
        groups = k // module.group_size
        perm = torch.randperm(groups, generator=generator, device=device)
        module.g_idx.copy_(perm.repeat_interleave(module.group_size))
    else:
        module.g_idx.zero_()

    scratch = ScratchSpace(
        module.scratch_space_fixed(max_input_len=max(MS), max_batch_size=1), device
    )
    module.post_init(scratch)
    return module


def _timed(module, x, mode, repeats, warmup):
    with torch.inference_mode():
        for _ in range(warmup):
            module(x, execution_mode=mode)
        torch.cuda.synchronize(x.device)

        samples = []
        peak = 0
        for _ in range(repeats):
            torch.cuda.reset_peak_memory_stats(x.device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            out = module(x, execution_mode=mode)
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end))
            peak = max(peak, torch.cuda.max_memory_allocated(x.device))
        metrics = {
            "mean_ms": statistics.mean(samples),
            "median_ms": statistics.median(samples),
            "stdev_ms": statistics.stdev(samples) if len(samples) > 1 else 0.0,
            "min_ms": min(samples),
            "max_ms": max(samples),
            "peak_memory_bytes": peak,
            "path": module.last_exllamav2_path,
            "output_checksum": float(out.float().abs().mean().item()),
        }
        return metrics, out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)
    props = torch.cuda.get_device_properties(device)
    print(json.dumps({
        "record": "device",
        "device": args.device,
        "name": props.name,
        "capability": [props.major, props.minor],
        "sm_count": props.multi_processor_count,
        "device_count": torch.cuda.device_count(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }), flush=True)

    case_count = 0
    combination_index = 0
    for k, n in SHAPES:
        for group_size in GROUP_SIZES:
            for desc_act in (False, True):
                selected_combination = combination_index % args.shards == args.shard
                combination_index += 1
                if not selected_combination:
                    continue
                setup_started = time.perf_counter()
                module = _module(k, n, group_size, desc_act, device)
                print(json.dumps({
                    "record": "setup",
                    "k": k,
                    "n": n,
                    "requested_group_size": group_size,
                    "desc_act": desc_act,
                    "seconds": time.perf_counter() - setup_started,
                }), flush=True)
                generator = torch.Generator(device=device).manual_seed(991 + k + n)
                for m in MS:
                    if args.limit is not None and case_count >= args.limit:
                        break
                    x = torch.randn((m, k), generator=generator, device=device, dtype=torch.float32)
                    row = {
                        "record": "case",
                        "m": m,
                        "n": n,
                        "k": k,
                        "group_size": group_size if group_size != -1 else k,
                        "requested_group_size": group_size,
                        "desc_act": desc_act,
                    }
                    timed = {}
                    outputs = {}
                    for mode in ("fused", "dense", "legacy", "auto"):
                        started = time.perf_counter()
                        row[mode], outputs[mode] = _timed(
                            module, x, mode, args.repeats, args.warmup
                        )
                        row[mode]["wall_setup_s"] = time.perf_counter() - started
                    dense_output = outputs["dense"]
                    for mode, output in outputs.items():
                        error = (output.float() - dense_output.float()).abs()
                        row[mode]["max_abs_error_vs_dense"] = float(error.max().item())
                        row[mode]["mean_abs_error_vs_dense"] = float(error.mean().item())
                    print(json.dumps(row), flush=True)
                    case_count += 1
                del module
                torch.cuda.empty_cache()
                if args.limit is not None and case_count >= args.limit:
                    break
            if args.limit is not None and case_count >= args.limit:
                break
        if args.limit is not None and case_count >= args.limit:
            break


if __name__ == "__main__":
    main()
