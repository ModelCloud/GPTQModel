# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Measure actual Marlin weight layouts, separately from source ownership."""

import argparse
import copy
import gc
import json
import os
import statistics
from pathlib import Path

import torch

from gptqmodel import GPTQModel
from gptqmodel.models.fusion import _SHARED_OPERANDS, _set_operand
from gptqmodel.utils.backend import BACKEND


def clone_executor(source):
    clone = copy.copy(source)
    clone._parameters = source._parameters.copy()
    clone._buffers = source._buffers.copy()
    clone._modules = source._modules.copy()
    clone.__dict__.pop("clear_weight_cache", None)
    clone.workspace = torch.zeros_like(source.workspace)
    return clone


def layout(tensor, fused):
    return {"shape": list(tensor.shape), "stride": list(tensor.stride()),
            "storage_offset": tensor.storage_offset(), "contiguous": tensor.is_contiguous(),
            "shares_fused_storage": tensor.untyped_storage().data_ptr() == fused.untyped_storage().data_ptr()}


def benchmark(method, path, args):
    model = GPTQModel.load(path, backend=BACKEND.AUTO, device="cuda:0", dtype=torch.float16,
                          fuse=True, attn_implementation="eager")
    assert model.quantize_config.bits == 4 and model.quantize_config.group_size == 128
    assert model.quantize_config.sym and not model.quantize_config.desc_act
    kernel = "MarlinLinear" if method == "gptq" else "AwqMarlinLinear"
    results = []
    plan = model.fuse()
    with torch.inference_mode():
        for group in plan.groups[:2]:
            fused = group.packed
            assert type(fused).__name__ == kernel
            with torch.inference_mode(False), torch.no_grad():
                fresh = clone_executor(fused)
                sources = [clone_executor(module) for module in group.modules]
                layouts = {}
                for name, dim in _SHARED_OPERANDS:
                    operand = getattr(fused, name)
                    if operand is None or not operand.numel():
                        continue
                    rebuilt = torch.cat([getattr(m, name) for m in group.modules], dim=dim)
                    assert operand.is_contiguous() and rebuilt.is_contiguous()
                    assert operand.is_inference() == rebuilt.is_inference()
                    assert operand.untyped_storage().data_ptr() != rebuilt.untyped_storage().data_ptr()
                    torch.testing.assert_close(rebuilt, operand, rtol=0, atol=0)
                    _set_operand(fresh, name, rebuilt)
                    for original, contiguous in zip(group.modules, sources):
                        _set_operand(contiguous, name, getattr(original, name).clone(memory_format=torch.contiguous_format))
                    layouts[name] = {"fused_current": layout(operand, operand),
                                     "fused_fresh_concat": layout(rebuilt, operand),
                                     "source_views": [layout(getattr(m, name), operand) for m in group.modules],
                                     "source_contiguous": [layout(getattr(m, name), operand) for m in sources]}

            for rows in (1, 128):
                torch.manual_seed(733 + rows)
                x = torch.randn(rows, fused.in_features, dtype=torch.float16, device="cuda:0")
                current_output = fused.forward(x)
                torch.testing.assert_close(fresh.forward(x), current_output, rtol=0, atol=0)
                source_output = torch.cat([m.forward(x) for m in group.modules], dim=-1)
                copied_output = torch.cat([m.forward(x) for m in sources], dim=-1)
                torch.testing.assert_close(source_output, copied_output, rtol=0, atol=0)
                torch.testing.assert_close(source_output, current_output, rtol=2e-3, atol=2e-3)
                calls = {
                    "fused_current": lambda: fused.forward(x),
                    "fused_fresh_concat": lambda executor=fresh: executor.forward(x),
                    "separate_source_views": lambda: tuple(m.forward(x) for m in group.modules),
                    "separate_source_contiguous": lambda modules=tuple(sources): tuple(m.forward(x) for m in modules),
                }
                samples = {name: [] for name in calls}
                for call in calls.values():
                    for _ in range(args.warmups):
                        call()
                torch.cuda.synchronize()
                names = list(calls)
                for repeat in range(args.repeats):
                    order = names[repeat % len(names):] + names[:repeat % len(names)]
                    for name in order:
                        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        start.record()
                        for _ in range(args.inner):
                            calls[name]()
                        end.record()
                        end.synchronize()
                        samples[name].append(start.elapsed_time(end) * 1000 / args.inner)
                result = {"family": group.decision.family, "parent": group.decision.parent,
                          "kernel": kernel, "rows": rows, "input_features": fused.in_features,
                          "output_features": fused.out_features, "operands_bitwise_equal": True,
                          "current_vs_fresh_logits_bitwise_equal": True,
                          "source_view_vs_contiguous_logits_bitwise_equal": True,
                          "layouts": layouts,
                          "median_us": {name: statistics.median(values) for name, values in samples.items()},
                          "samples_us": samples}
                results.append(result)
                print(method, group.decision.family, rows, result["median_us"], flush=True)
            # Drop copied quantized payloads before cyclic nn.Module GC runs.
            for executor in [fresh, *sources]:
                executor._parameters.clear()
                executor._buffers.clear()
                executor.__dict__.pop("workspace", None)
            del fresh, sources, calls
    model.fuse(False)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gptq-path", required=True)
    parser.add_argument("--awq-path", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=31)
    parser.add_argument("--inner", type=int, default=16)
    args = parser.parse_args()
    if min(args.repeats, args.inner) < 1 or args.warmups < 0:
        parser.error("repeats/inner must be positive; warmups must be nonnegative")
    result = {"hardware": torch.cuda.get_device_name(0), "torch": torch.__version__,
              "cpu_affinity": sorted(os.sched_getaffinity(0)), "dtype": "float16",
              "bits": 4, "group_size": 128, "sym": True, "desc_act": False,
              "backend": "auto", "warmups": args.warmups, "repeats": args.repeats, "inner": args.inner,
              "timing": "CUDA event median microseconds per call; mode order rotated per repeat; construction excluded",
              "scope": "first-layer QKV/gate-up; fused inputs are contiguous in both fused cases; source views are strided"}
    for method, path in (("gptq", args.gptq_path), ("awq", args.awq_path)):
        result[method] = benchmark(method, path, args)
        Path(args.output_json).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
