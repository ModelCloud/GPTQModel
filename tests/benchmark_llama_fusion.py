# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Reproducible local Llama 3.2 1B fusion prefill/decode and drift report."""

import argparse
import gc
import json
import statistics
import time
from pathlib import Path

import torch
from safetensors import safe_open

from gptqmodel import GPTQModel
from gptqmodel.utils.backend import BACKEND


def drift(actual, reference):
    difference = actual.float() - reference.float()
    return {
        "max_abs": difference.abs().max().item(),
        "rms": difference.square().mean().sqrt().item(),
        "relative_l2": (difference.norm() / reference.float().norm().clamp_min(1e-12)).item(),
        "top1_agreement": (actual.argmax(-1) == reference.argmax(-1)).float().mean().item(),
    }


def rounding_diagnostics(model):
    """Independent bit shifts + FP32 GEMM, separate from the inference kernel."""
    results = []
    source = Path(model.model_local_path)
    index = source / "model.safetensors.index.json"
    weight_map = json.loads(index.read_text())["weight_map"] if index.exists() else None

    def tensor(name):
        filename = weight_map[name] if weight_map else "model.safetensors"
        with safe_open(str(source / filename), framework="pt", device="cpu") as reader:
            return reader.get_tensor(name).to("cuda:0")

    torch.manual_seed(431)
    for group in model.fusion_plan().groups[:2]:
        first = group.modules[0]
        x = torch.randn(128, first.in_features, dtype=torch.float16, device=first.qweight.device)
        shifts = torch.arange(0, 32, 4, device=x.device)
        oracle, native = [], []
        for module, forward, relative in zip(group.modules, group.originals, group.decision.paths):
            prefix = group.decision.parent + "." + relative.rsplit(".", 1)[-1]
            raw_weight = tensor(prefix + ".qweight")
            raw_zeros = tensor(prefix + ".qzeros")
            scales = tensor(prefix + ".scales")
            zeros = ((raw_zeros.long().unsqueeze(-1) >> shifts) & 15).reshape(scales.shape)
            if model.quantize_config.method == "gptq":
                codes = ((raw_weight.long().unsqueeze(1) >> shifts.view(1, -1, 1)) & 15).reshape(
                    module.in_features, module.out_features)
                indices = tensor(prefix + ".g_idx").long()
                if str(model.quantize_config.format) in ("gptq", "FORMAT.GPTQ"):
                    zeros = (zeros + 1) & 15
            else:
                codes = ((raw_weight.long().unsqueeze(-1) >> shifts) & 15).reshape(
                    module.in_features, module.out_features)
                reverse = [0, 4, 1, 5, 2, 6, 3, 7]
                codes = codes.reshape(module.in_features, -1, 8)[..., reverse].reshape(codes.shape)
                zeros = zeros.reshape(zeros.shape[0], -1, 8)[..., reverse].reshape(zeros.shape)
                indices = torch.arange(module.in_features, device=x.device) // module.group_size
            weight = ((codes - zeros[indices]).float() * scales[indices].float()).half()
            reference = x.float() @ weight.float()
            reference = reference.half()  # Native matmul's specified output rounding.
            if module.bias is not None:
                reference = reference + module.bias
            oracle.append(reference)
            native.append(forward(x))
        reference = torch.cat(oracle, dim=-1)
        separate = torch.cat(native, dim=-1)
        combined = group.packed(x)
        torch.testing.assert_close(combined, reference, rtol=2e-3, atol=2e-3)
        torch.testing.assert_close(separate, reference, rtol=2e-3, atol=2e-3)
        results.append({"family": group.decision.family, "parent": group.decision.parent,
                        "packed_operands_equal": all(torch.equal(getattr(group.packed, name),
                            torch.cat([getattr(m, name) for m in group.modules], dim=1))
                            for name in ("qweight", "qzeros", "scales") if getattr(group.packed, name).numel()),
                        "on_vs_off": drift(combined.cpu(), separate.cpu()),
                        "off_vs_independent_oracle": drift(separate.cpu(), reference.cpu()),
                        "on_vs_independent_oracle": drift(combined.cpu(), reference.cpu())})
    return results


def benchmark(method, path, args):
    backend = BACKEND.AUTO
    model = GPTQModel.load(path, backend=backend, device="cuda:0", dtype=torch.float16,
                          fuse=False, attn_implementation="eager")
    assert model.quantize_config.bits == 4 and model.quantize_config.group_size == 128
    assert model.quantize_config.sym and not model.quantize_config.desc_act
    expected_kernel = "MarlinLinear" if method == "gptq" else "AwqMarlinLinear"
    assert type(model.model.model.layers[0].self_attn.q_proj).__name__ == expected_kernel
    text = "The capital of France is Paris. A small language model explains science and mathematics. "
    ids = model.tokenizer(text * args.prefill_tokens, return_tensors="pt").input_ids[:, :args.prefill_tokens].cuda()
    forced = torch.full((1, args.decode_steps), 42, dtype=torch.long, device="cuda:0")

    def prefill(wrapper):
        torch.cuda.synchronize()
        start = time.perf_counter()
        output = wrapper.model(input_ids=ids, use_cache=False)
        torch.cuda.synchronize()
        return output.logits, time.perf_counter() - start

    def decode(wrapper):
        cache = wrapper.model(input_ids=ids, use_cache=True).past_key_values
        outputs = []
        torch.cuda.synchronize()
        start = time.perf_counter()
        for index in range(args.decode_steps):
            output = wrapper.model(input_ids=forced[:, index:index + 1], past_key_values=cache, use_cache=True)
            cache = output.past_key_values
            outputs.append(output.logits)
        torch.cuda.synchronize()
        return torch.cat(outputs, dim=1), time.perf_counter() - start

    results = {}
    reference = {}
    with torch.inference_mode():
        for enabled in (False, True):
            model.fuse(enabled)
            for _ in range(args.warmups):
                prefill(model)
                decode(model)
            prefill_times, decode_times = [], []
            for _ in range(args.repeats):
                prefill_logits, seconds = prefill(model)
                prefill_times.append(seconds)
                decode_logits, seconds = decode(model)
                decode_times.append(seconds)
            label = "on" if enabled else "off"
            results[label] = {
                "kernel": expected_kernel,
                "prefill_ms": statistics.median(prefill_times) * 1000,
                "prefill_tokens_per_second": args.prefill_tokens / statistics.median(prefill_times),
                "decode_ms_per_token": statistics.median(decode_times) * 1000 / args.decode_steps,
                "decode_tokens_per_second": args.decode_steps / statistics.median(decode_times),
                "prefill_samples_ms": [seconds * 1000 for seconds in prefill_times],
                "decode_samples_ms_per_token": [seconds * 1000 / args.decode_steps for seconds in decode_times],
            }
            if not enabled:
                reference = {"prefill": prefill_logits.cpu(), "decode": decode_logits.cpu()}
            else:
                results["drift"] = {"prefill": drift(prefill_logits.cpu(), reference["prefill"]),
                                    "decode": drift(decode_logits.cpu(), reference["decode"])}
                results["eligible_groups"] = len(model.fusion_plan().groups)
                results["extra_packed_bytes"] = sum(
                    t.numel() * t.element_size() for g in model.fusion_plan().groups
                    for name, t in list(g.packed.named_parameters()) + list(g.packed.named_buffers())
                    if name not in ("g_idx", "g_idx_sort_indices")
                )
                results["decisions"] = model.fusion_plan().to_dict()
                results["projection_rounding"] = rounding_diagnostics(model)
    del model, prefill_logits, decode_logits
    gc.collect()
    torch.cuda.empty_cache()
    print(method, json.dumps({key: value for key, value in results.items() if key != "decisions"}), flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gptq-path", required=True)
    parser.add_argument("--awq-path", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--prefill-tokens", type=int, default=128)
    parser.add_argument("--decode-steps", type=int, default=16)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if min(args.prefill_tokens, args.decode_steps, args.repeats) < 1 or args.warmups < 0:
        parser.error("token counts and repeats must be positive; warmups must be nonnegative")
    result = {"hardware": torch.cuda.get_device_name(0), "torch": torch.__version__,
              "dtype": "float16", "batch_size": 1, "prefill_tokens": args.prefill_tokens,
              "decode_steps": args.decode_steps, "warmups": args.warmups, "repeats": args.repeats}
    for method, path in (("gptq", args.gptq_path), ("awq", args.awq_path)):
        result[method] = benchmark(method, path, args)
        Path(args.output_json).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
