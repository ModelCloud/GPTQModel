# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Trace decoder activations on one fixed prompt in a guarded GB10 process."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from .w4a_gb10_memory import require_w4a_test_headroom


def trace(checkpoint: Path, variant: str, prompt_result: Path, output: Path) -> None:
    require_w4a_test_headroom(require_scope=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from gptqmodel import BACKEND, GPTQModel
    from gptqmodel.nn_modules.qlinear.w4a_activation import W4AActivation

    if variant == "bf16":
        core = AutoModelForCausalLM.from_pretrained(
            checkpoint, dtype=torch.bfloat16, local_files_only=True,
            low_cpu_mem_usage=False,
        ).to("cuda:0")
    else:
        backend = {
            "w4a16": BACKEND.GPTQ_TRITON,
            "w4afp8": BACKEND.GPTQ_W4AFP8,
            "w4a_nvfp4": BACKEND.GPTQ_W4A_NVFP4,
        }[variant]
        core = GPTQModel.load(str(checkpoint), backend=backend, device="cuda:0", dtype=torch.bfloat16).model
    core.eval()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    prompt = json.loads(prompt_result.read_text())["tests"][0]["samples"][0]["prompt"]
    ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"].to("cuda:0")
    tensors = {"input_ids": ids.cpu()}
    handles = []

    def capture(index):
        def hook(_module, _args, result):
            value = result[0] if isinstance(result, tuple) else result
            if isinstance(value, W4AActivation):
                value = value.decode(torch.float32)
            tensors[f"layer_{index}"] = value.float().cpu().contiguous()
        return hook

    for index, layer in enumerate(core.model.layers):
        handles.append(layer.register_forward_hook(capture(index)))
    try:
        with torch.inference_mode():
            result = core(input_ids=ids, use_cache=False)
            tensors["logits"] = result.logits[:, -1, :].float().cpu().contiguous()
    finally:
        for handle in handles:
            handle.remove()
    if len(tensors) != 18 or not torch.isfinite(tensors["logits"]).all():
        raise AssertionError("The layer trace is incomplete or nonfinite.")
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(output))
    print(json.dumps({"variant": variant, "checkpoint": str(checkpoint),
                      "tokens": ids.shape[1], "next_token": int(tensors["logits"].argmax()),
                      "trace": str(output)}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--variant", choices=("bf16", "w4a16", "w4afp8", "w4a_nvfp4"), required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompt-result", type=Path)
    source.add_argument("--calibration", type=Path, help="Verified disjoint calibration artifact; selection partition only.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--probes", type=int, default=64)
    parser.add_argument("--experimental-token-energy", choices=("none", "all", "residual"), default="none",
                        help="Selection-only experiment preserving token norm with the encoded carrier multiplier.")
    parser.add_argument("--experimental-token-global", action="store_true",
                        help="Selection-only experiment using an FP32 outer NVFP4 scale per token.")
    parser.add_argument("--training-replay", action="store_true",
                        help="Audit weight-adaptation replay on disjoint selection articles.")
    parser.add_argument("--hardware-forward", action="store_true",
                        help="Use encoded hardware values inside the training-replay diagnostic.")
    args = parser.parse_args()
    if args.calibration is not None:
        from .w4a_heldout_trace import trace_heldout

        trace_heldout(args.checkpoint, args.variant, args.calibration, args.output,
                      rows=args.rows, sequence_length=args.sequence_length, probes=args.probes,
                      token_energy=args.experimental_token_energy, token_global=args.experimental_token_global,
                      training_replay=args.training_replay, hardware_forward=args.hardware_forward)
    else:
        if (args.experimental_token_energy != "none" or args.experimental_token_global
                or args.training_replay or args.hardware_forward):
            parser.error("Token-scale and replay diagnostics require the disjoint --calibration artifact")
        trace(args.checkpoint, args.variant, args.prompt_result, args.output)


if __name__ == "__main__":
    main()
