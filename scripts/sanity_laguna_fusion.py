#!/usr/bin/env python
"""Sanity-check Laguna-S-2.1-GPTQ-FIXED generation with FA2 + all fusions + grouped Marlin MoE."""
from __future__ import annotations

import argparse
import os
import sys
import time

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, repo_root)

# Parse --gpu before any CUDA context is created.
_gpu = "0"
for i, arg in enumerate(sys.argv):
    if arg == "--gpu" and i + 1 < len(sys.argv):
        _gpu = sys.argv[i + 1]
        break
    if arg.startswith("--gpu="):
        _gpu = arg.split("=", 1)[1]
        break
os.environ["CUDA_VISIBLE_DEVICES"] = _gpu
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import torch  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default="/monster/data/model/Laguna-S-2.1-GPTQ-FIXED")
    p.add_argument("--backend", default="GPTQ_MARLIN")
    p.add_argument("--gpu", type=str, default="0")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--attn-implementation", default="flash_attention_2")
    return p.parse_args()


PROMPTS = [
    "What is the capital of France?",
    "What is 7 times 8?",
    "Who wrote the play Hamlet?",
    "What is the largest planet in our solar system?",
    "Translate 'Hello' to Spanish.",
    "What is the boiling point of water in Celsius?",
    "Name a primary color.",
    "How many continents are there on Earth?",
    "What gas do plants absorb from the atmosphere?",
    "What is the square root of 64?",
]


def main():
    args = _parse_args()
    torch.manual_seed(args.seed)

    from gptqmodel import BACKEND, GPTQModel
    from gptqmodel.utils.moe_dispatch import (
        enable_grouped_dispatch_for_model,
        register_linear_loop_experts,
    )

    device_name = torch.cuda.get_device_name(torch.cuda.current_device())
    print(f"Visible CUDA device: {torch.cuda.current_device()} -> {device_name}")

    print("Loading model...")
    load_kwargs = {
        "trust_remote_code": True,
        "device": "cuda:0",
        "attn_implementation": args.attn_implementation,
    }
    backend = getattr(BACKEND, args.backend.upper(), BACKEND.GPTQ_MARLIN)
    model = GPTQModel.load(args.model_path, backend=backend, **load_kwargs)
    print("Model loaded.")

    attn_impl = getattr(model.config, "_attn_implementation", None) or getattr(model.config, "attn_implementation", None)
    print(f"Attention implementation: {attn_impl}")

    # Make generation deterministic and avoid speculative-decoding side effects.
    gen_cfg = getattr(model.model, "generation_config", None) or getattr(model, "generation_config", None)
    if gen_cfg is not None:
        gen_cfg.do_sample = False
        if getattr(gen_cfg, "speculative_config", None) is not None:
            gen_cfg.speculative_config = None
        print("Generation config: do_sample=False, speculative disabled.")

    print("\nFusing model...")
    counts = model.fuse(qkv=True, gate_up=True, gate_up_activation=True)
    print(f"  Fusion counts: {counts}")

    print("\nEnabling grouped MoE dispatch...")
    registered = register_linear_loop_experts()
    flagged = enable_grouped_dispatch_for_model(model.model)
    print(f"  Registered linear_loop experts: {registered}")
    print(f"  Flagged expert modules: {flagged}")

    # Warm-up forward to trigger backend selection and inspect one experts module.
    print("\nWarm-up forward + backend probe...")
    vocab_size = getattr(model.config, "vocab_size", 32000)
    with torch.inference_mode():
        _ = model(torch.randint(0, vocab_size, (1, 4), device="cuda:0"))

    backend_seen = None
    for name, module in model.model.named_modules():
        if getattr(module, "num_experts", 0) > 0 and "0" in module._modules:
            backend_seen = getattr(module, "_moe_dispatch_backend", None)
            print(f"  Experts module '{name}' -> _moe_dispatch_backend={backend_seen}")
            break

    tokenizer = model.tokenizer
    print("\n=== Sanity generations (max_new_tokens=%d, do_sample=False) ===" % args.max_new_tokens)
    results = []
    for idx, prompt in enumerate(PROMPTS, 1):
        messages = [{"role": "user", "content": prompt}]
        prompt_text = None
        try:
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            if not isinstance(prompt_text, str):
                raise ValueError(f"chat template returned {type(prompt_text)}")
        except Exception as e:
            print(f"Prompt {idx}: chat_template failed ({e}), falling back to raw prompt")
            prompt_text = prompt + "\n"

        input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to("cuda:0")

        start = time.perf_counter()
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        elapsed = time.perf_counter() - start
        generated = output_ids[0, input_ids.shape[1]:]
        text = tokenizer.decode(generated, skip_special_tokens=True)
        results.append({"prompt": prompt, "text": text, "time": elapsed})
        print(f"\n[{idx}] Prompt: {prompt}")
        print(f"     Time: {elapsed:.3f}s")
        print(f"     Output: {text[:500].replace(chr(10), ' ')}")

    print("\n=== Summary ===")
    print(f"Attention: {attn_impl}")
    print(f"Fusion: {counts}")
    print(f"Grouped MoE backend: {backend_seen}")
    print(f"Generated {len(results)} prompts, max {args.max_new_tokens} tokens each.")


if __name__ == "__main__":
    main()
