# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Explicit real-checkpoint gate; not a unit test or a checkpoint downloader.

Run from the repository root with PyTorch 2.15 and a local pinned checkpoint.
Tokenizers is used only for byte/token IO; all three model paths use pure Torch.
"""

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from tokenizers import Tokenizer

from gptqmodel_compiler import compile_program, llama32_1b
from gptqmodel_compiler.checkpoint import load_safetensors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dtype", choices=["float32"], default="float32")
    parser.add_argument("--new-tokens", type=int, default=6)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--artifacts", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    config = json.loads((args.checkpoint / "config.json").read_text())
    required = {
        "hidden_size": 2048,
        "intermediate_size": 8192,
        "num_hidden_layers": 16,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 64,
        "vocab_size": 128256,
        "tie_word_embeddings": True,
        "rms_norm_eps": 1e-5,
        "rope_theta": 500000.0,
    }
    for key, value in required.items():
        if config.get(key) != value:
            raise ValueError(f"checkpoint config mismatch: {key}")
    for key, value in {
        "factor": 32.0,
        "high_freq_factor": 4.0,
        "low_freq_factor": 1.0,
        "original_max_position_embeddings": 8192,
        "rope_type": "llama3",
    }.items():
        if config["rope_scaling"].get(key) != value:
            raise ValueError(f"RoPE config mismatch: {key}")
    spec = importlib.util.spec_from_file_location(
        "independent_llama_equations", Path(__file__).with_name("reference.py")
    )
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    program = llama32_1b(dtype=args.dtype, capacity=256)
    tokenizer = Tokenizer.from_file(str(args.checkpoint / "tokenizer.json"))
    prompt = (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        "What is 2 + 2? Answer with a single number.<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    tokens = torch.tensor([tokenizer.encode(prompt, add_special_tokens=False).ids])
    eager_artifact = compile_program(program)
    optimized_artifact = compile_program(program, target="optimized")
    if args.artifacts:
        eager_artifact.write(args.artifacts / "eager")
        optimized_artifact.write(args.artifacts / "optimized")
    print("Loading generated eager model", file=sys.stderr, flush=True)
    eager = load_safetensors(eager_artifact.instantiate(), args.checkpoint)
    raw = {p.name: eager.parameter_view(p.name).detach() for p in program.parameters}
    a = next(op.attributes for b in program.blocks for op in b.body if op.kind == "rope")
    oracle_config = {"layers": 16, "eps": 1e-5, **a}
    print("Independent equation oracle", file=sys.stderr, flush=True)
    expected, expected_cache = reference.llama_reference(tokens, raw, oracle_config)
    actual, eager_state = eager(tokens)
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)
    eager_error = (actual.float() - expected.float()).abs().max().item()
    print("Loading generated optimized model", file=sys.stderr, flush=True)
    optimized = load_safetensors(optimized_artifact.instantiate(), args.checkpoint)
    print("Compiling and executing whole-model Inductor prefill", file=sys.stderr, flush=True)
    started = time.monotonic()
    compiled, compiled_state = optimized(tokens)
    first_compile_seconds = time.monotonic() - started
    torch.testing.assert_close(compiled, expected, atol=2e-3, rtol=2e-3)
    compiled_error = (compiled.float() - expected.float()).abs().max().item()
    for ac, bc, ec in zip(eager_state.cache, compiled_state.cache, expected_cache):
        torch.testing.assert_close(ac, ec, atol=2e-3, rtol=2e-3)
        torch.testing.assert_close(bc, ec, atol=2e-3, rtol=2e-3)
    ids = []
    for step in range(args.new_tokens):
        chosen = expected[:, -1].argmax(-1, keepdim=True)
        assert torch.equal(actual[:, -1].argmax(-1, keepdim=True), chosen)
        assert torch.equal(compiled[:, -1].argmax(-1, keepdim=True), chosen)
        ids.append(chosen.item())
        if step + 1 == args.new_tokens:
            break
        offset = eager_state.length
        expected, expected_cache = reference.llama_reference(
            chosen, raw, oracle_config, past=expected_cache, offset=offset
        )
        eager_state.phase = compiled_state.phase = "decode"
        actual, eager_state = eager(chosen, eager_state)
        compiled, compiled_state = optimized(chosen, compiled_state)
        torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)
        torch.testing.assert_close(compiled, expected, atol=2e-3, rtol=2e-3)
        eager_error = max(eager_error, (actual.float() - expected.float()).abs().max().item())
        compiled_error = max(compiled_error, (compiled.float() - expected.float()).abs().max().item())
    from torch._dynamo import explain

    new_state = optimized.new_request()
    graph = explain(optimized._execute)(tokens, torch.arange(tokens.shape[1]), new_state.cache)
    assert graph.graph_count == 1 and graph.graph_break_count == 0
    checkpoint_hash = hashlib.file_digest((args.checkpoint / "model.safetensors").open("rb"), "sha256").hexdigest()
    report = {
        "passed": True,
        "utc": datetime.now(timezone.utc).isoformat(),
        "torch": torch.__version__,
        "torch_commit": torch.version.git_version,
        "python": sys.version,
        "device": "cpu",
        "threads": args.threads,
        "dtype": args.dtype,
        "program": program.fingerprint,
        "generated_source_sha256": {
            "eager": hashlib.sha256(eager_artifact.source.encode()).hexdigest(),
            "optimized": hashlib.sha256(optimized_artifact.source.encode()).hexdigest(),
        },
        "independent_reference_sha256": hashlib.sha256(
            Path(__file__).with_name("reference.py").read_bytes()
        ).hexdigest(),
        "checkpoint": "unsloth/Llama-3.2-1B-Instruct",
        "checkpoint_revision": "5a8abab4a5d6f164389b1079fb721cfab8d7126c",
        "checkpoint_sha256": checkpoint_hash,
        "prompt": prompt,
        "prompt_token_ids": tokens[0].tolist(),
        "generated_token_ids": ids,
        "generated_text": tokenizer.decode(ids, skip_special_tokens=False),
        "parameters": sum(p.numel() for p in eager.parameters()),
        "retained_native_weight_bytes_eager": sum(p.numel() * p.element_size() for p in eager.parameters()),
        "retained_native_weight_bytes_optimized": sum(p.numel() * p.element_size() for p in optimized.parameters()),
        "additional_expanded_weight_cache_bytes": 0,
        "eager_linear_modules": len(eager.projections),
        "optimized_linear_modules": len(optimized.projections),
        "max_abs_eager_vs_independent": eager_error,
        "max_abs_inductor_vs_independent": compiled_error,
        "atol": 2e-3,
        "rtol": 2e-3,
        "graph_count": graph.graph_count,
        "graph_break_count": graph.graph_break_count,
        "first_compile_and_prefill_seconds": first_compile_seconds,
        "transformers_imports": [m for m in sys.modules if m == "transformers" or m.startswith("transformers.")],
        "unqualified": [
            "optimized FP16/BF16 (compiler rejects these profiles)",
            "GPU execution",
            "CUDA graph capture",
            "packed quantized parameter lowerings",
            "padded/ragged batches",
            "other architecture builders",
        ],
    }
    assert not report["transformers_imports"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
