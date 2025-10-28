# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""
MiniMax-M2 Hugging Face checkpoint sanity check with streaming output.

Usage:
    python test_minimax_m2_hf.py \
        --model-path /monster/data/model/MiniMax-M2-bf16 \
        --question "How many letter A are there in the word Alphabet? Reply with the number only."
"""

from __future__ import annotations

import argparse
import threading
from pathlib import Path

import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

# from gptqmodel.hf_minimax_m2.modeling_minimax_m2 import (
#     MiniMaxAttention,
#     MiniMaxDecoderLayer,
#     MiniMaxForCausalLM,
#     MiniMaxMLP,
#     MiniMaxM2Attention,
#     MiniMaxM2DecoderLayer,
#     MiniMaxM2ForCausalLM,
#     MiniMaxM2MLP,
#     MiniMaxM2RMSNorm,
#     MiniMaxM2SparseMoeBlock,
#     MiniMaxRMSNorm,
#     MiniMaxSparseMoeBlock,
# )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MiniMax-M2 HF checkpoint smoke test.")
    parser.add_argument(
        "--model-path",
        type=str,
        default="/monster/data/model/MiniMax-M2-bf16",
        help="Path to the MiniMax-M2 Hugging Face checkpoint directory.",
    )
    parser.add_argument(
        "--question",
        type=str,
        default="How many letter A are there in the word Alphabet? Reply with the number only.",
        help="User question to send through the chat template.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum number of new tokens to sample from the model.",
    )
    return parser.parse_args()


def build_prompt(tokenizer: AutoTokenizer, question: str) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


# def assert_module_types(model: MiniMaxM2ForCausalLM) -> None:
#     causal_lm_types = (MiniMaxM2ForCausalLM, MiniMaxForCausalLM)
#     decoder_layer_types = (MiniMaxM2DecoderLayer, MiniMaxDecoderLayer)
#     attention_types = (MiniMaxM2Attention, MiniMaxAttention)
#     moe_block_types = (MiniMaxM2SparseMoeBlock, MiniMaxSparseMoeBlock)
#     norm_types = (MiniMaxM2RMSNorm, MiniMaxRMSNorm)
#     mlp_types = (MiniMaxM2MLP, MiniMaxMLP)
#
#     assert isinstance(
#         model, causal_lm_types
#     ), f"Expected MiniMaxM2ForCausalLM/MiniMaxForCausalLM, received {type(model).__name__}"
#
#     decoder = getattr(model, "model", None)
#     assert decoder is not None, "Model is missing the `model` attribute with decoder layers."
#
#     for layer_idx, layer in enumerate(decoder.layers):
#         assert isinstance(
#             layer, decoder_layer_types
#         ), f"Layer {layer_idx}: expected MiniMax(M2)DecoderLayer, got {type(layer).__name__}"
#         assert isinstance(
#             layer.self_attn, attention_types
#         ), f"Layer {layer_idx}: unexpected self_attn type {type(layer.self_attn).__name__}"
#         assert isinstance(
#             layer.block_sparse_moe, moe_block_types
#         ), f"Layer {layer_idx}: unexpected MoE block type {type(layer.block_sparse_moe).__name__}"
#         assert isinstance(
#             layer.input_layernorm, norm_types
#         ), f"Layer {layer_idx}: unexpected input_layernorm type {type(layer.input_layernorm).__name__}"
#         assert isinstance(
#             layer.post_attention_layernorm, norm_types
#         ), f"Layer {layer_idx}: unexpected post_attention_layernorm type {type(layer.post_attention_layernorm).__name__}"
#
#         moe_block = layer.block_sparse_moe
#         assert isinstance(
#             moe_block.experts, nn.ModuleList
#         ), f"Layer {layer_idx}: expected experts to be a ModuleList, got {type(moe_block.experts).__name__}"
#         for expert_idx, expert in enumerate(moe_block.experts):
#             assert isinstance(
#                 expert, mlp_types
#             ), f"Layer {layer_idx} expert {expert_idx}: expected MiniMax(M2)MLP, got {type(expert).__name__}"
#

def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path).expanduser().resolve()

    print(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    print(f"Loading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype="bfloat16",
        device_map="auto",
        trust_remote_code=True,
    )

    # Uncomment to enforce module type checks.
    # print("Validating module types...")
    # assert_module_types(model)

    prompt = build_prompt(tokenizer, args.question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    print("Running generation (streaming)...\n")
    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=False)
    eos_ids = model.generation_config.eos_token_id
    if eos_ids is None:
        eos_ids = []
    elif isinstance(eos_ids, int):
        eos_ids = [eos_ids]
    think_end_id = tokenizer.convert_tokens_to_ids("</think>")
    if think_end_id is not None and think_end_id not in eos_ids:
        eos_ids = eos_ids + [think_end_id]

    generation_kwargs = dict(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        streamer=streamer,
        eos_token_id=eos_ids if eos_ids else None,
    )

    generation_thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    generation_thread.start()

    completion = []
    first_chunk = True
    seen_end_reasoning = False
    for text in streamer:
        if first_chunk:
            print("<think>", end="", flush=True)
            completion.append("<think>")
            first_chunk = False
        print(text, end="", flush=True)
        completion.append(text)
        if "</think>" in text:
            seen_end_reasoning = True

    generation_thread.join()
    print("\n\n=== Completed Response ===")
    final_text = "".join(completion).strip()
    print(final_text or "<empty response>")
    if not seen_end_reasoning:
        print("\n[warning] No </think> token detected in streamed output.", flush=True)


if __name__ == "__main__":
    main()
