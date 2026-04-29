# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import copy
import io
import os
from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as hf_logging

from gptqmodel.nn_modules.qlinear.gguf import GGUFTorchLinear


MODEL_ID = Path("/monster/data/model/Llama-3.2-1B-Instruct")
PROMPT = "The capital city of France is Paris. The capital city of Germany is"
LAYER0_MODULES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
)
DIRECT_Q4_K_M_LIMITS = {
    "self_attn.q_proj": {
        "weight_mae": 0.0026,
        "weight_max": 0.035,
        "output_mae": 0.022,
        "output_max": 0.21,
    },
    "self_attn.k_proj": {
        "weight_mae": 0.0034,
        "weight_max": 0.031,
        "output_mae": 0.029,
        "output_max": 0.20,
    },
    "self_attn.v_proj": {
        "weight_mae": 0.0008,
        "weight_max": 0.004,
        "output_mae": 0.0065,
        "output_max": 0.032,
    },
    "self_attn.o_proj": {
        "weight_mae": 0.0010,
        "weight_max": 0.020,
        "output_mae": 0.0010,
        "output_max": 0.011,
    },
}


def _error_stats(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    diff = (candidate - reference).abs()
    return {
        "mae": diff.mean().item(),
        "max": diff.max().item(),
    }


def _skip_unavailable_environment(dtype: torch.dtype) -> torch.device:
    if not MODEL_ID.exists():
        pytest.skip(f"Missing local test model: {MODEL_ID}")
    if not torch.cuda.is_available():
        pytest.skip("Direct GGUF Llama layer test requires CUDA")
    if dtype == torch.float16 and not torch.cuda.is_available():
        pytest.skip("float16 path requires CUDA")
    return torch.device("cuda:0")


def _load_llama(dtype: torch.dtype):
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    hf_logging.disable_progress_bar()
    hf_logging.set_verbosity_error()

    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        model = AutoModelForCausalLM.from_pretrained(
            str(MODEL_ID),
            dtype=dtype,
            device_map="cuda:0",
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_ID), use_fast=True)

    return model, tokenizer


def _capture_layer0_module_io(model, tokenizer) -> dict[str, dict[str, torch.Tensor | None]]:
    layer0 = model.model.layers[0]
    captured: dict[str, dict[str, torch.Tensor | None]] = {}
    handles = []

    for module_name in LAYER0_MODULES:
        module = dict(layer0.named_modules())[module_name]

        def _hook(mod, args, out, *, module_name=module_name):
            captured[module_name] = {
                "input": args[0].detach().cpu(),
                "output": out.detach().cpu(),
                "weight": mod.weight.detach().cpu(),
                "bias": None if mod.bias is None else mod.bias.detach().cpu(),
            }

        handles.append(module.register_forward_hook(_hook))

    device = next(model.parameters()).device
    inputs = tokenizer(PROMPT, return_tensors="pt").to(device)
    with torch.inference_mode():
        model(**inputs)

    for handle in handles:
        handle.remove()

    return captured


def _build_direct_gguf_module(native_module: torch.nn.Linear) -> GGUFTorchLinear:
    module = GGUFTorchLinear(
        bits="q4_k_m",
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=native_module.in_features,
        out_features=native_module.out_features,
        bias=native_module.bias is not None,
        register_buffers=False,
    )
    module.pack_original(linear=native_module, scales=None, zeros=None, g_idx=None)
    module.post_init()
    return module


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
def test_direct_gguf_q4_k_m_llama3_2_layer0_attention_stays_close_to_native(dtype: torch.dtype):
    device = _skip_unavailable_environment(dtype)
    model, tokenizer = _load_llama(dtype)

    try:
        captured = _capture_layer0_module_io(model, tokenizer)
        layer0 = model.model.layers[0]

        for module_name in LAYER0_MODULES:
            native_module = copy.deepcopy(dict(layer0.named_modules())[module_name]).to(device=device, dtype=dtype).eval()
            gguf_module = _build_direct_gguf_module(native_module).to(device).eval()
            record = captured[module_name]

            module_input = record["input"].to(device=device, dtype=dtype)
            with torch.inference_mode():
                gguf_output = gguf_module(module_input).detach().cpu()

            dequant_weight = gguf_module.dequantize_weight().T.detach().cpu().to(record["weight"].dtype)
            weight_stats = _error_stats(record["weight"].to(torch.float32), dequant_weight.to(torch.float32))
            output_stats = _error_stats(record["output"].to(torch.float32), gguf_output.to(torch.float32))
            limits = DIRECT_Q4_K_M_LIMITS[module_name]

            assert weight_stats["mae"] < limits["weight_mae"], f"{dtype}: {module_name} weight MAE {weight_stats['mae']:.6f}"
            assert weight_stats["max"] < limits["weight_max"], f"{dtype}: {module_name} weight max {weight_stats['max']:.6f}"
            assert output_stats["mae"] < limits["output_mae"], f"{dtype}: {module_name} output MAE {output_stats['mae']:.6f}"
            assert output_stats["max"] < limits["output_max"], f"{dtype}: {module_name} output max {output_stats['max']:.6f}"
    finally:
        del model
        torch.cuda.empty_cache()
