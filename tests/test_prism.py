# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM
from transformers.integrations import PrismEmbedding, PrismLinear

from gptqmodel import GPTQModel


GROUP_SIZE = 128


def _pack_prism_weight(weight: torch.Tensor, group_size: int = GROUP_SIZE):
    rows, cols = weight.shape
    if cols % group_size != 0:
        raise ValueError(f"Expected columns divisible by {group_size}, got shape {tuple(weight.shape)}")

    num_groups = cols // group_size
    blocks = weight.reshape(rows, num_groups, group_size)
    mins = blocks.min(dim=-1, keepdim=True).values
    maxs = blocks.max(dim=-1, keepdim=True).values
    scales = (maxs - mins).clamp(min=1e-7)
    quantized = ((blocks - mins) / scales).round().clamp(0, 1).to(torch.uint8)

    shifts = torch.arange(32, dtype=torch.int64).view(1, 1, 1, 32)
    packed = ((quantized.reshape(rows, num_groups, group_size // 32, 32).to(torch.int64) << shifts).sum(dim=-1)).to(
        torch.uint32
    )
    return packed.reshape(rows, -1).contiguous(), scales.squeeze(-1).to(weight.dtype), mins.squeeze(-1).to(weight.dtype)


def _build_tiny_prism_checkpoint(model_dir: Path) -> PreTrainedTokenizerFast:
    torch.manual_seed(7)

    config = Qwen3Config(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        tie_word_embeddings=True,
    )
    config.architectures = ["Qwen3ForCausalLM"]
    config.quantization = {"bits": 1, "group_size": GROUP_SIZE}
    config.save_pretrained(model_dir)

    backend = Tokenizer(
        WordLevel(
            {
                "<pad>": 0,
                "<unk>": 1,
                "Hello": 2,
                "world": 3,
            },
            unk_token="<unk>",
        )
    )
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        eos_token="<pad>",
        unk_token="<unk>",
    )
    tokenizer.save_pretrained(model_dir)

    base_model = Qwen3ForCausalLM(config).eval()
    quantized_state = {}

    with torch.no_grad():
        for name, tensor in base_model.state_dict().items():
            if name == "lm_head.weight":
                continue

            if name.endswith(".weight"):
                module_name = name.removesuffix(".weight")
                module = base_model.get_submodule(module_name)
                if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
                    packed, scales, biases = _pack_prism_weight(tensor.detach().to(torch.float16))
                    quantized_state[name] = packed
                    quantized_state[f"{module_name}.scales"] = scales
                    quantized_state[f"{module_name}.biases"] = biases
                    continue

            quantized_state[name] = tensor.detach().clone()

    save_file(quantized_state, str(model_dir / "model.safetensors"))
    return tokenizer


def test_gptqmodel_loads_legacy_prism_checkpoint(tmp_path: Path):
    pytest.importorskip("accelerate")

    model_dir = tmp_path / "prism_qwen3"
    model_dir.mkdir()
    tokenizer = _build_tiny_prism_checkpoint(model_dir)

    target_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    load_dtype = torch.float16 if target_device.startswith("cuda") else torch.float32
    device_map = {"": target_device}

    hf_model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=load_dtype, device_map=device_map).eval()
    gptq_model = GPTQModel.load(str(model_dir), dtype=load_dtype, device_map=device_map)

    assert isinstance(gptq_model.model.model.embed_tokens, PrismEmbedding)
    assert isinstance(gptq_model.model.model.layers[0].self_attn.q_proj, PrismLinear)
    assert isinstance(gptq_model.model.lm_head, PrismLinear)

    inputs = tokenizer("Hello world", return_tensors="pt").to(target_device)
    with torch.inference_mode():
        expected = hf_model(**inputs).logits
        actual = gptq_model.model(**inputs).logits

    torch.testing.assert_close(actual.float(), expected.float(), atol=2e-3, rtol=2e-3)
