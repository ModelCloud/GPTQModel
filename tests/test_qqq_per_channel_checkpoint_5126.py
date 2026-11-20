# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Per-channel QQQ checkpoints must keep every empty ``s_group`` (#5126).

``s_group`` is ``torch.empty(0)`` for a layer whose group spans the whole input
(``group_size == in_features``). Every zero-element tensor reports storage
pointer 0, so the save path used to de-duplicate all of them as "shared
storage" and drop all but one. The fail-closed loader from #5032 then rejected
the file. These tests pin both halves: the writer keeps each named empty
buffer, and the loader still accepts a legacy file that omitted it.
"""

import json
import os

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


def _tiny_llama(path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
        pad_token_id=0,
    )
    config._attn_implementation = "eager"
    LlamaForCausalLM(config).save_pretrained(path)
    backend = Tokenizer(WordLevel({str(i): i for i in range(256)}, unk_token="0"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="0", pad_token="0"
    )
    tokenizer.save_pretrained(path)
    return path


def _quantize_per_channel(source, out, device):
    from gptqmodel import CalibrationConfig, GPTQModel
    from gptqmodel.quantization.config import QQQConfig

    model = GPTQModel.load(
        str(source),
        quantize_config=QQQConfig(
            bits=4,
            group_size=-1,
            desc_act=False,
            offload_to_disk=False,
            device=device,
        ),
        dtype=torch.float16,
    )
    model.quantize(
        CalibrationConfig(
            dataset=["the quick brown fox jumps over the lazy dog " * 4 for _ in range(4)],
            batch_size=1,
        )
    )
    model.save(str(out))
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def _s_group_keys(checkpoint):
    with safe_open(os.path.join(checkpoint, "model.safetensors"), framework="pt") as handle:
        return sorted(key for key in handle.keys() if key.endswith(".s_group"))


def _drop_s_group(checkpoint):
    path = os.path.join(checkpoint, "model.safetensors")
    tensors = {key: value for key, value in load_file(path).items() if not key.endswith(".s_group")}
    save_file(tensors, path)
    index = os.path.join(checkpoint, "model.safetensors.index.json")
    if os.path.exists(index):
        with open(index, encoding="utf-8") as handle:
            data = json.load(handle)
        data["weight_map"] = {
            key: value for key, value in data["weight_map"].items() if not key.endswith(".s_group")
        }
        with open(index, "w", encoding="utf-8") as handle:
            json.dump(data, handle)


def _generate(checkpoint, device):
    from gptqmodel import GPTQModel

    model = GPTQModel.load(str(checkpoint), device=device)
    ids = torch.randint(0, 256, (1, 16), device=device)
    output = model.generate(input_ids=ids, max_new_tokens=8, do_sample=False)
    assert tuple(output.shape) == (1, 24)
    return model


def test_per_channel_qqq_writes_every_empty_s_group_and_reloads(tmp_path):
    source = _tiny_llama(tmp_path / "dense")
    checkpoint = _quantize_per_channel(source, tmp_path / "quantized", "cpu")

    keys = _s_group_keys(checkpoint)
    # 2 decoder layers x 7 projections; every per-channel layer needs its own
    # empty entry, not one shared tensor.
    assert len(keys) == 14, keys
    assert len(set(keys)) == 14

    model = _generate(checkpoint, "cpu")
    assert any(type(module).__name__ == "QQQTorchLinear" for module in model.model.modules())


def test_legacy_per_channel_qqq_without_empty_s_group_still_loads(tmp_path):
    source = _tiny_llama(tmp_path / "dense")
    checkpoint = _quantize_per_channel(source, tmp_path / "quantized", "cpu")
    _drop_s_group(checkpoint)
    assert _s_group_keys(checkpoint) == []

    # A grouped layer's non-empty s_group must stay required.
    model = _generate(checkpoint, "cpu")
    assert any(type(module).__name__ == "QQQTorchLinear" for module in model.model.modules())


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_per_channel_qqq_cuda_roundtrip(tmp_path):
    source = _tiny_llama(tmp_path / "dense")
    checkpoint = _quantize_per_channel(source, tmp_path / "quantized", "cuda:0")

    assert len(_s_group_keys(checkpoint)) == 14

    model = _generate(checkpoint, "cuda:0")
    assert any(type(module).__name__ == "QQQLinear" for module in model.model.modules())
