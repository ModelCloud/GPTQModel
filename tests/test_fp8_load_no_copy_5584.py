# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""A block-FP8 load must not copy the non-FP8 tensors (#5584).

#4999 made the legacy-layout key converter read shards lazily, but a block-FP8
checkpoint still entered `_remap_safetensors_checkpoint_keys`: the copy existed
only to hide the packed FP8 weights from Accelerate. Every load therefore wrote
a copy of every dense tensor (embed, lm_head, norms, scales) into the
checkpoint directory (2.1 GB and 19 s of a 22 s Llama-3.1-8B load). A killed
load left a partial `gptqmodel-key-remap-*.safetensors` behind that
`is_embeddings_module_quantized` then tried to read.

These cases pin the contract:

* a successful FP8 load calls no remap copy and writes no checkpoint bytes,
* the checkpoint directory is byte-identical afterwards,
* the loaded FP8 payload and the forward match the copy fallback,
* a failed load starts no copy and leaves no partial file,
* embedding detection ignores loader scratch, including a stray partial file,
* a sharded FP8 checkpoint takes the same path.
"""

import json
import os

import pytest
import torch
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import LlamaConfig, LlamaForCausalLM

import gptqmodel.utils.model as gm
from gptqmodel import BACKEND, GPTQModel
from gptqmodel.nn_modules.qlinear.fp8 import TorchFP8Linear


_BLOCK = (128, 128)
_VOCAB = 1024
_LAYER_MODULES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


def _build_tokenizer(model_dir):
    vocab = {"<unk>": 0, "<s>": 1, "</s>": 2}
    vocab.update({f"t{i}": i for i in range(3, _VOCAB)})
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(os.path.join(model_dir, "tokenizer.json"))
    with open(os.path.join(model_dir, "tokenizer_config.json"), "w") as handle:
        json.dump(
            {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "unk_token": "<unk>",
                "bos_token": "<s>",
                "eos_token": "</s>",
                "pad_token": "<unk>",
                "model_max_length": 128,
                "clean_up_tokenization_spaces": False,
            },
            handle,
            indent=2,
        )


def _quantize_block_fp8(weight):
    """Encode one matrix like transformers' FineGrainedFP8 (128x128 blocks).

    ``weight_scale_inv`` is the dequantization multiplier (block abs-max /
    FP8 max), which is what a Hugging Face FP8 checkpoint stores and what the
    loader reads back.
    """

    rows, cols = weight.shape
    block_rows, block_cols = _BLOCK
    assert rows % block_rows == 0 and cols % block_cols == 0, (rows, cols)
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    blocks = weight.detach().to(torch.float32).reshape(
        rows // block_rows, block_rows, cols // block_cols, block_cols
    )
    abs_max = blocks.abs().amax(dim=(1, 3))
    scale = torch.where(
        abs_max > 0, abs_max / fp8_max, torch.ones_like(abs_max)
    )
    quantized = (
        torch.clamp(
            blocks / scale.unsqueeze(1).unsqueeze(3), min=-fp8_max, max=fp8_max
        )
        .to(torch.float8_e4m3fn)
        .reshape(rows, cols)
    )
    return quantized.contiguous(), scale.to(torch.float32).contiguous()


def _build_fp8_checkpoint(root, *, shards=None):
    """Write a tiny block-FP8 Llama checkpoint and return its directory."""

    config = LlamaConfig(
        vocab_size=_VOCAB,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        max_position_embeddings=64,
        tie_word_embeddings=False,
    )
    torch.manual_seed(0)
    model = LlamaForCausalLM(config)
    state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }
    for layer_index, layer in enumerate(model.model.layers):
        for module_name in _LAYER_MODULES:
            prefix = f"model.layers.{layer_index}.{module_name}"
            module = model.get_submodule(prefix)
            qweight, scale_inv = _quantize_block_fp8(module.weight)
            state[f"{prefix}.weight"] = qweight
            state[f"{prefix}.weight_scale_inv"] = scale_inv

    root.mkdir(parents=True, exist_ok=True)
    for name in list(state):
        if state[name].is_floating_point() and state[name].dtype == torch.float32:
            state[name] = state[name].to(torch.bfloat16)

    if shards is None:
        save_file(state, os.path.join(root, "model.safetensors"))
    else:
        names = sorted(state)
        weight_map = {}
        chunks = [names[i::shards] for i in range(shards)]
        for index, chunk in enumerate(chunks, start=1):
            filename = f"model-{index:05d}-of-{shards:05d}.safetensors"
            save_file(
                {name: state[name] for name in chunk},
                os.path.join(root, filename),
            )
            for name in chunk:
                weight_map[name] = filename
        index_payload = {
            "metadata": {"total_size": sum(t.numel() * t.element_size() for t in state.values())},
            "weight_map": weight_map,
        }
        with open(os.path.join(root, "model.safetensors.index.json"), "w") as handle:
            json.dump(index_payload, handle, indent=2)

    payload = config.to_dict()
    payload["dtype"] = "bfloat16"
    payload["quantization_config"] = {
        "activation_scheme": "dynamic",
        "dequantize": False,
        "expert_dtype": "fp8",
        "modules_to_convert": None,
        "modules_to_not_convert": None,
        "quant_method": "fp8",
        "scale_fmt": "float",
        "weight_block_size": list(_BLOCK),
    }
    with open(os.path.join(root, "config.json"), "w") as handle:
        json.dump(payload, handle, indent=2)
    _build_tokenizer(root)
    return root


def _snapshot(directory):
    out = {}
    for name in sorted(os.listdir(directory)):
        path = os.path.join(directory, name)
        if not os.path.isfile(path):
            continue
        with open(path, "rb") as handle:
            out[name] = (
                os.path.getsize(path),
                os.stat(path).st_mtime_ns,
                handle.read(),
            )
    return out


def _remap_names(directory):
    return sorted(
        name
        for name in os.listdir(directory)
        if name.startswith("gptqmodel-key-remap-")
    )


def _load(directory):
    return GPTQModel.load(
        str(directory),
        backend=BACKEND.TORCH,
        device="cpu",
        dtype=torch.float16,
    )


def _fp8_modules(model):
    return [
        (name, module)
        for name, module in model.model.named_modules()
        if isinstance(module, TorchFP8Linear)
    ]


def _assert_same(left, right):
    if left.dtype == torch.float8_e4m3fn:
        assert torch.equal(left.view(torch.uint8), right.view(torch.uint8))
    else:
        assert torch.equal(left, right)


def _forward(model):
    input_ids = torch.tensor([[1, 3, 4, 5]])
    with torch.inference_mode():
        logits = model.model(input_ids=input_ids, use_cache=False).logits
    assert logits.shape == (1, 4, _VOCAB)
    assert torch.isfinite(logits).all()
    return logits


def test_fp8_load_writes_no_remap_copy(tmp_path, monkeypatch):
    directory = _build_fp8_checkpoint(tmp_path / "fp8")
    before = _snapshot(directory)

    calls = []
    original = gm._remap_safetensors_checkpoint_keys

    def record(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append(result)
        return result

    writes = []
    original_write = gm._write_safetensors_with_converted_header

    def record_write(*args, **kwargs):
        result = original_write(*args, **kwargs)
        writes.append(result)
        return result

    monkeypatch.setattr(gm, "_remap_safetensors_checkpoint_keys", record)
    monkeypatch.setattr(gm, "_write_safetensors_with_converted_header", record_write)

    model = _load(directory)

    assert calls == [], "an FP8 load must not enter the remap copy path"
    assert writes == [], "an FP8 load must not write a converted safetensors copy"
    assert _remap_names(directory) == []
    assert _snapshot(directory) == before

    modules = _fp8_modules(model)
    assert len(modules) == 14
    source = load_file(os.path.join(str(directory), "model.safetensors"))
    state = model.model.state_dict()
    for name, _module in modules:
        assert state[f"{name}.weight"].dtype == torch.float8_e4m3fn
        _assert_same(state[f"{name}.weight"].cpu(), source[f"{name}.weight"])
        _assert_same(
            state[f"{name}.weight_scale_inv"].cpu(),
            source[f"{name}.weight_scale_inv"],
        )

    _forward(model)


def test_fp8_load_matches_copy_fallback(tmp_path, monkeypatch):
    directory = _build_fp8_checkpoint(tmp_path / "fp8")
    lazy = _load(directory)

    monkeypatch.setattr(gm, "_load_key_converted_safetensors", lambda *args, **kwargs: False)
    copied = _load(directory)

    lazy_state = {name: value.cpu() for name, value in lazy.model.state_dict().items()}
    copied_state = {name: value.cpu() for name, value in copied.model.state_dict().items()}
    assert set(lazy_state) == set(copied_state)
    for name in lazy_state:
        _assert_same(lazy_state[name], copied_state[name])

    assert torch.equal(_forward(lazy).float(), _forward(copied).float())


def test_fp8_load_failure_leaves_no_remap_file(tmp_path, monkeypatch):
    directory = _build_fp8_checkpoint(tmp_path / "fp8")
    before = _snapshot(directory)

    calls = []
    original = gm._remap_safetensors_checkpoint_keys

    def record(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append(result)
        return result

    monkeypatch.setattr(gm, "_remap_safetensors_checkpoint_keys", record)

    original_restore = gm._restore_checkpoint_buffers
    injections = []

    def fail_after_restore(*args, **kwargs):
        original_restore(*args, **kwargs)
        injections.append(True)
        raise RuntimeError("injected failure after checkpoint restore")

    monkeypatch.setattr(gm, "_restore_checkpoint_buffers", fail_after_restore)

    with pytest.raises(RuntimeError, match="injected failure"):
        _load(directory)

    assert injections, "the injected failure must run after a real checkpoint restore"
    assert calls == []
    assert _remap_names(directory) == []
    assert _snapshot(directory) == before


def test_embedding_detection_ignores_remap_scratch(tmp_path, monkeypatch, caplog):
    directory = _build_fp8_checkpoint(tmp_path / "fp8")
    stray_name = "gptqmodel-key-remap-abcd1234.safetensors"
    stray = os.path.join(str(directory), stray_name)
    with open(stray, "wb") as handle:
        handle.write(b"\x00" * 8 + b'{"model.embed_tokens.weight": {"dtype": "BF16", "shape": [1]')

    opened = []
    original_open = gm.safe_open

    def record_open(path, *args, **kwargs):
        opened.append(os.path.basename(str(path)))
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(gm, "safe_open", record_open)

    with caplog.at_level("WARNING"):
        quantized = gm.is_embeddings_module_quantized(str(directory), "model.embed_tokens", "lm_head")

    assert quantized == (False, False)
    assert stray_name not in opened
    assert "Failed to inspect" not in caplog.text

    opened.clear()
    model = _load(directory)
    assert stray_name not in opened
    assert os.path.isfile(stray)
    _forward(model)


def test_sharded_fp8_load_writes_no_remap_copy(tmp_path, monkeypatch):
    directory = _build_fp8_checkpoint(tmp_path / "fp8", shards=2)
    before = _snapshot(directory)

    calls = []
    original = gm._remap_safetensors_checkpoint_keys

    def record(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append(result)
        return result

    monkeypatch.setattr(gm, "_remap_safetensors_checkpoint_keys", record)

    model = _load(directory)

    assert calls == []
    assert _remap_names(directory) == []
    assert _snapshot(directory) == before
    assert len(_fp8_modules(model)) == 14
    _forward(model)


def test_fp8_load_without_writable_checkpoint_dir(tmp_path, monkeypatch):
    directory = _build_fp8_checkpoint(tmp_path / "fp8")
    before = _snapshot(directory)

    monkeypatch.setattr(gm, "_remap_temp_dir", lambda checkpoint: None)

    model = _load(directory)

    assert _remap_names(directory) == []
    assert _snapshot(directory) == before
    _forward(model)
