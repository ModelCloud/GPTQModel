# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Exact native W4A checkpoint re-export, including shards and public save()."""

import json

import pytest
import torch
from safetensors.torch import load_file, save_file
from transformers import LlamaConfig, LlamaForCausalLM

from gptqmodel.models.writer import _load_native_w4a_for_save
from gptqmodel.nn_modules.qlinear.torch import TorchLinear


def _shell(tied=False):
    config = LlamaConfig(vocab_size=32, hidden_size=128, intermediate_size=256,
                         num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
                         tie_word_embeddings=tied)
    core = LlamaForCausalLM(config).bfloat16()
    for layer in core.model.layers:
        for parent in (layer.self_attn, layer.mlp):
            for name, dense in list(parent.named_children()):
                if isinstance(dense, torch.nn.Linear):
                    module = TorchLinear(bits=4, group_size=128, sym=True, desc_act=False,
                                          in_features=dense.in_features, out_features=dense.out_features,
                                          bias=False)
                    module.scales.fill_(.1234)
                    module.qzeros.fill_(0x77777777)
                    setattr(parent, name, module)
    return core


def _source(core, mode="w4a_nvfp4"):
    state = {key: value.clone() for key, value in core.state_dict().items()}
    for name, module in core.named_modules():
        if mode == "w4a_nvfp4" and isinstance(module, TorchLinear):
            state[f"{name}.activation_global_scale_bits"] = torch.tensor(.01234567).view(torch.int32)
    return state


def _write(tmp_path, state, sharded=False):
    if not sharded:
        path = tmp_path / "model.safetensors"
        save_file(state, str(path))
        return str(path)
    midpoint = len(state) // 2
    names = list(state)
    mapping = {}
    for shard, keys in enumerate((names[:midpoint], names[midpoint:])):
        filename = f"model-{shard:05d}.safetensors"
        save_file({key: state[key] for key in keys}, str(tmp_path / filename))
        mapping.update(dict.fromkeys(keys, filename))
    path = tmp_path / "model.safetensors.index.json"
    path.write_text(json.dumps({"weight_map": mapping}))
    return str(path)


@pytest.mark.parametrize("sharded", [False, True])
@pytest.mark.parametrize("mode", ["w4afp8", "w4a_nvfp4"])
def test_native_writer_preserves_all_values_dtypes_and_scale_bits(tmp_path, sharded, mode):
    source = _source(_shell(), mode)
    assert {value.dtype for value in source.values()} >= {torch.bfloat16, torch.float16, torch.int32}
    path = _write(tmp_path, source, sharded)
    model = _shell().half()  # Deliberately different from the source dtype.
    _load_native_w4a_for_save(model, path, activation_mode=mode)
    restored = model.state_dict()
    assert set(restored) == set(source)
    for key, value in source.items():
        torch.testing.assert_close(restored[key], value, rtol=0, atol=0)
    assert not any("_weight_both" in key or "_quantizer" in key for key in restored)


@pytest.mark.parametrize("mode", ["w4afp8", "w4a_nvfp4"])
def test_native_writer_restores_omitted_tied_head(tmp_path, mode):
    source = _source(_shell(tied=True), mode)
    del source["lm_head.weight"]
    path = _write(tmp_path, source, sharded=True)
    model = _shell(tied=True)
    _load_native_w4a_for_save(model, path, activation_mode=mode)
    assert model.lm_head.weight is model.model.embed_tokens.weight
    torch.testing.assert_close(model.lm_head.weight, source["model.embed_tokens.weight"], rtol=0, atol=0)


@pytest.mark.parametrize("mode,suffix", [("w4afp8", "qweight"), ("w4a_nvfp4", "qweight"),
                                        ("w4a_nvfp4", "activation_global_scale_bits")])
def test_native_writer_rejects_missing_tensors(tmp_path, mode, suffix):
    source = _source(_shell(), mode)
    del source[f"model.layers.0.self_attn.q_proj.{suffix}"]
    with pytest.raises(ValueError, match="Missing native W4A"):
        _load_native_w4a_for_save(_shell(), _write(tmp_path, source), activation_mode=mode)


@pytest.mark.parametrize("mode", ["w4afp8", "w4a_nvfp4"])
def test_native_writer_rejects_unexpected_runtime_tensor(tmp_path, mode):
    source = _source(_shell(), mode)
    source["model.layers.0.self_attn.q_proj._weight_both"] = torch.ones(1)
    with pytest.raises(ValueError, match="Unrecognized native W4A"):
        _load_native_w4a_for_save(_shell(), _write(tmp_path, source), activation_mode=mode)


@pytest.mark.parametrize("mode", ["w4afp8", "w4a_nvfp4"])
def test_native_writer_rejects_duplicate_shard_entries(tmp_path, mode):
    source = _source(_shell(), mode)
    path = _write(tmp_path, source, sharded=True)
    save_file(source, str(tmp_path / "model-00001.safetensors"))
    with pytest.raises(ValueError, match="Duplicate native W4A"):
        _load_native_w4a_for_save(_shell(), path, activation_mode=mode)


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (12, 1),
                    reason="GB10 required")
@pytest.mark.parametrize("mode", ["w4afp8", "w4a_nvfp4"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("sharded", [False, True])
def test_public_load_save_preserves_native_w4a_tensors(tmp_path, mode, dtype, sharded):
    from gptqmodel import GPTQModel
    from gptqmodel.quantization.config import QuantizeConfig
    from tests.models.test_w4a_tiny_llama_lifecycle import _tokenizer

    source, saved = tmp_path / "source", tmp_path / "saved"
    source.mkdir()
    core = _shell().to(dtype=dtype)
    state = (_source(core) if mode == "w4a_nvfp4"
             else {name: value.clone() for name, value in core.state_dict().items()})
    if dtype == torch.bfloat16:
        # Both are finite BF16 values, but forcing an FP16 serialization shell
        # overflows one and underflows the other. No inference uses this fixture.
        state["model.embed_tokens.weight"][0, :2] = torch.tensor([1e5, 1e-12], dtype=dtype)
    _write(source, state, sharded=sharded)
    core.config.save_pretrained(source)
    _tokenizer().save_pretrained(source)
    QuantizeConfig(bits=4, group_size=128, desc_act=False, sym=True,
                   activation=mode).save_pretrained(str(source))
    wrapper = GPTQModel.load(str(source), device="cuda", dtype=dtype)
    wrapper.save(str(saved))
    actual = load_file(str(saved / "model.safetensors"))
    assert set(actual) == set(state)
    for name, value in state.items():
        torch.testing.assert_close(actual[name], value, rtol=0, atol=0, msg=name)
