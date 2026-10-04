# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for exact native NVFP4 checkpoint re-export, including shards."""

import json

import pytest
import torch
from safetensors.torch import save_file
from transformers import LlamaConfig, LlamaForCausalLM

from gptqmodel.models.writer import _load_native_nvfp4_for_save
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


def _source(core):
    state = {key: value.clone() for key, value in core.state_dict().items()}
    for name, module in core.named_modules():
        if isinstance(module, TorchLinear):
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
def test_native_writer_preserves_all_values_dtypes_and_scale_bits(tmp_path, sharded):
    source = _source(_shell())
    assert {value.dtype for value in source.values()} >= {torch.bfloat16, torch.float16, torch.int32}
    path = _write(tmp_path, source, sharded)
    model = _shell().half()  # Deliberately different from the source dtype.
    _load_native_nvfp4_for_save(model, path)
    restored = model.state_dict()
    assert set(restored) == set(source)
    for key, value in source.items():
        torch.testing.assert_close(restored[key], value, rtol=0, atol=0)
    assert not any("_weight_both" in key or "_quantizer" in key for key in restored)


def test_native_writer_restores_omitted_tied_head(tmp_path):
    source = _source(_shell(tied=True))
    del source["lm_head.weight"]
    path = _write(tmp_path, source, sharded=True)
    model = _shell(tied=True)
    _load_native_nvfp4_for_save(model, path)
    assert model.lm_head.weight is model.model.embed_tokens.weight
    torch.testing.assert_close(model.lm_head.weight, source["model.embed_tokens.weight"], rtol=0, atol=0)


@pytest.mark.parametrize("suffix", ["qweight", "activation_global_scale_bits"])
def test_native_writer_rejects_missing_tensors(tmp_path, suffix):
    source = _source(_shell())
    del source[f"model.layers.0.self_attn.q_proj.{suffix}"]
    with pytest.raises(ValueError, match="Missing native NVFP4"):
        _load_native_nvfp4_for_save(_shell(), _write(tmp_path, source))


def test_native_writer_rejects_unexpected_runtime_tensor(tmp_path):
    source = _source(_shell())
    source["model.layers.0.self_attn.q_proj._weight_both"] = torch.ones(1)
    with pytest.raises(ValueError, match="Unrecognized native NVFP4"):
        _load_native_nvfp4_for_save(_shell(), _write(tmp_path, source))


def test_native_writer_rejects_duplicate_shard_entries(tmp_path):
    source = _source(_shell())
    path = _write(tmp_path, source, sharded=True)
    save_file(source, str(tmp_path / "model-00001.safetensors"))
    with pytest.raises(ValueError, match="Duplicate native NVFP4"):
        _load_native_nvfp4_for_save(_shell(), path)
