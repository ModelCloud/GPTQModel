# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from gptqmodel import GPTQModel
from gptqmodel.quantization.config import AWQConfig, CalibrationConfig
from gptqmodel.utils.backend import BACKEND
from gptqmodel.utils.model import co_locate_shared_module_aliases


def resolved(device_map, name):
    return device_map[
        max((key for key in device_map if not key or key == name or name.startswith(f"{key}.")), key=len)
    ]


@pytest.mark.parametrize("devices", [("cuda:3", "cuda:7"), ("xpu:2", "xpu:5"), ("cpu", "meta"), (2, 7)])
@pytest.mark.parametrize("variant", ["embedding", "projection", "layer"])
def test_alias_map_keeps_real_physical_owners_together_without_repartitioning_siblings(devices, variant):
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    with torch.device("meta"):
        model = LlamaForCausalLM(config)
    mapping = {
        "model.embed_tokens": devices[0],
        "lm_head": devices[1],
        "model.norm": devices[1],
        **{f"model.layers.{i}": devices[i % 2] for i in range(4)},
    }
    if variant == "embedding":
        model.model.alias = torch.nn.ModuleDict({"word_embeddings": model.model.embed_tokens})
        model.model.separate_alias = torch.nn.Embedding(64, 32, device="meta")
        model.model.separate_alias.weight = model.model.embed_tokens.weight
        mapping["model.alias"] = devices[1]
        mapping["model.separate_alias"] = devices[1]
    elif variant == "projection":
        model.model.layers[1].self_attn.k_proj.weight = model.model.layers[0].self_attn.k_proj.weight
    else:
        model.model.layers[1] = model.model.layers[0]
    previous = dict(mapping)
    shapes = {key: tuple(value.shape) for key, value in model.state_dict().items()}
    result = co_locate_shared_module_aliases(model, mapping)
    assert mapping == previous
    assert {key: tuple(value.shape) for key, value in model.state_dict().items()} == shapes
    groups = {}
    for key, parameter in model.named_parameters(remove_duplicate=False):
        groups.setdefault(id(parameter), []).append(key)
    shared = [names for names in groups.values() if len(names) > 1]
    assert shared
    for names in shared:
        assert {resolved(result, name) for name in names} == {devices[0]}
    for i in [2, 3]:
        for key, _ in model.model.layers[i].named_parameters():
            assert resolved(result, f"model.layers.{i}.{key}") == devices[i % 2]
    if variant == "projection":
        for key in ["self_attn.q_proj.weight", "self_attn.o_proj.weight", "mlp.up_proj.weight"]:
            assert resolved(result, f"model.layers.1.{key}") == devices[1]
    assert co_locate_shared_module_aliases(model, result) == result
    assert co_locate_shared_module_aliases(model, {"": devices[1]}) == {"": devices[1]}


def test_nonalias_map_is_unchanged():
    model = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 8))
    mapping = {"0": "cuda:3", "1": "cuda:7"}
    assert co_locate_shared_module_aliases(model, mapping) == mapping


@pytest.mark.parametrize("alias", [True, False])
@pytest.mark.parametrize("explicit", [False, True])
def test_real_tied_embedding_alias_public_awq_default_and_explicit_reload(tmp_path, monkeypatch, alias, explicit):
    original_init = LlamaForCausalLM.__init__

    def init(self, config):
        original_init(self, config)
        if getattr(config, "fixture_embedding_alias", False):
            self.model.embedding_alias = torch.nn.ModuleDict({"word_embeddings": self.model.embed_tokens})

    monkeypatch.setattr(LlamaForCausalLM, "__init__", init)
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=True,
        fixture_embedding_alias=alias,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(419)
        model = LlamaForCausalLM(config).half().eval()
    source = tmp_path / "source"
    model.config.save_pretrained(source)
    save_file({key: value.detach().clone() for key, value in model.state_dict().items()}, source / "model.safetensors")
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({f"t{i}": i for i in range(64)}, unk_token="t3")),
        pad_token="t0",
        bos_token="t1",
        eos_token="t2",
        unk_token="t3",
    )
    tokenizer.save_pretrained(source)
    wrapped = GPTQModel.load(
        str(source),
        quantize_config=AWQConfig(device="cpu", group_size=16, fallback=None, offload_to_disk=True),
        dtype=torch.float16,
        backend=BACKEND.TORCH,
    )
    ids = torch.tensor([[1, 5, 7, 6, 8, 9], [1, 4, 6, 8, 3, 7]])
    wrapped.quantize(
        CalibrationConfig(
            dataset=[
                {"input_ids": row.unsqueeze(0), "attention_mask": torch.ones(1, 6, dtype=torch.long)} for row in ids
            ],
            batch_size=1,
            min_length=1,
        ),
        backend=BACKEND.TORCH,
    )
    with torch.inference_mode():
        expected = wrapped.model(ids, use_cache=False).logits
    path = tmp_path / "packed"
    wrapped.save(str(path))
    placement = {"device_map": {"": "cpu"}} if explicit else {"device": "cpu"}
    actual = GPTQModel.load(str(path), backend=BACKEND.TORCH, dtype=torch.float16, **placement)
    assert actual.model.model.embed_tokens.weight is actual.model.lm_head.weight
    if alias:
        assert actual.model.model.embedding_alias["word_embeddings"] is actual.model.model.embed_tokens
        assert "model.embedding_alias.word_embeddings.weight" in actual.model.state_dict()
    with torch.inference_mode():
        torch.testing.assert_close(actual.model(ids, use_cache=False).logits, expected, atol=0, rtol=0)
