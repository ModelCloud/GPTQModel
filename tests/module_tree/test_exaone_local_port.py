# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Actual EXAONE checkpoint owners, live plans, AWQ fold neutrality and packing."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from awq_neutrality import assert_awq_folds_neutral
from torch import nn

from gptqmodel import CalibrationConfig
from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models.definitions.exaone import ExaOneQModel
from gptqmodel.models.loader import _convert_model_with_defuser
from gptqmodel.models.ports.exaone.configuration_exaone import ExaoneConfig
from gptqmodel.models.ports.exaone.modeling_exaone import ExaoneForCausalLM
from tests.test_exaone_local_port import ROPE, tiny_config, tiny_model, write_tokenizer


# Geometry of each released config.json and the sha256/count of the sorted tensor
# names in its safetensors index (lm_head is absent from the tied 2.4B release).
RELEASES = {
    "exaone30": ((4096, 32, 32, 8, 14336, False), 291,
                 "7bd5647098a7a6eb57a2324a2ea49c59a28adab511cdf6a689919746755f542a"),
    "exaone35_2b": ((2560, 30, 32, 8, 7168, True), 272,
                    "e877c569323322a6e51e5b9ac7d2cd1d75086db12b144383d0b292947ca83b93"),
    "exaone35_7b": ((4096, 32, 32, 8, 14336, False), 291,
                    "7bd5647098a7a6eb57a2324a2ea49c59a28adab511cdf6a689919746755f542a"),
    "exaone35_32b": ((5120, 64, 40, 8, 27392, False), 579,
                     "f19c86853c665d8b7d48b455cb5c05f062c3c8951e53b627827605faaedb9ea8"),
}
QUANTIZED = {"attn.attention.q_proj", "attn.attention.k_proj", "attn.attention.v_proj",
             "attn.attention.out_proj", "mlp.c_fc_0", "mlp.c_fc_1", "mlp.c_proj"}
BLOCKS = [["ln_1:!"], ["attn.attention.q_proj", "attn.attention.k_proj", "attn.attention.v_proj"],
          ["attn.attention.out_proj"], ["ln_2:!"], ["mlp.c_fc_0", "mlp.c_fc_1"], ["mlp.c_proj"]]


@pytest.mark.parametrize("release", RELEASES)
def test_actual_released_checkpoint_exact_meta_ownership(release):
    (hidden, layers, heads, kv_heads, intermediate, tied), count, keys_sha256 = RELEASES[release]
    config = ExaoneConfig(vocab_size=102400, hidden_size=hidden, num_layers=layers, num_attention_heads=heads,
                          num_key_value_heads=kv_heads, intermediate_size=intermediate, tie_word_embeddings=tied)
    with torch.device("meta"):
        model = ExaoneForCausalLM(config)
    actual = set(model.state_dict())
    checkpoint = actual - {"lm_head.weight"} if tied else actual
    if tied:
        assert model.lm_head.weight is model.transformer.wte.weight
    assert len(checkpoint) == count
    assert hashlib.sha256("\n".join(sorted(checkpoint)).encode()).hexdigest() == keys_sha256
    assert model.transformer.h[0].attn.attention.k_proj.weight.shape == (
        config.num_key_value_heads * config.head_dim, config.hidden_size)
    assert len(model.transformer.h) == config.num_layers
    assert not any("inv_freq" in key for key in actual)
    assert {name for name, definition in MODEL_MAP.items() if definition is ExaOneQModel} == {"exaone"}


@pytest.mark.parametrize("kv_heads,tied", [(1, False), (2, True), (4, False)])
@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_live_decoder_plan_covers_all_checkpoint_owners(kv_heads, tied, device):
    with torch.device(device):
        model = ExaoneForCausalLM(tiny_config(num_key_value_heads=kv_heads, tie_word_embeddings=tied))
    _convert_model_with_defuser(ExaOneQModel, model, cleanup_original=False)
    wrapper = ExaOneQModel(model=model, quantized=False, quantize_config=None)
    plan = wrapper.module_plan()
    assert wrapper.extract_layers_node() == ["transformer.h"]
    assert wrapper.full_layer_modules() == BLOCKS
    for method in ("gptq", "awq"):
        assert {name for block in wrapper.simple_layer_modules(model.config, SimpleNamespace(method=method, dynamic=None))
                for name in block} == QUANTIZED
    for layer in model.transformer.h:
        assert set(layer.state_dict()) == {name + ".weight" for name in QUANTIZED | {"ln_1", "ln_2"}}
        for key in layer.state_dict():
            owner = key.rsplit(".", 1)[0]
            spec = plan.find(owner)
            assert spec is not None, key
            if spec.quantize:
                assert owner in QUANTIZED and isinstance(layer.get_submodule(owner), nn.Linear)
            else:
                assert owner in {"ln_1", "ln_2"} and "!" in spec.flags
            assert plan.find(owner + ".undeclared") is None
            assert plan.find(owner + "_extra") is None
    for projection, role in [("q_proj", "q"), ("k_proj", "k"), ("v_proj", "v")]:
        assert plan.roles("attn.attention." + projection) == (role,)
    registry = json.loads((Path(__file__).parent / "fixtures/registered_module_plans.json").read_text())
    assert registry["models"]["exaone"] == "ExaOneQModel"
    assert registry["cases"]["ExaOneQModel/gptq"] == {"layers": ["transformer.h"], "blocks": BLOCKS}


ATTENTION_GROUP = ("ln_1", ["attn.attention.q_proj", "attn.attention.k_proj", "attn.attention.v_proj"], [])
VALUE_GROUP = ("attn.attention.v_proj", ["attn.attention.out_proj"], [])
MLP_GROUPS = [("ln_2", ["mlp.c_fc_0", "mlp.c_fc_1"], []), ("mlp.c_fc_1", ["mlp.c_proj"], [])]


@pytest.mark.parametrize("kv_heads", [2, 4])
def test_awq_scale_folds_preserve_every_decoder_layer_output(kv_heads):
    model = tiny_model(num_key_value_heads=kv_heads)
    for name, value in model.named_parameters():
        if ".ln_" in name:
            with torch.no_grad():
                value.uniform_(0.5, 1.5)
    _convert_model_with_defuser(ExaOneQModel, model, cleanup_original=False)
    ids = torch.tensor([[4, 5, 6, 7, 8, 9, 10, 11]])

    def forward():
        return model(input_ids=ids, use_cache=False).logits

    # v_proj -> out_proj is only an exact fold without GQA head repetition.
    expected = [ATTENTION_GROUP, *([VALUE_GROUP] if kv_heads == 4 else []), *MLP_GROUPS]
    for layer in model.transformer.h:
        assert assert_awq_folds_neutral(ExaOneQModel, model, layer, forward) == expected


@pytest.mark.parametrize("tied,rope", [(False, "default"), (True, "llama3")])
@torch.no_grad()
def test_awq_scale_search_quantizes_packs_every_linear_and_round_trips(tmp_path, tied, rope):
    from gptqmodel import BACKEND, GPTQModel
    from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
    from gptqmodel.quantization.config import AWQConfig
    model = tiny_model(tie_word_embeddings=tied, rope_scaling=ROPE[rope])
    norms = {}
    for name, value in model.named_parameters():
        if ".ln_" in name:
            value.uniform_(0.5, 1.5)
            norms[name] = value.half().clone()
    model.save_pretrained(tmp_path)
    write_tokenizer(tmp_path)
    qcfg = AWQConfig(device="cpu", group_size=16, fallback=None)
    wrapper = GPTQModel.load(str(tmp_path), quantize_config=qcfg, backend=BACKEND.TORCH, dtype=torch.float16)
    calibration = [{"input_ids": torch.tensor([[4, 5, 6, 7, 8, 9, 10, 11]]),
                    "attention_mask": torch.ones(1, 8, dtype=torch.long)} for _ in range(2)]
    wrapper.quantize(
        CalibrationConfig(dataset=calibration, batch_size=1, min_length=1),
        backend=BACKEND.TORCH,
    )
    assert sum(isinstance(module, AwqTorchLinear) for module in wrapper.model.modules()) == 14
    for index in range(2):
        decoder = wrapper.model.transformer.h[index]
        assert {name for name, module in decoder.named_modules() if isinstance(module, AwqTorchLinear)} == QUANTIZED
    expected = wrapper.model(**calibration[0], use_cache=False).logits
    assert torch.isfinite(expected).all()
    saved = tmp_path / "quantized"
    wrapper.save_quantized(str(saved))
    assert list(saved.glob("*.safetensors")) and not list(saved.glob("*.bin"))
    restored = GPTQModel.load(str(saved), backend=BACKEND.TORCH, device="cpu", dtype=torch.float16)
    assert sum(isinstance(module, AwqTorchLinear) for module in restored.model.modules()) == 14
    # Scale search folds input scales into ln_1/ln_2, so they must differ from the source.
    assert any(not torch.equal(restored.model.state_dict()[name], value) for name, value in norms.items())
    torch.testing.assert_close(restored.model(**calibration[0], use_cache=False).logits, expected, atol=2e-4, rtol=2e-3)
