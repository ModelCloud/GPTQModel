# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Actual EXAONE checkpoint owners, live plans, and conservative AWQ packing."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models.definitions.exaone import ExaOneQModel
from gptqmodel.models.loader import _convert_model_with_defuser
from gptqmodel.models.ports.exaone import configuration_exaone
from gptqmodel.models.ports.exaone.configuration_exaone import ExaoneConfig
from gptqmodel.models.ports.exaone.modeling_exaone import ExaoneForCausalLM
from tests.test_exaone_local_port import ROPE, tiny_config, tiny_model, write_tokenizer


PORT = Path(configuration_exaone.__file__).parent
CONTRACTS = json.loads((PORT / "checkpoint_contracts.json").read_text())
QUANTIZED = {"attn.attention.q_proj", "attn.attention.k_proj", "attn.attention.v_proj",
             "attn.attention.out_proj", "mlp.c_fc_0", "mlp.c_fc_1", "mlp.c_proj"}
BLOCKS = [["ln_1:!"], ["attn.attention.q_proj", "attn.attention.k_proj", "attn.attention.v_proj"],
          ["attn.attention.out_proj"], ["ln_2:!"], ["mlp.c_fc_0", "mlp.c_fc_1"], ["mlp.c_proj"]]


@pytest.mark.parametrize("release", CONTRACTS)
def test_actual_released_checkpoint_exact_meta_ownership(release):
    contract = CONTRACTS[release]
    config = ExaoneConfig(**contract["config"])
    with torch.device("meta"):
        model = ExaoneForCausalLM(config)
    actual = set(model.state_dict())
    checkpoint = set()
    for key in contract["normalized_checkpoint_keys"]:
        if ".#." in key:
            checkpoint.update(key.replace(".#.", f".{i}.") for i in range(config.num_layers))
        else:
            checkpoint.add(key)
    assert hashlib.sha256("\n".join(sorted(checkpoint)).encode()).hexdigest() == contract["checkpoint_keys_sha256"]
    assert len(checkpoint) == contract["checkpoint_tensor_count"]
    if config.tie_word_embeddings:
        assert "lm_head.weight" not in checkpoint
        assert model.lm_head.weight is model.transformer.wte.weight
        checkpoint.add("lm_head.weight")
    assert actual == checkpoint
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
    assert wrapper.awq_get_modules_for_scaling(model.transformer.h[0], {}, {}) == []
    registry = json.loads((Path(__file__).parent / "fixtures/registered_module_plans.json").read_text())
    assert registry["models"]["exaone"] == "ExaOneQModel"
    assert registry["cases"]["ExaOneQModel/gptq"] == {"layers": ["transformer.h"], "blocks": BLOCKS}


@pytest.mark.parametrize("tied,rope", [(False, "default"), (True, "llama3")])
@torch.no_grad()
def test_plain_awq_quantizes_packs_every_linear_and_round_trips(tmp_path, tied, rope):
    from gptqmodel import BACKEND, GPTQModel
    from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
    from gptqmodel.quantization.config import AWQConfig, CalibrationConfig
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
    layer = wrapper.model.transformer.h[0]
    assert wrapper.awq_get_modules_for_scaling(layer, {}, {}) == []
    calibration = [{"input_ids": torch.tensor([[4, 5, 6, 7, 8, 9, 10, 11]]),
                    "attention_mask": torch.ones(1, 8, dtype=torch.long)} for _ in range(2)]
    wrapper.quantize(CalibrationConfig(dataset=calibration, batch_size=1, min_length=1), backend=BACKEND.TORCH)
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
    for name, value in norms.items():
        assert torch.equal(restored.model.state_dict()[name], value), "plain AWQ must not fold a norm"
    torch.testing.assert_close(restored.model(**calibration[0], use_cache=False).logits, expected, atol=2e-4, rtol=2e-3)
