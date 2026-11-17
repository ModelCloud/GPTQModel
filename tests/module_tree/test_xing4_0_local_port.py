# SPDX-License-Identifier: Apache-2.0
"""Actual Xing4.0 checkpoint inventory, live owners and conservative AWQ contract."""

import gzip
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models.definitions.xing4_0 import Xing4_0QModel
from gptqmodel.models.loader import _convert_model_with_defuser
from gptqmodel.models.ports.xing4_0.configuration_xing4_0 import Xing4_0Config
from gptqmodel.models.ports.xing4_0.modeling_xing4_0 import Xing4_0ForCausalLM
from gptqmodel.quantization.config import AWQConfig
from tests.test_xing4_0_local_port import ASSETS, tiny_config, tiny_model


def test_pinned_header_index_and_asset_hashes():
    manifest = json.loads((ASSETS / "source_manifest.json").read_text())
    inventory = json.loads(gzip.decompress((ASSETS / "inventory.json.gz").read_bytes()))
    index_bytes = gzip.decompress((ASSETS / "model.safetensors.index.json.gz").read_bytes())
    assert hashlib.sha256(index_bytes).hexdigest() == manifest["files"]["model.safetensors.index.json"]["sha256"]
    index = json.loads(index_bytes)["weight_map"]
    headers = json.loads((ASSETS / "header_manifest.json").read_text())
    reconstructed = {}
    for shard, entry in headers.items():
        raw = gzip.decompress((ASSETS / (shard + ".header.json.gz")).read_bytes())
        assert len(raw) == entry["header_bytes"]
        assert hashlib.sha256(raw).hexdigest() == entry["header_sha256"]
        for name, tensor in json.loads(raw).items():
            if name == "__metadata__":
                continue
            assert index[name] == shard
            assert name not in reconstructed
            reconstructed[name] = {"dtype": tensor["dtype"], "shape": tensor["shape"]}
    assert reconstructed == inventory
    assert hashlib.sha256((ASSETS.parent / "LICENSE").read_bytes()).hexdigest() == manifest["files"]["LICENSE"]["sha256"]
    for name in ["config.json", "generation_config.json", "chat_template.jinja", "tokenizer_config.json", "tokenizer.model"]:
        raw = gzip.decompress((ASSETS / (name + ".gz")).read_bytes()) if name == "tokenizer.model" else (ASSETS / name).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == manifest["files"][name]["sha256"]


def test_full_pinned_release_header_inventory_exact_meta():
    config = Xing4_0Config.from_dict(json.loads((ASSETS / "config.json").read_text()))
    released = json.loads(gzip.decompress((ASSETS / "inventory.json.gz").read_bytes()))
    executed = {name: entry for name, entry in released.items() if not name.startswith("model.layers.40.")}
    assert len(released) == 8307
    assert len(executed) == 8095
    assert sum(entry["dtype"] == "F32" for entry in released.values()) == 39
    with torch.device("meta"):
        model = Xing4_0QModel.loader.from_config(config, dtype=torch.bfloat16)
    assert type(model) is Xing4_0ForCausalLM
    assert {key: list(value.shape) for key, value in model.state_dict().items()} == {
        key: entry["shape"] for key, entry in executed.items()}
    for key, value in model.state_dict().items():
        assert value.dtype is (torch.float32 if key.endswith("e_score_correction_bias") else torch.bfloat16)
    assert all(entry["dtype"] == ("F32" if name.endswith("e_score_correction_bias") else "BF16")
               for name, entry in released.items())
    assert Xing4_0QModel.out_of_model_tensors == {"prefixes": ["model.layers.40"]}
    assert MODEL_MAP["xing4_0"] is Xing4_0QModel
    del model


@pytest.mark.parametrize("device", ["cpu", "meta"])
@pytest.mark.parametrize("qlora,bias,dense,hc", [(8, False, 1, 3), (None, True, 0, 1), (8, True, 2, 4)])
def test_real_live_decoder_plan_every_weight_owner_and_exclusion(device, qlora, bias, dense, hc):
    with torch.device(device):
        model = Xing4_0QModel.loader.from_config(tiny_config(q_lora_rank=qlora, attention_bias=bias,
                                                           first_k_dense_replace=dense, hc_mult=hc))
    _convert_model_with_defuser(Xing4_0QModel, model, cleanup_original=False)
    wrapper = Xing4_0QModel(model=model, quantized=False, quantize_config=None)
    plan = wrapper.module_plan()
    declared = {name for block in wrapper.simple_layer_modules(model.config, SimpleNamespace(dynamic=None)) for name in block}
    for layer in model.model.layers:
        quantized = set()
        for key in layer.state_dict():
            owner = key.rsplit(".", 1)[0]
            spec = plan.find(owner)
            assert spec is not None, key
            is_linear = isinstance(layer.get_submodule(owner), torch.nn.Linear)
            should_quantize = is_linear and owner != "mlp.gate"
            assert spec.quantize == should_quantize, key
            if spec.quantize:
                assert owner in declared
                quantized.add(owner)
            else:
                assert {"!", "?"}.intersection(spec.flags), key
        attention = ({"self_attn.q_proj"} if qlora is None else {"self_attn.q_a_proj", "self_attn.q_b_proj"}) | {
            "self_attn.kv_a_proj_with_mqa", "self_attn.kv_b_proj", "self_attn.o_proj"}
        mlp = {name for name, module in layer.mlp.named_modules() if isinstance(module, torch.nn.Linear) and name != "gate"}
        assert quantized == attention | {"mlp." + name for name in mlp}
        assert wrapper.awq_get_modules_for_scaling(layer, {}, {}) == []
    assert plan.find("self_attn.q_a_proj").flags == ("0", "in=h")
    assert not plan.find("attn_hc").quantize
    assert not plan.find("mlp.gate").quantize
    for invalid in ["self_attn.q_a_proj_suffix", "unrelated.mlp.gate_proj", "mlp.experts.x.gate_proj"]:
        assert plan.find(invalid) is None
    assert Xing4_0QModel.fusable is None


def test_registry_plan_is_unchanged_and_final_head_hook_collapses_streams():
    inventory = json.loads((ASSETS.parents[4] / "tests/module_tree/fixtures/registered_module_plans.json").read_text())
    expected = inventory["cases"]["Xing4_0QModel/gptq"]
    assert Xing4_0QModel.build_layer_modules(Xing4_0QModel.module_tree, include_capture_only=True) == expected["blocks"]
    model = tiny_model()
    wrapper = Xing4_0QModel(model=model, quantized=False, quantize_config=AWQConfig(device="cpu"))
    stream = torch.randn(2, 5, 3, 16)
    rows = [[stream.clone()]]
    result = wrapper.lm_head_pre_quantize_generate_hook(rows)
    merged = stream.mean(2)
    expected_norm = merged * torch.rsqrt(merged.square().mean(-1, keepdim=True) + model.config.rms_norm_eps)
    torch.testing.assert_close(result[0][0], expected_norm, rtol=0, atol=0)
