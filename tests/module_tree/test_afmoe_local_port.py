# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from awq_neutrality import assert_awq_folds_neutral
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import AfmoeForCausalLM, PreTrainedTokenizerFast

from gptqmodel.looper.awq_processor import AWQProcessor
from gptqmodel.looper.loop_processor import CalibrationConfig
from gptqmodel.looper.module_looper import ModuleLooper
from gptqmodel.looper.named_module import NamedModule
from gptqmodel.models.definitions.afmoe import AfMoeQModel
from gptqmodel.models.loader import _convert_model_with_defuser
from gptqmodel.quantization.config import AWQConfig
from tests.test_afmoe_local_port import split_expert_checkpoint, tiny_config


SCHEMA = json.loads(
    (Path(__file__).resolve().parents[1] / "fixtures/ports/afmoe/checkpoint-schema.json").read_text()
)


@pytest.mark.parametrize("checkpoint", SCHEMA["checkpoints"], ids=lambda entry: entry["repo"])
def test_full_native_meta_inventory_matches_pinned_checkpoint_schema(checkpoint):
    from transformers import AfmoeConfig

    config = AfmoeConfig(**checkpoint["config"])
    with torch.device("meta"):
        model = AfmoeForCausalLM(config)
    actual = set(split_expert_checkpoint(model))
    expected = set(SCHEMA["global_keys"])
    for index in range(config.num_hidden_layers):
        suffixes = (
            SCHEMA["common_layer_keys"]
            + SCHEMA["dense_mlp_keys" if index < config.num_dense_layers else "moe_mlp_keys"]
        )
        for suffix in suffixes:
            for expert in range(config.num_experts) if "#" in suffix else (0,):
                expected.add(f"model.layers.{index}.{suffix.replace('#', str(expert))}")
    assert len(expected) == checkpoint["key_count"]
    assert actual == expected
    del model


@pytest.mark.parametrize("device", ["cpu", "meta"])
@pytest.mark.parametrize("dense_layers", [0, 1, 2])
def test_real_checkpoint_owners_and_moe_consumers(device, dense_layers):
    config = tiny_config(num_dense_layers=dense_layers, num_attention_heads=4, num_key_value_heads=2)
    with torch.device(device):
        model = AfmoeForCausalLM(config)
    source_keys = set(split_expert_checkpoint(model))
    _convert_model_with_defuser(AfMoeQModel, model, cleanup_original=True)
    assert set(model.state_dict()) == source_keys
    wrapper = AfMoeQModel(model=model, quantized=False, quantize_config=None)
    plan = wrapper.module_plan()
    simple = {name for block in wrapper.simple_layer_modules(config, SimpleNamespace(dynamic=None)) for name in block}
    looper = ModuleLooper.__new__(ModuleLooper)
    looper.gptq_model = wrapper
    for index, layer in enumerate(model.model.layers):
        for key in layer.state_dict():
            # A direct parameter such as mlp.expert_bias belongs to mlp,
            # not to a fictitious child named expert_bias.
            owner = key.rsplit(".", 1)[0]
            spec = plan.find(owner)
            assert spec is not None, (index, key, owner)
            if spec.quantize:
                assert owner in simple
                assert isinstance(layer.get_submodule(owner), torch.nn.Linear)
            else:
                assert "!" in spec.flags or "?" in spec.flags
        for norm in ("self_attn.q_norm", "self_attn.k_norm"):
            assert not plan.find(norm).quantize
            assert norm not in simple
        for name, role in (("self_attn.q_proj", "q"), ("self_attn.k_proj", "k"), ("self_attn.v_proj", "v")):
            assert role in plan.find(name).roles
        if index >= dense_layers:
            assert plan.find("mlp").capture_only
            assert not plan.find("mlp.router.gate").quantize
            for expert in range(config.num_experts):
                name = f"mlp.experts.{expert}.gate_proj"
                assert plan.is_expert(name)
                assert looper._extract_moe_group_key(name) == f"mlp.experts.{expert}"
                assert name in simple
        assert plan.find("self_attn.q_norm.undeclared") is None
        assert plan.find("mlp.experts_extra.0.gate_proj") is None
    hooks = wrapper.moe_lifecycle_hooks
    assert hooks.module_plan is plan


def _routed_tiny_model(dense_layers, **overrides):
    torch.manual_seed(52)
    config = tiny_config(num_dense_layers=dense_layers, **overrides)
    model = AfmoeForCausalLM(config).eval()
    _convert_model_with_defuser(AfMoeQModel, model, cleanup_original=True)
    with torch.no_grad():
        for layer in model.model.layers:
            # Released checkpoints have trained gains and routers; the native
            # init leaves them at one and zero, which would hide a missing
            # router compensation behind tied routing scores.
            for norm in ("input_layernorm", "post_attention_layernorm", "pre_mlp_layernorm", "post_mlp_layernorm"):
                getattr(layer, norm).weight.uniform_(0.5, 1.5)
            if hasattr(layer.mlp, "router"):
                layer.mlp.router.gate.weight.normal_()
                layer.mlp.expert_bias.normal_(std=0.1)
    return model


ATTENTION_GROUP = (
    "input_layernorm",
    ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.gate_proj"],
)


def _expected_groups(layer, experts):
    if not hasattr(layer.mlp, "router"):
        # The dense MLP's `up_proj -> down_proj` fold is not emitted: the
        # shared planner merges it into the MoE down block and resolves its
        # source from the absent routed-expert `up_proj`.
        return [ATTENTION_GROUP, ("pre_mlp_layernorm", ["mlp.gate_proj", "mlp.up_proj"])]
    gate_up = ["mlp.shared_experts.gate_proj", "mlp.shared_experts.up_proj"]
    for expert in range(experts):
        gate_up += [f"mlp.experts.{expert}.gate_proj", f"mlp.experts.{expert}.up_proj"]
    return [ATTENTION_GROUP, ("pre_mlp_layernorm", gate_up)] + [
        (f"{owner}.up_proj", [f"{owner}.down_proj"])
        for owner in ["mlp.shared_experts"] + [f"mlp.experts.{expert}" for expert in range(experts)]
    ]


@pytest.mark.parametrize("kv_heads", [1, 2])
def test_awq_scale_folds_preserve_dense_and_moe_layer_output(kv_heads):
    model = _routed_tiny_model(1, num_key_value_heads=kv_heads)
    ids = torch.randint(3, 30, (1, 16), generator=torch.Generator().manual_seed(5))

    def forward():
        return model(input_ids=ids).logits

    for layer in model.model.layers:
        checked = assert_awq_folds_neutral(AfMoeQModel, model, layer, forward)
        assert [(source, targets) for source, targets, _ in checked] == _expected_groups(
            layer, model.config.num_experts
        )
        consumers = {source: consumers for source, _, consumers in checked}
        # The sigmoid router reads the same normalized MoE input as the experts.
        assert consumers["pre_mlp_layernorm"] == (["mlp.router.gate"] if hasattr(layer.mlp, "router") else [])
        # `o_proj` reads `attn * sigmoid(gate_proj(x))`; `gate_proj` is `nosrc`,
        # so neither it nor the post-attention sandwich norm becomes a source.
        assert all(source not in {"self_attn.gate_proj", "post_attention_layernorm"} for source, _, _ in checked)


@pytest.mark.parametrize("dense_layers", [0, 1, 2])
def test_awq_scale_search_quantizes_every_linear_and_reloads(tmp_path, monkeypatch, dense_layers):
    from gptqmodel import BACKEND, GPTQModel
    from gptqmodel.looper import awq_processor
    from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear

    # AwqTorchLinear needs a group size of 16 that divides every input width.
    model = _routed_tiny_model(dense_layers, intermediate_size=32, moe_intermediate_size=16)
    model.save_pretrained(tmp_path / "source", safe_serialization=True)
    PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"[PAD]": 0, "[EOS]": 1, "[BOS]": 2, "[UNK]": 3}, unk_token="[UNK]")),
        pad_token="[PAD]",
        eos_token="[EOS]",
        bos_token="[BOS]",
        unk_token="[UNK]",
    ).save_pretrained(tmp_path / "source")
    qcfg = AWQConfig(device="cpu", group_size=16, fallback=None)
    wrapper = GPTQModel.load(
        str(tmp_path / "source"), quantize_config=qcfg, backend=BACKEND.TORCH, dtype=torch.float16
    )
    applied = []
    original_apply_scale = awq_processor.apply_scale

    def record_apply_scale(module, scales_list, *args, **kwargs):
        applied.append([(source, list(targets)) for source, targets, _scales, _loss in scales_list])
        return original_apply_scale(module, scales_list, *args, **kwargs)

    monkeypatch.setattr(awq_processor, "apply_scale", record_apply_scale)
    calibration = [
        {"input_ids": torch.randint(3, 30, (1, 16), generator=torch.Generator().manual_seed(seed)),
         "attention_mask": torch.ones(1, 16, dtype=torch.long)}
        for seed in range(2)
    ]
    wrapper.quantize(CalibrationConfig(dataset=calibration, batch_size=1, min_length=1), backend=BACKEND.TORCH)

    config = model.config
    layers = wrapper.model.model.layers
    assert applied == [_expected_groups(layer, config.num_experts) for layer in layers]
    linears = sum(8 if index < dense_layers else 8 + 3 * config.num_experts for index in range(len(layers)))
    assert sum(isinstance(module, AwqTorchLinear) for module in wrapper.model.modules()) == linears
    with torch.inference_mode():
        expected = wrapper.model(**calibration[0], use_cache=False).logits
    assert torch.isfinite(expected).all()
    wrapper.save_quantized(str(tmp_path / "quantized"))
    reloaded = GPTQModel.load(str(tmp_path / "quantized"), backend=BACKEND.TORCH, device="cpu", dtype=torch.float16)
    assert sum(isinstance(module, AwqTorchLinear) for module in reloaded.model.modules()) == linears
    with torch.inference_mode():
        torch.testing.assert_close(
            reloaded.model(**calibration[0], use_cache=False).logits, expected, atol=2e-4, rtol=2e-3
        )
