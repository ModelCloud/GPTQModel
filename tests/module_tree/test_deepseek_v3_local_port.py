# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint-owned DeepSeek-V3 MLA norms, routers, dense Q and split Q."""

from types import SimpleNamespace

import pytest
import torch

from gptqmodel.looper.awq_processor import AWQProcessor
from gptqmodel.looper.loop_processor import CalibrationConfig
from gptqmodel.looper.module_looper import ModuleLooper
from gptqmodel.looper.named_module import NamedModule
from gptqmodel.models.definitions.deepseek_v3 import DeepSeekV3QModel
from gptqmodel.models.loader import _convert_model_with_defuser
from gptqmodel.models.ports.native.deepseek_v3.modeling_deepseek_v3 import DeepseekV3ForCausalLM
from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
from gptqmodel.quantization.config import AWQConfig
from gptqmodel.utils.hf import load_model_config
from tests.test_deepseek_v3_local_port import checkpoint_state_dict, poison_checkpoint, tiny_payload, write_checkpoint


@pytest.mark.parametrize("model_type", ["deepseek_v3", "kimi_k2"])
@pytest.mark.parametrize("q_lora_rank", [None, 8], ids=["dense_q", "split_q"])
def test_every_checkpoint_owner_is_declared_and_experts_survive_conversion(tmp_path, model_type, q_lora_rank):
    poison_checkpoint(tmp_path, tiny_payload(model_type, q_lora_rank))
    config = load_model_config(tmp_path)
    with torch.device("meta"):
        model = DeepSeekV3QModel.loader.from_config(config)
    assert isinstance(model, DeepseekV3ForCausalLM)
    expected_keys = set(checkpoint_state_dict(model))
    assert _convert_model_with_defuser(DeepSeekV3QModel, model, cleanup_original=True)
    assert set(model.state_dict()) == expected_keys
    wrapper = DeepSeekV3QModel(model=model, quantized=False, quantize_config=None)
    plan = wrapper.module_plan()
    declared = DeepSeekV3QModel.simple_layer_modules(config, SimpleNamespace(dynamic=None))
    # Quantization paths are independent of the router and the two latent RMSNorms.
    simple = {name for block in declared for name in block}
    for layer in model.model.layers:
        for key in layer.state_dict():
            owner = key.rsplit(".", 1)[0]
            spec = plan.find(owner)
            assert spec is not None, f"{model_type}: undeclared checkpoint weight {key}"
            if isinstance(layer.get_submodule(owner), torch.nn.Linear):
                assert spec.quantize, owner
                assert owner in simple, owner
            else:
                assert not spec.quantize, owner
    assert "self_attn.q_proj" in simple
    assert "self_attn.q_a_proj" in simple
    assert "self_attn.q_b_proj" in simple
    for name in ("self_attn.q_a_layernorm", "self_attn.kv_a_layernorm", "mlp.gate"):
        assert not plan.find(name).quantize
        assert name not in simple
        assert plan.find(name + ".undeclared") is None
    assert plan.roles("mlp.experts.2.gate_proj") == ("gate",)
    assert plan.roles("mlp.shared_experts.up_proj") == ("up",)
    assert plan.roles("self_attn.q_b_proj") == ("q",)
    assert plan.roles("self_attn.kv_b_proj") == ("k", "v")
    assert plan.expert_containers() == ("mlp.experts",)
    assert plan.shared_expert_containers() == ("mlp.shared_experts",)
    looper = object.__new__(ModuleLooper)
    looper.gptq_model = wrapper
    assert looper._extract_moe_group_key("mlp.experts.2.up_proj") == "mlp.experts.2"
    assert looper._extract_moe_group_key("mlp.shared_experts.up_proj") is None


@pytest.mark.parametrize("model_type", ["deepseek_v3", "kimi_k2"])
@pytest.mark.parametrize("q_lora_rank", [None, 16], ids=["dense_q", "split_q"])
@pytest.mark.parametrize("fallback", [None, {}], ids=["no_fallback", "fallback"])
def test_featureless_awq_quantizes_and_packs_every_real_projection(
    tmp_path, monkeypatch, model_type, q_lora_rank, fallback,
):
    poison_checkpoint(tmp_path, tiny_payload(
        model_type, q_lora_rank, hidden_size=32, intermediate_size=32, moe_intermediate_size=16,
        kv_lora_rank=16, qk_nope_head_dim=8, qk_rope_head_dim=8, v_head_dim=16,
    ))
    config = load_model_config(tmp_path)
    config._attn_implementation = "eager"
    model = DeepSeekV3QModel.loader.from_config(config).eval().half()
    _convert_model_with_defuser(DeepSeekV3QModel, model, cleanup_original=True)
    wrapper = DeepSeekV3QModel(model=model, quantized=False, quantize_config=None)
    processor = AWQProcessor(
        tokenizer=None, qcfg=AWQConfig(device="cpu", group_size=16, fallback=fallback),
        calibration=CalibrationConfig(), gptq_model=wrapper, model=model, calculate_w_wq_diff=False,
    )
    processor.draw_progress = lambda *args, **kwargs: None
    processor.log_new_row = lambda *args, **kwargs: None
    processor._feature_task_names = set()
    processor._awq_feature_kwargs = {}
    protected = {
        name: value.clone() for name, value in model.state_dict().items()
        if "layernorm" in name or ".gate." in name
    }
    packed_count = 0
    for i, layer in enumerate(model.model.layers):
        state = processor._get_layer_state(i)
        state.layer_module = layer
        state.modules = {
            name: NamedModule(module, name, f"model.layers.{i}.{name}", i)
            for name, module in layer.named_modules()
            if isinstance(module, torch.nn.Linear) and wrapper.module_plan().find(name).quantize
        }
        # Without captured activations no scale group can be built, so every
        # projection (including unobserved experts) takes the plain AWQ path.
        monkeypatch.setattr(processor, "_layer_input_features", lambda _: {})
        assert wrapper.awq_get_modules_for_scaling(layer, {}, {}) == []
        named_modules = dict(state.modules)
        processor._quantize_layer(i, state)
        assert state.quantized
        for name, named in named_modules.items():
            assert named.state["q_scales"].numel() and named.state["q_zeros"].numel()
            linear = named.module
            packed = AwqTorchLinear(
                bits=4, group_size=16, sym=False, desc_act=False,
                in_features=linear.in_features, out_features=linear.out_features, register_buffers=True,
            )
            packed.pack(linear, named.state["q_scales"], named.state["q_zeros"])
            assert packed.qweight.dtype == torch.int32 and packed.qweight.numel()
            # Packing preserves the pseudo-quantized linear's actual computation.
            x = torch.linspace(-1, 1, linear.in_features).half().expand(2, -1)
            with torch.inference_mode():
                torch.testing.assert_close(packed(x), linear(x), rtol=2e-3, atol=2e-3)
            parent, _, leaf = name.rpartition(".")
            setattr(layer.get_submodule(parent), leaf, packed)
            packed_count += 1
    assert packed_count == (26 if q_lora_rank is None else 28)
    for name, expected in protected.items():
        torch.testing.assert_close(model.state_dict()[name], expected, rtol=0, atol=0)
    with torch.inference_mode():
        assert torch.isfinite(model(torch.tensor([[1, 3, 6, 9]])).logits).all()


@pytest.mark.parametrize("model_type", ["deepseek_v3", "kimi_k2"])
@pytest.mark.parametrize("q_lora_rank", [None, 8], ids=["dense_q", "split_q"])
def test_awq_scale_folds_preserve_real_mla_dense_and_moe_layers(tmp_path, model_type, q_lora_rank):
    from awq_neutrality import assert_awq_folds_neutral

    poison_checkpoint(tmp_path, tiny_payload(model_type, q_lora_rank, n_shared_experts=2, num_experts_per_tok=4))
    config = load_model_config(tmp_path)
    config._attn_implementation = "eager"
    torch.manual_seed(3)
    model = DeepSeekV3QModel.loader.from_config(config).eval()
    assert _convert_model_with_defuser(DeepSeekV3QModel, model, cleanup_original=True)
    ids = torch.randint(3, 30, (1, 8))

    def forward():
        return model(input_ids=ids, use_cache=False).logits

    q_in = "self_attn.q_proj" if q_lora_rank is None else "self_attn.q_a_proj"
    for index, layer in enumerate(model.model.layers):
        checked = assert_awq_folds_neutral(DeepSeekV3QModel, model, layer, forward)
        groups = {(source, tuple(targets), tuple(consumers)) for source, targets, consumers in checked}
        # MLA: only the projections reading the input norm fold into it;
        # q_b_proj/kv_b_proj read the latents and o_proj the attention output.
        expected = {("input_layernorm", (q_in, "self_attn.kv_a_proj_with_mqa"), ())}
        if index < config.first_k_dense_replace:
            expected |= {
                ("post_attention_layernorm", ("mlp.gate_proj", "mlp.up_proj"), ()),
                ("mlp.up_proj", ("mlp.down_proj",), ()),
            }
        else:
            owners = [*(f"mlp.experts.{e}" for e in range(config.n_routed_experts)), "mlp.shared_experts"]
            # Routed and shared experts read one norm and fold as one group;
            # the router reads it too and is compensated.
            expected.add((
                "post_attention_layernorm",
                tuple(f"{owner}.{proj}_proj" for owner in owners for proj in ("gate", "up")),
                ("mlp.gate",),
            ))
            expected |= {(f"{owner}.up_proj", (f"{owner}.down_proj",), ()) for owner in owners}
        assert groups == expected
        folded = {target for _, targets, _ in checked for target in targets}
        assert not folded & {"self_attn.q_b_proj", "self_attn.kv_b_proj", "self_attn.o_proj"}


def test_public_awq_scale_search_quantizes_mla_and_moe_and_folds_norms(tmp_path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    from gptqmodel import BACKEND, GPTQModel

    reference, marker = write_checkpoint(tmp_path, tiny_payload(
        "deepseek_v3", 16, hidden_size=32, intermediate_size=32, moe_intermediate_size=16,
        kv_lora_rank=16, qk_nope_head_dim=8, qk_rope_head_dim=8, v_head_dim=16,
    ))
    PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"[PAD]": 0, "[BOS]": 1, "[EOS]": 2, "[UNK]": 3}, unk_token="[UNK]")),
        pad_token="[PAD]", bos_token="[BOS]", eos_token="[EOS]", unk_token="[UNK]",
    ).save_pretrained(tmp_path)
    wrapper = GPTQModel.load(
        str(tmp_path), quantize_config=AWQConfig(device="cpu", group_size=16, fallback=None),
        backend=BACKEND.TORCH, dtype=torch.float16,
    )
    norms = {key: value.half() for key, value in reference.state_dict().items() if key.endswith("layernorm.weight")}
    ids = torch.tensor([[1, 4, 5, 6, 7, 8, 9, 10, 11, 12]])
    calibration = [{"input_ids": ids, "attention_mask": torch.ones_like(ids)} for _ in range(2)]
    wrapper.quantize(CalibrationConfig(dataset=calibration, batch_size=1, min_length=1), backend=BACKEND.TORCH)
    packed = {name for name, module in wrapper.model.named_modules() if isinstance(module, AwqTorchLinear)}
    for index in range(2):
        for name in ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"):
            assert f"model.layers.{index}.self_attn.{name}" in packed
    assert {"model.layers.1.mlp.shared_experts.gate_proj", "model.layers.1.mlp.shared_experts.down_proj"} <= packed
    state = wrapper.model.state_dict()
    changed = {key for key, value in norms.items() if not torch.equal(state[key].half(), value)}
    # The input norm absorbs the q_a/kv_a fold; the MLA latent norms are never fold sources.
    assert {"model.layers.0.input_layernorm.weight", "model.layers.1.input_layernorm.weight"} <= changed
    assert not {key for key in changed if "_a_layernorm" in key}
    with torch.inference_mode():
        assert torch.isfinite(wrapper.model(ids).logits).all()
    assert not marker.exists()
