# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Intern-S1 real-model checkpoint ownership, consumers and plain AWQ contracts."""

import copy
import json
import re

import pytest
import torch
from PIL import Image
from torch import nn

from gptqmodel.looper.awq_processor import AWQProcessor
from gptqmodel.looper.loop_processor import CalibrationConfig
from gptqmodel.looper.module_looper import ModuleLooper
from gptqmodel.looper.named_module import NamedModule
from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models.definitions.interns1 import InternS1QModel
from gptqmodel.models.ports.interns1.configuration_interns1 import InternS1Config
from gptqmodel.models.ports.interns1.modeling_interns1 import InternS1ForConditionalGeneration
from gptqmodel.quantization.config import AWQConfig
from tests.test_interns1_local_port import tiny_config
from tests.test_interns1_local_port import tokenizer_assets as _tokenizer_assets  # noqa: F401


# Independently observed from the pinned published safetensors indexes, not the planner.
PUBLISHED = json.loads(r'''{
  "dense": {
    "revision": "0f25c8f60f4190a18dd6620c94bdccedbebc208d",
    "key_count": 841,
    "templates": [
      "lm_head.weight",
      "model.language_model.embed_tokens.weight",
      "model.language_model.layers.#.input_layernorm.weight",
      "model.language_model.layers.#.mlp.down_proj.weight",
      "model.language_model.layers.#.mlp.gate_proj.weight",
      "model.language_model.layers.#.mlp.up_proj.weight",
      "model.language_model.layers.#.post_attention_layernorm.weight",
      "model.language_model.layers.#.self_attn.k_norm.weight",
      "model.language_model.layers.#.self_attn.k_proj.weight",
      "model.language_model.layers.#.self_attn.o_proj.weight",
      "model.language_model.layers.#.self_attn.q_norm.weight",
      "model.language_model.layers.#.self_attn.q_proj.weight",
      "model.language_model.layers.#.self_attn.v_proj.weight",
      "model.language_model.norm.weight",
      "model.multi_modal_projector.layer_norm.bias",
      "model.multi_modal_projector.layer_norm.weight",
      "model.multi_modal_projector.linear_1.bias",
      "model.multi_modal_projector.linear_1.weight",
      "model.multi_modal_projector.linear_2.bias",
      "model.multi_modal_projector.linear_2.weight",
      "model.vision_tower.embeddings.cls_token",
      "model.vision_tower.embeddings.patch_embeddings.projection.bias",
      "model.vision_tower.embeddings.patch_embeddings.projection.weight",
      "model.vision_tower.embeddings.position_embeddings",
      "model.vision_tower.encoder.layer.#.attention.k_proj.bias",
      "model.vision_tower.encoder.layer.#.attention.k_proj.weight",
      "model.vision_tower.encoder.layer.#.attention.projection_layer.bias",
      "model.vision_tower.encoder.layer.#.attention.projection_layer.weight",
      "model.vision_tower.encoder.layer.#.attention.q_proj.bias",
      "model.vision_tower.encoder.layer.#.attention.q_proj.weight",
      "model.vision_tower.encoder.layer.#.attention.v_proj.bias",
      "model.vision_tower.encoder.layer.#.attention.v_proj.weight",
      "model.vision_tower.encoder.layer.#.lambda_1",
      "model.vision_tower.encoder.layer.#.lambda_2",
      "model.vision_tower.encoder.layer.#.layernorm_after.bias",
      "model.vision_tower.encoder.layer.#.layernorm_after.weight",
      "model.vision_tower.encoder.layer.#.layernorm_before.bias",
      "model.vision_tower.encoder.layer.#.layernorm_before.weight",
      "model.vision_tower.encoder.layer.#.mlp.fc1.bias",
      "model.vision_tower.encoder.layer.#.mlp.fc1.weight",
      "model.vision_tower.encoder.layer.#.mlp.fc2.bias",
      "model.vision_tower.encoder.layer.#.mlp.fc2.weight"
    ],
    "config": {
      "architectures": [
        "InternS1ForConditionalGeneration"
      ],
      "downsample_ratio": 0.5,
      "image_seq_length": 256,
      "image_token_id": 152957,
      "model_type": "interns1",
      "projector_hidden_act": "gelu",
      "text_config": {
        "_attn_implementation_autoset": true,
        "architectures": [
          "Qwen3ForCausalLM"
        ],
        "attention_bias": false,
        "attention_dropout": 0,
        "bos_token_id": 151643,
        "eos_token_id": 151645,
        "head_dim": 128,
        "hidden_act": "silu",
        "hidden_size": 4096,
        "initializer_range": 0.02,
        "intermediate_size": 12288,
        "max_position_embeddings": 65536,
        "max_window_layers": 36,
        "model_type": "qwen3",
        "num_attention_heads": 32,
        "num_hidden_layers": 36,
        "num_key_value_heads": 8,
        "rms_norm_eps": 0.000001,
        "rope_scaling": null,
        "rope_theta": 1000000,
        "sliding_window": null,
        "torch_dtype": "bfloat16",
        "use_cache": true,
        "use_sliding_window": false,
        "vocab_size": 153216
      },
      "torch_dtype": "bfloat16",
      "transformers_version": "4.53.0",
      "vision_config": {
        "_attn_implementation_autoset": true,
        "architectures": [
          "InternVisionModel"
        ],
        "attention_bias": true,
        "attention_dropout": 0,
        "auto_map": {
          "AutoConfig": "configuration_interns1.InternS1VisionConfig",
          "AutoModel": "modeling_interns1.InternS1VisionModel"
        },
        "drop_path_rate": 0,
        "dropout": 0,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0,
        "hidden_size": 1024,
        "image_size": [
          448,
          448
        ],
        "initializer_factor": 1,
        "initializer_range": 0.02,
        "intermediate_size": 4096,
        "layer_norm_eps": 0.000001,
        "layer_scale_init_value": 0.1,
        "model_type": "interns1_vision",
        "norm_type": "layer_norm",
        "num_attention_heads": 16,
        "num_channels": 3,
        "num_hidden_layers": 24,
        "patch_size": [
          14,
          14
        ],
        "projection_dropout": 0,
        "torch_dtype": "bfloat16",
        "use_absolute_position_embeddings": true,
        "use_mask_token": false,
        "use_mean_pooling": true,
        "use_qk_norm": false
      },
      "vision_feature_layer": -1,
      "vision_feature_select_strategy": "default",
      "auto_map": {
        "AutoConfig": "configuration_interns1.InternS1Config",
        "AutoModel": "modeling_interns1.InternS1Model",
        "AutoModelForCausalLM": "modeling_interns1.InternS1ForConditionalGeneration"
      }
    }
  },
  "moe": {
    "revision": "4ecad381e28293c4825793f113327cc016cdcbda",
    "key_count": 37630,
    "templates": [
      "lm_head.weight",
      "model.language_model.embed_tokens.weight",
      "model.language_model.layers.#.input_layernorm.weight",
      "model.language_model.layers.#.mlp.experts.#.down_proj.weight",
      "model.language_model.layers.#.mlp.experts.#.gate_proj.weight",
      "model.language_model.layers.#.mlp.experts.#.up_proj.weight",
      "model.language_model.layers.#.mlp.gate.weight",
      "model.language_model.layers.#.post_attention_layernorm.weight",
      "model.language_model.layers.#.self_attn.k_norm.weight",
      "model.language_model.layers.#.self_attn.k_proj.weight",
      "model.language_model.layers.#.self_attn.o_proj.weight",
      "model.language_model.layers.#.self_attn.q_norm.weight",
      "model.language_model.layers.#.self_attn.q_proj.weight",
      "model.language_model.layers.#.self_attn.v_proj.weight",
      "model.language_model.norm.weight",
      "model.multi_modal_projector.layer_norm.bias",
      "model.multi_modal_projector.layer_norm.weight",
      "model.multi_modal_projector.linear_1.bias",
      "model.multi_modal_projector.linear_1.weight",
      "model.multi_modal_projector.linear_2.bias",
      "model.multi_modal_projector.linear_2.weight",
      "model.vision_tower.embeddings.cls_token",
      "model.vision_tower.embeddings.patch_embeddings.projection.bias",
      "model.vision_tower.embeddings.patch_embeddings.projection.weight",
      "model.vision_tower.embeddings.position_embeddings",
      "model.vision_tower.encoder.layer.#.attention.k_norm.weight",
      "model.vision_tower.encoder.layer.#.attention.k_proj.weight",
      "model.vision_tower.encoder.layer.#.attention.projection_layer.bias",
      "model.vision_tower.encoder.layer.#.attention.projection_layer.weight",
      "model.vision_tower.encoder.layer.#.attention.q_norm.weight",
      "model.vision_tower.encoder.layer.#.attention.q_proj.weight",
      "model.vision_tower.encoder.layer.#.attention.v_proj.weight",
      "model.vision_tower.encoder.layer.#.lambda_1",
      "model.vision_tower.encoder.layer.#.lambda_2",
      "model.vision_tower.encoder.layer.#.layernorm_after.weight",
      "model.vision_tower.encoder.layer.#.layernorm_before.weight",
      "model.vision_tower.encoder.layer.#.mlp.fc1.bias",
      "model.vision_tower.encoder.layer.#.mlp.fc1.weight",
      "model.vision_tower.encoder.layer.#.mlp.fc2.bias",
      "model.vision_tower.encoder.layer.#.mlp.fc2.weight"
    ],
    "config": {
      "architectures": [
        "InternS1ForConditionalGeneration"
      ],
      "auto_map": {
        "AutoConfig": "configuration_interns1.InternS1Config",
        "AutoModel": "modeling_interns1.InternS1Model",
        "AutoModelForCausalLM": "modeling_interns1.InternS1ForConditionalGeneration"
      },
      "downsample_ratio": 0.5,
      "image_seq_length": 256,
      "image_token_id": 152957,
      "model_type": "interns1",
      "projector_hidden_act": "gelu",
      "text_config": {
        "architectures": [
          "Qwen3MoeForCausalLM"
        ],
        "attention_bias": false,
        "attention_dropout": 0,
        "bos_token_id": 151643,
        "decoder_sparse_step": 1,
        "eos_token_id": 151645,
        "head_dim": 128,
        "hidden_act": "silu",
        "hidden_size": 4096,
        "initializer_range": 0.02,
        "intermediate_size": 8192,
        "max_position_embeddings": 65536,
        "max_window_layers": 94,
        "mlp_only_layers": [],
        "model_type": "qwen3_moe",
        "moe_intermediate_size": 1536,
        "norm_topk_prob": true,
        "num_attention_heads": 64,
        "num_experts": 128,
        "num_experts_per_tok": 8,
        "num_hidden_layers": 94,
        "num_key_value_heads": 4,
        "output_router_logits": false,
        "rms_norm_eps": 0.000001,
        "rope_scaling": null,
        "rope_theta": 1000000,
        "router_aux_loss_coef": 0.001,
        "sliding_window": null,
        "torch_dtype": "bfloat16",
        "use_cache": true,
        "use_sliding_window": false,
        "vocab_size": 153216
      },
      "torch_dtype": "bfloat16",
      "transformers_version": "4.53.0",
      "vision_config": {
        "architectures": [
          "InternVisionModel"
        ],
        "attention_bias": false,
        "attention_dropout": 0,
        "auto_map": {
          "AutoConfig": "configuration_interns1.InternS1VisionConfig",
          "AutoModel": "modeling_interns1.InternS1VisionModel"
        },
        "drop_path_rate": 0.1,
        "dropout": 0,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0,
        "hidden_size": 3200,
        "image_size": [
          448,
          448
        ],
        "initializer_factor": 0.1,
        "initializer_range": 1e-10,
        "intermediate_size": 12800,
        "layer_norm_eps": 0.000001,
        "layer_scale_init_value": 0.1,
        "model_type": "interns1_vision",
        "norm_type": "rms_norm",
        "num_attention_heads": 25,
        "num_channels": 3,
        "num_hidden_layers": 45,
        "patch_size": [
          14,
          14
        ],
        "projection_dropout": 0,
        "torch_dtype": "bfloat16",
        "use_absolute_position_embeddings": true,
        "use_bfloat16": true,
        "use_flash_attn": true,
        "use_mask_token": false,
        "use_mean_pooling": true,
        "use_qk_norm": true
      },
      "vision_feature_layer": -1,
      "vision_feature_select_strategy": "default"
    }
  }
}''')


def normalize(path):
    return re.sub(r"\.\d+(?=\.)", ".#", path)


@pytest.mark.parametrize("layout", ["dense", "moe"])
def test_published_checkpoint_exact_ownership_on_real_meta_model(layout):
    observed = PUBLISHED[layout]
    config = InternS1Config(**observed["config"])
    with torch.device("meta"):
        model = InternS1QModel.loader.from_config(config)
    keys = model.state_dict()
    assert len(keys) == observed["key_count"]
    assert {normalize(name) for name in keys} == set(observed["templates"])
    assert all(tensor.device.type == "meta" for tensor in keys.values())


def wrapper_for(layout, tokenizer_assets):
    config = tiny_config(layout != "dense")
    if layout == "hybrid":
        config.text_config.mlp_only_layers = [0]
    elif layout == "moe_dense_only":
        config.text_config.mlp_only_layers = [0, 1]
    model = InternS1ForConditionalGeneration(config).eval()
    return InternS1QModel(model=model, quantized=False, quantize_config=AWQConfig(group_size=16, device="cpu"),
                         model_local_path=str(tokenizer_assets))


@pytest.mark.parametrize("layout", ["dense", "moe", "hybrid", "moe_dense_only"])
def test_real_wrapper_tree_owns_weights_roles_and_quantizer_consumers(layout, tokenizer_assets):
    declaration = copy.deepcopy(InternS1QModel.module_tree)
    wrapper = wrapper_for(layout, tokenizer_assets)
    assert {key for key, value in MODEL_MAP.items() if value is InternS1QModel} == {"interns1"}
    assert wrapper.extract_layers_node() == ["model.language_model.layers"]
    plan = wrapper.module_plan()
    specs = {name.split(":")[0] for block in wrapper.simple_layer_modules(
        wrapper.model.config, wrapper.quantize_config) for name in block}
    looper = ModuleLooper.__new__(ModuleLooper)
    looper.gptq_model = wrapper
    resolved = set()
    for layer in wrapper.model.model.language_model.layers:
        for name, module in layer.named_modules():
            if isinstance(module, nn.Linear) and name != "mlp.gate":
                assert name in specs
                assert plan.find(name).quantize
                expected_group = name.rsplit(".", 1)[0] if ".experts." in name else None
                assert looper._extract_moe_group_key(name) == expected_group
                resolved.add(name)
        for key in layer.state_dict():
            owner = key.rsplit(".", 1)[0]
            spec = plan.find(owner)
            assert spec is not None, key
            if spec.quantize:
                assert owner in specs
            else:
                assert set(spec.flags) & {"!", "?"}
        for path, role in (("self_attn.q_proj", "q"), ("self_attn.k_proj", "k"),
                           ("self_attn.v_proj", "v"), ("self_attn.o_proj", "o")):
            assert plan.roles(path) == (role,)
        if hasattr(layer.mlp, "experts"):
            assert plan.find("mlp").is_moe and plan.find("mlp").capture_only
            assert plan.find("mlp.gate").quantize is False
            assert wrapper.moe_lifecycle_hooks.module_plan is plan
            for expert in range(config_experts := wrapper.model.config.text_config.num_experts):
                for leaf, role in (("gate_proj", "gate"), ("up_proj", "up"), ("down_proj", "down")):
                    path = f"mlp.experts.{expert}.{leaf}"
                    assert plan.is_expert(path) and plan.roles(path) == (role,)
            assert config_experts == 3
    assert resolved == specs
    for invalid in ("self_attn.q_proj_extra", "self_attn.q_proj.child", "other.mlp.down_proj"):
        assert plan.find(invalid) is None
    assert InternS1QModel.module_tree == declaration
    assert wrapper.awq_get_modules_for_scaling(layer, {}, {}) == []


@pytest.mark.parametrize("layout", ["dense", "moe", "hybrid", "moe_dense_only"])
def test_plain_awq_real_consumer_quantizes_every_linear_and_retries(layout, tokenizer_assets, monkeypatch):
    torch.manual_seed(103)
    wrapper = wrapper_for(layout, tokenizer_assets)
    model = wrapper.model
    layer = model.model.language_model.layers[-1]
    originals = {key: value.clone() for key, value in layer.state_dict().items()}
    processor = AWQProcessor(tokenizer=None, qcfg=wrapper.quantize_config, calibration=CalibrationConfig(),
                             gptq_model=wrapper, model=model, calculate_w_wq_diff=False)
    processor.draw_progress = lambda *args, **kwargs: None
    processor.log_new_row = lambda *args, **kwargs: None
    processor.apply_clip = False
    processor._feature_task_names = set()
    processor._awq_feature_kwargs = {}
    state = processor._get_layer_state(1)
    state.layer_module = layer
    named = {name: NamedModule(module, name, f"model.language_model.layers.1.{name}", 1)
             for name, module in layer.named_modules() if isinstance(module, nn.Linear) and name != "mlp.gate"}
    state.modules.update(named)
    pseudo_quantize = processor.pseudo_quantize_tensor

    def fail_once(*args, **kwargs):
        raise ValueError("injected first tensor failure")

    monkeypatch.setattr(processor, "pseudo_quantize_tensor", fail_once)
    with pytest.raises(ValueError, match="injected first tensor"):
        processor._quantize_layer(1, state)
    assert not state.quantized
    assert state.modules
    for key, value in layer.state_dict().items():
        torch.testing.assert_close(value, originals[key], rtol=0, atol=0)
    monkeypatch.setattr(processor, "pseudo_quantize_tensor", pseudo_quantize)
    processor._quantize_layer(1, state)
    assert state.quantized
    assert not state.modules
    assert processor._awq_feature_kwargs == {}
    assert set(named) == {entry.rsplit(".", 1)[0] for entry in originals if entry.endswith(".weight")
                          and entry.rsplit(".", 1)[0] not in
                          {"input_layernorm", "post_attention_layernorm", "self_attn.q_norm",
                           "self_attn.k_norm", "mlp.gate"}}
    for name, module in named.items():
        assert module.state["q_scales"].numel() > 0
        assert module.state["q_zeros"].numel() > 0
        assert not torch.equal(module.weight, originals[name + ".weight"])
    for name in ("input_layernorm", "post_attention_layernorm", "self_attn.q_norm", "self_attn.k_norm"):
        torch.testing.assert_close(layer.get_submodule(name).weight, originals[name + ".weight"], rtol=0, atol=0)
    if hasattr(layer.mlp, "gate"):
        torch.testing.assert_close(layer.mlp.gate.weight, originals["mlp.gate.weight"], rtol=0, atol=0)
    after = {key: value.clone() for key, value in layer.state_dict().items()}
    processor._quantize_layer(1, state)
    for key, value in layer.state_dict().items():
        torch.testing.assert_close(value, after[key], rtol=0, atol=0)
    with torch.inference_mode():
        assert torch.isfinite(model(torch.tensor([[1, 17, 18, 19]])).logits).all()


@pytest.mark.parametrize("moe", [False, True, "hybrid", "moe_dense_only"])
def test_public_awq_plain_quantization_pack_media_and_safetensors_reload(tokenizer_assets, moe):
    from gptqmodel import BACKEND, GPTQModel
    from gptqmodel.models.ports.interns1.tokenization_interns1 import InternS1Tokenizer
    from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear

    tokenizer = InternS1Tokenizer.from_pretrained(tokenizer_assets)
    config = tiny_config(moe)
    if moe == "hybrid":
        config.text_config.mlp_only_layers = [0]
    elif moe == "moe_dense_only":
        config.text_config.mlp_only_layers = [0, 1]
    config.image_token_id = tokenizer.context_image_token_id
    model = InternS1ForConditionalGeneration._from_config(config).eval()
    model.save_pretrained(tokenizer_assets, safe_serialization=True)
    qcfg = AWQConfig(device="cpu", group_size=16, fallback=None, offload_to_disk=False)
    wrapper = GPTQModel.load(str(tokenizer_assets), quantize_config=qcfg, backend=BACKEND.TORCH,
                            dtype=torch.float16)
    samples = ["A scientific test about atoms and molecules.", "Proteins contain many different amino acids."]
    wrapper.quantize(
        CalibrationConfig(dataset=samples, batch_size=1, min_length=1),
        backend=BACKEND.TORCH,
    )
    expected_count = {False: 14, True: 26, "hybrid": 20, "moe_dense_only": 14}[moe]
    assert sum(isinstance(module, AwqTorchLinear) for module in wrapper.model.modules()) == expected_count
    assert wrapper.processor.tokenizer is wrapper.tokenizer.tokenizer
    inputs = wrapper.processor(text="<IMG_CONTEXT> science", images=Image.new("RGB", (8, 8), color=(55, 77, 99)),
                               return_tensors="pt")
    inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
    with torch.inference_mode():
        expected = wrapper.model(**inputs, use_cache=False).logits
        assert torch.isfinite(expected).all()
    saved = tokenizer_assets / "quantized"
    wrapper.save_quantized(str(saved))
    assert list(saved.glob("*.safetensors")) and not list(saved.glob("*.bin"))
    loaded = GPTQModel.load(str(saved), backend=BACKEND.TORCH, device="cpu", dtype=torch.float16)
    assert sum(isinstance(module, AwqTorchLinear) for module in loaded.model.modules()) == expected_count
    assert loaded.processor.tokenizer is loaded.tokenizer.tokenizer
    restored_inputs = loaded.processor(text="<IMG_CONTEXT> science", images=Image.new("RGB", (8, 8), color=(55, 77, 99)),
                                      return_tensors="pt")
    restored_inputs["pixel_values"] = restored_inputs["pixel_values"].to(torch.float16)
    for key in inputs:
        torch.testing.assert_close(restored_inputs[key], inputs[key], rtol=0, atol=0)
    assert loaded.model.state_dict().keys() == wrapper.model.state_dict().keys()
    for key, value in wrapper.model.state_dict().items():
        torch.testing.assert_close(loaded.model.state_dict()[key], value, rtol=0, atol=0)
    with torch.inference_mode():
        torch.testing.assert_close(loaded.model(**restored_inputs, use_cache=False).logits, expected, atol=2e-4, rtol=2e-3)
        torch.testing.assert_close(wrapper.model(**inputs, use_cache=False).logits, expected, rtol=0, atol=0)
    again_path = tokenizer_assets / "quantized_again"
    loaded.save_quantized(str(again_path))
    again = GPTQModel.load(str(again_path), backend=BACKEND.TORCH, device="cpu", dtype=torch.float16)
    for key, value in loaded.model.state_dict().items():
        torch.testing.assert_close(again.model.state_dict()[key], value, rtol=0, atol=0)
    with torch.inference_mode():
        torch.testing.assert_close(again.model(**restored_inputs, use_cache=False).logits, expected, atol=2e-4, rtol=2e-3)
