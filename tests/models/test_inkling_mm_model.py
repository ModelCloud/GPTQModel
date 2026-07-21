# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from pathlib import Path
from types import SimpleNamespace

import torch
from model_test import ModelTest
from safetensors.torch import save_file
from torch import nn
from transformers import AutoModelForMultimodalLM, InklingConfig
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

from gptqmodel.models import auto
from gptqmodel.models.definitions.inkling import InklingMMQModel
from gptqmodel.models.loader import _convert_model_with_defuser
from gptqmodel.utils.structure import LazyTurtle


MODEL_PATH = Path("/monster/data/model/Inkling-0.6B-A0.6B-BF16")


def _tiny_config() -> InklingConfig:
    return InklingConfig(
        text_config={
            "hidden_size": 16,
            "num_hidden_layers": 3,
            "layer_types": ["hybrid_sliding", "hybrid_sliding", "hybrid"],
            "mlp_layer_types": ["dense", "sparse", "sparse"],
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "swa_num_attention_heads": 2,
            "swa_num_key_value_heads": 1,
            "swa_head_dim": 8,
            "vocab_size": 64,
            "intermediate_size": 12,
            "moe_intermediate_size": 6,
            "n_routed_experts": 2,
            "num_experts_per_tok": 1,
            "n_shared_experts": 1,
            "d_rel": 2,
            "rel_extent": 8,
            "sliding_window_size": 4,
            "conv_kernel_size": 2,
            "max_position_embeddings": 32,
            "pad_token_id": 0,
            "bos_token_id": 1,
            "eos_token_id": 2,
        },
        audio_config={"n_mel_bins": 4, "mel_vocab_size": 8},
        vision_config={
            "hidden_size": 8,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "patch_size": 4,
            "temporal_patch_size": 2,
        },
        image_token_id=60,
        audio_token_id=61,
        image_bos_token_id=62,
        audio_bos_token_id=63,
    )


def _tiny_model():
    torch.manual_seed(0)
    model = AutoModelForMultimodalLM.from_config(_tiny_config(), dtype=torch.float32)
    # The optimized grouped-mm backend is CUDA-only; use the model's native
    # eager expert loop for the CPU equivalence assertion.
    model.config.text_config._experts_implementation = "eager"
    model.eval()
    return model


def _interleave(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    shape = list(tensor.shape)
    shape[dim : dim + 1] = [shape[dim] // 2, 2]
    return tensor.reshape(shape).transpose(dim, dim + 1).reshape(tensor.shape).contiguous()


def test_inkling_model_type_selects_definition(monkeypatch):
    fake_config = SimpleNamespace(model_type="inkling_mm_model")

    monkeypatch.setattr(auto, "resolve_trust_remote_code", lambda path, trust_remote_code=False: trust_remote_code)
    monkeypatch.setattr(auto.AutoConfig, "from_pretrained", lambda *args, **kwargs: fake_config)

    assert auto.check_and_get_model_definition("/tmp/inkling") is InklingMMQModel


def test_inkling_module_tree_covers_attention_dense_and_routed_experts():
    layer_modules = InklingMMQModel.simple_layer_modules(
        model_config=SimpleNamespace(text_config=SimpleNamespace(n_routed_experts=2)),
        quantize_config=SimpleNamespace(dynamic=None),
    )
    flat_modules = {name for block in layer_modules for name in block}

    assert InklingMMQModel.loader is AutoModelForMultimodalLM
    assert InklingMMQModel.layer_modules_strict is False
    assert InklingMMQModel.defuser_auto_detect_moe is True
    assert InklingMMQModel.extract_layers_node() == ["model.language_model.layers"]
    assert InklingMMQModel.pre_lm_head_norm_module == "model.language_model.norm"
    assert "self_attn.q_proj" in flat_modules
    assert "self_attn.k_proj" in flat_modules
    assert "self_attn.v_proj" in flat_modules
    assert "self_attn.r_proj" in flat_modules
    assert "self_attn.o_proj" in flat_modules
    assert "mlp.gate_proj" in flat_modules
    assert "mlp.up_proj" in flat_modules
    assert "mlp.down_proj" in flat_modules
    assert "mlp.experts.0.gate_proj" in flat_modules
    assert "mlp.experts.1.up_proj" in flat_modules
    assert "mlp.experts.0.down_proj" in flat_modules
    assert not any("shared_experts" in name for name in flat_modules)


def test_inkling_calibration_uses_native_multimodal_chat_template():
    class RecordingProcessor:
        def apply_chat_template(self, conversations, **kwargs):
            self.conversations = conversations
            self.kwargs = kwargs
            return {"input_ids": torch.tensor([[1, 2]])}

    processor = RecordingProcessor()
    conversations = [{"role": "user", "content": [{"type": "text", "text": "Describe the image."}]}]

    result = InklingMMQModel.prepare_inputs_for_conversations(processor, conversations)

    assert result["input_ids"].shape == (1, 2)
    assert processor.conversations is conversations
    assert processor.kwargs == {
        "tokenize": True,
        "add_generation_prompt": True,
        "reasoning_effort": "medium",
        "return_dict": True,
        "return_tensors": "pt",
    }


def test_inkling_multimodal_capture_materializes_preforward_modules_on_quant_device():
    class RecordingInklingMMQModel(InklingMMQModel):
        def shell_module_materialize(self, target_submodule, device, **kwargs):
            del kwargs
            self.materialize_calls.append((target_submodule, device))
            return target_submodule

    model = _tiny_model()
    wrapper = RecordingInklingMMQModel.__new__(RecordingInklingMMQModel)
    nn.Module.__init__(wrapper)
    wrapper.model = model
    wrapper.quantize_config = SimpleNamespace(device=torch.device("cuda:0"))
    wrapper.materialize_calls = []

    wrapper.pre_quantize_generate_hook_start()

    core_model = model.model
    assert [module for module, _device in wrapper.materialize_calls] == [
        core_model.language_model.embed_tokens,
        core_model.language_model.embed_norm,
        core_model.vision_tower,
        core_model.audio_tower,
    ]
    assert all(device == torch.device("cuda:0") for _module, device in wrapper.materialize_calls)


def test_inkling_defuser_expands_packed_routed_experts_without_changing_forward():
    model = _tiny_model()
    input_ids = torch.tensor([[1, 8, 9, 2]])
    packed_experts = model.model.language_model.layers[1].mlp.experts
    shared_experts = model.model.language_model.layers[1].mlp.shared_experts

    assert hasattr(packed_experts, "gate_up_proj")
    assert isinstance(shared_experts.gate_proj, nn.Parameter)
    with torch.inference_mode():
        expected = model(input_ids=input_ids, use_cache=False).logits

    assert _convert_model_with_defuser(InklingMMQModel, model, cleanup_original=False) is True

    experts = model.model.language_model.layers[1].mlp.experts
    assert not hasattr(experts, "gate_up_proj")
    assert isinstance(experts[0].gate_proj, nn.Linear)
    assert isinstance(experts[0].up_proj, nn.Linear)
    assert isinstance(experts[0].down_proj, nn.Linear)
    assert isinstance(shared_experts.gate_proj, nn.Parameter)
    with torch.inference_mode():
        actual = model(input_ids=input_ids, use_cache=False).logits

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-8)


def test_inkling_replay_rebuilds_sliding_and_full_attention_masks():
    model = _tiny_model()
    wrapper = InklingMMQModel.__new__(InklingMMQModel)
    hidden_states = torch.zeros(1, 6, model.config.text_config.hidden_size)
    padding_mask = torch.tensor([[1, 1, 1, 1, 0, 0]], dtype=torch.bool)
    sliding_layer = model.model.language_model.layers[0]
    full_layer = model.model.language_model.layers[2]

    mask_kwargs = {
        "config": model.config.text_config,
        "inputs_embeds": hidden_states,
        "attention_mask": padding_mask,
        "past_key_values": None,
        "position_ids": None,
    }
    captured_sliding_mask = create_sliding_window_causal_mask(
        **mask_kwargs,
        layer_idx=sliding_layer.self_attn.layer_idx,
    )
    base_inputs = {
        "attention_mask": captured_sliding_mask,
        "conv_mask": padding_mask,
        "past_key_values": None,
    }

    sliding_inputs = wrapper.prepare_layer_replay_kwargs(
        sliding_layer,
        [hidden_states],
        dict(base_inputs),
        torch.device("cpu"),
    )
    full_inputs = wrapper.prepare_layer_replay_kwargs(
        full_layer,
        [hidden_states],
        dict(base_inputs),
        torch.device("cpu"),
    )
    expected_full_mask = create_causal_mask(
        **mask_kwargs,
        layer_idx=full_layer.self_attn.layer_idx,
    )

    assert torch.equal(sliding_inputs["attention_mask"], captured_sliding_mask)
    assert torch.equal(full_inputs["attention_mask"], expected_full_mask)
    assert not torch.equal(full_inputs["attention_mask"], captured_sliding_mask)
    assert full_inputs["conv_mask"] is padding_mask


def test_inkling_lazy_turtle_applies_interleave_before_dense_and_expert_splits(tmp_path):
    model = _tiny_model()
    assert _convert_model_with_defuser(InklingMMQModel, model, cleanup_original=False) is True

    dense_w13 = torch.arange(24 * 16, dtype=torch.float32).reshape(24, 16)
    routed_w13 = torch.arange(2 * 12 * 16, dtype=torch.float32).reshape(2, 12, 16)
    shared_w13 = torch.arange(12 * 16, dtype=torch.float32).reshape(1, 12, 16)
    save_file(
        {
            "model.llm.layers.0.mlp.w13_dn.weight": dense_w13,
            "model.llm.layers.1.mlp.experts.w13_weight": routed_w13,
            "model.llm.layers.1.mlp.shared_experts.shared_w13_weight": shared_w13,
        },
        tmp_path / "model.safetensors",
    )
    turtle = LazyTurtle(
        model_local_path=str(tmp_path),
        config=model.config,
        module_tree=InklingMMQModel.module_tree,
        hf_conversion_map_reversed=InklingMMQModel.resolve_hf_conversion_map_reversed(target_model=model),
        target_model=model,
    )

    dense_source = turtle._resolve_checkpoint_tensor_source(
        "model.language_model.layers.0.mlp.gate_proj",
        "weight",
    )
    routed_source = turtle._resolve_checkpoint_tensor_source(
        "model.language_model.layers.1.mlp.experts.1.up_proj",
        "weight",
    )
    shared_source = turtle._resolve_checkpoint_tensor_source(
        "model.language_model.layers.1.mlp.shared_experts",
        "gate_proj",
    )

    assert dense_source[:3] == ("model.llm.layers.0.mlp.w13_dn.weight", None, 0)
    assert routed_source[:3] == ("model.llm.layers.1.mlp.experts.w13_weight", 1, 1)
    assert shared_source[:3] == (
        "model.llm.layers.1.mlp.shared_experts.shared_w13_weight",
        None,
        0,
    )
    assert getattr(dense_source[3], "interleave_dim", None) == 0
    assert getattr(routed_source[3], "interleave_dim", None) == 1
    assert getattr(shared_source[3], "interleave_dim", None) == 1

    dense_gate = turtle._transform_checkpoint_tensor(
        dense_w13,
        expert_index=dense_source[1],
        split_index=dense_source[2],
        split_dim=dense_source[3],
        expected_shape=(12, 16),
    )
    routed_up = turtle._transform_checkpoint_tensor(
        routed_w13,
        expert_index=routed_source[1],
        split_index=routed_source[2],
        split_dim=routed_source[3],
        expected_shape=(6, 16),
    )
    shared_gate = turtle._transform_checkpoint_tensor(
        shared_w13,
        expert_index=shared_source[1],
        split_index=shared_source[2],
        split_dim=shared_source[3],
        expected_shape=(1, 6, 16),
    )

    expected_dense_gate = _interleave(dense_w13, dim=0).chunk(2, dim=0)[0]
    expected_routed_up = _interleave(routed_w13[1], dim=0).chunk(2, dim=0)[1]
    expected_shared_gate = _interleave(shared_w13, dim=1).chunk(2, dim=1)[0]
    assert torch.equal(dense_gate, expected_dense_gate)
    assert torch.equal(routed_up, expected_routed_up)
    assert torch.equal(shared_gate, expected_shared_gate)

    # Exercise the real shell-materialization path as well as the resolver:
    # these are the three target layouts Inkling exposes after model creation
    # and routed-expert defusion.
    modules = dict(model.named_modules())
    materialization_cases = (
        (
            modules["model.language_model.layers.0.mlp.gate_proj"],
            expected_dense_gate,
        ),
        (
            modules["model.language_model.layers.1.mlp.experts.1.up_proj"],
            expected_routed_up,
        ),
    )
    for target_module, expected_weight in materialization_cases:
        target_module.weight = nn.Parameter(torch.empty_like(target_module.weight, device="meta"))
        turtle.materialize_direct_meta_tensors(
            target_model=model,
            target_submodule=target_module,
            device=torch.device("cpu"),
        )
        assert torch.equal(target_module.weight, expected_weight)

    shared_experts = modules["model.language_model.layers.1.mlp.shared_experts"]
    shared_experts.gate_proj = nn.Parameter(torch.empty_like(shared_experts.gate_proj, device="meta"))
    turtle.materialize_direct_meta_tensors(
        target_model=model,
        target_submodule=shared_experts,
        device=torch.device("cpu"),
    )
    assert torch.equal(shared_experts.gate_proj, expected_shared_gate)


class TestInklingMMModel(ModelTest):
    NATIVE_MODEL_ID = str(MODEL_PATH)
    TRUST_REMOTE_CODE = False
    USE_FLASH_ATTN = False
    EVAL_BATCH_SIZE = 1
    MODEL_COMPAT_FAST_LAYER_POSITION = "first"
    EVAL_TASKS_SLOW = {
        "arc_challenge": {
            "chat_template": True,
            "acc": {"value": 0.2098, "floor_pct": 0.04},
            "acc_norm": {"value": 0.2389, "floor_pct": 0.04},
        },
    }
    EVAL_TASKS_FAST = ModelTest.derive_fast_eval_tasks(EVAL_TASKS_SLOW)

    def test_inkling_mm_model(self):
        self.quantize_and_evaluate()


__all__ = ["TestInklingMMModel"]
