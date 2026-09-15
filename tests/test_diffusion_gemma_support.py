from types import SimpleNamespace

import torch
from torch import nn
from accelerate import init_empty_weights
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForMultimodalLM, DiffusionGemmaConfig
from transformers.models.diffusion_gemma.generation_diffusion_gemma import (
    DiffusionGemmaGenerationConfig,
    EntropyBoundSamplerConfig,
)

from gptqmodel.models import auto
from gptqmodel.models.definitions.diffusion_gemma import (
    _DIFFUSION_GEMMA_MASKS,
    DiffusionGemmaQModel,
    _sync_decoder_quantized_modules,
    _sync_decoder_tied_parameters,
)
from gptqmodel.models.loader import _convert_model_with_defuser
from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.utils.hf import autofix_hf_model_config
from gptqmodel.utils.model import MODALITY
from gptqmodel.utils.structure import LazyTurtle


MODEL_PATH = "/monster/data/model/diffusiongemma-26B-A4B-it"


def _tiny_config() -> DiffusionGemmaConfig:
    return DiffusionGemmaConfig(
        text_config={
            "vocab_size": 64,
            "hidden_size": 16,
            "intermediate_size": 24,
            "moe_intermediate_size": 8,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "num_global_key_value_heads": 1,
            "head_dim": 8,
            "global_head_dim": 8,
            "num_experts": 2,
            "top_k_experts": 1,
            "max_position_embeddings": 64,
            "sliding_window": 8,
            "layer_types": ["sliding_attention", "full_attention"],
            "use_bidirectional_attention": "vision",
        },
        vision_config={
            "model_type": "gemma4_vision",
            "hidden_size": 16,
            "intermediate_size": 24,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "max_position_embeddings": 64,
            "patch_size": 2,
            "pooling_kernel_size": 1,
            "position_embedding_size": 16,
        },
        canvas_length=4,
        image_token_id=60,
        boi_token_id=61,
        eoi_token_id=62,
    )


def _tiny_model():
    torch.manual_seed(0)
    return AutoModelForMultimodalLM.from_config(_tiny_config(), dtype=torch.float32).eval()


def test_diffusion_gemma_local_checkpoint_selects_definition():
    config = AutoConfig.from_pretrained(MODEL_PATH)

    assert config.model_type == "diffusion_gemma"
    assert auto.check_and_get_model_definition(MODEL_PATH) is DiffusionGemmaQModel


def test_diffusion_gemma_module_tree_covers_dense_and_packed_experts():
    config = _tiny_config()
    layer_modules = DiffusionGemmaQModel.simple_layer_modules(
        model_config=config,
        quantize_config=SimpleNamespace(dynamic=None),
    )
    flat_modules = {name for block in layer_modules for name in block}

    assert DiffusionGemmaQModel.loader is AutoModelForMultimodalLM
    assert DiffusionGemmaQModel.turtle_materialize_tie_weights is False
    assert DiffusionGemmaQModel.modality == [MODALITY.TEXT, MODALITY.IMAGE_TO_TEXT]
    assert DiffusionGemmaQModel.extract_layers_node() == ["model.encoder.language_model.layers"]
    assert "self_attn.q_proj" in flat_modules
    assert "self_attn.k_proj" in flat_modules
    assert "self_attn.v_proj" in flat_modules
    assert "self_attn.o_proj" in flat_modules
    assert "mlp.gate_proj" in flat_modules
    assert "mlp.up_proj" in flat_modules
    assert "mlp.down_proj" in flat_modules
    assert "experts.0.gate_proj" in flat_modules
    assert "experts.1.up_proj" in flat_modules
    assert "experts.0.down_proj" in flat_modules


def test_diffusion_gemma_excludes_decoder_from_capture_base_modules():
    model = _tiny_model()
    base_modules = DiffusionGemmaQModel.get_base_modules(model)

    assert "model.decoder" not in base_modules
    assert "model.encoder.vision_tower" in base_modules
    assert "model.encoder.embed_vision" in base_modules
    assert "model.encoder.language_model.embed_tokens" in base_modules
    assert "model.encoder.language_model.rotary_emb" in base_modules


def test_diffusion_gemma_pre_quantizes_capture_modules_and_patches_masks():
    class FakeLanguageModel(nn.Module):
        def forward(self, *args, **kwargs):
            return kwargs

    language_model = FakeLanguageModel()
    original_modules = [
        nn.Linear(2, 2),
        nn.Linear(2, 2),
        nn.Embedding(8, 2),
        nn.Identity(),
    ]
    encoder = SimpleNamespace(
        vision_tower=original_modules[0],
        embed_vision=original_modules[1],
        language_model=language_model,
    )
    language_model.embed_tokens = original_modules[2]
    language_model.rotary_emb = original_modules[3]

    wrapper = object.__new__(DiffusionGemmaQModel)
    nn.Module.__init__(wrapper)
    wrapper.model = SimpleNamespace(model=SimpleNamespace(encoder=encoder))
    seen = []

    def fake_pre_quantize(module):
        seen.append(module)
        return nn.Identity()

    wrapper.pre_quantize = fake_pre_quantize
    wrapper.pre_quantize_generate_hook_start()

    assert seen == original_modules
    assert encoder.vision_tower not in original_modules
    assert encoder.embed_vision not in original_modules
    assert language_model.embed_tokens not in original_modules
    assert language_model.rotary_emb not in original_modules

    mask_mapping = {"full_attention": torch.ones(1, 1, 2, 2, dtype=torch.bool)}
    language_model(attention_mask=mask_mapping)
    assert language_model._gptqmodel_attention_mask_mapping is mask_mapping


def test_diffusion_gemma_capture_inputs_match_vision_dtype():
    model = _tiny_model()
    wrapper = object.__new__(DiffusionGemmaQModel)
    nn.Module.__init__(wrapper)
    wrapper.model = model

    example = {
        "pixel_values": torch.ones(1, 3, 2, 2, dtype=torch.float64),
        "image_position_ids": torch.tensor([0, 1]),
    }
    moved = wrapper.move_input_capture_example(example, torch.device("cpu"))

    vision_parameter = next(model.model.encoder.vision_tower.parameters())
    assert moved["pixel_values"].dtype == vision_parameter.dtype
    assert moved["pixel_values"].device == vision_parameter.device
    assert moved["image_position_ids"].shape == (1, 2)
    assert moved["image_position_ids"].device == torch.device("cpu")


def test_diffusion_gemma_lazy_source_maps_tied_encoder_and_fused_experts_to_decoder():
    # The shell captures encoder layers, but the released checkpoint owns the
    # tied text tensors under decoder.layers. Verify both the namespace remap
    # and the packed-expert split that LazyTurtle performs on demand.
    config = AutoConfig.from_pretrained(MODEL_PATH, local_files_only=True)
    with init_empty_weights(include_buffers=True):
        shell = AutoModelForMultimodalLM.from_config(config)
    _convert_model_with_defuser(DiffusionGemmaQModel, shell, cleanup_original=False)
    DiffusionGemmaQModel.after_defuser_conversion(shell)

    tied_weights = shell.model._tied_weights_keys
    assert not any(
        "encoder.language_model.layers" in target or "decoder.layers" in target
        for target in tied_weights
    )
    assert not any("layers" in target for target in shell.model.all_tied_weights_keys)
    assert not any("layers" in target for target in shell.all_tied_weights_keys)
    assert tied_weights == {
        "encoder.language_model.norm.weight": "decoder.norm.weight",
        "encoder.language_model.embed_tokens.weight": "decoder.embed_tokens.weight",
    }

    turtle = LazyTurtle.maybe_create(
        model_local_path=MODEL_PATH,
        config=config,
        model_init_kwargs={"device_map": {"": "cpu"}},
        module_tree=DiffusionGemmaQModel.module_tree,
        hf_conversion_map_reversed=DiffusionGemmaQModel.resolve_hf_conversion_map_reversed(shell),
        target_model=shell,
    )

    assert turtle is not None
    assert turtle._resolve_checkpoint_tensor_name(
        "model.encoder.language_model.layers.0.self_attn.q_proj",
        "weight",
    ) == "model.decoder.layers.0.self_attn.q_proj.weight"
    assert turtle._resolve_fused_checkpoint_tensor_source(
        "model.encoder.language_model.layers.0",
        "experts.0.gate_proj.weight",
    ) == ("model.decoder.layers.0.experts.gate_up_proj", 0, 0, 1)


def test_diffusion_gemma_refreshes_ties_when_defuser_has_no_registration(monkeypatch):
    model = _tiny_model()
    assert any("layers" in target for target in model.model._tied_weights_keys)
    monkeypatch.setattr("gptqmodel.models.loader.defuser.convert_model", lambda *args, **kwargs: False)

    assert _convert_model_with_defuser(DiffusionGemmaQModel, model, cleanup_original=False) is False
    assert not any("layers" in target for target in model.model._tied_weights_keys)
    assert not any("layers" in target for target in model.model.all_tied_weights_keys)
    assert not any("layers" in target for target in model.all_tied_weights_keys)


def test_diffusion_gemma_generic_defusion_preserves_forward():
    # Defuser only changes the experts' storage/layout; this guards the model's
    # logits while checking that the packed module became ordinary Linear leaves.
    model = _tiny_model()
    input_ids = torch.tensor([[2, 7, 8, 1]])
    decoder_input_ids = torch.tensor([[4, 5, 6, 7]])
    packed_experts = model.model.encoder.language_model.layers[0].experts

    assert hasattr(packed_experts, "gate_up_proj")
    with torch.inference_mode():
        expected = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids).logits

    assert _convert_model_with_defuser(DiffusionGemmaQModel, model, cleanup_original=False) is True
    assert not hasattr(packed_experts, "gate_up_proj")
    assert isinstance(packed_experts[0].gate_proj, nn.Linear)
    assert isinstance(packed_experts[0].up_proj, nn.Linear)
    assert isinstance(packed_experts[0].down_proj, nn.Linear)

    with torch.inference_mode():
        actual = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids).logits

    torch.testing.assert_close(actual, expected)


def test_diffusion_gemma_syncs_quantized_encoder_projection_to_decoder():
    # Quantization walks the encoder once, so generation must receive the same
    # QuantLinear object in the decoder rather than a second independently
    # quantized copy.
    model = _tiny_model()
    encoder_layer = model.model.encoder.language_model.layers[0]
    decoder_layer = model.model.decoder.layers[0]
    quantized = TorchLinear(
        bits=4,
        group_size=16,
        sym=True,
        desc_act=False,
        in_features=16,
        out_features=16,
    )
    encoder_layer.self_attn.q_proj = quantized

    assert _sync_decoder_quantized_modules(model, encoder_layer=encoder_layer) == 1
    assert decoder_layer.self_attn.q_proj is quantized


def test_diffusion_gemma_restores_dense_decoder_parameter_aliases():
    # Dense checkpoint loading can break Python-level aliases even when shapes
    # match; this is distinct from sharing a quantized module instance above.
    model = _tiny_model()
    encoder_layer = model.model.encoder.language_model.layers[0]
    decoder_layer = model.model.decoder.layers[0]
    encoder_weight = encoder_layer.self_attn.q_proj.weight
    decoder_layer.self_attn.q_proj.weight = nn.Parameter(encoder_weight.detach().clone())

    assert decoder_layer.self_attn.q_proj.weight is not encoder_weight
    assert _sync_decoder_tied_parameters(model) > 0
    assert decoder_layer.self_attn.q_proj.weight is encoder_weight


def test_diffusion_gemma_partial_quantization_can_tie_weights():
    model = _tiny_model()
    DiffusionGemmaQModel.after_defuser_conversion(model)
    model.model.encoder.language_model.layers[0].self_attn.q_proj = TorchLinear(
        bits=4,
        group_size=16,
        sym=True,
        desc_act=False,
        in_features=16,
        out_features=16,
    )

    model.tie_weights()


def test_diffusion_gemma_post_quantize_syncs_encoder_layer():
    model = _tiny_model()
    wrapper = object.__new__(DiffusionGemmaQModel)
    nn.Module.__init__(wrapper)
    wrapper.model = model
    encoder_layer = model.model.encoder.language_model.layers[0]
    quantized = TorchLinear(
        bits=4,
        group_size=16,
        sym=True,
        desc_act=False,
        in_features=16,
        out_features=16,
    )
    encoder_layer.self_attn.q_proj = quantized

    wrapper.post_quantize(encoder_layer)

    assert model.model.decoder.layers[0].self_attn.q_proj is quantized


def test_diffusion_gemma_quantized_ties_save_only_encoder_state(tmp_path):
    # Saving should emit the encoder-owned quantized tensors once, while the
    # decoder aliases remain recoverable from all_tied_weights_keys.
    model = _tiny_model()
    DiffusionGemmaQModel.after_defuser_conversion(model)
    wrapper = object.__new__(DiffusionGemmaQModel)
    nn.Module.__init__(wrapper)
    wrapper.model = model
    quantized = TorchLinear(
        bits=4,
        group_size=16,
        sym=True,
        desc_act=False,
        in_features=16,
        out_features=16,
    )
    model.model.encoder.language_model.layers[0].self_attn.q_proj = quantized

    original_state_dict = model.model.state_dict

    def reject_full_state_dict(*args, **kwargs):
        raise AssertionError("tied-weight refresh must not materialize the full LazyTurtle model")

    model.model.state_dict = reject_full_state_dict
    try:
        wrapper.after_quantize()
    finally:
        model.model.state_dict = original_state_dict

    target_prefix = "model.decoder.layers.0.self_attn.q_proj."
    source_prefix = "model.encoder.language_model.layers.0.self_attn.q_proj."
    for name in ("qweight", "qzeros", "scales", "g_idx"):
        assert model.all_tied_weights_keys[target_prefix + name] == source_prefix + name
    model.save_pretrained(tmp_path, safe_serialization=True)
    with safe_open(tmp_path / "model.safetensors", framework="pt") as checkpoint:
        checkpoint_keys = set(checkpoint.keys())

    assert source_prefix + "qweight" in checkpoint_keys
    assert target_prefix + "qweight" not in checkpoint_keys


def test_diffusion_gemma_finalizes_decoder_quantized_modules():
    model = _tiny_model()
    wrapper = object.__new__(DiffusionGemmaQModel)
    nn.Module.__init__(wrapper)
    wrapper.model = model
    encoder_layer = model.model.encoder.language_model.layers[0]
    decoder_layer = model.model.decoder.layers[0]
    quantized = TorchLinear(
        bits=4,
        group_size=16,
        sym=True,
        desc_act=False,
        in_features=16,
        out_features=16,
    )
    encoder_layer.self_attn.q_proj = quantized

    wrapper.after_quantize()

    assert decoder_layer.self_attn.q_proj is quantized


def test_diffusion_gemma_replay_refreshes_rope_for_full_attention():
    model = _tiny_model()
    wrapper = object.__new__(DiffusionGemmaQModel)
    nn.Module.__init__(wrapper)
    wrapper.model = model
    layer = model.model.encoder.language_model.layers[1]
    hidden_states = torch.randn(1, 4, 16)
    sliding_mask = torch.ones(1, 1, 4, 4, dtype=torch.bool)
    full_mask = torch.zeros(1, 1, 4, 4, dtype=torch.bool)

    refreshed = wrapper.prepare_layer_replay_kwargs(
        layer=layer,
        layer_input=[hidden_states],
        additional_inputs={
            "position_ids": torch.arange(4).unsqueeze(0),
            _DIFFUSION_GEMMA_MASKS: {
                "sliding_attention": sliding_mask,
                "full_attention": full_mask,
            },
        },
        target_device=torch.device("cpu"),
    )

    cos, sin = refreshed["position_embeddings"]
    assert cos.shape == (1, 4, 8)
    assert sin.shape == (1, 4, 8)
    assert torch.equal(refreshed["attention_mask"], full_mask)


def test_diffusion_gemma_calibration_uses_processor_chat_template():
    class RecordingProcessor:
        def apply_chat_template(self, conversations, **kwargs):
            self.conversations = conversations
            self.kwargs = kwargs
            return {"input_ids": torch.tensor([[1, 2]])}

    processor = RecordingProcessor()
    wrapper = object.__new__(DiffusionGemmaQModel)
    nn.Module.__init__(wrapper)
    wrapper.load_processor = lambda: processor
    conversations = [[{"role": "user", "content": [{"type": "text", "text": "Describe the image."}]}]]

    result = wrapper.prepare_dataset(conversations, batch_size=1)

    assert result[0]["input_ids"].shape == (1, 2)
    assert processor.conversations == conversations
    assert processor.kwargs == {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_dict": True,
        "return_tensors": "pt",
    }


def test_diffusion_gemma_specialized_generation_config_is_preserved():
    # DiffusionGemma's sampler fields are not part of the generic autoregressive
    # GenerationConfig; the config fixer must leave this subclass untouched.
    generation_config = DiffusionGemmaGenerationConfig(
        max_new_tokens=256,
        max_denoising_steps=48,
        sampler_config=EntropyBoundSamplerConfig(entropy_bound=0.1),
        t_max=0.8,
        t_min=0.4,
        confidence_threshold=0.005,
        stability_threshold=1,
    )
    model = SimpleNamespace(
        generation_config=generation_config,
        can_generate=lambda: True,
    )

    autofix_hf_model_config(model, path=MODEL_PATH)

    assert model.generation_config is generation_config
    assert isinstance(model.generation_config, DiffusionGemmaGenerationConfig)
    assert isinstance(model.generation_config.sampler_config, EntropyBoundSamplerConfig)
    assert model.generation_config.sampler_config.entropy_bound == 0.1
    assert not hasattr(model.generation_config, "do_sample")
