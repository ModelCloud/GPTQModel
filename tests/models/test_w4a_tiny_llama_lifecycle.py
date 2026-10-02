# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Local Llama quantize/save/load smoke test for both GB10 W4A policies."""

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from gptqmodel import BACKEND, GPTQModel
from gptqmodel.nn_modules.qlinear.w4a_activation import W4AActivation
from gptqmodel.nn_modules.qlinear.w4a_floatx import W4AFP8Linear
from gptqmodel.nn_modules.qlinear.w4a_nvfp4 import W4ANVFP4Linear
from gptqmodel.quantization.config import GPTAQConfig, QuantizeConfig


def _tokenizer():
    words = ("[UNK]", "[PAD]", "<s>", "</s>", "A", "short", "calibration", "sentence", "about", "fox", "Another", "moon", ".", "runs")
    inner = Tokenizer(models.WordLevel(vocab={word: i for i, word in enumerate(words)}, unk_token="[UNK]"))
    inner.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=inner, unk_token="[UNK]", pad_token="[PAD]",
        bos_token="<s>", eos_token="</s>",
    )


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (12, 1),
    reason="GB10 / SM121 required",
)
@pytest.mark.parametrize(
    "activation,recipe,backend,kernel",
    [
        ("w4afp8", None, BACKEND.GPTQ_W4AFP8, W4AFP8Linear),
        ("w4a_nvfp4", None, BACKEND.GPTQ_W4A_NVFP4, W4ANVFP4Linear),
        ("w4a_nvfp4", "nvidia_headroom", BACKEND.GPTQ_W4A_NVFP4, W4ANVFP4Linear),
    ],
)
@pytest.mark.parametrize("version", [3, 4])
@pytest.mark.parametrize("rotation", [None, "hadamard"])
def test_tiny_llama_quantize_save_reload(tmp_path, activation, recipe, backend, kernel, rotation, version, gptaq=False):
    if version == 4 and (activation != "w4a_nvfp4" or rotation is None):
        pytest.skip("Version 4 requires NVFP4 with fused rotated norms")
    torch.manual_seed(41)
    source = tmp_path / "source"
    saved = tmp_path / "quantized"
    source.mkdir()
    tokenizer = _tokenizer()
    config = LlamaConfig(
        vocab_size=32, hidden_size=128, intermediate_size=256,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=256, bos_token_id=2, eos_token_id=3, pad_token_id=1,
    )
    LlamaForCausalLM(config).save_pretrained(source)
    tokenizer.save_pretrained(source)
    qcfg = QuantizeConfig(
        bits=4, group_size=128, sym=True, desc_act=False,
        activation={"version": version, "mode": activation, **({"recipe": recipe} if recipe else {})},
        rotation=rotation, offload_to_disk=False,
        gptaq=GPTAQConfig(alpha=1.0, device="cpu") if gptaq else None,
    )
    model = GPTQModel.load(str(source), qcfg, device="cuda", dtype=torch.bfloat16)
    calibration = [
        "A short calibration sentence about a fox. A short calibration sentence about a fox.",
        "Another sentence about the moon. Another sentence about the moon.",
    ]
    model.quantize(calibration, batch_size=1, backend=backend)
    selected = [(name, module) for name, module in model.model.named_modules() if isinstance(module, kernel)]
    assert len(selected) == 14
    assert not isinstance(model.model.lm_head, kernel)
    assert not isinstance(model.model.model.layers[0].input_layernorm, kernel)
    assert all(module.qweight.dtype == torch.int32 for _, module in selected)
    if activation == "w4a_nvfp4":
        assert all(module.activation_global_scale.item() > 0 for _, module in selected)
    model.save(str(saved))
    loaded = GPTQModel.load(str(saved), backend=BACKEND.AUTO, device="cuda", dtype=torch.bfloat16)
    assert loaded.quantize_config.rotation == rotation
    if recipe:
        assert loaded.quantize_config.activation_recipe == recipe
    loaded_selected = [(name, module) for name, module in loaded.model.named_modules() if isinstance(module, kernel)]
    assert len(loaded_selected) == len(selected)
    for (name, original), (loaded_name, restored) in zip(selected, loaded_selected):
        assert name == loaded_name
        torch.testing.assert_close(restored.qweight.cpu(), original.qweight.cpu(), rtol=0, atol=0)
        if activation == "w4a_nvfp4":
            torch.testing.assert_close(restored.activation_global_scale.cpu(), original.activation_global_scale.cpu(), rtol=0, atol=0)
        assert not any(key.startswith("_weight_") for key in restored.state_dict())
    down_projections = [module for name, module in loaded_selected if name.endswith("mlp.down_proj")]
    assert len(down_projections) == 2
    assert all(module.online_full_had is (rotation is not None) for module in down_projections)
    with pytest.raises(TypeError, match="requires encoded"):
        loaded_selected[0][1](torch.zeros((1, loaded_selected[0][1].in_features),
                                      device="cuda", dtype=torch.bfloat16))
    with pytest.raises(TypeError, match="must receive encoded"):
        loaded.model.model.layers[1](torch.zeros((1, 1, 128), device="cuda", dtype=torch.bfloat16))
    ids = tokenizer("A fox runs", return_tensors="pt").input_ids.cuda()
    layer_outputs = []
    next_inputs = []
    final_inputs = []
    handles = [
        loaded.model.model.layers[0].register_forward_hook(
            lambda _module, _args, output: layer_outputs.append(output)
        ),
        loaded.model.model.layers[1].register_forward_pre_hook(
            lambda _module, args: next_inputs.append(args[0])
        ),
        loaded.model.model.norm.register_forward_pre_hook(
            lambda _module, args: final_inputs.append(args[0])
        ),
    ]
    with torch.inference_mode():
        logits = loaded.model(input_ids=ids).logits
    for handle in handles:
        handle.remove()
    assert all(isinstance(x, W4AActivation) for x in (layer_outputs[0], next_inputs[0], final_inputs[0]))
    assert layer_outputs[0].codes.data_ptr() == next_inputs[0].codes.data_ptr()
    assert layer_outputs[0].scales.data_ptr() == next_inputs[0].scales.data_ptr()
    assert final_inputs[0].mode == activation
    assert final_inputs[0].codes.dtype == (
        torch.float8_e4m3fn if activation == "w4afp8" else torch.float4_e2m1fn_x2
    )
    assert torch.isfinite(logits).all()
    assert logits.shape == (1, ids.shape[1], 32)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (12, 1),
    reason="GB10 / SM121 required",
)
def test_tiny_llama_mixed_fp8_attention_nvfp4_mlp(tmp_path):
    """A mixed stream carries FP8 on attention and NVFP4 on the MLP.

    NVIDIA's W4A4 model recipes step attention down to FP8 while the MLP keeps
    NVFP4. Both operands describe the same native GPTQ INT4 weights, so the
    split only changes the activation staging, never the saved weight format.
    """
    source = tmp_path / "source"
    saved = tmp_path / "quantized"
    source.mkdir()
    tokenizer = _tokenizer()
    config = LlamaConfig(
        vocab_size=32, hidden_size=128, intermediate_size=256,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=256, bos_token_id=2, eos_token_id=3, pad_token_id=1,
    )
    LlamaForCausalLM(config).save_pretrained(source)
    tokenizer.save_pretrained(source)
    qcfg = QuantizeConfig(
        bits=4, group_size=128, sym=True, desc_act=False,
        activation={"version": 3, "mode": "w4a_nvfp4", "attention": {"mode": "w4afp8"}},
        rotation=None, offload_to_disk=False,
    )
    model = GPTQModel.load(str(source), qcfg, device="cuda", dtype=torch.bfloat16)
    calibration = [
        "A short calibration sentence about a fox. A short calibration sentence about a fox.",
        "Another sentence about the moon. Another sentence about the moon.",
    ]
    model.quantize(calibration, batch_size=1, backend=BACKEND.GPTQ_W4A_NVFP4)
    selected = [(name, module) for name, module in model.model.named_modules()
                if isinstance(module, W4ANVFP4Linear)]
    assert len(selected) == 14
    # The INT4 weights are untouched; only the derived E4M3 staging is added.
    assert all(module.qweight.dtype == torch.int32 for _, module in selected)
    assert all(module._weight_e4m3.numel() == module.in_features * module.out_features
               for _, module in selected)
    model.save(str(saved))
    loaded = GPTQModel.load(str(saved), backend=BACKEND.AUTO, device="cuda", dtype=torch.bfloat16)
    assert loaded.quantize_config.activation_attention_mode == "w4afp8"
    assert loaded.quantize_config.activation_attention_recipe is None
    seen = {}
    layer = loaded.model.model.layers[0]
    def _record_q(_module, args):
        seen.setdefault("q", args[0].mode)

    def _record_gate(_module, args):
        seen.setdefault("gate", args[0].mode)

    handles = [
        layer.self_attn.q_proj.register_forward_pre_hook(_record_q),
        layer.mlp.gate_proj.register_forward_pre_hook(_record_gate),
    ]
    ids = tokenizer("A fox runs", return_tensors="pt").input_ids.cuda()
    with torch.inference_mode():
        logits = loaded.model(input_ids=ids).logits
    for handle in handles:
        handle.remove()
    assert seen == {"q": "w4afp8", "gate": "w4a_nvfp4"}
    assert torch.isfinite(logits).all()
    assert logits.shape == (1, ids.shape[1], 32)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (12, 1),
    reason="GB10 / SM121 required",
)
def test_tiny_llama_per_layer_mlp_fp8_override(tmp_path):
    """Per-layer MLP promotion carries FP8 on the named layers only.

    ModelOpt keeps the most sensitive blocks out of the 4-bit activation grid
    by promoting selected layers. The saved INT4 weights are identical; only
    the activation staging of the named layers changes.
    """
    source = tmp_path / "source"
    saved = tmp_path / "quantized"
    source.mkdir()
    tokenizer = _tokenizer()
    config = LlamaConfig(
        vocab_size=32, hidden_size=128, intermediate_size=256,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=256, bos_token_id=2, eos_token_id=3, pad_token_id=1,
    )
    LlamaForCausalLM(config).save_pretrained(source)
    tokenizer.save_pretrained(source)
    qcfg = QuantizeConfig(
        bits=4, group_size=128, sym=True, desc_act=False,
        activation={
            "version": 3, "mode": "w4a_nvfp4",
            "attention": {"mode": "w4afp8"},
            "mlp": {"mode": "w4afp8", "layers": [0]},
        },
        rotation=None, offload_to_disk=False,
    )
    model = GPTQModel.load(str(source), qcfg, device="cuda", dtype=torch.bfloat16)
    calibration = [
        "A short calibration sentence about a fox. A short calibration sentence about a fox.",
        "Another sentence about the moon. Another sentence about the moon.",
    ]
    model.quantize(calibration, batch_size=1, backend=BACKEND.GPTQ_W4A_NVFP4)
    selected = [module for _, module in model.model.named_modules()
                if isinstance(module, W4ANVFP4Linear)]
    assert len(selected) == 14
    assert all(module.qweight.dtype == torch.int32 for module in selected)
    model.save(str(saved))

    loaded = GPTQModel.load(str(saved), backend=BACKEND.AUTO, device="cuda", dtype=torch.bfloat16)
    assert loaded.quantize_config.activation_mlp_fp8_layers == (0,)
    seen = {}

    def _record(layer_index, kind):
        def hook(_module, args):
            seen.setdefault((layer_index, kind), set()).add(args[0].mode)
        return hook

    handles = []
    for index, layer in enumerate(loaded.model.model.layers):
        handles.append(layer.mlp.gate_proj.register_forward_pre_hook(_record(index, "gate")))
        handles.append(layer.self_attn.q_proj.register_forward_pre_hook(_record(index, "q")))
    ids = tokenizer("A fox runs", return_tensors="pt").input_ids.cuda()
    with torch.inference_mode():
        logits = loaded.model(input_ids=ids).logits
    for handle in handles:
        handle.remove()
    # Layer 0's MLP was promoted; layer 1 keeps the stream default.
    assert seen[(0, "gate")] == {"w4afp8"}
    assert seen[(1, "gate")] == {"w4a_nvfp4"}
    # Attention is unaffected by the MLP override.
    assert seen[(0, "q")] == {"w4afp8"}
    assert seen[(1, "q")] == {"w4afp8"}
    assert torch.isfinite(logits).all()
    assert logits.shape == (1, ids.shape[1], 32)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (12, 1),
    reason="GB10 / SM121 required",
)
def test_tiny_llama_v4_gptaq_retains_native_reference(tmp_path, monkeypatch):
    from gptqmodel.quantization.gptaq import GPTAQ

    original = GPTAQ.process_batch
    cross_terms = []
    def record_cross_term(self, inp):
        original(self, inp)
        cross_terms.append(float(self.dXXT.abs().max()))
    monkeypatch.setattr(GPTAQ, "process_batch", record_cross_term)
    test_tiny_llama_quantize_save_reload(
        tmp_path, "w4a_nvfp4", "least_squares", BACKEND.GPTQ_W4A_NVFP4,
        W4ANVFP4Linear, "hadamard", 4, gptaq=True,
    )
    assert len(cross_terms) >= 14
    assert all(value > 0 for value in cross_terms)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (12, 1),
    reason="GB10 / SM121 required",
)
def test_calibrated_producers_survive_standard_model_save(tmp_path):
    """Cover the ordinary loaded-model writer, not just metadata-view export."""
    from safetensors.torch import load_file

    test_tiny_llama_quantize_save_reload(
        tmp_path, "w4a_nvfp4", "least_squares", BACKEND.GPTQ_W4A_NVFP4,
        W4ANVFP4Linear, "hadamard", 4,
    )
    source = tmp_path / "quantized"
    calibrated = tmp_path / "calibrated"
    wrapper = GPTQModel.load(str(source), backend=BACKEND.GPTQ_W4A_NVFP4,
                            device="cuda", dtype=torch.bfloat16)
    wrapper.model.eval()
    samples = [torch.arange(17), torch.arange(23)]
    report = wrapper.calibrate_activations(samples)
    assert wrapper.quantize_config.activation_global_scales == report["global_scales"]
    with torch.inference_mode():
        expected = wrapper.model(input_ids=samples[0][None].cuda(), use_cache=False).logits.cpu()
    original_tensors = load_file(str(source / "model.safetensors"))
    wrapper.save(str(calibrated), max_shard_size=None)
    saved_tensors = load_file(str(calibrated / "model.safetensors"))
    assert set(saved_tensors) == set(original_tensors)
    for name in original_tensors:
        torch.testing.assert_close(saved_tensors[name], original_tensors[name], rtol=0, atol=0)
    restored = GPTQModel.load(str(calibrated), backend=BACKEND.GPTQ_W4A_NVFP4,
                             device="cuda", dtype=torch.bfloat16)
    assert restored.quantize_config.activation_global_scales == report["global_scales"]
    with torch.inference_mode():
        actual = restored.model(input_ids=samples[0][None].cuda(), use_cache=False).logits.cpu()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
