# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Local Llama quantize/save/load smoke test for the GB10 W4AFP8 policy."""

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from gptqmodel import BACKEND, GPTQModel
from gptqmodel.nn_modules.qlinear.w4a_activation import W4AActivation
from gptqmodel.nn_modules.qlinear.w4a_floatx import W4AFP8Linear
from gptqmodel.quantization.config import QuantizeConfig


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
    ],
)
@pytest.mark.parametrize("rotation", [None, "hadamard"])
def test_tiny_llama_quantize_save_reload(tmp_path, activation, recipe, backend, kernel, rotation):
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
        activation=({"version": 3, "mode": activation, "recipe": recipe}
                    if recipe else activation),
        rotation=rotation, offload_to_disk=False,
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
        torch.float8_e4m3fn
    )
    assert torch.isfinite(logits).all()
    assert logits.shape == (1, ids.shape[1], 32)
