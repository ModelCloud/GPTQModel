# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Real Dream checkpoint ownership and AWQ scale-search qualification."""

from types import SimpleNamespace

import pytest
import torch
from awq_neutrality import assert_awq_folds_neutral

from gptqmodel import CalibrationConfig
from gptqmodel.models.definitions.dream import DreamQModel
from gptqmodel.models.loader import _convert_model_with_defuser
from gptqmodel.models.ports.dream.configuration_dream import DreamConfig
from gptqmodel.models.ports.dream.modeling_dream import DreamModel
from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
from gptqmodel.quantization.config import AWQConfig


QUANTIZED = {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
             "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"}
NORMS = {"input_layernorm", "post_attention_layernorm"}
# Audited against all 339 keys of Dream-org/Dream-v0-Instruct-7B's pinned index:
# 12 tensors/layer * 28 layers + embedding, final norm and head.
LAYER_KEYS = {name + ".weight" for name in QUANTIZED | NORMS} | {
    "self_attn.q_proj.bias", "self_attn.k_proj.bias", "self_attn.v_proj.bias"}


def model_config(kv_heads=2, **kwargs):
    return DreamConfig(vocab_size=32, hidden_size=32, intermediate_size=64,
                       num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=kv_heads,
                       pad_token_id=0, mask_token_id=31, max_position_embeddings=32, **kwargs)


@pytest.mark.parametrize("kv_heads", [1, 2, 4])
@pytest.mark.parametrize("method", ["gptq", "awq"])
def test_real_state_dict_owners_tags_roles_and_aliases(kv_heads, method):
    with torch.device("meta"):
        model = DreamModel(model_config(kv_heads))
    _convert_model_with_defuser(DreamQModel, model, cleanup_original=False)
    wrapper = DreamQModel(model=model, quantized=False, quantize_config=None)
    plan = wrapper.module_plan()
    expected = {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"} | {
        f"model.layers.{index}.{key}" for index in range(2) for key in LAYER_KEYS}
    assert set(model.state_dict()) == expected
    assert len(model.state_dict()) == 27
    blocks = DreamQModel.simple_layer_modules(model.config, SimpleNamespace(method=method, dynamic=None))
    assert {name for block in blocks for name in block} == QUANTIZED
    for layer in model.model.layers:
        for key in layer.state_dict():
            owner = key.rsplit(".", 1)[0]
            spec = plan.find(owner)
            assert spec is not None, key
            assert spec.quantize is (owner in QUANTIZED)
            assert spec.quantize or "!" in spec.flags
    for name, role in [("q_proj", "q"), ("k_proj", "k"), ("v_proj", "v"), ("o_proj", "o")]:
        assert role in plan.find("self_attn." + name).flags
    assert plan.find("self_attn.q_proj.undeclared") is None
    assert plan.find("self_attn.q_proj_extra") is None
    assert wrapper.support_batch_quantize is False


def test_published_checkpoint_meta_geometry_and_complete_key_inventory():
    config = DreamConfig(vocab_size=152064, hidden_size=3584, intermediate_size=18944,
                         num_hidden_layers=28, num_attention_heads=28, num_key_value_heads=4,
                         rope_theta=1000000.0, max_position_embeddings=131072)
    with torch.device("meta"):
        model = DreamModel(config)
    assert set(model.state_dict()) == {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"} | {
        f"model.layers.{index}.{key}" for index in range(28) for key in LAYER_KEYS}
    assert len(model.state_dict()) == 339
    assert model.model.layers[0].self_attn.k_proj.weight.shape == (512, 3584)


# Dream-v0 ships GQA (4 KV heads for 28 query heads), where v_proj -> o_proj is
# width-ineligible; equal KV heads additionally qualify that fold.
@pytest.mark.parametrize("kv_heads", [2, 4])
def test_awq_scale_folds_preserve_bidirectional_layer_output(kv_heads):
    torch.manual_seed(29)
    model = DreamModel(model_config(kv_heads)).eval()
    ids = torch.tensor([[1, 2, 3, 31, 31, 7, 9, 4]])
    mask = torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0]])

    def forward():
        # Diffusion denoising attends bidirectionally over masked positions.
        return model(ids, attention_mask=mask).logits

    expected = [("input_layernorm", ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"])]
    if kv_heads == 4:
        expected.append(("self_attn.v_proj", ["self_attn.o_proj"]))
    expected += [("post_attention_layernorm", ["mlp.gate_proj", "mlp.up_proj"]), ("mlp.up_proj", ["mlp.down_proj"])]
    for layer in model.model.layers:
        checked = assert_awq_folds_neutral(DreamQModel, model, layer, forward)
        assert [(source, targets) for source, targets, _ in checked] == expected


def test_public_awq_scale_search_packs_every_linear_and_reloads(tmp_path, monkeypatch):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    from gptqmodel import BACKEND, GPTQModel
    from gptqmodel.looper import awq_processor

    torch.manual_seed(29)
    DreamModel(model_config()).eval().save_pretrained(tmp_path)
    vocab = {"<pad>": 0, "<unk>": 1} | {f"t{index}": index for index in range(2, 31)}
    PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(vocab, unk_token="<unk>")),
                            pad_token="<pad>", unk_token="<unk>").save_pretrained(tmp_path)
    wrapper = GPTQModel.load(str(tmp_path), quantize_config=AWQConfig(device="cpu", group_size=16, fallback=None),
                             backend=BACKEND.TORCH, dtype=torch.float16)
    applied = []
    original_apply_scale = awq_processor.apply_scale

    def record_apply_scale(module, scales_list, *args, **kwargs):
        applied.extend((source, tuple(targets)) for source, targets, _scales, _loss in scales_list)
        return original_apply_scale(module, scales_list, *args, **kwargs)

    monkeypatch.setattr(awq_processor, "apply_scale", record_apply_scale)
    calibration = [{"input_ids": torch.tensor([[4, 5, 6, 7, 31, 31, 10, 11]]),
                    "attention_mask": torch.ones(1, 8, dtype=torch.long)} for _ in range(2)]
    wrapper.quantize(CalibrationConfig(dataset=calibration, batch_size=1, min_length=1), backend=BACKEND.TORCH)
    assert applied == 2 * [
        ("input_layernorm", ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")),
        ("post_attention_layernorm", ("mlp.gate_proj", "mlp.up_proj")),
        ("mlp.up_proj", ("mlp.down_proj",)),
    ]
    assert sum(isinstance(module, AwqTorchLinear) for module in wrapper.model.modules()) == 14
    ids = calibration[0]["input_ids"]
    with torch.inference_mode():
        expected = wrapper.model(ids).logits
    assert torch.isfinite(expected).all()
    saved = tmp_path / "quantized"
    wrapper.save_quantized(str(saved))
    reloaded = GPTQModel.load(str(saved), backend=BACKEND.TORCH, device="cpu", dtype=torch.float16)
    assert sum(isinstance(module, AwqTorchLinear) for module in reloaded.model.modules()) == 14
    with torch.inference_mode():
        torch.testing.assert_close(reloaded.model(ids).logits, expected, atol=2e-4, rtol=2e-3)
