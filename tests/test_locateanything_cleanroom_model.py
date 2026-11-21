# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Black-box contract checks for the LocateAnything port (loading, saving, validation, quantization)."""

import json
import shutil

import pytest
import torch
from test_locateanything_cleanroom_fixtures import FIXTURES, expected, load, tiny_config
from transformers.cache_utils import DynamicCache

from gptqmodel import BACKEND, GPTQModel
from gptqmodel.models.definitions.locateanything import LocateAnythingQModel
from gptqmodel.models.ports.locateanything.configuration_locateanything import LocateAnythingConfig
from gptqmodel.models.ports.locateanything.modeling_locateanything import LocateAnythingForConditionalGeneration
from gptqmodel.models.ports.native.qwen2.configuration_qwen2 import Qwen2Config
from gptqmodel.models.ports.native.qwen3.configuration_qwen3 import Qwen3Config


def _tokens(model, ids):
    with torch.no_grad():
        return model(input_ids=torch.tensor([ids])).logits


@pytest.mark.parametrize("variant", ["qwen2_gqa", "qwen2_mqa_tied", "qwen3_bias"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_save_reload_is_exact_and_safetensors_only(tmp_path, variant, dtype):
    model = LocateAnythingQModel.loader.from_pretrained(FIXTURES / variant, dtype=dtype, attn_implementation="eager")
    tied = model.config.text_config.tie_word_embeddings
    lm = model.language_model
    assert (lm.lm_head.weight is lm.model.embed_tokens.weight) is tied
    assert "language_model.lm_head.weight" in model.state_dict()
    reference = _tokens(model.eval(), [1, 5, 9, 11, 6])

    states = []
    for round_ in range(2):
        out = tmp_path / f"save{round_}"
        model.save_pretrained(out)
        files = sorted(p.name for p in out.iterdir())
        assert "model.safetensors" in files and not any(name.endswith((".bin", ".pt", ".pth")) for name in files)
        model = LocateAnythingQModel.loader.from_pretrained(out, dtype=dtype, attn_implementation="eager").eval()
        lm = model.language_model
        assert (lm.lm_head.weight is lm.model.embed_tokens.weight) is tied
        states.append({k: v.clone() for k, v in model.state_dict().items()})
        assert torch.equal(_tokens(model, [1, 5, 9, 11, 6]), reference)
    assert states[0].keys() == states[1].keys()
    for key, value in states[0].items():
        assert value.dtype == states[1][key].dtype == dtype
        assert torch.equal(value, states[1][key]), key


def test_config_types_and_validation():
    assert type(tiny_config().text_config) is Qwen2Config
    assert type(tiny_config("Qwen3ForCausalLM").text_config) is Qwen3Config
    with pytest.raises(ValueError, match="Unsupported LocateAnything decoder"):
        tiny_config("LlamaForCausalLM")
    with pytest.raises(ValueError, match="supports eager or sdpa"):
        LocateAnythingConfig(attn_implementation="flash_attention_2")
    with pytest.raises(ValueError, match="MoonViT"):
        LocateAnythingConfig(vision_config={"model_type": "siglip"})
    config = tiny_config()
    for cls in type(LocateAnythingForConditionalGeneration(config).language_model).mro():
        assert not cls.__module__.startswith("transformers.models.")


def test_model_attention_policy():
    model = LocateAnythingQModel.loader.from_pretrained(FIXTURES / "qwen2_gqa", attn_implementation="magi")
    assert model.config._attn_implementation == "sdpa"
    assert model.config.text_config._attn_implementation == "sdpa"
    with pytest.raises(ValueError, match="supports eager or sdpa"):
        LocateAnythingQModel.loader.from_pretrained(FIXTURES / "qwen2_gqa", attn_implementation="flash_attention_2")


@torch.no_grad()
def test_cache_type_and_feature_mismatch_leave_cache_untouched():
    model = load("qwen2_gqa", "sdpa")
    lm = model.language_model
    ids = torch.tensor([[1, 59, 59, 7]])
    with pytest.raises(ValueError, match="DynamicCache only"):
        lm(input_ids=ids, past_key_values=(), use_cache=True)
    cache = DynamicCache()
    with pytest.raises(ValueError, match="Image placeholder count does not match visual features"):
        lm(input_ids=ids, visual_features=torch.ones(3, 32), image_token_index=59, past_key_values=cache, use_cache=True)
    assert cache.get_seq_length() == 0
    features = torch.randn(2, 32)
    retry = lm(input_ids=ids, visual_features=features, image_token_index=59, past_key_values=cache, use_cache=True)
    fresh = lm(input_ids=ids, visual_features=features, image_token_index=59, use_cache=True)
    assert torch.equal(retry.logits, fresh.logits)
    assert retry.logits.dtype is torch.float32
    assert not torch.equal(fresh.logits, lm(input_ids=ids).logits)


def test_mixed_sliding_layers_are_rejected():
    config = tiny_config(use_sliding_window=True, sliding_window=2, layer_types=["full_attention", "sliding_attention"])
    model = LocateAnythingForConditionalGeneration(config).eval()
    with pytest.raises(NotImplementedError, match="mixed"):
        model(input_ids=torch.tensor([[1, 2, 3]]))


def test_generate_validation():
    model = load("qwen2_gqa", "sdpa")
    with pytest.raises(ValueError, match="input_ids is required"):
        model.generate(tokenizer=object(), use_cache=True)
    with pytest.raises(ValueError):
        model.generate(input_ids=torch.tensor([[1, 2]]), n_future_tokens=4, use_cache=True)
    with pytest.raises(ValueError):
        model.generate(input_ids=torch.tensor([[1, 2]]), use_cache=True)


def _checkpoint_with_hostile_code(tmp_path, variant="qwen2_gqa"):
    target = tmp_path / "ckpt"
    shutil.copytree(FIXTURES / variant, target)
    for name in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copy(FIXTURES / "processor" / name, target / name)
    marker = tmp_path / "executed"
    (target / "hostile.py").write_text(f"open({str(marker)!r}, 'w').write('ran')\nclass Hostile: pass\n")
    config = json.loads((target / "config.json").read_text())
    config["auto_map"] = {"AutoConfig": "hostile.Hostile", "AutoModel": "hostile.Hostile"}
    (target / "config.json").write_text(json.dumps(config))
    for name in (target / "expected.safetensors", target / "expected_meta.json"):
        name.unlink()
    return target, marker


@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", ["0", "1"])
def test_public_load_ignores_hostile_auto_map(tmp_path, monkeypatch, trust, bypass):
    monkeypatch.setenv("GPTQ_BYPASS_SAFETY_CHECKS", bypass)
    monkeypatch.setattr(
        "transformers.dynamic_module_utils.get_class_from_dynamic_module",
        lambda *_args, **_kwargs: pytest.fail("checkpoint Python was imported"),
    )
    target, marker = _checkpoint_with_hostile_code(tmp_path)
    loaded = GPTQModel.load(str(target), trust_remote_code=trust, device="cpu", dtype=torch.float32)
    assert type(loaded.model) is LocateAnythingForConditionalGeneration
    assert type(loaded.model.config) is LocateAnythingConfig
    assert loaded.trust_remote_code is False
    assert not marker.exists()
    reference = expected("qwen2_gqa")
    with torch.no_grad():
        logits = loaded.model(input_ids=reference["in.text_ids"]).logits
    torch.testing.assert_close(logits, reference["sdpa.text_logits"], atol=1e-5, rtol=1e-5)


@pytest.mark.cpu
@pytest.mark.parametrize("variant", ["qwen2_gqa", "qwen3_bias"])
def test_awq_quantize_save_reload(tmp_path, variant):
    from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
    from gptqmodel.quantization.config import AWQConfig, CalibrationConfig

    target, _marker = _checkpoint_with_hostile_code(tmp_path, variant)
    dense = LocateAnythingQModel.loader.from_pretrained(target, dtype=torch.float16)
    kept = {
        name: tensor.clone()
        for name, tensor in dense.state_dict().items()
        if name.startswith(("vision_model.", "mlp1.", "language_model.model.embed_tokens.", "language_model.lm_head."))
    }
    model = GPTQModel.load(
        str(target),
        quantize_config=AWQConfig(device="cpu", group_size=16),
        backend=BACKEND.TORCH,
        dtype=torch.float16,
        trust_remote_code=False,
    )
    from PIL import Image
    from transformers.tokenization_utils_tokenizers import TokenizersBackend

    from gptqmodel.models.ports.locateanything.image_processing_locateanything import LocateAnythingImageProcessor
    from gptqmodel.models.ports.locateanything.processing_locateanything import LocateAnythingProcessor

    LocateAnythingProcessor(
        LocateAnythingImageProcessor(patch_size=2), TokenizersBackend.from_pretrained(str(FIXTURES / "processor"))
    ).save_pretrained(target)
    generator = torch.Generator().manual_seed(7)
    conversations = []
    for size in ((8, 8), (8, 12)):
        pixels = torch.randint(0, 256, (*size, 3), dtype=torch.uint8, generator=generator).numpy()
        conversations.append([
            {"role": "user", "content": [{"type": "image", "image": Image.fromarray(pixels)}, {"type": "text", "text": "a b"}]},
            {"role": "assistant", "content": "c d"},
        ])
    calibration = conversations
    model.quantize(CalibrationConfig(dataset=calibration, batch_size=1, min_length=1), backend=BACKEND.TORCH)
    quantized = [name for name, module in model.model.named_modules() if isinstance(module, AwqTorchLinear)]
    assert len(quantized) == 14 and all(name.startswith("language_model.model.layers.") for name in quantized)
    state = model.model.state_dict()
    for name, tensor in kept.items():
        assert torch.equal(state[name].cpu(), tensor), name

    out = tmp_path / "quantized"
    model.save(str(out))
    reloaded = GPTQModel.load(str(out), backend=BACKEND.TORCH, device="cpu")
    reloaded_quantized = [n for n, m in reloaded.model.named_modules() if isinstance(m, AwqTorchLinear)]
    assert sorted(reloaded_quantized) == sorted(quantized)
    with torch.inference_mode():
        logits = reloaded.model(input_ids=torch.tensor([[1, 4, 6, 9, 11]])).logits
    assert torch.isfinite(logits).all()
