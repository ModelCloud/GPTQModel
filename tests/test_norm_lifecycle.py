# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Local CPU public-API regressions; no downloads or inference kernel stubs."""

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

from gptqmodel import BACKEND, GPTQModel, NormConfig, NormType, QuantizeConfig, QuantizeEmbed
from gptqmodel.quantization import FORMAT


def _checkpoint_tensors(directory, basename):
    index = directory / f"{basename}.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
        tensors = {}
        for shard in set(weight_map.values()):
            tensors.update(load_file(str(directory / shard)))
        assert set(tensors) == set(weight_map)
        for name, shard in weight_map.items():
            assert name in load_file(str(directory / shard))
        return tensors
    return load_file(str(directory / f"{basename}.safetensors"))


def _write_gpt2_source(directory):
    """Tiny GPT-2 checkpoint + tokenizer: the LayerNorm definition behind `:norm`."""
    config = GPT2Config(vocab_size=64, n_positions=32, n_embd=32, n_layer=1, n_head=2, n_inner=64)
    teacher = GPT2LMHeadModel(config).half().eval()
    teacher.save_pretrained(directory)
    tokenizer = Tokenizer(models.WordLevel(
        vocab={"[PAD]": 0, "[BOS]": 1, "[EOS]": 2, "[UNK]": 3, "hello": 4}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token="[PAD]", bos_token="[BOS]",
        eos_token="[EOS]", unk_token="[UNK]").save_pretrained(directory)
    return teacher


@pytest.mark.parametrize("layout", ["canonical", "legacy", "legacy-sharded"])
@pytest.mark.parametrize("format", [FORMAT.GPTQ, FORMAT.GPTQ_V2])
def test_fresh_quantization_and_embedding_requantization_tuning_lifecycle(tmp_path, monkeypatch, format, layout):
    # Compilation is immaterial to this CPU storage/lifecycle test.
    monkeypatch.setattr("gptqmodel.nn_modules.qlinear.torch.torch_compile", lambda fn, **kwargs: fn)
    torch.manual_seed(898)
    source, packed, embedded, tuned = [tmp_path / name for name in ("source", "packed", "embedded", "tuned")]
    config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=16,
        max_position_embeddings=64, pad_token_id=0, bos_token_id=1, eos_token_id=2,
        tie_word_embeddings=False)
    teacher = Qwen3ForCausalLM(config).half().eval()
    teacher.save_pretrained(source)
    tokenizer = Tokenizer(models.WordLevel(
        vocab={"[PAD]": 0, "[BOS]": 1, "[EOS]": 2, "[UNK]": 3, "hello": 4}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token="[PAD]", bos_token="[BOS]",
        eos_token="[EOS]", unk_token="[UNK]").save_pretrained(source)
    calibration = [{"input_ids": torch.randint(4, 64, (1, 16)),
        "attention_mask": torch.ones(1, 16, dtype=torch.long)} for _ in range(2)]
    qcfg = QuantizeConfig(bits=4, group_size=16, desc_act=False, device="cpu",
        format=format, pack_impl="original")  # Exercise the default disk-offload lifecycle.
    model = GPTQModel.load(str(source), quantize_config=qcfg, backend=BACKEND.TORCH, dtype=torch.float16)
    model.quantize(calibration, batch_size=1, backend=BACKEND.TORCH, calibration_data_min_length=1)
    assert any(t.is_meta or torch.is_inference(t) for t in model.model.parameters())
    with pytest.raises(ValueError, match="reload.*BACKEND.TORCH"):
        model.requantize(calibration, norm=NormConfig(teacher="must-not-load", steps=1),
            calibration_data_min_length=1)
    model.save(str(packed))
    basename = "model" if layout == "canonical" else "gptq_model-4bit-16g"
    if layout == "legacy":
        (packed / "model.safetensors").rename(packed / f"{basename}.safetensors")
    elif layout == "legacy-sharded":
        tensors = load_file(str(packed / "model.safetensors"))
        weight_map = {}
        for part in range(2):
            shard = f"{basename}-{part + 1:05}-of-00002.safetensors"
            values = dict(list(tensors.items())[part::2])
            save_file(values, str(packed / shard), metadata={"format": "pt"})
            weight_map.update(dict.fromkeys(values, shard))
        (packed / f"{basename}.safetensors.index.json").write_text(json.dumps({
            "metadata": {"total_size": sum(t.numel() * t.element_size() for t in tensors.values())},
            "weight_map": weight_map,
        }))
        (packed / "model.safetensors").unlink()
    model = GPTQModel.load(str(packed), backend=BACKEND.TORCH, device="cpu", dtype=torch.float16)

    # Losing the selected source after loading must fail before teacher loading
    # or any gain updates.
    resolved = Path(model.checkpoint_file_name)
    assert resolved.name.startswith(basename)
    before = {k: v.clone() for k, v in model.model.state_dict().items()}
    unavailable = resolved.with_name(resolved.name + ".unavailable")
    resolved.rename(unavailable)
    try:
        with pytest.raises(FileNotFoundError):
            model.requantize(calibration, norm=NormConfig(teacher="must-not-load", steps=2),
                calibration_data_min_length=1)
        assert all(torch.equal(v, before[k]) for k, v in model.model.state_dict().items())
        assert not getattr(model, "_module_replacement_prefixes", set())
    finally:
        unavailable.rename(resolved)

    tune = NormConfig(teacher=teacher, steps=2, lr=1e-2, reg=0.0)
    args = {"calibration": calibration, "embed_quant_mode": QuantizeEmbed.OUTPUT,
        "backend": BACKEND.TORCH, "calibration_data_min_length": 1}
    if format == FORMAT.GPTQ:
        before = {k: v.clone() for k, v in model.model.state_dict().items()}
        with pytest.raises(ValueError, match="Requantize embeddings separately"):
            model.requantize(**args, norm=tune)
        assert all(torch.equal(v, before[k]) for k, v in model.model.state_dict().items())
        model.requantize(**args)
        assert model.model.lm_head.qzero_format() == 1
        with pytest.raises(ValueError, match="reload.*BACKEND.TORCH"):
            model.requantize(calibration, norm=tune, calibration_data_min_length=1)
        model.save(str(embedded))
        model = GPTQModel.load(str(embedded), backend=BACKEND.TORCH, device="cpu", dtype=torch.float16)
    else:
        # GPTQ-v2 remains a supported combined operation.
        result = model.requantize(**args, norm=tune)
        assert result["norm"]["skipped_steps"] == 0
        model.save(str(embedded))
        model = GPTQModel.load(str(embedded), backend=BACKEND.TORCH, device="cpu", dtype=torch.float16)

    # The documented save/reload recovery must really train and remain serializable.
    result = model.requantize(calibration, norm=tune, calibration_data_min_length=1)
    assert result["norm"]["skipped_steps"] == 0
    with torch.inference_mode():
        logits = model.model(**calibration[0], use_cache=False).logits.clone()
    model.save(str(tuned))
    original, saved = _checkpoint_tensors(embedded, basename), _checkpoint_tensors(tuned, basename)
    assert original.keys() == saved.keys()
    allowed = {name + ".weight" for name in result["norm"]["modules"]}
    changed = set()
    for name, value in original.items():
        assert saved[name].dtype == value.dtype
        if not torch.equal(value, saved[name]):
            changed.add(name)
    assert changed and changed <= allowed  # All packed tensors remain byte-identical.
    reloaded = GPTQModel.load(str(tuned), backend=BACKEND.TORCH, device="cpu", dtype=torch.float16)
    with torch.inference_mode():
        restored = reloaded.model(**calibration[0], use_cache=False).logits
    assert torch.equal(logits, restored)


def test_layer_norm_tuning_through_requantize_on_quantized_gpt2(tmp_path, monkeypatch):
    """`NormType.LAYERNORM` trains weight+bias through the public requantize API."""
    monkeypatch.setattr("gptqmodel.nn_modules.qlinear.torch.torch_compile", lambda fn, **kwargs: fn)
    torch.manual_seed(1312)
    source, packed, tuned = [tmp_path / name for name in ("source", "packed", "tuned")]
    teacher = _write_gpt2_source(source)
    calibration = [{"input_ids": torch.randint(4, 64, (1, 16)),
        "attention_mask": torch.ones(1, 16, dtype=torch.long)} for _ in range(2)]
    qcfg = QuantizeConfig(bits=4, group_size=16, desc_act=False, device="cpu",
        format=FORMAT.GPTQ_V2, pack_impl="original")
    model = GPTQModel.load(str(source), quantize_config=qcfg, backend=BACKEND.TORCH, dtype=torch.float16)
    model.quantize(calibration, batch_size=1, backend=BACKEND.TORCH, calibration_data_min_length=1)
    model.save(str(packed))
    model = GPTQModel.load(str(packed), backend=BACKEND.TORCH, device="cpu", dtype=torch.float16)

    names = model.norm_module_names((NormType.LAYERNORM,))
    assert names == ["transformer.h.0.ln_1", "transformer.h.0.ln_2", "transformer.ln_f"]
    # The first-release default stays RMSNorm-only, so this definition has nothing to tune.
    with pytest.raises(ValueError, match="declares no `:norm` modules matching"):
        model.requantize(calibration, norm=NormConfig(teacher=teacher, steps=1),
            calibration_data_min_length=1)

    before = _checkpoint_tensors(packed, "model")
    result = model.requantize(calibration, norm=NormConfig(teacher=teacher, types=(NormType.LAYERNORM,),
        steps=2, lr=1e-2, reg=0.0), calibration_data_min_length=1)
    assert result["norm"]["types"] == ["layernorm"]
    assert result["norm"]["skipped_steps"] == 0
    expected = {f"{name}.{kind}" for name in names for kind in ("weight", "bias")}
    assert set(result["norm"]["params"]) == expected

    model.save(str(tuned))
    after = _checkpoint_tensors(tuned, "model")
    changed = {name for name, value in before.items() if not torch.equal(after[name], value)}
    assert changed == expected  # Every packed quantized tensor stays byte-identical.

    with torch.inference_mode():
        logits = model.model(**calibration[0], use_cache=False).logits.clone()
    reloaded = GPTQModel.load(str(tuned), backend=BACKEND.TORCH, device="cpu", dtype=torch.float16)
    with torch.inference_mode():
        restored = reloaded.model(**calibration[0], use_cache=False).logits
    assert torch.equal(logits, restored)
