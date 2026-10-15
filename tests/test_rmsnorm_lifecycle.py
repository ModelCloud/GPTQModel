# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Local CPU public-API regressions; no downloads or inference kernel stubs."""

import pytest
import torch
from safetensors.torch import load_file
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

from gptqmodel import BACKEND, GPTQModel, QuantizeConfig, QuantizeEmbed, RMSNormConfig
from gptqmodel.quantization import FORMAT


@pytest.mark.parametrize("format", [FORMAT.GPTQ, FORMAT.GPTQ_V2])
def test_fresh_quantization_and_embedding_requantization_tuning_lifecycle(tmp_path, monkeypatch, format):
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
        model.requantize(calibration, rmsnorm=RMSNormConfig(teacher="must-not-load", steps=1),
            calibration_data_min_length=1)
    model.save(str(packed))
    model = GPTQModel.load(str(packed), backend=BACKEND.TORCH, device="cpu", dtype=torch.float16)
    tune = RMSNormConfig(teacher=teacher, steps=2, lr=1e-2, reg=0.0)
    args = {"calibration": calibration, "embed_quant_mode": QuantizeEmbed.OUTPUT,
        "backend": BACKEND.TORCH, "calibration_data_min_length": 1}
    if format == FORMAT.GPTQ:
        before = {k: v.clone() for k, v in model.model.state_dict().items()}
        with pytest.raises(ValueError, match="Requantize embeddings separately"):
            model.requantize(**args, rmsnorm=tune)
        assert all(torch.equal(v, before[k]) for k, v in model.model.state_dict().items())
        model.requantize(**args)
        assert model.model.lm_head.qzero_format() == 1
        with pytest.raises(ValueError, match="reload.*BACKEND.TORCH"):
            model.requantize(calibration, rmsnorm=tune, calibration_data_min_length=1)
        model.save(str(embedded))
        model = GPTQModel.load(str(embedded), backend=BACKEND.TORCH, device="cpu", dtype=torch.float16)
    else:
        # GPTQ-v2 remains a supported combined operation.
        result = model.requantize(**args, rmsnorm=tune)
        assert result["rmsnorm"]["skipped_steps"] == 0
        model.save(str(embedded))
        model = GPTQModel.load(str(embedded), backend=BACKEND.TORCH, device="cpu", dtype=torch.float16)

    # The documented save/reload recovery must really train and remain serializable.
    result = model.requantize(calibration, rmsnorm=tune, calibration_data_min_length=1)
    assert result["rmsnorm"]["skipped_steps"] == 0
    with torch.inference_mode():
        logits = model.model(**calibration[0], use_cache=False).logits.clone()
    model.save(str(tuned))
    original, saved = load_file(str(embedded / "model.safetensors")), load_file(str(tuned / "model.safetensors"))
    assert original.keys() == saved.keys()
    allowed = {name + ".weight" for name in result["rmsnorm"]["modules"]}
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
