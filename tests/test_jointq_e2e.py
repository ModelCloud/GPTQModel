# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Whole-model end-to-end validation of the JointQ refinement.

Quantize a tiny Llama with ``jointq=JointQConfig()``, save it, and reload it
from disk. The checkpoint must keep the GPTQ layout and ``meta.jointq``, the
reloaded model must reproduce the in-memory quantized logits and greedy
generation, and the refinement must change at least one packed tensor relative
to a plain GPTQ run on the same fixture.
"""

import json
from pathlib import Path

import pytest
import torch


_CALIBRATION_TEXTS = [
    "tiny jointq calibration sample one with enough tokens to survive minimum length filtering",
    "tiny jointq calibration sample two exercising the joint scale and integer refinement",
    "another synthetic calibration example that is intentionally verbose so filtering keeps it",
] * 2


def _build_tiny_llama_fixture(model_dir: Path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.trainers import WordLevelTrainer
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"])
    tokenizer.train_from_iterator(_CALIBRATION_TEXTS, trainer=trainer)
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="[BOS]",
        eos_token="[EOS]",
        unk_token="[UNK]",
        pad_token="[PAD]",
    )
    fast_tokenizer.save_pretrained(model_dir)

    config = LlamaConfig(
        num_hidden_layers=1,
        hidden_size=64,
        intermediate_size=96,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=128,
        max_position_embeddings=128,
        pad_token_id=0,
        bos_token_id=2,
        eos_token_id=3,
    )
    torch.manual_seed(0)
    model = LlamaForCausalLM(config)
    model.save_pretrained(model_dir)
    return fast_tokenizer


def _calibration(tokenizer):
    calibration = []
    for text in _CALIBRATION_TEXTS:
        encoded = tokenizer(text, return_tensors="pt")
        calibration.append(
            {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"]}
        )
    return calibration


def _quantize(model_dir: Path, output_dir: Path, jointq, calibration):
    from gptqmodel import BACKEND, CalibrationConfig, GPTQModel
    from gptqmodel.quantization.config import QuantizeConfig

    quantize_config = QuantizeConfig(
        bits=4,
        group_size=32,
        desc_act=False,
        jointq=jointq,
        device="cpu",
    )
    model = GPTQModel.load(str(model_dir), quantize_config=quantize_config, backend=BACKEND.TORCH)
    model.quantize(
        CalibrationConfig(dataset=calibration, batch_size=1, min_length=1),
        backend=BACKEND.TORCH,
    )
    model.save(output_dir)
    return model


def _packed_tensors(path: Path):
    from safetensors.torch import load_file

    return load_file(str(path / "model.safetensors"))


@pytest.mark.slow
def test_jointq_model_quantize_save_reload_inference(tmp_path: Path):
    from gptqmodel import BACKEND, GPTQModel
    from gptqmodel.quantization import JointQConfig

    model_dir = tmp_path / "native"
    gptq_dir = tmp_path / "gptq"
    jointq_dir = tmp_path / "jointq"
    model_dir.mkdir()

    tokenizer = _build_tiny_llama_fixture(model_dir)
    calibration = _calibration(tokenizer)

    # Baseline on the same fixture: JointQ must change the packed result.
    _quantize(model_dir, gptq_dir, None, calibration)

    jointq_config = JointQConfig()
    model = _quantize(model_dir, jointq_dir, jointq_config, calibration)

    encoded = tokenizer("tiny jointq calibration sample", return_tensors="pt")
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    model.model.eval()
    with torch.inference_mode():
        ref_logits = model.model(input_ids=input_ids, attention_mask=attention_mask).logits.float()
    ref_generate = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=8,
        do_sample=False,
    )
    del model

    with open(jointq_dir / "config.json") as handle:
        saved = json.load(handle)
    quantization_config = saved["quantization_config"]
    assert quantization_config["quant_method"] == "gptq"
    assert quantization_config["meta"]["jointq"] == jointq_config.meta_payload()

    gptq_tensors = _packed_tensors(gptq_dir)
    jointq_tensors = _packed_tensors(jointq_dir)
    assert set(gptq_tensors) == set(jointq_tensors)
    assert any(
        not torch.equal(gptq_tensors[key], jointq_tensors[key])
        for key in gptq_tensors
        if gptq_tensors[key].shape == jointq_tensors[key].shape
    ), "JointQ produced the same packed weights as plain GPTQ on the same fixture"

    reloaded = GPTQModel.load(str(jointq_dir), backend=BACKEND.TORCH, device="cpu", dtype=torch.float16)
    assert isinstance(reloaded.quantize_config.jointq, JointQConfig)
    assert reloaded.quantize_config.jointq == jointq_config

    reloaded.model.eval()
    with torch.inference_mode():
        reload_logits = reloaded.model(input_ids=input_ids, attention_mask=attention_mask).logits.float()
    assert torch.allclose(reload_logits, ref_logits, atol=1e-2, rtol=0), (
        "reloaded JointQ logits diverge from the in-memory quantized model "
        f"(max diff {(reload_logits - ref_logits).abs().max().item():.3e})"
    )

    reload_generate = reloaded.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=8,
        do_sample=False,
    )
    assert torch.equal(reload_generate, ref_generate)
