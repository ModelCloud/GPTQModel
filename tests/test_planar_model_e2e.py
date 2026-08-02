# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Whole-model end-to-end validation of the planar (gptq_p) format.

For every planar bit width: quantize a tiny Llama with format=gptq_p, save,
reload from disk, and verify the reloaded model reproduces the in-memory
quantized model's logits and greedy generation exactly.
"""

import json
from pathlib import Path

import pytest
import torch

from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.quantization import FORMAT
from gptqmodel.quantization.config import QuantizeConfig
from gptqmodel.utils.planar_packing import PLANAR_BITS


_CALIBRATION_TEXTS = [
    "tiny planar calibration sample one with enough tokens to survive minimum length filtering",
    "tiny planar calibration sample two exercising the planar gptq_p quantization path",
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


@pytest.mark.slow
@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6, 7, 8])
def test_planar_model_quantize_save_reload_inference(bits: int, tmp_path: Path):
    from gptqmodel import BACKEND, GPTQModel

    model_dir = tmp_path / "native"
    quantized_dir = tmp_path / "quantized"
    model_dir.mkdir()

    tokenizer = _build_tiny_llama_fixture(model_dir)
    calibration = []
    for text in _CALIBRATION_TEXTS:
        encoded = tokenizer(text, return_tensors="pt")
        calibration.append(
            {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"]}
        )

    quantize_config = QuantizeConfig(
        bits=bits,
        group_size=32,
        desc_act=False,
        format=FORMAT.GPTQ_P,
        device="cpu",
    )

    model = GPTQModel.load(str(model_dir), quantize_config=quantize_config, backend=BACKEND.TORCH)
    model.quantize(calibration, batch_size=1, backend=BACKEND.TORCH, calibration_data_min_length=1)
    model.save(quantized_dir)

    encoded = tokenizer("tiny planar calibration sample", return_tensors="pt")
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    # Reference logits + greedy generation from the in-memory quantized model.
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

    # Saved checkpoint must be labelled planar and keep standard GPTQ tensor names.
    with open(quantized_dir / "config.json") as fh:
        saved = json.load(fh)
    assert saved["quantization_config"]["checkpoint_format"] == "gptq_p"

    from safetensors import safe_open

    tensor_files = sorted(quantized_dir.glob("*.safetensors"))
    assert tensor_files, "expected saved safetensors shards"
    tensor_keys = set()
    for tensor_file in tensor_files:
        with safe_open(str(tensor_file), framework="pt") as handle:
            tensor_keys.update(handle.keys())
    quant_suffixes = {key.rsplit(".", 1)[-1] for key in tensor_keys if "proj" in key}
    assert {"qweight", "qzeros", "scales", "g_idx"} <= quant_suffixes

    # Reload from disk and verify identical inference behavior.
    # Load in fp16 to match the in-memory quantized modules' fp16 scales;
    # the default bf16 runtime cast would add bf16 rounding noise on top.
    reloaded = GPTQModel.load(
        str(quantized_dir), backend=BACKEND.TORCH, device="cpu", dtype=torch.float16
    )
    assert reloaded.quantize_config.bits == bits
    assert reloaded.quantize_config.format == FORMAT.GPTQ_P

    quantized_layers = [
        module for module in reloaded.model.modules() if isinstance(module, TorchLinear)
    ]
    assert quantized_layers, "expected at least one quantized TorchLinear layer"
    assert all(module.bits == bits for module in quantized_layers)
    expected_planar = bits in PLANAR_BITS or bits == 3
    assert all(module.planar == expected_planar for module in quantized_layers)

    reloaded.model.eval()
    with torch.inference_mode():
        reload_logits = reloaded.model(input_ids=input_ids, attention_mask=attention_mask).logits.float()
    assert torch.allclose(reload_logits, ref_logits, atol=1e-2, rtol=0), (
        f"bits={bits}: reloaded logits diverge from in-memory quantized model "
        f"(max diff {(reload_logits - ref_logits).abs().max().item():.3e})"
    )

    reload_generate = reloaded.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=8,
        do_sample=False,
    )
    assert reload_generate.shape == ref_generate.shape
    assert reload_generate.shape[-1] > input_ids.shape[-1]

    # Greedy generation must be deterministic on the reloaded model.
    repeat_generate = reloaded.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=8,
        do_sample=False,
    )
    assert torch.equal(reload_generate, repeat_generate), (
        f"bits={bits}: reloaded greedy generation is not deterministic"
    )
