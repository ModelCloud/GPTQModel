# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""The live quantized model must decode the same zero points as its checkpoint.

``pack_module`` used to leave a ``REQUIRES_FORMAT_V2`` kernel holding the v1
``zero - 1`` checkpoint words. ``TorchLinear.forward`` decodes those words as
v2, so every in-process evaluation after ``quantize()`` measured a model whose
zero points were one step low, while the saved checkpoint (v1) reloaded
correctly (issue #4063). The invariant pinned here is that the in-memory model
and the reloaded checkpoint produce identical logits.
"""

from __future__ import annotations

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from gptqmodel import BACKEND, CalibrationConfig, GPTQModel
from gptqmodel.quantization import FORMAT
from gptqmodel.quantization.config import RTNConfig


pytestmark = pytest.mark.slow

_CALIBRATION_TEXTS = [
    "the quick brown fox jumps over the lazy dog and keeps running through the field",
    "a tiny llama calibration sample with enough tokens to survive length filtering",
    "another synthetic calibration example deliberately verbose so filtering keeps it",
] * 3

_MODULE = "model.layers.0.self_attn.q_proj"


def _build_tokenizer(model_dir) -> PreTrainedTokenizerFast:
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train_from_iterator(
        _CALIBRATION_TEXTS,
        WordLevelTrainer(special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"]),
    )
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="[BOS]",
        eos_token="[EOS]",
        unk_token="[UNK]",
        pad_token="[PAD]",
    )
    fast.save_pretrained(model_dir)
    return fast


def _build_fp16_llama(model_dir) -> None:
    """An FP16 source keeps the reloaded scales in FP16, so logits must match exactly."""

    torch.manual_seed(1234)
    config = LlamaConfig(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
    )
    LlamaForCausalLM(config).to(torch.float16).eval().save_pretrained(model_dir)
    _build_tokenizer(model_dir)


def _unpack_4bit_fields(words: torch.Tensor) -> torch.Tensor:
    """Independent little-endian nibble unpacker for 4-bit packed qzeros words."""

    unsigned = words.to(torch.int64) & 0xFFFFFFFF
    shifts = torch.arange(0, 32, 4, dtype=torch.int64)
    return (unsigned.unsqueeze(-1) >> shifts) & 0xF


@pytest.fixture(scope="module")
def quantized_model(tmp_path_factory):
    model_dir = tmp_path_factory.mktemp("inmemory-qzero-native")
    _build_fp16_llama(model_dir)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_dir)
    calibration = []
    for text in _CALIBRATION_TEXTS:
        encoded = tokenizer(text, return_tensors="pt")
        calibration.append(
            {
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
            }
        )

    qcfg = RTNConfig(bits=4, group_size=128, offload_to_disk=False, device="cpu")
    # Pin FP16 so the live model matches the FP16 reload below bit-for-bit;
    # without it a CPU load picks bfloat16 for the unquantized layers.
    model = GPTQModel.load(
        str(model_dir),
        quantize_config=qcfg,
        backend=BACKEND.TORCH,
        dtype=torch.float16,
    )
    model.quantize(
        CalibrationConfig(
            dataset=calibration,
            batch_size=1,
            min_length=1,
        ),
        backend=BACKEND.TORCH,
    )
    checkpoint = tmp_path_factory.mktemp("inmemory-qzero-checkpoint") / "quantized"
    model.save(str(checkpoint))
    return model, model_dir, checkpoint, tokenizer


def _logits(model, tokenizer) -> torch.Tensor:
    ids = tokenizer("the quick brown fox", return_tensors="pt")["input_ids"]
    device = model.model.get_submodule(_MODULE).qweight.device
    with torch.inference_mode():
        return model.model(input_ids=ids.to(device)).logits.float().cpu()


def test_pack_keeps_the_kernel_runtime_qzero_layout(quantized_model):
    model, _, _, _ = quantized_model
    module = model.model.get_submodule(_MODULE)

    assert getattr(type(module), "REQUIRES_FORMAT_V2", False)
    assert module.qzero_format() == 2
    assert module._checkpoint_format_on_disk == FORMAT.GPTQ


def test_state_dict_emits_the_checkpoint_qzero_layout(quantized_model):
    """Third-party readers get v1 words even though the live kernel reads v2."""

    model, _, _, _ = quantized_model
    module = model.model.get_submodule(_MODULE)
    live = _unpack_4bit_fields(module.qzeros.detach().cpu())
    serialized = _unpack_4bit_fields(model.model.state_dict()[f"{_MODULE}.qzeros"].cpu())

    # The independent oracle only needs the field relation: v1 stores zero - 1.
    assert torch.equal((serialized + 1) & 0xF, live)


def test_in_memory_logits_match_the_reloaded_checkpoint(quantized_model):
    model, _, checkpoint, tokenizer = quantized_model
    live_logits = _logits(model, tokenizer)

    reloaded = GPTQModel.load(
        str(checkpoint),
        backend=BACKEND.TORCH,
        device="cpu",
        dtype=torch.float16,
    )
    reloaded_logits = _logits(reloaded, tokenizer)

    live_module = model.model.get_submodule(_MODULE)
    reloaded_module = reloaded.model.get_submodule(_MODULE)
    assert reloaded_module.qzero_format() == 2
    assert torch.equal(live_module.qzeros.cpu(), reloaded_module.qzeros.cpu())
    assert torch.equal(live_logits, reloaded_logits)
