# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""EXL3 must survive the default ``auto_dtype_cast=auto_nf12`` on bf16 models.

The layer inventory used to replace an AUTO_NF12 projection with its raw module
after the capture forward. ``prepare_layer_runtime_weights_for_forward`` only
asks ``NamedModule`` wrappers for deferred replay state, so EXL3's META
placeholder from the previous subset reached the next layer forward and the run
died with ``NotImplementedError: Cannot copy out of meta tensor`` (issue #4063).
The existing deferred-weight coverage used an fp32 source with
``auto_dtype_cast="off"``, which never takes the AUTO_NF12 inventory path.
"""

from __future__ import annotations

import pytest
import torch
from safetensors.torch import load_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from gptqmodel import BACKEND, CalibrationConfig, GPTQModel
from gptqmodel.quantization import AutoDtypeCast
from gptqmodel.quantization.config import EXL3Config
from gptqmodel.utils import auto_dtype_cast


pytestmark = pytest.mark.slow

_CALIBRATION_TEXTS = [
    "the quick brown fox jumps over the lazy dog and keeps running through the field",
    "a tiny llama calibration sample with enough tokens to survive length filtering",
    "another synthetic calibration example deliberately verbose so filtering keeps it",
] * 3


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


@pytest.fixture(scope="module")
def bf16_llama(tmp_path_factory):
    """Two-layer BF16 Llama; EXL3 needs every input width to be 128-aligned."""

    model_dir = tmp_path_factory.mktemp("exl3-auto-nf12-native")
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
    LlamaForCausalLM(config).to(torch.bfloat16).eval().save_pretrained(model_dir)
    _build_tokenizer(model_dir)
    return model_dir


def _require_nf12_gpu() -> None:
    if not torch.cuda.is_available():
        pytest.skip("AUTO_NF12 eligibility requires a CUDA/HIP device")
    if not auto_dtype_cast.nf12_supported_device(torch.device("cuda:0")):
        pytest.skip("AUTO_NF12 is not supported on this accelerator")


def _calibration(native):
    tokenizer = PreTrainedTokenizerFast.from_pretrained(native)
    calibration = []
    for text in _CALIBRATION_TEXTS:
        encoded = tokenizer(text, return_tensors="pt")
        calibration.append(
            {
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
            }
        )
    return calibration


def _quantize(native, output, cast: str | None):
    settings = {"bits": 4.0, "device": "cuda:0"}
    if cast is not None:
        settings["auto_dtype_cast"] = cast
    model = GPTQModel.load(
        str(native),
        quantize_config=EXL3Config(**settings),
        backend=BACKEND.AUTO,
    )
    model.quantize(
        CalibrationConfig(
            dataset=_calibration(native),
            batch_size=1,
            min_length=1,
        ),
        backend=BACKEND.AUTO,
    )
    model.save(str(output))
    return output


def _reload_logits(checkpoint, native):
    tokenizer = PreTrainedTokenizerFast.from_pretrained(native)
    ids = tokenizer("the quick brown fox", return_tensors="pt")["input_ids"].to("cuda:0")
    reloaded = GPTQModel.load(str(checkpoint), backend=BACKEND.AUTO, device="cuda:0")
    with torch.inference_mode():
        return reloaded.model(input_ids=ids).logits.float()


def _packed_tensor_names(checkpoint) -> list[str]:
    names: list[str] = []
    for shard in sorted(checkpoint.glob("*.safetensors")):
        names.extend(load_file(str(shard)).keys())
    return names


def test_default_auto_nf12_is_auto_nf12():
    assert EXL3Config(bits=4.0).auto_dtype_cast is AutoDtypeCast.AUTO_NF12


@pytest.mark.gpu
@pytest.mark.parametrize("cast", [None, "off"], ids=["default-auto-nf12", "off"])
def test_exl3_bf16_quantizes_with_the_default_source_cast(tmp_path, bf16_llama, cast):
    """A bf16 source takes the AUTO_NF12 path unless it is explicitly disabled."""

    _require_nf12_gpu()
    checkpoint = _quantize(bf16_llama, tmp_path / "quantized", cast)

    names = _packed_tensor_names(checkpoint)
    trellis = [name for name in names if name.endswith(".trellis")]
    assert trellis, "EXL3 checkpoint has no packed trellis payload"

    logits = _reload_logits(checkpoint, bf16_llama)
    assert logits.shape[0] == 1
    assert torch.isfinite(logits).all(), "reloaded EXL3 logits are not finite"
