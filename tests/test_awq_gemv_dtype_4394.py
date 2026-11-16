# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""AWQ GEMV dtype regression for issue #4394.

``FORMAT.GEMV`` is read by kernels that only support ``torch.float16``. A bf16
model used to load fine and only fail in layer-0 finalize, after the AWQ search
had already run. The loader must resolve the dtype before calibration: cast an
auto-selected dtype for the whole model, or reject an explicit bf16 request.
"""

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedTokenizerFast,
)

from gptqmodel import GPTQModel
from gptqmodel.models import loader
from gptqmodel.quantization.config import FORMAT, AWQConfig


def _write_tiny_llama_bf16(path):
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_hidden_layers=1,
    )
    config._attn_implementation = "eager"
    config.torch_dtype = torch.bfloat16
    torch.manual_seed(0)
    LlamaForCausalLM(config).to(torch.bfloat16).save_pretrained(path)

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            WordLevel({str(i): i for i in range(128)}, unk_token="0")
        ),
        unk_token="0",
        pad_token="0",
        eos_token="0",
    )
    tokenizer.save_pretrained(path)
    return path


def _gemv_config():
    return AWQConfig(bits=4, group_size=128, format=FORMAT.GEMV)


@pytest.mark.cpu
def test_awq_gemv_load_casts_auto_dtype_to_fp16(tmp_path, monkeypatch):
    model_path = _write_tiny_llama_bf16(tmp_path / "tiny-llama")
    messages = []
    monkeypatch.setattr(
        loader.log,
        "info",
        lambda message, *args, **kwargs: messages.append(
            message % args if args else message
        ),
    )

    model = GPTQModel.load(
        str(model_path),
        quantize_config=_gemv_config(),
        device="cpu",
        trust_remote_code=False,
    )

    # The cast must cover the whole model, including the non-quantized modules,
    # so the later AWQ forward cannot mix fp16 and bf16.
    assert {parameter.dtype for parameter in model.model.parameters()} == {
        torch.float16
    }
    assert any("Auto fix `dtype`" in message for message in messages)
    assert any("torch.float16" in message for message in messages)


@pytest.mark.cpu
def test_awq_gemv_explicit_bf16_is_rejected_before_calibration(tmp_path):
    model_path = _write_tiny_llama_bf16(tmp_path / "tiny-llama")

    with pytest.raises(ValueError, match="float16") as excinfo:
        GPTQModel.load(
            str(model_path),
            quantize_config=_gemv_config(),
            device="cpu",
            dtype=torch.bfloat16,
            trust_remote_code=False,
        )

    message = str(excinfo.value)
    assert "FORMAT.GEMV" in message
    assert "FORMAT.GEMM" in message
