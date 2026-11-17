# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""QQQ on a BF16 model must build a kernel that can run the model dtype.

Issue #4074 item 2: on CUDA, ``QQQProcessor._quant_linear_kernel`` always chose
the FP16-only ``QQQLinear`` whatever the model dtype. Since #4001 that kernel's
``forward`` raises for non-FP16 activations, so ``quantize()`` with
``offload_to_disk=False`` produced an in-memory model whose first forward
failed. The reload path already selected ``QQQTorchLinear`` through
``SUPPORTS_DTYPES``; this test pins that both paths agree.
"""

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from gptqmodel import CalibrationConfig, GPTQModel, QQQConfig
from gptqmodel.nn_modules.qlinear.qqq import QQQLinear, QQQTorchLinear


pytestmark = [pytest.mark.gpu]

VOCAB = 256
HIDDEN = 128
INTERMEDIATE = 256


_WORDS = "the quick brown fox jumps over lazy dog".split()


def _tiny_tokenizer(directory):
    vocab = {"<unk>": 0, "<s>": 1, "</s>": 2, **{word: index + 3 for index, word in enumerate(_WORDS)}}
    backend = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<unk>",
    )
    tokenizer.save_pretrained(directory)


def _tiny_llama(directory):
    torch.manual_seed(0)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=VOCAB,
            hidden_size=HIDDEN,
            intermediate_size=INTERMEDIATE,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=128,
            pad_token_id=0,
        )
    )
    model.save_pretrained(directory)
    _tiny_tokenizer(directory)
    return directory


def _calibration():
    return ["the quick brown fox jumps over the lazy dog " * 4 for _ in range(4)]


def _kernels(model):
    modules = [module for _, module in model.named_modules()]
    fp16_only = [
        module
        for module in modules
        if isinstance(module, QQQLinear) and not isinstance(module, QQQTorchLinear)
    ]
    torch_kernels = [module for module in modules if isinstance(module, QQQTorchLinear)]
    return fp16_only, torch_kernels


def _assert_forward(model, input_ids):
    out = model.model(input_ids=input_ids, use_cache=False)
    assert out.logits.shape == (1, input_ids.shape[1], VOCAB)
    assert torch.isfinite(out.logits).all()


def test_qqq_bf16_in_memory_model_uses_the_torch_kernel(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required: CPU and NPU already select QQQTorchLinear")

    source = _tiny_llama(tmp_path / "source")
    model = GPTQModel.load(
        str(source),
        quantize_config=QQQConfig(
            bits=4,
            group_size=128,
            offload_to_disk=False,
            device="cuda:0",
        ),
        dtype=torch.bfloat16,
    )
    model.quantize(CalibrationConfig(dataset=_calibration(), batch_size=1))

    fp16_only, torch_kernels = _kernels(model.model)
    assert fp16_only == [], "BF16 QQQ must not build the FP16-only kernel"
    assert torch_kernels, "QQQ must have packed at least one linear"

    # The quantization flow leaves non-quantized modules on CPU; place the
    # in-memory model on the execution device before running it.
    model.model.to("cuda:0")
    input_ids = torch.randint(0, VOCAB, (1, 16), device="cuda:0")
    _assert_forward(model, input_ids)

    save_dir = tmp_path / "quantized"
    model.save(str(save_dir))
    del model
    torch.cuda.empty_cache()

    reloaded = GPTQModel.load(str(save_dir), device="cuda:0", dtype=torch.bfloat16)
    fp16_only, torch_kernels = _kernels(reloaded.model)
    assert fp16_only == [], "the reload path must keep the portable QQQ kernel"
    assert torch_kernels
    _assert_forward(reloaded, input_ids)
