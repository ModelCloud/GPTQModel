# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""A static-cache `generate()` must not compile a graph-breaking GPT-QModel kernel (#4669).

Transformers auto-compiles the forward pass of a static-cache generation. Every
default CUDA kernel here breaks the graph on each quantized linear, so capture
never covers a layer and the decode step pays for the fragments: the static
cache measured 2-4x slower than eager on Llama-1B. `GPTQModel.generate()` now
turns that auto-compile off unless the kernels declare `SUPPORTS_COMPILE`.

The check is `torch._dynamo.utils.counters`, not wall-clock: CI timing is
noisy, but a compiled frame is unambiguous. Greedy tokens must also match the
eager dynamic-cache path.
"""

from __future__ import annotations

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from transformers import PreTrainedTokenizerFast
from transformers.models.llama.modeling_llama import LlamaConfig, LlamaForCausalLM

from gptqmodel import BACKEND, GPTQModel
from gptqmodel.quantization.config import CalibrationConfig, GPTQConfig
from tests.accelerator_gates import requires_nvidia_cuda


pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not requires_nvidia_cuda(),
        reason="transformers only auto-compiles a static cache on a CUDA accelerator",
    ),
]


_TEXTS = [
    "the quick brown fox jumps over the lazy dog and then keeps running through the field",
    "a second calibration sentence that is long enough to survive the minimum length filter",
    "quantization calibration data needs more than ten tokens for this tiny model to run",
    "another example sentence with several words so that the tokenizer has enough to encode",
] * 4

_PROMPTS = [
    "the quick brown fox jumps over",
    "a second sentence here for the batch",
    "hello",
    "another longer prompt for batching",
]


def _build_fixture(root):
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train_from_iterator(
        _TEXTS, WordLevelTrainer(special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"])
    )
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="[BOS]",
        eos_token="[EOS]",
        unk_token="[UNK]",
        pad_token="[PAD]",
    )
    fast.save_pretrained(root)

    # Every projection width is a multiple of 64 so Marlin accepts the layers,
    # and every contraction axis is a multiple of the group size.
    config = LlamaConfig(
        vocab_size=256,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=2048,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
    )
    LlamaForCausalLM(config).save_pretrained(root)
    return fast


def _quantized_model(root, tmp_path):
    _build_fixture(root)
    model = GPTQModel.load(
        str(root),
        quantize_config=GPTQConfig(bits=4, group_size=128, desc_act=False, device="cuda:0"),
        backend=BACKEND.AUTO,
    )
    model.quantize(
        CalibrationConfig(dataset=_TEXTS, batch_size=1, min_length=1), backend=BACKEND.AUTO
    )
    model.save(str(tmp_path / "quantized"))
    # Marlin cannot pack raw weights; it is a load target for the checkpoint.
    return GPTQModel.load(str(tmp_path / "quantized"), backend=BACKEND.MARLIN, device="cuda:0")


def _generate(model, inputs, **kwargs):
    from torch._dynamo.utils import counters

    counters.clear()
    tokens = model.generate(
        inputs, max_new_tokens=32, min_new_tokens=32, do_sample=False, **kwargs
    )
    torch.cuda.synchronize()
    return tokens, dict(counters["stats"])


def test_static_cache_does_not_compile_a_graph_breaking_kernel(tmp_path):
    model = _quantized_model(tmp_path / "dense", tmp_path)
    tokenizer = model.tokenizer
    tokenizer.padding_side = "left"
    encoded = dict(tokenizer(_PROMPTS, return_tensors="pt", padding=True))
    inputs = {name: value.to("cuda:0") for name, value in encoded.items()}

    eager, _ = _generate(model, inputs)  # warm up lazy kernels
    eager, _ = _generate(model, inputs)

    static, stats = _generate(model, inputs, cache_implementation="static")
    assert stats == {}, f"the static-cache path compiled {stats} despite breaking kernels"
    assert torch.equal(static, eager), "the eager static cache must reproduce the dynamic tokens"

    # An explicit opt-in still compiles: the policy only changes the default.
    compiled, stats = _generate(model, inputs, cache_implementation="static", disable_compile=False)
    assert stats.get("unique_graphs", 0) > 0, "disable_compile=False must still compile"

    # The declaration that drives the policy, pinned last so the behavioural
    # assertion above is the one that fails on the unfixed tree.
    assert model._kernels_support_compile() is False
