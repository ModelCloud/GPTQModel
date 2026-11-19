# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""``SUPPORTS_COMPILE`` must mean trace-clean *and* measured faster (#4739).

#4727 (the #4669 fix) opted `AwqTorchLinear` in and left `TorchFP8Linear` out.
The FP8 half was wrong on every architecture: `TorchFP8Linear.forward` branches
only on host-visible constants and its lazy decoded-weight bound is a load-time
cache, so `torch.compile(fullgraph=True)` traces it with zero graph breaks and
the compiled static-cache decode is 3x faster than eager. Leaving it out made
FP8 static-cache generation silently eager, the exact 4x regression #4669 was
supposed to remove.

The declaration now requires both halves -- a zero-graph-break `fullgraph=True`
trace *and* a measured win on the hardware it claims. These tests pin the trace
half at the module level (cheap, deterministic) and the full-model behaviour for
FP8.

`AwqTorchLinear` keeps its opt-in: since #4737 the decode-range guard checks
`torch.compiler.is_compiling()` before the `tensor_cache_key()` call, so the
forward traces clean, and a full-model static-cache decode measured 0.174 s
eager -> 0.047 s compiled (1 graph, 0 breaks) on sm_121.
"""

from __future__ import annotations

import statistics
import time

import pytest
import torch
from torch import nn

from gptqmodel import BACKEND, GPTQModel
from gptqmodel.nn_modules.qlinear.fp8 import TorchFP8Linear
from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
from gptqmodel.quantization.config import CalibrationConfig, FP8Config
from tests.fp8_test_utils import pack_fp8_module


pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="compile capture needs a CUDA device"),
]

DEVICE = "cuda:0"
K = N = 2048
M = 512

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


@pytest.fixture(autouse=True)
def _fresh_dynamo():
    """A failed ``fullgraph`` trace must not leak into the next case."""

    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def _fp8_module() -> TorchFP8Linear:
    linear = nn.Linear(K, N, bias=True, dtype=torch.float16)
    with torch.no_grad():
        linear.weight.normal_(0, 0.02)
        linear.bias.normal_(0, 0.02)
    module = TorchFP8Linear(
        bits=8,
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=K,
        out_features=N,
        bias=True,
        register_buffers=False,
        format="float8_e4m3fn",
    )
    pack_fp8_module(module, linear)
    return module.half().to(DEVICE)


def _awq_module() -> AwqTorchLinear:
    linear = nn.Linear(K, N, bias=True).half()
    with torch.no_grad():
        linear.weight.normal_(0, 0.02)
        linear.bias.normal_(0, 0.02)
    module = AwqTorchLinear(
        bits=4,
        group_size=128,
        sym=False,
        desc_act=False,
        in_features=K,
        out_features=N,
        bias=True,
        pack_dtype=torch.int32,
        register_buffers=True,
    )
    module.pack(
        linear,
        torch.full((N, K // 128), 0.02, dtype=torch.float16),
        torch.full((N, K // 128), 8.0, dtype=torch.float16),
    )
    return module.to(DEVICE)


def _median_ms(call, x, *, iters=20, warmup=3):
    with torch.inference_mode():
        for _ in range(warmup):
            call(x)
        torch.cuda.synchronize()
        samples = []
        for _ in range(5):
            start = time.perf_counter()
            for _ in range(iters):
                call(x)
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - start) / iters * 1e3)
    return statistics.median(samples)


def test_fp8_is_trace_clean_and_compiled_decode_is_faster():
    """The FP8 declaration rests on a clean full graph *and* a measured win."""

    module = _fp8_module()
    x = torch.randn(M, K, dtype=torch.float16, device=DEVICE)

    with torch.inference_mode():
        module(x)  # warms the load-time decoded-weight bound
        eager = module(x)

    explained = torch._dynamo.explain(module)(x)
    assert explained.graph_break_count == 0, [
        reason.reason for reason in explained.break_reasons
    ]

    compiled = torch.compile(module, fullgraph=True)
    with torch.inference_mode():
        output = compiled(x)
    # Inductor picks a different FP16 reduction order, so the compiled output is
    # close but not bitwise equal to eager. Tokens can therefore differ from the
    # eager path on a greedy decode (documented in the policy doc).
    torch.testing.assert_close(output, eager, rtol=0, atol=2e-2)

    eager_ms = _median_ms(module, x)
    compiled_ms = _median_ms(compiled, x)
    assert compiled_ms < eager_ms * 0.9, (
        f"compiled FP8 decode must beat eager ({compiled_ms:.3f} ms vs {eager_ms:.3f} ms); "
        "the declaration claims a measured win"
    )


def test_awq_torch_is_trace_clean():
    """`AwqTorchLinear`'s opt-in rests on a zero-graph-break full graph.

    #4737's decode-range guard returns early under `torch.compiler.is_compiling()`
    before `tensor_cache_key()` can read `.is_inference()` on the host, so the
    break that #4739 first measured is gone. A full-model static-cache decode
    measured 0.174 s eager -> 0.047 s compiled (1 graph, 0 breaks) on sm_121, so
    the kernel keeps its policy opt-in.
    """

    module = _awq_module()
    x = torch.randn(M, K, dtype=torch.float16, device=DEVICE)

    with torch.inference_mode():
        module(x)

    explained = torch._dynamo.explain(module)(x)
    assert explained.graph_break_count == 0, [
        reason.reason for reason in explained.break_reasons
    ]

    compiled = torch.compile(module, fullgraph=True)
    with torch.inference_mode():
        output = compiled(x)
    with torch.inference_mode():
        eager = module(x)
    torch.testing.assert_close(output, eager, rtol=0, atol=2e-2)


def _build_fp8_model(root, out):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.trainers import WordLevelTrainer
    from transformers import PreTrainedTokenizerFast
    from transformers.models.llama.modeling_llama import LlamaConfig, LlamaForCausalLM

    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train_from_iterator(
        _TEXTS, WordLevelTrainer(special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"])
    )
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="[BOS]",
        eos_token="[EOS]",
        unk_token="[UNK]",
        pad_token="[PAD]",
    ).save_pretrained(root)

    LlamaForCausalLM(
        LlamaConfig(
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
    ).save_pretrained(root)

    model = GPTQModel.load(str(root), quantize_config=FP8Config(device=DEVICE), backend=BACKEND.AUTO)
    model.quantize(
        CalibrationConfig(dataset=_TEXTS, batch_size=1, min_length=1), backend=BACKEND.AUTO
    )
    model.save(str(out))
    return GPTQModel.load(str(out), backend=BACKEND.FP8_TORCH, device=DEVICE)


def _generate(model, inputs, **kwargs):
    from torch._dynamo.utils import counters

    counters.clear()
    tokens = model.generate(inputs, max_new_tokens=32, min_new_tokens=32, do_sample=False, **kwargs)
    torch.cuda.synchronize()
    return tokens, dict(counters["stats"]), dict(counters["graph_break"])


def test_fp8_static_cache_keeps_the_compile_and_beats_eager(tmp_path):
    """An FP8 model must not lose the static-cache auto-compile (#4739)."""

    model = _build_fp8_model(tmp_path / "fp8", tmp_path / "fp8-quantized")
    assert type(model.model.model.layers[0].self_attn.q_proj).__name__ == "TorchFP8Linear"
    assert model._kernels_support_compile() is True

    tokenizer = model.tokenizer
    tokenizer.padding_side = "left"
    inputs = {
        name: value.to(DEVICE)
        for name, value in dict(tokenizer(_PROMPTS, return_tensors="pt", padding=True)).items()
    }

    dynamic, _, _ = _generate(model, inputs)
    dynamic, _, _ = _generate(model, inputs)  # warm lazy kernels

    eager_static, _, _ = _generate(model, inputs, cache_implementation="static", disable_compile=True)
    assert torch.equal(eager_static, dynamic), "the eager static cache must reproduce the dynamic tokens"

    compiled_static, stats, breaks = _generate(model, inputs, cache_implementation="static")
    assert stats.get("unique_graphs", 0) >= 1, "FP8 must keep the static-cache compile"
    assert not breaks, f"the FP8 declaration claims a clean trace, got {breaks}"
    # Deterministic run to run even though inductor changes the reduction order.
    again, _, _ = _generate(model, inputs, cache_implementation="static")
    assert torch.equal(compiled_static, again)

    eager_ms = _median_generate_ms(model, inputs, disable_compile=True)
    compiled_ms = _median_generate_ms(model, inputs)
    assert compiled_ms < eager_ms, (
        f"the compiled FP8 static cache must beat eager ({compiled_ms:.3f} s vs {eager_ms:.3f} s)"
    )


def _median_generate_ms(model, inputs, **kwargs):
    samples = []
    for _ in range(3):
        start = time.perf_counter()
        model.generate(
            inputs,
            max_new_tokens=32,
            min_new_tokens=32,
            do_sample=False,
            cache_implementation="static",
            **kwargs,
        )
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)
