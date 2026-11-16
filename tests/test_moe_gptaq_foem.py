# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""GPTAQ/FOEM must not pair native and quantized routed-expert inputs by call order.

`NativeProcessor` records each routed expert's inputs during the full-precision
forward. During the quantized replay, earlier modules in the same layer are
already quantized, so the router sends a different token subset to each expert.
Popping the native entries by call order then pairs tensors with different row
counts (a crash) or, when the counts happen to match, rows from different tokens
(silent `dXXT` corruption).

Routed experts are now marked from the model's module plan and quantize with
`dXXT = 0` (plain GPTQ for GPTAQ, FOEM's beta term only). Experts that receive
no calibration rows fall back to data-independent RTN quantization instead of
dereferencing a missing Hessian.
"""

import math
from pathlib import Path

import pytest
import torch
from torch import nn

from gptqmodel.looper.named_module import NamedModule
from gptqmodel.looper.native_processor import NATIVE_INPUTS_STATE_KEY
from gptqmodel.quantization.config import FOEMConfig, GPTAQConfig, QuantizeConfig
from gptqmodel.quantization.foem import FOEM
from gptqmodel.quantization.gptaq import GPTAQ


def _routed_task(cls, qcfg, native_entries):
    module = NamedModule(
        nn.Linear(64, 32, bias=False),
        name="mlp.experts.0.gate_proj",
        full_name="model.layers.0.mlp.experts.0.gate_proj",
        layer_index=0,
    )
    module.state["is_routed_expert"] = True
    module.state[NATIVE_INPUTS_STATE_KEY] = native_entries
    task = cls(module, qcfg)
    task.quantizer.configure(perchannel=True)
    return task


_GPTAQ_CFG = QuantizeConfig(
    bits=4,
    group_size=32,
    desc_act=False,
    gptaq=GPTAQConfig(alpha=0.25),
)
_FOEM_CFG = QuantizeConfig(
    bits=4,
    group_size=32,
    desc_act=False,
    foem=FOEMConfig(alpha=0.25, beta=0.2),
)


@pytest.mark.parametrize(
    ("cls", "qcfg", "token_normalized"),
    [
        # GPTAQ keeps its upstream request-count Hessian recurrence and only
        # reports a token-normalized loss.
        (GPTAQ, _GPTAQ_CFG, False),
        # FOEM normalizes its Hessian by calibration token count since 60b420d
        # (the #4179 loss-denominator fix), matching GPTQ.
        (FOEM, _FOEM_CFG, True),
    ],
)
def test_routed_expert_ignores_mismatched_native_inputs(cls, qcfg, token_normalized):
    """A routed expert must build its Hessian from the quantized-path rows only."""

    torch.manual_seed(0)
    # Deliberately the wrong row count: the old code popped this and crashed.
    task = _routed_task(cls, qcfg, [torch.randn(99, 64)])
    rows = [torch.randn(3, 64) for _ in range(4)]
    for row in rows:
        task.add_batch(row, None)

    assert task.dXXT is None
    hessian = task.H.to(torch.float64)
    samples = torch.cat(rows).to(torch.float64)
    # The oracle must follow the quantizer's internal normalization: FOEM scales
    # by the number of calibration tokens (2 / total tokens), GPTAQ by requests.
    # FOEM's token normalization is the documented contract (docs/quantizer-moe.md)
    # and is what makes its Hessian, `nsamples` and reported loss match GPTQ for
    # the same rows; `2 / len(rows)` would scale with the number of calibration
    # sequences instead. `dXXT` stays on the same scale as `H`.
    denominator = sum(row.shape[0] for row in rows) if token_normalized else len(rows)
    oracle = (2.0 / denominator) * (samples.T @ samples)
    relative_error = torch.linalg.vector_norm(hessian - oracle) / torch.linalg.vector_norm(oracle)
    assert float(relative_error) <= 1e-6

    quantized = task.quantize()[0]
    assert torch.isfinite(quantized).all()


@pytest.mark.parametrize(("cls", "qcfg"), [(GPTAQ, _GPTAQ_CFG), (FOEM, _FOEM_CFG)])
def test_routed_expert_with_zero_samples_uses_rtn_fallback(cls, qcfg):
    """An expert that receives no tokens must not dereference a missing Hessian."""

    task = _routed_task(cls, qcfg, [])

    quantized, scale, zero, _g_idx, _duration, avg_loss, damp, samples = task.quantize()

    assert samples == 0
    assert damp == 0.0
    assert isinstance(avg_loss, str) and avg_loss.startswith("fallback(rtn)")
    assert torch.isfinite(quantized).all()
    assert torch.isfinite(scale).all()
    assert torch.isfinite(zero).all()


def _build_tiny_qwen2_moe(model_dir: Path):
    """Save a three-layer, 16-expert Qwen2-MoE fixture with a local tokenizer."""

    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.trainers import WordLevelTrainer
    from transformers import PreTrainedTokenizerFast
    from transformers.models.qwen2_moe.modeling_qwen2_moe import (
        Qwen2MoeConfig,
        Qwen2MoeForCausalLM,
    )

    texts = [
        "the quick brown fox jumps over the lazy dog while the sun sets behind the distant hills and the river flows quietly",
        "machine learning models require careful calibration data that covers many token patterns so every expert receives samples",
        "a second synthetic calibration example with deliberately different vocabulary to exercise routing decisions across experts",
        "quantization reduces memory usage but must preserve the numerical behaviour of the original model as closely as possible",
        "the committee reviewed the proposal and agreed that further experiments were needed before the final decision was announced",
        "routing decisions depend on the hidden state produced by the attention block which is already quantized in this layer",
        "experts that receive no calibration tokens still reach the quantizer and must not crash when the hessian is unavailable",
        "the final sample closes the calibration set and provides enough rows for the inverse hessian to remain numerically stable",
    ]

    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train_from_iterator(
        texts, trainer=WordLevelTrainer(special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"])
    )
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="[BOS]",
        eos_token="[EOS]",
        unk_token="[UNK]",
        pad_token="[PAD]",
    )
    fast_tokenizer.save_pretrained(model_dir)

    config = Qwen2MoeConfig(
        num_hidden_layers=3,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_experts=16,
        num_experts_per_tok=2,
        vocab_size=128,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        max_position_embeddings=256,
        tie_word_embeddings=False,
        decoder_sparse_step=1,
    )
    torch.manual_seed(0)
    Qwen2MoeForCausalLM(config).save_pretrained(model_dir)

    calibration = [dict(fast_tokenizer(text, return_tensors="pt")) for text in texts]
    return config, fast_tokenizer, calibration


def _quantize_and_load(model_dir: Path, output_dir: Path, calibration, qcfg, device, dtype):
    from gptqmodel import BACKEND, GPTQModel

    model = GPTQModel.load(
        str(model_dir),
        quantize_config=qcfg,
        backend=BACKEND.TORCH,
        dtype=dtype,
    )
    model.quantize(
        calibration,
        batch_size=1,
        backend=BACKEND.TORCH,
        calibration_data_min_length=1,
    )
    model.save(output_dir)
    return GPTQModel.load(str(output_dir), backend=BACKEND.TORCH, device=device)


def _perplexity(model, tokenizer, text: str) -> float:
    encoded = tokenizer(text, return_tensors="pt")
    input_ids = encoded["input_ids"].to(model.model.device)
    with torch.no_grad():
        loss = model.model(input_ids=input_ids, labels=input_ids).loss
    return float(torch.exp(loss).item())


@pytest.mark.slow
def test_tiny_qwen2_moe_gptaq_and_foem_quantize(tmp_path: Path):
    """The issue's repro: GPTAQ and FOEM must complete on a routed Qwen2-MoE."""

    from gptqmodel import QuantizeConfig

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    model_dir = tmp_path / "native"
    config, tokenizer, calibration = _build_tiny_qwen2_moe(model_dir)

    controls = {
        "gptq": QuantizeConfig(bits=4, group_size=32, desc_act=False, device=device),
        "gptaq": QuantizeConfig(
            bits=4, group_size=32, desc_act=False, device=device, gptaq=GPTAQConfig(alpha=0.25)
        ),
        "foem": QuantizeConfig(
            bits=4,
            group_size=32,
            desc_act=False,
            device=device,
            foem=FOEMConfig(alpha=0.25, beta=0.2),
        ),
    }

    loaded = {}
    for name, qcfg in controls.items():
        loaded[name] = _quantize_and_load(
            model_dir, tmp_path / name, calibration, qcfg, device, dtype
        )

    eval_text = (
        "quantization reduces memory usage but must preserve the numerical behaviour of the "
        "original model as closely as possible while the committee reviewed the proposal"
    )
    ppl = {name: _perplexity(model, tokenizer, eval_text) for name, model in loaded.items()}

    for name in ("gptaq", "foem"):
        assert math.isfinite(ppl[name])
        # The routed-expert delta is disabled, so the method must stay in the
        # same quality band as the plain GPTQ control on this fixture.
        assert abs(ppl[name] - ppl["gptq"]) / ppl["gptq"] < 0.1

    for name in ("gptaq", "foem"):
        model = loaded[name]
        expert_modules = 0
        for module_name, module in model.model.named_modules():
            if ".mlp.experts." not in module_name or not hasattr(module, "dequantize_weight"):
                continue
            expert_modules += 1
            weight = module.dequantize_weight()
            assert torch.isfinite(weight).all(), module_name
        assert expert_modules == config.num_experts * 3 * config.num_hidden_layers

