# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1

"""One starved MoE expert must not send its whole gate/up group to RTN (#4675).

AWQ builds one scaling group per MoE block's routed-expert ``gate_proj`` and
``up_proj``. The group shares one scale vector, so the fallback predicate was
all-or-nothing: one expert below ``Fallback.threshold``, one expert native
routing never selects, or one ``dynamic``-excluded expert moved every expert in
the layer to plain quantization. On Qwen3-30B-A3B that silently RTNs all 256
routed gate/up projections per layer.

A starved member contributes no rows to the scale search, but the norm it reads
is still divided by the folded scale, so its weights must still be multiplied
by it. The fix drops only the starved members from the search, folds the scale
into every member, and plain-quantizes the starved ones after the fold.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from transformers import PreTrainedTokenizerFast
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeConfig,
    Qwen3MoeForCausalLM,
)

from gptqmodel import BACKEND, GPTQModel, QuantizeConfig
from gptqmodel.looper.awq_processor import AWQProcessor, _AWQLayerState
from gptqmodel.looper.loop_processor import CalibrationConfig
from gptqmodel.looper.named_module import NamedModule
from gptqmodel.quantization import FORMAT, METHOD
from gptqmodel.quantization.config import AWQConfig


pytestmark = [pytest.mark.cpu]


# ---------------------------------------------------------------------------
# Unit coverage: the group is split, not vetoed.
# ---------------------------------------------------------------------------


class _Expert(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.gate = nn.Linear(width, width)
        self.up = nn.Linear(width, width)
        self.down = nn.Linear(width, width)


class _MoELayer(nn.Module):
    def __init__(self, num_experts=3, width=4):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.block = nn.Module()
        self.block.experts = nn.ModuleList([_Expert(width) for _ in range(num_experts)])

    def forward(self, x):
        return x


def _unit_processor(layer):
    processor = AWQProcessor(
        tokenizer=None,
        qcfg=AWQConfig(device="cpu", group_size=2, sym=True, fallback={}),
        calibration=CalibrationConfig(),
        gptq_model=SimpleNamespace(rotary_embedding=None, model=layer),
        model=layer,
        calculate_w_wq_diff=False,
    )
    processor._quant_batch_size = 0
    processor.apply_clip = False
    return processor


def test_starved_member_is_folded_but_not_searched(monkeypatch):
    layer = _MoELayer(num_experts=3, width=4)
    processor = _unit_processor(layer)
    state = _AWQLayerState(
        layer_module=layer,
        modules={
            f"block.experts.{index}.gate": NamedModule(
                expert.gate,
                name=f"experts.{index}.gate",
                full_name=f"layer.block.experts.{index}.gate",
                layer_index=0,
            )
            for index, expert in enumerate(layer.block.experts)
        },
    )
    features = {
        f"block.experts.{index}.gate": torch.ones(2, 4)
        for index in range(len(layer.block.experts))
    }
    # Native routing never selected expert 2, so it captured no rows.
    features["block.experts.2.gate"] = torch.empty(0)

    monkeypatch.setattr(processor, "_layer_input_features", lambda _state: features)
    processor.gptq_model.awq_get_modules_for_scaling = lambda *_args: [
        {
            "prev_op": layer.norm,
            "layers": [expert.gate for expert in layer.block.experts],
            "inp": features["block.experts.0.gate"],
            "module2inspect": layer.block,
        }
    ]

    searched = {}
    folded = {}
    quantized = []

    def fake_search(module, prev_op, layers, inp, module2inspect=None, **kwargs):
        searched["layers"] = [get_name(layer, item) for item in layers]
        searched["fold_layers"] = [
            get_name(layer, item) for item in (kwargs.get("fold_layers") or layers)
        ]
        return (
            get_name(module, prev_op),
            tuple(searched["fold_layers"]),
            torch.ones(inp.shape[-1]),
            0.0,
        )

    def fake_apply_scale(module, scales_list, input_feat_dict=None, consumer_map=None):
        for _prev, names, _scales, _loss in scales_list:
            folded["layers"] = list(names)

    def fake_apply_quant(named_linears, scales_list):
        quantized.append((set(named_linears), list(scales_list)))

    def get_name(root, module):
        for name, candidate in root.named_modules():
            if candidate is module:
                return name
        raise AssertionError("module is not attached to the layer")

    monkeypatch.setattr(processor, "_search_best_scale", fake_search)
    monkeypatch.setattr("gptqmodel.looper.awq_processor.apply_scale", fake_apply_scale)
    monkeypatch.setattr(processor, "apply_quant", fake_apply_quant)

    processor._quantize_layer(0, state)

    assert searched["layers"] == ["block.experts.0.gate", "block.experts.1.gate"]
    # The fold still reaches the starved expert: the norm is divided by s.
    assert folded["layers"] == [
        "block.experts.0.gate",
        "block.experts.1.gate",
        "block.experts.2.gate",
    ]
    awq_names, awq_scales = quantized[0]
    assert awq_names == {"block.experts.0.gate", "block.experts.1.gate"}
    assert awq_scales
    fallback_names, fallback_scales = quantized[1]
    assert fallback_names == {"block.experts.2.gate"}
    assert fallback_scales == []
    assert state.quantized


def test_group_falls_back_only_when_every_member_is_starved(monkeypatch):
    layer = _MoELayer(num_experts=2, width=4)
    processor = _unit_processor(layer)
    state = _AWQLayerState(
        layer_module=layer,
        modules={
            f"block.experts.{index}.gate": NamedModule(
                expert.gate,
                name=f"experts.{index}.gate",
                full_name=f"layer.block.experts.{index}.gate",
                layer_index=0,
            )
            for index, expert in enumerate(layer.block.experts)
        },
    )
    features = {f"block.experts.{index}.gate": torch.empty(0) for index in range(2)}
    monkeypatch.setattr(processor, "_layer_input_features", lambda _state: features)
    processor.gptq_model.awq_get_modules_for_scaling = lambda *_args: [
        {
            "prev_op": layer.norm,
            "layers": [expert.gate for expert in layer.block.experts],
            "inp": torch.ones(2, 4),
            "module2inspect": layer.block,
        }
    ]
    monkeypatch.setattr(
        processor, "_search_best_scale", lambda *_args, **_kwargs: pytest.fail("no search")
    )
    quantized = []
    monkeypatch.setattr(
        processor, "apply_quant", lambda modules, scales_list: quantized.append(set(modules))
    )

    processor._quantize_layer(0, state)

    # No member can be searched, so the group is plain-quantized as a whole.
    assert quantized == [set(), {"block.experts.0.gate", "block.experts.1.gate"}]


# ---------------------------------------------------------------------------
# End to end: a tiny Qwen3-MoE whose router never selects one expert.
# ---------------------------------------------------------------------------


_CALIBRATION_TEXTS = [
    "tiny moe calibration sample one with enough tokens to survive minimum length filtering and trigger expert routing",
    "tiny moe calibration sample two with repeated expert words to exercise the moe quantization smoke path cleanly",
    "another synthetic calibration example that is intentionally verbose so token filtering does not remove it",
] * 2


def _build_local_tokenizer(model_dir: Path) -> PreTrainedTokenizerFast:
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
    return fast_tokenizer


def _build_tiny_fixture(model_dir: Path, *, num_experts: int = 8) -> tuple[Qwen3MoeConfig, PreTrainedTokenizerFast]:
    config = Qwen3MoeConfig(
        num_hidden_layers=1,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_experts=num_experts,
        num_experts_per_tok=2,
        vocab_size=128,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    Qwen3MoeForCausalLM(config).save_pretrained(model_dir)
    tokenizer = _build_local_tokenizer(model_dir)
    return config, tokenizer


def _starve_expert(model: torch.nn.Module, expert_index: int) -> None:
    """Add a router bias that makes ``expert_index`` unreachable.

    ``Qwen3MoeTopKRouter`` has no bias parameter, so install the equivalent:
    a constant bias of ``-1e9`` on the starved expert's logit, applied before
    the top-k. Quantization capture and the AWQ replay both call this forward.
    """

    gate = model.model.layers[0].mlp.gate

    def forward(hidden_states):
        hidden_states = hidden_states.reshape(-1, gate.hidden_dim)
        logits = torch.nn.functional.linear(hidden_states, gate.weight)
        bias = torch.zeros(gate.num_experts, dtype=logits.dtype, device=logits.device)
        bias[expert_index] = -1e9
        logits = logits + bias
        probs = torch.softmax(logits, dtype=torch.float, dim=-1)
        top_value, top_index = torch.topk(probs, gate.top_k, dim=-1)
        if gate.norm_topk_prob:
            top_value /= top_value.sum(dim=-1, keepdim=True)
        return logits, top_value.to(logits.dtype), top_index

    gate.forward = forward


@pytest.mark.slow
def test_starved_expert_keeps_awq_for_its_group(tmp_path: Path, caplog):
    model_dir = tmp_path / "native"
    config, tokenizer = _build_tiny_fixture(model_dir)
    calibration = [
        {
            "input_ids": tokenizer(text, return_tensors="pt")["input_ids"],
            "attention_mask": tokenizer(text, return_tensors="pt")["attention_mask"],
        }
        for text in _CALIBRATION_TEXTS
    ]

    quantize_config = QuantizeConfig(
        bits=4,
        group_size=32,
        desc_act=False,
        device="cpu",
        method=METHOD.AWQ,
        format=FORMAT.GEMM,
    )
    model = GPTQModel.load(str(model_dir), quantize_config=quantize_config, backend=BACKEND.TORCH)
    starved = 3
    _starve_expert(model.model, starved)

    with caplog.at_level(logging.WARNING):
        model.quantize(
            calibration,
            batch_size=1,
            backend=BACKEND.TORCH,
            calibration_data_min_length=1,
        )

    rows = {str(row["module"]): row for row in model.quant_log}
    for expert_index in range(config.num_experts):
        for suffix in ("gate_proj", "up_proj"):
            name = f"mlp.experts.{expert_index}.{suffix}"
            assert name in rows, name
            if expert_index == starved:
                # Folded, then plain-quantized because it has no rows.
                assert rows[name]["loss"] == "not applicable", name
            else:
                assert rows[name]["loss"] != "not applicable", name

    messages = [record.getMessage() for record in caplog.records]
    # The whole group must not be vetoed, and the partial split must be logged.
    assert not [message for message in messages if "plain-quantization fallback" in message], messages
    assert any(
        "keeps AWQ for" in message and "starved member(s)" in message for message in messages
    ), messages
