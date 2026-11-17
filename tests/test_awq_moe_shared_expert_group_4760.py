# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1

"""A shared-expert MoE gate/up group must search the MoE root feature (#4760).

#4720 (the fix for #4675) keeps AWQ for the covered experts of a routed-expert
gate/up group only when the group's search input is not one of its members:

    shared_search_input = cfg.get("_input_feature_name") not in layer_names

On Ling 2.0 (``bailing_moe``) the 514-member gate/up group searched
``mlp.shared_experts.gate_proj``, a member, instead of the MoE root ``mlp`` that
Qwen1.5-MoE uses. Both features hold the same post-norm activations, but the
name check vetoed the split, so one starved routed expert still sent all 514
projections to RTN in every MoE layer.

The tree declares a dense ``mlp`` block next to ``mlp:moe``. The shared-expert
and dense blocks are rooted at ``mlp``, so ``last_module_root`` is already
``mlp`` when the routed-expert block runs. The ``root != last_module_root``
gate then left ``module2inspect`` unset and anchored the group on its first
member. A MoE gate/up block always reads the MoE block input, so it now prefers
the root capture whenever one exists.
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
from transformers import BailingHybridConfig, BailingHybridForCausalLM, PreTrainedTokenizerFast

from gptqmodel import BACKEND, GPTQModel, QuantizeConfig
from gptqmodel.looper.awq_processor import AWQProcessor, _AWQLayerState
from gptqmodel.looper.named_module import NamedModule
from gptqmodel.models.definitions.bailing_moe import BailingMoeQModel
from gptqmodel.models.definitions.deepseek_v2 import DeepSeekV2QModel
from gptqmodel.models.definitions.qwen2_moe import Qwen2MoeQModel
from gptqmodel.quantization import FORMAT, METHOD
from gptqmodel.quantization.config import AWQConfig, CalibrationConfig


pytestmark = [pytest.mark.cpu]

WIDTH = 8
INTERMEDIATE = 16


# ---------------------------------------------------------------------------
# Unit coverage: the group searches the MoE root and splits, not vetoes.
# ---------------------------------------------------------------------------


class _Projection(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(WIDTH, INTERMEDIATE, bias=False)
        self.up_proj = nn.Linear(WIDTH, INTERMEDIATE, bias=False)
        self.down_proj = nn.Linear(INTERMEDIATE, WIDTH, bias=False)


class _Router(nn.Module):
    def __init__(self, rows: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(rows, WIDTH))


class _Layer(nn.Module):
    """The MoE half of one decoder layer for each definition under test."""

    def __init__(self, kind: str, num_experts: int):
        super().__init__()
        self.post_attention_layernorm = nn.LayerNorm(WIDTH)
        self.mlp = nn.Module()
        self.mlp.experts = nn.ModuleList([_Projection() for _ in range(num_experts)])
        if kind == "bailing":
            # Ling 2.0: a router, a shared expert, and the routed experts.
            self.mlp.gate = _Router(num_experts)
            self.mlp.shared_experts = _Projection()
        elif kind == "qwen2":
            self.mlp.gate = _Router(num_experts)
            self.mlp.shared_expert_gate = _Router(1)
            self.mlp.shared_expert = _Projection()
        elif kind == "deepseek":
            # DeepSeek V2 MoE layers own no dense `mlp` projections; the tree
            # declares them only for the alternative layouts.
            self.mlp.shared_experts = _Projection()
        else:  # pragma: no cover - programming error
            raise AssertionError(kind)


class _Wrapper(nn.Module):
    def __init__(self, layer: nn.Module, num_experts: int, model_type: str):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([layer])
        self.config = SimpleNamespace(
            model_type=model_type,
            num_experts=num_experts,
            num_local_experts=num_experts,
            n_routed_experts=num_experts,
            n_shared_experts=1,
        )


_DEFINITIONS = {
    "bailing": (BailingMoeQModel, "bailing_moe", "mlp.shared_experts"),
    "qwen2": (Qwen2MoeQModel, "qwen2_moe", "mlp.shared_expert"),
    "deepseek": (DeepSeekV2QModel, "deepseek_v2", "mlp.shared_experts"),
}


def _build_layer(kind: str, num_experts: int = 3):
    definition, model_type, shared_prefix = _DEFINITIONS[kind]
    layer = _Layer(kind, num_experts)
    qmodel = definition.__new__(definition)
    nn.Module.__init__(qmodel)
    qmodel.__dict__["model"] = _Wrapper(layer, num_experts, model_type)
    qmodel.__dict__["quantize_config"] = SimpleNamespace(dynamic=None)
    # `layer_modules_strict=False` lets the fake layer omit the attention
    # branches; the MoE declarations are what this test exercises.
    qmodel.__dict__["layer_modules_strict"] = False
    qmodel.__dict__["dynamic_expert_index"] = definition.dynamic_expert_index
    qmodel.__dict__["awq_scale_optimize_shape_dependent_modules"] = None
    return layer, qmodel, shared_prefix


def _features(layer, kind: str, num_experts: int, starved: int, shared_prefix: str):
    features = {"mlp": torch.randn(4, WIDTH)}
    for index in range(num_experts):
        for suffix in ("gate_proj", "up_proj", "down_proj"):
            name = f"mlp.experts.{index}.{suffix}"
            if index == starved:
                # Native routing never selected this expert.
                features[name] = torch.empty(0)
                continue
            width = INTERMEDIATE if suffix == "down_proj" else WIDTH
            features[name] = torch.randn(4, width)
    for suffix in ("gate_proj", "up_proj", "down_proj"):
        width = INTERMEDIATE if suffix == "down_proj" else WIDTH
        features[f"{shared_prefix}.{suffix}"] = torch.randn(4, width)
    return features


def _layer_state(layer, num_experts: int, shared_prefix: str) -> _AWQLayerState:
    modules = {}
    for index in range(num_experts):
        for suffix in ("gate_proj", "up_proj", "down_proj"):
            name = f"mlp.experts.{index}.{suffix}"
            modules[name] = NamedModule(
                getattr(layer.mlp.experts[index], suffix),
                name=name,
                full_name=f"layer.{name}",
                layer_index=0,
            )
    for suffix in ("gate_proj", "up_proj", "down_proj"):
        name = f"{shared_prefix}.{suffix}"
        module = layer
        for part in name.split("."):
            module = getattr(module, part)
        modules[name] = NamedModule(module, name=name, full_name=f"layer.{name}", layer_index=0)
    return _AWQLayerState(layer_module=layer, modules=modules)


def _processor(layer, qmodel) -> AWQProcessor:
    processor = AWQProcessor(
        tokenizer=None,
        qcfg=AWQConfig(device="cpu", group_size=2, sym=True, fallback={}),
        calibration=CalibrationConfig(),
        gptq_model=qmodel,
        model=layer,
        calculate_w_wq_diff=False,
    )
    processor._quant_batch_size = 0
    processor.apply_clip = False
    return processor


def _module_name(layer, module) -> str:
    for name, candidate in layer.named_modules():
        if candidate is module:
            return name
    raise AssertionError("module is not attached to the layer")


@pytest.mark.parametrize("kind", ["bailing", "qwen2", "deepseek"])
def test_shared_expert_gate_up_group_searches_the_moe_root(kind):
    num_experts, starved = 3, 2
    layer, qmodel, shared_prefix = _build_layer(kind, num_experts)
    features = _features(layer, kind, num_experts, starved, shared_prefix)

    nodes = qmodel.awq_get_modules_for_scaling(layer, features, {})
    groups = [node for node in nodes if node.get("prev_op") is layer.post_attention_layernorm]
    assert len(groups) == 1, [node.get("prev_op") for node in nodes]

    group = groups[0]
    member_names = {_module_name(layer, module) for module in group["layers"]}
    assert len(member_names) == 2 * num_experts + 2
    assert f"{shared_prefix}.gate_proj" in member_names
    assert f"{shared_prefix}.up_proj" in member_names
    # The search input is the MoE root capture, not one of the group's members,
    # so `_quantize_layer` can keep AWQ for the covered members (#4760).
    assert group["_input_feature_name"] == "mlp"
    assert group["_input_feature_name"] not in member_names


@pytest.mark.parametrize("kind", ["bailing", "qwen2", "deepseek"])
def test_shared_expert_gate_up_group_keeps_awq_for_covered_members(kind, monkeypatch):
    num_experts, starved = 3, 2
    layer, qmodel, shared_prefix = _build_layer(kind, num_experts)
    features = _features(layer, kind, num_experts, starved, shared_prefix)
    processor = _processor(layer, qmodel)
    state = _layer_state(layer, num_experts, shared_prefix)

    monkeypatch.setattr(processor, "_layer_input_features", lambda _state: features)

    starved_names = {
        f"mlp.experts.{starved}.gate_proj",
        f"mlp.experts.{starved}.up_proj",
    }
    expected_members = {
        f"mlp.experts.{index}.{suffix}"
        for index in range(num_experts)
        for suffix in ("gate_proj", "up_proj")
    } | {f"{shared_prefix}.gate_proj", f"{shared_prefix}.up_proj"}
    assert expected_members - starved_names

    searched = []
    folded = []
    quantized = []

    def fake_search(module, prev_op, layers, inp, module2inspect=None, **kwargs):
        searched.append(
            {
                "prev": _module_name(layer, prev_op),
                "layers": sorted(_module_name(layer, item) for item in layers),
                "fold": sorted(
                    _module_name(layer, item)
                    for item in (kwargs.get("fold_layers") or layers)
                ),
            }
        )
        fold_layers = kwargs.get("fold_layers") or layers
        return (
            _module_name(layer, prev_op),
            tuple(_module_name(layer, item) for item in fold_layers),
            torch.ones(inp.shape[-1]),
            0.0,
        )

    def fake_apply_scale(module, scales_list, input_feat_dict=None, consumer_map=None):
        for _prev, names, _scales, _loss in scales_list:
            folded.append(sorted(names))

    def fake_apply_quant(named_linears, scales_list):
        quantized.append((set(named_linears), list(scales_list)))

    monkeypatch.setattr(processor, "_search_best_scale", fake_search)
    monkeypatch.setattr("gptqmodel.looper.awq_processor.apply_scale", fake_apply_scale)
    monkeypatch.setattr(processor, "apply_quant", fake_apply_quant)

    processor._quantize_layer(0, state)

    norm_searches = [entry for entry in searched if entry["prev"] == "post_attention_layernorm"]
    assert len(norm_searches) == 1, searched
    # Only the starved routed member is dropped from the search.
    assert norm_searches[0]["layers"] == sorted(expected_members - starved_names)
    # The fold still reaches every member: the norm they read is divided by s.
    assert norm_searches[0]["fold"] == sorted(expected_members)
    assert sorted(expected_members) in [sorted(names) for names in folded]

    awq_names, awq_scales = quantized[0]
    fallback_names, fallback_scales = quantized[1]
    assert expected_members - starved_names <= awq_names
    assert awq_names.isdisjoint(starved_names)
    assert awq_scales
    # The starved expert alone is folded then plain-quantized. Its `down_proj`
    # group is a single-member group, so it falls back with the gate/up pair.
    assert fallback_names == starved_names | {f"mlp.experts.{starved}.down_proj"}
    assert fallback_scales == []
    assert state.quantized


# ---------------------------------------------------------------------------
# End to end: a tiny Ling-style sparse MoE whose router never selects one expert.
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


def _build_tiny_bailing(model_dir: Path, *, num_experts: int = 4, starved: int = 3):
    """A one-layer sparse `bailing_hybrid` whose router can never pick ``starved``.

    ``BailingHybrid`` shares the ``bailing_moe`` module tree, including the dense
    ``mlp`` block declared next to ``mlp:moe``. ``num_experts_per_tok`` is one
    below the expert count and the starved expert gets a large negative router
    correction bias, so the other experts are selected by every token and the
    starved one is selected by none. The bias is written before
    ``save_pretrained`` because ``GPTQModel.load`` keeps the checkpoint on the
    meta device until quantization materializes it.
    """

    config = BailingHybridConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=8,
        num_key_value_heads=8,
        head_dim=8,
        num_local_experts=num_experts,
        num_experts_per_tok=num_experts - 1,
        n_shared_experts=1,
        n_group=1,
        topk_group=1,
        first_k_dense_replace=0,
        max_position_embeddings=128,
        tie_word_embeddings=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        layer_types=["full_attention"],
        mlp_layer_types=["sparse"],
    )
    config._attn_implementation = "eager"
    torch.manual_seed(7)
    model = BailingHybridForCausalLM(config)
    with torch.no_grad():
        model.model.layers[0].mlp.gate.weight.normal_(mean=0.0, std=0.02)
    model.model.layers[0].mlp.gate.e_score_correction_bias[starved] = -1e9
    model.save_pretrained(model_dir)
    tokenizer = _build_local_tokenizer(model_dir)
    return config, tokenizer


@pytest.mark.slow
def test_bailing_moe_shared_expert_keeps_awq_for_its_group(tmp_path: Path, caplog):
    model_dir = tmp_path / "native"
    config, tokenizer = _build_tiny_bailing(model_dir)
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

    with caplog.at_level(logging.WARNING):
        model.quantize(
            CalibrationConfig(dataset=calibration, batch_size=1, min_length=1),
            backend=BACKEND.TORCH,
        )

    rows = {str(row["module"]): row for row in model.quant_log}
    for expert_index in range(config.num_local_experts):
        for suffix in ("gate_proj", "up_proj"):
            name = f"mlp.experts.{expert_index}.{suffix}"
            assert name in rows, name
            if expert_index == starved:
                # Folded, then plain-quantized because it has no rows.
                assert rows[name]["loss"] == "not applicable", name
            else:
                assert rows[name]["loss"] != "not applicable", name
    # The shared expert always executes; it must keep AWQ too.
    for suffix in ("gate_proj", "up_proj"):
        name = f"mlp.shared_experts.{suffix}"
        assert name in rows, name
        assert rows[name]["loss"] != "not applicable", name

    messages = [record.getMessage() for record in caplog.records]
    assert not [message for message in messages if "plain-quantization fallback" in message], messages
    assert any(
        "keeps AWQ for" in message and "starved member(s)" in message for message in messages
    ), messages
