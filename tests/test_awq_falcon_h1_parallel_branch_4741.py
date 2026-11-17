# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Falcon-H1's parallel norm branches must still be AWQ-scaled (#4741).

`input_layernorm` feeds attention and the Mamba mixer in parallel, so its
scaling group is `[q_proj, k_proj, v_proj, mamba.in_proj]`. The group's replay
boundary used to be `self_attn`, and `_validate_scale_group` rejected the whole
group because `mamba.in_proj` lives in the sibling `mamba` branch. Every layer
then fell back to plain quantization, so an "AWQ" Falcon-H1 checkpoint had RTN
q/k/v and RTN `mamba.in_proj`.

The replay input is the post-norm activation, so the decoder layer cannot be
the boundary (it would apply the norm twice). The group is now split: the
members the boundary covers are scored, and the sibling-branch member is
fold-only. The norm is divided once for all of its consumers, so the same
folded scale is correct for every member.
"""

import logging

import pytest
import torch
from test_awq_scaling_groups import _BASE_CONFIG, _features, _field_names, _tiny_model
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedTokenizerFast

from gptqmodel import BACKEND, CalibrationConfig, GPTQModel
from gptqmodel.looper.awq_processor import AWQProcessor
from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models.module_plan import ModulePlan
from gptqmodel.quantization.awq.utils.module import get_op_name
from gptqmodel.quantization.config import AWQConfig


def _falcon_h1_groups(layer, qmodel, tree):
    plan = ModulePlan.from_tree(tree)
    return qmodel.awq_get_modules_for_scaling(layer, _features(layer, plan), {})


def test_parallel_branch_group_scores_only_the_replay_boundary():
    """The merged norm group searches `self_attn` and folds `mamba.in_proj`."""

    _, qmodel, layers, tree = _tiny_model("falcon_h1")
    layer = layers[0]
    nodes = _falcon_h1_groups(layer, qmodel, tree)

    by_prev = {}
    for node in nodes:
        prev_op = node.get("prev_op")
        if prev_op is not None:
            by_prev[get_op_name(layer, prev_op)] = node

    group = by_prev["input_layernorm"]
    members = [get_op_name(layer, module) for module in group["layers"]]
    assert set(members) == {
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "mamba.in_proj",
    }
    assert group["_awq_allow_outside_fold"] is True, (
        "the group spans sibling branches, so the outside member must be fold-only"
    )
    scored = {get_op_name(layer, module) for module in group["_awq_search_layers"]}
    assert scored == {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"}
    assert "mamba.in_proj" not in scored
    # The Mamba SSM between `conv1d` and `out_proj` is not homogeneous, so a
    # `conv1d -> out_proj` group must not be built at all.
    assert "mamba.conv1d" not in by_prev


def _tiny_falcon_h1_checkpoint(directory):
    directory.mkdir(parents=True, exist_ok=True)
    base = AutoConfig.for_model("falcon_h1")
    fields = _field_names(base)
    kwargs = {key: value for key, value in _BASE_CONFIG.items() if key in fields}
    torch.manual_seed(0)
    AutoModelForCausalLM.from_config(
        AutoConfig.for_model("falcon_h1", **kwargs)
    ).eval().save_pretrained(directory)
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="[BOS]",
        eos_token="[EOS]",
        unk_token="[UNK]",
        pad_token="[PAD]",
    ).save_pretrained(directory)
    assert MODEL_MAP["falcon_h1"] is not None


class _RecordHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


@pytest.mark.slow
def test_awq_scales_the_parallel_branch_instead_of_falling_back(tmp_path, monkeypatch):
    """No fallback, a split search, and a non-trivial scale for `mamba.in_proj`."""

    _tiny_falcon_h1_checkpoint(tmp_path / "source")
    handler = _RecordHandler()
    root = logging.getLogger()
    root.addHandler(handler)

    calls = []
    original = AWQProcessor._search_best_scale

    def spy(self, module, prev_op, layers, inp, **kwargs):
        result = original(self, module, prev_op, layers, inp, **kwargs)
        calls.append(
            {
                "prev": get_op_name(module, prev_op),
                "scored": [get_op_name(module, layer) for layer in layers],
                "folded": [
                    get_op_name(module, layer)
                    for layer in kwargs.get("fold_layers") or layers
                ],
                "allow_outside_fold": kwargs.get("allow_outside_fold", False),
                "scales": result[2],
            }
        )
        return result

    monkeypatch.setattr(AWQProcessor, "_search_best_scale", spy)

    try:
        model = GPTQModel.load(
            str(tmp_path / "source"),
            AWQConfig(bits=4, group_size=32, device="cpu", offload_to_disk=False),
            backend=BACKEND.TORCH,
        )
        layer0 = model.model.model.layers[0]
        norm_before = layer0.input_layernorm.weight.detach().clone()
        calibration = [{"input_ids": torch.randint(0, 128, (1, 8))} for _ in range(2)]
        model.quantize(
            CalibrationConfig(dataset=calibration, batch_size=1, min_length=1),
            backend=BACKEND.TORCH,
        )
    finally:
        root.removeHandler(handler)

    fallbacks = [
        message
        for message in handler.messages
        if "plain-quantization fallback" in message and "input_layernorm" in message
    ]
    assert not fallbacks, fallbacks
    assert not any("mamba.conv1d" in message for message in handler.messages)

    groups = [call for call in calls if call["prev"] == "input_layernorm"]
    assert groups, "the input_layernorm scaling group never reached the search"
    for call in groups:
        assert call["allow_outside_fold"] is True
        assert "mamba.in_proj" in call["folded"]
        assert "mamba.in_proj" not in call["scored"]
        assert set(call["scored"]) == {
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
        }
        assert (call["scales"] - 1).abs().max() > 1e-3, "the search produced an all-ones scale"

    norm_after = layer0.input_layernorm.weight.detach()
    assert not torch.equal(norm_before, norm_after), "the fold never divided the norm gain"
