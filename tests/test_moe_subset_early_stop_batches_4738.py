# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Subset early-stop state must be rebuilt for every calibration batch (#4738).

#4703 replaced the declared-last ``forward_hook_last`` with
``_mark_subset_member``: the member whose hook empties the pending set flips the
stop flag, so a subset is cut off only after every member fired once, whatever
order the model calls them in (#4679).

The pending set was built once per subset pass, and the flag the completing
member set was never cleared until the pass ended. From the second calibration
batch on, the set was already empty, so the *first* member to fire flipped the
flag and stopped the forward. Every other member of the subset - every other
expert of a MoE layer - then received calibration from exactly one batch.

These tests run a real AWQ pass over a tiny Qwen3-MoE with three batches and
assert every routed expert projection saw every batch.
"""

import os
from pathlib import Path


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from transformers import PreTrainedTokenizerFast
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeConfig, Qwen3MoeForCausalLM

from gptqmodel import BACKEND, CalibrationConfig, GPTQModel
from gptqmodel.looper.awq_processor import AWQProcessor
from gptqmodel.quantization.config import AWQConfig, QuantizeConfig


pytestmark = [pytest.mark.cpu, pytest.mark.slow]


_NUM_EXPERTS = 8
_NUM_BATCHES = 3
_TEXTS = [
    "first tiny moe calibration sample with enough tokens to survive the minimum length filter and reach every expert",
    "second tiny moe calibration sample that routes the same way but is a different batch of tokens for the layer",
    "third tiny moe calibration sample so the subset early stop has to be rebuilt more than once during the pass",
]


def _build_fixture(model_dir: Path) -> PreTrainedTokenizerFast:
    model_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train_from_iterator(
        _TEXTS,
        trainer=WordLevelTrainer(special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"]),
    )
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="[BOS]",
        eos_token="[EOS]",
        unk_token="[UNK]",
        pad_token="[PAD]",
    )
    fast.save_pretrained(model_dir)

    config = Qwen3MoeConfig(
        num_hidden_layers=1,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_experts=_NUM_EXPERTS,
        # Route every expert on every token so "routed" is not a per-batch
        # accident: a member that misses a batch is a dropped capture, not a
        # router choice.
        num_experts_per_tok=_NUM_EXPERTS,
        vocab_size=128,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    Qwen3MoeForCausalLM(config).save_pretrained(model_dir)
    return fast


def _calibration(tokenizer: PreTrainedTokenizerFast) -> list[dict[str, object]]:
    dataset = []
    for text in _TEXTS:
        encoded = tokenizer(text, return_tensors="pt")
        dataset.append(
            {
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
            }
        )
    return dataset


def test_every_routed_expert_sees_every_calibration_batch(tmp_path: Path, monkeypatch):
    model_dir = tmp_path / "native"
    tokenizer = _build_fixture(model_dir)

    # `_layer_input_features` is the one place the per-module capture lists are
    # collapsed, so its snapshot is the authoritative batch count per module.
    snapshots = []
    original = AWQProcessor._layer_input_features

    def spy(self, state):
        features = original(self, state)
        snapshots.append(dict(self._feature_stats))
        return features

    monkeypatch.setattr(AWQProcessor, "_layer_input_features", spy)

    model = GPTQModel.load(
        str(model_dir),
        quantize_config=AWQConfig(
            bits=4,
            group_size=32,
            desc_act=False,
            device="cpu",
        ),
        backend=BACKEND.TORCH,
    )
    model.quantize(
        CalibrationConfig(
            dataset=_calibration(tokenizer),
            batch_size=1,
            min_length=1,
            sort=None,
        ),
        backend=BACKEND.TORCH,
    )

    assert snapshots, "AWQ must publish feature statistics"
    stats = max(
        snapshots,
        key=lambda snapshot: sum(name.startswith("mlp.experts.") for name in snapshot),
    )
    expert_names = [
        name
        for name in stats
        if name.startswith("mlp.experts.") and name.endswith(("gate_proj", "up_proj", "down_proj"))
    ]
    assert len(expert_names) == _NUM_EXPERTS * 3

    missing = {
        name: stats[name].get("batches")
        for name in expert_names
        if stats[name].get("batches") != _NUM_BATCHES
    }
    assert not missing, (
        "every routed expert projection must capture every calibration batch; "
        f"these saw fewer than {_NUM_BATCHES}: {missing}"
    )


def test_every_routed_expert_hessian_sees_every_batch(tmp_path: Path, monkeypatch):
    """The same per-batch rebuild must hold for the GPTQ Hessian path."""

    from gptqmodel.quantization.gptq import GPTQ

    model_dir = tmp_path / "native-gptq"
    tokenizer = _build_fixture(model_dir)

    # ``GPTQ.add_batch`` is where a module's Hessian accumulates one
    # calibration batch, so the set of batch indices it saw is the direct
    # evidence that the forward reached the module in that batch.
    batches = {}
    original = GPTQ.add_batch

    def spy(self, inp, out, batch_index=None):
        batches.setdefault(self.name, set()).add(batch_index)
        return original(self, inp, out, batch_index=batch_index)

    monkeypatch.setattr(GPTQ, "add_batch", spy)

    model = GPTQModel.load(
        str(model_dir),
        quantize_config=QuantizeConfig(
            bits=4,
            group_size=32,
            desc_act=False,
            device="cpu",
        ),
        backend=BACKEND.TORCH,
    )
    model.quantize(
        CalibrationConfig(
            dataset=_calibration(tokenizer),
            batch_size=1,
            min_length=1,
            sort=None,
        ),
        backend=BACKEND.TORCH,
    )

    expert_names = [
        name
        for name in batches
        if "experts." in name and name.endswith(("gate_proj", "up_proj", "down_proj"))
    ]
    # GPTQ quantizes the fused gate/up pair as one module named `gate_proj`, so
    # each expert contributes two Hessians, not three.
    assert len(expert_names) == _NUM_EXPERTS * 2

    missing = {
        name: sorted(batches[name]) for name in expert_names if len(batches[name]) != _NUM_BATCHES
    }
    assert not missing, (
        "every routed expert projection must accumulate every calibration batch; "
        f"these saw fewer than {_NUM_BATCHES}: {missing}"
    )
