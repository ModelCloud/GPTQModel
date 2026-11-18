# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""#4223: bypass-mode MoE AWQ replay must feed every expert.

`MoEConfig(routing=ExpertsRoutingBypass())` makes the calibration forward enter
`ModuleLooper.MoELifecycleContext`, which feeds the block input to every expert.
AWQ replays `module2inspect` directly, so the replay has to re-enter the same
all-experts lifecycle; otherwise only natively routed experts execute, the
coverage guard rejects the whole group, and every expert gate/up falls back to
plain quantization.
"""

import inspect
import logging
import os
from pathlib import Path


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest
import torch
from torch import nn

from gptqmodel import BACKEND, CalibrationConfig, GPTQModel, QuantizeConfig
from gptqmodel.looper.awq_processor import (
    AWQProcessor,
    _AWQScaleGroupError,
    _forward_accepts_var_kwargs,
)
from gptqmodel.quantization import FORMAT, METHOD
from gptqmodel.quantization.config import ExpertsRoutingBypass, MoEConfig
from tests.test_tiny_moe_quant_smoke import (
    _build_calibration_dataset,
    _build_tiny_qwen3_moe_fixture,
)


pytestmark = [pytest.mark.cpu, pytest.mark.slow]


class _BypassExpert(nn.Module):
    def __init__(self, hidden_size: int = 8, intermediate_size: int = 16):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(
            nn.functional.silu(self.gate_proj(x)) * self.up_proj(x)
        )


class _StandardMoeBlock(nn.Module):
    """A standard MoE block whose forward accepts only ``hidden_states``.

    This is the contract ``OlmoeSparseMoeBlock`` / ``LinearQwen3MoeSparseMoeBlock``
    have, and the one AWQ's instance-level all-experts wrapper must not widen.
    """

    def __init__(self, hidden_size: int = 8, num_experts: int = 2):
        super().__init__()
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [_BypassExpert(hidden_size) for _ in range(num_experts)]
        )
        self.calls = 0
        self.last_hidden_states = None

    def forward(self, hidden_states):
        self.calls += 1
        self.last_hidden_states = hidden_states
        probs = torch.softmax(self.gate(hidden_states), dim=-1)
        routed = sum(
            expert(hidden_states) * probs[..., index : index + 1]
            for index, expert in enumerate(self.experts)
        )
        return routed


class _BypassHooks:
    gate_proj_name = "gate_proj"
    up_proj_name = "up_proj"
    down_proj_name = "down_proj"

    @staticmethod
    def apply_expert_activation(*, experts_module, expert, gate_out, up_out):
        return nn.functional.silu(gate_out) * up_out


def _bypass_processor():
    processor = AWQProcessor.__new__(AWQProcessor)
    processor.gptq_model = type("_Model", (), {})()
    processor._get_root_rotary = lambda: None
    return processor


def _bypass_targets(block: _StandardMoeBlock):
    targets = []
    for index, expert in enumerate(block.experts):
        for name in ("gate_proj", "up_proj", "down_proj"):
            targets.append((f"experts.{index}.{name}", getattr(expert, name)))
    return targets


def test_bypass_wrapper_keeps_the_block_signature_and_filters_kwargs():
    """#5095: the replay wrapper must not widen the original block's contract.

    The wrapper has to stay callable with the decoder layer's kwargs, but the
    block it replaces only declares ``hidden_states``. ``functools.wraps``
    keeps that signature visible, and the wrapper forwards only accepted keys.
    """

    processor = _bypass_processor()
    block = _StandardMoeBlock()
    targets = _bypass_targets(block)
    x = torch.randn(2, 5, 8)

    with processor._all_experts_replay_forward(
        _BypassHooks(), block, block.experts, targets
    ):
        # `inspect.signature` follows `__wrapped__`; the replay machinery must
        # not mistake this wrapper for a real `**kwargs` block.
        assert list(inspect.signature(block.forward).parameters) == ["hidden_states"]
        assert not _forward_accepts_var_kwargs(block)

        output = block(
            x,
            position_ids=torch.arange(5).unsqueeze(0),
            attention_mask=torch.ones(1, 5),
            position_embeddings=(torch.zeros(1, 5, 4), torch.zeros(1, 5, 4)),
        )

    assert block.calls == 1
    assert output.shape == x.shape
    # The all-experts replay must keep the block input's 3-D geometry; the
    # wrapper flattens only its private expert calibration copy.
    assert block.last_hidden_states.shape == x.shape


def test_bypass_wrapper_is_not_a_var_keyword_block_for_kwarg_sanitizing():
    """#5095: sanitizing and position promotion must see the real signature."""

    processor = _bypass_processor()
    block = _StandardMoeBlock()
    targets = _bypass_targets(block)
    layer_kwargs = {
        "attention_mask": torch.ones(1, 6),
        "position_ids": torch.arange(6).unsqueeze(0),
    }

    with processor._all_experts_replay_forward(
        _BypassHooks(), block, block.experts, targets
    ):
        sanitized = processor._sanitize_kwargs(layer_kwargs, block)
        prepared_x, prepared_kwargs = processor._prepare_module_forward(
            torch.randn(6, 8), block, dict(sanitized)
        )

    assert sanitized == {}
    assert prepared_kwargs == {}
    # A 2-D replay stays 2-D: `position_ids` was never part of this block's
    # contract, so promoting the activation is not required (and would crash
    # the original forward).
    assert prepared_x.shape == (6, 8)


def test_awq_bypass_replay_scales_every_expert(tmp_path: Path, monkeypatch, caplog):
    model_dir = tmp_path / "native"
    config, tokenizer = _build_tiny_qwen3_moe_fixture(
        model_dir,
        num_experts=32,
        num_experts_per_tok=1,
        moe_intermediate_size=64,
    )
    calibration = _build_calibration_dataset(tokenizer)
    # Native top-1 over these few tokens cannot route to all 32 experts; the
    # bypass lifecycle is what makes every expert observable.
    assert config.num_experts == 32
    assert config.num_experts_per_tok == 1

    raised = []
    original_search = AWQProcessor._search_best_scale

    def record_group_errors(self, *args, **kwargs):
        try:
            return original_search(self, *args, **kwargs)
        except _AWQScaleGroupError as exc:
            raised.append(str(exc))
            raise

    monkeypatch.setattr(AWQProcessor, "_search_best_scale", record_group_errors)

    quantize_config = QuantizeConfig(
        bits=4,
        group_size=32,
        desc_act=False,
        device="cpu",
        method=METHOD.AWQ,
        format=FORMAT.GEMM,
        moe=MoEConfig(routing=ExpertsRoutingBypass()),
    )
    model = GPTQModel.load(
        str(model_dir),
        quantize_config=quantize_config,
        backend=BACKEND.TORCH,
    )
    with caplog.at_level(logging.WARNING):
        model.quantize(
            CalibrationConfig(
                dataset=calibration,
                batch_size=1,
                min_length=1,
            ),
            backend=BACKEND.TORCH,
        )

    assert raised == []
    rows = {str(row["module"]): row for row in model.quant_log}
    for expert_index in range(config.num_experts):
        for suffix in ("gate_proj", "up_proj"):
            module_name = f"mlp.experts.{expert_index}.{suffix}"
            assert module_name in rows
            assert rows[module_name]["loss"] != "not applicable", module_name

    fallback_messages = [
        record.getMessage()
        for record in caplog.records
        if "plain-quantization fallback" in record.getMessage()
    ]
    assert fallback_messages == []
