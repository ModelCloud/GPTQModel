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

import logging
import os
from pathlib import Path


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest

from gptqmodel import BACKEND, CalibrationConfig, GPTQModel, QuantizeConfig
from gptqmodel.looper.awq_processor import AWQProcessor, _AWQScaleGroupError
from gptqmodel.quantization import FORMAT, METHOD
from gptqmodel.quantization.config import ExpertsRoutingBypass, MoEConfig
from tests.test_tiny_moe_quant_smoke import (
    _build_calibration_dataset,
    _build_tiny_qwen3_moe_fixture,
)


pytestmark = [pytest.mark.cpu, pytest.mark.slow]


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


def test_all_experts_replay_keeps_the_block_signature():
    """The bypass replay wrapper must not look like a `**kwargs` forward.

    `_prepare_module_forward` synthesizes `position_ids` for a catch-all
    forward (#4891). A bare `(hidden_states, **kwargs)` wrapper around a
    `forward(hidden_states)` MoE block was read that way, so the replay
    passed `position_ids` into Qwen3-MoE's sparse block and raised.
    """

    import inspect
    from types import SimpleNamespace

    import torch
    from torch import nn

    from gptqmodel.looper.awq_processor import _forward_accepts_var_kwargs
    from gptqmodel.models.moe_lifecycle import GateUpDownMoELifecycleHooks

    class Expert(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = nn.Linear(4, 8, bias=False)
            self.up_proj = nn.Linear(4, 8, bias=False)
            self.down_proj = nn.Linear(8, 4, bias=False)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = nn.ModuleList([Expert(), Expert()])

        def forward(self, hidden_states):
            return hidden_states * 2

    block = Block()
    calls = []
    for expert in block.experts:
        expert.gate_proj.register_forward_pre_hook(lambda module, args: calls.append(module))
    processor = AWQProcessor.__new__(AWQProcessor)
    processor.gptq_model = SimpleNamespace()
    targets = [(f"experts.{index}.gate_proj", expert.gate_proj) for index, expert in enumerate(block.experts)]
    hidden = torch.randn(3, 4)
    with processor._all_experts_replay_forward(GateUpDownMoELifecycleHooks(), block, block.experts, targets):
        assert list(inspect.signature(block.forward).parameters) == ["hidden_states"]
        assert not _forward_accepts_var_kwargs(block)
        assert processor._sanitize_kwargs({"position_ids": torch.arange(3)}, block) == {}
        torch.testing.assert_close(block(hidden), hidden * 2)
    # The wrapper really ran: every replayed expert saw the block input.
    assert calls == [expert.gate_proj for expert in block.experts]
    assert list(inspect.signature(block.forward).parameters) == ["hidden_states"]
