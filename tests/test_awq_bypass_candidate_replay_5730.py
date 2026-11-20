# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""#5730: bypass-mode candidate replays must not run every expert.

`MoEConfig(routing=ExpertsRoutingBypass())` needs the all-experts lifecycle for
the AWQ reference pass so every member of a routed group is observed (#4223).
That observation ends before the candidate scale grid, but the wrapper used to
stay installed for all 20 candidates: every expert gate/up ran on every replay
token and the result was discarded. Candidates must keep native routing plus
the captured top-k override instead.
"""

import os
from pathlib import Path


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest

from gptqmodel import BACKEND, CalibrationConfig, GPTQModel, QuantizeConfig
from gptqmodel.looper.awq_processor import AWQProcessor
from gptqmodel.quantization import FORMAT, METHOD
from gptqmodel.quantization.config import ExpertsRoutingBypass, MoEConfig
from tests.test_tiny_moe_quant_smoke import (
    _build_calibration_dataset,
    _build_tiny_qwen3_moe_fixture,
)


pytestmark = [pytest.mark.cpu, pytest.mark.slow]

NUM_EXPERTS = 8


def test_bypass_candidate_grid_keeps_native_routing(tmp_path: Path, monkeypatch):
    model_dir = tmp_path / "native"
    config, tokenizer = _build_tiny_qwen3_moe_fixture(
        model_dir,
        num_experts=NUM_EXPERTS,
        num_experts_per_tok=1,
        moe_intermediate_size=32,
    )
    calibration = _build_calibration_dataset(tokenizer)
    quantize_config = QuantizeConfig(
        bits=4,
        group_size=16,
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

    # A `_compute_best_scale` call runs the whole candidate grid over one scale
    # group. Count the expert gate rows and the rows entering the MoE block
    # during that call: native top-1 routing observes one expert row per block
    # row, while the all-experts wrapper observes one per expert. The live
    # expert modules are resolved from the replay boundary of each call because
    # the lazy loader materializes (and replaces) them during `quantize()`.
    counters = {"expert_rows": 0, "moe_rows": 0}
    calls = []
    original_compute = AWQProcessor._compute_best_scale

    def count_rows(module, args):
        del module
        if args:
            counters["expert_rows"] += int(args[0].numel() // args[0].shape[-1])

    def count_moe_rows(module, args):
        del module
        if args:
            counters["moe_rows"] += int(args[0].numel() // args[0].shape[-1])

    def counting_compute(self, *args, **kwargs):
        inspect_module = args[3]
        experts = getattr(inspect_module, "experts", None)
        counters["expert_rows"] = 0
        counters["moe_rows"] = 0
        handles = [
            expert.gate_proj.register_forward_pre_hook(count_rows)
            for expert in (experts or ())
            if getattr(expert, "gate_proj", None) is not None
        ]
        if experts:
            handles.append(inspect_module.register_forward_pre_hook(count_moe_rows))
        try:
            return original_compute(self, *args, **kwargs)
        finally:
            for handle in handles:
                handle.remove()
            calls.append((counters["moe_rows"], counters["expert_rows"]))

    monkeypatch.setattr(AWQProcessor, "_compute_best_scale", counting_compute)

    model.quantize(
        CalibrationConfig(
            dataset=calibration,
            batch_size=1,
            min_length=1,
        ),
        backend=BACKEND.TORCH,
    )

    # The reference pass still observes and scores every expert.
    rows = {str(row["module"]): row for row in model.quant_log}
    for index in range(config.num_experts):
        for suffix in ("gate_proj", "up_proj"):
            module_name = f"mlp.experts.{index}.{suffix}"
            assert module_name in rows, module_name
            assert rows[module_name]["loss"] != "not applicable", module_name

    # Native top-1 routing processes each replayed token through exactly one
    # expert, so a candidate grid that crosses the MoE block observes one
    # expert gate row per block row. The old all-experts wrapper additionally
    # ran every expert on every token (NUM_EXPERTS x) before the native
    # forward, i.e. NUM_EXPERTS + 1 expert rows per block row.
    moe_calls = [(moe_rows, expert_rows) for moe_rows, expert_rows in calls if moe_rows > 0]
    assert moe_calls, "the candidate grid never replayed the MoE block"
    worst_moe_rows, worst_expert_rows = max(
        moe_calls, key=lambda call: call[1] / call[0],
    )
    ratio = worst_expert_rows / worst_moe_rows
    assert ratio <= 1.5, (
        f"candidate replay observed {worst_expert_rows} expert gate rows for "
        f"{worst_moe_rows} MoE block rows ({ratio:.2f} rows/token); native "
        f"top-1 routing should be about 1.0, the all-experts wrapper "
        f"{NUM_EXPERTS + 1}.0"
    )
