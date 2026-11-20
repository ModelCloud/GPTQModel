# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1

"""Paro must honour ``Fallback.threshold`` (#5510).

``ParoProcessor`` only checked for empty captures, so a sparsely routed MoE
expert with a handful of rows still ran the full rotation + weight/quantizer
optimization on those rows while every other calibrated method fell back to
RTN. The threshold is now resolved from the rows that actually routed to the
module (before the bounded capture sampling), and starved modules use the
identity-rotation RTN result in both module and grouped scopes.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest
import torch

from gptqmodel.looper import paro_processor as paro_processor_module
from gptqmodel.looper.named_module import NamedModule
from gptqmodel.quantization.config import Fallback, FallbackStrategy, ParoConfig
from gptqmodel.quantization.paro.optimization import (
    ParoOptimizationResult,
    identity_paro_result,
    optimize_paro_linear,
)


def _processor(*, threshold, total_tokens=1000, opt_scope="module"):
    processor = object.__new__(paro_processor_module.ParoProcessor)
    processor.lock = threading.Lock()
    processor._batch_tls = threading.local()
    processor.tasks = {}
    processor.qcfg = ParoConfig(
        bits=4,
        group_size=8,
        fallback=Fallback(strategy=FallbackStrategy.RTN, threshold=threshold),
        opt_scope=opt_scope,
        opt_stage_impl="reference",
        opt_train_samples=64,
        opt_validation_samples=16,
    )
    processor.fallback = processor.qcfg.fallback
    processor.total_calibration_tokens = total_tokens
    processor.calculate_w_wq_diff = False
    processor.inputs_cache = None
    processor.calibration = None
    processor._rotary_cache = {}
    return processor


def _module(name="proj", rows=4, columns=8):
    layer = torch.nn.Linear(columns, rows, bias=True, dtype=torch.float32)
    with torch.no_grad():
        layer.weight.copy_(
            torch.arange(layer.weight.numel(), dtype=torch.float32).view(rows, columns) / 100.0
        )
        layer.bias.zero_()
    named = NamedModule(layer, name=name, full_name=f"model.layers.0.{name}", layer_index=0)
    named.state["resolved_module_config"] = {"bits": 4, "group_size": 8, "sym": True}
    return named


def _identity(module):
    return identity_paro_result(
        weight=module.weight.detach(),
        bias=module.bias.detach(),
        bits=4,
        group_size=8,
        sym=True,
        krot=8,
        quantizer_impl="reference",
    )


def _identity_result(weight):
    zeros = torch.zeros_like(weight)
    return ParoOptimizationResult(
        pseudo_weight=zeros,
        pack_weight=zeros,
        q_scales=torch.zeros(1, dtype=weight.dtype),
        q_zeros=torch.zeros(1, dtype=torch.int8),
        pairs=torch.zeros((8, 8), dtype=torch.int16),
        theta=torch.zeros((8, 4), dtype=weight.dtype),
        channel_scales=torch.ones(1, dtype=weight.dtype),
        train_loss=0.5,
        val_loss=0.25,
        used_identity=False,
    )


def test_threshold_uses_raw_routed_rows_not_the_sampling_bound():
    processor = _processor(threshold="50%", total_tokens=1000)
    processor._ensure_task_bucket("proj", 0)
    processor._set_current_batch_index(0)

    processor._record_input_feature("proj", torch.zeros(1, 400, 8))

    # The bounded capture keeps 80 rows for optimization, but 400 rows routed.
    assert processor.tasks["proj"]["inputs"][0].shape[1] == 80
    assert processor._captured_rows("proj") == 400
    assert processor._module_fallback_requested("proj")  # 400 < 50% of 1000

    processor.tasks["proj"]["raw_tokens"] = 600
    assert not processor._module_fallback_requested("proj")


@pytest.mark.parametrize(
    "threshold,observed,falls_back",
    [
        (0, 10, False),
        (None, 10, False),
        (100, 40, True),
        (100, 200, False),
        ("50%", 400, True),
        ("50%", 600, False),
    ],
)
def test_threshold_values(threshold, observed, falls_back):
    processor = _processor(threshold=threshold, total_tokens=1000)
    processor.tasks["proj"] = {"inputs": [], "raw_tokens": observed}

    assert processor._module_fallback_requested("proj") is falls_back


def test_empty_capture_matches_the_public_identity_helper():
    module = _module()
    expected = _identity(module)

    result = optimize_paro_linear(
        weight=module.weight.detach(),
        bias=module.bias.detach(),
        inputs=torch.empty(0, 8),
        bits=4,
        group_size=8,
        sym=True,
        krot=8,
        pair_ratio=1.0,
        train_rows=4,
        val_rows=1,
        batch_size=1,
        rotation_epochs=1,
        finetune_epochs=1,
        rotation_lr=0.01,
        weight_lr=0.01,
        quantizer_lr=0.01,
        seed=0,
        quantizer_impl="reference",
    )

    assert result.used_identity
    assert torch.allclose(result.pseudo_weight, expected.pseudo_weight)


def test_module_scope_uses_identity_fallback(monkeypatch):
    processor = _processor(threshold=100, total_tokens=1000)
    module = _module()
    processor.tasks["proj"] = {"inputs": [], "raw_tokens": 10}

    calls = []

    def _unexpected(**kwargs):
        calls.append(kwargs)
        raise AssertionError("the optimizer must not run for a starved module")

    monkeypatch.setattr(paro_processor_module, "optimize_paro_linear", _unexpected)

    train_loss, val_loss = processor._quantize_one_module(module, torch.zeros(1, 4, 8))

    expected = _identity(_module())
    assert calls == []
    assert torch.allclose(module.weight, expected.pseudo_weight)
    assert "pack_weight" in module.state
    assert (train_loss, val_loss) == (0.0, 0.0)


def test_module_scope_runs_the_optimizer_above_the_threshold(monkeypatch):
    processor = _processor(threshold=100, total_tokens=1000)
    module = _module()
    processor.tasks["proj"] = {"inputs": [], "raw_tokens": 900}

    seen = []

    def _fake(**kwargs):
        seen.append(kwargs)
        return _identity_result(kwargs["weight"])

    monkeypatch.setattr(paro_processor_module, "optimize_paro_linear", _fake)

    train_loss, val_loss = processor._quantize_one_module(module, torch.zeros(1, 4, 8))

    assert len(seen) == 1
    assert torch.count_nonzero(module.weight) == 0
    assert (train_loss, val_loss) == (0.5, 0.25)


def test_grouped_scope_falls_back_starved_members_only(monkeypatch):
    processor = _processor(threshold=100, total_tokens=1000, opt_scope="compute_block")
    starved = _module("gate_proj")
    covered = _module("up_proj")
    processor.tasks = {
        "gate_proj": {"inputs": [], "raw_tokens": 10},
        "up_proj": {"inputs": [], "raw_tokens": 900},
    }

    optimized = []

    def _fake_group(state, group_modules):
        optimized.append([module.name for module in group_modules])
        return {module.name: _identity_result(module.weight) for module in group_modules}, 0.25

    monkeypatch.setattr(processor, "_optimize_group", _fake_group)
    monkeypatch.setattr(
        processor,
        "_layer_input_features",
        lambda state: {"gate_proj": torch.zeros(1, 4, 8), "up_proj": torch.zeros(1, 4, 8)},
    )
    monkeypatch.setattr(processor, "_log_quant_result", lambda *args, **kwargs: None)

    state = SimpleNamespace(
        modules={"gate_proj": starved, "up_proj": covered},
        quantized=False,
        layer_inputs=[torch.zeros(1, 4, 8)],
        layer_outputs=[torch.zeros(1, 4, 8)],
        layer_input_kwargs=[{}],
        pending_modules=set(),
        processed_subsets=set(),
        subset_total=None,
        pristine_layer_module=None,
        prepared_group_source_module=None,
        prepared_group_source_module_by_device=None,
        grouped_dataset=None,
        grouped_dataset_by_device=None,
        replay_batches=None,
    )

    expected_starved = _identity(_module("gate_proj"))
    processor._quantize_layer(0, state)

    assert optimized == [["up_proj"]]
    assert torch.allclose(starved.weight, expected_starved.pseudo_weight)
    assert torch.count_nonzero(covered.weight) == 0
