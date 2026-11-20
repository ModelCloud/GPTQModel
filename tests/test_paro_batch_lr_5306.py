# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1

"""Paro's stage LRs must follow the optimizer batch size (#5306).

#5279 raised the default ``opt_batch_size`` from 64 to 128, halving the number
of optimizer steps per epoch. At a fixed LR that also halves the per-epoch
update; on Llama-3.2-1B it cost 2.2% PPL. Scaling the stage LRs from the
64-row reference recovers most of that (+0.44% at the best coefficient), but
not all of it, so the default goes back to the accuracy-tuned 64 rows and the
scaling keeps larger opt-in batches close to it.
"""

import threading
from types import SimpleNamespace

import pytest
import torch

from gptqmodel.looper import paro_processor as paro_processor_module
from gptqmodel.looper.paro_processor import ParoProcessor
from gptqmodel.quantization.config import FORMAT, METHOD, QuantizeConfig


def _processor(**overrides):
    processor = ParoProcessor.__new__(ParoProcessor)
    processor.qcfg = QuantizeConfig(
        quant_method=METHOD.PARO,
        format=FORMAT.PARO,
        bits=4,
        group_size=128,
        **overrides,
    )
    return processor


def test_reference_batch_keeps_the_configured_lrs():
    """A 64-row step is the reference the defaults were tuned on."""

    assert _processor(opt_batch_size=64)._stage_learning_rates() == pytest.approx(
        (0.05, 1e-5, 1e-6)
    )


def test_default_batch_is_the_accuracy_tuned_reference():
    """The default must stay on the reference that was validated for accuracy.

    LR scaling closes most of the gap but not all of it (+0.44% PPL at the
    best coefficient), so the accuracy-first default is the 64-row step.
    """

    processor = _processor()
    assert processor.qcfg.opt_batch_size == 64
    assert processor._stage_learning_rates() == pytest.approx((0.05, 1e-5, 1e-6))


def test_opt_in_batch_scales_the_lrs_linearly():
    """Opting into 128 rows must still keep the per-epoch update (#5306)."""

    assert _processor(opt_batch_size=128)._stage_learning_rates() == pytest.approx(
        (0.1, 2e-5, 2e-6)
    )


def test_small_batch_keeps_the_configured_lrs():
    """Batches at or below the reference keep the LR they always had."""

    assert _processor(opt_batch_size=16)._stage_learning_rates() == pytest.approx(
        (0.05, 1e-5, 1e-6)
    )


def test_configured_lrs_are_scaled_not_replaced():
    """An explicit LR is still the reference at 64 rows and scales from there."""

    processor = _processor(
        opt_batch_size=128,
        opt_rotation_lr=0.02,
        opt_weight_lr=2e-5,
        opt_quantizer_lr=4e-6,
    )
    assert processor._stage_learning_rates() == pytest.approx((0.04, 4e-5, 8e-6))

def _resolved_module_config(_module_name):
    return SimpleNamespace(get=lambda _key=None, default=None, sub_key=None: default)


def _module_scope_processor(**overrides):
    """A processor whose per-linear optimizer call can be captured."""

    fields = {
        "opt_scope": "module",
        "opt_stage_cudagraph": False,
        "runtime_bits": 4,
        "group_size": 128,
        "sym": True,
        "krot": 8,
        "opt_pair_ratio": 0.25,
        "gsq": None,
        "opt_train_samples": 128,
        "opt_validation_samples": 32,
        "opt_batch_size": 64,
        "opt_rotation_epochs": 1,
        "opt_finetune_epochs": 1,
        "opt_rotation_lr": 0.05,
        "opt_weight_lr": 1e-5,
        "opt_quantizer_lr": 1e-6,
        "opt_seed": 0,
        "opt_fused_rotation": True,
        "opt_stage_impl": "reference",
        "opt_pair_impl": "reference",
        "opt_quantizer_impl": "reference",
        "opt_channel_scale_clamp_min": 1e-2,
        "opt_channel_scale_clamp_max": 1e2,
        "dynamic_get": lambda _name, _field, default=None: default,
        "resolve_module_config": _resolved_module_config,
    }
    fields.update(overrides)
    processor = ParoProcessor.__new__(ParoProcessor)
    processor.qcfg = SimpleNamespace(**fields)
    processor.calculate_w_wq_diff = False
    processor.lock = threading.Lock()
    return processor


def _fake_optimize_paro_linear(*, weight, **kwargs):
    zeros = torch.zeros((weight.shape[0], weight.shape[1] // 128), dtype=weight.dtype)
    return SimpleNamespace(
        train_loss=0.0,
        val_loss=0.0,
        pseudo_weight=weight.detach().clone(),
        pack_weight=weight.detach().clone(),
        q_scales=torch.ones_like(zeros),
        q_zeros=torch.zeros_like(zeros, dtype=torch.int32),
        pairs=torch.zeros((1, weight.shape[1]), dtype=torch.int16),
        theta=torch.zeros((1, weight.shape[1] // 2), dtype=weight.dtype),
        channel_scales=torch.ones((1, weight.shape[1]), dtype=weight.dtype),
    )


@pytest.mark.parametrize("batch_size,scale", [(16, 1.0), (64, 1.0), (128, 2.0), (256, 4.0)])
def test_module_scope_passes_the_scaled_lrs(monkeypatch, batch_size, scale):
    """The default module scope must scale the stage LRs too (#5517).

    #5356 scaled them only for the grouped and layer scopes, so the default
    ``opt_scope="module"`` kept the raw values and ``opt_batch_size=128`` was
    bit-identical to the unscaled pre-fix run (#5306).
    """

    calls = []
    monkeypatch.setattr(
        paro_processor_module,
        "optimize_paro_linear",
        lambda **kwargs: (calls.append(kwargs), _fake_optimize_paro_linear(**kwargs))[1],
    )

    processor = _module_scope_processor(opt_batch_size=batch_size)
    module = SimpleNamespace(
        name="self_attn.q_proj",
        full_name="model.layers.0.self_attn.q_proj",
        layer_index=0,
        weight=torch.nn.Parameter(torch.randn((8, 128), dtype=torch.float32)),
        bias=None,
        state={},
    )

    processor._quantize_one_module(module, torch.randn((32, 128), dtype=torch.float32))

    assert len(calls) == 1
    assert calls[0]["rotation_lr"] == pytest.approx(0.05 * scale)
    assert calls[0]["weight_lr"] == pytest.approx(1e-5 * scale)
    assert calls[0]["quantizer_lr"] == pytest.approx(1e-6 * scale)


def test_module_scope_scales_an_explicit_lr_from_the_same_reference(monkeypatch):
    """An explicit LR is still the 64-row reference, not a fixed value (#5517)."""

    calls = []
    monkeypatch.setattr(
        paro_processor_module,
        "optimize_paro_linear",
        lambda **kwargs: (calls.append(kwargs), _fake_optimize_paro_linear(**kwargs))[1],
    )

    processor = _module_scope_processor(
        opt_batch_size=128,
        opt_rotation_lr=0.02,
        opt_weight_lr=2e-5,
        opt_quantizer_lr=4e-6,
    )
    module = SimpleNamespace(
        name="mlp.down_proj",
        full_name="model.layers.0.mlp.down_proj",
        layer_index=0,
        weight=torch.nn.Parameter(torch.randn((16, 128), dtype=torch.float32)),
        bias=None,
        state={},
    )

    processor._quantize_one_module(module, torch.randn((32, 128), dtype=torch.float32))

    assert calls[0]["rotation_lr"] == pytest.approx(0.04)
    assert calls[0]["weight_lr"] == pytest.approx(4e-5)
    assert calls[0]["quantizer_lr"] == pytest.approx(8e-6)

