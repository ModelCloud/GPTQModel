# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# ParoQuant test coverage adapted from the ParoQuant paper and public project:
# https://arxiv.org/html/2511.10645v2
# https://github.com/z-lab/paroquant

"""Unit tests for ParoQuant config, optimizer, and lifecycle invariants."""

import copy
import inspect
import sys
import threading
import time
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from transformers.quantizers.auto import AutoQuantizationConfig
from transformers.utils.quantization_config import GPTQConfig

import gptqmodel.looper.paroquant_processor as paroquant_processor_module
import gptqmodel.utils.paroquant as paroquant_utils_module
from gptqmodel.looper.awq_processor import AWQProcessor
from gptqmodel.looper.input_cache import InputCache
from gptqmodel.looper.module_looper import _restrict_quant_devices_for_method
from gptqmodel.looper.named_module import NamedModule
from gptqmodel.looper.paroquant_processor import ParoQuantProcessor
from gptqmodel.looper.stage_layer import _capture_pristine_group_context
from gptqmodel.nn_modules.hooked_linear import replace_module_with_hooked_legacy
from gptqmodel.nn_modules.qlinear.paroquant import ParoLinear
from gptqmodel.quantization.config import FORMAT, METHOD, ParoConfig, QuantizeConfig
from gptqmodel.quantization.paroquant import optimization as paroquant_optimization
from gptqmodel.quantization.paroquant.optimization import (
    GroupLinearQuantizer,
    _apply_rotation,
    _ParoQuantOptimLinear,
    build_random_rotation_buffers,
    optimize_paroquant_linear,
    pseudo_quantize_dequant,
)
from gptqmodel.utils.backend import BACKEND
from gptqmodel.utils.importer import get_kernel_for_backend
from gptqmodel.utils.paroquant import (
    _rotation_launch_config,
    apply_paroquant_rotation,
    apply_paroquant_rotation_reference,
    build_identity_rotation_buffers,
    clear_paroquant_rotation_autotune_cache,
    clear_paroquant_rotation_extension_cache,
    prewarm_paroquant_rotation_extension,
)
from gptqmodel.utils.paroquant_benchmark import make_paroquant_config


def test_paroquant_quantize_config_dispatches_constructor():
    """Guard that ParoQuant config fields survive direct construction."""
    cfg = QuantizeConfig(
        quant_method=METHOD.PARO,
        format=FORMAT.PAROQUANT,
        bits=4,
        group_size=128,
        krot=8,
    )

    assert cfg.quant_method == METHOD.PARO
    assert cfg.format == FORMAT.PAROQUANT
    assert cfg.krot == 8
    assert cfg.opt_batch_size == 64
    assert cfg.opt_optimizer == "adamw"
    assert cfg.opt_weight_decay == pytest.approx(0.01)
    assert cfg.opt_betas == pytest.approx((0.9, 0.95))
    assert cfg.opt_eps == pytest.approx(1e-10)
    assert cfg.opt_amsgrad is False
    assert cfg.opt_sgd_momentum == pytest.approx(0.0)
    assert cfg.opt_sgd_dampening == pytest.approx(0.0)
    assert cfg.opt_sgd_nesterov is False
    assert cfg.opt_scope == "module"
    assert cfg.opt_stage_impl == "fast"
    assert cfg.opt_pair_impl == "fast"
    assert cfg.opt_quantizer_impl == "reference"
    assert cfg.opt_stage_cudagraph is True
    assert cfg.opt_gradient_checkpointing is False
    assert cfg.opt_best_state_dtype == "fp32"
    assert cfg.opt_train_on_noisy_inputs is False
    assert cfg.opt_channel_scale_clamp_min == 1e-2
    assert cfg.opt_channel_scale_clamp_max == 1e2
    assert cfg.export_quant_method() == METHOD.PARO


def test_paroquant_quantize_config_enables_gradient_checkpointing_by_default_for_layer_scope():
    """Layer scope should opt into activation checkpointing by default because it is the only measured memory win."""

    cfg = ParoConfig(
        bits=4,
        group_size=128,
        opt_scope="layer",
    )

    assert cfg.opt_gradient_checkpointing is True


def test_paroquant_quantize_config_from_external_payload_round_trips():
    """Guard import/export of ParoQuant metadata from serialized payloads."""
    cfg = QuantizeConfig.from_quant_config(
        {
            "quant_method": "paroquant",
            "bits": 4,
            "group_size": 128,
            "krot": 8,
            "meta": {
                "opt_rotation_epochs": 10,
                "opt_finetune_epochs": 10,
                "opt_train_samples": 2048,
                "opt_validation_samples": 64,
                "opt_batch_size": 16,
                "opt_rotation_lr": 0.05,
                "opt_weight_lr": 1e-5,
                "opt_quantizer_lr": 1e-6,
                "opt_pair_ratio": 0.5,
                "opt_seed": 0,
                "opt_optimizer": "sgd",
                "opt_weight_decay": 0.02,
                "opt_betas": [0.8, 0.9],
                "opt_eps": 1e-8,
                "opt_amsgrad": True,
                "opt_sgd_momentum": 0.85,
                "opt_sgd_dampening": 0.0,
                "opt_sgd_nesterov": True,
                "opt_fused_rotation": False,
                "opt_gradient_checkpointing": False,
                "opt_stage_cudagraph": False,
                "opt_best_state_dtype": "fp16",
                "opt_train_on_noisy_inputs": True,
                "opt_scope": "compute_block",
                "opt_stage_impl": "reference",
                "opt_pair_impl": "fast",
                "opt_quantizer_impl": "reference",
                "opt_channel_scale_clamp_min": 0.02,
                "opt_channel_scale_clamp_max": 50.0,
            },
        }
    )

    assert isinstance(cfg, ParoConfig)
    assert cfg.quant_method == METHOD.PARO
    assert cfg.format == FORMAT.PAROQUANT
    assert cfg.krot == 8
    assert cfg.opt_rotation_epochs == 10
    assert cfg.opt_finetune_epochs == 10
    assert cfg.opt_train_samples == 2048
    assert cfg.opt_validation_samples == 64
    assert cfg.opt_batch_size == 16
    assert cfg.opt_rotation_lr == 0.05
    assert cfg.opt_weight_lr == 1e-5
    assert cfg.opt_quantizer_lr == 1e-6
    assert cfg.opt_pair_ratio == 0.5
    assert cfg.opt_seed == 0
    assert cfg.opt_optimizer == "sgd"
    assert cfg.opt_weight_decay == pytest.approx(0.02)
    assert cfg.opt_betas == pytest.approx((0.8, 0.9))
    assert cfg.opt_eps == pytest.approx(1e-8)
    assert cfg.opt_amsgrad is True
    assert cfg.opt_sgd_momentum == pytest.approx(0.85)
    assert cfg.opt_sgd_dampening == pytest.approx(0.0)
    assert cfg.opt_sgd_nesterov is True
    assert cfg.opt_fused_rotation is False
    assert cfg.opt_gradient_checkpointing is False
    assert cfg.opt_stage_cudagraph is False
    assert cfg.opt_best_state_dtype == "fp16"
    assert cfg.opt_train_on_noisy_inputs is True
    assert cfg.opt_scope == "compute_block"
    assert cfg.opt_stage_impl == "reference"
    assert cfg.opt_pair_impl == "fast"
    assert cfg.opt_quantizer_impl == "reference"
    assert cfg.opt_channel_scale_clamp_min == 0.02
    assert cfg.opt_channel_scale_clamp_max == 50.0
    assert cfg.to_dict()["meta"]["opt_fused_rotation"] is False
    assert cfg.to_dict()["meta"]["opt_gradient_checkpointing"] is False
    assert cfg.to_dict()["meta"]["opt_stage_cudagraph"] is False
    assert cfg.to_dict()["meta"]["opt_best_state_dtype"] == "fp16"
    assert cfg.to_dict()["meta"]["opt_train_on_noisy_inputs"] is True
    assert cfg.to_dict()["meta"]["opt_scope"] == "compute_block"
    assert cfg.to_dict()["meta"]["opt_stage_impl"] == "reference"
    assert cfg.to_dict()["meta"]["opt_pair_impl"] == "fast"
    assert cfg.to_dict()["meta"]["opt_quantizer_impl"] == "reference"
    assert cfg.to_dict()["meta"]["opt_channel_scale_clamp_min"] == 0.02
    assert cfg.to_dict()["meta"]["opt_channel_scale_clamp_max"] == 50.0
    assert cfg.to_dict()["meta"]["opt_optimizer"] == "sgd"
    assert cfg.to_dict()["meta"]["opt_weight_decay"] == pytest.approx(0.02)
    assert cfg.to_dict()["meta"]["opt_betas"] == [0.8, 0.9]
    assert cfg.to_dict()["meta"]["opt_eps"] == pytest.approx(1e-8)
    assert cfg.to_dict()["meta"]["opt_amsgrad"] is True
    assert cfg.to_dict()["meta"]["opt_sgd_momentum"] == pytest.approx(0.85)
    assert cfg.to_dict()["meta"]["opt_sgd_dampening"] == pytest.approx(0.0)
    assert cfg.to_dict()["meta"]["opt_sgd_nesterov"] is True


def test_paroquant_quantize_config_rejects_invalid_scale_clamp_range():
    """Guard that ParoQuant scale-clamp overrides remain numerically valid."""
    with pytest.raises(ValueError, match="scale clamp bounds must be positive"):
        ParoConfig(
            bits=4,
            group_size=128,
            opt_channel_scale_clamp_min=0.0,
            opt_channel_scale_clamp_max=10.0,
        )

    with pytest.raises(ValueError, match="opt_channel_scale_clamp_min"):
        ParoConfig(
            bits=4,
            group_size=128,
            opt_channel_scale_clamp_min=10.0,
            opt_channel_scale_clamp_max=10.0,
        )


def test_paroquant_quantize_config_rejects_invalid_opt_scope():
    """Guard that ParoQuant optimize-scope selection stays within supported modes."""
    with pytest.raises(ValueError, match="opt_scope"):
        ParoConfig(
            bits=4,
            group_size=128,
            opt_scope="block",
        )


def test_paroquant_quantize_config_rejects_invalid_opt_optimizer():
    """Guard that ParoQuant optimizer selection stays within supported modes."""
    with pytest.raises(ValueError, match="opt_optimizer"):
        ParoConfig(
            bits=4,
            group_size=128,
            opt_optimizer="lion",
        )


def test_paroquant_quantize_config_rejects_invalid_best_state_dtype():
    """Guard best-state snapshot compression against unsupported dtype strings."""
    with pytest.raises(ValueError, match="opt_best_state_dtype"):
        ParoConfig(
            bits=4,
            group_size=128,
            opt_best_state_dtype="int8",
        )


def test_paroquant_quantize_config_rejects_invalid_optimizer_hyperparameters():
    """Guard optimizer hyperparameter validation against invalid stage settings."""
    with pytest.raises(ValueError, match="opt_betas"):
        ParoConfig(
            bits=4,
            group_size=128,
            opt_betas=(0.9,),
        )

    with pytest.raises(ValueError, match="opt_eps"):
        ParoConfig(
            bits=4,
            group_size=128,
            opt_eps=0.0,
        )

    with pytest.raises(ValueError, match="opt_sgd_nesterov"):
        ParoConfig(
            bits=4,
            group_size=128,
            opt_sgd_nesterov=True,
        )

    with pytest.raises(ValueError, match="opt_sgd_dampening"):
        ParoConfig(
            bits=4,
            group_size=128,
            opt_sgd_momentum=0.9,
            opt_sgd_dampening=0.1,
            opt_sgd_nesterov=True,
        )


def test_paroquant_benchmark_config_preserves_opt_scope():
    """Benchmark helpers should propagate the requested optimization scope."""
    cfg = make_paroquant_config(dynamic={}, opt_scope="compute_block")

    assert cfg.quant_method == METHOD.PARO
    assert cfg.opt_scope == "compute_block"
    assert cfg.opt_gradient_checkpointing is False


def test_paroquant_quantize_config_preserves_explicit_gradient_checkpointing_override():
    """Explicit checkpointing overrides must win over the scope-derived default."""

    layer_cfg = ParoConfig(
        bits=4,
        group_size=128,
        opt_scope="layer",
        opt_gradient_checkpointing=False,
    )
    compute_block_cfg = ParoConfig(
        bits=4,
        group_size=128,
        opt_scope="compute_block",
        opt_gradient_checkpointing=True,
    )

    assert layer_cfg.opt_gradient_checkpointing is False
    assert compute_block_cfg.opt_gradient_checkpointing is True


def test_paroquant_rotation_toggle_prefers_explicit_config_over_env(monkeypatch):
    """Guard that the config-backed fused toggle overrides the legacy env fallback."""
    x = torch.randn(4, 8, dtype=torch.float32)
    pairs, theta, channel_scales = build_identity_rotation_buffers(
        in_features=8,
        group_size=8,
        krot=1,
        dtype=torch.float32,
    )
    theta = theta.clone().requires_grad_(True)

    calls = []

    def fake_fused_rotation(x, pairs, theta, *, scales, group_size):
        del pairs, theta, scales, group_size
        calls.append("fused")
        return x + 123.0

    monkeypatch.setattr(paroquant_optimization, "apply_paroquant_rotation_autograd", fake_fused_rotation)
    monkeypatch.setenv("GPTQMODEL_PAROQUANT_OPT_FUSED_ROTATION", "1")

    reference_out = _apply_rotation(
        x,
        pairs,
        theta,
        scales=channel_scales,
        group_size=8,
        fused_rotation=False,
    )
    assert calls == []
    torch.testing.assert_close(reference_out, x, atol=0, rtol=0)

    fused_out = _apply_rotation(
        x,
        pairs,
        theta,
        scales=channel_scales,
        group_size=8,
        fused_rotation=True,
    )
    assert calls == ["fused"]
    torch.testing.assert_close(fused_out, x + 123.0)


def test_paroquant_fused_rotation_uses_forward_only_path_without_grad_inputs(monkeypatch):
    """Guard that inactive rotation grads do not route through the autograd wrapper."""
    x = torch.randn(4, 8, dtype=torch.float32)
    pairs, theta, channel_scales = build_identity_rotation_buffers(
        in_features=8,
        group_size=8,
        krot=1,
        dtype=torch.float32,
    )

    calls = []

    def fake_forward_only(x, pairs, theta, *, scales, group_size):
        del pairs, theta, scales, group_size
        calls.append("forward")
        return x + 1.0

    def fake_autograd(x, pairs, theta, *, scales, group_size):
        del pairs, theta, scales, group_size
        calls.append("autograd")
        return x + 2.0

    monkeypatch.setattr(paroquant_optimization, "apply_paroquant_rotation", fake_forward_only)
    monkeypatch.setattr(paroquant_optimization, "apply_paroquant_rotation_autograd", fake_autograd)

    out = _apply_rotation(
        x,
        pairs,
        theta,
        scales=channel_scales,
        group_size=8,
        fused_rotation=True,
    )
    torch.testing.assert_close(out, x + 1.0)
    assert calls == ["forward"]

    calls.clear()
    theta = theta.clone().requires_grad_(True)
    out = _apply_rotation(
        x,
        pairs,
        theta,
        scales=channel_scales,
        group_size=8,
        fused_rotation=True,
    )
    torch.testing.assert_close(out, x + 2.0)
    assert calls == ["autograd"]


@pytest.mark.parametrize(
    ("device_type", "expected_use_amp"),
    [
        ("cpu", False),
        ("cuda", True),
    ],
)
def test_paroquant_fast_stage_matches_reference_amp_eval_flag(monkeypatch, device_type, expected_use_amp):
    """Guard that the fast stage uses CUDA AMP for eval bookkeeping like reference."""
    calls = []

    class _DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

    class _FakeTensor:
        def __init__(self, kind: str):
            self.device = SimpleNamespace(type=kind)

    def fake_evaluate(_model, _inputs, _targets, *, use_amp=False):
        calls.append(use_amp)
        return 0.0

    monkeypatch.setattr(paroquant_optimization, "_evaluate_model", fake_evaluate)

    train_loss, val_loss = paroquant_optimization._run_stage_gptqmodel(
        model=_DummyModel(),
        inputs_train=_FakeTensor(device_type),
        targets_train=_FakeTensor(device_type),
        inputs_val=_FakeTensor(device_type),
        targets_val=_FakeTensor(device_type),
        param_groups=[],
        epochs=0,
        batch_size=16,
    )

    assert train_loss == 0.0
    assert val_loss == 0.0
    assert calls == [expected_use_amp, expected_use_amp]


def test_paroquant_evaluate_model_keeps_loss_inside_cuda_autocast(monkeypatch):
    """Guard the PR-18 fix so validation loss stays inside the CUDA autocast region."""
    state = {"autocast_active": False, "loss_saw_autocast": None}

    @contextmanager
    def fake_autocast(device_type: str):
        assert device_type == "cuda"
        previous = state["autocast_active"]
        state["autocast_active"] = True
        try:
            yield
        finally:
            state["autocast_active"] = previous

    class _FakeInput:
        def __init__(self):
            self.device = SimpleNamespace(type="cuda")

        def numel(self):
            return 1

    class _DummyModel(torch.nn.Module):
        def forward(self, _inputs):
            return torch.tensor([1.0], dtype=torch.float32)

    def fake_loss(preds, targets):
        del preds, targets
        state["loss_saw_autocast"] = state["autocast_active"]
        return torch.tensor(0.25, dtype=torch.float32)

    monkeypatch.setattr(torch.amp, "autocast", fake_autocast)
    monkeypatch.setattr(paroquant_optimization.F, "smooth_l1_loss", fake_loss)

    loss = paroquant_optimization._evaluate_model(
        _DummyModel(),
        _FakeInput(),
        torch.tensor([0.0], dtype=torch.float32),
        use_amp=True,
    )

    assert loss == 0.25
    assert state["loss_saw_autocast"] is True


def test_paroquant_fast_stage_uses_cuda_amp_training(monkeypatch):
    """Guard that the fast stage now mirrors upstream AMP training on CUDA."""
    state = {"autocast_active": False, "loss_saw_autocast": []}
    scaler_events = []

    @contextmanager
    def fake_autocast(device_type: str):
        assert device_type == "cuda"
        previous = state["autocast_active"]
        state["autocast_active"] = True
        try:
            yield
        finally:
            state["autocast_active"] = previous

    class _FakeScaler:
        def __init__(self, *, enabled: bool):
            scaler_events.append(("init", enabled))
            self.enabled = enabled

        def scale(self, loss):
            scaler_events.append(("scale", self.enabled))

            class _ScaledLoss:
                def __init__(self, wrapped_loss):
                    self.wrapped_loss = wrapped_loss

                def backward(self):
                    scaler_events.append(("backward", self.wrapped_loss.detach().item()))
                    self.wrapped_loss.backward()

            return _ScaledLoss(loss)

        def step(self, optimizer):
            scaler_events.append(("step", self.enabled))
            optimizer.step()

        def update(self):
            scaler_events.append(("update", self.enabled))

    class _TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([[1.0]], dtype=torch.float32))

        def forward(self, x):
            return x @ self.weight

        def reset_masked_angles(self):
            return None

    class _FakeRows:
        def __init__(self, rows: int):
            self.device = SimpleNamespace(type="cuda")
            self.shape = (rows, 1)

    train_inputs = _FakeRows(rows=2)
    train_targets = _FakeRows(rows=2)
    val_inputs = _FakeRows(rows=1)
    val_targets = _FakeRows(rows=1)

    train_input_batches = [torch.tensor([[1.0]], dtype=torch.float32), torch.tensor([[2.0]], dtype=torch.float32)]
    train_target_batches = [torch.tensor([[0.0]], dtype=torch.float32), torch.tensor([[0.0]], dtype=torch.float32)]

    def fake_chunk_rows(rows, batch_size):
        del batch_size
        if rows is train_inputs:
            return train_input_batches
        if rows is train_targets:
            return train_target_batches
        raise AssertionError("unexpected rows object")

    def fake_evaluate(_model, _inputs, _targets, *, use_amp=False):
        assert use_amp is True
        return 0.0

    original_loss = paroquant_optimization.F.smooth_l1_loss

    def wrapped_loss(preds, targets):
        state["loss_saw_autocast"].append(state["autocast_active"])
        return original_loss(preds, targets)

    monkeypatch.setattr(paroquant_optimization, "_chunk_rows", fake_chunk_rows)
    monkeypatch.setattr(paroquant_optimization, "_evaluate_model", fake_evaluate)
    monkeypatch.setattr(paroquant_optimization.F, "smooth_l1_loss", wrapped_loss)
    monkeypatch.setattr(torch.amp, "autocast", fake_autocast)
    monkeypatch.setattr(torch.amp, "GradScaler", _FakeScaler)

    model = _TinyModel()
    train_loss, val_loss = paroquant_optimization._run_stage_gptqmodel(
        model=model,
        inputs_train=train_inputs,
        targets_train=train_targets,
        inputs_val=val_inputs,
        targets_val=val_targets,
        param_groups=[{"params": [model.weight], "lr": 0.1}],
        epochs=1,
        batch_size=1,
    )

    assert train_loss >= 0.0
    assert val_loss == 0.0
    assert state["loss_saw_autocast"] == [True, True]
    assert scaler_events[0] == ("init", True)
    assert [event[0] for event in scaler_events[1:]] == [
        "scale",
        "backward",
        "step",
        "update",
        "scale",
        "backward",
        "step",
        "update",
    ]
    assert all(event[1] is True for event in scaler_events if event[0] != "backward")
    assert all(event[1] >= 0.0 for event in scaler_events if event[0] == "backward")


def test_paroquant_fast_pair_builder_emits_disjoint_matchings():
    """Guard the fast pair builder so each rotation remains kernel-legal."""
    pairs, masks = build_random_rotation_buffers(
        in_features=8,
        group_size=8,
        krot=3,
        pair_ratio=0.5,
        seed=0,
        device=torch.device("cpu"),
    )

    assert pairs.shape == (3, 8)
    assert masks.shape == (3, 4)
    assert torch.count_nonzero(masks).item() == 0

    seen_edges = set()
    for rotation_pairs in pairs.view(3, 4, 2).tolist():
        used_channels = set()
        for left, right in rotation_pairs:
            assert left != right
            assert left not in used_channels
            assert right not in used_channels
            used_channels.add(left)
            used_channels.add(right)
            edge = tuple(sorted((left, right)))
            assert edge not in seen_edges
            seen_edges.add(edge)


def test_paroquant_optim_forward_matches_pseudo_weight_contract():
    """Guard the stage-time forward rewrite against the original pseudo-weight contract."""
    torch.manual_seed(0)
    weight = torch.randn((16, 8), dtype=torch.float32)
    inputs = torch.randn((5, 8), dtype=torch.float32)
    pairs, theta_mask = build_random_rotation_buffers(
        in_features=8,
        group_size=8,
        krot=3,
        pair_ratio=0.5,
        seed=0,
        device=torch.device("cpu"),
    )
    model = _ParoQuantOptimLinear(
        weight,
        None,
        bits=4,
        group_size=8,
        quantizer_sym=True,
        pairs=pairs,
        theta_mask=theta_mask,
        fused_rotation=False,
    )
    with torch.no_grad():
        model.theta.uniform_(-0.2, 0.2)
        model.channel_scales_opt.uniform_(0.8, 1.2)
    model.init_quantizer()

    expected = F.linear(inputs, model.pseudo_weight(), model.bias)
    actual = model(inputs)

    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)


def test_paroquant_optim_forward_matches_pseudo_weight_contract_for_rank3():
    """Guard grouped/layer optimization forwards that pass [batch, seq, hidden] activations."""
    torch.manual_seed(0)
    weight = torch.randn((16, 8), dtype=torch.float32)
    inputs = torch.randn((2, 3, 8), dtype=torch.float32)
    pairs, theta_mask = build_random_rotation_buffers(
        in_features=8,
        group_size=8,
        krot=3,
        pair_ratio=0.5,
        seed=0,
        device=torch.device("cpu"),
    )
    model = _ParoQuantOptimLinear(
        weight,
        None,
        bits=4,
        group_size=8,
        quantizer_sym=True,
        pairs=pairs,
        theta_mask=theta_mask,
        fused_rotation=False,
    )
    with torch.no_grad():
        model.theta.uniform_(-0.2, 0.2)
        model.channel_scales_opt.uniform_(0.8, 1.2)
    model.init_quantizer()

    expected = F.linear(inputs, model.pseudo_weight(), model.bias)
    actual = model(inputs)

    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)


def test_paroquant_materialized_sym_scale_ste_matches_legacy_gradients():
    """Guard the stage2 symmetric quantizer rewrite against the legacy STE math."""
    torch.manual_seed(0)
    group_size = 8
    bits = 4
    qmin = -(2 ** (bits - 1))
    qmax = 2 ** (bits - 1) - 1

    weight = torch.randn((6, group_size), dtype=torch.float32, requires_grad=True)
    scale = (torch.rand((6, 1), dtype=torch.float32) + 0.1).requires_grad_()

    legacy_scale = scale.clone().detach().requires_grad_(True)
    legacy_weight = weight.clone().detach().requires_grad_(True)
    legacy_scale_safe = paroquant_optimization._clamp_ste(legacy_scale, min_value=1e-5, max_value=1e5)
    legacy_quant = paroquant_optimization._clamp_ste(
        paroquant_optimization._round_ste(legacy_weight / legacy_scale_safe),
        qmin,
        qmax,
    )
    legacy_output = (legacy_quant * legacy_scale_safe).reshape_as(legacy_weight)

    actual_output = pseudo_quantize_dequant(
        weight,
        bits=bits,
        group_size=group_size,
        sym=True,
        scale=scale,
        use_ste=True,
    )

    torch.testing.assert_close(actual_output, legacy_output, atol=0, rtol=0)

    legacy_output.sum().backward()
    actual_output.sum().backward()

    torch.testing.assert_close(weight.grad, legacy_weight.grad, atol=0, rtol=0)
    torch.testing.assert_close(scale.grad, legacy_scale.grad, atol=0, rtol=0)


def test_paroquant_large_train_quant_compile_dispatch(monkeypatch):
    """Guard that only large CUDA training-time quant calls route into the compiled helper."""
    class _FakeWeight:
        def __init__(self, numel: int, device_type: str = "cuda"):
            self._numel = numel
            self.device = SimpleNamespace(type=device_type)

        def numel(self) -> int:
            return self._numel

    monkeypatch.setattr(paroquant_optimization, "_PAROQUANT_LARGE_TRAIN_QUANT_COMPILE_MIN_NUMEL", 16)
    monkeypatch.setattr(paroquant_optimization, "env_flag", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(paroquant_optimization, "_get_large_train_quant_compile", lambda: lambda *_args, **_kwargs: "compiled")
    monkeypatch.setattr(paroquant_optimization, "pseudo_quantize_dequant", lambda *_args, **_kwargs: "eager")

    assert (
        paroquant_optimization._maybe_compile_large_train_quant(
            _FakeWeight(32),
            bits=4,
            group_size=8,
            sym=True,
        )
        == "compiled"
    )
    assert (
        paroquant_optimization._maybe_compile_large_train_quant(
            _FakeWeight(8),
            bits=4,
            group_size=8,
            sym=True,
        )
        == "eager"
    )
    assert (
        paroquant_optimization._maybe_compile_large_train_quant(
            _FakeWeight(32),
            bits=4,
            group_size=8,
            sym=False,
        )
        == "compiled"
    )
    assert (
        paroquant_optimization._maybe_compile_large_train_quant(
            _FakeWeight(32, device_type="cpu"),
            bits=4,
            group_size=8,
            sym=True,
        )
        == "eager"
    )


def test_paroquant_stage_cudagraph_gate_requires_real_cuda_tensor(monkeypatch):
    """Guard that CUDA-graph replay only activates for real CUDA tensor stages."""
    class _DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones((2, 4), dtype=torch.float32))
            self.fused_rotation = True

    class _FakeRows:
        def __init__(self, device_type: str):
            self.shape = (2048, 4)
            self.device = SimpleNamespace(type=device_type, index=0 if device_type == "cuda" else None)

    monkeypatch.delenv("GPTQMODEL_PAROQUANT_OPT_STAGE_CUDAGRAPH", raising=False)

    model = _DummyModel()
    real_cpu_rows = torch.ones((2048, 4), dtype=torch.float32)
    fake_cuda_rows = _FakeRows("cuda")

    assert paroquant_optimization._should_use_paroquant_stage_cudagraph(model, inputs_train=real_cpu_rows, batch_size=64) is False
    assert paroquant_optimization._should_use_paroquant_stage_cudagraph(model, inputs_train=fake_cuda_rows, batch_size=64) is False


def test_paroquant_stage_cudagraph_falls_back_to_eager_on_runtime_error(monkeypatch):
    """Guard that a CUDA-graph stage failure restores model state and reruns eagerly."""
    class _DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([[1.0]], dtype=torch.float32))
            self.fused_rotation = True

        def reset_masked_angles(self):
            return None

    model = _DummyModel()
    call_order = []

    def fake_impl(*, model, **kwargs):
        del kwargs
        call_order.append(model)
        return 1.25, 2.5

    def fake_cudagraph(*, model, **kwargs):
        del kwargs
        call_order.append(("graph", model))
        raise RuntimeError("graph failed at runtime")

    monkeypatch.setattr(paroquant_optimization, "_should_use_paroquant_stage_cudagraph", lambda *args, **kwargs: True)
    monkeypatch.setattr(paroquant_optimization, "_run_stage_gptqmodel_cudagraph", fake_cudagraph)
    monkeypatch.setattr(paroquant_optimization, "_run_stage_gptqmodel_impl", fake_impl)

    train_loss, val_loss = paroquant_optimization._run_stage_gptqmodel(
        model=model,
        inputs_train=torch.ones((2, 1), dtype=torch.float32),
        targets_train=torch.zeros((2, 1), dtype=torch.float32),
        inputs_val=torch.ones((1, 1), dtype=torch.float32),
        targets_val=torch.zeros((1, 1), dtype=torch.float32),
        param_groups=[],
        epochs=1,
        batch_size=1,
    )

    assert (train_loss, val_loss) == (1.25, 2.5)
    assert call_order == [("graph", model), model]


def test_optimize_paroquant_linear_forwards_stage_cudagraph(monkeypatch):
    """Guard that explicit CUDA-graph policy is forwarded into both optimization stages."""
    stage_cudagraph_calls = []
    original_run_stage = paroquant_optimization._run_stage

    def spy_run_stage(*, stage_cudagraph=None, **kwargs):
        stage_cudagraph_calls.append(stage_cudagraph)
        return original_run_stage(stage_cudagraph=stage_cudagraph, **kwargs)

    monkeypatch.setattr(paroquant_optimization, "_run_stage", spy_run_stage)

    weight = torch.randn((8, 8), dtype=torch.float32)
    inputs = torch.randn((64, 8), dtype=torch.float32)

    result = optimize_paroquant_linear(
        weight=weight,
        bias=None,
        inputs=inputs,
        bits=4,
        group_size=8,
        sym=True,
        krot=1,
        pair_ratio=0.5,
        train_rows=32,
        val_rows=16,
        batch_size=16,
        rotation_epochs=1,
        finetune_epochs=1,
        rotation_lr=0.05,
        weight_lr=1e-5,
        quantizer_lr=1e-6,
        seed=0,
        fused_rotation=True,
        stage_cudagraph=False,
        stage_impl="fast",
        pair_impl="fast",
        quantizer_impl="reference",
    )

    assert result.val_loss >= 0.0
    assert stage_cudagraph_calls == [False, False]


def test_optimize_paroquant_linear_forwards_optimizer_name(monkeypatch):
    """Guard that the selected stage optimizer is forwarded into both optimization stages."""
    optimizer_name_calls = []
    original_run_stage = paroquant_optimization._run_stage

    def spy_run_stage(*, optimizer_name="adamw", **kwargs):
        optimizer_name_calls.append(optimizer_name)
        return original_run_stage(optimizer_name=optimizer_name, **kwargs)

    monkeypatch.setattr(paroquant_optimization, "_run_stage", spy_run_stage)

    weight = torch.randn((8, 8), dtype=torch.float32)
    inputs = torch.randn((64, 8), dtype=torch.float32)

    result = optimize_paroquant_linear(
        weight=weight,
        bias=None,
        inputs=inputs,
        bits=4,
        group_size=8,
        sym=True,
        krot=1,
        pair_ratio=0.5,
        train_rows=32,
        val_rows=16,
        batch_size=16,
        rotation_epochs=1,
        finetune_epochs=1,
        rotation_lr=0.05,
        weight_lr=1e-5,
        quantizer_lr=1e-6,
        seed=0,
        optimizer_name="sgd",
        fused_rotation=True,
        stage_cudagraph=False,
        stage_impl="fast",
        pair_impl="fast",
        quantizer_impl="reference",
    )

    assert result.val_loss >= 0.0
    assert optimizer_name_calls == ["sgd", "sgd"]


def test_optimize_paroquant_linear_forwards_best_state_dtype(monkeypatch):
    """Guard that explicit best-state snapshot dtype policy is forwarded into both optimization stages."""
    best_state_dtype_calls = []
    original_run_stage = paroquant_optimization._run_stage

    def spy_run_stage(*, best_state_dtype="fp32", **kwargs):
        best_state_dtype_calls.append(best_state_dtype)
        return original_run_stage(best_state_dtype=best_state_dtype, **kwargs)

    monkeypatch.setattr(paroquant_optimization, "_run_stage", spy_run_stage)

    weight = torch.randn((8, 8), dtype=torch.float32)
    inputs = torch.randn((64, 8), dtype=torch.float32)

    result = optimize_paroquant_linear(
        weight=weight,
        bias=None,
        inputs=inputs,
        bits=4,
        group_size=8,
        sym=True,
        krot=1,
        pair_ratio=0.5,
        train_rows=32,
        val_rows=16,
        batch_size=16,
        rotation_epochs=1,
        finetune_epochs=1,
        rotation_lr=0.05,
        weight_lr=1e-5,
        quantizer_lr=1e-6,
        seed=0,
        fused_rotation=True,
        stage_cudagraph=False,
        best_state_dtype="fp16",
        stage_impl="fast",
        pair_impl="fast",
        quantizer_impl="reference",
    )

    assert result.val_loss >= 0.0
    assert best_state_dtype_calls == ["fp16", "fp16"]


def test_optimize_paroquant_linear_supports_sgd_optimizer():
    """Guard the direct ParoQuant path against rejecting valid SGD hyperparameters."""
    weight = torch.randn((16, 16), dtype=torch.float32)
    inputs = torch.randn((96, 16), dtype=torch.float32)

    result = optimize_paroquant_linear(
        weight=weight,
        bias=None,
        inputs=inputs,
        bits=4,
        group_size=8,
        sym=True,
        krot=1,
        pair_ratio=0.5,
        train_rows=64,
        val_rows=32,
        batch_size=16,
        rotation_epochs=1,
        finetune_epochs=1,
        rotation_lr=0.05,
        weight_lr=1e-4,
        quantizer_lr=1e-4,
        seed=0,
        optimizer_name="sgd",
        optimizer_weight_decay=0.02,
        sgd_momentum=0.85,
        sgd_dampening=0.0,
        sgd_nesterov=True,
        fused_rotation=False,
        stage_cudagraph=False,
        stage_impl="fast",
        pair_impl="fast",
        quantizer_impl="reference",
    )

    assert result.val_loss >= 0.0
    assert result.pseudo_weight.shape == weight.shape


def test_paroquant_run_stage_only_enables_active_gradients(monkeypatch):
    """Guard that each stage only backpropagates through the parameters it optimizes."""
    pairs, theta_mask = build_random_rotation_buffers(
        in_features=8,
        group_size=8,
        krot=1,
        pair_ratio=0.5,
        seed=0,
        device=torch.device("cpu"),
    )
    model = _ParoQuantOptimLinear(
        torch.randn((8, 8), dtype=torch.float32),
        torch.randn((8,), dtype=torch.float32),
        bits=4,
        group_size=8,
        quantizer_sym=True,
        pairs=pairs,
        theta_mask=theta_mask,
        fused_rotation=False,
    )
    original_flags = {name: param.requires_grad for name, param in model.named_parameters()}
    seen_flags = {}

    def fake_stage_impl(**kwargs):
        del kwargs
        seen_flags.update({name: param.requires_grad for name, param in model.named_parameters()})
        return 0.0, 0.0

    monkeypatch.setattr(paroquant_optimization, "_run_stage_gptqmodel", fake_stage_impl)

    paroquant_optimization._run_stage(
        model=model,
        inputs_train=torch.randn((4, 8), dtype=torch.float32),
        targets_train=torch.randn((4, 8), dtype=torch.float32),
        inputs_val=torch.randn((2, 8), dtype=torch.float32),
        targets_val=torch.randn((2, 8), dtype=torch.float32),
        param_groups=[
            {"params": [model.channel_scales_opt], "lr": 0.05},
            {"params": [model.theta], "lr": 0.05},
        ],
        epochs=1,
        batch_size=2,
        stage_impl="fast",
    )

    assert seen_flags["theta"] is True
    assert seen_flags["channel_scales_opt"] is True
    assert seen_flags["weight"] is False
    assert seen_flags["bias"] is False
    assert {name: param.requires_grad for name, param in model.named_parameters()} == original_flags


def test_paroquant_registers_with_transformers_gptq_quantizer():
    """Guard the HF quantization registry alias used by Evalution loaders."""
    cfg = AutoQuantizationConfig.from_dict(
        {
            "quant_method": "paroquant",
            "bits": 4,
            "group_size": 128,
            "sym": True,
            "format": "paroquant",
        }
    )

    assert isinstance(cfg, GPTQConfig)
    assert getattr(cfg.quant_method, "value", cfg.quant_method) == "gptq"


def test_paroquant_kernel_mapping_uses_paroquant_backend():
    """Guard backend dispatch so ParoQuant does not silently fall back to AWQ."""
    from gptqmodel.nn_modules.qlinear.paroquant import ParoLinear

    assert (
        get_kernel_for_backend(BACKEND.PAROQUANT_CUDA, METHOD.PARO, FORMAT.PAROQUANT)
        is ParoLinear
    )


def test_paroquant_kernel_mapping_uses_paroquant_triton_backend():
    """Guard Triton backend dispatch for ParoQuant-specific runtime modules."""
    from gptqmodel.nn_modules.qlinear.paroquant_triton import ParoQuantTritonLinear

    assert (
        get_kernel_for_backend(BACKEND.PAROQUANT_TRITON, METHOD.PARO, FORMAT.PAROQUANT)
        is ParoQuantTritonLinear
    )


def test_paroquant_identity_rotation_buffers_preserve_input():
    """Guard the identity buffer builder used by no-op and fallback paths."""
    x = torch.randn(3, 128, dtype=torch.float16)
    pairs, theta, channel_scales = build_identity_rotation_buffers(
        in_features=128,
        group_size=128,
        krot=8,
        dtype=torch.float16,
    )

    rotated = apply_paroquant_rotation_reference(
        x,
        pairs,
        theta,
        scales=channel_scales,
        group_size=128,
    )

    torch.testing.assert_close(rotated, x, atol=0, rtol=0)


def test_paroquant_module_default_rotation_buffers_are_identity():
    """Guard fresh runtime modules against invalid all-zero pair buffers."""
    module = ParoLinear(
        bits=4,
        group_size=128,
        sym=True,
        desc_act=False,
        in_features=128,
        out_features=128,
        bias=False,
        register_buffers=True,
    )
    pairs, theta, channel_scales = build_identity_rotation_buffers(
        in_features=module.in_features,
        group_size=module.group_size,
        krot=module.krot,
        dtype=module.theta.dtype,
    )

    assert torch.equal(module.pairs, pairs)
    assert torch.equal(module.theta, theta)
    assert torch.equal(module.channel_scales, channel_scales)


def test_paroquant_processor_is_not_awq_subclass():
    """Guard the dedicated lifecycle split from AWQ requested by the user."""
    assert not issubclass(ParoQuantProcessor, AWQProcessor)


def test_paroquant_processor_resets_reused_module_buckets_per_layer():
    """Guard against cross-layer activation reuse for repeated relative module names."""
    processor = object.__new__(ParoQuantProcessor)
    processor.lock = threading.Lock()
    processor.tasks = {}

    processor._ensure_task_bucket("mlp.gate_proj", layer_index=0)
    processor.tasks["mlp.gate_proj"]["inputs"].append(torch.randn(1, 8))
    processor._ensure_task_bucket("mlp.gate_proj", layer_index=0)
    assert len(processor.tasks["mlp.gate_proj"]["inputs"]) == 1

    processor._ensure_task_bucket("mlp.gate_proj", layer_index=1)
    assert processor.tasks["mlp.gate_proj"]["layer_index"] == 1
    assert processor.tasks["mlp.gate_proj"]["inputs"] == []


def test_paroquant_processor_groups_common_llama_compute_blocks():
    """Guard the planned compute_block optimizer buckets for attention and MLP projections."""
    processor = object.__new__(ParoQuantProcessor)
    processor.qcfg = SimpleNamespace(opt_scope="compute_block")

    state = SimpleNamespace(
        modules={
            "self_attn.q_proj": SimpleNamespace(name="self_attn.q_proj"),
            "self_attn.k_proj": SimpleNamespace(name="self_attn.k_proj"),
            "self_attn.v_proj": SimpleNamespace(name="self_attn.v_proj"),
            "self_attn.o_proj": SimpleNamespace(name="self_attn.o_proj"),
            "mlp.gate_proj": SimpleNamespace(name="mlp.gate_proj"),
            "mlp.up_proj": SimpleNamespace(name="mlp.up_proj"),
            "mlp.down_proj": SimpleNamespace(name="mlp.down_proj"),
        }
    )

    groups = processor._optimization_groups_for_layer(state)

    assert [(label, [module.name for module in modules]) for label, modules in groups] == [
        ("attn_o", ["self_attn.o_proj"]),
        ("attn_qkv", ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]),
        ("mlp_down", ["mlp.down_proj"]),
        ("mlp_gate_up", ["mlp.gate_proj", "mlp.up_proj"]),
    ]


def test_paroquant_processor_module_scope_seed_uses_full_module_name():
    """Guard module scope against collapsing different layer linears onto one archetype seed."""
    full_name_a = "model.layers.0.self_attn.q_proj"
    full_name_b = "model.layers.0.block.self_attn.q_proj"

    module_scope = object.__new__(ParoQuantProcessor)
    module_scope.qcfg = SimpleNamespace(opt_scope="module", opt_seed=3141592653)
    assert module_scope._module_seed(0, full_name_a) != module_scope._module_seed(0, full_name_b)

    grouped_scope = object.__new__(ParoQuantProcessor)
    grouped_scope.qcfg = SimpleNamespace(opt_scope="compute_block", opt_seed=3141592653)
    assert grouped_scope._module_seed(0, full_name_a) == grouped_scope._module_seed(0, full_name_b)


def test_paroquant_prewarm_rotation_extension_skips_unsupported_configs(monkeypatch):
    """Guard the explicit prewarm helper so startup only pays for real fused-kernel cases."""
    calls = []

    monkeypatch.setattr(
        paroquant_utils_module._PAROQUANT_ROTATION_EXTENSION,
        "load",
        lambda: calls.append("load") or True,
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert prewarm_paroquant_rotation_extension(fused_rotation=False, group_size=128, krot=8) is False
    assert prewarm_paroquant_rotation_extension(fused_rotation=True, group_size=64, krot=8) is False
    assert prewarm_paroquant_rotation_extension(fused_rotation=True, group_size=128, krot=4) is False
    assert (
        prewarm_paroquant_rotation_extension(
            fused_rotation=True,
            group_size=128,
            krot=8,
            device=torch.device("cpu"),
        )
        is False
    )
    assert prewarm_paroquant_rotation_extension(fused_rotation=True, group_size=128, krot=8) is True
    assert calls == ["load"]


def test_paroquant_clear_rotation_extension_cache_delegates_to_shared_loader(monkeypatch):
    """Guard the public cache-clear helper so benchmarks can force fresh torch.ops rebuilds."""

    calls = []
    monkeypatch.setattr(
        paroquant_utils_module._PAROQUANT_ROTATION_EXTENSION,
        "clear_cache",
        lambda: calls.append("clear"),
    )

    clear_paroquant_rotation_extension_cache()

    assert calls == ["clear"]


def test_paroquant_rotation_launch_config_honors_env_overrides(monkeypatch):
    """Guard manual launch-shape overrides so benchmarking can pin one kernel variant."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required to validate the ParoQuant launch-config path.")

    monkeypatch.setenv("GPTQMODEL_PAROQUANT_ROTATE_CTA_M", "16")
    monkeypatch.setenv("GPTQMODEL_PAROQUANT_ROTATE_ROW_PAD", "0")

    assert prewarm_paroquant_rotation_extension(
        fused_rotation=True,
        group_size=128,
        krot=8,
        device=torch.device("cuda"),
    )

    cta_m, row_pad = _rotation_launch_config(torch.empty((1, 128), device="cuda", dtype=torch.float16))
    assert (cta_m, row_pad) == (16, 0)


def test_paroquant_rotation_launch_config_autotunes_once_per_shape(monkeypatch):
    """Guard fused rotation autotune so one native shape plan is cached and reused."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required to validate the ParoQuant launch-config path.")

    clear_paroquant_rotation_autotune_cache()
    monkeypatch.delenv("GPTQMODEL_PAROQUANT_ROTATE_CTA_M", raising=False)
    monkeypatch.delenv("GPTQMODEL_PAROQUANT_ROTATE_ROW_PAD", raising=False)
    monkeypatch.setenv("GPTQMODEL_PAROQUANT_ROTATE_AUTOTUNE", "1")
    monkeypatch.setenv("GPTQMODEL_PAROQUANT_ROTATE_AUTOTUNE_WARMUP", "1")
    monkeypatch.setenv("GPTQMODEL_PAROQUANT_ROTATE_AUTOTUNE_ITERS", "1")

    assert prewarm_paroquant_rotation_extension(
        fused_rotation=True,
        group_size=128,
        krot=8,
        device=torch.device("cuda"),
    )

    x = torch.empty((32, 128), device="cuda", dtype=torch.float16)
    pairs = torch.zeros((8, 128), device="cuda", dtype=torch.int16)
    theta = torch.zeros((8, 64), device="cuda", dtype=torch.float16)
    scales = torch.ones((1, 128), device="cuda", dtype=torch.float16)
    x_other = torch.empty((64, 128), device="cuda", dtype=torch.float16)

    assert paroquant_utils_module._rotation_autotune_cache_size() == 0
    first = _rotation_launch_config(x, pairs, theta, scales=scales, group_size=128)
    assert first in {(4, 0), (4, 2), (8, 0), (8, 2), (16, 0), (16, 2)}
    assert paroquant_utils_module._rotation_autotune_cache_size() == 1
    second = _rotation_launch_config(x, pairs, theta, scales=scales, group_size=128)
    third = _rotation_launch_config(x_other, pairs, theta, scales=scales, group_size=128)

    assert second == first
    assert third in {(4, 0), (4, 2), (8, 0), (8, 2), (16, 0), (16, 2)}
    assert paroquant_utils_module._rotation_autotune_cache_size() == 2
    clear_paroquant_rotation_autotune_cache()
    assert paroquant_utils_module._rotation_autotune_cache_size() == 0


def test_paroquant_rotation_launch_config_serializes_concurrent_autotune(monkeypatch):
    """Guard free-threaded launch autotune so one cold shape is measured once at a time."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required to validate the ParoQuant launch-config path.")

    clear_paroquant_rotation_autotune_cache()
    monkeypatch.setattr(paroquant_utils_module, "_load_rotation_extension", lambda: True)
    monkeypatch.setattr(paroquant_utils_module, "_rotation_requested_launch", lambda: (-2, -2))

    state_lock = threading.Lock()
    result_lock = threading.Lock()
    calls = {"launch": 0, "active": 0, "max_active": 0}
    results = []
    failures = []

    def fake_launch_config(x, krot, has_scale, group_size, cta_m, row_pad):
        del x, krot, has_scale, group_size, cta_m, row_pad
        with state_lock:
            calls["launch"] += 1
            calls["active"] += 1
            calls["max_active"] = max(calls["max_active"], calls["active"])
        try:
            time.sleep(0.05)
            return (8, 2)
        finally:
            with state_lock:
                calls["active"] -= 1

    def fake_op(name):
        if name == "launch_config":
            return fake_launch_config
        raise AssertionError(f"unexpected op lookup: {name}")

    monkeypatch.setattr(paroquant_utils_module._PAROQUANT_ROTATION_EXTENSION, "op", fake_op)

    x = torch.empty((32, 128), device="cuda", dtype=torch.float16)
    pairs = torch.zeros((8, 128), device="cuda", dtype=torch.int16)
    theta = torch.zeros((8, 64), device="cuda", dtype=torch.float16)
    scales = torch.ones((1, 128), device="cuda", dtype=torch.float16)
    start_barrier = threading.Barrier(4)

    def worker():
        try:
            start_barrier.wait()
            resolved = _rotation_launch_config(x, pairs, theta, scales=scales, group_size=128)
            with result_lock:
                results.append(resolved)
        except BaseException as exc:  # pragma: no cover - test should fail below instead.
            with result_lock:
                failures.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert failures == []
    assert results == [(8, 2)] * 4
    assert calls["launch"] == 1
    assert calls["max_active"] == 1

    clear_paroquant_rotation_autotune_cache()


def test_paroquant_rotation_helper_reuses_resolved_autotune_launch(monkeypatch):
    """Guard the fused helper so autotune resolves once and steady-state runs explicitly."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required to validate the ParoQuant launch-config path.")

    clear_paroquant_rotation_autotune_cache()
    monkeypatch.setattr(paroquant_utils_module, "_load_rotation_extension", lambda: True)
    monkeypatch.setattr(paroquant_utils_module, "_rotation_requested_launch", lambda: (-2, -2))

    calls = {"launch": 0, "rotate": 0, "configs": []}

    def fake_launch_config(x, krot, has_scale, group_size, cta_m, row_pad):
        del x, krot, has_scale, group_size, cta_m, row_pad
        calls["launch"] += 1
        return (16, 2)

    def fake_rotate(x, pairs, theta, scales, group_size, cta_m, row_pad):
        del pairs, theta, scales, group_size
        calls["rotate"] += 1
        calls["configs"].append((cta_m, row_pad))
        return x.clone()

    def fake_op(name):
        if name == "launch_config":
            return fake_launch_config
        if name == "rotate":
            return fake_rotate
        raise AssertionError(f"unexpected op lookup: {name}")

    monkeypatch.setattr(paroquant_utils_module._PAROQUANT_ROTATION_EXTENSION, "op", fake_op)

    x = torch.randn((32, 128), device="cuda", dtype=torch.float16)
    pairs = torch.zeros((8, 128), device="cuda", dtype=torch.int16)
    theta = torch.zeros((8, 64), device="cuda", dtype=torch.float16)
    scales = torch.ones((1, 128), device="cuda", dtype=torch.float16)

    first = apply_paroquant_rotation(x, pairs, theta, scales=scales, group_size=128)
    second = apply_paroquant_rotation(x, pairs, theta, scales=scales, group_size=128)

    assert torch.equal(first, x)
    assert torch.equal(second, x)
    assert calls["launch"] == 1
    assert calls["rotate"] == 2
    assert calls["configs"] == [(16, 2), (16, 2)]

    clear_paroquant_rotation_autotune_cache()
    third = apply_paroquant_rotation(x, pairs, theta, scales=scales, group_size=128)

    assert torch.equal(third, x)
    assert calls["launch"] == 2
    assert calls["rotate"] == 3
    assert calls["configs"][-1] == (16, 2)


def test_paroquant_processor_prewarm_runtime_runs_once(monkeypatch):
    """Guard startup prewarm so the looper does not retry the fused extension every layer."""
    calls = []

    monkeypatch.setattr(
        paroquant_processor_module,
        "prewarm_paroquant_rotation_extension",
        lambda **kwargs: calls.append(kwargs) or True,
    )

    processor = object.__new__(ParoQuantProcessor)
    processor.qcfg = SimpleNamespace(opt_fused_rotation=True, group_size=128, krot=8)
    processor._runtime_prewarmed = False

    processor.prewarm_runtime()
    processor.prewarm_runtime()

    assert len(calls) == 1
    assert calls[0] == {
        "fused_rotation": True,
        "group_size": 128,
        "krot": 8,
    }


def test_paroquant_processor_grouped_modes_capture_pristine_context_outside_subset_forward():
    """Guard grouped modes against treating early-stopped subset forwards as full-layer targets."""
    processor = object.__new__(ParoQuantProcessor)
    processor.qcfg = SimpleNamespace(opt_scope="compute_block")
    assert processor.uses_grouped_optimization() is True
    assert processor.capture_layer_forward_context_during_subset() is False

    processor.qcfg = SimpleNamespace(opt_scope="layer")
    assert processor.uses_grouped_optimization() is True
    assert processor.capture_layer_forward_context_during_subset() is False

    processor.qcfg = SimpleNamespace(opt_scope="module")
    assert processor.uses_grouped_optimization() is False
    assert processor.capture_layer_forward_context_during_subset() is False


def test_paroquant_processor_disables_stage_cudagraph_for_module_scope_loop():
    """Guard module scope against CUDA-graph private-pool growth across many linear optimizations."""
    processor = object.__new__(ParoQuantProcessor)

    processor.qcfg = SimpleNamespace(opt_scope="module", opt_stage_cudagraph=True)
    assert processor._module_scope_stage_cudagraph_enabled() is False

    processor.qcfg = SimpleNamespace(opt_scope="module", opt_stage_cudagraph=False)
    assert processor._module_scope_stage_cudagraph_enabled() is False

    processor.qcfg = SimpleNamespace(opt_scope="compute_block", opt_stage_cudagraph=True)
    assert processor._module_scope_stage_cudagraph_enabled() is True

    processor.qcfg = SimpleNamespace(opt_scope="layer", opt_stage_cudagraph=True)
    assert processor._module_scope_stage_cudagraph_enabled() is True


def test_paroquant_processor_module_quantize_forces_stage_cudagraph_off(monkeypatch):
    """Guard the full model module loop against per-linear CUDA-graph pool retention."""
    stage_cudagraph_calls = []
    optimizer_name_calls = []
    optimizer_kwargs_calls = []

    def fake_optimize_paroquant_linear(*, weight, stage_cudagraph=None, optimizer_name="adamw", **kwargs):
        stage_cudagraph_calls.append(stage_cudagraph)
        optimizer_name_calls.append(optimizer_name)
        optimizer_kwargs_calls.append(kwargs)
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

    monkeypatch.setattr(paroquant_processor_module, "optimize_paroquant_linear", fake_optimize_paroquant_linear)

    processor = object.__new__(ParoQuantProcessor)
    processor.qcfg = SimpleNamespace(
        opt_scope="module",
        opt_stage_cudagraph=True,
        dynamic_get=lambda _name, _field, default=None: default,
        runtime_bits=4,
        group_size=128,
        sym=True,
        krot=8,
        opt_pair_ratio=0.25,
        opt_train_samples=128,
        opt_validation_samples=32,
        opt_batch_size=16,
        opt_rotation_epochs=1,
        opt_finetune_epochs=1,
        opt_rotation_lr=0.05,
        opt_weight_lr=1e-5,
        opt_quantizer_lr=1e-6,
        opt_seed=0,
        opt_optimizer="sgd",
        opt_weight_decay=0.02,
        opt_betas=(0.8, 0.9),
        opt_eps=1e-8,
        opt_amsgrad=True,
        opt_sgd_momentum=0.85,
        opt_sgd_dampening=0.0,
        opt_sgd_nesterov=True,
        opt_fused_rotation=True,
        opt_best_state_dtype="fp16",
        opt_stage_impl="fast",
        opt_pair_impl="fast",
        opt_quantizer_impl="reference",
        opt_channel_scale_clamp_min=1e-2,
        opt_channel_scale_clamp_max=1e2,
    )
    processor.calculate_w_wq_diff = False
    processor.lock = threading.Lock()

    module = SimpleNamespace(
        name="self_attn.q_proj",
        full_name="model.layers.0.self_attn.q_proj",
        layer_index=0,
        weight=torch.nn.Parameter(torch.randn((8, 128), dtype=torch.float32)),
        bias=None,
        state={},
    )

    processor._quantize_one_module(module, torch.randn((32, 128), dtype=torch.float32))

    assert stage_cudagraph_calls == [False]
    assert optimizer_name_calls == ["sgd"]
    assert len(optimizer_kwargs_calls) == 1
    forwarded_kwargs = optimizer_kwargs_calls[0]
    assert forwarded_kwargs["bias"] is None
    assert isinstance(forwarded_kwargs["inputs"], torch.Tensor)
    assert tuple(forwarded_kwargs["inputs"].shape) == (32, 128)
    assert forwarded_kwargs["bits"] == 4
    assert forwarded_kwargs["group_size"] == 128
    assert forwarded_kwargs["sym"] is True
    assert forwarded_kwargs["krot"] == 8
    assert forwarded_kwargs["pair_ratio"] == pytest.approx(0.25)
    assert forwarded_kwargs["train_rows"] == 128
    assert forwarded_kwargs["val_rows"] == 32
    assert forwarded_kwargs["batch_size"] == 16
    assert forwarded_kwargs["rotation_epochs"] == 1
    assert forwarded_kwargs["finetune_epochs"] == 1
    assert forwarded_kwargs["rotation_lr"] == pytest.approx(0.05)
    assert forwarded_kwargs["weight_lr"] == pytest.approx(1e-5)
    assert forwarded_kwargs["quantizer_lr"] == pytest.approx(1e-6)
    assert isinstance(forwarded_kwargs["seed"], int)
    assert forwarded_kwargs["optimizer_weight_decay"] == pytest.approx(0.02)
    assert forwarded_kwargs["optimizer_betas"] == pytest.approx((0.8, 0.9))
    assert forwarded_kwargs["optimizer_eps"] == pytest.approx(1e-8)
    assert forwarded_kwargs["optimizer_amsgrad"] is True
    assert forwarded_kwargs["sgd_momentum"] == pytest.approx(0.85)
    assert forwarded_kwargs["sgd_dampening"] == pytest.approx(0.0)
    assert forwarded_kwargs["sgd_nesterov"] is True
    assert forwarded_kwargs["fused_rotation"] is True
    assert forwarded_kwargs["gradient_checkpointing"] is False
    assert forwarded_kwargs["best_state_dtype"] == "fp16"
    assert forwarded_kwargs["stage_impl"] == "fast"
    assert forwarded_kwargs["pair_impl"] == "fast"
    assert forwarded_kwargs["quantizer_impl"] == "reference"
    assert forwarded_kwargs["scale_clamp_min"] == pytest.approx(1e-2)
    assert forwarded_kwargs["scale_clamp_max"] == pytest.approx(1e2)


def test_paroquant_processor_layer_scope_live_path_is_dense_only():
    """Guard the official-like live layer path for dense decoder layers only."""
    dense_modules = [
        SimpleNamespace(name="self_attn.q_proj"),
        SimpleNamespace(name="self_attn.k_proj"),
        SimpleNamespace(name="self_attn.v_proj"),
        SimpleNamespace(name="self_attn.o_proj"),
        SimpleNamespace(name="mlp.gate_proj"),
        SimpleNamespace(name="mlp.up_proj"),
        SimpleNamespace(name="mlp.down_proj"),
    ]
    moe_modules = [
        SimpleNamespace(name="self_attn.q_proj"),
        SimpleNamespace(name="mlp.experts.0.gate_up_proj"),
        SimpleNamespace(name="mlp.experts.0.down_proj"),
    ]

    assert ParoQuantProcessor._supports_live_layer_scope(dense_modules) is True
    assert ParoQuantProcessor._supports_live_layer_scope(moe_modules) is False


@pytest.mark.parametrize(
    ("opt_scope", "expected_capture"),
    [
        ("module", False),
        ("compute_block", True),
        ("layer", True),
    ],
)
def test_paroquant_processor_enables_layer_context_capture_only_for_grouped_scopes(opt_scope, expected_capture):
    """Guard module scope against retaining pristine layer IO it never consumes."""
    processor = ParoQuantProcessor(
        tokenizer=None,
        qcfg=ParoConfig(bits=4, group_size=128, opt_scope=opt_scope),
        calibration=None,
        prepare_dataset_func=None,
        calibration_concat_size=None,
        calibration_sort=None,
        batch_size=1,
        gptq_model=None,
        model=None,
    )

    assert processor.execution_config.capture_layer_forward_context is expected_capture


@pytest.mark.parametrize(
    ("opt_scope", "opt_gradient_checkpointing", "expected"),
    [
        ("module", None, False),
        ("compute_block", None, False),
        ("layer", None, True),
        ("module", True, True),
        ("layer", False, False),
    ],
)
def test_paroquant_processor_resolves_gradient_checkpointing_by_scope(opt_scope, opt_gradient_checkpointing, expected):
    """Processor runtime should mirror the config default and explicit override semantics."""

    processor = object.__new__(ParoQuantProcessor)
    processor.qcfg = SimpleNamespace(
        opt_scope=opt_scope,
        opt_gradient_checkpointing=opt_gradient_checkpointing,
    )

    assert processor._gradient_checkpointing_enabled() is expected


def test_paroquant_processor_skips_pristine_layer_clone_for_layer_scope():
    """Guard layer scope against retaining an unused pristine layer clone on CPU."""

    processor = object.__new__(ParoQuantProcessor)
    processor.qcfg = SimpleNamespace(opt_scope="layer")
    processor._layer_states = {}
    processor._layer_states_lock = threading.Lock()

    layer = torch.nn.Linear(4, 4, bias=False)
    processor.receive_pristine_layer_module(layer_index=0, layer_module=layer)

    assert processor._layer_states == {}


def test_paroquant_processor_routes_non_module_units_through_group_optimizer():
    """Guard that compute_block/layer modes now use grouped optimization instead of raising."""
    processor = object.__new__(ParoQuantProcessor)
    processor.qcfg = SimpleNamespace(opt_scope="compute_block")
    processor.fallback = True
    processor.lock = threading.Lock()
    processor.tasks = {}
    processor.calculate_w_wq_diff = False
    processor._log_quant_result = lambda *args, **kwargs: None  # type: ignore[method-assign]

    layer = torch.nn.Module()
    layer.self_attn = torch.nn.Module()
    layer.self_attn.q_proj = torch.nn.Linear(8, 8, bias=False)
    layer.self_attn.k_proj = torch.nn.Linear(8, 8, bias=False)
    layer.self_attn.v_proj = torch.nn.Linear(8, 8, bias=False)

    q_proj = NamedModule(layer.self_attn.q_proj, "self_attn.q_proj", "model.layers.0.self_attn.q_proj", 0)
    k_proj = NamedModule(layer.self_attn.k_proj, "self_attn.k_proj", "model.layers.0.self_attn.k_proj", 0)
    v_proj = NamedModule(layer.self_attn.v_proj, "self_attn.v_proj", "model.layers.0.self_attn.v_proj", 0)
    processor._layer_input_features = lambda _state: {  # type: ignore[method-assign]
        q_proj.name: torch.randn(4, 8),
        k_proj.name: torch.randn(4, 8),
        v_proj.name: torch.randn(4, 8),
    }

    observed_groups = []

    def fake_optimize_group(state, group_modules):
        del state
        observed_groups.append([module.name for module in group_modules])
        results = {}
        for module in group_modules:
            weight = module.weight.data.detach()
            results[module.name] = SimpleNamespace(
                pseudo_weight=weight + 1.0,
                pack_weight=weight + 2.0,
                q_scales=torch.ones((weight.shape[0], max(1, weight.shape[1] // 8)), dtype=weight.dtype),
                q_zeros=torch.zeros((weight.shape[0], max(1, weight.shape[1] // 8)), dtype=torch.int32),
                pairs=torch.zeros((1, 8), dtype=torch.int16),
                theta=torch.zeros((1, weight.shape[1] // 2), dtype=weight.dtype),
                channel_scales=torch.ones((1, weight.shape[1]), dtype=weight.dtype),
            )
        return results, 0.25

    processor._optimize_group = fake_optimize_group  # type: ignore[method-assign]

    state = SimpleNamespace(
        quantized=False,
        modules={q_proj.name: q_proj, k_proj.name: k_proj, v_proj.name: v_proj},
        layer_inputs=[[torch.randn(1, 2, 8)]],
        layer_outputs=[[torch.randn(1, 2, 8)]],
        pending_modules=set(),
        processed_subsets={0},
        subset_total=1,
    )

    original_q_weight = q_proj.weight.data.clone()
    processor._quantize_layer(layer_index=0, state=state)

    assert observed_groups == [["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]]
    torch.testing.assert_close(q_proj.weight.data, original_q_weight + 1.0)
    assert state.quantized is True
    assert state.modules == {}
    assert state.pending_modules == set()
    assert state.processed_subsets == set()


def test_paroquant_processor_compute_block_scope_flushes_cuda_cache_between_groups(monkeypatch):
    """Guard compute_block scope against carrying allocator cache forward between grouped passes when offload is on."""

    processor = object.__new__(ParoQuantProcessor)
    processor.qcfg = SimpleNamespace(opt_scope="compute_block", offload_to_disk=True)
    processor.fallback = True
    processor.lock = threading.Lock()
    processor.tasks = {}
    processor.calculate_w_wq_diff = False
    processor._log_quant_result = lambda *args, **kwargs: None  # type: ignore[method-assign]

    layer = torch.nn.Module()
    layer.self_attn = torch.nn.Module()
    layer.self_attn.q_proj = torch.nn.Linear(8, 8, bias=False)
    layer.self_attn.k_proj = torch.nn.Linear(8, 8, bias=False)
    layer.self_attn.v_proj = torch.nn.Linear(8, 8, bias=False)

    q_proj = NamedModule(layer.self_attn.q_proj, "self_attn.q_proj", "model.layers.0.self_attn.q_proj", 0)
    k_proj = NamedModule(layer.self_attn.k_proj, "self_attn.k_proj", "model.layers.0.self_attn.k_proj", 0)
    v_proj = NamedModule(layer.self_attn.v_proj, "self_attn.v_proj", "model.layers.0.self_attn.v_proj", 0)
    processor._layer_input_features = lambda _state: {  # type: ignore[method-assign]
        q_proj.name: torch.randn(4, 8),
        k_proj.name: torch.randn(4, 8),
        v_proj.name: torch.randn(4, 8),
    }

    def fake_optimize_group(state, group_modules):
        del state
        results = {}
        for module in group_modules:
            weight = module.weight.data.detach()
            results[module.name] = SimpleNamespace(
                pseudo_weight=weight + 1.0,
                pack_weight=weight + 2.0,
                q_scales=torch.ones((weight.shape[0], max(1, weight.shape[1] // 8)), dtype=weight.dtype),
                q_zeros=torch.zeros((weight.shape[0], max(1, weight.shape[1] // 8)), dtype=torch.int32),
                pairs=torch.zeros((1, 8), dtype=torch.int16),
                theta=torch.zeros((1, weight.shape[1] // 2), dtype=weight.dtype),
                channel_scales=torch.ones((1, weight.shape[1]), dtype=weight.dtype),
            )
        return results, 0.25

    processor._optimize_group = fake_optimize_group  # type: ignore[method-assign]

    empty_cache_calls = []
    monkeypatch.setattr(
        paroquant_processor_module,
        "torch_empty_cache",
        lambda device=None, gc=True, sync=False: empty_cache_calls.append(
            {"device": device, "gc": gc, "sync": sync}
        ),
    )

    state = SimpleNamespace(
        quantized=False,
        modules={q_proj.name: q_proj, k_proj.name: k_proj, v_proj.name: v_proj},
        layer_inputs=[[torch.randn(1, 2, 8)]],
        layer_outputs=[[torch.randn(1, 2, 8)]],
        pending_modules=set(),
        processed_subsets={0},
        subset_total=1,
    )

    processor._quantize_layer(layer_index=0, state=state)

    assert empty_cache_calls == [{"device": torch.device("cpu"), "gc": False, "sync": True}]


def test_paroquant_processor_layer_scope_falls_back_to_clone_for_expert_like_groups(monkeypatch):
    """Guard expert-like layer groups against accidentally taking the dense live path."""
    import gptqmodel.looper.paroquant_processor as paroquant_processor_module

    processor = object.__new__(ParoQuantProcessor)
    processor.qcfg = SimpleNamespace(
        opt_scope="layer",
        opt_rotation_lr=0.05,
        opt_weight_lr=1e-5,
        opt_quantizer_lr=1e-6,
        opt_rotation_epochs=0,
        opt_finetune_epochs=0,
    )
    state = SimpleNamespace()
    group_modules = [
        SimpleNamespace(name="self_attn.q_proj"),
        SimpleNamespace(name="mlp.experts.0.gate_up_proj"),
    ]

    live_calls = []
    clone_calls = []

    def fake_live(_state, _modules):
        live_calls.append(True)
        return {}, 0.0

    processor._optimize_live_layer = fake_live  # type: ignore[method-assign]

    def _fake_optim_module():
        return SimpleNamespace(
            channel_scales_opt=torch.nn.Parameter(torch.ones(1)),
            theta=torch.nn.Parameter(torch.zeros(1)),
            weight=torch.nn.Parameter(torch.ones(1, 1)),
            quantizer=None,
            init_quantizer=lambda: None,
        )

    def fake_build_group_optim_layer(_state, _modules):
        clone_calls.append(True)
        return torch.nn.Linear(4, 4, bias=False), {
            module.name: _fake_optim_module() for module in _modules
        }

    processor._build_group_optim_layer = fake_build_group_optim_layer  # type: ignore[method-assign]
    processor._group_dataset_for_device = lambda *_args, **_kwargs: ([], [], [], [], [], [], [], [], [], [])  # type: ignore[method-assign]
    processor._run_group_stage = lambda *args, **kwargs: (0.0, 0.0)  # type: ignore[method-assign]
    monkeypatch.setattr(
        paroquant_processor_module,
        "_result_from_model",
        lambda _optim_module, **_kwargs: SimpleNamespace(ok=True),
    )

    results, val_loss = processor._optimize_group(state, group_modules)

    assert live_calls == []
    assert clone_calls == [True]
    assert set(results) == {"self_attn.q_proj", "mlp.experts.0.gate_up_proj"}
    assert val_loss == 0.0


def test_paroquant_processor_captures_first_layer_forward_context():
    """Guard that grouped optimization modes keep the original float layer IO once."""
    processor = object.__new__(ParoQuantProcessor)
    processor._layer_states = {}
    processor._layer_states_lock = threading.Lock()

    first_inputs = [[torch.randn(1, 4)]]
    first_kwargs = [{"attention_mask": torch.ones((1, 4), dtype=torch.int64)}]
    first_outputs = [[torch.randn(1, 4)]]
    second_inputs = [[torch.randn(1, 4)]]
    second_kwargs = [{"attention_mask": torch.zeros((1, 4), dtype=torch.int64)}]
    second_outputs = [[torch.randn(1, 4)]]

    processor.receive_layer_forward_context(
        layer_index=0,
        layer_inputs=first_inputs,
        layer_input_kwargs=first_kwargs,
        layer_outputs=first_outputs,
        subset_index=0,
        subset_total=2,
    )
    processor.receive_layer_forward_context(
        layer_index=0,
        layer_inputs=second_inputs,
        layer_input_kwargs=second_kwargs,
        layer_outputs=second_outputs,
        subset_index=1,
        subset_total=2,
    )

    state = processor._get_layer_state(0)
    assert state.layer_inputs is first_inputs
    assert state.layer_input_kwargs is first_kwargs
    assert state.layer_outputs is first_outputs
    assert state.subset_total == 2


def test_paroquant_processor_group_clean_inputs_seed_from_input_cache():
    """Guard grouped clean targets against aliasing the noisy replay cache."""
    processor = object.__new__(ParoQuantProcessor)
    processor.qcfg = SimpleNamespace(opt_scope="layer", opt_train_on_noisy_inputs=True)
    clean_inputs = [[torch.randn(1, 4)]]
    noisy_inputs = [[torch.randn(1, 4)]]
    cache = InputCache(
        layer_inputs=clean_inputs,
        layer_input_kwargs=[{}],
        position_ids=[None],
        attention_masks=[None],
    )

    processor.receive_input_cache(cache)

    assert processor.inputs_cache.layer_inputs is clean_inputs
    assert processor.clean_group_layer_inputs(layer_index=0, layer_inputs=noisy_inputs) is clean_inputs


def test_paroquant_processor_group_clean_inputs_default_to_noisy_stream():
    """Guard default grouped behavior against enabling train-on-noisy-inputs implicitly."""
    processor = object.__new__(ParoQuantProcessor)
    processor.qcfg = SimpleNamespace(opt_scope="layer", opt_train_on_noisy_inputs=False)
    clean_inputs = [[torch.randn(1, 4)]]
    noisy_inputs = [[torch.randn(1, 4)]]
    processor.receive_input_cache(
        InputCache(
            layer_inputs=clean_inputs,
            layer_input_kwargs=[{}],
            position_ids=[None],
            attention_masks=[None],
        )
    )

    assert processor.clean_group_layer_inputs(layer_index=0, layer_inputs=noisy_inputs) is noisy_inputs


def test_paroquant_processor_group_capture_uses_pristine_module_and_clean_inputs_for_targets():
    """Guard grouped capture so targets come from the untouched module on the clean stream."""

    class _PristineLayer(torch.nn.Module):
        def forward(self, x, attention_mask=None, position_ids=None, use_cache=False):
            del attention_mask, position_ids, use_cache
            return x + 1.0

    class _HookedLikeLayer(torch.nn.Module):
        def forward(self, x, attention_mask=None, position_ids=None, use_cache=False):
            del attention_mask, position_ids, use_cache
            return x + 100.0

    class _DummyPB:
        def manual(self):
            return self

        def set(self, **kwargs):
            del kwargs
            return self

        def title(self, _value):
            return self

        def subtitle(self, _value):
            return self

        def draw(self):
            return self

        def close(self):
            return None

    class _DummyLog:
        def pb(self, _iterable):
            return _DummyPB()

    class _DummyLooper:
        def _resolve_batch_total(self, _num_batches, layer_inputs):
            return len(layer_inputs)

        def _collect_row_counts(self, layer_inputs):
            return [1 for _ in layer_inputs]

        def _run_forward_batches(
            self,
            *,
            module,
            processor,
            layer_inputs,
            layer_input_kwargs,
            position_ids,
            attention_masks,
            cur_layer_device,
            is_lm_head_module,
            shared_kv_cache_dict,
            layer_index,
            need_outputs,
            reuse_kv,
            progress_pb,
            progress_title,
            progress_stage,
            progress_rows_per_batch,
            progress_total_rows,
            force_serial,
            preserve_module_devices,
        ):
            del (
                processor,
                position_ids,
                attention_masks,
                cur_layer_device,
                is_lm_head_module,
                shared_kv_cache_dict,
                layer_index,
                need_outputs,
                reuse_kv,
                progress_pb,
                progress_title,
                progress_stage,
                progress_rows_per_batch,
                progress_total_rows,
                force_serial,
                preserve_module_devices,
            )
            outputs = []
            for batch_inputs, batch_kwargs in zip(layer_inputs, layer_input_kwargs):
                outputs.append([module(batch_inputs[0], **batch_kwargs)])
            return outputs

    processor = object.__new__(ParoQuantProcessor)
    processor.qcfg = SimpleNamespace(opt_scope="layer", opt_train_on_noisy_inputs=True)
    processor._layer_states = {}
    processor._layer_states_lock = threading.Lock()

    clean_inputs = [[torch.tensor([[1.0]])]]
    noisy_inputs = [[torch.tensor([[3.0]])]]
    clean_cache = InputCache(
        layer_inputs=clean_inputs,
        layer_input_kwargs=[{}],
        position_ids=[None],
        attention_masks=[None],
    )
    processor.receive_input_cache(clean_cache)

    _capture_pristine_group_context(
        _DummyLooper(),
        processor=processor,
        module=_HookedLikeLayer(),
        pristine_module=_PristineLayer(),
        subset_plans=[SimpleNamespace()],
        layer_inputs=noisy_inputs,
        layer_input_kwargs=[{}],
        position_ids=[None],
        attention_masks=[None],
        cur_layer_device=torch.device("cpu"),
        is_lm_head_module=False,
        shared_kv_cache_dict={},
        layer_index=0,
        layer_descriptor="model.layers.0",
        full={},
        log=_DummyLog(),
        region_timer=None,
    )

    state = processor._get_layer_state(0)
    assert state.layer_inputs is noisy_inputs
    torch.testing.assert_close(state.layer_outputs[0][0], torch.tensor([[2.0]]))
    torch.testing.assert_close(processor.clean_group_layer_inputs(layer_index=1, layer_inputs=noisy_inputs)[0][0], torch.tensor([[2.0]]))


def test_paroquant_processor_group_capture_advances_clean_stream_without_subset_plans():
    """Guard clean/noisy replay semantics for grouped layers that are dynamically skipped."""

    class _ToyLayer(torch.nn.Module):
        def forward(self, x, attention_mask=None, position_ids=None, use_cache=False):
            del attention_mask, position_ids, use_cache
            return x + 5.0

    class _DummyPB:
        def manual(self):
            return self

        def set(self, **kwargs):
            del kwargs
            return self

        def title(self, _value):
            return self

        def subtitle(self, _value):
            return self

        def draw(self):
            return self

        def close(self):
            return None

    class _DummyLog:
        def pb(self, _iterable):
            return _DummyPB()

    class _DummyLooper:
        def _resolve_batch_total(self, _num_batches, layer_inputs):
            return len(layer_inputs)

        def _collect_row_counts(self, layer_inputs):
            return [1 for _ in layer_inputs]

        def _run_forward_batches(self, **kwargs):
            layer_inputs = kwargs["layer_inputs"]
            module = kwargs["module"]
            layer_input_kwargs = kwargs["layer_input_kwargs"]
            return [[module(batch[0], **batch_kwargs)] for batch, batch_kwargs in zip(layer_inputs, layer_input_kwargs)]

    processor = object.__new__(ParoQuantProcessor)
    processor.qcfg = SimpleNamespace(opt_scope="layer", opt_train_on_noisy_inputs=True)
    processor._layer_states = {}
    processor._layer_states_lock = threading.Lock()
    processor.receive_input_cache(
        InputCache(
            layer_inputs=[[torch.tensor([[2.0]])]],
            layer_input_kwargs=[{}],
            position_ids=[None],
            attention_masks=[None],
        )
    )

    noisy_inputs = [[torch.tensor([[9.0]])]]
    _capture_pristine_group_context(
        _DummyLooper(),
        processor=processor,
        module=_ToyLayer(),
        pristine_module=None,
        subset_plans=[],
        layer_inputs=noisy_inputs,
        layer_input_kwargs=[{}],
        position_ids=[None],
        attention_masks=[None],
        cur_layer_device=torch.device("cpu"),
        is_lm_head_module=False,
        shared_kv_cache_dict={},
        layer_index=0,
        layer_descriptor="model.layers.0",
        full={},
        log=_DummyLog(),
        region_timer=None,
    )

    state = processor._get_layer_state(0)
    assert state.layer_inputs is None
    assert state.layer_outputs is None
    torch.testing.assert_close(processor.clean_group_layer_inputs(layer_index=1, layer_inputs=noisy_inputs)[0][0], torch.tensor([[7.0]]))


def test_paroquant_quantize_layer_clears_stored_forward_context():
    """Guard that transient grouped-optimization IO snapshots do not leak across layers."""
    module = SimpleNamespace(
        name="mlp.gate_proj",
        full_name="model.layers.0.mlp.gate_proj",
        weight=SimpleNamespace(data=torch.randn(8, 8)),
        bias=None,
    )
    processor = object.__new__(ParoQuantProcessor)
    processor.qcfg = SimpleNamespace(opt_scope="module")
    processor.fallback = True
    processor.lock = threading.Lock()
    processor.tasks = {"mlp.gate_proj": {"inputs": [torch.randn(1, 8)], "layer_index": 0}}
    processor._layer_input_features = lambda _state: {"mlp.gate_proj": torch.randn(4, 8)}
    processor._quantize_one_module = lambda named_module, feat: (0.0, float(feat.numel() > 0))  # type: ignore[method-assign]
    processor._log_quant_result = lambda *args, **kwargs: None  # type: ignore[method-assign]

    state = SimpleNamespace(
        quantized=False,
        modules={"mlp.gate_proj": module},
        pending_modules=set(),
        processed_subsets={0},
        pristine_layer_module=torch.nn.Linear(8, 8, bias=False),
        prepared_group_source_module=torch.nn.Linear(8, 8, bias=False),
        prepared_group_source_module_by_device={"cpu": torch.nn.Linear(8, 8, bias=False)},
        layer_inputs=[[torch.randn(1, 8)]],
        layer_input_kwargs=[{"attention_mask": torch.ones((1, 8), dtype=torch.int64)}],
        layer_outputs=[[torch.randn(1, 8)]],
        grouped_dataset=("cached",),
        grouped_dataset_by_device={"cpu": ("cached",)},
        subset_total=1,
    )

    processor._quantize_layer(layer_index=0, state=state)

    assert state.quantized is True
    assert state.modules == {}
    assert state.pending_modules == set()
    assert state.processed_subsets == set()
    assert state.layer_inputs is None
    assert state.layer_input_kwargs is None
    assert state.layer_outputs is None
    assert state.pristine_layer_module is None
    assert state.prepared_group_source_module is None
    assert state.prepared_group_source_module_by_device is None
    assert state.grouped_dataset is None
    assert state.grouped_dataset_by_device is None
    assert state.subset_total is None
    assert processor.tasks["mlp.gate_proj"]["inputs"] == []


def test_paroquant_processor_builds_group_optim_layer_clone():
    """Guard that compute_block/layer modes can swap selected modules into a cloned float layer."""

    class _ToyAttn(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(_attn_implementation="sdpa")
            self.q_proj = torch.nn.Linear(8, 8, bias=False)
            self.k_proj = torch.nn.Linear(8, 8, bias=False)
            self.v_proj = torch.nn.Linear(8, 8, bias=False)
            self.o_proj = torch.nn.Linear(8, 8, bias=False)

    class _ToyMlp(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = torch.nn.Linear(8, 8, bias=False)
            self.up_proj = torch.nn.Linear(8, 8, bias=False)
            self.down_proj = torch.nn.Linear(8, 8, bias=False)

    class _ToyLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = _ToyAttn()
            self.mlp = _ToyMlp()

    def _dynamic_get(_module_name, _key, default=None):
        return default

    layer = _ToyLayer().half()
    q_proj = NamedModule(layer.self_attn.q_proj, "self_attn.q_proj", "model.layers.0.self_attn.q_proj", 0)
    k_proj = NamedModule(layer.self_attn.k_proj, "self_attn.k_proj", "model.layers.0.self_attn.k_proj", 0)

    processor = object.__new__(ParoQuantProcessor)
    processor.qcfg = SimpleNamespace(
        opt_scope="compute_block",
        runtime_bits=4,
        group_size=8,
        sym=True,
        krot=1,
        opt_seed=0,
        opt_pair_ratio=0.5,
        opt_pair_impl="fast",
        opt_quantizer_impl="reference",
        opt_fused_rotation=False,
        opt_channel_scale_clamp_min=1e-2,
        opt_channel_scale_clamp_max=1e2,
        dynamic_get=_dynamic_get,
    )

    state = SimpleNamespace(layer_module=layer)
    layer_clone, optim_modules = processor._build_group_optim_layer(state, [q_proj, k_proj])

    assert layer_clone is not layer
    assert isinstance(layer.self_attn.q_proj, torch.nn.Linear)
    assert isinstance(layer.self_attn.k_proj, torch.nn.Linear)
    assert isinstance(layer_clone.self_attn.q_proj, _ParoQuantOptimLinear)
    assert isinstance(layer_clone.self_attn.k_proj, _ParoQuantOptimLinear)
    assert isinstance(layer_clone.self_attn.v_proj, torch.nn.Linear)
    assert set(optim_modules) == {"self_attn.q_proj", "self_attn.k_proj"}
    assert layer_clone.self_attn.q_proj.weight.dtype == torch.float32
    assert layer_clone.self_attn.k_proj.weight.dtype == torch.float32
    assert layer_clone.self_attn.config._attn_implementation == "eager"


def test_paroquant_processor_builds_group_optim_layer_from_pristine_snapshot():
    """Guard grouped clones against inheriting HookedLinear-style mutations from the live layer."""

    class _HookedLike(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(8, 8))

        def forward(self, x):
            return x

    class _ToyAttn(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(_attn_implementation="sdpa")
            self.q_proj = torch.nn.Linear(8, 8, bias=False)
            self.k_proj = torch.nn.Linear(8, 8, bias=False)
            self.v_proj = torch.nn.Linear(8, 8, bias=False)
            self.o_proj = torch.nn.Linear(8, 8, bias=False)

    class _ToyLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = _ToyAttn()

    def _dynamic_get(_module_name, _key, default=None):
        return default

    pristine_layer = _ToyLayer().half()
    live_layer = copy.deepcopy(pristine_layer)
    live_layer.self_attn.o_proj = _HookedLike()

    q_proj = NamedModule(
        live_layer.self_attn.q_proj,
        "self_attn.q_proj",
        "model.layers.0.self_attn.q_proj",
        0,
    )
    k_proj = NamedModule(
        live_layer.self_attn.k_proj,
        "self_attn.k_proj",
        "model.layers.0.self_attn.k_proj",
        0,
    )

    processor = object.__new__(ParoQuantProcessor)
    processor.qcfg = SimpleNamespace(
        opt_scope="compute_block",
        runtime_bits=4,
        group_size=8,
        sym=True,
        krot=1,
        opt_seed=0,
        opt_pair_ratio=0.5,
        opt_pair_impl="fast",
        opt_quantizer_impl="reference",
        opt_fused_rotation=False,
        opt_channel_scale_clamp_min=1e-2,
        opt_channel_scale_clamp_max=1e2,
        dynamic_get=_dynamic_get,
    )

    state = SimpleNamespace(
        layer_module=live_layer,
        pristine_layer_module=pristine_layer,
    )
    layer_clone, _optim_modules = processor._build_group_optim_layer(state, [q_proj, k_proj])

    assert isinstance(layer_clone.self_attn.q_proj, _ParoQuantOptimLinear)
    assert isinstance(layer_clone.self_attn.k_proj, _ParoQuantOptimLinear)
    assert isinstance(layer_clone.self_attn.o_proj, torch.nn.Linear)
    assert not isinstance(layer_clone.self_attn.o_proj, _HookedLike)


def test_paroquant_processor_reuses_cached_group_source_clone():
    """Guard grouped clone preparation against rebuilding from a later-mutated pristine snapshot."""

    class _HookedLike(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(8, 8))

        def forward(self, x):
            return x

    class _ToyAttn(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(_attn_implementation="sdpa")
            self.q_proj = torch.nn.Linear(8, 8, bias=False)
            self.k_proj = torch.nn.Linear(8, 8, bias=False)
            self.v_proj = torch.nn.Linear(8, 8, bias=False)
            self.o_proj = torch.nn.Linear(8, 8, bias=False)

    class _ToyLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = _ToyAttn()

    def _dynamic_get(_module_name, _key, default=None):
        return default

    pristine_layer = _ToyLayer().half()
    live_layer = copy.deepcopy(pristine_layer)
    q_proj = NamedModule(
        live_layer.self_attn.q_proj,
        "self_attn.q_proj",
        "model.layers.0.self_attn.q_proj",
        0,
    )

    processor = object.__new__(ParoQuantProcessor)
    processor.qcfg = SimpleNamespace(
        opt_scope="compute_block",
        runtime_bits=4,
        group_size=8,
        sym=True,
        krot=1,
        opt_seed=0,
        opt_pair_ratio=0.5,
        opt_pair_impl="fast",
        opt_quantizer_impl="reference",
        opt_fused_rotation=False,
        opt_channel_scale_clamp_min=1e-2,
        opt_channel_scale_clamp_max=1e2,
        dynamic_get=_dynamic_get,
    )

    state = SimpleNamespace(
        layer_module=live_layer,
        pristine_layer_module=pristine_layer,
        prepared_group_source_module=None,
    )
    processor._build_group_optim_layer(state, [q_proj])
    assert state.prepared_group_source_module is not None

    state.pristine_layer_module.self_attn.o_proj = _HookedLike()
    layer_clone, _optim_modules = processor._build_group_optim_layer(state, [q_proj])

    assert isinstance(layer_clone.self_attn.o_proj, torch.nn.Linear)
    assert not isinstance(layer_clone.self_attn.o_proj, _HookedLike)


def test_paroquant_processor_reuses_cached_group_source_clone_per_device():
    """Guard grouped clone preparation against repeating the same device-local source setup."""

    class _ToyAttn(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(_attn_implementation="sdpa")
            self.q_proj = torch.nn.Linear(8, 8, bias=False)
            self.k_proj = torch.nn.Linear(8, 8, bias=False)
            self.v_proj = torch.nn.Linear(8, 8, bias=False)
            self.o_proj = torch.nn.Linear(8, 8, bias=False)

    class _ToyLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = _ToyAttn()

    def _dynamic_get(_module_name, _key, default=None):
        return default

    layer = _ToyLayer().half()
    q_proj = NamedModule(
        layer.self_attn.q_proj,
        "self_attn.q_proj",
        "model.layers.0.self_attn.q_proj",
        0,
    )

    processor = object.__new__(ParoQuantProcessor)
    processor.qcfg = SimpleNamespace(
        opt_scope="compute_block",
        runtime_bits=4,
        group_size=8,
        sym=True,
        krot=1,
        opt_seed=0,
        opt_pair_ratio=0.5,
        opt_pair_impl="fast",
        opt_quantizer_impl="reference",
        opt_fused_rotation=False,
        opt_channel_scale_clamp_min=1e-2,
        opt_channel_scale_clamp_max=1e2,
        dynamic_get=_dynamic_get,
    )

    state = SimpleNamespace(
        layer_module=layer,
        prepared_group_source_module=None,
        prepared_group_source_module_by_device=None,
    )
    processor._build_group_optim_layer(state, [q_proj])
    cached_source = state.prepared_group_source_module_by_device["cpu"]

    processor._build_group_optim_layer(state, [q_proj])

    assert state.prepared_group_source_module is not None
    assert state.prepared_group_source_module_by_device is not None
    assert state.prepared_group_source_module_by_device["cpu"] is cached_source


def test_paroquant_processor_device_group_source_cache_is_compute_block_only():
    """Guard grouped source-device caching so whole-layer mode keeps the simpler one-shot path."""

    class _ToyAttn(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(_attn_implementation="sdpa")
            self.q_proj = torch.nn.Linear(8, 8, bias=False)
            self.k_proj = torch.nn.Linear(8, 8, bias=False)
            self.v_proj = torch.nn.Linear(8, 8, bias=False)
            self.o_proj = torch.nn.Linear(8, 8, bias=False)

    class _ToyLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = _ToyAttn()

    def _dynamic_get(_module_name, _key, default=None):
        return default

    layer = _ToyLayer().half()
    q_proj = NamedModule(
        layer.self_attn.q_proj,
        "self_attn.q_proj",
        "model.layers.0.self_attn.q_proj",
        0,
    )

    processor = object.__new__(ParoQuantProcessor)
    processor.qcfg = SimpleNamespace(
        opt_scope="layer",
        runtime_bits=4,
        group_size=8,
        sym=True,
        krot=1,
        opt_seed=0,
        opt_pair_ratio=0.5,
        opt_pair_impl="fast",
        opt_quantizer_impl="reference",
        opt_fused_rotation=False,
        opt_channel_scale_clamp_min=1e-2,
        opt_channel_scale_clamp_max=1e2,
        dynamic_get=_dynamic_get,
    )

    state = SimpleNamespace(
        layer_module=layer,
        prepared_group_source_module=None,
        prepared_group_source_module_by_device=None,
    )
    processor._build_group_optim_layer(state, [q_proj])

    assert state.prepared_group_source_module is not None
    assert state.prepared_group_source_module_by_device is None


def test_paroquant_processor_caches_group_dataset_split():
    """Guard grouped dataset slicing against recomputing the same train/val split every group."""
    processor = object.__new__(ParoQuantProcessor)
    processor.qcfg = SimpleNamespace(opt_train_samples=4, opt_validation_samples=2)
    processor.inputs_cache = SimpleNamespace(position_ids=[], attention_masks=[])

    inputs = [[torch.randn(1, 4)], [torch.randn(1, 4)]]
    outputs = [[torch.randn(1, 4)], [torch.randn(1, 4)]]
    state = SimpleNamespace(
        layer_inputs=inputs,
        layer_input_kwargs=[{}, {}],
        layer_outputs=outputs,
        grouped_dataset=None,
    )

    first = processor._group_dataset_from_state(state)
    assert state.grouped_dataset is first

    state.layer_inputs = []
    state.layer_input_kwargs = []
    state.layer_outputs = []
    second = processor._group_dataset_from_state(state)

    assert second is first


def test_paroquant_processor_merges_equivalent_group_optimizer_param_groups():
    """Guard grouped AdamW setup against spawning redundant one-parameter optimizer groups."""
    p1 = torch.nn.Parameter(torch.randn(4))
    p2 = torch.nn.Parameter(torch.randn(4))
    p3 = torch.nn.Parameter(torch.randn(4))
    processor = object.__new__(ParoQuantProcessor)

    groups = processor._normalize_group_optimizer_param_groups(
        [
            {"params": [p1], "lr": 0.05, "weight_decay": 0.01, "betas": (0.9, 0.95), "eps": 1e-10},
            {"params": [p2], "lr": 0.05, "weight_decay": 0.01, "betas": (0.9, 0.95), "eps": 1e-10},
            {"params": [p1], "lr": 0.05, "weight_decay": 0.01, "betas": (0.9, 0.95), "eps": 1e-10},
            {"params": [p3], "lr": 1e-5, "weight_decay": 0.01, "betas": (0.9, 0.95), "eps": 1e-10},
        ]
    )

    assert len(groups) == 2
    assert groups[0]["params"] == [p1, p2]
    assert groups[1]["params"] == [p3]


def test_paroquant_processor_group_adamw_uses_merged_groups(monkeypatch):
    """Guard grouped optimizer setup against re-expanding merged parameter buckets."""
    processor = object.__new__(ParoQuantProcessor)
    param_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    param = torch.nn.Parameter(torch.randn(4, device=param_device))
    calls = []

    class _FakeOptimizer:
        def __init__(self):
            self.param_groups = [{"params": [param], "lr": 0.05}]

    def fake_adamw(param_groups, **kwargs):
        calls.append((param_groups, kwargs.copy()))
        return _FakeOptimizer()

    monkeypatch.setattr(torch.optim, "AdamW", fake_adamw)
    optimizer = processor._build_group_adamw(
        [{"params": [param], "lr": 0.05, "weight_decay": 0.01, "betas": (0.9, 0.95), "eps": 1e-10}],
        device=torch.device("cuda"),
    )

    assert isinstance(optimizer, _FakeOptimizer)
    expected_kwargs = {"fused": True} if torch.cuda.is_available() else {}
    expected_param_group = {
        "params": [param],
        "lr": 0.05,
        "weight_decay": 0.01,
        "betas": (0.9, 0.95),
        "eps": 1e-10,
        "amsgrad": False,
    }
    assert calls == [([expected_param_group], expected_kwargs)]


def test_paroquant_processor_group_adamw_falls_back_when_fused_cuda_is_unsupported(monkeypatch):
    """Guard grouped CUDA optimizer setup against fused AdamW support gaps."""

    processor = object.__new__(ParoQuantProcessor)
    param_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    param = torch.nn.Parameter(torch.randn(4, device=param_device))
    calls = []

    class _FakeOptimizer:
        def __init__(self):
            self.param_groups = [{"params": [param], "lr": 0.05}]

    def fake_adamw(param_groups, **kwargs):
        calls.append(kwargs.copy())
        if kwargs.get("fused"):
            raise TypeError("fused unsupported")
        return _FakeOptimizer()

    monkeypatch.setattr(torch.optim, "AdamW", fake_adamw)
    optimizer = processor._build_group_adamw(
        [{"params": [param], "lr": 0.05, "weight_decay": 0.01, "betas": (0.9, 0.95), "eps": 1e-10}],
        device=torch.device("cuda"),
    )

    assert isinstance(optimizer, _FakeOptimizer)
    expected = [{"fused": True}, {}] if torch.cuda.is_available() else [{}]
    assert calls == expected


def test_paroquant_processor_group_optimizer_uses_selected_sgd(monkeypatch):
    """Guard grouped optimizer setup against ignoring the selected stage optimizer."""
    processor = object.__new__(ParoQuantProcessor)
    param_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    param = torch.nn.Parameter(torch.randn(4, device=param_device))
    calls = []

    class _FakeOptimizer:
        def __init__(self):
            self.param_groups = [{"params": [param], "lr": 0.05}]

    def fake_sgd(param_groups, **kwargs):
        calls.append((param_groups, kwargs.copy()))
        return _FakeOptimizer()

    monkeypatch.setattr(torch.optim, "SGD", fake_sgd)
    optimizer = processor._build_group_optimizer(
        [
            {
                "params": [param],
                "lr": 0.05,
                "weight_decay": 0.01,
                "momentum": 0.85,
                "dampening": 0.0,
                "nesterov": True,
            }
        ],
        device=torch.device("cuda"),
        optimizer_name="sgd",
    )

    assert isinstance(optimizer, _FakeOptimizer)
    expected_kwargs = {"fused": True} if torch.cuda.is_available() else {}
    expected_param_group = {
        "params": [param],
        "lr": 0.05,
        "weight_decay": 0.01,
        "momentum": 0.85,
        "dampening": 0.0,
        "nesterov": True,
    }
    assert calls == [([expected_param_group], expected_kwargs)]


def test_paroquant_processor_group_adamw_passes_amsgrad(monkeypatch):
    """Guard grouped AdamW setup against dropping optimizer-specific hyperparameters."""
    processor = object.__new__(ParoQuantProcessor)
    param_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    param = torch.nn.Parameter(torch.randn(4, device=param_device))
    calls = []

    class _FakeOptimizer:
        def __init__(self):
            self.param_groups = [{"params": [param], "lr": 0.05}]

    def fake_adamw(param_groups, **kwargs):
        calls.append(kwargs.copy())
        return _FakeOptimizer()

    monkeypatch.setattr(torch.optim, "AdamW", fake_adamw)
    optimizer = processor._build_group_optimizer(
        [{"params": [param], "lr": 0.05, "weight_decay": 0.01, "amsgrad": True}],
        device=torch.device("cuda"),
        optimizer_name="adamw",
    )

    assert isinstance(optimizer, _FakeOptimizer)
    expected = [{"fused": True}] if torch.cuda.is_available() else [{}]
    assert calls == expected


def test_paroquant_processor_group_stage_skips_redundant_initial_train_eval(monkeypatch):
    """Guard grouped stage timing against paying an extra full train-set eval before epoch 1."""

    class _ToyLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 4, bias=False)

        def forward(self, x, **_kwargs):
            return self.linear(x)

    processor = object.__new__(ParoQuantProcessor)
    processor.qcfg = SimpleNamespace(opt_stage_impl="fast")
    processor.gptq_model = None
    processor.model = None
    calls = []

    def fake_evaluate(*_args, **_kwargs):
        calls.append("eval")
        return 0.0

    monkeypatch.setattr(processor, "_evaluate_group_layer", fake_evaluate)
    layer = _ToyLayer()
    input_batch = [[torch.randn(2, 4)]]
    target_batch = [[torch.randn(2, 4)]]

    processor._run_group_stage(
        layer,
        optim_modules={},
        input_batches_train=input_batch,
        input_kwargs_train=[{}],
        target_batches_train=target_batch,
        position_ids_train=[None],
        attention_masks_train=[None],
        input_batches_val=input_batch,
        input_kwargs_val=[{}],
        target_batches_val=target_batch,
        position_ids_val=[None],
        attention_masks_val=[None],
        param_groups=[{"params": [layer.linear.weight], "lr": 0.05, "weight_decay": 0.01, "betas": (0.9, 0.95), "eps": 1e-10}],
        epochs=1,
    )

    assert calls == ["eval"]


def test_paroquant_processor_group_stage_defers_best_state_snapshot_until_first_val(monkeypatch):
    """Guard grouped stage setup against cloning the full layer before the first val result exists."""

    class _ToyLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 4, bias=False)
            self.state_dict_calls = 0

        def forward(self, x, **_kwargs):
            return self.linear(x)

        def state_dict(self, *args, **kwargs):
            self.state_dict_calls += 1
            return super().state_dict(*args, **kwargs)

    processor = object.__new__(ParoQuantProcessor)
    processor.qcfg = SimpleNamespace(opt_stage_impl="fast")
    processor.gptq_model = None
    processor.model = None
    layer = _ToyLayer()
    input_batch = [[torch.randn(2, 4)]]
    target_batch = [[torch.randn(2, 4)]]

    processor._run_group_stage(
        layer,
        optim_modules={},
        input_batches_train=input_batch,
        input_kwargs_train=[{}],
        target_batches_train=target_batch,
        position_ids_train=[None],
        attention_masks_train=[None],
        input_batches_val=input_batch,
        input_kwargs_val=[{}],
        target_batches_val=target_batch,
        position_ids_val=[None],
        attention_masks_val=[None],
        param_groups=[{"params": [layer.linear.weight], "lr": 0.05, "weight_decay": 0.01, "betas": (0.9, 0.95), "eps": 1e-10}],
        epochs=1,
    )

    assert layer.state_dict_calls == 1


def test_paroquant_processor_group_dataset_for_device_caches_per_device():
    """Guard grouped dataset replay so repeated requests on the same device reuse one cached copy."""

    processor = object.__new__(ParoQuantProcessor)
    processor.qcfg = SimpleNamespace(opt_train_samples=8, opt_validation_samples=8)
    processor.inputs_cache = InputCache(
        layer_inputs=[[torch.randn(1, 4)]],
        layer_input_kwargs=[{"position_ids": None}],
        position_ids=[torch.arange(4).unsqueeze(0)],
        attention_masks=[None],
    )
    state = SimpleNamespace(
        layer_inputs=[[torch.randn(1, 4)]],
        layer_input_kwargs=[{"position_ids": None}],
        layer_outputs=[[torch.randn(1, 4)]],
        grouped_dataset=None,
        grouped_dataset_by_device=None,
    )

    first = processor._group_dataset_for_device(state, torch.device("cpu"))
    second = processor._group_dataset_for_device(state, torch.device("cpu"))

    assert first is second
    assert state.grouped_dataset is not None
    assert state.grouped_dataset_by_device is not None
    assert state.grouped_dataset_by_device["cpu"] is first


def test_paroquant_processor_replay_batches_cache_cpu_splits():
    """Guard layer-scope replay batches so they stay on CPU and cache their train/val split."""

    processor = object.__new__(ParoQuantProcessor)
    processor.qcfg = SimpleNamespace(opt_train_samples=4, opt_validation_samples=2)
    processor.inputs_cache = InputCache(
        layer_inputs=[[torch.randn(2, 4)] for _ in range(3)],
        layer_input_kwargs=[{} for _ in range(3)],
        position_ids=[None, None, None],
        attention_masks=[None, None, None],
    )
    state = SimpleNamespace(
        layer_inputs=[[torch.randn(2, 4)] for _ in range(3)],
        layer_input_kwargs=[{} for _ in range(3)],
        layer_outputs=[[torch.randn(2, 4)] for _ in range(3)],
        replay_batches=None,
    )

    first_train, first_val = processor._replay_batches_from_state(state)
    second_train, second_val = processor._replay_batches_from_state(state)

    assert first_train is second_train
    assert first_val is second_val
    assert len(first_train) == 2
    assert len(first_val) == 1
    assert all(batch.inputs[0].device.type == "cpu" for batch in first_train + first_val)
    assert all(batch.target.device.type == "cpu" for batch in first_train + first_val)
    if torch.cuda.is_available():
        assert all(batch.inputs[0].is_pinned() for batch in first_train + first_val)
        assert all(batch.target.is_pinned() for batch in first_train + first_val)


def test_paroquant_processor_replay_batches_strip_inference_tensors():
    """Replay-cache tensors must be recreated outside inference mode before layer training."""

    processor = object.__new__(ParoQuantProcessor)
    processor.qcfg = SimpleNamespace(opt_train_samples=2, opt_validation_samples=1)

    with torch.inference_mode():
        cached_input = torch.randn(2, 4)
        cached_kwarg = torch.randn(2, 4)
        cached_output = torch.randn(2, 4)
        cached_pos = torch.arange(4).unsqueeze(0)
        cached_mask = torch.ones(1, 4)

    processor.inputs_cache = InputCache(
        layer_inputs=[[cached_input]],
        layer_input_kwargs=[{"cache_position": cached_kwarg}],
        position_ids=[cached_pos],
        attention_masks=[cached_mask],
    )
    state = SimpleNamespace(
        layer_inputs=[[cached_input]],
        layer_input_kwargs=[{"cache_position": cached_kwarg}],
        layer_outputs=[[cached_output]],
        replay_batches=None,
    )

    train_batches, val_batches = processor._replay_batches_from_state(state)
    replay_batch = train_batches[0]

    assert not replay_batch.inputs[0].is_inference()
    assert not replay_batch.input_kwargs["cache_position"].is_inference()
    assert not replay_batch.target.is_inference()
    assert replay_batch.position_ids is not None
    assert replay_batch.attention_mask is not None
    assert not replay_batch.position_ids.is_inference()
    assert not replay_batch.attention_mask.is_inference()
    assert len(train_batches) == 1
    assert len(val_batches) == 1


def test_paroquant_processor_layer_shard_loader_normalizes_inference_inputs():
    """Materialized replay batches must hand autograd normal tensors even if cache input is inference-mode."""

    class _ToyNorm(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(4))

        def forward(self, hidden_states):
            return self.weight * hidden_states.to(hidden_states.dtype)

    with torch.inference_mode():
        replay_batch = paroquant_processor_module._ParoQuantReplayBatch(
            inputs=[torch.randn(2, 4)],
            input_kwargs={},
            target=torch.randn(2, 4),
            position_ids=None,
            attention_mask=None,
            row_count=2,
        )

    loader = paroquant_processor_module._LayerShardLoader(
        [replay_batch],
        target_device=torch.device("cpu"),
        shard_batches=1,
    )
    materialized_batch = next(loader.iter_shards())[0]
    layer = _ToyNorm()

    assert not materialized_batch.inputs[0].is_inference()
    assert not materialized_batch.target.is_inference()

    with torch.inference_mode(False), torch.enable_grad():
        output = layer(materialized_batch.inputs[0])

    assert output.shape == materialized_batch.inputs[0].shape


def test_paroquant_processor_group_checkpoint_normalizes_inference_inputs():
    """Grouped checkpoint training must rebuild inference-mode inputs into autograd-safe tensors."""

    processor = object.__new__(ParoQuantProcessor)
    processor._gradient_checkpointing_enabled = lambda: True

    captured = {}
    layer_scale = torch.nn.Parameter(torch.tensor(2.0))

    def _fake_forward_group_batch(
        layer,
        *,
        batch_index,
        input_batch,
        input_kwargs,
        attention_mask,
        position_ids,
    ):
        captured["inputs_inference"] = [tensor.is_inference() for tensor in input_batch]
        captured["kwargs_inference"] = paroquant_processor_module._value_has_inference_tensor(input_kwargs)
        captured["mask_inference"] = attention_mask.is_inference() if attention_mask is not None else False
        captured["pos_inference"] = position_ids.is_inference() if position_ids is not None else False
        return input_batch[0] * layer_scale

    processor._forward_group_batch = _fake_forward_group_batch

    with torch.inference_mode():
        input_batch = [torch.randn(2, 4)]
        input_kwargs = {"cache_position": torch.arange(4)}
        attention_mask = torch.ones(1, 4)
        position_ids = torch.arange(4).unsqueeze(0)

    output = processor._forward_group_batch_train(
        object(),
        batch_index=0,
        input_batch=input_batch,
        input_kwargs=input_kwargs,
        attention_mask=attention_mask,
        position_ids=position_ids,
    )
    output.sum().backward()

    assert captured["inputs_inference"] == [False]
    assert captured["kwargs_inference"] is False
    assert captured["mask_inference"] is False
    assert captured["pos_inference"] is False
    assert layer_scale.grad is not None


def test_paroquant_processor_cached_group_position_ids_are_autograd_safe():
    """Generated position-id cache entries must stay reusable outside worker inference mode."""

    processor = object.__new__(ParoQuantProcessor)

    with torch.inference_mode():
        cached = processor._cached_group_position_ids(
            device=torch.device("cpu"),
            batch_dim=2,
            seq_len=4,
        )

    assert not cached.is_inference()
    assert processor._cached_group_position_ids(device=torch.device("cpu"), batch_dim=2, seq_len=4) is cached


def test_paroquant_processor_cached_rotary_embeddings_are_autograd_safe():
    """Rotary cache entries created during inference replay must not leak inference tensors into training."""

    class _ToyRotary(torch.nn.Module):
        def forward(self, x, position_ids):
            pos = position_ids.unsqueeze(-1).to(dtype=x.dtype)
            return x + pos, x - pos

    processor = object.__new__(ParoQuantProcessor)
    rotary = _ToyRotary()

    with torch.inference_mode():
        x = torch.randn(1, 4, 8)
        position_ids = torch.arange(4).unsqueeze(0)
        cached = processor._cached_group_rotary_position_embeddings(
            rotary=rotary,
            x=x,
            position_ids=position_ids,
            rotary_device=torch.device("cpu"),
        )

    assert not paroquant_processor_module._value_has_inference_tensor(cached)

    with torch.inference_mode(False), torch.enable_grad():
        q = torch.randn(1, 4, 8, requires_grad=True)
        cos, sin = processor._cached_group_rotary_position_embeddings(
            rotary=rotary,
            x=x,
            position_ids=position_ids,
            rotary_device=torch.device("cpu"),
        )
        loss = (q * cos).sum() + (q * sin).sum()
        loss.backward()

    assert q.grad is not None


def test_paroquant_processor_layer_shard_loader_reuses_metadata_tensors():
    """Guard streamed layer replay so shared position/mask tensors can stay cached on one device."""

    if not torch.cuda.is_available():
        return

    cpu_pos = torch.arange(8).unsqueeze(0)
    cpu_mask = torch.ones(1, 8)
    if not cpu_pos.is_pinned():
        cpu_pos = cpu_pos.pin_memory()
    if not cpu_mask.is_pinned():
        cpu_mask = cpu_mask.pin_memory()

    replay_batch = paroquant_processor_module._ParoQuantReplayBatch(
        inputs=[torch.randn(1, 8).pin_memory()],
        input_kwargs={},
        target=torch.randn(1, 8).pin_memory(),
        position_ids=cpu_pos,
        attention_mask=cpu_mask,
        row_count=1,
    )
    metadata_cache = {}

    loader_a = paroquant_processor_module._LayerShardLoader(
        [replay_batch],
        target_device=torch.device("cuda"),
        shard_batches=1,
        metadata_cache=metadata_cache,
    )
    loader_b = paroquant_processor_module._LayerShardLoader(
        [replay_batch],
        target_device=torch.device("cuda"),
        shard_batches=1,
        metadata_cache=metadata_cache,
    )

    batch_a = next(loader_a.iter_shards())[0]
    batch_b = next(loader_b.iter_shards())[0]

    assert batch_a.position_ids is batch_b.position_ids
    assert batch_a.attention_mask is batch_b.attention_mask
    assert batch_a.target is not batch_b.target


def test_paroquant_processor_group_best_state_tracks_only_active_prefixes():
    """Guard grouped best-state snapshots against cloning untouched layer state."""

    class _ToyLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = torch.nn.Linear(4, 4, bias=False)
            self.b = torch.nn.Linear(4, 4, bias=False)

    processor = object.__new__(ParoQuantProcessor)
    layer = _ToyLayer()
    original_b = layer.b.weight.detach().clone()

    best_state = processor._snapshot_group_best_state(layer, active_prefixes=("a",))

    assert sorted(best_state.keys()) == ["a.weight"]

    with torch.no_grad():
        layer.a.weight.zero_()
        layer.b.weight.fill_(7.0)

    processor._restore_group_best_state(layer, best_state=best_state)

    assert torch.allclose(layer.a.weight, best_state["a.weight"])
    assert torch.allclose(layer.b.weight, torch.full_like(layer.b.weight, 7.0))
    assert not torch.allclose(layer.b.weight, original_b)


def test_paroquant_processor_group_best_state_can_snapshot_to_cpu():
    """Guard streamed layer checkpoints so best-state snapshots can live off device."""

    class _ToyLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = torch.nn.Linear(4, 4, bias=False)
            self.b = torch.nn.Linear(4, 4, bias=False)

    processor = object.__new__(ParoQuantProcessor)
    layer = _ToyLayer()

    best_state = processor._snapshot_group_best_state(
        layer,
        active_prefixes=("a",),
        target_device=torch.device("cpu"),
    )

    assert sorted(best_state.keys()) == ["a.weight"]
    assert all(tensor.device.type == "cpu" for tensor in best_state.values())


def test_paroquant_processor_group_best_state_can_cast_float_snapshots_without_touching_int_buffers():
    """Guard grouped best-state compression so float tensors shrink without corrupting integer buffers."""

    class _ToyBranch(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(4, 4, dtype=torch.float32))
            self.register_buffer("index", torch.tensor([1, 2], dtype=torch.int32))

    class _ToyLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = _ToyBranch()
            self.b = _ToyBranch()

    processor = object.__new__(ParoQuantProcessor)
    layer = _ToyLayer()

    best_state = processor._snapshot_group_best_state(
        layer,
        active_prefixes=("a",),
        target_device=torch.device("cpu"),
        target_dtype=torch.bfloat16,
    )

    assert sorted(best_state.keys()) == ["a.index", "a.weight"]
    assert best_state["a.weight"].dtype == torch.bfloat16
    assert best_state["a.index"].dtype == torch.int32


def test_paroquant_quantize_config_accepts_torch_float16_best_state_dtype():
    """Guard direct config construction so torch.float16 snapshots serialize as fp16."""
    cfg = ParoConfig(
        bits=4,
        group_size=128,
        opt_best_state_dtype=torch.float16,
    )

    assert cfg.opt_best_state_dtype == "fp16"


def test_paroquant_best_state_dtype_resolves_explicit_fp16():
    """Guard explicit fp16 snapshot selection."""
    resolved = paroquant_optimization._resolve_best_state_snapshot_dtype(
        best_state_dtype="fp16",
        device=torch.device("cuda"),
    )

    assert resolved == torch.float16


def test_paroquant_best_state_dtype_resolves_explicit_bf16():
    """Guard explicit bf16 snapshot selection after removing the auto policy."""
    resolved = paroquant_optimization._resolve_best_state_snapshot_dtype(
        best_state_dtype="bf16",
        device=torch.device("cuda"),
    )

    assert resolved == torch.bfloat16


def test_paroquant_best_state_dtype_defaults_to_fp32():
    """Guard the no-auto default so missing best-state dtype configuration stays on fp32."""
    resolved = paroquant_optimization._resolve_best_state_snapshot_dtype(
        best_state_dtype=None,
        device=torch.device("cpu"),
    )

    assert resolved == torch.float32


def test_paroquant_processor_caches_group_forward_signature_flags(monkeypatch):
    """Guard grouped replay kwargs against repeated forward-signature introspection."""

    class _ToyLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 4, bias=False)

        def forward(self, x, attention_mask=None, position_ids=None):
            del attention_mask, position_ids
            return self.linear(x)

    processor = object.__new__(ParoQuantProcessor)
    processor.gptq_model = None
    processor.model = None
    layer = _ToyLayer()
    signature_calls = []
    original_signature = inspect.signature

    def counting_signature(obj):
        signature_calls.append(obj)
        return original_signature(obj)

    monkeypatch.setattr(inspect, "signature", counting_signature)

    kwargs_a = processor._prepare_group_forward_kwargs(
        layer,
        x=torch.randn(2, 4),
        input_kwargs={},
        attention_mask=torch.ones(1, 2),
        position_ids=torch.arange(2).unsqueeze(0),
    )
    kwargs_b = processor._prepare_group_forward_kwargs(
        layer,
        x=torch.randn(2, 4),
        input_kwargs={},
        attention_mask=torch.ones(1, 2),
        position_ids=torch.arange(2).unsqueeze(0),
    )

    assert len(signature_calls) == 1
    assert kwargs_a.keys() == kwargs_b.keys()
    assert processor._group_forward_signature_cache[type(layer)] == (True, False, True)


def test_paroquant_processor_caches_group_forward_base_kwargs(monkeypatch):
    """Guard grouped replay kwargs against repeated nested device moves for the same cached dict."""

    class _ToyLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 4, bias=False)

        def forward(self, x, attention_mask=None, position_ids=None):
            del attention_mask, position_ids
            return self.linear(x)

    processor = object.__new__(ParoQuantProcessor)
    processor.gptq_model = None
    processor.model = None
    layer = _ToyLayer()
    move_calls = []
    original_nested_move_to = paroquant_processor_module.nested_move_to

    def counting_nested_move_to(value, *, device):
        move_calls.append((type(value).__name__, str(device)))
        return original_nested_move_to(value, device=device)

    monkeypatch.setattr(paroquant_processor_module, "nested_move_to", counting_nested_move_to)
    shared_kwargs = {"foo": {"bar": torch.randn(1)}}

    kwargs_a = processor._prepare_group_forward_kwargs(
        layer,
        x=torch.randn(2, 4),
        input_kwargs=shared_kwargs,
        attention_mask=torch.ones(1, 2),
        position_ids=torch.arange(2).unsqueeze(0),
    )
    kwargs_b = processor._prepare_group_forward_kwargs(
        layer,
        x=torch.randn(2, 4),
        input_kwargs=shared_kwargs,
        attention_mask=torch.ones(1, 2),
        position_ids=torch.arange(2).unsqueeze(0),
    )

    assert len(move_calls) == 1
    assert kwargs_a.keys() == kwargs_b.keys()
    assert torch.allclose(kwargs_a["foo"]["bar"], kwargs_b["foo"]["bar"])


def test_paroquant_processor_caches_full_group_forward_kwargs(monkeypatch):
    """Guard grouped replay kwargs against recomputing rotary-derived inputs for identical batches."""

    class _ToyLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 4, bias=False)

        def forward(self, x, attention_mask=None, position_ids=None, position_embeddings=None):
            del attention_mask, position_ids, position_embeddings
            return self.linear(x)

    class _FakeRotary(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, x, position_ids):
            self.calls += 1
            return x + position_ids.unsqueeze(-1).to(dtype=x.dtype)

    processor = object.__new__(ParoQuantProcessor)
    processor.gptq_model = None
    processor.model = None
    layer = _ToyLayer()
    rotary = _FakeRotary()

    monkeypatch.setattr(processor, "_get_root_rotary", lambda: rotary)
    monkeypatch.setattr(processor, "_get_rotary_for_device", lambda device: rotary)
    monkeypatch.setattr(processor, "_get_rotary_device", lambda module, fallback=None: torch.device("cpu"))

    x = torch.randn(1, 2, 4)
    attention_mask = torch.ones(1, 2)
    position_ids = torch.arange(2).unsqueeze(0)
    shared_kwargs = {}

    kwargs_a = processor._prepare_group_forward_kwargs(
        layer,
        x=x,
        input_kwargs=shared_kwargs,
        attention_mask=attention_mask,
        position_ids=position_ids,
    )
    kwargs_b = processor._prepare_group_forward_kwargs(
        layer,
        x=x,
        input_kwargs=shared_kwargs,
        attention_mask=attention_mask,
        position_ids=position_ids,
    )

    assert rotary.calls == 1
    assert kwargs_a is not kwargs_b
    assert kwargs_a.keys() == kwargs_b.keys()
    assert torch.allclose(kwargs_a["position_embeddings"], kwargs_b["position_embeddings"])


def test_paroquant_processor_caches_rotary_position_embeddings_across_distinct_inputs(monkeypatch):
    """Guard streamed grouped replay against recomputing HF rotary embeddings for identical ids/device/dtype."""

    class _ToyLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 4, bias=False)

        def forward(self, x, attention_mask=None, position_ids=None, position_embeddings=None):
            del attention_mask, position_ids, position_embeddings
            return self.linear(x)

    class _FakeLlamaRotaryEmbedding(torch.nn.Module):
        __module__ = "transformers.models.llama.modeling_llama"

        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, x, position_ids):
            self.calls += 1
            base = position_ids.unsqueeze(-1).to(device=x.device, dtype=x.dtype)
            return (base.cos(), base.sin())

    processor = object.__new__(ParoQuantProcessor)
    processor.gptq_model = None
    processor.model = None
    layer = _ToyLayer()
    rotary = _FakeLlamaRotaryEmbedding()

    monkeypatch.setattr(processor, "_get_root_rotary", lambda: rotary)
    monkeypatch.setattr(processor, "_get_rotary_for_device", lambda device: rotary)
    monkeypatch.setattr(processor, "_get_rotary_device", lambda module, fallback=None: torch.device("cpu"))

    shared_position_ids = torch.arange(4).unsqueeze(0)

    kwargs_a = processor._prepare_group_forward_kwargs(
        layer,
        x=torch.randn(1, 4, 4),
        input_kwargs={},
        attention_mask=torch.ones(1, 4),
        position_ids=shared_position_ids,
        cache=False,
    )
    kwargs_b = processor._prepare_group_forward_kwargs(
        layer,
        x=torch.randn(1, 4, 4),
        input_kwargs={},
        attention_mask=torch.ones(1, 4),
        position_ids=shared_position_ids,
        cache=False,
    )

    assert rotary.calls == 1
    assert torch.allclose(kwargs_a["position_embeddings"][0], kwargs_b["position_embeddings"][0])
    assert torch.allclose(kwargs_a["position_embeddings"][1], kwargs_b["position_embeddings"][1])


def test_paroquant_processor_caches_generated_position_ids():
    """Guard grouped replay against rebuilding deterministic synthetic position ids."""

    processor = object.__new__(ParoQuantProcessor)

    first = processor._cached_group_position_ids(device=torch.device("cpu"), batch_dim=2, seq_len=8)
    second = processor._cached_group_position_ids(device=torch.device("cpu"), batch_dim=2, seq_len=8)
    third = processor._cached_group_position_ids(device=torch.device("cpu"), batch_dim=1, seq_len=8)

    assert first is second
    assert third is not first
    assert first.shape == (2, 8)
    assert third.shape == (1, 8)


def test_paroquant_processor_streamed_group_forward_kwargs_skip_redundant_moves(monkeypatch):
    """Guard streamed layer replay against recursively re-moving already device-ready kwargs."""

    class _ToyLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 4, bias=False)

        def forward(self, x, attention_mask=None, position_ids=None):
            del attention_mask, position_ids
            return self.linear(x)

    processor = object.__new__(ParoQuantProcessor)
    processor.gptq_model = None
    processor.model = None
    layer = _ToyLayer()
    moved_values = []

    original_move_value = paroquant_processor_module._LayerShardLoader._move_value_to_device

    def counting_move_value(value, device):
        moved_values.append((type(value).__name__, str(device)))
        return original_move_value(value, device)

    monkeypatch.setattr(paroquant_processor_module._LayerShardLoader, "_move_value_to_device", counting_move_value)

    attention_mask = torch.ones(1, 2)
    position_ids = torch.arange(2).unsqueeze(0)
    shared_kwargs = {"foo": {"bar": torch.randn(1)}}
    kwargs = processor._prepare_group_forward_kwargs(
        layer,
        x=torch.randn(2, 4),
        input_kwargs=shared_kwargs,
        attention_mask=attention_mask,
        position_ids=position_ids,
        cache=False,
    )

    assert kwargs["foo"]["bar"] is shared_kwargs["foo"]["bar"]
    assert kwargs["attention_mask"] is attention_mask
    assert kwargs["position_ids"] is position_ids


def test_paroquant_processor_group_forward_kwargs_drop_past_key_values():
    """Layer-scope replay must mirror the normal forward executor and omit KV-cache objects."""

    class _ToyLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 4, bias=False)

        def forward(
            self,
            x,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            past_key_value=None,
            use_cache=False,
        ):
            del attention_mask, position_ids, past_key_values, past_key_value, use_cache
            return self.linear(x)

    processor = object.__new__(ParoQuantProcessor)
    processor.gptq_model = None
    processor.model = None
    layer = _ToyLayer()

    kwargs = processor._prepare_group_forward_kwargs(
        layer,
        x=torch.randn(2, 4),
        input_kwargs={
            "past_key_values": object(),
            "past_key_value": object(),
            "cache_position": torch.arange(2),
        },
        attention_mask=torch.ones(1, 2),
        position_ids=torch.arange(2).unsqueeze(0),
        cache=False,
    )

    assert "past_key_values" not in kwargs
    assert "past_key_value" not in kwargs
    assert "cache_position" in kwargs


def test_paroquant_processor_force_layer_eager_attention_restores_shared_config():
    """Live-layer optimization should temporarily switch shared attention config to eager and restore it."""

    class _Config:
        def __init__(self):
            self._attn_implementation = "flash_attention_2"
            self.attn_implementation = "flash_attention_2"

    class _ToyAttention(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config

    class _ToyLayer(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.self_attn = _ToyAttention(config)

    processor = object.__new__(ParoQuantProcessor)
    shared_config = _Config()
    layer = _ToyLayer(shared_config)

    overrides = processor._force_layer_eager_attention(layer)

    assert shared_config._attn_implementation == "eager"
    assert shared_config.attn_implementation == "eager"
    assert len(overrides) == 2

    processor._restore_layer_attention_impl(overrides)

    assert shared_config._attn_implementation == "flash_attention_2"
    assert shared_config.attn_implementation == "flash_attention_2"


def test_paroquant_processor_prepare_group_forward_kwargs_normalizes_inference_position_embeddings(monkeypatch):
    """Grouped replay kwargs must clone rotary metadata back to normal tensors before live-layer forward."""

    class _ToyLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 4, bias=False)

        def forward(self, x, attention_mask=None, position_ids=None, position_embeddings=None):
            del attention_mask, position_ids, position_embeddings
            return self.linear(x)

    class _FakeLlamaRotaryEmbedding(torch.nn.Module):
        __module__ = "transformers.models.llama.modeling_llama"

        def forward(self, x, position_ids):
            del x, position_ids
            with torch.inference_mode():
                return (torch.randn(1, 2, 4), torch.randn(1, 2, 4))

    processor = object.__new__(ParoQuantProcessor)
    processor.gptq_model = None
    processor.model = None
    layer = _ToyLayer()
    rotary = _FakeLlamaRotaryEmbedding()

    monkeypatch.setattr(processor, "_get_root_rotary", lambda: rotary)
    monkeypatch.setattr(processor, "_get_rotary_for_device", lambda device: rotary)
    monkeypatch.setattr(processor, "_get_rotary_device", lambda module, fallback=None: torch.device("cpu"))

    kwargs = processor._prepare_group_forward_kwargs(
        layer,
        x=torch.randn(1, 2, 4),
        input_kwargs={},
        attention_mask=torch.ones(1, 2),
        position_ids=torch.arange(2).unsqueeze(0),
        cache=False,
    )

    assert "position_embeddings" in kwargs
    assert not paroquant_processor_module._value_has_inference_tensor(kwargs["position_embeddings"])


def test_paroquant_processor_caches_group_targets_by_dtype_and_device():
    """Guard grouped replay targets against repeated device/dtype conversions."""

    processor = object.__new__(ParoQuantProcessor)
    target_batch = [torch.randn(2, 4)]

    first = processor._prepare_group_target(target_batch, device=torch.device("cpu"), dtype=torch.float32)
    second = processor._prepare_group_target(target_batch, device=torch.device("cpu"), dtype=torch.float32)
    third = processor._prepare_group_target(target_batch, device=torch.device("cpu"), dtype=torch.float16)

    assert first is second
    assert third.dtype == torch.float16
    assert third is not first


def test_paroquant_processor_optimize_group_runs_on_toy_layer():
    """Guard the grouped optimizer path on a tiny layer without needing the full looper."""

    class _ToyAttn(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = torch.nn.Linear(8, 8, bias=False)
            self.k_proj = torch.nn.Linear(8, 8, bias=False)
            self.v_proj = torch.nn.Linear(8, 8, bias=False)
            self.o_proj = torch.nn.Linear(8, 8, bias=False)

        def forward(self, x):
            return self.o_proj(self.q_proj(x) + self.k_proj(x) + self.v_proj(x))

    class _ToyMlp(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = torch.nn.Linear(8, 8, bias=False)
            self.up_proj = torch.nn.Linear(8, 8, bias=False)
            self.down_proj = torch.nn.Linear(8, 8, bias=False)

        def forward(self, x):
            return self.down_proj(torch.sigmoid(self.gate_proj(x)) * self.up_proj(x))

    class _ToyLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = _ToyAttn()
            self.mlp = _ToyMlp()

        def forward(self, x, attention_mask=None, position_ids=None, use_cache=False):
            del attention_mask, position_ids, use_cache
            x = self.self_attn(x)
            return self.mlp(x)

    def _dynamic_get(_module_name, _key, default=None):
        return default

    layer = _ToyLayer()
    x = torch.randn(1, 2, 8)
    y = layer(x).detach()
    q_proj = NamedModule(layer.self_attn.q_proj, "self_attn.q_proj", "model.layers.0.self_attn.q_proj", 0)
    k_proj = NamedModule(layer.self_attn.k_proj, "self_attn.k_proj", "model.layers.0.self_attn.k_proj", 0)
    v_proj = NamedModule(layer.self_attn.v_proj, "self_attn.v_proj", "model.layers.0.self_attn.v_proj", 0)

    processor = object.__new__(ParoQuantProcessor)
    processor.qcfg = SimpleNamespace(
        runtime_bits=4,
        group_size=8,
        sym=True,
        krot=1,
        opt_seed=0,
        opt_pair_ratio=0.5,
        opt_pair_impl="fast",
        opt_quantizer_impl="reference",
        opt_fused_rotation=False,
        opt_channel_scale_clamp_min=1e-2,
        opt_channel_scale_clamp_max=1e2,
        opt_train_samples=2,
        opt_validation_samples=2,
        opt_stage_impl="fast",
        opt_rotation_lr=0.05,
        opt_weight_lr=1e-5,
        opt_quantizer_lr=1e-6,
        opt_rotation_epochs=0,
        opt_finetune_epochs=0,
        dynamic_get=_dynamic_get,
    )
    processor.gptq_model = SimpleNamespace(support_batch_quantize=True)
    processor._batch_tls = threading.local()
    processor.inputs_cache = InputCache(
        layer_inputs=[[x]],
        layer_input_kwargs=[{}],
        position_ids=[None],
        attention_masks=[None],
    )

    state = SimpleNamespace(
        layer_module=layer,
        layer_inputs=[[x]],
        layer_input_kwargs=[{}],
        layer_outputs=[[y]],
    )

    results, val_loss = processor._optimize_group(state, [q_proj, k_proj, v_proj])

    assert set(results) == {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"}
    assert val_loss >= 0.0
    assert results["self_attn.q_proj"].pseudo_weight.shape == q_proj.weight.shape
    assert results["self_attn.k_proj"].pseudo_weight.shape == k_proj.weight.shape
    assert results["self_attn.v_proj"].pseudo_weight.shape == v_proj.weight.shape


def test_paroquant_processor_layer_scope_streams_without_device_dataset(monkeypatch):
    """Guard layer scope against materializing the full grouped dataset on device."""

    class _ToyAttn(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = torch.nn.Linear(8, 8, bias=False)
            self.k_proj = torch.nn.Linear(8, 8, bias=False)
            self.v_proj = torch.nn.Linear(8, 8, bias=False)
            self.o_proj = torch.nn.Linear(8, 8, bias=False)

        def forward(self, x):
            return self.o_proj(self.q_proj(x) + self.k_proj(x) + self.v_proj(x))

    class _ToyMlp(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = torch.nn.Linear(8, 8, bias=False)
            self.up_proj = torch.nn.Linear(8, 8, bias=False)
            self.down_proj = torch.nn.Linear(8, 8, bias=False)

        def forward(self, x):
            return self.down_proj(torch.sigmoid(self.gate_proj(x)) * self.up_proj(x))

    class _ToyLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = _ToyAttn()
            self.mlp = _ToyMlp()

        def forward(self, x, attention_mask=None, position_ids=None, use_cache=False):
            del attention_mask, position_ids, use_cache
            return self.mlp(self.self_attn(x))

    def _dynamic_get(_module_name, _key, default=None):
        return default

    layer = _ToyLayer()
    x = torch.randn(1, 2, 8)
    y = layer(x).detach()
    q_proj = NamedModule(layer.self_attn.q_proj, "self_attn.q_proj", "model.layers.0.self_attn.q_proj", 0)
    k_proj = NamedModule(layer.self_attn.k_proj, "self_attn.k_proj", "model.layers.0.self_attn.k_proj", 0)
    v_proj = NamedModule(layer.self_attn.v_proj, "self_attn.v_proj", "model.layers.0.self_attn.v_proj", 0)

    processor = object.__new__(ParoQuantProcessor)
    processor.qcfg = SimpleNamespace(
        opt_scope="layer",
        runtime_bits=4,
        group_size=8,
        sym=True,
        krot=1,
        opt_seed=0,
        opt_pair_ratio=0.5,
        opt_pair_impl="fast",
        opt_quantizer_impl="reference",
        opt_fused_rotation=False,
        opt_channel_scale_clamp_min=1e-2,
        opt_channel_scale_clamp_max=1e2,
        opt_train_samples=2,
        opt_validation_samples=2,
        opt_stage_impl="fast",
        opt_rotation_lr=0.05,
        opt_weight_lr=1e-5,
        opt_quantizer_lr=1e-6,
        opt_rotation_epochs=0,
        opt_finetune_epochs=0,
        dynamic_get=_dynamic_get,
    )
    processor.gptq_model = SimpleNamespace(support_batch_quantize=True)
    processor._batch_tls = threading.local()
    processor.inputs_cache = InputCache(
        layer_inputs=[[x]],
        layer_input_kwargs=[{}],
        position_ids=[None],
        attention_masks=[None],
    )

    def fail_group_dataset_for_device(*args, **kwargs):
        raise AssertionError("layer scope should stream replay batches instead of caching a full device dataset")

    monkeypatch.setattr(processor, "_group_dataset_for_device", fail_group_dataset_for_device)

    state = SimpleNamespace(
        layer_module=layer,
        layer_inputs=[[x]],
        layer_input_kwargs=[{}],
        layer_outputs=[[y]],
        replay_batches=None,
        grouped_dataset_by_device=None,
    )

    results, val_loss = processor._optimize_group(state, [q_proj, k_proj, v_proj])

    assert set(results) == {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"}
    assert val_loss >= 0.0
    assert state.grouped_dataset_by_device is None
    assert state.replay_batches is not None
    replay_train, replay_val = state.replay_batches
    assert all(batch.inputs[0].device.type == "cpu" for batch in replay_train + replay_val)


def test_paroquant_processor_layer_scope_skips_angle_reset_when_theta_frozen(monkeypatch):
    """Guard finetune-only layer scope against redundant masked-angle resets."""

    class _ToyAttn(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = torch.nn.Linear(8, 8, bias=False)
            self.k_proj = torch.nn.Linear(8, 8, bias=False)
            self.v_proj = torch.nn.Linear(8, 8, bias=False)
            self.o_proj = torch.nn.Linear(8, 8, bias=False)

        def forward(self, x):
            return self.o_proj(self.q_proj(x) + self.k_proj(x) + self.v_proj(x))

    class _ToyMlp(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = torch.nn.Linear(8, 8, bias=False)
            self.up_proj = torch.nn.Linear(8, 8, bias=False)
            self.down_proj = torch.nn.Linear(8, 8, bias=False)

        def forward(self, x):
            return self.down_proj(torch.sigmoid(self.gate_proj(x)) * self.up_proj(x))

    class _ToyLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = _ToyAttn()
            self.mlp = _ToyMlp()

        def forward(self, x, attention_mask=None, position_ids=None, use_cache=False):
            del attention_mask, position_ids, use_cache
            return self.mlp(self.self_attn(x))

    def _dynamic_get(_module_name, _key, default=None):
        return default

    layer = _ToyLayer()
    x = torch.randn(1, 2, 8)
    y = layer(x).detach()
    q_proj = NamedModule(layer.self_attn.q_proj, "self_attn.q_proj", "model.layers.0.self_attn.q_proj", 0)
    k_proj = NamedModule(layer.self_attn.k_proj, "self_attn.k_proj", "model.layers.0.self_attn.k_proj", 0)
    v_proj = NamedModule(layer.self_attn.v_proj, "self_attn.v_proj", "model.layers.0.self_attn.v_proj", 0)

    processor = object.__new__(ParoQuantProcessor)
    processor.qcfg = SimpleNamespace(
        opt_scope="layer",
        runtime_bits=4,
        group_size=8,
        sym=True,
        krot=1,
        opt_seed=0,
        opt_pair_ratio=0.5,
        opt_pair_impl="fast",
        opt_quantizer_impl="reference",
        opt_fused_rotation=False,
        opt_channel_scale_clamp_min=1e-2,
        opt_channel_scale_clamp_max=1e2,
        opt_train_samples=2,
        opt_validation_samples=2,
        opt_stage_impl="fast",
        opt_rotation_lr=0.05,
        opt_weight_lr=1e-5,
        opt_quantizer_lr=1e-6,
        opt_rotation_epochs=0,
        opt_finetune_epochs=1,
        dynamic_get=_dynamic_get,
    )
    processor.gptq_model = SimpleNamespace(support_batch_quantize=True)
    processor.model = None
    processor.lock = threading.Lock()
    processor.calculate_w_wq_diff = False
    processor._batch_tls = threading.local()
    processor.inputs_cache = InputCache(
        layer_inputs=[[x]],
        layer_input_kwargs=[{}],
        position_ids=[None],
        attention_masks=[None],
    )

    state = SimpleNamespace(
        layer_module=layer,
        layer_inputs=[[x]],
        layer_input_kwargs=[{}],
        layer_outputs=[[y]],
        replay_batches=None,
    )

    reset_calls = []

    def counting_reset(optim_modules):
        reset_calls.append(tuple(sorted(optim_modules)))

    monkeypatch.setattr(processor, "_reset_group_angles", counting_reset)

    results, val_loss = processor._optimize_group(state, [q_proj, k_proj, v_proj])

    assert set(results) == {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"}
    assert val_loss >= 0.0
    assert reset_calls == []


def test_paroquant_processor_optimize_group_reenables_grad_inside_inference_mode():
    """Guard grouped optimization under the worker lifecycle, which runs process() inside inference mode."""

    class _ToyLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = torch.nn.Module()
            self.self_attn.q_proj = torch.nn.Linear(8, 8, bias=False)
            self.self_attn.k_proj = torch.nn.Linear(8, 8, bias=False)
            self.self_attn.v_proj = torch.nn.Linear(8, 8, bias=False)

        def forward(self, x, attention_mask=None, position_ids=None, use_cache=False):
            del attention_mask, position_ids, use_cache
            return self.self_attn.q_proj(x) + self.self_attn.k_proj(x) + self.self_attn.v_proj(x)

    def _dynamic_get(_module_name, _key, default=None):
        return default

    layer = _ToyLayer()
    x = torch.randn(1, 2, 8)
    y = layer(x).detach()
    q_proj = NamedModule(layer.self_attn.q_proj, "self_attn.q_proj", "model.layers.0.self_attn.q_proj", 0)
    k_proj = NamedModule(layer.self_attn.k_proj, "self_attn.k_proj", "model.layers.0.self_attn.k_proj", 0)
    v_proj = NamedModule(layer.self_attn.v_proj, "self_attn.v_proj", "model.layers.0.self_attn.v_proj", 0)

    processor = object.__new__(ParoQuantProcessor)
    processor.qcfg = SimpleNamespace(
        runtime_bits=4,
        group_size=8,
        sym=True,
        krot=1,
        opt_seed=0,
        opt_pair_ratio=0.5,
        opt_pair_impl="fast",
        opt_quantizer_impl="reference",
        opt_fused_rotation=False,
        opt_channel_scale_clamp_min=1e-2,
        opt_channel_scale_clamp_max=1e2,
        opt_train_samples=2,
        opt_validation_samples=2,
        opt_stage_impl="fast",
        opt_rotation_lr=0.05,
        opt_weight_lr=1e-5,
        opt_quantizer_lr=1e-6,
        opt_rotation_epochs=1,
        opt_finetune_epochs=0,
        dynamic_get=_dynamic_get,
    )
    processor.gptq_model = SimpleNamespace(support_batch_quantize=True)
    processor._batch_tls = threading.local()
    processor.inputs_cache = InputCache(
        layer_inputs=[[x]],
        layer_input_kwargs=[{}],
        position_ids=[None],
        attention_masks=[None],
    )

    state = SimpleNamespace(
        layer_module=layer,
        layer_inputs=[[x]],
        layer_input_kwargs=[{}],
        layer_outputs=[[y]],
    )

    with torch.inference_mode():
        results, val_loss = processor._optimize_group(state, [q_proj, k_proj, v_proj])

    assert set(results) == {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"}
    assert val_loss >= 0.0


def test_paroquant_processor_compute_block_scope_strips_hooked_linear_wrappers():
    """ComputeBlock clone optimization must unwrap HookedLinear so backward survives cloned full-layer replay."""

    class _ToyLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = torch.nn.Module()
            self.self_attn.q_proj = torch.nn.Linear(8, 8, bias=False)
            self.self_attn.k_proj = torch.nn.Linear(8, 8, bias=False)
            self.self_attn.v_proj = torch.nn.Linear(8, 8, bias=False)
            self.self_attn.o_proj = torch.nn.Linear(8, 8, bias=False)

        def forward(self, x, attention_mask=None, position_ids=None, use_cache=False):
            del attention_mask, position_ids, use_cache
            hidden_states = (
                self.self_attn.q_proj(x)
                + self.self_attn.k_proj(x)
                + self.self_attn.v_proj(x)
            )
            return self.self_attn.o_proj(hidden_states)

    def _dynamic_get(_module_name, _key, default=None):
        return default

    float_layer = _ToyLayer()
    x = torch.randn(1, 2, 8)
    y = float_layer(x).detach()

    hooked_layer = copy.deepcopy(float_layer)
    replace_module_with_hooked_legacy(hooked_layer)

    q_proj = NamedModule(hooked_layer.self_attn.q_proj, "self_attn.q_proj", "model.layers.0.self_attn.q_proj", 0)
    k_proj = NamedModule(hooked_layer.self_attn.k_proj, "self_attn.k_proj", "model.layers.0.self_attn.k_proj", 0)
    v_proj = NamedModule(hooked_layer.self_attn.v_proj, "self_attn.v_proj", "model.layers.0.self_attn.v_proj", 0)

    processor = object.__new__(ParoQuantProcessor)
    processor.qcfg = SimpleNamespace(
        opt_scope="compute_block",
        runtime_bits=4,
        group_size=8,
        sym=True,
        krot=1,
        opt_seed=0,
        opt_pair_ratio=0.5,
        opt_pair_impl="fast",
        opt_quantizer_impl="reference",
        opt_fused_rotation=False,
        opt_channel_scale_clamp_min=1e-2,
        opt_channel_scale_clamp_max=1e2,
        opt_train_samples=2,
        opt_validation_samples=2,
        opt_stage_impl="fast",
        opt_rotation_lr=0.05,
        opt_weight_lr=1e-5,
        opt_quantizer_lr=1e-6,
        opt_rotation_epochs=1,
        opt_finetune_epochs=0,
        dynamic_get=_dynamic_get,
    )
    processor.gptq_model = SimpleNamespace(support_batch_quantize=True)
    processor._batch_tls = threading.local()
    processor.inputs_cache = InputCache(
        layer_inputs=[[x]],
        layer_input_kwargs=[{}],
        position_ids=[None],
        attention_masks=[None],
    )

    state = SimpleNamespace(
        layer_module=hooked_layer,
        layer_inputs=[[x]],
        layer_input_kwargs=[{}],
        layer_outputs=[[y]],
        pristine_layer_module=None,
        prepared_group_source_module=None,
    )

    with torch.inference_mode():
        results, val_loss = processor._optimize_group(state, [q_proj, k_proj, v_proj])

    assert set(results) == {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"}
    assert val_loss >= 0.0
    assert hooked_layer.self_attn.o_proj.__class__.__name__ == "HookedLinear"
    assert state.prepared_group_source_module is not None
    assert state.prepared_group_source_module.self_attn.o_proj.__class__ is torch.nn.Linear


def test_paroquant_processor_layer_scope_strips_hooked_linear_wrappers():
    """Layer-scope live optimization must unwrap HookedLinear so training does not re-enter inference mode."""

    class _ToyNorm(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(8))

        def forward(self, hidden_states):
            input_dtype = hidden_states.dtype
            hidden_states = hidden_states.to(torch.float32)
            return self.weight * hidden_states.to(input_dtype)

    class _ToyLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = torch.nn.Module()
            self.self_attn.q_proj = torch.nn.Linear(8, 8, bias=False)
            self.self_attn.k_proj = torch.nn.Linear(8, 8, bias=False)
            self.self_attn.v_proj = torch.nn.Linear(8, 8, bias=False)
            self.self_attn.o_proj = torch.nn.Linear(8, 8, bias=False)
            self.post_attention_layernorm = _ToyNorm()

        def forward(self, x, attention_mask=None, position_ids=None, use_cache=False):
            del attention_mask, position_ids, use_cache
            hidden_states = (
                self.self_attn.q_proj(x)
                + self.self_attn.k_proj(x)
                + self.self_attn.v_proj(x)
            )
            hidden_states = self.self_attn.o_proj(hidden_states)
            return self.post_attention_layernorm(hidden_states)

    def _dynamic_get(_module_name, _key, default=None):
        return default

    float_layer = _ToyLayer()
    x = torch.randn(1, 2, 8)
    y = float_layer(x).detach()

    hooked_layer = copy.deepcopy(float_layer)
    replace_module_with_hooked_legacy(hooked_layer)

    q_proj = NamedModule(hooked_layer.self_attn.q_proj, "self_attn.q_proj", "model.layers.0.self_attn.q_proj", 0)
    k_proj = NamedModule(hooked_layer.self_attn.k_proj, "self_attn.k_proj", "model.layers.0.self_attn.k_proj", 0)
    v_proj = NamedModule(hooked_layer.self_attn.v_proj, "self_attn.v_proj", "model.layers.0.self_attn.v_proj", 0)

    processor = object.__new__(ParoQuantProcessor)
    processor.qcfg = SimpleNamespace(
        opt_scope="layer",
        runtime_bits=4,
        group_size=8,
        sym=True,
        krot=1,
        opt_seed=0,
        opt_pair_ratio=0.5,
        opt_pair_impl="fast",
        opt_quantizer_impl="reference",
        opt_fused_rotation=False,
        opt_channel_scale_clamp_min=1e-2,
        opt_channel_scale_clamp_max=1e2,
        opt_train_samples=2,
        opt_validation_samples=2,
        opt_stage_impl="fast",
        opt_rotation_lr=0.05,
        opt_weight_lr=1e-5,
        opt_quantizer_lr=1e-6,
        opt_rotation_epochs=1,
        opt_finetune_epochs=0,
        dynamic_get=_dynamic_get,
    )
    processor.gptq_model = SimpleNamespace(support_batch_quantize=True)
    processor.model = None
    processor.lock = threading.Lock()
    processor.calculate_w_wq_diff = False
    processor._batch_tls = threading.local()
    processor.inputs_cache = InputCache(
        layer_inputs=[[x]],
        layer_input_kwargs=[{}],
        position_ids=[None],
        attention_masks=[None],
    )

    state = SimpleNamespace(
        layer_module=hooked_layer,
        layer_inputs=[[x]],
        layer_input_kwargs=[{}],
        layer_outputs=[[y]],
        replay_batches=None,
    )

    with torch.inference_mode():
        results, val_loss = processor._optimize_group(state, [q_proj, k_proj, v_proj])

    assert set(results) == {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"}
    assert val_loss >= 0.0
    assert hooked_layer.self_attn.o_proj.__class__ is torch.nn.Linear
    assert q_proj.module is hooked_layer.self_attn.q_proj
    assert k_proj.module is hooked_layer.self_attn.k_proj
    assert v_proj.module is hooked_layer.self_attn.v_proj
    assert q_proj.module.__class__ is torch.nn.Linear
    assert k_proj.module.__class__ is torch.nn.Linear
    assert v_proj.module.__class__ is torch.nn.Linear

    q_proj_result = results["self_attn.q_proj"]
    original_weight = processor._module_weight_matrix(q_proj).detach().clone()
    processor._apply_optimization_result(q_proj, q_proj_result, original_weight)
    assert torch.allclose(
        hooked_layer.self_attn.q_proj.weight.detach(),
        q_proj_result.pseudo_weight.to(dtype=hooked_layer.self_attn.q_proj.weight.dtype),
        atol=1e-5,
        rtol=1e-5,
    )


def test_paroquant_processor_layer_scope_restores_live_layer_dtype_after_fp32_training():
    """Layer-scope live optimization must downcast the layer back to its original replay dtype."""

    class _ToyConfig:
        def __init__(self):
            self._attn_implementation = "flash_attention_2"
            self.attn_implementation = "flash_attention_2"
            self.dtype = torch.bfloat16

    class _ToyNorm(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(8, dtype=torch.bfloat16))

        def forward(self, hidden_states):
            input_dtype = hidden_states.dtype
            hidden_states = hidden_states.to(torch.float32)
            return (self.weight.to(torch.float32) * hidden_states).to(input_dtype)

    class _ToyAttn(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.q_proj = torch.nn.Linear(8, 8, bias=False, dtype=torch.bfloat16)
            self.k_proj = torch.nn.Linear(8, 8, bias=False, dtype=torch.bfloat16)
            self.v_proj = torch.nn.Linear(8, 8, bias=False, dtype=torch.bfloat16)
            self.o_proj = torch.nn.Linear(8, 8, bias=False, dtype=torch.bfloat16)

        def forward(self, hidden_states, attention_mask=None, position_ids=None, use_cache=False):
            del attention_mask, position_ids, use_cache
            hidden_states = self.q_proj(hidden_states) + self.k_proj(hidden_states) + self.v_proj(hidden_states)
            return self.o_proj(hidden_states), None

    class _ToyLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = _ToyConfig()
            self.self_attn = _ToyAttn(self.config)
            self.post_attention_layernorm = _ToyNorm()

        def forward(self, x, attention_mask=None, position_ids=None, use_cache=False):
            attn_out, _ = self.self_attn(
                x,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=use_cache,
            )
            return self.post_attention_layernorm(attn_out)

    def _dynamic_get(_module_name, _key, default=None):
        return default

    layer = _ToyLayer()
    x = torch.randn(1, 2, 8, dtype=torch.bfloat16)
    y = layer(x).detach()

    q_proj = NamedModule(layer.self_attn.q_proj, "self_attn.q_proj", "model.layers.0.self_attn.q_proj", 0)
    k_proj = NamedModule(layer.self_attn.k_proj, "self_attn.k_proj", "model.layers.0.self_attn.k_proj", 0)
    v_proj = NamedModule(layer.self_attn.v_proj, "self_attn.v_proj", "model.layers.0.self_attn.v_proj", 0)

    processor = object.__new__(ParoQuantProcessor)
    processor.qcfg = SimpleNamespace(
        opt_scope="layer",
        runtime_bits=4,
        group_size=8,
        sym=True,
        krot=1,
        opt_seed=0,
        opt_pair_ratio=0.5,
        opt_pair_impl="fast",
        opt_quantizer_impl="reference",
        opt_fused_rotation=False,
        opt_channel_scale_clamp_min=1e-2,
        opt_channel_scale_clamp_max=1e2,
        opt_train_samples=2,
        opt_validation_samples=2,
        opt_stage_impl="fast",
        opt_rotation_lr=0.05,
        opt_weight_lr=1e-5,
        opt_quantizer_lr=1e-6,
        opt_rotation_epochs=1,
        opt_finetune_epochs=0,
        dynamic_get=_dynamic_get,
    )
    processor.gptq_model = SimpleNamespace(support_batch_quantize=True)
    processor.model = None
    processor._batch_tls = threading.local()
    processor.inputs_cache = InputCache(
        layer_inputs=[[x]],
        layer_input_kwargs=[{}],
        position_ids=[None],
        attention_masks=[None],
    )

    state = SimpleNamespace(
        layer_module=layer,
        layer_inputs=[[x]],
        layer_input_kwargs=[{}],
        layer_outputs=[[y]],
        replay_batches=None,
    )

    with torch.inference_mode():
        results, val_loss = processor._optimize_group(state, [q_proj, k_proj, v_proj])

    assert set(results) == {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"}
    assert val_loss >= 0.0
    assert layer.self_attn.q_proj.weight.dtype == torch.bfloat16
    assert layer.self_attn.k_proj.weight.dtype == torch.bfloat16
    assert layer.self_attn.v_proj.weight.dtype == torch.bfloat16
    assert layer.self_attn.o_proj.weight.dtype == torch.bfloat16
    assert layer.post_attention_layernorm.weight.dtype == torch.bfloat16
    assert layer(x).dtype == torch.bfloat16


def test_paroquant_quant_device_selection_forces_single_gpu():
    """Guard against multi-GPU ParoQuant worker fan-out and sync hazards."""
    cuda_devices = [torch.device("cuda:0"), torch.device("cuda:1"), torch.device("cuda:2")]
    mixed_devices = [torch.device("cpu"), torch.device("cuda:3"), torch.device("cuda:4")]

    assert _restrict_quant_devices_for_method(METHOD.PARO, cuda_devices) == [torch.device("cuda:0")]
    assert _restrict_quant_devices_for_method(METHOD.PARO, mixed_devices) == [torch.device("cuda:3")]
    assert _restrict_quant_devices_for_method(METHOD.GPTQ, cuda_devices) == cuda_devices


def test_paroquant_kernel_rejects_sym_false():
    """Guard that runtime capability flags disable asymmetric ParoQuant."""
    ok, err = ParoLinear.validate(
        bits=4,
        group_size=128,
        desc_act=False,
        sym=False,
        in_features=128,
        out_features=128,
        pack_dtype=torch.int32,
        dtype=torch.float16,
    )

    assert not ok
    assert isinstance(err, NotImplementedError)
    assert "actual sym = `False`" in str(err)


def test_paroquant_kernel_accepts_bf16():
    """Guard that saved ParoQuant checkpoints can be reloaded for bf16 inference."""
    ok, err = ParoLinear.validate(
        bits=4,
        group_size=128,
        desc_act=False,
        sym=True,
        in_features=128,
        out_features=128,
        pack_dtype=torch.int32,
        dtype=torch.bfloat16,
    )

    assert ok
    assert err is None


def test_paroquant_cuda_awq_kernel_preserves_bf16(monkeypatch):
    """Guard that the CUDA AWQ fast path does not silently downcast bf16 inputs."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required to validate the ParoQuant AWQ bf16 kernel path.")

    paroquant_module = sys.modules[ParoLinear.__module__]

    module = ParoLinear(
        bits=4,
        group_size=128,
        sym=True,
        desc_act=False,
        in_features=128,
        out_features=128,
        bias=False,
        register_buffers=True,
    ).to("cuda")
    module.scales.fill_(1)

    seen = {}

    def fake_awq_cuda_gemm_forward(input, qweight, scales, qzeros, split_k_iters, fp32_accum=True):
        del qweight, qzeros, fp32_accum
        seen["input_dtype"] = input.dtype
        seen["scales_dtype"] = scales.dtype
        seen["split_k_iters"] = split_k_iters
        return torch.zeros((input.shape[0], module.out_features), device=input.device, dtype=input.dtype)

    monkeypatch.setattr(paroquant_module, "_awq_cuda_gemm_forward", fake_awq_cuda_gemm_forward)

    x = torch.randn((2, module.in_features), device="cuda", dtype=torch.bfloat16)
    out = module._forward_cuda_awq_kernel(x)

    assert seen["input_dtype"] == torch.bfloat16
    assert seen["scales_dtype"] == torch.bfloat16
    assert seen["split_k_iters"] == 4
    assert module.scales.dtype == torch.bfloat16
    assert out is not None
    assert out.dtype == torch.bfloat16


def test_paroquant_rotation_helper_dispatches_fused_kernel_for_bf16(monkeypatch):
    """Guard bf16 activations onto the fused CUDA rotation path when ready."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required to validate the ParoQuant rotation bf16 fused path.")

    module = ParoLinear(
        bits=4,
        group_size=128,
        sym=True,
        desc_act=False,
        in_features=128,
        out_features=128,
        bias=False,
        register_buffers=True,
        auto_cache_bf16_rotation_dtype=True,
    ).to("cuda")
    module.theta.uniform_(-0.2, 0.2)
    module.channel_scales.uniform_(0.75, 1.25)
    module.post_init()

    calls = {}

    def spy_load_rotation_extension():
        calls["load_count"] = calls.get("load_count", 0) + 1
        return True

    def fake_rotate(x, pairs, theta, scales, group_size, cta_m, row_pad):
        calls["x_dtype"] = x.dtype
        calls["pairs_device"] = pairs.device.type
        calls["theta_dtype"] = theta.dtype
        calls["scales_dtype"] = None if scales is None else scales.dtype
        calls["group_size"] = group_size
        calls["cta_m"] = cta_m
        calls["row_pad"] = row_pad
        return x.clone()

    monkeypatch.setattr(paroquant_utils_module, "_load_rotation_extension", spy_load_rotation_extension)
    monkeypatch.setattr(paroquant_utils_module, "_rotation_requested_launch", lambda: (8, 2))
    monkeypatch.setattr(paroquant_utils_module._PAROQUANT_ROTATION_EXTENSION, "op", lambda name: fake_rotate)

    x = torch.randn((2, module.in_features), device="cuda", dtype=torch.bfloat16)
    theta = module.theta.to(device=x.device, dtype=torch.bfloat16)
    channel_scales = module.channel_scales.to(device=x.device, dtype=torch.bfloat16)

    actual = apply_paroquant_rotation(
        x,
        module.pairs,
        theta,
        scales=channel_scales,
        group_size=module.group_size,
    )

    assert calls["load_count"] == 1
    assert calls["x_dtype"] == torch.bfloat16
    assert calls["pairs_device"] == "cuda"
    assert calls["theta_dtype"] == torch.bfloat16
    assert calls["scales_dtype"] == torch.bfloat16
    assert calls["group_size"] == 128
    assert calls["cta_m"] == 8
    assert calls["row_pad"] == 2
    assert actual.dtype == torch.bfloat16
    assert torch.equal(actual, x)


def test_paroquant_rotation_fused_bf16_uses_fp16_workspace_contract():
    """Guard bf16 fused rotation against regressing back to bf16 workspace accumulation."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required to validate the ParoQuant rotation bf16 fused workspace path.")

    assert (
        prewarm_paroquant_rotation_extension(
            fused_rotation=True,
            group_size=128,
            krot=8,
            device="cuda",
        )
        is True
    )

    in_features = 128
    group_size = 128
    krot = 8
    pairs, _mask = build_random_rotation_buffers(
        in_features=in_features,
        group_size=group_size,
        krot=krot,
        pair_ratio=0.5,
        seed=13,
        device=torch.device("cuda"),
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(7)
    x = torch.randn((4, in_features), generator=generator, dtype=torch.float32).to(device="cuda", dtype=torch.bfloat16)
    theta = torch.empty((krot, in_features // 2), dtype=torch.float32)
    theta.uniform_(-0.25, 0.25, generator=generator)
    theta = theta.to(device="cuda", dtype=torch.bfloat16)
    scales = torch.empty((1, in_features), dtype=torch.float32)
    scales.uniform_(0.75, 1.25, generator=generator)
    scales = scales.to(device="cuda", dtype=torch.bfloat16)

    actual = apply_paroquant_rotation(x, pairs, theta, scales=scales, group_size=group_size)
    bf16_reference = apply_paroquant_rotation_reference(x, pairs, theta, scales=scales, group_size=group_size)
    fp16_workspace_reference = apply_paroquant_rotation_reference(
        x.to(dtype=torch.float16),
        pairs,
        theta.to(dtype=torch.float16),
        scales=scales.to(dtype=torch.float16),
        group_size=group_size,
    ).to(dtype=torch.bfloat16)
    fp32_reference = apply_paroquant_rotation_reference(
        x.to(dtype=torch.float32),
        pairs,
        theta.to(dtype=torch.float32),
        scales=scales.to(dtype=torch.float32),
        group_size=group_size,
    )

    actual_fp16_workspace = (actual.float() - fp16_workspace_reference.float()).abs()
    bf16_fp16_workspace = (bf16_reference.float() - fp16_workspace_reference.float()).abs()
    actual_fp32 = (actual.float() - fp32_reference).abs()
    bf16_fp32 = (bf16_reference.float() - fp32_reference).abs()

    assert actual.dtype == torch.bfloat16
    assert actual_fp16_workspace.mean().item() < bf16_fp16_workspace.mean().item()
    assert actual_fp16_workspace.max().item() < bf16_fp16_workspace.max().item()
    assert actual_fp32.mean().item() <= bf16_fp32.mean().item()
    assert actual_fp32.max().item() <= bf16_fp32.max().item()


def test_paroquant_rotation_cache_preserves_bf16(monkeypatch):
    """Guard that cached BF16 rotation metadata preserves the runtime dtype and values."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required to validate the ParoQuant rotation cache path.")

    paroquant_module = sys.modules[ParoLinear.__module__]
    from gptqmodel.utils import paroquant as paroquant_utils

    module = ParoLinear(
        bits=4,
        group_size=128,
        sym=True,
        desc_act=False,
        in_features=128,
        out_features=128,
        bias=False,
        register_buffers=True,
        auto_cache_bf16_rotation_dtype=True,
    ).to("cuda")
    module.theta.uniform_(-0.2, 0.2)
    module.channel_scales.uniform_(0.75, 1.25)
    module.post_init()

    seen = {}
    original_rotate = paroquant_utils.apply_paroquant_rotation

    def spy_rotate(x, pairs, theta, scales=None, group_size=128):
        del pairs, group_size
        seen["x_dtype"] = x.dtype
        seen["theta_dtype"] = theta.dtype
        seen["scales_dtype"] = None if scales is None else scales.dtype
        return original_rotate(x, module.pairs, theta, scales=scales, group_size=module.group_size)

    monkeypatch.setattr(paroquant_module, "apply_paroquant_rotation", spy_rotate)

    x = torch.randn((2, module.in_features), device="cuda", dtype=torch.bfloat16)
    module._rotate_inputs(x)
    baseline = module._rotate_inputs(x)
    cached = module._rotate_inputs(x)

    assert seen["x_dtype"] == torch.bfloat16
    assert seen["theta_dtype"] == torch.bfloat16
    assert seen["scales_dtype"] == torch.bfloat16
    assert module._runtime_theta is not None
    assert module._runtime_channel_scales is not None
    assert module._runtime_theta.dtype == torch.bfloat16
    assert module._runtime_channel_scales.dtype == torch.bfloat16
    assert torch.equal(baseline, cached)


def test_paroquant_optimizer_improves_over_identity_quantization():
    """Guard that learned rotations beat naive identity-domain quantization."""
    in_features = 128
    out_features = 12
    group_size = 128
    bits = 4
    seed = 11
    pair_ratio = 1.0 / group_size

    pairs, mask = build_random_rotation_buffers(
        in_features=in_features,
        group_size=group_size,
        krot=1,
        pair_ratio=pair_ratio,
        seed=seed,
        device=torch.device("cpu"),
    )

    theta = torch.zeros((1, in_features // 2), dtype=torch.float32)
    theta[~mask] = 0.65
    channel_scales_opt = torch.linspace(0.75, 1.25, steps=in_features, dtype=torch.float32).view(1, -1)
    transformed_weight = (torch.randint(-7, 8, (out_features, in_features), dtype=torch.int32).to(torch.float32)) * 0.25

    original_weight = apply_paroquant_rotation_reference(
        transformed_weight,
        pairs.flip(0),
        -theta.flip(0),
        scales=None,
        group_size=group_size,
    ) / channel_scales_opt

    inputs = torch.randn(256, in_features, dtype=torch.float32)
    targets = F.linear(inputs, original_weight)

    baseline_weight = pseudo_quantize_dequant(
        original_weight,
        bits=bits,
        group_size=group_size,
        sym=True,
        use_ste=False,
    )
    baseline_loss = F.smooth_l1_loss(F.linear(inputs, baseline_weight), targets)

    result = optimize_paroquant_linear(
        weight=original_weight,
        bias=None,
        inputs=inputs,
        bits=bits,
        group_size=group_size,
        sym=True,
        krot=1,
        pair_ratio=pair_ratio,
        train_rows=192,
        val_rows=64,
        batch_size=32,
        rotation_epochs=24,
        finetune_epochs=16,
        rotation_lr=0.05,
        weight_lr=5e-4,
        quantizer_lr=5e-4,
        seed=seed,
    )

    optimized_loss = F.smooth_l1_loss(F.linear(inputs, result.pseudo_weight), targets)
    assert optimized_loss < baseline_loss


def test_paroquant_exported_runtime_state_matches_paper_contract():
    """Guard that export tensors reproduce the pseudo-quantized optimization model.

    This is the key paper-contract regression test. It checks that we optimize
    in the transformed domain, inverse-map back to the input domain for replay,
    and then export runtime tensors whose rotated-input execution matches the
    pseudo-quantized layer exactly.
    """
    in_features = 128
    out_features = 10
    group_size = 128
    bits = 4

    weight = torch.randn(out_features, in_features, dtype=torch.float32) * 0.3
    bias = torch.randn(out_features, dtype=torch.float32) * 0.1
    pairs, theta_mask = build_random_rotation_buffers(
        in_features=in_features,
        group_size=group_size,
        krot=2,
        pair_ratio=2.0 / group_size,
        seed=7,
        device=torch.device("cpu"),
    )

    model = _ParoQuantOptimLinear(
        weight,
        bias,
        bits=bits,
        group_size=group_size,
        quantizer_sym=True,
        pairs=pairs,
        theta_mask=theta_mask,
    )

    with torch.no_grad():
        model.theta.uniform_(-0.35, 0.35)
        model.reset_masked_angles()
        model.channel_scales_opt.copy_(torch.linspace(0.8, 1.2, steps=in_features))
        model.weight.add_(torch.linspace(-0.05, 0.05, steps=in_features).view(1, -1))

    model.quantizer = GroupLinearQuantizer(
        model.transformed_weight().detach(),
        bits=bits,
        group_size=group_size,
        sym=True,
    )
    with torch.no_grad():
        model.quantizer.scale.mul_(1.05)

    transformed = apply_paroquant_rotation_reference(
        model.weight.detach() * model.channel_scales_opt.detach().view(1, -1),
        model.pairs,
        model.theta.detach(),
        scales=None,
        group_size=group_size,
    )
    quantized_transformed = pseudo_quantize_dequant(
        transformed,
        bits=bits,
        group_size=group_size,
        sym=True,
        scale=model.quantizer.scale.detach(),
        zero_point_float=None,
        use_ste=False,
    )
    expected_pseudo_weight = apply_paroquant_rotation_reference(
        quantized_transformed,
        model.pairs.flip(0),
        -model.theta.detach().flip(0),
        scales=None,
        group_size=group_size,
    ) / model.channel_scales_opt.detach().view(1, -1)

    torch.testing.assert_close(model.pseudo_weight().detach(), expected_pseudo_weight, atol=1e-5, rtol=1e-5)

    pack_weight, _q_scales, _q_zeros, theta, runtime_channel_scales = model.export_pack_state()
    inputs = torch.randn(32, in_features, dtype=torch.float32)
    runtime_outputs = F.linear(
        apply_paroquant_rotation_reference(
            inputs,
            model.pairs,
            theta,
            scales=runtime_channel_scales,
            group_size=group_size,
        ),
        pack_weight,
        bias,
    )

    torch.testing.assert_close(
        runtime_outputs,
        F.linear(inputs, model.pseudo_weight().detach(), bias),
        atol=1e-5,
        rtol=1e-5,
    )


def test_paroquant_result_from_model_matches_direct_export():
    """Guard the fused result export against the original direct pseudo/export path."""
    torch.manual_seed(7)
    in_features = 128
    out_features = 64
    group_size = 128
    bits = 4
    weight = torch.randn((out_features, in_features), dtype=torch.float32)
    bias = torch.randn((out_features,), dtype=torch.float32)
    pairs, theta_mask = build_random_rotation_buffers(
        in_features=in_features,
        group_size=group_size,
        krot=2,
        pair_ratio=2.0 / in_features,
        seed=7,
        device=torch.device("cpu"),
    )
    model = _ParoQuantOptimLinear(
        weight,
        bias,
        bits=bits,
        group_size=group_size,
        quantizer_sym=True,
        pairs=pairs,
        theta_mask=theta_mask,
        fused_rotation=False,
    )
    model.init_quantizer()
    with torch.no_grad():
        model.theta.normal_(mean=0.0, std=0.05)
        model.channel_scales_opt.uniform_(0.8, 1.2)

    direct_pseudo_weight = model.pseudo_weight().detach()
    direct_pack_weight, direct_q_scales, direct_q_zeros, direct_theta, direct_channel_scales = model.export_pack_state()
    result = paroquant_optimization._result_from_model(
        model,
        train_loss=0.123,
        val_loss=0.456,
        used_identity=False,
    )

    torch.testing.assert_close(result.pseudo_weight, direct_pseudo_weight, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(result.pack_weight, direct_pack_weight, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(result.q_scales, direct_q_scales, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(result.q_zeros, direct_q_zeros, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(result.theta, direct_theta, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(result.channel_scales, direct_channel_scales, atol=1e-5, rtol=1e-5)
    assert result.train_loss == pytest.approx(0.123)
    assert result.val_loss == pytest.approx(0.456)
    assert result.used_identity is False


def test_paroquant_reference_quantizer_exports_affine_qzeros():
    """Guard that reference optimizer mode uses affine qparams despite sym runtime config."""
    torch.manual_seed(5)
    weight = torch.randn(32, 128, dtype=torch.float32) * 0.25 + 0.1
    inputs = torch.randn(96, 128, dtype=torch.float32)

    result = optimize_paroquant_linear(
        weight=weight,
        bias=None,
        inputs=inputs,
        bits=4,
        group_size=128,
        sym=True,
        krot=2,
        pair_ratio=2.0 / 128.0,
        train_rows=64,
        val_rows=32,
        batch_size=16,
        rotation_epochs=1,
        finetune_epochs=1,
        rotation_lr=0.05,
        weight_lr=1e-5,
        quantizer_lr=1e-6,
        seed=11,
        stage_impl="reference",
        pair_impl="reference",
        quantizer_impl="reference",
    )

    midpoint = 2 ** (4 - 1)
    assert not torch.all(result.q_zeros == midpoint)
