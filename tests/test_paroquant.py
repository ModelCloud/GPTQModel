# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# ParoQuant test coverage adapted from the ParoQuant paper and public project:
# https://arxiv.org/html/2511.10645v2
# https://github.com/z-lab/paroquant

"""Unit tests for ParoQuant config, optimizer, and lifecycle invariants."""

from contextlib import contextmanager
import sys
import threading
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from transformers.quantizers.auto import AutoQuantizationConfig
from transformers.utils.quantization_config import GPTQConfig

from gptqmodel.looper.awq_processor import AWQProcessor
from gptqmodel.looper.module_looper import _restrict_quant_devices_for_method
from gptqmodel.looper.paroquant_processor import ParoQuantProcessor
from gptqmodel.nn_modules.qlinear.paroquant import ParoQuantQuantLinear
from gptqmodel.quantization.config import FORMAT, METHOD, ParoQuantizeConfig, QuantizeConfig
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
    apply_paroquant_rotation_reference,
    build_identity_rotation_buffers,
)


def test_paroquant_quantize_config_dispatches_constructor():
    """Guard that ParoQuant config fields survive direct construction."""
    cfg = QuantizeConfig(
        quant_method=METHOD.PAROQUANT,
        format=FORMAT.PAROQUANT,
        bits=4,
        group_size=128,
        krot=8,
    )

    assert cfg.quant_method == METHOD.PAROQUANT
    assert cfg.format == FORMAT.PAROQUANT
    assert cfg.krot == 8
    assert cfg.opt_batch_size == 64
    assert cfg.opt_stage_impl == "fast"
    assert cfg.opt_pair_impl == "fast"
    assert cfg.opt_quantizer_impl == "reference"
    assert cfg.opt_channel_scale_clamp_min == 1e-2
    assert cfg.opt_channel_scale_clamp_max == 1e2
    assert cfg.export_quant_method() == METHOD.PAROQUANT


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
                "opt_fused_rotation": False,
                "opt_stage_impl": "reference",
                "opt_pair_impl": "fast",
                "opt_quantizer_impl": "reference",
                "opt_channel_scale_clamp_min": 0.02,
                "opt_channel_scale_clamp_max": 50.0,
            },
        }
    )

    assert isinstance(cfg, ParoQuantizeConfig)
    assert cfg.quant_method == METHOD.PAROQUANT
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
    assert cfg.opt_fused_rotation is False
    assert cfg.opt_stage_impl == "reference"
    assert cfg.opt_pair_impl == "fast"
    assert cfg.opt_quantizer_impl == "reference"
    assert cfg.opt_channel_scale_clamp_min == 0.02
    assert cfg.opt_channel_scale_clamp_max == 50.0
    assert cfg.to_dict()["meta"]["opt_fused_rotation"] is False
    assert cfg.to_dict()["meta"]["opt_stage_impl"] == "reference"
    assert cfg.to_dict()["meta"]["opt_pair_impl"] == "fast"
    assert cfg.to_dict()["meta"]["opt_quantizer_impl"] == "reference"
    assert cfg.to_dict()["meta"]["opt_channel_scale_clamp_min"] == 0.02
    assert cfg.to_dict()["meta"]["opt_channel_scale_clamp_max"] == 50.0


def test_paroquant_quantize_config_rejects_invalid_scale_clamp_range():
    """Guard that ParoQuant scale-clamp overrides remain numerically valid."""
    with pytest.raises(ValueError, match="scale clamp bounds must be positive"):
        ParoQuantizeConfig(
            bits=4,
            group_size=128,
            opt_channel_scale_clamp_min=0.0,
            opt_channel_scale_clamp_max=10.0,
        )

    with pytest.raises(ValueError, match="opt_channel_scale_clamp_min"):
        ParoQuantizeConfig(
            bits=4,
            group_size=128,
            opt_channel_scale_clamp_min=10.0,
            opt_channel_scale_clamp_max=10.0,
        )


def test_paroquant_rotation_toggle_prefers_explicit_config_over_env(monkeypatch):
    """Guard that the config-backed fused toggle overrides the legacy env fallback."""
    x = torch.randn(4, 8, dtype=torch.float32)
    pairs, theta, channel_scales = build_identity_rotation_buffers(
        in_features=8,
        group_size=8,
        krot=1,
        dtype=torch.float32,
    )

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
    from gptqmodel.nn_modules.qlinear.paroquant import ParoQuantQuantLinear

    assert (
        get_kernel_for_backend(BACKEND.PAROQUANT_CUDA, METHOD.PAROQUANT, FORMAT.PAROQUANT)
        is ParoQuantQuantLinear
    )


def test_paroquant_kernel_mapping_uses_paroquant_triton_backend():
    """Guard Triton backend dispatch for ParoQuant-specific runtime modules."""
    from gptqmodel.nn_modules.qlinear.paroquant_triton import ParoQuantTritonQuantLinear

    assert (
        get_kernel_for_backend(BACKEND.PAROQUANT_TRITON, METHOD.PAROQUANT, FORMAT.PAROQUANT)
        is ParoQuantTritonQuantLinear
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


def test_paroquant_quant_device_selection_forces_single_gpu():
    """Guard against multi-GPU ParoQuant worker fan-out and sync hazards."""
    cuda_devices = [torch.device("cuda:0"), torch.device("cuda:1"), torch.device("cuda:2")]
    mixed_devices = [torch.device("cpu"), torch.device("cuda:3"), torch.device("cuda:4")]

    assert _restrict_quant_devices_for_method(METHOD.PAROQUANT, cuda_devices) == [torch.device("cuda:0")]
    assert _restrict_quant_devices_for_method(METHOD.PAROQUANT, mixed_devices) == [torch.device("cuda:3")]
    assert _restrict_quant_devices_for_method(METHOD.GPTQ, cuda_devices) == cuda_devices


def test_paroquant_kernel_rejects_sym_false():
    """Guard that runtime capability flags disable asymmetric ParoQuant."""
    ok, err = ParoQuantQuantLinear.validate(
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
    ok, err = ParoQuantQuantLinear.validate(
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

    paroquant_module = sys.modules[ParoQuantQuantLinear.__module__]

    module = ParoQuantQuantLinear(
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
        del qweight, qzeros, split_k_iters, fp32_accum
        seen["input_dtype"] = input.dtype
        seen["scales_dtype"] = scales.dtype
        return torch.zeros((input.shape[0], module.out_features), device=input.device, dtype=input.dtype)

    monkeypatch.setattr(paroquant_module, "awq_ext", object())
    monkeypatch.setattr(paroquant_module, "_awq_cuda_gemm_forward", fake_awq_cuda_gemm_forward)

    x = torch.randn((2, module.in_features), device="cuda", dtype=torch.bfloat16)
    out = module._forward_cuda_awq_kernel(x)

    assert seen["input_dtype"] == torch.bfloat16
    assert seen["scales_dtype"] == torch.bfloat16
    assert out is not None
    assert out.dtype == torch.bfloat16


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
