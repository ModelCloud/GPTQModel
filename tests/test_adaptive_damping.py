# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch
import torch.nn as nn

from gptqmodel.quantization.config import AdaptiveDampingConfig, DampConfig, QuantizeConfig
from gptqmodel.quantization.gptq import GPTQ


def _make_spd_matrix(d: int, device) -> torch.Tensor:
    """Return a small positive-definite matrix with a spread spectrum."""
    diag = torch.linspace(0.001, 10.0, d, device=device, dtype=torch.float32)
    H = torch.diag(diag)
    H = H + torch.randn(d, d, device=device, dtype=torch.float32) * 0.0001
    return H @ H.t()


def test_adaptive_damping_is_opt_in():
    cfg = QuantizeConfig(method="gptq")
    assert isinstance(cfg.adaptive_damping, DampConfig)
    assert cfg.adaptive_damping.min == 0.05
    assert cfg.adaptive_damping.max == 0.05
    assert cfg.adaptive_damping.step == 0.01
    assert cfg.adaptive_clipping is None
    assert cfg.damp is cfg.adaptive_damping


def test_adaptive_damping_config_round_trip():
    payload = {
        "method": "gptq",
        "bits": 4,
        "group_size": 128,
        "adaptive_damping": {
            "enabled": True,
            "base_percdamp": 0.04,
            "min": 0.02,
            "max": 0.08,
            "eigen_iterations": 15,
            "spectral_alpha": 0.30,
        },
    }
    cfg = QuantizeConfig.from_quant_config(payload)
    assert cfg.adaptive_damping.enabled is True
    assert cfg.adaptive_damping.base_percdamp == 0.04
    assert cfg.adaptive_damping.min == 0.02
    assert cfg.adaptive_damping.max == 0.08
    assert cfg.adaptive_damping.eigen_iterations == 15
    assert cfg.adaptive_damping.spectral_alpha == 0.30
    assert cfg.to_dict()["meta"]["adaptive_damping"] == cfg.adaptive_damping.to_dict()


def test_adaptive_damping_config_validation():
    with pytest.raises(ValueError):
        AdaptiveDampingConfig(min=0.0)
    with pytest.raises(ValueError):
        AdaptiveDampingConfig(max=1.0)
    with pytest.raises(ValueError):
        AdaptiveDampingConfig(min=0.05, max=0.01)
    with pytest.raises(ValueError):
        AdaptiveDampingConfig(base_percdamp=0.0)
    with pytest.raises(ValueError):
        AdaptiveDampingConfig(spectral_alpha=0.0)
    with pytest.raises(ValueError):
        AdaptiveDampingConfig(eigen_iterations=0)
    with pytest.raises(ValueError):
        AdaptiveDampingConfig(method="lanczos2")
    with pytest.raises(ValueError):
        AdaptiveDampingConfig(method="gershgorin")
    with pytest.raises(ValueError):
        AdaptiveDampingConfig(module_factors={"down": -0.5})
    with pytest.raises(ValueError):
        AdaptiveDampingConfig(group_error_ema_decay=1.0)
    with pytest.raises(ValueError):
        AdaptiveDampingConfig(group_error_scale_min=0.0)
    with pytest.raises(ValueError):
        AdaptiveDampingConfig(group_error_scale_min=1.5, group_error_scale_max=1.2)
    # Valid methods should instantiate without error.
    AdaptiveDampingConfig(method="power_iteration")
    AdaptiveDampingConfig(method="lanczos")
    AdaptiveDampingConfig(method="diagonal")


@pytest.mark.parametrize("device", [
    "cpu",
    pytest.param(
        "cuda",
        marks=[
            pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
            pytest.mark.gpu,
        ],
    ),
])
def test_adaptive_damping_disabled_uses_legacy_damp_percent(device):
    cfg = QuantizeConfig(method="gptq", damp_percent=0.02, adaptive_damping={"enabled": False})
    assert isinstance(cfg.damp, DampConfig)
    m = nn.Linear(64, 64, bias=False).to(device)
    q = GPTQ(m, qcfg=cfg)
    H = torch.eye(64, device=device, dtype=torch.float32)
    diag = H.diagonal().clone()
    mean = diag.mean()
    damp = q._resolve_initial_damp(H, diag, mean)
    assert damp == pytest.approx(0.02)


@pytest.mark.parametrize("device", [
    "cpu",
    pytest.param(
        "cuda",
        marks=[
            pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
            pytest.mark.gpu,
        ],
    ),
])
def test_adaptive_damping_estimates_and_clamps(device):
    cfg = QuantizeConfig(
        method="gptq",
        bits=4,
        group_size=16,
        adaptive_damping={
            "enabled": True,
            "base_percdamp": 0.05,
            "min": 0.02,
            "max": 0.08,
            "eigen_iterations": 30,
        },
    )
    m = nn.Linear(64, 64, bias=False).to(device)
    q = GPTQ(m, qcfg=cfg)
    H = _make_spd_matrix(64, device)
    diag = H.diagonal().clone()
    mean = diag.mean()
    damp = q._resolve_initial_damp(H, diag, mean)
    assert 0.02 <= damp <= 0.08


@pytest.mark.parametrize("device", [
    "cpu",
    pytest.param(
        "cuda",
        marks=[
            pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
            pytest.mark.gpu,
        ],
    ),
])
def test_adaptive_damping_diagonal_method_is_cost_free_and_clamped(device):
    cfg = QuantizeConfig(
        method="gptq",
        bits=4,
        group_size=16,
        damp_percent=0.001,
        adaptive_damping={
            "enabled": True,
            "method": "diagonal",
            "min": 0.001,
            "max": 0.04,
        },
    )
    m = nn.Linear(64, 64, bias=False).to(device)
    q = GPTQ(m, qcfg=cfg)
    H = _make_spd_matrix(64, device)
    diag = H.diagonal().clone()
    mean = diag.mean()
    damp = q._resolve_initial_damp(H, diag, mean)
    assert 0.001 <= damp <= 0.04


@pytest.mark.parametrize("device", [
    "cpu",
    pytest.param(
        "cuda",
        marks=[
            pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
            pytest.mark.gpu,
        ],
    ),
])
def test_adaptive_damping_gptq_quantize_produces_finite_result(device):
    cfg = QuantizeConfig(
        method="gptq",
        bits=4,
        group_size=16,
        desc_act=False,
        adaptive_damping={
            "enabled": True,
            "method": "power_iteration",
            "eigen_iterations": 20,
            "min": 0.001,
            "max": 0.10,
        },
    )
    torch.manual_seed(0)
    dtype = torch.float16 if device == "cuda" else torch.float32
    layer = nn.Linear(16, 12, bias=False, dtype=dtype).eval().to(device)
    gptq = GPTQ(layer, qcfg=cfg)
    gptq.quantizer.configure(perchannel=True)
    calibration = torch.randn(8, 16, device=device, dtype=dtype)
    gptq.add_batch(calibration, None)
    gptq.finalize_hessian()
    Q, scale, zero, g_idx, duration, avg_loss, damp, nsamples = gptq.quantize(blocksize=8)
    assert Q.shape == layer.weight.shape
    assert torch.isfinite(Q).all()
    assert torch.isfinite(scale).all()
    assert 0 < damp < 1


@pytest.mark.parametrize("device", [
    "cpu",
    pytest.param(
        "cuda",
        marks=[
            pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
            pytest.mark.gpu,
        ],
    ),
])
def test_adaptive_damping_online_group_feedback(device):
    """v4.2: online per-group residual feedback runs inside the GPTQ group loop."""
    cfg = QuantizeConfig(
        method="gptq",
        bits=4,
        group_size=8,
        desc_act=False,
        adaptive_damping={
            "enabled": True,
            "group_error_enabled": True,
            "group_size_prior_enabled": True,
            "min": 0.02,
            "max": 0.08,
        },
    )
    torch.manual_seed(0)
    dtype = torch.float16 if device == "cuda" else torch.float32
    layer = nn.Linear(16, 12, bias=False, dtype=dtype).eval().to(device)
    gptq = GPTQ(layer, qcfg=cfg)
    gptq.quantizer.configure(perchannel=True)
    calibration = torch.randn(8, 16, device=device, dtype=dtype)
    gptq.add_batch(calibration, None)
    gptq.finalize_hessian()
    Q, scale, zero, g_idx, duration, avg_loss, damp, nsamples = gptq.quantize(blocksize=8)
    assert Q.shape == layer.weight.shape
    assert torch.isfinite(Q).all()
    assert torch.isfinite(scale).all()
    assert torch.isfinite(zero).all()
    assert 0 < damp < 1


@pytest.mark.parametrize("device", [
    "cpu",
    pytest.param(
        "cuda",
        marks=[
            pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
            pytest.mark.gpu,
        ],
    ),
])
def test_adaptive_damping_respects_high_damp_percent(device):
    """Explicit high `damp_percent` must not be silently lowered by the adaptive ceiling."""
    cfg = QuantizeConfig(
        method="gptq",
        damp_percent=0.15,
        adaptive_damping={
            "enabled": True,
            "min": 0.02,
            "max": 0.08,
        },
    )
    m = nn.Linear(64, 64, bias=False).to(device)
    q = GPTQ(m, qcfg=cfg)
    H = _make_spd_matrix(64, device)
    diag = H.diagonal().clone()
    mean = diag.mean()
    damp = q._resolve_initial_damp(H, diag, mean)
    assert damp == pytest.approx(0.15)


def test_group_error_scale_is_clamped():
    """v4.2: the final error scale is clamped to avoid extreme updates."""
    # Small group with no feedback would give 1.41, clamped to 1.2.
    assert GPTQ._group_error_scale(1.0, 0.71) == pytest.approx(1.2)
    # Large group with no feedback would give ~0.84, not clamped.
    assert GPTQ._group_error_scale(1.0, 1.19) == pytest.approx(1.0 / 1.19)
    # A strong feedback signal is still bounded by the scale clamp.
    assert GPTQ._group_error_scale(1.1, 0.71) == pytest.approx(1.2)
    assert GPTQ._group_error_scale(0.9, 1.19) == pytest.approx(0.8)


def test_adaptive_damping_hessian_weighting_and_raw_residual_config():
    """Hessian-weighted loss and raw-residual measurement can be toggled via config."""
    cfg = QuantizeConfig(
        method="gptq",
        adaptive_damping={
            "enabled": True,
            "group_error_enabled": True,
            "group_error_use_hessian_weighting": True,
            "group_error_measure_raw_residual": True,
        },
    )
    d = cfg.to_dict()["meta"]["adaptive_damping"]
    assert d["group_error_use_hessian_weighting"] is True
    assert d["group_error_measure_raw_residual"] is True

    cfg2 = QuantizeConfig(
        method="gptq",
        adaptive_damping={
            "enabled": True,
            "group_error_enabled": True,
            "group_error_use_hessian_weighting": False,
            "group_error_measure_raw_residual": False,
        },
    )
    assert cfg2.adaptive_damping.group_error_use_hessian_weighting is False
    assert cfg2.adaptive_damping.group_error_measure_raw_residual is False


def test_online_feedback_enabled_gates_online_group_damping():
    """online_feedback_enabled can disable per-group scaling independently."""
    cfg = QuantizeConfig(
        method="gptq",
        group_size=8,
        adaptive_damping={
            "enabled": True,
            "online_feedback_enabled": False,
            "group_error_enabled": True,
            "group_size_prior_enabled": True,
        },
    )
    assert cfg.adaptive_damping.online_feedback_enabled is False
    m = nn.Linear(16, 12, bias=False)
    gptq = GPTQ(m, qcfg=cfg)
    gptq.quantizer.configure(perchannel=True)
    calibration = torch.randn(8, 16)
    gptq.add_batch(calibration, None)
    gptq.finalize_hessian()
    Q, *_ = gptq.quantize(blocksize=8)
    assert torch.isfinite(Q).all()


def test_group_feedback_update_direction():
    """v4.2: high group loss reduces the feedback scale, low loss increases it."""
    target, feedback = GPTQ._update_group_feedback(1.0, None, 0.9, 0.1, 0.9, 1.1)
    assert target == 1.0
    assert feedback == 1.0

    # Loss twice the target -> feedback < 1 (more damping).
    target, feedback = GPTQ._update_group_feedback(2.0, 1.0, 0.9, 0.1, 0.9, 1.1)
    assert target > 1.0
    assert 0.9 <= feedback < 1.0

    # Loss half the target -> feedback > 1 (less damping).
    target, feedback = GPTQ._update_group_feedback(0.5, 1.0, 0.9, 0.1, 0.9, 1.1)
    assert target < 1.0
    assert 1.0 < feedback <= 1.1

    # Clamping is respected.
    _, feedback = GPTQ._update_group_feedback(1e9, 1.0, 0.9, 0.1, 0.9, 1.1)
    assert feedback == pytest.approx(0.9)
    _, feedback = GPTQ._update_group_feedback(1e-9, 1.0, 0.9, 0.1, 0.9, 1.1)
    assert feedback == pytest.approx(1.1)


def test_group_error_scale_size_prior_direction():
    """v4.2: larger groups receive a smaller error update (more damping)."""
    # No feedback and reference size 128: larger group -> smaller scale.
    assert GPTQ._group_error_scale(1.0, 0.71) > 1.0  # group smaller than 128
    assert GPTQ._group_error_scale(1.0, 1.19) < 1.0  # group larger than 128
    # High loss feedback further reduces the scale.
    assert GPTQ._group_error_scale(0.8, 1.19) < GPTQ._group_error_scale(1.0, 1.19)


@pytest.mark.parametrize("device", [
    "cpu",
    pytest.param(
        "cuda",
        marks=[
            pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
            pytest.mark.gpu,
        ],
    ),
])
def test_adaptive_damping_online_group_feedback_invokes_helper(device, monkeypatch):
    """v4.2: online group feedback exercises the helper path during quantize()."""
    cfg = QuantizeConfig(
        method="gptq",
        bits=4,
        group_size=8,
        desc_act=False,
        adaptive_damping={
            "enabled": True,
            "group_error_enabled": True,
            "group_size_prior_enabled": True,
            "min": 0.02,
            "max": 0.08,
        },
    )
    calls = []
    original = GPTQ._update_group_feedback

    def counting(L_g, feedback_target, ema_decay, gamma, factor_min, factor_max):
        calls.append((L_g, feedback_target))
        return original(L_g, feedback_target, ema_decay, gamma, factor_min, factor_max)

    monkeypatch.setattr(GPTQ, "_update_group_feedback", staticmethod(counting))

    torch.manual_seed(0)
    dtype = torch.float16 if device == "cuda" else torch.float32
    layer = nn.Linear(16, 12, bias=False, dtype=dtype).eval().to(device)
    gptq = GPTQ(layer, qcfg=cfg)
    gptq.quantizer.configure(perchannel=True)
    calibration = torch.randn(8, 16, device=device, dtype=dtype)
    gptq.add_batch(calibration, None)
    gptq.finalize_hessian()
    Q, scale, zero, g_idx, duration, avg_loss, damp, nsamples = gptq.quantize(blocksize=8)
    assert Q.shape == layer.weight.shape
    assert torch.isfinite(Q).all()
    assert torch.isfinite(scale).all()
    assert torch.isfinite(zero).all()
    assert 0 < damp < 1
    assert len(calls) >= 1


def _make_ill_conditioned_hessian(d: int, device, condition: float = 1e6) -> torch.Tensor:
    """Return a positive-definite matrix with the requested condition number."""
    log_min = -math.log10(condition)
    diag = torch.logspace(log_min, 0.0, d, device=device, dtype=torch.float32)
    A = torch.diag(diag)
    A = A + torch.randn(d, d, device=device, dtype=torch.float32) * (diag.max() * 1e-4)
    return A @ A.t()


@pytest.mark.parametrize("device", [
    "cpu",
    pytest.param(
        "cuda",
        marks=[
            pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
            pytest.mark.gpu,
        ],
    ),
])
def test_adaptive_damping_identity_hessian(device):
    """An identity Hessian has spectral ratio ~1, so damping stays near base."""
    cfg = QuantizeConfig(
        method="gptq",
        adaptive_damping={"enabled": True, "base_percdamp": 0.05, "min": 0.02, "max": 0.08},
    )
    m = nn.Linear(32, 32, bias=False).to(device)
    q = GPTQ(m, qcfg=cfg)
    H = torch.eye(32, device=device, dtype=torch.float32)
    diag = H.diagonal().clone()
    damp = q._resolve_initial_damp(H, diag, diag.mean())
    assert damp == pytest.approx(0.05, abs=0.005)


@pytest.mark.parametrize("device", [
    "cpu",
    pytest.param(
        "cuda",
        marks=[
            pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
            pytest.mark.gpu,
        ],
    ),
])
def test_adaptive_damping_ill_conditioned_hessian(device):
    """A wide spectrum increases the spectral correction above the base damping."""
    cfg = QuantizeConfig(
        method="gptq",
        adaptive_damping={"enabled": True, "base_percdamp": 0.05, "min": 0.02, "max": 0.2},
    )
    m = nn.Linear(64, 64, bias=False).to(device)
    q = GPTQ(m, qcfg=cfg)
    H = _make_ill_conditioned_hessian(64, device, condition=1e6)
    diag = H.diagonal().clone()
    damp = q._resolve_initial_damp(H, diag, diag.mean())
    # Spectral factor should push damping above the 0.05 baseline.
    assert damp > 0.055
    assert 0.02 <= damp <= 0.2


@pytest.mark.parametrize("device", [
    "cpu",
    pytest.param(
        "cuda",
        marks=[
            pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
            pytest.mark.gpu,
        ],
    ),
])
def test_adaptive_damping_singular_hessian_recovery(device):
    """Fewer calibration samples than columns produces a singular H; quantize must recover."""
    cfg = QuantizeConfig(
        method="gptq",
        bits=4,
        group_size=16,
        adaptive_damping={"enabled": True, "min": 0.001, "max": 0.2},
    )
    torch.manual_seed(0)
    dtype = torch.float32
    layer = nn.Linear(64, 64, bias=False, dtype=dtype).eval().to(device)
    gptq = GPTQ(layer, qcfg=cfg)
    gptq.quantizer.configure(perchannel=True)
    calibration = torch.randn(8, 64, device=device, dtype=dtype)
    gptq.add_batch(calibration, None)
    gptq.finalize_hessian()
    Q, scale, zero, g_idx, duration, avg_loss, damp, nsamples = gptq.quantize(blocksize=16)
    assert Q.shape == layer.weight.shape
    assert torch.isfinite(Q).all()
    assert torch.isfinite(scale).all()
    assert torch.isfinite(zero).all()
    assert 0 < damp <= 1.0


@pytest.mark.parametrize("method", ["power_iteration", "lanczos", "diagonal"])
@pytest.mark.parametrize("device", [
    "cpu",
    pytest.param(
        "cuda",
        marks=[
            pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
            pytest.mark.gpu,
        ],
    ),
])
def test_adaptive_damping_eigen_estimator_finite(device, method):
    """Each eigen estimator returns a finite lambda_max and treats lambda_min as 0."""
    cfg = QuantizeConfig(method="gptq", adaptive_damping={"enabled": True, "method": method})
    m = nn.Linear(64, 64, bias=False).to(device)
    q = GPTQ(m, qcfg=cfg)
    H = _make_spd_matrix(64, device)
    diag = H.diagonal().clone()
    lambda_max, lambda_min = q._estimate_hessian_eigen_spectrum(
        H, method, 10, effective_diag=diag
    )
    assert lambda_max is not None
    assert math.isfinite(lambda_max) and lambda_max > 0
    assert lambda_min == 0.0


@pytest.mark.parametrize("device", [
    "cpu",
    pytest.param(
        "cuda",
        marks=[
            pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
            pytest.mark.gpu,
        ],
    ),
])
def test_adaptive_damping_power_iteration_is_deterministic(device):
    """Power iteration uses a deterministic module seed, yielding the same lambda_max."""
    cfg = QuantizeConfig(method="gptq")
    m = nn.Linear(64, 64, bias=False).to(device)
    q = GPTQ(m, qcfg=cfg)
    H = _make_spd_matrix(64, device)
    l1, _ = q._estimate_hessian_eigen_spectrum(H, "power_iteration", 20)
    l2, _ = q._estimate_hessian_eigen_spectrum(H, "power_iteration", 20)
    assert l1 == pytest.approx(l2, rel=1e-4)


@pytest.mark.parametrize(("name", "factor", "expected"), [
    ("q", 1.0, 0.05),
    ("k", 1.0, 0.05),
    ("v", 1.0, 0.05),
    ("o", 1.0, 0.05),
    ("gate", 1.1, 0.055),
    ("up", 1.1, 0.055),
    ("down", 0.9, 0.045),
])
def test_adaptive_damping_module_factor_matrix(name, factor, expected):
    """The per-module damping factor scales the base percdamp on an identity Hessian."""
    cfg = QuantizeConfig(
        method="gptq",
        adaptive_damping={"enabled": True, "base_percdamp": 0.05, "min": 0.02, "max": 0.08},
    )
    m = nn.Linear(64, 64, bias=False)
    q = GPTQ(m, qcfg=cfg)
    q.name = name
    H = torch.eye(64, dtype=torch.float32)
    diag = H.diagonal().clone()
    damp = q._resolve_initial_damp(H, diag, diag.mean())
    assert damp == pytest.approx(expected, abs=0.005)


def test_adaptive_damping_group_size_prior_is_monotonic():
    """Larger groups receive a smaller error-scale update (more damping)."""
    group_sizes = [32, 64, 128, 256, 512]
    beta = 0.25
    reference = 128
    scales = [
        GPTQ._group_error_scale(1.0, (g / reference) ** beta)
        for g in group_sizes
    ]
    for i in range(len(scales) - 1):
        assert scales[i] > scales[i + 1]


def test_adaptive_damping_feedback_edge_cases():
    """The feedback controller handles zero, infinite, extreme small and large losses."""
    _, feedback = GPTQ._update_group_feedback(0.0, 1.0, 0.9, 0.1, 0.9, 1.1)
    assert feedback == 1.0

    _, feedback = GPTQ._update_group_feedback(float("inf"), 1.0, 0.9, 0.1, 0.9, 1.1)
    assert feedback == pytest.approx(0.9)

    _, feedback = GPTQ._update_group_feedback(1e-20, 1.0, 0.9, 0.1, 0.9, 1.1)
    assert feedback == pytest.approx(1.1)

    _, feedback = GPTQ._update_group_feedback(1e20, 1.0, 0.9, 0.1, 0.9, 1.1)
    assert feedback == pytest.approx(0.9)

    target, feedback = GPTQ._update_group_feedback(5.0, None, 0.9, 0.1, 0.9, 1.1)
    assert target == 5.0
    assert feedback == 1.0


def test_compute_group_loss_unweighted_and_hessian_weighted():
    """Hessian weighting turns L_g from E^T E into E^T diag(H) E."""
    err = torch.tensor([[1.0, 2.0]])
    h_diag = torch.tensor([1.0, 10.0])
    assert GPTQ._compute_group_loss(err, None, False) == pytest.approx(5.0)
    assert GPTQ._compute_group_loss(err, h_diag, True) == pytest.approx(41.0)
    # Missing Hessian diagonal with weighting requested falls back to the unweighted norm.
    assert GPTQ._compute_group_loss(err, None, True) == pytest.approx(5.0)


def test_adaptive_damping_raw_residual_measurement_math():
    """Dividing the residual by the applied scale before loss recovers the raw GPTQ error."""
    raw_err = torch.tensor([[1.0, 2.0]])
    scale = 2.0
    measured = GPTQ._compute_group_loss(raw_err / scale, None, False)
    expected_raw = float(torch.sum(raw_err ** 2).item())
    assert measured == pytest.approx(expected_raw / (scale ** 2))


@pytest.mark.parametrize("actorder", ["none", "desc_act", "act_group_aware"])
@pytest.mark.parametrize("device", [
    "cpu",
    pytest.param(
        "cuda",
        marks=[
            pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
            pytest.mark.gpu,
        ],
    ),
])
def test_adaptive_damping_activation_ordering_with_online_feedback(device, actorder):
    """Online group feedback must remain finite with desc_act and act_group_aware permutations."""
    adaptive = {
        "enabled": True,
        "group_error_enabled": True,
        "group_size_prior_enabled": True,
        "min": 0.02,
        "max": 0.08,
    }
    kwargs = {"method": "gptq", "bits": 4, "group_size": 8, "adaptive_damping": adaptive}
    if actorder == "desc_act":
        kwargs["desc_act"] = True
    elif actorder == "act_group_aware":
        kwargs["act_group_aware"] = True

    cfg = QuantizeConfig(**kwargs)
    torch.manual_seed(0)
    dtype = torch.float16 if device == "cuda" else torch.float32
    layer = nn.Linear(16, 12, bias=False, dtype=dtype).eval().to(device)
    gptq = GPTQ(layer, qcfg=cfg)
    gptq.quantizer.configure(perchannel=True)
    calibration = torch.randn(8, 16, device=device, dtype=dtype)
    gptq.add_batch(calibration, None)
    gptq.finalize_hessian()
    Q, scale, zero, g_idx, duration, avg_loss, damp, nsamples = gptq.quantize(blocksize=8)
    assert Q.shape == layer.weight.shape
    assert torch.isfinite(Q).all()
    assert torch.isfinite(scale).all()
    assert torch.isfinite(zero).all()
    assert torch.isfinite(torch.tensor(avg_loss))
    assert 0 < damp < 1


def test_adaptive_damping_shared_hessian_cache_key_includes_damp():
    """The shared Hessian inverse cache key distinguishes different resolved damping values."""
    cfg = QuantizeConfig(method="gptq")
    m = nn.Linear(64, 64, bias=False)
    q = GPTQ(m, qcfg=cfg)
    q._shared_hessian_inverse_key = ("shared_group_0",)
    H = torch.eye(64, dtype=torch.float32)
    key_a = q._shared_hessian_inverse_cache_key(H, damp=0.05)
    key_b = q._shared_hessian_inverse_cache_key(H, damp=0.10)
    assert key_a is not None and key_b is not None
    assert key_a != key_b


@pytest.mark.parametrize("bits", [3, 4, 8])
@pytest.mark.parametrize("group_size", [8, 16])
@pytest.mark.parametrize("device", [
    "cpu",
    pytest.param(
        "cuda",
        marks=[
            pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
            pytest.mark.gpu,
        ],
    ),
])
def test_adaptive_damping_quantize_integration_matrix(device, bits, group_size):
    """Adaptive damping must produce finite quantized weights across bits and group sizes."""
    cfg = QuantizeConfig(
        method="gptq",
        bits=bits,
        group_size=group_size,
        adaptive_damping={"enabled": True, "min": 0.001, "max": 0.2},
    )
    torch.manual_seed(0)
    dtype = torch.float16 if device == "cuda" else torch.float32
    layer = nn.Linear(16, 12, bias=False, dtype=dtype).eval().to(device)
    gptq = GPTQ(layer, qcfg=cfg)
    gptq.quantizer.configure(perchannel=True)
    calibration = torch.randn(8, 16, device=device, dtype=dtype)
    gptq.add_batch(calibration, None)
    gptq.finalize_hessian()
    Q, scale, zero, g_idx, duration, avg_loss, damp, nsamples = gptq.quantize(blocksize=group_size)
    assert Q.shape == layer.weight.shape
    assert torch.isfinite(Q).all()
    assert torch.isfinite(scale).all()
    assert torch.isfinite(zero).all()
    assert 0 < damp < 1


def test_adaptive_damping_dict_routes_ema_decay_to_adaptive_config():
    """A plain dict containing only `group_error_ema_decay` must select AdaptiveDampingConfig."""
    cfg = QuantizeConfig(method="gptq", adaptive_damping={"group_error_ema_decay": 0.8})
    assert isinstance(cfg.adaptive_damping, AdaptiveDampingConfig)
    assert cfg.adaptive_damping.group_error_ema_decay == pytest.approx(0.8)


def test_adaptive_damping_damp_auto_increment_sync():
    """Legacy `damp_auto_increment` is propagated into the default static damping config."""
    cfg = QuantizeConfig(method="gptq", damp_auto_increment=0.007)
    assert isinstance(cfg.adaptive_damping, DampConfig)
    assert cfg.adaptive_damping.step == pytest.approx(0.007)
    assert cfg.damp_auto_increment == pytest.approx(0.007)


def test_adaptive_damping_explicit_step_not_overwritten():
    """An explicit AdaptiveDampingConfig.step must not be overwritten by legacy damp_auto_increment."""
    cfg = QuantizeConfig(
        method="gptq",
        damp_auto_increment=0.007,
        adaptive_damping=AdaptiveDampingConfig(step=0.003),
    )
    assert cfg.adaptive_damping.step == pytest.approx(0.003)
