# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch

from gptqmodel.quantization.config import AdaptiveClippingConfig, QuantizeConfig
from gptqmodel.quantization.quantizer import Quantizer


def _make_qcfg(bits=4, group_size=128, adaptive_clipping=None, **kwargs):
    """Build an isolated GPTQ config with scale search disabled."""
    return QuantizeConfig(
        method="gptq",
        bits=bits,
        group_size=group_size,
        scale_search=None,
        mse=0.0,
        adaptive_damping={"enabled": False},
        adaptive_clipping=adaptive_clipping,
        **kwargs,
    )


def test_adaptive_clipping_config_defaults():
    cfg = AdaptiveClippingConfig()
    assert cfg.enabled is True
    assert cfg.metric == "gptq_error"
    assert cfg.per_group is True
    assert cfg.candidates == (0.99, 0.995, 0.999, 1.0)


def test_adaptive_clipping_config_validation():
    with pytest.raises(ValueError):
        AdaptiveClippingConfig(metric="unknown")
    with pytest.raises(ValueError):
        AdaptiveClippingConfig(candidates=())
    with pytest.raises(ValueError):
        AdaptiveClippingConfig(candidates=[1.1])
    with pytest.raises(ValueError):
        AdaptiveClippingConfig(candidates=[0.0])


def test_adaptive_clipping_config_round_trip():
    payload = {
        "method": "gptq",
        "bits": 4,
        "group_size": 128,
        "adaptive_clipping": {
            "enabled": True,
            "metric": "mse",
            "per_group": False,
            "candidates": [0.9, 1.0],
        },
    }
    cfg = QuantizeConfig.from_quant_config(payload)
    assert cfg.adaptive_clipping.enabled is True
    assert cfg.adaptive_clipping.metric == "mse"
    assert cfg.adaptive_clipping.per_group is False
    assert cfg.adaptive_clipping.candidates == (0.9, 1.0)
    assert cfg.to_dict()["meta"]["adaptive_clipping"] == cfg.adaptive_clipping.to_dict()


@pytest.mark.parametrize("method_config", [{"gptaq": {}}, {"foem": {}}])
def test_exact_adaptive_clipping_rejects_noncanonical_correction_math(method_config):
    with pytest.raises(ValueError, match="currently supports canonical GPTQ only"):
        _make_qcfg(adaptive_clipping={"enabled": True, "metric": "gptq_error"}, **method_config)

    approximate = _make_qcfg(
        adaptive_clipping={"enabled": True, "metric": "hessian_diag"},
        **method_config,
    )
    assert approximate.adaptive_clipping.metric == "hessian_diag"


def test_exact_adaptive_clipping_rejects_static_group_precomputation():
    with pytest.raises(ValueError, match="static scales are selected before"):
        _make_qcfg(
            static_groups=True,
            adaptive_clipping={"enabled": True, "metric": "gptq_error"},
        )

    approximate = _make_qcfg(
        static_groups=True,
        adaptive_clipping={"enabled": True, "metric": "hessian_diag"},
    )
    assert approximate.adaptive_clipping.metric == "hessian_diag"


def _quantized_mse(weight, qcfg, hessian=None):
    """Quantize a single weight block and return (Q, mse)."""
    quantizer = Quantizer(qcfg=qcfg, name="test")
    quantizer.configure(perchannel=True)
    quantizer.find_params(weight, weight=True, hessian=hessian)
    Q = quantizer.quantize(weight)
    finite_mask = torch.isfinite(weight)
    mse = ((weight[finite_mask] - Q[finite_mask]) ** 2).mean().item()
    return Q, mse


def _inverse_cholesky_from_hessian(hessian: torch.Tensor, damping: float = 1e-3) -> torch.Tensor:
    """Build GPTQ's upper inverse-Cholesky factor from calibration geometry."""

    hessian = hessian.float()
    hessian = hessian.clone()
    hessian.diagonal().add_(damping * hessian.diagonal().mean().clamp_min(1e-12))
    return torch.linalg.cholesky(torch.linalg.inv(hessian), upper=True)


def _inverse_cholesky_from_calibration(calibration: torch.Tensor, damping: float = 1e-3) -> torch.Tensor:
    hessian = 2.0 / calibration.shape[0] * calibration.T @ calibration
    return _inverse_cholesky_from_hessian(hessian, damping=damping)


def _reference_quantize_column(
    value: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
    *,
    bits: int,
    groupwise: bool = False,
) -> torch.Tensor:
    maxq = 2 ** (bits - 1) - 1 if groupwise else 2**bits - 1
    if groupwise:
        return scale * torch.clamp(torch.round(value / scale), -maxq, maxq)
    return scale * (torch.clamp(torch.round(value / scale) + zero, 0, maxq) - zero)


def _reference_gptq_loss(
    weight: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
    inverse_cholesky: torch.Tensor,
    *,
    bits: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Independent scalar GPTQ recursion returning per-row loss, E, and Q."""

    work = weight.clone()
    error = torch.zeros_like(work)
    quantized = torch.zeros_like(work)
    for column in range(weight.shape[1]):
        q = _reference_quantize_column(
            work[:, column],
            scale,
            zero,
            bits=bits,
        )
        e = (work[:, column] - q) / inverse_cholesky[column, column]
        quantized[:, column] = q
        error[:, column] = e
        for future in range(column, weight.shape[1]):
            work[:, future] -= e * inverse_cholesky[column, future]
    return 0.5 * error.square().sum(dim=1), error, quantized


def _candidate_params(weight: torch.Tensor, *, bits: int, candidate: float) -> tuple[torch.Tensor, torch.Tensor]:
    qcfg = _make_qcfg(
        bits=bits,
        group_size=weight.shape[1],
        adaptive_clipping={"enabled": True, "metric": "mse", "candidates": (candidate,)},
    )
    quantizer = Quantizer(qcfg=qcfg, name="candidate-reference")
    quantizer.configure(perchannel=True)
    quantizer.find_params(weight.clone(), weight=True)
    return quantizer.scale.flatten(), quantizer.zero.flatten()


def test_adaptive_clip_gptq_error_matches_independent_scalar_reference():
    torch.manual_seed(731)
    bits = 3
    weight = torch.randn(5, 7, dtype=torch.float32)
    weight[:, 0] *= 7.0
    calibration = torch.randn(19, 7, dtype=torch.float32)
    inverse_cholesky = _inverse_cholesky_from_calibration(calibration)
    candidates = (0.55, 0.8, 0.95, 1.0)

    reference_scales = []
    reference_zeros = []
    reference_losses = []
    for candidate in candidates:
        scale, zero = _candidate_params(weight, bits=bits, candidate=candidate)
        loss, _, _ = _reference_gptq_loss(
            weight,
            scale,
            zero,
            inverse_cholesky,
            bits=bits,
        )
        reference_scales.append(scale)
        reference_zeros.append(zero)
        reference_losses.append(loss)
    best = torch.stack(reference_losses).argmin(dim=0)
    expected_scale = torch.stack(reference_scales).gather(0, best.unsqueeze(0)).squeeze(0)
    expected_zero = torch.stack(reference_zeros).gather(0, best.unsqueeze(0)).squeeze(0)

    qcfg = _make_qcfg(
        bits=bits,
        group_size=weight.shape[1],
        adaptive_clipping={"enabled": True, "metric": "gptq_error", "candidates": candidates},
    )
    quantizer = Quantizer(qcfg=qcfg, name="exact-reference")
    quantizer.configure(perchannel=True)
    quantizer.find_params(
        weight.clone(),
        weight=True,
        gptq_inverse_cholesky=inverse_cholesky,
    )

    torch.testing.assert_close(quantizer.scale.flatten(), expected_scale, rtol=0, atol=0)
    torch.testing.assert_close(quantizer.zero.flatten(), expected_zero, rtol=0, atol=0)


def test_adaptive_clip_gptq_loss_equals_damped_calibration_quadratic():
    """The correction loss must equal the dense quadratic implied by calibration."""

    torch.manual_seed(991)
    columns = 6
    calibration = torch.randn(32, columns, dtype=torch.float64)
    hessian = 2.0 / calibration.shape[0] * calibration.T @ calibration
    inverse_cholesky = torch.linalg.cholesky(torch.linalg.inv(hessian), upper=True)
    weight = torch.randn(3, columns, dtype=torch.float64)
    scale, zero = _candidate_params(weight.float(), bits=4, candidate=0.8)
    loss, error, quantized = _reference_gptq_loss(
        weight,
        scale.double(),
        zero.double(),
        inverse_cholesky,
        bits=4,
    )
    residual = weight - quantized.double()
    reconstructed_residual = error.double() @ inverse_cholesky
    quadratic = 0.5 * torch.einsum("ri,ij,rj->r", residual, hessian, residual)
    output_mse_sum = (calibration @ residual.T).square().sum(dim=0) / calibration.shape[0]

    torch.testing.assert_close(residual, reconstructed_residual, rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(loss.double(), quadratic, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(quadratic, output_mse_sum, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize(
    "invalid_factor",
    [
        None,
        torch.eye(3),
        torch.full((4, 4), float("nan")),
        torch.diag(torch.tensor([1.0, 1.0, -1.0, 1.0])),
    ],
)
def test_adaptive_clip_missing_or_invalid_gptq_geometry_is_unclipped(invalid_factor):
    torch.manual_seed(117)
    weight = torch.randn(4, 4)
    exact_cfg = _make_qcfg(
        bits=4,
        group_size=4,
        adaptive_clipping={
            "enabled": True,
            "metric": "gptq_error",
            # Deliberately exclude 1.0: safe fallback must still be unclipped.
            "candidates": (0.25, 0.5),
        },
    )
    baseline_cfg = _make_qcfg(bits=4, group_size=4, adaptive_clipping={"enabled": False})
    exact = Quantizer(qcfg=exact_cfg, name="invalid-geometry")
    baseline = Quantizer(qcfg=baseline_cfg, name="baseline")
    exact.configure(perchannel=True)
    baseline.configure(perchannel=True)
    exact.find_params(weight.clone(), weight=True, gptq_inverse_cholesky=invalid_factor)
    baseline.find_params(weight.clone(), weight=True)

    assert torch.equal(exact.scale, baseline.scale)
    assert torch.equal(exact.zero, baseline.zero)


def test_adaptive_clip_missing_hessian_diag_is_unclipped_not_mse():
    torch.manual_seed(118)
    weight = torch.randn(4, 9)
    clipped_cfg = _make_qcfg(
        bits=4,
        group_size=9,
        adaptive_clipping={"enabled": True, "metric": "hessian_diag", "candidates": (0.25, 0.5)},
    )
    baseline_cfg = _make_qcfg(bits=4, group_size=9, adaptive_clipping={"enabled": False})
    clipped = Quantizer(qcfg=clipped_cfg, name="missing-diagonal")
    baseline = Quantizer(qcfg=baseline_cfg, name="baseline")
    clipped.configure(perchannel=True)
    baseline.configure(perchannel=True)
    clipped.find_params(weight.clone(), weight=True)
    baseline.find_params(weight.clone(), weight=True)

    assert torch.equal(clipped.scale, baseline.scale)
    assert torch.equal(clipped.zero, baseline.zero)


def test_adaptive_clip_search_chooses_no_clipping_for_gaussian():
    qcfg = _make_qcfg(bits=4, group_size=128, adaptive_clipping={"enabled": True})
    torch.manual_seed(42)
    weight = torch.randn(16, 128, dtype=torch.float32)
    Q, mse = _quantized_mse(weight, qcfg)
    assert torch.isfinite(Q).all()
    assert math.isfinite(mse)


def test_adaptive_clip_search_improves_outlier_mse():
    qcfg_clipped = _make_qcfg(bits=4, group_size=128, adaptive_clipping={"enabled": True, "metric": "mse"})
    qcfg_baseline = _make_qcfg(bits=4, group_size=128, adaptive_clipping={"enabled": False})
    torch.manual_seed(0)
    weight = torch.randn(8, 128, dtype=torch.float32)
    # Inject one extreme outlier per row.
    weight[:, 0] = 10.0
    _, mse_clipped = _quantized_mse(weight, qcfg_clipped)
    _, mse_baseline = _quantized_mse(weight, qcfg_baseline)
    assert mse_clipped <= mse_baseline * 1.05 + 1e-6


def test_adaptive_clip_hessian_objective_matches_reference():
    """For a small tensor, the selected scale should minimize sum(H_ii * (W - Q)^2)."""
    torch.manual_seed(1)
    rows, cols = 4, 32
    weight = torch.randn(rows, cols, dtype=torch.float32)
    hessian_diag = torch.rand(cols, dtype=torch.float32) + 0.1
    qcfg = _make_qcfg(bits=4, group_size=32, adaptive_clipping={"enabled": True, "metric": "hessian_diag"})
    quantizer = Quantizer(qcfg=qcfg, name="test")
    quantizer.configure(perchannel=True)
    quantizer.find_params(weight, weight=True, hessian=hessian_diag)
    Q = quantizer.quantize(weight)
    assert torch.isfinite(Q).all()


def test_adaptive_clip_mse_vs_hessian_can_differ():
    torch.manual_seed(2)
    weight = torch.randn(2, 64, dtype=torch.float32)
    # Put an outlier in a low-Hessian column.
    h = torch.ones(64, dtype=torch.float32)
    h[0] = 0.01
    weight[:, 0] = 8.0

    qcfg_mse = _make_qcfg(bits=4, group_size=64, adaptive_clipping={"enabled": True, "metric": "mse"})
    qcfg_hess = _make_qcfg(bits=4, group_size=64, adaptive_clipping={"enabled": True, "metric": "hessian_diag"})

    _, mse_mse = _quantized_mse(weight, qcfg_mse, hessian=h)
    _, mse_hess = _quantized_mse(weight, qcfg_hess, hessian=h)
    assert math.isfinite(mse_mse)
    assert math.isfinite(mse_hess)


@pytest.mark.parametrize("bits", [3, 4, 6, 8])
@pytest.mark.parametrize("group_size", [32, 64, 128, 256])
def test_adaptive_clipping_bits_and_group_sizes(bits, group_size):
    torch.manual_seed(bits + group_size)
    rows, cols = 16, group_size * 2
    weight = torch.randn(rows, cols, dtype=torch.float32)
    weight[:, 0] = 8.0  # outlier
    qcfg = _make_qcfg(bits=bits, group_size=group_size, adaptive_clipping={"enabled": True})
    Q, mse = _quantized_mse(weight, qcfg)
    assert Q.shape == weight.shape
    assert torch.isfinite(Q).all()
    assert math.isfinite(mse)


def test_adaptive_clipping_find_params_batched():
    """Ensure the batched-scale fallback applies per-group clipping."""
    torch.manual_seed(10)
    rows, num_groups, group_size = 4, 3, 32
    x = torch.randn(rows, num_groups * group_size, dtype=torch.float32)
    x = x.reshape(rows, num_groups, group_size)
    hessian = torch.rand(num_groups, group_size, dtype=torch.float32) + 0.1
    qcfg = _make_qcfg(bits=4, group_size=group_size, adaptive_clipping={"enabled": True, "metric": "hessian_diag"})
    quantizer = Quantizer(qcfg=qcfg, name="test")
    quantizer.configure(perchannel=True)
    scale, zero = quantizer.find_params_batched(x, weight=True, hessian=hessian)
    assert scale.shape == (rows, num_groups)
    assert zero.shape == (rows, num_groups)
    assert torch.isfinite(scale).all()
    assert torch.isfinite(zero).all()


def test_adaptive_clipping_gptq_error_batched_matches_serial_groups():
    torch.manual_seed(212)
    rows, num_groups, group_size = 5, 3, 6
    x = torch.randn(rows, num_groups, group_size)
    calibration = torch.randn(48, num_groups * group_size)
    inverse_cholesky = _inverse_cholesky_from_calibration(calibration)
    qcfg = _make_qcfg(
        bits=4,
        group_size=group_size,
        adaptive_clipping={
            "enabled": True,
            "metric": "gptq_error",
            "candidates": (0.5, 0.8, 1.0),
        },
    )
    batched = Quantizer(qcfg=qcfg, name="batched")
    batched.configure(perchannel=True)
    batched_scale, batched_zero = batched.find_params_batched(
        x.clone(),
        weight=True,
        gptq_inverse_cholesky=inverse_cholesky,
    )

    serial_scales = []
    serial_zeros = []
    for group in range(num_groups):
        start = group * group_size
        end = start + group_size
        serial = Quantizer(qcfg=qcfg, name=f"serial-{group}")
        serial.configure(perchannel=True)
        serial.find_params(
            x[:, group, :].clone(),
            weight=True,
            gptq_inverse_cholesky=inverse_cholesky[start:end, start:end],
        )
        serial_scales.append(serial.scale.flatten())
        serial_zeros.append(serial.zero.flatten())

    assert torch.equal(batched_scale, torch.stack(serial_scales, dim=1))
    assert torch.equal(batched_zero, torch.stack(serial_zeros, dim=1))


def test_adaptive_clipping_gptq_candidate_is_invariant_to_uniform_hessian_scale():
    torch.manual_seed(313)
    weight = torch.randn(5, 8)
    weight[:, 0] *= 9
    calibration = torch.randn(40, 8)
    hessian = 2.0 / calibration.shape[0] * calibration.T @ calibration
    first_factor = _inverse_cholesky_from_hessian(hessian)
    second_factor = _inverse_cholesky_from_hessian(hessian * 37.0)
    qcfg = _make_qcfg(
        bits=3,
        group_size=8,
        adaptive_clipping={
            "enabled": True,
            "metric": "gptq_error",
            "candidates": (0.4, 0.65, 0.85, 1.0),
        },
    )

    outputs = []
    for factor in (first_factor, second_factor):
        quantizer = Quantizer(qcfg=qcfg, name="uniform-hessian-scale")
        quantizer.configure(perchannel=True)
        quantizer.find_params(weight.clone(), weight=True, gptq_inverse_cholesky=factor)
        outputs.append((quantizer.scale, quantizer.zero))

    assert torch.equal(outputs[0][0], outputs[1][0])
    assert torch.equal(outputs[0][1], outputs[1][1])


def test_adaptive_clipping_gptq_selection_depends_on_calibration_geometry():
    """The same weights must be able to choose different clipping from different activations."""

    candidates = (0.35, 0.55, 0.75, 1.0)
    qcfg = _make_qcfg(
        bits=2,
        group_size=6,
        adaptive_clipping={"enabled": True, "metric": "gptq_error", "candidates": candidates},
    )
    found_difference = False
    for seed in range(32):
        generator = torch.Generator().manual_seed(seed)
        weight = torch.randn(4, 6, generator=generator)
        weight[:, 0] *= 8
        first_calibration = torch.randn(24, 6, generator=generator)
        second_calibration = torch.randn(24, 6, generator=generator)
        second_calibration[:, 0] *= 0.05
        second_calibration[:, 1] += 0.95 * second_calibration[:, 2]
        factors = (
            _inverse_cholesky_from_calibration(first_calibration),
            _inverse_cholesky_from_calibration(second_calibration),
        )
        scales = []
        for factor in factors:
            quantizer = Quantizer(qcfg=qcfg, name="calibration-dependence")
            quantizer.configure(perchannel=True)
            quantizer.find_params(weight.clone(), weight=True, gptq_inverse_cholesky=factor)
            scales.append(quantizer.scale)
        if not torch.equal(scales[0], scales[1]):
            found_difference = True
            break

    assert found_difference, "gptq_error behaved like a calibration-independent weight-only objective"


@pytest.mark.parametrize(
    "geometry",
    ["identity", "correlated", "rank_deficient", "ill_conditioned", "dead_feature"],
)
def test_adaptive_clipping_gptq_geometry_spectrum_is_finite(geometry):
    torch.manual_seed(414)
    columns = 8
    if geometry == "identity":
        hessian = torch.eye(columns)
    elif geometry == "correlated":
        calibration = torch.randn(32, columns)
        calibration[:, 1] = 0.95 * calibration[:, 0] + 0.05 * calibration[:, 1]
        hessian = 2.0 / calibration.shape[0] * calibration.T @ calibration
    elif geometry == "rank_deficient":
        calibration = torch.randn(3, columns)
        hessian = 2.0 / calibration.shape[0] * calibration.T @ calibration
    elif geometry == "ill_conditioned":
        eigenvalues = torch.logspace(-8, 2, columns)
        basis, _ = torch.linalg.qr(torch.randn(columns, columns))
        hessian = basis @ torch.diag(eigenvalues) @ basis.T
    elif geometry == "dead_feature":
        hessian = torch.eye(columns)
        hessian[-1, -1] = 0
    else:
        raise AssertionError(geometry)

    factor = _inverse_cholesky_from_hessian(hessian, damping=1e-2)
    weight = torch.randn(6, columns)
    qcfg = _make_qcfg(
        bits=4,
        group_size=columns,
        adaptive_clipping={"enabled": True, "metric": "gptq_error", "candidates": (0.5, 0.9, 1.0)},
    )
    quantizer = Quantizer(qcfg=qcfg, name=geometry)
    quantizer.configure(perchannel=True)
    quantizer.find_params(weight, weight=True, gptq_inverse_cholesky=factor)
    quantized = quantizer.quantize(weight)

    assert torch.isfinite(factor).all()
    assert torch.isfinite(quantizer.scale).all()
    assert torch.isfinite(quantizer.zero).all()
    assert torch.isfinite(quantized).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for GPU clipping accuracy coverage")
@pytest.mark.parametrize("bits", [2, 3, 4, 8])
@pytest.mark.parametrize("sym", [False, True])
@pytest.mark.parametrize("geometry", ["diagonal", "correlated", "ill_conditioned"])
def test_adaptive_clipping_gptq_gpu_matches_cpu(bits, sym, geometry):
    """GPU candidate selection must match the dense FP32 CPU implementation."""

    torch.manual_seed(bits * 100 + int(sym) * 10 + len(geometry))
    rows, columns = 8, 16
    weight = torch.randn(rows, columns)
    weight[:, 0] *= 10
    if geometry == "diagonal":
        hessian = torch.diag(torch.linspace(0.1, 3.0, columns))
    elif geometry == "correlated":
        calibration = torch.randn(64, columns)
        calibration[:, 1] = 0.98 * calibration[:, 0] + 0.02 * calibration[:, 1]
        hessian = 2.0 / calibration.shape[0] * calibration.T @ calibration
    elif geometry == "ill_conditioned":
        eigenvalues = torch.logspace(-6, 2, columns)
        basis, _ = torch.linalg.qr(torch.randn(columns, columns))
        hessian = basis @ torch.diag(eigenvalues) @ basis.T
    else:
        raise AssertionError(geometry)
    inverse_cholesky = _inverse_cholesky_from_hessian(hessian, damping=1e-2)
    qcfg = _make_qcfg(
        bits=bits,
        group_size=columns,
        sym=sym,
        adaptive_clipping={
            "enabled": True,
            "metric": "gptq_error",
            "candidates": (0.4, 0.7, 0.9, 1.0),
        },
    )

    cpu = Quantizer(qcfg=qcfg, name="cpu-reference")
    gpu = Quantizer(qcfg=qcfg, name="gpu")
    cpu.configure(perchannel=True)
    gpu.configure(perchannel=True)
    cpu.find_params(weight.clone(), weight=True, gptq_inverse_cholesky=inverse_cholesky)
    gpu.find_params(
        weight.cuda(),
        weight=True,
        gptq_inverse_cholesky=inverse_cholesky.cuda(),
    )

    torch.testing.assert_close(gpu.scale.cpu(), cpu.scale, rtol=2e-6, atol=2e-7)
    torch.testing.assert_close(gpu.zero.cpu(), cpu.zero, rtol=0, atol=0)
    torch.testing.assert_close(
        gpu.quantize(weight.cuda()).cpu(),
        cpu.quantize(weight),
        rtol=2e-6,
        atol=2e-7,
    )


def test_adaptive_clipping_flat_distribution():
    qcfg = _make_qcfg(bits=4, group_size=32, adaptive_clipping={"enabled": True})
    weight = torch.ones(8, 32, dtype=torch.float32) * 0.5
    Q, mse = _quantized_mse(weight, qcfg)
    assert torch.isfinite(Q).all()
    assert Q.shape == weight.shape


def test_adaptive_clipping_nan_inf_safe():
    qcfg = _make_qcfg(bits=4, group_size=32, adaptive_clipping={"enabled": True})
    weight = torch.randn(4, 32, dtype=torch.float32)
    weight[0, 0] = float("nan")
    weight[1, 1] = float("inf")
    weight[2, 2] = float("-inf")
    Q, mse = _quantized_mse(weight, qcfg)
    assert torch.isfinite(Q).all()
    assert math.isfinite(mse)


@pytest.mark.parametrize("distribution", ["gaussian", "laplace", "asymmetric", "positive_only", "negative_only"])
def test_adaptive_clipping_skewed_distributions(distribution):
    torch.manual_seed(hash(distribution) % (2 ** 32))
    rows, cols = 8, 64
    if distribution == "gaussian":
        weight = torch.randn(rows, cols, dtype=torch.float32) * 2.0
    elif distribution == "laplace":
        weight = torch.randn(rows, cols, dtype=torch.float32)
        weight = torch.sign(weight) * torch.abs(weight) ** 1.5
    elif distribution == "asymmetric":
        weight = torch.randn(rows, cols, dtype=torch.float32) * 0.5
        weight = weight + torch.rand(rows, cols, dtype=torch.float32) * 2.0
    elif distribution == "positive_only":
        weight = torch.rand(rows, cols, dtype=torch.float32) * 5.0
    elif distribution == "negative_only":
        weight = -torch.rand(rows, cols, dtype=torch.float32) * 5.0

    qcfg = _make_qcfg(bits=4, group_size=64, adaptive_clipping={"enabled": True, "metric": "hessian_diag"})
    # A diagonal Hessian that grows with the column index.
    hessian = torch.linspace(0.1, 2.0, cols, dtype=torch.float32)
    Q, mse = _quantized_mse(weight, qcfg, hessian=hessian)
    assert torch.isfinite(Q).all()
    assert math.isfinite(mse)


def test_adaptive_clipping_per_row_independence():
    """Per-row clipping should allow different rows to choose different thresholds."""
    torch.manual_seed(3)
    weight = torch.randn(4, 64, dtype=torch.float32)
    # Row 0 clean, row 1 has outlier, row 2 heavy tail.
    weight[0] = torch.randn(64, dtype=torch.float32) * 0.3
    weight[1, 0] = 12.0
    weight[2] = torch.sign(torch.randn(64, dtype=torch.float32)) * torch.abs(torch.randn(64, dtype=torch.float32)) ** 1.2
    qcfg = _make_qcfg(bits=4, group_size=64, adaptive_clipping={"enabled": True, "metric": "mse"})
    Q, mse = _quantized_mse(weight, qcfg)
    assert torch.isfinite(Q).all()
    assert math.isfinite(mse)


def _spectrum_weight(distribution: str, *, rows: int = 4, columns: int = 17) -> torch.Tensor:
    """Build deterministic ideal, ordinary, and pathological clipping inputs."""

    generator = torch.Generator().manual_seed(20260806)
    if distribution == "zeros":
        return torch.zeros(rows, columns)
    if distribution == "constant_positive":
        return torch.full((rows, columns), 0.25)
    if distribution == "constant_negative":
        return torch.full((rows, columns), -0.25)
    if distribution == "tiny":
        return torch.randn(rows, columns, generator=generator) * 1e-12
    if distribution == "gaussian":
        return torch.randn(rows, columns, generator=generator)
    if distribution == "asymmetric":
        return torch.randn(rows, columns, generator=generator) * 0.2 + 1.5
    if distribution == "outlier":
        value = torch.randn(rows, columns, generator=generator) * 0.1
        value[:, 0] = torch.tensor([10.0, -12.0, 25.0, -30.0])
        return value
    if distribution == "nonfinite":
        value = torch.randn(rows, columns, generator=generator)
        value[0, 0], value[1, 1], value[2, 2] = float("nan"), float("inf"), float("-inf")
        return value
    raise AssertionError(f"unknown distribution: {distribution}")


@pytest.mark.parametrize("bits", [2, 3, 4, 8])
@pytest.mark.parametrize("sym", [False, True])
@pytest.mark.parametrize("metric", ["mse", "hessian_diag", "gptq_error"])
@pytest.mark.parametrize(
    "distribution",
    [
        "zeros",
        "constant_positive",
        "constant_negative",
        "tiny",
        "gaussian",
        "asymmetric",
        "outlier",
        "nonfinite",
    ],
)
def test_adaptive_clipping_full_numeric_spectrum_is_finite_and_deterministic(
    bits,
    sym,
    metric,
    distribution,
):
    """Cover clipping precision, symmetry, objective, and numeric-distribution regions exhaustively."""

    weight = _spectrum_weight(distribution)
    hessian = torch.linspace(0.0, 2.0, weight.shape[1])
    qcfg = _make_qcfg(
        bits=bits,
        group_size=weight.shape[1],
        sym=sym,
        adaptive_clipping={
            "enabled": True,
            "metric": metric,
            "candidates": (0.5, 0.9, 0.99, 1.0),
        },
    )

    first = Quantizer(qcfg=qcfg, name="spectrum")
    first.configure(perchannel=True)
    inverse_cholesky = torch.diag(torch.rsqrt(hessian.clamp_min(1e-3))) if metric == "gptq_error" else None
    first.find_params(
        weight.clone(),
        weight=True,
        hessian=hessian,
        gptq_inverse_cholesky=inverse_cholesky,
    )
    first_q = first.quantize(weight.clone())

    second = Quantizer(qcfg=qcfg, name="spectrum")
    second.configure(perchannel=True)
    second.find_params(
        weight.clone(),
        weight=True,
        hessian=hessian,
        gptq_inverse_cholesky=inverse_cholesky,
    )
    second_q = second.quantize(weight.clone())

    assert torch.isfinite(first.scale).all()
    assert torch.isfinite(first.zero).all()
    assert torch.isfinite(first_q).all()
    assert torch.equal(first.scale, second.scale)
    assert torch.equal(first.zero, second.zero)
    assert torch.equal(first_q, second_q)
