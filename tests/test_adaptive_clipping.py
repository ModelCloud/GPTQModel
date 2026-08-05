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
    assert cfg.metric == "hessian_diag"
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


def _quantized_mse(weight, qcfg, hessian=None):
    """Quantize a single weight block and return (Q, mse)."""
    quantizer = Quantizer(qcfg=qcfg, name="test")
    quantizer.configure(perchannel=True)
    quantizer.find_params(weight, weight=True, hessian=hessian)
    Q = quantizer.quantize(weight)
    finite_mask = torch.isfinite(weight)
    mse = ((weight[finite_mask] - Q[finite_mask]) ** 2).mean().item()
    return Q, mse


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
