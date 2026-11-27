# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""ADMM-Q weight quantizer: config surface, grid properties and Torch oracle.

The quality oracle recomputes ``0.5 * Tr(dW H dW^T)`` directly from the returned
weights, so it is independent of the solver's internal bookkeeping.
"""

import math
from dataclasses import replace

import pytest
import torch
import torch.nn as nn

from gptqmodel.quantization.admmq import (
    ADMMQ,
    _Grid,
    admm_penalty_schedule,
    admm_rho_at,
    admmq_diagnostics_alert,
    pairswap_delta,
    projection_objective,
    refresh_accepts,
    summarize_admmq_diagnostics,
)
from gptqmodel.quantization.config import (
    FORMAT,
    METHOD,
    METHOD_SUPPORTED_BITS,
    QUANT_METHOD_FORMAT_MAPPING,
    ADMMQConfig,
    FOEMConfig,
    GPTAQConfig,
    GSQConfig,
    MSEConfig,
    QuantizeConfig,
    resolve_quant_format,
)
from gptqmodel.quantization.gptq import GPTQ
from gptqmodel.quantization.quantizer import FitLimits, Quantizer


IN_FEATURES = 16
OUT_FEATURES = 8
CALIBRATION_SAMPLES = 64


def _make_layer(weight: torch.Tensor) -> nn.Linear:
    layer = nn.Linear(weight.shape[1], weight.shape[0], bias=False, dtype=weight.dtype)
    layer.weight.data.copy_(weight)
    return layer.eval()


def _calibration() -> torch.Tensor:
    generator = torch.Generator().manual_seed(1234)
    x = torch.randn(CALIBRATION_SAMPLES, IN_FEATURES, generator=generator)
    # Channel outliers make the Hessian ill-conditioned; this is the regime the
    # paper's diagonal preconditioning targets.
    x[:, 0] *= 40.0
    x[:, 3] *= 8.0
    return x


def _run_solver(cls, layer, qcfg, x):
    solver = cls(layer, qcfg=qcfg)
    solver.quantizer.configure(perchannel=True)
    solver.add_batch(x, None)
    return solver, solver.quantize()


def _damped_hessian(x: torch.Tensor, damp: float) -> torch.Tensor:
    h = x.t() @ x
    return h + damp * torch.diagonal(h).mean() * torch.eye(h.shape[0], dtype=h.dtype, device=h.device)


def _reconstruction_error(weight: torch.Tensor, quantized: torch.Tensor, hessian: torch.Tensor) -> float:
    delta = weight - quantized
    return float(0.5 * torch.sum(delta * (delta @ hessian)))


def _reconstruction_error_rows(weight: torch.Tensor, quantized: torch.Tensor, hessian: torch.Tensor) -> torch.Tensor:
    delta = weight - quantized
    return 0.5 * torch.sum(delta * (delta @ hessian), dim=1)


def _rtn_baseline(weight: torch.Tensor, qcfg) -> torch.Tensor:
    quantizer = Quantizer(qcfg=qcfg)
    quantizer.configure(perchannel=True)
    quantizer.find_params(weight, weight=True)
    return quantizer.quantize(weight)


def _rtn_grid_projection(weight: torch.Tensor, qcfg, module: nn.Module) -> torch.Tensor:
    """Independent per-group RTN projection for the solver's fallback path."""

    quantizer = Quantizer(qcfg=qcfg)
    quantizer.configure(perchannel=True)
    group_size = qcfg.group_size
    if group_size is None or int(group_size) <= 0:
        ranges = [(0, weight.shape[1])]
    else:
        ranges = [
            (start, min(start + int(group_size), weight.shape[1]))
            for start in range(0, weight.shape[1], int(group_size))
        ]
    limits = FitLimits.for_module(module)
    projected = torch.empty_like(weight)
    for start, end in ranges:
        quantizer.fit_weight(weight[:, start:end], limits)
        projected[:, start:end] = quantizer.quantize(weight[:, start:end])
    return projected


# ---------------------------------------------------------------------------
# Config surface
# ---------------------------------------------------------------------------


def test_admmq_method_metadata_and_export():
    assert METHOD.ADMMQ.value == "admmq"
    assert METHOD_SUPPORTED_BITS[METHOD.ADMMQ] == (3, 4, 5, 6, 7, 8)
    assert QUANT_METHOD_FORMAT_MAPPING[METHOD.ADMMQ] == {
        FORMAT.GPTQ,
        FORMAT.GPTQ_V2,
        FORMAT.GPTQ_P,
        FORMAT.MARLIN,
        FORMAT.BITBLAS,
    }
    assert resolve_quant_format(FORMAT.GPTQ, METHOD.ADMMQ) == FORMAT.GPTQ
    assert resolve_quant_format(FORMAT.GPTQ_P, METHOD.ADMMQ) == FORMAT.GPTQ_P


def test_admmq_two_bit_is_rejected():
    with pytest.raises(ValueError, match="supported bit widths"):
        ADMMQConfig(bits=2, group_size=128)

    with pytest.raises(ValueError, match="supported bit widths"):
        QuantizeConfig(method="admmq", bits=2, group_size=128)


def test_admmq_dynamic_two_bit_is_rejected():
    with pytest.raises(ValueError, match="supported bit widths"):
        ADMMQConfig(bits=4, group_size=128, dynamic={"down_proj": {"bits": 2}})


def test_admmq_three_bit_is_accepted():
    assert ADMMQConfig(bits=3, group_size=128).bits == 3


def test_admmq_config_defaults():
    cfg = ADMMQConfig()
    assert cfg.method == METHOD.ADMMQ
    assert cfg.format == FORMAT.GPTQ
    assert cfg.export_quant_method() == METHOD.GPTQ
    assert cfg.max_iter == 300
    assert cfg.rho == pytest.approx(0.1)
    assert cfg.rho_update == pytest.approx(1.1)
    assert cfg.adaptive_penalty is True
    assert cfg.update_iter == 3
    assert cfg.switch_iter == 30
    assert cfg.damp_percent == pytest.approx(0.01)
    assert isinstance(cfg.mse, MSEConfig)
    assert cfg.mse.norm == pytest.approx(0.0)
    assert cfg.pairswap_mode == "always"
    assert cfg.static_groups is False
    assert cfg.act_group_aware is False


@pytest.mark.parametrize(
    ("bits", "expected"),
    [(3, 2.0), (4, 0.0), (5, 0.0), (6, 0.0), (7, 0.0), (8, 0.0)],
)
def test_admmq_mse_default_follows_bit_width(bits, expected):
    cfg = ADMMQConfig(bits=bits)
    assert cfg.mse.norm == pytest.approx(expected)


@pytest.mark.parametrize("value", [0, 0.0, {"norm": 0.0}, MSEConfig(), MSEConfig(norm=0.0)])
def test_admmq_explicit_mse_disable_wins_at_low_bits(value):
    assert ADMMQConfig(bits=3, mse=value).mse.norm == pytest.approx(0.0)


def test_admmq_explicit_mse_enable_wins_at_four_bits():
    assert ADMMQConfig(bits=4, mse=MSEConfig(norm=1.5)).mse.norm == pytest.approx(1.5)
    assert ADMMQConfig(bits=4, mse=2.0).mse.norm == pytest.approx(2.0)


def test_admmq_bit_adaptive_mse_survives_factory_and_roundtrip():
    factory = QuantizeConfig(method="admmq", bits=3, group_size=128)
    assert isinstance(factory, ADMMQConfig)
    assert factory.mse.norm == pytest.approx(2.0)

    payload = ADMMQConfig(bits=3, group_size=128).to_dict()
    assert payload["meta"]["mse"]["norm"] == pytest.approx(2.0)
    restored = QuantizeConfig(**payload)
    assert isinstance(restored, ADMMQConfig)
    assert restored.mse.norm == pytest.approx(2.0)

    off_payload = ADMMQConfig(bits=3, group_size=128, mse=0.0).to_dict()
    assert off_payload["meta"]["mse"]["norm"] == pytest.approx(0.0)
    restored_off = QuantizeConfig(**off_payload)
    assert restored_off.mse.norm == pytest.approx(0.0)


def test_admmq_three_bit_default_uses_the_mse_grid():
    torch.manual_seed(0)
    weight = torch.randn(OUT_FEATURES, IN_FEATURES) * 0.2
    x = _calibration()

    def solve(mse):
        layer = _make_layer(weight)
        cfg = ADMMQConfig(
            bits=3,
            group_size=-1,
            sym=True,
            max_iter=40,
            pairswap_mode="off",
            offload_to_disk=False,
            mse=mse,
        )
        _, result = _run_solver(ADMMQ, layer, cfg, x)
        return result

    default = solve(None)
    disabled = solve(0.0)
    enabled = solve(MSEConfig(norm=2.0))

    assert torch.equal(default[0], enabled[0])
    assert torch.equal(default[1], enabled[1])
    assert not torch.equal(default[0], disabled[0])


def test_admmq_config_roundtrip_keeps_method_and_export_split():
    cfg = ADMMQConfig(
        bits=3,
        group_size=32,
        sym=False,
        max_iter=77,
        rho=0.05,
        rho_update=1.2,
        adaptive_penalty=False,
        update_iter=5,
        switch_iter=11,
        pairswap_mode="off",
        pairswap_rounds=2,
        damp_percent=0.02,
        mse=MSEConfig(norm=1.0),
    )
    payload = cfg.to_dict()
    assert payload["method"] == "admmq"
    assert payload["quant_method"] == "gptq"
    assert payload["format"] == "gptq"
    assert payload["meta"]["max_iter"] == 77
    assert payload["meta"]["pairswap_mode"] == "off"
    assert payload["meta"]["adaptive_penalty"] is False

    rebuilt = QuantizeConfig(**payload)
    assert isinstance(rebuilt, ADMMQConfig)
    for name in (
        "bits",
        "group_size",
        "sym",
        "max_iter",
        "rho",
        "rho_update",
        "adaptive_penalty",
        "update_iter",
        "switch_iter",
        "pairswap_mode",
        "pairswap_rounds",
        "damp_percent",
    ):
        assert getattr(rebuilt, name) == getattr(cfg, name), name
    assert rebuilt.mse.norm == pytest.approx(1.0)
    assert rebuilt.export_quant_method() == METHOD.GPTQ


def test_admmq_applies_one_damp_pass_and_rejects_the_recovery_grid():
    """`damp_auto_increment` drives GPTQ's damp-recovery grid, which ADMM-Q
    never runs: the solver damps once with `damp_percent`. Accepting a
    non-zero value would persist a setting that changed no weight (#5583).
    """

    assert ADMMQConfig().damp_auto_increment == 0.0
    assert ADMMQConfig(bits=4, group_size=128, damp_auto_increment=0.0).damp_auto_increment == 0.0
    # An explicit `None` resolves to the ADMM-Q default, not GPTQ's 0.01.
    assert ADMMQConfig(bits=4, group_size=128, damp_auto_increment=None).damp_auto_increment == 0.0
    for value in (0.01, 0.25, 1.0):
        with pytest.raises(ValueError) as excinfo:
            ADMMQConfig(bits=4, group_size=128, damp_auto_increment=value)
        message = str(excinfo.value)
        assert "`damp_auto_increment`" in message, message
        assert "is not applied by the ADMMQ quantizer" in message, message

    # The factory accepts checkpoint-shaped payloads, so it drops the stale
    # claim (with a warning) like the other pre-fix options; the concrete
    # class above is the construction-time rejection.
    factory = QuantizeConfig(method="admmq", bits=4, group_size=128, damp_auto_increment=0.01)
    assert isinstance(factory, ADMMQConfig)
    assert factory.damp_auto_increment == 0.0


def test_admmq_rejects_dynamic_damp_auto_increment():
    with pytest.raises(ValueError) as excinfo:
        ADMMQConfig(bits=4, group_size=128, dynamic={"down_proj": {"damp_auto_increment": 0.01}})
    message = str(excinfo.value)
    assert "dynamic['down_proj'].damp_auto_increment" in message, message
    assert "is not applied by the" in message, message

    # Zero is the value ADMM-Q actually applies, so a zero rule is harmless.
    config = ADMMQConfig(bits=4, group_size=128, dynamic={"down_proj": {"damp_auto_increment": 0.0}})
    assert config.dynamic["down_proj"]["damp_auto_increment"] == 0.0


def test_pre_fix_admmq_checkpoint_drops_the_damp_recovery_grid():
    """A checkpoint written before the rejection still loads with its weights.

    The solver never read the field, so the stored tensors are unchanged; the
    restored config must stop claiming the option and re-save without it.
    """

    payload = ADMMQConfig(bits=4, group_size=128).to_dict()
    payload["damp_auto_increment"] = 0.01
    payload["meta"]["damp_auto_increment"] = 0.01
    payload["dynamic"] = {"down_proj": {"damp_auto_increment": 0.01}}

    for restored in (
        QuantizeConfig.from_quant_config(payload),
        QuantizeConfig(**payload),
    ):
        assert isinstance(restored, ADMMQConfig)
        assert restored.damp_auto_increment == 0.0
        assert restored.dynamic == {"down_proj": {}}
        resaved = restored.to_dict()
        assert resaved["meta"].get("damp_auto_increment") == 0.0
        assert "damp_auto_increment" not in resaved["dynamic"]["down_proj"]


def test_pre_fix_admmq_checkpoint_defaults_adaptive_penalty_on():
    payload = ADMMQConfig(bits=4, group_size=128).to_dict()
    payload["meta"].pop("adaptive_penalty")
    payload.pop("adaptive_penalty", None)

    restored = QuantizeConfig.from_quant_config(payload)
    assert isinstance(restored, ADMMQConfig)
    assert restored.adaptive_penalty is True


def test_gptq_keeps_the_damp_recovery_grid_it_applies():
    """Control: GPTQ does run the recovery grid, so its value must survive."""

    config = QuantizeConfig(bits=4, group_size=128, damp_auto_increment=0.25)
    assert config.damp_auto_increment == pytest.approx(0.25)

    payload = config.to_dict()
    payload["damp_auto_increment"] = 0.25
    payload["meta"]["damp_auto_increment"] = 0.25
    restored = QuantizeConfig.from_quant_config(payload)
    assert restored.damp_auto_increment == pytest.approx(0.25)


def test_admmq_config_planar_bits_select_gptq_p():
    cfg = ADMMQConfig(bits=6)
    assert cfg.format == FORMAT.GPTQ_P
    cfg = ADMMQConfig(bits=4, dynamic={"model.layers.0.mlp": {"bits": 5}})
    assert cfg.format == FORMAT.GPTQ_P


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mock_quantization": True},
        {"gptaq": GPTAQConfig()},
        {"foem": FOEMConfig()},
        {"gsq": GSQConfig(enabled=True)},
        {"static_groups": True},
        {"act_group_aware": True},
        {"max_iter": 0},
        {"rho": 0.0},
        {"rho_update": 1.0},
        {"adaptive_penalty": "yes"},
        {"update_iter": 0},
        {"switch_iter": -1},
        {"stagnation_window": 1},
        {"stagnation_tol": -1.0},
        {"refresh_accept_rel": -1.0},
        {"pairswap_mode": "sometimes"},
        {"pairswap_rounds": -1},
        {"pairswap_pairs": 0},
        {"pairswap_batch": 0},
    ],
)
def test_admmq_config_rejects_unsupported_options(kwargs):
    with pytest.raises((ValueError, TypeError)):
        ADMMQConfig(**kwargs)


@pytest.mark.parametrize("g2ptq", [True, {}])
def test_admmq_rejects_g2ptq(g2ptq):
    with pytest.raises(ValueError, match="g2ptq"):
        ADMMQConfig(g2ptq=g2ptq)


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("activation", "w4afp8"),
        ("mock_quantization", True),
        ("gptaq", {}),
        ("foem", {}),
        ("g2ptq", True),
        ("g2ptq", {}),
        ("gsq", {"enabled": True}),
        ("gsq_training", {"enabled": True}),
        ("static_groups", True),
        ("act_group_aware", True),
    ],
)
def test_admmq_rejects_incompatible_dynamic_options(option, value):
    with pytest.raises(ValueError, match=rf"dynamic.*{option}"):
        ADMMQConfig(dynamic={"down_proj": {option: value}})


def test_admmq_rejects_a_non_boolean_dynamic_adaptive_penalty():
    with pytest.raises(TypeError, match=r"dynamic\['down_proj'\]\.adaptive_penalty.*bool"):
        ADMMQConfig(dynamic={"down_proj": {"adaptive_penalty": "off"}})


def test_admmq_keeps_supported_dynamic_options():
    from gptqmodel.looper.gptq_processor import clone_gptq_config_for_module

    config = ADMMQConfig(g2ptq=False, dynamic={"down_proj": {
        "desc_act": True, "max_iter": 17, "pairswap_mode": "off", "gsq": {"enabled": False},
        "g2ptq": False, "adaptive_penalty": False,
    }})
    resolved = clone_gptq_config_for_module(config, "down_proj")
    assert resolved.method is METHOD.ADMMQ
    assert resolved.desc_act is True
    assert resolved.max_iter == 17
    assert resolved.pairswap_mode == "off"
    assert resolved.adaptive_penalty is False
    assert resolved.g2ptq is None


def test_admmq_config_rejects_activation_stream():
    with pytest.raises(ValueError, match="activation"):
        ADMMQConfig(bits=4, group_size=128, sym=True, activation="w4afp8")


def test_admmq_packing_and_run_plan_reuse_the_gptq_contract():
    from gptqmodel.models.quantization_run_plan import _CALIBRATED_PRIMARY, ProcessorStep
    from gptqmodel.utils.packing import _normalize_method

    # The checkpoint is a GPTQ artifact: packing must not look for an ADMM-Q
    # kernel family, and the run plan must use the GPTQ calibration processor.
    assert _normalize_method("admmq") == METHOD.GPTQ
    assert _normalize_method(METHOD.ADMMQ) == METHOD.GPTQ
    assert _CALIBRATED_PRIMARY[METHOD.ADMMQ] == ProcessorStep.GPTQ


# ---------------------------------------------------------------------------
# Pure algorithm helpers
# ---------------------------------------------------------------------------


def test_admm_rho_schedule_follows_growth_factor():
    assert admm_rho_at(0, 0.1, 1.1, 3) == pytest.approx(0.1)
    assert admm_rho_at(2, 0.1, 1.1, 3) == pytest.approx(0.1)
    assert admm_rho_at(3, 0.1, 1.1, 3) == pytest.approx(0.11)
    assert admm_rho_at(30, 0.1, 1.1, 3) == pytest.approx(0.1 * 1.1**10)
    # The penalty is capped so a long run cannot overflow.
    assert admm_rho_at(10**6, 0.1, 1.1, 3) == pytest.approx(1e6)


@pytest.mark.parametrize(
    ("iteration", "rho", "growth", "expected"),
    [
        (10, 0.1, 1e100, 1e6),
        (6000, 0.1, 1.0001, 0.1 * 1.0001**6000),
        (2, 1e-308, 1e155, 100.0),
    ],
)
def test_admm_rho_schedule_saturates_only_after_reaching_cap(iteration, rho, growth, expected):
    assert admm_rho_at(iteration, rho, growth, 1) == pytest.approx(expected, rel=1e-12)


def test_admm_penalty_schedule_uses_the_paper_thresholds():
    assert admm_penalty_schedule(5e4, 0.1, 1.1) == (0.1, 1.1)
    assert admm_penalty_schedule(math.nextafter(5e4, math.inf), 0.1, 1.1) == (0.05, 1.05)
    assert admm_penalty_schedule(3e5, 0.1, 1.1) == (0.05, 1.05)
    assert admm_penalty_schedule(math.nextafter(3e5, math.inf), 0.1, 1.1) == (0.01, 1.05)
    assert admm_penalty_schedule(math.inf, 0.1, 1.1) == (0.01, 1.05)
    assert admm_penalty_schedule(1e9, 0.1, 1.1, enabled=False) == (0.1, 1.1)


def test_refresh_accepts_only_strict_improvement():
    assert refresh_accepts(new=9.0, old=10.0, rel_margin=0.0) is True
    assert refresh_accepts(new=10.0, old=10.0, rel_margin=0.0) is False
    assert refresh_accepts(new=10.5, old=10.0, rel_margin=0.0) is False
    assert refresh_accepts(new=9.9, old=10.0, rel_margin=0.02) is False
    assert refresh_accepts(new=9.7, old=10.0, rel_margin=0.02) is True


def test_pairswap_delta_matches_bruteforce_objective_change():
    torch.manual_seed(7)
    columns = 4
    rows = 3
    basis = torch.randn(columns, columns)
    hessian = basis @ basis.t() + 0.1 * torch.eye(columns)
    weight = torch.randn(rows, columns)
    scale = torch.tensor([[0.25], [0.5], [0.125]])
    codes = torch.randint(0, 16, (rows, columns))
    quantized = (codes - 8) * scale

    first, second = 0, 2
    gradient = (weight - quantized) @ hessian
    deltas = pairswap_delta(
        gradient[:, first],
        gradient[:, second],
        hessian[first, first],
        hessian[second, second],
        hessian[first, second],
        scale[:, 0],
        scale[:, 0],
    )

    base = _reconstruction_error_rows(weight, quantized, hessian)
    combos = ((1, 1), (1, -1), (-1, 1), (-1, -1))
    for delta, (sign1, sign2) in zip(deltas, combos):
        moved = quantized.clone()
        moved[:, first] += sign1 * scale[:, 0]
        moved[:, second] += sign2 * scale[:, 0]
        expected = _reconstruction_error_rows(weight, moved, hessian) - base
        assert torch.allclose(delta, expected, atol=1e-5, rtol=1e-4), (sign1, sign2)


def test_pairswap_reconstructs_exact_grid_values_from_selected_codes():
    scale = torch.tensor([[0.1], [0.1]])
    weight = torch.tensor([[0.72, 0.72], [0.6, 0.6]])
    hessian = torch.tensor([[1.0, 0.25], [0.25, 1.0]])
    initial = torch.full((2, 2), 6.0) * scale
    solver = ADMMQ(_make_layer(weight), qcfg=ADMMQConfig(
        bits=3, group_size=-1, sym=False, pairswap_rounds=1, pairswap_batch=1,
    ))
    grid = _Grid(0, 2, scale, torch.zeros_like(scale), None, None, 7)

    actual, flips = solver._pairswap(weight, initial.clone(), hessian, [grid])
    expected = torch.tensor([[7.0, 7.0], [6.0, 6.0]]) * scale
    assert flips == 1
    assert _reconstruction_error(weight, actual, hessian) < _reconstruction_error(weight, initial, hessian)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_projection_objective_is_squared_frobenius_norm():
    target = torch.tensor([[1.0, -2.0], [3.0, 0.5]])
    projected = torch.tensor([[1.5, -2.0], [2.0, 1.5]])
    assert projection_objective(target, projected) == pytest.approx(0.25 + 1.0 + 1.0)


# ---------------------------------------------------------------------------
# Solver properties
# ---------------------------------------------------------------------------


def _small_config(**overrides):
    base = {
        "bits": 3,
        "group_size": 4,
        "sym": False,
        "max_iter": 40,
        "switch_iter": 10,
        "pairswap_mode": "off",
        # Solver-mechanics fixtures are tuned without the shrink grid; the
        # bit-adaptive default is covered separately.
        "mse": 0.0,
    }
    base.update(overrides)
    return ADMMQConfig(**base)


def test_admmq_early_stop_uses_refreshed_iterate():
    generator = torch.Generator().manual_seed(4)
    weight = torch.randn(3, 8, generator=generator)
    x = torch.randn(24, 8, generator=generator)
    x[:, 1] += 2 * x[:, 0]
    x[:, 4] -= 2 * x[:, 2]
    config = _small_config(
        max_iter=10,
        rho=2.0,
        update_iter=1,
        switch_iter=2,
        stagnation_window=2,
        stagnation_tol=1.0,
        offload_to_disk=False,
    )

    control, _ = _run_solver(ADMMQ, _make_layer(weight), replace(config, refresh_on_stagnation=False), x)
    capped, capped_result = _run_solver(ADMMQ, _make_layer(weight), replace(config, max_iter=2), x)
    solver, result = _run_solver(ADMMQ, _make_layer(weight), config, x)

    assert control.admmq_diagnostics["iterations"] == 2
    assert capped.admmq_diagnostics["refresh_accepted"]
    assert solver.admmq_diagnostics["refresh_accepted"]
    assert solver.admmq_diagnostics["iterations"] > 2
    hessian = _damped_hessian(x.double(), config.damp_percent)
    capped_error = _reconstruction_error(weight.double(), capped_result[0].double(), hessian)
    error = _reconstruction_error(weight.double(), result[0].double(), hessian)
    assert error < 0.9 * capped_error


def test_admmq_projection_stays_exactly_on_the_fitted_grid():
    torch.manual_seed(0)
    weight = torch.randn(OUT_FEATURES, IN_FEATURES) * 0.2
    layer = _make_layer(weight)
    x = _calibration()

    solver, result = _run_solver(ADMMQ, layer, _small_config(), x)
    quantized, scale, zero, g_idx, duration, avg_loss, damp, samples = result

    assert quantized.shape == weight.shape
    assert samples == x.shape[0]
    assert damp == pytest.approx(0.01)
    assert math.isfinite(avg_loss)
    assert duration >= 0.0

    per_column_scale = scale[:, g_idx]
    per_column_zero = zero[:, g_idx]
    codes = torch.round(quantized / per_column_scale) + per_column_zero
    assert torch.allclose(codes, codes.round())
    assert torch.allclose(quantized, per_column_scale * (codes - per_column_zero))
    assert int(codes.min()) >= 0
    assert int(codes.max()) <= 2**3 - 1

    # The solver must not retain dense Hessian/workspace state after the solve.
    assert solver.H is None


def test_admmq_beats_rtn_and_is_competitive_with_gptq_on_ill_conditioned_fixture():
    torch.manual_seed(0)
    weight = torch.randn(OUT_FEATURES, IN_FEATURES) * 0.2
    x = _calibration()
    hessian = _damped_hessian(x, damp=0.01)

    rtn_cfg = ADMMQConfig(bits=3, group_size=-1, sym=True, max_iter=1, pairswap_mode="off")
    quantized_rtn = _rtn_baseline(weight, rtn_cfg)

    gptq_layer = _make_layer(weight)
    gptq_cfg = QuantizeConfig(bits=3, group_size=-1, sym=True, damp_percent=0.01, mse=MSEConfig(norm=2.0))
    _, gptq_result = _run_solver(GPTQ, gptq_layer, gptq_cfg, x)

    admm_layer = _make_layer(weight)
    admm_cfg = ADMMQConfig(bits=3, group_size=-1, sym=True, max_iter=250, pairswap_mode="always")
    _, admm_result = _run_solver(ADMMQ, admm_layer, admm_cfg, x)

    rtn_error = _reconstruction_error(weight, quantized_rtn, hessian)
    gptq_error = _reconstruction_error(weight, gptq_result[0], hessian)
    admm_error = _reconstruction_error(weight, admm_result[0], hessian)

    assert admm_error < rtn_error, (admm_error, rtn_error)
    assert admm_error <= gptq_error * 1.10, (admm_error, gptq_error)


def test_admmq_reports_diagnostics_and_runs_pairswap_only_when_enabled():
    torch.manual_seed(3)
    weight = torch.randn(OUT_FEATURES, IN_FEATURES) * 0.2
    x = _calibration()

    _, off_result = _run_solver(
        ADMMQ,
        _make_layer(weight),
        _small_config(pairswap_mode="off"),
        x,
    )
    off_solver = ADMMQ(_make_layer(weight), qcfg=_small_config(pairswap_mode="off"))
    off_solver.quantizer.configure(perchannel=True)
    off_solver.add_batch(x, None)
    off_solver.quantize()
    assert off_solver.admmq_diagnostics["pairswap_flips"] == 0

    solver = ADMMQ(
        _make_layer(weight),
        qcfg=_small_config(
            pairswap_mode="always",
            max_iter=60,
            refresh_on_stagnation=True,
            stagnation_window=2,
            stagnation_tol=1.0,
        ),
    )
    solver.quantizer.configure(perchannel=True)
    solver.add_batch(x, None)
    solver.quantize()
    diagnostics = solver.admmq_diagnostics

    assert 0 < diagnostics["iterations"] <= 60
    assert diagnostics["condition_number"] > 0.0
    assert diagnostics["refresh_attempts"] == 1
    assert isinstance(diagnostics["refresh_accepted"], bool)
    assert diagnostics["pairswap_flips"] >= 0
    assert math.isfinite(diagnostics["final_objective"])
    assert diagnostics["final_objective"] <= diagnostics["initial_objective"] * 1.001
    assert diagnostics["fallback_reason"] is None
    assert off_result[0].shape == weight.shape


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_admmq_eigh_failure_falls_back_to_rtn(monkeypatch, device):
    """A failed float64 eigendecomposition must not abort the model.

    The layer keeps the same per-group RTN projection the solver uses for its
    "did not improve" exit, so packing and replay see a normal grid.
    """

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires an accelerator")
    torch.manual_seed(11)
    weight = torch.randn(OUT_FEATURES, IN_FEATURES) * 0.2
    x = _calibration().to(device)
    config = _small_config()
    attempts = []

    def fail_eigh(matrix):
        attempts.append(tuple(matrix.shape))
        raise torch._C._LinAlgError("injected eigendecomposition failure")

    monkeypatch.setattr(torch.linalg, "eigh", fail_eigh)
    solver, result = _run_solver(ADMMQ, _make_layer(weight).to(device), config, x)

    assert attempts, "the eigendecomposition was not attempted"
    diagnostics = solver.admmq_diagnostics
    assert diagnostics["fallback_to_rtn"] is True
    assert diagnostics["fallback_reason"] == "eigh_failure"
    assert diagnostics["status"] == "solved"
    assert diagnostics["converged"] is False
    assert diagnostics["stop_reason"] == "fallback"
    assert diagnostics["iterations"] == 0
    assert diagnostics["max_iter"] == config.max_iter
    assert diagnostics["pairswap_ran"] is False

    quantized, scale, zero, g_idx, duration, avg_loss, damp, samples = result
    assert bool(torch.isfinite(quantized).all())
    assert samples == x.shape[0]
    assert damp == 0.0
    torch.testing.assert_close(
        quantized,
        _rtn_grid_projection(weight.to(device), config, solver.module),
        rtol=0,
        atol=0,
    )
    assert solver.H is None


@pytest.mark.parametrize(
    ("corruption", "reason", "iterations"),
    [
        ("eigenvalues", "non_finite_eigenvalues", 0),
        ("eigenvectors", "non_finite_iterate", 1),
    ],
)
def test_admmq_non_finite_factorization_falls_back_to_rtn(
    monkeypatch, corruption, reason, iterations
):
    """A solver that returns NaN instead of raising must also recover."""

    torch.manual_seed(12)
    weight = torch.randn(OUT_FEATURES, IN_FEATURES) * 0.2
    x = _calibration()
    config = _small_config()
    real_eigh = torch.linalg.eigh
    attempts = []

    def corrupt_eigh(matrix):
        values, vectors = real_eigh(matrix)
        attempts.append(1)
        values = values.clone()
        vectors = vectors.clone()
        if corruption == "eigenvalues":
            values[0] = float("nan")
        else:
            vectors[0, 0] = float("nan")
        return values, vectors

    monkeypatch.setattr(torch.linalg, "eigh", corrupt_eigh)
    solver, result = _run_solver(ADMMQ, _make_layer(weight), config, x)

    assert attempts
    diagnostics = solver.admmq_diagnostics
    assert diagnostics["fallback_to_rtn"] is True
    assert diagnostics["fallback_reason"] == reason
    assert diagnostics["iterations"] == iterations
    quantized = result[0]
    assert bool(torch.isfinite(quantized).all())
    torch.testing.assert_close(
        quantized,
        _rtn_grid_projection(weight, config, solver.module),
        rtol=0,
        atol=0,
    )
    assert solver.H is None


def test_admmq_non_finite_loss_falls_back_to_rtn():
    """A finite but extreme weight/Hessian scale can overflow the objective."""

    torch.manual_seed(13)
    weight = torch.randn(OUT_FEATURES, IN_FEATURES) * 1e19
    x = torch.randn(CALIBRATION_SAMPLES, IN_FEATURES) * 1e18
    config = _small_config(bits=4, group_size=8, sym=True, max_iter=4, switch_iter=2)
    solver, result = _run_solver(ADMMQ, _make_layer(weight), config, x)

    diagnostics = solver.admmq_diagnostics
    assert diagnostics["fallback_to_rtn"] is True
    assert diagnostics["fallback_reason"] == "non_finite_loss"
    quantized = result[0]
    assert bool(torch.isfinite(quantized).all())
    torch.testing.assert_close(
        quantized,
        _rtn_grid_projection(weight, config, solver.module),
        rtol=0,
        atol=0,
    )
    assert solver.H is None


def test_admmq_rtn_fallback_raises_when_it_is_also_non_finite():
    """A corrupt source weight must not be published as a finite checkpoint."""

    torch.manual_seed(14)
    weight = torch.randn(OUT_FEATURES, IN_FEATURES) * 0.2
    weight[0, 0] = float("inf")
    x = _calibration()
    config = _small_config(bits=4, group_size=8, sym=True, max_iter=4, switch_iter=2)

    with pytest.raises(ValueError, match="RTN fallback also produced non-finite"):
        _run_solver(ADMMQ, _make_layer(weight), config, x)
def test_admmq_reports_convergence_and_stop_reason():
    torch.manual_seed(0)
    weight = torch.randn(OUT_FEATURES, IN_FEATURES) * 0.2
    x = _calibration()

    capped, _ = _run_solver(
        ADMMQ,
        _make_layer(weight),
        _small_config(max_iter=3, switch_iter=1, early_stop_tol=0.0),
        x,
    )
    capped_diagnostics = capped.admmq_diagnostics
    assert capped_diagnostics.get("status") == "solved"
    assert capped_diagnostics.get("iterations") == 3
    assert capped_diagnostics.get("max_iter") == 3
    assert capped_diagnostics.get("converged") is False
    assert capped_diagnostics.get("stop_reason") == "max_iter"

    early, _ = _run_solver(
        ADMMQ,
        _make_layer(weight),
        _small_config(max_iter=10, switch_iter=1, early_stop_tol=1e9),
        x,
    )
    early_diagnostics = early.admmq_diagnostics
    assert early_diagnostics.get("iterations") == 1
    assert early_diagnostics.get("max_iter") == 10
    assert early_diagnostics.get("converged") is True
    assert early_diagnostics.get("stop_reason") == "early_stop"


@pytest.mark.parametrize(
    ("replacement", "reason"),
    [
        (lambda quantized: quantized + 1.0, "worse_than_rtn"),
        (lambda quantized: torch.full_like(quantized, float("nan")), "non_finite_objective"),
    ],
)
def test_admmq_records_rtn_fallback_reason(monkeypatch, replacement, reason):
    torch.manual_seed(0)
    weight = torch.randn(OUT_FEATURES, IN_FEATURES) * 0.2
    x = _calibration()
    solver = ADMMQ(
        _make_layer(weight),
        qcfg=_small_config(
            max_iter=2,
            switch_iter=1,
            early_stop_tol=0.0,
            pairswap_mode="always",
        ),
    )
    solver.quantizer.configure(perchannel=True)
    solver.add_batch(x, None)

    def replace_pairswap(weight, quantized, hessian, grids):
        return replacement(quantized), 0

    monkeypatch.setattr(solver, "_pairswap", replace_pairswap)
    solver.quantize()
    diagnostics = solver.admmq_diagnostics
    assert diagnostics.get("fallback_to_rtn") is True
    assert diagnostics.get("fallback_reason") == reason
    assert diagnostics.get("stop_reason") == "max_iter"
    assert diagnostics.get("pairswap_ran") is True
    assert diagnostics.get("pairswap_flips") == 0


def test_admmq_records_skipped_calibration_fallback():
    torch.manual_seed(0)
    weight = torch.randn(OUT_FEATURES, IN_FEATURES) * 0.2
    solver = ADMMQ(_make_layer(weight), qcfg=_small_config())
    solver.quantizer.configure(perchannel=True)
    solver.quantize()

    diagnostics = getattr(solver, "admmq_diagnostics", None) or {}
    assert diagnostics.get("status") == "skipped"
    assert diagnostics.get("reason") == "insufficient_calibration"
    assert diagnostics.get("stop_reason") == "skipped"
    assert diagnostics.get("iterations") == 0
    assert diagnostics.get("fallback_to_rtn") is True
    assert diagnostics.get("fallback_reason") == "insufficient_calibration"


def test_admmq_embedding_diagnostics_are_skipped():
    embedding = nn.Embedding(8, 16, dtype=torch.float32).eval()
    solver = ADMMQ(embedding, qcfg=_small_config(group_size=4))
    solver.quantizer.configure(perchannel=True)
    solver.add_batch(torch.randint(0, 8, (4, 5)), None)
    solver.quantize()

    diagnostics = getattr(solver, "admmq_diagnostics", None) or {}
    assert diagnostics.get("status") == "skipped"
    assert diagnostics.get("reason") == "embedding"
    assert diagnostics.get("stop_reason") == "skipped"
    assert diagnostics.get("fallback_to_rtn") is False
    assert diagnostics.get("fallback_reason") is None


def test_admmq_diagnostics_summary_names_stop_reasons_and_distributions():
    layers = {
        "a": {
            "status": "solved",
            "converged": True,
            "stop_reason": "early_stop",
            "iterations": 5,
            "max_iter": 10,
            "condition_number": 10.0,
            "refresh_attempts": 1,
            "refresh_accepted": True,
            "pairswap_ran": True,
            "pairswap_flips": 3,
            "fallback_to_rtn": False,
            "fallback_reason": None,
        },
        "b": {
            "status": "solved",
            "converged": False,
            "stop_reason": "max_iter",
            "iterations": 10,
            "max_iter": 10,
            "condition_number": float("inf"),
            "refresh_attempts": 0,
            "refresh_accepted": False,
            "pairswap_ran": False,
            "pairswap_flips": 0,
            "fallback_to_rtn": False,
            "fallback_reason": None,
        },
        "c": {
            "status": "solved",
            "converged": False,
            "stop_reason": "max_iter",
            "iterations": 10,
            "max_iter": 10,
            "condition_number": 30.0,
            "refresh_attempts": 0,
            "refresh_accepted": False,
            "pairswap_ran": False,
            "pairswap_flips": 0,
            "fallback_to_rtn": True,
            "fallback_reason": "worse_than_rtn",
        },
        "d": {
            "status": "skipped",
            "converged": False,
            "stop_reason": "skipped",
            "iterations": 0,
            "max_iter": 10,
            "condition_number": None,
            "refresh_attempts": 0,
            "refresh_accepted": False,
            "pairswap_ran": False,
            "pairswap_flips": 0,
            "fallback_to_rtn": True,
            "fallback_reason": "insufficient_calibration",
        },
    }

    summary = summarize_admmq_diagnostics(layers)
    assert summary["layers"] == 4
    assert summary["solved"] == 3
    assert summary["skipped"] == 1
    assert summary["early_stop_layers"] == ["a"]
    assert summary["max_iter_layers"] == ["b", "c"]
    assert summary["rtn_fallback_layers"] == ["c", "d"]
    assert summary["rtn_fallback_reasons"] == {
        "insufficient_calibration": 1,
        "worse_than_rtn": 1,
    }
    assert summary["refresh_attempted"] == 1
    assert summary["refresh_accepted"] == 1
    assert summary["refresh_accepted_layers"] == ["a"]
    assert summary["pairswap_layers"] == 1
    assert summary["pairswap_flips"] == 3
    assert summary["pairswap_layers_with_flips"] == ["a"]
    assert summary["condition_number"] == {"min": 10.0, "median": 20.0, "max": 30.0}
    assert summary["condition_number_nonfinite"] == 1
    assert summary["iterations"] == {"min": 5.0, "median": 10.0, "max": 10.0}

    alert = admmq_diagnostics_alert(summary)
    assert alert is not None
    assert "skipped=1" in alert
    assert "rtn_fallback=2" in alert
    assert "condition_number_nonfinite=1" in alert
    assert "insufficient_calibration:1" in alert
    assert "worse_than_rtn:1" in alert


def test_admmq_diagnostics_alert_is_silent_for_a_clean_run():
    summary = {
        "layers": 2,
        "solved": 2,
        "skipped": 0,
        "rtn_fallback": 0,
        "rtn_fallback_reasons": {},
        "condition_number_nonfinite": 0,
    }

    assert admmq_diagnostics_alert(summary) is None


def test_admmq_pairswap_auto_uses_the_condition_number_gate():
    layer = _make_layer(torch.zeros(OUT_FEATURES, IN_FEATURES))
    always = ADMMQ(layer, qcfg=ADMMQConfig(pairswap_mode="always"))
    off = ADMMQ(layer, qcfg=ADMMQConfig(pairswap_mode="off"))
    auto = ADMMQ(
        layer,
        qcfg=ADMMQConfig(pairswap_mode="auto", pairswap_condition_threshold=1e4),
    )

    assert always._should_pairswap(1.0) is True
    assert off._should_pairswap(1e9) is False
    assert auto._should_pairswap(1e3) is False
    assert auto._should_pairswap(1e5) is True
    assert auto._should_pairswap(float("inf")) is True
    assert auto._should_pairswap(float("nan")) is False


def test_admmq_desc_act_round_trips_to_natural_column_order():
    torch.manual_seed(5)
    weight = torch.randn(OUT_FEATURES, IN_FEATURES) * 0.2
    x = _calibration()
    cfg = _small_config(desc_act=True)

    solver, result = _run_solver(ADMMQ, _make_layer(weight), cfg, x)
    quantized, scale, zero, g_idx, *_ = result

    assert quantized.shape == weight.shape
    assert g_idx.shape == (IN_FEATURES,)
    codes = torch.round(quantized / scale[:, g_idx]) + zero[:, g_idx]
    assert torch.allclose(quantized, scale[:, g_idx] * (codes - zero[:, g_idx]))
    assert solver.H is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an accelerator")
@pytest.mark.parametrize("desc_act", [False, True])
@pytest.mark.parametrize("error_type", [torch.OutOfMemoryError, RuntimeError])
def test_admmq_retries_only_allocation_failures_from_pristine_state(monkeypatch, desc_act, error_type):
    cpu_weight = torch.randn(OUT_FEATURES, IN_FEATURES, generator=torch.Generator().manual_seed(37))
    x = _calibration().round()
    x[:, 7] = 0
    x[:, 2] += x[:, 1]
    assert (x[:, 2] @ x[:, 1]).abs() > 0
    config = _small_config(desc_act=desc_act, max_iter=8, refresh_on_stagnation=False)
    device = torch.device("cuda", torch.cuda.current_device())
    solver = ADMMQ(_make_layer(cpu_weight).to(device), qcfg=config)
    solver.quantizer.configure(perchannel=True)
    solver.add_batch(x.to(device), None)
    _, expected = _run_solver(ADMMQ, _make_layer(cpu_weight), config, x)

    project = solver._project
    attempts = []

    def project_with_failure(weight, grids):
        attempts.append(weight.device.type)
        if attempts == ["cuda"] * 4:
            # RTN plus two completed ADMM iterations preceded the failure.
            raise error_type("injected out of memory" if error_type is torch.OutOfMemoryError else "injected solver error")
        return project(weight, grids)

    monkeypatch.setattr(solver, "_project", project_with_failure)
    if error_type is RuntimeError:
        with pytest.raises(RuntimeError, match="injected solver error"):
            solver.quantize()
        assert attempts == ["cuda"] * 4
    else:
        actual = solver.quantize()
        assert attempts[:4] == ["cuda"] * 4
        assert set(attempts[4:]) == {"cpu"}
        assert actual[0].device == device
        for tensor, reference in zip(actual[:4], expected[:4]):
            torch.testing.assert_close(tensor.cpu(), reference, rtol=0, atol=0)
        assert actual[5] == pytest.approx(expected[5], rel=1e-6)
    assert solver.H is None
    torch.testing.assert_close(solver.module.weight.cpu(), cpu_weight, rtol=0, atol=0)


@pytest.mark.parametrize("device_type", ["mps", "xpu", "xla"])
def test_admmq_solve_device_keeps_qualified_families_and_hosts_the_rest(device_type):
    solver = ADMMQ(_make_layer(torch.ones(2, 4)), qcfg=_small_config())

    assert solver._solve_device(torch.device("cpu")) == torch.device("cpu")
    assert solver._solve_device(torch.device("cuda", 1)) == torch.device("cuda", 1)
    assert solver._solve_device(torch.device(device_type)) == torch.device("cpu")


def test_admmq_solve_device_hosts_npu():
    try:
        device = torch.device("npu")
    except (RuntimeError, ValueError):
        pytest.skip("this Torch build has no NPU device type")

    solver = ADMMQ(_make_layer(torch.ones(2, 4)), qcfg=_small_config())
    assert solver._solve_device(device) == torch.device("cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an accelerator")
@pytest.mark.parametrize("desc_act", [False, True])
def test_admmq_unqualified_device_restarts_the_solve_on_the_host(monkeypatch, desc_act):
    cpu_weight = torch.randn(OUT_FEATURES, IN_FEATURES, generator=torch.Generator().manual_seed(41))
    x = _calibration().round()
    x[:, 7] = 0
    x[:, 2] += x[:, 1]
    config = _small_config(desc_act=desc_act, max_iter=8, refresh_on_stagnation=False)
    _, expected = _run_solver(ADMMQ, _make_layer(cpu_weight), config, x)

    class HostSolveADMMQ(ADMMQ):
        def _solve_device(self, source_device):
            # Stand in for a device family that cannot run the float64
            # factorization, without requiring that family in CI.
            return torch.device("cpu")

    device = torch.device("cuda", torch.cuda.current_device())
    solver = HostSolveADMMQ(_make_layer(cpu_weight).to(device), qcfg=config)
    solver.quantizer.configure(perchannel=True)
    solver.add_batch(x.to(device), None)

    solve = solver._admm_weight_solve
    seen = []

    def record_solve(weight, hessian):
        seen.append((weight.device.type, hessian.device.type))
        return solve(weight, hessian)

    monkeypatch.setattr(solver, "_admm_weight_solve", record_solve)
    actual = solver.quantize()

    assert seen == [("cpu", "cpu")]
    assert actual[0].device == device
    for tensor, reference in zip(actual[:4], expected[:4]):
        torch.testing.assert_close(tensor.cpu(), reference, rtol=0, atol=0)
    assert actual[5] == pytest.approx(expected[5], rel=1e-6)
    assert solver.H is None
    torch.testing.assert_close(solver.module.weight.cpu(), cpu_weight, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("spectrum", "expected"),
    [
        ([0.0, 0.0, 2.0, 2.0], math.inf),
        ([0.0, 0.0], math.inf),
        ([1e-20, 2.0], 2e20),
        ([0.5, 2.0], 4.0),
    ],
)
def test_admmq_condition_number_keeps_null_and_small_eigenvalues(spectrum, expected):
    solver = ADMMQ(_make_layer(torch.ones(2, 4)), qcfg=_small_config())
    actual = solver._condition_number(torch.tensor(spectrum, dtype=torch.float64))
    assert actual == pytest.approx(expected, rel=1e-12)


def _penalty_schedule_fixture(correlation: float, **overrides):
    hessian = torch.tensor(
        [[1.0, correlation], [correlation, 1.0]], dtype=torch.float64
    )
    weight = torch.tensor(
        [[0.7, -0.2], [0.1, 0.4], [-0.3, 0.8], [0.2, -0.6]], dtype=torch.float64
    )
    config = ADMMQConfig(
        bits=3,
        group_size=-1,
        sym=True,
        damp_percent=1e-12,
        max_iter=3,
        update_iter=1,
        switch_iter=2,
        pairswap_mode="off",
        # The penalty schedule is independent of the MSE search, so pin it off
        # instead of inheriting the bit-adaptive default at 3 bits.
        mse=0,
        rho=0.1,
        rho_update=1.1,
        **overrides,
    )
    solver = ADMMQ(_make_layer(weight), qcfg=config)
    solver.quantizer.configure(perchannel=True)
    solver._admm_weight_solve(weight, hessian)
    return solver.admmq_diagnostics


def test_admmq_ill_conditioned_layers_use_the_adaptive_penalty_schedule():
    healthy = _penalty_schedule_fixture(0.0)
    moderate = _penalty_schedule_fixture(0.999993)
    severe = _penalty_schedule_fixture(0.999999)

    assert healthy["condition_number"] == pytest.approx(1.0)
    assert 5e4 < moderate["condition_number"] <= 3e5
    assert severe["condition_number"] > 3e5

    # The configured 0.1 / 1.1 schedule must survive on a well-conditioned
    # layer. Ill-conditioned layers take the paper's 0.05 (or 0.01 above
    # 3e5) initial penalty with the slower 1.05 growth.
    assert healthy["ill_conditioned"] is False
    assert moderate["ill_conditioned"] is True
    assert severe["ill_conditioned"] is True
    assert healthy["rho_initial"] == pytest.approx(0.1)
    assert moderate["rho_initial"] == pytest.approx(0.05)
    assert severe["rho_initial"] == pytest.approx(0.01)
    assert moderate["rho_update"] == pytest.approx(1.05)
    assert severe["rho_update"] == pytest.approx(1.05)

    # The recorded final penalty must be the selected schedule evaluated at the
    # last iteration the solver actually ran. Deriving it keeps the assertion
    # valid when convergence, not the schedule, changes.
    for diagnostics, rho_initial, rho_update in (
        (healthy, 0.1, 1.1),
        (moderate, 0.05, 1.05),
        (severe, 0.01, 1.05),
    ):
        expected = admm_rho_at(diagnostics["iterations"] - 1, rho_initial, rho_update, 1)
        assert diagnostics["rho_final"] == pytest.approx(expected)
    # Growth must be exercised, not only the first-iteration penalty.
    assert healthy["iterations"] >= 2
    assert moderate["iterations"] >= 2
    assert severe["iterations"] >= 2


def test_admmq_adaptive_penalty_can_be_disabled_for_an_ablation():
    diagnostics = _penalty_schedule_fixture(0.999999, adaptive_penalty=False)
    assert diagnostics["ill_conditioned"] is True
    assert diagnostics["rho_initial"] == pytest.approx(0.1)
    assert diagnostics["rho_update"] == pytest.approx(1.1)
    expected = admm_rho_at(diagnostics["iterations"] - 1, 0.1, 1.1, 1)
    assert diagnostics["rho_final"] == pytest.approx(expected)


def test_admmq_auto_pairswap_refines_rank_deficient_calibration():
    # Both column pairs are correlated, with no dead input columns. The
    # positive damping is below FP32 resolution, leaving two null directions.
    generator = torch.Generator().manual_seed(2)
    weight = torch.rand(4, 4, generator=generator) - 0.5
    x = torch.tensor([[1., 1., 0., 0.], [0., 0., 1., 1.]]).repeat(16, 1)
    hessian = 2.0 * x.double().t() @ x.double() / x.shape[0]
    assert torch.linalg.matrix_rank(hessian) == 2
    assert (hessian.diagonal() > 0).all()
    results = {}
    for mode in ("off", "auto", "always"):
        config = _small_config(
            bits=3, group_size=2, sym=True, damp_percent=1e-10,
            max_iter=1, pairswap_mode=mode, pairswap_rounds=2, pairswap_batch=8,
        )
        solver, result = _run_solver(ADMMQ, _make_layer(weight), config, x)
        results[mode] = result
        if mode != "off":
            assert solver.admmq_diagnostics["pairswap_flips"] > 0
            assert math.isinf(solver.admmq_diagnostics["condition_number"])
        assert solver.H is None

    # Compare actual returned weights against an independent FP64 objective,
    # not the solver's recorded error. AUTO must take the same refinement as
    # the explicit policy and retain its quantized weights and group metadata.
    before = _reconstruction_error(weight.double(), results["off"][0].double(), hessian)
    after = _reconstruction_error(weight.double(), results["auto"][0].double(), hessian)
    # PairSwap's gain is bit-width dependent; at 3 bits the measured
    # reconstruction error drops by about 20%, so require a clear improvement
    # rather than the 2-bit-specific 50% threshold.
    assert after < before * 0.9
    for actual, expected in zip(results["auto"][:4], results["always"][:4]):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
