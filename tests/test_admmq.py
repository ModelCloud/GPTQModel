# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""ADMM-Q weight quantizer: config surface, grid properties and Torch oracle.

The quality oracle recomputes ``0.5 * Tr(dW H dW^T)`` directly from the returned
weights, so it is independent of the solver's internal bookkeeping.
"""

import math

import pytest
import torch
import torch.nn as nn

from gptqmodel.quantization.admmq import (
    ADMMQ,
    _Grid,
    admm_rho_at,
    pairswap_delta,
    projection_objective,
    refresh_accepts,
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
from gptqmodel.quantization.quantizer import Quantizer


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
    return h + damp * torch.diagonal(h).mean() * torch.eye(h.shape[0])


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


# ---------------------------------------------------------------------------
# Config surface
# ---------------------------------------------------------------------------


def test_admmq_method_metadata_and_export():
    assert METHOD.ADMMQ.value == "admmq"
    assert METHOD_SUPPORTED_BITS[METHOD.ADMMQ] == (2, 3, 4, 5, 6, 7, 8)
    assert QUANT_METHOD_FORMAT_MAPPING[METHOD.ADMMQ] == {
        FORMAT.GPTQ,
        FORMAT.GPTQ_V2,
        FORMAT.GPTQ_P,
        FORMAT.MARLIN,
        FORMAT.BITBLAS,
    }
    assert resolve_quant_format(FORMAT.GPTQ, METHOD.ADMMQ) == FORMAT.GPTQ
    assert resolve_quant_format(FORMAT.GPTQ_P, METHOD.ADMMQ) == FORMAT.GPTQ_P


def test_admmq_config_defaults():
    cfg = ADMMQConfig()
    assert cfg.method == METHOD.ADMMQ
    assert cfg.format == FORMAT.GPTQ
    assert cfg.export_quant_method() == METHOD.GPTQ
    assert cfg.max_iter == 300
    assert cfg.rho == pytest.approx(0.1)
    assert cfg.rho_update == pytest.approx(1.1)
    assert cfg.update_iter == 3
    assert cfg.switch_iter == 30
    assert cfg.damp_percent == pytest.approx(0.01)
    assert isinstance(cfg.mse, MSEConfig)
    assert cfg.mse.norm == pytest.approx(0.0)
    assert cfg.pairswap_mode == "always"
    assert cfg.static_groups is False
    assert cfg.act_group_aware is False


def test_admmq_config_roundtrip_keeps_method_and_export_split():
    cfg = ADMMQConfig(
        bits=3,
        group_size=32,
        sym=False,
        max_iter=77,
        rho=0.05,
        rho_update=1.2,
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

    rebuilt = QuantizeConfig(**payload)
    assert isinstance(rebuilt, ADMMQConfig)
    for name in (
        "bits",
        "group_size",
        "sym",
        "max_iter",
        "rho",
        "rho_update",
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


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("activation", "w4afp8"),
        ("mock_quantization", True),
        ("gptaq", {}),
        ("foem", {}),
        ("gsq", {"enabled": True}),
        ("gsq_training", {"enabled": True}),
        ("static_groups", True),
        ("act_group_aware", True),
    ],
)
def test_admmq_rejects_incompatible_dynamic_options(option, value):
    with pytest.raises(ValueError, match=rf"dynamic.*{option}"):
        ADMMQConfig(dynamic={"down_proj": {option: value}})


def test_admmq_keeps_supported_dynamic_options():
    from gptqmodel.looper.gptq_processor import clone_gptq_config_for_module

    config = ADMMQConfig(dynamic={"down_proj": {
        "desc_act": True, "max_iter": 17, "pairswap_mode": "off", "gsq": {"enabled": False},
    }})
    resolved = clone_gptq_config_for_module(config, "down_proj")
    assert resolved.method is METHOD.ADMMQ
    assert resolved.desc_act is True
    assert resolved.max_iter == 17
    assert resolved.pairswap_mode == "off"


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
    }
    base.update(overrides)
    return ADMMQConfig(**base)


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
    assert off_result[0].shape == weight.shape


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
            bits=2, group_size=2, sym=True, damp_percent=1e-10,
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
    assert after < before * 0.5
    for actual, expected in zip(results["auto"][:4], results["always"][:4]):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
