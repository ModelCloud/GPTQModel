# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest
import torch

from gptqmodel.looper import gptq_processor
from gptqmodel.looper.gptq_processor import clone_gptq_config_for_module, log_scale_search_config
from gptqmodel.quantization import QuantizeConfig, Quantizer, ScaleSearchConfig
from gptqmodel.quantization.config import AdaptiveClippingConfig
from gptqmodel.quantization.gptq import GPTQ


def _run_scale_search(
    weights: torch.Tensor,
    method: ScaleSearchConfig | str | None,
    *,
    hessian: torch.Tensor | None = None,
    mse: float = 0.0,
    bits: int = 2,
    grid: int = 20,
):
    qcfg = QuantizeConfig(
        bits=bits,
        group_size=weights.shape[1],
        sym=True,
        act_group_aware=False,
        offload_to_disk=False,
        mse=mse,
        scale_search=method,
    )
    quantizer = Quantizer(qcfg)
    quantizer.configure(perchannel=True, grid=grid, maxshrink=0.8)
    quantizer.find_params(weights, weight=True, hessian=hessian)
    return quantizer.quantize(weights), quantizer.scale.clone(), quantizer.zero.clone()


def _quadratic_error(reference: torch.Tensor, candidate: torch.Tensor, hessian: torch.Tensor) -> float:
    error = candidate.float() - reference.float()
    return torch.einsum("bi,ij,bj->", error, hessian.float(), error).item()


def _hybrid_error(reference: torch.Tensor, candidate: torch.Tensor, hessian: torch.Tensor) -> float:
    """Independent oracle for the 50/50 diagonal/full-Hessian objective."""

    error = candidate.float() - reference.float()
    diagonal = (error.square() * hessian.float().diagonal()).sum()
    return 0.5 * (diagonal.item() + _quadratic_error(reference, candidate, hessian))


def _run_scalar_activation_reference(weights: torch.Tensor, hessian: torch.Tensor):
    """Retain the pre-vectorization candidate loop as a numerical oracle."""

    qcfg = QuantizeConfig(
        bits=4,
        group_size=weights.shape[1],
        sym=True,
        act_group_aware=False,
        offload_to_disk=False,
        scale_search=ScaleSearchConfig.ACTIVATION,
    )
    quantizer = Quantizer(qcfg)
    quantizer.configure(perchannel=True, grid=100, maxshrink=0.8)
    quantizer.maxq = quantizer.maxq.to(weights.device)

    x = weights.flatten(1)
    zero_range = torch.zeros(x.shape[0], device=x.device)
    xmin = torch.minimum(x.min(dim=1).values, zero_range)
    xmax = torch.maximum(x.max(dim=1).values, zero_range)
    xmax = torch.maximum(xmin.abs(), xmax)
    xmin = torch.where(xmin < 0, -xmax, xmin)
    empty = (xmin == 0) & (xmax == 0)
    xmin = torch.where(empty, -torch.ones_like(xmin), xmin)
    xmax = torch.where(empty, torch.ones_like(xmax), xmax)

    scale = (xmax - xmin) / quantizer.maxq
    zero = torch.full_like(scale, (quantizer.maxq + 1) / 2)
    importance = hessian.detach().to(device=x.device, dtype=torch.float32)
    importance = torch.nan_to_num(importance, nan=0.0, posinf=0.0, neginf=0.0)
    importance = ((importance + importance.t()) * 0.5).diagonal().clamp_min(0)
    importance = importance / importance.mean()
    best = torch.full([x.shape[0]], float("inf"), device=x.device)

    for shrink in 1 - torch.arange(80, device=x.device, dtype=torch.float32) / 100:
        scale_candidate = (shrink * xmax - shrink * xmin) / quantizer.maxq
        levels = torch.clamp(
            torch.round(x / scale_candidate.unsqueeze(1)) + zero.unsqueeze(1),
            0,
            quantizer.maxq,
        )
        candidate = scale_candidate.unsqueeze(1) * (levels - zero.unsqueeze(1))
        error = (candidate - x).float().square().mul(importance).sum(dim=1)
        take = error < best
        if torch.any(take):
            best[take] = error[take]
            scale[take] = scale_candidate[take]

    quantizer.scale = scale.reshape(-1, 1)
    quantizer.zero = zero.reshape(-1, 1)
    return quantizer.quantize(weights), quantizer.scale.clone(), quantizer.zero.clone()


def _run_scalar_correlated_reference(
    weights: torch.Tensor,
    hessian: torch.Tensor,
    method: ScaleSearchConfig,
):
    """Retain the eager one-candidate Hessian loop as a numerical oracle."""

    qcfg = QuantizeConfig(
        bits=4,
        group_size=weights.shape[1],
        sym=True,
        act_group_aware=False,
        offload_to_disk=False,
        scale_search=method,
    )
    quantizer = Quantizer(qcfg)
    quantizer.configure(perchannel=True, grid=100, maxshrink=0.8)
    quantizer.maxq = quantizer.maxq.to(weights.device)

    x = weights.flatten(1)
    zero_range = torch.zeros(x.shape[0], device=x.device)
    xmin = torch.minimum(x.min(dim=1).values, zero_range)
    xmax = torch.maximum(x.max(dim=1).values, zero_range)
    xmax = torch.maximum(xmin.abs(), xmax)
    xmin = torch.where(xmin < 0, -xmax, xmin)
    empty = (xmin == 0) & (xmax == 0)
    xmin = torch.where(empty, -torch.ones_like(xmin), xmin)
    xmax = torch.where(empty, torch.ones_like(xmax), xmax)

    scale = (xmax - xmin) / quantizer.maxq
    zero = torch.full_like(scale, (quantizer.maxq + 1) / 2)
    prepared_hessian = hessian.detach().to(device=x.device, dtype=torch.float32)
    prepared_hessian = torch.nan_to_num(prepared_hessian, nan=0.0, posinf=0.0, neginf=0.0)
    prepared_hessian = (prepared_hessian + prepared_hessian.t()) * 0.5
    prepared_hessian = prepared_hessian / prepared_hessian.diagonal().clamp_min(0).mean()
    best = torch.full([x.shape[0]], float("inf"), device=x.device)

    for shrink in 1 - torch.arange(80, device=x.device, dtype=torch.float32) / 100:
        scale_candidate = (shrink * xmax - shrink * xmin) / quantizer.maxq
        levels = torch.clamp(
            torch.round(x / scale_candidate.unsqueeze(1)) + zero.unsqueeze(1),
            0,
            quantizer.maxq,
        )
        candidate = scale_candidate.unsqueeze(1) * (levels - zero.unsqueeze(1))
        error = (candidate - x).float()
        objective = (error.matmul(prepared_hessian) * error).sum(dim=1)
        if method == ScaleSearchConfig.HYBRID:
            diagonal = (error.square() * prepared_hessian.diagonal().clamp_min(0)).sum(dim=1)
            objective = 0.5 * (objective + diagonal)
        take = objective < best
        if torch.any(take):
            best[take] = objective[take]
            scale[take] = scale_candidate[take]

    quantizer.scale = scale.reshape(-1, 1)
    quantizer.zero = zero.reshape(-1, 1)
    return quantizer.quantize(weights), quantizer.scale.clone(), quantizer.zero.clone()


def test_scale_search_values_are_stable():
    assert ScaleSearchConfig.MSE.value == "mse"
    assert ScaleSearchConfig.ACTIVATION.value == "activation"
    assert ScaleSearchConfig.HESSIAN.value == "hessian"
    assert ScaleSearchConfig.HYBRID.value == "hybrid"


def test_scale_search_config_normalization_and_round_trip():
    default = QuantizeConfig(offload_to_disk=False)
    assert default.scale_search is ScaleSearchConfig.ACTIVATION
    assert default.mse == 2.0
    assert default.to_dict()["meta"]["scale_search"] == "activation"

    reloaded_default = QuantizeConfig.from_quant_config(default.to_dict())
    assert reloaded_default.scale_search is ScaleSearchConfig.ACTIVATION
    assert reloaded_default.mse == 2.0

    disabled = QuantizeConfig(scale_search=None, offload_to_disk=False)
    assert disabled.scale_search is None
    assert disabled.mse == 0.0

    reloaded_disabled = QuantizeConfig.from_quant_config(disabled.to_dict())
    assert reloaded_disabled.scale_search is None
    assert reloaded_disabled.mse == 0.0

    hybrid = QuantizeConfig(scale_search="hybrid", offload_to_disk=False)
    assert hybrid.scale_search is ScaleSearchConfig.HYBRID
    assert hybrid.mse == 2.0
    assert hybrid.to_dict()["meta"]["scale_search"] == "hybrid"

    reloaded_hybrid = QuantizeConfig.from_quant_config(hybrid.to_dict())
    assert reloaded_hybrid.scale_search is ScaleSearchConfig.HYBRID
    assert reloaded_hybrid.mse == 2.0


def test_scale_search_cli_summary_reports_default_and_dynamic_overrides():
    pattern = r"+:^model\.layers\.\d+\.self_attn\.(?:q_proj|k_proj|v_proj|o_proj)$"
    default = QuantizeConfig(offload_to_disk=False)
    mixed = QuantizeConfig(
        scale_search=ScaleSearchConfig.HESSIAN,
        dynamic={pattern: {"scale_search": ScaleSearchConfig.ACTIVATION}},
        offload_to_disk=False,
    )

    assert default.scale_search_cli_summary() == "global=activation; dynamic_overrides=none"
    assert mixed.scale_search_cli_summary() == (
        f"global=hessian; dynamic_overrides=1 [{pattern} -> activation]"
    )


def test_scale_search_startup_log_is_clear_for_cli_users(monkeypatch):
    messages = []
    qcfg = QuantizeConfig(scale_search=None, offload_to_disk=False)
    monkeypatch.setattr(gptq_processor.log, "info", messages.append)

    message = log_scale_search_config(qcfg)

    assert message == "ScaleSearch config: global=disabled; dynamic_overrides=none"
    assert messages == [message]


def test_gptq_processor_logs_scale_search_during_startup(monkeypatch):
    summaries = []
    qcfg = QuantizeConfig(offload_to_disk=False)
    monkeypatch.setattr(
        gptq_processor,
        "log_scale_search_config",
        lambda active_qcfg: summaries.append(active_qcfg.scale_search_cli_summary()),
    )
    def _mock_loop_processor_init(self, **kwargs):
        self.calibration_dataset = []
        self.qcfg = kwargs["qcfg"]

    monkeypatch.setattr(gptq_processor.LoopProcessor, "__init__", _mock_loop_processor_init)

    gptq_processor.GPTQProcessor(
        tokenizer=None,
        qcfg=qcfg,
        calibration=None,
        prepare_dataset_func=None,
        calibration_concat_size=None,
        calibration_sort=None,
        batch_size=1,
    )

    assert summaries == ["global=activation; dynamic_overrides=none"]


def test_legacy_mse_selects_mse_strategy_without_changing_exponent():
    qcfg = QuantizeConfig(mse=2.4, offload_to_disk=False)

    assert qcfg.scale_search is ScaleSearchConfig.MSE
    assert qcfg.mse == 2.4


def test_dynamic_scale_search_override_is_normalized_per_module():
    qcfg = QuantizeConfig(
        mse=2.4,
        dynamic={"+:model.layers.0.self_attn.q_proj": {"scale_search": "hessian"}},
        offload_to_disk=False,
    )

    cloned = clone_gptq_config_for_module(qcfg, "model.layers.0.self_attn.q_proj")

    assert cloned is not None
    assert cloned.scale_search is ScaleSearchConfig.HESSIAN
    assert cloned.mse == 2.0


def test_dynamic_scale_search_none_disables_search_per_module():
    qcfg = QuantizeConfig(
        scale_search="activation",
        dynamic={"+:model.layers.0.self_attn.q_proj": {"scale_search": None}},
        offload_to_disk=False,
    )

    cloned = clone_gptq_config_for_module(qcfg, "model.layers.0.self_attn.q_proj")

    assert cloned is not None
    assert cloned.scale_search is None
    assert cloned.mse == 0.0


def test_dynamic_scale_search_projection_scopes_are_disjoint():
    qcfg = QuantizeConfig(
        scale_search=None,
        dynamic={
            r"+:^model\.layers\.\d+\.self_attn\.(?:q_proj|k_proj|v_proj)$": {
                "scale_search": "activation",
            },
            r"+:^model\.layers\.\d+\.self_attn\.o_proj$": {
                "scale_search": "hybrid",
            },
            r"+:^model\.layers\.\d+\.mlp\.(?:gate_proj|up_proj|down_proj)$": {
                "scale_search": "hessian",
            },
        },
        offload_to_disk=False,
    )

    q_proj = clone_gptq_config_for_module(qcfg, "model.layers.3.self_attn.q_proj")
    o_proj = clone_gptq_config_for_module(qcfg, "model.layers.3.self_attn.o_proj")
    down_proj = clone_gptq_config_for_module(qcfg, "model.layers.3.mlp.down_proj")

    assert q_proj is not None and q_proj.scale_search is ScaleSearchConfig.ACTIVATION
    assert down_proj is not None and down_proj.scale_search is ScaleSearchConfig.HESSIAN
    assert o_proj is not None and o_proj.scale_search is ScaleSearchConfig.HYBRID


def test_qkvo_activation_else_hessian_policy_covers_every_qwen_projection():
    """Keep the combined Qwen policy explicit and prevent unmatched quantized projections."""

    qcfg = QuantizeConfig(
        scale_search=ScaleSearchConfig.HESSIAN,
        dynamic={
            r"+:^model\.layers\.\d+\.self_attn\.(?:q_proj|k_proj|v_proj|o_proj)$": {
                "scale_search": ScaleSearchConfig.ACTIVATION,
            },
        },
        offload_to_disk=False,
    )

    for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
        cloned = clone_gptq_config_for_module(qcfg, f"model.layers.3.self_attn.{projection}")
        assert cloned is not None and cloned.scale_search is ScaleSearchConfig.ACTIVATION
    for projection in ("gate_proj", "up_proj", "down_proj"):
        cloned = clone_gptq_config_for_module(qcfg, f"model.layers.3.mlp.{projection}")
        assert cloned is not None and cloned.scale_search is ScaleSearchConfig.HESSIAN


def test_legacy_dynamic_mse_zero_disables_scale_search_per_module():
    qcfg = QuantizeConfig(
        scale_search="activation",
        dynamic={"+:model.layers.0.self_attn.q_proj": {"mse": 0.0}},
        offload_to_disk=False,
    )

    cloned = clone_gptq_config_for_module(qcfg, "model.layers.0.self_attn.q_proj")

    assert cloned is not None
    assert cloned.scale_search is None
    assert cloned.mse == 0.0


@pytest.mark.parametrize("method", ["activation", "hessian", "hybrid"])
def test_activation_aware_methods_require_squared_error(method):
    with pytest.raises(ValueError, match="require `mse=2.0`"):
        QuantizeConfig(scale_search=method, mse=2.4, offload_to_disk=False)


def test_explicit_mse_strategy_matches_legacy_mse_ab():
    weights = torch.tensor(
        [
            [0.13, -0.72, 0.91, 0.04, 0.27, -0.58],
            [-0.08, 0.18, 0.31, -0.49, 0.07, 0.81],
        ],
        dtype=torch.float32,
    )

    legacy_q, legacy_scale, legacy_zero = _run_scale_search(weights, None, mse=2.0)
    explicit_q, explicit_scale, explicit_zero = _run_scale_search(weights, ScaleSearchConfig.MSE)

    assert torch.equal(explicit_scale, legacy_scale)
    assert torch.equal(explicit_zero, legacy_zero)
    assert torch.equal(explicit_q, legacy_q)


def test_activation_scale_search_reduces_activation_weighted_error_ab():
    weights = torch.tensor(
        [
            [0.04221606, -0.68705058, -0.80969417, 0.04280474, 0.18644902, 0.57719105],
            [-0.06110157, 0.13952501, 0.13988575, -0.04203784, -0.01532374, -0.38614652],
        ],
        dtype=torch.float32,
    )
    hessian = torch.diag(
        torch.tensor([1.0728676, 11.582974, 2.112485, 3.0850899, 0.3360269, 46.948254])
    )

    mse_q, mse_scale, _ = _run_scale_search(weights, ScaleSearchConfig.MSE, hessian=hessian)
    activation_q, activation_scale, _ = _run_scale_search(
        weights,
        ScaleSearchConfig.ACTIVATION,
        hessian=hessian,
    )

    assert not torch.equal(activation_scale, mse_scale)
    assert _quadratic_error(weights, activation_q, hessian) < _quadratic_error(weights, mse_q, hessian)


def test_activation_diagonal_input_matches_full_hessian():
    """Ultra's compact grouped path must preserve activation-search results."""

    weights = torch.tensor(
        [
            [0.04221606, -0.68705058, -0.80969417, 0.04280474, 0.18644902, 0.57719105],
            [-0.06110157, 0.13952501, 0.13988575, -0.04203784, -0.01532374, -0.38614652],
        ],
        dtype=torch.float32,
    )
    diagonal = torch.tensor([1.0728676, 11.582974, 2.112485, 3.0850899, 0.3360269, 46.948254])

    full_q, full_scale, full_zero = _run_scale_search(
        weights,
        ScaleSearchConfig.ACTIVATION,
        hessian=torch.diag(diagonal),
    )
    compact_q, compact_scale, compact_zero = _run_scale_search(
        weights,
        ScaleSearchConfig.ACTIVATION,
        hessian=diagonal,
    )

    assert torch.equal(compact_scale, full_scale)
    assert torch.equal(compact_zero, full_zero)
    assert torch.equal(compact_q, full_q)


@pytest.mark.parametrize(
    ("device", "dtype"),
    [("cpu", torch.float32), ("cuda", torch.bfloat16)],
)
def test_vectorized_activation_search_matches_scalar_reference_ab(device, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is required for the BF16 scale-search A/B")

    generator = torch.Generator(device=device).manual_seed(1907)
    weights = torch.randn((64, 128), generator=generator, device=device, dtype=dtype)
    activations = torch.randn((128, 16), generator=generator, device=device, dtype=torch.float32)
    hessian = activations.matmul(activations.t())

    reference_q, reference_scale, reference_zero = _run_scalar_activation_reference(weights, hessian)
    vectorized_q, vectorized_scale, vectorized_zero = _run_scale_search(
        weights,
        ScaleSearchConfig.ACTIVATION,
        hessian=hessian,
        bits=4,
        grid=100,
    )

    assert torch.equal(vectorized_scale, reference_scale)
    assert torch.equal(vectorized_zero, reference_zero)
    assert torch.equal(vectorized_q, reference_q)


@pytest.mark.parametrize("method", [ScaleSearchConfig.HESSIAN, ScaleSearchConfig.HYBRID])
@pytest.mark.parametrize(
    ("device", "dtype"),
    [("cpu", torch.float32), ("cuda", torch.bfloat16)],
)
def test_vectorized_correlated_search_matches_scalar_reference_ab(method, device, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is required for the BF16 correlated scale-search A/B")

    generator = torch.Generator(device=device).manual_seed(2719)
    weights = torch.randn((64, 128), generator=generator, device=device, dtype=dtype)
    activations = torch.randn((128, 32), generator=generator, device=device, dtype=torch.float32)
    hessian = activations.matmul(activations.t())

    reference_q, reference_scale, reference_zero = _run_scalar_correlated_reference(weights, hessian, method)
    vectorized_q, vectorized_scale, vectorized_zero = _run_scale_search(
        weights,
        method,
        hessian=hessian,
        bits=4,
        grid=100,
    )

    assert torch.equal(vectorized_scale, reference_scale)
    assert torch.equal(vectorized_zero, reference_zero)
    assert torch.equal(vectorized_q, reference_q)


def test_vectorized_shrink_grid_is_bitwise_equal_to_pre_optimization_grid():
    """Candidate construction must preserve the FP32 grid used by 66a14182."""

    expected = 1 - torch.arange(80, dtype=torch.float32) / 100
    actual = Quantizer._scale_search_shrink_factors(80, 100, torch.device("cpu"))

    assert torch.equal(actual, expected)


def test_scale_search_shrink_grid_cache_reuses_exact_task_local_tensor():
    qcfg = QuantizeConfig(scale_search=ScaleSearchConfig.ACTIVATION, offload_to_disk=False)
    quantizer = Quantizer(qcfg)

    first = quantizer._cached_scale_search_shrink_factors(80, 100, torch.device("cpu"))
    second = quantizer._cached_scale_search_shrink_factors(80, 100, torch.device("cpu"))

    assert first is second
    assert torch.equal(first, Quantizer._scale_search_shrink_factors(80, 100, torch.device("cpu")))
    assert all("shrink" not in key for key in quantizer.state_dict())


def test_scale_search_shrink_grid_cache_invalidates_for_changed_grid():
    qcfg = QuantizeConfig(scale_search=ScaleSearchConfig.ACTIVATION, offload_to_disk=False)
    quantizer = Quantizer(qcfg)

    first = quantizer._cached_scale_search_shrink_factors(80, 100, torch.device("cpu"))
    changed = quantizer._cached_scale_search_shrink_factors(80, 200, torch.device("cpu"))

    assert changed is not first
    assert torch.equal(changed, 1 - torch.arange(80, dtype=torch.float32) / 200)


def test_scale_search_shrink_grid_cache_invalidates_for_candidate_count():
    qcfg = QuantizeConfig(scale_search=ScaleSearchConfig.ACTIVATION, offload_to_disk=False)
    quantizer = Quantizer(qcfg)

    first = quantizer._cached_scale_search_shrink_factors(80, 100, torch.device("cpu"))
    changed = quantizer._cached_scale_search_shrink_factors(79, 100, torch.device("cpu"))

    assert changed is not first
    assert torch.equal(changed, 1 - torch.arange(79, dtype=torch.float32) / 100)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two visible CUDA devices")
def test_scale_search_shrink_grid_cache_is_device_local():
    qcfg = QuantizeConfig(scale_search=ScaleSearchConfig.ACTIVATION, offload_to_disk=False)
    quantizer = Quantizer(qcfg)

    first = quantizer._cached_scale_search_shrink_factors(80, 100, torch.device("cuda:0"))
    changed = quantizer._cached_scale_search_shrink_factors(80, 100, torch.device("cuda:1"))

    assert changed is not first
    assert first.device == torch.device("cuda:0")
    assert changed.device == torch.device("cuda:1")


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_batched_activation_search_prepared_hessian_is_bitwise_exact(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is required for the prepared-Hessian A/B")

    generator = torch.Generator(device=device).manual_seed(9102)
    weights = torch.randn((32, 2, 64), generator=generator, device=device, dtype=torch.float32)
    hessian = torch.rand((2, 64), generator=generator, device=device, dtype=torch.float32)
    qcfg = QuantizeConfig(
        bits=4,
        group_size=64,
        sym=True,
        mse=2.0,
        scale_search=ScaleSearchConfig.ACTIVATION,
        scale_search_candidate_chunk_size=80,
        offload_to_disk=False,
    )
    reference = Quantizer(qcfg)
    reference.configure(perchannel=True, grid=100, maxshrink=0.8)
    expected_scale, expected_zero = reference.find_params_batched(
        weights,
        weight=True,
        hessian=hessian,
    )

    candidate = Quantizer(qcfg)
    candidate.configure(perchannel=True, grid=100, maxshrink=0.8)
    prepared = candidate._prepare_scale_search_hessian_batched(
        hessian,
        method=ScaleSearchConfig.ACTIVATION,
    )
    actual_scale, actual_zero = candidate.find_params_batched(
        weights,
        weight=True,
        hessian=prepared,
        hessian_prepared=True,
    )

    assert torch.equal(actual_scale, expected_scale)
    assert torch.equal(actual_zero, expected_zero)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("sym,bits", [(True, 2), (False, 4), (True, 8)])
@pytest.mark.parametrize("importance_kind", ["zero", "constant", "skewed", "invalid"])
def test_prepared_activation_hessian_matches_raw_across_accuracy_spectrum(
    device, dtype, sym, bits, importance_kind
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is required for the prepared-Hessian A/B")

    generator = torch.Generator(device=device).manual_seed(7219)
    storage = torch.randn((3, 2, 128), generator=generator, device=device, dtype=dtype)
    # Exercise the non-contiguous input canonicalization path while retaining
    # an exact raw/prepared comparison for a final partial-like odd row count.
    weights = storage[..., ::2]
    assert not weights.is_contiguous()
    if importance_kind == "zero":
        hessian = torch.zeros((2, 64), device=device)
    elif importance_kind == "constant":
        hessian = torch.full((2, 64), 3.25, device=device)
    elif importance_kind == "skewed":
        hessian = torch.logspace(-6, 6, 128, device=device).reshape(2, 64)
    else:
        hessian = torch.ones((2, 64), device=device)
        hessian[0, 0] = torch.nan
        hessian[0, 1] = torch.inf
        hessian[1, 0] = -4.0

    qcfg = QuantizeConfig(
        bits=bits,
        group_size=64,
        sym=sym,
        mse=2.0,
        scale_search=ScaleSearchConfig.ACTIVATION,
        scale_search_candidate_chunk_size=17,
        offload_to_disk=False,
    )
    raw = Quantizer(qcfg)
    raw.configure(perchannel=True, grid=31, maxshrink=0.8)
    expected_scale, expected_zero = raw.find_params_batched(weights, weight=True, hessian=hessian)

    candidate = Quantizer(qcfg)
    candidate.configure(perchannel=True, grid=31, maxshrink=0.8)
    prepared = candidate._prepare_scale_search_hessian_batched(
        hessian,
        method=ScaleSearchConfig.ACTIVATION,
    )
    actual_scale, actual_zero = candidate.find_params_batched(
        weights,
        weight=True,
        hessian=prepared,
        hessian_prepared=True,
    )

    assert torch.equal(actual_scale, expected_scale)
    assert torch.equal(actual_zero, expected_zero)


@pytest.mark.parametrize(
    "method,hessian,match",
    [
        (ScaleSearchConfig.MSE, torch.ones(2, 64), "only accepts"),
        (ScaleSearchConfig.ACTIVATION, None, "must have shape"),
        (ScaleSearchConfig.ACTIVATION, torch.ones(1, 64), "must have shape"),
        (ScaleSearchConfig.ACTIVATION, torch.ones(2, 64, dtype=torch.bfloat16), "must be FP32"),
    ],
)
def test_prepared_hessian_contract_rejects_ambiguous_or_inexact_state(method, hessian, match):
    qcfg = QuantizeConfig(
        bits=4,
        group_size=64,
        sym=True,
        mse=2.0,
        scale_search=method,
        offload_to_disk=False,
    )
    quantizer = Quantizer(qcfg)
    quantizer.configure(perchannel=True, grid=10, maxshrink=0.8)
    with pytest.raises(ValueError, match=match):
        quantizer.find_params_batched(
            torch.randn(3, 2, 64),
            weight=True,
            hessian=hessian,
            hessian_prepared=True,
        )


@pytest.mark.parametrize("method,mse", [(None, 2.0), ("activation", 2.0), (None, 0.0)])
def test_batched_scale_search_resolves_implicit_string_and_disabled_methods(method, mse):
    qcfg = QuantizeConfig(
        bits=4,
        group_size=64,
        sym=True,
        mse=mse,
        scale_search=ScaleSearchConfig.MSE,
        offload_to_disk=False,
    )
    quantizer = Quantizer(qcfg)
    quantizer.qcfg.scale_search = method
    quantizer.qcfg.mse = mse
    quantizer.configure(perchannel=True, grid=10, maxshrink=0.8)

    scale, zero = quantizer.find_params_batched(
        torch.randn(3, 2, 64),
        weight=True,
        hessian=torch.ones(2, 64),
    )

    assert scale.shape == (3, 2)
    assert zero.shape == (3, 2)


@pytest.mark.parametrize(
    "mse,adaptive_clipping",
    [
        (0.0, AdaptiveClippingConfig(enabled=False)),
        (2.0, AdaptiveClippingConfig(enabled=True, per_group=True)),
    ],
)
def test_prepared_hessian_contract_is_enforced_before_disabled_or_adaptive_paths(mse, adaptive_clipping):
    qcfg = QuantizeConfig(
        bits=4,
        group_size=64,
        sym=True,
        mse=mse,
        scale_search=ScaleSearchConfig.ACTIVATION,
        adaptive_clipping=adaptive_clipping,
        offload_to_disk=False,
    )
    quantizer = Quantizer(qcfg)
    quantizer.configure(perchannel=True, grid=10, maxshrink=0.8)
    hessian = torch.ones(2, 64, dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="active activation|must be FP32"):
        quantizer.find_params_batched(
            torch.randn(3, 2, 64),
            weight=True,
            hessian=hessian,
            hessian_prepared=True,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_prepared_hessian_contract_rejects_cross_device_state():
    qcfg = QuantizeConfig(
        bits=4,
        group_size=64,
        sym=True,
        mse=2.0,
        scale_search=ScaleSearchConfig.ACTIVATION,
        offload_to_disk=False,
    )
    quantizer = Quantizer(qcfg)
    quantizer.configure(perchannel=True, grid=10, maxshrink=0.8)
    with pytest.raises(ValueError, match="colocated"):
        quantizer.find_params_batched(
            torch.randn(3, 2, 64, device="cuda"),
            weight=True,
            hessian=torch.ones(2, 64),
            hessian_prepared=True,
        )


def _small_grouped_activation_gptq(device="cpu", method=ScaleSearchConfig.ACTIVATION, group_size=4):
    torch.manual_seed(8042)
    layer = torch.nn.Linear(10, 5, bias=False, dtype=torch.float32, device=device).eval()
    qcfg = QuantizeConfig(
        bits=4,
        group_size=group_size,
        sym=True,
        act_group_aware=False,
        mse=2.0,
        scale_search=method,
        offload_to_disk=False,
    )
    task = GPTQ(layer, qcfg=qcfg)
    task.quantizer.configure(perchannel=True, grid=20, maxshrink=0.8)
    task.add_batch(torch.randn(3, 10, device=device), None)
    return task


def test_gptq_prepares_full_groups_once_and_keeps_raw_partial_group(monkeypatch):
    task = _small_grouped_activation_gptq()
    prepare_calls = []
    batched_calls = []
    scalar_hessian_shapes = []
    original_prepare = task.quantizer._prepare_scale_search_hessian_batched
    original_batched = task.quantizer.find_params_batched
    original_scalar = task.quantizer.find_params

    def record_prepare(hessian, *, method):
        prepare_calls.append((tuple(hessian.shape), method))
        return original_prepare(hessian, method=method)

    def record_batched(*args, **kwargs):
        batched_calls.append((tuple(kwargs["hessian"].shape), kwargs["hessian_prepared"]))
        return original_batched(*args, **kwargs)

    def record_scalar(*args, **kwargs):
        hessian = kwargs.get("hessian")
        scalar_hessian_shapes.append(None if hessian is None else tuple(hessian.shape))
        return original_scalar(*args, **kwargs)

    monkeypatch.setattr(task.quantizer, "_prepare_scale_search_hessian_batched", record_prepare)
    monkeypatch.setattr(task.quantizer, "find_params_batched", record_batched)
    monkeypatch.setattr(task.quantizer, "find_params", record_scalar)
    result = task.quantize(blocksize=8)

    assert prepare_calls == [((2, 4), ScaleSearchConfig.ACTIVATION)]
    assert batched_calls == [((2, 4), True)]
    assert scalar_hessian_shapes == [(2,)]
    assert torch.isfinite(result[0]).all()


def test_gptq_prepared_hessian_is_bitwise_repeatable_with_partial_group():
    first = _small_grouped_activation_gptq().quantize(blocksize=8)
    second = _small_grouped_activation_gptq().quantize(blocksize=8)

    assert torch.equal(first[0], second[0])
    assert all(torch.equal(left, right) for left, right in zip(first[1], second[1]))
    assert all(torch.equal(left, right) for left, right in zip(first[2], second[2]))
    assert torch.equal(first[3], second[3])
    assert first[5:] == second[5:]


def test_gptq_activation_search_with_only_a_partial_group_stays_on_raw_hessian_path(monkeypatch):
    task = _small_grouped_activation_gptq(group_size=16)
    prepare = Mock(wraps=task.quantizer._prepare_scale_search_hessian_batched)
    monkeypatch.setattr(task.quantizer, "_prepare_scale_search_hessian_batched", prepare)

    result = task.quantize(blocksize=8)

    prepare.assert_not_called()
    assert torch.isfinite(result[0]).all()


def test_gptq_activation_search_fails_closed_to_raw_hessian_when_preparation_is_unavailable(monkeypatch):
    task = _small_grouped_activation_gptq()
    batched_calls = []
    original_batched = task.quantizer.find_params_batched

    monkeypatch.setattr(task.quantizer, "_prepare_scale_search_hessian_batched", lambda *_args, **_kwargs: None)

    def record_batched(*args, **kwargs):
        batched_calls.append((kwargs["hessian"].clone(), kwargs["hessian_prepared"]))
        return original_batched(*args, **kwargs)

    monkeypatch.setattr(task.quantizer, "find_params_batched", record_batched)
    result = task.quantize(blocksize=8)

    assert batched_calls[0][0].shape == (2, 4)
    assert batched_calls[0][1] is False
    assert torch.isfinite(result[0]).all()


def test_gptq_drops_prepared_hessian_when_inverse_is_unavailable(monkeypatch):
    task = _small_grouped_activation_gptq()
    batched_calls = []
    original_batched = task.quantizer.find_params_batched

    monkeypatch.setattr(task, "hessian_inverse", lambda *_args, **_kwargs: (None, 0.0))

    def record_batched(*args, **kwargs):
        batched_calls.append((kwargs["hessian"], kwargs["hessian_prepared"]))
        return original_batched(*args, **kwargs)

    monkeypatch.setattr(task.quantizer, "find_params_batched", record_batched)
    result = task.quantize(blocksize=8)

    assert batched_calls == [(None, False), (None, False)]
    assert torch.isfinite(result[0]).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_gptq_cpu_oom_fallback_moves_prepared_hessian_with_weights(monkeypatch):
    task = _small_grouped_activation_gptq(device="cuda")
    original_inverse = task.hessian_inverse
    inverse_devices = []
    search_devices = []
    original_batched = task.quantizer.find_params_batched

    def fail_once_then_invert(hessian, *args, **kwargs):
        inverse_devices.append(hessian.device.type)
        if len(inverse_devices) == 1:
            raise RuntimeError("CUDA out of memory")
        return original_inverse(hessian, *args, **kwargs)

    monkeypatch.setattr(task, "hessian_inverse", fail_once_then_invert)

    def record_batched(*args, **kwargs):
        search_devices.append((args[0].device.type, kwargs["hessian"].device.type))
        return original_batched(*args, **kwargs)

    monkeypatch.setattr(task.quantizer, "find_params_batched", record_batched)
    result = task.quantize(blocksize=8)

    assert inverse_devices == ["cuda", "cpu"]
    assert search_devices == [("cpu", "cpu")]
    assert result[0].device.type == "cuda"
    assert torch.isfinite(result[0]).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_gptq_correlated_search_cpu_oom_fallback_has_no_prepared_diagonal(monkeypatch):
    task = _small_grouped_activation_gptq(device="cuda", method=ScaleSearchConfig.HESSIAN)
    original_inverse = task.hessian_inverse
    inverse_calls = 0

    def fail_once_then_invert(hessian, *args, **kwargs):
        nonlocal inverse_calls
        inverse_calls += 1
        if inverse_calls == 1:
            raise RuntimeError("CUDA out of memory")
        return original_inverse(hessian, *args, **kwargs)

    monkeypatch.setattr(task, "hessian_inverse", fail_once_then_invert)
    result = task.quantize(blocksize=8)

    assert inverse_calls == 2
    assert result[0].device.type == "cuda"
    assert torch.isfinite(result[0]).all()


def test_correlated_scale_search_uses_larger_bounded_candidate_chunks():
    qcfg = QuantizeConfig(scale_search=ScaleSearchConfig.HESSIAN, offload_to_disk=False)
    quantizer = Quantizer(qcfg)

    expected_chunks = {
        512: (16, 80),
        8192: (16, 80),
        16384: (16, 64),
        65536: (8, 16),
    }
    for rows, (activation_chunk, hessian_chunk) in expected_chunks.items():
        weights = torch.empty((rows, 128))
        assert quantizer._scale_search_candidate_chunk_size(
            weights,
            80,
            ScaleSearchConfig.ACTIVATION,
        ) == activation_chunk
        assert quantizer._scale_search_candidate_chunk_size(
            weights,
            80,
            ScaleSearchConfig.HESSIAN,
        ) == hessian_chunk


def test_explicit_scale_search_candidate_chunk_size_overrides_bounded_policy():
    qcfg = QuantizeConfig(
        scale_search=ScaleSearchConfig.ACTIVATION,
        scale_search_candidate_chunk_size=80,
        offload_to_disk=False,
    )
    quantizer = Quantizer(qcfg)
    weights = torch.empty((65536, 128))

    assert quantizer._scale_search_candidate_chunk_size(weights, 80, ScaleSearchConfig.ACTIVATION) == 80
    assert quantizer._scale_search_candidate_chunk_size(weights, 40, ScaleSearchConfig.ACTIVATION) == 40


@pytest.mark.parametrize("value", [0, -1, False])
def test_scale_search_candidate_chunk_size_rejects_non_positive_values(value):
    with pytest.raises(ValueError, match="positive integer"):
        QuantizeConfig(scale_search_candidate_chunk_size=value, offload_to_disk=False)


@pytest.mark.parametrize("fill_value", [0.0, float("nan")])
def test_activation_search_invalid_hessian_falls_back_to_mse(fill_value):
    weights = torch.tensor([[0.17, -0.91, 0.38, 0.04, -0.55, 0.73]], dtype=torch.float32)
    hessian = torch.full((weights.shape[1], weights.shape[1]), fill_value)

    mse_q, mse_scale, _ = _run_scale_search(weights, ScaleSearchConfig.MSE)
    fallback_q, fallback_scale, _ = _run_scale_search(
        weights,
        ScaleSearchConfig.ACTIVATION,
        hessian=hessian,
    )

    assert torch.equal(fallback_scale, mse_scale)
    assert torch.equal(fallback_q, mse_q)


def test_hessian_scale_search_uses_off_diagonal_correlations_ab():
    weights = torch.tensor(
        [
            [1.8885934, 0.9970499, -0.5151259, 1.7492554, 0.6894920, 2.0634480],
            [-0.0878053, -0.0962911, 2.5050144, 0.9097772, 0.1909927, 3.1856225],
        ],
        dtype=torch.float32,
    )
    hessian = torch.tensor(
        [
            [5.0774817, 0.3797653, 1.1295515, -1.3739724, 4.2334814, 1.4466956],
            [0.3797653, 10.7409353, 8.4417381, 15.9138947, 5.7092881, 19.5332298],
            [1.1295515, 8.4417381, 15.7778111, 13.4487381, 7.3816881, 21.8914757],
            [-1.3739724, 15.9138947, 13.4487381, 24.6508999, 7.2371798, 29.6403923],
            [4.2334814, 5.7092881, 7.3816881, 7.2371798, 6.8846903, 12.6048203],
            [1.4466956, 19.5332298, 21.8914757, 29.6403923, 12.6048203, 40.3601532],
        ],
        dtype=torch.float32,
    )

    activation_q, activation_scale, _ = _run_scale_search(
        weights,
        ScaleSearchConfig.ACTIVATION,
        hessian=hessian,
    )
    hessian_q, hessian_scale, _ = _run_scale_search(
        weights,
        ScaleSearchConfig.HESSIAN,
        hessian=hessian,
    )
    hybrid_q, _, _ = _run_scale_search(
        weights,
        ScaleSearchConfig.HYBRID,
        hessian=hessian,
    )

    assert not torch.equal(hessian_scale, activation_scale)
    assert _quadratic_error(weights, hessian_q, hessian) < _quadratic_error(weights, activation_q, hessian)
    assert _hybrid_error(weights, hybrid_q, hessian) <= _hybrid_error(weights, activation_q, hessian)
    assert _hybrid_error(weights, hybrid_q, hessian) <= _hybrid_error(weights, hessian_q, hessian)


def test_activation_hessian_and_hybrid_match_for_diagonal_hessian():
    weights = torch.tensor([[0.1, -0.8, 0.35, 1.2, -0.42, 0.07]], dtype=torch.float32)
    hessian = torch.diag(torch.tensor([1.0, 7.0, 0.5, 11.0, 2.0, 0.25]))

    activation_q, activation_scale, _ = _run_scale_search(
        weights,
        ScaleSearchConfig.ACTIVATION,
        hessian=hessian,
    )
    hessian_q, hessian_scale, _ = _run_scale_search(
        weights,
        ScaleSearchConfig.HESSIAN,
        hessian=hessian,
    )
    hybrid_q, hybrid_scale, _ = _run_scale_search(
        weights,
        ScaleSearchConfig.HYBRID,
        hessian=hessian,
    )

    assert torch.equal(hessian_scale, activation_scale)
    assert torch.equal(hessian_q, activation_q)
    assert torch.equal(hybrid_scale, activation_scale)
    assert torch.equal(hybrid_q, activation_q)


@pytest.mark.parametrize(
    "method",
    [ScaleSearchConfig.ACTIVATION, ScaleSearchConfig.HESSIAN, ScaleSearchConfig.HYBRID],
)
def test_activation_aware_search_without_hessian_falls_back_to_mse(method):
    weights = torch.tensor([[0.17, -0.91, 0.38, 0.04, -0.55, 0.73]], dtype=torch.float32)

    mse_q, mse_scale, _ = _run_scale_search(weights, ScaleSearchConfig.MSE)
    fallback_q, fallback_scale, _ = _run_scale_search(weights, method)

    assert torch.equal(fallback_scale, mse_scale)
    assert torch.equal(fallback_q, mse_q)


def test_scale_search_rejects_misaligned_hessian_shape():
    weights = torch.tensor([[0.1, -0.2, 0.3, -0.4]], dtype=torch.float32)

    with pytest.raises(ValueError, match="must have shape"):
        _run_scale_search(
            weights,
            ScaleSearchConfig.ACTIVATION,
            hessian=torch.eye(3),
        )
