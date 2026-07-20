# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from gptqmodel.looper.gptq_processor import clone_gptq_config_for_module
from gptqmodel.quantization import QuantizeConfig, Quantizer, ScaleSearch


def _run_scale_search(
    weights: torch.Tensor,
    method: ScaleSearch | str | None,
    *,
    hessian: torch.Tensor | None = None,
    mse: float = 0.0,
):
    qcfg = QuantizeConfig(
        bits=2,
        group_size=weights.shape[1],
        sym=True,
        act_group_aware=False,
        offload_to_disk=False,
        mse=mse,
        scale_search=method,
    )
    quantizer = Quantizer(qcfg)
    quantizer.configure(perchannel=True, grid=20, maxshrink=0.8)
    quantizer.find_params(weights, weight=True, hessian=hessian)
    return quantizer.quantize(weights), quantizer.scale.clone(), quantizer.zero.clone()


def _quadratic_error(reference: torch.Tensor, candidate: torch.Tensor, hessian: torch.Tensor) -> float:
    error = candidate.float() - reference.float()
    return torch.einsum("bi,ij,bj->", error, hessian.float(), error).item()


def test_scale_search_values_are_stable():
    assert ScaleSearch.MSE.value == "mse"
    assert ScaleSearch.ACTIVATION.value == "activation"
    assert ScaleSearch.HESSIAN.value == "hessian"


def test_scale_search_config_normalization_and_round_trip():
    disabled = QuantizeConfig(offload_to_disk=False)
    assert disabled.scale_search is None
    assert disabled.mse == 0.0

    activation = QuantizeConfig(scale_search="activation", offload_to_disk=False)
    assert activation.scale_search is ScaleSearch.ACTIVATION
    assert activation.mse == 2.0
    assert activation.to_dict()["meta"]["scale_search"] == "activation"

    reloaded = QuantizeConfig.from_quant_config(activation.to_dict())
    assert reloaded.scale_search is ScaleSearch.ACTIVATION
    assert reloaded.mse == 2.0


def test_legacy_mse_selects_mse_strategy_without_changing_exponent():
    qcfg = QuantizeConfig(mse=2.4, offload_to_disk=False)

    assert qcfg.scale_search is ScaleSearch.MSE
    assert qcfg.mse == 2.4


def test_dynamic_scale_search_override_is_normalized_per_module():
    qcfg = QuantizeConfig(
        mse=2.4,
        dynamic={"+:model.layers.0.self_attn.q_proj": {"scale_search": "hessian"}},
        offload_to_disk=False,
    )

    cloned = clone_gptq_config_for_module(qcfg, "model.layers.0.self_attn.q_proj")

    assert cloned is not None
    assert cloned.scale_search is ScaleSearch.HESSIAN
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


@pytest.mark.parametrize("method", ["activation", "hessian"])
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
    explicit_q, explicit_scale, explicit_zero = _run_scale_search(weights, ScaleSearch.MSE)

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

    mse_q, mse_scale, _ = _run_scale_search(weights, ScaleSearch.MSE, hessian=hessian)
    activation_q, activation_scale, _ = _run_scale_search(
        weights,
        ScaleSearch.ACTIVATION,
        hessian=hessian,
    )

    assert not torch.equal(activation_scale, mse_scale)
    assert _quadratic_error(weights, activation_q, hessian) < _quadratic_error(weights, mse_q, hessian)


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
        ScaleSearch.ACTIVATION,
        hessian=hessian,
    )
    hessian_q, hessian_scale, _ = _run_scale_search(
        weights,
        ScaleSearch.HESSIAN,
        hessian=hessian,
    )

    assert not torch.equal(hessian_scale, activation_scale)
    assert _quadratic_error(weights, hessian_q, hessian) < _quadratic_error(weights, activation_q, hessian)


def test_activation_and_hessian_match_for_diagonal_hessian():
    weights = torch.tensor([[0.1, -0.8, 0.35, 1.2, -0.42, 0.07]], dtype=torch.float32)
    hessian = torch.diag(torch.tensor([1.0, 7.0, 0.5, 11.0, 2.0, 0.25]))

    activation_q, activation_scale, _ = _run_scale_search(
        weights,
        ScaleSearch.ACTIVATION,
        hessian=hessian,
    )
    hessian_q, hessian_scale, _ = _run_scale_search(
        weights,
        ScaleSearch.HESSIAN,
        hessian=hessian,
    )

    assert torch.equal(hessian_scale, activation_scale)
    assert torch.equal(hessian_q, activation_q)


@pytest.mark.parametrize("method", [ScaleSearch.ACTIVATION, ScaleSearch.HESSIAN])
def test_activation_aware_search_without_hessian_falls_back_to_mse(method):
    weights = torch.tensor([[0.17, -0.91, 0.38, 0.04, -0.55, 0.73]], dtype=torch.float32)

    mse_q, mse_scale, _ = _run_scale_search(weights, ScaleSearch.MSE)
    fallback_q, fallback_scale, _ = _run_scale_search(weights, method)

    assert torch.equal(fallback_scale, mse_scale)
    assert torch.equal(fallback_q, mse_q)


def test_scale_search_rejects_misaligned_hessian_shape():
    weights = torch.tensor([[0.1, -0.2, 0.3, -0.4]], dtype=torch.float32)

    with pytest.raises(ValueError, match="must have shape"):
        _run_scale_search(
            weights,
            ScaleSearch.ACTIVATION,
            hessian=torch.eye(3),
        )
