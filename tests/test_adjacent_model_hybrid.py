# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import copy

import pytest
import torch

from gptqmodel.quantization.adjacent_model import (
    AdjacentModelConfig,
    _adjacent_group_candidate,
    _coordinate_descent_batch_cuda_masked,
    _coordinate_descent_batch_reference,
    _resolve_executor,
    apply_adjacent_model_hybrid,
)
from gptqmodel.quantization.config import AWQConfig, GPTQConfig, METHOD, QuantizeConfig
from gptqmodel.quantization.gptq import GPTQ


def test_adjacent_model_config_rejects_invalid_budgets():
    with pytest.raises(ValueError, match="max_coordinate_flips"):
        AdjacentModelConfig(max_coordinate_flips=0)
    with pytest.raises(ValueError, match="native_refinements_per_module"):
        AdjacentModelConfig(native_refinements_per_module=-1)
    with pytest.raises(ValueError, match="executor"):
        AdjacentModelConfig(executor="tpu")
    with pytest.raises(ValueError, match="cpu_workers"):
        AdjacentModelConfig(cpu_workers=0)
    with pytest.raises(ValueError, match="cpu_min_row_groups"):
        AdjacentModelConfig(cpu_min_row_groups=0)
    with pytest.raises(ValueError, match="activation_chunk_size"):
        AdjacentModelConfig(activation_chunk_size=0)
    with pytest.raises(TypeError, match="dictionary"):
        AdjacentModelConfig.from_dict([])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unknown fields"):
        AdjacentModelConfig.from_dict({"future_policy": True})
    with pytest.raises(TypeError, match="not a string"):
        AdjacentModelConfig(coordinate_starts="nearest")  # type: ignore[arg-type]
    config = AdjacentModelConfig()
    assert copy.deepcopy(config) is config


def test_adjacent_model_is_an_opt_in_serialized_gptq_or_awq_config():
    adjacent_model = AdjacentModelConfig(
        coordinate_starts=("linear", "nearest"),
        max_coordinate_flips=17,
        coordinate_rebase_interval=5,
        batch_coordinate_starts_on_cuda=False,
        executor="auto",
        row_chunk_size=321,
        cpu_row_chunk_size=123,
        cpu_workers=7,
        cpu_min_row_groups=456,
        objective_row_chunk_size=89,
        activation_chunk_size=144,
        native_refinements_per_module=3,
        native_split_depth=4,
        native_max_nodes_per_worker=211,
        certificate_tolerance=2e-11,
        selection_tolerance=3e-6,
    )
    gptq_qcfg = QuantizeConfig(
        bits=4,
        group_size=128,
        desc_act=False,
        adjacent_model=adjacent_model,
    )
    awq_qcfg = QuantizeConfig(
        bits=4,
        group_size=128,
        method=METHOD.AWQ,
        adjacent_model=adjacent_model,
    )

    assert GPTQConfig().adjacent_model is None
    assert AWQConfig().adjacent_model is None
    assert gptq_qcfg.adjacent_model is adjacent_model
    assert awq_qcfg.adjacent_model is adjacent_model
    assert copy.deepcopy(gptq_qcfg).adjacent_model is adjacent_model
    assert copy.deepcopy(awq_qcfg).adjacent_model is adjacent_model
    assert "adjacent_model" not in GPTQConfig().to_dict().get("meta", {})
    assert "adjacent_model" not in AWQConfig().to_dict().get("meta", {})
    for qcfg in (gptq_qcfg, awq_qcfg):
        payload = qcfg.to_dict()
        assert "adjacent_model" not in payload
        assert payload["meta"]["adjacent_model"] == adjacent_model.to_dict()
        loaded = QuantizeConfig.from_quant_config(payload)
        assert isinstance(loaded.adjacent_model, AdjacentModelConfig)
        assert loaded.adjacent_model is not adjacent_model
        assert loaded.adjacent_model.to_dict() == adjacent_model.to_dict()
        assert loaded.adjacent_model.snapshot() == []
    for legacy_name in ("_adjacent_model_config", "adjacent_config"):
        with pytest.raises(ValueError, match=f"{legacy_name}.*adjacent_model"):
            QuantizeConfig(**{legacy_name: adjacent_model})
    with pytest.raises(ValueError, match="adjacent_model.*GPTQ and AWQ"):
        QuantizeConfig(method=METHOD.FP8, adjacent_model=adjacent_model)
    unsupported_payload = QuantizeConfig(method=METHOD.FP8).to_dict()
    unsupported_payload.setdefault("meta", {})["adjacent_model"] = adjacent_model.to_dict()
    with pytest.raises(ValueError, match="adjacent_model.*GPTQ and AWQ"):
        QuantizeConfig.from_quant_config(unsupported_payload)


def test_auto_executor_is_conservative_and_runtime_probed():
    config = AdjacentModelConfig(
        executor="auto",
        cpu_workers=8,
        cpu_min_row_groups=100,
    )
    assert _resolve_executor(
        config,
        row_group_count=100,
        bits=4,
        group_size=128,
        gil_enabled=False,
        available_cpu_count=8,
    ) == "cpu"
    assert _resolve_executor(
        config,
        row_group_count=99,
        bits=4,
        group_size=128,
        gil_enabled=False,
        available_cpu_count=8,
    ) == "cuda"
    assert _resolve_executor(
        config,
        row_group_count=100,
        bits=4,
        group_size=128,
        gil_enabled=True,
        available_cpu_count=8,
    ) == "cuda"
    assert _resolve_executor(
        config,
        row_group_count=100,
        bits=4,
        group_size=128,
        gil_enabled=False,
        available_cpu_count=7,
    ) == "cuda"
    assert _resolve_executor(
        config,
        row_group_count=100,
        bits=3,
        group_size=128,
        gil_enabled=False,
        available_cpu_count=8,
    ) == "cuda"
    assert _resolve_executor(
        config,
        row_group_count=100,
        bits=4,
        group_size=64,
        gil_enabled=False,
        available_cpu_count=8,
    ) == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_masked_coordinate_descent_exactly_matches_dynamic_reference():
    device = torch.device("cuda")
    generator = torch.Generator().manual_seed(898)
    rows = 19
    columns = 13
    residual = torch.randn(rows, columns, generator=generator, dtype=torch.float64).to(device)
    steps = (0.2 * torch.rand(rows, columns, generator=generator, dtype=torch.float64)).to(device)
    steps[:, -1] = 0
    activations = torch.randn(48, columns, generator=generator, dtype=torch.float64).to(device)
    hessian = activations.mT @ activations / activations.shape[0]
    initial_state = torch.randint(
        0,
        2,
        (rows, columns),
        generator=generator,
        dtype=torch.int64,
    ).to(device=device, dtype=torch.float64)

    kwargs = {
        "residual": residual,
        "steps": steps,
        "hessian": hessian,
        "initial_state": initial_state,
        "max_flips": 16,
        "rebase_interval": 4,
    }
    expected = _coordinate_descent_batch_reference(**kwargs)
    actual = _coordinate_descent_batch_cuda_masked(**kwargs)
    for expected_tensor, actual_tensor in zip(expected, actual, strict=True):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0.0, atol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("bits", [2, 3, 4, 8])
def test_cuda_batched_coordinate_starts_exactly_match_sequential_starts(bits: int):
    device = torch.device("cuda")
    generator = torch.Generator().manual_seed(1000 + bits)
    rows = 23
    columns = 16
    weight = torch.randn(rows, columns, generator=generator, dtype=torch.float32).to(device)
    activations = torch.randn(64, columns, generator=generator, dtype=torch.float64).to(device)
    hessian = activations.mT @ activations / activations.shape[0]
    maxq = (1 << bits) - 1
    absolute_max = weight.abs().amax(dim=1, keepdim=True).clamp_min_(1e-8)
    scale = 2.0 * absolute_max / maxq
    zero = torch.full_like(scale, (maxq + 1) / 2)

    kwargs = {
        "weight": weight,
        "hessian": hessian,
        "scale": scale,
        "zero": zero,
        "bits": bits,
    }
    sequential = _adjacent_group_candidate(
        **kwargs,
        config=AdjacentModelConfig(
            max_coordinate_flips=16,
            coordinate_rebase_interval=4,
            batch_coordinate_starts_on_cuda=False,
        ),
    )
    batched = _adjacent_group_candidate(
        **kwargs,
        config=AdjacentModelConfig(
            max_coordinate_flips=16,
            coordinate_rebase_interval=4,
            batch_coordinate_starts_on_cuda=True,
        ),
    )
    for expected_tensor, actual_tensor in zip(sequential, batched, strict=True):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0.0, atol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_adjacent_model_hybrid_is_packable_deterministic_and_no_worse_than_classic():
    device = torch.device("cuda")
    generator = torch.Generator().manual_seed(20260723)
    rows = 11
    columns = 16
    group_size = 8
    weight = torch.randn(rows, columns, generator=generator, dtype=torch.float32).to(device)
    activations = torch.randn(64, columns, generator=generator, dtype=torch.float32).to(device)
    hessian = activations.mT @ activations / activations.shape[0]
    scales = [
        torch.full((rows, 1), 0.2, device=device),
        torch.full((rows, 1), 0.25, device=device),
    ]
    zeros = [torch.full((rows, 1), 8.0, device=device) for _ in range(2)]
    scale_tensor = torch.cat(scales, dim=1)
    zero_tensor = torch.cat(zeros, dim=1)
    group_ids = torch.arange(columns, device=device) // group_size
    per_column_scale = scale_tensor.index_select(1, group_ids)
    per_column_zero = zero_tensor.index_select(1, group_ids)
    classic = (
        torch.round(weight / per_column_scale + per_column_zero)
        .clamp_(0, 15)
        .sub_(per_column_zero)
        .mul_(per_column_scale)
        .to(torch.bfloat16)
    )
    config = AdjacentModelConfig(
        max_coordinate_flips=16,
        row_chunk_size=6,
        objective_row_chunk_size=5,
        native_refinements_per_module=2,
        native_split_depth=3,
        native_max_nodes_per_worker=100,
    )

    first, first_stats = apply_adjacent_model_hybrid(
        module_name="test.linear",
        weight=weight,
        hessian=hessian,
        classic_quantized=classic,
        scale_parts=scales,
        zero_parts=zeros,
        bits=4,
        group_size=group_size,
        config=config,
    )
    second, second_stats = apply_adjacent_model_hybrid(
        module_name="test.linear",
        weight=weight,
        hessian=hessian,
        classic_quantized=classic,
        scale_parts=scales,
        zero_parts=zeros,
        bits=4,
        group_size=group_size,
        config=config,
    )

    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
    assert first.dtype == classic.dtype
    codes = first.to(torch.float32) / per_column_scale + per_column_zero
    torch.testing.assert_close(codes, codes.round(), rtol=0.0, atol=2e-2)
    assert first_stats["hybrid_full_hessian_error"] <= first_stats["classic_full_hessian_error"] + 1e-8
    assert first_stats["native_refinements_attempted"] == 2
    assert first_stats["hybrid_selected_rows"] == second_stats["hybrid_selected_rows"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_parallel_cpu_executor_matches_cuda_hybrid():
    device = torch.device("cuda")
    generator = torch.Generator().manual_seed(1234)
    rows = 17
    columns = 24
    group_size = 8
    bits = 4
    weight = torch.randn(rows, columns, generator=generator, dtype=torch.float32).to(device)
    activations = torch.randn(96, columns, generator=generator, dtype=torch.float32).to(device)
    hessian = activations.mT @ activations / activations.shape[0]
    scales = [torch.full((rows, 1), 0.2 + 0.025 * index, device=device) for index in range(3)]
    zeros = [torch.full((rows, 1), 8.0, device=device) for _ in range(3)]
    scale_tensor = torch.cat(scales, dim=1)
    zero_tensor = torch.cat(zeros, dim=1)
    group_ids = torch.arange(columns, device=device) // group_size
    per_column_scale = scale_tensor.index_select(1, group_ids)
    per_column_zero = zero_tensor.index_select(1, group_ids)
    classic = (
        torch.round(weight / per_column_scale + per_column_zero)
        .clamp_(0, (1 << bits) - 1)
        .sub_(per_column_zero)
        .mul_(per_column_scale)
        .to(torch.bfloat16)
    )
    common = {
        "max_coordinate_flips": 16,
        "coordinate_rebase_interval": 4,
        "objective_row_chunk_size": 7,
        "native_refinements_per_module": 0,
    }
    cuda_config = AdjacentModelConfig(
        **common,
        executor="cuda",
        row_chunk_size=5,
    )
    cpu_config = AdjacentModelConfig(
        **common,
        executor="cpu",
        cpu_row_chunk_size=5,
        cpu_workers=2,
    )
    kwargs = {
        "module_name": "test.linear",
        "weight": weight,
        "hessian": hessian,
        "classic_quantized": classic,
        "scale_parts": scales,
        "zero_parts": zeros,
        "bits": bits,
        "group_size": group_size,
    }

    cuda_hybrid, cuda_stats = apply_adjacent_model_hybrid(**kwargs, config=cuda_config)
    cpu_hybrid, cpu_stats = apply_adjacent_model_hybrid(**kwargs, config=cpu_config)

    torch.testing.assert_close(cpu_hybrid, cuda_hybrid, rtol=0.0, atol=0.0)
    assert cpu_stats["executor"] == "cpu"
    assert cpu_stats["executor_requested"] == "cpu"
    assert cpu_stats["candidate_workers"] == 2
    assert cpu_stats["candidate_row_chunk_size"] == 5
    assert cuda_stats["executor"] == "cuda"
    assert cpu_stats["coordinate_converged_row_groups"] == cuda_stats["coordinate_converged_row_groups"]
    assert cpu_stats["coordinate_total_flips"] == cuda_stats["coordinate_total_flips"]
    assert cpu_stats["hybrid_selected_rows"] == cuda_stats["hybrid_selected_rows"]
    assert cpu_stats["local_group_hessian_error"] == pytest.approx(
        cuda_stats["local_group_hessian_error"],
        rel=0.0,
        abs=1e-12,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_gptq_adjacent_model_hook_preserves_scales_and_records_full_hessian_improvement(monkeypatch):
    device = torch.device("cuda")
    generator = torch.Generator().manual_seed(77)
    source = torch.nn.Linear(16, 9, bias=False, dtype=torch.float16).to(device)
    source.weight.data.copy_(
        torch.randn(source.weight.shape, generator=generator, dtype=torch.float32).to(device)
    )
    classic_module = copy.deepcopy(source)
    adjacent_module = copy.deepcopy(source)
    calibration = torch.randn(96, 16, generator=generator, dtype=torch.float16).to(device)

    classic_config = QuantizeConfig(
        bits=4,
        group_size=8,
        desc_act=False,
        act_group_aware=True,
        sym=True,
        offload_to_disk=False,
    )
    adjacent_qcfg = copy.deepcopy(classic_config)
    adjacent_model = AdjacentModelConfig(
        max_coordinate_flips=16,
        row_chunk_size=9,
        objective_row_chunk_size=9,
        native_refinements_per_module=0,
    )
    adjacent_qcfg.adjacent_model = adjacent_model

    classic_task = GPTQ(classic_module, qcfg=classic_config)
    adjacent_task = GPTQ(adjacent_module, qcfg=adjacent_qcfg)
    classic_task.quantizer.configure(perchannel=True)
    adjacent_task.quantizer.configure(perchannel=True)
    classic_task.add_batch(calibration, classic_module(calibration))
    adjacent_task.add_batch(calibration, adjacent_module(calibration))
    monkeypatch.setattr(
        "gptqmodel.quantization.adjacent_model.apply_adjacent_model_hybrid",
        lambda **_kwargs: pytest.fail("disabled GPTQ unexpectedly invoked AdjacentExact"),
    )
    classic_quantized, classic_scales, classic_zeros, classic_g_idx, *_ = classic_task.quantize()
    monkeypatch.setattr(
        "gptqmodel.quantization.adjacent_model.apply_adjacent_model_hybrid",
        apply_adjacent_model_hybrid,
    )
    adjacent_quantized, adjacent_scales, adjacent_zeros, adjacent_g_idx, *_ = adjacent_task.quantize()

    torch.testing.assert_close(adjacent_scales, classic_scales, rtol=0.0, atol=0.0)
    torch.testing.assert_close(adjacent_zeros, classic_zeros, rtol=0.0, atol=0.0)
    torch.testing.assert_close(adjacent_g_idx, classic_g_idx, rtol=0.0, atol=0.0)
    assert adjacent_quantized.shape == classic_quantized.shape == source.weight.shape
    stats = adjacent_model.snapshot()
    assert len(stats) == 1
    assert stats[0]["module"] == "hf_optimum"
    assert stats[0]["hybrid_full_hessian_error"] <= stats[0]["classic_full_hessian_error"] + 1e-8
