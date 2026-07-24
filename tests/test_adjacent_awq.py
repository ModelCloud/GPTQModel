# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import types

import pytest
import torch
from torch import nn

from gptqmodel.looper.awq_processor import AWQProcessor
from gptqmodel.looper.named_module import NamedModule
from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
from gptqmodel.quantization.adjacent_model import AdjacentModelConfig, apply_adjacent_awq_hybrid
from gptqmodel.quantization.awq.utils.packing_utils import dequantize_gemm
from gptqmodel.quantization.config import AWQConfig, FORMAT


class _TestAWQProcessor(AWQProcessor):
    def __init__(self, qcfg: AWQConfig):
        super().__init__(
            tokenizer=None,
            qcfg=qcfg,
            calibration=None,
            prepare_dataset_func=None,
            calibration_concat_size=None,
            calibration_sort=None,
            batch_size=1,
            gptq_model=types.SimpleNamespace(rotary_embedding=None),
            model=None,
            require_fwd=True,
            calculate_w_wq_diff=False,
            calibration_concat_separator=None,
        )


def test_awq_adjacent_processor_validates_explicit_runtime_config():
    with pytest.raises(TypeError, match="AdjacentModelConfig"):
        _TestAWQProcessor(
            AWQConfig(bits=4, group_size=16, format=FORMAT.GEMM, adjacent_model=object())
        )
    with pytest.raises(ValueError, match=r"group_size.*\[1, 128\]"):
        _TestAWQProcessor(
            AWQConfig(
                bits=4,
                group_size=256,
                format=FORMAT.GEMM,
                adjacent_model=AdjacentModelConfig(),
            )
        )


def _affine_quantize(
    weight: torch.Tensor,
    *,
    bits: int,
    group_size: int,
    sym: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows, columns = weight.shape
    grouped = weight.to(torch.float32).reshape(rows, columns // group_size, group_size)
    maxq = (1 << bits) - 1
    if sym:
        max_int = (1 << (bits - 1)) - 1
        scales = grouped.abs().amax(dim=2).clamp_min_(1e-5) / max_int
        zeros = torch.full_like(scales, 1 << (bits - 1))
    else:
        maximum = grouped.amax(dim=2)
        minimum = grouped.amin(dim=2)
        scales = (maximum - minimum).clamp_min_(1e-5) / maxq
        zeros = (-torch.round(minimum / scales)).clamp_(0, maxq)
    scales = scales.to(weight.dtype)
    zeros = zeros.to(weight.dtype)

    column_scales = scales.repeat_interleave(group_size, dim=1)
    column_zeros = zeros.repeat_interleave(group_size, dim=1)
    quantized = (
        torch.round(weight.to(torch.float32) / column_scales + column_zeros)
        .clamp_(0, maxq)
        .sub_(column_zeros)
        .mul_(column_scales)
        .to(weight.dtype)
    )
    return quantized, scales, zeros


def test_awq_adjacent_reference_capture_is_disabled_by_default_and_precedes_clipping(monkeypatch):
    linear = nn.Linear(16, 8, bias=False)
    named = NamedModule(linear, name="proj", full_name="layers.0.proj", layer_index=0)
    baseline = _TestAWQProcessor(AWQConfig(bits=4, group_size=16, format=FORMAT.GEMM))

    assert baseline.adjacent_model is None
    assert baseline._capture_adjacent_reference_weights({"proj": named}) == {}
    monkeypatch.setattr(
        "gptqmodel.quantization.adjacent_model.apply_adjacent_awq_hybrid",
        lambda **_kwargs: pytest.fail("disabled AWQ unexpectedly invoked AdjacentExact"),
    )
    expected_baseline = baseline.pseudo_quantize_tensor(linear.weight.detach())[0]
    baseline.apply_quant({"proj": named}, scales_list=[])
    torch.testing.assert_close(linear.weight, expected_baseline, rtol=0.0, atol=0.0)

    adjacent = AdjacentModelConfig(native_refinements_per_module=0)
    processor = _TestAWQProcessor(
        AWQConfig(bits=4, group_size=16, format=FORMAT.GEMM, adjacent_model=adjacent)
    )
    linear.weight.data.normal_()
    expected = linear.weight.detach().clone()
    references = processor._capture_adjacent_reference_weights({"proj": named})
    linear.weight.data.clamp_(-0.01, 0.01)

    assert references["proj"].device.type == "cpu"
    torch.testing.assert_close(references["proj"], expected, rtol=0.0, atol=0.0)
    assert not torch.equal(references["proj"], linear.weight)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("bits", [2, 3, 4, 8])
@pytest.mark.parametrize("sym", [True, False])
def test_awq_adjacent_hybrid_is_on_grid_and_no_worse_than_rtn(bits: int, sym: bool):
    device = torch.device("cuda")
    generator = torch.Generator().manual_seed(47_000 + bits + int(sym))
    rows = 12
    columns = 32
    group_size = 16
    samples = 83
    reference = torch.randn(rows, columns, generator=generator, dtype=torch.float32).to(
        device=device,
        dtype=torch.float16,
    )
    latent = torch.randn(samples, 5, generator=generator, dtype=torch.float32)
    mixing = torch.randn(5, columns, generator=generator, dtype=torch.float32)
    activations = latent @ mixing + 0.03 * torch.randn(
        samples,
        columns,
        generator=generator,
        dtype=torch.float32,
    )
    classic, scales, zeros = _affine_quantize(
        reference,
        bits=bits,
        group_size=group_size,
        sym=sym,
    )
    config = AdjacentModelConfig(
        executor="cuda",
        max_coordinate_flips=32,
        row_chunk_size=7,
        objective_row_chunk_size=5,
        activation_chunk_size=17,
        native_refinements_per_module=0,
    )

    hybrid, stats = apply_adjacent_awq_hybrid(
        module_name="layers.0.proj",
        reference_weight=reference,
        activations=activations,
        classic_quantized=classic,
        scales=scales,
        zeros=zeros,
        bits=bits,
        group_size=group_size,
        config=config,
    )

    column_scales = scales.repeat_interleave(group_size, dim=1)
    column_zeros = zeros.repeat_interleave(group_size, dim=1)
    codes = hybrid.to(torch.float32) / column_scales + column_zeros
    rounded_codes = codes.round()
    assert float((codes - rounded_codes).abs().amax()) < 0.125
    reconstructed = ((rounded_codes - column_zeros) * column_scales).to(hybrid.dtype)
    torch.testing.assert_close(reconstructed, hybrid, rtol=0.0, atol=0.0)
    assert int(rounded_codes.amin().item()) >= 0
    assert int(rounded_codes.amax().item()) <= (1 << bits) - 1

    activation_device = activations.to(device)
    classic_error = activation_device @ (reference.to(torch.float32) - classic.to(torch.float32)).mT
    hybrid_error = activation_device @ (reference.to(torch.float32) - hybrid.to(torch.float32)).mT
    assert float(hybrid_error.square().sum()) <= float(classic_error.square().sum()) + 1e-4
    assert stats["quant_method"] == "awq"
    assert stats["status"] == "applied"
    assert stats["reference_stage"] == "post_scale_pre_clip"
    assert stats["baseline"] == "awq_post_clip_rtn"
    assert stats["activation_samples"] == samples
    assert stats["hybrid_activation_error"] <= stats["classic_activation_error"] + 1e-8


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("bits", [2, 3])
@pytest.mark.parametrize("group_size", [32, 64, 128])
def test_awq_adjacent_ultralow_bits_support_native_symmetric_group_sizes(bits: int, group_size: int):
    generator = torch.Generator().manual_seed(910_000 + 100 * bits + group_size)
    device = torch.device("cuda")
    rows = 4
    reference = torch.randn(rows, group_size, generator=generator, dtype=torch.float32).to(
        device=device,
        dtype=torch.float16,
    )
    activations = torch.randn(73, group_size, generator=generator, dtype=torch.float32)
    classic, scales, zeros = _affine_quantize(
        reference,
        bits=bits,
        group_size=group_size,
        sym=True,
    )

    hybrid, stats = apply_adjacent_awq_hybrid(
        module_name=f"layers.0.group_{group_size}",
        reference_weight=reference,
        activations=activations,
        classic_quantized=classic,
        scales=scales,
        zeros=zeros,
        bits=bits,
        group_size=group_size,
        config=AdjacentModelConfig(
            executor="cuda",
            max_coordinate_flips=24,
            row_chunk_size=4,
            objective_row_chunk_size=4,
            activation_chunk_size=19,
            native_refinements_per_module=0,
        ),
    )

    activation_device = activations.to(device)
    classic_error = activation_device @ (reference.to(torch.float32) - classic.to(torch.float32)).mT
    hybrid_error = activation_device @ (reference.to(torch.float32) - hybrid.to(torch.float32)).mT
    assert float(hybrid_error.square().sum()) <= float(classic_error.square().sum()) + 1e-4
    assert stats["group_size"] == group_size
    assert stats["group_count"] == 1
    assert stats["quant_method"] == "awq"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_awq_adjacent_cpu_and_cuda_executors_match():
    generator = torch.Generator().manual_seed(8181)
    device = torch.device("cuda")
    reference = torch.randn(9, 32, generator=generator, dtype=torch.float32).to(
        device=device,
        dtype=torch.float16,
    )
    activations = torch.randn(67, 32, generator=generator, dtype=torch.float32)
    classic, scales, zeros = _affine_quantize(
        reference,
        bits=4,
        group_size=16,
        sym=True,
    )
    common = {
        "max_coordinate_flips": 24,
        "objective_row_chunk_size": 5,
        "activation_chunk_size": 19,
        "native_refinements_per_module": 0,
    }
    kwargs = {
        "module_name": "layers.0.proj",
        "reference_weight": reference,
        "activations": activations,
        "classic_quantized": classic,
        "scales": scales,
        "zeros": zeros,
        "bits": 4,
        "group_size": 16,
    }

    cuda_hybrid, cuda_stats = apply_adjacent_awq_hybrid(
        **kwargs,
        config=AdjacentModelConfig(executor="cuda", row_chunk_size=5, **common),
    )
    cpu_hybrid, cpu_stats = apply_adjacent_awq_hybrid(
        **kwargs,
        config=AdjacentModelConfig(
            executor="cpu",
            cpu_row_chunk_size=5,
            cpu_workers=2,
            **common,
        ),
    )

    torch.testing.assert_close(cpu_hybrid, cuda_hybrid, rtol=0.0, atol=0.0)
    assert cuda_stats["executor"] == "cuda"
    assert cpu_stats["executor"] == "cpu"
    assert cpu_stats["candidate_workers"] == 2
    assert cpu_stats["hybrid_selected_rows"] == cuda_stats["hybrid_selected_rows"]
    assert cpu_stats["hybrid_activation_error"] == pytest.approx(
        cuda_stats["hybrid_activation_error"],
        rel=0.0,
        abs=1e-12,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_awq_processor_applies_adjacent_then_preserves_awq_pack_dequant_contract():
    device = torch.device("cuda")
    generator = torch.Generator().manual_seed(20260724)
    bits = 4
    group_size = 16
    in_features = 32
    out_features = 32
    adjacent = AdjacentModelConfig(
        executor="cuda",
        max_coordinate_flips=32,
        row_chunk_size=8,
        objective_row_chunk_size=8,
        activation_chunk_size=19,
        native_refinements_per_module=0,
    )
    processor = _TestAWQProcessor(
        AWQConfig(
            bits=bits,
            group_size=group_size,
            sym=True,
            format=FORMAT.GEMM,
            adjacent_model=adjacent,
        )
    )
    linear = nn.Linear(
        in_features,
        out_features,
        bias=False,
        device=device,
        dtype=torch.float16,
    )
    linear.weight.data.copy_(
        torch.randn(
            linear.weight.shape,
            generator=generator,
            dtype=torch.float32,
        ).to(device)
    )
    named = NamedModule(linear, name="proj", full_name="layers.0.proj", layer_index=0)
    reference = linear.weight.detach().cpu().clone()
    linear.weight.data.clamp_(-0.7, 0.7)
    classic, classic_scales, classic_zeros = processor.pseudo_quantize_tensor(linear.weight.detach())
    activations = torch.randn(
        97,
        in_features,
        generator=generator,
        dtype=torch.float32,
    )

    processor.apply_quant(
        {"proj": named},
        scales_list=[],
        input_features={"proj": activations},
        adjacent_references={"proj": reference},
    )
    named.stream_sync()

    stats = adjacent.snapshot()
    assert len(stats) == 1
    assert stats[0]["module"] == "layers.0.proj"
    assert stats[0]["status"] == "applied"
    assert named.state["q_scales"].device.type == "cpu"
    assert named.state["q_zeros"].device.type == "cpu"
    assert torch.all(named.state["q_zeros"] == 8)
    torch.testing.assert_close(named.state["q_scales"], classic_scales.cpu(), rtol=0.0, atol=0.0)
    torch.testing.assert_close(named.state["q_zeros"], classic_zeros.cpu(), rtol=0.0, atol=0.0)
    activation_device = activations.to(device)
    reference_device = reference.to(device=device, dtype=torch.float32)
    classic_error = activation_device @ (reference_device - classic.to(torch.float32)).mT
    hybrid_error = activation_device @ (reference_device - linear.weight.detach().to(torch.float32)).mT
    assert float(hybrid_error.square().sum()) <= float(classic_error.square().sum()) + 1e-4

    expected = linear.weight.detach().cpu()
    linear.cpu()
    packed = AwqTorchLinear(
        bits=bits,
        group_size=group_size,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=False,
        dtype=torch.float16,
    )
    packed.pack(
        linear=linear,
        scales=named.state["q_scales"],
        zeros=named.state["q_zeros"],
    )
    dequantized = dequantize_gemm(
        qweight=packed.qweight,
        qzeros=packed.qzeros,
        scales=packed.scales,
        bits=bits,
        group_size=group_size,
    ).mT
    torch.testing.assert_close(dequantized, expected, rtol=0.0, atol=2e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_awq_adjacent_records_fallback_modules_without_changing_rtn():
    adjacent = AdjacentModelConfig(
        executor="cuda",
        native_refinements_per_module=0,
    )
    processor = _TestAWQProcessor(
        AWQConfig(
            bits=4,
            group_size=16,
            sym=True,
            format=FORMAT.GEMM,
            adjacent_model=adjacent,
        )
    )
    linear = nn.Linear(16, 8, bias=False, device="cuda", dtype=torch.float16)
    named = NamedModule(linear, name="proj", full_name="layers.0.proj", layer_index=0)
    expected = processor.pseudo_quantize_tensor(linear.weight.detach())[0]

    processor.apply_quant({"proj": named}, scales_list=[])

    torch.testing.assert_close(linear.weight, expected, rtol=0.0, atol=0.0)
    stats = adjacent.snapshot()
    assert stats == [
        {
            "module": "layers.0.proj",
            "quant_method": "awq",
            "status": "skipped",
            "reason": "AWQ fallback quantization has no scaled activation/reference pair.",
            "bits": 4,
            "group_size": 16,
        }
    ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_awq_adjacent_supports_tensor_parallel_zero_padding():
    adjacent = AdjacentModelConfig(
        executor="cuda",
        row_chunk_size=8,
        objective_row_chunk_size=8,
        activation_chunk_size=13,
        native_refinements_per_module=0,
    )
    processor = _TestAWQProcessor(
        AWQConfig(
            bits=4,
            group_size=16,
            sym=True,
            format=FORMAT.GEMM,
            adjacent_model=adjacent,
        )
    )
    linear = nn.Linear(30, 8, bias=False, device="cuda", dtype=torch.float16)
    named = NamedModule(linear, name="proj", full_name="layers.0.proj", layer_index=0)
    named.state["tp_pad_info"] = {
        "pad_cols": 2,
        "original_columns": 30,
    }
    reference = linear.weight.detach().cpu().clone()
    activations = torch.randn(41, 30, dtype=torch.float32)

    processor.apply_quant(
        {"proj": named},
        scales_list=[],
        input_features={"proj": activations},
        adjacent_references={"proj": reference},
    )
    named.stream_sync()

    assert linear.weight.shape == (8, 30)
    assert named.state["q_scales"].shape == (8, 2)
    assert named.state["q_zeros"].shape == (8, 2)
    assert "tp_pad_info" not in named.state
    stats = adjacent.snapshot()
    assert len(stats) == 1
    assert stats[0]["columns"] == 32
    assert stats[0]["activation_columns"] == 30
