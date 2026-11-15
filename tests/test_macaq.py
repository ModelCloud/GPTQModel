# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""MACaQ MXFP4 coverage.

The port has four independently checkable pieces:

1. the block-Hadamard rotation is orthogonal and block-local
2. the D1 balance pre-scale follows ``w_std^alpha / act_rms^(1-alpha)``
3. RTN / exponent screen / flip refinement produce E2M1 codes on the OCP grid
4. pack -> unpack -> dequantize is exact, and the Triton kernels agree with
   the Torch reference for both activations and the W4A4 GEMM

The Torch oracles below are written from the OCP MXFP4 definition rather than
calling the production helpers, so a packer/rotation regression cannot hide
behind the implementation it is checking.
"""

import math
import sys
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest
import torch

from gptqmodel.nn_modules.qlinear.macaq import MacaqLinear, MacaqTritonLinear
from gptqmodel.quantization import FORMAT, METHOD, MacaqConfig, QuantizeConfig
from gptqmodel.quantization import macaq as macaq
from gptqmodel.utils.backend import BACKEND
from gptqmodel.utils.macaq_triton import triton_mxfp4_available


def _device():
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def _hessian_weighted_error(weight, dequantized, hessian):
    delta = dequantized.to(torch.float64) - weight.to(torch.float64)
    return float((delta @ hessian.to(torch.float64) * delta).sum().item())


def _reference_rotate(x, bh):
    block = bh.shape[0]
    lead = x.shape[:-1]
    return (x.reshape(*lead, x.shape[-1] // block, block) @ bh.to(x.dtype)).reshape(*lead, x.shape[-1])


# ---------------------------------------------------------------------------
# rotation / balance
# ---------------------------------------------------------------------------
def test_hadamard_is_orthogonal_and_block_local():
    for size in (32, 64):
        bh = macaq.make_hadamard(size)
        identity = torch.eye(size)
        assert torch.allclose(bh @ bh.T, identity, atol=1e-6)
        assert torch.allclose(bh.T @ bh, identity, atol=1e-6)

    x = torch.randn(3, 128)
    bh = macaq.make_hadamard(32)
    rotated = macaq.rotate_blocks(x, bh)
    expected = _reference_rotate(x, bh)
    assert torch.allclose(rotated, expected, atol=1e-6)
    # Rotation is block-local: changing one 32-wide group cannot touch another.
    x2 = x.clone()
    x2[:, :32] += 10.0
    rotated2 = macaq.rotate_blocks(x2, bh)
    assert torch.allclose(rotated[:, 32:], rotated2[:, 32:], atol=1e-6)


def test_balance_pre_scale_matches_definition():
    torch.manual_seed(0)
    weight = torch.randn(32, 64)
    activations = torch.randn(128, 64)
    hessian = activations.T @ activations
    alpha = 0.5

    scale = macaq.balance_pre_scale(weight, hessian, activations.shape[0], alpha=alpha)

    w_std = weight.std(dim=0)
    act_rms = (hessian.diagonal() / activations.shape[0]).sqrt()
    expected = (w_std.pow(alpha) / act_rms.pow(1.0 - alpha)).clamp(1e-4, 1e4)
    assert torch.allclose(scale, expected, atol=1e-6)


def test_macaq_default_balance_matches_reference_recipe():
    config = MacaqConfig()
    assert config.pre_mode == "balance"
    assert config.pre_alpha == 0.75
    assert config.had_block == config.block_size == 32
    assert config.compile_flip is True
    assert config.to_dict()["compile_flip"] is True


def test_macaq_weight_quantizer_requests_compiled_flips_by_default():
    weight = torch.randn(8, 64)
    hessian = torch.eye(64)
    with patch.object(macaq, "flip_refine_levels", wraps=macaq.flip_refine_levels) as sweep:
        macaq.macaq_quantize_weight(weight, hessian, 64, n_flip_iters=1)
    assert sweep.call_args.kwargs["compile_flip"] is True


# ---------------------------------------------------------------------------
# MXFP4 round-to-nearest and packing
# ---------------------------------------------------------------------------
def test_rtn_codes_are_on_the_e2m1_grid():
    torch.manual_seed(0)
    weight = torch.randn(8, 64)
    signed_index, log2_scale, scale_element = macaq.mxfp4_rtn_signed_index(weight, 32)
    values = macaq.E2M1_VALUES.to(weight.device)[signed_index] * scale_element
    assert signed_index.min() >= 0 and signed_index.max() <= 14
    assert torch.isfinite(values).all()
    # Every value is exactly representable as E2M1 * 2^k.
    grid = macaq.E2M1_VALUES.to(weight.device)
    ratio = (values / scale_element).reshape(-1)
    assert torch.allclose(
        ratio,
        grid[torch.argmin((ratio[:, None] - grid[None, :]).abs(), dim=1)],
        atol=1e-5,
    )


def test_pack_unpack_roundtrip_is_exact():
    torch.manual_seed(0)
    weight = torch.randn(16, 128)
    signed_index, log2_scale, scale_element = macaq.mxfp4_rtn_signed_index(weight, 32)
    packed, scales = macaq.pack_mxfp4_from_signed_index(signed_index, log2_scale, 32)

    assert packed.dtype == torch.uint8
    assert packed.shape == (16, 64)
    assert scales.dtype == torch.float8_e8m0fnu
    assert scales.shape == (16, 4)

    dequantized = macaq.dequantize_mxfp4(packed, scales, weight.shape[1])
    expected = macaq.E2M1_VALUES.to(weight.device)[signed_index] * scale_element
    assert torch.equal(dequantized, expected)

    # The uint8 view used by the kernels is the same storage.
    dequantized_bytes = macaq.dequantize_mxfp4(packed, scales.view(torch.uint8), weight.shape[1])
    assert torch.equal(dequantized, dequantized_bytes)


def test_single_offset_screen_matches_plain_rtn():
    torch.manual_seed(0)
    weight = torch.randn(8, 64)
    hessian = torch.eye(64)
    screened = macaq.screen_block_exponents(weight, hessian.diagonal(), 32, offsets=(0,))
    plain = macaq.mxfp4_rtn_signed_index(weight, 32, 0)
    assert torch.equal(screened[0], plain[0])
    assert torch.equal(screened[1], plain[1])


def test_flip_refinement_reduces_hessian_weighted_error():
    torch.manual_seed(0)
    weight = torch.randn(32, 128)
    activations = torch.randn(256, 128)
    hessian = activations.T @ activations
    hessian.diagonal().add_(hessian.diagonal().mean())

    signed_index, log2_scale, scale_element = macaq.mxfp4_rtn_signed_index(weight, 32)
    before = macaq.E2M1_VALUES.to(weight.device)[signed_index] * scale_element
    error_before = _hessian_weighted_error(weight, before, hessian)

    macaq.flip_refine_levels(signed_index, scale_element, weight, hessian, 64, fast=True)
    after = macaq.E2M1_VALUES.to(weight.device)[signed_index] * scale_element
    error_after = _hessian_weighted_error(weight, after, hessian)

    assert error_after <= error_before
    assert error_after < error_before * 0.999


@pytest.mark.skipif(not torch.cuda.is_available(), reason="compiled flip parity needs CUDA")
@pytest.mark.parametrize("fast", [True, False])
def test_compiled_flip_matches_eager_packed_codes(fast):
    if not macaq._compiled_flip_supported(torch.device("cuda:0")):
        pytest.skip("compiled flips are not qualified on this GPU/runtime")
    torch.manual_seed(71)
    weight = torch.randn(48, 128, device="cuda")
    activations = torch.randn(256, 128, device="cuda")
    hessian = activations.T @ activations
    hessian.diagonal().add_(hessian.diagonal().mean())
    initial, log2_scale, scale_element = macaq.mxfp4_rtn_signed_index(weight, 32)
    eager = initial.clone()
    compiled = initial.clone()

    macaq.flip_refine_levels(eager, scale_element, weight, hessian, 24,
                             fast=fast, compile_flip=False)
    macaq.flip_refine_levels(
        compiled, scale_element, weight, hessian, 24,
        fast=fast, compile_flip=True,
    )
    torch.cuda.synchronize()
    eager_packed = macaq.pack_mxfp4_from_signed_index(eager, log2_scale, 32)
    compiled_packed = macaq.pack_mxfp4_from_signed_index(compiled, log2_scale, 32)
    assert torch.equal(eager, compiled)
    assert all(torch.equal(a.view(torch.uint8), b.view(torch.uint8))
               for a, b in zip(eager_packed, compiled_packed))


@pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is not None,
                    reason="NVIDIA CUDA free-threaded fallback check")
def test_compiled_flip_uses_eager_on_free_threaded_nvidia():
    torch.manual_seed(73)
    weight = torch.randn(8, 32, device="cuda")
    hessian = torch.eye(32, device="cuda")
    initial, _, scale_element = macaq.mxfp4_rtn_signed_index(weight)
    eager, guarded = initial.clone(), initial.clone()
    macaq.flip_refine_levels(eager, scale_element, weight, hessian, 4,
                             compile_flip=False)
    with patch.object(sys, "_is_gil_enabled", return_value=False, create=True), patch.object(
        torch, "compile", side_effect=AssertionError("compiler used from GIL=0 worker")
    ):
        macaq.flip_refine_levels(guarded, scale_element, weight, hessian, 4, compile_flip=True)
    assert torch.equal(guarded, eager)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="compiled flips need a GPU")
def test_compiled_flip_is_default_on_supported_gpu():
    if not macaq._compiled_flip_supported(torch.device("cuda:0")):
        pytest.skip("compiled flips are not qualified on this GPU/runtime")
    weight = torch.randn(8, 32, device="cuda")
    hessian = torch.eye(32, device="cuda")
    initial, _, scale_element = macaq.mxfp4_rtn_signed_index(weight)
    with patch.object(macaq, "_compiled_flip_sweep", wraps=macaq._flip_sweep) as sweep:
        macaq.flip_refine_levels(initial, scale_element, weight, hessian, 1)
    assert sweep.called


@pytest.mark.skipif(not torch.cuda.is_available(), reason="compiled flip needs CUDA")
def test_compiled_flip_concurrent_first_use_matches_eager():
    """Quantization workers can enter the same Inductor autotuner together."""

    if not macaq._compiled_flip_supported(torch.device("cuda:0")):
        pytest.skip("compiled flips are not qualified on this GPU/runtime")

    def run(seed):
        generator = torch.Generator(device="cuda").manual_seed(seed)
        weight = torch.randn(24, 96, generator=generator, device="cuda")
        x = torch.randn(128, 96, generator=generator, device="cuda")
        hessian = x.T @ x
        initial, _, scale = macaq.mxfp4_rtn_signed_index(weight)
        eager, compiled = initial.clone(), initial.clone()
        macaq.flip_refine_levels(eager, scale, weight, hessian, 12,
                                 compile_flip=False)
        macaq.flip_refine_levels(compiled, scale, weight, hessian, 12, compile_flip=True)
        return torch.equal(eager, compiled)

    with patch.object(macaq, "_COMPILED_FLIP_SWEEP", None), ThreadPoolExecutor(max_workers=4) as pool:
        assert all(pool.map(run, (1, 2, 3, 4)))


def test_quantize_weight_reports_finite_loss_and_shapes():
    torch.manual_seed(0)
    device = _device()
    weight = torch.randn(32, 128, device=device)
    activations = torch.randn(128, 128, device=device)
    hessian = activations.T @ activations

    result = macaq.macaq_quantize_weight(
        weight, hessian, activations.shape[0], n_flip_iters=8, return_dequantized=True
    )
    assert result.W_packed.shape == (32, 64)
    assert result.W_scales.shape == (32, 4)
    assert result.pre_scale.shape == (128,)
    assert result.BH.shape == (32, 32)
    assert result.dequantized_weight.shape == weight.shape
    assert math.isfinite(result.stats.get("loss", 0.0))


# ---------------------------------------------------------------------------
# quantizer interface and config
# ---------------------------------------------------------------------------
def test_macaq_quantizer_interface():
    torch.manual_seed(0)
    device = _device()
    layer = torch.nn.Linear(64, 32, bias=True).to(device)
    activations = torch.randn(128, 64, device=device)

    quantizer = macaq.Macaq(layer, MacaqConfig(n_flip_iters=4, offload_to_disk=False))
    quantizer.add_batch(activations)
    result, duration, loss = quantizer.quantize()

    assert quantizer.nsamples == 128
    assert duration >= 0
    assert math.isfinite(loss)
    assert result.W_packed.dtype == torch.uint8
    assert result.W_scales.dtype == torch.float8_e8m0fnu
    quantizer.free()
    assert quantizer.H2 is None


def test_macaq_config_roundtrip():
    config = MacaqConfig(
        n_flip_iters=12,
        compile_flip=True,
        had_block=64,
        ridge=0.5,
        pre_alpha=0.75,
        exp_offsets=(1, 0),
        activation_precision="w4a16",
        offload_to_disk=False,
    )
    payload = config.to_dict()
    assert payload["method"] == METHOD.MACAQ
    assert payload["format"] == FORMAT.MXFP4_MACAQ
    assert FORMAT.MXFP4_MACAQ.value == "mxfp4_macaq"
    assert payload["n_flip_iters"] == 12
    assert payload["compile_flip"] is True
    assert payload["had_block"] == 64
    assert payload["exp_offsets"] == [1, 0]
    assert payload["activation_precision"] == "w4a16"
    assert payload["sym"] is True

    restored = QuantizeConfig.from_quant_config(payload)
    assert isinstance(restored, MacaqConfig)
    assert restored.n_flip_iters == 12
    assert restored.compile_flip is True
    assert restored.had_block == 64
    assert restored.ridge == 0.5
    assert restored.pre_alpha == 0.75
    assert restored.exp_offsets == (1, 0)
    assert restored.activation_precision == "w4a16"


def test_macaq_config_rejects_unsupported_settings():
    with pytest.raises(ValueError):
        MacaqConfig(block_size=16)
    with pytest.raises(ValueError):
        MacaqConfig(had_block=24)
    with pytest.raises(ValueError):
        MacaqConfig(pre_mode="invalid")
    with pytest.raises(ValueError):
        MacaqConfig(activation_precision="w8a8")
    with pytest.raises(ValueError):
        MacaqConfig(lm_head=True)
    with pytest.raises(ValueError, match="unsupported dynamic keys"):
        MacaqConfig(dynamic={"layer": {"bits": 3}})


def test_macaq_processor_keeps_resolved_pack_backend_and_dynamic_config():
    from gptqmodel.looper.macaq_processor import MacaqProcessor
    from gptqmodel.looper.named_module import NamedModule
    from gptqmodel.quantization.module_config import resolve_module_constructor_config

    config = MacaqConfig(
        had_block=32,
        activation_precision="w4a4",
        dynamic={"model.layer": {"had_block": 64, "activation_precision": "w4a16"}},
        offload_to_disk=False,
    )
    wrapped = NamedModule(
        torch.nn.Linear(128, 32), name="layer", full_name="model.layer", layer_index=0
    )
    processor = MacaqProcessor(
        tokenizer=None, qcfg=config, calibration=None, backend=BACKEND.MACAQ_TORCH
    )
    assert processor.lifecycle.forward.all_modules_in_single_pass
    processor.preprocess(wrapped)
    task = processor.tasks[wrapped.name]
    assert task.qcfg.had_block == 64
    assert task.qcfg.activation_precision == "w4a16"
    assert processor._linear_cls() == (MacaqLinear, BACKEND.MACAQ_TORCH)

    module_config = resolve_module_constructor_config(
        bits=4, group_size=32, desc_act=False, sym=True, pack_dtype=torch.int32,
        format=FORMAT.MXFP4_MACAQ, init_kwargs=config.quant_linear_init_kwargs(),
        overrides=config.resolve_module_config(wrapped.full_name).overrides,
    )
    assert module_config.options_dict()["had_block"] == 64
    assert module_config.options_dict()["activation_precision"] == "w4a16"

    processor_triton = MacaqProcessor(
        tokenizer=None, qcfg=config, calibration=None, backend=BACKEND.MACAQ_TRITON
    )
    assert processor_triton._linear_cls() == (MacaqTritonLinear, BACKEND.MACAQ_TRITON)
    with pytest.raises(ValueError, match="resolved Torch or Triton"):
        MacaqProcessor(tokenizer=None, qcfg=config, calibration=None, backend=BACKEND.AUTO)


@pytest.mark.parametrize("activation_precision", ["w4a4", "w4a16"])
def test_macaq_processor_replays_through_packed_runtime(activation_precision):
    from gptqmodel.looper.macaq_processor import MacaqProcessor
    from gptqmodel.looper.named_module import NamedModule

    torch.manual_seed(29)
    wrapped = NamedModule(
        torch.nn.Linear(64, 32, bias=True),
        name="proj", full_name="model.proj", layer_index=0,
    )
    config = MacaqConfig(
        activation_precision=activation_precision, n_flip_iters=4,
        offload_to_disk=False,
    )
    processor = MacaqProcessor(
        tokenizer=None, qcfg=config, calibration=None, backend=BACKEND.MACAQ_TORCH,
    )
    processor.preprocess(wrapped)
    processor.tasks[wrapped.name].add_batch(torch.randn(128, 64))
    processor.process(wrapped)

    x = torch.randn(12, 64) * 3
    packed_output = wrapped.module._macaq_replay(x)
    torch.testing.assert_close(wrapped.module(x), packed_output, rtol=0, atol=0)
    dense_output = torch.nn.functional.linear(x, wrapped.module.weight, wrapped.module.bias)
    assert not torch.equal(packed_output, dense_output)

    processor.prepare_layer_post_quantize(
        model=None, layer_module=wrapped.module, layer_index=0,
        processed_modules={wrapped.name: wrapped}, is_lm_head_module=False,
    )
    assert "_macaq_replay" not in wrapped.module._modules
    assert "forward" not in wrapped.module.__dict__
    torch.testing.assert_close(wrapped.module(x), dense_output, rtol=0, atol=0)


def test_macaq_processor_conv1d_replay_preserves_weight_layout():
    from transformers import Conv1D

    from gptqmodel.looper.macaq_processor import MacaqProcessor
    from gptqmodel.looper.named_module import NamedModule

    torch.manual_seed(31)
    wrapped = NamedModule(
        Conv1D(32, 64), name="proj", full_name="model.proj", layer_index=0,
    )
    original_shape = wrapped.weight.shape
    processor = MacaqProcessor(
        tokenizer=None,
        qcfg=MacaqConfig(n_flip_iters=4, offload_to_disk=False),
        calibration=None,
        backend=BACKEND.MACAQ_TORCH,
    )
    processor.preprocess(wrapped)
    processor.tasks[wrapped.name].add_batch(torch.randn(128, 64))
    processor.process(wrapped)

    assert wrapped.weight.shape == original_shape
    x = torch.randn(7, 64)
    torch.testing.assert_close(wrapped.module(x), wrapped.module._macaq_replay(x), rtol=0, atol=0)

    processor.prepare_layer_post_quantize(
        model=None, layer_module=wrapped.module, layer_index=0,
        processed_modules={wrapped.name: wrapped}, is_lm_head_module=False,
    )
    expected = torch.nn.functional.linear(x, wrapped.weight.T, wrapped.bias)
    torch.testing.assert_close(wrapped.module(x), expected, rtol=1e-6, atol=1e-6)


def test_macaq_triton_dequant_accepts_padded_e8m0_grid():
    from gptqmodel.utils.macaq_triton import dequantize_mxfp4

    packed = torch.zeros(2, 48, dtype=torch.uint8)
    scales = torch.full((2, 4), 127, dtype=torch.uint8)
    packed[:, :48] = 0x22  # E2M1 1.0 on both nibbles
    decoded = dequantize_mxfp4(packed, scales, 96)
    assert decoded.shape == (2, 96)
    assert torch.equal(decoded, torch.ones_like(decoded))


@pytest.mark.skipif(
    not triton_mxfp4_available(),
    reason="MACaQ Triton kernels require NVIDIA CUDA or ROCm gfx1201",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_triton_w4a16_matches_torch_with_bias(dtype):
    from gptqmodel.utils.macaq_triton import mxfp4_w4a16_linear, quantize_mxfp4

    torch.manual_seed(17)
    x = torch.randn(3, 64, device="cuda", dtype=dtype)
    weight = torch.randn(8, 64, device="cuda", dtype=torch.float32)
    packed, scales = quantize_mxfp4(weight)
    bias = torch.randn(8, device="cuda", dtype=dtype)
    pre_scale = torch.rand(64, device="cuda", dtype=dtype) + 0.5
    bh = macaq.make_hadamard(32, device=x.device, dtype=dtype)

    actual = mxfp4_w4a16_linear(
        x, packed, scales, pre_scale=pre_scale, bh=bh,
        bias=bias, compute_dtype=dtype,
    )
    expected_input = macaq.rotate_blocks(x * pre_scale, bh)
    expected_weight = macaq.dequantize_mxfp4(packed, scales, 64, dtype=dtype)
    expected = torch.nn.functional.linear(expected_input, expected_weight, bias)
    assert actual.dtype == dtype
    assert torch.allclose(actual, expected, atol=2e-3, rtol=2e-3)


@pytest.mark.skipif(
    not triton_mxfp4_available(),
    reason="MACaQ Triton kernels require NVIDIA CUDA or ROCm gfx1201",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_triton_activation_packs_exact_e2m1_bytes(dtype):
    from gptqmodel.utils.macaq_triton import rot_scale_quantize_mxfp4

    torch.manual_seed(101)
    x = torch.randn(7, 64, device="cuda", dtype=dtype)
    pre_scale = (torch.rand(64, device="cuda") + 0.5).to(dtype)
    bh = macaq.make_hadamard(32, device=x.device, dtype=torch.bfloat16)
    packed, scales = rot_scale_quantize_mxfp4(x, pre_scale, bh)

    work = (x.float() * pre_scale.float()).to(torch.bfloat16).float()
    work = macaq.rotate_blocks(work, bh.float()).to(torch.bfloat16).float()
    grouped = work.reshape(7, 2, 32)
    exponent = torch.floor(torch.log2(grouped.abs().amax(dim=-1).clamp_min(1e-12))) - 2
    scale = torch.exp2(exponent)
    normalized = grouped.abs() / scale.unsqueeze(-1)
    thresholds = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=x.device)
    magnitude = (normalized.unsqueeze(-1) >= thresholds).sum(dim=-1).to(torch.uint8)
    nibble = magnitude | ((grouped < 0).to(torch.uint8) << 3)
    expected_packed = (nibble[..., 1::2] << 4 | nibble[..., 0::2]).reshape(7, 32)
    expected_scales = (exponent + 127).to(torch.uint8)
    assert torch.equal(packed, expected_packed)
    assert torch.equal(scales, expected_scales)


@pytest.mark.skipif(
    not triton_mxfp4_available(),
    reason="MACaQ Triton kernels require NVIDIA CUDA or ROCm gfx1201",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("out_scale", [1.0, 0.5])
def test_triton_w4a4_accumulates_bias_before_output_rounding(dtype, out_scale):
    from gptqmodel.utils.macaq_triton import mxfp4_w4a4_linear, quantize_mxfp4

    torch.manual_seed(21)
    x = torch.randn(7, 128, device="cuda", dtype=dtype)
    weight = torch.randn(64, 128, device="cuda")
    packed, scales = quantize_mxfp4(weight)
    pre_scale = (torch.rand(128, device="cuda") + 0.5).to(dtype)
    bh = macaq.make_hadamard(32, device=x.device, dtype=dtype)
    bias = torch.randn(64, device="cuda", dtype=dtype)

    actual = mxfp4_w4a4_linear(
        x, packed, scales, pre_scale=pre_scale, bh=bh, bias=bias, out_scale=out_scale,
    )
    activation = macaq.macaq_quantize_activations(x, pre_scale, bh)
    decoded_weight = macaq.dequantize_mxfp4(packed, scales, 128)
    expected = (out_scale * (activation.float() @ decoded_weight.T) + bias.float()).to(dtype)
    assert actual.dtype == dtype
    assert torch.allclose(actual.float(), expected.float(), atol=2e-3, rtol=2e-3)


@pytest.mark.skipif(
    not triton_mxfp4_available(), reason="MACaQ Triton kernels require NVIDIA CUDA or ROCm gfx1201"
)
@pytest.mark.parametrize("rows", [32, 64, 65])
def test_triton_tile_boundary_matches_torch(rows):
    from gptqmodel.utils.macaq_triton import mxfp4_w4a4_linear, quantize_mxfp4

    torch.manual_seed(41)
    k, n = 896, 96
    x = torch.randn(rows, k, device="cuda", dtype=torch.bfloat16) * 0.1
    weight = torch.randn(n, k, device="cuda") * 0.1
    packed, scales = quantize_mxfp4(weight)
    bias = torch.randn(n, device="cuda", dtype=torch.bfloat16) * 0.1

    actual = mxfp4_w4a4_linear(x, packed, scales, bias=bias)
    activation = macaq.macaq_quantize_activations(x)
    decoded_weight = macaq.dequantize_mxfp4(packed, scales, k)
    expected = (activation.float() @ decoded_weight.T + bias.float()).to(x.dtype)
    assert actual.dtype == x.dtype
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)


def test_macaq_module_does_not_retain_dense_weight_between_forwards():
    _, _, module, _ = _quantized_module_pair()
    x = torch.randn(3, module.in_features, device=module.W_packed.device, dtype=torch.bfloat16)
    for _ in range(2):
        module(x)
        assert set(module._buffers) == {"BH", "pre_scale", "bias", "W_packed", "W_scales"}
        assert all(
            tensor.numel() < module.in_features * module.out_features
            for tensor in module._buffers.values() if tensor is not None
        )


@pytest.mark.skipif(
    not triton_mxfp4_available(), reason="MACaQ Triton kernels require NVIDIA CUDA or ROCm gfx1201"
)
def test_macaq_triton_module_keeps_only_packed_weight_after_repeated_forwards():
    _, _, _, module = _quantized_module_pair()
    for rows in (1, 65):
        x = torch.randn(rows, module.in_features, device="cuda", dtype=torch.bfloat16)
        output = module(x)
        assert output.shape == (rows, module.out_features)
        assert set(module._buffers) == {"BH", "pre_scale", "bias", "W_packed", "W_scales"}
        assert all(
            tensor.numel() < module.in_features * module.out_features
            for tensor in module._buffers.values() if tensor is not None
        )


# ---------------------------------------------------------------------------
# module forward
# ---------------------------------------------------------------------------
def _quantized_module_pair():
    torch.manual_seed(0)
    device = _device()
    layer = torch.nn.Linear(128, 64, bias=True).to(device)
    activations = torch.randn(256, 128, device=device)
    quantizer = macaq.Macaq(layer, MacaqConfig(n_flip_iters=8, offload_to_disk=False))
    quantizer.add_batch(activations)
    result, _, _ = quantizer.quantize()

    def build(cls, backend):
        module = cls(
            bits=4,
            group_size=32,
            in_features=128,
            out_features=64,
            bias=True,
            pack_dtype=torch.int32,
            block_size=32,
            had_block=32,
            activation_precision="w4a4",
            dtype=torch.bfloat16,
            backend=backend,
        ).to(device)
        module.pack(layer, macaq_result=result)
        return module.to(device)

    return layer, result, build(MacaqLinear, BACKEND.MACAQ_TORCH), build(MacaqTritonLinear, BACKEND.MACAQ_TRITON)


def test_macaq_linear_torch_matches_reference():
    layer, result, module, _ = _quantized_module_pair()
    x = torch.randn(2, 5, 128, device=layer.weight.device, dtype=torch.bfloat16)

    output = module(x)
    reference_weight = macaq.dequantize_mxfp4(
        result.W_packed, result.W_scales, 128, dtype=torch.bfloat16
    )
    reference_input = macaq.macaq_quantize_activations(x, result.pre_scale, result.BH).to(torch.bfloat16)
    expected = torch.nn.functional.linear(
        reference_input, reference_weight, layer.bias.to(torch.bfloat16)
    )

    assert output.dtype == torch.bfloat16
    assert torch.equal(output, expected)


@pytest.mark.skipif(
    not triton_mxfp4_available(),
    reason="MACaQ Triton kernels require NVIDIA CUDA or ROCm gfx1201",
)
def test_triton_kernels_match_torch():
    layer, result, torch_module, triton_module = _quantized_module_pair()
    x = torch.randn(4, 128, device=layer.weight.device, dtype=torch.bfloat16)

    from gptqmodel.utils import macaq_triton

    # Activation quantizer parity (including the fused rotation).
    packed, scales = macaq_triton.rot_scale_quantize_mxfp4(
        x, result.pre_scale, result.BH
    )
    triton_activations = macaq_triton.dequantize_mxfp4(packed, scales, 128).to(torch.bfloat16)
    torch_activations = macaq.macaq_quantize_activations(
        x, result.pre_scale, result.BH
    ).to(torch.bfloat16)
    assert torch.equal(triton_activations, torch_activations)

    # Weight dequantization parity.
    assert torch.equal(
        macaq_triton.dequant_w4_to_bf16(result.W_packed, result.W_scales),
        macaq.dequantize_mxfp4(result.W_packed, result.W_scales, 128, dtype=torch.bfloat16),
    )

    # W4A4 GEMM parity.
    triton_output = macaq_triton.mxfp4_w4a4_linear(
        x,
        result.W_packed,
        result.W_scales,
        pre_scale=result.pre_scale,
        bh=result.BH,
        bias=layer.bias,
    )
    torch_output = torch_module(x)
    assert triton_output.dtype == torch.bfloat16
    assert torch.allclose(triton_output, torch_output, atol=0.02, rtol=0.01)

    # The registered Triton module takes the same path and preserves dtype.
    module_output = triton_module(x)
    assert module_output.dtype == torch.bfloat16
    assert torch.allclose(module_output, torch_output, atol=0.02, rtol=0.01)


@pytest.mark.skipif(
    not triton_mxfp4_available(),
    reason="MACaQ Triton kernels require NVIDIA CUDA or ROCm gfx1201",
)
@pytest.mark.parametrize("k", [896, 1152, 2176])
def test_triton_dense_kernel_masks_k_tail(k):
    """``BLOCK_K=256`` does not divide these hidden sizes.

    The tail tile must be masked with the loop offset.  Without it the kernel
    reads the next row's packed bytes and E8M0 exponents, which is a
    catastrophic error rather than a rounding difference.
    """

    import triton

    from gptqmodel.utils import macaq_triton

    device = "cuda"
    rows, cols = 8, 128
    torch.manual_seed(k)
    weight = torch.randn(cols, k, device=device) * 0.1
    activations = torch.randn(rows, k, device=device, dtype=torch.bfloat16)

    w_packed, w_scales = macaq_triton.quantize_mxfp4(weight)
    a_packed, a_scales = macaq_triton.quantize_mxfp4(activations.float())

    out = torch.empty(rows, cols, device=device, dtype=torch.float32)
    macaq_triton._dense_w4a4_kernel[
        (triton.cdiv(rows, 64), triton.cdiv(cols, 128))
    ](
        a_packed, w_packed, a_scales, w_scales, out, w_scales,
        rows, cols, k,
        a_packed.stride(0), w_packed.stride(0),
        a_scales.stride(0), w_scales.stride(0), out.stride(0),
        OUT_SCALE=1.0, HAS_BIAS=False, BLOCK_M=64, BLOCK_N=128, BLOCK_K=256,
        num_warps=8, num_stages=3,
    )

    expected = macaq.dequantize_mxfp4(a_packed, a_scales, k) @ macaq.dequantize_mxfp4(
        w_packed, w_scales, k
    ).T
    assert torch.allclose(out, expected, atol=0.02, rtol=0.01)

    # The public entry point reduces BLOCK_K to a divisor; both paths must agree.
    linear = macaq_triton.mxfp4_w4a4_linear(
        activations, w_packed, w_scales, pre_scale=None, bh=None
    )
    assert torch.allclose(linear.float(), expected, atol=0.05, rtol=0.02)


def test_checkpoint_module_config_roundtrips_macaq_kernel():
    from gptqmodel.looper.checkpoint_modules import packed_module_config, restore_packed_module

    config = MacaqConfig(bits=4, group_size=32, had_block=64, pre_alpha=0.7)
    module = MacaqLinear(
        bits=4,
        group_size=32,
        in_features=64,
        out_features=8,
        bias=False,
        had_block=64,
        pre_alpha=0.7,
    )

    module_config = packed_module_config(module, config, module_name="layer")
    assert module_config["kind"] == "macaq"
    assert module_config["options"]["had_block"] == 64
    assert module_config["kernel"].endswith("MacaqLinear")

    restored = restore_packed_module(
        module_config, name="layer", config=config, kernel=MacaqLinear, lm_head_name="lm_head"
    )
    assert type(restored) is MacaqLinear
    assert restored.had_block == 64
    assert restored.pre_alpha == 0.7


def test_checkpoint_restores_recorded_macaq_triton_kernel():
    from gptqmodel.looper.checkpoint_modules import packed_module_config, restore_packed_module

    config = MacaqConfig(bits=4, group_size=32)
    module = MacaqTritonLinear(bits=4, group_size=32, in_features=64, out_features=8, bias=False)

    module_config = packed_module_config(module, config, module_name="layer")
    assert module_config["kernel"].endswith("MacaqTritonLinear")

    restored = restore_packed_module(
        module_config, name="layer", config=config, kernel=MacaqLinear, lm_head_name="lm_head"
    )
    assert type(restored) is MacaqTritonLinear
