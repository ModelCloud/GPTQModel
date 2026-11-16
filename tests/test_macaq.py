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

import copy
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from gptqmodel.nn_modules.qlinear.macaq import MacaqLinear, MacaqTritonLinear
from gptqmodel.quantization import FORMAT, METHOD, MacaqConfig, QuantizeConfig
from gptqmodel.quantization import macaq as macaq
from gptqmodel.quantization.config import GGUFConfig, resolve_model_rotation, resolve_quant_format
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


@pytest.mark.parametrize("plain", [False, True], ids=["mxfp4_macaq", "mxfp4"])
@pytest.mark.parametrize("activation_precision", ["w4a4", "w4a16"])
def test_macaq_processor_replays_through_packed_runtime(activation_precision, plain):
    from gptqmodel.looper.macaq_processor import MacaqProcessor
    from gptqmodel.looper.named_module import NamedModule

    torch.manual_seed(29)
    wrapped = NamedModule(
        torch.nn.Linear(64, 32, bias=True),
        name="proj", full_name="model.proj", layer_index=0,
    )
    transforms = {"rotation": "none", "pre_mode": "none"} if plain else {}
    config = MacaqConfig(
        activation_precision=activation_precision, n_flip_iters=4,
        offload_to_disk=False, **transforms,
    )
    assert config.format == (FORMAT.MXFP4 if plain else FORMAT.MXFP4_MACAQ)
    processor = MacaqProcessor(
        tokenizer=None, qcfg=config, calibration=None, backend=BACKEND.MACAQ_TORCH,
    )
    processor.preprocess(wrapped)
    processor.tasks[wrapped.name].add_batch(torch.randn(128, 64))
    processor.process(wrapped)
    assert wrapped.module._macaq_replay.plain_mxfp4 is plain

    x = torch.randn(12, 64) * 3
    packed_output = wrapped.module._macaq_replay(x)
    torch.testing.assert_close(wrapped.module(x), packed_output, rtol=0, atol=0)
    dense_output = torch.nn.functional.linear(x, wrapped.module.weight, wrapped.module.bias)
    if plain and activation_precision == "w4a16":
        # No activation transform: the replay is exactly a matmul against the
        # dequantized weight the processor wrote back.
        assert torch.equal(packed_output, dense_output)
    else:
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
@pytest.mark.parametrize("k", [896, 1152, 2176, 96, 288, 160])
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


# ---------------------------------------------------------------------------
# issue #4090 regressions
# ---------------------------------------------------------------------------
_E2M1_VALUES = torch.tensor(
    [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
    dtype=torch.float32,
)
_E2M1_MAG = torch.tensor([7, 6, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.uint8)
_E2M1_SIGN = torch.tensor([1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.uint8)
_E2M1_THRESHOLDS = ((0.25, 1), (0.75, 2), (1.25, 3), (1.75, 4), (2.5, 5), (3.5, 6), (5.0, 7))


def _reference_hadamard(block, *, device="cpu"):
    """Independent normalized Sylvester Hadamard matrix, rounded like the port."""

    matrix = torch.ones((1, 1), dtype=torch.float32)
    while matrix.shape[0] < block:
        matrix = torch.cat(
            [torch.cat([matrix, matrix], dim=1), torch.cat([matrix, -matrix], dim=1)], dim=0
        )
    return (matrix / math.sqrt(block)).to(device=device, dtype=torch.bfloat16)


def _reference_weight_mxfp4(weight, bh, offset=0):
    """Independent MXFP4 RTN of the block-rotated weight (strict tie rule)."""

    out_features, in_features = weight.shape
    rotated = _reference_rotate(weight.to(torch.float32), bh.float())
    grouped = rotated.reshape(out_features, in_features // 32, 32)
    amax = grouped.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    exponent = (torch.floor(torch.log2(amax)) - 2.0 + offset).clamp(-127.0, 127.0)
    scale = torch.exp2(exponent)
    normalized = (grouped / scale).abs()
    code = torch.zeros_like(grouped, dtype=torch.long)
    for threshold, level in _E2M1_THRESHOLDS:
        code = torch.where(normalized > threshold, torch.full_like(code, level), code)
    negative = grouped < 0
    signed = torch.where(
        code == 0, torch.full_like(code, 7), torch.where(negative, 7 - code, 7 + code)
    ).reshape(out_features, in_features)
    values = _E2M1_VALUES.to(weight.device)[signed] * scale.expand(-1, -1, 32).reshape(
        out_features, in_features
    )
    nibbles = (_E2M1_SIGN.to(weight.device)[signed] << 3) | _E2M1_MAG.to(weight.device)[signed]
    packed = (nibbles[:, 1::2] << 4) | nibbles[:, 0::2]
    scales = (exponent.squeeze(-1) + 127).clamp(0, 254).to(torch.uint8)
    return packed.contiguous(), scales, values


def _reference_min_mse_mxfp4(weight, bh, offsets):
    """Per-block exponent with the lowest plain squared error (first wins ties)."""

    out_features, in_features = weight.shape
    groups = in_features // 32
    rotated = _reference_rotate(weight.to(torch.float32), bh.float()).reshape(out_features, groups, 32)
    best = None
    for offset in offsets:
        packed, scales, values = _reference_weight_mxfp4(weight, bh, offset=offset)
        error = (rotated - values.reshape(out_features, groups, 32)).pow(2).sum(-1)
        candidate = (error, packed.reshape(out_features, groups, 16), scales, values.reshape(out_features, groups, 32))
        if best is None:
            best = candidate
            continue
        better = candidate[0] < best[0]
        best = (
            torch.where(better, candidate[0], best[0]),
            torch.where(better.unsqueeze(-1), candidate[1], best[1]),
            torch.where(better, candidate[2], best[2]),
            torch.where(better.unsqueeze(-1), candidate[3], best[3]),
        )
    _, packed, scales, values = best
    return (
        packed.reshape(out_features, in_features // 2).contiguous(),
        scales.contiguous(),
        values.reshape(out_features, in_features).contiguous(),
    )


def _reference_activation_mxfp4(values):
    """Independent MXFP4 packer for the activation kernel (>= tie rule)."""

    rows, k = values.shape
    grouped = values.to(torch.float32).reshape(rows, k // 32, 32)
    amax = grouped.abs().amax(dim=-1, keepdim=True).clamp_min(1e-20)
    exponent = (torch.floor(torch.log2(amax)) - 2.0).clamp(-127.0, 127.0)
    scale = torch.exp2(exponent)
    normalized = (grouped / scale).abs()
    code = torch.zeros_like(grouped, dtype=torch.long)
    for threshold, level in _E2M1_THRESHOLDS:
        code = torch.where(normalized >= threshold, torch.full_like(code, level), code)
    negative = grouped < 0
    signed = torch.where(
        code == 0, torch.full_like(code, 7), torch.where(negative, 7 - code, 7 + code)
    ).reshape(rows, k)
    nibbles = (_E2M1_SIGN.to(values.device)[signed] << 3) | _E2M1_MAG.to(values.device)[signed]
    packed = (nibbles[:, 1::2] << 4) | nibbles[:, 0::2]
    scales = (exponent.squeeze(-1) + 127).clamp(0, 254).to(torch.uint8)
    return packed.contiguous(), scales


@pytest.mark.skipif(
    not triton_mxfp4_available(), reason="MACaQ Triton kernels require NVIDIA CUDA or ROCm gfx1201"
)
def test_triton_row_offsets_are_int64_past_2gb():
    """Rows past 2^31/K must read their own row, not a wrapped 32-bit offset.

    A 1e9-row tensor is not runnable, so the input is a view whose storage
    starts 2 GiB into a larger allocation: the buggy int32 offset for the
    first overflowing row wraps to ``data_ptr - 2**31`` and silently reads
    row 0's neighbourhood instead of the requested row.
    """

    from gptqmodel.utils.macaq_triton import quantize_mxfp4

    free, _ = torch.cuda.mem_get_info()
    if free < 8 * 2**30:
        pytest.skip("int64 row-offset regression needs about 6 GiB of free GPU memory")

    k = 8192
    offset = 2**31
    rows = offset // k + 2  # first row whose offset is 2**31
    storage = torch.empty(offset + rows * k, dtype=torch.uint8, device="cuda")
    try:
        activations = storage[offset:offset + rows * k].view(rows, k)
        torch.manual_seed(4090)
        first = torch.randint(1, 256, (k,), dtype=torch.uint8, device="cuda")
        last = torch.randint(1, 256, (k,), dtype=torch.uint8, device="cuda")
        activations[0].copy_(first)
        activations[-1].copy_(last)

        packed, scales = quantize_mxfp4(activations)

        expected_packed, expected_scales = _reference_activation_mxfp4(last.float().unsqueeze(0))
        assert torch.equal(packed[-1], expected_packed[0])
        assert torch.equal(scales[-1], expected_scales[0])

        first_packed, first_scales = _reference_activation_mxfp4(first.float().unsqueeze(0))
        assert torch.equal(packed[0], first_packed[0])
        assert torch.equal(scales[0], first_scales[0])
    finally:
        del storage
        torch.cuda.empty_cache()


def test_macaq_row_offsets_widen_to_int64():
    """All three row-indexed kernels must widen the program id.

    The runtime regression above covers the activation quantizer; the weight
    dequantizer and fused rotate+quantize use the same row offset and are
    pinned statically because their 2^31 boundary needs a much larger tensor.
    """

    import inspect

    from gptqmodel.utils import macaq_triton

    for kernel in (
        macaq_triton._quant_mxfp4_kernel,
        macaq_triton._rot_scale_quant_kernel,
        macaq_triton._dequant_w4_to_bf16_kernel,
    ):
        assert "tl.program_id(0).to(tl.int64)" in inspect.getsource(kernel.fn)


@pytest.mark.skipif(
    not triton_mxfp4_available(), reason="MACaQ Triton kernels require NVIDIA CUDA or ROCm gfx1201"
)
def test_macaq_triton_launches_bind_the_operand_device(monkeypatch):
    """Every launch must enter the device that owns its tensors (#4090 item 3)."""

    from gptqmodel.utils import macaq_triton

    entered = []

    class SpyDevice(torch.cuda.device):
        def __init__(self, device):
            entered.append(torch.device(device))
            super().__init__(device)

    monkeypatch.setattr(torch.cuda, "device", SpyDevice)
    weight = torch.randn(8, 64, device="cuda")
    activations = torch.randn(4, 64, device="cuda", dtype=torch.bfloat16)
    bh = macaq.make_hadamard(32, device="cuda", dtype=torch.bfloat16)

    entered.clear()
    packed, scales = macaq_triton.quantize_mxfp4(weight)
    assert entered == [weight.device]

    entered.clear()
    macaq_triton.rot_scale_quantize_mxfp4(activations, None, bh)
    assert entered == [activations.device]

    entered.clear()
    macaq_triton.dequant_w4_to_bf16(packed, scales)
    assert entered == [packed.device]

    entered.clear()
    macaq_triton.mxfp4_w4a4_linear(activations, packed, scales, bh=bh)
    assert entered.count(activations.device) >= 2


@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="requires two visible CUDA/HIP devices"
)
@pytest.mark.skipif(
    not triton_mxfp4_available(), reason="MACaQ Triton kernels require NVIDIA CUDA or ROCm gfx1201"
)
def test_macaq_module_on_non_current_gpu_stays_on_triton():
    """A layer on ``cuda:1`` must not fall back when ``cuda:0`` is current."""

    _, _, _, module = _quantized_module_pair()
    other = copy.deepcopy(module).to("cuda:1")
    x = torch.randn(4, module.in_features, device="cuda:0", dtype=torch.bfloat16)
    torch.cuda.set_device(0)

    expected = module(x)
    actual = other(x.to("cuda:1"))

    assert other._triton_disabled is False
    torch.testing.assert_close(actual.cpu(), expected.cpu(), atol=2e-3, rtol=2e-3)


def _cpu_macaq_module():
    torch.manual_seed(11)
    layer = torch.nn.Linear(64, 8, bias=False)
    quantizer = macaq.Macaq(
        layer,
        MacaqConfig(
            n_flip_iters=0, do_flip=False, exp_screen=False, pre_mode="none", offload_to_disk=False
        ),
    )
    quantizer.add_batch(torch.randn(32, 64))
    result, _, _ = quantizer.quantize()
    module = MacaqLinear(bits=4, group_size=32, in_features=64, out_features=8, bias=False)
    module.pack(layer, macaq_result=result)
    module.backend = BACKEND.MACAQ_TRITON
    # Keep the latch in the decision so the tests exercise the real gate.
    module._can_use_triton = lambda x: not module._triton_disabled
    return module


def test_macaq_transient_triton_oom_does_not_latch():
    module = _cpu_macaq_module()
    calls = []

    def fail(x):
        calls.append(1)
        raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")

    module._forward_triton = fail
    x = torch.randn(2, 64, dtype=torch.bfloat16)
    module(x)
    assert module._triton_disabled is False
    module(x)
    assert len(calls) == 2


def test_macaq_permanent_triton_error_latches():
    module = _cpu_macaq_module()
    calls = []

    def fail(x):
        calls.append(1)
        raise ValueError("Pointer argument cannot be accessed from Triton (cpu tensor?)")

    module._forward_triton = fail
    x = torch.randn(2, 64, dtype=torch.bfloat16)
    module(x)
    assert module._triton_disabled is True
    module(x)
    assert len(calls) == 1


def test_macaq_tiles_never_use_block_k_below_64():
    """sm_120's native MXFP4 mma needs a K tile of at least 64 (#4090 item 4)."""

    from gptqmodel.utils import macaq_triton

    for rows, k in ((1, 32), (1, 96), (64, 96), (1, 64), (1, 128), (1, 256), (8, 160), (8, 288)):
        assert macaq_triton._pick_tiles(rows, k)[2] >= 64


def test_macaq_zero_sample_expert_uses_configured_fallback():
    """An unrouted expert must RTN-quantize instead of crashing (#4090 item 5)."""

    from gptqmodel.quantization import Fallback

    torch.manual_seed(23)
    layer = torch.nn.Linear(64, 8, bias=False)
    quantizer = macaq.Macaq(layer, MacaqConfig(n_flip_iters=16, offload_to_disk=False))
    quantizer.fallback = Fallback()
    quantizer.expected_nsamples = 4096

    result, _, loss = quantizer.quantize()

    assert result.stats["fallback"] is True
    assert math.isfinite(loss)
    # Fallback is data-independent: no D1 and no flip, but the per-block
    # exponent is still screened under the identity (plain squared-error)
    # metric instead of taking the clipping OCP exponent unconditionally.
    assert result.pre_scale is None
    bh = _reference_hadamard(32, device=layer.weight.device)
    expected_packed, expected_scales, expected_values = _reference_min_mse_mxfp4(
        layer.weight.detach(), bh, offsets=(1, 0, -1)
    )
    assert torch.equal(result.W_packed, expected_packed)
    assert torch.equal(result.W_scales.view(torch.uint8), expected_scales)
    assert torch.equal(result.dequantized_weight, expected_values)

    # Never worse than the fixed OCP exponent, and strictly better here.
    rotated = _reference_rotate(layer.weight.detach().float(), bh.float())
    _, _, rtn_values = _reference_weight_mxfp4(layer.weight.detach(), bh, offset=0)
    screened_error = (rotated - expected_values).pow(2).sum()
    rtn_error = (rotated - rtn_values).pow(2).sum()
    assert screened_error < rtn_error


def test_macaq_low_sample_expert_uses_configured_fallback():
    from gptqmodel.quantization import Fallback

    torch.manual_seed(29)
    layer = torch.nn.Linear(64, 8, bias=False)
    quantizer = macaq.Macaq(layer, MacaqConfig(n_flip_iters=16, offload_to_disk=False))
    quantizer.fallback = Fallback(threshold="0.5%")
    quantizer.expected_nsamples = 1_000_000
    quantizer.add_batch(torch.randn(1, 64))

    result, _, _ = quantizer.quantize()

    assert result.stats["fallback"] is True
    assert result.pre_scale is None


def test_macaq_zero_sample_expert_without_fallback_raises():
    layer = torch.nn.Linear(64, 8, bias=False)
    quantizer = macaq.Macaq(layer, MacaqConfig(n_flip_iters=0, offload_to_disk=False))
    quantizer.fallback = None
    quantizer.expected_nsamples = 4096

    with pytest.raises(ValueError, match="no usable calibration signal"):
        quantizer.quantize()


@pytest.mark.parametrize("offset", [-1, 2])
def test_single_offset_is_applied_to_the_rtn_path(offset):
    """A one-element `exp_offsets` must pick that exponent (#4090 item 6)."""

    torch.manual_seed(31)
    weight = torch.randn(8, 64)
    hessian = torch.eye(64)

    result = macaq.macaq_quantize_weight(
        weight, hessian, 64, exp_offsets=(offset,), do_flip=False, pre_mode="none"
    )
    plain_zero = macaq.macaq_quantize_weight(
        weight, hessian, 64, exp_offsets=(0,), do_flip=False, pre_mode="none"
    )

    bh = _reference_hadamard(32, device=weight.device)
    expected_packed, expected_scales, _ = _reference_weight_mxfp4(weight, bh, offset=offset)
    assert torch.equal(result.W_packed, expected_packed)
    assert torch.equal(result.W_scales.view(torch.uint8), expected_scales)
    assert not torch.equal(
        result.W_scales.view(torch.uint8), plain_zero.W_scales.view(torch.uint8)
    )


def test_macaq_dynamic_false_excludes_modules():
    """`dynamic={pattern: False}`, as documented, must not crash (#4090 item 7)."""

    config = MacaqConfig(
        dynamic={
            ".*down_proj": False,
            "model.layers.0.mlp.up_proj": {"do_flip": False},
        }
    )
    assert config.dynamic_get("model.layers.0.mlp.down_proj") is False
    assert config.dynamic_get("model.layers.0.mlp.up_proj") == {"do_flip": False}
    assert config.dynamic_get("model.layers.0.mlp.gate_proj") is None

    payload = config.to_dict()
    assert payload["dynamic"]["-:.*down_proj"] == {}
    restored = QuantizeConfig.from_quant_config(payload)
    assert restored.dynamic_get("model.layers.0.mlp.down_proj") is False
    assert restored.dynamic_get("model.layers.0.mlp.up_proj") == {"do_flip": False}


@pytest.mark.parametrize("method", ["macaq", "paroquant", "qqq", "nf12", "tt-blockfloat"])
def test_transformers_refuses_gptqmodel_only_quant_methods(method):
    """An unknown HF quant method only warns and random-inits (#4090 item 2)."""

    from transformers.quantizers import auto as hf_quant_auto
    from transformers.quantizers.auto import AutoHfQuantizer, AutoQuantizationConfig

    from gptqmodel.utils.hf_refusing_quantizer import (
        GPTQModelRefusingHfQuantizer,
        GPTQModelRefusingQuantizationConfig,
    )

    assert hf_quant_auto.AUTO_QUANTIZATION_CONFIG_MAPPING[method] is GPTQModelRefusingQuantizationConfig
    assert hf_quant_auto.AUTO_QUANTIZER_MAPPING[method] is GPTQModelRefusingHfQuantizer

    config = AutoQuantizationConfig.from_dict({"quant_method": method, "bits": 4})
    assert isinstance(config, GPTQModelRefusingQuantizationConfig)
    with pytest.raises(ValueError, match="cannot be loaded by `transformers`"):
        AutoHfQuantizer.from_config(config)


def test_hf_from_pretrained_refuses_macaq_checkpoint(tmp_path):
    from transformers import AutoModelForCausalLM, LlamaConfig

    model_config = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    model_config.quantization_config = {
        "quant_method": "macaq",
        "bits": 4,
        "format": "mxfp4_macaq",
    }
    model_config.save_pretrained(str(tmp_path))

    with pytest.raises(ValueError, match="MACaQ MXFP4 checkpoints cannot be loaded"):
        AutoModelForCausalLM.from_pretrained(str(tmp_path), local_files_only=True)


# ---------------------------------------------------------------------------
# rotation-free (rotation="none") weight-only MXFP4
# ---------------------------------------------------------------------------
def _identity_bh(device="cpu"):
    return torch.eye(32, dtype=torch.bfloat16, device=device)


def test_rotation_none_quantizes_in_the_original_frame():
    torch.manual_seed(41)
    weight = torch.randn(16, 128)
    hessian = torch.eye(128)

    result = macaq.macaq_quantize_weight(
        weight, hessian, 128, rotation="none", pre_mode="none",
        do_flip=False, exp_offsets=(0,), return_dequantized=True,
    )

    assert torch.equal(result.BH, _identity_bh())
    expected_packed, expected_scales, expected_values = _reference_weight_mxfp4(
        weight, _identity_bh(), offset=0
    )
    assert torch.equal(result.W_packed, expected_packed)
    assert torch.equal(result.W_scales.view(torch.uint8), expected_scales)
    assert torch.equal(result.dequantized_weight, expected_values)


def test_rotation_none_flip_refinement_reduces_hessian_error():
    torch.manual_seed(43)
    weight = torch.randn(32, 128)
    calibration = torch.randn(512, 128) * torch.linspace(0.1, 3.0, 128)
    hessian = calibration.T @ calibration

    common = {"rotation": "none", "pre_mode": "none", "return_dequantized": True}
    rtn = macaq.macaq_quantize_weight(weight, hessian, 512, do_flip=False, exp_screen=False, **common)
    refined = macaq.macaq_quantize_weight(weight, hessian, 512, n_flip_iters=64, **common)

    # No rotation: the stored frame is the original frame.
    assert _hessian_weighted_error(weight, refined.dequantized_weight, hessian) < (
        _hessian_weighted_error(weight, rtn.dequantized_weight, hessian)
    )


def test_macaq_quantizer_honors_rotation_none():
    torch.manual_seed(47)
    layer = torch.nn.Linear(128, 16, bias=False)
    config = MacaqConfig(
        rotation="none", pre_mode="none", activation_precision="w4a16",
        n_flip_iters=16, offload_to_disk=False,
    )
    quantizer = macaq.Macaq(layer, config)
    quantizer.add_batch(torch.randn(256, 128))

    result, _, loss = quantizer.quantize()

    assert math.isfinite(loss)
    assert result.pre_scale is None
    assert torch.equal(result.BH, _identity_bh())
    restored = macaq.dequantize_mxfp4(result.W_packed, result.W_scales, 128)
    relative = (restored - layer.weight.detach()).norm() / layer.weight.detach().norm()
    assert relative < 0.2


def test_macaq_config_rotation_roundtrip_and_legacy_default():
    assert MacaqConfig(offload_to_disk=False).rotation == "hadamard"

    config = MacaqConfig(rotation="none", activation_precision="w4a16", offload_to_disk=False)
    payload = config.to_dict()
    assert payload["rotation"] == "none"
    assert QuantizeConfig.from_quant_config(payload).rotation == "none"

    # Checkpoints written before the field was configurable store the
    # inherited `null` and were always rotated.
    legacy = dict(payload, rotation=None)
    assert QuantizeConfig.from_quant_config(legacy).rotation == "hadamard"
    legacy.pop("rotation")
    assert QuantizeConfig.from_quant_config(legacy).rotation == "hadamard"

    assert MacaqConfig(rotation=" None ", offload_to_disk=False).rotation == "none"
    with pytest.raises(ValueError, match="rotation"):
        MacaqConfig(rotation="random", offload_to_disk=False)

    # Accepted as a per-module `dynamic` override.
    MacaqConfig(dynamic={"mlp": {"rotation": "none"}}, offload_to_disk=False)


@pytest.mark.parametrize("rotation", ["hadamard", "none"])
def test_macaq_rotation_is_not_a_model_wide_rotation(rotation):
    # The inherited `rotation` drives model-wide rotation (fused norms, online
    # Hadamard at load, no disk offload). MACaQ's per-layer value must not.
    config = MacaqConfig(rotation=rotation)
    assert config.model_rotation is None
    assert resolve_model_rotation(config) is None
    assert config.offload_to_disk is True

    assert resolve_model_rotation(QuantizeConfig(rotation="hadamard", offload_to_disk=False)) == "hadamard"
    assert resolve_model_rotation(SimpleNamespace(rotation="random")) == "random"
    assert resolve_model_rotation(None) is None


@pytest.mark.parametrize(
    ("settings", "expected"),
    [
        ({}, FORMAT.MXFP4_MACAQ),
        ({"rotation": "none"}, FORMAT.MXFP4_MACAQ),
        ({"pre_mode": "none"}, FORMAT.MXFP4_MACAQ),
        ({"rotation": "none", "pre_mode": "none"}, FORMAT.MXFP4),
        # An explicit format never overrides what the settings store.
        ({"format": FORMAT.MXFP4}, FORMAT.MXFP4_MACAQ),
        ({"rotation": "none", "pre_mode": "none", "format": FORMAT.MXFP4_MACAQ}, FORMAT.MXFP4),
        # Any module that keeps a transform makes the checkpoint MACaQ-specific.
        ({"rotation": "none", "pre_mode": "none", "dynamic": {"mlp": {"rotation": "hadamard"}}},
         FORMAT.MXFP4_MACAQ),
        ({"rotation": "none", "pre_mode": "none", "dynamic": {"mlp": {"pre_mode": "balance"}}},
         FORMAT.MXFP4_MACAQ),
        ({"rotation": "none", "pre_mode": "none", "dynamic": {"mlp": {"n_flip_iters": 4}, "attn": False}},
         FORMAT.MXFP4),
        ({"dynamic": {"mlp": {"rotation": "none", "pre_mode": "none"}}}, FORMAT.MXFP4_MACAQ),
    ],
)
def test_macaq_format_follows_rotation_and_pre_mode(settings, expected):
    assert MacaqConfig(offload_to_disk=False, **settings).format == expected


def test_plain_mxfp4_config_round_trips_as_macaq():
    assert FORMAT.MXFP4.value == "mxfp4"
    config = MacaqConfig(rotation="none", pre_mode="none", activation_precision="w4a16",
                         offload_to_disk=False)
    payload = config.to_dict()
    assert payload["format"] == "mxfp4"
    assert payload["method"] == "macaq"
    assert config.export_quant_method() == METHOD.MACAQ
    assert resolve_quant_format(config.format, config.method) == FORMAT.MXFP4
    assert config.quant_linear_init_kwargs()["rotation"] == "none"

    restored = QuantizeConfig.from_quant_config(payload)
    assert isinstance(restored, MacaqConfig)
    assert restored.format == FORMAT.MXFP4
    assert (restored.rotation, restored.pre_mode) == ("none", "none")


def test_gguf_mxfp4_subtype_is_not_claimed_by_macaq():
    # GGUF also names a subtype `mxfp4`; only an explicit MACaQ method maps
    # the plain `mxfp4` format to MACaQ.
    for payload in (
        {"method": "gguf", "format": "mxfp4", "bits": 4},
        {"format": "mxfp4", "bits": 4},
        {"method": "gguf", "bits": "mxfp4"},
    ):
        config = QuantizeConfig.from_quant_config(payload)
        assert isinstance(config, GGUFConfig)
        assert config.format == "mxfp4"
        assert resolve_quant_format(config.format, config.method) == FORMAT.GGUF
    assert GGUFConfig(format="mxfp4").format == "mxfp4"


def _packed_module(result, *, activation_precision, rotation="hadamard", pre_mode="balance",
                   backend=BACKEND.MACAQ_TORCH, cls=MacaqLinear,
                   in_features=128, out_features=16, bias=None, device="cpu"):
    module = cls(
        bits=4, group_size=32, in_features=in_features, out_features=out_features,
        bias=bias is not None, pack_dtype=torch.int32, block_size=32, had_block=32,
        activation_precision=activation_precision, rotation=rotation, pre_mode=pre_mode,
        dtype=torch.bfloat16, backend=backend,
    ).to(device)
    holder = torch.nn.Linear(in_features, out_features, bias=bias is not None).to(device)
    if bias is not None:
        holder.bias.data.copy_(bias)
    module.pack(holder, macaq_result=result)
    return module


def _rotation_free_result(device="cpu", in_features=128, out_features=16):
    torch.manual_seed(53)
    weight = torch.randn(out_features, in_features, device=device)
    calibration = torch.randn(256, in_features, device=device)
    return macaq.macaq_quantize_weight(
        weight, calibration.T @ calibration, 256, rotation="none", pre_mode="none",
        n_flip_iters=8,
    )


def _plain_and_forced_modules(result, **kwargs):
    """The plain layer and the same result loaded with explicit identity transforms."""
    plain = _packed_module(result, rotation="none", pre_mode="none", **kwargs)
    forced = _packed_module(result, rotation="hadamard", pre_mode="balance", **kwargs)
    assert torch.equal(forced.BH, _identity_bh(forced.BH.device))
    assert bool((forced.pre_scale == 1).all())
    return plain, forced


@pytest.mark.parametrize("activation_precision", ["w4a16", "w4a4"])
def test_plain_mxfp4_layer_stores_only_mxfp4_tensors(activation_precision):
    result = _rotation_free_result()
    bias = torch.randn(16)
    plain, forced = _plain_and_forced_modules(result, activation_precision=activation_precision, bias=bias)

    assert plain.plain_mxfp4 and not forced.plain_mxfp4
    assert plain.BH is None and plain.pre_scale is None
    assert set(plain.state_dict()) == {"W_packed", "W_scales", "bias"}

    # Dropping the identity transforms changes no output bit.
    x = torch.randn(3, 7, 128, dtype=torch.bfloat16)
    fast = plain(x)
    assert torch.equal(forced(x), fast)

    if activation_precision == "w4a16":
        weight = macaq.dequantize_mxfp4(result.W_packed, result.W_scales, 128, dtype=torch.bfloat16)
        expected = torch.nn.functional.linear(x, weight, bias.to(torch.bfloat16))
        assert torch.equal(fast, expected)


def test_plain_mxfp4_state_dict_round_trips_strictly():
    source = _packed_module(_rotation_free_result(), activation_precision="w4a16",
                            rotation="none", pre_mode="none")
    target = MacaqLinear(
        bits=4, group_size=32, in_features=128, out_features=16, bias=False,
        pack_dtype=torch.int32, activation_precision="w4a16", rotation="none", pre_mode="none",
        dtype=torch.bfloat16,
    )
    target.load_state_dict(source.state_dict(), strict=True)
    target.post_init()
    x = torch.randn(5, 128, dtype=torch.bfloat16)
    assert torch.equal(target(x), source(x))


def test_rotated_layer_keeps_its_transforms():
    _, result, module, _ = _quantized_module_pair()
    assert not module.plain_mxfp4
    assert {"BH", "pre_scale"} <= set(module.state_dict())


def test_plain_layer_rejects_transformed_results():
    rotated = macaq.macaq_quantize_weight(torch.randn(16, 128), torch.eye(128), 128, n_flip_iters=0)
    with pytest.raises(ValueError, match="rotated result"):
        _packed_module(rotated, activation_precision="w4a16", rotation="none", pre_mode="none")

    torch.manual_seed(59)
    calibration = torch.randn(256, 128)
    scaled = macaq.macaq_quantize_weight(
        torch.randn(16, 128), calibration.T @ calibration, 256, rotation="none", n_flip_iters=0,
    )
    with pytest.raises(ValueError, match="pre-scale"):
        _packed_module(scaled, activation_precision="w4a16", rotation="none", pre_mode="none")
    # The same result fits a rotation-free layer that keeps D1.
    module = _packed_module(scaled, activation_precision="w4a16", rotation="none", pre_mode="balance")
    assert module.BH is None and module.pre_scale is not None


@pytest.mark.skipif(
    not triton_mxfp4_available(),
    reason="MACaQ Triton kernels require NVIDIA CUDA or ROCm gfx1201",
)
@pytest.mark.parametrize("activation_precision", ["w4a16", "w4a4"])
def test_triton_plain_mxfp4_matches_forced_transforms(activation_precision):
    device = torch.device("cuda:0")
    result = _rotation_free_result(device=device, in_features=256, out_features=128)
    plain, forced = _plain_and_forced_modules(
        result, activation_precision=activation_precision, backend=BACKEND.MACAQ_TRITON,
        cls=MacaqTritonLinear, in_features=256, out_features=128, device=device,
    )
    x = torch.randn(9, 256, device=device, dtype=torch.bfloat16)
    fast = plain(x)
    assert not plain._triton_disabled
    assert torch.equal(forced(x), fast)
