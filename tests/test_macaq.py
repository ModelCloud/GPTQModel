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

import pytest
import torch

from gptqmodel.nn_modules.qlinear.macaq import MacaqLinear, MacaqTritonLinear
from gptqmodel.quantization import FORMAT, METHOD, MacaqConfig, QuantizeConfig
from gptqmodel.quantization import macaq as macaq
from gptqmodel.utils.backend import BACKEND
from gptqmodel.utils.macaq_triton import TRITON_AVAILABLE


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
        had_block=64,
        ridge=0.5,
        pre_alpha=0.75,
        exp_offsets=(1, 0),
        activation_precision="w4a16",
        offload_to_disk=False,
    )
    payload = config.to_dict()
    assert payload["method"] == METHOD.MACAQ
    assert payload["format"] == FORMAT.MACAQ
    assert payload["n_flip_iters"] == 12
    assert payload["had_block"] == 64
    assert payload["exp_offsets"] == [1, 0]
    assert payload["activation_precision"] == "w4a16"
    assert payload["sym"] is True

    restored = QuantizeConfig.from_quant_config(payload)
    assert isinstance(restored, MacaqConfig)
    assert restored.n_flip_iters == 12
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
    not (TRITON_AVAILABLE and torch.cuda.is_available()),
    reason="MACaQ Triton kernels require CUDA and Triton",
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
    not (TRITON_AVAILABLE and torch.cuda.is_available()),
    reason="MACaQ Triton kernels require CUDA and Triton",
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
        a_packed, w_packed, a_scales, w_scales, out,
        rows, cols, k,
        a_packed.stride(0), w_packed.stride(0),
        a_scales.stride(0), w_scales.stride(0), out.stride(0),
        OUT_SCALE=1.0, BLOCK_M=64, BLOCK_N=128, BLOCK_K=256,
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
