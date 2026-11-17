# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""AWQ packing must not destroy a finite bias in a narrow storage dtype.

The native AWQ GEMM, GEMV and GEMV_FAST packers used to store the
non-quantized bias in the scale dtype (FP16). An FP32 ``-100000`` is ``-inf``
in FP16, so every packed layer with such a bias shipped an infinite offset and
no forward-time fix could recover it. The bias is not a quantization operand:
it is added after the kernel's reduction, so the packers now keep the source
width and each forward widens it in the FP32 epilogue instead.
"""

from __future__ import annotations

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from gptqmodel.nn_modules.qlinear.gemm_awq import AwqGEMMLinear
from gptqmodel.nn_modules.qlinear.gemm_awq_triton import AwqGEMMTritonLinear
from gptqmodel.nn_modules.qlinear.gemv_awq import (
    AwqGEMVLinear,
    AwqGEMVTorchLinear,
    AwqGEMVTritonLinear,
)
from gptqmodel.nn_modules.qlinear.gemv_fast_awq import AwqGEMVFastLinear, LLMAwqLinear
from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
from gptqmodel.utils.model import load_checkpoint_in_model_then_tie_weights
from gptqmodel.utils.packing import hf_pack_layer


IN_FEATURES = 128
OUT_FEATURES = 128
GROUP_SIZE = 128

NATIVE_AWQ_KERNELS = (AwqGEMMLinear, AwqGEMVLinear, AwqGEMVFastLinear)


def _zeros_name(cls) -> str:
    if cls is LLMAwqLinear:
        return "scaled_zeros"
    return "qzeros"


def _pack_dtype(cls) -> torch.dtype:
    return torch.int16 if cls in (AwqGEMVFastLinear, LLMAwqLinear) else torch.int32


def _bare_module(cls) -> nn.Module:
    """Build the attributes each native packer reads, without its GPU probe."""

    module = cls.__new__(cls)
    nn.Module.__init__(module)
    module.bits = 4
    module.group_size = GROUP_SIZE
    module.in_features = IN_FEATURES
    module.out_features = OUT_FEATURES
    module.pack_dtype = _pack_dtype(cls)
    module.adapter = None
    module.zeros_name = _zeros_name(cls)
    module.sym = True
    module.desc_act = False
    return module


def _linear(bias, *, bias_dtype=None) -> nn.Linear:
    linear = nn.Linear(IN_FEATURES, OUT_FEATURES, bias=bias is not None, dtype=torch.float32)
    with torch.no_grad():
        linear.weight.fill_(1.0)
        if bias is not None:
            linear.bias.fill_(bias)
    if bias_dtype is not None and linear.bias is not None:
        linear.bias.data = linear.bias.data.to(bias_dtype)
    return linear


def _affine():
    scales = torch.ones((OUT_FEATURES, IN_FEATURES // GROUP_SIZE), dtype=torch.float32)
    zeros = torch.zeros((OUT_FEATURES, IN_FEATURES // GROUP_SIZE), dtype=torch.int32)
    return scales, zeros


# Every value is finite in its source dtype. ``65520`` and ``-65520`` overflow
# FP16, ``65504`` is the largest finite FP16 value, ``nextafter`` steps just
# inside/outside it, and ``1e-45`` underflows FP16 to zero. All of them must
# survive packing at the source width.
BIAS_VALUES = (
    -100000.0,
    100000.0,
    65520.0,
    -65520.0,
    65504.0,
    float(torch.nextafter(torch.tensor(65520.0), torch.tensor(0.0))),
    float(torch.nextafter(torch.tensor(-65520.0), torch.tensor(0.0))),
    -1e-45,
    0.0,
)


@pytest.mark.cpu
@pytest.mark.parametrize("cls", NATIVE_AWQ_KERNELS + (LLMAwqLinear,))
@pytest.mark.parametrize("bias_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("bias_value", BIAS_VALUES)
def test_pack_keeps_source_bias_width(cls, bias_dtype, bias_value):
    module = _bare_module(cls)
    linear = _linear(bias_value, bias_dtype=bias_dtype)
    scales, zeros = _affine()

    module.pack(linear, scales, zeros)

    assert module.bias is not None
    assert module.bias.dtype is bias_dtype, (cls.__name__, module.bias.dtype)
    assert torch.isfinite(module.bias).all()
    assert torch.equal(module.bias.cpu(), linear.bias.detach().cpu())


@pytest.mark.cpu
@pytest.mark.parametrize("cls", NATIVE_AWQ_KERNELS + (LLMAwqLinear,))
def test_hf_pack_layer_keeps_unrepresentable_bias(cls):
    """The public AWQ branch must not add a second narrowing step."""

    module = _bare_module(cls)
    linear = _linear(-100000.0)
    scales, zeros = _affine()

    hf_pack_layer(module, linear, scales, zeros)

    assert module.bias.dtype is torch.float32
    assert torch.equal(module.bias, torch.full_like(module.bias, -100000.0))


@pytest.mark.cpu
@pytest.mark.parametrize("cls", NATIVE_AWQ_KERNELS + (LLMAwqLinear,))
def test_pack_without_bias_registers_no_buffer(cls):
    module = _bare_module(cls)
    linear = _linear(None)
    scales, zeros = _affine()

    module.pack(linear, scales, zeros)

    assert module.bias is None
    assert "bias" not in module._buffers


@pytest.mark.cpu
def test_packed_bias_survives_safetensors_round_trip(tmp_path):
    """Pack -> save -> reload must not narrow the stored bias."""

    module = _bare_module(AwqGEMVLinear)
    linear = _linear(-65520.0)
    scales, zeros = _affine()
    module.pack(linear, scales, zeros)

    checkpoint = tmp_path / "model.safetensors"
    save_file(
        {
            f"proj.{name}": tensor.contiguous()
            for name, tensor in module.state_dict().items()
        },
        str(checkpoint),
        metadata={"format": "pt"},
    )

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = AwqGEMVLinear(
                bits=4,
                group_size=GROUP_SIZE,
                sym=True,
                desc_act=False,
                in_features=IN_FEATURES,
                out_features=OUT_FEATURES,
                bias=True,
                register_buffers=True,
            )

        def tie_weights(self):
            pass

    model = _Model()
    assert model.proj.bias.dtype is torch.float16  # placeholder width
    load_checkpoint_in_model_then_tie_weights(
        model,
        checkpoint=str(checkpoint),
        dtype=torch.float16,
        device_map={"": "cpu"},
    )

    loaded = model.proj.bias
    assert loaded.dtype is torch.float32, loaded.dtype
    assert torch.equal(loaded.cpu(), torch.full_like(loaded.cpu(), -65520.0))


# --------------------------------------------------------------------------
# Forward: a wide bias must cancel a finite product before the output rounds.
# --------------------------------------------------------------------------

def _packed_unit_module(cls, bias_value: float, *, module_cls=None):
    """Pack a layer whose dequantized weight is exactly one."""

    module = _bare_module(cls) if module_cls is None else module_cls
    if cls is AwqGEMMLinear or module_cls is not None:
        # `_bare_module` skips the constructor, which also owns the resolved
        # split-K policy; the bare kernel reads it at forward time.
        module.fp32_accum = True
        module.kernel = None
    linear = _linear(bias_value)
    scales, zeros = _affine()
    module.pack(linear, scales, zeros)
    return module, linear.bias.detach().clone()


def _assert_cancellation(module, bias, *, rows: int, input_value: float, expected: float, dtype):
    x = torch.full((rows, IN_FEATURES), input_value, dtype=dtype, device=module.bias.device)
    oracle = torch.full(
        (rows, OUT_FEATURES),
        input_value * IN_FEATURES + bias.double().mean().item(),
        dtype=torch.float64,
    )
    out = module(x)

    assert out.dtype is dtype
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out.cpu().double(), oracle, rtol=0, atol=1e-2)
    assert out.flatten()[0].item() == pytest.approx(expected, abs=1e-2)
    # The stored bias is still the checkpoint value, not a rounded copy.
    assert module.bias.dtype is bias.dtype
    assert torch.equal(module.bias.cpu(), bias.cpu())


@pytest.mark.cpu
def test_portable_torch_forward_cancels_wide_bias():
    module = AwqTorchLinear(
        bits=4,
        group_size=GROUP_SIZE,
        sym=True,
        desc_act=False,
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        bias=True,
        register_buffers=False,
    )
    module.pack(_linear(-100000.0), *_affine())
    module.post_init()
    bias = module.bias.detach().clone()

    _assert_cancellation(
        module, bias,
        rows=1, input_value=1000.0,
        expected=1000.0 * IN_FEATURES - 100000.0,
        dtype=torch.float16,
    )


@pytest.mark.cpu
def test_portable_gemv_forward_cancels_wide_bias():
    module = AwqGEMVTorchLinear(
        bits=4,
        group_size=GROUP_SIZE,
        sym=True,
        desc_act=False,
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        bias=True,
        register_buffers=False,
    )
    module.pack(_linear(-100000.0), *_affine())
    bias = module.bias.detach().clone()

    _assert_cancellation(
        module, bias,
        rows=1, input_value=1000.0,
        expected=1000.0 * IN_FEATURES - 100000.0,
        dtype=torch.float16,
    )


def _native_available(cls) -> tuple[bool, str]:
    """Whether the native CUDA AWQ extension can actually launch here.

    ``validate_once`` reports the device family the kernel declares; the JIT
    extension itself is CUDA-only, so a HIP host must skip rather than assert.
    """

    from gptqmodel.utils.awq import awq_runtime_available

    ok, error = cls.validate_once()
    if not ok:
        return False, str(error)
    if not torch.cuda.is_available():
        return False, "CUDA device required"
    if not awq_runtime_available():
        return False, "native CUDA AWQ extension unavailable"
    return True, ""


@pytest.mark.parametrize("cls", NATIVE_AWQ_KERNELS)
def test_native_forward_cancels_wide_bias(cls):
    ok, reason = _native_available(cls)
    if not ok:
        pytest.skip(reason)

    # GEMM forms the reduction in FP32, so the whole 128000 product can
    # cancel. The native GEMV kernels round their own partial result to FP16
    # before the Python epilogue, so keep that product inside the FP16 range.
    bias_value, input_value = (
        (-100000.0, 1000.0) if cls is AwqGEMMLinear else (-10000.0, 100.0)
    )
    module, bias = _packed_unit_module(cls, bias_value)
    module = module.cuda()

    _assert_cancellation(
        module, bias,
        rows=1, input_value=input_value,
        expected=input_value * IN_FEATURES + bias_value,
        dtype=torch.float16,
    )


def _assert_triton_gemm_cancellation():
    if not torch.cuda.is_available():
        pytest.skip("accelerator required")

    # The Triton GEMM reader has no packer of its own: it reads the buffer
    # layout `AwqGEMMLinear.pack` produces, so pack there and copy the state.
    # `AwqGEMMLinear` refuses to construct on ROCm (the native kernel is
    # CUDA-only) while the Triton reader serves the same packed layout there,
    # so build the packing-only instance directly.
    packed = _bare_module(AwqGEMMLinear)
    packed.pack(_linear(-100000.0), *_affine())
    bias = packed.bias.detach().clone()

    module = AwqGEMMTritonLinear(
        bits=4,
        group_size=GROUP_SIZE,
        sym=True,
        desc_act=False,
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        bias=True,
        register_buffers=True,
    )
    # Allocate the bias at the packed width: `load_state_dict` copies into the
    # existing buffer, so an FP16 placeholder would narrow the value first.
    # This is the same width `restore_packed_module` rebuilds from the
    # checkpoint's recorded `bias_dtype`.
    module.bias = module.bias.to(dtype=packed.bias.dtype)
    module.load_state_dict(packed.state_dict(), strict=False)
    module = module.cuda()

    _assert_cancellation(
        module, bias,
        rows=1, input_value=1000.0,
        expected=1000.0 * IN_FEATURES - 100000.0,
        dtype=torch.float16,
    )


def test_triton_gemm_forward_cancels_wide_bias():
    """The Triton GEMM reader is the AWQ GEMM path on ROCm."""

    _assert_triton_gemm_cancellation()


def test_triton_gemv_forward_cancels_wide_bias():
    """The Triton GEMV reader inherits the fixed packer on ROCm and CUDA."""

    if not torch.cuda.is_available():
        pytest.skip("accelerator required")

    module = AwqGEMVTritonLinear(
        bits=4,
        group_size=GROUP_SIZE,
        sym=True,
        desc_act=False,
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        bias=True,
        register_buffers=False,
    )
    module.pack(_linear(-10000.0), *_affine())
    bias = module.bias.detach().clone()
    module = module.cuda()

    _assert_cancellation(
        module, bias,
        rows=1, input_value=100.0,
        expected=100.0 * IN_FEATURES - 10000.0,
        dtype=torch.float16,
    )
