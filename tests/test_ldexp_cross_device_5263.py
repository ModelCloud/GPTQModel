# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""``torch.ldexp`` faults on a device other than the current one.

``torch.ldexp`` launches on the *current* CUDA device, so a tensor that lives
on another GPU -- ``device="cuda:1"``, or any ``device_map`` split while the
current device stays 0 -- aborts the process with an illegal memory access
(#5263). EXL3's BF16 pre-scale hit it on the default bf16 dtype of the
checkpoints, before the first GEMM.

The paths that only need a power of two now use ``torch.pow(2, exponent)``,
which is bit-exact over the whole exponent range. The NVFP4 candidate search
keeps ``ldexp`` because it scales arbitrary values exactly (a ``2**-e``
multiplier can overflow where ``ldexp`` does not) and enters the tensor's
device instead.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from gptqmodel.nn_modules.exllamav3 import ExllamaV3Linear
from gptqmodel.nn_modules.exllamav3_bf16_kernels import FP16_SAFE_AMAX
from gptqmodel.nn_modules.qlinear._numerics import tensor_device_context
from gptqmodel.quantization.activation_floatx import _nvfp4_mse_quantize_blocks
from gptqmodel.quantization.dtype import decode_e8m0_scale
from gptqmodel.utils.kimi_k3 import decode_mxfp4


REPO_ROOT = Path(__file__).resolve().parents[1]
HAS_CUDA = torch.cuda.is_available()

requires_cuda = pytest.mark.skipif(not HAS_CUDA, reason="CUDA device required")
requires_two_gpus = pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="needs a second CUDA device"
)


def _forbid_ldexp(monkeypatch) -> None:
    """Make any ``torch.ldexp`` call fail the test."""

    def _ldexp(*args, **kwargs):
        raise AssertionError(
            "torch.ldexp launches on the current device and faults on another "
            "GPU (#5263); use torch.pow(2, exponent) or a device guard"
        )

    monkeypatch.setattr(torch, "ldexp", _ldexp)


def _bf16_prescale_module() -> ExllamaV3Linear:
    """An ``ExllamaV3Linear`` without ``nn.Module.__init__`` for the prescale."""
    module = object.__new__(ExllamaV3Linear)
    module.out_features = 3
    module.bias = None
    return module


def _bf16_wide_input(device) -> torch.Tensor:
    """One row above the FP16-safe bound, one row below it."""
    return torch.tensor(
        [[1e4, -2e4, 3.0], [1.0, 2.0, 3.0]], dtype=torch.bfloat16, device=device
    )


def _prescale_reference(module, x):
    """The pre-#5263 eager expression, evaluated on the tensors' own device."""
    row_amax = module._row_amax(x)
    _, exponent = torch.frexp(row_amax / FP16_SAFE_AMAX)
    # ``torch.ldexp`` launches on the *current* device, so this reference must
    # enter the operand's device too; otherwise the oracle faults instead of
    # the code under test (#5263). CPU needs no context.
    with tensor_device_context(row_amax):
        scale = torch.where(
            row_amax > FP16_SAFE_AMAX,
            torch.ldexp(torch.ones_like(row_amax), exponent),
            torch.ones_like(row_amax),
        )
    return (x / scale).to(torch.float16), scale


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_bf16_prescale_never_calls_ldexp(monkeypatch, device):
    if device == "cuda" and not HAS_CUDA:
        pytest.skip("CUDA device required")

    module = _bf16_prescale_module()
    x = _bf16_wide_input(device)
    expected_input, expected_scale = _prescale_reference(module, x)

    _forbid_ldexp(monkeypatch)
    native_input, output_scale = module._prescale_from_row_amax(x, module._row_amax(x))

    # ``pow(2, exponent)`` reproduces the old ``ldexp`` scale bit for bit.
    assert torch.equal(output_scale.view(torch.int32), expected_scale.view(torch.int32))
    assert torch.equal(native_input.view(torch.int16), expected_input.view(torch.int16))
    # Only the wide row is rescaled, and the result stays inside FP16 range.
    assert output_scale[0, 0].item() == 4.0
    assert output_scale[1, 0].item() == 1.0
    assert float(native_input.float().abs().max()) <= FP16_SAFE_AMAX


@requires_two_gpus
def test_bf16_prescale_on_a_non_current_device():
    """The issue's repro: bf16 pre-scale on ``cuda:1`` while device 0 is current."""
    module = _bf16_prescale_module()
    x = _bf16_wide_input("cuda:1")

    original = torch.cuda.current_device()
    try:
        torch.cuda.set_device(0)
        assert torch.cuda.current_device() == 0
        native_input, output_scale = module._prescale_from_row_amax(x, module._row_amax(x))
        torch.cuda.synchronize(1)
    finally:
        torch.cuda.set_device(original)

    assert native_input.device.index == 1
    expected_input, expected_scale = _prescale_reference(module, x)
    assert torch.equal(output_scale.view(torch.int32), expected_scale.view(torch.int32))
    assert torch.equal(native_input.view(torch.int16), expected_input.view(torch.int16))


def test_e8m0_decode_never_calls_ldexp(monkeypatch):
    """Every E8M0 byte except 255 must decode bit-exactly without ``ldexp``."""
    encoded = torch.arange(255, dtype=torch.uint8)
    # FP64 ``2**exponent`` is exact for the whole E8M0 range.
    expected = torch.pow(2.0, encoded.double() - 127).float()

    _forbid_ldexp(monkeypatch)
    decoded = decode_e8m0_scale(encoded)

    assert torch.equal(decoded.view(torch.int32), expected.view(torch.int32))
    # Byte 0 is the denormal 2**-127 that ``exp2`` rounds one ulp low on CUDA.
    assert decoded[0].item() == 2.0**-127


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_mxfp4_decode_never_calls_ldexp(monkeypatch, device):
    if device == "cuda" and not HAS_CUDA:
        pytest.skip("CUDA device required")

    packed = torch.full((2, 16), 0x99, dtype=torch.uint8, device=device)  # two -0.5
    scales = torch.tensor([[0], [254]], dtype=torch.uint8, device=device)
    expected = (-0.5 * torch.pow(2.0, scales.double() - 127)).expand(2, 32).float()

    _forbid_ldexp(monkeypatch)
    decoded = decode_mxfp4(packed, scales, dtype=torch.float32)

    assert torch.equal(decoded, expected)


def test_nvfp4_block_search_enters_the_tensor_device(monkeypatch):
    """The one remaining ``ldexp`` site must select the tensors' device."""
    recorded = []
    real_device = torch.cuda.device

    class _RecordingDevice(real_device):
        # A subclass keeps ``isinstance(..., torch.cuda.device)`` working for
        # torch's own device-index helper while recording each guard.
        def __init__(self, device=None, *args, **kwargs):
            recorded.append(device)
            super().__init__(device, *args, **kwargs)

    monkeypatch.setattr(torch.cuda, "device", _RecordingDevice)
    blocks = torch.linspace(1.0, 1e4, 16).reshape(1, 1, 16)
    global_scale = torch.tensor(2.0**-9, dtype=torch.float32)

    _nvfp4_mse_quantize_blocks(blocks, global_scale)

    # A CPU tensor must not enter a CUDA device context at all.
    assert recorded == []

    if not HAS_CUDA:
        return
    blocks = blocks.cuda()
    global_scale = global_scale.cuda()
    _nvfp4_mse_quantize_blocks(blocks, global_scale)
    # ``evaluate`` runs once per candidate seed and refinement.
    assert recorded
    assert all(device == blocks.device for device in recorded)


def test_no_unguarded_ldexp_outside_the_nvfp4_search():
    """Only the device-guarded sites may call ``torch.ldexp`` (#5263)."""
    offenders = sorted(
        path.relative_to(REPO_ROOT).as_posix()
        for path in (REPO_ROOT / "gptqmodel").rglob("*.py")
        if "torch.ldexp(" in path.read_text(encoding="utf-8")
    )

    # The NVFP4 candidate search scales arbitrary values exactly and keeps
    # ``ldexp``. The bitsandbytes INT8 FP32 epilogue (#5694) splits both absmaxes
    # into mantissa/exponent and needs the same exact rescaling. MACAQ's
    # residual screen normalizes the residual to the block's own power of two.
    # All three must enter the tensors' device.
    assert offenders == [
        "gptqmodel/nn_modules/qlinear/bitsandbytes.py",
        "gptqmodel/quantization/activation_floatx.py",
        "gptqmodel/quantization/macaq.py",
    ]
    nvfp4 = (REPO_ROOT / "gptqmodel/quantization/activation_floatx.py").read_text(
        encoding="utf-8"
    )
    # The NVFP4 site keeps its inline device guard.
    assert "with torch.cuda.device(" in nvfp4
    assert "nullcontext()" in nvfp4
    bnb = (REPO_ROOT / "gptqmodel/nn_modules/qlinear/bitsandbytes.py").read_text(
        encoding="utf-8"
    )
    # The bitsandbytes site routes through the shared guard helper.
    assert "tensor_device_context(" in bnb
    macaq = (REPO_ROOT / "gptqmodel/quantization/macaq.py").read_text(
        encoding="utf-8"
    )
    assert "tensor_device_context(" in macaq
    helper = (REPO_ROOT / "gptqmodel/nn_modules/qlinear/_numerics.py").read_text(
        encoding="utf-8"
    )
    assert "def tensor_device_context(" in helper
    assert "torch.cuda.device(" in helper
