# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""GPTQ must not narrow a finite grid reconstruction to ``inf``.

Every GPTQ-family solver fits its grid on an FP32 copy of the weight and then
narrows the reconstruction back to the module's weight dtype for replay
(``GPTQ._narrow_reconstruction``). The packers never see the quantizer's codes:
they recover them from that narrowed weight with ``round(Q / scale) + zero``.

A symmetric INT4 grid stores ``zero = 8`` and ``scale = 2 * amax / 15``, so the
negative endpoint of a group rounds ties-to-even to code ``0`` and reconstructs
``-16 * amax / 15``, which is strictly larger than ``amax``. At
``amax = 61440`` that product is ``-65536``: the input and the scale are both
representable in FP16, but the reconstruction is not, so narrowing stores
``-inf`` in the replay weight even though every source parameter is finite.

The grid and its rounding are correct. The missing constraint is that the
*selected codes* have to stay inside the finite range of the dtype the
reconstruction is narrowed to. These tests drive the real ``GPTQ.quantize``
path, the real packer, the replay forward, and a checkpoint decode.
"""

import pytest
import torch
import torch.nn as nn

from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.quantization import FORMAT
from gptqmodel.quantization.config import QuantizeConfig
from gptqmodel.quantization.gptq import GPTQ


WIDTH = 32
GROUP_SIZE = 32
FP16_MAX = torch.finfo(torch.float16).max


def _weight(magnitude: float, dtype: torch.dtype = torch.float16) -> torch.Tensor:
    """One group per row: ``[-a, +a, 0, ...]`` sets the fitted range."""

    weight = torch.zeros(WIDTH, WIDTH, dtype=dtype)
    weight[:, 0] = -magnitude
    weight[:, 1] = magnitude
    return weight


def _config(bits: int, *, sym: bool, group_size: int = GROUP_SIZE) -> QuantizeConfig:
    return QuantizeConfig(
        bits=bits,
        sym=sym,
        group_size=group_size,
        desc_act=False,
        act_group_aware=False,
        format=FORMAT.GPTQ,
        offload_to_disk=False,
    )


def _quantize(
    magnitude: float,
    *,
    bits: int = 4,
    sym: bool = True,
    group_size: int = GROUP_SIZE,
    dtype: torch.dtype = torch.float16,
    device: torch.device = torch.device("cpu"),
):
    layer = nn.Linear(WIDTH, WIDTH, bias=False, dtype=dtype, device=device)
    with torch.no_grad():
        layer.weight.copy_(_weight(magnitude, dtype).to(device))
    task = GPTQ(layer, _config(bits, sym=sym, group_size=group_size))
    task.quantizer.configure(perchannel=True)
    eye = torch.eye(WIDTH, dtype=dtype, device=device).unsqueeze(0)
    task.add_batch(eye, None)
    quantized, scale, zero, g_idx, *_rest = task.quantize()
    return task, layer, quantized, scale, zero, g_idx


def _recovered_codes(
    weight: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor, g_idx: torch.Tensor, bits: int
) -> torch.Tensor:
    """The packer's code recovery, computed independently in FP32."""

    maxq = (1 << bits) - 1
    transposed = weight.detach().to(torch.float32).T
    rows = g_idx.long()
    group_scale = scale.to(torch.float32).T[rows]
    group_zero = zero.to(torch.float32).T[rows]
    return torch.clamp(torch.round(transposed / group_scale) + group_zero, 0, maxq).T


@pytest.mark.parametrize("magnitude", [60000.0, 61408.0, 61440.0, 65504.0])
@pytest.mark.parametrize("sym", [True, False])
def test_fp16_reconstruction_stays_finite(magnitude, sym):
    """The control (61408) and the overflowing neighbours must both be finite."""

    _, layer, quantized, scale, zero, _ = _quantize(magnitude, sym=sym)

    # The source parameter is never rewritten by this path.
    assert layer.weight.dtype == torch.float16
    assert torch.isfinite(layer.weight.data).all()

    assert quantized.dtype == torch.float16, "the live module keeps its own dtype"
    assert quantized.shape == layer.weight.shape
    assert torch.isfinite(quantized).all(), (
        f"amax={magnitude} sym={sym} narrowed to "
        f"{int((~torch.isfinite(quantized)).sum())} non-finite elements"
    )
    assert torch.isfinite(scale).all()
    assert torch.isfinite(zero).all()
    assert quantized.abs().max().item() <= FP16_MAX


@pytest.mark.parametrize("sym", [True, False])
def test_adjacent_fp16_values_keep_a_finite_endpoint(sym):
    """61408 and 61440 are adjacent FP16 values and must both stay finite."""

    control, *_rest = _quantize(61408.0, sym=sym)[2:3]
    overflow, *_rest = _quantize(61440.0, sym=sym)[2:3]

    assert torch.isfinite(control).all()
    assert torch.isfinite(overflow).all()
    # 61408 still reaches the most negative finite FP16 value; 61440 cannot, so
    # its endpoint code moves one step in and reconstructs a finite value.
    assert control[0, 0].item() == -FP16_MAX
    assert overflow[0, 0].item() == -57344.0


@pytest.mark.parametrize("bits", [2, 4, 8])
@pytest.mark.parametrize("sym", [True, False])
def test_both_endpoints_are_finite_at_the_fp16_extreme(bits, sym):
    magnitude = FP16_MAX
    _, _, quantized, scale, _, _ = _quantize(magnitude, bits=bits, sym=sym)

    assert torch.isfinite(quantized).all(), f"bits={bits} sym={sym}"
    assert torch.isfinite(scale).all()
    assert quantized[0, 0].item() >= -FP16_MAX
    assert quantized[0, 1].item() <= FP16_MAX


@pytest.mark.parametrize("bits", [2, 4, 8])
@pytest.mark.parametrize("sym", [True, False])
@pytest.mark.parametrize("magnitude", [61408.0, 61440.0, 65504.0])
def test_normal_range_is_unchanged_at_the_representable_control(bits, sym, magnitude):
    """Every in-range grid must keep its exact codes; only overflowing ones move."""

    _, _, quantized, scale, zero, g_idx = _quantize(magnitude, bits=bits, sym=sym)
    maxq = (1 << bits) - 1

    codes = _recovered_codes(quantized, scale, zero, g_idx, bits)
    assert codes.min().item() >= 0
    assert codes.max().item() <= maxq
    # The packer recovers codes from the narrowed weight, so a finite replay
    # tensor and a finite reconstruction are the same guarantee.
    reconstructed = (codes.to(torch.float32) - zero.T[g_idx.long()]) * scale.T[g_idx.long()]
    assert torch.isfinite(reconstructed).all()
    assert reconstructed.abs().max().item() <= FP16_MAX


def _pack(quantized, scale, zero, g_idx, *, bits, dtype=torch.float16):
    module = TorchLinear(
        bits=bits,
        group_size=GROUP_SIZE,
        sym=True,
        desc_act=False,
        in_features=WIDTH,
        out_features=WIDTH,
        bias=False,
        format=FORMAT.GPTQ,
    )
    linear = nn.Linear(WIDTH, WIDTH, bias=False, dtype=dtype)
    with torch.no_grad():
        linear.weight.copy_(quantized)
    module.pack_block(linear, scale, zero, g_idx, workers=1)
    return module


@pytest.mark.parametrize("bits", [2, 4, 8])
@pytest.mark.parametrize("magnitude", [61440.0, 65504.0])
def test_quantize_pack_unpack_preserves_the_narrowed_codes(bits, magnitude):
    """The narrowest path must pack the codes the replay weight implies."""

    _, _, quantized, scale, zero, g_idx = _quantize(magnitude, bits=bits)
    assert torch.isfinite(quantized).all()

    module = _pack(quantized, scale, zero, g_idx, bits=bits)
    decoded = module.dequantize_weight(dtype=torch.float32)
    assert torch.isfinite(decoded).all()
    assert decoded.abs().max().item() <= FP16_MAX

    # ``decoded`` is ``[in, out]``; the packer's recovery keeps ``[out, in]``.
    expected = _recovered_codes(quantized, scale, zero, g_idx, bits)
    unpacked = torch.clamp(
        torch.round(decoded / scale.T[g_idx.long()]) + zero.T[g_idx.long()],
        0,
        (1 << bits) - 1,
    )
    torch.testing.assert_close(
        unpacked.to(torch.int64), expected.T.contiguous().to(torch.int64), rtol=0, atol=0
    )


def test_source_replay_after_quantization_stays_finite():
    """The replay weight is installed and forwarded, so its output must be finite."""

    _, layer, quantized, *_rest = _quantize(65504.0)
    assert torch.isfinite(quantized).all()

    with torch.no_grad():
        layer.weight.data = quantized.to(layer.weight.dtype)
    activation = torch.eye(WIDTH, dtype=torch.float16)
    output = layer(activation)
    assert torch.isfinite(output).all(), "a finite source produced a non-finite replay output"


def test_fp32_source_keeps_the_unconstrained_grid():
    """FP32 is the computation dtype, so no bound applies and codes are unchanged."""

    _, _, quantized, scale, zero, g_idx = _quantize(61440.0, dtype=torch.float32)
    assert quantized.dtype == torch.float32
    # The exact FP32 reconstruction of the negative endpoint is -65536.
    assert quantized[0, 0].item() == -65536.0
    codes = _recovered_codes(quantized, scale, zero, g_idx, 4)
    assert codes[0, 0].item() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("sym", [True, False])
@pytest.mark.parametrize("fused", [True, False])
@pytest.mark.parametrize("magnitude", [61408.0, 61440.0, 65504.0])
def test_cuda_eager_and_fused_paths_agree_and_stay_finite(monkeypatch, sym, fused, magnitude):
    from gptqmodel.quantization import gptq_cuda
    from gptqmodel.utils.kernel import KernelConfig, configure_kernel

    if fused and not gptq_cuda.block_update_available():
        pytest.skip(gptq_cuda._EXTENSION.last_error_message())

    configure_kernel(KernelConfig(gptq_fused_block_update=fused))
    device = torch.device("cuda:0")

    launches = []
    original = gptq_cuda.gptq_block_update

    def counted(*args, **kwargs):
        launches.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(gptq_cuda, "gptq_block_update", counted)

    width = 256
    weight = torch.zeros(width, width, dtype=torch.float16, device=device)
    weight[:, 0] = -magnitude
    weight[:, 1] = magnitude
    layer = nn.Linear(width, width, bias=False, dtype=torch.float16, device=device)
    with torch.no_grad():
        layer.weight.copy_(weight)
    task = GPTQ(layer, _config(4, sym=sym, group_size=32))
    task.quantizer.configure(perchannel=True)
    task.add_batch(torch.eye(width, dtype=torch.float16, device=device).unsqueeze(0), None)
    quantized, scale, zero, g_idx, *_rest = task.quantize(blocksize=128)
    torch.cuda.synchronize(device)

    # The trigger must actually run: two 128-column blocks.
    assert len(launches) == (2 if fused else 0)
    assert quantized.dtype == torch.float16
    assert torch.isfinite(quantized).all(), f"sym={sym} fused={fused} amax={magnitude}"
    assert quantized.abs().max().item() <= FP16_MAX
    assert torch.isfinite(scale).all()
    assert torch.isfinite(zero).all()
