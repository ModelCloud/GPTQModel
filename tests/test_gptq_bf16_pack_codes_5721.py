# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""GPTQ packers must store the codes the quantizer chose (#5721).

The three packers never receive the quantizer's integer codes: they recover
them from ``linear.weight`` with ``round(Q / scale) + zero``. ``Q`` is
``(q - zero) * scale`` narrowed to the module dtype for the replay forward.
BF16's 8-bit mantissa cannot hold that product once ``|q - zero|`` passes 128,
so an 8-bit asymmetric code can round to its neighbour and the checkpoint
stores a code GPTQ never chose. This module pins the packed words to the
codes recovered from the exact FP32 reconstruction: the FP32 twin run below
narrows to itself, so its weight *is* that reconstruction.
"""

# GPU=-1

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from gptqmodel.looper.gptq_processor import GPTQProcessor
from gptqmodel.looper.loop_processor import ForwardConfig
from gptqmodel.looper.named_module import NamedModule
from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.quantization import FORMAT
from gptqmodel.quantization.config import CalibrationConfig, QuantizeConfig


IN_FEATURES = 256
OUT_FEATURES = 64
GROUP_SIZE = 128
BITS = 8
BATCH_ROWS = 16
BATCHES = 4


class _TinyLinearModel(nn.Module):
    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.linear = nn.Linear(IN_FEATURES, OUT_FEATURES, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(weight)


class _DummyProgressBar:
    def title(self, _):
        return self

    def subtitle(self, _):
        return self

    def draw(self, *args, **kwargs):
        return None


def _one_sided_weight() -> torch.Tensor:
    """One-sided groups: the zero point lands at 0 and codes span 0..255."""

    generator = torch.Generator().manual_seed(0)
    return (torch.randn(OUT_FEATURES, IN_FEATURES, generator=generator) * 0.02).abs() + 0.001


def _batches() -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(1)
    return [
        torch.randn(BATCH_ROWS, IN_FEATURES, generator=generator).to(torch.bfloat16)
        for _ in range(BATCHES)
    ]


# Both the FP32 twin and the narrow run quantize the same BF16 weights from the
# same BF16 activations, so they produce the same GPTQ codes; only the final
# narrowing differs.
_WEIGHT = _one_sided_weight().to(torch.bfloat16)
_BATCHES = _batches()


def _quantize(dtype: torch.dtype):
    model = _TinyLinearModel(_WEIGHT.to(dtype)).to(dtype)
    named = NamedModule(model.linear, name="linear", full_name="linear", layer_index=0)
    named.target_device = torch.device("cpu")
    named.module.target_device = torch.device("cpu")

    qcfg = QuantizeConfig(
        bits=BITS,
        group_size=GROUP_SIZE,
        desc_act=False,
        sym=False,
        format=FORMAT.GPTQ,
        pack_impl="block",
        offload_to_disk=False,
    )
    processor = GPTQProcessor(
        tokenizer=None,
        qcfg=qcfg,
        calibration=CalibrationConfig(),
        forward=ForwardConfig(
            required=False,
            replay_after_process=True,
            subset_early_stop=True,
        ),
        calculate_w_wq_diff=False,
    )
    processor.pb = _DummyProgressBar()
    processor.preprocess(named)

    hook = processor.pre_process_fwd_hook(named.name)
    captured_out = torch.empty(BATCH_ROWS, OUT_FEATURES, dtype=torch.bfloat16)
    for batch in _BATCHES:
        hook(named.module, (batch,), captured_out)
    processor.process(named)
    return model, named, processor


def _pack(named: NamedModule, processor: GPTQProcessor, model: nn.Module) -> None:
    """Finalize exactly like the looper: pack, then install the quantized layer."""

    quant_model = SimpleNamespace(
        model=model,
        quantize_config=processor.qcfg,
        qlinear_kernel=TorchLinear,
        lm_head="lm_head",
        quant_region_timer=None,
        quantized=False,
    )
    processor.submodule_finalize(named, quant_model)


def _codes(weight: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor, g_idx: torch.Tensor):
    """The packer's code formula, computed in FP32: ``round(Q / s) + z``."""

    maxq = (1 << BITS) - 1
    transposed = weight.detach().to(torch.float32).T
    scale = scales.to(torch.float32).T[g_idx.long()]
    zero = zeros.to(torch.float32).T[g_idx.long()]
    return torch.clamp(torch.round(transposed / scale) + zero, 0, maxq).T


def _unpacked_codes(layer: TorchLinear) -> torch.Tensor:
    """Decode the packed words back to ``[in, out]`` integer codes."""

    pack_factor = 32 // BITS
    maxq = (1 << BITS) - 1
    words = layer.qweight.to(torch.int64) & 0xFFFFFFFF
    return torch.stack(
        [(words >> (BITS * index)) & maxq for index in range(pack_factor)], dim=1
    ).reshape(IN_FEATURES, OUT_FEATURES)


def _exact_reconstruction() -> torch.Tensor:
    """FP32 twin: narrowing is a no-op, so the returned weight is exact."""

    _, named, _ = _quantize(torch.float32)
    return named.weight.data.clone()


@pytest.fixture(scope="module")
def exact_reconstruction() -> torch.Tensor:
    return _exact_reconstruction()


def _pack_and_compare(dtype: torch.dtype, exact: torch.Tensor):
    model, named, processor = _quantize(dtype)
    replay_copy = named.weight.data.clone()
    assert replay_copy.dtype == dtype

    scales = named.state["q_scales"].clone()
    zeros = named.state["q_zeros"].clone()
    g_idx = named.state["q_g_idx"].clone()
    _pack(named, processor, model)

    layer = model.linear
    assert isinstance(layer, TorchLinear)
    packed = _unpacked_codes(layer)
    # ``_codes`` keeps the packer's ``[out, in]`` layout; the packed words are
    # decoded as ``[in, out]``, so compare against the transpose.
    expected = _codes(exact, scales, zeros, g_idx)
    return packed, expected, replay_copy, scales, zeros, g_idx


def test_bf16_eight_bit_asymmetric_packs_the_quantizer_codes(exact_reconstruction):
    packed, expected, replay_copy, scales, zeros, g_idx = _pack_and_compare(
        torch.bfloat16, exact_reconstruction
    )

    # The old path packed from the BF16 replay copy; those codes are not the
    # ones GPTQ chose, so the fixture still covers the regression.
    narrowed = _codes(replay_copy, scales, zeros, g_idx)
    assert not torch.equal(narrowed, expected)

    torch.testing.assert_close(
        packed.to(torch.int64), expected.T.contiguous().to(torch.int64), atol=0, rtol=0
    )


def test_fp16_eight_bit_asymmetric_control(exact_reconstruction):
    packed, expected, replay_copy, scales, zeros, g_idx = _pack_and_compare(
        torch.float16, exact_reconstruction
    )

    # FP16 has enough mantissa bits for every 8-bit code to round-trip, so the
    # replay copy is exact and the packer needs no FP32 source.
    torch.testing.assert_close(
        _codes(replay_copy, scales, zeros, g_idx).to(torch.int64),
        expected.to(torch.int64),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        packed.to(torch.int64), expected.T.contiguous().to(torch.int64), atol=0, rtol=0
    )


def test_narrowing_predicate():
    # Imported here so the behavioural tests still collect on a tree that
    # predates the helper; only this assertion needs the new predicate.
    from gptqmodel.quantization.quantizer import narrowing_loses_codes

    assert narrowing_loses_codes(torch.bfloat16, 8, sym=False) is True
    assert narrowing_loses_codes(torch.bfloat16, 8, sym=True) is False
    assert narrowing_loses_codes(torch.bfloat16, 4, sym=False) is False
    assert narrowing_loses_codes(torch.bfloat16, 3, sym=False) is False
    assert narrowing_loses_codes(torch.bfloat16, 2, sym=False) is False
    assert narrowing_loses_codes(torch.float16, 8, sym=False) is False
    assert narrowing_loses_codes(torch.float32, 8, sym=False) is False
