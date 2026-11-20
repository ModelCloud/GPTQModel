# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""GPTAQ/FOEM must pack BF16 asymmetric 8-bit codes from the exact reconstruction (#5819).

The packers recover codes from ``linear.weight`` with ``round(Q / scale) + zero``.
``GPTQProcessor`` installs the replay copy (``Q`` narrowed to the module dtype)
in that parameter. BF16's 8-bit mantissa cannot round-trip 8-bit asymmetric
codes, so GPTAQ and FOEM must keep GPTQ's exact FP32 reconstruction for packing
through ``GPTQ._narrow_reconstruction``. This is the issue fixture: a 32x32
projection whose input channel 1 selects code 207, while the BF16 replay copy
alone rounds to 206.
"""

# GPU=-1

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from gptqmodel.looper.gptq_processor import GPTQProcessor
from gptqmodel.looper.loop_processor import ForwardConfig
from gptqmodel.looper.named_module import NamedModule
from gptqmodel.looper.native_processor import NATIVE_INPUTS_STATE_KEY
from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.quantization.config import CalibrationConfig, FOEMConfig, GPTAQConfig, QuantizeConfig


N = 32
BITS = 8
WEIGHT_COLUMNS = (0.00128936767578125, 0.6484375, 0.80078125)


class _Progress:
    def title(self, *args):
        return self

    subtitle = title
    draw = title


def _variant(kind):
    if kind == "gptaq":
        return {"gptaq": GPTAQConfig(alpha=0.25)}, 0.0
    return {"foem": FOEMConfig(alpha=0.25, beta=0.25)}, 0.25


def _run(dtype, kind):
    """Quantize the issue fixture and return the packed layer and its inputs."""

    x = torch.eye(N, dtype=dtype)
    x[0, 1] = 0.5
    source = torch.zeros(N, N, dtype=dtype)
    source[:, :3] = torch.tensor(WEIGHT_COLUMNS, dtype=dtype)

    model = nn.Module()
    model.linear = nn.Linear(N, N, bias=False, dtype=dtype)
    with torch.no_grad():
        model.linear.weight.copy_(source)

    variant, beta = _variant(kind)
    qcfg = QuantizeConfig(
        bits=BITS,
        group_size=N,
        sym=False,
        desc_act=False,
        act_group_aware=False,
        damp_percent=0.01,
        offload_to_disk=False,
        pack_impl="block",
        **variant,
    )
    named = NamedModule(model.linear, name="linear", full_name="linear", layer_index=0)
    # GPTAQ and FOEM pair the quantized pass with the native one.
    named.state[NATIVE_INPUTS_STATE_KEY] = [x]
    named.module.target_device = torch.device("cpu")
    processor = GPTQProcessor(
        tokenizer=None,
        qcfg=qcfg,
        calibration=CalibrationConfig(),
        forward=ForwardConfig(required=False, replay_after_process=True, subset_early_stop=True),
        calculate_w_wq_diff=False,
    )
    processor.pb = _Progress()
    processor.preprocess(named)
    processor.pre_process_fwd_hook(named.name)(model.linear, (x,), None)
    processor.process(named)

    replay = named.weight.data.clone()
    scale = named.state["q_scales"].flatten()[0].item()
    processor.submodule_finalize(
        named,
        SimpleNamespace(
            model=model,
            quantize_config=qcfg,
            qlinear_kernel=TorchLinear,
            lm_head="lm_head",
            quant_region_timer=None,
            quantized=False,
        ),
    )
    return model.linear, replay, scale, source, x, beta


def _fp64_code(x, source, scale, beta):
    """Independent FP64 code for input channel 1 (only feature 0 feeds it error)."""

    h = x.double().T @ x.double()
    damp = 0.01 * h.diag().mean().item()
    w0, w1 = source[0, :2].tolist()
    residual = w0 - round(w0 / scale) * scale
    working = (w1 + (1 - beta) * residual * h[0, 1].item() / (h[1, 1].item() + damp)) / scale
    return round(working)


def _packed_channel_one(layer):
    """Decode input channel 1 for every output; 8-bit packs four channels per word."""

    return (layer.qweight[0].long() >> 8) & 0xFF


@pytest.mark.parametrize("kind", ["gptaq", "foem"])
def test_bf16_asymmetric_eight_bit_packs_the_exact_code(kind):
    layer, replay, scale, source, x, beta = _run(torch.bfloat16, kind)

    oracle = _fp64_code(x, source, scale, beta)
    assert oracle == 207

    # The trigger: the BF16 replay copy alone rounds to the neighbour, so the
    # packed code cannot be recovered from `linear.weight`.
    assert round((replay[0, 1].float() / scale).item()) == 206

    assert _packed_channel_one(layer).unique().tolist() == [oracle]


@pytest.mark.parametrize("kind", ["gptaq", "foem"])
def test_fp16_asymmetric_eight_bit_control(kind):
    layer, replay, scale, source, x, beta = _run(torch.float16, kind)

    oracle = _fp64_code(x, source, scale, beta)
    assert oracle == 207

    # FP16 holds every 8-bit asymmetric code, so the replay copy is exact.
    assert round((replay[0, 1].float() / scale).item()) == 207

    assert _packed_channel_one(layer).unique().tolist() == [oracle]
