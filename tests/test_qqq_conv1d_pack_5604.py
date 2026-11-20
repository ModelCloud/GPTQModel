# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1

"""Regression tests for #5604: QQQ must pack ``transformers.Conv1D`` projections.

GPT-2-style projections store their weight as ``[in_features, out_features]``,
while QQQ's tiling and checkpoint layout address the logical
``[out_features, in_features]`` matrix. A missing transpose silently packed a
square projection transposed and raised a size mismatch on a rectangular one.
The grouped INT8 channel scale (``scale_extra``) had the same axis bug in both
quantization epilogues: it was fitted on the source ``[in, out]`` orientation,
so a rectangular layer failed at pack time and a square one used the input
axis as the output normalisation.
"""

import gc

import pytest
import torch
import transformers
from tokenizers import Tokenizer, models, pre_tokenizers
from torch import nn
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from gptqmodel.nn_modules.qlinear.qqq import QQQTorchLinear
from gptqmodel.quantization.config import QQQConfig
from gptqmodel.quantization.qqq import QQQ


def _conv1d(in_features, out_features, group_size):
    """A Conv1D whose integer weights are exactly representable at unit scale."""

    generator = torch.Generator().manual_seed(5604)
    conv = transformers.Conv1D(out_features, in_features).half()
    if group_size == -1:
        # Per-channel codes are signed INT4, so [-7, 7] survives unit scales.
        low, high = -7, 7
    else:
        # Grouped codes are biased by 8; [-8, 7] maps onto [0, 15].
        low, high = -8, 7
    with torch.no_grad():
        conv.weight.copy_(
            torch.randint(low, high + 1, (in_features, out_features), generator=generator).half()
        )
        conv.bias.copy_(
            torch.randint(-4, 5, (out_features,), generator=generator).half()
        )
    return conv


def _pack(source, in_features, out_features, group_size, scales, s_extra):
    module = QQQTorchLinear(
        bits=4,
        group_size=group_size,
        desc_act=False,
        sym=True,
        in_features=in_features,
        out_features=out_features,
        bias=True,
    )
    module.pack(source, scales, s_extra=s_extra)
    return module


@pytest.mark.cpu
@pytest.mark.parametrize(
    "in_features,out_features,group_size",
    [
        (128, 128, -1),
        (128, 256, -1),
        (256, 128, 128),
        (256, 256, 128),
    ],
)
def test_qqq_pack_conv1d_matches_linear_transpose(
    in_features, out_features, group_size
):
    """Conv1D must pack exactly like the equivalent ``[out, in]`` Linear.

    The square case is the dangerous one: a missing transpose does not raise,
    it silently packs the transposed matrix.
    """

    conv = _conv1d(in_features, out_features, group_size)
    linear = nn.Linear(in_features, out_features, bias=True)
    with torch.no_grad():
        linear.weight.copy_(conv.weight.t())
        linear.bias.copy_(conv.bias)

    if group_size == -1:
        groups = 1
        s_extra = None
    else:
        groups = in_features // group_size
        s_extra = torch.ones(out_features, dtype=torch.float32)
    scales = torch.ones(out_features, groups, dtype=torch.float32)

    conv_module = _pack(
        conv, in_features, out_features, group_size, scales, s_extra
    )
    linear_module = _pack(
        linear, in_features, out_features, group_size, scales, s_extra
    )

    assert torch.equal(conv_module.B, linear_module.B)
    assert torch.equal(conv_module.s_channel, linear_module.s_channel)
    assert torch.equal(conv_module.s_group, linear_module.s_group)
    assert torch.equal(conv_module.bias, linear_module.bias)

    x = torch.randn(3, in_features, dtype=torch.float16)
    packed = conv_module(x)
    reference = conv(x)
    # The unit scales make the packed codes exact, so only the W4A8 activation
    # quantization separates the packed forward from the dense one. A
    # transposed pack changes the whole matrix and cannot stay under this.
    relative_rmse = (
        (packed.float() - reference.float()).square().mean().sqrt()
        / reference.float().square().mean().sqrt()
    ).item()
    assert relative_rmse < 0.05, relative_rmse
    torch.testing.assert_close(packed, linear_module(x), rtol=0, atol=0)


@pytest.mark.cpu
@pytest.mark.parametrize(
    "in_features,out_features,group_size,calibrate",
    [
        (128, 128, -1, True),
        (128, 256, -1, True),
        (256, 128, 128, True),
        (256, 128, 128, False),
        (256, 256, 128, True),
        (256, 256, 128, False),
    ],
    ids=[
        "square-per-channel",
        "rect-per-channel",
        "rect-grouped-calibrated",
        "rect-grouped-fallback",
        "square-grouped-calibrated",
        "square-grouped-fallback",
    ],
)
def test_qqq_quantize_conv1d_fits_scale_extra_on_output_axis(
    in_features, out_features, group_size, calibrate
):
    """Both quantization epilogues must fit ``scale_extra`` per output channel."""

    torch.manual_seed(5604)
    conv = transformers.Conv1D(out_features, in_features).half()
    with torch.no_grad():
        conv.weight.copy_(
            torch.randint(-40, 41, (in_features, out_features)).half() / 8
        )
        conv.bias.copy_(torch.randint(-8, 9, (out_features,)).half() / 8)

    qcfg = QQQConfig(
        bits=4,
        group_size=group_size,
        desc_act=False,
        sym=True,
        offload_to_disk=False,
    )
    qqq = QQQ(conv, qcfg)
    qqq.quantizer.configure(4, perchannel=True, sym=True, groupsize=group_size)
    if calibrate:
        qqq.add_batch(torch.randn(8, in_features, dtype=torch.float16), None)

    wq, scales, _, _, _, _, _, s_extra, nsamples = qqq.quantize(blocksize=128)
    assert nsamples == (8 if calibrate else 0)

    if group_size == -1:
        assert s_extra is None
    else:
        assert s_extra is not None
        # ``Q`` comes back in the source [in, out] orientation; the INT8
        # channel scale belongs to the output axis.
        # ``find_params`` clamps the row extrema with a float32 zero before
        # dividing, so mirror that promotion.
        expected = torch.maximum(
            wq.abs().amax(dim=0), torch.zeros(out_features)
        ) / 127
        assert s_extra.shape == (out_features, 1)
        torch.testing.assert_close(s_extra.flatten(), expected, rtol=1e-4, atol=1e-4)

    module = _pack(conv, in_features, out_features, group_size, scales, s_extra)
    x = torch.randn(4, in_features, dtype=torch.float16)
    packed = module(x).float()
    reference = conv(x).float()
    relative_rmse = (
        (packed - reference).square().mean().sqrt()
        / reference.square().mean().sqrt()
    ).item()
    assert relative_rmse < 0.15, relative_rmse


def _tiny_gpt2(path):
    vocab_size = 256
    backend = Tokenizer(
        models.WordLevel({f"tok{i}": i for i in range(vocab_size)}, unk_token="tok0")
    )
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="tok0", pad_token="tok0"
    )
    torch.manual_seed(5604)
    config = GPT2Config(
        n_embd=128,
        n_head=4,
        n_layer=2,
        n_positions=64,
        vocab_size=vocab_size,
    )
    model = GPT2LMHeadModel(config).to(torch.float16).eval()
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    return path, model


@pytest.mark.cpu
@pytest.mark.integration
def test_qqq_gpt2_conv1d_quantize_save_reload(tmp_path):
    """A real GPT-2 QQQ round trip must replace and reload every Conv1D projection."""

    from gptqmodel import BACKEND, CalibrationConfig, GPTQModel

    source, reference_model = _tiny_gpt2(tmp_path / "dense")
    generator = torch.Generator().manual_seed(5604)
    input_ids = torch.randint(3, 256, (1, 16), generator=generator)
    with torch.inference_mode():
        reference = reference_model(input_ids=input_ids, use_cache=False).logits.float()

    model = GPTQModel.load(
        str(source),
        QQQConfig(
            bits=4,
            group_size=128,
            desc_act=False,
            offload_to_disk=False,
            device="cpu",
        ),
        dtype=torch.float16,
        backend=BACKEND.QQQ_TORCH,
    )
    calibration = [
        {
            "input_ids": input_ids[0].tolist(),
            "attention_mask": [1] * input_ids.shape[1],
        }
        for _ in range(4)
    ]
    model.quantize(CalibrationConfig(dataset=calibration, batch_size=1))

    packed = [module for module in model.model.modules() if isinstance(module, QQQTorchLinear)]
    # GPT-2 has four Conv1D projections per block: c_attn and c_proj in
    # attention, then c_fc and c_proj in the MLP.
    assert len(packed) == 4 * 2
    for module in packed:
        assert module.B.shape == (module.in_features // 16, module.out_features * 2)

    with torch.inference_mode():
        actual = model.model(input_ids=input_ids, use_cache=False).logits.float()
    relative_rmse = (
        (actual - reference).square().mean().sqrt()
        / reference.square().mean().sqrt()
    ).item()
    assert relative_rmse < 0.15, relative_rmse

    destination = tmp_path / "quantized"
    model.save(str(destination))
    del model, packed
    gc.collect()

    restored = GPTQModel.load(
        str(destination), device="cpu", dtype=torch.float16, backend=BACKEND.QQQ_TORCH
    )
    with torch.inference_mode():
        reloaded = restored.model(input_ids=input_ids, use_cache=False).logits.float()
    torch.testing.assert_close(reloaded, actual, rtol=0, atol=0)
    del restored
    gc.collect()
