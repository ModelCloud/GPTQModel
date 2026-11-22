# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Cold INT8 source decode must crop the padded input-channel rows (#4483).

The packed GPTQ layout pads the input-channel axis up to the pack factor, so
``in_features=17`` stores 20 decoded rows. ``transform_cpu`` cropped them to
``in_features`` when it built the runtime view, but
``_dequantize_gptq_source_weight`` did not. Once the runtime view was dropped
(any ``.to(...)``/``_apply``), ``dequantize_weight()`` decoded all 20 rows and
then subtracted per-channel zeros with 17 rows, raising a row mismatch. Explicit
dequantization/export therefore depended on call order.
"""

import pytest
import torch

from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.nn_modules.qlinear.torch_int8 import TorchInt8Linear
from gptqmodel.quantization import FORMAT


def _pack_words(words: torch.Tensor) -> torch.Tensor:
    return (
        (words[..., 0] << 0)
        | (words[..., 1] << 8)
        | (words[..., 2] << 16)
        | (words[..., 3] << 24)
    ).to(torch.int32).contiguous()


def _build(in_features: int, out_features: int, group_size: int, seed: int) -> tuple[TorchInt8Linear, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    pack_factor = 4
    padded = -(-in_features // pack_factor) * pack_factor
    num_groups = 1 if group_size == -1 else -(-in_features // group_size)

    # Padding rows are real packed data; only the first `in_features` are logical.
    codes = torch.randint(0, 256, (padded, out_features), generator=generator)
    zeros = torch.randint(0, 256, (num_groups, out_features), generator=generator)
    scales = (torch.rand(num_groups, out_features, generator=generator) * 0.01 + 1e-3).half()
    groups = (
        torch.zeros(in_features, dtype=torch.int32)
        if group_size == -1
        else (torch.arange(in_features) // group_size).int()
    )

    weight = (codes[:in_features] - zeros[groups].long()).double() * scales.double()[groups]

    module = TorchInt8Linear(
        bits=8,
        group_size=group_size,
        sym=False,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=False,
    )
    module.load_state_dict(
        {
            "qweight": _pack_words(codes.reshape(padded // pack_factor, pack_factor, out_features).permute(0, 2, 1)),
            "qzeros": _pack_words(zeros.reshape(num_groups, out_features // pack_factor, pack_factor)),
            "scales": scales,
            "g_idx": groups,
        },
        strict=True,
    )
    module.post_init()
    module.eval()
    return module, weight


@pytest.mark.parametrize("in_features", [17, 18, 19])
def test_cold_source_decode_matches_the_runtime_view(in_features):
    module, expected = _build(in_features, out_features=8, group_size=-1, seed=4483)

    # Warm the runtime view, then drop it the way any dtype/device apply does.
    warm = module.dequantize_weight().clone()
    assert module.int8_module is not None

    module.to(dtype=torch.float32)
    assert module.int8_module is None, "the runtime view must be invalidated for this regression"

    cold = module.dequantize_weight()

    assert tuple(cold.shape) == (in_features, 8)
    torch.testing.assert_close(cold, warm, rtol=0, atol=0)
    torch.testing.assert_close(cold.double(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("in_features", [17, 18, 19])
def test_cold_source_decode_keeps_the_forward_oracle(in_features):
    module, expected = _build(in_features, out_features=8, group_size=32, seed=4483 + in_features)
    activation = torch.randn(3, in_features)

    module.to(dtype=torch.float32)
    assert module.int8_module is None

    with torch.inference_mode():
        output = module(activation)

    torch.testing.assert_close(output.double(), activation.double() @ expected, rtol=2e-3, atol=2e-3)


def test_aligned_width_is_unchanged():
    """K divisible by the pack factor must keep decoding every row."""

    module, expected = _build(16, out_features=8, group_size=-1, seed=7)
    module.to(dtype=torch.float32)

    torch.testing.assert_close(module.dequantize_weight().double(), expected, rtol=0, atol=0)


# --- Output-axis padding (#5851) -------------------------------------------
#
# GPTQ packs four zero-point lanes per int32 word along the output axis, so
# ``out_features=5`` stores two words (8 lanes) and the lanes past the logical
# width are padding. Both ``_dequantize_gptq_source_weight`` and
# ``transform_cpu`` used to reshape every decoded lane to ``scales.shape``,
# which raised ``RuntimeError: shape '[1, 5]' is invalid for input of size 8``
# during ``post_init()`` for any layer whose output width was not word aligned.


def _new_layer(in_features: int, out_features: int, *, group_size: int = 32, sym: bool = True) -> TorchInt8Linear:
    return TorchInt8Linear(
        bits=8,
        group_size=group_size,
        sym=sym,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=False,
        format=FORMAT.GPTQ_V2,
    ).eval()


def _build_uniform(
    in_features: int,
    out_features: int,
    *,
    group_size: int = 32,
    code: int = 129,
    zero: int = 128,
    scale: float = 0.25,
    sym: bool = True,
    pad_lane: int = None,
) -> TorchInt8Linear:
    """Build a layer whose every logical code and zero point are identical.

    The padding lanes of the zero words are set to ``pad_lane`` (a different
    value by default in the odd-width tests) so a decode that keeps them
    cannot silently match the oracle.
    """

    pack_factor = 4
    padded_in = -(-in_features // pack_factor) * pack_factor
    padded_out = -(-out_features // pack_factor) * pack_factor
    num_groups = 1 if group_size == -1 else -(-in_features // group_size)

    codes = torch.full((padded_in, out_features), code, dtype=torch.int64)
    zeros = torch.full((num_groups, padded_out), zero if pad_lane is None else pad_lane, dtype=torch.int64)
    zeros[:, :out_features] = zero
    groups = (
        torch.zeros(in_features, dtype=torch.int32)
        if group_size == -1
        else (torch.arange(in_features) // group_size).int()
    )

    module = _new_layer(in_features, out_features, group_size=group_size, sym=sym)
    module.load_state_dict(
        {
            "qweight": _pack_words(
                codes.reshape(padded_in // pack_factor, pack_factor, out_features).permute(0, 2, 1)
            ),
            "qzeros": _pack_words(zeros.reshape(num_groups, padded_out // pack_factor, pack_factor)),
            "scales": torch.full((num_groups, out_features), scale, dtype=torch.float16),
            "g_idx": groups,
        },
        strict=True,
    )
    module.qzero_format(2)
    return module


def _expected_uniform(in_features: int, out_features: int, code: int, zero: int, scale: float) -> torch.Tensor:
    return torch.full((in_features, out_features), (code - zero) * scale)


@pytest.mark.parametrize("out_features", [5, 7, 65])
def test_partial_qzero_words_post_init_and_forward(out_features):
    module = _build_uniform(32, out_features, pad_lane=127)
    module.post_init()
    assert module.int8_module is not None

    with torch.inference_mode():
        output = module(torch.ones(2, 32))

    expected = torch.full((2, out_features), 32 * (129 - 128) * 0.25)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


@pytest.mark.parametrize("out_features", [5, 7, 65])
def test_partial_qzero_words_cold_source_decode(out_features):
    module = _build_uniform(32, out_features, pad_lane=127)
    module.post_init()

    # Drop the derived INT8 view the way a dtype/device apply does, so the
    # explicit decode reads the checkpoint words instead of the runtime cache.
    module.to(dtype=torch.float32)
    assert module.int8_module is None

    decoded = module.dequantize_weight()
    torch.testing.assert_close(decoded, _expected_uniform(32, out_features, 129, 128, 0.25), rtol=0, atol=0)


@pytest.mark.parametrize("out_features", [4, 8])
def test_word_aligned_qzero_words_are_unchanged(out_features):
    module = _build_uniform(32, out_features)
    module.post_init()

    with torch.inference_mode():
        output = module(torch.ones(2, 32))

    torch.testing.assert_close(output, torch.full((2, out_features), 8.0), rtol=0, atol=0)


def test_partial_qzero_words_asymmetric_offset():
    # A nonzero per-(group, output) offset is built from the decoded zeros, so
    # the crop must also hold on the asymmetric tile path.
    module = _build_uniform(32, 5, zero=130, sym=False, pad_lane=1)
    module.post_init()

    with torch.inference_mode():
        output = module(torch.ones(2, 32))

    torch.testing.assert_close(output, torch.full((2, 5), 32 * (129 - 130) * 0.25), rtol=0, atol=0)


@pytest.mark.parametrize("out_features", [5, 7, 65])
def test_partial_qzero_words_state_dict_round_trip(out_features):
    module = _build_uniform(32, out_features, pad_lane=127)
    module.post_init()

    fresh = _new_layer(32, out_features)
    fresh.load_state_dict(module.state_dict(), strict=True)
    fresh.post_init()

    with torch.inference_mode():
        output = fresh(torch.ones(2, 32))

    torch.testing.assert_close(output, torch.full((2, out_features), 8.0), rtol=0, atol=0)


def test_partial_qzero_words_portable_consumer_matches():
    module = _build_uniform(32, 5, pad_lane=127)
    portable = TorchLinear(
        bits=8,
        group_size=32,
        sym=True,
        desc_act=False,
        in_features=32,
        out_features=5,
        bias=False,
        format=FORMAT.GPTQ_V2,
    ).eval()
    portable.load_state_dict(module.state_dict(), strict=True)

    with torch.inference_mode():
        output = portable(torch.ones(2, 32))

    torch.testing.assert_close(output, torch.full((2, 5), 8.0), rtol=0, atol=0)
