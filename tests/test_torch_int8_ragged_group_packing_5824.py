# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""CPU INT8 groups must store only their own kernel-aligned width (#5824).

``transform_cpu`` used to pad every group to the widest group's aligned width
and allocate ``[groups, out_features, width]``. A checkpoint whose groups are
unequal therefore inflated every short group to the longest one, and an empty
group still owned a full panel. The runtime layout now keeps the dense tensor
only when one width covers every group; otherwise it stores one flat buffer
whose per-group element offsets point at panels of the group's own aligned
width, and empty groups own no panel and launch no kernel.
"""

from __future__ import annotations

import torch

from gptqmodel.nn_modules.qlinear.torch_int8 import (
    INT8_WEIGHT_BUFFER_NAME,
    TorchInt8Linear,
    _align_int8_group_width,
)
from gptqmodel.quantization import FORMAT


IN_FEATURES = 256
OUT_FEATURES = 64
GROUP_SIZE = 32
WEIGHT_CODE = 129
ZERO_CODE = 128
SCALE = 0.25


def _packed_word(code: int) -> int:
    """Pack one 8-bit value into every lane of a GPTQ V2 int32 word."""

    word = sum(code << (8 * lane) for lane in range(4))
    return word - (1 << 32) if word >= (1 << 31) else word


def _build_layer(
    g_idx: torch.Tensor, in_features: int = IN_FEATURES
) -> tuple[TorchInt8Linear, torch.Tensor]:
    layer = TorchInt8Linear(
        bits=8,
        group_size=GROUP_SIZE,
        sym=True,
        desc_act=True,
        in_features=in_features,
        out_features=OUT_FEATURES,
        format=FORMAT.GPTQ_V2,
    ).eval()
    layer.qweight.fill_(_packed_word(WEIGHT_CODE))
    layer.qzeros.fill_(_packed_word(ZERO_CODE))
    layer.scales.fill_(SCALE)
    layer.g_idx.copy_(g_idx.to(torch.int32))
    layer.qzero_format(2)
    words = layer.qweight.clone()
    layer.post_init()
    return layer, words


def _groups_from_lengths(lengths: tuple[int, ...]) -> torch.Tensor:
    return torch.repeat_interleave(torch.arange(len(lengths)), torch.tensor(lengths))


def _oracle_weight(in_features: int = IN_FEATURES) -> torch.Tensor:
    """Independent FP64 decode of the checkpoint's ``[K, N]`` matrix."""

    return (
        torch.full((in_features, OUT_FEATURES), WEIGHT_CODE - ZERO_CODE, dtype=torch.float64)
        * SCALE
    )


def _assert_forward(layer: TorchInt8Linear, in_features: int = IN_FEATURES) -> None:
    activation = torch.ones((2, in_features), dtype=torch.float32)
    expected = activation.double() @ _oracle_weight(in_features)
    with torch.inference_mode():
        actual = layer(activation)
    torch.testing.assert_close(actual.double(), expected, rtol=1e-6, atol=1e-6)


def _assert_round_trip(
    layer: TorchInt8Linear, words: torch.Tensor, in_features: int = IN_FEATURES
) -> None:
    """The flat layout must rebuild the checkpoint words and decode exactly."""

    checkpoint = {name: value.clone() for name, value in layer.state_dict().items()}
    # The checkpoint words were released after ``post_init``; the runtime
    # layout must invert them byte-exactly, not merely reproduce its own decode.
    torch.testing.assert_close(checkpoint["qweight"], words, rtol=0, atol=0)
    torch.testing.assert_close(
        layer.dequantize_weight().double(),
        _oracle_weight(in_features),
        rtol=0,
        atol=0,
    )
    restored = TorchInt8Linear(
        bits=8,
        group_size=GROUP_SIZE,
        sym=True,
        desc_act=True,
        in_features=in_features,
        out_features=OUT_FEATURES,
        format=FORMAT.GPTQ_V2,
    ).eval()
    restored.load_state_dict(checkpoint, strict=True)
    restored.post_init()
    for name, tensor in checkpoint.items():
        torch.testing.assert_close(restored.state_dict()[name], tensor, rtol=0, atol=0)
    _assert_forward(restored, in_features)


def test_uniform_groups_keep_the_dense_layout():
    """A single width is still one dense tensor; this is the fast path."""

    layer, words = _build_layer(torch.arange(IN_FEATURES) // GROUP_SIZE)
    weight = getattr(layer, INT8_WEIGHT_BUFFER_NAME)
    assert layer._int8_group_lengths == (32,) * (IN_FEATURES // GROUP_SIZE)
    assert weight.dim() == 3
    assert tuple(weight.shape) == (IN_FEATURES // GROUP_SIZE, OUT_FEATURES, 32)
    assert weight.nbytes == 16_384
    assert getattr(layer, "int8_group_offsets", None) is None
    assert getattr(layer.int8_module, "_group_plan", None) is None
    _assert_forward(layer)
    _assert_round_trip(layer, words)


def test_all_in_one_group_does_not_pad_empty_groups():
    """``g_idx`` all zero: one 256-channel group, seven empty groups."""

    layer, words = _build_layer(torch.zeros(IN_FEATURES, dtype=torch.int32))
    weight = getattr(layer, INT8_WEIGHT_BUFFER_NAME)
    assert layer._int8_group_lengths == (IN_FEATURES,) + (0,) * (IN_FEATURES // GROUP_SIZE - 1)
    # Only the single nonempty group is resident; the seven empty groups are
    # not padded to its width (the dense layout would store 8 x 64 x 256 B).
    assert weight.dim() == 1
    assert weight.nbytes == OUT_FEATURES * IN_FEATURES == 16_384
    offsets = getattr(layer, "int8_group_offsets", None)
    assert offsets.tolist() == [0] + [16_384] * (IN_FEATURES // GROUP_SIZE)
    assert [entry[0] for entry in layer.int8_module._group_plan] == [0]
    _assert_forward(layer)
    _assert_round_trip(layer, words)


def test_empty_groups_do_not_launch_the_kernel(monkeypatch):
    layer, _ = _build_layer(torch.zeros(IN_FEATURES, dtype=torch.int32))
    calls = []
    original = torch.ops.aten._weight_int8pack_mm

    def spy(activations, weight, scale):
        calls.append(int(weight.shape[0]))
        return original(activations, weight, scale)

    monkeypatch.setattr(torch.ops.aten, "_weight_int8pack_mm", spy)
    with torch.inference_mode():
        layer(torch.ones((1, IN_FEATURES)))
    assert calls == [OUT_FEATURES]


def test_ragged_groups_store_their_own_aligned_width():
    """31/33/32... groups: 16,896 B instead of padding every group to 40."""

    lengths = (31, 33, 32, 32, 32, 32, 32, 32)
    assert sum(lengths) == IN_FEATURES
    widths = tuple(_align_int8_group_width(length) for length in lengths)
    assert widths == (32, 40, 32, 32, 32, 32, 32, 32)
    layer, words = _build_layer(_groups_from_lengths(lengths))
    weight = getattr(layer, INT8_WEIGHT_BUFFER_NAME)
    assert layer._int8_group_lengths == lengths
    assert weight.dim() == 1
    assert weight.nbytes == OUT_FEATURES * sum(widths) == 16_896
    assert weight.nbytes < OUT_FEATURES * max(widths) * len(lengths) == 20_480
    offsets = getattr(layer, "int8_group_offsets", None).tolist()
    expected_offsets = [0]
    for width in widths:
        expected_offsets.append(expected_offsets[-1] + OUT_FEATURES * width)
    assert offsets == expected_offsets
    assert [entry[2] for entry in layer.int8_module._group_plan] == list(lengths)
    _assert_forward(layer)
    _assert_round_trip(layer, words)


def test_ragged_forward_matches_the_checkpoint_decode():
    """Unequal widths must still reproduce every checkpoint weight exactly."""

    lengths = (30, 34, 32, 32, 32, 32, 32, 32)
    layer, words = _build_layer(_groups_from_lengths(lengths))
    weight = getattr(layer, INT8_WEIGHT_BUFFER_NAME)
    assert weight.dim() == 1
    assert weight.nbytes == OUT_FEATURES * sum(
        _align_int8_group_width(length) for length in lengths
    )
    assert layer.dequantize_weight().dtype == torch.float32
    torch.testing.assert_close(
        layer.dequantize_weight().double(),
        _oracle_weight(),
        rtol=0,
        atol=0,
    )
    activation = torch.arange(3 * IN_FEATURES).reshape(3, -1).remainder(5).sub(2).float() / 8
    expected = activation.double() @ _oracle_weight()
    with torch.inference_mode():
        actual = layer(activation)
    torch.testing.assert_close(actual.double(), expected, rtol=1e-6, atol=1e-6)
    _assert_round_trip(layer, words)


def test_flat_layout_compiles_and_matches_eager():
    """A ragged layout must stay traceable; the class declares INDUCTOR."""

    lengths = (31, 33, 32, 32, 32, 32, 32, 32)
    layer, _ = _build_layer(_groups_from_lengths(lengths))
    activation = torch.ones((3, IN_FEATURES), dtype=torch.float32)
    with torch.inference_mode():
        eager = layer(activation)
    compiled = torch.compile(layer, backend="inductor", fullgraph=True)
    with torch.inference_mode():
        traced = compiled(activation)
    torch.testing.assert_close(traced, eager, rtol=0, atol=0)


def test_fusion_declines_a_ragged_layout():
    """Fusion concatenates dense codes on the output axis; ragged stays direct."""

    from gptqmodel.models.fusion_kernels import kernel_spec

    layer, _ = _build_layer(_groups_from_lengths((31, 33, 32, 32, 32, 32, 32, 32)))
    error = kernel_spec(layer).shape_error(layer)
    assert "per-group widths" in error


def test_short_last_group_with_one_common_width_stays_dense():
    """A 26-channel last group aligns to 32, so the dense layout still fits."""

    in_features = 250
    lengths = (32,) * 7 + (26,)
    layer, words = _build_layer(_groups_from_lengths(lengths), in_features=in_features)
    weight = getattr(layer, INT8_WEIGHT_BUFFER_NAME)
    assert layer._int8_group_lengths == lengths
    assert weight.dim() == 3
    assert tuple(weight.shape) == (8, OUT_FEATURES, 32)
    assert weight.nbytes == 8 * OUT_FEATURES * 32
    assert getattr(layer, "int8_group_offsets", None) is None
    _assert_forward(layer, in_features)
    _assert_round_trip(layer, words, in_features)
