# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Issue #4304: ``load_state_dict(..., assign=True)`` must convert v1 qzeros.

A v1 checkpoint stores ``zero_point - 1`` while a v2-runtime kernel (the CPU
Torch kernel, Marlin, ExLlamaV2, ...) decodes logical words, so the load hook
converts the copied words. ``assign=True`` rebinds ``self.qzeros`` to the
incoming checkpoint tensor, which made the load hook mistake a genuine
checkpoint load for an ``accelerate`` refill of the module's own buffer and
skip the conversion (silent corruption). ``torch.distributed.checkpoint``
``set_model_state_dict`` uses ``assign=True`` for meta-device models, so the
same path is hit there.
"""

import pytest
import torch

from gptqmodel import BACKEND
from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.quantization import FORMAT
from gptqmodel.utils.checkpoint_state import checkpoint_serialization
from gptqmodel.utils.packing import hf_pack_layer


IN_FEATURES = 32
OUT_FEATURES = 8
GROUP = 32
BITS = 4

# 4-bit packing stores eight zero points per int32 word. The v1 checkpoint
# convention stores `zero_point - 1`, so `0x77777777` decodes to zero points of
# 8 and the logical (v2) words are `0x88888888`.
V1_WORD = 0x77777777
V2_WORD = 0x88888888


def _module(fmt: FORMAT = FORMAT.GPTQ_V2):
    return TorchLinear(
        bits=BITS,
        group_size=GROUP,
        sym=True,
        desc_act=False,
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        pack_dtype=torch.int32,
        backend=BACKEND.GPTQ_TORCH,
        bias=False,
        register_buffers=True,
        format=fmt,
    )


def _v1_qzeros():
    """The raw v1 qzeros words a v1 checkpoint carries (no layout marker)."""

    return torch.full(
        (IN_FEATURES // GROUP, OUT_FEATURES * BITS // 32),
        V1_WORD,
        dtype=torch.int32,
    )


def _receiver():
    """A v2-runtime kernel freshly loaded from a v1 checkpoint on disk."""

    module = _module()
    module._checkpoint_format_on_disk = FORMAT.GPTQ
    return module


def test_assign_true_converts_v1_qzeros_like_assign_false():
    reference = _receiver()
    reference.load_state_dict({"qzeros": _v1_qzeros()}, strict=False)
    assert reference.qzero_format() == 2
    assert torch.all(reference.qzeros == V2_WORD)

    assigned = _receiver()
    assigned.load_state_dict({"qzeros": _v1_qzeros()}, strict=False, assign=True)
    assert assigned.qzero_format() == 2
    assert torch.all(assigned.qzeros == V2_WORD), (
        "assign=True skipped the v1->v2 conversion: qzeros still hold v1 words"
    )
    assert torch.equal(assigned.qzeros, reference.qzeros)


def test_assign_true_does_not_double_convert_refilled_live_buffer():
    """Issue #4166: a refill of the module's own words must stay untouched."""

    module = _receiver()
    module.load_state_dict({"qzeros": _v1_qzeros()}, strict=False)
    assert module.qzero_format() == 2
    reference = module.qzeros.detach().clone()

    # accelerate hands back the module's own live buffer (or a view sharing
    # its storage) with no layout marker. The load must not shift it again.
    module.load_state_dict({"qzeros": module.qzeros}, strict=False)
    assert torch.equal(module.qzeros, reference), "assign=False refilled load shifted zeros"
    module.load_state_dict({"qzeros": module.qzeros}, strict=False, assign=True)
    assert torch.equal(module.qzeros, reference), "assign=True refilled load shifted zeros"


def test_assign_true_is_idempotent_with_marker_carrying_refill():
    """A marker-carrying refill keeps working under both assign modes."""

    module = _receiver()
    module.load_state_dict({"qzeros": _v1_qzeros()}, strict=False)
    reference = module.qzeros.detach().clone()

    # accelerate/DCP rebuild a full payload from the module's own state dict,
    # including the persistent layout marker (here: live runtime layout = 2).
    merged = {
        name: value.detach().clone()
        for name, value in module.state_dict().items()
        if name in ("qzeros", "qzeros_format")
    }
    merged["qzeros"] = reference.clone()
    merged["qzeros_format"] = torch.tensor(2, dtype=torch.int64)
    for assign in (False, True):
        target = _receiver()
        target.load_state_dict({"qzeros": reference.clone()}, strict=False)
        target.load_state_dict(merged, strict=False, assign=assign)
        assert torch.equal(target.qzeros, reference), (
            f"assign={assign}: marker-carrying refill shifted the live zeros"
        )


# ---------------------------------------------------------------------------
# Issue #5805: an assign load must not rewrite the caller's qzero payload.
#
# ``assign=True`` rebinds ``self.qzeros`` to the incoming tensor, so the v1->v2
# converter used to write the corrected words into the caller's checkpoint
# storage. Reusing that payload converted it again and every zero point moved
# one step per load. The layout marker was aliased the same way.
# ---------------------------------------------------------------------------


class _V1RuntimeTorchLinear(TorchLinear):
    """A buffer-backed GPTQ kernel that decodes v1 words, like Marlin."""

    # This stand-in exists only to exercise the conversion path in-place. Kernel
    # discovery walks every ``BaseQuantLinear`` subclass, so without the opt-out
    # this class becomes a second selectable ``BACKEND.GPTQ_TORCH`` kernel and
    # every later ``from_quantized``/``quantize`` call in the process fails with
    # "Multiple kernels matched".
    SUPPORTS_BACKEND_SELECTION = False
    REQUIRES_FORMAT_V2 = False
    SUPPORTS_BACKENDS = TorchLinear.SUPPORTS_BACKENDS
    SUPPORTS_METHODS = TorchLinear.SUPPORTS_METHODS
    SUPPORTS_FORMATS = TorchLinear.SUPPORTS_FORMATS
    SUPPORTS_BITS = TorchLinear.SUPPORTS_BITS
    SUPPORTS_GROUP_SIZE = TorchLinear.SUPPORTS_GROUP_SIZE
    SUPPORTS_DESC_ACT = TorchLinear.SUPPORTS_DESC_ACT
    SUPPORTS_SYM = TorchLinear.SUPPORTS_SYM
    SUPPORTS_SHARDS = TorchLinear.SUPPORTS_SHARDS
    SUPPORTS_TRAINING = TorchLinear.SUPPORTS_TRAINING
    SUPPORTS_AUTO_PADDING = TorchLinear.SUPPORTS_AUTO_PADDING
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = TorchLinear.SUPPORTS_IN_FEATURES_DIVISIBLE_BY
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = TorchLinear.SUPPORTS_OUT_FEATURES_DIVISIBLE_BY
    SUPPORTS_DEVICES = TorchLinear.SUPPORTS_DEVICES
    SUPPORTS_PLATFORM = TorchLinear.SUPPORTS_PLATFORM
    SUPPORTS_PACK_DTYPES = TorchLinear.SUPPORTS_PACK_DTYPES
    SUPPORTS_ADAPTERS = TorchLinear.SUPPORTS_ADAPTERS
    SUPPORTS_DTYPES = TorchLinear.SUPPORTS_DTYPES


def _packed_module(fmt):
    """A packed module whose live words are logical (v2)."""

    layer = torch.nn.Linear(IN_FEATURES, OUT_FEATURES, bias=False)
    with torch.no_grad():
        layer.weight.fill_(1)
    return hf_pack_layer(
        _module(fmt),
        layer,
        torch.ones(OUT_FEATURES, 1),
        torch.full((OUT_FEATURES, 1), 8),
        torch.zeros(IN_FEATURES, dtype=torch.int32),
        checkpoint_format=fmt,
        post_init=True,
    )


def _markerless_v1_state():
    """The markerless v1 payload a ``gptq`` checkpoint writes."""

    packed = _packed_module(FORMAT.GPTQ)
    with checkpoint_serialization():
        state = packed.state_dict()
    assert "qzeros_format" not in state
    assert int(state["qzeros"][0, 0]) & 0xFFFFFFFF == V1_WORD
    return state


@pytest.mark.parametrize("assign", [False, True])
def test_reused_markerless_v1_payload_stays_v1(assign):
    state = _markerless_v1_state()
    x = torch.eye(IN_FEATURES, dtype=torch.float16)[:1]
    expected = torch.ones(1, OUT_FEATURES, dtype=torch.float16)

    first = _module(FORMAT.GPTQ)
    first.load_state_dict(state, assign=assign)
    first.post_init()
    torch.testing.assert_close(first(x), expected, rtol=0, atol=0)

    second = _module(FORMAT.GPTQ)
    second.load_state_dict(state, assign=assign)
    second.post_init()
    torch.testing.assert_close(second(x), expected, rtol=0, atol=0)

    assert int(state["qzeros"][0, 0]) & 0xFFFFFFFF == V1_WORD, (
        f"assign={assign}: the load rewrote the caller's v1 qzero words"
    )
    assert int(first.qzeros[0, 0]) & 0xFFFFFFFF == V2_WORD
    assert int(first.state_dict()["qzeros"][0, 0]) & 0xFFFFFFFF == V1_WORD
    # A cold copy of the first receiver must decode the same zero points.
    first.clear_weight_cache()
    torch.testing.assert_close(first(x), expected, rtol=0, atol=0)


def test_assign_load_keeps_the_marker_carrying_payload_consistent():
    state = _packed_module(FORMAT.GPTQ).state_dict()
    assert state["qzeros_format"].item() == 1
    assert int(state["qzeros"][0, 0]) & 0xFFFFFFFF == V1_WORD

    receiver = _module(FORMAT.GPTQ)
    receiver.load_state_dict(state, assign=True)

    assert int(state["qzeros"][0, 0]) & 0xFFFFFFFF == V1_WORD, (
        "assign=True converted the caller's qzero words in place"
    )
    assert state["qzeros_format"].item() == 1, (
        "assign=True relabelled the caller's marker while its words stayed v1"
    )
    assert int(receiver.qzeros[0, 0]) & 0xFFFFFFFF == V2_WORD
    assert receiver.qzero_format() == 2
    assert receiver._buffers["qzeros_format"].item() == 2


def test_assign_v2_to_v1_conversion_keeps_the_source_words():
    packed = _packed_module(FORMAT.GPTQ_V2)
    state = packed.state_dict()
    assert state["qzeros_format"].item() == 2
    assert int(state["qzeros"][0, 0]) & 0xFFFFFFFF == V2_WORD

    receiver = _V1RuntimeTorchLinear(
        bits=BITS,
        group_size=GROUP,
        sym=True,
        desc_act=False,
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        pack_dtype=torch.int32,
        backend=BACKEND.GPTQ_TORCH,
        bias=False,
        register_buffers=True,
        format=FORMAT.GPTQ,
    )
    receiver.load_state_dict(state, assign=True)

    assert int(state["qzeros"][0, 0]) & 0xFFFFFFFF == V2_WORD, (
        "assign=True converted the caller's v2 qzero words in place"
    )
    assert state["qzeros_format"].item() == 2
    assert int(receiver.qzeros[0, 0]) & 0xFFFFFFFF == V1_WORD
    assert receiver.qzero_format() == 1
    assert receiver._buffers["qzeros_format"].item() == 1


def test_own_assigned_qzeros_preserves_a_parameter_buffer():
    """Marlin-style kernels register ``qzeros`` as a Parameter, not a buffer.

    The clone must stay assignable to that name; a plain tensor would raise
    ``cannot assign 'torch.IntTensor' as parameter`` for those kernels.
    """

    module = _module(FORMAT.GPTQ)
    module.qzeros = torch.nn.Parameter(module.qzeros.detach().clone(), requires_grad=False)
    original = module.qzeros
    before = original.detach().clone()

    module._own_assigned_qzeros()

    assert isinstance(module.qzeros, torch.nn.Parameter)
    assert module.qzeros is not original
    torch.testing.assert_close(module.qzeros, before, rtol=0, atol=0)
