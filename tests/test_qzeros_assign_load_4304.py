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

import torch

from gptqmodel import BACKEND
from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.quantization import FORMAT


IN_FEATURES = 32
OUT_FEATURES = 8
GROUP = 32
BITS = 4

# 4-bit packing stores eight zero points per int32 word. The v1 checkpoint
# convention stores `zero_point - 1`, so `0x77777777` decodes to zero points of
# 8 and the logical (v2) words are `0x88888888`.
V1_WORD = 0x77777777
V2_WORD = 0x88888888


def _module():
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
        format=FORMAT.GPTQ_V2,
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
