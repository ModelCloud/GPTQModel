# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""A CPU int4pack scale change must rebuild the affine offsets (#5372).

The CPU int4pack transform replaces the checkpoint's integer zero codes with
the affine offsets ``(2**(bits-1) - zero_code) * scale`` that the fused kernel
consumes. Those offsets are derived from ``scales``, so changing the scales
leaves them based on the old value: the layer then decodes with a new scale and
a stale offset, and even ``clear_weight_cache()`` republished the stale pair.

The fixture deliberately uses zero code 3, not 8. An offset of
``(8 - 8) * scale`` is zero for every scale, which is why the existing
cache-invalidation coverage never noticed the defect.
"""

from __future__ import annotations

import pytest
import torch

from gptqmodel.nn_modules.qlinear.torch_aten_kernel import TorchAtenLinear
from gptqmodel.nn_modules.qlinear.torch_aten_kernel_awq import TorchAtenAwqLinear
from gptqmodel.nn_modules.qlinear.torch_fused import TorchFusedLinear
from gptqmodel.nn_modules.qlinear.torch_fused_awq import TorchFusedAwqLinear
from gptqmodel.utils.torch import TORCH_HAS_FUSED_OPS


pytestmark = pytest.mark.skipif(
    not TORCH_HAS_FUSED_OPS,
    reason="the CPU int4pack kernels require torch>=2.8 fused ops",
)

KERNELS = [TorchFusedLinear, TorchAtenLinear, TorchFusedAwqLinear, TorchAtenAwqLinear]

BITS = 4
GROUP_SIZE = 32
IN_FEATURES = 32
OUT_FEATURES = 16
PACK_FACTOR = 32 // BITS
WEIGHT_CODE = 7
ZERO_CODE = 3
SCALE = 0.5
ZERO_CENTER = 1 << (BITS - 1)


def _module(cls):
    awq = "Awq" in cls.__name__
    # AWQ packs columnwise (``[in, out/pack]``); GPTQ packs the input axis.
    qweight_shape = (IN_FEATURES, OUT_FEATURES // PACK_FACTOR) if awq else (
        IN_FEATURES // PACK_FACTOR,
        OUT_FEATURES,
    )
    module = cls(
        bits=BITS,
        group_size=GROUP_SIZE,
        sym=False,
        desc_act=False,
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        bias=False,
        pack_dtype=torch.int32,
        register_buffers=True,
    )
    module.qweight = torch.full(qweight_shape, 0x77777777, dtype=torch.int32)
    module.qzeros = torch.full(
        (IN_FEATURES // GROUP_SIZE, OUT_FEATURES // PACK_FACTOR), 0x33333333, dtype=torch.int32
    )
    module.scales = torch.full(
        (IN_FEATURES // GROUP_SIZE, OUT_FEATURES), SCALE, dtype=torch.bfloat16
    )
    module.post_init()
    module.eval()
    return module


def _activation():
    x = torch.zeros(1, IN_FEATURES, dtype=torch.bfloat16)
    x[0, 0] = 1.0
    return x


def _decode(scale, offset):
    """The int4pack CPU op dequantizes ``scale * (code - 8) + offset``."""

    return scale * (WEIGHT_CODE - ZERO_CENTER) + offset


def _derived_offset(scale):
    return (ZERO_CENTER - ZERO_CODE) * scale


@pytest.mark.parametrize("cls", KERNELS, ids=lambda cls: cls.__name__)
def test_scale_change_rebuilds_the_affine_offsets(cls):
    """Doubling the scales must double the decoded weight, not leave it stale."""

    module = _module(cls)
    x = _activation()

    with torch.no_grad():
        assert module(x)[0, 0].item() == pytest.approx(_decode(SCALE, _derived_offset(SCALE)))
        module.scales.mul_(2)
        module.clear_weight_cache()
        assert module(x)[0, 0].item() == pytest.approx(
            _decode(SCALE * 2, _derived_offset(SCALE * 2))
        )


# ``TorchFusedAwqLinear`` publishes buffer changes through
# ``clear_weight_cache()``; the other three also refresh inside the forward.
SELF_REFRESHING_KERNELS = [TorchFusedLinear, TorchAtenLinear, TorchAtenAwqLinear]


@pytest.mark.parametrize("cls", SELF_REFRESHING_KERNELS, ids=lambda cls: cls.__name__)
def test_scale_change_is_observed_without_an_explicit_clear(cls):
    """A forward-only scale change is rebuilt where the kernel refreshes eagerly."""

    module = _module(cls)
    x = _activation()

    with torch.no_grad():
        module(x)
        module.scales.mul_(2)
        assert module(x)[0, 0].item() == pytest.approx(
            _decode(SCALE * 2, _derived_offset(SCALE * 2))
        )


@pytest.mark.parametrize("cls", KERNELS, ids=lambda cls: cls.__name__)
def test_an_explicit_offset_replacement_is_kept(cls):
    """Replacing ``qzeros`` is an explicit runtime offset, not a stale cache."""

    module = _module(cls)
    x = _activation()
    explicit = 1.0

    with torch.no_grad():
        module(x)
        module.scales.mul_(2)
        module.qzeros = torch.full_like(module.qzeros, explicit)
        module.clear_weight_cache()
        assert module(x)[0, 0].item() == pytest.approx(_decode(SCALE * 2, explicit))


@pytest.mark.parametrize("cls", KERNELS, ids=lambda cls: cls.__name__)
def test_a_grad_forward_observes_a_scale_change(cls):
    """A grad forward decodes the live layout and must see the new scales.

    The no-grad forward refreshes the affine offsets in ``_refresh_fused_state``
    (#5372); the eval-mode grad forward decodes the runtime layout directly, so
    it must refresh them first instead of pairing the new scales with the
    offsets derived from the old ones (#5571).
    """

    module = _module(cls)
    x = _activation()

    with torch.no_grad():
        module(x)
        module.scales.mul_(2)

    xx = x.clone().requires_grad_(True)
    assert module(xx)[0, 0].item() == pytest.approx(
        _decode(SCALE * 2, _derived_offset(SCALE * 2))
    )


@pytest.mark.parametrize("cls", KERNELS, ids=lambda cls: cls.__name__)
def test_a_compiled_grad_forward_observes_a_scale_change(cls):
    """A traced grad forward decodes fresh offsets without eager state writes.

    ``_refresh_fused_state`` cannot run inside a trace (cache keys and module
    writes), so the compiled branch decodes the affine offsets from the
    checkpoint zero codes and the current scales (#5571).
    """

    module = _module(cls)
    x = _activation()

    with torch.no_grad():
        module(x)
    comp = torch.compile(module, backend="eager")
    with torch.no_grad():
        comp(x)

    module.scales.mul_(2)
    xx = x.clone().requires_grad_(True)
    assert comp(xx)[0, 0].item() == pytest.approx(
        _decode(SCALE * 2, _derived_offset(SCALE * 2))
    )
