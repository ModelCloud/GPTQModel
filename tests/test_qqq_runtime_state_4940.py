# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""A class-swapped ``QQQTorchLinear`` must not depend on ``__init__`` state.

``rehome_quantized_kernel`` runs ``_switch_quant_linear_class`` at the end of
``quantize()``: it assigns ``module.__class__`` so the packed module keeps its
buffers while the compute path changes. That never runs the target kernel's
``__init__``, so any runtime attribute only ``__init__`` created is missing and
the following ``clear_weight_cache()``/``post_init()`` raises. This broke #4451
and again #4856 (``_torch_dropped_checkpoint``), which made every QQQ
``quantize()`` fail at the very end with ``AttributeError`` (#4940).

The fix installs the state through an optional ``_init_runtime_state`` hook that
the swap calls, and keeps class-level defaults so a direct attribute read cannot
raise either.
"""

from __future__ import annotations

import torch

from gptqmodel.models.base import _switch_quant_linear_class
from gptqmodel.nn_modules.qlinear.qqq import QQQLinear, QQQTorchLinear
from tests.test_qqq_kernel_memory_4387 import _build_torch_module


def _stripped_swapped_module(**kwargs) -> QQQTorchLinear:
    """A packed QQQ module whose ``QQQTorchLinear`` runtime state was never set."""

    module = _build_torch_module(**kwargs)
    # Emulate ``rehome_quantized_kernel``: the module arrives as a plain
    # ``QQQLinear`` (no Torch-kernel attributes), then the class is swapped.
    module.__class__ = QQQLinear
    for name in (
        "_torch_packed_cache",
        "_torch_packed_cache_key",
        "_torch_dropped_checkpoint",
        "_torch_int_mm_failed_shapes",
    ):
        module.__dict__.pop(name, None)
    return module


def test_switch_quant_linear_class_installs_runtime_state():
    module = _stripped_swapped_module()
    assert "_torch_dropped_checkpoint" not in module.__dict__

    _switch_quant_linear_class(module, QQQTorchLinear)

    assert type(module) is QQQTorchLinear
    assert module._torch_dropped_checkpoint is None
    assert module._torch_packed_cache is None
    assert module._torch_packed_cache_key is None
    assert module._torch_int_mm_failed_shapes == set()


def test_clear_weight_cache_after_swap_does_not_raise():
    module = _stripped_swapped_module()

    _switch_quant_linear_class(module, QQQTorchLinear)
    # The end-of-quantize path calls both, in this order.
    module.clear_weight_cache()
    module.post_init()

    with torch.inference_mode():
        output = module(torch.randn(3, module.in_features, dtype=torch.float16))
    assert output.shape == (3, module.out_features)


def test_clear_weight_cache_survives_a_bare_class_swap():
    """Even without the hook, the class-level defaults keep the reads safe."""

    module = _stripped_swapped_module()
    module.__class__ = QQQTorchLinear

    module.clear_weight_cache()
    module.post_init()

    with torch.inference_mode():
        output = module(torch.randn(2, module.in_features, dtype=torch.float16))
    assert output.shape == (2, module.out_features)


def test_failed_int_mm_memo_is_per_instance():
    """The failure memo must not be shared through the class-level default."""

    first = _stripped_swapped_module()
    second = _stripped_swapped_module()
    _switch_quant_linear_class(first, QQQTorchLinear)
    _switch_quant_linear_class(second, QQQTorchLinear)

    first._failed_int_mm_shapes().add(("cpu", 1, 2, 3))

    assert first._failed_int_mm_shapes() != second._failed_int_mm_shapes()
    assert second._failed_int_mm_shapes() == set()
