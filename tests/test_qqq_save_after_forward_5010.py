# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""A QQQ save after an in-session forward must still publish its weights (#5010).

Since #4856 the first CPU forward of ``QQQTorchLinear`` drops the checkpoint
``B``/``s_group``/``s_channel`` buffers once the packed cache can rebuild them.
``GPTQModel.save()`` does not call ``state_dict()``; it walks
``named_buffers(recurse=False)``, and PyTorch skips a ``None`` buffer. The
ordinary flow quantize -> generate -> save therefore wrote a checkpoint with no
quantized weights at all, and the reload crashed the CUDA context (or returned
all-NaN logits on ``QQQ_TORCH``).
"""

from __future__ import annotations

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from gptqmodel.nn_modules.qlinear.qqq import QQQTorchLinear
from gptqmodel.quantization import FORMAT
from gptqmodel.utils.model import (
    _restore_checkpoint_layout_for_export,
    get_state_dict_for_save,
    validate_checkpoint_qweights,
)


IN_FEATURES = 256
OUT_FEATURES = 128
GROUP_SIZE = 128
DTYPE = torch.float16


def _build_module() -> QQQTorchLinear:
    torch.manual_seed(5010)
    linear = nn.Linear(IN_FEATURES, OUT_FEATURES, bias=False, dtype=DTYPE)
    linear.weight.data.normal_(0, 0.08)
    groups = IN_FEATURES // GROUP_SIZE
    scales = torch.rand((OUT_FEATURES, groups), dtype=DTYPE) * 0.02 + 0.01
    s_extra = torch.rand(OUT_FEATURES, dtype=torch.float32) * 0.001 + 0.001

    module = QQQTorchLinear(
        bits=4,
        group_size=GROUP_SIZE,
        sym=True,
        desc_act=False,
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        bias=False,
        register_buffers=True,
    )
    module.pack(linear, scales, s_extra)
    module.post_init()
    return module.eval().to(dtype=DTYPE)


def _dropped_module() -> QQQTorchLinear:
    """A module in the state the first AVX512 CPU forward leaves behind.

    That forward calls ``_packed_weight_for_cpu()``, which builds the cache and
    then drops the checkpoint buffers. It is called directly here because this
    host is ARM64 and has no AVX512 packed kernel, so a plain forward takes the
    generic matmul path and never drops anything.
    """

    module = _build_module()
    module._packed_weight_for_cpu()
    assert module.B is None, "the fixture did not drop the checkpoint buffers"
    assert module.s_channel is None
    return module


def test_the_packed_build_drops_the_checkpoint_buffers():
    """Guard the fixture: without the drop this file cannot reproduce #5010."""

    module = _dropped_module()

    assert module._torch_dropped_checkpoint is not None
    assert module._torch_packed_cache is not None


def test_export_materializes_the_dropped_buffers():
    module = _dropped_module()
    model = nn.Module()
    model.layer = module

    restored = _restore_checkpoint_layout_for_export(model)

    assert restored == 1
    for name in ("B", "s_group", "s_channel"):
        buffer = getattr(module, name)
        assert isinstance(buffer, torch.Tensor), f"{name} was not published for export"
    assert module._torch_dropped_checkpoint is None


def test_save_manifest_contains_every_quantized_tensor():
    module = _dropped_module()
    model = nn.Module()
    model.layer = module

    state = get_state_dict_for_save(model)
    keys = set(state)

    for name in ("B", "s_group", "s_channel"):
        assert f"layer.{name}" in keys, (
            f"`layer.{name}` missing from the save manifest: {sorted(keys)}"
        )


def test_forward_still_works_after_export():
    module = _dropped_module()
    model = nn.Module()
    model.layer = module
    activations = torch.randn(2, IN_FEATURES, dtype=DTYPE)
    with torch.inference_mode():
        before = module(activations).clone()

    _restore_checkpoint_layout_for_export(model)
    with torch.inference_mode():
        after = module(activations)

    assert torch.equal(before, after)


def test_loading_a_qqq_checkpoint_without_its_weights_fails_closed(tmp_path):
    """A truncated QQQ checkpoint must raise, not load uninitialized buffers."""

    module = _build_module()
    model = nn.Module()
    model.layer = module
    # Everything a truncated save would still contain: the dense leftovers.
    save_file(
        {"other": torch.zeros(4)},
        str(tmp_path / "model.safetensors"),
        metadata={"format": "pt"},
    )

    with pytest.raises(ValueError, match=r"layer\.B"):
        validate_checkpoint_qweights(model, str(tmp_path), FORMAT.QQQ)


def test_a_complete_qqq_checkpoint_passes_validation(tmp_path):
    module = _build_module()
    model = nn.Module()
    model.layer = module
    state = {
        f"layer.{name}": getattr(module, name)
        for name in ("B", "s_group", "s_channel")
        if getattr(module, name) is not None
    }
    save_file(
        {name: value.contiguous() for name, value in state.items()},
        str(tmp_path / "model.safetensors"),
        metadata={"format": "pt"},
    )

    validate_checkpoint_qweights(model, str(tmp_path), FORMAT.QQQ)
