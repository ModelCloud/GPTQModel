# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""An int-ordinal device map must not be tested for truthiness (#4964).

#4915 spells ``cuda:0`` as the int ordinal ``0`` so ``safetensors`` only touches
the requested device. ``_restore_checkpoint_buffers`` then placed a packed
(FP8) weight, which Accelerate never sees, with

    target_device = mapped_device(name) or target_device

and ``0`` is falsy, so the fallback won: every packed FP8 weight stayed on the
CPU slot while its scales reached the GPU. Decode was ~4x slower and the
reported VRAM was a fraction of the checkpoint, with no error.

The resolver now returns a ``torch.device`` (or ``None``), and the call site
tests ``is not None``, so ordinal ``0`` and ordinal ``1`` behave alike.
"""

from __future__ import annotations

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from gptqmodel.utils.model import _device_map_target, _restore_checkpoint_buffers


def test_int_zero_resolves_to_the_first_accelerator():
    assert _device_map_target({"": 0}, "a.weight") == torch.device("cuda", 0)
    assert _device_map_target({"": 1}, "a.weight") == torch.device("cuda", 1)


def test_string_and_device_forms_are_unchanged():
    assert _device_map_target({"": "cuda:0"}, "a.weight") == torch.device("cuda", 0)
    assert _device_map_target({"": torch.device("cuda", 1)}, "a.weight") == torch.device("cuda", 1)
    assert _device_map_target({"": "cpu"}, "a.weight") == torch.device("cpu")


def test_offload_and_absent_entries_resolve_to_none():
    assert _device_map_target({"": "disk"}, "a.weight") is None
    assert _device_map_target({"": "meta"}, "a.weight") is None
    assert _device_map_target({}, "a.weight") is None
    assert _device_map_target(None, "a.weight") is None


def test_longest_prefix_wins():
    device_map = {"": 0, "layers.0": 1, "layers.0.mlp": 2}
    assert _device_map_target(device_map, "layers.0.mlp.down_proj.weight") == torch.device("cuda", 2)
    assert _device_map_target(device_map, "layers.0.self_attn.q_proj.weight") == torch.device("cuda", 1)
    assert _device_map_target(device_map, "embed_tokens.weight") == torch.device("cuda", 0)


class _Packed(nn.Module):
    """A module shaped like a packed FP8 projection: weight + scale."""

    def __init__(self):
        super().__init__()
        self.register_buffer("weight", torch.zeros(4, 4, dtype=torch.uint8))
        self.register_buffer("weight_scale_inv", torch.zeros(1, dtype=torch.float32))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA target device")
def test_packed_weight_lands_on_the_mapped_gpu_for_ordinal_zero(tmp_path):
    """The #4964 regression: ordinal ``0`` must place the packed weight on the GPU."""

    model = _Packed()
    # Accelerate leaves the slot on CPU because packed FP8 weights are not in
    # the checkpoint it reads (#4509); the restore has to move it.
    assert model.weight.device.type == "cpu"

    checkpoint = tmp_path / "model.safetensors"
    save_file({"weight": torch.ones(4, 4, dtype=torch.uint8)}, checkpoint)

    _restore_checkpoint_buffers(
        model,
        str(checkpoint),
        {"weight": torch.uint8},
        exact_source_names={"weight"},
        device_map={"": 0},
    )

    assert model.weight.device == torch.device("cuda:0")
    assert torch.equal(model.weight.cpu(), torch.ones(4, 4, dtype=torch.uint8))


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="needs two visible CUDA devices",
)
def test_ordinal_one_is_unchanged(tmp_path):
    model = _Packed()
    checkpoint = tmp_path / "model.safetensors"
    save_file({"weight": torch.ones(4, 4, dtype=torch.uint8)}, checkpoint)

    _restore_checkpoint_buffers(
        model,
        str(checkpoint),
        {"weight": torch.uint8},
        exact_source_names={"weight"},
        device_map={"": 1},
    )

    assert model.weight.device == torch.device("cuda:1")
