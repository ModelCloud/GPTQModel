# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for #4777.

Resuming a QQQ quantization restores finished modules on ``meta`` and registers
them with an accelerate disk-offload hook. The hook's index is built from
``state_dict()``, which excludes non-persistent buffers, so ``QQQLinear``'s
split-K ``workspace`` stayed on ``meta`` with no stored value and accelerate
raised

    ValueError: workspace is on the meta device, we need a `value` to put in on cpu.

Every resume path for QQQ failed this way. The hook now materializes
non-persistent meta buffers first; they are runtime scratch, never checkpoint
state.
"""

import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch import nn

from gptqmodel.nn_modules.qlinear.qqq import QQQTorchLinear
from gptqmodel.utils.offload import (
    attach_disk_offload_hook,
    is_meta_module,
    restore_disk_offloaded_modules,
)


IN_FEATURES = 128
OUT_FEATURES = 128
GROUP_SIZE = 128


def _kernel_kwargs() -> dict:
    return {
        "bits": 4,
        "group_size": GROUP_SIZE,
        "desc_act": False,
        "sym": True,
        "in_features": IN_FEATURES,
        "out_features": OUT_FEATURES,
        "bias": True,
        "register_buffers": True,
    }


def _seeded_source() -> QQQTorchLinear:
    """A packed QQQ module with finite scales and codes (``__init__`` empties)."""

    torch.manual_seed(4777)
    module = QQQTorchLinear(**_kernel_kwargs())
    module.B.copy_(torch.randint(0, 2**31 - 1, module.B.shape, dtype=torch.int32))
    module.s_channel.copy_(torch.rand(module.s_channel.shape) * 0.02 + 0.001)
    module.s_group.copy_((torch.rand(module.s_group.shape) * 0.02 + 0.001).half())
    module.bias.copy_(torch.randn(module.bias.shape) * 0.01)
    return module


def _write_resume_bundle(source: nn.Module, directory: Path) -> Path:
    """Write the bundle + index that ``GPTQCheckpoint.restore`` produces."""

    directory.mkdir(parents=True, exist_ok=True)
    state = source.state_dict()
    bundle = directory / "module.safetensors"
    save_file(state, str(bundle))

    raw = bundle.read_bytes()
    header_size = int.from_bytes(raw[:8], "little")
    header = json.loads(raw[8 : 8 + header_size])
    index = {}
    for key, tensor in state.items():
        start, end = header[key]["data_offsets"]
        index[key] = {
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "shape": list(tensor.shape),
            "safetensors_file": str(bundle.absolute()),
            "weight_name": key,
            "data_offsets": [8 + header_size + start, 8 + header_size + end],
        }
    (directory / "index.json").write_text(json.dumps(index))
    return directory


def _meta_restore(directory: Path) -> QQQTorchLinear:
    """Rebuild the module the way ``restore_packed_module`` does, then hook it."""

    with torch.device("meta"):
        module = QQQTorchLinear(**_kernel_kwargs())
    assert module.workspace.is_meta
    assert "workspace" not in module.state_dict()

    attach_disk_offload_hook(module, str(directory))
    return module


def test_resume_materializes_the_nonpersistent_workspace(tmp_path):
    source = _seeded_source()
    directory = _write_resume_bundle(source, tmp_path / "proj")

    module = _meta_restore(directory)

    # Used to raise `ValueError: workspace is on the meta device ...`.
    assert not module.workspace.is_meta
    assert module.workspace.shape == source.workspace.shape
    assert module.workspace.dtype is torch.int32
    assert torch.equal(module.workspace, torch.zeros_like(module.workspace))


def test_resumed_module_materializes_and_runs(tmp_path):
    source = _seeded_source()
    directory = _write_resume_bundle(source, tmp_path / "proj")

    class _Tree(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = _meta_restore(directory)

    tree = _Tree()

    # The loop-exit restore is what a resumed run reaches after this point.
    assert restore_disk_offloaded_modules(tree, device=torch.device("cpu")) == []
    assert not is_meta_module(tree.proj)

    x = torch.randn(2, 3, IN_FEATURES, dtype=torch.float16)
    with torch.inference_mode():
        restored = tree.proj(x)
        expected = source(x)

    assert restored.shape == (2, 3, OUT_FEATURES)
    assert torch.equal(restored, expected)


def test_persistent_meta_buffers_are_left_to_the_offload_index(tmp_path):
    """Only non-persistent scratch is allocated; real state still comes from disk."""

    source = _seeded_source()
    directory = _write_resume_bundle(source, tmp_path / "proj")

    module = _meta_restore(directory)
    # The hook serves the packed weight; it must not be replaced by zeros.
    assert module.B.is_meta or not torch.equal(module.B, torch.zeros_like(module.B))

    restore_disk_offloaded_modules(module, device=torch.device("cpu"))
    assert torch.equal(module.B, source.B)
    assert torch.equal(module.s_channel, source.s_channel)
    assert torch.equal(module.s_group, source.s_group)
