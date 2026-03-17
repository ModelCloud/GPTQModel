# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import json
import struct
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from tabulate import tabulate
from torch import nn

from gptqmodel.utils.model import get_state_dict_for_save, move_to, streaming_state_dict_to_shards
from gptqmodel.utils.offload import offload_to_disk, undo_offload_to_disk
from gptqmodel.utils.structure import alias_all_from_turtle_if_meta


class _LinearWithBuffers(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)
        self.register_buffer("scale_buffer", torch.linspace(0.0, 1.0, out_features))
        self.register_buffer("mask_buffer", torch.randint(0, 2, (out_features, in_features)).bool())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x * self.mask_buffer.float()) * self.scale_buffer


def _clone_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


class _HybridBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.inner = nn.Linear(width, width, bias=False)
        self.dt_bias = nn.Parameter(torch.randn(width))
        self.register_buffer("dt_scale", torch.linspace(0.0, 1.0, width))


class _HybridWrapper(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.block = _HybridBlock(width)


def test_offload_to_disk_writes_single_dat_file(tmp_path):
    model = _LinearWithBuffers(in_features=128, out_features=96)
    original_state = _clone_state_dict(model.linear)

    offload_root = tmp_path / "offload_root"
    offload_to_disk(module=model.linear, model=model, disk_path=str(offload_root))

    module_dir = offload_root / "linear"
    assert module_dir.is_dir(), "Expected per-module directory to exist"

    files = sorted(module_dir.iterdir(), key=lambda p: p.name)
    rows = [(path.name, path.stat().st_size) for path in files]
    print(tabulate(rows, headers=["file", "bytes"], tablefmt="github"))

    safetensor_files = [path for path in files if path.suffix == ".safetensors"]
    assert len(safetensor_files) == 1, "offload_to_disk should produce exactly one safetensors file per module"
    assert safetensor_files[0].name == "module.safetensors"

    with open(module_dir / "index.json", encoding="utf-8") as fp:
        index = json.load(fp)

    expected_keys = set(model.linear.state_dict().keys())
    assert set(index.keys()) == expected_keys
    assert all(Path(entry.get("safetensors_file")).name == "module.safetensors" for entry in index.values())
    assert all(entry.get("data_offsets") is not None for entry in index.values())

    save_dir = tmp_path / "saved"
    save_dir.mkdir()
    state_dict = get_state_dict_for_save(model, offload_root=str(offload_root))
    expected_files, tensor_to_filename, _ = streaming_state_dict_to_shards(
        state_dict,
        save_dir=str(save_dir),
        model_base_name="model",
        single_file_name="model.safetensors",
        metadata={},
        max_shard_size=None,
    )

    assert len(expected_files) == 1
    shard_path = save_dir / expected_files[0]
    with safe_open(str(shard_path), framework="pt", device="cpu") as handler:
        for name, tensor in original_state.items():
            saved = handler.get_tensor(f"linear.{name}")
            torch.testing.assert_close(saved, tensor)

    # Materialize the module back and ensure values match the snapshot captured before offload.
    undo_offload_to_disk(model.linear, delete_offload_folders=False)
    for name, tensor in model.linear.state_dict().items():
        torch.testing.assert_close(tensor, original_state[name])


def test_alias_all_from_turtle_restores_direct_meta_tensors_with_offloaded_children(tmp_path):
    turtle_model = _HybridWrapper(width=16)
    shell_model = _HybridWrapper(width=16)
    shell_model.load_state_dict(turtle_model.state_dict())

    original_state = _clone_state_dict(turtle_model)
    offload_root = tmp_path / "offload_root"
    offload_to_disk(module=shell_model.block.inner, model=shell_model, disk_path=str(offload_root))

    shell_model.block.dt_bias = nn.Parameter(
        torch.empty_like(shell_model.block.dt_bias, device="meta"),
        requires_grad=shell_model.block.dt_bias.requires_grad,
    )
    shell_model.block.register_buffer(
        "dt_scale",
        torch.empty_like(shell_model.block.dt_scale, device="meta"),
        persistent=True,
    )

    alias_all_from_turtle_if_meta(shell_model=shell_model, turtle_model=turtle_model)

    state_dict = get_state_dict_for_save(shell_model, offload_root=str(offload_root))
    save_dir = tmp_path / "saved"
    save_dir.mkdir()
    expected_files, _, _ = streaming_state_dict_to_shards(
        state_dict,
        save_dir=str(save_dir),
        model_base_name="model",
        single_file_name="model.safetensors",
        metadata={},
        max_shard_size=None,
    )

    shard_path = save_dir / expected_files[0]
    with safe_open(str(shard_path), framework="pt", device="cpu") as handler:
        for name, tensor in original_state.items():
            saved = handler.get_tensor(name)
            torch.testing.assert_close(saved, tensor)


def test_streaming_state_dict_pads_safetensors_header_to_8_bytes(tmp_path):
    model = nn.Linear(3, 5, bias=False)
    state_dict = get_state_dict_for_save(model)

    # Force an unaligned raw header so the regression test would fail without padding.
    metadata = {"format": "pt"}
    for size in range(1, 33):
        candidate = dict(metadata, pad=("x" * size))
        raw_header = {
            "__metadata__": candidate,
            "weight": {
                "dtype": "F32",
                "shape": list(model.weight.shape),
                "data_offsets": [0, model.weight.numel() * model.weight.element_size()],
            },
        }
        raw_header_len = len(json.dumps(raw_header, separators=(",", ":")).encode("utf-8"))
        if raw_header_len % 8 != 0:
            metadata = candidate
            break
    else:
        raise AssertionError("Failed to construct an unaligned safetensors header for the regression test.")

    save_dir = tmp_path / "saved"
    save_dir.mkdir()
    expected_files, _, _ = streaming_state_dict_to_shards(
        state_dict,
        save_dir=str(save_dir),
        model_base_name="model",
        single_file_name="model.safetensors",
        metadata=metadata,
        max_shard_size=None,
    )

    shard_path = save_dir / expected_files[0]
    with shard_path.open("rb") as handle:
        stored_header_len = struct.unpack("<Q", handle.read(8))[0]

    assert raw_header_len % 8 != 0
    assert stored_header_len % 8 == 0

    with safe_open(str(shard_path), framework="pt", device="cpu") as handler:
        torch.testing.assert_close(handler.get_tensor("weight"), model.weight.detach())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for offload rematerialization test")
def test_move_to_restores_offloaded_meta_leaves_before_gpu_transfer(tmp_path):
    model = _LinearWithBuffers(in_features=32, out_features=24)
    original_state = _clone_state_dict(model.linear)

    offload_root = tmp_path / "offload_root"
    offload_to_disk(module=model.linear, model=model, disk_path=str(offload_root))

    restored = move_to(model.linear, device=torch.device("cuda", 0))

    for name, tensor in restored.state_dict().items():
        assert tensor.device.type == "cuda"
        torch.testing.assert_close(tensor.cpu(), original_state[name])
