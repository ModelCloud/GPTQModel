# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import json
from pathlib import Path

import torch
from safetensors import safe_open
from tabulate import tabulate
from torch import nn

from gptqmodel.utils.model import get_state_dict_for_save, streaming_state_dict_to_shards
from gptqmodel.utils.offload import offload_to_disk, undo_offload_to_disk


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
