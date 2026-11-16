# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint metadata must describe the artifact, not the quantizer's host."""

from __future__ import annotations

import json
import os

import pytest
import torch

from gptqmodel.models.writer import _parse_max_shard_size
from gptqmodel.quantization.config import QuantizeConfig
from gptqmodel.utils.model import TensorSource, streaming_state_dict_to_shards


def test_max_shard_size_uses_hugging_face_units():
    assert _parse_max_shard_size("4GB") == 4_000_000_000
    assert _parse_max_shard_size("4MB") == 4_000_000
    assert _parse_max_shard_size("4GiB") == 4 * 1024 ** 3
    assert _parse_max_shard_size("4MiB") == 4 * 1024 ** 2
    # Bare suffixes keep their historical binary meaning.
    assert _parse_max_shard_size("4G") == 4 * 1024 ** 3
    assert _parse_max_shard_size(1234) == 1234
    with pytest.raises(ValueError):
        _parse_max_shard_size("4iB")


def test_index_total_size_counts_tensor_bytes_not_file_bytes(tmp_path):
    entries = {
        f"layer.{index}.weight": TensorSource(
            name=f"layer.{index}.weight",
            torch_dtype=torch.float32,
            shape=(8,),
            source=torch.arange(8, dtype=torch.float32),
        )
        for index in range(3)
    }
    filenames, weight_map, total_size = streaming_state_dict_to_shards(
        entries,
        save_dir=str(tmp_path),
        model_base_name="model",
        single_file_name="model.safetensors",
        metadata={"format": "pt"},
        max_shard_size=16,
    )

    assert len(filenames) == 3
    assert set(weight_map) == set(entries)
    assert total_size == sum(entry.num_bytes for entry in entries.values())
    assert total_size < sum(os.path.getsize(tmp_path / name) for name in filenames)


def test_quantize_config_does_not_publish_the_local_offload_path():
    config = QuantizeConfig(bits=4, offload_to_disk=True)
    config._ensure_offload_temp_dir()
    assert config.offload_to_disk_path

    payload = config.to_dict()

    assert payload["meta"]["offload_to_disk"] is True
    assert "offload_to_disk_path" not in payload["meta"]


def test_legacy_offload_path_is_still_read():
    config = QuantizeConfig.from_quant_config(
        {
            "bits": 4,
            "group_size": 128,
            "quant_method": "gptq",
            "meta": {"offload_to_disk": True, "offload_to_disk_path": "/tmp/legacy"},
        }
    )
    assert config.offload_to_disk_path == "/tmp/legacy"
