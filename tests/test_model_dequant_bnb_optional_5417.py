# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""The bitsandbytes converter must not import its optional dependency early.

``convert_bitsandbytes_shard`` resolved bitsandbytes before looking at the
keys, so a shard that contains only ignored tensors failed on a host without
the package even though no packed branch runs (issue #5417). The dependency is
now resolved inside the packed 4-bit and 8-bit decode branches only; the
ignored-tensor path stays in the base test environment.
"""

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from gptqmodel.utils import model_dequant
from gptqmodel.utils.model_dequant import convert_bitsandbytes_shard


def _save_shard(tmp_path, tensors):
    shard_path = tmp_path / "shard.safetensors"
    save_file(tensors, str(shard_path))
    return shard_path


def _ignored_only_tensors(weight):
    return {
        "ignored.weight": weight,
        "ignored.qweight": torch.ones(1, 1, dtype=torch.int32),
        "ignored.qzeros": torch.ones(1, 1, dtype=torch.int32),
        "ignored.scales": torch.ones(1, 1, dtype=torch.float32),
        "ignored.g_idx": torch.zeros(1, dtype=torch.int32),
    }


def test_ignored_tensors_decode_without_bitsandbytes(tmp_path, monkeypatch):
    """An ignored-only shard never touches the optional dependency."""

    def missing():
        raise RuntimeError(
            "Support for bitsandbytes quantized checkpoints requires the "
            "'bitsandbytes' package."
        )

    monkeypatch.setattr(model_dequant, "_get_bitsandbytes_dependencies", missing)

    weight = torch.randn(2, 2, dtype=torch.bfloat16)
    shard_path = _save_shard(tmp_path, _ignored_only_tensors(weight))

    with safe_open(shard_path, framework="pt", device="cpu") as reader:
        out = convert_bitsandbytes_shard(reader, torch.bfloat16, quant_cfg={}, ignored_layers={"ignored"})

    assert set(out) == {"ignored.weight"}
    torch.testing.assert_close(out["ignored.weight"], weight)


def test_packed_4bit_shard_still_requires_bitsandbytes(tmp_path, monkeypatch):
    """The packed 4-bit branch still fails closed without the dependency."""

    def missing():
        raise RuntimeError("bitsandbytes is required")

    monkeypatch.setattr(model_dequant, "_get_bitsandbytes_dependencies", missing)

    tensors = _ignored_only_tensors(torch.randn(2, 2, dtype=torch.bfloat16))
    tensors.update(
        {
            "model.weight": torch.ones(2, 2, dtype=torch.uint8),
            "model.weight_absmax": torch.ones(1, dtype=torch.float32),
            "model.weight_quant_map": torch.ones(1, dtype=torch.float32),
            "model.weight_quant_state": torch.ones(1, dtype=torch.float32),
        }
    )
    shard_path = _save_shard(tmp_path, tensors)

    with safe_open(shard_path, framework="pt", device="cpu") as reader:
        with pytest.raises(RuntimeError, match="bitsandbytes"):
            convert_bitsandbytes_shard(reader, torch.bfloat16, quant_cfg={}, ignored_layers={"ignored"})


def test_packed_8bit_shard_still_requires_bitsandbytes(tmp_path, monkeypatch):
    """The packed int8 branch still fails closed without the dependency."""

    def missing():
        raise RuntimeError("bitsandbytes is required")

    monkeypatch.setattr(model_dequant, "_get_bitsandbytes_dependencies", missing)

    tensors = _ignored_only_tensors(torch.randn(2, 2, dtype=torch.bfloat16))
    tensors.update(
        {
            "model.weight": torch.ones(2, 2, dtype=torch.int8),
            "model.weight_scb": torch.ones(2, dtype=torch.float32),
        }
    )
    shard_path = _save_shard(tmp_path, tensors)

    with safe_open(shard_path, framework="pt", device="cpu") as reader:
        with pytest.raises(RuntimeError, match="bitsandbytes"):
            convert_bitsandbytes_shard(reader, torch.bfloat16, quant_cfg={}, ignored_layers={"ignored"})
