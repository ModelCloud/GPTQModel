# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Issue #5417: an ignored-only shard must not import the optional
``bitsandbytes`` package.

``convert_bitsandbytes_shard`` imported the dependency before it looked at the
tensors, so a shard whose only module is ignored failed in a clean test
environment even though no packed decode ran.
"""

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from gptqmodel.utils import model_dequant
from gptqmodel.utils.model_dequant import convert_bitsandbytes_shard


pytestmark = [pytest.mark.cpu]


def _ignored_shard(tmp_path):
    weight = torch.randn(2, 2, dtype=torch.bfloat16)
    path = tmp_path / "ignored.safetensors"
    save_file(
        {
            "ignored.weight": weight,
            "ignored.weight_scale": torch.ones(1, dtype=torch.float32),
            "ignored.qweight": torch.ones(1, 1, dtype=torch.int32),
        },
        str(path),
    )
    return path, weight


def test_ignored_only_shard_does_not_import_bitsandbytes(tmp_path, monkeypatch):
    path, weight = _ignored_shard(tmp_path)

    def _unavailable():
        raise RuntimeError("bitsandbytes is not installed")

    monkeypatch.setattr(model_dequant, "_get_bitsandbytes_dependencies", _unavailable)

    with safe_open(str(path), framework="pt", device="cpu") as reader:
        out = convert_bitsandbytes_shard(
            reader, torch.bfloat16, quant_cfg={}, ignored_layers={"ignored"}
        )

    assert set(out) == {"ignored.weight"}
    torch.testing.assert_close(out["ignored.weight"], weight)


def test_packed_shard_still_requires_bitsandbytes(tmp_path, monkeypatch):
    path = tmp_path / "packed.safetensors"
    save_file(
        {
            "m.weight": torch.ones(1, 1, dtype=torch.uint8),
            "m.weight_absmax": torch.ones(1, dtype=torch.float32),
            "m.weight_quant_map": torch.ones(1, dtype=torch.float32),
            "m.weight_quant_state": torch.ones(1, dtype=torch.uint8),
        },
        str(path),
    )

    calls = []

    def _unavailable():
        calls.append(1)
        raise RuntimeError("bitsandbytes is not installed")

    monkeypatch.setattr(model_dequant, "_get_bitsandbytes_dependencies", _unavailable)

    with safe_open(str(path), framework="pt", device="cpu") as reader:
        with pytest.raises(RuntimeError, match="bitsandbytes is not installed"):
            convert_bitsandbytes_shard(reader, torch.bfloat16, quant_cfg={})

    assert calls == [1]
