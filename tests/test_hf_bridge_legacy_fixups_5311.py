# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Issue #5311: the Transformers bridge must run the legacy remote-code repairs.

``BaseQModel.__init__`` restores rotary buffers that transformers 5 materializes
with ``torch.empty_like`` (a remote-code class that overrides ``_init_weights``
never re-derives ``inv_freq``), and installs the transformers-4 ``generate()``
adapter for remote ``prepare_inputs_for_generation`` implementations. The bridge
(``hf_gptqmodel_post_init_for_load``, used by
``AutoModelForCausalLM.from_pretrained`` and ``pipeline()``) mirrored the other
post-load steps but skipped these two, so a checkpoint that ``GPTQModel.load``
reads at PPL 13 produced garbage RoPE frequencies (PPL 478) through transformers.

Both repairs now live in ``apply_legacy_remote_code_fixups`` and both load paths
call it, so a new repair cannot be added to one path only.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import gptqmodel.utils.hf as hf_utils
import gptqmodel.utils.structure as structure_utils
from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.quantization import QuantizeConfig
from gptqmodel.utils.hf_load import (
    HFGPTQModelLoadContext,
    hf_gptqmodel_post_init_for_load,
)


REPO_ROOT = Path(__file__).resolve().parents[1]

pytestmark = [pytest.mark.cpu]


class _LegacyRotary(nn.Module):
    """Remote-code rotary module: ``inv_freq`` is a non-persistent buffer."""

    def __init__(self, dim: int, max_position_embeddings: int = 32, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)


class _LegacyRemoteModel(nn.Module):
    """Model whose ``_init_weights`` override never re-derives the rotary buffer."""

    def __init__(self, dim: int = 8):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=dim * 2, num_attention_heads=2)
        self.rotary_emb = _LegacyRotary(dim)

    def _init_weights(self, module):  # remote-code override
        pass


def _bridge_model() -> nn.Module:
    model = torch.nn.Module()
    model.config = SimpleNamespace(_attn_implementation="sdpa")
    model.hf_device_map = {"": "cpu"}
    return model


def _context() -> HFGPTQModelLoadContext:
    return HFGPTQModelLoadContext(
        quantize_config=QuantizeConfig(bits=4, group_size=128),
        quant_linear=TorchLinear,
    )


def _stub_native_post_init(monkeypatch) -> None:
    """Skip the kernel-building steps; only the fix-up wiring is under test."""

    import gptqmodel.utils.model as model_utils

    monkeypatch.setattr(
        model_utils,
        "hf_convert_gptq_v1_to_v2_format",
        lambda model, **kwargs: (model, False),
    )
    monkeypatch.setattr(model_utils, "gptqmodel_post_init", lambda model, **kwargs: model)


def test_bridge_runs_the_legacy_remote_code_fixes(monkeypatch):
    _stub_native_post_init(monkeypatch)
    calls = []
    monkeypatch.setattr(
        structure_utils,
        "restore_init_owned_rotary_buffers",
        lambda model: calls.append(("rotary", model)) or 2,
    )
    monkeypatch.setattr(
        hf_utils,
        "install_legacy_generation_compat",
        lambda model: calls.append(("generation", model)) or True,
    )

    model = _bridge_model()

    assert hf_gptqmodel_post_init_for_load(model, _context()) is model
    assert [name for name, _target in calls] == ["rotary", "generation"]
    assert all(target is model for _name, target in calls)


def test_helper_restores_a_corrupted_rotary_buffer(monkeypatch):
    monkeypatch.setattr(hf_utils, "install_legacy_generation_compat", lambda model: False)
    model = _LegacyRemoteModel(dim=8)
    expected = model.rotary_emb.inv_freq.clone()
    # transformers 5 materializes checkpoint-absent non-persistent buffers with
    # `torch.empty_like`; the contents are whatever was in memory.
    model.rotary_emb.inv_freq.fill_(float("nan"))

    restored, installed = hf_utils.apply_legacy_remote_code_fixups(model)

    assert restored == 1
    assert installed is False
    assert torch.equal(model.rotary_emb.inv_freq, expected)


def test_helper_reports_the_generation_adapter(monkeypatch):
    monkeypatch.setattr(structure_utils, "restore_init_owned_rotary_buffers", lambda model: 0)
    monkeypatch.setattr(hf_utils, "install_legacy_generation_compat", lambda model: True)

    assert hf_utils.apply_legacy_remote_code_fixups(_LegacyRemoteModel()) == (0, True)


def test_both_load_paths_call_the_shared_helper():
    for relative in ("gptqmodel/models/base.py", "gptqmodel/utils/hf_load.py"):
        tree = ast.parse((REPO_ROOT / relative).read_text(encoding="utf-8"))
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and getattr(node.func, "id", None) == "apply_legacy_remote_code_fixups"
        ]
        assert calls, f"{relative} does not run the shared post-load fix-ups (#5311)"
