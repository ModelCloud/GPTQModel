# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for #5348: name lookups must not scan the model per linear.

Disk offload and shell materialization each recovered a module's dotted name by
walking ``named_modules()`` from the root.  That ran three times per quantized
linear, so per-layer cost grew with model depth and expert count.  Callers that
already hold the name must pass it through; the full scan stays only as a
fallback for stale or missing names.
"""

from types import SimpleNamespace

import torch
from tokenizers import Tokenizer, models
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

import gptqmodel.models.base as base_mod
import gptqmodel.utils.offload as offload_mod
import gptqmodel.utils.structure as structure_mod
from gptqmodel import BACKEND, CalibrationConfig, GPTQModel, QuantizeConfig
from gptqmodel.looper.named_module import NamedModule
from gptqmodel.models.base import BaseQModel
from gptqmodel.quantization.config import AutoDtypeCast
from gptqmodel.utils.structure import _resolve_qualified_name


class _NestedModel(torch.nn.Module):
    def __init__(self, layers: int = 3) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [torch.nn.Linear(4, 4, bias=False) for _ in range(layers)]
        )


def _record_offload(monkeypatch):
    recorded = {}

    def fake_offload_disk(*, module, name, **kwargs):
        recorded["module"] = module
        recorded["name"] = name

    monkeypatch.setattr(offload_mod, "_offload_disk", fake_offload_disk)
    return recorded


def test_offload_to_disk_uses_known_module_name(monkeypatch, tmp_path):
    model = _NestedModel()
    recorded = _record_offload(monkeypatch)

    def unexpected_scan(*args, **kwargs):
        raise AssertionError("offload scanned named_modules() despite a known name")

    monkeypatch.setattr(offload_mod, "get_module_fullname", unexpected_scan)

    offload_mod.offload_to_disk(
        model=model,
        module=model.layers[1],
        module_name="layers.1",
        disk_path=str(tmp_path),
    )

    assert recorded == {"module": model.layers[1], "name": "layers.1"}


def test_offload_to_disk_uses_named_module_full_name(monkeypatch, tmp_path):
    model = _NestedModel()
    named = NamedModule(
        model.layers[2],
        name="2",
        full_name="layers.2",
        layer_index=2,
    )
    recorded = _record_offload(monkeypatch)

    def unexpected_scan(*args, **kwargs):
        raise AssertionError("offload scanned named_modules() despite NamedModule.full_name")

    monkeypatch.setattr(offload_mod, "get_module_fullname", unexpected_scan)

    offload_mod.offload_to_disk(
        model=model,
        module=named,
        disk_path=str(tmp_path),
    )

    assert recorded == {"module": model.layers[2], "name": "layers.2"}


def test_offload_to_disk_falls_back_for_stale_name(monkeypatch, tmp_path):
    model = _NestedModel()
    recorded = _record_offload(monkeypatch)
    scans = []

    def counted_scan(*, model, module):
        scans.append(module)
        return "layers.1"

    monkeypatch.setattr(offload_mod, "get_module_fullname", counted_scan)

    offload_mod.offload_to_disk(
        model=model,
        module=model.layers[1],
        module_name="layers.2",
        disk_path=str(tmp_path),
    )

    assert scans == [model.layers[1]]
    assert recorded["name"] == "layers.1"


def test_resolve_qualified_name_uses_known_path(monkeypatch):
    model = _NestedModel()

    def unexpected_scan(*args, **kwargs):
        raise AssertionError("resolved a valid path with a full named_modules() scan")

    monkeypatch.setattr(structure_mod, "_get_qualified_name", unexpected_scan)

    assert _resolve_qualified_name(model, model.layers[1], "layers.1") == "layers.1"


def test_resolve_qualified_name_falls_back_for_stale_path(monkeypatch):
    model = _NestedModel()
    scans = []
    original = structure_mod._get_qualified_name

    def counted_scan(root, obj):
        scans.append(obj)
        return original(root, obj)

    monkeypatch.setattr(structure_mod, "_get_qualified_name", counted_scan)

    assert _resolve_qualified_name(model, model.layers[1], "layers.2") == "layers.1"
    assert scans == [model.layers[1]]


def test_replace_live_submodule_uses_known_path(monkeypatch):
    model = _NestedModel()
    replacement = torch.nn.Linear(4, 4, bias=False)

    def unexpected_scan(*args, **kwargs):
        raise AssertionError("replacement scanned named_modules() despite a known path")

    monkeypatch.setattr(structure_mod, "_get_qualified_name", unexpected_scan)

    BaseQModel._replace_live_submodule(
        SimpleNamespace(model=model),
        model.layers[1],
        replacement,
        module_path="layers.1",
    )

    assert model.layers[1] is replacement


def _build_tiny_llama_source(source) -> None:
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=32,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    config._attn_implementation = "eager"
    model = LlamaForCausalLM(config).eval()
    model.save_pretrained(source)

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            models.WordLevel({"[PAD]": 0, "[UNK]": 1, "a": 2}, unk_token="[UNK]")
        ),
        pad_token="[PAD]",
        unk_token="[UNK]",
    )
    tokenizer.save_pretrained(source)


def test_quantize_does_not_scan_named_modules_per_linear(monkeypatch, tmp_path):
    """A tiny disk-offloaded quantization must not pay one full-tree scan per linear."""

    scans = {"offload": 0, "qualified": 0}
    original_offload_scan = offload_mod.get_module_fullname
    original_qualified_scan = structure_mod._get_qualified_name

    def counted_offload_scan(*args, **kwargs):
        scans["offload"] += 1
        return original_offload_scan(*args, **kwargs)

    def counted_qualified_scan(*args, **kwargs):
        scans["qualified"] += 1
        return original_qualified_scan(*args, **kwargs)

    monkeypatch.setattr(offload_mod, "get_module_fullname", counted_offload_scan)
    monkeypatch.setattr(structure_mod, "_get_qualified_name", counted_qualified_scan)
    monkeypatch.setattr(base_mod, "_get_qualified_name", counted_qualified_scan)

    source = tmp_path / "source"
    _build_tiny_llama_source(source)
    qcfg = QuantizeConfig(
        bits=4,
        group_size=32,
        desc_act=False,
        device="cpu",
        offload_to_disk=True,
        offload_to_disk_path=str(tmp_path / "offload"),
        auto_dtype_cast=AutoDtypeCast.OFF,
    )
    qmodel = GPTQModel.load(str(source), quantize_config=qcfg, dtype=torch.float16)
    qmodel.quantize(
        CalibrationConfig(
            dataset=[
                {
                    "input_ids": [1, 3, 5, 7, 9, 11, 13, 15],
                    "attention_mask": [1] * 8,
                }
            ],
            batch_size=1,
            min_length=1,
        ),
        backend=BACKEND.GPTQ_TORCH,
    )

    assert scans["offload"] == 0, (
        "disk offload recovered names with full-tree scans: "
        f"{scans['offload']} calls"
    )
    # Endpoint modules may still fall back once; the count must not scale with
    # the number of quantized linears (14 in this model).
    assert scans["qualified"] <= 2, (
        "shell materialization recovered names with full-tree scans: "
        f"{scans['qualified']} calls"
    )
