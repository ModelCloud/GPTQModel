# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for #4368.

``bind_private_offload_root`` gives every quantization run its own
``run-<uuid>`` directory so two models sharing a config/path cannot overwrite
each other's bundles (#4202). Nothing removed that directory afterwards, so an
explicit ``offload_to_disk_path`` grew by one full model per run, across
processes.

The bundles are the live storage for the quantized modules: a forward after
``save()`` still materializes weights from them. Cleanup therefore happens when
the model is collected (or the interpreter exits), when the run fails, and
before the same model starts another run.
"""

from __future__ import annotations

import gc
import os
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import LlamaConfig, PreTrainedModel

from gptqmodel.looper.module_looper import ModuleLooper
from gptqmodel.looper.weight_only_looper import WeightOnlyLooper
from gptqmodel.models.definitions.llama import LlamaQModel
from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.quantization import QuantizeConfig
from gptqmodel.quantization.config import QuantizeEmbed, QuantizeEmbedConfig, RTNConfig
from gptqmodel.utils.offload import (
    bind_private_offload_root,
    offload_to_disk,
    release_private_offload_root,
)


class _TinyModel(PreTrainedModel):
    def __init__(self) -> None:
        super().__init__(LlamaConfig(num_hidden_layers=1, hidden_size=32))
        self.norm = torch.nn.LayerNorm(32)
        self.proj = TorchLinear(
            bits=4,
            group_size=32,
            sym=True,
            desc_act=False,
            in_features=32,
            out_features=32,
            bias=False,
        )


def _build_writer(tmp_path: Path, offload_root: Path, value: float) -> LlamaQModel:
    source = tmp_path / "source"
    source.mkdir(exist_ok=True)
    model = _TinyModel()
    model.config.save_pretrained(source)
    save_file({"norm.weight": model.norm.weight.detach()}, source / "model.safetensors")
    with torch.no_grad():
        model.proj.qweight.fill_(value)

    return LlamaQModel(
        model=model,
        quantized=True,
        quantize_config=QuantizeConfig(
            group_size=32,
            offload_to_disk=True,
            offload_to_disk_path=str(offload_root),
        ),
        qlinear_kernel=TorchLinear,
        model_local_path=str(source),
    )


def _saved_tensor(save_dir: Path, name: str) -> torch.Tensor:
    for shard in sorted(save_dir.glob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as handle:
            if name in handle.keys():
                return handle.get_tensor(name)
    raise AssertionError(f"{name} was not written to {save_dir}")


def test_release_private_offload_root_removes_bundles(tmp_path):
    offload_root = tmp_path / "offload"
    writer = _build_writer(tmp_path, offload_root, value=1.0)
    run_root = bind_private_offload_root(writer)
    assert run_root is not None and os.path.isdir(run_root)

    offload_to_disk(
        model=writer.model,
        module=["proj"],
        disk_path=run_root,
        force=True,
    )
    assert (Path(run_root) / "proj" / "module.safetensors").is_file()

    release_private_offload_root(writer)
    assert not os.path.exists(run_root)
    assert list(offload_root.iterdir()) == []


def test_run_directory_survives_save_and_is_removed_at_teardown(tmp_path):
    offload_root = tmp_path / "offload"
    save_dir = tmp_path / "saved"

    writer = _build_writer(tmp_path, offload_root, value=3.0)
    run_root = bind_private_offload_root(writer)
    assert run_root is not None
    expected = writer.model.proj.qweight.detach().clone()
    offload_to_disk(
        model=writer.model,
        module=["proj"],
        disk_path=run_root,
        force=True,
    )

    writer.save(str(save_dir))

    # The saved checkpoint resolved the bundle from the private run directory,
    # and that directory still backs the in-session (meta) module afterwards.
    torch.testing.assert_close(_saved_tensor(save_dir, "proj.qweight"), expected)
    assert os.path.isdir(run_root)
    assert writer.model.proj.qweight.device.type == "meta"

    del writer
    gc.collect()

    assert not os.path.exists(run_root)
    assert list(offload_root.iterdir()) == []


def test_sequential_runs_share_path_without_accumulating(tmp_path):
    offload_root = tmp_path / "shared-offload"

    for index, value in enumerate((1.0, 2.0)):
        writer = _build_writer(tmp_path, offload_root, value=value)
        run_root = bind_private_offload_root(writer)
        assert run_root is not None
        expected = writer.model.proj.qweight.detach().clone()
        offload_to_disk(
            model=writer.model,
            module=["proj"],
            disk_path=run_root,
            force=True,
        )
        save_dir = tmp_path / f"saved-{index}"
        writer.save(str(save_dir))
        torch.testing.assert_close(
            _saved_tensor(save_dir, "proj.qweight"),
            expected,
        )
        assert os.path.isdir(run_root)

        del writer
        gc.collect()
        assert not os.path.exists(run_root)
        assert list(offload_root.iterdir()) == []


class _FakeModel:
    def __init__(self, qcfg):
        self.model = torch.nn.Module()
        self.model.config = type("Config", (), {"use_cache": True})()
        self.model.layers = torch.nn.ModuleList([torch.nn.Linear(4, 4, bias=False)])
        self.model.embed_tokens = torch.nn.Embedding(8, 4)
        self.model.lm_head = torch.nn.Linear(4, 8, bias=False)
        self.quantize_config = qcfg
        self.layer_modules_strict = True
        self.lm_head = "lm_head"
        self.tokenizer = None
        self.quant_log = None
        self.support_batch_quantize = False
        self.layer_callback = None

    def extract_layers_node(self):
        return ["layers"]

    def get_modules_with_direct_meta_tensors(self, _model):
        return []

    def simple_layer_modules(self, **_kwargs):
        return [["linear"]]

    def pre_quantize(self, module):
        return module

    def post_quantize(self, module):
        return module

    def get_input_embeddings_name(self):
        return "embed_tokens"

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def get_output_embeddings_name(self):
        return "lm_head"

    def get_output_embeddings(self):
        return self.model.lm_head


class _FakeProcessor:
    def __init__(self, qcfg):
        self.qcfg = qcfg

    def name(self):
        return "fake_weight_only"

    def is_skipped(self, module):
        return False


def test_run_directory_is_removed_when_weight_only_run_fails(tmp_path):
    offload_root = tmp_path / "offload"
    qcfg = RTNConfig(
        bits=4,
        group_size=4,
        offload_to_disk=True,
        offload_to_disk_path=str(offload_root),
        device="cpu",
    )
    model = _FakeModel(qcfg)
    # Tied embeddings are rejected inside the loop, after the run directory has
    # been bound.
    model.model.lm_head.weight = model.model.embed_tokens.weight

    looper = WeightOnlyLooper(
        model=model,
        processor=_FakeProcessor(qcfg),
        embed_quant_config=QuantizeEmbedConfig(embed_quant_mode=QuantizeEmbed.BOTH),
    )
    run_root = model.quantize_config.offload_to_disk_path
    assert Path(run_root).parent == offload_root
    assert os.path.isdir(run_root)

    with pytest.raises(NotImplementedError):
        looper.loop()

    assert not os.path.exists(run_root)
    assert list(offload_root.iterdir()) == []


def test_run_directory_is_removed_when_module_looper_fails(tmp_path, monkeypatch):
    offload_root = tmp_path / "offload"
    qcfg = QuantizeConfig(
        group_size=32,
        offload_to_disk=True,
        offload_to_disk_path=str(offload_root),
    )
    model = _FakeModel(qcfg)
    looper = ModuleLooper(model, processors=[])
    run_root = model.quantize_config.offload_to_disk_path
    assert os.path.isdir(run_root)

    def _boom(**_kwargs):
        raise RuntimeError("forced quantization failure")

    monkeypatch.setattr(looper, "_loop_impl", _boom)
    with pytest.raises(RuntimeError, match="forced quantization failure"):
        looper.loop()

    assert not os.path.exists(run_root)
    assert list(offload_root.iterdir()) == []


def test_rebinding_same_model_releases_previous_run(tmp_path):
    offload_root = tmp_path / "offload"
    writer = _build_writer(tmp_path, offload_root, value=1.0)

    first = bind_private_offload_root(writer)
    second = bind_private_offload_root(writer)

    assert first is not None and second is not None and first != second
    assert not os.path.exists(first)
    assert os.path.isdir(second)
    assert [entry.name for entry in offload_root.iterdir()] == [Path(second).name]
