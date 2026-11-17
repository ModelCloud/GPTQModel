# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for #4368: per-run offload directories must not accumulate.

``bind_private_offload_root`` gives every quantization run its own
``run-<uuid>`` directory so two models sharing a config/path cannot overwrite
each other's bundles (#4202). Nothing removed that directory afterwards, so an
explicit ``offload_to_disk_path`` grew by one full model per run.

The bundles are the live storage for the quantized modules: a forward after
``save()`` still materializes weights from them. Cleanup therefore happens when
the model is collected, when the run fails, and before the same model starts
another run -- never at the end of a successful ``save()``.
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

from gptqmodel.models.definitions.llama import LlamaQModel
from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.quantization import QuantizeConfig
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


def _offload_proj(writer: LlamaQModel, run_root: str) -> torch.Tensor:
    expected = writer.model.proj.qweight.detach().clone()
    offload_to_disk(
        model=writer.model,
        module=["proj"],
        disk_path=run_root,
        force=True,
    )
    assert writer.model.proj.qweight.device.type == "meta"
    return expected


def test_run_directory_survives_save_and_is_removed_at_teardown(tmp_path):
    offload_root = tmp_path / "offload"
    save_dir = tmp_path / "saved"

    writer = _build_writer(tmp_path, offload_root, value=3.0)
    run_root = bind_private_offload_root(writer)
    assert run_root is not None
    expected = _offload_proj(writer, run_root)

    writer.save(str(save_dir))

    # The saved checkpoint resolved the bundle from the private run directory,
    # and that directory still backs the in-session (meta) module afterwards.
    torch.testing.assert_close(_saved_tensor(save_dir, "proj.qweight"), expected)
    assert os.path.isdir(run_root)
    assert writer.model.proj.qweight.device.type == "meta"

    # A second save must keep working: the bundles were not deleted.
    writer.save(str(tmp_path / "saved-again"))
    torch.testing.assert_close(
        _saved_tensor(tmp_path / "saved-again", "proj.qweight"),
        expected,
    )
    assert os.path.isdir(run_root)

    del writer
    gc.collect()

    assert not os.path.exists(run_root)
    assert list(offload_root.iterdir()) == []


def test_failed_save_keeps_run_directory_for_retry(tmp_path):
    offload_root = tmp_path / "offload"

    writer = _build_writer(tmp_path, offload_root, value=4.0)
    run_root = bind_private_offload_root(writer)
    assert run_root is not None
    expected = _offload_proj(writer, run_root)

    def _boom(**_kwargs):
        raise RuntimeError("disk full")

    original = writer.save_quantized
    writer.save_quantized = _boom
    with pytest.raises(RuntimeError, match="disk full"):
        writer.save(str(tmp_path / "saved"))
    assert os.path.isdir(run_root)

    # The retry after the transient failure still resolves the bundles.
    writer.save_quantized = original
    writer.save(str(tmp_path / "saved"))
    torch.testing.assert_close(
        _saved_tensor(tmp_path / "saved", "proj.qweight"),
        expected,
    )

    release_private_offload_root(writer)
    assert not os.path.exists(run_root)


def test_sequential_runs_share_path_without_accumulating(tmp_path):
    offload_root = tmp_path / "shared-offload"

    for index, value in enumerate((1.0, 2.0)):
        writer = _build_writer(tmp_path, offload_root, value=value)
        run_root = bind_private_offload_root(writer)
        assert run_root is not None
        expected = _offload_proj(writer, run_root)

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
