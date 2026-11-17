# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for #4202.

Offload bundles used to be keyed by module name alone in a directory shared by
every copy of a ``QuantizeConfig``. A second quantization run overwrote the
first model's bundles and its ``save()`` wrote the other model's weights. Each
run must get a private namespace that both the writer and the save-time reader
agree on.
"""

import gc
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from torch import nn

from gptqmodel.looper.module_looper import ModuleLooper
from gptqmodel.quantization import QuantizeConfig
from gptqmodel.utils.model import OffloadTensorRef, get_state_dict_for_save
from gptqmodel.utils.offload import (
    bind_private_offload_root,
    offload_to_disk,
    release_private_offload_root,
)


class _TinyModel(nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Module()])
        self.model.layers[0].linear = nn.Linear(8, 8, bias=False)
        with torch.no_grad():
            self.model.layers[0].linear.weight.fill_(value)


def _dummy_qmodel(config):
    return SimpleNamespace(
        support_batch_quantize=False,
        quantize_config=config,
        layer_callback=None,
    )


def _shared_config(tmp_path) -> QuantizeConfig:
    return QuantizeConfig(
        bits=4,
        group_size=32,
        offload_to_disk=True,
        offload_to_disk_path=str(tmp_path / "shared-offload"),
        device="cpu",
    )


def _resolved_weight(model, config) -> torch.Tensor:
    state = get_state_dict_for_save(
        model, offload_root=config.offload_to_disk_path
    )
    source = state["model.layers.0.linear.weight"].source
    assert isinstance(source, OffloadTensorRef)
    with safe_open(source.path, framework="pt", device="cpu") as handle:
        return handle.get_tensor(source.weight_name)


def test_bind_private_offload_root_isolates_shared_config(tmp_path):
    shared = _shared_config(tmp_path)
    first = _dummy_qmodel(shared)
    second = _dummy_qmodel(shared)

    first_root = bind_private_offload_root(first)
    second_root = bind_private_offload_root(second)

    assert first_root is not None and second_root is not None
    assert first_root != second_root
    assert Path(first_root).parent == Path(second_root).parent == tmp_path / "shared-offload"
    # Each model keeps its own config copy; the caller's object is untouched.
    assert first.quantize_config is not second.quantize_config
    assert first.quantize_config is not shared
    assert shared.offload_to_disk_path == str(tmp_path / "shared-offload")
    assert first.quantize_config.offload_to_disk_path == first_root
    assert second.quantize_config.offload_to_disk_path == second_root


def test_module_looper_rebinds_processor_to_private_config(tmp_path):
    shared = _shared_config(tmp_path)
    model = _dummy_qmodel(shared)
    processor = SimpleNamespace(qcfg=shared)

    ModuleLooper(model, processors=[processor])

    assert processor.qcfg is model.quantize_config
    assert processor.qcfg is not shared
    assert processor.qcfg.offload_to_disk_path.startswith(str(tmp_path / "shared-offload"))


def test_offload_bundles_do_not_cross_run_namespaces(tmp_path):
    shared = _shared_config(tmp_path)
    first_model = _TinyModel(1.0)
    second_model = _TinyModel(2.0)
    first = _dummy_qmodel(shared)
    second = _dummy_qmodel(shared)

    first_root = bind_private_offload_root(first)
    second_root = bind_private_offload_root(second)
    for holder, model in ((first, first_model), (second, second_model)):
        offload_to_disk(
            model=model,
            module=["model.layers.0.linear"],
            disk_path=holder.quantize_config.offload_to_disk_path,
            force=True,
        )
        assert os.path.isfile(
            os.path.join(
                holder.quantize_config.offload_to_disk_path,
                "model.layers.0.linear",
                "module.safetensors",
            )
        )

    assert first_root != second_root
    torch.testing.assert_close(
        _resolved_weight(first_model, first.quantize_config),
        torch.ones(8, 8),
    )
    torch.testing.assert_close(
        _resolved_weight(second_model, second.quantize_config),
        torch.full((8, 8), 2.0),
    )


def test_release_private_offload_root_removes_explicit_run_dir_only(tmp_path):
    shared = _shared_config(tmp_path)
    model = _dummy_qmodel(shared)
    base = Path(shared.offload_to_disk_path)
    run_root = Path(bind_private_offload_root(model))
    sibling = base / "run-sibling"
    sibling.mkdir()
    (run_root / "model.layers.0.linear").mkdir(parents=True)

    released = release_private_offload_root(model)

    assert released == str(run_root)
    assert not run_root.exists()
    assert sibling.is_dir()
    assert base.is_dir()


def test_release_private_offload_root_removes_auto_temp_run_dir(tmp_path):
    config = QuantizeConfig(bits=4, group_size=32, offload_to_disk=True, device="cpu")
    base = Path(config.offload_to_disk_path)
    model = _dummy_qmodel(config)

    run_root = Path(bind_private_offload_root(model))
    assert run_root.parent == base
    assert run_root.is_dir()

    release_private_offload_root(model)

    assert not run_root.exists()
    assert base.is_dir()


def test_private_offload_run_dir_is_removed_when_model_is_collected(tmp_path):
    shared = _shared_config(tmp_path)
    model = _TinyModel(1.0)
    model.quantize_config = shared

    run_root = Path(bind_private_offload_root(model))
    assert run_root.is_dir()

    del model
    gc.collect()

    assert not run_root.exists()


def test_rebinding_private_offload_root_releases_the_previous_run_dir(tmp_path):
    shared = _shared_config(tmp_path)
    model = _dummy_qmodel(shared)
    first_root = Path(bind_private_offload_root(model))
    assert first_root.is_dir()

    second_root = Path(bind_private_offload_root(model))

    assert second_root != first_root
    assert not first_root.exists()
    assert second_root.is_dir()
    assert model.quantize_config.offload_to_disk_path == str(second_root)


def test_module_looper_failure_releases_private_run_dir(tmp_path, monkeypatch):
    shared = _shared_config(tmp_path)
    model = _dummy_qmodel(shared)
    released = []
    # ``loop()`` releases every processor's run workspace on failure too
    # (``LoopProcessor.release_run_workspace``), so the stub needs the hook.
    processor = SimpleNamespace(qcfg=shared, release_run_workspace=lambda: released.append(True))
    looper = ModuleLooper(model, processors=[processor])
    run_root = Path(model.quantize_config.offload_to_disk_path)
    assert run_root.is_dir()

    def fail(**_kwargs):
        raise RuntimeError("quantization failed")

    monkeypatch.setattr(looper, "_loop_impl", fail)

    with pytest.raises(RuntimeError, match="quantization failed"):
        looper.loop()

    assert not run_root.exists()
    assert released == [True]


def test_base_model_save_keeps_private_run_dir_for_in_session_use(tmp_path):
    """#4368: save() must not delete bundles the live model still reads."""
    from gptqmodel.models.base import BaseQModel

    class _SaveModel(BaseQModel):
        def __init__(self, config):
            nn.Module.__init__(self)
            self.quantized = True
            self.quantize_config = config
            self.quant_override_files = {}
            self._model_free_weight_only_replacements_only = False

        def _warn_fused_unit_dynamic_overrides(self):
            return None

        def save_quantized(self, **_kwargs):
            return None

    shared = _shared_config(tmp_path)
    model = _SaveModel(shared)
    run_root = Path(bind_private_offload_root(model))
    assert run_root.is_dir()
    (run_root / "model.layers.0.linear").mkdir(parents=True)

    model.save(str(tmp_path / "saved"))

    # The quantized modules stay meta-backed by this directory, so an in-session
    # forward (and a second save) still resolves its bundles.
    assert run_root.is_dir()
    assert (run_root / "model.layers.0.linear").is_dir()

    model.save(str(tmp_path / "saved-again"))

    assert run_root.is_dir()

    del model
    gc.collect()

    assert not run_root.exists()
