# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Every model-construction site must take the shared model-load lock (#4204).

transformers' ``from_config`` wraps construction in ``local_torch_dtype``, a
process-global ``torch.set_default_dtype``. Issue #4129 serialized the
``from_pretrained`` / ``from_quantized`` entry points; the save-side shell
rebuild (``BaseQModel.get_model_with_quantize``), the compressed-tensors dequant
path and the TT block-float ``_from_config`` path were still unlocked and could
reopen the race. These tests spy on the one shared lock object rather than
building real models.
"""

from __future__ import annotations

import json
import threading
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from gptqmodel.models import loader as loader_module
from gptqmodel.models.base import BaseQModel
from gptqmodel.utils import model_dequant
from gptqmodel.utils import tt_blockfloat_quant as tt_module


class _Stop(Exception):
    """Abort a construction site once it has reached ``from_config``."""


class _TrackingLock:
    """RLock stand-in that records every ``with`` entry and current depth."""

    def __init__(self):
        self._lock = threading.RLock()
        self.enter_count = 0
        self.depth = 0

    def __enter__(self):
        self._lock.acquire()
        self.enter_count += 1
        self.depth += 1
        return self

    def __exit__(self, *_exc):
        self.depth -= 1
        self._lock.release()
        return False

    def held(self) -> bool:
        return self.depth > 0


def _write_tiny_config(tmp_path: Path) -> Path:
    """A local config-only checkpoint: AutoConfig can parse it, nothing loads."""

    checkpoint = tmp_path / "tiny-checkpoint"
    checkpoint.mkdir()
    config = {
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 8,
        "intermediate_size": 8,
        "num_hidden_layers": 1,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "vocab_size": 8,
    }
    (checkpoint / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return checkpoint


def test_save_shell_build_enters_model_load_lock(tmp_path, monkeypatch):
    checkpoint = _write_tiny_config(tmp_path)
    lock = _TrackingLock()
    monkeypatch.setattr(loader_module, "_MODEL_LOAD_LOCK", lock)
    observed = []

    class _Loader:
        def from_config(self, *_args, **_kwargs):
            observed.append(lock.held())
            raise _Stop

    owner = SimpleNamespace(trust_remote_code=False, loader=_Loader())

    with pytest.raises(_Stop):
        BaseQModel.get_model_with_quantize(
            owner, qcfg=None, model_id_or_path=str(checkpoint)
        )

    assert lock.enter_count == 1
    assert observed == [True], "save-side from_config ran without the load lock"


def test_dequant_from_config_enters_model_load_lock(monkeypatch):
    lock = _TrackingLock()
    monkeypatch.setattr(loader_module, "_MODEL_LOAD_LOCK", lock)
    monkeypatch.setattr(
        "gptqmodel.utils.hf.load_model_config",
        lambda *_args, **_kwargs: SimpleNamespace(model_type="lock_fixture"),
    )
    observed = []

    class _Loader:
        __name__ = "FakeAutoModel"

        def from_config(self, *_args, **_kwargs):
            observed.append(lock.held())
            return SimpleNamespace()

    deps = {
        "init_empty_weights": lambda **_kwargs: nullcontext(),
        "AutoModelForCausalLM": _Loader(),
        "AutoModelForSeq2SeqLM": None,
        "AutoModel": None,
        "apply_quantization_config": lambda *_args, **_kwargs: None,
        "map_module_to_scheme": lambda _model: {},
    }

    schemes = model_dequant._discover_compressed_tensors_module_schemes(
        Path("unused"), {}, deps=deps
    )

    assert schemes == {}
    assert lock.enter_count == 1
    assert observed == [True], "dequant from_config ran without the load lock"


def test_tt_blockfloat_from_config_enters_model_load_lock(tmp_path, monkeypatch):
    checkpoint = _write_tiny_config(tmp_path)
    lock = _TrackingLock()
    monkeypatch.setattr(loader_module, "_MODEL_LOAD_LOCK", lock)
    monkeypatch.setattr(
        tt_module,
        "load_hf_state_dict",
        lambda _dir: {"model.layers.0.mlp.down_proj.codes": torch.zeros(1, dtype=torch.uint8)},
    )
    monkeypatch.setattr(
        tt_module,
        "_resolve_layer_widths",
        lambda _state, _quant_cfg: {"model.layers.0.mlp.down_proj": 4},
    )
    observed = []

    def _fake_from_config(cls, config, **_kwargs):
        observed.append(lock.held())
        raise _Stop

    from gptqmodel.models.ports.native.llama.modeling_llama import LlamaForCausalLM

    monkeypatch.setattr(LlamaForCausalLM, "_from_config", classmethod(_fake_from_config))

    with pytest.raises(_Stop):
        tt_module.build_blockfloat_model(str(checkpoint), dtype=torch.float32)

    assert lock.enter_count == 1
    assert observed == [True], "TT block-float _from_config ran without the load lock"


def test_concurrent_from_config_cannot_change_default_dtype(tmp_path):
    """A save-side shell build must not leak bf16 into a concurrent load."""

    checkpoint = _write_tiny_config(tmp_path)
    original = torch.get_default_dtype()
    in_window = threading.Event()
    release = threading.Event()
    builds = []

    class _Loader:
        def from_config(self, *_args, **_kwargs):
            # Mimic transformers' local_torch_dtype: set the process-global
            # default dtype, construct, then restore it.
            builds.append(True)
            torch.set_default_dtype(torch.bfloat16)
            in_window.set()
            try:
                release.wait(timeout=10)
            finally:
                torch.set_default_dtype(original)
            raise _Stop

    owner = SimpleNamespace(trust_remote_code=False, loader=_Loader())

    def _save_side_shell():
        try:
            BaseQModel.get_model_with_quantize(
                owner, qcfg=None, model_id_or_path=str(checkpoint)
            )
        except _Stop:
            pass

    @loader_module._serialized_model_load
    def _load_observe_default_dtype():
        return torch.get_default_dtype()

    observed = {}
    load_done = threading.Event()

    def _load():
        observed["dtype"] = _load_observe_default_dtype()
        load_done.set()

    shell_thread = threading.Thread(target=_save_side_shell, daemon=True)
    shell_thread.start()
    assert in_window.wait(timeout=10), "the save-side shell build never ran"
    assert torch.get_default_dtype() == torch.bfloat16

    load_thread = threading.Thread(target=_load, daemon=True)
    load_thread.start()
    # The load must wait for the in-flight construction instead of observing
    # the bf16 window. Pre-fix it returns immediately with bf16.
    waited = not load_done.wait(timeout=0.25)
    release.set()
    load_thread.join(timeout=10)
    shell_thread.join(timeout=10)

    assert builds, "the save-side shell build never ran"
    assert waited, "a load did not wait for the in-flight save-side construction"
    assert observed == {"dtype": original}, (
        f"a load observed a default dtype changed by the save-side from_config: {observed}"
    )
    assert torch.get_default_dtype() == original
