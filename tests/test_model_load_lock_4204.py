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
import sysconfig
import threading
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from gptqmodel import BACKEND, GPTQModel
from gptqmodel.models import loader as loader_module
from gptqmodel.models.base import BaseQModel
from gptqmodel.models.ports.native.llama.configuration_llama import LlamaConfig
from gptqmodel.models.ports.native.llama.modeling_llama import LlamaForCausalLM
from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.quantization import RTNConfig
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


class _SignallingLock:
    """Re-entrant lock that records when another thread has to wait for it.

    The compiled-call side asks for ``reader()``; this stand-in grants it the
    same exclusive lock so a test can still observe the wait. The real
    ``_ModelGlobalStateLock`` grants shared readers.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self.waited = threading.Event()

    def acquire(self, blocking=True, timeout=-1):
        if not self._lock.acquire(blocking=False):
            self.waited.set()
            return self._lock.acquire(blocking, timeout)
        return True

    def release(self):
        self._lock.release()

    def __enter__(self):
        self.acquire()
        return self

    def reader(self):
        return self

    def holds_reader(self) -> bool:
        return False

    def __exit__(self, *_exc):
        self.release()
        return False


def _write_tiny_rtn_checkpoint(tmp_path: Path) -> Path:
    """Build a one-layer TorchLinear checkpoint so a real load can run."""

    source = tmp_path / "tiny-dense"
    checkpoint = tmp_path / "tiny-rtn"
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=32,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
    )
    torch.manual_seed(0)
    LlamaForCausalLM(config).half().eval().save_pretrained(str(source))
    PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            WordLevel({"[UNK]": 0, "[BOS]": 1, "[EOS]": 2}, unk_token="[UNK]")
        ),
        unk_token="[UNK]",
        bos_token="[BOS]",
        eos_token="[EOS]",
    ).save_pretrained(str(source))

    qcfg = RTNConfig(bits=4, group_size=32, offload_to_disk=False, device="cpu")
    model = GPTQModel.load(
        str(source), quantize_config=qcfg, backend=BACKEND.GPTQ_TORCH
    )
    model.quantize(backend=BACKEND.GPTQ_TORCH)
    model.save(str(checkpoint))
    return checkpoint


@pytest.mark.skipif(
    sysconfig.get_config_var("Py_GIL_DISABLED") != 1,
    reason="the compile/load default-dtype race is qualified on free-threaded Python",
)
def test_first_compile_cannot_restore_dtype_over_a_concurrent_load(tmp_path, monkeypatch):
    """A first TorchLinear compile must not save/restore dtype across a load (#5127).

    The load thread is held inside transformers' bf16 window while holding
    ``_MODEL_LOAD_LOCK``. The forward thread then triggers dynamo's lazy first
    trace, whose ``preserve_global_state`` saves bf16. If that trace is not
    serialized with the load, it restores bf16 *after* the load restored fp32,
    leaving the process default dtype wrong. The barriers make the interleaving
    deterministic instead of relying on the race window.
    """

    dynamo = pytest.importorskip("torch._dynamo")
    checkpoint = _write_tiny_rtn_checkpoint(tmp_path)
    warm = GPTQModel.load(
        str(checkpoint),
        device="cpu",
        dtype=torch.bfloat16,
        backend=BACKEND.GPTQ_TORCH,
    )
    compiled_modules = []
    for module in warm.model.modules():
        if not isinstance(module, TorchLinear):
            continue
        # Eager backend keeps the trace fast; dynamo still saves/restores the
        # process-global default dtype around it.
        module.optimized = False
        module.optimize(backend="eager")
        compiled = getattr(module.dequantize_weight, "_compiled", None)
        if getattr(compiled, "_torchdynamo_orig_callable", None) is None:
            pytest.skip("torch.compile is unavailable on this interpreter/torch build")
        compiled_modules.append(module)
    assert compiled_modules, "the checkpoint has no TorchLinear modules"
    dynamo.reset()

    original_dtype = torch.float32
    torch.set_default_dtype(original_dtype)

    lock = _SignallingLock()
    monkeypatch.setattr(loader_module, "_MODEL_GLOBAL_STATE_LOCK", lock)

    load_bf16 = threading.Event()
    compile_restore = threading.Event()
    release_load = threading.Event()
    release_compile = threading.Event()
    load_done = threading.Event()
    load_ident = {}
    forward_ident = {}
    load_errors = []
    forward_errors = []

    real_set_default_dtype = torch.set_default_dtype

    def guarded_set_default_dtype(dtype):
        real_set_default_dtype(dtype)
        ident = threading.get_ident()
        if (
            ident == load_ident.get("id")
            and dtype == torch.bfloat16
            and not load_bf16.is_set()
        ):
            # Hold the load inside transformers' bf16 window while it owns
            # the model-load lock.
            load_bf16.set()
            release_load.wait(timeout=60)
        elif (
            ident == forward_ident.get("id")
            and dtype == torch.bfloat16
            and load_bf16.is_set()
            and not load_done.is_set()
        ):
            # Dynamo is about to restore the dtype it captured before tracing.
            compile_restore.set()
            release_compile.wait(timeout=60)

    monkeypatch.setattr(torch, "set_default_dtype", guarded_set_default_dtype)

    def _load():
        load_ident["id"] = threading.get_ident()
        try:
            GPTQModel.load(
                str(checkpoint),
                device="cpu",
                dtype=torch.bfloat16,
                backend=BACKEND.GPTQ_TORCH,
            )
        except BaseException as exc:  # noqa: BLE001 - reported below
            load_errors.append(exc)
        finally:
            load_done.set()

    ids = torch.tensor([[1, 2, 3, 4]])

    def _first_forward():
        forward_ident["id"] = threading.get_ident()
        try:
            with torch.inference_mode():
                warm.model(ids)
        except BaseException as exc:  # noqa: BLE001 - reported below
            forward_errors.append(exc)

    load_thread = threading.Thread(target=_load, daemon=True)
    load_thread.start()
    assert load_bf16.wait(timeout=60), "the load never entered its bf16 window"

    forward_thread = threading.Thread(target=_first_forward, daemon=True)
    forward_thread.start()

    # Pre-fix the compile reaches its dtype restore while the load is still
    # blocked; post-fix it waits for the load lock and signals ``waited``.
    saw_restore = False
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if compile_restore.wait(timeout=0.05):
            saw_restore = True
            break
        if lock.waited.wait(timeout=0.05):
            break

    release_load.set()
    load_thread.join(timeout=60)
    assert not load_thread.is_alive(), "the load thread did not finish"
    if saw_restore:
        release_compile.set()
    forward_thread.join(timeout=60)
    assert not forward_thread.is_alive(), "the forward thread did not finish"

    assert torch.get_default_dtype() == original_dtype, (
        "a concurrent first compile restored its bf16 snapshot after the load "
        f"restored fp32: {torch.get_default_dtype()}"
    )
    assert not load_errors, f"concurrent load failed: {load_errors!r}"
    assert not forward_errors, f"concurrent first forward failed: {forward_errors!r}"
