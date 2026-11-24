# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""transformers' class-wide construction patches may not cross threads (#5421).

transformers builds a model inside one init-context group: the default-dtype
window, a class-wide ``PreTrainedModel.tie_weights`` no-op
(``init.no_tie_weights``) and the registered class swaps (``apply_patches``).
Only the dtype window held the model-global-state writer, so a plain
``from_pretrained`` in another thread could disable tying while a GPT-QModel
load built or re-tied its shell. The load then failed with
``does not provide tensors for ... lm_head.weight``, or was rejected as a
corrupted state because the load's default-dtype check observed the other
thread's temporary dtype.

These tests pin the writer to every consumer of those patches: the init-context
group, ``from_config``, the patches themselves, the post-load tie, and the
default-dtype capture/check. They are deterministic -- the foreign thread holds
the patch window open until the test releases it -- so a broken guard fails
immediately instead of depending on a race.
"""

from __future__ import annotations

import threading
from contextlib import ExitStack
from types import SimpleNamespace

import torch
from transformers import AutoModelForCausalLM, Qwen2Config, Qwen2ForCausalLM, initialization, modeling_utils

import gptqmodel  # noqa: F401 - installs the import-time construction guards
from gptqmodel.models.loader import (
    _default_dtype_outside_construction,
    model_global_state_lock,
)
from gptqmodel.utils.hf import assert_loaded_model_state
from gptqmodel.utils.model import _tie_weights_after_checkpoint_load


def _tiny_config() -> Qwen2Config:
    return Qwen2Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        tie_word_embeddings=True,
    )


def _is_tied(model) -> bool:
    return model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()


def _wait_alive(thread: threading.Thread, seconds: float) -> bool:
    """Whether ``thread`` is still running after ``seconds`` (bounded wait)."""

    thread.join(timeout=seconds)
    return thread.is_alive()


def test_import_installs_every_construction_guard():
    from gptqmodel.utils import torch as torch_utils

    assert torch_utils.model_global_state_dtype_guard_installed()
    assert torch_utils.model_global_state_init_context_guard_installed()
    assert getattr(
        modeling_utils.PreTrainedModel.get_init_context.__func__,
        "_gptqmodel_global_state_guarded",
        False,
    )
    assert getattr(
        modeling_utils.PreTrainedModel._from_config.__func__,
        "_gptqmodel_global_state_guarded",
        False,
    )
    assert getattr(initialization.no_tie_weights, "_gptqmodel_global_state_guarded", False)
    assert getattr(modeling_utils.apply_patches, "_gptqmodel_global_state_guarded", False)


class _ForeignWindow:
    """Hold a transformers construction patch open on another thread."""

    def __init__(self, factory):
        self._factory = factory
        self.holding = threading.Event()
        self.release = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        with self._factory():
            self.holding.set()
            self.release.wait(30)

    def __enter__(self):
        self.thread.start()
        assert self.holding.wait(30), "the foreign construction window never opened"
        return self

    def __exit__(self, *_exc):
        self.release.set()
        self.thread.join(30)
        return False


def test_init_context_group_holds_the_writer_and_disables_tying():
    """The whole init-context group, not only the dtype window, is exclusive."""

    lock = model_global_state_lock()
    contexts = modeling_utils.PreTrainedModel.get_init_context(
        torch.float16, False, False, False
    )
    assert contexts[0] is lock, "the writer must be entered before every context"

    started = threading.Event()
    release = threading.Event()
    observed = {}

    def construct():
        stack = ExitStack()
        try:
            for context in contexts:
                if isinstance(context, torch.device):
                    # The meta default device is process-wide; the other
                    # contexts are what this test is about.
                    continue
                stack.enter_context(context)
            observed["writer_held"] = lock.holds_writer()
            observed["tie_is_noop"] = (
                modeling_utils.PreTrainedModel.tie_weights.__name__ == "empty_func"
            )
            started.set()
            release.wait(30)
        finally:
            stack.close()

    thread = threading.Thread(target=construct, daemon=True)
    thread.start()
    try:
        assert started.wait(30), "the construction window never opened"
        assert observed == {"writer_held": True, "tie_is_noop": True}

        entered = threading.Event()

        def compiled_call():
            with lock.reader():
                entered.set()

        reader = threading.Thread(target=compiled_call, daemon=True)
        reader.start()
        try:
            assert not entered.wait(0.5), (
                "a compiled call entered while the class-wide no-tie patch was active"
            )
        finally:
            release.set()
            thread.join(30)
            reader.join(30)
        assert entered.wait(5)
    finally:
        release.set()
        thread.join(30)
    assert lock._writer is None


def test_foreign_tie_window_blocks_a_from_config_construction():
    """A construction may not run inside another thread's no-tie window."""

    result = {}
    errors = []

    def build():
        try:
            result["model"] = AutoModelForCausalLM.from_config(_tiny_config())
        except BaseException as exc:  # noqa: BLE001 - reported below
            errors.append(exc)

    with _ForeignWindow(initialization.no_tie_weights):
        builder = threading.Thread(target=build, daemon=True)
        builder.start()
        assert _wait_alive(builder, 0.5), (
            "a from_config construction ran inside a foreign no-tie window; "
            "its shell would stay untied (#5421)"
        )

    builder.join(30)
    assert not errors, f"the construction failed: {errors!r}"
    assert _is_tied(result["model"])


def test_post_load_tie_is_not_skipped_by_a_foreign_window():
    """The tie after a checkpoint load must reach the real ``tie_weights``."""

    model = AutoModelForCausalLM.from_config(_tiny_config())
    # A quantized shell is built with weight init suspended, so its head is
    # still separate until the post-load tie.
    model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.detach().clone())
    assert not _is_tied(model)

    errors = []

    def tie():
        try:
            _tie_weights_after_checkpoint_load(model, None)
        except BaseException as exc:  # noqa: BLE001 - reported below
            errors.append(exc)

    with _ForeignWindow(initialization.no_tie_weights):
        tier = threading.Thread(target=tie, daemon=True)
        tier.start()
        assert _wait_alive(tier, 0.5), (
            "the post-load tie ran inside a foreign no-tie window and was "
            "silently skipped (#5421)"
        )

    tier.join(30)
    assert not errors, f"the tie failed: {errors!r}"
    assert _is_tied(model)


def test_default_dtype_capture_waits_out_a_foreign_window():
    """The dtype captured for the load invariant is the settled process value."""

    captured = {}

    def capture():
        captured["dtype"] = _default_dtype_outside_construction()

    with _ForeignWindow(
        lambda: modeling_utils.local_torch_dtype(torch.bfloat16, model_class_name="Tiny")
    ):
        thread = threading.Thread(target=capture, daemon=True)
        thread.start()
        assert _wait_alive(thread, 0.5), (
            "the load captured another thread's temporary default dtype (#5421)"
        )

    thread.join(30)
    assert captured["dtype"] is torch.float32


def test_load_state_check_waits_out_a_foreign_window():
    """A correct load is not rejected because another thread changed the dtype."""

    model = AutoModelForCausalLM.from_config(_tiny_config())
    errors = []

    def check():
        try:
            assert_loaded_model_state(
                model, api_name="test.load", default_dtype_before=torch.float32
            )
        except BaseException as exc:  # noqa: BLE001 - reported below
            errors.append(exc)

    with _ForeignWindow(
        lambda: modeling_utils.local_torch_dtype(torch.bfloat16, model_class_name="Tiny")
    ):
        thread = threading.Thread(target=check, daemon=True)
        thread.start()
        assert _wait_alive(thread, 0.5), (
            "the load-state check observed another thread's dtype window and "
            "reported a change the load did not make (#5421)"
        )

    thread.join(30)
    assert not errors, f"the check rejected a correct load: {errors!r}"


def test_norm_teacher_load_takes_the_model_load_lock(monkeypatch):
    """The ``requantize(norm=...)`` teacher is a GPT-QModel-owned construction."""

    from gptqmodel.models import base as base_module
    from gptqmodel.quantization.config import NormConfig

    class _TrackingLock:
        def __init__(self):
            self.depth = 0

        def __enter__(self):
            self.depth += 1
            return self

        def __exit__(self, *_exc):
            self.depth -= 1
            return False

    lock = _TrackingLock()
    seen = {}

    class _RecordingAutoModel:
        @staticmethod
        def from_pretrained(path, **kwargs):
            seen["path"] = path
            seen["lock_depth"] = lock.depth
            return torch.nn.Linear(4, 4)

    monkeypatch.setattr(base_module, "model_load_lock", lambda: lock)
    monkeypatch.setattr(base_module, "AutoModelForCausalLM", _RecordingAutoModel)
    # ``_requantize_norm`` resolves the teacher config before the model load;
    # a synthetic id must not reach the Hub there either.
    monkeypatch.setattr(base_module, "load_model_config", lambda *_a, **_k: _tiny_config())
    # Transformers 5.23 routes every model through a ``LocalModelLoader``, so
    # the Qwen2 teacher resolves through the port loader, not the auto class.
    # Pin the loader factory the production path actually calls.
    monkeypatch.setattr(
        base_module, "local_model_loader_for_config", lambda _config: _RecordingAutoModel
    )
    monkeypatch.setattr(
        "gptqmodel.quantization.norm_tuning.tune_norm_params",
        lambda *_args, **_kwargs: {"modules": []},
    )

    class _Owner:
        trust_remote_code = False
        load_quantized_model = True
        _module_replacement_prefixes = set()

        def __init__(self):
            self.model = Qwen2ForCausalLM(_tiny_config())

        def get_input_embeddings(self):
            return self.model.get_input_embeddings()

        def prepare_dataset(self, **_kwargs):
            return []

        def _calibration_batch_size(self, _calibration):
            return 1

    owner = _Owner()
    base_module.BaseQModel._requantize_norm(
        owner,
        calibration=SimpleNamespace(
            dataset=[],
            concat_size=None,
            sort=None,
            min_length=None,
            concat_separator=None,
        ),
        norm=NormConfig(teacher="fake-teacher", steps=1),
        tokenizer=None,
        names=["model.norm"],
    )

    assert seen["path"] == "fake-teacher"
    assert seen["lock_depth"] == 1, (
        "the norm-teacher from_pretrained ran outside the model-load lock (#5421)"
    )
    assert lock.depth == 0


def test_concurrent_foreign_construction_does_not_break_quantized_loads(tmp_path):
    """The reported scenario: loads of a tied checkpoint next to a plain load."""

    from gptqmodel import BACKEND, GPTQModel
    from gptqmodel.quantization import RTNConfig

    torch.manual_seed(0)
    source = tmp_path / "dense"
    Qwen2ForCausalLM(_tiny_config()).to(torch.float16).save_pretrained(source)

    quantized = tmp_path / "quantized"
    model = GPTQModel.load(
        str(source),
        RTNConfig(bits=4, group_size=32),
        backend=BACKEND.TORCH,
        device="cpu",
        dtype=torch.float16,
    )
    model.quantize()
    model.save(str(quantized))
    del model

    stop = threading.Event()
    foreign_errors = []
    foreign_loads = [0]

    def foreign_construction():
        while not stop.is_set():
            try:
                loaded = AutoModelForCausalLM.from_pretrained(
                    source, dtype=torch.bfloat16, device_map="cpu"
                )
                del loaded
                foreign_loads[0] += 1
            except BaseException as exc:  # noqa: BLE001 - reported below
                foreign_errors.append(exc)

    thread = threading.Thread(target=foreign_construction, daemon=True)
    thread.start()
    failures = []
    try:
        for _ in range(5):
            try:
                loaded = GPTQModel.load(
                    str(quantized),
                    backend=BACKEND.TORCH,
                    device="cpu",
                    dtype=torch.float16,
                )
            except BaseException as exc:  # noqa: BLE001 - reported below
                failures.append(exc)
                continue
            assert _is_tied(loaded.model), "the loaded shell lost its tied lm_head"
            del loaded
    finally:
        stop.set()
        thread.join(60)

    assert foreign_loads[0] > 0, "the foreign construction never ran"
    assert not foreign_errors, f"the foreign construction failed: {foreign_errors!r}"
    assert not failures, f"{len(failures)} load(s) failed: {failures!r}"
