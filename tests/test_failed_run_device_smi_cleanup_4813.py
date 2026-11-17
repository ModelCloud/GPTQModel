# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""A cancelled or failed ``quantize()`` must stop its Device-SMI sampler threads (#4813).

``LoopProcessor`` opens a ``device_smi.Device`` handle for every accelerator plus
the CPU, and every handle starts a daemon ``device-smi-metrics-*`` thread that
polls at 5 Hz. The handles were closed only in ``finalize()``, which runs after a
successful loop, so the failure branch of ``ModuleLooper.loop()`` -- a
``StopMainLoop`` from a subset callback, a ``KeyboardInterrupt``, or a callback
or kernel exception -- permanently leaked those threads for the life of the
process.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from tokenizers import Tokenizer, models
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from gptqmodel import BACKEND, CalibrationConfig
from gptqmodel.looper import module_looper as module_looper_module
from gptqmodel.looper.module_looper import ModuleLooper, StopMainLoop
from gptqmodel.models.definitions.llama import LlamaQModel
from gptqmodel.quantization import QuantizeConfig
from gptqmodel.quantization.config import AutoDtypeCast


CALIBRATION = [{"input_ids": [1, 3, 5, 7, 9, 11, 13, 15], "attention_mask": [1] * 8}]
SMI_THREAD_PREFIX = "device-smi"


def _smi_threads() -> list[threading.Thread]:
    return [
        thread
        for thread in threading.enumerate()
        if thread.name.startswith(SMI_THREAD_PREFIX)
    ]


def _new_smi_threads(baseline_idents: set[int | None]) -> list[threading.Thread]:
    return [thread for thread in _smi_threads() if thread.ident not in baseline_idents]


def _wait_for(predicate, timeout: float = 10.0, interval: float = 0.05) -> bool:
    """Poll ``predicate`` so a sampler thread that is still stopping is not a flake."""

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def _build_qmodel(tmp_path: Path) -> LlamaQModel:
    source = tmp_path / "source"
    config = LlamaConfig(
        vocab_size=32,
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
    return LlamaQModel(
        model=model,
        quantized=False,
        tokenizer=tokenizer,
        quantize_config=QuantizeConfig(
            bits=4,
            group_size=32,
            desc_act=False,
            device="cpu",
            offload_to_disk=False,
            auto_dtype_cast=AutoDtypeCast.OFF,
        ),
        model_local_path=str(source),
    )


@pytest.mark.parametrize("error", [StopMainLoop, RuntimeError], ids=["cancel", "failure"])
def test_failed_quantize_stops_device_smi_threads(tmp_path, error):
    """No sampler thread may outlive a cancelled or failed run."""

    baseline = {thread.ident for thread in _smi_threads()}
    qmodel = _build_qmodel(tmp_path)

    def stop_or_fail(**kwargs):
        if kwargs["stage"] == "forward_start" and kwargs["layer_idx"] == 0:
            if error is StopMainLoop:
                return StopMainLoop
            raise RuntimeError("injected mid-layer failure")
        return None

    qmodel.subset_callback = stop_or_fail

    with pytest.raises(error):
        qmodel.quantize(
            CalibrationConfig(
                dataset=CALIBRATION,
                batch_size=1,
                min_length=1,
            ),
            backend=BACKEND.GPTQ_TORCH,
        )

    assert _wait_for(lambda: not _new_smi_threads(baseline)), [
        thread.name for thread in _new_smi_threads(baseline)
    ]


# ---------------------------------------------------------------------------
# Failure-branch wiring and idempotency.
# ---------------------------------------------------------------------------


class _RecordingProcessor:
    def __init__(self, calls: list[str], *, fail: bool = False):
        self.calls = calls
        self.fail = fail

    def release_failed_run(self) -> None:
        self.calls.append("release")
        if self.fail:
            raise RuntimeError("processor release failed")


def _failing_looper(calls: list[str], processors) -> ModuleLooper:
    looper = object.__new__(ModuleLooper)
    looper.gptq_model = SimpleNamespace()

    def _raise(**_kwargs):
        raise RuntimeError("layer callback error")

    looper._loop_impl = _raise
    looper.processors = processors
    looper.wait_dangling_threads = lambda: calls.append("dangling")
    looper._restore_inflight_layer = lambda: calls.append("inflight")
    looper._restore_offloaded_base_modules = lambda strict: calls.append(f"base:{strict}")
    looper._release_run_workspace = lambda: calls.append("workspace")
    return looper


def _patch_run_pools(monkeypatch, calls: list[str]) -> None:
    class _Pool:
        def wait(self):
            calls.append("device_pool")

    class _StreamPool:
        def wait(self):
            calls.append("stream_pool")

    monkeypatch.setattr(module_looper_module, "DEVICE_THREAD_POOL", _Pool())
    monkeypatch.setattr(
        "gptqmodel.utils.stream.STREAM_DEVICE_POOL", _StreamPool(), raising=True
    )
    monkeypatch.setattr(
        module_looper_module,
        "release_private_offload_root",
        lambda _model: calls.append("release_root"),
    )


def test_failed_run_releases_processors_after_the_fence(monkeypatch):
    calls: list[str] = []
    _patch_run_pools(monkeypatch, calls)

    looper = _failing_looper(calls, [_RecordingProcessor(calls)])

    with pytest.raises(RuntimeError, match="layer callback error"):
        looper.loop()

    # The release happens once, after the fence drained the workers that use the
    # handles, and before the model is restored/offload is released.
    assert calls == [
        "device_pool",
        "dangling",
        "stream_pool",
        "release",
        "inflight",
        "base:False",
        "release_root",
        "workspace",
    ]


def test_processor_release_failure_does_not_mask_the_run_error(monkeypatch):
    calls: list[str] = []
    _patch_run_pools(monkeypatch, calls)

    looper = _failing_looper(calls, [_RecordingProcessor(calls, fail=True)])

    with pytest.raises(RuntimeError, match="layer callback error"):
        looper.loop()

    # The failed release is contained; the rest of the teardown still runs.
    assert calls == [
        "device_pool",
        "dangling",
        "stream_pool",
        "release",
        "inflight",
        "base:False",
        "release_root",
        "workspace",
    ]


def test_release_failed_run_is_idempotent():
    """A second release (or a later ``finalize()``) must not raise or double-close."""

    from device_smi import Device

    from gptqmodel.looper.loop_processor import LoopProcessor

    baseline = {thread.ident for thread in _smi_threads()}
    processor = object.__new__(LoopProcessor)
    processor._device_smi_lock = threading.RLock()
    processor._device_smi_handles = {}
    processor._cpu_device_smi = Device("cpu")
    try:
        assert _wait_for(lambda: bool(_new_smi_threads(baseline))), "device-smi sampler did not start"

        processor.release_failed_run()
        processor.release_failed_run()

        assert _wait_for(lambda: not _new_smi_threads(baseline)), [
            thread.name for thread in _new_smi_threads(baseline)
        ]
    finally:
        processor._close_device_smi_handles()
