# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""A failed ``quantize()`` must fence its submitted work before teardown (#4595).

#4515 fixed the capture-hook half of #4470. The other half stayed: the failure
branch of ``ModuleLooper.loop()`` released the model and deleted the per-run
offload directory while already-submitted submodule finalizers were still
running in ``DEVICE_THREAD_POOL``. Those finalizers re-created the directory
(one leaked bundle set per failed run) and kept packing/replacing modules after
``quantize()`` had raised to the caller.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from tokenizers import Tokenizer, models
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from gptqmodel import BACKEND, CalibrationConfig
from gptqmodel.looper import module_looper as module_looper_module
from gptqmodel.looper.module_looper import ModuleLooper
from gptqmodel.models.definitions.llama import LlamaQModel
from gptqmodel.quantization import QuantizeConfig
from gptqmodel.quantization.config import AutoDtypeCast


CALIBRATION = [{"input_ids": [1, 3, 5, 7, 9, 11, 13, 15], "attention_mask": [1] * 8}]


def _failing_looper(calls: list[str]) -> ModuleLooper:
    """Build a ``ModuleLooper`` whose run always fails, without touching a model."""

    looper = object.__new__(ModuleLooper)
    looper.gptq_model = SimpleNamespace()

    def _raise(**_kwargs):
        raise RuntimeError("layer callback error")

    looper._loop_impl = _raise
    looper.wait_dangling_threads = lambda: calls.append("dangling")
    looper._restore_inflight_layer = lambda: calls.append("inflight")
    looper._restore_offloaded_base_modules = lambda strict: calls.append(f"base:{strict}")
    looper._release_run_workspace = lambda: calls.append("workspace")
    return looper


def test_failed_run_fences_submitted_work_before_teardown(tmp_path, monkeypatch):
    run_root = tmp_path / "run-abc"
    run_root.mkdir()
    calls: list[str] = []

    class _Pool:
        def wait(self):
            calls.append("device_pool")
            # A finalizer this run already submitted is still writing here.
            (run_root / "layer.bundle").write_text("packed")

    class _StreamPool:
        def wait(self):
            calls.append("stream_pool")

    def _release_root(_model):
        calls.append("release_root")
        # The run directory may only be deleted once nothing is writing to it.
        assert (run_root / "layer.bundle").exists()
        shutil.rmtree(run_root)

    monkeypatch.setattr(module_looper_module, "DEVICE_THREAD_POOL", _Pool())
    monkeypatch.setattr(
        "gptqmodel.utils.stream.STREAM_DEVICE_POOL", _StreamPool(), raising=True
    )
    monkeypatch.setattr(module_looper_module, "release_private_offload_root", _release_root)

    looper = _failing_looper(calls)

    with pytest.raises(RuntimeError, match="layer callback error"):
        looper.loop()

    assert calls == [
        "device_pool",
        "dangling",
        "stream_pool",
        "inflight",
        "base:False",
        "release_root",
        "workspace",
    ]
    assert not run_root.exists()


def test_drain_failure_does_not_skip_the_other_waits_or_mask_the_run_error(
    tmp_path, monkeypatch
):
    calls: list[str] = []

    class _BrokenPool:
        def wait(self):
            calls.append("device_pool")
            raise RuntimeError("pool is wedged")

    class _StreamPool:
        def wait(self):
            calls.append("stream_pool")

    monkeypatch.setattr(module_looper_module, "DEVICE_THREAD_POOL", _BrokenPool())
    monkeypatch.setattr(
        "gptqmodel.utils.stream.STREAM_DEVICE_POOL", _StreamPool(), raising=True
    )
    monkeypatch.setattr(
        module_looper_module,
        "release_private_offload_root",
        lambda _model: calls.append("release_root"),
    )

    looper = _failing_looper(calls)

    with pytest.raises(RuntimeError, match="layer callback error"):
        looper.loop()

    # Every wait still ran, teardown still ran, and the original failure won.
    assert calls == [
        "device_pool",
        "dangling",
        "stream_pool",
        "inflight",
        "base:False",
        "release_root",
        "workspace",
    ]


def _build_qmodel(tmp_path: Path) -> LlamaQModel:
    source = tmp_path / "source"
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
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
            offload_to_disk=True,
            offload_to_disk_path=str(tmp_path / "offload"),
            auto_dtype_cast=AutoDtypeCast.OFF,
        ),
        model_local_path=str(source),
    )


def test_failed_quantize_does_not_leak_offload_bundles(tmp_path, monkeypatch):
    """The end-to-end shape of #4595: the deleted run dir must not come back."""

    from gptqmodel.looper import stage_layer

    qmodel = _build_qmodel(tmp_path)
    offload_root = Path(qmodel.quantize_config.offload_to_disk_path)
    real_offload = stage_layer.offload_to_disk

    def slow_offload(**kwargs):
        # Keep the finalizer in flight while the main loop fails, so the
        # failure path has real submitted work to fence.
        time.sleep(0.4)
        return real_offload(**kwargs)

    monkeypatch.setattr(stage_layer, "offload_to_disk", slow_offload)

    def fail_after_first_layer(*, layer_idx, submodule_finalized):
        if layer_idx == 0 and not submodule_finalized:
            raise RuntimeError("user layer callback error")
        return None

    qmodel.layer_callback = fail_after_first_layer

    with pytest.raises(RuntimeError, match="user layer callback error"):
        qmodel.quantize(
            CalibrationConfig(
                dataset=CALIBRATION,
                batch_size=1,
                min_length=1,
            ),
            backend=BACKEND.GPTQ_TORCH,
        )

    # Any finalizer still running at the raise must have been fenced before
    # the run directory was released; nothing may recreate it afterwards.
    from gptqmodel import DEVICE_THREAD_POOL

    DEVICE_THREAD_POOL.wait()
    time.sleep(0.5)
    assert not offload_root.exists() or list(offload_root.iterdir()) == []
