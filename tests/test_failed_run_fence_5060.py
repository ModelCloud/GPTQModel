# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Both loopers must fence submitted work before a failed run tears down (#5060).

``ModuleLooper`` has drained ``DEVICE_THREAD_POOL``, its finalizer watcher
threads and ``STREAM_DEVICE_POOL`` since #4595. ``WeightOnlyLooper`` submits
submodule finalizers to the same pool (the pack/replace steps) but its failure
branch went straight to ``_restore_inflight_layer()``,
``rehome_failed_run_kernels()`` and ``release_private_offload_root()``.

That is the #4595 defect class in the other looper: a finalizer this run
already submitted keeps packing and replacing modules after ``quantize()`` has
raised, and re-creates the per-run offload directory that the teardown just
removed (one leaked bundle set per failed run). #5049 made the same
"the calibrated looper already guarantees this" argument for the in-flight
placement, and this issue's contract states the obligation for both loopers.

The two tests below pin the ordering (fence first, then repairs, then release)
and the failure policy (a wedged drain must not skip the repairs or replace the
run's exception). They drive the exit path directly, without a model, so they
assert the sequence rather than a quantization result.
"""

from __future__ import annotations

import shutil
from types import SimpleNamespace

import pytest

from gptqmodel.looper import weight_only_looper as weight_only_looper_module
from gptqmodel.looper.weight_only_looper import WeightOnlyLooper


def _failing_looper(calls: list[str]) -> WeightOnlyLooper:
    """Build a ``WeightOnlyLooper`` whose run always fails, without a model."""

    looper = object.__new__(WeightOnlyLooper)
    looper.gptq_model = SimpleNamespace(model=SimpleNamespace())
    looper.processor = SimpleNamespace()

    def _raise(**_kwargs):
        raise RuntimeError("weight-only layer error")

    looper._loop_impl = _raise
    looper._restore_inflight_layer = lambda: calls.append("inflight")
    looper._restore_offloaded_base_modules = lambda strict: calls.append(f"base:{strict}")
    looper._materialize_unreached_meta_modules = lambda: calls.append("materialize")
    return looper


def _patch_exit_steps(monkeypatch, calls: list[str], *, release_root):
    monkeypatch.setattr(
        weight_only_looper_module,
        "rehome_failed_run_kernels",
        lambda _model: calls.append("rehome"),
    )
    monkeypatch.setattr(
        weight_only_looper_module,
        "ensure_version_counters",
        lambda _model: calls.append("counters"),
    )
    monkeypatch.setattr(
        weight_only_looper_module,
        "has_disk_offloaded_modules",
        lambda _model: False,
    )
    monkeypatch.setattr(
        weight_only_looper_module,
        "release_private_offload_root",
        release_root,
    )


def test_weight_only_failed_run_fences_submitted_work_before_teardown(
    tmp_path, monkeypatch
):
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

    monkeypatch.setattr(weight_only_looper_module, "DEVICE_THREAD_POOL", _Pool())
    monkeypatch.setattr(
        "gptqmodel.utils.stream.STREAM_DEVICE_POOL", _StreamPool(), raising=True
    )
    _patch_exit_steps(monkeypatch, calls, release_root=_release_root)

    looper = _failing_looper(calls)

    with pytest.raises(RuntimeError, match="weight-only layer error"):
        looper.loop()

    assert calls == [
        "device_pool",
        "stream_pool",
        "inflight",
        "base:False",
        "materialize",
        "rehome",
        "counters",
        "release_root",
    ]
    # The released directory must not be re-created by a late finalizer.
    assert not run_root.exists()


def test_weight_only_drain_failure_still_repairs_and_keeps_the_run_error(
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

    monkeypatch.setattr(weight_only_looper_module, "DEVICE_THREAD_POOL", _BrokenPool())
    monkeypatch.setattr(
        "gptqmodel.utils.stream.STREAM_DEVICE_POOL", _StreamPool(), raising=True
    )
    _patch_exit_steps(
        monkeypatch, calls, release_root=lambda _model: calls.append("release_root")
    )

    looper = _failing_looper(calls)

    # A wedged drain is reported, not raised: the repairs still run and the
    # caller sees the error that failed the run.
    with pytest.raises(RuntimeError, match="weight-only layer error"):
        looper.loop()

    assert calls == [
        "device_pool",
        "stream_pool",
        "inflight",
        "base:False",
        "materialize",
        "rehome",
        "counters",
        "release_root",
    ]


def test_both_loopers_share_one_fence_implementation():
    """The calibrated looper's drain must not be a second copy of the fence."""

    from gptqmodel.looper import module_looper as module_looper_module

    calibrated = module_looper_module.ModuleLooper._fence_run_work
    weight_only = WeightOnlyLooper._fence_run_work

    # Both delegate to the shared helper, so a new wait is added once.
    assert "fence_run_work" in calibrated.__code__.co_names
    assert "fence_run_work" in weight_only.__code__.co_names
