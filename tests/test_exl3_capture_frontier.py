# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import gc
import json
from pathlib import Path
from types import SimpleNamespace
import threading
import weakref

import pytest
import torch

from gptqmodel.utils.exl3_capture_frontier import (
    EXL3CaptureDescriptor,
    EXL3CaptureFrontierError,
    EXL3CaptureFrontierStore,
    EXL3CaptureState,
)
from gptqmodel.looper.exllamav3_processor import EXL3Processor


FAMILY_JOIN = {"source": {"revision": "test"}, "corpus": {"sha256": "a" * 64}}


def _subset():
    return {
        "mlp.experts.7.gate_proj": SimpleNamespace(
            full_name="model.layers.3.mlp.experts.7.gate_proj"
        ),
        "mlp.experts.7.up_proj": SimpleNamespace(
            full_name="model.layers.3.mlp.experts.7.up_proj"
        ),
    }


def _mtp_subset():
    return {
        "mlp.experts.7.gate_proj": SimpleNamespace(
            full_name="mtp.0.mlp.experts.7.gate_proj"
        ),
        "mlp.experts.7.up_proj": SimpleNamespace(
            full_name="mtp.0.mlp.experts.7.up_proj"
        ),
    }


def _states(hessian: torch.Tensor):
    evidence = {"schema": "test-route", "expert": 7, "count": 32}
    recovery = {"schema": "test-zero-route-recovery", "count": 32}
    return [
        EXL3CaptureState(
            module=named.full_name,
            hessian=hessian.clone(),
            sample_count=32,
            route_evidence=evidence,
            zero_route_recovery=recovery,
        )
        for named in _subset().values()
    ]


def test_capture_frontier_round_trip_deduplicates_gate_up(tmp_path) -> None:
    store = EXL3CaptureFrontierStore(tmp_path / "frontier", family_join=FAMILY_JOIN)
    hessian = torch.eye(4, dtype=torch.float32)
    manifest = store.commit(
        layer_index=3,
        subset_index=1,
        subset_total=4,
        subset=_subset(),
        states=_states(hessian),
    )
    assert manifest["payload_hash_algorithm"] == "xxh3-128"
    assert len({record["hessian"]["file"] for record in manifest["captures"]}) == 1
    assert len(list(store.root.rglob("*.safetensors"))) == 1

    restored = store.restore(
        layer_index=3,
        subset_index=1,
        subset_total=4,
        subset=_subset(),
    )
    assert restored is not None
    gate = restored["model.layers.3.mlp.experts.7.gate_proj"]
    up = restored["model.layers.3.mlp.experts.7.up_proj"]
    assert torch.equal(gate.hessian, hessian)
    assert gate.hessian.data_ptr() == up.hessian.data_ptr()
    assert gate.sample_count == up.sample_count == 32
    assert gate.zero_route_recovery == up.zero_route_recovery == {
        "schema": "test-zero-route-recovery",
        "count": 32,
    }

    store.discard_through(2)
    assert list(store.root.iterdir())
    store.discard_through(3)
    assert list(store.root.iterdir()) == []


def test_scoped_discard_keeps_other_block_namespace(tmp_path) -> None:
    store = EXL3CaptureFrontierStore(tmp_path / "frontier", family_join=FAMILY_JOIN)
    hessian = torch.eye(4, dtype=torch.float32)
    store.commit(
        layer_index=3,
        subset_index=0,
        subset_total=1,
        subset=_subset(),
        states=_states(hessian),
    )
    mtp_subset = _mtp_subset()
    mtp_states = [
        EXL3CaptureState(
            module=named.full_name,
            hessian=hessian.clone(),
            sample_count=32,
            route_evidence={"schema": "test-route", "expert": 7, "count": 32},
        )
        for named in mtp_subset.values()
    ]
    store.commit(
        layer_index=0,
        subset_index=0,
        subset_total=1,
        subset=mtp_subset,
        states=mtp_states,
    )
    assert len(list(store.root.iterdir())) == 2

    store.discard_through(42, block_namespace="base")

    remaining = list(store.root.iterdir())
    assert len(remaining) == 1
    manifest = json.loads((remaining[0] / "manifest.json").read_text())
    assert all(
        capture["expert_identity"]["block_namespace"] == "mtp"
        for capture in manifest["captures"]
    )
    assert store.restore(
        layer_index=0,
        subset_index=0,
        subset_total=1,
        subset=mtp_subset,
    )

    store.discard_through(0, block_namespace="mtp")
    assert list(store.root.iterdir()) == []


def test_scoped_discard_rejects_unknown_namespace(tmp_path) -> None:
    store = EXL3CaptureFrontierStore(tmp_path / "frontier", family_join=FAMILY_JOIN)
    with pytest.raises(EXL3CaptureFrontierError, match="namespace is invalid"):
        store.discard_through(0, block_namespace="other")


def test_capture_frontier_rejects_gate_up_drift(tmp_path) -> None:
    store = EXL3CaptureFrontierStore(tmp_path / "frontier", family_join=FAMILY_JOIN)
    states = _states(torch.eye(4, dtype=torch.float32))
    states[1].hessian[0, 0] += 1
    with pytest.raises(EXL3CaptureFrontierError, match="not identical"):
        store.commit(
            layer_index=3,
            subset_index=1,
            subset_total=4,
            subset=_subset(),
            states=states,
        )


def test_capture_frontier_rejects_payload_tampering(tmp_path) -> None:
    store = EXL3CaptureFrontierStore(tmp_path / "frontier", family_join=FAMILY_JOIN)
    store.commit(
        layer_index=3,
        subset_index=1,
        subset_total=4,
        subset=_subset(),
        states=_states(torch.eye(4, dtype=torch.float32)),
    )
    payload = next(store.root.rglob("*.safetensors"))
    damaged = bytearray(payload.read_bytes())
    damaged[-1] ^= 1
    payload.write_bytes(damaged)
    with pytest.raises(EXL3CaptureFrontierError, match="failed validation"):
        store.restore(
            layer_index=3,
            subset_index=1,
            subset_total=4,
            subset=_subset(),
        )


def test_streaming_commit_bounds_live_hessians_to_one_family(tmp_path) -> None:
    store = EXL3CaptureFrontierStore(tmp_path / "frontier", family_join=FAMILY_JOIN)
    subset = {}
    descriptors = []
    for expert in (7, 8, 9):
        evidence = {"schema": "test-route", "expert": expert, "count": 32}
        for projection in ("gate_proj", "up_proj"):
            full_name = f"model.layers.3.mlp.experts.{expert}.{projection}"
            subset[f"mlp.experts.{expert}.{projection}"] = SimpleNamespace(
                full_name=full_name
            )
            descriptors.append(
                EXL3CaptureDescriptor(
                    module=full_name,
                    sample_count=32,
                    route_evidence=evidence,
                )
            )

    live = 0
    maximum_live = 0

    def load_hessian(_module):
        nonlocal live, maximum_live
        tensor = torch.eye(4, dtype=torch.float32)
        live += 1
        maximum_live = max(maximum_live, live)

        def release():
            nonlocal live
            live -= 1

        weakref.finalize(tensor, release)
        return tensor

    manifest = store.commit_streaming(
        layer_index=3,
        subset_index=1,
        subset_total=4,
        subset=subset,
        descriptors=descriptors,
        hessian_loader=load_hessian,
    )
    gc.collect()

    assert len(manifest["captures"]) == 6
    assert len(list(store.root.rglob("*.safetensors"))) == 3
    assert maximum_live <= 2
    assert live == 0


class _Capture:
    def __init__(self, hessian: torch.Tensor | None, sample_count: int) -> None:
        self.H = hessian
        self.nsamples = sample_count
        self._device_hessian_partials = {}
        self._device_sample_counts = {}
        self._hessian_dirty = False
        self._final_hessian_device_hint = None

    def finalize_hessian(self, target_device=None):
        self.H = self.H.to(device=target_device)
        return self.H

    def snapshot_hessian(self, target_device=None):
        return self.H.to(device=target_device).clone()


def _processor(subset, *, captured: bool, inline_mixed: bool = False) -> EXL3Processor:
    processor = EXL3Processor.__new__(EXL3Processor)
    meta = {"ds4rt_error_ledger": {"family_join": FAMILY_JOIN}}
    if inline_mixed:
        meta["ds4rt_inline_mixed"] = {
            "schema": "gptqmodel.exl3-inline-mixed",
            "schema_version": 1,
            "namespace": "base",
            "base_bits": 2,
            "upgrade_bits": 3,
            "extra_bits": {"numerator": 1, "denominator": 10},
            "target_bpw": "21/10",
            "projection_ratio": {"w1": 3, "w3": 5, "w2": 8},
            "score_kind": "k2-hessian-weighted-relative-error-times-natural-gate-squared-mass-v1",
            "tier_plan_root": str(Path.cwd() / "tier-plans"),
        }
    processor.qcfg = SimpleNamespace(
        meta=meta
    )
    processor._stats_lock = threading.Lock()
    processor._natural_route_evidence_cache = {}
    processor.tasks = {}
    for task_name in subset:
        processor.tasks[task_name] = {
            "capture": _Capture(
                torch.eye(4, dtype=torch.float32) if captured else None,
                32 if captured else 0,
            ),
            "route_evidence": {"expert": 7} if captured else None,
            "zero_route_recovery": (
                {"schema": "test-zero-route-recovery", "count": 32}
                if captured
                else None
            ),
        }
    return processor


def test_processor_commits_and_restores_before_replay(tmp_path, monkeypatch) -> None:
    subset = _subset()
    root = tmp_path / "frontier"
    monkeypatch.setenv("GPTQMODEL_EXL3_CAPTURE_FRONTIER", str(root))
    captured = _processor(subset, captured=True)
    captured.commit_subset_capture_frontier(
        layer_index=3,
        subset_index=1,
        subset_total=4,
        subset=subset,
    )
    captured_tasks = list(captured.tasks.values())
    assert all(task["capture"].H.device.type == "cpu" for task in captured_tasks)

    resumed = _processor(subset, captured=False)
    assert resumed.restore_subset_capture_frontier(
        layer_index=3,
        subset_index=1,
        subset_total=4,
        subset=subset,
    )
    load_devices = []
    load_record_hessian = EXL3CaptureFrontierStore.load_record_hessian

    def tracked_load(record, *, device="cpu"):
        load_devices.append(str(device))
        return load_record_hessian(record, device=device)

    monkeypatch.setattr(
        EXL3CaptureFrontierStore,
        "load_record_hessian",
        staticmethod(tracked_load),
    )
    for task in resumed.tasks.values():
        capture = task["capture"]
        assert capture.nsamples == 32
        assert capture.H is None
        assert task["capture_frontier_record"].sample_count == 32
        resumed._hydrate_capture_frontier(
            task_entry=task,
            capture=capture,
            target_device=torch.device("cpu"),
        )
        assert torch.equal(capture.H, torch.eye(4, dtype=torch.float32))
        assert capture._hessian_dirty is False
        assert "capture_frontier_record" not in task
        assert task["route_evidence"] == {"expert": 7}
        assert task["zero_route_recovery"] == {
            "schema": "test-zero-route-recovery",
            "count": 32,
        }
    assert load_devices == ["cpu", "cpu"]


def test_inline_commit_demotes_live_hessians_to_lazy_frontier(
    tmp_path, monkeypatch
) -> None:
    subset = _subset()
    root = tmp_path / "frontier"
    monkeypatch.setenv("GPTQMODEL_EXL3_CAPTURE_FRONTIER", str(root))
    captured = _processor(subset, captured=True, inline_mixed=True)
    for task in captured.tasks.values():
        capture = task["capture"]
        capture._device_hessian_partials[torch.device("cpu")] = torch.eye(4)
        capture._device_sample_counts[torch.device("cpu")] = 32

    captured.commit_subset_capture_frontier(
        layer_index=3,
        subset_index=1,
        subset_total=4,
        subset=subset,
    )

    for task in captured.tasks.values():
        capture = task["capture"]
        assert capture.H is None
        assert capture._device_hessian_partials == {}
        assert capture._device_sample_counts == {}
        assert capture.nsamples == 32
        assert task["capture_frontier_record"].sample_count == 32
        captured._hydrate_capture_frontier(
            task_entry=task,
            capture=capture,
            target_device=torch.device("cpu"),
        )
        assert torch.equal(capture.H, torch.eye(4))
        assert "capture_frontier_record" in task
