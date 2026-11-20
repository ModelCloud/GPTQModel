# SPDX-License-Identifier: Apache-2.0
"""Issue #5749: continuation capture must not clone or reread its payload."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors import safe_open

from gptqmodel.looper.checkpoint import (
    CheckpointConfig,
    CheckpointExtension,
    CheckpointStopped,
    LoopPlan,
)
from gptqmodel.looper.checkpoint_store import CheckpointCorrupt, CheckpointStore
from gptqmodel.looper.continuation import ContinuationCodec
from gptqmodel.looper.extension import LoopExtensions, LoopStep


pytestmark = [pytest.mark.cpu]

REPO_ROOT = Path(__file__).resolve().parents[1]
PLAN = LoopPlan((LoopStep("layer", 0, "model.layers.0"),))
MIB = 2**20


class Adapter:
    def __init__(self, state=None):
        self.state = state

    def capture(self):
        return self.state, {}

    def restore(self, state, artifacts):
        self.state = state


def _state():
    source = torch.arange(64, dtype=torch.float32).reshape(8, 8)
    return {
        "view": source[:, ::2],
        "source": source,
        "alias_a": source,
        "alias_b": source,
        "row": source[:1],
        "dense": source + 1,
    }


def _bytes(tensor):
    return tensor.contiguous().reshape(-1).view(torch.uint8)


def _commit(tmp_path, state, monkeypatch=None):
    with CheckpointExtension(CheckpointConfig(tmp_path), Adapter(state)) as extension:
        extension.prepare(PLAN, {})
        extension.request_stop()
        with pytest.raises(CheckpointStopped):
            LoopExtensions([extension]).publish(PLAN.steps[0])


def test_file_codec_matches_in_memory_bytes_and_restores_bitwise(tmp_path):
    state = _state()
    data = ContinuationCodec.dumps(state)
    path = tmp_path / "continuation.safetensors"
    ContinuationCodec.dump_file(state, path)

    # The staged file is byte-identical, so the calibration identity digest
    # already stored in existing manifests stays valid.
    assert path.read_bytes() == data
    assert ContinuationCodec.sha256(state, tmp_path) == hashlib.sha256(data).hexdigest()

    with safe_open(path, framework="pt", device="cpu") as tensors:
        # Repeated references share one payload entry; views that overlap
        # another entry are the only ones copied.
        assert len(tensors.keys()) == 4

    restored = ContinuationCodec.load_file(path)
    for key, value in state.items():
        assert torch.equal(_bytes(restored[key]), _bytes(value))
    assert restored["alias_a"] is restored["alias_b"]


def test_encoder_does_not_clone_contiguous_cpu_tensors(monkeypatch):
    state = {"a": torch.ones(4), "b": torch.ones(4) + 1}

    def fail(*args, **kwargs):
        raise AssertionError("continuation encoder cloned a CPU tensor")

    monkeypatch.setattr(torch.Tensor, "clone", fail)
    restored = ContinuationCodec.loads(ContinuationCodec.dumps(state))
    assert torch.equal(restored["a"], state["a"])
    assert torch.equal(restored["b"], state["b"])


def test_commit_does_not_read_continuation_into_memory(tmp_path, monkeypatch):
    state = {"replay": torch.ones(1024, 1024)}
    full_reads = []
    original = CheckpointStore.get

    def observed(store, digest):
        data = original(store, digest)
        if len(data) >= MIB:
            full_reads.append(len(data))
        return data

    monkeypatch.setattr(CheckpointStore, "get", observed)
    _commit(tmp_path, state)
    assert full_reads == []


def test_resume_restores_all_tensors_bitwise(tmp_path):
    source = torch.arange(64, dtype=torch.float32).reshape(8, 8)
    state = {
        "view": source[:, ::2],
        "source": source,
        "alias_a": source,
        "alias_b": source,
        "dense": source + 1,
    }
    _commit(tmp_path, state)

    resumed = Adapter()
    with CheckpointExtension(
        CheckpointConfig(tmp_path, resume="required"), resumed
    ) as extension:
        assert extension.prepare(PLAN, {}) == 1
    for key, value in state.items():
        assert torch.equal(_bytes(resumed.state[key]), _bytes(value))
        assert resumed.state[key].device.type == "cpu"


def test_corrupted_continuation_fails_integrity(tmp_path):
    _commit(tmp_path, {"replay": torch.ones(8)})
    refs = json.loads((tmp_path / "CURRENT").read_bytes())["generations"]
    manifest = json.loads((tmp_path / "objects" / refs[0]).read_bytes())
    object_path = tmp_path / "objects" / manifest["continuation"]
    object_path.write_bytes(object_path.read_bytes()[:-1] + b"\x00")

    with (
        pytest.raises(CheckpointCorrupt, match="checksum mismatch"),
        CheckpointExtension(
            CheckpointConfig(tmp_path, resume="required"), Adapter()
        ) as extension,
    ):
        extension.prepare(PLAN, {})


_RSS_DRIVER = r"""
import gc
import json
import resource
import sys

import torch

from gptqmodel.looper.checkpoint import (
    CheckpointConfig,
    CheckpointExtension,
    CheckpointStopped,
    LoopPlan,
)
from gptqmodel.looper.checkpoint_store import CheckpointStore
from gptqmodel.looper.extension import LoopExtensions, LoopStep
from gptqmodel.looper.input_cache import InputCache


def peak_mib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


class Adapter:
    def __init__(self, state):
        self.state = state

    def capture(self):
        return self.state, {}

    def restore(self, state, artifacts):
        self.state = state


def build_state():
    replay = torch.ones((16, 2048, 2048), dtype=torch.bfloat16)
    dense = torch.ones((8192, 4096), dtype=torch.bfloat16)
    cache = InputCache([[replay]], [{}], [], [], src_inputs=[])
    return {"processors": [{"cache": cache}], "dense": {"projection": {"weight": dense}}}


mode, root = sys.argv[1], sys.argv[2]
state = build_state()
gc.collect()
base = peak_mib()
full_reads = []
if mode == "commit":
    original = CheckpointStore.get

    def observed(store, digest):
        data = original(store, digest)
        if len(data) >= 2**20:
            full_reads.append(len(data))
        return data

    CheckpointStore.get = observed
    plan = LoopPlan((LoopStep("layer", 0, "model.layers.0"),))
    with CheckpointExtension(CheckpointConfig(path=root), Adapter(state)) as ext:
        ext.prepare(plan, {})
        ext.request_stop()
        try:
            LoopExtensions([ext]).publish(plan.steps[0])
        except CheckpointStopped:
            pass
print(json.dumps({"peak_mib": peak_mib(), "base_mib": base, "full_reads": len(full_reads)}))
"""


def _run_rss_driver(tmp_path, mode):
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    result = subprocess.run(
        [sys.executable, "-c", _RSS_DRIVER, mode, str(tmp_path / mode)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_checkpoint_commit_peak_rss_is_bounded(tmp_path):
    baseline = _run_rss_driver(tmp_path, "baseline")
    commit = _run_rss_driver(tmp_path, "commit")
    # 192 MiB of live replay plus dense state. Main held several full copies.
    added = commit["peak_mib"] - baseline["peak_mib"]
    assert commit["full_reads"] == 0
    assert added < 256, f"commit added {added:.1f} MiB of peak RSS"
