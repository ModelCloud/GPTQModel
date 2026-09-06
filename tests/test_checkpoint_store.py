# SPDX-License-Identifier: Apache-2.0
import json
import os
import signal
import subprocess
import sys
import sysconfig
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from gptqmodel.looper.checkpoint_store import (
    CheckpointConfig,
    CheckpointCorrupt,
    CheckpointError,
    CheckpointStore,
)

IDENTITY = {"model": "dense", "algorithm": {"bits": 4}}


def commit(store, cursor):
    return store.commit(
        identity=IDENTITY,
        cursor=cursor,
        continuation=store.put(f"state-{cursor}".encode()),
        artifacts={"layer": store.put(f"weights-{cursor}".encode())},
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"resume": "sometimes"},
        {"every_layers": 0},
        {"every_layers": True},
        {"keep_last": 1},
    ],
)
def test_invalid_config(tmp_path, kwargs):
    with pytest.raises(ValueError):
        CheckpointConfig(tmp_path, **kwargs)


def test_commit_load_and_retention(tmp_path):
    with CheckpointStore(CheckpointConfig(tmp_path)) as store:
        assert store.load(IDENTITY) is None
        first = commit(store, 1)
        second = commit(store, 2)
        third = commit(store, 3)
        assert store.load(IDENTITY) == third
        store.collect()
        with pytest.raises(CheckpointCorrupt):
            store.get(first["continuation"])
        assert store.get(second["continuation"]) == b"state-2"


def test_streamed_artifact_is_immutable_and_reused(tmp_path):
    source = tmp_path / "source"
    source.write_bytes(b"packed weights")
    with CheckpointStore(CheckpointConfig(tmp_path / "checkpoint")) as store:
        ref = store.put_file(source)
        source.write_bytes(b"changed speculative weights")
        assert store.get(ref) == b"packed weights"
        assert store.put_file(store.object_path(ref)) == ref


def test_missing_and_corrupt_generations_fail_closed(tmp_path):
    with CheckpointStore(CheckpointConfig(tmp_path)) as store:
        manifest = commit(store, 1)
        (tmp_path / "objects" / manifest["continuation"]).unlink()
        with pytest.raises(CheckpointCorrupt, match="no complete published"):
            store.load(IDENTITY)


def test_recomputed_artifact_repairs_corruption_after_fallback(tmp_path):
    with CheckpointStore(CheckpointConfig(tmp_path)) as store:
        first = commit(store, 1)
        second = commit(store, 2)
    (tmp_path / "objects" / second["artifacts"]["layer"]).write_bytes(b"corrupt")
    with CheckpointStore(CheckpointConfig(tmp_path)) as store:
        assert store.load(IDENTITY) == first
        assert commit(store, 2) == second
        assert store.load(IDENTITY) == second


def test_corruption_falls_back_without_promoting_orphan(tmp_path):
    with CheckpointStore(CheckpointConfig(tmp_path)) as store:
        first = commit(store, 1)
        second = commit(store, 2)
        (tmp_path / "objects" / second["artifacts"]["layer"]).write_bytes(b"corrupt")
        # Even a valid manifest cannot be recovered unless CURRENT published it.
        orphan = {**first, "cursor": 999}
        store.put(json.dumps(orphan).encode())
        assert store.load(IDENTITY) == first
        commit(store, 3)
        refs = json.loads((tmp_path / "CURRENT").read_bytes())["generations"]
        assert [store._manifest(ref)["cursor"] for ref in refs] == [3, 1]


def test_identity_mismatch_is_not_fallback(tmp_path):
    with CheckpointStore(CheckpointConfig(tmp_path)) as store:
        commit(store, 1)
        with pytest.raises(CheckpointError, match="algorithm"):
            store.load({**IDENTITY, "algorithm": {"bits": 8}})


def test_resume_policies(tmp_path):
    with (
        CheckpointStore(CheckpointConfig(tmp_path, resume="required")) as store,
        pytest.raises(CheckpointError, match="no checkpoint"),
    ):
        store.load(IDENTITY)
    with CheckpointStore(CheckpointConfig(tmp_path, resume="never")) as store:
        commit(store, 1)
        commit(store, 2)
    with (
        CheckpointStore(CheckpointConfig(tmp_path, resume="never")) as store,
        pytest.raises(CheckpointError, match="requires a new directory"),
    ):
        commit(store, 3)


def test_exclusive_lease_and_thread_ownership(tmp_path):
    store = CheckpointStore(CheckpointConfig(tmp_path))
    with pytest.raises(CheckpointError, match="lease-owning"):
        store.put(b"bad")
    with store:
        with (
            pytest.raises(CheckpointError, match="another writer"),
            CheckpointStore(CheckpointConfig(tmp_path)),
        ):
            pytest.fail("duplicate lease")
        with (
            ThreadPoolExecutor(1) as pool,
            pytest.raises(CheckpointError, match="lease-owning"),
        ):
            pool.submit(store.put, b"bad").result()
    with CheckpointStore(CheckpointConfig(tmp_path)):
        pass


def test_failed_publication_preserves_current(tmp_path, monkeypatch):
    with CheckpointStore(CheckpointConfig(tmp_path)) as store:
        first = commit(store, 1)
        original = os.replace

        def fail(src, dst):
            if Path(dst).name == "CURRENT":
                raise OSError("simulated disk failure")
            return original(src, dst)

        monkeypatch.setattr(os, "replace", fail)
        with pytest.raises(OSError, match="disk failure"):
            commit(store, 2)
        assert store.load(IDENTITY) == first


def test_invalid_current_fails_closed(tmp_path):
    with CheckpointStore(CheckpointConfig(tmp_path)) as store:
        commit(store, 1)
        (tmp_path / "CURRENT").write_text("{")
        with pytest.raises(CheckpointCorrupt, match="refusing to scan"):
            store.load(IDENTITY)


def test_path_traversal_rejected(tmp_path):
    with (
        CheckpointStore(CheckpointConfig(tmp_path)) as store,
        pytest.raises(CheckpointCorrupt, match="invalid object reference"),
    ):
        store.get("../../secret")


# Import the storage module directly: subprocess fault injection should not
# initialize CUDA, transformers, or the quantization worker pool.
DRIVER = r"""
import importlib.util, json, os, signal, sys, uuid
spec = importlib.util.spec_from_file_location("checkpoint_store", sys.argv[1])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
if hasattr(sys, "_is_gil_enabled"):
    if os.environ.get("PYTHON_GIL") == "0":
        assert not sys._is_gil_enabled()
with module.CheckpointStore(module.CheckpointConfig(sys.argv[2])) as store:
    original = store._atomic_write
    def interrupt(destination, data):
        if destination.parent.name == "objects" and sys.argv[3] == "partial-object":
            temporary = destination.parent / f".{uuid.uuid4().hex}.tmp"
            with temporary.open("wb") as stream:
                stream.write(data[:max(1, len(data) // 2)])
                stream.flush()
                os.kill(os.getpid(), int(sys.argv[4]))
        if destination.name == "CURRENT":
            if sys.argv[3] == "after":
                original(destination, data)
            os.kill(os.getpid(), int(sys.argv[4]))
        original(destination, data)
    store._atomic_write = interrupt
    store.commit(identity={"model": "dense", "algorithm": {"bits": 4}}, cursor=2,
                 continuation=store.put(b"state-2"), artifacts={"layer": store.put(b"weights-2")})
"""


@pytest.mark.skipif(os.name != "posix", reason="POSIX process termination")
@pytest.mark.parametrize(
    "when,expected", [("partial-object", 1), ("before", 1), ("after", 2)]
)
@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGKILL])
def test_process_death_at_publication(tmp_path, when, expected, sig):
    import gptqmodel.looper.checkpoint_store as module

    with CheckpointStore(CheckpointConfig(tmp_path)) as store:
        commit(store, 1)
    environment = dict(os.environ)
    if sysconfig.get_config_var("Py_GIL_DISABLED"):
        environment["PYTHON_GIL"] = "0"
    result = subprocess.run(
        [sys.executable, "-c", DRIVER, module.__file__, str(tmp_path), when, str(sig)],
        env=environment,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == -sig, result.stderr.decode()
    with CheckpointStore(CheckpointConfig(tmp_path, resume="required")) as store:
        manifest = store.load(IDENTITY)
        assert manifest["cursor"] == expected
        assert store.get(manifest["continuation"]) == f"state-{expected}".encode()
        assert (
            store.get(manifest["artifacts"]["layer"]) == f"weights-{expected}".encode()
        )
        store.collect()
        assert not list((tmp_path / "objects").glob(".*.tmp"))
