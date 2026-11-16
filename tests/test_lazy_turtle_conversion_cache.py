# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for issue #4205: cache and release LazyTurtle conversions."""

import gc
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from torch import nn

from gptqmodel.utils.structure import LazyTurtle


class _FakeCheckpointModel(nn.Module):
    """Minimal `from_pretrained`/`save_pretrained` stand-in for `.bin` exports."""

    save_calls = 0
    save_delay = 0.0
    _counter_lock = threading.Lock()

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(model_type="fake_lazy_turtle")

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        del path, kwargs
        return cls()

    def save_pretrained(self, save_directory, safe_serialization=True):
        del safe_serialization
        if self.save_delay:
            time.sleep(self.save_delay)
        with type(self)._counter_lock:
            type(self).save_calls += 1
        (Path(save_directory) / "model.safetensors").write_bytes(b"stub")


def _patch_bin_source(monkeypatch, source_dir: Path) -> None:
    archive = source_dir / "pytorch_model.bin"
    archive.write_bytes(b"placeholder")

    def _get_checkpoints(*args, **kwargs):
        del args, kwargs
        return False, str(archive), None

    monkeypatch.setattr("gptqmodel.utils.model.get_checkpoints", _get_checkpoints)

    def _load_weight_map(self, model_local_path):
        del self
        if Path(model_local_path) == source_dir:
            raise FileNotFoundError(str(source_dir))
        return {"weight": "model.safetensors"}

    monkeypatch.setattr(LazyTurtle, "_load_weight_map", _load_weight_map)


def _create_turtle(
    source_dir: Path,
    *,
    offload_path: Path | None = None,
    target_model: nn.Module | None = None,
) -> LazyTurtle:
    turtle = LazyTurtle.maybe_create(
        model_local_path=str(source_dir),
        config=SimpleNamespace(model_type="fake_lazy_turtle"),
        model_init_kwargs={"device_map": {"": "cpu"}},
        target_model=target_model or _FakeCheckpointModel(),
        offload_path=None if offload_path is None else str(offload_path),
    )
    assert turtle is not None
    return turtle


@pytest.fixture(autouse=True)
def _reset_fake_model():
    _FakeCheckpointModel.save_calls = 0
    _FakeCheckpointModel.save_delay = 0.0
    yield
    _FakeCheckpointModel.save_calls = 0
    _FakeCheckpointModel.save_delay = 0.0


def test_same_source_reuses_one_conversion_and_last_release_removes_it(tmp_path, monkeypatch):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    _patch_bin_source(monkeypatch, source_dir)
    offload = tmp_path / "offload"

    first = _create_turtle(source_dir, offload_path=offload)
    converted = first.model_local_path
    assert Path(converted).is_dir()
    assert Path(converted).parent == offload
    assert _FakeCheckpointModel.save_calls == 1

    second = _create_turtle(source_dir, offload_path=offload)
    assert second.model_local_path == converted
    # The second load reuses the already-published conversion.
    assert _FakeCheckpointModel.save_calls == 1

    # Releasing one turtle must not delete a directory still in use.
    del first
    gc.collect()
    assert Path(converted).is_dir()

    # Dropping the last turtle frees the converted directory.
    del second
    gc.collect()
    assert not Path(converted).exists()


def test_dropping_last_turtle_runs_weakref_finalizer(tmp_path, monkeypatch):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    _patch_bin_source(monkeypatch, source_dir)

    turtle = _create_turtle(source_dir)
    converted = turtle.model_local_path
    assert Path(converted).is_dir()

    del turtle
    gc.collect()
    assert not Path(converted).exists()


def test_explicit_offload_path_is_honoured(tmp_path, monkeypatch):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    _patch_bin_source(monkeypatch, source_dir)
    offload = tmp_path / "configured-offload"

    turtle = _create_turtle(source_dir, offload_path=offload)
    converted = Path(turtle.model_local_path)

    assert converted.is_dir()
    assert converted.parent == offload
    assert offload in converted.parents
    turtle.close()


def test_close_releases_conversion_while_turtle_is_alive(tmp_path, monkeypatch):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    _patch_bin_source(monkeypatch, source_dir)

    turtle = _create_turtle(source_dir)
    converted = turtle.model_local_path
    assert Path(converted).is_dir()

    turtle.close()

    assert not Path(converted).exists()
    # Closing twice is a no-op.
    turtle.close()


def test_concurrent_loads_convert_once(tmp_path, monkeypatch):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    _patch_bin_source(monkeypatch, source_dir)
    _FakeCheckpointModel.save_delay = 0.3
    offload = tmp_path / "offload"

    turtles: list[LazyTurtle] = []
    errors: list[BaseException] = []
    start = threading.Barrier(3)

    def _worker():
        start.wait()
        try:
            turtles.append(_create_turtle(source_dir, offload_path=offload))
        except BaseException as exc:  # pragma: no cover - surfaced by assertion below
            errors.append(exc)

    threads = [threading.Thread(target=_worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    start.wait()
    for thread in threads:
        thread.join(timeout=30)

    assert not errors
    assert len(turtles) == 2
    assert turtles[0].model_local_path == turtles[1].model_local_path
    assert _FakeCheckpointModel.save_calls == 1
    for turtle in turtles:
        turtle.close()
