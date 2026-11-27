# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1
"""NF12 JIT resolution is a process-level answer, not a per-call one.

``library_directory`` is reached from every NF12 call. It used to memoize only
a successful build: a toolchain that cannot build (an old CMake, a broken
compiler) was retried on every call - once per module and per capture subset
during quantization. One 0.5B GPTQ run logged 1008 doomed cmake configures.

These tests pin: the cache target is resolved once, a failed build is attempted
once, ``reset_runtime_cache`` makes the next call try again, and the state is
safe for the free-threaded lane (``PYTHON_GIL=0``) - one lock, and a cached
failure re-raised as a fresh instance so threads never share a traceback.
"""


import threading
import time

import pytest

from gptqmodel_nf12 import _jit


@pytest.fixture(autouse=True)
def _fresh_jit_state(monkeypatch, tmp_path):
    _jit.reset_runtime_cache()
    monkeypatch.setattr(_jit, "_source_root", lambda: tmp_path)
    monkeypatch.setattr(_jit, "_architectures", lambda: [89])
    monkeypatch.setattr(_jit, "_build_tools", lambda: ("/tools/cmake", "/tools/ninja", "/tools/nvcc"))
    monkeypatch.setattr(_jit, "_cache_key", lambda source, archs: "fixture")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    yield
    _jit.reset_runtime_cache()


def test_a_failed_build_is_attempted_once_per_process(monkeypatch):
    attempts = []

    def broken(source, directory, archs):
        attempts.append(directory)
        raise RuntimeError("nf12: JIT compilation failed")

    monkeypatch.setattr(_jit, "_build", broken)

    for _ in range(3):
        with pytest.raises(RuntimeError, match="JIT compilation failed"):
            _jit.library_directory()

    assert len(attempts) == 1


def test_reset_runtime_cache_allows_a_retry(monkeypatch):
    attempts = []

    def broken(source, directory, archs):
        attempts.append(directory)
        raise RuntimeError("nf12: JIT compilation failed")

    monkeypatch.setattr(_jit, "_build", broken)
    with pytest.raises(RuntimeError):
        _jit.library_directory()

    _jit.reset_runtime_cache()
    with pytest.raises(RuntimeError):
        _jit.library_directory()

    assert len(attempts) == 2


def test_the_cache_target_is_hashed_once(monkeypatch, tmp_path):
    keys = []
    monkeypatch.setattr(_jit, "_cache_key", lambda source, archs: keys.append(1) or "key")

    _jit.cache_directory()
    _jit.cache_directory()

    assert len(keys) == 1


def test_a_successful_build_is_still_memoized(monkeypatch):
    builds = []
    monkeypatch.setattr(_jit, "_ready", lambda directory: True)

    def unexpected(*args, **kwargs):
        builds.append(args)
        raise AssertionError("a ready cache must not build")

    monkeypatch.setattr(_jit, "_build", unexpected)

    first = _jit.library_directory()
    second = _jit.library_directory()

    assert first == second
    assert not builds


def test_an_old_cmake_is_reported_before_any_build(monkeypatch):
    monkeypatch.setattr(_jit, "_build_tools", lambda: ("/usr/bin/cmake", "/usr/bin/ninja", "/usr/local/cuda/bin/nvcc"))
    monkeypatch.setattr(_jit, "_tool_version", lambda tool: (3, 22, 1))
    monkeypatch.setattr(_jit, "_architectures", lambda: [89])

    problem = _jit._toolchain_problem()

    assert problem is not None and "3.22.1" in problem and "3.26" in problem
    assert _jit.source_build_available() is False

    # The real ``_build`` refuses at the toolchain check, before any configure.
    with pytest.raises(_jit._MissingBuildTools, match="older than 3.26"):
        _jit.library_directory()


def test_a_toolchain_without_tools_is_reported():
    _jit.reset_runtime_cache()
    import gptqmodel_nf12._jit as jit

    problem = None
    original = jit._build_tools
    try:
        jit._build_tools = lambda: (None, None, None)
        problem = jit._toolchain_problem()
    finally:
        jit._build_tools = original

    assert problem is not None
    for tool in ("cmake", "ninja", "nvcc"):
        assert tool in problem

def _run_threads(work, count: int = 8):
    """Run ``work`` on ``count`` threads through a barrier and collect errors."""

    barrier = threading.Barrier(count)
    errors = []

    def runner(index):
        try:
            barrier.wait(timeout=30)
            work(index)
        except BaseException as exc:  # noqa: BLE001 - the assertions inspect these
            errors.append(exc)

    threads = [threading.Thread(target=runner, args=(index,)) for index in range(count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
    assert not any(thread.is_alive() for thread in threads), "a worker hung"
    return errors


def test_concurrent_build_failure_is_attempted_once(monkeypatch):
    """Eight threads racing a cold cache must not each start a build."""

    attempts = []

    def broken(source, directory, archs):
        attempts.append(1)
        time.sleep(0.05)
        raise RuntimeError("nf12: JIT compilation failed")

    monkeypatch.setattr(_jit, "_build", broken)

    errors = _run_threads(lambda _index: _jit.library_directory())

    assert len(errors) == 8
    assert all(isinstance(exc, RuntimeError) for exc in errors)
    assert len(attempts) == 1


def test_a_cached_failure_is_raised_as_a_fresh_instance(monkeypatch):
    """A shared exception instance would race on ``__traceback__`` under gil=0."""

    monkeypatch.setattr(_jit, "_build", lambda *args: (_ for _ in ()).throw(RuntimeError("boom")))

    seen = []
    for _ in range(2):
        with pytest.raises(RuntimeError, match="boom") as info:
            _jit.library_directory()
        seen.append(info.value)

    assert seen[0] is not seen[1]


def test_concurrent_toolchain_probe_runs_once(monkeypatch):
    """The ``cmake --version`` subprocess is shared, not run per thread."""

    probes = []

    def probe(tool):
        probes.append(tool)
        time.sleep(0.05)
        return (3, 26)

    monkeypatch.setattr(_jit, "_tool_version", probe)
    monkeypatch.setattr(_jit, "_architectures", lambda: [89])

    errors = _run_threads(lambda _index: _jit._toolchain_problem())

    assert not errors
    assert len(probes) == 1

def test_a_transient_filesystem_error_is_not_cached(monkeypatch):
    """Only ``_build`` failures are remembered; a bad cache dir can recover.

    The reviewer reproduced a ``PermissionError`` from the cache directory: the
    first call must raise it, and a later call whose directory is writable must
    be retried rather than re-raise a process-wide failure.
    """

    ready_calls = []

    def flaky_ready(directory):
        ready_calls.append(directory)
        if len(ready_calls) == 1:
            raise PermissionError("cache directory is not writable")
        return True

    monkeypatch.setattr(_jit, "_ready", flaky_ready)
    monkeypatch.setattr(_jit, "_build", lambda *args: (_ for _ in ()).throw(
        AssertionError("a ready cache must not build")
    ))

    with pytest.raises(PermissionError):
        _jit.library_directory()

    assert _jit.library_directory() == _jit.cache_directory() / "build"
    assert len(ready_calls) == 2

