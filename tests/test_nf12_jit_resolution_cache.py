# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1
"""NF12 JIT resolution is a process-level answer, not a per-call one.

``library_directory`` is reached from every NF12 call. It used to memoize only
a successful build: a toolchain that cannot build (an old CMake, a broken
compiler) was retried on every call - once per module and per capture subset
during quantization. One 0.5B GPTQ run logged 672 doomed cmake configures.

These tests pin: the cache target is resolved once, a failed build is attempted
once, and ``reset_runtime_cache`` makes the next call try again.
"""


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
