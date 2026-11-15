# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import ast
import threading
import time
from pathlib import Path

import pytest

import gptqmodel
from gptqmodel.utils import nogil_patcher


def test_triton_patch_apply_runs_once(monkeypatch):
    call_count = 0

    def fake_patch():
        nonlocal call_count
        call_count += 1

    monkeypatch.setattr(nogil_patcher, "patch_triton_autotuner", fake_patch)
    monkeypatch.setattr(nogil_patcher.TritonPatch, "_applied", False)

    nogil_patcher.TritonPatch.apply()
    nogil_patcher.TritonPatch.apply()
    nogil_patcher.TritonPatch.apply()

    assert call_count == 1


def test_triton_patch_apply_runs_once_across_threads(monkeypatch):
    call_count = 0
    call_count_lock = threading.Lock()

    def fake_patch():
        nonlocal call_count
        time.sleep(0.01)
        with call_count_lock:
            call_count += 1

    monkeypatch.setattr(nogil_patcher, "patch_triton_autotuner", fake_patch)
    monkeypatch.setattr(nogil_patcher.TritonPatch, "_applied", False)

    threads = [threading.Thread(target=nogil_patcher.TritonPatch.apply) for _ in range(12)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert call_count == 1


def test_triton_patch_public_api_export():
    assert gptqmodel.TritonPatch is nogil_patcher.TritonPatch


def test_triton_patch_external_usage_snippet():
    # Run the exact user-facing API snippet.
    from gptqmodel import TritonPatch
    TritonPatch.apply()


def test_patched_autotuner_constructor_owns_cache_lock(monkeypatch):
    triton_autotuner = pytest.importorskip("triton.runtime.autotuner")
    original_lock = threading.Lock()
    existing_cache_value = object()

    class _FakeAutotuner:
        def __init__(self):
            self.cache = {"existing": existing_cache_value}
            self._cache_lock = original_lock

    monkeypatch.setattr(triton_autotuner, "Autotuner", _FakeAutotuner)
    monkeypatch.setattr(triton_autotuner, "CacheFuture", None, raising=False)
    monkeypatch.setattr(nogil_patcher, "version", lambda _package_name: "3.7.0")

    nogil_patcher.patch_triton_autotuner()
    autotuner = _FakeAutotuner()

    assert autotuner._cache_lock is not original_lock
    assert isinstance(autotuner._cache_lock, type(threading.RLock()))
    assert autotuner.cache is autotuner._cache
    assert autotuner._cache == {"existing": existing_cache_value}
    assert autotuner._cache_futures == {}


def test_patched_autotuner_lazily_initializes_preexisting_instance(monkeypatch):
    triton_autotuner = pytest.importorskip("triton.runtime.autotuner")
    existing_cache_value = object()

    class _FakeAutotuner:
        def __init__(self):
            self.cache = {"existing": existing_cache_value}

    autotuner = _FakeAutotuner()

    monkeypatch.setattr(triton_autotuner, "Autotuner", _FakeAutotuner)
    monkeypatch.setattr(triton_autotuner, "CacheFuture", None, raising=False)
    monkeypatch.setattr(nogil_patcher, "version", lambda _package_name: "3.7.0")

    nogil_patcher.patch_triton_autotuner()
    cached, used_cached_result, bench_time = autotuner._get_config_for_key("existing", (), {})

    assert cached is existing_cache_value
    assert used_cached_result is True
    assert bench_time is None
    assert isinstance(autotuner._cache_lock, type(threading.RLock()))
    assert autotuner.cache is autotuner._cache
    assert autotuner._cache_futures == {}


def test_patched_autotuner_lazily_initializes_single_config_run(monkeypatch):
    triton_autotuner = pytest.importorskip("triton.runtime.autotuner")

    class _FakeConfig:
        pre_hook = None

        def all_kwargs(self):
            return {"BLOCK_SIZE": 16}

    class _FakeFn:
        def run(self, *args, **kwargs):
            return args, kwargs

    class _FakeAutotuner:
        def __init__(self):
            self.arg_names = ["value"]
            self.base_fn = _FakeFn.run
            self.cache = {}
            self.configs = [_FakeConfig()]
            self.fn = _FakeFn()

    autotuner = _FakeAutotuner()

    monkeypatch.setattr(triton_autotuner, "Autotuner", _FakeAutotuner)
    monkeypatch.setattr(triton_autotuner, "CacheFuture", None, raising=False)
    monkeypatch.setattr(nogil_patcher, "version", lambda _package_name: "3.7.0")

    nogil_patcher.patch_triton_autotuner()
    result = autotuner.run("input")

    assert result == (("input",), {"BLOCK_SIZE": 16})
    assert autotuner.best_config is autotuner.configs[0]
    assert isinstance(autotuner._cache_lock, type(threading.RLock()))
    assert autotuner.cache is autotuner._cache
    assert autotuner._cache_futures == {}


def test_patched_autotuner_lazily_initializes_concurrent_first_lookup(monkeypatch):
    triton_autotuner = pytest.importorskip("triton.runtime.autotuner")
    benchmark_count = 0
    benchmark_count_lock = threading.Lock()

    class _FakeConfig:
        def all_kwargs(self):
            return {}

    config = _FakeConfig()

    class _FakeAutotuner:
        def __init__(self):
            self.cache = {}
            self.cache_results = False
            self.nargs = {}

        def prune_configs(self, _kwargs):
            return [config]

        def _bench(self, *args, config, **kwargs):
            nonlocal benchmark_count
            with benchmark_count_lock:
                benchmark_count += 1
            time.sleep(0.01)
            return 1.0

        def pre_hook(self, _nargs, reset_only):
            assert reset_only is True

    autotuner = _FakeAutotuner()

    monkeypatch.setattr(triton_autotuner, "Autotuner", _FakeAutotuner)
    monkeypatch.setattr(triton_autotuner, "CacheFuture", None, raising=False)
    monkeypatch.setattr(nogil_patcher, "version", lambda _package_name: "3.7.0")

    nogil_patcher.patch_triton_autotuner()
    barrier = threading.Barrier(16)
    results = [None] * barrier.parties
    errors = []

    def lookup(index):
        try:
            barrier.wait()
            results[index] = autotuner._get_config_for_key("new", (), {})
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=lookup, args=(index,)) for index in range(barrier.parties)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert benchmark_count == 1
    assert all(result[0] is config for result in results)
    assert all(result[1] is False for result in results)
    assert autotuner._cache == {"new": config}
    assert autotuner._cache_futures == {}


def test_package_init_uses_triton_patch_api():
    init_py = Path(__file__).resolve().parents[1] / "gptqmodel" / "__init__.py"
    tree = ast.parse(init_py.read_text(encoding="utf-8"))

    call_names = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            call_names.append((node.func.value.id, node.func.attr))
        elif isinstance(node.func, ast.Name):
            call_names.append((node.func.id, ""))

    assert ("TritonPatch", "apply") in call_names
    assert ("patch_triton_autotuner", "") not in call_names


def _fake_jit_function(kernel_cache):
    class _FakeJITFunction:
        def __init__(self):
            self.device_caches = {0: (kernel_cache, None, None, None, None)}

    return _FakeJITFunction()


def test_single_flight_triton_compile_compiles_one_key_once():
    kernel_cache = {}
    jit_function = _fake_jit_function(kernel_cache)
    calls = []
    calls_lock = threading.Lock()
    start = threading.Barrier(8)

    def compile_fn():
        with calls_lock:
            calls.append(1)
        time.sleep(0.05)
        kernel = object()
        kernel_cache[("same",)] = kernel
        return kernel

    results = []
    results_lock = threading.Lock()

    def worker():
        start.wait(timeout=10)
        kernel = nogil_patcher._single_flight_triton_compile(
            jit_function, ("same",), 0, compile_fn
        )
        with results_lock:
            results.append(kernel)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert len(calls) == 1
    assert len(results) == 8
    assert all(result is results[0] for result in results)


def test_single_flight_triton_compile_serializes_distinct_keys():
    kernel_cache = {}
    jit_function = _fake_jit_function(kernel_cache)
    active = 0
    max_active = 0
    state_lock = threading.Lock()

    def compile_fn():
        nonlocal active, max_active
        with state_lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.03)
        with state_lock:
            active -= 1
        return object()

    threads = [
        threading.Thread(
            target=nogil_patcher._single_flight_triton_compile,
            args=(jit_function, (index,), 0, compile_fn),
        )
        for index in range(6)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    # libtriton's compiler is process-global, so unrelated kernels must not
    # compile concurrently either.
    assert max_active == 1


def test_triton_jit_compile_patch_is_installed():
    jit_module = pytest.importorskip("triton.runtime.jit")
    nogil_patcher.patch_triton_jit_compile()

    assert getattr(jit_module.JITFunction, "_gptqmodel_compile_serialized", False)


def test_macaq_triton_kernels_use_the_patched_jit_function():
    jit_module = pytest.importorskip("triton.runtime.jit")
    macaq = pytest.importorskip("gptqmodel.utils.macaq_triton")
    if not getattr(macaq, "TRITON_AVAILABLE", False):
        pytest.skip("triton is not available")

    # MACaQ launches through the same JITFunction.run path as AWQ, so the
    # process-wide compile lock covers it without a second patch.
    assert isinstance(macaq._dense_w4a4_kernel, jit_module.JITFunction)
    assert getattr(jit_module.JITFunction, "_gptqmodel_compile_serialized", False)
