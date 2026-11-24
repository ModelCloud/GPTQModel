# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""GPT-QModel kernels serialize their own cold Triton compiles (issue #4094).

The compile lock lives on a GPT-QModel-owned ``JITFunction`` subclass built by
``gptqmodel.utils.triton_jit.jit``; Triton's own classes are never modified.
"""

import ast
import threading
import time
from pathlib import Path

import pytest

from gptqmodel.utils import triton_jit


_PACKAGE = Path(__file__).resolve().parents[1] / "gptqmodel"


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
        kernel = triton_jit._single_flight_triton_compile(
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
            target=triton_jit._single_flight_triton_compile,
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


def test_package_kernels_never_use_stock_triton_jit():
    offenders = []
    for path in sorted(_PACKAGE.rglob("*.py")):
        if path.name == "triton_jit.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in {"jit", "autotune"}:
                if isinstance(node.value, ast.Name) and node.value.id == "triton":
                    offenders.append(f"{path.relative_to(_PACKAGE.parent)}:{node.lineno}")
    assert offenders == [], offenders


def test_gptqmodel_jit_builds_a_package_owned_kernel_and_leaves_triton_alone():
    jit_module = pytest.importorskip("triton.runtime.jit")
    stock = dict(vars(jit_module.JITFunction))

    @triton_jit.jit
    def _kernel(x):
        pass

    @triton_jit.jit(do_not_specialize=["x"])
    def _kernel_with_options(x):
        pass

    owned = triton_jit.jit_function_class()
    assert type(_kernel) is owned and type(_kernel_with_options) is owned
    assert issubclass(owned, jit_module.JITFunction)
    assert owned.__module__ == "gptqmodel.utils.triton_jit"
    assert "_do_compile" in vars(owned)
    assert dict(vars(jit_module.JITFunction)) == stock


def test_gptqmodel_jit_routes_compiles_through_the_single_flight_lock(monkeypatch):
    jit_module = pytest.importorskip("triton.runtime.jit")
    seen = []

    def fake_parent(self, key, *args, **kwargs):
        return ("compiled", key)

    def spy(self, key, device, compile_fn):
        seen.append((key, device))
        return compile_fn()

    monkeypatch.setattr(jit_module.JITFunction, "_do_compile", fake_parent)
    monkeypatch.setattr(triton_jit, "_single_flight_triton_compile", spy)

    @triton_jit.jit
    def _kernel(x):
        pass

    assert _kernel._do_compile("k", "sig", 3) == ("compiled", "k")
    assert seen == [("k", 3)]


def test_macaq_triton_kernels_use_the_package_jit_function():
    pytest.importorskip("triton.runtime.jit")
    macaq = pytest.importorskip("gptqmodel.utils.macaq_triton")
    if not getattr(macaq, "TRITON_AVAILABLE", False):
        pytest.skip("triton is not available")
    jit_function = triton_jit.jit_function_class()
    assert type(macaq._dense_mx_gemm_kernel) is jit_function
    assert type(macaq._fp4_to_fp8_codes) is jit_function


def test_nvfp4_and_tt_block_float_kernels_use_the_package_jit_function():
    """A plain ``@triton.jit`` bypasses the single-flight cold-compile lock."""

    pytest.importorskip("triton.runtime.jit")
    if triton_jit._interpret_mode():
        pytest.skip("TRITON_INTERPRET returns Triton's own JITFunction")
    from gptqmodel.utils import nvfp4_triton, tt_blockfloat_triton

    jit_function = triton_jit.jit_function_class()
    bindings = (
        (
            nvfp4_triton,
            (
                "_decode_e2m1",
                "_decode_e4m3",
                "_nvfp4_w4a16_kernel",
                "_mxfp8_w8a16_kernel",
            ),
        ),
        (
            tt_blockfloat_triton,
            ("_decode_tile", "_gemv_partials", "_split_k_epilogue", "_gemm"),
        ),
    )
    for module, names in bindings:
        for name in names:
            kernel = getattr(module, name, None)
            assert kernel is not None, f"{module.__name__}.{name} was not built"
            assert type(kernel) is jit_function, f"{module.__name__}.{name}"
