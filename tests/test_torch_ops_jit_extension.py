# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

from gptqmodel.utils import cpp as cpp_module


class _FakeSpinner:
    """Capture spinner lifecycle so tests can assert compile-stage UX hooks."""

    def __init__(self, title: str):
        self.title = title
        self.closed = False

    def close(self):
        self.closed = True


class _FakeLogger:
    """Collect durable info logs and spinner titles without rendering output."""

    def __init__(self):
        self.info_messages: list[str] = []
        self.spinners: list[_FakeSpinner] = []

    def info(self, message: str):
        self.info_messages.append(message)

    def spinner(self, title: str = "", *, interval: float = 0.5, tail_length: int = 4):
        del interval, tail_length
        spinner = _FakeSpinner(title)
        self.spinners.append(spinner)
        return spinner


def _make_loader(tmp_path: Path, **overrides) -> cpp_module.TorchOpsJitExtension:
    """Construct a shared torch.ops loader with a disposable build root."""

    params = {
        "name": "unit_test_ops",
        "namespace": "unit_test_ns",
        "required_ops": ("kernel",),
        "sources": ["unit_test.cpp"],
        "build_root_env": "UNIT_TEST_BUILD_ROOT",
        "default_build_root": lambda: tmp_path / "jit_build",
        "display_name": "Unit Test Kernel",
        "requires_cuda": False,
    }
    params.update(overrides)
    return cpp_module.TorchOpsJitExtension(**params)


def test_default_jit_cflags_allow_noopt(monkeypatch):
    """Guard the noopt path so callers can intentionally omit all `-O*` flags."""

    monkeypatch.delenv("GPTQMODEL_NVCC_COMPILE_LEVEL", raising=False)

    flags = cpp_module.default_jit_cflags(opt_level=None)

    assert "-std=c++17" in flags
    assert not any(flag.startswith("-O") for flag in flags)


def test_default_jit_cuda_cflags_respect_o2_override(monkeypatch):
    """Guard the global override so kernels can be forced onto one explicit optimization level."""

    monkeypatch.setenv("GPTQMODEL_NVCC_COMPILE_LEVEL", "O2")

    flags = cpp_module.default_jit_cuda_cflags(
        opt_level="O3",
        include_nvcc_threads=True,
        include_ptxas_optimizations=True,
    )

    assert "-O2" in flags
    assert "-O3" not in flags
    assert "--optimize=2" in flags
    assert flags[flags.index("-Xptxas") + 1] == "-v,-O2,-dlcm=ca"


def test_default_jit_cuda_cflags_respect_noopt_override(monkeypatch):
    """Guard the noopt override so users can disable every emitted `-O*` flag when needed."""

    monkeypatch.setenv("GPTQMODEL_NVCC_COMPILE_LEVEL", "NONE")

    flags = cpp_module.default_jit_cuda_cflags(
        opt_level="O3",
        include_nvcc_threads=True,
        include_ptxas_optimizations=True,
    )

    assert not any(flag.startswith("-O") for flag in flags)
    assert not any(flag.startswith("--optimize=") for flag in flags)
    assert flags[flags.index("-Xptxas") + 1] == "-v,-dlcm=ca"


def test_default_jit_cuda_cflags_allow_quiet_ptxas(monkeypatch):
    """Guard per-kernel PTXAS verbosity overrides so AWQ can suppress giant compile logs."""

    monkeypatch.delenv("GPTQMODEL_NVCC_COMPILE_LEVEL", raising=False)

    flags = cpp_module.default_jit_cuda_cflags(
        include_ptxas_optimizations=True,
        include_ptxas_verbosity=False,
    )

    assert flags[flags.index("-Xptxas") + 1] == "-O3,-dlcm=ca"


def test_detected_cuda_wheel_include_paths_discovers_merged_and_split_layouts(monkeypatch, tmp_path):
    """Guard CUDA wheel header discovery so JIT kernels can see NVIDIA pip headers."""

    nvidia_root = tmp_path / "site-packages" / "nvidia"
    (nvidia_root / "cu13" / "include").mkdir(parents=True)
    (nvidia_root / "cusparse" / "include").mkdir(parents=True)
    (nvidia_root / "cublas" / "include").mkdir(parents=True)

    fake_nvidia = type("FakeNvidia", (), {"__path__": [str(nvidia_root)]})()
    monkeypatch.setitem(sys.modules, "nvidia", fake_nvidia)

    assert cpp_module.detected_cuda_wheel_include_paths() == [
        str(nvidia_root / "cu13" / "include"),
        str(nvidia_root / "cublas" / "include"),
        str(nvidia_root / "cusparse" / "include"),
    ]


def test_detected_local_cuda_include_paths_prefers_cuda_home(monkeypatch, tmp_path):
    """Guard local CUDA header discovery so JIT builds can skip wheel headers when toolkit headers exist."""

    cuda_home = tmp_path / "cuda-toolkit"
    (cuda_home / "include").mkdir(parents=True)

    monkeypatch.setattr(cpp_module, "CUDA_HOME", str(cuda_home))
    monkeypatch.delenv("CUDA_PATH", raising=False)

    assert cpp_module.detected_local_cuda_include_paths() == [str(cuda_home / "include")]


def test_cuda_include_paths_with_fallback_use_wheel_headers_when_local_cuda_is_incomplete(monkeypatch, tmp_path):
    """Guard shared CUDA header fallback so incomplete local toolkits still build JIT extensions."""

    local_cuda_include = tmp_path / "local_cuda_include"
    wheel_cuda_include = tmp_path / "wheel_cuda_include"
    local_cuda_include.mkdir()
    wheel_cuda_include.mkdir()
    (wheel_cuda_include / "cusparse.h").write_text("// stub", encoding="utf-8")

    monkeypatch.setattr(cpp_module, "detected_local_cuda_include_paths", lambda: [str(local_cuda_include)])
    monkeypatch.setattr(cpp_module, "detected_cuda_wheel_include_paths", lambda: [str(wheel_cuda_include)])

    include_paths = cpp_module.cuda_include_paths_with_fallback(
        ["/tmp/extension"],
        required_header_names=("cusparse.h",),
    )

    assert include_paths == ["/tmp/extension", str(wheel_cuda_include)]


def test_cuda_include_paths_with_fallback_skip_wheel_headers_when_local_cuda_has_required_headers(
    monkeypatch,
    tmp_path,
):
    """Guard shared CUDA header fallback so complete local toolkits do not mix in wheel headers."""

    local_cuda_include = tmp_path / "local_cuda_include"
    wheel_cuda_include = tmp_path / "wheel_cuda_include"
    local_cuda_include.mkdir()
    wheel_cuda_include.mkdir()
    (local_cuda_include / "cusparse.h").write_text("// stub", encoding="utf-8")

    monkeypatch.setattr(cpp_module, "detected_local_cuda_include_paths", lambda: [str(local_cuda_include)])
    monkeypatch.setattr(cpp_module, "detected_cuda_wheel_include_paths", lambda: [str(wheel_cuda_include)])

    include_paths = cpp_module.cuda_include_paths_with_fallback(
        ["/tmp/extension"],
        required_header_names=("cusparse.h",),
    )

    assert include_paths == ["/tmp/extension"]


def test_cuda_cache_fingerprint_payload_includes_resolved_arch_flags(monkeypatch, tmp_path):
    """Guard CUDA cache keys so stale binaries cannot cross architecture targets."""

    loader = _make_loader(tmp_path, requires_cuda=True)

    monkeypatch.delenv("TORCH_CUDA_ARCH_LIST", raising=False)
    monkeypatch.setattr(cpp_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(cpp_module.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(cpp_module.torch.cuda, "get_device_capability", lambda _index: (12, 0))
    monkeypatch.setattr(
        cpp_module,
        "resolved_cuda_arch_flags",
        lambda: [
            "-gencode=arch=compute_120,code=compute_120",
            "-gencode=arch=compute_120,code=sm_120",
        ],
    )

    payload = loader._cuda_cache_fingerprint_payload()

    assert payload == [
        "cuda_ext=1",
        "visible_caps=12.0",
        (
            "resolved_arch_flags="
            "-gencode=arch=compute_120,code=compute_120,"
            "-gencode=arch=compute_120,code=sm_120"
        ),
    ]


def test_default_torch_ops_build_root_ignores_removed_global_override(monkeypatch):
    monkeypatch.setenv("GPTQMODEL_EXT_BUILD_BASE", "/tmp/obsolete-jit-root")

    assert cpp_module.default_torch_ops_build_root("marlin") == (
        Path.home() / ".cache" / "gptqmodel" / "torch_extensions" / "marlin"
    )


def test_default_torch_ops_build_root_respects_ci_override(monkeypatch):
    monkeypatch.setenv("GPTQMODEL_TORCH_EXTENSIONS_DIR", "/tmp/gptqmodel-ci")

    assert cpp_module.default_torch_ops_build_root("marlin") == Path("/tmp/gptqmodel-ci") / "marlin"


def test_torch_ops_jit_extension_prefers_explicit_build_root_over_global_default(monkeypatch, tmp_path):
    loader = _make_loader(
        tmp_path,
        default_build_root=lambda: cpp_module.default_torch_ops_build_root("unit_test_ops"),
    )

    monkeypatch.setenv("GPTQMODEL_TORCH_EXTENSIONS_DIR", "/tmp/gptqmodel-ci")
    monkeypatch.setenv("UNIT_TEST_BUILD_ROOT", "/tmp/unit-test-override")

    assert loader.base_build_root() == Path("/tmp/unit-test-override")


def test_torch_ops_jit_extension_prefers_cached_binary(monkeypatch, tmp_path):
    """Guard cache reuse so startup skips expensive JIT rebuilds when ops are already built."""

    loader = _make_loader(tmp_path)
    build_root = loader.build_root()
    build_root.mkdir(parents=True)
    library_path = build_root / "unit_test_ops.so"
    library_path.write_bytes(b"placeholder")

    state = {"ready": False}
    load_library_calls = []
    compile_calls = []

    monkeypatch.setattr(loader, "_ops_available", lambda: state["ready"])

    def fake_load_library(path: str):
        load_library_calls.append(path)
        state["ready"] = True

    monkeypatch.setattr(cpp_module.torch.ops, "load_library", fake_load_library, raising=False)
    monkeypatch.setattr(cpp_module, "load", lambda **kwargs: compile_calls.append(kwargs) or None)

    assert loader.load() is True
    assert load_library_calls == [str(library_path)]
    assert compile_calls == []


def test_torch_ops_jit_extension_force_rebuild_clears_cache(monkeypatch, tmp_path):
    """Guard force-rebuild mode so stale cached libraries never short-circuit a requested rebuild."""

    loader = _make_loader(tmp_path, force_rebuild_env="UNIT_TEST_FORCE_REBUILD")
    build_root = loader.build_root()
    build_root.mkdir(parents=True)
    stale_library = build_root / "unit_test_ops.so"
    stale_library.write_bytes(b"stale")

    state = {"ready": False}
    compile_calls = []
    logger = _FakeLogger()
    runtime = type("RuntimeNamespace", (), {"kernel": object()})()

    monkeypatch.setenv("UNIT_TEST_FORCE_REBUILD", "1")
    monkeypatch.setattr(loader, "_ops_available", lambda: state["ready"])
    monkeypatch.setattr(cpp_module, "setup_logger", lambda: logger)
    monkeypatch.setattr(
        cpp_module.torch.ops,
        "load_library",
        lambda path: (_ for _ in ()).throw(AssertionError(f"unexpected cached load: {path}")),
        raising=False,
    )

    def fake_compile(**kwargs):
        compile_calls.append(kwargs)
        state["ready"] = True
        monkeypatch.setattr(cpp_module.torch.ops, "unit_test_ns", runtime, raising=False)

    monkeypatch.setattr(cpp_module, "load", fake_compile)

    assert loader.load() is True
    assert len(compile_calls) == 1
    assert stale_library.exists() is False
    assert any("clearing cached JIT extension" in message for message in logger.info_messages)


def test_torch_ops_jit_extension_global_kernel_rebuild_clears_cache(monkeypatch, tmp_path):
    """Guard the umbrella rebuild flag so every torch.ops extension gets cold-build behavior."""

    loader = _make_loader(tmp_path)
    build_root = loader.build_root()
    build_root.mkdir(parents=True)
    stale_library = build_root / "unit_test_ops.so"
    stale_library.write_bytes(b"stale")

    state = {"ready": False}
    compile_calls = []
    logger = _FakeLogger()
    runtime = type("RuntimeNamespace", (), {"kernel": object()})()

    monkeypatch.setenv("GPTQMODEL_KERNEL_REBUILD", "1")
    monkeypatch.setattr(loader, "_ops_available", lambda: state["ready"])
    monkeypatch.setattr(cpp_module, "setup_logger", lambda: logger)
    monkeypatch.setattr(
        cpp_module.torch.ops,
        "load_library",
        lambda path: (_ for _ in ()).throw(AssertionError(f"unexpected cached load: {path}")),
        raising=False,
    )

    def fake_compile(**kwargs):
        compile_calls.append(kwargs)
        state["ready"] = True
        monkeypatch.setattr(cpp_module.torch.ops, "unit_test_ns", runtime, raising=False)

    monkeypatch.setattr(cpp_module, "load", fake_compile)

    assert loader.load() is True
    assert len(compile_calls) == 1
    assert stale_library.exists() is False
    assert any("clearing cached JIT extension" in message for message in logger.info_messages)


def test_torch_ops_jit_extension_emits_spinner_logs_around_compile(monkeypatch, tmp_path):
    """Guard compile UX so users get explicit progress feedback before and after JIT build stalls."""

    loader = _make_loader(
        tmp_path,
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-lineinfo"],
        extra_include_paths=["/tmp/include"],
        extra_ldflags=["-lm"],
    )

    state = {"ready": False}
    logger = _FakeLogger()
    compile_calls = []
    runtime = type("RuntimeNamespace", (), {"kernel": object()})()

    monkeypatch.setattr(loader, "_ops_available", lambda: state["ready"])
    monkeypatch.setattr(cpp_module, "setup_logger", lambda: logger)

    def fake_compile(**kwargs):
        compile_calls.append(kwargs)
        state["ready"] = True
        monkeypatch.setattr(cpp_module.torch.ops, "unit_test_ns", runtime, raising=False)

    monkeypatch.setattr(cpp_module, "load", fake_compile)

    assert loader.load() is True
    assert len(compile_calls) == 1
    assert compile_calls[0]["is_python_module"] is False
    assert compile_calls[0]["sources"] == ["unit_test.cpp"]
    assert compile_calls[0]["extra_cflags"] == ["-O3"]
    assert compile_calls[0]["extra_cuda_cflags"] == ["-lineinfo"]
    assert compile_calls[0]["extra_include_paths"] == ["/tmp/include"]
    assert compile_calls[0]["extra_ldflags"] == ["-lm"]
    assert logger.spinners
    assert logger.spinners[0].title == "Compiling extension: Unit Test Kernel..."
    assert logger.spinners[0].closed is True
    assert any("compiling torch.ops JIT extension" in message for message in logger.info_messages)
    assert any("torch.ops JIT extension ready" in message for message in logger.info_messages)


def test_torch_ops_jit_extension_appends_detected_cuda_include_paths(monkeypatch, tmp_path):
    """Guard CUDA JIT kwargs so detected NVIDIA wheel headers reach the compiler."""

    loader = _make_loader(
        tmp_path,
        requires_cuda=True,
        extra_include_paths=["/tmp/include"],
    )

    state = {"ready": False}
    compile_calls = []
    runtime = type("RuntimeNamespace", (), {"kernel": object()})()

    monkeypatch.setattr(loader, "_ops_available", lambda: state["ready"])
    monkeypatch.setattr(cpp_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(cpp_module, "detected_local_cuda_include_paths", lambda: [])
    monkeypatch.setattr(
        cpp_module,
        "detected_cuda_wheel_include_paths",
        lambda: ["/tmp/nvidia/cu13/include", "/tmp/nvidia/cusparse/include"],
    )

    def fake_compile(**kwargs):
        compile_calls.append(kwargs)
        state["ready"] = True
        monkeypatch.setattr(cpp_module.torch.ops, "unit_test_ns", runtime, raising=False)

    monkeypatch.setattr(cpp_module, "load", fake_compile)

    assert loader.load() is True
    assert compile_calls[0]["extra_include_paths"] == [
        "/tmp/include",
        "/tmp/nvidia/cu13/include",
        "/tmp/nvidia/cusparse/include",
    ]


def test_torch_ops_jit_extension_merges_visible_capability_into_compile_override(monkeypatch, tmp_path):
    """Guard CUDA JIT builds so manual arch overrides still compile for the visible GPU."""

    loader = _make_loader(
        tmp_path,
        requires_cuda=True,
    )

    state = {"ready": False}
    compile_arch_lists = []
    runtime = type("RuntimeNamespace", (), {"kernel": object()})()

    monkeypatch.setattr(loader, "_ops_available", lambda: state["ready"])
    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "8.9+PTX")
    monkeypatch.setattr(cpp_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(cpp_module.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(cpp_module.torch.cuda, "get_device_capability", lambda device_index=0: (8, 0))
    monkeypatch.setattr(cpp_module.torch.cuda, "get_arch_list", lambda: ["sm_80", "sm_89"])
    monkeypatch.setattr(cpp_module, "_get_cuda_arch_flags", lambda: [os.environ["TORCH_CUDA_ARCH_LIST"]])

    def fake_compile(**kwargs):
        del kwargs
        compile_arch_lists.append(os.environ["TORCH_CUDA_ARCH_LIST"])
        state["ready"] = True
        monkeypatch.setattr(cpp_module.torch.ops, "unit_test_ns", runtime, raising=False)

    monkeypatch.setattr(cpp_module, "load", fake_compile)

    assert loader.load() is True
    assert compile_arch_lists == ["8.9+PTX;8.0"]
    assert os.environ["TORCH_CUDA_ARCH_LIST"] == "8.9+PTX"


def test_torch_ops_jit_extension_can_skip_visible_capability_merge_when_requested(monkeypatch, tmp_path):
    """Guard Hopper-only build flows so forced arch overrides can stay isolated from local GPUs."""

    loader = _make_loader(
        tmp_path,
        requires_cuda=True,
        merge_visible_cuda_arch_override=False,
    )

    state = {"ready": False}
    compile_arch_lists = []
    runtime = type("RuntimeNamespace", (), {"kernel": object()})()

    monkeypatch.setattr(loader, "_ops_available", lambda: state["ready"])
    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "9.0a")
    monkeypatch.setattr(cpp_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(cpp_module.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(cpp_module.torch.cuda, "get_device_capability", lambda device_index=0: (8, 0))
    monkeypatch.setattr(cpp_module.torch.cuda, "get_arch_list", lambda: ["sm_80", "sm_90a"])
    monkeypatch.setattr(cpp_module, "_get_cuda_arch_flags", lambda: [os.environ["TORCH_CUDA_ARCH_LIST"]])

    def fake_compile(**kwargs):
        del kwargs
        compile_arch_lists.append(os.environ["TORCH_CUDA_ARCH_LIST"])
        state["ready"] = True
        monkeypatch.setattr(cpp_module.torch.ops, "unit_test_ns", runtime, raising=False)

    monkeypatch.setattr(cpp_module, "load", fake_compile)

    assert loader.load() is True
    assert compile_arch_lists == ["9.0a"]
    assert os.environ["TORCH_CUDA_ARCH_LIST"] == "9.0a"


def test_torch_ops_jit_extension_uses_original_compile_paths(monkeypatch, tmp_path):
    local_root = tmp_path / "repo"
    (local_root / "src").mkdir(parents=True)
    (local_root / "include").mkdir(parents=True)
    source_path = local_root / "src" / "unit_test.cpp"
    include_path = local_root / "include"
    source_path.write_text('#include "unit_test.h"\nint kernel() { return 1; }\n', encoding="utf-8")
    (include_path / "unit_test.h").write_text("inline int unit_test_header() { return 1; }\n", encoding="utf-8")

    loader = _make_loader(
        tmp_path,
        sources=[str(source_path)],
        extra_include_paths=[str(include_path), "/usr/local/cuda/include"],
    )

    state = {"ready": False}
    compile_calls = []
    runtime = type("RuntimeNamespace", (), {"kernel": object()})()

    monkeypatch.setattr(loader, "_ops_available", lambda: state["ready"])

    def fake_compile(**kwargs):
        compile_calls.append(kwargs)
        state["ready"] = True
        monkeypatch.setattr(cpp_module.torch.ops, "unit_test_ns", runtime, raising=False)

    monkeypatch.setattr(cpp_module, "load", fake_compile)

    assert loader.load() is True
    assert compile_calls[0]["sources"] == [str(source_path)]
    assert compile_calls[0]["extra_include_paths"] == [str(include_path), "/usr/local/cuda/include"]


def test_torch_ops_jit_extension_skips_cuda_wheel_include_paths_when_local_headers_exist(monkeypatch, tmp_path):
    """Guard local-toolkit builds so wheel headers do not get mixed into one CUDA compile invocation."""

    loader = _make_loader(
        tmp_path,
        requires_cuda=True,
        extra_include_paths=["/tmp/include"],
    )

    state = {"ready": False}
    compile_calls = []
    runtime = type("RuntimeNamespace", (), {"kernel": object()})()

    monkeypatch.setattr(loader, "_ops_available", lambda: state["ready"])
    monkeypatch.setattr(cpp_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(cpp_module, "detected_local_cuda_include_paths", lambda: ["/usr/local/cuda/include"])
    monkeypatch.setattr(
        cpp_module,
        "detected_cuda_wheel_include_paths",
        lambda: ["/tmp/nvidia/cu13/include", "/tmp/nvidia/cusparse/include"],
    )

    def fake_compile(**kwargs):
        compile_calls.append(kwargs)
        state["ready"] = True
        monkeypatch.setattr(cpp_module.torch.ops, "unit_test_ns", runtime, raising=False)

    monkeypatch.setattr(cpp_module, "load", fake_compile)

    assert loader.load() is True
    assert compile_calls[0]["extra_include_paths"] == ["/tmp/include"]


def test_torch_ops_jit_extension_reuses_cached_namespace_after_first_load(monkeypatch, tmp_path):
    """Guard steady-state hot paths so repeated runtime checks skip torch.ops probing after first success."""

    loader = _make_loader(tmp_path)
    runtime = type("RuntimeNamespace", (), {"kernel": object()})()
    monkeypatch.setattr(cpp_module.torch.ops, "unit_test_ns", runtime, raising=False)

    state = {"ready": True}
    monkeypatch.setattr(loader, "_ops_available", lambda: state["ready"])

    assert loader.load() is True
    assert loader.op("kernel") is runtime.kernel

    def unexpected_probe():
        raise AssertionError("steady-state load should not re-probe torch.ops after first success")

    monkeypatch.setattr(loader, "_ops_available", unexpected_probe)

    assert loader.load() is True
    assert loader.namespace_object() is runtime
    assert loader.op("kernel") is runtime.kernel


def test_torch_ops_jit_extension_serializes_different_extensions_with_one_shared_lock(monkeypatch, tmp_path):
    """Guard that different JIT extensions do not compile in parallel."""

    loader_a = _make_loader(
        tmp_path,
        name="unit_test_ops_a",
        namespace="unit_test_ns_a",
    )
    loader_b = _make_loader(
        tmp_path,
        name="unit_test_ops_b",
        namespace="unit_test_ns_b",
    )

    states = {
        "unit_test_ops_a": False,
        "unit_test_ops_b": False,
    }
    runtime_a = type("RuntimeNamespaceA", (), {"kernel": object()})()
    runtime_b = type("RuntimeNamespaceB", (), {"kernel": object()})()
    logger = _FakeLogger()
    compile_tracker = {
        "active": 0,
        "max_active": 0,
    }
    compile_tracker_lock = threading.Lock()
    start_barrier = threading.Barrier(3)
    errors: list[Exception] = []

    monkeypatch.setattr(loader_a, "_ops_available", lambda: states["unit_test_ops_a"])
    monkeypatch.setattr(loader_b, "_ops_available", lambda: states["unit_test_ops_b"])
    monkeypatch.setattr(cpp_module.torch.ops, "unit_test_ns_a", runtime_a, raising=False)
    monkeypatch.setattr(cpp_module.torch.ops, "unit_test_ns_b", runtime_b, raising=False)
    monkeypatch.setattr(cpp_module, "setup_logger", lambda: logger)

    def fake_compile(**kwargs):
        extension_name = kwargs["name"]
        with compile_tracker_lock:
            compile_tracker["active"] += 1
            compile_tracker["max_active"] = max(compile_tracker["max_active"], compile_tracker["active"])
        time.sleep(0.02)
        states[extension_name] = True
        with compile_tracker_lock:
            compile_tracker["active"] -= 1

    monkeypatch.setattr(cpp_module, "load", fake_compile)

    def runner(loader):
        try:
            start_barrier.wait(timeout=1.0)
            assert loader.load() is True
        except Exception as exc:  # pragma: no cover - assertion path below
            errors.append(exc)

    threads = [
        threading.Thread(target=runner, args=(loader_a,)),
        threading.Thread(target=runner, args=(loader_b,)),
    ]
    for thread in threads:
        thread.start()
    start_barrier.wait(timeout=1.0)
    for thread in threads:
        thread.join()

    assert errors == []
    assert compile_tracker["max_active"] == 1


def test_torch_ops_jit_extension_cuda_fingerprint_tracks_visible_capabilities(monkeypatch, tmp_path):
    """Guard CUDA cache keys so binaries do not get reused across incompatible GPU architectures."""

    loader = _make_loader(tmp_path, requires_cuda=True)

    monkeypatch.delenv("TORCH_CUDA_ARCH_LIST", raising=False)
    monkeypatch.setattr(cpp_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(cpp_module.torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(
        cpp_module.torch.cuda,
        "get_device_capability",
        lambda device_index=0: (8, 9) if device_index == 0 else (8, 0),
    )
    first_build_root = loader.build_root()

    monkeypatch.setattr(
        cpp_module.torch.cuda,
        "get_device_capability",
        lambda device_index=0: (8, 9) if device_index == 0 else (9, 0),
    )
    second_build_root = loader.build_root()

    assert first_build_root != second_build_root


def test_torch_ops_jit_extension_cuda_fingerprint_prefers_arch_override(monkeypatch, tmp_path):
    """Guard explicit arch overrides so manual build targets produce isolated caches."""

    loader = _make_loader(tmp_path, requires_cuda=True)

    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "8.0")
    first_build_root = loader.build_root()

    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "8.9+PTX")
    second_build_root = loader.build_root()

    assert first_build_root != second_build_root


def test_resolved_cuda_arch_flags_appends_visible_capability_missing_from_override(monkeypatch):
    """Guard JIT arch resolution so manual overrides still build for the currently visible GPU."""

    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "8.0 8.6")
    monkeypatch.setattr(cpp_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(cpp_module.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(cpp_module.torch.cuda, "get_device_capability", lambda device_index=0: (7, 5))
    monkeypatch.setattr(cpp_module.torch.cuda, "get_arch_list", lambda: ["sm_75", "sm_80", "sm_86"])

    seen = {}

    def fake_get_cuda_arch_flags():
        seen["arch_list"] = os.environ["TORCH_CUDA_ARCH_LIST"]
        return [seen["arch_list"]]

    monkeypatch.setattr(cpp_module, "_get_cuda_arch_flags", fake_get_cuda_arch_flags)

    assert cpp_module.resolved_cuda_arch_flags() == ["8.0;8.6;7.5"]
    assert seen["arch_list"] == "8.0;8.6;7.5"
    assert os.environ["TORCH_CUDA_ARCH_LIST"] == "8.0 8.6"


def test_torch_ops_jit_extension_cuda_fingerprint_tracks_visible_capabilities_even_with_override(monkeypatch, tmp_path):
    """Guard cache keys so fixed arch overrides cannot reuse binaries across omitted visible GPU targets."""

    loader = _make_loader(tmp_path, requires_cuda=True)

    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "8.0")
    monkeypatch.setattr(cpp_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(cpp_module.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(cpp_module.torch.cuda, "get_arch_list", lambda: ["sm_75", "sm_80", "sm_89"])
    monkeypatch.setattr(cpp_module, "_get_cuda_arch_flags", lambda: [os.environ["TORCH_CUDA_ARCH_LIST"]])

    monkeypatch.setattr(cpp_module.torch.cuda, "get_device_capability", lambda device_index=0: (7, 5))
    first_build_root = loader.build_root()

    monkeypatch.setattr(cpp_module.torch.cuda, "get_device_capability", lambda device_index=0: (8, 9))
    second_build_root = loader.build_root()

    assert first_build_root != second_build_root


def test_torch_ops_jit_extension_cuda_fingerprint_tracks_detected_include_paths(monkeypatch, tmp_path):
    """Guard cache keys so CUDA wheel header layout changes invalidate old JIT binaries."""

    loader = _make_loader(tmp_path, requires_cuda=True)

    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "8.0")
    monkeypatch.setattr(cpp_module, "detected_local_cuda_include_paths", lambda: [])
    monkeypatch.setattr(cpp_module, "detected_cuda_wheel_include_paths", lambda: ["/tmp/nvidia/cu13/include"])
    first_build_root = loader.build_root()

    monkeypatch.setattr(
        cpp_module,
        "detected_cuda_wheel_include_paths",
        lambda: ["/tmp/nvidia/cu13/include", "/tmp/nvidia/cusparse/include"],
    )
    second_build_root = loader.build_root()

    assert first_build_root != second_build_root


def test_torch_ops_jit_extension_fingerprint_tracks_transitive_local_includes(tmp_path):
    """Guard cache keys so changes under quoted transitive includes rebuild stale entrypoint binaries."""

    source_root = tmp_path / "src"
    source_root.mkdir()
    entry = source_root / "entry.cpp"
    middle = source_root / "middle.h"
    leaf = source_root / "leaf.inc"

    entry.write_text('#include "middle.h"\nint kernel() { return answer(); }\n', encoding="utf-8")
    middle.write_text('#include "leaf.inc"\ninline int answer() { return ANSWER_VALUE; }\n', encoding="utf-8")
    leaf.write_text("#define ANSWER_VALUE 1\n", encoding="utf-8")

    loader = _make_loader(tmp_path, sources=[str(entry)])
    first_build_root = loader.build_root()

    leaf.write_text("#define ANSWER_VALUE 12345\n", encoding="utf-8")
    second_build_root = loader.build_root()

    assert first_build_root != second_build_root
