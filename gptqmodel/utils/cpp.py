# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import hashlib
import logging
import math
import os
import shutil
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Optional, Sequence

import pcre
import torch
from torch.utils.cpp_extension import CUDA_HOME, _get_build_directory, _get_cuda_arch_flags, load

from .env import env_flag
from .jit_compile_baselines import get_jit_compile_baseline_seconds
from .logger import setup_logger


log = logging.getLogger(__name__)

# One process-local lock serializes every torch.ops JIT cache mutation and
# compile so concurrent startup paths never overlap toolchain work across
# different extensions.
_TORCH_OPS_JIT_LOCK = threading.Lock()

_PACK_BLOCK_EXTENSION: Optional[bool] = None
_PACK_BLOCK_EXTENSION_INITIALISED = False

_PACK_BLOCK_TORCH_OPS_EXTENSION = None
_FLOATX_CPU_TORCH_OPS_EXTENSION = None

_cpp_ext_lock = _TORCH_OPS_JIT_LOCK

# Used to track whether cleanup has been done already
_cpp_ext_initialized = False

_SHARED_LIBRARY_SUFFIXES = (".so", ".pyd", ".dylib", ".dll")
_COMPILE_PROGRESS_TOTAL_STEPS = 100
_COMPILE_PROGRESS_INTERVAL_SECONDS = 1.0
_LOCAL_INCLUDE_PATTERN = pcre.compile(
    r'^\s*#\s*include\s+"([^"]+)"',
    flags=pcre.Flag.MULTILINE,
)
_NVCC_RELEASE_PATTERN = pcre.compile(r"release\s+(\d+)\.(\d+)")
_NVCC_VERSION_LOCK = threading.Lock()
_NVCC_VERSION_CACHE: tuple[int, int] | None = None
# Default NVCC internal threading for JIT builds. This is based on clean-build
# timings collected on an AMD Zen 3 CPU running at 2.2 GHz, where 8 threads was
# the best overall tradeoff across Marlin, AWQ, QQQ, ExLlama, and ParoQuant.
_DEFAULT_NVCC_THREADS = "8"
_GLOBAL_KERNEL_REBUILD_ENV = "GPTQMODEL_KERNEL_REBUILD"
_TORCH_OPS_BUILD_ROOT_ENV = "GPTQMODEL_TORCH_EXTENSIONS_DIR"


def _nvcc_path() -> Optional[str]:
    return shutil.which("nvcc")


def _nvcc_text(*args: str) -> str:
    nvcc_path = _nvcc_path()
    if not nvcc_path:
        return ""

    try:
        result = subprocess.run(
            [nvcc_path, *args],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return ""

    return ((result.stdout or "") + "\n" + (result.stderr or "")).strip()


def _nvcc_version() -> tuple[int, int]:
    global _NVCC_VERSION_CACHE

    with _NVCC_VERSION_LOCK:
        if _NVCC_VERSION_CACHE is not None:
            return _NVCC_VERSION_CACHE

        version_text = _nvcc_text("--version")
        match = _NVCC_RELEASE_PATTERN.search(version_text)
        if match:
            _NVCC_VERSION_CACHE = (int(match.group(1)), int(match.group(2)))
        else:
            _NVCC_VERSION_CACHE = (0, 0)
        return _NVCC_VERSION_CACHE


def nvcc_version_at_least(major: int, minor: int) -> bool:
    return _nvcc_version() >= (major, minor)


def is_nvcc_compatible() -> bool:
    """Return whether nvcc is new enough for required JIT kernel flags."""

    return nvcc_version_at_least(12, 8)


def _format_compile_duration_seconds(seconds: float) -> str:
    """Format one duration compactly for user-facing compile progress text."""

    seconds_value = max(0.0, float(seconds))
    if seconds_value < 10.0:
        return f"{seconds_value:.1f}s"
    return f"{seconds_value:.0f}s"


def _compile_progress_ratio(elapsed_seconds: float, baseline_seconds: float) -> float:
    """Map elapsed compile time onto a progress ratio that never reaches 100% early."""

    baseline = max(float(baseline_seconds), 0.0)
    elapsed = max(float(elapsed_seconds), 0.0)
    if baseline <= 0.0 or elapsed <= 0.0:
        return 0.0
    if elapsed <= baseline:
        return min(0.95 * (elapsed / baseline), 0.95)

    overrun = elapsed - baseline
    tail_ratio = 1.0 - math.exp(-overrun / max(baseline, 1.0))
    return min(0.95 + (0.04 * tail_ratio), 0.99)


def _compile_progress_step(
    elapsed_seconds: float,
    baseline_seconds: float,
    *,
    total_steps: int = _COMPILE_PROGRESS_TOTAL_STEPS,
) -> int:
    """Convert one elapsed/baseline pair into a bounded manual progress step."""

    if total_steps <= 1:
        return 0
    ratio = _compile_progress_ratio(elapsed_seconds, baseline_seconds)
    return max(0, min(total_steps - 1, int(math.floor(ratio * (total_steps - 1)))))


def _compile_progress_subtitle(elapsed_seconds: float, baseline_seconds: float) -> str:
    """Describe compile elapsed time against the recorded reference baseline."""

    elapsed = max(float(elapsed_seconds), 0.0)
    baseline = max(float(baseline_seconds), 0.0)
    if baseline <= 0.0:
        return f"elapsed {_format_compile_duration_seconds(elapsed)}"
    if elapsed <= baseline:
        return (
            f"elapsed {_format_compile_duration_seconds(elapsed)} / "
            f"estimated ~{_format_compile_duration_seconds(baseline)}"
        )
    return (
        f"elapsed {_format_compile_duration_seconds(elapsed)} / "
        f"estimated ~{_format_compile_duration_seconds(baseline)} "
        f"(+{_format_compile_duration_seconds(elapsed - baseline)})"
    )


def _compile_baseline_summary(elapsed_seconds: float, baseline_seconds: Optional[float]) -> str:
    """Format a concise compile-vs-baseline summary for durable log lines."""

    elapsed = _format_compile_duration_seconds(elapsed_seconds)
    if baseline_seconds is None or baseline_seconds <= 0:
        return f"in {elapsed}"

    baseline = _format_compile_duration_seconds(baseline_seconds)
    delta = elapsed_seconds - baseline_seconds
    delta_text = _format_compile_duration_seconds(abs(delta))
    if abs(delta) < 0.05:
        return f"in {elapsed} (estimated ~{baseline})"
    sign = "+" if delta >= 0 else "-"
    return f"in {elapsed} (estimated ~{baseline}, {sign}{delta_text})"


class _CompileProgressDisplay:
    """Render either a baseline-backed progress bar or a fallback spinner."""

    def __init__(
        self,
        *,
        logger,
        title: str,
        baseline_seconds: Optional[float],
    ) -> None:
        self._logger = logger
        self._title = title
        self._baseline_seconds = (
            None if baseline_seconds is None or baseline_seconds <= 0 else float(baseline_seconds)
        )
        self._started = time.perf_counter()
        self._stop_event: Optional[threading.Event] = None
        self._thread: Optional[threading.Thread] = None
        self._progress = None
        self._spinner = None
        self._render_lock = threading.Lock()
        self._closed = False

        if self._baseline_seconds is None:
            self._spinner = logger.spinner(title=title, interval=_COMPILE_PROGRESS_INTERVAL_SECONDS)
            return

        progress = logger.pb(range(_COMPILE_PROGRESS_TOTAL_STEPS)).manual().set(show_left_steps=False)
        progress.title(title)
        progress.subtitle(_compile_progress_subtitle(0.0, self._baseline_seconds))
        progress.draw(force=True)
        self._progress = progress
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._refresh_loop,
            name=f"jit-compile-progress-{title}",
            daemon=True,
        )
        self._thread.start()

    def elapsed_seconds(self) -> float:
        return max(0.0, time.perf_counter() - self._started)

    def _refresh_loop(self) -> None:
        assert self._stop_event is not None
        while not self._stop_event.wait(_COMPILE_PROGRESS_INTERVAL_SECONDS):
            self._draw_current(force=False)

    def _draw_current(self, *, force: bool) -> None:
        if self._progress is None or self._baseline_seconds is None:
            return
        if self._closed:
            return
        with self._render_lock:
            if self._closed:
                return
            elapsed = self.elapsed_seconds()
            self._progress.current_iter_step = _compile_progress_step(elapsed, self._baseline_seconds)
            self._progress.subtitle(_compile_progress_subtitle(elapsed, self._baseline_seconds))
            self._progress.draw(force=force)

    def close(self, *, succeeded: bool, elapsed_seconds: Optional[float] = None) -> None:
        elapsed = self.elapsed_seconds() if elapsed_seconds is None else max(0.0, float(elapsed_seconds))
        if self._spinner is not None:
            self._spinner.close()
            return
        if self._stop_event is not None:
            self._stop_event.set()
        if self._progress is None or self._baseline_seconds is None:
            return
        # Completion is driven by the actual build result and elapsed time, not
        # by the estimated baseline. A faster-than-expected compile should exit
        # immediately and force the bar to its final state.
        self._closed = True
        with self._render_lock:
            self._progress.current_iter_step = (
                _COMPILE_PROGRESS_TOTAL_STEPS if succeeded else _compile_progress_step(elapsed, self._baseline_seconds)
            )
            self._progress.subtitle(_compile_progress_subtitle(elapsed, self._baseline_seconds))
            self._progress.draw(force=True)
            self._progress.close()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=0.05)


def default_torch_ops_build_root(subdir: str) -> Path:
    """Return the default on-disk cache root for torch.ops JIT extensions."""

    override_root = os.getenv(_TORCH_OPS_BUILD_ROOT_ENV)
    if override_root:
        return Path(override_root).expanduser() / subdir
    return Path.home() / ".cache" / "gptqmodel" / "torch_extensions" / subdir


def _dedupe_path_strings(paths: Sequence[str]) -> list[str]:
    """Normalize and deduplicate include/library path strings while preserving order."""

    deduped: list[str] = []
    seen: set[str] = set()
    for raw_path in paths:
        normalized = str(Path(raw_path).expanduser())
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def detected_cuda_wheel_include_paths() -> list[str]:
    """Discover CUDA developer headers shipped via NVIDIA Python wheels."""

    try:
        import nvidia  # type: ignore
    except Exception:
        return []

    include_paths: list[str] = []
    for base_text in getattr(nvidia, "__path__", []):
        base_path = Path(base_text)
        candidate_paths = list(base_path.glob("cu*/include"))
        candidate_paths.extend(base_path.glob("*/include"))
        for candidate in sorted(candidate_paths):
            if candidate.is_dir():
                include_paths.append(str(candidate))
    return _dedupe_path_strings(include_paths)


def _resolve_local_include_path(
    include_name: str,
    *,
    including_path: Path,
    include_search_roots: Sequence[Path],
) -> Optional[Path]:
    """Resolve one quoted local include against the current file and explicit include roots."""

    include_path = Path(include_name).expanduser()
    if include_path.is_absolute():
        resolved = include_path.resolve(strict=False)
        return resolved if resolved.exists() else None

    search_roots = [including_path.parent, *include_search_roots]
    for root in search_roots:
        candidate = (root / include_path).resolve(strict=False)
        if candidate.exists():
            return candidate
    return None


def detected_local_cuda_include_paths() -> list[str]:
    """Discover CUDA developer headers from the active local CUDA toolkit."""

    include_paths: list[str] = []

    if CUDA_HOME:
        candidate = Path(CUDA_HOME).expanduser() / "include"
        if candidate.is_dir():
            include_paths.append(str(candidate))

    cuda_path = os.getenv("CUDA_PATH")
    if cuda_path:
        candidate = Path(cuda_path).expanduser() / "include"
        if candidate.is_dir():
            include_paths.append(str(candidate))

    return _dedupe_path_strings(include_paths)


def _detected_local_cuda_has_required_headers(required_header_names: Sequence[str]) -> bool:
    """Return whether the detected local CUDA toolkit exposes every required header."""

    local_cuda_include_paths = detected_local_cuda_include_paths()
    if not local_cuda_include_paths:
        return False
    return all(
        any((Path(include_path) / header_name).is_file() for include_path in local_cuda_include_paths)
        for header_name in required_header_names
    )


def cuda_include_paths_with_fallback(
    include_paths: Sequence[str],
    *,
    required_header_names: Sequence[str] = (),
) -> list[str]:
    """Append NVIDIA wheel headers when the local CUDA toolkit is absent or incomplete."""

    resolved_include_paths = _dedupe_path_strings(include_paths)
    if not _detected_local_cuda_has_required_headers(required_header_names):
        resolved_include_paths.extend(detected_cuda_wheel_include_paths())
    return _dedupe_path_strings(resolved_include_paths)


_CUDA_ARCH_TOKEN_RE = pcre.compile(r"^(?P<major>\d+)\.(?P<minor>\d+)(?:\+PTX)?$")


def _supported_cuda_arch_pairs() -> list[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for arch in getattr(torch.cuda, "get_arch_list", lambda: [])():
        if not isinstance(arch, str) or not arch.startswith("sm_"):
            continue
        sm = arch.split("_", 1)[1].rstrip("af")
        if not sm.isdigit() or len(sm) < 2:
            continue
        pairs.add((int(sm[:-1]), int(sm[-1])))
    return sorted(pairs)


def _clamp_visible_cuda_capability(capability: tuple[int, int]) -> tuple[int, int]:
    supported = _supported_cuda_arch_pairs()
    if not supported:
        return capability
    return min(max(supported), capability)


def _visible_cuda_arch_tokens() -> list[str]:
    if not torch.cuda.is_available():
        return []

    tokens: list[str] = []
    seen: set[str] = set()
    for device_index in range(torch.cuda.device_count()):
        major, minor = _clamp_visible_cuda_capability(torch.cuda.get_device_capability(device_index))
        token = f"{major}.{minor}"
        if token not in seen:
            seen.add(token)
            tokens.append(token)
    return tokens


def _merge_cuda_arch_override_with_visible_caps(raw_override: str) -> str:
    requested_tokens: list[str] = []
    requested_bases: set[str] = set()
    for token in pcre.split(r"[;\s,]+", raw_override.strip()):
        if not token:
            continue
        requested_tokens.append(token)
        match = _CUDA_ARCH_TOKEN_RE.match(token)
        if match:
            requested_bases.add(f"{int(match.group('major'))}.{int(match.group('minor'))}")

    for token in _visible_cuda_arch_tokens():
        if token not in requested_bases:
            requested_tokens.append(token)
            requested_bases.add(token)

    return ";".join(requested_tokens)


def _effective_cuda_arch_flags(*, merge_visible_caps: bool) -> list[str]:
    """Return the effective NVCC arch flags Torch will emit for this host."""

    override = os.getenv("TORCH_CUDA_ARCH_LIST")
    try:
        if override and merge_visible_caps:
            merged_override = _merge_cuda_arch_override_with_visible_caps(override)
            if merged_override != override:
                os.environ["TORCH_CUDA_ARCH_LIST"] = merged_override
                try:
                    return list(_get_cuda_arch_flags())
                finally:
                    os.environ["TORCH_CUDA_ARCH_LIST"] = override
        return list(_get_cuda_arch_flags())
    except Exception:
        return []


@contextmanager
def _temporary_merged_cuda_arch_override(*, enabled: bool = True):
    """Temporarily include the visible CUDA capability in manual arch overrides."""

    override = os.getenv("TORCH_CUDA_ARCH_LIST")
    if not enabled or not override:
        yield
        return

    merged_override = _merge_cuda_arch_override_with_visible_caps(override)
    if merged_override == override:
        yield
        return

    os.environ["TORCH_CUDA_ARCH_LIST"] = merged_override
    try:
        yield
    finally:
        os.environ["TORCH_CUDA_ARCH_LIST"] = override


def resolved_cuda_arch_flags() -> list[str]:
    """Return the effective NVCC arch flags Torch will emit for this host."""

    return _effective_cuda_arch_flags(merge_visible_caps=True)


def torch_cxx11_abi_flag() -> int:
    """Return the ABI mode local JIT extensions must match for this torch build."""

    return int(getattr(torch._C, "_GLIBCXX_USE_CXX11_ABI", 1))


def torch_cxx11_abi_define() -> str:
    """Return the compiler define that keeps local extensions ABI-compatible."""

    return f"-D_GLIBCXX_USE_CXX11_ABI={torch_cxx11_abi_flag()}"


def resolved_jit_opt_level(opt_level: str | None = "O3") -> str | None:
    """Resolve the effective JIT optimization level, honoring the global env override."""

    override = os.getenv("GPTQMODEL_NVCC_COMPILE_LEVEL")
    raw_level = override if override is not None else opt_level
    if raw_level is None:
        return None

    normalized = str(raw_level).strip()
    if not normalized:
        return None
    if normalized.startswith("-"):
        normalized = normalized[1:]
    normalized = normalized.upper()

    if normalized in {"NONE", "NOOPT", "NO_OPT", "OFF", "DISABLE", "0"}:
        return None
    if normalized in {"O0", "O1", "O2", "O3"}:
        return normalized
    raise ValueError(
        "GPTQMODEL_NVCC_COMPILE_LEVEL must be one of O0/O1/O2/O3 or NONE/NOOPT/OFF."
    )


def default_jit_cflags(
    *,
    opt_level: str | None = "O3",
    enable_bf16: bool = False,
    include_abi: bool = True,
) -> list[str]:
    """Return the common C++ compiler flags for torch.ops JIT extensions."""

    resolved_opt_level = resolved_jit_opt_level(opt_level)
    flags = ["-std=c++17"]
    if resolved_opt_level is not None:
        flags.insert(0, f"-{resolved_opt_level}")
    if enable_bf16:
        flags.append("-DENABLE_BF16")
    if include_abi:
        flags.append(torch_cxx11_abi_define())
    return flags


def default_jit_cuda_cflags(
    *,
    opt_level: str | None = "O3",
    enable_bf16: bool = False,
    include_abi: bool = True,
    include_lineinfo: bool = False,
    include_nvcc_threads: bool = True,
    include_ptxas_optimizations: bool = False,
    include_ptxas_verbosity: bool = True,
    include_fatbin_compression: bool = False,
    include_diag_suppress: bool = False,
    nvcc_threads: str | int | None = None,
) -> list[str]:
    """Return the common NVCC flags for torch.ops JIT CUDA extensions."""

    resolved_opt_level = resolved_jit_opt_level(opt_level)
    flags = default_jit_cflags(
        opt_level=resolved_opt_level,
        enable_bf16=enable_bf16,
        include_abi=include_abi,
    )

    if include_nvcc_threads:
        resolved_nvcc_threads = str(nvcc_threads) if nvcc_threads is not None else os.getenv("NVCC_THREADS", _DEFAULT_NVCC_THREADS)
        flags.extend(["--threads", resolved_nvcc_threads])
        if resolved_opt_level is not None:
            optimization_level = (
                resolved_opt_level[1:] if resolved_opt_level.startswith("O") else resolved_opt_level
            )
            flags.append(f"--optimize={optimization_level}")
    if include_ptxas_optimizations:
        ptxas_flags = ["-v"] if include_ptxas_verbosity else []
        if resolved_opt_level is not None:
            ptxas_flags.append(f"-{resolved_opt_level}")
        ptxas_flags.append("-dlcm=ca")
        flags.extend(["-Xptxas", ",".join(ptxas_flags)])
    if include_lineinfo:
        flags.append("-lineinfo")
    if include_fatbin_compression:
        flags.extend(["-Xfatbin", "-compress-all"])
    if include_diag_suppress:
        flags.append("-diag-suppress=179,39,177")
    return flags


class TorchOpsJitExtension:
    """Manage one torch.ops JIT extension with shared cache and rebuild policy."""

    def __init__(
        self,
        *,
        name: str,
        namespace: str,
        required_ops: Sequence[str],
        sources: Sequence[str] | Callable[[], Sequence[str]],
        build_root_env: Optional[str],
        default_build_root: Path | str | Callable[[], Path | str],
        display_name: str,
        extra_cflags: Optional[Sequence[str] | Callable[[], Sequence[str]]] = None,
        extra_cuda_cflags: Optional[Sequence[str] | Callable[[], Sequence[str]]] = None,
        extra_include_paths: Optional[Sequence[str] | Callable[[], Sequence[str]]] = None,
        extra_ldflags: Optional[Sequence[str] | Callable[[], Sequence[str]]] = None,
        force_rebuild_env: Optional[str] = None,
        verbose_env: Optional[str] = None,
        requires_cuda: bool = False,
        merge_visible_cuda_arch_override: bool = True,
        binary_names: Optional[Sequence[str]] = None,
    ) -> None:
        self.name = name
        self.namespace = namespace
        self.required_ops = tuple(required_ops)
        self.sources = sources
        self.build_root_env = build_root_env
        self.default_build_root = default_build_root
        self.display_name = display_name
        self.extra_cflags = extra_cflags
        self.extra_cuda_cflags = extra_cuda_cflags
        self.extra_include_paths = extra_include_paths
        self.extra_ldflags = extra_ldflags
        self.force_rebuild_env = force_rebuild_env
        self.verbose_env = verbose_env
        self.requires_cuda = bool(requires_cuda)
        self.merge_visible_cuda_arch_override = bool(merge_visible_cuda_arch_override)
        self.binary_names = tuple(binary_names or (name,))
        self.compile_baseline_seconds = get_jit_compile_baseline_seconds(name)
        self._load_attempted = False
        self._load_result = False
        self._last_error = ""
        self._namespace_cache: Optional[object] = None
        self._op_cache: dict[str, object] = {}
        self._lock = self._get_shared_lock()

    @classmethod
    def _get_shared_lock(cls) -> threading.Lock:
        """Reuse the single process-local lock for every JIT extension."""
        return _TORCH_OPS_JIT_LOCK

    def _resolve_path(self, value: Path | str | Callable[[], Path | str]) -> Path:
        resolved = value() if callable(value) else value
        return Path(resolved).expanduser()

    def _resolve_sequence(
        self,
        value: Optional[Sequence[str] | Callable[[], Sequence[str]]],
    ) -> list[str]:
        if value is None:
            return []
        resolved = value() if callable(value) else value
        return [str(item) for item in resolved]

    def _resolved_extra_include_paths(self) -> list[str]:
        """Resolve explicit include paths and append CUDA wheel headers when needed."""

        include_paths = self._resolve_sequence(self.extra_include_paths)
        if not self.requires_cuda:
            return _dedupe_path_strings(include_paths)
        return cuda_include_paths_with_fallback(include_paths)

    def base_build_root(self) -> Path:
        """Return the user-visible cache root before applying the loader fingerprint."""

        override = os.getenv(self.build_root_env) if self.build_root_env else None
        if override:
            return Path(override).expanduser()
        return self._resolve_path(self.default_build_root)

    def _source_cache_fingerprint_payload(self, source: str, include_paths: Sequence[str]) -> list[str]:
        """Hash one source file plus recursively discovered quoted local includes."""

        payload: list[str] = []
        visited: set[Path] = set()
        include_search_roots = [Path(path).expanduser().resolve(strict=False) for path in include_paths]

        def visit(path: Path) -> None:
            normalized = path.expanduser().resolve(strict=False)
            if normalized in visited:
                return
            visited.add(normalized)
            payload.append(str(normalized))

            if not normalized.exists():
                payload.append("missing")
                return

            try:
                source_bytes = normalized.read_bytes()
            except OSError as exc:
                payload.append(f"read_error={type(exc).__name__}")
                return

            payload.append(hashlib.sha256(source_bytes).hexdigest())

            source_text = source_bytes.decode("utf-8", errors="ignore")
            for include_name in _LOCAL_INCLUDE_PATTERN.findall(source_text):
                included_path = _resolve_local_include_path(
                    include_name,
                    including_path=normalized,
                    include_search_roots=include_search_roots,
                )
                if included_path is None:
                    payload.append(f"missing_include={normalized}:{include_name}")
                    continue
                visit(included_path)

        visit(Path(source))
        return payload

    def _cache_fingerprint(self) -> str:
        """Hash the effective op surface and source metadata to avoid stale cache reuse."""

        payload: list[str] = [self.name, self.namespace, *self.required_ops]
        payload.append(f"python={sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
        payload.append(f"torch={torch.__version__}")
        payload.append(f"torch_cuda={torch.version.cuda or 'none'}")
        payload.extend(self._cuda_cache_fingerprint_payload())
        include_paths = self._resolved_extra_include_paths()
        for source in self._resolve_sequence(self.sources):
            payload.extend(self._source_cache_fingerprint_payload(source, include_paths))

        payload.extend(self._resolve_sequence(self.extra_cflags))
        payload.extend(self._resolve_sequence(self.extra_cuda_cflags))
        payload.extend(include_paths)
        payload.extend(self._resolve_sequence(self.extra_ldflags))
        digest = hashlib.sha256("\0".join(payload).encode("utf-8")).hexdigest()
        return digest[:16]

    def _cuda_cache_fingerprint_payload(self) -> list[str]:
        """Capture the effective CUDA target set so cached binaries stay device-compatible."""

        if not self.requires_cuda:
            return ["cuda_ext=0"]

        payload = ["cuda_ext=1"]
        override = os.getenv("TORCH_CUDA_ARCH_LIST")
        if override:
            payload.append(f"arch_list={override}")
            arch_flags = _effective_cuda_arch_flags(
                merge_visible_caps=self.merge_visible_cuda_arch_override
            )
            if arch_flags:
                payload.append(f"resolved_arch_flags={','.join(arch_flags)}")
            return payload

        if not torch.cuda.is_available():
            payload.append("cuda_available=0")
            return payload

        capabilities: set[str] = set()
        for device_index in range(torch.cuda.device_count()):
            major, minor = torch.cuda.get_device_capability(device_index)
            capabilities.add(f"{major}.{minor}")

        if not capabilities:
            payload.append("visible_caps=none")
        else:
            payload.append(f"visible_caps={','.join(sorted(capabilities))}")

        arch_flags = resolved_cuda_arch_flags()
        if arch_flags:
            payload.append(f"resolved_arch_flags={','.join(arch_flags)}")
        return payload

    def build_root(self) -> Path:
        """Return the fingerprinted filesystem directory that caches this JIT extension."""

        return self.base_build_root() / self._cache_fingerprint()

    def force_rebuild_enabled(self) -> bool:
        """Check whether this extension should ignore and replace cached binaries."""

        if env_flag(_GLOBAL_KERNEL_REBUILD_ENV, default=False):
            return True
        if not self.force_rebuild_env:
            return False
        return env_flag(self.force_rebuild_env, default=False)

    def _ops_available(self) -> bool:
        namespace = getattr(torch.ops, self.namespace, None)
        return namespace is not None and all(hasattr(namespace, op_name) for op_name in self.required_ops)

    def _refresh_runtime_cache(self) -> bool:
        namespace = getattr(torch.ops, self.namespace, None)
        if namespace is None:
            self._namespace_cache = None
            self._op_cache = {}
            return False
        missing = [op_name for op_name in self.required_ops if not hasattr(namespace, op_name)]
        if missing:
            self._namespace_cache = None
            self._op_cache = {}
            return False
        self._namespace_cache = namespace
        self._op_cache = {op_name: getattr(namespace, op_name) for op_name in self.required_ops}
        return True

    def _candidate_binary_paths(self, build_root: Path) -> list[Path]:
        seen: set[Path] = set()
        candidates: list[Path] = []
        for binary_name in self.binary_names:
            for suffix in _SHARED_LIBRARY_SUFFIXES:
                exact = build_root / f"{binary_name}{suffix}"
                if exact not in seen:
                    seen.add(exact)
                    candidates.append(exact)
                for match in sorted(build_root.glob(f"{binary_name}*{suffix}")):
                    if match not in seen:
                        seen.add(match)
                        candidates.append(match)
        return candidates

    def _try_load_prebuilt_library(self, build_root: Path) -> bool:
        for library_path in self._candidate_binary_paths(build_root):
            if not library_path.is_file():
                continue
            try:
                torch.ops.load_library(str(library_path))
                if self._ops_available():
                    return True
            except Exception as exc:  # pragma: no cover - binary/runtime mismatch depends on host
                log.debug("%s: failed to load cached torch.ops library %s: %s", self.display_name, library_path, exc)
        return False

    def clear_cache(self) -> None:
        """Best-effort cache clear for the next process-local JIT load attempt."""

        with self._lock:
            self._load_attempted = False
            self._load_result = False
            self._last_error = ""
            self._namespace_cache = None
            self._op_cache = {}
            build_root = self.base_build_root()
            if build_root.exists():
                shutil.rmtree(build_root, ignore_errors=True)

    def last_error_message(self) -> str:
        """Return the most recent human-readable load failure."""

        return self._last_error

    def load(self) -> bool:
        """Load the extension from cache or JIT-compile it on first use."""

        if self._load_attempted and self._load_result and not self.force_rebuild_enabled():
            return True

        if self._namespace_cache is not None and not self.force_rebuild_enabled():
            self._load_attempted = True
            self._load_result = True
            self._last_error = ""
            return True

        if self._ops_available():
            self._refresh_runtime_cache()
            self._load_attempted = True
            self._load_result = True
            self._last_error = ""
            return True

        if self.requires_cuda and not torch.cuda.is_available():
            self._load_attempted = True
            self._load_result = False
            self._last_error = f"{self.display_name}: CUDA is not available."
            return False

        with self._lock:
            force_rebuild = self.force_rebuild_enabled()
            if self._load_attempted and self._load_result and not force_rebuild:
                return True
            if self._namespace_cache is not None and not force_rebuild:
                self._load_attempted = True
                self._load_result = True
                self._last_error = ""
                return True
            if self._ops_available():
                self._refresh_runtime_cache()
                self._load_attempted = True
                self._load_result = True
                self._last_error = ""
                return True
            if self._load_attempted and not force_rebuild:
                return self._load_result
            build_root = self.build_root()
            base_build_root = self.base_build_root()

            if force_rebuild and base_build_root.exists():
                setup_logger().info(f"{self.display_name}: clearing cached JIT extension at `{base_build_root}`.")
                shutil.rmtree(base_build_root, ignore_errors=True)

            build_root.mkdir(parents=True, exist_ok=True)

            if not force_rebuild and self._try_load_prebuilt_library(build_root):
                self._load_attempted = True
                self._load_result = True
                self._last_error = ""
                return True

            logger = setup_logger()
            logger.info(f"{self.display_name}: compiling torch.ops JIT extension in `{build_root}`.")
            progress_display = _CompileProgressDisplay(
                logger=logger,
                title=f"Compiling extension: {self.display_name}...",
                baseline_seconds=self.compile_baseline_seconds,
            )
            started = time.perf_counter()
            build_invocation_succeeded = False
            try:
                resolved_sources = self._resolve_sequence(self.sources)
                extra_include_paths = self._resolved_extra_include_paths()
                kwargs = {
                    "name": self.name,
                    "sources": resolved_sources,
                    "build_directory": str(build_root),
                    "is_python_module": False,
                    "verbose": env_flag(self.verbose_env, default=False) if self.verbose_env else False,
                }
                extra_cflags = self._resolve_sequence(self.extra_cflags)
                if extra_cflags:
                    kwargs["extra_cflags"] = extra_cflags
                extra_cuda_cflags = self._resolve_sequence(self.extra_cuda_cflags)
                if extra_cuda_cflags:
                    kwargs["extra_cuda_cflags"] = extra_cuda_cflags
                if extra_include_paths:
                    kwargs["extra_include_paths"] = extra_include_paths
                extra_ldflags = self._resolve_sequence(self.extra_ldflags)
                if extra_ldflags:
                    kwargs["extra_ldflags"] = extra_ldflags
                with _temporary_merged_cuda_arch_override(
                    enabled=self.merge_visible_cuda_arch_override
                ):
                    load(**kwargs)
                build_invocation_succeeded = True
            except Exception as exc:  # pragma: no cover - build depends on host toolchain
                elapsed = time.perf_counter() - started
                self._load_attempted = True
                self._load_result = False
                self._last_error = f"{self.display_name}: failed to build torch.ops JIT extension: {exc}"
                log.debug("%s", self._last_error, exc_info=True)
                logger.info(
                    f"{self.display_name}: torch.ops JIT compilation failed "
                    f"{_compile_baseline_summary(elapsed, self.compile_baseline_seconds)}; using fallback path."
                )
                return False
            finally:
                elapsed = time.perf_counter() - started
                progress_display.close(succeeded=build_invocation_succeeded, elapsed_seconds=elapsed)

            elapsed = time.perf_counter() - started
            ready = self._refresh_runtime_cache() or self._try_load_prebuilt_library(build_root)
            self._load_attempted = True
            self._load_result = ready
            if ready:
                self._refresh_runtime_cache()
                self._last_error = ""
                logger.info(
                    f"{self.display_name}: torch.ops JIT extension ready "
                    f"{_compile_baseline_summary(elapsed, self.compile_baseline_seconds)}."
                )
                return True

            self._last_error = f"{self.display_name}: build completed but required torch.ops were not registered."
            logger.info(f"{self.display_name}: torch.ops JIT build finished without registering required ops.")
            return False

    def namespace_object(self) -> object:
        """Return the cached torch.ops namespace after loading this extension."""

        if self._namespace_cache is not None:
            return self._namespace_cache
        if not self.load():
            raise RuntimeError(self.last_error_message() or f"{self.display_name}: runtime unavailable.")
        if self._refresh_runtime_cache():
            return self._namespace_cache
        raise RuntimeError(f"{self.display_name}: required torch.ops namespace `{self.namespace}` is unavailable.")

    def op(self, op_name: str) -> object:
        """Return a cached torch.ops handle for one registered op."""

        cached = self._op_cache.get(op_name)
        if cached is not None:
            return cached
        namespace = self.namespace_object()
        if not hasattr(namespace, op_name):
            raise AttributeError(f"{self.display_name}: torch.ops `{self.namespace}.{op_name}` is unavailable.")
        op = getattr(namespace, op_name)
        self._op_cache[op_name] = op
        return op


def safe_load_cpp_ext(
        name,
        build_directory=None,
        verbose=False,
        **kwargs
):
    """
    A safe wrapper for torch.utils.cpp_extension.load() that removed old cached kernel on first load. Pytorch has a bug where it does not check previous cached kernel for current pytorch compatibility as it
    only does it a naive name check based on pytorch/cuda version but same pytorch version can have multiple variants (cpu, cuda, etc) which are not cross compatible resulting in hidden errors.
    """
    global _cpp_ext_initialized

    with _cpp_ext_lock:
        # First-time initialization cleanup
        if not _cpp_ext_initialized:
            build_directory = build_directory or _get_build_directory(name, verbose=verbose)
            if os.path.exists(build_directory):
                try:
                    shutil.rmtree(build_directory)
                    if verbose:
                        log.debug(f"[safe_cpp_extension_load] Removed old build directory: {build_directory}")
                except Exception as e:
                    log.error(f"[safe_cpp_extension_load] Failed to remove build directory: {e}")

            if not os.path.exists(build_directory):
                if verbose:
                    log.debug('Creating extension directory %s...', build_directory)
                # This is like mkdir -p, i.e. will also create parent directories.
                os.makedirs(build_directory, exist_ok=True)

            # Load the extension (JIT)
            load(
                name=name,
                build_directory=build_directory,
                **kwargs
            )

            _cpp_ext_initialized = True

        return


def _pack_block_source_path() -> Path:
    """Resolve the pack_block custom-op source file from source or editable installs."""

    project_root = Path(__file__).resolve().parents[2]
    source_path = project_root / "pack_block_cpu.cpp"
    if source_path.exists():
        return source_path
    return project_root / "gptqmodel_ext" / "pack_block_cpu.cpp"


def _floatx_cpu_source_path() -> Path:
    """Resolve the floatx CPU custom-op source file from source or editable installs."""

    project_root = Path(__file__).resolve().parents[2]
    source_path = project_root / "floatx_cpu.cpp"
    if source_path.exists():
        return source_path
    return project_root / "gptqmodel_ext" / "floatx_cpu.cpp"


def _pack_block_extension() -> TorchOpsJitExtension:
    """Build the shared pack_block torch.ops loader on first use."""

    global _PACK_BLOCK_TORCH_OPS_EXTENSION
    if _PACK_BLOCK_TORCH_OPS_EXTENSION is None:
        _PACK_BLOCK_TORCH_OPS_EXTENSION = TorchOpsJitExtension(
            name="gptqmodel_pack_block_cpu",
            namespace="gptqmodel",
            required_ops=("pack_block_cpu",),
            sources=lambda: [str(_pack_block_source_path())],
            build_root_env="GPTQMODEL_EXT_BUILD",
            default_build_root=lambda: default_torch_ops_build_root("pack_block_cpu"),
            display_name="pack_block_cpu",
            extra_cflags=["-O3", "-std=c++17"],
            extra_ldflags=[],
            verbose_env="GPTQMODEL_EXT_VERBOSE",
            requires_cuda=False,
        )
    return _PACK_BLOCK_TORCH_OPS_EXTENSION


def _floatx_cpu_extension() -> TorchOpsJitExtension:
    """Build the shared floatx CPU torch.ops loader on first use."""

    global _FLOATX_CPU_TORCH_OPS_EXTENSION
    if _FLOATX_CPU_TORCH_OPS_EXTENSION is None:
        _FLOATX_CPU_TORCH_OPS_EXTENSION = TorchOpsJitExtension(
            name="gptqmodel_floatx_cpu",
            namespace="gptqmodel_floatx",
            required_ops=("dequantize_fp8_cpu", "dequantize_fp4_cpu"),
            sources=lambda: [str(_floatx_cpu_source_path())],
            build_root_env="GPTQMODEL_EXT_BUILD",
            default_build_root=lambda: default_torch_ops_build_root("floatx_cpu"),
            display_name="floatx_cpu",
            extra_cflags=["-O3", "-std=c++17"],
            extra_ldflags=[],
            verbose_env="GPTQMODEL_EXT_VERBOSE",
            requires_cuda=False,
        )
    return _FLOATX_CPU_TORCH_OPS_EXTENSION


def load_pack_block_extension(*, verbose: bool = False) -> Optional[object]:
    """Ensure the pack_block CPU extension is built and loaded."""

    global _PACK_BLOCK_EXTENSION, _PACK_BLOCK_EXTENSION_INITIALISED

    if hasattr(torch.ops.gptqmodel, "pack_block_cpu"):
        _PACK_BLOCK_EXTENSION_INITIALISED = True
        _PACK_BLOCK_EXTENSION = True
        return _PACK_BLOCK_EXTENSION

    if _PACK_BLOCK_EXTENSION_INITIALISED and _PACK_BLOCK_EXTENSION:
        return _PACK_BLOCK_EXTENSION

    source_path = _pack_block_source_path()
    if not source_path.exists():
        log.debug("pack_block_cpu extension source not found at %s", source_path)
        _PACK_BLOCK_EXTENSION = None
        _PACK_BLOCK_EXTENSION_INITIALISED = True
        return None

    try:
        from gptqmodel import extension as extension_api

        previous_verbose = os.environ.get("GPTQMODEL_EXT_VERBOSE")
        if verbose:
            os.environ["GPTQMODEL_EXT_VERBOSE"] = "1"
        try:
            _PACK_BLOCK_EXTENSION = extension_api.load(name="pack_block_cpu")["pack_block_cpu"]
        finally:
            if verbose:
                if previous_verbose is None:
                    os.environ.pop("GPTQMODEL_EXT_VERBOSE", None)
                else:
                    os.environ["GPTQMODEL_EXT_VERBOSE"] = previous_verbose
        log.debug("pack_block_cpu extension loaded from %s", source_path)
    except Exception as exc:  # pragma: no cover - environment-specific
        log.debug("pack_block_cpu extension build failed: %s", exc)
        _PACK_BLOCK_EXTENSION = None
    _PACK_BLOCK_EXTENSION_INITIALISED = True
    return _PACK_BLOCK_EXTENSION


def load_floatx_cpu_extension(*, verbose: bool = False) -> Optional[object]:
    """Ensure the floatx CPU extension is built and loaded."""

    namespace = getattr(torch.ops, "gptqmodel_floatx", None)
    if namespace is not None and hasattr(namespace, "dequantize_fp8_cpu") and hasattr(namespace, "dequantize_fp4_cpu"):
        return namespace

    source_path = _floatx_cpu_source_path()
    if not source_path.exists():
        log.debug("floatx_cpu extension source not found at %s", source_path)
        return None

    try:
        from gptqmodel import extension as extension_api

        previous_verbose = os.environ.get("GPTQMODEL_EXT_VERBOSE")
        if verbose:
            os.environ["GPTQMODEL_EXT_VERBOSE"] = "1"
        try:
            extension = extension_api.load(name="floatx_cpu")["floatx_cpu"]
        finally:
            if verbose:
                if previous_verbose is None:
                    os.environ.pop("GPTQMODEL_EXT_VERBOSE", None)
                else:
                    os.environ["GPTQMODEL_EXT_VERBOSE"] = previous_verbose
        log.debug("floatx_cpu extension loaded from %s", source_path)
        return extension
    except Exception as exc:  # pragma: no cover - environment-specific
        log.debug("floatx_cpu extension build failed: %s", exc)
    return None
