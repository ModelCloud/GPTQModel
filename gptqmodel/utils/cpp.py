# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import logging
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Callable, Optional, Sequence

import torch
from torch.utils.cpp_extension import _get_build_directory, load

from .env import env_flag
from .logger import setup_logger


log = logging.getLogger(__name__)

# Shared per-extension locks serialize cache deletion and JIT compilation so
# concurrent startup paths do not race on the same build root.
_TORCH_OPS_EXTENSION_LOCKS: dict[str, threading.Lock] = {}
_TORCH_OPS_EXTENSION_LOCKS_GUARD = threading.Lock()

_PACK_BLOCK_EXTENSION: Optional[bool] = None
_PACK_BLOCK_EXTENSION_INITIALISED = False

_PACK_BLOCK_TORCH_OPS_EXTENSION = None

_cpp_ext_lock = threading.Lock()

# Used to track whether cleanup has been done already
_cpp_ext_initialized = False

_SHARED_LIBRARY_SUFFIXES = (".so", ".pyd", ".dylib", ".dll")


def default_torch_ops_build_root(subdir: str) -> Path:
    """Return the default on-disk cache root for torch.ops JIT extensions."""

    return Path.home() / ".cache" / "gptqmodel" / "torch_extensions" / subdir


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
        self.binary_names = tuple(binary_names or (name,))
        self._load_attempted = False
        self._load_result = False
        self._last_error = ""
        self._namespace_cache: Optional[object] = None
        self._op_cache: dict[str, object] = {}
        self._lock = self._get_shared_lock(name)

    @classmethod
    def _get_shared_lock(cls, extension_name: str) -> threading.Lock:
        """Reuse one process-local lock per extension build directory."""

        with _TORCH_OPS_EXTENSION_LOCKS_GUARD:
            lock = _TORCH_OPS_EXTENSION_LOCKS.get(extension_name)
            if lock is None:
                lock = threading.Lock()
                _TORCH_OPS_EXTENSION_LOCKS[extension_name] = lock
            return lock

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

    def build_root(self) -> Path:
        """Return the filesystem directory that caches this JIT extension."""

        override = os.getenv(self.build_root_env) if self.build_root_env else None
        if override:
            return Path(override).expanduser()
        return self._resolve_path(self.default_build_root)

    def force_rebuild_enabled(self) -> bool:
        """Check whether this extension should ignore and replace cached binaries."""

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
            build_root = self.build_root()
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

            if force_rebuild and build_root.exists():
                setup_logger().info(f"{self.display_name}: clearing cached JIT extension at `{build_root}`.")
                shutil.rmtree(build_root, ignore_errors=True)

            build_root.mkdir(parents=True, exist_ok=True)

            if not force_rebuild and self._try_load_prebuilt_library(build_root):
                self._load_attempted = True
                self._load_result = True
                self._last_error = ""
                return True

            logger = setup_logger()
            logger.info(f"{self.display_name}: compiling torch.ops JIT extension in `{build_root}`.")
            spinner = logger.spinner(title=f"{self.display_name}: compiling kernel...", interval=0.1)
            started = time.perf_counter()
            try:
                kwargs = {
                    "name": self.name,
                    "sources": self._resolve_sequence(self.sources),
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
                extra_include_paths = self._resolve_sequence(self.extra_include_paths)
                if extra_include_paths:
                    kwargs["extra_include_paths"] = extra_include_paths
                extra_ldflags = self._resolve_sequence(self.extra_ldflags)
                if extra_ldflags:
                    kwargs["extra_ldflags"] = extra_ldflags

                load(**kwargs)
            except Exception as exc:  # pragma: no cover - build depends on host toolchain
                self._load_attempted = True
                self._load_result = False
                self._last_error = f"{self.display_name}: failed to build torch.ops JIT extension: {exc}"
                log.debug("%s", self._last_error, exc_info=True)
                logger.info(f"{self.display_name}: torch.ops JIT compilation failed; using fallback path.")
                return False
            finally:
                spinner.close()

            elapsed = time.perf_counter() - started
            ready = self._refresh_runtime_cache() or self._try_load_prebuilt_library(build_root)
            self._load_attempted = True
            self._load_result = ready
            if ready:
                self._refresh_runtime_cache()
                self._last_error = ""
                logger.info(f"{self.display_name}: torch.ops JIT extension ready in {elapsed:.1f}s.")
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
        previous_verbose = os.environ.get("GPTQMODEL_EXT_VERBOSE")
        if verbose:
            os.environ["GPTQMODEL_EXT_VERBOSE"] = "1"
        try:
            _PACK_BLOCK_EXTENSION = _pack_block_extension().load()
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
