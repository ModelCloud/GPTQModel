# SPDX-FileCopyrightText: 2024-2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path, PurePosixPath
from typing import List, Optional

import pcre
import torch
from filelock import FileLock

from .cpp import (
    TorchOpsJitExtension,
    cuda_include_paths_with_fallback,
    default_jit_cflags,
    default_jit_cuda_cflags,
    default_torch_ops_build_root,
    is_nvcc_compatible,
    resolved_cuda_arch_flags,
)
from .env import env_flag
from .logger import setup_logger
from .marlin_scalar_type import ScalarType, scalar_types
from .rocm import IS_ROCM


log = setup_logger()

_MACHETE_OPS_NAME = "gptqmodel_machete_ops"
_MACHETE_OPS_NAMESPACE = "gptqmodel_machete"

_CUTLASS_VERSION = "4.7.1"
_CUTLASS_RELEASE_URL = f"https://github.com/NVIDIA/cutlass/archive/refs/tags/v{_CUTLASS_VERSION}.tar.gz"
_CUTLASS_ARCHIVE_SHA256 = "8290eb914cd5aaf4c665ee4108ba5bd65383cfee1296286a42a7ef711554d365"
_CUTLASS_VERSION_MARKER = ".gptqmodel_cutlass_version"
_MACHETE_COMPLETE_MARKER = ".gptqmodel_complete"
_MACHETE_MANIFEST_NAME = ".gptqmodel_manifest.json"
_MACHETE_GENERATION_CACHE_VERSION = 1
_MACHETE_CACHE_LOCK_TIMEOUT_SECONDS = 600
_CUTLASS_VERSION_DEFINE_PATTERN = pcre.compile(
    r"^\s*#define\s+CUTLASS_(MAJOR|MINOR|PATCH)\s+(\d+)\s*$",
    flags=pcre.Flag.MULTILINE,
)
_MACHETE_REQUIRED_COMPUTE_CAPABILITY = (9, 0)
_MACHETE_MIN_SHARED_MEMORY_PER_BLOCK_OPTIN = 204800
_MACHETE_SM90A_ARCH_FLAGS = (
    "-gencode=arch=compute_90a,code=sm_90a",
    "-gencode=arch=compute_90a,code=compute_90a",
)
_MACHETE_JIT_NVCC_THREADS = "16"
_MACHETE_REQUIRED_TORCH_NVCC_UNDEFINES = (
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
)
_MACHETE_REQUIRED_CUDA_HEADERS = (
    "cuda_runtime_api.h",
    "cusparse.h",
    "cublas_v2.h",
    "cublasLt.h",
    "cusolverDn.h",
)

MACHETE_PREPACKED_BLOCK_SHAPE = (64, 128)


def _machete_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _machete_source_root() -> Path:
    return _machete_project_root() / "gptqmodel_ext" / "machete"


def _repo_local_cutlass_root() -> Path:
    return _machete_project_root() / "cutlass"


def _gptqmodel_cache_dir() -> Path:
    """Return the user cache root used by downloaded source artifacts."""

    configured = os.getenv("GPTQMODEL_CACHE_DIR")
    if configured:
        return Path(configured).expanduser()
    xdg_cache_home = os.getenv("XDG_CACHE_HOME")
    if xdg_cache_home:
        return Path(xdg_cache_home).expanduser() / "gptqmodel"
    return Path.home() / ".cache" / "gptqmodel"


def _cutlass_cache_key() -> str:
    return f"{_CUTLASS_VERSION}-{_CUTLASS_ARCHIVE_SHA256}"


def _cutlass_cache_dir() -> Path:
    return _gptqmodel_cache_dir() / "cutlass" / _cutlass_cache_key()


def _cutlass_download_cache_dir() -> Path:
    return _gptqmodel_cache_dir() / "downloads"


def _cutlass_archive_path() -> Path:
    return _cutlass_download_cache_dir() / f"cutlass-v{_cutlass_cache_key()}.tar.gz"


def _machete_cache_lock_path(name: str) -> Path:
    return _gptqmodel_cache_dir() / "locks" / f"{name}.lock"


def _offline_mode() -> bool:
    return env_flag("GPTQMODEL_OFFLINE", default=False)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _acquire_machete_cache_lock(name: str) -> FileLock:
    lock_path = _machete_cache_lock_path(name)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    return FileLock(str(lock_path), timeout=_MACHETE_CACHE_LOCK_TIMEOUT_SECONDS)


def _cutlass_python_bindings_present(cutlass_root: Path) -> bool:
    python_dir = cutlass_root / "python"
    return (
        (python_dir / "cutlass_library.py").is_file()
        or (python_dir / "cutlass_library" / "__init__.py").is_file()
    )


def _cutlass_checkout_complete(cutlass_root: Path) -> bool:
    common_include_dir = cutlass_root / "examples" / "common" / "include"
    util_include_dir = cutlass_root / "tools" / "util" / "include"
    return (
        (cutlass_root / "include" / "cutlass" / "cutlass.h").is_file()
        and (cutlass_root / "tools" / "library" / "include").is_dir()
        and (common_include_dir.is_dir() or util_include_dir.is_dir())
        and _cutlass_python_bindings_present(cutlass_root)
    )


def _repo_local_cutlass_version_marker(cutlass_root: Path) -> Path:
    return cutlass_root / _CUTLASS_VERSION_MARKER


def _cutlass_checkout_version(cutlass_root: Path) -> Optional[str]:
    version_header = cutlass_root / "include" / "cutlass" / "version.h"
    if not version_header.is_file():
        return None

    macros = dict(_CUTLASS_VERSION_DEFINE_PATTERN.findall(version_header.read_text(encoding="utf-8")))
    required_macros = {"MAJOR", "MINOR", "PATCH"}
    if macros.keys() < required_macros:
        return None

    return f"{macros['MAJOR']}.{macros['MINOR']}.{macros['PATCH']}"


def _cutlass_checkout_version_error(cutlass_root: Path) -> Optional[str]:
    version = _cutlass_checkout_version(cutlass_root)
    if version is None:
        return (
            f"`{cutlass_root}` is missing a readable `include/cutlass/version.h`; "
            f"GPTQModel requires CUTLASS v{_CUTLASS_VERSION}."
        )
    if version != _CUTLASS_VERSION:
        return (
            f"`{cutlass_root}` contains CUTLASS v{version}, but GPTQModel requires v{_CUTLASS_VERSION}."
        )
    return None


def _mark_cutlass_cache(cutlass_root: Path) -> None:
    """Mark a cache-owned checkout complete; never call this for user sources."""

    _repo_local_cutlass_version_marker(cutlass_root).write_text(
        json.dumps(
            {
                "version": _CUTLASS_VERSION,
                "archive_sha256": _CUTLASS_ARCHIVE_SHA256,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _download_cutlass_archive(url: str, destination: Path) -> None:
    """Download and atomically publish the pinned CUTLASS archive."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        if partial.exists():
            partial.unlink()

        log.info("Machete: downloading CUTLASS v%s into `%s`.", _CUTLASS_VERSION, destination)
        with urllib.request.urlopen(url) as response, partial.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        actual = _sha256(partial)
        if actual != _CUTLASS_ARCHIVE_SHA256:
            raise RuntimeError(
                "Machete: CUTLASS archive checksum mismatch: "
                f"expected {_CUTLASS_ARCHIVE_SHA256}, got {actual}."
            )
        os.replace(partial, destination)
    finally:
        try:
            partial.unlink()
        except FileNotFoundError:
            pass


def _extract_cutlass_archive(archive_path: Path, destination_parent: Path) -> None:
    """Extract a tarball after rejecting traversal, links, and special files."""

    with tarfile.open(archive_path, "r:gz") as archive:
        root = destination_parent.resolve()
        members = archive.getmembers()
        for member in members:
            member_path = PurePosixPath(member.name)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise RuntimeError(f"Machete: unsafe CUTLASS archive member `{member.name}`.")
            target = (destination_parent / member.name).resolve(strict=False)
            try:
                target.relative_to(root)
            except ValueError as exc:
                raise RuntimeError(f"Machete: unsafe CUTLASS archive member `{member.name}`.") from exc
            if member.issym() or member.islnk() or not (member.isdir() or member.isfile()):
                raise RuntimeError(
                    f"Machete: unsupported or unsafe CUTLASS archive member `{member.name}`."
                )
        extract_kwargs = {"path": destination_parent}
        if sys.version_info >= (3, 12):
            extract_kwargs["filter"] = "data"
        archive.extractall(members=members, **extract_kwargs)


def _validate_cutlass_checkout(cutlass_root: Path, *, require_marker: bool = False) -> None:
    if not _cutlass_checkout_complete(cutlass_root):
        raise RuntimeError(
            f"Machete: CUTLASS checkout `{cutlass_root}` is incomplete; "
            f"GPTQModel requires CUTLASS v{_CUTLASS_VERSION}."
        )
    version_error = _cutlass_checkout_version_error(cutlass_root)
    if version_error:
        raise RuntimeError(f"Machete: {version_error}")
    if require_marker:
        marker = _repo_local_cutlass_version_marker(cutlass_root)
        try:
            marker_data = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError) as exc:
            raise RuntimeError(
                f"Machete: CUTLASS cache `{cutlass_root}` is missing a valid completion marker."
            ) from exc
        expected = {
            "version": _CUTLASS_VERSION,
            "archive_sha256": _CUTLASS_ARCHIVE_SHA256,
        }
        if marker_data != expected:
            raise RuntimeError(
                f"Machete: CUTLASS cache `{cutlass_root}` has a stale or incompatible completion marker."
            )


def _cache_cutlass_checkout(archive_path: Path) -> Path:
    archive_digest = _sha256(archive_path) if archive_path.is_file() else "missing"
    if archive_digest != _CUTLASS_ARCHIVE_SHA256:
        raise RuntimeError(
            "Machete: refusing to extract an unverified CUTLASS archive: "
            f"expected {_CUTLASS_ARCHIVE_SHA256}, got {archive_digest}."
        )

    checkout = _cutlass_cache_dir()
    if checkout.is_dir():
        try:
            _validate_cutlass_checkout(checkout, require_marker=True)
            return checkout.resolve()
        except RuntimeError:
            log.warning("Machete: ignoring incomplete CUTLASS cache `%s`.", checkout)
    elif checkout.exists():
        raise RuntimeError(f"Machete: CUTLASS cache path `{checkout}` is not a directory.")

    parent = checkout.parent
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=parent, prefix=f".{_CUTLASS_VERSION}.unpack-") as temp_dir:
        temp_root = Path(temp_dir)
        _extract_cutlass_archive(archive_path, temp_root)
        extracted_root = temp_root / f"cutlass-{_CUTLASS_VERSION}"
        _validate_cutlass_checkout(extracted_root)
        _mark_cutlass_cache(extracted_root)

        stale = checkout.with_name(f".{checkout.name}.stale-{os.getpid()}")
        if stale.exists():
            shutil.rmtree(stale, ignore_errors=True)
        if checkout.exists():
            os.replace(checkout, stale)
        try:
            os.replace(extracted_root, checkout)
        except Exception:
            if stale.exists() and not checkout.exists():
                os.replace(stale, checkout)
            raise
        if stale.exists():
            shutil.rmtree(stale, ignore_errors=True)
    _validate_cutlass_checkout(checkout, require_marker=True)
    return checkout.resolve()


def _ensure_cutlass_source() -> Path:
    """Resolve CUTLASS from a strict override, compatible repo checkout, or cache."""

    configured_root = os.getenv("GPTQMODEL_CUTLASS_DIR")
    if configured_root:
        configured_path = Path(configured_root).expanduser().resolve()
        try:
            _validate_cutlass_checkout(configured_path)
        except RuntimeError as exc:
            raise RuntimeError(
                "Machete: GPTQMODEL_CUTLASS_DIR is authoritative and must point to a read-only, "
                f"compatible CUTLASS v{_CUTLASS_VERSION} checkout: {exc}"
            ) from exc
        return configured_path

    repo_local_root = _repo_local_cutlass_root().resolve()
    if repo_local_root.is_dir():
        try:
            _validate_cutlass_checkout(repo_local_root)
            return repo_local_root
        except RuntimeError:
            log.info("Machete: ignoring incompatible repo-local CUTLASS checkout `%s`.", repo_local_root)

    with _acquire_machete_cache_lock(f"cutlass-{_CUTLASS_VERSION}"):
        cached_checkout = _cutlass_cache_dir()
        if cached_checkout.is_dir():
            try:
                _validate_cutlass_checkout(cached_checkout, require_marker=True)
                return cached_checkout.resolve()
            except RuntimeError:
                log.warning("Machete: ignoring incomplete CUTLASS cache `%s`.", cached_checkout)
        archive_path = _cutlass_archive_path()
        archive_digest = _sha256(archive_path) if archive_path.is_file() else None
        archive_valid = archive_digest == _CUTLASS_ARCHIVE_SHA256
        if not archive_valid:
            if _offline_mode():
                archive_status = (
                    f"has SHA256 {archive_digest}, expected {_CUTLASS_ARCHIVE_SHA256}"
                    if archive_digest is not None
                    else "is missing"
                )
                raise RuntimeError(
                    "Machete: GPTQMODEL_OFFLINE=1 and no verified CUTLASS v%s source cache is available; "
                    "archive `%s` %s. Set GPTQMODEL_CUTLASS_DIR to a compatible checkout, or run "
                    "`python -c \"from gptqmodel import extension; extension.load('machete')\"` once "
                    "while online to populate the cache. To skip JIT entirely, set "
                    "GPTQMODEL_MACHETE_PRECOMPILED_LIBRARY to a compatible shared library."
                    % (_CUTLASS_VERSION, archive_path, archive_status)
                )
            _download_cutlass_archive(_CUTLASS_RELEASE_URL, archive_path)
            if not archive_path.is_file() or _sha256(archive_path) != _CUTLASS_ARCHIVE_SHA256:
                actual = _sha256(archive_path) if archive_path.is_file() else "missing"
                raise RuntimeError(
                    "Machete: downloaded CUTLASS archive checksum mismatch: "
                    f"expected {_CUTLASS_ARCHIVE_SHA256}, got {actual}."
                )
        return _cache_cutlass_checkout(archive_path)


def _machete_generated_dir(fingerprint: Optional[str] = None) -> Path:
    root = _gptqmodel_cache_dir() / "machete" / "generated"
    return root if fingerprint is None else root / fingerprint


def _machete_generation_marker(generated_dir: Optional[Path] = None) -> Path:
    return (generated_dir or _machete_generated_dir()) / _MACHETE_COMPLETE_MARKER


def _machete_generation_manifest(generated_dir: Optional[Path] = None) -> Path:
    return (generated_dir or _machete_generated_dir()) / _MACHETE_MANIFEST_NAME


def _cutlass_python_binding_inputs(cutlass_root: Path) -> list[Path]:
    python_dir = cutlass_root / "python"
    module = python_dir / "cutlass_library.py"
    if module.is_file():
        return [module]
    package = python_dir / "cutlass_library"
    if (package / "__init__.py").is_file():
        # A configured checkout is allowed to differ from the pinned cache, so
        # fingerprint the complete importable generator package rather than
        # only __init__.py. The pinned checkout remains content-addressed by the
        # archive SHA, while this also prevents stale reuse for local edits.
        return sorted(package.rglob("*.py"))
    return []


def _machete_generation_signature(cutlass_root: Path) -> str:
    payload = {
        "cache_version": _MACHETE_GENERATION_CACHE_VERSION,
        "cutlass_version": _CUTLASS_VERSION,
        "cutlass_archive_sha256": _CUTLASS_ARCHIVE_SHA256,
        "jinja2_version": importlib.metadata.version("jinja2"),
        "inputs": [],
    }
    for index, path in enumerate(_machete_generator_inputs(cutlass_root)):
        resolved = path.resolve()
        if not resolved.is_file():
            payload["inputs"].append({"index": index, "name": path.name, "missing": True})
            continue
        payload["inputs"].append(
            {"index": index, "name": path.name, "sha256": _sha256(resolved), "size": resolved.stat().st_size}
        )
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _machete_generator_inputs(cutlass_root: Path) -> list[Path]:
    project_root = _machete_project_root()
    return [
        _machete_source_root() / "generate.py",
        project_root / "gptqmodel_ext" / "cutlass_extensions" / "vllm_cutlass_library_extension.py",
        *_cutlass_python_binding_inputs(cutlass_root),
    ]


def _generated_machete_sources(generated_dir: Optional[Path] = None) -> list[Path]:
    return sorted((generated_dir or _machete_generated_dir()).glob("*.cu"))


def _generated_machete_sources_current(cutlass_root: Path, generated_dir: Optional[Path] = None) -> bool:
    generated_dir = generated_dir or _machete_generated_dir()
    marker = _machete_generation_marker(generated_dir)
    manifest_path = _machete_generation_manifest(generated_dir)
    generated_sources = _generated_machete_sources(generated_dir)
    if (
        not marker.exists()
        or not manifest_path.exists()
        or not generated_sources
        or any(not path.is_file() or path.is_symlink() for path in generated_sources)
    ):
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("version") != _MACHETE_GENERATION_CACHE_VERSION:
            return False
        if marker.read_text(encoding="utf-8").strip() != manifest.get("signature"):
            return False
        if manifest.get("signature") != _machete_generation_signature(cutlass_root):
            return False
        files = manifest.get("files")
        if not isinstance(files, dict) or set(files) != {path.name for path in generated_sources}:
            return False
        for path in generated_sources:
            if _sha256(path) != files.get(path.name):
                return False
    except (OSError, ValueError, TypeError):
        return False
    return True


def _run_machete_generator(cutlass_root: Path, output_dir: Optional[Path] = None) -> None:
    generator = _machete_source_root() / "generate.py"
    output_dir = output_dir or _machete_generated_dir()
    env = os.environ.copy()
    env["GPTQMODEL_CUTLASS_DIR"] = str(cutlass_root)

    log.info("Machete: generating CUTLASS-backed kernel sources in `%s`.", output_dir)
    result = subprocess.run(
        [sys.executable, str(generator), "--output-dir", str(output_dir)],
        cwd=str(_machete_project_root()),
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Machete: failed to generate kernel sources.\n"
            f"Return code: {result.returncode}\n"
            f"Stdout: {result.stdout}\n"
            f"Stderr: {result.stderr}"
        )


def _ensure_generated_machete_sources() -> list[Path]:
    cutlass_root = _ensure_cutlass_source()
    fingerprint = _machete_generation_signature(cutlass_root)
    generated_dir = _machete_generated_dir(fingerprint)
    with _acquire_machete_cache_lock("machete-generated"):
        if _generated_machete_sources_current(cutlass_root, generated_dir):
            return _generated_machete_sources(generated_dir)

        generated_root = generated_dir.parent
        generated_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=generated_root, prefix=f".{fingerprint}.tmp-") as temp_dir:
            temp_output = Path(temp_dir)
            _run_machete_generator(cutlass_root, temp_output)
            generated_sources = _generated_machete_sources(temp_output)
            if not generated_sources or any(path.is_symlink() or not path.is_file() for path in generated_sources):
                raise RuntimeError("Machete: generator completed without producing any CUDA sources.")
            unexpected = [path for path in temp_output.iterdir() if path.suffix != ".cu"]
            if unexpected:
                raise RuntimeError(
                    "Machete: generator produced unexpected files: "
                    + ", ".join(path.name for path in unexpected)
                )
            manifest = {
                "version": _MACHETE_GENERATION_CACHE_VERSION,
                "signature": fingerprint,
                "files": {path.name: _sha256(path) for path in generated_sources},
            }
            _machete_generation_manifest(temp_output).write_text(
                json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
            )
            _machete_generation_marker(temp_output).write_text(fingerprint + "\n", encoding="utf-8")

            stale = generated_dir.with_name(f".{generated_dir.name}.stale-{os.getpid()}")
            if stale.exists():
                shutil.rmtree(stale, ignore_errors=True)
            if generated_dir.exists():
                os.replace(generated_dir, stale)
            try:
                os.replace(temp_output, generated_dir)
            except Exception:
                if stale.exists() and not generated_dir.exists():
                    os.replace(stale, generated_dir)
                raise
            if stale.exists():
                shutil.rmtree(stale, ignore_errors=True)
    if not _generated_machete_sources_current(cutlass_root, generated_dir):
        raise RuntimeError(f"Machete: generated source cache `{generated_dir}` failed integrity validation.")
    return _generated_machete_sources(generated_dir)


def _machete_sources() -> list[str]:
    machete_root = _machete_source_root()
    generated_sources = _ensure_generated_machete_sources()
    return [str(machete_root / "machete_pytorch.cu"), *[str(path) for path in generated_sources]]


def _machete_include_paths() -> list[str]:
    project_root = _machete_project_root()
    cutlass_root = _ensure_cutlass_source()
    include_paths = [
        str(_machete_source_root().resolve()),
        str((project_root / "gptqmodel_ext").resolve()),
        str((project_root / "gptqmodel_ext" / "cutlass_extensions").resolve()),
        str((cutlass_root / "include").resolve()),
        str((cutlass_root / "tools" / "library" / "include").resolve()),
    ]
    common_include_dir = cutlass_root / "examples" / "common" / "include"
    util_include_dir = cutlass_root / "tools" / "util" / "include"
    if common_include_dir.is_dir():
        include_paths.append(str(common_include_dir.resolve()))
    if util_include_dir.is_dir():
        include_paths.append(str(util_include_dir.resolve()))
    return cuda_include_paths_with_fallback(
        include_paths,
        required_header_names=_MACHETE_REQUIRED_CUDA_HEADERS,
    )


def _machete_extra_cflags() -> list[str]:
    return default_jit_cflags(enable_bf16=True)


def _machete_hopper_arch_cuda_cflags() -> list[str]:
    if _machete_static_runtime_error():
        return []

    # vLLM builds Machete only for Hopper-compatible sm90a targets. Torch's
    # default JIT arch detection resolves H100/H200 to sm_90, which compiles
    # but triggers CUTLASS runtime abort spam for sm90a-only instructions.
    if any("90a" in flag for flag in resolved_cuda_arch_flags()):
        return []
    return list(_MACHETE_SM90A_ARCH_FLAGS)


def _machete_extra_cuda_cflags() -> list[str]:
    flags = [
        *_MACHETE_REQUIRED_TORCH_NVCC_UNDEFINES,
        *default_jit_cuda_cflags(
            enable_bf16=True,
            include_lineinfo=True,
            include_nvcc_threads=True,
            include_ptxas_optimizations=True,
            include_ptxas_verbosity=False,
            include_fatbin_compression=True,
            include_diag_suppress=True,
            nvcc_threads=_MACHETE_JIT_NVCC_THREADS,
        ),
        *_machete_hopper_arch_cuda_cflags(),
    ]
    if is_nvcc_compatible():
        flags.insert(0, "-static-global-template-stub=false")
    return flags


def _machete_extra_ldflags() -> list[str]:
    # Hopper tensor-map entry points such as cuTensorMapEncodeTiled live in the
    # CUDA driver library, not libcudart. Link libcuda explicitly so the JIT
    # extension remains loadable after a successful compile on non-SM90 hosts.
    return ["-lcuda"]


_MACHETE_TORCH_OPS_EXTENSION = TorchOpsJitExtension(
    name=_MACHETE_OPS_NAME,
    namespace=_MACHETE_OPS_NAMESPACE,
    required_ops=("machete_prepack_B", "machete_mm", "machete_supported_schedules"),
    sources=_machete_sources,
    build_root_env="GPTQMODEL_MACHETE_BUILD_ROOT",
    default_build_root=lambda: default_torch_ops_build_root("machete"),
    display_name="Machete",
    extra_cflags=_machete_extra_cflags,
    extra_cuda_cflags=_machete_extra_cuda_cflags,
    extra_include_paths=_machete_include_paths,
    extra_ldflags=_machete_extra_ldflags,
    force_rebuild_env="GPTQMODEL_MACHETE_FORCE_REBUILD",
    prebuilt_library_env="GPTQMODEL_MACHETE_PRECOMPILED_LIBRARY",
    verbose_env="GPTQMODEL_EXT_VERBOSE",
    requires_cuda=True,
    python_abi_dependent=False,
    # Machete kernels are Hopper-only, so compile-only workflows may need to
    # force a non-local target such as `TORCH_CUDA_ARCH_LIST=9.0a`.
    merge_visible_cuda_arch_override=False,
)


def _extension_api():
    from gptqmodel import extension as extension_api

    return extension_api


def _machete_static_runtime_error() -> str:
    if IS_ROCM:
        return "Machete kernel is not supported on ROCm."
    if not torch.cuda.is_available():
        return "Machete kernel requires CUDA."
    capability = torch.cuda.get_device_capability()
    if capability != _MACHETE_REQUIRED_COMPUTE_CAPABILITY:
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        return (
            "Machete kernel is Hopper-only (SM90); its generated CUTLASS kernels "
            f"target arch::Sm90 and have no Blackwell image. Found `{props.name}` with "
            f"compute capability {capability[0]}.{capability[1]}."
        )
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    shared_memory_per_block_optin = getattr(
        props,
        "shared_memory_per_block_optin",
        props.shared_memory_per_block,
    )
    if shared_memory_per_block_optin < _MACHETE_MIN_SHARED_MEMORY_PER_BLOCK_OPTIN:
        return (
            "Machete kernel requires at least "
            f"{_MACHETE_MIN_SHARED_MEMORY_PER_BLOCK_OPTIN} bytes of opt-in shared memory per block; "
            f"`{props.name}` exposes {shared_memory_per_block_optin}."
        )
    return ""


def clear_machete_extension_cache() -> None:
    _MACHETE_TORCH_OPS_EXTENSION.clear_cache()


def machete_runtime_available() -> bool:
    static_error = _machete_static_runtime_error()
    if static_error:
        return False
    return _extension_api().is_available("machete")


def machete_runtime_error() -> str:
    static_error = _machete_static_runtime_error()
    if static_error:
        return static_error

    extension_api = _extension_api()
    if extension_api.is_available("machete"):
        return ""
    return extension_api.error("machete") or "Machete runtime unavailable."


def prewarm_machete_extension() -> bool:
    return _extension_api().load(name="machete")["machete"]


def _validate_machete_device_support() -> bool:
    return _machete_static_runtime_error() == ""


def query_machete_supported_quant_types(zero_points: bool) -> List[ScalarType]:
    if zero_points:
        return [scalar_types.uint4, scalar_types.uint8]
    return [scalar_types.uint4b8, scalar_types.uint8b128]


def query_machete_supported_act_types(_zero_points: bool) -> List[torch.dtype]:
    return [torch.float16, torch.bfloat16]


def query_machete_supported_group_sizes(act_type: torch.dtype) -> List[int]:
    if act_type in (torch.float16, torch.bfloat16):
        return [-1, 64, 128]
    return [-1, 128]


def check_machete_supports_shape(
    in_features: int,
    out_features: int,
) -> tuple[bool, Optional[str]]:
    if in_features % MACHETE_PREPACKED_BLOCK_SHAPE[0] != 0:
        return (
            False,
            f"Input features size must be divisible by {MACHETE_PREPACKED_BLOCK_SHAPE[0]}",
        )
    if out_features % MACHETE_PREPACKED_BLOCK_SHAPE[1] != 0:
        return (
            False,
            f"Output features size must be divisible by {MACHETE_PREPACKED_BLOCK_SHAPE[1]}",
        )
    return (True, None)


def machete_prepack_B(
    weight: torch.Tensor,
    a_type: torch.dtype,
    b_type: ScalarType,
    group_scales_type: Optional[torch.dtype],
) -> torch.Tensor:
    return _extension_api().op("machete", "machete_prepack_B")(
        weight,
        a_type,
        b_type.id,
        group_scales_type,
    )


def machete_supported_schedules(
    a_type: torch.dtype,
    b_type: ScalarType,
    group_scales_type: Optional[torch.dtype] = None,
    group_zeros_type: Optional[torch.dtype] = None,
    channel_scales_type: Optional[torch.dtype] = None,
    token_scales_type: Optional[torch.dtype] = None,
    out_type: Optional[torch.dtype] = None,
) -> List[str]:
    return _extension_api().op("machete", "machete_supported_schedules")(
        a_type,
        b_type.id,
        group_scales_type,
        group_zeros_type,
        channel_scales_type,
        token_scales_type,
        out_type,
    )


def machete_mm(
    *,
    a: torch.Tensor,
    b_q: torch.Tensor,
    b_type: ScalarType,
    b_group_scales: Optional[torch.Tensor] = None,
    b_group_zeros: Optional[torch.Tensor] = None,
    b_group_size: Optional[int] = None,
    b_channel_scales: Optional[torch.Tensor] = None,
    a_token_scales: Optional[torch.Tensor] = None,
    out_type: Optional[torch.dtype] = None,
    schedule: Optional[str] = None,
) -> torch.Tensor:
    return _extension_api().op("machete", "machete_mm")(
        a,
        b_q,
        b_type.id,
        out_type,
        b_group_scales,
        b_group_zeros,
        b_group_size,
        b_channel_scales,
        a_token_scales,
        schedule,
    )


def pack_quantized_values_into_int32(
    tensor: torch.Tensor,
    qtype: ScalarType,
    packed_dim: int = 0,
) -> torch.Tensor:
    perm = tuple(i for i in range(tensor.ndim) if i != packed_dim) + (packed_dim,)
    inv_perm = tuple(perm.index(i) for i in range(len(perm)))
    temp = tensor.permute(perm)

    pack_factor = 32 // qtype.size_bits
    mask = (1 << qtype.size_bits) - 1

    assert temp.shape[-1] % pack_factor == 0
    new_shape = list(temp.shape)
    new_shape[-1] //= pack_factor

    result = torch.zeros(new_shape, dtype=torch.int32, device=tensor.device)
    for i in range(pack_factor):
        result |= ((temp[..., i::pack_factor] & mask) << (qtype.size_bits * i))

    return result.permute(inv_perm)


def unpack_quantized_values_into_int32(
    tensor: torch.Tensor,
    qtype: ScalarType,
    packed_dim: int = 0,
) -> torch.Tensor:
    perm = tuple(i for i in range(tensor.ndim) if i != packed_dim) + (packed_dim,)
    inv_perm = tuple(perm.index(i) for i in range(len(perm)))
    temp = tensor.permute(perm)

    pack_factor = 32 // qtype.size_bits
    mask = (1 << qtype.size_bits) - 1

    new_shape = list(temp.shape)
    new_shape[-1] *= pack_factor

    result = torch.zeros(new_shape, dtype=torch.int32, device=tensor.device)
    for i in range(pack_factor):
        result[..., i::pack_factor] = (temp >> (qtype.size_bits * i)) & mask

    return result.permute(inv_perm)


__all__ = [
    "_ensure_cutlass_source",
    "_ensure_generated_machete_sources",
    "_validate_machete_device_support",
    "check_machete_supports_shape",
    "clear_machete_extension_cache",
    "machete_mm",
    "machete_prepack_B",
    "machete_runtime_available",
    "machete_runtime_error",
    "machete_supported_schedules",
    "pack_quantized_values_into_int32",
    "prewarm_machete_extension",
    "query_machete_supported_act_types",
    "query_machete_supported_group_sizes",
    "query_machete_supported_quant_types",
    "unpack_quantized_values_into_int32",
]
