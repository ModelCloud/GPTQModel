# SPDX-FileCopyrightText: 2024-2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from typing import List, Optional

import pcre
import torch

from .cpp import (
    TorchOpsJitExtension,
    cuda_include_paths_with_fallback,
    default_jit_cflags,
    default_jit_cuda_cflags,
    default_torch_ops_build_root,
    is_nvcc_compatible,
    resolved_cuda_arch_flags,
)
from .logger import setup_logger
from .marlin_scalar_type import ScalarType, scalar_types
from .rocm import IS_ROCM


log = setup_logger()

_MACHETE_OPS_NAME = "gptqmodel_machete_ops"
_MACHETE_OPS_NAMESPACE = "gptqmodel_machete"

_CUTLASS_VERSION = "4.4.2"
_CUTLASS_RELEASE_URL = f"https://github.com/NVIDIA/cutlass/archive/refs/tags/v{_CUTLASS_VERSION}.tar.gz"
_CUTLASS_VERSION_MARKER = ".gptqmodel_cutlass_version"
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


def _cutlass_download_cache_dir() -> Path:
    return _machete_project_root() / "build" / "_deps"


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


def _repo_local_cutlass_version_matches(cutlass_root: Path) -> bool:
    marker = _repo_local_cutlass_version_marker(cutlass_root)
    return (
        _cutlass_checkout_version(cutlass_root) == _CUTLASS_VERSION
        and marker.is_file()
        and marker.read_text(encoding="utf-8").strip() == _CUTLASS_VERSION
    )


def _mark_repo_local_cutlass_version(cutlass_root: Path) -> None:
    _repo_local_cutlass_version_marker(cutlass_root).write_text(f"{_CUTLASS_VERSION}\n", encoding="utf-8")


def _use_repo_local_cutlass(cutlass_root: Path) -> Path:
    if not _repo_local_cutlass_version_matches(cutlass_root):
        _mark_repo_local_cutlass_version(cutlass_root)
    os.environ["GPTQMODEL_CUTLASS_DIR"] = str(cutlass_root)
    return cutlass_root


def _download_cutlass_archive(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    if partial.exists():
        partial.unlink()

    log.info("Machete: downloading CUTLASS v%s into `%s`.", _CUTLASS_VERSION, destination)
    with urllib.request.urlopen(url) as response, partial.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    partial.replace(destination)


def _extract_cutlass_archive(archive_path: Path, destination_parent: Path) -> None:
    with tarfile.open(archive_path, "r:gz") as archive:
        extract_kwargs = {"path": destination_parent}
        if sys.version_info >= (3, 12):
            extract_kwargs["filter"] = "data"
        archive.extractall(**extract_kwargs)


def _ensure_cutlass_source() -> Path:
    repo_local_root = _repo_local_cutlass_root().resolve()
    configured_root = os.getenv("GPTQMODEL_CUTLASS_DIR")
    if configured_root:
        configured_path = Path(configured_root).expanduser().resolve()
        if _cutlass_checkout_complete(configured_path):
            version_error = _cutlass_checkout_version_error(configured_path)
            if version_error is None:
                if configured_path == repo_local_root:
                    return _use_repo_local_cutlass(configured_path)
                return configured_path
            if configured_path != repo_local_root:
                raise RuntimeError(
                    "Machete: GPTQMODEL_CUTLASS_DIR points to an incompatible CUTLASS checkout. "
                    f"{version_error} Unset GPTQMODEL_CUTLASS_DIR to allow auto-download, or point it at a "
                    f"CUTLASS v{_CUTLASS_VERSION} checkout."
                )
            log.info(
                "Machete: GPTQMODEL_CUTLASS_DIR points to stale repo-local CUTLASS checkout `%s`; refreshing to v%s.",
                configured_path,
                _CUTLASS_VERSION,
            )
        else:
            log.info(
                "Machete: GPTQMODEL_CUTLASS_DIR=`%s` is incomplete; falling back to repo-local CUTLASS checkout.",
                configured_path,
            )

    if _cutlass_checkout_complete(repo_local_root):
        if _cutlass_checkout_version_error(repo_local_root) is None:
            return _use_repo_local_cutlass(repo_local_root)
    if repo_local_root.exists():
        current_version = _cutlass_checkout_version(repo_local_root)
        log.info(
            "Machete: refreshing repo-local CUTLASS checkout at `%s`%s to v%s.",
            repo_local_root,
            f" from v{current_version}" if current_version else "",
            _CUTLASS_VERSION,
        )

    archive_path = _cutlass_download_cache_dir() / f"cutlass-v{_CUTLASS_VERSION}.tar.gz"
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if not archive_path.exists():
        _download_cutlass_archive(_CUTLASS_RELEASE_URL, archive_path)

    parent = repo_local_root.parent
    parent.mkdir(parents=True, exist_ok=True)
    if repo_local_root.exists():
        shutil.rmtree(repo_local_root, ignore_errors=True)

    with tempfile.TemporaryDirectory(dir=parent, prefix="cutlass-unpack-") as temp_dir:
        temp_root = Path(temp_dir)
        _extract_cutlass_archive(archive_path, temp_root)
        extracted_root = temp_root / f"cutlass-{_CUTLASS_VERSION}"
        if not extracted_root.exists():
            raise RuntimeError(f"Machete: failed to extract CUTLASS archive `{archive_path}`.")
        extracted_root.replace(repo_local_root)
        _mark_repo_local_cutlass_version(repo_local_root)

    return _use_repo_local_cutlass(repo_local_root)


def _machete_generated_dir() -> Path:
    return _machete_source_root() / "generated"


def _machete_generation_marker() -> Path:
    return _machete_generated_dir() / ".gptqmodel_complete"


def _cutlass_python_binding_inputs(cutlass_root: Path) -> list[Path]:
    python_dir = cutlass_root / "python"
    candidates = [
        python_dir / "cutlass_library.py",
        python_dir / "cutlass_library" / "__init__.py",
    ]
    return [candidate for candidate in candidates if candidate.exists()]


def _machete_generation_signature(cutlass_root: Path) -> str:
    return json.dumps(
        {
            "cutlass_root": str(cutlass_root.resolve()),
            "cutlass_version": _CUTLASS_VERSION,
        },
        sort_keys=True,
    )


def _machete_generator_inputs(cutlass_root: Path) -> list[Path]:
    project_root = _machete_project_root()
    return [
        _machete_source_root() / "generate.py",
        project_root / "gptqmodel_ext" / "cutlass_extensions" / "vllm_cutlass_library_extension.py",
        *_cutlass_python_binding_inputs(cutlass_root),
    ]


def _generated_machete_sources() -> list[Path]:
    return sorted(_machete_generated_dir().glob("*.cu"))


def _generated_machete_sources_current(cutlass_root: Path) -> bool:
    marker = _machete_generation_marker()
    generated_sources = _generated_machete_sources()
    if not marker.exists() or not generated_sources:
        return False
    if marker.read_text(encoding="utf-8").strip() != _machete_generation_signature(cutlass_root):
        return False
    marker_mtime_ns = marker.stat().st_mtime_ns
    return not any(path.stat().st_mtime_ns > marker_mtime_ns for path in _machete_generator_inputs(cutlass_root))


def _run_machete_generator(cutlass_root: Path) -> None:
    generator = _machete_source_root() / "generate.py"
    env = os.environ.copy()
    env["GPTQMODEL_CUTLASS_DIR"] = str(cutlass_root)

    log.info("Machete: generating CUTLASS-backed kernel sources in `%s`.", _machete_generated_dir())
    result = subprocess.run(
        [sys.executable, str(generator)],
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
    if _generated_machete_sources_current(cutlass_root):
        return _generated_machete_sources()

    generated_dir = _machete_generated_dir()
    if generated_dir.exists():
        shutil.rmtree(generated_dir, ignore_errors=True)

    _run_machete_generator(cutlass_root)

    generated_sources = _generated_machete_sources()
    if not generated_sources:
        raise RuntimeError(
            "Machete: generator completed without producing any CUDA sources."
        )

    _machete_generation_marker().write_text(
        _machete_generation_signature(cutlass_root),
        encoding="utf-8",
    )
    return generated_sources


def _machete_sources() -> list[str]:
    machete_root = _machete_source_root()
    generated_sources = _ensure_generated_machete_sources()
    return [str(machete_root / "machete_pytorch.cu"), *[str(path) for path in generated_sources]]


def _machete_include_paths() -> list[str]:
    project_root = _machete_project_root()
    cutlass_root = _ensure_cutlass_source()
    include_paths = [
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
    verbose_env="GPTQMODEL_EXT_VERBOSE",
    requires_cuda=True,
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
            "Machete kernel currently supports Hopper-class SM90 GPUs only; "
            f"found `{props.name}` with compute capability {capability[0]}.{capability[1]}."
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
