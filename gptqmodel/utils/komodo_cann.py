# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from pathlib import Path

import torch

from .cpp import TorchOpsJitExtension, default_jit_cflags, default_torch_ops_build_root


_KOMODO_CANN_V3_OPS_NAME = "gptqmodel_cannoe_v3_ops"
_KOMODO_CANN_ASCENDC_OPS_NAME = "gptqmodel_cannoe_ascendc_ops"
_KOMODO_CANN_V3_NAMESPACE = "gptqmodel_cannoe"
_KOMODO_CANN_ASCENDC_NAMESPACE = "gptqmodel_cannoe"
_CANNOE_ENV_BY_LEGACY = {
    "GPTQMODEL_KOMODO_CANN_ASCENDC_OPAPI_LIB": "GPTQMODEL_CANNOE_ASCENDC_OPAPI_LIB",
    "GPTQMODEL_KOMODO_CANN_ASCENDC_BUILD_ROOT": "GPTQMODEL_CANNOE_ASCENDC_BUILD_ROOT",
    "GPTQMODEL_KOMODO_CANN_ASCENDC_FORCE_REBUILD": "GPTQMODEL_CANNOE_ASCENDC_FORCE_REBUILD",
    "GPTQMODEL_KOMODO_CANN_ASCENDC_VERBOSE": "GPTQMODEL_CANNOE_ASCENDC_VERBOSE",
    "GPTQMODEL_KOMODO_CANN_V3_BUILD_ROOT": "GPTQMODEL_CANNOE_V3_BUILD_ROOT",
    "GPTQMODEL_KOMODO_CANN_V3_FORCE_REBUILD": "GPTQMODEL_CANNOE_V3_FORCE_REBUILD",
    "GPTQMODEL_KOMODO_CANN_V3_VERBOSE": "GPTQMODEL_CANNOE_V3_VERBOSE",
}
_DEFAULT_CANN_HOME_CANDIDATES = (
    "/usr/local/Ascend/ascend-toolkit/latest",
    "/usr/local/Ascend/cann",
    "/usr/local/Ascend/cann-9.0.0-beta.2",
    "/usr/local/Ascend/cann-8.5.1",
)


def _cannoe_env(legacy_name: str) -> str | None:
    current_name = _CANNOE_ENV_BY_LEGACY.get(legacy_name, legacy_name)
    value = os.getenv(current_name)
    if value is not None:
        return value
    return os.getenv(legacy_name)


def _sync_cannoe_env_aliases() -> None:
    for legacy_name, current_name in _CANNOE_ENV_BY_LEGACY.items():
        if os.getenv(legacy_name) is None:
            value = os.getenv(current_name)
            if value is not None:
                os.environ[legacy_name] = value


def _komodo_cann_root() -> Path:
    return Path(__file__).resolve().parents[2] / "gptqmodel_ext" / "komodo_cann"


def _komodo_cann_v3_sources() -> list[str]:
    return [str(_komodo_cann_root() / "wq_bmm_v3_probe.cpp")]


def _komodo_cann_ascendc_sources() -> list[str]:
    return [str(_komodo_cann_root() / "w4a16_ascendc_bridge.cpp")]


def _torch_npu_root() -> Path:
    import torch_npu

    return Path(torch_npu.__file__).resolve().parent


def _cann_root() -> Path:
    for env_name in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME"):
        raw = os.getenv(env_name)
        if raw:
            return Path(raw).expanduser()

    for raw in _DEFAULT_CANN_HOME_CANDIDATES:
        path = Path(raw).expanduser()
        if path.exists():
            return path
    return Path(_DEFAULT_CANN_HOME_CANDIDATES[0]).expanduser()


def _cann_arch_root() -> Path:
    return _cann_root() / "aarch64-linux"


def _komodo_cann_v3_include_paths() -> list[str]:
    torch_npu_root = _torch_npu_root()
    cann_arch_root = _cann_arch_root()
    return [
        str(torch_npu_root / "include"),
        str(torch_npu_root / "include" / "third_party" / "acl" / "inc"),
        str(cann_arch_root / "include"),
    ]


def _komodo_cann_v3_extra_cflags() -> list[str]:
    return default_jit_cflags()


def _komodo_cann_v3_extra_ldflags() -> list[str]:
    torch_npu_lib = _torch_npu_root() / "lib"
    cann_lib = _cann_arch_root() / "lib64"
    return [
        f"-L{torch_npu_lib}",
        f"-Wl,-rpath,{torch_npu_lib}",
        "-ltorch_npu",
        f"-L{cann_lib}",
        f"-Wl,-rpath,{cann_lib}",
        "-lopapi",
        "-lacl_rt",
        "-lacl_op_executor",
    ]


def _komodo_cann_ascendc_extra_ldflags() -> list[str]:
    flags = _komodo_cann_v3_extra_ldflags()
    flags.append("-ldl")
    return flags


def _find_komodo_cann_ascendc_opapi_lib() -> Path | None:
    explicit = _cannoe_env("GPTQMODEL_KOMODO_CANN_ASCENDC_OPAPI_LIB")
    if explicit:
        path = Path(explicit).expanduser()
        return path if path.exists() else None

    search_roots: list[Path] = []
    for raw in os.getenv("ASCEND_CUSTOM_OPP_PATH", "").split(os.pathsep):
        if raw:
            search_roots.append(Path(raw).expanduser() / "op_api" / "lib")
    for raw in os.getenv("LD_LIBRARY_PATH", "").split(os.pathsep):
        if raw:
            search_roots.append(Path(raw).expanduser())

    for root in search_roots:
        candidate = root / "libcust_opapi.so"
        if candidate.exists():
            return candidate
    return None


def komodo_cann_v3_environment_error() -> str:
    try:
        torch_npu_root = _torch_npu_root()
    except Exception as exc:  # pragma: no cover - depends on optional torch_npu package
        return f"Cannoe V3 failed to import torch_npu: {exc}"

    if not hasattr(torch, "npu"):
        return "Cannoe V3 requires a Torch-NPU runtime."
    try:
        if not torch.npu.is_available():
            return "Cannoe V3 requires an available NPU device."
    except Exception as exc:  # pragma: no cover - depends on Torch-NPU runtime
        return f"Cannoe V3 failed to query NPU availability: {exc}"

    cann_arch_root = _cann_arch_root()
    required_paths = (
        _komodo_cann_root() / "wq_bmm_v3_probe.cpp",
        torch_npu_root / "include" / "torch_npu" / "csrc" / "core" / "npu" / "NPUStream.h",
        torch_npu_root / "lib" / "libtorch_npu.so",
        cann_arch_root / "include" / "aclnnop" / "aclnn_weight_quant_batch_matmul_v3.h",
        cann_arch_root / "lib64" / "libopapi.so",
    )
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        return "Cannoe V3 missing required files: " + ", ".join(missing)
    return ""


def komodo_cann_ascendc_environment_error() -> str:
    try:
        torch_npu_root = _torch_npu_root()
    except Exception as exc:  # pragma: no cover - depends on optional torch_npu package
        return f"Cannoe Ascend C failed to import torch_npu: {exc}"

    if not hasattr(torch, "npu"):
        return "Cannoe Ascend C requires a Torch-NPU runtime."
    try:
        if not torch.npu.is_available():
            return "Cannoe Ascend C requires an available NPU device."
    except Exception as exc:  # pragma: no cover - depends on Torch-NPU runtime
        return f"Cannoe Ascend C failed to query NPU availability: {exc}"

    cann_arch_root = _cann_arch_root()
    required_paths = (
        _komodo_cann_root() / "w4a16_ascendc_bridge.cpp",
        torch_npu_root / "include" / "torch_npu" / "csrc" / "core" / "npu" / "NPUStream.h",
        torch_npu_root / "lib" / "libtorch_npu.so",
        cann_arch_root / "include" / "aclnn" / "acl_meta.h",
        cann_arch_root / "lib64" / "libacl_op_executor.so",
    )
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        return "Cannoe Ascend C missing required files: " + ", ".join(missing)

    opapi_lib = _find_komodo_cann_ascendc_opapi_lib()
    if opapi_lib is None:
        return (
            "Cannoe Ascend C requires libcust_opapi.so from the built custom OPP. "
            "Source the package set_env.bash or set GPTQMODEL_CANNOE_ASCENDC_OPAPI_LIB."
        )
    return ""


def _komodo_cann_v3_supported() -> bool:
    return komodo_cann_v3_environment_error() == ""


def _komodo_cann_ascendc_supported() -> bool:
    return komodo_cann_ascendc_environment_error() == ""


_KOMODO_CANN_V3_TORCH_OPS_EXTENSION = TorchOpsJitExtension(
    name=_KOMODO_CANN_V3_OPS_NAME,
    namespace=_KOMODO_CANN_V3_NAMESPACE,
    required_ops=("w4a16_matmul",),
    sources=_komodo_cann_v3_sources,
    build_root_env="GPTQMODEL_KOMODO_CANN_V3_BUILD_ROOT",
    default_build_root=lambda: default_torch_ops_build_root("cannoe_v3"),
    display_name="Cannoe V3",
    extra_cflags=_komodo_cann_v3_extra_cflags,
    extra_include_paths=_komodo_cann_v3_include_paths,
    extra_ldflags=_komodo_cann_v3_extra_ldflags,
    force_rebuild_env="GPTQMODEL_KOMODO_CANN_V3_FORCE_REBUILD",
    verbose_env="GPTQMODEL_KOMODO_CANN_V3_VERBOSE",
    requires_cuda=False,
    merge_visible_cuda_arch_override=False,
)


_KOMODO_CANN_ASCENDC_TORCH_OPS_EXTENSION = TorchOpsJitExtension(
    name=_KOMODO_CANN_ASCENDC_OPS_NAME,
    namespace=_KOMODO_CANN_ASCENDC_NAMESPACE,
    required_ops=("cannoe_w4_a16_matmul",),
    sources=_komodo_cann_ascendc_sources,
    build_root_env="GPTQMODEL_KOMODO_CANN_ASCENDC_BUILD_ROOT",
    default_build_root=lambda: default_torch_ops_build_root("cannoe_ascendc"),
    display_name="Cannoe Ascend C",
    extra_cflags=_komodo_cann_v3_extra_cflags,
    extra_include_paths=_komodo_cann_v3_include_paths,
    extra_ldflags=_komodo_cann_ascendc_extra_ldflags,
    force_rebuild_env="GPTQMODEL_KOMODO_CANN_ASCENDC_FORCE_REBUILD",
    verbose_env="GPTQMODEL_KOMODO_CANN_ASCENDC_VERBOSE",
    requires_cuda=False,
    merge_visible_cuda_arch_override=False,
)


def load_komodo_cann_v3() -> bool:
    _sync_cannoe_env_aliases()
    if not _komodo_cann_v3_supported():
        _KOMODO_CANN_V3_TORCH_OPS_EXTENSION._last_error = komodo_cann_v3_environment_error()
        return False
    return _KOMODO_CANN_V3_TORCH_OPS_EXTENSION.load()


def load_komodo_cann_ascendc() -> bool:
    _sync_cannoe_env_aliases()
    if not _komodo_cann_ascendc_supported():
        _KOMODO_CANN_ASCENDC_TORCH_OPS_EXTENSION._last_error = komodo_cann_ascendc_environment_error()
        return False
    if not _cannoe_env("GPTQMODEL_KOMODO_CANN_ASCENDC_OPAPI_LIB"):
        opapi_lib = _find_komodo_cann_ascendc_opapi_lib()
        if opapi_lib is not None:
            os.environ["GPTQMODEL_CANNOE_ASCENDC_OPAPI_LIB"] = str(opapi_lib)
            os.environ["GPTQMODEL_KOMODO_CANN_ASCENDC_OPAPI_LIB"] = str(opapi_lib)
    return _KOMODO_CANN_ASCENDC_TORCH_OPS_EXTENSION.load()


def komodo_cann_v3_runtime_error() -> str:
    return _KOMODO_CANN_V3_TORCH_OPS_EXTENSION.last_error_message() or komodo_cann_v3_environment_error()


def komodo_cann_ascendc_runtime_error() -> str:
    return _KOMODO_CANN_ASCENDC_TORCH_OPS_EXTENSION.last_error_message() or komodo_cann_ascendc_environment_error()


def clear_komodo_cann_v3_extension_cache() -> None:
    _KOMODO_CANN_V3_TORCH_OPS_EXTENSION.clear_cache()


def clear_komodo_cann_ascendc_extension_cache() -> None:
    _KOMODO_CANN_ASCENDC_TORCH_OPS_EXTENSION.clear_cache()


_CANNOE_ASCENDC_TORCH_OPS_EXTENSION = _KOMODO_CANN_ASCENDC_TORCH_OPS_EXTENSION
_CANNOE_V3_TORCH_OPS_EXTENSION = _KOMODO_CANN_V3_TORCH_OPS_EXTENSION
cannoe_ascendc_environment_error = komodo_cann_ascendc_environment_error
cannoe_ascendc_runtime_error = komodo_cann_ascendc_runtime_error
cannoe_v3_environment_error = komodo_cann_v3_environment_error
cannoe_v3_runtime_error = komodo_cann_v3_runtime_error
clear_cannoe_ascendc_extension_cache = clear_komodo_cann_ascendc_extension_cache
clear_cannoe_v3_extension_cache = clear_komodo_cann_v3_extension_cache
load_cannoe_ascendc = load_komodo_cann_ascendc
load_cannoe_v3 = load_komodo_cann_v3


__all__ = [
    "_CANNOE_ASCENDC_TORCH_OPS_EXTENSION",
    "_CANNOE_V3_TORCH_OPS_EXTENSION",
    "_KOMODO_CANN_ASCENDC_TORCH_OPS_EXTENSION",
    "_KOMODO_CANN_V3_TORCH_OPS_EXTENSION",
    "_komodo_cann_ascendc_supported",
    "_komodo_cann_v3_supported",
    "cannoe_ascendc_environment_error",
    "cannoe_ascendc_runtime_error",
    "cannoe_v3_environment_error",
    "cannoe_v3_runtime_error",
    "clear_cannoe_ascendc_extension_cache",
    "clear_cannoe_v3_extension_cache",
    "clear_komodo_cann_ascendc_extension_cache",
    "clear_komodo_cann_v3_extension_cache",
    "komodo_cann_ascendc_environment_error",
    "komodo_cann_ascendc_runtime_error",
    "komodo_cann_v3_environment_error",
    "komodo_cann_v3_runtime_error",
    "load_cannoe_ascendc",
    "load_cannoe_v3",
    "load_komodo_cann_ascendc",
    "load_komodo_cann_v3",
]
