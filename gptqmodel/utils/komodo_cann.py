# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from pathlib import Path

import torch

from .cpp import TorchOpsJitExtension, default_jit_cflags, default_torch_ops_build_root


_KOMODO_CANN_V3_OPS_NAME = "gptqmodel_komodo_cann_v3_ops"
_KOMODO_CANN_ASCENDC_OPS_NAME = "gptqmodel_komodo_cann_ascendc_ops"
_KOMODO_CANN_V3_NAMESPACE = "gptqmodel_komodo_cann"
_KOMODO_CANN_ASCENDC_NAMESPACE = "gptqmodel_komodo_cann"
_DEFAULT_CANN_HOME = "/usr/local/Ascend/cann-8.5.1"


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
    return Path(os.getenv("ASCEND_HOME_PATH", _DEFAULT_CANN_HOME)).expanduser()


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
    explicit = os.getenv("GPTQMODEL_KOMODO_CANN_ASCENDC_OPAPI_LIB")
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
        return f"Komodo-CANN V3 failed to import torch_npu: {exc}"

    if not hasattr(torch, "npu"):
        return "Komodo-CANN V3 requires a Torch-NPU runtime."
    try:
        if not torch.npu.is_available():
            return "Komodo-CANN V3 requires an available NPU device."
    except Exception as exc:  # pragma: no cover - depends on Torch-NPU runtime
        return f"Komodo-CANN V3 failed to query NPU availability: {exc}"

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
        return "Komodo-CANN V3 missing required files: " + ", ".join(missing)
    return ""


def komodo_cann_ascendc_environment_error() -> str:
    try:
        torch_npu_root = _torch_npu_root()
    except Exception as exc:  # pragma: no cover - depends on optional torch_npu package
        return f"Komodo-CANN Ascend C failed to import torch_npu: {exc}"

    if not hasattr(torch, "npu"):
        return "Komodo-CANN Ascend C requires a Torch-NPU runtime."
    try:
        if not torch.npu.is_available():
            return "Komodo-CANN Ascend C requires an available NPU device."
    except Exception as exc:  # pragma: no cover - depends on Torch-NPU runtime
        return f"Komodo-CANN Ascend C failed to query NPU availability: {exc}"

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
        return "Komodo-CANN Ascend C missing required files: " + ", ".join(missing)

    opapi_lib = _find_komodo_cann_ascendc_opapi_lib()
    if opapi_lib is None:
        return (
            "Komodo-CANN Ascend C requires libcust_opapi.so from the built custom OPP. "
            "Source the package set_env.bash or set GPTQMODEL_KOMODO_CANN_ASCENDC_OPAPI_LIB."
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
    default_build_root=lambda: default_torch_ops_build_root("komodo_cann_v3"),
    display_name="Komodo-CANN V3",
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
    required_ops=("komodo_cann_w4_a16_matmul",),
    sources=_komodo_cann_ascendc_sources,
    build_root_env="GPTQMODEL_KOMODO_CANN_ASCENDC_BUILD_ROOT",
    default_build_root=lambda: default_torch_ops_build_root("komodo_cann_ascendc"),
    display_name="Komodo-CANN Ascend C",
    extra_cflags=_komodo_cann_v3_extra_cflags,
    extra_include_paths=_komodo_cann_v3_include_paths,
    extra_ldflags=_komodo_cann_ascendc_extra_ldflags,
    force_rebuild_env="GPTQMODEL_KOMODO_CANN_ASCENDC_FORCE_REBUILD",
    verbose_env="GPTQMODEL_KOMODO_CANN_ASCENDC_VERBOSE",
    requires_cuda=False,
    merge_visible_cuda_arch_override=False,
)


def load_komodo_cann_v3() -> bool:
    if not _komodo_cann_v3_supported():
        _KOMODO_CANN_V3_TORCH_OPS_EXTENSION._last_error = komodo_cann_v3_environment_error()
        return False
    return _KOMODO_CANN_V3_TORCH_OPS_EXTENSION.load()


def load_komodo_cann_ascendc() -> bool:
    if not _komodo_cann_ascendc_supported():
        _KOMODO_CANN_ASCENDC_TORCH_OPS_EXTENSION._last_error = komodo_cann_ascendc_environment_error()
        return False
    if not os.getenv("GPTQMODEL_KOMODO_CANN_ASCENDC_OPAPI_LIB"):
        opapi_lib = _find_komodo_cann_ascendc_opapi_lib()
        if opapi_lib is not None:
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


__all__ = [
    "_KOMODO_CANN_ASCENDC_TORCH_OPS_EXTENSION",
    "_KOMODO_CANN_V3_TORCH_OPS_EXTENSION",
    "_komodo_cann_ascendc_supported",
    "_komodo_cann_v3_supported",
    "clear_komodo_cann_ascendc_extension_cache",
    "clear_komodo_cann_v3_extension_cache",
    "komodo_cann_ascendc_environment_error",
    "komodo_cann_ascendc_runtime_error",
    "komodo_cann_v3_environment_error",
    "komodo_cann_v3_runtime_error",
    "load_komodo_cann_ascendc",
    "load_komodo_cann_v3",
]
