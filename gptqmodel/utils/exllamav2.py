# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import torch

from .cpp import (
    TorchOpsJitExtension,
    cuda_include_paths_with_fallback,
    default_jit_cflags,
    default_jit_cuda_cflags,
    default_torch_ops_build_root,
)


# These values are deliberately strings at the Python boundary.  Apart from
# making the public knob readable, this prevents callers from expressing
# different paths through combinations of booleans whose meaning depends on
# the current kernel implementation.
EXLLAMAV2_PATH_LEGACY = "legacy"
EXLLAMAV2_PATH_FUSED = "fused"
EXLLAMAV2_PATH_DENSE = "dense"
EXLLAMAV2_PATH_AUTO = "auto"
EXLLAMAV2_PATHS = frozenset(
    {
        EXLLAMAV2_PATH_LEGACY,
        EXLLAMAV2_PATH_FUSED,
        EXLLAMAV2_PATH_DENSE,
        EXLLAMAV2_PATH_AUTO,
    }
)

# The C++ ABI keeps the old four-argument op intact by making this a trailing
# argument with a default.  The Python codes are not exposed to users.
_EXLLAMAV2_PATH_CODE = {
    EXLLAMAV2_PATH_LEGACY: -1,
    EXLLAMAV2_PATH_FUSED: 0,
    EXLLAMAV2_PATH_DENSE: 1,
}


def _legacy_exllamav2_path(m: int) -> str:
    return EXLLAMAV2_PATH_DENSE if m > 50 else EXLLAMAV2_PATH_FUSED


def _device_metadata(device) -> Optional[tuple[str, tuple[int, int], int, int]]:
    """Return stable device facts used by the offline dispatch table.

    Failure to obtain CUDA metadata is intentionally a normal fallback: the
    selector must not initialize CUDA, synchronize, or tune during forward.
    The tuple is (name, compute capability, SM count, visible device count).
    """

    try:
        if device is None or device.type != "cuda" or not torch.cuda.is_available():
            return None
        index = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        return (
            props.name,
            (props.major, props.minor),
            props.multi_processor_count,
            torch.cuda.device_count(),
        )
    except (RuntimeError, AttributeError, AssertionError):
        return None


def _rule_matches(rule, *, metadata, m, n, k, group_size, desc_act, sym, layout):
    if metadata is None:
        return False
    name, capability, sm_count, device_count = metadata
    return (
        rule["device_name"] == name
        and rule["capability"] == capability
        and rule["sm_count"] == sm_count
        and rule["device_count"] == device_count
        and rule["n"] == n
        and rule["k"] == k
        and rule["group_size"] == group_size
        and rule["desc_act"] == desc_act
        and rule["sym"] == sym
        and rule["layout"] == layout
        and any(start <= m <= end for start, end, _ in rule["m_ranges"])
    )


def _make_rtx4090_measured_rules() -> tuple[dict, ...]:
    """Build the reviewed singleton buckets from the RTX 4090 measurements.

    Only buckets where the measured winner differs from the legacy M > 50
    decision are listed.  Singleton buckets are intentional: the benchmark
    covered the M values below and, where the kernel has a boundary, we do not
    extrapolate an unmeasured interval.  Unsupported device counts, layouts,
    or shapes use the legacy selector.
    """

    # (K, N) -> measured M/path overrides.  These are conservative overrides;
    # a tie or noisy crossover is omitted rather than made into a default.
    overrides = {
        (2048, 512): {
            51: EXLLAMAV2_PATH_FUSED,
            52: EXLLAMAV2_PATH_FUSED,
            64: EXLLAMAV2_PATH_FUSED,
            96: EXLLAMAV2_PATH_FUSED,
            128: EXLLAMAV2_PATH_FUSED,
            256: EXLLAMAV2_PATH_FUSED,
        },
        (2048, 8192): {
            40: EXLLAMAV2_PATH_DENSE,
            48: EXLLAMAV2_PATH_DENSE,
            49: EXLLAMAV2_PATH_DENSE,
            50: EXLLAMAV2_PATH_DENSE,
        },
        (8192, 2048): {
            49: EXLLAMAV2_PATH_DENSE,
            50: EXLLAMAV2_PATH_DENSE,
        },
        (4096, 4096): {
            48: EXLLAMAV2_PATH_DENSE,
            49: EXLLAMAV2_PATH_DENSE,
            50: EXLLAMAV2_PATH_DENSE,
        },
    }
    rules = []
    for (k, n), m_paths in overrides.items():
        for requested_group_size in (32, 64, 128, -1):
            group_size = k if requested_group_size == -1 else requested_group_size
            for desc_act in (False, True):
                rules.append(
                    {
                        "device_name": "NVIDIA GeForce RTX 4090",
                        "capability": (8, 9),
                        "sm_count": 128,
                        "device_count": 4,
                        "n": n,
                        "k": k,
                        "group_size": group_size,
                        "desc_act": desc_act,
                        "sym": True,
                        "layout": "gptq4:qzero1",
                        "m_ranges": tuple(
                            (m, m, selected_path)
                            for m, selected_path in sorted(m_paths.items())
                        ),
                    }
                )
    return tuple(rules)


# This table contains only repeated measurements from the four RTX 4090
# machine used for this change.  No default is inferred for other devices.
EXLLAMAV2_MEASURED_RULES: tuple[dict, ...] = _make_rtx4090_measured_rules()


def select_exllamav2_path(
    *,
    path: str,
    m: int,
    n: int,
    k: int,
    device=None,
    group_size: int,
    desc_act: bool,
    sym: bool,
    layout: str = "gptq4",
    force_cuda: bool = False,
    device_metadata=None,
) -> str:
    """Select a concrete GPTQ ExLlamaV2 execution path.

    ``m`` is the flattened Linear row count.  Auto selection is a pure lookup
    over the measured table and falls back to the historical ``M > 50`` rule.
    No CUDA calls, synchronization, or candidate probing occur here unless
    the caller supplies a CUDA device and the already-available CUDA metadata
    is needed for a table lookup.
    """

    if path not in EXLLAMAV2_PATHS:
        raise ValueError(
            f"Unsupported ExLlamaV2 GPTQ path {path!r}; expected one of "
            f"{sorted(EXLLAMAV2_PATHS)}"
        )
    if force_cuda:
        return EXLLAMAV2_PATH_FUSED
    if path == EXLLAMAV2_PATH_FUSED or path == EXLLAMAV2_PATH_DENSE:
        return path
    if path == EXLLAMAV2_PATH_LEGACY:
        return _legacy_exllamav2_path(m)

    metadata = _device_metadata(device) if device_metadata is None else device_metadata
    for rule in EXLLAMAV2_MEASURED_RULES:
        if _rule_matches(
            rule,
            metadata=metadata,
            m=m,
            n=n,
            k=k,
            group_size=group_size,
            desc_act=desc_act,
            sym=sym,
            layout=layout,
        ):
            for start, end, selected_path in rule["m_ranges"]:
                if start <= m <= end:
                    return selected_path
    return _legacy_exllamav2_path(m)


class ScratchSpace:
    def __init__(self, scratch_bytes, dev):
        self.scratch_bytes = scratch_bytes
        self.scratch = torch.empty(
            self.scratch_bytes // 2,
            dtype=torch.float16,
            device=dev,
        )

    def get_slice(self, size_bytes):
        # This allocator intentionally returns a view into one shared backing
        # storage.  ExLlamaV2 uses it sequentially on one CUDA stream during a
        # normal model forward; overlapping use from independent streams is
        # not supported because reconstruct and cuBLAS may alias this region.
        size_halfs = next_multiple(size_bytes, 128) // 2
        scratch_slice = self.scratch.narrow(0, 0, size_halfs)

        return scratch_slice


def next_multiple(x, multiple):
    return ((x + multiple - 1) // multiple) * multiple


def _exllamav2_root() -> Path:
    return Path(__file__).resolve().parents[2] / "gptqmodel_ext" / "exllamav2"


def _exllamav2_gptq_sources() -> list[str]:
    root = _exllamav2_root()
    return [
        str(root / "ext_gptq.cpp"),
        str(root / "cuda" / "q_matrix.cu"),
        str(root / "cuda" / "q_gemm.cu"),
    ]


def _exllamav2_required_cuda_headers() -> tuple[str, ...]:
    return ("cusparse.h",)


def _exllamav2_include_paths() -> list[str]:
    return cuda_include_paths_with_fallback(
        [str(_exllamav2_root())],
        required_header_names=_exllamav2_required_cuda_headers(),
    )


def _exllamav2_gptq_extra_cflags() -> list[str]:
    return default_jit_cflags(opt_level="O2", enable_bf16=True)


def _exllamav2_gptq_extra_cuda_cflags() -> list[str]:
    return default_jit_cuda_cflags(
        opt_level="O2",
        enable_bf16=True,
        include_lineinfo=True,
        include_nvcc_threads=True,
        include_ptxas_optimizations=True,
        include_fatbin_compression=True,
        include_diag_suppress=True,
    )


def _exllamav2_extra_cflags() -> list[str]:
    return default_jit_cflags(enable_bf16=True)


def _exllamav2_extra_cuda_cflags() -> list[str]:
    return default_jit_cuda_cflags(
        enable_bf16=True,
        include_lineinfo=True,
        include_nvcc_threads=True,
        include_ptxas_optimizations=True,
        include_fatbin_compression=True,
        include_diag_suppress=True,
    )


def _exllamav2_extra_ldflags() -> list[str]:
    # q_gemm.cu / q_gemm_awq.cu call cublasHgemm. On Linux the symbol resolves
    # at load time, but the MSVC linker needs the import library explicitly or
    # the build fails with `LNK2019: unresolved external symbol cublasHgemm`.
    # Mirrors _extra_ldflags in gptqmodel/exllamav3/ext.py.
    if sys.platform != "win32":
        return []
    flags = ["cublas.lib"]
    if sys.base_prefix != sys.prefix:
        flags.append(f"/LIBPATH:{os.path.join(sys.base_prefix, 'libs')}")
    return flags


def _exllamav2_awq_extra_cflags() -> list[str]:
    return default_jit_cflags(opt_level=None, enable_bf16=True)


def _exllamav2_awq_extra_cuda_cflags() -> list[str]:
    return default_jit_cuda_cflags(
        opt_level=None,
        enable_bf16=True,
        include_lineinfo=True,
        include_nvcc_threads=True,
        include_ptxas_optimizations=True,
        include_fatbin_compression=True,
        include_diag_suppress=True,
    )


_EXLLAMAV2_GPTQ_TORCH_OPS_EXTENSION = TorchOpsJitExtension(
    name="gptqmodel_exllamav2_ops",
    namespace="gptqmodel_exllamav2",
    required_ops=("make_q_matrix", "gemm_half_q_half"),
    sources=_exllamav2_gptq_sources,
    build_root_env="GPTQMODEL_EXLLAMAV2_BUILD_ROOT",
    default_build_root=lambda: default_torch_ops_build_root("exllamav2"),
    display_name="ExLlamaV2 GPTQ",
    extra_cflags=_exllamav2_gptq_extra_cflags,
    extra_cuda_cflags=_exllamav2_gptq_extra_cuda_cflags,
    extra_include_paths=_exllamav2_include_paths,
    extra_ldflags=_exllamav2_extra_ldflags,
    force_rebuild_env="GPTQMODEL_EXLLAMAV2_FORCE_REBUILD",
    verbose_env="GPTQMODEL_EXT_VERBOSE",
    requires_cuda=True,
    python_abi_dependent=False,
)

# Shared AWQ singleton so every caller reuses the same torch.ops cache and
# first-use build policy instead of depending on setup-time wheels.
_EXLLAMAV2_AWQ_TORCH_OPS_EXTENSION = TorchOpsJitExtension(
    name="gptqmodel_exllamav2_awq_ops",
    namespace="gptqmodel_exllamav2_awq",
    required_ops=("make_q_matrix_awq", "gemm_half_q_half_awq"),
    sources=lambda: [
        str(_exllamav2_root() / "ext_awq.cpp"),
        str(_exllamav2_root() / "cuda" / "q_matrix_awq.cu"),
        str(_exllamav2_root() / "cuda" / "q_gemm_awq.cu"),
    ],
    build_root_env="GPTQMODEL_EXLLAMAV2_AWQ_BUILD_ROOT",
    default_build_root=lambda: default_torch_ops_build_root("exllamav2_awq"),
    display_name="ExLlamaV2 AWQ",
    extra_cflags=_exllamav2_awq_extra_cflags,
    extra_cuda_cflags=_exllamav2_awq_extra_cuda_cflags,
    extra_include_paths=_exllamav2_include_paths,
    extra_ldflags=_exllamav2_extra_ldflags,
    force_rebuild_env="GPTQMODEL_EXLLAMAV2_AWQ_FORCE_REBUILD",
    verbose_env="GPTQMODEL_EXT_VERBOSE",
    requires_cuda=True,
    python_abi_dependent=False,
)


def _extension_api():
    from gptqmodel import extension as extension_api

    return extension_api


def clear_exllamav2_gptq_extension_cache() -> None:
    _EXLLAMAV2_GPTQ_TORCH_OPS_EXTENSION.clear_cache()


def exllamav2_gptq_runtime_available() -> bool:
    return _extension_api().is_available("exllamav2")


def exllamav2_gptq_runtime_error() -> str:
    extension_api = _extension_api()
    if extension_api.is_available("exllamav2"):
        return ""
    return (
        extension_api.error("exllamav2")
        or "ExLlamaV2 GPTQ CUDA runtime unavailable."
    )


def prewarm_exllamav2_gptq_extension() -> bool:
    return _extension_api().load(name="exllamav2")["exllamav2"]


def exllamav2_make_q_matrix(
    q_weight,
    q_perm,
    q_invperm,
    q_scale,
    q_scale_max,
    q_groups,
    gptq_qzeros,
    gptq_scales,
    gptq_g_idx,
    temp_dq,
) -> int:
    return int(
        _extension_api().op("exllamav2", "make_q_matrix")(
            q_weight,
            q_perm,
            q_invperm,
            q_scale,
            q_scale_max,
            q_groups,
            gptq_qzeros,
            gptq_scales,
            gptq_g_idx,
            temp_dq,
        )
    )


def exllamav2_gemm_half_q_half(
    a, q_handle: int, c, force_cuda: bool = False, path: Optional[str] = None
) -> None:
    op = _extension_api().op("exllamav2", "gemm_half_q_half")
    if path is None or path == EXLLAMAV2_PATH_LEGACY:
        # Preserve the old call ABI for integrations which provide the
        # original four-argument torch op.
        op(a, int(q_handle), c, bool(force_cuda))
    else:
        op(a, int(q_handle), c, bool(force_cuda), _EXLLAMAV2_PATH_CODE[path])


def clear_exllamav2_awq_extension_cache() -> None:
    _EXLLAMAV2_AWQ_TORCH_OPS_EXTENSION.clear_cache()


def exllamav2_awq_runtime_available() -> bool:
    return _extension_api().is_available("exllamav2_awq")


def exllamav2_awq_runtime_error() -> str:
    extension_api = _extension_api()
    if extension_api.is_available("exllamav2_awq"):
        return ""
    return (
        extension_api.error("exllamav2_awq")
        or "ExLlamaV2 AWQ CUDA runtime unavailable."
    )


def prewarm_exllamav2_awq_extension() -> bool:
    return _extension_api().load(name="exllamav2_awq")["exllamav2_awq"]


def exllamav2_awq_make_q_matrix(
    q_weight,
    q_perm,
    q_invperm,
    q_scale,
    q_scale_max,
    q_groups,
    gptq_qzeros,
    gptq_scales,
    gptq_g_idx,
    temp_dq,
) -> int:
    return int(
        _extension_api().op("exllamav2_awq", "make_q_matrix_awq")(
            q_weight,
            q_perm,
            q_invperm,
            q_scale,
            q_scale_max,
            q_groups,
            gptq_qzeros,
            gptq_scales,
            gptq_g_idx,
            temp_dq,
        )
    )


def exllamav2_awq_gemm_half_q_half(a, q_handle: int, c, force_cuda: bool = False) -> None:
    _extension_api().op("exllamav2_awq", "gemm_half_q_half_awq")(a, int(q_handle), c, bool(force_cuda))
