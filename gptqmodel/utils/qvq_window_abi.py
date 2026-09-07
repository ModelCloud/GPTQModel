# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Native C boundary for external P32 window consumers; no alternate format."""

import ctypes
import hashlib
import sys
from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.cpp_extension import include_paths, library_paths

from .cpp import TorchOpsJitExtension, default_jit_cflags, default_torch_ops_build_root

_ROOT = Path(__file__).resolve().parents[2] / "gptqmodel_ext" / "qvq"
_NAME = "gptqmodel_qvq_window_abi_ops"
_EXTENSION = TorchOpsJitExtension(
    name=_NAME,
    namespace="gptqmodel_qvq_window_abi",
    required_ops=("version",),
    sources=[
        str(_ROOT / "qvq_window_abi.cpp"),
        str(_ROOT / "qvq_window_rank8_fused.cu"),
    ],
    build_root_env="GPTQMODEL_QVQ_WINDOW_ABI_BUILD_ROOT",
    default_build_root=lambda: default_torch_ops_build_root(_NAME),
    display_name="P32 window native ABI",
    extra_cflags=lambda: [
        *default_jit_cflags(),
        "-DQVQ_WINDOW_ABI_HEADER_HASH=0x" + hashlib.sha256((_ROOT / "qvq_window_abi.h").read_bytes()).hexdigest()[:8],
    ],
    extra_include_paths=lambda: include_paths(device_type="cuda"),
    extra_ldflags=lambda: [*("-L" + path for path in library_paths(device_type="cuda")),
                           "-lc10_cuda", "-ltorch_cuda", "-lcudart"],
    requires_cuda=True,
)


def native_window_abi_supported():
    return (
        sys.platform == "linux" and torch.version.hip is None and torch.cuda.is_available()
        and any(torch.cuda.get_device_capability(i) == (9, 0) for i in range(torch.cuda.device_count()))
    )


class WindowBuffer(ctypes.Structure):
    _fields_ = [("data", ctypes.c_void_p), ("bytes", ctypes.c_uint64)]

    @classmethod
    def tensor(cls, tensor):
        if tensor is None:
            return cls(None, 0)
        if not tensor.is_contiguous():
            raise ValueError("native window buffers must be contiguous")
        return cls(tensor.data_ptr(), tensor.numel() * tensor.element_size())


class WindowConfig(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint32) for name in (
        "abi_version", "struct_bytes", "m", "k", "n", "transition_bits", "bank_alt_id",
        "algorithm", "block_m", "block_n", "block_k", "warp_groups", "pipeline_stages", "split_k",
        "min_m", "max_m", "input_hadamard", "output_hadamard", "rank8_enabled",
    )]


@lru_cache(maxsize=1)
def native_window_library():
    from .. import extension
    from .qvq_wgmma_cuda import _QVQ_WGMMA_EXTENSION

    if not extension.load("qvq_cuda")["qvq_cuda"] or not _QVQ_WGMMA_EXTENSION.load():
        raise RuntimeError("native window ABI requires loaded QVQ CUDA and Hopper operators")
    if not extension.load("qvq_window_abi")["qvq_window_abi"]:
        raise RuntimeError(_EXTENSION.last_error_message())
    library = ctypes.CDLL(str(_EXTENSION.build_root() / (_NAME + ".so")))
    library.qvq_p32_window_linear.argtypes = (
        [WindowBuffer] * 10
        + [ctypes.POINTER(WindowConfig), ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64]
    )
    library.qvq_p32_window_linear.restype = ctypes.c_int
    library.qvq_p32_window_graph_create.argtypes = [
        ctypes.POINTER(WindowBuffer), ctypes.POINTER(WindowConfig), ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_uint64,
    ]
    library.qvq_p32_window_graph_run.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64,
    ]
    library.qvq_p32_window_graph_destroy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64]
    for name in ("create", "run", "destroy"):
        getattr(library, "qvq_p32_window_graph_" + name).restype = ctypes.c_int
    return library


def native_window_linear(layer, x, config):
    """Validate an artifact and call the same operator through its C ABI.

    This setup/reference helper is not a hot-path dispatcher. External hosts
    load/validate once, retain their buffers, and call the library directly.
    Initial ABI coverage is explicit Hopper geometry and reference correction;
    unsupported policies fail rather than selecting an implicit substitute.
    """
    from ..quantization.qvq_rank8 import prepare_rank8
    from .qvq_cuda import _pgc16_levels

    if (config.algorithm not in ("hopper_m16", "hopper_direct_decode_mma")
            or (config.algorithm == "hopper_direct_decode_mma" and not config.block_m)
            or config.recovery_kernel != "separate_reference"
            or config.recovery_projection != "separate_reference" or config.chunk_m):
        raise ValueError("native ABI requires explicit Hopper geometry and reference rank8 kernels")
    # Window-only deployments intentionally release the planar trellis after
    # CPU repacking.  Validate against the live payload that this ABI will
    # actually consume, while keeping legacy planar layers supported.
    payload = layer.window_words if getattr(layer, "window_only", False) else layer.trellis
    if payload is None:
        raise ValueError("native ABI layer has no P32 window or planar payload")
    if (x.ndim != 2 or x.dtype != torch.float16 or x.device != payload.device
            or x.shape[1] != layer.in_features or not x.is_contiguous()):
        raise ValueError("native ABI input must be a contiguous FP16 module matrix")
    # This entry point allocates an output tensor and calls the raw native ABI,
    # whose workspace is intentionally not capture-safe.  Reject before rank8
    # preparation, metadata packing, or allocation so a caller cannot enter a
    # capture with a partially initialized path.  Use the prepared graph API
    # for command-buffer/CUDA-graph execution instead.
    if x.device.type == "cuda" and torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "native_window_linear cannot run during CUDA Graph capture; "
            "prepare and replay a native window graph instead"
        )
    library = native_window_library()
    prepare_rank8(layer, config)
    window, banks, alt_id = layer._prepare_amd_p32_metadata(x.device)
    enabled = layer._p32_rank8_enabled
    output = torch.empty((x.shape[0], layer.out_features), device=x.device, dtype=x.dtype)
    buffers = (
        x, window, banks, _pgc16_levels(x.device, layer.codebook_version),
        layer._cached_cast("SU", torch.float16), layer._cached_cast("SV", torch.float16),
        layer._cached_cast("bias", torch.float16),
        layer.rank8_A if enabled else None, layer.rank8_B if enabled else None, output,
    )
    native = WindowConfig(
        3, ctypes.sizeof(WindowConfig), x.shape[0], layer.in_features, layer.out_features,
        round(2 * layer.bits), alt_id, 1 if config.algorithm == "hopper_m16" else 2,
        config.block_m, config.block_n, config.block_k, config.warp_groups,
        config.pipeline_stages, config.split_k, config.min_m, config.max_m,
        int(layer.input_hadamard), int(layer.output_hadamard), int(enabled),
    )
    error = ctypes.create_string_buffer(4096)
    status = library.qvq_p32_window_linear(
        *(WindowBuffer.tensor(value) for value in buffers), ctypes.byref(native),
        torch.cuda.current_stream(x.device).cuda_stream, error, len(error),
    )
    if status:
        raise RuntimeError(error.value.decode(errors="replace"))
    return output
