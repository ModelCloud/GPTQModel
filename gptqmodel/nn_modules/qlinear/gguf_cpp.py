# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ctypes
import os
import threading
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ...adapter.adapter import Adapter, Lora
from ...models._const import DEVICE, PLATFORM
from ...quantization.config import FORMAT, METHOD
from ...utils.backend import BACKEND
from .gguf import _GGUF_TYPE_INFO, GGUFTorchLinear


# Imported lazily by ``_require_llama_cpp``: importing llama-cpp-python costs
# tens of milliseconds and must not happen while the kernel registry is merely
# being discovered. The module attributes stay available for callers that read
# them after an explicit import.
_llama_cpp_pkg = None
_llama_cpp_lib = None
_llama_cpp_load_shared_library = None
_LLAMA_CPP_IMPORT_ERROR = None
_LLAMA_CPP_LOADED = False


def _require_llama_cpp():
    """Import llama-cpp-python once and return its module objects."""

    global _LLAMA_CPP_LOADED, _LLAMA_CPP_IMPORT_ERROR
    global _llama_cpp_pkg, _llama_cpp_lib, _llama_cpp_load_shared_library

    if _LLAMA_CPP_LOADED:
        return _llama_cpp_pkg, _llama_cpp_lib, _llama_cpp_load_shared_library

    _LLAMA_CPP_LOADED = True
    try:
        import llama_cpp as llama_cpp_pkg
        from llama_cpp import llama_cpp as llama_cpp_lib
        from llama_cpp._ctypes_extensions import (
            load_shared_library as llama_cpp_load_shared_library,
        )

        _llama_cpp_pkg = llama_cpp_pkg
        _llama_cpp_lib = llama_cpp_lib
        _llama_cpp_load_shared_library = llama_cpp_load_shared_library
        _LLAMA_CPP_IMPORT_ERROR = None
    except Exception as exc:  # pragma: no cover - optional dependency
        _LLAMA_CPP_IMPORT_ERROR = exc

    return _llama_cpp_pkg, _llama_cpp_lib, _llama_cpp_load_shared_library


class _GGMLInitParams(ctypes.Structure):
    _fields_ = [
        ("mem_size", ctypes.c_size_t),
        ("mem_buffer", ctypes.c_void_p),
        ("no_alloc", ctypes.c_bool),
    ]


class _GGMLMatmulPlan:
    def __init__(
        self,
        *,
        ctx,
        buffer,
        weight_tensor,
        input_tensor,
        output_tensor,
        graph,
        rows: int,
        out_features: int,
        output_nbytes: int,
        output_dtype: torch.dtype,
        backend_buffer_free,
        ctx_free,
    ) -> None:
        self.ctx = ctx
        self.buffer = buffer
        self.weight_tensor = weight_tensor
        self.input_tensor = input_tensor
        self.output_tensor = output_tensor
        self.graph = graph
        self.rows = rows
        self.out_features = out_features
        self.output_nbytes = output_nbytes
        self.output_dtype = output_dtype
        self._backend_buffer_free = backend_buffer_free
        self._ctx_free = ctx_free

    def close(self) -> None:
        if self.buffer:
            self._backend_buffer_free(self.buffer)
            self.buffer = None
        if self.ctx:
            self._ctx_free(self.ctx)
            self.ctx = None

    def __del__(self) -> None:  # pragma: no cover - best effort cleanup
        try:
            self.close()
        except Exception:
            # Destructors must not raise during GC or interpreter shutdown.
            pass


# Upstream enum ggml_type values for constants llama-cpp-python does not export
# (kept in sync with ggml/include/ggml.h; Q8_1=9, TQ1_0=34, TQ2_0=35).
_GGML_TYPE_ID_FALLBACK = {
    "Q8_1": 9,
    "TQ1_0": 34,
    "TQ2_0": 35,
}


class _GGMLBridge:
    GGML_METADATA_BYTES = 1 << 20

    def __init__(self) -> None:
        llama_cpp_pkg, llama_cpp_lib, load_shared_library = _require_llama_cpp()
        if _LLAMA_CPP_IMPORT_ERROR is not None:
            raise ModuleNotFoundError(
                "GGUFCppKernel requires `llama-cpp-python` to be installed."
            ) from _LLAMA_CPP_IMPORT_ERROR

        lib_dir = Path(llama_cpp_pkg.__file__).resolve().parent / "lib"
        self._ggml_base = load_shared_library("ggml-base", lib_dir)
        self._ggml_cpu = load_shared_library("ggml-cpu", lib_dir)
        self._ggml_cuda = None
        self._ggml_cuda_error: Optional[Exception] = None
        try:
            self._ggml_cuda = load_shared_library("ggml-cuda", lib_dir)
        except Exception as exc:  # pragma: no cover - optional shared library
            self._ggml_cuda_error = exc
        self._bind_functions()

        self.ggml_type_f32 = int(llama_cpp_lib.GGML_TYPE_F32)
        self.ggml_type_f16 = int(llama_cpp_lib.GGML_TYPE_F16)
        self.ggml_qtypes = {}
        for name in (
            "F32", "F16", "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "Q8_1",
            "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_K",
            "IQ2_XXS", "IQ2_XS", "IQ3_XXS", "IQ1_S", "IQ4_NL", "IQ3_S",
            "IQ2_S", "IQ4_XS", "I8", "I16", "I32", "I64", "F64", "IQ1_M",
            "BF16", "TQ1_0", "TQ2_0", "MXFP4", "NVFP4", "Q1_0", "Q2_0",
        ):
            value = getattr(llama_cpp_lib, f"GGML_TYPE_{name}", None)
            if value is None:
                value = _GGML_TYPE_ID_FALLBACK.get(name)
            if value is not None:
                self.ggml_qtypes[name] = int(value)
        # PRISM's Q1_0_g128 shares the upstream Q1_0 sign-only layout.
        if "Q1_0" in self.ggml_qtypes:
            self.ggml_qtypes["Q1_0_g128"] = self.ggml_qtypes["Q1_0"]
        self._cpu_backend: Optional[int] = None
        self._cpu_backend_lock = threading.RLock()
        self._cuda_backends: Dict[int, int] = {}
        self._quantize_initialized: set[int] = set()
        self._quantize_init_lock = threading.Lock()
        self._dequantize_fns: Dict[str, object] = {}

    def _bind(self, lib, name: str, argtypes, restype) -> None:
        fn = getattr(lib, name)
        fn.argtypes = argtypes
        fn.restype = restype

    def _bind_functions(self) -> None:
        self._bind(self._ggml_base, "ggml_init", [_GGMLInitParams], ctypes.c_void_p)
        self._bind(self._ggml_base, "ggml_free", [ctypes.c_void_p], None)
        self._bind(
            self._ggml_base,
            "ggml_new_tensor_2d",
            [ctypes.c_void_p, ctypes.c_int, ctypes.c_int64, ctypes.c_int64],
            ctypes.c_void_p,
        )
        self._bind(self._ggml_base, "ggml_mul_mat", [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p], ctypes.c_void_p)
        self._bind(self._ggml_base, "ggml_set_input", [ctypes.c_void_p], None)
        self._bind(self._ggml_base, "ggml_new_graph", [ctypes.c_void_p], ctypes.c_void_p)
        self._bind(self._ggml_base, "ggml_build_forward_expand", [ctypes.c_void_p, ctypes.c_void_p], None)
        self._bind(self._ggml_base, "ggml_nbytes", [ctypes.c_void_p], ctypes.c_size_t)
        self._bind(self._ggml_base, "ggml_element_size", [ctypes.c_void_p], ctypes.c_size_t)
        self._bind(self._ggml_base, "ggml_backend_alloc_ctx_tensors", [ctypes.c_void_p, ctypes.c_void_p], ctypes.c_void_p)
        self._bind(self._ggml_base, "ggml_backend_buffer_free", [ctypes.c_void_p], None)
        self._bind(self._ggml_base, "ggml_backend_free", [ctypes.c_void_p], None)
        self._bind(
            self._ggml_base,
            "ggml_backend_tensor_set",
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t],
            None,
        )
        self._bind(
            self._ggml_base,
            "ggml_backend_tensor_get",
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t],
            None,
        )
        self._bind(self._ggml_base, "ggml_backend_graph_compute", [ctypes.c_void_p, ctypes.c_void_p], ctypes.c_int)
        self._bind(self._ggml_base, "ggml_backend_synchronize", [ctypes.c_void_p], None)
        self._bind(self._ggml_cpu, "ggml_backend_cpu_init", [], ctypes.c_void_p)
        self._bind(self._ggml_cpu, "ggml_backend_cpu_set_n_threads", [ctypes.c_void_p, ctypes.c_int], None)
        # Native ggml quantize/dequantize entry points (public ggml API).
        self._bind(self._ggml_base, "ggml_is_quantized", [ctypes.c_int], ctypes.c_bool)
        self._bind(self._ggml_base, "ggml_blck_size", [ctypes.c_int], ctypes.c_int64)
        self._bind(self._ggml_base, "ggml_row_size", [ctypes.c_int, ctypes.c_int64], ctypes.c_size_t)
        self._bind(self._ggml_base, "ggml_quantize_init", [ctypes.c_int], None)
        self._bind(self._ggml_base, "ggml_quantize_requires_imatrix", [ctypes.c_int], ctypes.c_bool)
        self._bind(
            self._ggml_base,
            "ggml_quantize_chunk",
            [
                ctypes.c_int,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.c_void_p,
            ],
            ctypes.c_size_t,
        )
        if self._ggml_cuda is not None:
            self._bind(self._ggml_cuda, "ggml_backend_cuda_init", [ctypes.c_int], ctypes.c_void_p)
            self._bind(self._ggml_cuda, "ggml_backend_cuda_get_device_count", [], ctypes.c_int)

    # Types ggml_quantize_chunk can produce: the switch in ggml.c covers every
    # IQ/K/microscaling type plus Q1_0/Q2_0/Q4_*/Q5_*/Q8_0/F16/BF16. Q8_1 and
    # Q8_K are inference-only upstream (no quantizer case) and calling them
    # aborts the process via GGML_ASSERT, so they are excluded explicitly.
    _GGML_QUANTIZABLE_QTYPES = frozenset({
        "Q1_0", "Q2_0", "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "MXFP4", "NVFP4",
        "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "TQ1_0", "TQ2_0",
        "IQ2_XXS", "IQ2_XS", "IQ3_XXS", "IQ3_S", "IQ2_S", "IQ1_S", "IQ1_M",
        "IQ4_NL", "IQ4_XS",
    })

    def quantize_chunk(self, weight: np.ndarray, gguf_tensor_qtype: str) -> np.ndarray:
        """Quantize a 2D fp32 matrix into ggml-packed rows via `ggml_quantize_chunk`."""

        if gguf_tensor_qtype not in self._GGML_QUANTIZABLE_QTYPES:
            raise NotImplementedError(f"ggml has no quantizer for `{gguf_tensor_qtype}`.")
        type_id = self.ggml_qtypes.get(gguf_tensor_qtype)
        if type_id is None or not self._ggml_base.ggml_is_quantized(type_id):
            raise NotImplementedError(f"ggml does not support quantizing `{gguf_tensor_qtype}`.")
        w = np.ascontiguousarray(weight, dtype=np.float32)
        if w.ndim != 2:
            raise ValueError(f"ggml quantize expects a 2D matrix, got shape {w.shape}.")
        nrows, ncols = w.shape
        blck = int(self._ggml_base.ggml_blck_size(type_id))
        if blck > 1 and ncols % blck != 0:
            raise ValueError(
                f"GGUF `{gguf_tensor_qtype}` requires columns divisible by {blck}, got {ncols}."
            )
        row_size = int(self._ggml_base.ggml_row_size(type_id, ncols))
        dst = np.empty((nrows, row_size), dtype=np.uint8)

        imatrix = None
        imatrix_ptr = None
        if self._ggml_base.ggml_quantize_requires_imatrix(type_id):
            # Importance-matrix formats accept a uniform matrix as a valid
            # (non activation-aware) fallback.
            imatrix = np.ones(ncols, dtype=np.float32)
            imatrix_ptr = imatrix.ctypes.data_as(ctypes.c_void_p)

        with self._quantize_init_lock:
            if type_id not in self._quantize_initialized:
                self._ggml_base.ggml_quantize_init(type_id)
                self._quantize_initialized.add(type_id)

        result = self._ggml_base.ggml_quantize_chunk(
            type_id,
            w.ctypes.data_as(ctypes.c_void_p),
            dst.ctypes.data_as(ctypes.c_void_p),
            0,
            nrows,
            ncols,
            imatrix_ptr,
        )
        if result != dst.nbytes:
            raise RuntimeError(
                f"ggml_quantize_chunk({gguf_tensor_qtype}) produced {result} bytes, expected {dst.nbytes}."
            )
        return dst

    _GGML_DEQUANT_FN_NAMES = {
        "Q4_0": "dequantize_row_q4_0",
        "Q4_1": "dequantize_row_q4_1",
        "Q5_0": "dequantize_row_q5_0",
        "Q5_1": "dequantize_row_q5_1",
        "Q8_0": "dequantize_row_q8_0",
        "Q8_1": "dequantize_row_q8_1",
        "Q2_K": "dequantize_row_q2_K",
        "Q3_K": "dequantize_row_q3_K",
        "Q4_K": "dequantize_row_q4_K",
        "Q5_K": "dequantize_row_q5_K",
        "Q6_K": "dequantize_row_q6_K",
        "Q8_K": "dequantize_row_q8_K",
        "IQ2_XXS": "dequantize_row_iq2_xxs",
        "IQ2_XS": "dequantize_row_iq2_xs",
        "IQ3_XXS": "dequantize_row_iq3_xxs",
        "IQ1_S": "dequantize_row_iq1_s",
        "IQ4_NL": "dequantize_row_iq4_nl",
        "IQ3_S": "dequantize_row_iq3_s",
        "IQ2_S": "dequantize_row_iq2_s",
        "IQ4_XS": "dequantize_row_iq4_xs",
        "IQ1_M": "dequantize_row_iq1_m",
        "TQ1_0": "dequantize_row_tq1_0",
        "TQ2_0": "dequantize_row_tq2_0",
        "MXFP4": "dequantize_row_mxfp4",
        "NVFP4": "dequantize_row_nvfp4",
        "Q1_0": "dequantize_row_q1_0",
        "Q1_0_g128": "dequantize_row_q1_0",
        "Q2_0": "dequantize_row_q2_0",
    }

    def dequantize_rows(
        self,
        data: np.ndarray,
        gguf_tensor_qtype: str,
        out_shape: Tuple[int, ...],
    ) -> np.ndarray:
        """Dequantize packed ggml rows via the native `dequantize_row_*` kernels."""

        info = _GGUF_TYPE_INFO.get(gguf_tensor_qtype)
        if info is None:
            raise NotImplementedError(f"Unknown GGUF tensor type `{gguf_tensor_qtype}`.")
        raw = np.ascontiguousarray(data, dtype=np.uint8)
        if raw.ndim == 0 or raw.shape[-1] % info["type_size"]:
            raise ValueError("GGML dequantization requires complete packed rows.")
        expected_shape = (*raw.shape[:-1], raw.shape[-1] // info["type_size"] * info["block_size"])
        if tuple(out_shape) != expected_shape:
            raise ValueError(f"GGML dequantized shape must be {expected_shape}, got {out_shape}.")

        fn = self._dequantize_fns.get(gguf_tensor_qtype)
        if fn is None:
            name = self._GGML_DEQUANT_FN_NAMES.get(gguf_tensor_qtype)
            if name is None:
                raise NotImplementedError(
                    f"ggml has no dequantize_row kernel for `{gguf_tensor_qtype}`."
                )
            fn = getattr(self._ggml_base, name, None)
            if fn is None:
                raise NotImplementedError(
                    f"ggml shared library lacks `{name}` for `{gguf_tensor_qtype}`."
                )
            fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
            fn.restype = None
            self._dequantize_fns[gguf_tensor_qtype] = fn

        output = np.empty(out_shape, dtype=np.float32)
        fn(
            raw.ctypes.data_as(ctypes.c_void_p),
            output.ctypes.data_as(ctypes.c_void_p),
            int(output.size),
        )
        return output

    @staticmethod
    def _cpu_threads() -> int:
        override = os.environ.get("GPTQMODEL_GGUF_CPP_THREADS")
        if override is not None:
            try:
                return max(1, int(override))
            except ValueError:
                pass
        return max(1, torch.get_num_threads())

    def cpu_available(self) -> Tuple[bool, Optional[Exception]]:
        try:
            self._get_cpu_backend()
        except Exception as exc:
            return False, exc
        return True, None

    def cuda_available(self) -> Tuple[bool, Optional[Exception]]:
        try:
            if self._ggml_cuda is None:
                raise RuntimeError("llama-cpp-python was built without GGML CUDA support.")
            if not torch.cuda.is_available():
                raise RuntimeError("Torch CUDA is unavailable.")
            device_count = int(self._ggml_cuda.ggml_backend_cuda_get_device_count())
            if device_count <= 0:
                raise RuntimeError("GGML CUDA backend found no CUDA devices.")
            self._get_cuda_backend(0)
        except Exception as exc:
            return False, exc
        return True, None

    def _get_cpu_backend(self) -> int:
        with self._cpu_backend_lock:
            if self._cpu_backend is None:
                backend = self._ggml_cpu.ggml_backend_cpu_init()
                if not backend:
                    raise RuntimeError("GGUFCppKernel failed to initialize GGML CPU backend.")
                self._ggml_cpu.ggml_backend_cpu_set_n_threads(backend, self._cpu_threads())
                self._cpu_backend = backend
            return self._cpu_backend

    def _get_cuda_backend(self, device_index: int) -> int:
        if self._ggml_cuda is None:
            raise RuntimeError("llama-cpp-python was built without GGML CUDA support.") from self._ggml_cuda_error
        if device_index < 0:
            device_index = 0
        device_count = int(self._ggml_cuda.ggml_backend_cuda_get_device_count())
        if device_index >= device_count:
            raise RuntimeError(
                f"GGML CUDA backend device index `{device_index}` is out of range for `{device_count}` devices."
            )
        if device_index not in self._cuda_backends:
            backend = self._ggml_cuda.ggml_backend_cuda_init(device_index)
            if not backend:
                raise RuntimeError(f"GGUFCudaKernel failed to initialize GGML CUDA backend for device `{device_index}`.")
            self._cuda_backends[device_index] = backend
        return self._cuda_backends[device_index]

    @staticmethod
    def _normalize_qweight_cpu(qweight: torch.Tensor) -> torch.Tensor:
        qweight_cpu = qweight.detach()
        if qweight_cpu.device.type != "cpu":
            qweight_cpu = qweight_cpu.to(device="cpu")
        if not qweight_cpu.is_contiguous():
            qweight_cpu = qweight_cpu.contiguous()
        return qweight_cpu

    @staticmethod
    def _normalize_input_cpu(x: torch.Tensor, padded_in_features: int) -> torch.Tensor:
        x_cpu = x.detach().to(device="cpu", dtype=torch.float32)
        if x_cpu.shape[-1] != padded_in_features:
            x_cpu = F.pad(x_cpu, (0, padded_in_features - x_cpu.shape[-1]))
        if not x_cpu.is_contiguous():
            x_cpu = x_cpu.contiguous()
        return x_cpu

    def _normalize_input_cuda(
        self,
        x: torch.Tensor,
        padded_in_features: int,
    ) -> tuple[torch.Tensor, int]:
        x_cuda = x.detach()
        if x_cuda.dtype == torch.float16:
            ggml_input_type = self.ggml_type_f16
        elif x_cuda.dtype == torch.float32:
            ggml_input_type = self.ggml_type_f32
        elif x_cuda.dtype == torch.bfloat16:
            # ggml in llama-cpp-python does not expose BF16 here, so use native CUDA fp16.
            x_cuda = x_cuda.to(dtype=torch.float16)
            ggml_input_type = self.ggml_type_f16
        else:
            raise RuntimeError(
                "GGUFCudaKernel only supports float16, bfloat16, or float32 inputs."
            )

        if x_cuda.shape[-1] != padded_in_features:
            x_cuda = F.pad(x_cuda, (0, padded_in_features - x_cuda.shape[-1]))
        if not x_cuda.is_contiguous():
            x_cuda = x_cuda.contiguous()
        return x_cuda, ggml_input_type

    @staticmethod
    def _normalize_qweight_cuda(qweight: torch.Tensor, device: torch.device) -> torch.Tensor:
        qweight_cuda = qweight.detach()
        if qweight_cuda.device != device:
            qweight_cuda = qweight_cuda.to(device=device)
        if not qweight_cuda.is_contiguous():
            qweight_cuda = qweight_cuda.contiguous()
        return qweight_cuda

    @staticmethod
    def _torch_dtype_from_ggml_element_size(output_element_size: int, *, kernel_name: str) -> torch.dtype:
        if output_element_size == 4:
            return torch.float32
        if output_element_size == 2:
            return torch.float16
        raise RuntimeError(
            f"{kernel_name} received unsupported GGML output element size `{output_element_size}`."
        )

    def _run_quantized_matmul(
        self,
        *,
        backend: int,
        qweight_cpu: torch.Tensor,
        x_cpu: torch.Tensor,
        gguf_tensor_qtype: str,
        padded_in_features: int,
        out_features: int,
        kernel_name: str,
    ) -> torch.Tensor:

        ctx = self._ggml_base.ggml_init(
            _GGMLInitParams(
                mem_size=self.GGML_METADATA_BYTES,
                mem_buffer=None,
                no_alloc=True,
            )
        )
        if not ctx:
            raise RuntimeError(f"{kernel_name} failed to initialize GGML metadata context.")

        buffer = None
        try:
            weight_tensor = self._ggml_base.ggml_new_tensor_2d(
                ctx,
                self.ggml_qtypes[gguf_tensor_qtype],
                padded_in_features,
                out_features,
            )
            input_tensor = self._ggml_base.ggml_new_tensor_2d(
                ctx,
                self.ggml_type_f32,
                padded_in_features,
                x_cpu.shape[0],
            )
            if not weight_tensor or not input_tensor:
                raise RuntimeError(f"{kernel_name} failed to create GGML tensors.")

            self._ggml_base.ggml_set_input(input_tensor)
            output_tensor = self._ggml_base.ggml_mul_mat(ctx, weight_tensor, input_tensor)
            if not output_tensor:
                raise RuntimeError(f"{kernel_name} failed to create GGML matmul node.")

            graph = self._ggml_base.ggml_new_graph(ctx)
            if not graph:
                raise RuntimeError(f"{kernel_name} failed to allocate GGML graph.")
            self._ggml_base.ggml_build_forward_expand(graph, output_tensor)

            buffer = self._ggml_base.ggml_backend_alloc_ctx_tensors(ctx, backend)
            if not buffer:
                raise RuntimeError(f"{kernel_name} failed to allocate GGML backend tensors.")

            self._ggml_base.ggml_backend_tensor_set(
                weight_tensor,
                ctypes.c_void_p(qweight_cpu.data_ptr()),
                0,
                qweight_cpu.numel() * qweight_cpu.element_size(),
            )
            self._ggml_base.ggml_backend_tensor_set(
                input_tensor,
                ctypes.c_void_p(x_cpu.data_ptr()),
                0,
                x_cpu.numel() * x_cpu.element_size(),
            )

            status = self._ggml_base.ggml_backend_graph_compute(backend, graph)
            if status != 0:
                raise RuntimeError(f"{kernel_name} GGML graph compute failed with status={status}.")
            self._ggml_base.ggml_backend_synchronize(backend)

            output_nbytes = self._ggml_base.ggml_nbytes(output_tensor)
            output_element_size = self._ggml_base.ggml_element_size(output_tensor)
            if output_element_size == 4:
                output_dtype = torch.float32
            elif output_element_size == 2:
                output_dtype = torch.float16
            else:
                raise RuntimeError(
                    f"{kernel_name} received unsupported GGML output element size `{output_element_size}`."
                )

            output = torch.empty((x_cpu.shape[0], out_features), dtype=output_dtype, device="cpu")
            expected_nbytes = output.numel() * output.element_size()
            if expected_nbytes != output_nbytes:
                raise RuntimeError(
                    f"{kernel_name} GGML output size mismatch: expected {expected_nbytes}, got {output_nbytes}."
                )

            self._ggml_base.ggml_backend_tensor_get(
                output_tensor,
                ctypes.c_void_p(output.data_ptr()),
                0,
                output_nbytes,
            )
            return output
        finally:
            if buffer:
                self._ggml_base.ggml_backend_buffer_free(buffer)
            self._ggml_base.ggml_free(ctx)

    def build_quantized_matmul_cuda_plan(
        self,
        *,
        backend: int,
        qweight: torch.Tensor,
        gguf_tensor_qtype: str,
        padded_in_features: int,
        out_features: int,
        rows: int,
        input_ggml_type: int,
        kernel_name: str,
    ) -> _GGMLMatmulPlan:
        ctx = self._ggml_base.ggml_init(
            _GGMLInitParams(
                mem_size=self.GGML_METADATA_BYTES,
                mem_buffer=None,
                no_alloc=True,
            )
        )
        if not ctx:
            raise RuntimeError(f"{kernel_name} failed to initialize GGML metadata context.")

        buffer = None
        try:
            weight_tensor = self._ggml_base.ggml_new_tensor_2d(
                ctx,
                self.ggml_qtypes[gguf_tensor_qtype],
                padded_in_features,
                out_features,
            )
            input_tensor = self._ggml_base.ggml_new_tensor_2d(
                ctx,
                input_ggml_type,
                padded_in_features,
                rows,
            )
            if not weight_tensor or not input_tensor:
                raise RuntimeError(f"{kernel_name} failed to create GGML tensors.")

            self._ggml_base.ggml_set_input(input_tensor)
            output_tensor = self._ggml_base.ggml_mul_mat(ctx, weight_tensor, input_tensor)
            if not output_tensor:
                raise RuntimeError(f"{kernel_name} failed to create GGML matmul node.")

            graph = self._ggml_base.ggml_new_graph(ctx)
            if not graph:
                raise RuntimeError(f"{kernel_name} failed to allocate GGML graph.")
            self._ggml_base.ggml_build_forward_expand(graph, output_tensor)

            buffer = self._ggml_base.ggml_backend_alloc_ctx_tensors(ctx, backend)
            if not buffer:
                raise RuntimeError(f"{kernel_name} failed to allocate GGML backend tensors.")

            self._ggml_base.ggml_backend_tensor_set(
                weight_tensor,
                ctypes.c_void_p(qweight.data_ptr()),
                0,
                qweight.numel() * qweight.element_size(),
            )
            output_nbytes = self._ggml_base.ggml_nbytes(output_tensor)
            output_dtype = self._torch_dtype_from_ggml_element_size(
                int(self._ggml_base.ggml_element_size(output_tensor)),
                kernel_name=kernel_name,
            )
            return _GGMLMatmulPlan(
                ctx=ctx,
                buffer=buffer,
                weight_tensor=weight_tensor,
                input_tensor=input_tensor,
                output_tensor=output_tensor,
                graph=graph,
                rows=rows,
                out_features=out_features,
                output_nbytes=output_nbytes,
                output_dtype=output_dtype,
                backend_buffer_free=self._ggml_base.ggml_backend_buffer_free,
                ctx_free=self._ggml_base.ggml_free,
            )
        except Exception:
            if buffer:
                self._ggml_base.ggml_backend_buffer_free(buffer)
            self._ggml_base.ggml_free(ctx)
            raise

    def run_quantized_matmul_cuda_plan(
        self,
        *,
        backend: int,
        plan: _GGMLMatmulPlan,
        x: torch.Tensor,
        kernel_name: str,
    ) -> torch.Tensor:
        self._ggml_base.ggml_backend_tensor_set(
            plan.input_tensor,
            ctypes.c_void_p(x.data_ptr()),
            0,
            x.numel() * x.element_size(),
        )
        status = self._ggml_base.ggml_backend_graph_compute(backend, plan.graph)
        if status != 0:
            raise RuntimeError(f"{kernel_name} GGML graph compute failed with status={status}.")
        self._ggml_base.ggml_backend_synchronize(backend)

        output = torch.empty((plan.rows, plan.out_features), dtype=plan.output_dtype, device=x.device)
        expected_nbytes = output.numel() * output.element_size()
        if expected_nbytes != plan.output_nbytes:
            raise RuntimeError(
                f"{kernel_name} GGML output size mismatch: expected {expected_nbytes}, got {plan.output_nbytes}."
            )
        self._ggml_base.ggml_backend_tensor_get(
            plan.output_tensor,
            ctypes.c_void_p(output.data_ptr()),
            0,
            plan.output_nbytes,
        )
        return output

    # Q8_1 and Q8_K are KV-cache/auxiliary types upstream with no vec_dot
    # kernel for an F32 source, and ggml_mul_mat dereferences the missing
    # pointer instead of erroring, so they must never reach graph compute.
    _GGML_MATMUL_UNSUPPORTED = frozenset({"Q8_1", "Q8_K"})

    def quantized_matmul_cpu(
        self,
        *,
        qweight: torch.Tensor,
        x: torch.Tensor,
        gguf_tensor_qtype: str,
        padded_in_features: int,
        out_features: int,
    ) -> torch.Tensor:
        if x.device.type != "cpu":
            raise RuntimeError("GGUFCppKernel only supports CPU input tensors.")
        if gguf_tensor_qtype in self._GGML_MATMUL_UNSUPPORTED:
            raise NotImplementedError(
                f"ggml CPU matmul does not support `{gguf_tensor_qtype}` weights."
            )
        info = _GGUF_TYPE_INFO.get(gguf_tensor_qtype)
        if info is None or gguf_tensor_qtype not in self.ggml_qtypes:
            raise NotImplementedError(f"ggml CPU matmul does not support `{gguf_tensor_qtype}` weights.")
        if padded_in_features <= 0 or padded_in_features % info["block_size"] or out_features <= 0:
            raise ValueError("GGML matmul requires positive dimensions and complete weight blocks.")
        expected_shape = (out_features, padded_in_features // info["block_size"] * info["type_size"])
        if qweight.dtype != torch.uint8 or tuple(qweight.shape) != expected_shape:
            raise ValueError(f"GGML matmul requires packed uint8 weights of shape {expected_shape}.")
        if x.ndim != 2 or not 0 < x.shape[-1] <= padded_in_features:
            raise ValueError("GGML matmul input must be a 2D matrix within the padded weight width.")
        if x.shape[0] == 0:
            return torch.empty((0, out_features), device=x.device, dtype=torch.float32)

        # ggml's CPU backend owns mutable graph scratch shared by all callers.
        with self._cpu_backend_lock:
            return self._run_quantized_matmul(
                backend=self._get_cpu_backend(),
                qweight_cpu=self._normalize_qweight_cpu(qweight),
                x_cpu=self._normalize_input_cpu(x, padded_in_features),
                gguf_tensor_qtype=gguf_tensor_qtype,
                padded_in_features=padded_in_features,
                out_features=out_features,
                kernel_name="GGUFCppKernel",
            )

    def quantized_matmul_cuda(
        self,
        *,
        qweight: torch.Tensor,
        x: torch.Tensor,
        gguf_tensor_qtype: str,
        padded_in_features: int,
        out_features: int,
        plan: _GGMLMatmulPlan | None = None,
    ) -> tuple[torch.Tensor, _GGMLMatmulPlan]:
        if x.device.type != "cuda":
            raise RuntimeError("GGUFCudaKernel only supports CUDA input tensors.")

        device = x.device
        backend = self._get_cuda_backend(0 if device.index is None else device.index)
        x_cuda, input_ggml_type = self._normalize_input_cuda(x, padded_in_features)
        if plan is None:
            qweight_cuda = self._normalize_qweight_cuda(qweight, device=device)
            plan = self.build_quantized_matmul_cuda_plan(
                backend=backend,
                qweight=qweight_cuda,
                gguf_tensor_qtype=gguf_tensor_qtype,
                padded_in_features=padded_in_features,
                out_features=out_features,
                rows=x_cuda.shape[0],
                input_ggml_type=input_ggml_type,
                kernel_name="GGUFCudaKernel",
            )
        output = self.run_quantized_matmul_cuda_plan(
            backend=backend,
            plan=plan,
            x=x_cuda,
            kernel_name="GGUFCudaKernel",
        )
        return output, plan


_GGML_BRIDGE: Optional[_GGMLBridge] = None
_GGML_BRIDGE_LOCK = threading.Lock()


def _get_ggml_bridge() -> _GGMLBridge:
    global _GGML_BRIDGE
    with _GGML_BRIDGE_LOCK:
        if _GGML_BRIDGE is None:
            _GGML_BRIDGE = _GGMLBridge()
        return _GGML_BRIDGE


class GGUFCppKernel(GGUFTorchLinear):
    SUPPORTS_BACKENDS = [BACKEND.GGUF_CPP_CPU]
    SUPPORTS_METHODS = [METHOD.GGUF]
    SUPPORTS_FORMATS = {FORMAT.GGUF: 25}
    SUPPORTS_BITS = [1, 2, 3, 4, 5, 6, 8]
    SUPPORTS_SHARDS = True
    SUPPORTS_TRAINING = False
    SUPPORTS_AUTO_PADDING = True
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [1]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [1]
    SUPPORTS_DEVICES = [DEVICE.CPU]
    SUPPORTS_PLATFORM = [PLATFORM.ALL]
    SUPPORTS_PACK_DTYPES = [torch.int8, torch.int16, torch.int32]
    SUPPORTS_ADAPTERS = [Lora]
    SUPPORTS_DTYPES = [torch.float16, torch.bfloat16, torch.float32]

    REQUIRES_FORMAT_V2 = False
    AUTOTUNE = False

    QUANT_TYPE = "gguf"

    pack = None
    pack_block = None
    pack_gpu = None
    pack_original = None

    def __init__(
        self,
        bits,
        group_size: int,
        sym: bool,
        desc_act: bool,
        in_features: int,
        out_features: int,
        bias: bool = False,
        pack_dtype: torch.dtype = torch.int32,
        adapter: Adapter = None,
        register_buffers: bool = True,
        **kwargs,
    ):
        kwargs.setdefault("backend", BACKEND.GGUF_CPP_CPU)
        super().__init__(
            bits=bits,
            group_size=group_size,
            sym=sym,
            desc_act=desc_act,
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            pack_dtype=pack_dtype,
            adapter=adapter,
            register_buffers=register_buffers,
            **kwargs,
        )
        self._gguf_cpu_autotune_cache: Dict[tuple, bool] = {}

    @classmethod
    def validate_once(cls) -> Tuple[bool, Optional[Exception]]:
        try:
            return _get_ggml_bridge().cpu_available()
        except Exception as exc:
            return False, exc

    def _forward_ggml_cpu(self, x_flat: torch.Tensor) -> torch.Tensor:
        return _get_ggml_bridge().quantized_matmul_cpu(
            qweight=self.qweight,
            x=x_flat,
            gguf_tensor_qtype=self.gguf_tensor_qtype,
            padded_in_features=self.padded_in_features,
            out_features=self.out_features,
        )

    def _forward_torch_cpu(self, x_flat: torch.Tensor) -> torch.Tensor:
        if self._should_use_fused_k_forward(x_flat):
            return self._forward_fused_k(x_flat)
        return self._forward_dequant_matmul(x_flat)

    def _use_ggml_cpu_matmul(self, x_flat: torch.Tensor) -> bool:
        if self.gguf_tensor_qtype in _GGMLBridge._GGML_MATMUL_UNSUPPORTED:
            return False
        override = os.environ.get("GPTQMODEL_GGUF_CPP_NATIVE_MATMUL")
        if override is not None:
            return override.strip().lower() not in {"0", "false", "off"}

        key = (x_flat.shape[0], x_flat.dtype, self.gguf_tensor_qtype)
        cached = self._gguf_cpu_autotune_cache.get(key)
        if cached is None:
            try:
                ggml_time = self._benchmark_forward_runner(
                    lambda: self._forward_ggml_cpu(x_flat), device=x_flat.device
                )
                torch_time = self._benchmark_forward_runner(
                    lambda: self._forward_torch_cpu(x_flat), device=x_flat.device
                )
                cached = ggml_time <= torch_time
            except Exception:
                # Any benchmark failure (including qtypes ggml cannot matmul)
                # falls back to the always-correct torch path.
                cached = False
            self._gguf_cpu_autotune_cache[key] = cached
        return cached

    def clear_autotune(self):
        super().clear_autotune()
        self._gguf_cpu_autotune_cache = {}

    def forward(self, x: torch.Tensor):
        if x.ndim == 0 or x.shape[-1] != self.in_features:
            raise ValueError(f"Expected input with last dimension {self.in_features}, got {tuple(x.shape)}.")
        original_shape = x.shape[:-1] + (self.out_features,)
        x_flat = x.reshape(-1, x.shape[-1])
        if x_flat.device.type != "cpu":
            raise RuntimeError(
                f"{self.__class__.__name__} only supports CPU inference. "
                "Load GGUF models on CPU or use BACKEND.GGUF_CPP_CUDA or BACKEND.GGUF_TORCH for CUDA inference."
            )

        if self._use_ggml_cpu_matmul(x_flat):
            output = self._forward_ggml_cpu(x_flat)
        else:
            output = self._forward_torch_cpu(x_flat)
        if output.dtype != x_flat.dtype:
            output = output.to(dtype=x_flat.dtype)

        if self.bias is not None:
            bias = self.bias
            if bias.device != output.device or bias.dtype != output.dtype:
                bias = bias.to(device=output.device, dtype=output.dtype)
            output = output + bias

        if self.adapter:
            output = self.adapter.apply(x=x_flat, out=output)

        return output.reshape(original_shape)

class GGUFCudaKernel(GGUFTorchLinear):
    SUPPORTS_BACKENDS = [BACKEND.GGUF_CPP_CUDA]
    SUPPORTS_METHODS = [METHOD.GGUF]
    SUPPORTS_FORMATS = {FORMAT.GGUF: 35}
    SUPPORTS_BITS = [1, 2, 3, 4, 5, 6, 8]
    SUPPORTS_SHARDS = True
    SUPPORTS_TRAINING = False
    SUPPORTS_AUTO_PADDING = True
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [1]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [1]
    SUPPORTS_DEVICES = [DEVICE.CUDA]
    SUPPORTS_PLATFORM = [PLATFORM.ALL]
    SUPPORTS_PACK_DTYPES = [torch.int8, torch.int16, torch.int32]
    SUPPORTS_ADAPTERS = [Lora]
    SUPPORTS_DTYPES = [torch.float16, torch.bfloat16, torch.float32]

    REQUIRES_FORMAT_V2 = False
    AUTOTUNE = False

    QUANT_TYPE = "gguf"

    pack = None
    pack_block = None
    pack_gpu = None
    pack_original = None

    def __init__(
        self,
        bits,
        group_size: int,
        sym: bool,
        desc_act: bool,
        in_features: int,
        out_features: int,
        bias: bool = False,
        pack_dtype: torch.dtype = torch.int32,
        adapter: Adapter = None,
        register_buffers: bool = True,
        **kwargs,
    ):
        kwargs.setdefault("backend", BACKEND.GGUF_CPP_CUDA)
        super().__init__(
            bits=bits,
            group_size=group_size,
            sym=sym,
            desc_act=desc_act,
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            pack_dtype=pack_dtype,
            adapter=adapter,
            register_buffers=register_buffers,
            **kwargs,
        )
        self._ggml_cuda_plans: Dict[tuple[int, int, torch.dtype, int], _GGMLMatmulPlan] = {}

    @classmethod
    def validate_once(cls) -> Tuple[bool, Optional[Exception]]:
        try:
            return _get_ggml_bridge().cuda_available()
        except Exception as exc:
            return False, exc

    def clear_weight_cache(self) -> None:
        for plan in self._ggml_cuda_plans.values():
            plan.close()
        self._ggml_cuda_plans.clear()
        return super().clear_weight_cache()

    def _cuda_plan_key(self, x_flat: torch.Tensor) -> tuple[int, int, torch.dtype, int]:
        device_index = 0 if x_flat.device.index is None else x_flat.device.index
        return (
            device_index,
            x_flat.shape[0],
            x_flat.dtype,
            self.qweight.data_ptr(),
        )

    def forward(self, x: torch.Tensor):
        original_shape = x.shape[:-1] + (self.out_features,)
        x_flat = x.reshape(-1, x.shape[-1])
        if x_flat.device.type != "cuda":
            raise RuntimeError(
                f"{self.__class__.__name__} only supports CUDA inference. "
                "Load GGUF models on CUDA or use BACKEND.GGUF_CPP_CPU or BACKEND.GGUF_TORCH for CPU inference."
            )

        plan_key = self._cuda_plan_key(x_flat)
        output, plan = _get_ggml_bridge().quantized_matmul_cuda(
            qweight=self.qweight,
            x=x_flat,
            gguf_tensor_qtype=self.gguf_tensor_qtype,
            padded_in_features=self.padded_in_features,
            out_features=self.out_features,
            plan=self._ggml_cuda_plans.get(plan_key),
        )
        self._ggml_cuda_plans.setdefault(plan_key, plan)
        if output.device != x_flat.device or output.dtype != x_flat.dtype:
            output = output.to(device=x_flat.device, dtype=x_flat.dtype)

        if self.bias is not None:
            bias = self.bias
            if bias.device != output.device or bias.dtype != output.dtype:
                bias = bias.to(device=output.device, dtype=output.dtype)
            output = output + bias

        if self.adapter:
            output = self.adapter.apply(x=x_flat, out=output)

        return output.reshape(original_shape)


__all__ = ["GGUFCppKernel", "GGUFCudaKernel"]
