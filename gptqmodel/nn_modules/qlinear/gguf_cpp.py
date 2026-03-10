# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from ...adapter.adapter import Adapter, Lora
from ...models._const import DEVICE, PLATFORM
from ...quantization.config import FORMAT, METHOD
from ...utils.backend import BACKEND
from .gguf import GGUFTorchQuantLinear


try:
    import llama_cpp as _llama_cpp_pkg
    from llama_cpp import llama_cpp as _llama_cpp_lib
    from llama_cpp._ctypes_extensions import load_shared_library as _llama_cpp_load_shared_library

    _LLAMA_CPP_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - optional dependency
    _llama_cpp_pkg = None
    _llama_cpp_lib = None
    _llama_cpp_load_shared_library = None
    _LLAMA_CPP_IMPORT_ERROR = exc


class _GGMLInitParams(ctypes.Structure):
    _fields_ = [
        ("mem_size", ctypes.c_size_t),
        ("mem_buffer", ctypes.c_void_p),
        ("no_alloc", ctypes.c_bool),
    ]


class _GGMLBridge:
    GGML_METADATA_BYTES = 1 << 20

    def __init__(self) -> None:
        if _LLAMA_CPP_IMPORT_ERROR is not None:
            raise ModuleNotFoundError(
                "GGUFCppKernel requires `llama-cpp-python` to be installed."
            ) from _LLAMA_CPP_IMPORT_ERROR

        lib_dir = Path(_llama_cpp_pkg.__file__).resolve().parent / "lib"
        self._ggml_base = _llama_cpp_load_shared_library("ggml-base", lib_dir)
        self._ggml_cpu = _llama_cpp_load_shared_library("ggml-cpu", lib_dir)
        self._bind_functions()

        self.ggml_type_f32 = int(_llama_cpp_lib.GGML_TYPE_F32)
        self.ggml_qtypes = {
            "Q4_0": int(_llama_cpp_lib.GGML_TYPE_Q4_0),
            "Q8_0": int(_llama_cpp_lib.GGML_TYPE_Q8_0),
            "Q4_K": int(_llama_cpp_lib.GGML_TYPE_Q4_K),
            "Q5_K": int(_llama_cpp_lib.GGML_TYPE_Q5_K),
            "Q6_K": int(_llama_cpp_lib.GGML_TYPE_Q6_K),
        }

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

    @staticmethod
    def _cpu_threads() -> int:
        override = os.environ.get("GPTQMODEL_GGUF_CPP_THREADS")
        if override is not None:
            try:
                return max(1, int(override))
            except ValueError:
                pass
        return max(1, torch.get_num_threads())

    def quantized_matmul(
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

        qweight_cpu = qweight.detach()
        if qweight_cpu.device.type != "cpu":
            qweight_cpu = qweight_cpu.to(device="cpu")
        if not qweight_cpu.is_contiguous():
            qweight_cpu = qweight_cpu.contiguous()

        x_cpu = x.detach().to(dtype=torch.float32)
        if x_cpu.shape[-1] != padded_in_features:
            x_cpu = F.pad(x_cpu, (0, padded_in_features - x_cpu.shape[-1]))
        if not x_cpu.is_contiguous():
            x_cpu = x_cpu.contiguous()

        ctx = self._ggml_base.ggml_init(
            _GGMLInitParams(
                mem_size=self.GGML_METADATA_BYTES,
                mem_buffer=None,
                no_alloc=True,
            )
        )
        if not ctx:
            raise RuntimeError("GGUFCppKernel failed to initialize GGML metadata context.")

        backend = self._ggml_cpu.ggml_backend_cpu_init()
        if not backend:
            self._ggml_base.ggml_free(ctx)
            raise RuntimeError("GGUFCppKernel failed to initialize GGML CPU backend.")

        buffer = None
        try:
            self._ggml_cpu.ggml_backend_cpu_set_n_threads(backend, self._cpu_threads())

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
                x.shape[0],
            )
            if not weight_tensor or not input_tensor:
                raise RuntimeError("GGUFCppKernel failed to create GGML tensors.")

            self._ggml_base.ggml_set_input(input_tensor)
            output_tensor = self._ggml_base.ggml_mul_mat(ctx, weight_tensor, input_tensor)
            if not output_tensor:
                raise RuntimeError("GGUFCppKernel failed to create GGML matmul node.")

            graph = self._ggml_base.ggml_new_graph(ctx)
            if not graph:
                raise RuntimeError("GGUFCppKernel failed to allocate GGML graph.")
            self._ggml_base.ggml_build_forward_expand(graph, output_tensor)

            buffer = self._ggml_base.ggml_backend_alloc_ctx_tensors(ctx, backend)
            if not buffer:
                raise RuntimeError("GGUFCppKernel failed to allocate GGML backend tensors.")

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
                raise RuntimeError(f"GGUFCppKernel GGML graph compute failed with status={status}.")
            self._ggml_base.ggml_backend_synchronize(backend)

            output_nbytes = self._ggml_base.ggml_nbytes(output_tensor)
            output_element_size = self._ggml_base.ggml_element_size(output_tensor)
            if output_element_size == 4:
                output_dtype = torch.float32
            elif output_element_size == 2:
                output_dtype = torch.float16
            else:
                raise RuntimeError(
                    f"GGUFCppKernel received unsupported GGML output element size `{output_element_size}`."
                )

            output = torch.empty((x.shape[0], out_features), dtype=output_dtype, device="cpu")
            expected_nbytes = output.numel() * output.element_size()
            if expected_nbytes != output_nbytes:
                raise RuntimeError(
                    f"GGUFCppKernel GGML output size mismatch: expected {expected_nbytes}, got {output_nbytes}."
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
            self._ggml_base.ggml_backend_free(backend)
            self._ggml_base.ggml_free(ctx)


_GGML_BRIDGE: Optional[_GGMLBridge] = None


def _get_ggml_bridge() -> _GGMLBridge:
    global _GGML_BRIDGE
    if _GGML_BRIDGE is None:
        _GGML_BRIDGE = _GGMLBridge()
    return _GGML_BRIDGE


class GGUFCppKernel(GGUFTorchQuantLinear):
    SUPPORTS_BACKENDS = [BACKEND.GGUF_CPP]
    SUPPORTS_METHODS = [METHOD.GGUF]
    SUPPORTS_FORMATS = {FORMAT.GGUF: 25}
    SUPPORTS_BITS = [4, 5, 6, 8]
    SUPPORTS_GROUP_SIZE = [-1]
    SUPPORTS_DESC_ACT = [False]
    SUPPORTS_SYM = [True]
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
        kwargs.setdefault("backend", BACKEND.GGUF_CPP)
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

    @classmethod
    def validate_once(cls) -> Tuple[bool, Optional[Exception]]:
        try:
            _get_ggml_bridge()
        except Exception as exc:
            return False, exc
        return True, None

    def forward(self, x: torch.Tensor):
        original_shape = x.shape[:-1] + (self.out_features,)
        x_flat = x.reshape(-1, x.shape[-1])
        if x_flat.device.type != "cpu":
            raise RuntimeError(
                f"{self.__class__.__name__} only supports CPU inference. "
                "Load GGUF models on CPU or use BACKEND.GGUF_TORCH for non-CPU inference."
            )

        output = _get_ggml_bridge().quantized_matmul(
            qweight=self.qweight,
            x=x_flat,
            gguf_tensor_qtype=self.gguf_tensor_qtype,
            padded_in_features=self.padded_in_features,
            out_features=self.out_features,
        )
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


__all__ = ["GGUFCppKernel"]
