# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from ...adapter.adapter import Adapter, Lora
from ...models._const import DEVICE, PLATFORM
from ...quantization.config import FORMAT, METHOD
from ...utils.backend import BACKEND
from .gguf import GGUFTorchLinear


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
        self._ggml_cuda = None
        self._ggml_cuda_error: Optional[Exception] = None
        try:
            self._ggml_cuda = _llama_cpp_load_shared_library("ggml-cuda", lib_dir)
        except Exception as exc:  # pragma: no cover - optional shared library
            self._ggml_cuda_error = exc
        self._bind_functions()

        self.ggml_type_f32 = int(_llama_cpp_lib.GGML_TYPE_F32)
        self.ggml_type_f16 = int(_llama_cpp_lib.GGML_TYPE_F16)
        self.ggml_qtypes = {
            "Q4_0": int(_llama_cpp_lib.GGML_TYPE_Q4_0),
            "Q8_0": int(_llama_cpp_lib.GGML_TYPE_Q8_0),
            "Q4_K": int(_llama_cpp_lib.GGML_TYPE_Q4_K),
            "Q5_K": int(_llama_cpp_lib.GGML_TYPE_Q5_K),
            "Q6_K": int(_llama_cpp_lib.GGML_TYPE_Q6_K),
        }
        self._cpu_backend: Optional[int] = None
        self._cuda_backends: Dict[int, int] = {}

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
        if self._ggml_cuda is not None:
            self._bind(self._ggml_cuda, "ggml_backend_cuda_init", [ctypes.c_int], ctypes.c_void_p)
            self._bind(self._ggml_cuda, "ggml_backend_cuda_get_device_count", [], ctypes.c_int)

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


def _get_ggml_bridge() -> _GGMLBridge:
    global _GGML_BRIDGE
    if _GGML_BRIDGE is None:
        _GGML_BRIDGE = _GGMLBridge()
    return _GGML_BRIDGE


class GGUFCppKernel(GGUFTorchLinear):
    SUPPORTS_BACKENDS = [BACKEND.GGUF_CPP_CPU]
    SUPPORTS_METHODS = [METHOD.GGUF]
    SUPPORTS_FORMATS = {FORMAT.GGUF: 25}
    SUPPORTS_BITS = [4, 5, 6, 8]
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

    @classmethod
    def validate_once(cls) -> Tuple[bool, Optional[Exception]]:
        try:
            return _get_ggml_bridge().cpu_available()
        except Exception as exc:
            return False, exc

    def forward(self, x: torch.Tensor):
        original_shape = x.shape[:-1] + (self.out_features,)
        x_flat = x.reshape(-1, x.shape[-1])
        if x_flat.device.type != "cpu":
            raise RuntimeError(
                f"{self.__class__.__name__} only supports CPU inference. "
                "Load GGUF models on CPU or use BACKEND.GGUF_CPP_CUDA or BACKEND.GGUF_TORCH for CUDA inference."
            )

        output = _get_ggml_bridge().quantized_matmul_cpu(
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

class GGUFCudaKernel(GGUFTorchLinear):
    SUPPORTS_BACKENDS = [BACKEND.GGUF_CPP_CUDA]
    SUPPORTS_METHODS = [METHOD.GGUF]
    SUPPORTS_FORMATS = {FORMAT.GGUF: 35}
    SUPPORTS_BITS = [4, 5, 6, 8]
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
