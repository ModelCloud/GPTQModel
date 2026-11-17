# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ctypes
import functools
import mmap
import struct
import sys
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ...adapter.adapter import Adapter, Lora
from ...models._const import DEVICE, PLATFORM
from ...quantization.config import FORMAT, METHOD
from ...utils.backend import BACKEND
from ...utils.kernel import get_kernel_config
from ...utils.logger import setup_logger
from ._compile_support import COMPILE_LEVEL
from ._runtime_cache import tensor_cache_key
from .gguf import _GGUF_TYPE_INFO, GGUFTorchLinear


log = setup_logger()

# ggml's CPU buffer-from-pointer requires this alignment.
_GGML_CPU_WEIGHT_ALIGNMENT = 32

# Linux ``madvise`` advice. On a file mapping it only unmaps the range: the
# pages stay in the page cache and re-fault from it. It would discard a private
# copy-on-write page, so ``_reclaim_file_pages`` checks for those first.
_MADV_DONTNEED = 4
# ``/proc/self/pagemap`` entry bits.
_PAGEMAP_PRESENT = 1 << 63
_PAGEMAP_SWAPPED = 1 << 62
_PAGEMAP_FILE_OR_SHARED = 1 << 61


@functools.lru_cache(maxsize=1)
def _libc_madvise():
    madvise = ctypes.CDLL(None, use_errno=True).madvise
    madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    madvise.restype = ctypes.c_int
    return madvise


def _file_backed_range(start: int, end: int) -> bool:
    """Whether ``[start, end)`` lies inside one file mapping of this process."""

    try:
        with open("/proc/self/maps", encoding="ascii", errors="replace") as maps:
            for line in maps:
                fields = line.split(maxsplit=5)
                low, _, high = fields[0].partition("-")
                if int(high, 16) <= start:
                    continue
                if int(low, 16) > start or int(high, 16) < end:
                    return False
                # ``inode`` is 0 for anonymous memory, the heap and the stack.
                return len(fields) == 6 and fields[4] != "0" and not fields[5].startswith("[")
    except (OSError, ValueError, IndexError):
        return False
    return False


def _only_file_pages(start: int, end: int) -> bool:
    """Whether every page in ``[start, end)`` is unmapped or a page-cache page.

    A present anonymous page (or a swapped one) inside a file mapping is a
    private copy-on-write page: somebody wrote to the mapping, and dropping
    it would revert their bytes to the file contents.
    """

    page = mmap.PAGESIZE
    try:
        with open("/proc/self/pagemap", "rb") as pagemap:
            pagemap.seek(start // page * 8)
            entries = pagemap.read((end - start) // page * 8)
    except OSError:
        return False
    if len(entries) != (end - start) // page * 8:
        return False
    for (entry,) in struct.iter_unpack("<Q", entries):
        if entry & _PAGEMAP_SWAPPED:
            return False
        if entry & _PAGEMAP_PRESENT and not entry & _PAGEMAP_FILE_OR_SHARED:
            return False
    return True


def _reclaim_file_pages(tensor: torch.Tensor) -> int:
    """Unmap the file pages that only ``tensor`` occupies.

    A safetensors checkpoint is loaded as an mmap of the whole file. Once a
    tensor's bytes have been copied elsewhere, the pages it faulted in stay
    mapped (counted in ``RssFile``) next to the copy for the life of the
    model. Only whole pages inside the tensor are advised, so a neighbouring
    tensor that is still read in place is untouched, and the view itself stays
    valid (it re-faults from the page cache). Returns the advised byte count:
    0 when the tensor is not file backed, holds copy-on-write pages, or the
    platform has no ``madvise``.
    """

    if not sys.platform.startswith("linux") or tensor.device.type != "cpu":
        return 0
    start = tensor.data_ptr()
    end = start + tensor.numel() * tensor.element_size()
    page = mmap.PAGESIZE
    low = -(-start // page) * page
    high = end // page * page
    if high <= low or not _file_backed_range(low, high) or not _only_file_pages(low, high):
        return 0
    if _libc_madvise()(ctypes.c_void_p(low), ctypes.c_size_t(high - low), _MADV_DONTNEED) != 0:
        return 0
    return high - low


# Imported lazily by ``_require_llama_cpp``: importing llama-cpp-python costs
# tens of milliseconds and must not happen while the kernel registry is merely
# being discovered. The module attributes stay available for callers that read
# them after an explicit import.
_llama_cpp_pkg = None
_llama_cpp_lib = None
_llama_cpp_load_shared_library = None
# The failure is kept as text, not as the exception: its traceback frames chain
# through ``f_back`` to whichever caller first probed the backend (for example a
# model loader), and a process-lifetime global would keep that model alive.
_LLAMA_CPP_IMPORT_ERROR: Optional[str] = None
_LLAMA_CPP_LOADED = False
_LLAMA_CPP_LOCK = threading.Lock()


def _require_llama_cpp():
    """Import llama-cpp-python once and return its module objects.

    The lock keeps a concurrent caller from observing the "already loaded" flag
    before the module objects are assigned, which used to hand back
    ``(None, None, None)`` and turn a missing backend into an ``AttributeError``
    on ``Path(None.__file__)``.
    """

    global _LLAMA_CPP_LOADED, _LLAMA_CPP_IMPORT_ERROR
    global _llama_cpp_pkg, _llama_cpp_lib, _llama_cpp_load_shared_library

    if _LLAMA_CPP_LOADED:
        return _llama_cpp_pkg, _llama_cpp_lib, _llama_cpp_load_shared_library

    with _LLAMA_CPP_LOCK:
        if not _LLAMA_CPP_LOADED:
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
                _LLAMA_CPP_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
            _LLAMA_CPP_LOADED = True

    return _llama_cpp_pkg, _llama_cpp_lib, _llama_cpp_load_shared_library


class _GGMLInitParams(ctypes.Structure):
    _fields_ = [
        ("mem_size", ctypes.c_size_t),
        ("mem_buffer", ctypes.c_void_p),
        ("no_alloc", ctypes.c_bool),
    ]


class _GGMLWeightView:
    """``__cuda_array_interface__`` exporter for a :class:`_GGMLSharedWeight`.

    ``torch.as_tensor`` keeps the exporter alive for as long as any tensor (or
    view of one) shares the memory, and the exporter keeps the ggml weight
    alive, so the ggml buffer cannot be freed under a live ``qweight``.
    """

    def __init__(self, weight: "_GGMLSharedWeight") -> None:
        self.weight = weight
        self.__cuda_array_interface__ = {
            "shape": tuple(weight.shape),
            "typestr": "|u1",
            "data": (int(weight.data_ptr), False),
            "strides": None,
            "version": 3,
        }


class _GGMLSharedWeight:
    """One ggml weight tensor per device, referenced by every plan on it.

    The packed weight used to be created (and uploaded) inside each
    batch-size plan, so a layer with up to ``_MAX_GGML_CUDA_PLANS`` plans kept
    that many device copies of it (#4216). A plan's graph now references this
    tensor, and only its activation/output tensors live in the plan's buffer.

    The ggml buffer is also the storage of the layer's ``qweight`` once the
    kernel adopts :meth:`as_tensor` (#4387): Torch and ggml then read the same
    device bytes instead of keeping one copy each. ggml cannot read Torch's
    allocation instead: when a row is not a multiple of 512 columns, the CUDA
    buffer type reserves zero-filled padding past ``ggml_nbytes``
    (``ggml_backend_buft_get_alloc_size``) for the kernels' row over-read. The
    buffer is freed when the last reference goes away: the kernel, a plan, or
    any Torch tensor sharing the memory.
    """

    def __init__(
        self,
        *,
        ctx,
        buffer,
        tensor,
        nbytes: int,
        backend_buffer_free,
        ctx_free,
        data_ptr: int,
        shape: tuple[int, ...],
        device: torch.device,
        lock: threading.RLock | None,
    ) -> None:
        self.ctx = ctx
        self.buffer = buffer
        self.tensor = tensor
        self.nbytes = nbytes
        self.data_ptr = int(data_ptr)
        self.shape = tuple(int(dim) for dim in shape)
        self.device = device
        self._lock = lock
        self._exported = False
        self._backend_buffer_free = backend_buffer_free
        self._ctx_free = ctx_free

    def as_tensor(self) -> torch.Tensor:
        """Return a zero-copy Torch ``uint8`` tensor over the ggml weight bytes."""

        if self.device.type != "cuda" or not self.buffer:
            raise RuntimeError("only a live CUDA ggml weight can back a Torch tensor.")
        # A view created inside ``inference_mode`` would be an inference
        # tensor, and a later ``load_state_dict`` outside it could not copy
        # into the buffer.
        with torch.inference_mode(False):
            view = torch.as_tensor(_GGMLWeightView(self), device=self.device)
        if view.data_ptr() != self.data_ptr or tuple(view.shape) != self.shape:
            raise RuntimeError("ggml weight view does not alias the ggml buffer.")
        self._exported = True
        return view

    def backs(self, tensor: torch.Tensor | None) -> bool:
        """Whether ``tensor`` is the Torch view of this weight's bytes."""

        return (
            isinstance(tensor, torch.Tensor)
            and self.buffer is not None
            and tensor.device == self.device
            and tensor.dtype == torch.uint8
            and tensor.data_ptr() == self.data_ptr
            and tuple(tensor.shape) == self.shape
        )

    def close(self) -> None:
        """Free the ggml buffer; only valid once no Torch view of it is alive."""

        lock = self._lock
        if lock is None:
            self._close()
            return
        with lock:
            self._close()

    def _close(self) -> None:
        if self.buffer:
            if self._exported:
                # Torch may still have kernels in flight that read the view.
                torch.cuda.synchronize(self.device)
            self._backend_buffer_free(self.buffer)
            self.buffer = None
        if self.ctx:
            self._ctx_free(self.ctx)
            self.ctx = None
        self.tensor = None

    def __del__(self) -> None:  # pragma: no cover - best effort cleanup
        try:
            self.close()
        except Exception:
            # Destructors must not raise during GC or interpreter shutdown.
            pass


class _GGMLCudaScratch:
    """One ggml backend buffer per device, shared by every CUDA plan.

    A plan's input/output/intermediate tensors are pure scratch: the forward
    copies the activation in and the result out.  Keeping a backend buffer per
    module *and* per row count retained gigabytes outside the Torch allocator
    (issue #4387).  All plans now point at offsets inside this single arena.

    ``ggml_backend_tensor_alloc`` requires ``tensor->data == NULL``, so tensors
    cannot be moved to a grown arena.  Growth therefore frees the old buffer and
    bumps ``generation``; ``GGUFCudaKernel`` closes any plan whose generation is
    stale before it is used again.
    """

    def __init__(
        self,
        *,
        backend,
        alloc_buffer,
        buffer_free,
        buffer_get_base,
        buffer_get_size,
    ) -> None:
        self.backend = backend
        self.buffer = None
        self.base = None
        self.size = 0
        self.generation = 0
        self._alloc_buffer = alloc_buffer
        self._buffer_free = buffer_free
        self._buffer_get_base = buffer_get_base
        self._buffer_get_size = buffer_get_size

    def ensure(self, nbytes: int) -> "_GGMLCudaScratch":
        if self.buffer is not None and self.size >= nbytes:
            return self
        if self.buffer is not None:
            self._buffer_free(self.buffer)
            self.buffer = None
            self.base = None
            self.size = 0
        buffer = self._alloc_buffer(self.backend, max(int(nbytes), 1))
        if not buffer:
            raise RuntimeError(
                f"ggml CUDA scratch allocation of {nbytes} bytes failed."
            )
        self.buffer = buffer
        self.base = int(self._buffer_get_base(buffer))
        self.size = int(self._buffer_get_size(buffer))
        self.generation += 1
        return self

    def close(self) -> None:
        if self.buffer:
            self._buffer_free(self.buffer)
            self.buffer = None
        self.base = None
        self.size = 0


class _GGMLMatmulPlan:
    """Per-batch-size ggml graph.

    The quantized weight is owned by a :class:`_GGMLSharedWeight`; this plan's
    activation, output and intermediate tensors are slices of the per-device
    :class:`_GGMLCudaScratch`.  The plan keeps only graph metadata, never its
    own backend buffer.
    """

    def __init__(
        self,
        *,
        ctx,
        weight: _GGMLSharedWeight,
        input_tensor,
        output_tensor,
        graph,
        rows: int,
        out_features: int,
        device_index: int,
        backend,
        tensors,
        tensor_offsets,
        scratch_nbytes: int,
        output_nbytes: int,
        output_dtype: torch.dtype,
        ctx_free,
    ) -> None:
        self.ctx = ctx
        # The graph points at ``weight.tensor``; holding the owner keeps its
        # buffer alive for as long as this plan can run.
        self.weight = weight
        self.input_tensor = input_tensor
        self.output_tensor = output_tensor
        self.graph = graph
        self.rows = rows
        self.out_features = out_features
        self.device_index = device_index
        self.backend = backend
        self.tensors = tensors
        self.tensor_offsets = tensor_offsets
        self.scratch_nbytes = scratch_nbytes
        self.output_nbytes = output_nbytes
        self.output_dtype = output_dtype
        self._ctx_free = ctx_free
        self.scratch = None
        self.scratch_generation = -1

    def close(self) -> None:
        if self.ctx:
            self._ctx_free(self.ctx)
            self.ctx = None
        self.tensors = None
        self.weight = None

    def __del__(self) -> None:  # pragma: no cover - best effort cleanup
        try:
            self.close()
        except Exception:
            # Destructors must not raise during GC or interpreter shutdown.
            pass


# Q8_1 and Q8_K are KV-cache/auxiliary types upstream with no vec_dot kernel
# for an F32 source, so the CPU backend must never hand them to ggml either:
# ggml_mul_mat dereferences the missing pointer instead of raising.
_GGML_CPU_MATMUL_UNSUPPORTED = frozenset({"Q8_1", "Q8_K"})

# The CUDA backend has no convert/vec_dot kernel for the ternary types on top of
# the two above, and `GGML_ASSERT(convert_func != nullptr)` in ggml-cuda.cu
# aborts the whole process rather than raising. The CPU backend does handle
# TQ1_0/TQ2_0, so this set must not be reused there.
_GGML_CUDA_MATMUL_UNSUPPORTED = _GGML_CPU_MATMUL_UNSUPPORTED | frozenset({"TQ1_0", "TQ2_0"})
_MAX_GGML_CUDA_PLANS = 8

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
                "GGUFCppKernel requires `llama-cpp-python` to be installed "
                f"(import failed with {_LLAMA_CPP_IMPORT_ERROR})."
            )

        lib_dir = Path(llama_cpp_pkg.__file__).resolve().parent / "lib"
        self._ggml_base = load_shared_library("ggml-base", lib_dir)
        self._ggml_cpu = load_shared_library("ggml-cpu", lib_dir)
        self._ggml_cuda = None
        # Text only: the bridge is a process-lifetime singleton, so a stored
        # exception would pin the frames of the first caller that built it.
        self._ggml_cuda_error: Optional[str] = None
        try:
            self._ggml_cuda = load_shared_library("ggml-cuda", lib_dir)
        except Exception as exc:  # pragma: no cover - optional shared library
            self._ggml_cuda_error = f"{type(exc).__name__}: {exc}"
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
        # One scratch arena per device; see _GGMLCudaScratch.
        self._cuda_scratch: Dict[int, _GGMLCudaScratch] = {}
        # ggml-cuda keeps process-global state (cuBLAS handles, memory pools)
        # that is shared across devices, so entry must serialize process-wide,
        # not per device (#4378). One re-entrant lock covers every device.
        self._cuda_backend_lock = threading.RLock()
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
        self._bind(self._ggml_base, "ggml_get_data", [ctypes.c_void_p], ctypes.c_void_p)
        self._bind(self._ggml_base, "ggml_element_size", [ctypes.c_void_p], ctypes.c_size_t)
        self._bind(self._ggml_base, "ggml_backend_alloc_ctx_tensors", [ctypes.c_void_p, ctypes.c_void_p], ctypes.c_void_p)
        self._bind(self._ggml_base, "ggml_backend_get_default_buffer_type", [ctypes.c_void_p], ctypes.c_void_p)
        self._bind(self._ggml_base, "ggml_backend_buft_get_alignment", [ctypes.c_void_p], ctypes.c_size_t)
        self._bind(self._ggml_base, "ggml_backend_buft_get_alloc_size", [ctypes.c_void_p, ctypes.c_void_p], ctypes.c_size_t)
        self._bind(self._ggml_base, "ggml_backend_alloc_buffer", [ctypes.c_void_p, ctypes.c_size_t], ctypes.c_void_p)
        self._bind(self._ggml_base, "ggml_backend_buffer_get_base", [ctypes.c_void_p], ctypes.c_void_p)
        self._bind(self._ggml_base, "ggml_backend_buffer_get_size", [ctypes.c_void_p], ctypes.c_size_t)
        self._bind(self._ggml_base, "ggml_backend_tensor_alloc", [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p], ctypes.c_int)
        self._bind(self._ggml_base, "ggml_get_first_tensor", [ctypes.c_void_p], ctypes.c_void_p)
        self._bind(self._ggml_base, "ggml_get_next_tensor", [ctypes.c_void_p, ctypes.c_void_p], ctypes.c_void_p)
        self._bind(self._ggml_base, "ggml_backend_buffer_free", [ctypes.c_void_p], None)
        self._bind(self._ggml_base, "ggml_backend_free", [ctypes.c_void_p], None)
        # Wrap caller-owned host storage instead of copying it into a fresh
        # backend buffer. ``ggml_backend_tensor_alloc`` binds a tensor to an
        # already-allocated address; ``ggml_backend_cpu_buffer_from_ptr``
        # returns a non-owning CPU buffer over that address. Both live in
        # ggml-base (``ggml-backend.cpp``).
        # Optional zero-copy symbols. Older llama.cpp builds may not export
        # them; the CPU matmul then keeps using the copied-buffer path.
        self._cpu_buffer_from_ptr = getattr(
            self._ggml_base, "ggml_backend_cpu_buffer_from_ptr", None
        )
        if self._cpu_buffer_from_ptr is not None:
            self._cpu_buffer_from_ptr.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            self._cpu_buffer_from_ptr.restype = ctypes.c_void_p
        self._cpu_tensor_alloc = getattr(
            self._ggml_base, "ggml_backend_tensor_alloc", None
        )
        if self._cpu_tensor_alloc is not None:
            self._cpu_tensor_alloc.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
            ]
            self._cpu_tensor_alloc.restype = ctypes.c_int
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
        override = get_kernel_config().gguf_cpp_threads
        if override is not None:
            return max(1, int(override))
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

    def cuda_backend_lock(self, device_index: int) -> threading.RLock:
        """Return the process-wide lock serializing entry into ggml-cuda.

        ggml-cuda shares cuBLAS handles and memory pools across devices, so a
        per-device lock still lets cuda:0 and cuda:1 run inside the backend
        together and abort the process (#4378). ``device_index`` is accepted
        for call-site clarity; every device takes the same re-entrant lock.
        """

        del device_index
        return self._cuda_backend_lock

    def _get_cuda_backend(self, device_index: int) -> int:
        if self._ggml_cuda is None:
            raise RuntimeError(
                "llama-cpp-python was built without GGML CUDA support "
                f"(loading ggml-cuda failed with {self._ggml_cuda_error})."
            )
        if device_index < 0:
            device_index = 0
        device_count = int(self._ggml_cuda.ggml_backend_cuda_get_device_count())
        if device_index >= device_count:
            raise RuntimeError(
                f"GGML CUDA backend device index `{device_index}` is out of range for `{device_count}` devices."
            )
        # The dict is shared by every layer; initializing a device backend is
        # also not safe to race (it allocates the backend memory pool) and
        # ggml-cuda state is process-wide (#4378).
        with self.cuda_backend_lock(device_index):
            backend = self._cuda_backends.get(device_index)
            if backend is None:
                backend = self._ggml_cuda.ggml_backend_cuda_init(device_index)
                if not backend:
                    raise RuntimeError(
                        f"GGUFCudaKernel failed to initialize GGML CUDA backend for device `{device_index}`."
                    )
                self._cuda_backends[device_index] = backend
            return backend

    @staticmethod
    def _normalize_qweight_cpu(qweight: torch.Tensor) -> torch.Tensor:
        qweight_cpu = qweight.detach()
        if qweight_cpu.device.type != "cpu":
            qweight_cpu = qweight_cpu.to(device="cpu")
        if not qweight_cpu.is_contiguous():
            qweight_cpu = qweight_cpu.contiguous()
        return qweight_cpu

    def _can_wrap_cpu_weight(self, qweight_cpu: torch.Tensor) -> bool:
        """Whether the packed weight can be read in place by the CPU backend.

        ``ggml_backend_cpu_buffer_from_ptr`` aborts (GGML_ASSERT) on a pointer
        that is not 32-byte aligned, so alignment must be checked here, not
        recovered from an exception.
        """

        return (
            self._cpu_buffer_from_ptr is not None
            and self._cpu_tensor_alloc is not None
            and qweight_cpu.device.type == "cpu"
            and qweight_cpu.dtype == torch.uint8
            and qweight_cpu.numel() > 0
            and qweight_cpu.is_contiguous()
            and qweight_cpu.data_ptr() % _GGML_CPU_WEIGHT_ALIGNMENT == 0
        )

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
            # ggml has no BF16 input type here. FP32 preserves the BF16 range;
            # casting to FP16 can turn finite values into inf or zero.
            x_cuda = x_cuda.to(dtype=torch.float32)
            ggml_input_type = self.ggml_type_f32
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
        zero_copy_weight: bool = False,
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
        weight_buffer = None
        try:
            weight_tensor = self._ggml_base.ggml_new_tensor_2d(
                ctx,
                self.ggml_qtypes[gguf_tensor_qtype],
                padded_in_features,
                out_features,
            )
            if zero_copy_weight:
                if not weight_tensor:
                    raise RuntimeError(f"{kernel_name} failed to create the GGML weight tensor.")
                # The CPU backend can read the packed weight in place. Bind the
                # caller's storage to the tensor instead of letting
                # ``ggml_backend_alloc_ctx_tensors`` allocate and then copy the
                # whole weight on every forward.
                weight_nbytes = qweight_cpu.numel() * qweight_cpu.element_size()
                weight_ptr = ctypes.c_void_p(qweight_cpu.data_ptr())
                weight_buffer = self._cpu_buffer_from_ptr(
                    weight_ptr,
                    weight_nbytes,
                )
                if not weight_buffer:
                    raise RuntimeError(
                        f"{kernel_name} failed to wrap the packed weight storage."
                    )
                status = self._cpu_tensor_alloc(
                    weight_buffer,
                    weight_tensor,
                    weight_ptr,
                )
                if status != 0:
                    raise RuntimeError(
                        f"{kernel_name} failed to bind the packed weight storage "
                        f"(status={status})."
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

            if not zero_copy_weight:
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
            if weight_buffer:
                self._ggml_base.ggml_backend_buffer_free(weight_buffer)
            self._ggml_base.ggml_free(ctx)

    def build_cuda_weight(
        self,
        *,
        device: torch.device,
        qweight: torch.Tensor,
        gguf_tensor_qtype: str,
        padded_in_features: int,
        out_features: int,
        kernel_name: str,
    ) -> _GGMLSharedWeight:
        """Upload the packed weight once per device for all batch-size plans."""

        device_index = 0 if device.index is None else device.index
        previous_device = torch.cuda.current_device() if torch.cuda.is_available() else None
        try:
            backend = self._get_cuda_backend(device_index)
            qweight_device = self._normalize_qweight_cuda(qweight, device=device)
            ctx = self._ggml_base.ggml_init(
                _GGMLInitParams(
                    mem_size=self.GGML_METADATA_BYTES,
                    mem_buffer=None,
                    no_alloc=True,
                )
            )
            if not ctx:
                raise RuntimeError(f"{kernel_name} failed to initialize GGML weight context.")

            buffer = None
            try:
                weight_tensor = self._ggml_base.ggml_new_tensor_2d(
                    ctx,
                    self.ggml_qtypes[gguf_tensor_qtype],
                    padded_in_features,
                    out_features,
                )
                if not weight_tensor:
                    raise RuntimeError(f"{kernel_name} failed to create the GGML weight tensor.")
                buffer = self._ggml_base.ggml_backend_alloc_ctx_tensors(ctx, backend)
                if not buffer:
                    raise RuntimeError(f"{kernel_name} failed to allocate the GGML weight buffer.")
                weight_nbytes = qweight_device.numel() * qweight_device.element_size()
                self._ggml_base.ggml_backend_tensor_set(
                    weight_tensor,
                    ctypes.c_void_p(qweight_device.data_ptr()),
                    0,
                    weight_nbytes,
                )
                return _GGMLSharedWeight(
                    ctx=ctx,
                    buffer=buffer,
                    tensor=weight_tensor,
                    nbytes=weight_nbytes,
                    backend_buffer_free=self._ggml_base.ggml_backend_buffer_free,
                    ctx_free=self._ggml_base.ggml_free,
                    data_ptr=int(self._ggml_base.ggml_get_data(weight_tensor) or 0),
                    shape=tuple(qweight_device.shape),
                    device=qweight_device.device,
                    lock=self.cuda_backend_lock(device_index),
                )
            except Exception:
                if buffer:
                    self._ggml_base.ggml_backend_buffer_free(buffer)
                self._ggml_base.ggml_free(ctx)
                raise
        finally:
            # ggml's CUDA backend calls `cudaSetDevice`; see `quantized_matmul_cuda`.
            if previous_device is not None and torch.cuda.current_device() != previous_device:
                torch.cuda.set_device(previous_device)

    def cuda_scratch_generation(self, device_index: int) -> int:
        """Generation of the device scratch arena (0 when not allocated yet)."""

        scratch = self._cuda_scratch.get(device_index)
        return 0 if scratch is None else scratch.generation

    def _cuda_scratch_for(
        self, device_index: int, backend: int, nbytes: int
    ) -> _GGMLCudaScratch:
        scratch = self._cuda_scratch.get(device_index)
        if scratch is None or scratch.backend != backend:
            if scratch is not None:
                scratch.close()
            scratch = _GGMLCudaScratch(
                backend=backend,
                alloc_buffer=self._ggml_base.ggml_backend_alloc_buffer,
                buffer_free=self._ggml_base.ggml_backend_buffer_free,
                buffer_get_base=self._ggml_base.ggml_backend_buffer_get_base,
                buffer_get_size=self._ggml_base.ggml_backend_buffer_get_size,
            )
            self._cuda_scratch[device_index] = scratch
        return scratch.ensure(nbytes)

    def _bind_cuda_plan(self, plan: _GGMLMatmulPlan) -> _GGMLCudaScratch:
        """Point a plan's scratch tensors at the shared device arena."""

        scratch = self._cuda_scratch_for(
            plan.device_index, plan.backend, plan.scratch_nbytes
        )
        if plan.scratch is scratch and plan.scratch_generation == scratch.generation:
            return scratch
        if plan.scratch is not None:
            # ggml_backend_tensor_alloc only accepts unbound tensors.  Growth
            # frees the old arena, so the caller must rebuild stale plans.
            raise RuntimeError("stale ggml CUDA plan: rebuild it after scratch growth")
        base = int(scratch.base)
        for tensor, offset in zip(plan.tensors, plan.tensor_offsets):
            status = self._ggml_base.ggml_backend_tensor_alloc(
                scratch.buffer, tensor, base + offset
            )
            if status != 0:
                raise RuntimeError(
                    f"ggml backend tensor allocation failed with status={status}."
                )
        plan.scratch = scratch
        plan.scratch_generation = scratch.generation
        return scratch

    def build_quantized_matmul_cuda_plan(
        self,
        *,
        backend: int,
        weight: _GGMLSharedWeight,
        padded_in_features: int,
        out_features: int,
        rows: int,
        input_ggml_type: int,
        kernel_name: str,
        device_index: int = 0,
    ) -> _GGMLMatmulPlan:
        """Build the batch-size-specific graph around a shared weight tensor.

        The plan owns no backend buffer: its input/output/intermediate tensors
        are bound into the single per-device scratch arena, so a layer with
        many batch sizes no longer retains one ggml buffer per plan.
        """

        ctx = self._ggml_base.ggml_init(
            _GGMLInitParams(
                mem_size=self.GGML_METADATA_BYTES,
                mem_buffer=None,
                no_alloc=True,
            )
        )
        if not ctx:
            raise RuntimeError(f"{kernel_name} failed to initialize GGML metadata context.")

        try:
            weight_tensor = weight.tensor
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

            # Compute the same per-tensor layout `ggml_backend_alloc_ctx_tensors`
            # would use, but leave the tensors unbound for the shared arena.
            buft = self._ggml_base.ggml_backend_get_default_buffer_type(backend)
            alignment = max(int(self._ggml_base.ggml_backend_buft_get_alignment(buft)), 1)
            tensors = []
            tensor_offsets = []
            offset = 0
            tensor = self._ggml_base.ggml_get_first_tensor(ctx)
            while tensor:
                tensors.append(tensor)
                tensor_offsets.append(offset)
                alloc_size = int(
                    self._ggml_base.ggml_backend_buft_get_alloc_size(buft, tensor)
                )
                offset += ((alloc_size + alignment - 1) // alignment) * alignment
                tensor = self._ggml_base.ggml_get_next_tensor(ctx, tensor)
            if offset <= 0:
                raise RuntimeError(f"{kernel_name} GGML plan has no scratch tensors.")

            output_nbytes = self._ggml_base.ggml_nbytes(output_tensor)
            output_dtype = self._torch_dtype_from_ggml_element_size(
                int(self._ggml_base.ggml_element_size(output_tensor)),
                kernel_name=kernel_name,
            )
            plan = _GGMLMatmulPlan(
                ctx=ctx,
                weight=weight,
                input_tensor=input_tensor,
                output_tensor=output_tensor,
                graph=graph,
                rows=rows,
                out_features=out_features,
                device_index=device_index,
                backend=backend,
                tensors=tensors,
                tensor_offsets=tensor_offsets,
                scratch_nbytes=offset,
                output_nbytes=output_nbytes,
                output_dtype=output_dtype,
                ctx_free=self._ggml_base.ggml_free,
            )
            self._bind_cuda_plan(plan)
            return plan
        except Exception:
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
        self._bind_cuda_plan(plan)
        # ggml runs the graph on its own CUDA stream. Any work the caller
        # enqueued on its current stream (for example under
        # `torch.cuda.stream(s)`) must be visible before ggml copies the input
        # from `x.data_ptr()`, otherwise the copy reads stale memory and the
        # result is wrong. A synchronize is the portable ordering primitive
        # here: ggml's stream is not exposed.
        torch.cuda.current_stream(x.device).synchronize()
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

    # CPU-side set, kept as a class attribute because callers read
    # `_GGMLBridge._GGML_MATMUL_UNSUPPORTED`.
    _GGML_MATMUL_UNSUPPORTED = _GGML_CPU_MATMUL_UNSUPPORTED

    def quantized_matmul_cpu(
        self,
        *,
        qweight: torch.Tensor,
        x: torch.Tensor,
        gguf_tensor_qtype: str,
        padded_in_features: int,
        out_features: int,
        threads: Optional[int] = None,
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
            backend = self._get_cpu_backend()
            self._ggml_cpu.ggml_backend_cpu_set_n_threads(
                backend, self._cpu_threads() if threads is None else threads,
            )
            qweight_cpu = self._normalize_qweight_cpu(qweight)
            x_cpu = self._normalize_input_cpu(x, padded_in_features)
            # ``ggml_backend_cpu_buffer_from_ptr`` asserts 32-byte alignment and
            # the backend may still refuse a particular layout. Keep the copy
            # path as the fallback so an unaligned or rejected weight stays
            # correct instead of aborting.
            if self._can_wrap_cpu_weight(qweight_cpu):
                try:
                    return self._run_quantized_matmul(
                        backend=backend,
                        qweight_cpu=qweight_cpu,
                        x_cpu=x_cpu,
                        gguf_tensor_qtype=gguf_tensor_qtype,
                        padded_in_features=padded_in_features,
                        out_features=out_features,
                        kernel_name="GGUFCppKernel",
                        zero_copy_weight=True,
                    )
                except Exception as exc:  # pragma: no cover - backend-specific
                    log.debug(
                        "GGUFCppKernel: zero-copy packed weight rejected (%s); "
                        "falling back to a copied weight buffer.",
                        exc,
                    )
            return self._run_quantized_matmul(
                backend=backend,
                qweight_cpu=qweight_cpu,
                x_cpu=x_cpu,
                gguf_tensor_qtype=gguf_tensor_qtype,
                padded_in_features=padded_in_features,
                out_features=out_features,
                kernel_name="GGUFCppKernel",
            )

    def quantized_matmul_cuda(
        self,
        *,
        x: torch.Tensor,
        gguf_tensor_qtype: str,
        padded_in_features: int,
        out_features: int,
        plan: _GGMLMatmulPlan | None = None,
        weight: _GGMLSharedWeight | None = None,
    ) -> tuple[torch.Tensor, _GGMLMatmulPlan]:
        if x.device.type != "cuda":
            raise RuntimeError("GGUFCudaKernel only supports CUDA input tensors.")
        if gguf_tensor_qtype in _GGML_CUDA_MATMUL_UNSUPPORTED:
            # ggml-cuda aborts the process on these (no convert/vec_dot kernel),
            # so refuse them instead of building a graph.
            raise NotImplementedError(
                f"ggml CUDA matmul does not support `{gguf_tensor_qtype}` weights; "
                "use BACKEND.GGUF_TORCH instead."
            )

        device = x.device
        previous_device = torch.cuda.current_device()
        try:
            backend = self._get_cuda_backend(0 if device.index is None else device.index)
            x_cuda, input_ggml_type = self._normalize_input_cuda(x, padded_in_features)
            if plan is None:
                if weight is None:
                    raise RuntimeError(
                        "GGML CUDA matmul needs the shared weight tensor to build a plan."
                    )
                plan = self.build_quantized_matmul_cuda_plan(
                    backend=backend,
                    weight=weight,
                    padded_in_features=padded_in_features,
                    out_features=out_features,
                    rows=x_cuda.shape[0],
                    input_ggml_type=input_ggml_type,
                    kernel_name="GGUFCudaKernel",
                    device_index=0 if device.index is None else device.index,
                )
            output = self.run_quantized_matmul_cuda_plan(
                backend=backend,
                plan=plan,
                x=x_cuda,
                kernel_name="GGUFCudaKernel",
            )
            return output, plan
        finally:
            # ggml's CUDA backend calls `cudaSetDevice`; without restoring it
            # a forward on cuda:1 leaves the process's current device on 1, so
            # later `torch.empty(device="cuda")` allocations land on the wrong
            # GPU.
            if torch.cuda.current_device() != previous_device:
                torch.cuda.set_device(previous_device)


_GGML_BRIDGE: Optional[_GGMLBridge] = None
_GGML_BRIDGE_LOCK = threading.Lock()


def _get_ggml_bridge() -> _GGMLBridge:
    global _GGML_BRIDGE
    with _GGML_BRIDGE_LOCK:
        if _GGML_BRIDGE is None:
            _GGML_BRIDGE = _GGMLBridge()
        return _GGML_BRIDGE


# The native ggml matmul quantizes activations to Q8_0/Q8_1 (legacy layouts) or
# Q8_K (K-quant layouts) before the integer dot product; ``ggml_mul_mat`` has no
# full-precision activation path, so the loss is a property of the kernel rather
# than of the packed weights or of the Python glue. Measured against an FP64
# matmul of exactly dequantized weights, over every supported layout, M in
# {1, 8, 32} and FP32/FP16/BF16 activations:
#
# * Gaussian activations, CPU: 4.9e-3 .. 7.9e-3. The native result reproduces
#   an independent Q8 round trip of the activations to ~1e-7, so the whole CPU
#   error is activation quantization.
# * Gaussian activations, CUDA: 4.9e-3 .. 1.5e-2 for FP32/BF16 activations
#   (ggml-cuda adds low-precision accumulation), and 5.9e-4 .. 9.3e-4 for FP16
#   activations, which take a different kernel.
# * Outlier channels, CPU: 3.3e-2 .. 3.8e-2 relative, 7.6e-2 .. 8.4e-2 worst
#   row for 16 of 4096 channels scaled by 100 (#4316). Q8_K stores one scale
#   per 256-element super-block, so one large channel widens the step for every
#   element of its block. There is no activation-independent bound.
#
# ``GGML_NATIVE_MATMUL_ERROR_BOUND`` is the Gaussian envelope the native path is
# validated against, not a bound for every input. ``GGUF_TORCH`` is the exact
# path, and ``GGUFCppKernel(exact_matmul=True)`` forces it for a CPU layer. See
# ``docs/gguf-cpp-activation-quantization.md`` and
# ``tests/kernels/test_gguf_cpp_activation_quantization.py``.
GGML_NATIVE_MATMUL_ERROR_BOUND = 2e-2
# Worst-row envelope measured on the #4316 outlier repro (8.4e-2). Recorded so
# the regression test can pin the observed magnitude; it is not a universal
# bound either.
GGML_NATIVE_MATMUL_OUTLIER_ERROR_ENVELOPE = 1e-1


def _ggml_native_output_needs_exact_epilogue(
    *,
    device_type: str,
    bias_present: bool,
    native_dtype: torch.dtype,
) -> bool:
    """Whether the native ggml product must be recomputed in FP32 for the bias.

    The rule is a pure function of static configuration: the backend, whether
    the layer owns a bias buffer, and the dtype ggml chose for the product. It
    never inspects activation values, never times the two paths, and never
    depends on call order, so a given checkpoint takes the same path in every
    process (#4219).

    Only a biased layer needs it: with ``bias is None`` the native product *is*
    the answer, and its Q8 activation loss stays within
    ``GGML_NATIVE_MATMUL_ERROR_BOUND`` for Gaussian activations (outlier
    channels can exceed it; see the module constant). A biased layer cannot add
    the bias to a product that has already been rounded (FP16/BF16 output) or
    that overflowed inside ggml's low-precision accumulation -- ggml-cuda's FP32
    output tensor
    can still hold ``inf``/``nan`` for a finite result (measured: Q4_0 with FP32
    activations of magnitude 1e4 returns NaN), so an FP32 bias would not cancel
    it. Those cases recompute exactly instead (#3789).
    """

    if not bias_present:
        return False
    return device_type == "cuda" or native_dtype in (torch.float16, torch.bfloat16)


def _finish_ggml_output(
    module: GGUFTorchLinear, x_flat: torch.Tensor, output: torch.Tensor
) -> torch.Tensor:
    """Add the stored bias before rounding a ggml result to the input dtype.

    The native ggml kernel quantizes activations to Q8_0/Q8_1/Q8_K before the
    dot product, so its result differs from an FP32 dequantized-weight oracle.
    On CPU it stays within 1e-2 for Gaussian activations (measured 4.9e-3 ..
    7.9e-3); the documented ``GGML_NATIVE_MATMUL_ERROR_BOUND`` is 2e-2 because
    the CUDA FP32/BF16 envelope reaches 1.5e-2 from ggml-cuda's low-precision
    accumulation. Outlier channels can exceed 2e-2 on both backends because
    Q8_K shares one scale per 256-element block (#4316). That loss is inherent
    to ``GGUF_CPP``; ``GGUF_TORCH`` and ``GGUFCppKernel(exact_matmul=True)`` are
    the exact paths. Biased layers take the exact FP32 epilogue described by
    ``_ggml_native_output_needs_exact_epilogue``.
    """

    bias = module.bias
    if _ggml_native_output_needs_exact_epilogue(
        device_type=x_flat.device.type,
        bias_present=bias is not None,
        native_dtype=output.dtype,
    ):
        # A low-precision ggml output has already rounded the dot product.
        # GGML CUDA can also return an FP32 tensor whose product has already
        # overflowed or lost precision, including inf for a finite result.
        # Recompute these biased cases before rounding to the input dtype.
        with torch.autocast(device_type=x_flat.device.type, enabled=False):
            weight = module.dequantize_weight(device=x_flat.device, dtype=torch.float32)
            output = x_flat.float() @ weight
            output = output + bias.to(device=x_flat.device, dtype=torch.float32)
            del weight
        return output.to(x_flat.dtype)

    if output.device != x_flat.device:
        output = output.to(device=x_flat.device)
    if bias is not None:
        output = output + bias.to(device=output.device, dtype=output.dtype)
    return output.to(x_flat.dtype)


class GGUFCppKernel(GGUFTorchLinear):
    # `llama-cpp-python` is not installed on this host, so the ggml CPU bridge
    # cannot build; the CPU-only forward also rejects CUDA input. Static: EAGER.
    SUPPORTS_COMPILE_LEVEL = COMPILE_LEVEL.EAGER
    SUPPORTS_GRAPH_CAPTURE = False

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
        exact_matmul: Optional[bool] = None,
        **kwargs,
    ):
        """CPU GGUF kernel backed by ggml.

        ``exact_matmul`` selects the activation path per layer: ``True`` uses
        the exact torch dequantize-then-matmul path (``GGUF_TORCH`` numerics),
        ``False`` requests the lossy native ggml matmul, and ``None`` (default)
        keeps the capability-based native choice controlled by
        ``GGUFCPPConfig.native_matmul``. The constructor argument wins.
        """
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
        self.exact_matmul = None if exact_matmul is None else bool(exact_matmul)
        if self.kernel.gguf_cpp_threads is None:
            self.kernel.gguf_cpp_threads = torch.get_num_threads()
        self._gguf_cpu_autotune_cache: Dict[tuple, bool] = {}

    def post_init(self):
        super().post_init()
        self._align_packed_weight()

    def _align_packed_weight(self) -> None:
        """Give ``qweight`` the 32-byte alignment the zero-copy CPU path needs.

        Checkpoints loaded from safetensors are mmap views whose storage offset
        is arbitrary, so the packed buffer can start on any byte. Copy it once
        into a padded allocation during initialization; the forward path then
        reads it in place instead of copying it every call.

        The mmap view's pages would otherwise stay resident next to the copy
        (+522 MiB ``RssFile`` on Llama-3.2-1B q4_k_m), so they are reclaimed
        once the copy exists. A mapping cannot be moved onto a 32-byte
        boundary instead: ``mmap`` keeps a page offset, and the safetensors
        header fixes the data offset modulo 32.
        """

        qweight = getattr(self, "qweight", None)
        if not isinstance(qweight, torch.Tensor):
            return
        if qweight.device.type != "cpu" or qweight.dtype != torch.uint8:
            return
        if qweight.numel() == 0 or not qweight.is_contiguous():
            return
        if qweight.data_ptr() % _GGML_CPU_WEIGHT_ALIGNMENT == 0:
            return

        nbytes = qweight.numel() * qweight.element_size()
        storage = torch.empty(
            nbytes + _GGML_CPU_WEIGHT_ALIGNMENT - 1,
            dtype=torch.uint8,
        )
        offset = (-storage.data_ptr()) % _GGML_CPU_WEIGHT_ALIGNMENT
        aligned = storage.narrow(0, offset, nbytes).view(qweight.shape)
        aligned.copy_(qweight)
        # ``aligned`` keeps ``storage`` alive, and the extra <32 bytes are not
        # part of the buffer shape or the checkpoint payload.
        self.qweight = aligned
        _reclaim_file_pages(qweight)

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
            threads=self.kernel.gguf_cpp_threads,
        )

    def _forward_torch_cpu(
        self, x_flat: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self._should_use_fused_k_forward(x_flat):
            return self._forward_fused_k(x_flat, bias=bias)
        return self._forward_dequant_matmul(x_flat, bias=bias)

    def _ggml_cpu_matmul_key(self, x_flat: torch.Tensor) -> tuple:
        return (x_flat.shape[0], x_flat.dtype, self.gguf_tensor_qtype)

    def _use_ggml_cpu_matmul(self, x_flat: torch.Tensor) -> bool:
        """Whether this forward should use the native ggml CPU matmul.

        The default choice is capability-based and therefore stable for a given
        config: the native kernel is used whenever the CPU backend has a vec_dot
        for the weight layout. It used to be decided by timing the two paths,
        which made the numerics (ggml quantizes activations, the torch path
        does not) depend on the machine and on the run (#4219).

        An explicit ``exact_matmul`` constructor argument selects the path for
        one layer (``True`` = exact torch path, ``False`` = native). Without it,
        ``KernelConfig.gguf_cpp_native_matmul`` selects the path; ``None`` keeps
        the capability default. A layout that fails at compute time is recorded
        as unusable so the following calls take the exact torch path.
        """

        if self.gguf_tensor_qtype in _GGMLBridge._GGML_MATMUL_UNSUPPORTED:
            return False
        if self.exact_matmul is not None:
            return not self.exact_matmul
        override = self.kernel.gguf_cpp_native_matmul
        if override is not None:
            return override
        return self._gguf_cpu_autotune_cache.get(self._ggml_cpu_matmul_key(x_flat), True)

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
            try:
                output = _finish_ggml_output(self, x_flat, self._forward_ggml_cpu(x_flat))
            except Exception:
                # A layout the backend cannot actually run must not make the
                # path depend on timing or on call order: remember it and take
                # the exact torch path from here on.
                self._gguf_cpu_autotune_cache[self._ggml_cpu_matmul_key(x_flat)] = False
                output = self._forward_torch_cpu(x_flat, bias=self.bias).to(x_flat.dtype)
        else:
            output = self._forward_torch_cpu(x_flat, bias=self.bias).to(x_flat.dtype)

        if self.adapter:
            output = self.adapter.apply(x=x_flat, out=output)

        return output.reshape(original_shape)

class GGUFCudaKernel(GGUFTorchLinear):
    # `llama-cpp-python` is not installed on this host, so the ggml CUDA bridge
    # cannot build. The forward also takes a device lock and mutates the plan
    # cache on the host. Static: EAGER, no capture.
    SUPPORTS_COMPILE_LEVEL = COMPILE_LEVEL.EAGER
    SUPPORTS_GRAPH_CAPTURE = False

    SUPPORTS_BACKENDS = [BACKEND.GGUF_CPP_CUDA]
    SUPPORTS_METHODS = [METHOD.GGUF]
    SUPPORTS_FORMATS = {FORMAT.GGUF: 35}
    # `bits` alone cannot express this: 1-bit Q1_0/IQ1_S/IQ1_M and 2-bit
    # Q2_K/IQ2_* run through ggml CUDA, but TQ1_0 (1-bit) and TQ2_0 (2-bit)
    # have no GPU convert kernel and abort the process. `__init__` rejects them
    # with NotImplementedError so AUTO can fall back to GGUF_TORCH, and
    # `quantized_matmul_cuda` re-checks as defence in depth.
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
        self._ggml_cuda_plans: OrderedDict[
            tuple[int, int, torch.dtype, tuple], _GGMLMatmulPlan
        ] = OrderedDict()
        # One device copy of the packed weight, shared by every plan. Plans hold
        # pointers into its buffer, so it may only be replaced after they are
        # closed. Once uploaded from a same-device ``qweight`` it is also the
        # storage of ``qweight`` (see ``_cuda_weight``).
        self._ggml_cuda_weight: _GGMLSharedWeight | None = None
        self._ggml_cuda_weight_key: tuple | None = None
        # Generation of the shared per-device scratch arena the cached plans
        # were bound to. Growth invalidates every previously bound plan.
        self._ggml_cuda_plan_generation = -1
        # Reject at construction: `create_quant_layer` retries the next AUTO
        # candidate only on a NotImplementedError raised here. A guard in
        # `forward()` is too late to fall back and would surface as a runtime
        # failure on an already-built layer.
        if self.gguf_tensor_qtype in _GGML_CUDA_MATMUL_UNSUPPORTED:
            raise NotImplementedError(
                f"ggml CUDA matmul does not support `{self.gguf_tensor_qtype}` weights; "
                "use BACKEND.GGUF_TORCH instead."
            )

    @classmethod
    def validate_once(cls) -> Tuple[bool, Optional[Exception]]:
        try:
            return _get_ggml_bridge().cuda_available()
        except Exception as exc:
            return False, exc

    def _close_cuda_plans(self) -> None:
        for plan in self._ggml_cuda_plans.values():
            plan.close()
        self._ggml_cuda_plans.clear()
        self._ggml_cuda_plan_generation = -1

    def _release_cuda_weight(self) -> None:
        # Drop the reference instead of closing: a Torch tensor that still
        # shares the buffer (``qweight`` itself, or a ``state_dict()`` entry)
        # keeps it alive, and the last reference frees it.
        self._ggml_cuda_weight = None
        self._ggml_cuda_weight_key = None

    @staticmethod
    def _cuda_lock_for_device(device_index: int) -> threading.RLock:
        return _get_ggml_bridge().cuda_backend_lock(device_index)

    def _cuda_weight_lock(self) -> threading.RLock:
        key = self._ggml_cuda_weight_key
        # `_cuda_weight` sets the key and the weight together before a plan is
        # ever cached, so a missing key means there is nothing to free.
        return self._cuda_lock_for_device(0 if key is None else key[0])

    def clear_weight_cache(self) -> None:
        if self._ggml_cuda_weight_key is not None:
            with self._cuda_weight_lock():
                self._close_cuda_plans()
                weight = self._ggml_cuda_weight
                # A weight that backs ``qweight`` is its storage, not a cache:
                # in-place writes (``load_state_dict``) already landed in it.
                if weight is None or not weight.backs(self.qweight):
                    self._release_cuda_weight()
        return super().clear_weight_cache()

    def _cuda_weight(self, x_flat: torch.Tensor) -> _GGMLSharedWeight:
        device_index = 0 if x_flat.device.index is None else x_flat.device.index
        # Closing plans and freeing the shared weight is only safe while no
        # other thread is computing with it; every forward takes the same
        # process-wide (re-entrant) lock, because ggml-cuda state is shared
        # across devices (#4378).
        with self._cuda_lock_for_device(device_index):
            qweight = self.qweight
            weight = self._ggml_cuda_weight
            if weight is not None and weight.device == x_flat.device:
                if weight.backs(qweight):
                    return weight
                if self._ggml_cuda_weight_key == (device_index, tensor_cache_key(qweight)):
                    return weight
            # A new weight (or device) invalidates every plan built against the
            # old buffer, so drop them before releasing it.
            self._close_cuda_plans()
            self._release_cuda_weight()
            weight = _get_ggml_bridge().build_cuda_weight(
                device=x_flat.device,
                qweight=qweight,
                gguf_tensor_qtype=self.gguf_tensor_qtype,
                padded_in_features=self.padded_in_features,
                out_features=self.out_features,
                kernel_name=self.__class__.__name__,
            )
            if qweight.device == x_flat.device:
                # The Torch ``qweight`` on this device would duplicate the
                # upload (#4387). Make the ggml buffer its storage: state_dict,
                # save, ``.to()`` and the exact FP32 epilogue keep reading the
                # same bytes through the zero-copy view, and the Torch
                # allocation is released.
                qweight = weight.as_tensor()
                self.qweight = qweight
            self._ggml_cuda_weight = weight
            self._ggml_cuda_weight_key = (device_index, tensor_cache_key(qweight))
            return weight

    def __getstate__(self):
        """Leave the ggml weight and plans out of a copy or pickle.

        They are raw process-local handles owned by this instance; a copy that
        shared them would free them twice. The copied ``qweight`` holds the
        bytes, and the copy uploads its own weight on its first forward.
        """

        state = super().__getstate__()
        state["_ggml_cuda_plans"] = OrderedDict()
        state["_ggml_cuda_weight"] = None
        state["_ggml_cuda_weight_key"] = None
        state["_ggml_cuda_plan_generation"] = -1
        return state

    def __del__(self) -> None:  # pragma: no cover - best effort cleanup
        try:
            if self._ggml_cuda_weight_key is not None:
                with self._cuda_weight_lock():
                    self._close_cuda_plans()
                    self._release_cuda_weight()
        except Exception:
            # Destructors must not raise during GC or interpreter shutdown.
            pass

    def _cuda_plan_key(
        self,
        x_flat: torch.Tensor,
    ) -> tuple[int, int, torch.dtype, tuple]:
        device_index = 0 if x_flat.device.index is None else x_flat.device.index
        return (
            device_index,
            x_flat.shape[0],
            x_flat.dtype,
            tensor_cache_key(self.qweight),
        )

    def _remember_cuda_plan(
        self,
        plan_key: tuple[int, int, torch.dtype, tuple],
        plan: _GGMLMatmulPlan,
        *,
        cached: bool,
    ) -> None:
        if cached:
            self._ggml_cuda_plans.move_to_end(plan_key)
            return
        self._ggml_cuda_plans[plan_key] = plan
        while len(self._ggml_cuda_plans) > _MAX_GGML_CUDA_PLANS:
            _, evicted = self._ggml_cuda_plans.popitem(last=False)
            evicted.close()

    def forward(self, x: torch.Tensor):
        original_shape = x.shape[:-1] + (self.out_features,)
        x_flat = x.reshape(-1, x.shape[-1])
        if x_flat.device.type != "cuda":
            raise RuntimeError(
                f"{self.__class__.__name__} only supports CUDA inference. "
                "Load GGUF models on CUDA or use BACKEND.GGUF_CPP_CPU or BACKEND.GGUF_TORCH for CPU inference."
            )

        device_index = 0 if x_flat.device.index is None else x_flat.device.index
        # ggml-cuda keeps cuBLAS handles and memory pools in process-global
        # state shared by every device, so weight/plan cache mutation and the
        # backend call must not interleave with any other forward, not just one
        # on the same device (#4378). The lock is re-entrant: `_cuda_weight`
        # takes it again for the swap itself.
        with self._cuda_lock_for_device(device_index):
            bridge = _get_ggml_bridge()
            # Resolve the shared weight first: it may drop plans built against
            # a stale weight buffer and replace ``qweight`` with its view, so
            # the plan key and lookup below must happen after it.
            weight = self._cuda_weight(x_flat)
            plan_key = self._cuda_plan_key(x_flat)
            generation = bridge.cuda_scratch_generation(device_index)
            if self._ggml_cuda_plan_generation != generation:
                # The device arena grew since these plans were bound; their
                # tensors point into the freed buffer.
                self._close_cuda_plans()
                self._ggml_cuda_plan_generation = generation
            cached_plan = self._ggml_cuda_plans.get(plan_key)
            output, plan = bridge.quantized_matmul_cuda(
                x=x_flat,
                gguf_tensor_qtype=self.gguf_tensor_qtype,
                padded_in_features=self.padded_in_features,
                out_features=self.out_features,
                plan=cached_plan,
                weight=weight,
            )
            current_generation = bridge.cuda_scratch_generation(device_index)
            if current_generation != generation:
                # Building this plan grew the arena and invalidated every plan
                # bound to the old one. Keep only the plan just built.
                self._close_cuda_plans()
            self._remember_cuda_plan(
                plan_key,
                plan,
                cached=cached_plan is not None,
            )
            self._ggml_cuda_plan_generation = current_generation
        output = _finish_ggml_output(self, x_flat, output)

        if self.adapter:
            output = self.adapter.apply(x=x_flat, out=output)

        return output.reshape(original_shape)


__all__ = ["GGUFCppKernel", "GGUFCudaKernel"]
