# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import torch

from ...adapter.adapter import Adapter, Lora
from ...models._const import DEVICE, PLATFORM
from ...nn_modules.qlinear import AWQuantLinear
from ...quantization import FORMAT, METHOD
from ...quantization.awq.utils.packing_utils import unpack_reorder_pack
from ...utils.backend import BACKEND
from ...utils.exllamav2 import (
    EXLLAMAV2_AWQ_ROCM_UNSUPPORTED,
    ScratchSpace,
    exllamav2_awq_gemm_half_q_half,
    exllamav2_awq_make_q_matrix,
    exllamav2_awq_runtime_available,
    exllamav2_awq_runtime_error,
    exllamav2_free_q_matrix,
)
from ...utils.logger import setup_logger
from ...utils.rocm import IS_ROCM
from ._runtime_cache import tensor_cache_key


log = setup_logger()


class AwqExllamaV2Linear(AWQuantLinear):
    SUPPORTS_BACKENDS = [BACKEND.AWQ_EXLLAMA_V2]
    SUPPORTS_METHODS = [METHOD.AWQ]
    SUPPORTS_FORMATS = {FORMAT.GEMM: 80}
    SUPPORTS_BITS = [4]
    # TODO: intel is reporting v2 has accuracy issues with group-size == 16 for this kernel
    # disable for now until we can validate this issue: ref https://github.com/ModelCloud/GPTQModel/issues/1515
    SUPPORTS_GROUP_SIZE = [-1, 32, 64, 128]
    SUPPORTS_DESC_ACT = [True, False]
    SUPPORTS_SYM = [True, False]
    SUPPORTS_SHARDS = True
    SUPPORTS_TRAINING = False
    SUPPORTS_AUTO_PADDING = False
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [32]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [32]

    SUPPORTS_DEVICES = [DEVICE.CUDA] # ROCm has broken accuracies issues
    SUPPORTS_PLATFORM = [PLATFORM.LINUX, PLATFORM.WIN32]
    SUPPORTS_PACK_DTYPES = [torch.int32]
    SUPPORTS_ADAPTERS = [Lora]

    SUPPORTS_DTYPES = [torch.float16, torch.bfloat16, torch.float32]

    REQUIRES_FORMAT_V2 = True

    # for transformers/optimum tests compat
    QUANT_TYPE = "awq_exllamav2"

    def __init__(
            self,
            bits: int,
            group_size: int,
            sym: bool,
            desc_act: bool,
            in_features: int,
            out_features: int,
            bias: bool = False,
            pack_dtype: torch.dtype = torch.int32,
            adapter: Adapter = None,
            **kwargs,
    ):
        super().__init__(
            bits=bits,
            group_size=group_size,
            sym=sym,
            desc_act=desc_act,
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            pack_dtype=pack_dtype,
            backend=kwargs.pop("backend", BACKEND.AWQ_EXLLAMA_V2),
            adapter=adapter,
            **kwargs)
        self.q_handle = None
        self._awq_repacked = False
        self._q_handle_key = None

    @classmethod
    def validate_once(cls):
        # The native ExLlamaV2 AWQ extension is CUDA-only: hipcc cannot build it
        # and the kernel had accuracy regressions on ROCm. Fail fast with an
        # actionable error instead of attempting the JIT build.
        if IS_ROCM:
            return False, NotImplementedError(EXLLAMAV2_AWQ_ROCM_UNSUPPORTED)
        # Validate through the shared JIT loader so import-time checks match the
        # runtime build path used on first kernel access.
        if not exllamav2_awq_runtime_available():
            return False, ImportError(exllamav2_awq_runtime_error())
        return True, None

    def ext_make_q_matrix_awq(self, qweight, qzeros, scales) -> int:
        runtime_scales = scales if scales.dtype == torch.float16 else scales.to(dtype=torch.float16)
        return exllamav2_awq_make_q_matrix(
            qweight,
            None,
            None,
            None,
            None,
            None,
            qzeros,
            runtime_scales,
            None,
        )

    def post_init(self, scratch_space: ScratchSpace):
        # if self.padded_infeatures != self.in_features:
        #     self.qweight.resize_(self.padded_infeatures // self.pack_dtype_bits * self.bits, self.out_features)
        #     self.qzeros.resize_(
        #         math.ceil(self.padded_infeatures / self.group_size),
        #         self.out_features // self.pack_dtype_bits * self.bits
        #     )
        #     self.scales.resize_((math.ceil(self.padded_infeatures / self.group_size), self.out_features), )
        #     self.g_idx = torch.tensor([i // self.group_size for i in range(self.padded_infeatures)], dtype=torch.int32,
        #                               device=self.g_idx.device)

        if not exllamav2_awq_runtime_available():
            raise ModuleNotFoundError("ExLlamaV2 AWQ torch.ops kernels are not properly installed. Error: " + exllamav2_awq_runtime_error())

        self._release_q_handle()
        # awq only accepts float16
        self.scales = self.scales.to(dtype=torch.float16)

        assert self.qweight.device.type == "cuda"
        assert self.qweight.device.index is not None

        if not self._awq_repacked:
            self.qweight, self.qzeros = unpack_reorder_pack(
                self.qweight, self.qzeros, self.bits
            )
            self._awq_repacked = True
        self._ensure_q_handle()

        super().post_init()

    def _release_q_handle(self) -> None:
        if self.q_handle is not None:
            exllamav2_free_q_matrix(self.q_handle, awq=True)
            self.q_handle = None
        self._q_handle_key = None

    def clear_weight_cache(self) -> None:
        """Release native pointers derived from the current packed buffers."""

        self._release_q_handle()

    def _ensure_q_handle(self) -> int:
        key = (
            tensor_cache_key(self.qweight),
            tensor_cache_key(self.qzeros),
            tensor_cache_key(self.scales),
        )
        if self.q_handle is not None and self._q_handle_key != key:
            self._release_q_handle()
        if self.q_handle is None:
            self.q_handle = self.ext_make_q_matrix_awq(
                self.qweight,
                self.qzeros,
                self.scales,
            )
            self._q_handle_key = key
        return self.q_handle

    def forward(self, x: torch.Tensor):
        if self.input_rows(x) == 0:
            return self.empty_linear_output(x)
        self._ensure_q_handle()
        if not exllamav2_awq_runtime_available():
            raise ModuleNotFoundError("ExLlamaV2 AWQ torch.ops kernels are not properly installed. Error: " + exllamav2_awq_runtime_error())

        input_dtype = x.dtype
        out_shape = x.shape[:-1] + (self.out_features,)

        x = x.reshape(-1, x.shape[-1]).contiguous()

        out = torch.empty(
            (x.shape[0], self.out_features),
            dtype=x.dtype,
            device=x.device,
        )
        handle, temporaries = self.q_handle, None
        if getattr(self, "_fusion_native_view", False):
            from ...utils.exllamav2 import exllamav2_clone_q_matrix

            handle, temporaries = exllamav2_clone_q_matrix(self, awq=True, compact=True)
        try:
            exllamav2_awq_gemm_half_q_half(x, handle, out, False)
        finally:
            if temporaries is not None:
                from ...utils.exllamav2 import exllamav2_free_q_matrix

                exllamav2_free_q_matrix(handle, awq=True)

        if input_dtype != torch.float16:
            out = out.to(dtype=input_dtype)

        if self.bias is not None:
            out.add_(self.bias)

        if self.adapter:
            out = self.adapter.apply(x=x, out=out)

        return out.view(out_shape)

    def temp_dq_size(self):
        """Bytes of retained dense-weight scratch: zero by design.

        The quantized kernel decodes inside its launch and the row-count
        fallback allocates a transient tensor per call, so no dense weight
        buffer is attached to the handle or the model.
        """
        return 0

    def temp_fwd_size(self, max_input_len, max_batch_size):
        """
        Returns the size of the temporary buffer required for the fwd kernel.
        """
        return self.out_features * max_input_len * max_batch_size * 4 + 128

    def scratch_space_fixed(self, max_input_len=2048, max_batch_size=8):
        """
        Returns the size of the fixed scratch space required for the kernel.
        """
        return self.temp_dq_size() + self.temp_fwd_size(max_input_len, max_batch_size)


def next_multiple(x, multiple):
    return ((x + multiple - 1) // multiple) * multiple


__all__ = ["AwqExllamaV2Linear"]
