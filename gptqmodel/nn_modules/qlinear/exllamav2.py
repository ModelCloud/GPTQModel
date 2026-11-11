# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

# Adapted from turboderp exllama: https://github.com/turboderp/exllamav2

from typing import List, Optional, Tuple

import torch

from ...adapter.adapter import Adapter, Lora
from ...models._const import DEVICE, PLATFORM
from ...nn_modules.qlinear import GPTQQuantLinear
from ...quantization import FORMAT, METHOD
from ...utils.backend import BACKEND
from ...utils.exllamav2 import (
    EXLLAMAV2_GPTQ_ROCM_UNSUPPORTED,
    ScratchSpace,
    exllamav2_free_q_matrix,
    exllamav2_gemm_half_q_half,
    exllamav2_gptq_runtime_available,
    exllamav2_gptq_runtime_error,
    exllamav2_make_q_matrix,
)
from ...utils.logger import setup_logger
from ...utils.rocm import IS_ROCM
from ._runtime_cache import tensor_cache_key


log = setup_logger()


def _torch_device(idx):
    if idx == -1:
        return "cpu"
    return f"cuda:{idx}"

class ExllamaV2Linear(GPTQQuantLinear):
    SUPPORTS_BACKENDS = [BACKEND.GPTQ_EXLLAMA_V2]
    SUPPORTS_METHODS = [METHOD.GPTQ]
    SUPPORTS_FORMATS = {FORMAT.GPTQ: 80, FORMAT.GPTQ_V2: 80}
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
    QUANT_TYPE = "exllamav2"

    """Linear layer implementation with per-group 4-bit quantization of the weights"""

    def __init__(
        self,
        bits: int,
        group_size: int,
        desc_act: bool,
        sym: bool,
        in_features: int,
        out_features: int,
        bias: bool = False,
        pack_dtype: torch.dtype = torch.int32,
        adapter: Adapter = None,
        register_buffers: bool = True,
        **kwargs, ):
        # backup original values
        # self.original_out_features = out_features
        # self.original_in_features = in_features
        #
        # # auto pad
        # group_size = group_size if group_size != -1 else in_features
        # out_features = out_features + (-out_features % 32)
        # in_features = in_features + (-in_features % group_size)
        # self.in_features_padding_size = in_features - self.original_in_features
        # self.in_features_padding_shape = (0, self.in_features_padding_size)

        super().__init__(
            bits=bits,
            group_size=group_size,
            sym=sym,
            desc_act=desc_act,
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            pack_dtype=pack_dtype,
            backend=kwargs.pop("backend", BACKEND.GPTQ_EXLLAMA_V2),
            adapter=adapter,
            register_buffers=register_buffers,
            register_buffers_in_features=in_features,
            register_buffers_out_feature=out_features,
            **kwargs)

        self.q_handle = None
        self.q_tensors = None
        self._q_handle_key = None
        self._native_shuffled = False

    @classmethod
    def validate_once(cls) -> Tuple[bool, Optional[Exception]]:
        # The native ExLlamaV2 extension is CUDA-only: hipcc cannot build it and
        # the kernel had accuracy regressions on ROCm. Fail fast with an
        # actionable error instead of attempting the JIT build (~15 s of hipcc
        # errors) and surfacing a build failure.
        if IS_ROCM:
            return False, NotImplementedError(EXLLAMAV2_GPTQ_ROCM_UNSUPPORTED)
        if not exllamav2_gptq_runtime_available():
            return False, ImportError(exllamav2_gptq_runtime_error())
        return True, None

    def post_init(self, scratch_space: ScratchSpace):
        # resize due to padding after model weights have been loaded
        # if self.out_features != self.original_out_features or self.in_features != self.original_in_features:
        #     self.qweight.resize_(self.in_features // self.pack_dtype_bits * self.bits, self.out_features)
        #     self.qzeros.resize_(
        #         math.ceil(self.in_features / self.group_size),
        #         self.out_features // self.pack_dtype_bits * self.bits
        #     )
        #     self.scales.resize_(math.ceil(self.in_features / self.group_size), self.out_features)
        #     self.g_idx = torch.tensor([i // self.group_size for i in range(self.in_features)], dtype=torch.int32, device=self.g_idx.device)
        #     if self.bias is not None:
        #         self.bias.resize_(self.out_features)

        self._release_q_handle()
        # ext_make_q_matrix only accepts float16
        self.scales = self.scales.to(dtype=torch.float16)
        self.q_tensors = None
        self._ensure_q_handle()

        super().post_init()

    def _release_q_handle(self) -> None:
        if self.q_handle is not None:
            exllamav2_free_q_matrix(self.q_handle)
            self.q_handle = None
        self._q_handle_key = None

    def clear_weight_cache(self) -> None:
        """Release native pointers derived from the current packed buffers."""

        self._release_q_handle()
        self.q_tensors = None

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        metadata = getattr(destination, "_metadata", None)
        if metadata is not None:
            metadata.setdefault(prefix[:-1], {})["exllamav2_native_shuffled"] = self._native_shuffled

    def _load_from_state_dict(self, *args, **kwargs):
        state = args[0] if args else kwargs.get("state_dict", {})
        prefix = args[1] if len(args) > 1 else kwargs.get("prefix", "")
        local_metadata = args[2] if len(args) > 2 else kwargs.get("local_metadata", {})
        native_shuffled = bool(local_metadata.get("exllamav2_native_shuffled", False))
        if prefix + "qweight" in state:
            self._release_q_handle()
            if getattr(self, "_fusion_native_view", False):
                # A checkpoint must not be copied into a strided slice of the
                # group's finalized words. Detach every borrowed operand.
                for name in ("qweight", "qzeros", "scales", "bias"):
                    tensor = getattr(self, name, None)
                    if tensor is not None:
                        self.register_buffer(name, tensor.clone(memory_format=torch.contiguous_format))
                self._fusion_native_view = False
            self.q_tensors = None
        super()._load_from_state_dict(*args, **kwargs)
        if prefix + "qweight" in state:
            # Standard checkpoints contain raw words; a Torch state_dict from
            # this finalized module carries its layout in local metadata.
            self._native_shuffled = native_shuffled

    def _ensure_q_handle(self) -> int:
        if getattr(self, "_fusion_native_view", False):
            if self.q_handle is None:
                raise RuntimeError("ExLlamaV2 fusion view lost its native descriptor")
            return self.q_handle
        if self.scales.dtype != torch.float16:
            self.scales = self.scales.to(dtype=torch.float16)
        tensors = {
            "qweight": self.qweight,
            "qzeros": self.qzeros,
            "scales": self.scales,
            "g_idx": self.g_idx,
        }
        if self.q_tensors is None:
            self.q_tensors = tensors
        else:
            self.q_tensors.update(tensors)
        key = tuple(tensor_cache_key(tensor) for tensor in self.q_tensors.values())
        if (
            self.q_handle is not None
            and self._q_handle_key is not None
            and self._q_handle_key != key
        ):
            self._release_q_handle()
        if self.q_handle is None:
            already_shuffled = getattr(self, "_native_shuffled", False)
            if already_shuffled:
                self.q_handle = self.ext_make_q_matrix(self.q_tensors, already_shuffled=True)
            else:
                self.q_handle = self.ext_make_q_matrix(self.q_tensors)
            self._native_shuffled = True
            # Act-order matrix creation adds/fills q_perm and q_invperm in
            # ``q_tensors``. Key the completed dictionary so the next forward
            # does not mistake those newly inserted tensors for a replacement
            # and rebuild the native matrix every call.
            self._q_handle_key = tuple(
                tensor_cache_key(tensor) for tensor in self.q_tensors.values()
            )
        return self.q_handle

    def list_buffers(self) -> List:
        buf = super().list_buffers()
        if hasattr(self, "q_tensors") and self.q_tensors is not None:
            buf.append(self.q_tensors)
        return buf

    def forward(self, x: torch.Tensor, force_cuda=False):
        # TODO FIXME: parent should never call us if there is no data to process
        # check: https://github.com/ModelCloud/GPTQModel/issues/1361
        if self.input_rows(x) == 0:
            return self.empty_linear_output(x)

        x_dtype = x.dtype
        # TODO: need to run checks to make sure there is no performance regression padding with F.pad
        # if in_features is padded, we need to pad the input as well
        # if x.size(-1) != self.in_features:
        #     x = F.pad(x, self.in_features_padding_shape)


        handle, temporaries = self._ensure_q_handle(), None
        if getattr(self, "_fusion_native_view", False):
            from ...utils.exllamav2 import exllamav2_clone_q_matrix

            handle, temporaries = exllamav2_clone_q_matrix(self, compact=True)
        try:
            out = self.ext_gemm_half_q_half(x, handle, self.out_features, force_cuda)
        finally:
            if temporaries is not None:
                from ...utils.exllamav2 import exllamav2_free_q_matrix

                exllamav2_free_q_matrix(handle)

        if self.bias is not None:
            out.add_(self.bias)

        if self.adapter:
            out = self.adapter.apply(x=x, out=out)

        return out.to(dtype=x_dtype)

    def temp_dq_size(self):
        """Bytes of retained dense-weight scratch: zero by design.

        The kernel never keeps a reconstructed FP16 weight. The row-count
        fallback in the extension allocates a transient tensor per call, so no
        dense buffer is attached to the handle or the model.
        """
        return 0

    def temp_fwd_size(self, max_input_len, max_batch_size):
        return self.out_features * max_input_len * max_batch_size * 4 + 128

    def scratch_space_fixed(self, max_input_len=2048, max_batch_size=8):
        return self.temp_dq_size() + self.temp_fwd_size(max_input_len, max_batch_size)

    def ext_gemm_half_q_half(self, x, q_handle, q4_width, force_cuda):
        """Matrix multiplication, returns x @ q4"""
        output_shape = x.shape[:-1] + (q4_width,)
        x = x.reshape(-1, x.shape[-1]).contiguous()
        output = torch.empty((x.shape[0], q4_width), dtype=x.dtype, device=x.device)
        exllamav2_gemm_half_q_half(x, q_handle, output, force_cuda)
        return output.view(output_shape)

    def ext_make_q_matrix(self, w: dict, key: str = None, *, already_shuffled: bool = False):
        """
        Create Q matrix
        """
        # EXL2
        # won't work as the moment because the tensors are not the same.
        if "q_weight" in w:
            if not already_shuffled:
                w["q_scale_max"] /= 256
            w["q_perm"] = w["q_perm"].short()
            w["q_invperm"] = w["q_invperm"].short()
            return exllamav2_make_q_matrix(
                w["q_weight"],
                w["q_perm"],
                w["q_invperm"],
                w["q_scale"],
                w["q_scale_max"],
                w["q_groups"],
                None,
                None,
                None,
                already_shuffled=already_shuffled,
            )
        # GPTQ
        elif "qweight" in w:
            if w["scales"].dtype == torch.float:
                w["scales"] = w["scales"].half()

            # GPTQ with g_idx (act_order)
            if "g_idx" in w and not (w["g_idx"] == 0).all().item():
                if already_shuffled:
                    perm = torch.argsort(w["g_idx"], stable=True).to(device=w["qweight"].device, dtype=torch.short)
                    invperm = torch.argsort(perm, stable=True).to(dtype=torch.short)
                    w["q_perm"], w["q_invperm"] = perm, invperm
                else:
                    w["q_perm"] = torch.empty(
                        (w["qweight"].shape[0] * 8,),
                        dtype=torch.short,
                        device=w["qweight"].device,
                    )
                    w["q_invperm"] = torch.empty_like(w["q_perm"])
                # make_q4 segfaults if g_idx is not on cpu in the act-order case. In the non act-order case, None needs to be passed for g_idx.
                return exllamav2_make_q_matrix(
                    w["qweight"],
                    w["q_perm"],
                    w["q_invperm"],
                    None,
                    None,
                    None,
                    w["qzeros"],
                    w["scales"],
                    None if already_shuffled else w["g_idx"].cpu(),
                    **({"already_shuffled": True} if already_shuffled else {}),
                )
            # GPTQ without g_idx
            else:
                return exllamav2_make_q_matrix(
                    w["qweight"],
                    None,
                    None,
                    None,
                    None,
                    None,
                    w["qzeros"],
                    w["scales"],
                    None,
                    **({"already_shuffled": True} if already_shuffled else {}),
                )
        else:
            raise ValueError("q_weight not found in exllama v2 quantized weights")
