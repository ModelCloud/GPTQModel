# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import os

import torch

from ...adapter.adapter import Adapter, Lora
from ...models._const import DEVICE, PLATFORM
from ...quantization import FORMAT, METHOD
from ...utils.amplin import (
    _build_kernel_family,
    _call_kernel,
    _call_kernel_chunked,
    _select_family_member,
    amplin_runtime_available,
    amplin_runtime_error,
    dynamic,
    select_kernel,
)
from ...utils.backend import BACKEND
from ...utils.logger import setup_logger
from . import PackableQuantLinear


log = setup_logger()


class AmplinLinear(PackableQuantLinear):
    """GPTQ W4A16 linear layer routed through Amplin dynamic micro-kernels.

    Amplin dynamically selects the fastest Ampere/SM80 kernel for each
    ``(M, K, N, dtype)``.  `post_init` pre-builds a packed ``_KernelDispatch``
    for the batch size given by ``KERNEL_BATCH_HINT`` (default 1) so matching
    decode ``forward`` calls are a direct kernel launch.  Runtime batches that
    fit the selected kernel's M-constraints reuse the same dispatch; larger or
    in-between batches are handled by ``_call_kernel_chunked``.
    """

    SUPPORTS_BACKENDS = [BACKEND.GPTQ_AMPLIN, BACKEND.AMPLIN]
    SUPPORTS_METHODS = [METHOD.GPTQ]
    SUPPORTS_FORMATS = {FORMAT.GPTQ: 0, FORMAT.GPTQ_V2: 0}
    SUPPORTS_BITS = [4]
    SUPPORTS_GROUP_SIZE = [128]
    SUPPORTS_DESC_ACT = [False]
    SUPPORTS_SYM = [True]
    SUPPORTS_SHARDS = False
    SUPPORTS_TRAINING = False
    SUPPORTS_AUTO_PADDING = False
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [128]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [1]
    SUPPORTS_PACK_DTYPES = [torch.int32]
    SUPPORTS_ADAPTERS = [Lora]
    SUPPORTS_DEVICES = [DEVICE.CUDA]
    SUPPORTS_PLATFORM = [PLATFORM.LINUX]
    SUPPORTS_DTYPES = [torch.float16, torch.bfloat16]

    REQUIRES_FORMAT_V2 = True
    QUANT_TYPE = "amplin"

    @classmethod
    def validate_once(cls):
        if not amplin_runtime_available():
            return False, ImportError(amplin_runtime_error())
        return True, None

    @classmethod
    def validate_device(cls, device: DEVICE):
        super().validate_device(device)
        if not torch.cuda.is_available():
            raise NotImplementedError(f"{cls.__name__} requires a CUDA device.")
        if not any(torch.cuda.get_device_capability(i) == (8, 0) for i in range(torch.cuda.device_count())):
            raise NotImplementedError(
                f"{cls.__name__} currently requires compute capability 8.0 (Ampere)."
            )

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
        register_buffers: bool = True,
        **kwargs,
    ):
        if kwargs.get("backend") is None:
            kwargs["backend"] = BACKEND.GPTQ_AMPLIN
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
            enable_wf_unsqueeze=False,
            **kwargs,
        )

    def post_init(self):
        super().post_init()

        device = self.runtime_device()
        if device is None or device.type != "cuda":
            return

        # Use KERNEL_BATCH_HINT to pick the micro-kernel that will be used at
        # inference time; default to batch 1 for decode.
        batch_hint = int(os.environ.get("KERNEL_BATCH_HINT", "1"))
        dtype = self.scales.dtype
        self._amplin_dispatch = select_kernel(
            self.qweight,
            self.scales,
            size_m=batch_hint,
            dtype=dtype,
            device=device,
            logical_n=self.out_features,
            update_dynamic_table=False,
        )
        self._amplin_batch_hint = batch_hint
        self._amplin_family = _build_kernel_family(self._amplin_dispatch)

        if os.environ.get("AMPLIN_DEBUG_POST_INIT", "0") == "1":
            dispatch = self._amplin_dispatch
            family = self._amplin_family
            log.info(
                "amplin post_init: module=%s batch_hint=%d op=%s packer=%s "
                "in=%d out=%d dtype=%s min_m=%d max_m=%s m_multiple=%d "
                "is_marlin_style=%s family_size=%d layout_id=%s",
                self.name,
                batch_hint,
                dispatch.name,
                dispatch.packer,
                self.in_features,
                self.out_features,
                dtype,
                dispatch.min_m,
                dispatch.max_m,
                dispatch.m_multiple,
                dispatch.is_marlin_style,
                len(family.members),
                family.layout_id,
            )

        # The packed execution weights are owned by the dispatch object.  Once
        # packing is complete, the canonical unpacked qweight/scales can be
        # released to a zero-memory meta tensor while preserving shape/dtype.
        if self._amplin_dispatch.packer != "none":
            torch.cuda.synchronize(device)
            self.qweight = torch.empty_like(self.qweight, device="meta")
            self.scales = torch.empty_like(self.scales, device="meta")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"{self.__class__.__name__} expected input last dim {self.in_features}, got {x.shape[-1]}"
            )

        out_shape = x.shape[:-1] + (self.out_features,)
        x = x.reshape(-1, x.shape[-1])
        x = self._apply_rotation_to_input(x)

        family = getattr(self, "_amplin_family", None)
        if family is not None:
            batch = x.size(0)
            member = _select_family_member(family, batch)
            if member is not None:
                out = _call_kernel(member, x)
            else:
                # Runtime M is not supported by any single family member; chunk
                # with the decode-time dispatch (it shares the same layout).
                out = _call_kernel_chunked(family.default, x)
        else:
            out = dynamic(x, self.qweight, self.scales)

        if self.bias is not None:
            out = out + self.bias.to(device=out.device, dtype=out.dtype)

        if self.adapter:
            out = self.adapter.apply(x=x, out=out)

        return out.reshape(out_shape)
