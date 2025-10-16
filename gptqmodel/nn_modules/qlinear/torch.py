# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium


import math
import os
from collections.abc import Iterable

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from ...adapter.adapter import Adapter, Lora
from ...models._const import DEVICE, HAS_CUDA, PLATFORM
from ...nn_modules.qlinear import BaseQuantLinear, PackableQuantLinear
from ...utils.backend import BACKEND
from ...utils.logger import setup_logger
from ...utils.torch import torch_compile


try:
    from ..triton_utils.dequant import dequant as triton_dequant

    _TRITON_DEQUANT_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    triton_dequant = None
    _TRITON_DEQUANT_AVAILABLE = False


log = setup_logger()

class TorchQuantLinear(PackableQuantLinear):
    SUPPORTS_BITS = [2, 3, 4, 8]
    SUPPORTS_GROUP_SIZE = [-1, 16, 32, 64, 128, 256, 512, 1024]
    SUPPORTS_DESC_ACT = [True, False]
    SUPPORTS_SYM = [True, False]
    SUPPORTS_SHARDS = True
    SUPPORTS_TRAINING = True
    SUPPORTS_AUTO_PADDING = True
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [1]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [1]

    SUPPORTS_DEVICES = [DEVICE.ALL]
    SUPPORTS_PLATFORM = [PLATFORM.ALL]
    SUPPORTS_PACK_DTYPES = [torch.int8, torch.int16, torch.int32]
    SUPPORTS_ADAPTERS = [Lora]

    SUPPORTS_DTYPES = [torch.float16, torch.bfloat16]

    REQUIRES_FORMAT_V2 = True

    # for transformers/optimum tests compat
    QUANT_TYPE = "torch"

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
        super().__init__(
            bits=bits,
            group_size=group_size,
            sym=sym,
            desc_act=desc_act,
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            pack_dtype=pack_dtype,
            backend=kwargs.pop("backend", BACKEND.TORCH),
            adapter=adapter,
            register_buffers=register_buffers,
            **kwargs)

        self.dequant_dtype = torch.int16 if self.bits == 8 else torch.int8
        self._streaming_enabled = bool(int(os.environ.get("GPTQ_TORCH_STREAMING", "0")))
        self._stream_tile_cols = int(os.environ.get("GPTQ_TORCH_STREAM_TILE", "512"))
        self._stream_double_buffers = 2
        self._g_idx_long_cache = None
        self._zeros_cache = None
        self._cache_enabled = bool(int(os.environ.get("GPTQ_TORCH_CACHE_WEIGHTS", "0")))
        triton_flag = os.environ.get("GPTQ_TORCH_TRITON_DEQUANT")
        if triton_flag is None:
            self._triton_dequant_enabled = _TRITON_DEQUANT_AVAILABLE
        else:
            self._triton_dequant_enabled = (
                triton_flag not in {"0", "false", "False"}
            ) and _TRITON_DEQUANT_AVAILABLE
        self._cached_weights = {}
        self._lookahead_enabled = bool(int(os.environ.get("GPTQ_TORCH_LOOKAHEAD", "0")))
        self._lookahead_next = None
        self._prefetched_weights = {}
        self._prefetch_events = {}

        # if self.group_size != self.in_features:
        #     self.padded_infeatures = self.in_features + (-self.in_features % self.group_size)
        # else:
        #     self.padded_infeatures = self.in_features

    def post_init(self):
        # if self.padded_infeatures != self.in_features:
        #     self.qweight.resize_(self.padded_infeatures // self.pack_dtype_bits * self.bits, self.out_features)
        #     self.qzeros.resize_(
        #         math.ceil(self.padded_infeatures / self.group_size),
        #         self.out_features // self.pack_dtype_bits * self.bits
        #     )
        #     self.scales.resize_((math.ceil(self.padded_infeatures / self.group_size), self.out_features), )
        #     self.g_idx = torch.tensor([i // self.group_size for i in range(self.padded_infeatures)], dtype=torch.int32,
        #                               device=self.g_idx.device)

        super().post_init()

        # torch benefits the most from torch.compile, enable it by default
        self.optimize()
        self._stream_reset_cache()
        self.clear_weight_cache()
        self._reset_prefetch_state()

    def dequantize_weight(self, num_itr: int = 1):
        if (
            num_itr == 1
            and self._triton_dequant_enabled
            and self._can_use_triton_dequant()
        ):
            return self._dequantize_weight_triton()

        return super().dequantize_weight(num_itr=num_itr)

    def optimize(self, backend: str = None, mode: str = None, fullgraph: bool = False):
        if self.optimized:
            return

        if backend is None:
            # MPS doesn't support inductor.
            backend = "inductor" if self.list_buffers()[0].device.type != "mps" else "aot_eager"

        # compile dequantize
        self.dequantize_weight = torch_compile(self.dequantize_weight, backend=backend, mode=mode, fullgraph=fullgraph)

        if self.adapter:
            self.adapter.optimize(backend=backend, mode=mode, fullgraph=fullgraph)

        super().optimize()

    def train(self, mode: bool = True):
        old_train = self.training
        if mode == old_train:
            return self

        from ...utils.model import convert_gptq_v1_to_v2_format_module

        # IPEX kernel will use Torch for training only and switches back to IPEX for eval/inference
        # If the kernel inherits Torch kernel only for training and can do its own inference in v1 (IPEX, Marlin) then
        # we can support training for all these v1 kernels by enabling this flag. We need to switch qzero states
        # by overriding module train() and swapping qzero back between v1 and v2 (Torch kernel requires v2)
        if self.SUPPORTS_TRAINING_USE_TORCH_KERNEL:
            # training starts
            if mode:
                # one time clone v1 qzeros and save both v1 and v2 qzeros in memory
                if self.qzero_format() == 1:
                    if not hasattr(self, "qzeros_data_v1"):
                        self.qzeros_data_v1 = self.qzeros.data.clone()
                        convert_gptq_v1_to_v2_format_module(self, bits=self.bits, pack_dtype=self.pack_dtype)
                        self.qzeros_data_v2 = self.qzeros.data
                    else:
                        self.qzeros.data = self.qzeros_data_v2
                        self.qzero_format(format=2)

            # training switching to inference/eval
            else:
                if hasattr(self, "qzeros_data_v1"):
                    # switch qzero back to v1 for inference/eval
                    self.qzeros.data = self.qzeros_data_v1
                    self.qzero_format(format=1)

        return super().train(mode=mode)

    def forward(self, x: torch.Tensor):
        # if x.size(-1) != self.padded_infeatures:
        #     x = F.pad(x, (0, self.padded_infeatures - self.in_features))

        out_shape = x.shape[:-1] + (self.out_features,)
        x = x.reshape(-1, x.shape[-1])
        out = self._forward(x, out_shape)
        return out

    def _forward(self, x, out_shape):
        cached = self._maybe_get_cached_weights(x)
        if cached is not None:
            out = torch.matmul(x, cached).reshape(out_shape)
        elif self._should_use_streaming(x):
            out = self._forward_streaming(x, out_shape)
        else:
            out = self._forward_eager(x, out_shape)

        self._maybe_schedule_lookahead(x.dtype)
        return out

    def _forward_eager(self, x: torch.Tensor, out_shape):
        num_itr = self.g_idx.shape[0] // x.shape[-1]
        weights = self._consume_prefetched_weights(x.dtype)
        if weights is None:
            weights = self.dequantize_weight(num_itr=num_itr).to(x.dtype)
        self._update_cached_weights(weights)
        out = torch.matmul(x, weights).reshape(out_shape)

        if self.bias is not None:
            out.add_(self.bias)

        if self.adapter:
            out = self.adapter.apply(x=x, out=out)

        return out

    def _forward_streaming(self, x: torch.Tensor, out_shape):
        tile = max(64, min(self._stream_tile_cols, self.out_features))
        total_tiles = math.ceil(self.out_features / tile)
        device = x.device

        out = torch.empty((x.shape[0], self.out_features), dtype=x.dtype, device=device)
        buffers = [
            torch.empty((self.in_features, tile), dtype=x.dtype, device=device)
            for _ in range(self._stream_double_buffers)
        ]
        widths = [0 for _ in range(self._stream_double_buffers)]

        stream_dequant = torch.cuda.Stream(device=device)
        zeros = self._stream_decode_qzeros()
        g_idx = self._stream_g_idx_long()

        def schedule(tile_idx: int, buffer_idx: int):
            start = tile_idx * tile
            end = min(start + tile, self.out_features)
            with torch.cuda.stream(stream_dequant):
                widths[buffer_idx] = self._stream_dequantize_tile(
                    buffer=buffers[buffer_idx],
                    zeros=zeros,
                    g_idx=g_idx,
                    start=start,
                    end=end,
                    dtype=x.dtype,
                )

        schedule(0, 0)
        compute_stream = torch.cuda.current_stream()

        for tile_idx in range(total_tiles):
            buffer_idx = tile_idx % self._stream_double_buffers
            compute_stream.wait_stream(stream_dequant)
            width = widths[buffer_idx]
            start = tile_idx * tile
            start + width

            out_slice = out.narrow(1, start, width)
            out_slice.zero_()
            torch.addmm(
                out_slice,
                x,
                buffers[buffer_idx].narrow(1, 0, width),
                beta=0.0,
                alpha=1.0,
                out=out_slice,
            )

            next_tile = tile_idx + 1
            if next_tile < total_tiles:
                next_buffer_idx = next_tile % self._stream_double_buffers
                schedule(next_tile, next_buffer_idx)

        out = out.reshape(out_shape)

        if self.bias is not None:
            out.add_(self.bias)

        if self.adapter:
            out = self.adapter.apply(x=x, out=out)

        return out

    def _maybe_get_cached_weights(self, x: torch.Tensor):
        if not self._cache_enabled or self.training:
            return None
        cached = self._cached_weights.get(x.dtype)
        if cached is not None:
            if cached.device != x.device:
                self._cached_weights.pop(x.dtype, None)
            else:
                return cached
        return None

    def _update_cached_weights(self, weights: torch.Tensor):
        if not self._cache_enabled or self.training:
            return
        self._cached_weights[weights.dtype] = weights.detach()

    def _consume_prefetched_weights(self, dtype: torch.dtype):
        if not self._lookahead_enabled or self.training:
            return None
        tensor = self._prefetched_weights.pop(dtype, None)
        if tensor is None:
            return None
        event = self._prefetch_events.pop(dtype, None)
        if event is not None and HAS_CUDA:
            event.synchronize()
        return tensor

    def _stream_dequantize_tile(
        self,
        buffer: torch.Tensor,
        zeros: torch.Tensor,
        g_idx: torch.Tensor,
        start: int,
        end: int,
        dtype: torch.dtype,
    ) -> int:
        width = end - start
        qweight_tile = self.qweight.narrow(1, start, width)
        weight = torch.bitwise_right_shift(
            qweight_tile.unsqueeze(1).expand(-1, self.pack_factor, -1),
            self.wf_unsqueeze_neg_one,
        ).to(self.dequant_dtype)
        weight = torch.bitwise_and(weight, self.maxq)
        weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])

        zeros_tile = zeros.narrow(1, start, width)
        scales_tile = self.scales.narrow(1, start, width)

        dequant = scales_tile[g_idx] * (weight - zeros_tile[g_idx])
        buffer.narrow(1, 0, width).copy_(dequant.to(dtype))
        return width

    def _stream_decode_qzeros(self):
        if self._zeros_cache is not None and self._zeros_cache.device == self.qzeros.device:
            return self._zeros_cache

        zeros = torch.bitwise_right_shift(
            self.qzeros.unsqueeze(2).expand(-1, -1, self.pack_factor),
            self.wf_unsqueeze_zero,
        ).to(self.dequant_dtype)
        zeros = torch.bitwise_and(zeros, self.maxq).reshape(self.scales.shape)
        self._zeros_cache = zeros
        return zeros

    def _stream_g_idx_long(self):
        if self._g_idx_long_cache is None or self._g_idx_long_cache.device != self.g_idx.device:
            self._g_idx_long_cache = self.g_idx.long()
        return self._g_idx_long_cache

    def _stream_reset_cache(self):
        self._zeros_cache = None
        self._g_idx_long_cache = None

    def _should_use_streaming(self, x: torch.Tensor) -> bool:
        if not self._streaming_enabled:
            return False
        if x.device.type != "cuda":
            return False
        if not HAS_CUDA:
            return False
        # Torch kernels with num_itr > 1 already tiled differently.
        if self.g_idx.shape[0] // x.shape[-1] != 1:
            return False
        return True

    def enable_streaming(self, enabled: bool = True, tile_cols: int = None):
        self._streaming_enabled = enabled
        if tile_cols is not None:
            self._stream_tile_cols = tile_cols
        self._stream_reset_cache()
        return self

    def enable_weight_cache(self, enabled: bool = True):
        self._cache_enabled = enabled
        if not enabled:
            self.clear_weight_cache()
        return self

    def clear_weight_cache(self):
        self._cached_weights.clear()

    def enable_lookahead(self, enabled: bool = True):
        self._lookahead_enabled = enabled
        if not enabled:
            self._reset_prefetch_state()
        return self

    def set_lookahead_next(self, module: "TorchQuantLinear"):
        if module is None:
            self._lookahead_next = None
            self._reset_prefetch_state()
            return self

        if isinstance(module, TorchQuantLinear):
            self._lookahead_next = module
            return self

        if isinstance(module, Iterable):
            targets = tuple(m for m in module if m is not None)
            if not targets:
                self._lookahead_next = None
                self._reset_prefetch_state()
                return self
            for target in targets:
                if not isinstance(target, TorchQuantLinear):
                    raise TypeError("lookahead targets must be TorchQuantLinear modules or None")
            self._lookahead_next = targets
            return self

        raise TypeError("lookahead target must be TorchQuantLinear, iterable of TorchQuantLinear, or None")

    def _reset_prefetch_state(self):
        for event in self._prefetch_events.values():
            if hasattr(event, "destroy"):
                event.destroy()
        self._prefetch_events.clear()
        self._prefetched_weights.clear()

    def _maybe_schedule_lookahead(self, dtype: torch.dtype):
        if not self._lookahead_enabled or self.training:
            return
        next_module = self._lookahead_next
        if next_module is None:
            return
        if self.qweight.device.type != "cuda":
            return
        if isinstance(next_module, tuple):
            for module in next_module:
                module._prefetch(dtype)
        else:
            next_module._prefetch(dtype)

    def _prefetch(self, dtype: torch.dtype):
        if not self._lookahead_enabled or self.training:
            return
        if dtype in self._prefetched_weights:
            return
        device = self.list_buffers()[0].device
        if device.type != "cuda":
            return
        stream = torch.cuda.Stream(device=device)
        with torch.cuda.stream(stream):
            num_itr = max(1, self.g_idx.shape[0] // self.in_features)
            weights = self.dequantize_weight(num_itr=num_itr).to(dtype)
        event = torch.cuda.Event(enable_timing=False)
        event.record(stream)
        self._prefetched_weights[dtype] = weights
        self._prefetch_events[dtype] = event

    # clear gptq only weights: useful in de-quantization
    def _empty_gptq_only_weights(self):
        self.qzeros = None
        self.qweight = None
        self.g_idx = None
        self.scales = None

    def _can_use_triton_dequant(self) -> bool:
        if not _TRITON_DEQUANT_AVAILABLE:
            return False
        if self.training:
            return False
        if self.qweight is None or self.qzeros is None or self.scales is None or self.g_idx is None:
            return False
        if self.qweight.device.type != "cuda":
            return False
        if self.bits not in (2, 3, 4, 8):
            return False
        if not (self.qweight.is_contiguous() and self.qzeros.is_contiguous() and self.scales.is_contiguous()):
            return False
        # g_idx is stored as int32 tensor; ensure it resides on the same device.
        if self.g_idx.device != self.qweight.device:
            return False
        return True

    def _dequantize_weight_triton(self) -> torch.Tensor:
        # Use the Triton helper to decode weights directly on device.
        dtype = self.scales.dtype
        weights = triton_dequant(
            dtype,
            self.qweight,
            self.scales,
            self.qzeros,
            self.g_idx,
            self.bits,
            self.pack_dtype_bits,
            self.maxq,
        )
        return weights

def dequantize_model(model: PreTrainedModel):
    for name, module in model.named_modules():
        if isinstance(module, BaseQuantLinear) and not isinstance(module, TorchQuantLinear):
            raise ValueError(
                "Only models loaded using TorchQuantLinear are supported for dequantization. "
                "Please load model using backend=BACKEND.TORCH."
            )

        if isinstance(module, TorchQuantLinear):
            # Create a new Linear layer with dequantized weights
            new_module = nn.Linear(module.in_features, module.out_features)
            new_module.weight = nn.Parameter(module.dequantize_weight().T.detach().to("cpu", torch.float16))
            new_module.bias = torch.nn.Parameter(module.bias)

            # Replace the module in the model
            parent = model
            if '.' in name:
                parent_name, module_name = name.rsplit('.', 1)
                parent = dict(model.named_modules())[parent_name]
            else:
                module_name = name

            setattr(parent, module_name, new_module)

    del model.config.quantization_config
    return model


__all__ = ["TorchQuantLinear", "dequantize_model"]
