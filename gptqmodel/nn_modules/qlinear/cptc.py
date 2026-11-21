# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from ...adapter.adapter import Lora
from ...models._const import DEVICE, PLATFORM
from ...quantization import FORMAT, METHOD
from ...quantization.cptc import CPTCSpec, decode_weight, quantize_weight
from ...quantization.cptc_yaqa import (
    TILE_COLUMNS,
    hadamard,
    quantize_calibrated_weight,
    rotate_matrix,
    validate_tile_shape,
)
from ...quantization.nf12 import weight_to_matrix
from ...utils.backend import BACKEND
from ...utils.cptc import native_ops
from ...utils.torch import as_torch_device
from . import COMPILE_LEVEL, WeightOnlyQuantLinear
from ._numerics import linear_wide_bias


class CPTCLinear(WeightOnlyQuantLinear):
    """Linear layer over row-major tail-biting CPTC blocks and an E2M5 table."""

    QUANT_TYPE = "cptc"
    SUPPORTS_BACKENDS = [BACKEND.CPTC]
    SUPPORTS_METHODS = [METHOD.CPTC]
    SUPPORTS_FORMATS = {FORMAT.CPTC: 100}
    SUPPORTS_BITS = [1, 2, 3, 4, 5, 6]
    SUPPORTS_SHARDS = True
    SUPPORTS_TRAINING = False
    SUPPORTS_AUTO_PADDING = False
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [1]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [1]
    SUPPORTS_DEVICES = [DEVICE.CPU, DEVICE.CUDA, DEVICE.ROCM]
    SUPPORTS_PLATFORM = [PLATFORM.ALL]
    SUPPORTS_PACK_DTYPES = [torch.int8, torch.int16, torch.int32, torch.int64]
    SUPPORTS_ADAPTERS = [Lora]
    SUPPORTS_DTYPES = [torch.float16, torch.bfloat16, torch.float32]
    SUPPORTS_CPU_OFFLOAD = True
    SUPPORTS_COMPILE_LEVEL = COMPILE_LEVEL.EAGER
    SUPPORTS_GRAPH_CAPTURE = False
    CHECKPOINT_REQUIRED_BUFFERS = ("cptc_packed", "cptc_table", "cptc_coeffs", "cptc_scale_bits")
    YAQA_BUFFERS = ("cptc_input_signs", "cptc_output_signs")
    MODEL_DTYPE_BUFFERS = ()
    CHECKPOINT_EXACT_DTYPE_BUFFERS = ("bias",)

    def __init__(
        self,
        bits: int,
        in_features: int,
        out_features: int,
        bias: bool = False,
        *,
        backend: BACKEND = BACKEND.CPTC,
        adapter=None,
        register_buffers: bool = True,
        pack_dtype: torch.dtype = torch.int32,
        group_size: int = -1,
        desc_act: bool = False,
        sym: bool = True,
        trellis_depth: int = 16,
        table_q: int = 6,
        block_size: int = 128,
        yaqa: bool = False,
        yaqa_seed: int = 0,
        dtype: torch.dtype | None = None,
        bias_dtype: torch.dtype | None = None,
        **kwargs,
    ):
        if group_size != -1 or desc_act or not sym:
            raise ValueError("CPTC requires group_size=-1, desc_act=False and sym=True")
        if in_features < 1 or out_features < 1:
            raise ValueError("CPTC requires positive in_features and out_features")
        super().__init__(
            bits=bits, in_features=in_features, out_features=out_features, bias=bias,
            backend=backend, adapter=adapter, register_buffers=False, pack_dtype=pack_dtype,
            dtype=dtype, bias_dtype=bias_dtype or dtype or torch.float32, **kwargs,
        )
        self.spec = CPTCSpec(bits, trellis_depth, table_q, block_size)
        self.yaqa = yaqa
        self.yaqa_seed = yaqa_seed
        self.tile_rows = validate_tile_shape(in_features, out_features, block_size) if yaqa else 1
        self.bias = None
        if register_buffers:
            for name, (shape, storage_dtype) in self._payload_layout().items():
                self.register_buffer(name, torch.zeros(shape, dtype=storage_dtype))
            if bias:
                del self.bias
                self.register_buffer("bias", torch.zeros(out_features, dtype=self.bias_dtype))

    def _payload_layout(self):
        if self.yaqa:
            shape = (self.out_features // self.tile_rows, self.in_features // TILE_COLUMNS)
        else:
            shape = (self.out_features, self.spec.blocks(self.in_features))
        layout = {
            "cptc_packed": ((*shape, self.spec.block_bytes), torch.uint8),
            "cptc_table": ((self.spec.table_size,), torch.uint8),
            "cptc_coeffs": ((2,), torch.int32),
            # IEEE FP32 scale bits survive Accelerate/model dtype conversion.
            "cptc_scale_bits": (shape, torch.int32),
        }
        if self.yaqa:
            layout[self.YAQA_BUFFERS[0]] = ((self.in_features,), torch.int8)
            layout[self.YAQA_BUFFERS[1]] = ((self.out_features,), torch.int8)
        return layout

    def checkpoint_required_buffers(self):
        return self.CHECKPOINT_REQUIRED_BUFFERS + (self.YAQA_BUFFERS if self.yaqa else ())

    def post_init(self):
        for name, (shape, dtype) in self._payload_layout().items():
            tensor = self._buffers.get(name)
            if tensor is None or tensor.shape != shape or tensor.dtype != dtype or not tensor.is_contiguous():
                raise ValueError(f"CPTCLinear `{self.name}` requires contiguous `{name}` with shape {shape} and {dtype}")
        super().post_init()

    def _weight_to_matrix(self, linear):
        return weight_to_matrix(linear)

    def pack(self, linear, scales=None, zeros=None, g_idx=None, **kwargs):
        self.pack_original(linear, scales, zeros, g_idx, **kwargs)

    @torch.no_grad()
    def pack_original(self, linear, scales=None, zeros=None, g_idx=None, *, smooth=None, device=None, sketch=None):
        if any(value is not None for value in (scales, zeros, g_idx, smooth)):
            raise ValueError("CPTC packs original weights and does not use external scales, zeros, indices or smoothing")
        weight = self._weight_to_matrix(linear)
        if weight.shape != (self.out_features, self.in_features):
            raise ValueError(f"CPTCLinear `{self.name}` cannot pack weight shape {tuple(weight.shape)}")
        target = self.runtime_device() or weight.device
        if target.type == "meta":
            target = weight.device
        device = weight.device if device is None else as_torch_device(device)
        if self.yaqa:
            if sketch is None:
                raise ValueError("CPTC YAQA packing requires a two-sided Hessian sketch")
            packed, table, coeffs, scale, *signs = quantize_calibrated_weight(
                weight, self.spec, sketch, device=device, seed=self.yaqa_seed, name=self.name or "",
            )
        else:
            packed, table, coeffs, scale = quantize_weight(weight, self.spec, device=device)
            signs = ()
        # Direct-pack may be called inside the looper's inference_mode scope.
        # Keep checkpoint buffers usable by ordinary state_dict and move APIs.
        with torch.inference_mode(False):
            tensors = (packed, table, coeffs, scale.view(torch.int32), *signs)
            for name, tensor in zip(self.checkpoint_required_buffers(), tensors, strict=True):
                value = tensor.to(target, copy=tensor.is_inference()).contiguous()
                self.register_buffer(name, value)
            bias = getattr(linear, "bias", None)
            if bias is not None:
                if "bias" not in self._buffers:
                    del self.bias
                self.register_buffer("bias", bias.detach().to(target, copy=True).contiguous())
            else:
                self.bias = None

    def _decode_rows(self, start, end, device, ops):
        tiles = slice(start // self.tile_rows, end // self.tile_rows)
        packed = self.cptc_packed[tiles].to(device)
        table = self.cptc_table.to(device)
        coefficients = self.cptc_coeffs.to(device)
        scales = self.cptc_scale_bits[tiles].to(device).view(torch.float32)
        if ops is not None:
            return ops.decode(
                packed, table, coefficients, scales, self.bits, self.spec.trellis_depth,
                self.spec.shift, self.in_features, self.tile_rows,
            )
        if not self.yaqa:
            return decode_weight(packed, table, coefficients, scales, self.spec, self.in_features)
        n, k, _ = packed.shape
        decoded = decode_weight(packed, table, coefficients, scales, self.spec, k * self.spec.block_size)
        return decoded.reshape(n, k, self.tile_rows, TILE_COLUMNS).transpose(1, 2).reshape(end - start, self.in_features)

    @torch.no_grad()
    def dequantize_weight(self, *, device=None, dtype=torch.float32):
        """Return a dense [out_features, in_features] matrix for explicit export."""
        target = self.cptc_packed.device if device is None else as_torch_device(device)
        weight = self._decode_rows(0, self.out_features, target, native_ops(target))
        if self.yaqa:
            weight = rotate_matrix(weight, self.cptc_input_signs.to(target),
                                   self.cptc_output_signs.to(target), inverse=True)
        return weight.to(dtype=dtype)

    def dequantize(self, *, dtype=torch.float32):
        return self.dequantize_weight(dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rows = self.input_rows(x)
        if x.shape[-1] != self.in_features:
            raise ValueError(f"CPTCLinear expected {self.in_features} input features, got {x.shape[-1]}")
        if rows == 0:
            return self.empty_linear_output(x)
        if x.dtype not in self.SUPPORTS_DTYPES:
            raise TypeError(f"CPTCLinear does not support input dtype {x.dtype}")
        activations = hadamard(x.float() * self.cptc_input_signs.to(x.device)).to(x.dtype) if self.yaqa else x
        bias = None if self.yaqa else self.bias
        ops = native_ops(x.device)
        if (ops is not None and self.cptc_packed.device == x.device and rows <= 16
                and x.dtype in (torch.float16, torch.bfloat16)
                and not (torch.is_grad_enabled() and x.requires_grad)):
            output = ops.mma(
                activations.reshape(rows, self.in_features).contiguous(), self.cptc_packed, self.cptc_table,
                self.cptc_coeffs, self.cptc_scale_bits.view(torch.float32), self.bits,
                self.spec.trellis_depth, self.spec.shift, self.tile_rows,
            )
            if bias is not None:
                output = output + bias.float()
            output = output.reshape(*x.shape[:-1], self.out_features)
        else:
            # Process the out_features axis in tiles so even a wide prefill
            # only holds a bounded dense FP32 weight scratch tile at a time.
            tile_values = (1 << 22) if ops is not None else (1 << 18)
            tile_rows = max(1, tile_values // (self.in_features * self.tile_rows)) * self.tile_rows
            wide = activations.float() if self.yaqa else activations
            tiles = []
            for start in range(0, self.out_features, tile_rows):
                end = min(start + tile_rows, self.out_features)
                weight = self._decode_rows(start, end, x.device, ops)
                tile_bias = bias[start:end] if bias is not None else None
                tiles.append(linear_wide_bias(wide, weight, tile_bias))
                del weight
            output = tiles[0] if len(tiles) == 1 else torch.cat(tiles, dim=-1)
        if self.yaqa:
            output = hadamard(output.float(), transpose=True) * self.cptc_output_signs.to(x.device)
            if self.bias is not None:
                output = output + self.bias.float().to(x.device)
        output = output.to(x.dtype)
        if self.adapter is not None:
            output = self.adapter.apply(x=x, out=output)
        return output

    def extra_repr(self):
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, "
            f"bits={self.bits}, depth={self.spec.trellis_depth}, table_q={self.spec.table_q}, "
            f"block_size={self.spec.block_size}"
        )
