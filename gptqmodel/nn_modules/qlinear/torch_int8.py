# SPDX-FileCopyrightText: 2024-2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
# Credit: int8 kernel sync adapted from Yintong Lu (yintong-lu), vLLM PR #35697.


import threading
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel

from ...adapter.adapter import Adapter, Lora
from ...models._const import DEVICE, PLATFORM
from ...nn_modules.qlinear import BaseQuantLinear, GPTQQuantLinear
from ...quantization import FORMAT, METHOD
from ...utils.backend import BACKEND
from ._runtime_cache import tensor_cache_key


INT8_WEIGHT_BUFFER_NAME = "int8_weight_gnk"
INT8_GROUP_SCALE_BUFFER_NAME = "int8_group_scale"
INT8_GROUP_OFFSET_BUFFER_NAME = "int8_group_offset"
INT8_INPUT_PERM_BUFFER_NAME = "int8_input_perm"
# Legacy alias kept so external integrations importing the old name resolve to
# the group-scale buffer rather than silently reading different semantics.
INT8_SCALE_BUFFER_NAME = INT8_GROUP_SCALE_BUFFER_NAME
_INT8_REBUILD_LOCK = threading.Lock()


def _require_native_int8(bits: int) -> None:
    if bits != 8:
        raise NotImplementedError(
            "Torch INT8 requires 8-bit source weights; expanding lower-bit weights to persistent INT8 is disabled. "
            "Use BACKEND.AUTO to select a packed kernel, or BACKEND.TORCH for a portable fallback."
        )


def _has_int8_mm_op() -> bool:
    return hasattr(torch.ops.aten, "_weight_int8pack_mm")


def _normalize_g_idx(g_idx: torch.Tensor, num_groups: int) -> torch.Tensor:
    """Resolve the checkpoint's negative group indices to explicit group rows.

    GPTQ packers may mark padded rows with ``-1`` (see
    ``GPTQQuantLinear.pack_block``), which means "the last group" when indexing
    ``scales``/``zeros``. Both the per-group channel counts and the channel
    permutation must use that same normalized value, otherwise a negative entry
    is counted in the last group but sorted before the first group.
    """
    groups = g_idx.long()
    return torch.where(groups < 0, groups + num_groups, groups)


def _group_layout(g_idx: torch.Tensor, num_groups: int) -> tuple[tuple[int, ...], int]:
    """Return per-group channel counts after sorting channels into group order."""
    groups = _normalize_g_idx(g_idx, num_groups)
    counts = torch.bincount(groups, minlength=num_groups)[:num_groups].tolist()
    return tuple(int(count) for count in counts), int(max(counts) if counts else 0)


def _dequantize_group_layout(
    *,
    int8_weight: torch.Tensor,
    group_scale: torch.Tensor,
    group_offset: Optional[torch.Tensor],
    input_perm: Optional[torch.Tensor],
    group_lengths: Optional[tuple[int, ...]],
    in_features: Optional[int],
) -> torch.Tensor:
    """Reconstruct the checkpoint ``[K, N]`` matrix from group-preserving codes.

    Codes stay in the checkpoint's own ``w - zero`` integer space, so this is a
    storage repack rather than a second quantization: every reconstructed
    weight equals the checkpoint decode exactly.
    """
    if int8_weight.dim() != 3:
        raise ValueError(f"Expected 3D group codes, got shape={tuple(int8_weight.shape)}")
    num_groups, out_features, group_width = int8_weight.shape
    lengths = tuple(group_lengths or (group_width,) * num_groups)
    if len(lengths) != num_groups:
        raise ValueError(f"Group length metadata has {len(lengths)} entries for {num_groups} groups")

    weight = int8_weight.to(torch.float32) * group_scale.to(torch.float32).unsqueeze(-1)
    if group_offset is not None:
        weight = weight + group_offset.to(torch.float32).unsqueeze(-1)

    rows = weight.permute(0, 2, 1).reshape(num_groups * group_width, out_features)
    if any(length != group_width for length in lengths[:-1]):
        # A short/empty non-trailing group leaves padding rows inside the flat
        # layout. Gathering each group's exact rows is required; truncating the
        # tail would keep padding rows from earlier groups and drop valid rows
        # from later ones. A ragged final group is already handled by the
        # trailing truncation below.
        index = torch.cat([
            torch.arange(group * group_width, group * group_width + length, device=rows.device)
            for group, length in enumerate(lengths)
        ])
        rows = rows.index_select(0, index)
    keep = sum(lengths)
    if in_features is not None:
        keep = min(keep, in_features)
    rows = rows[:keep]
    if input_perm is not None:
        if input_perm.numel() != rows.shape[0]:
            raise ValueError(
                f"Activation permutation has {input_perm.numel()} entries for {rows.shape[0]} decoded rows"
            )
        rows = rows.index_select(0, torch.argsort(input_perm.long()))
    return rows


def _cached_int8_dequantize(module: nn.Module) -> Optional[torch.Tensor]:
    """Reconstruct ``[K, N]`` checkpoint weights from the current runtime layout."""
    int8_weight = getattr(module, INT8_WEIGHT_BUFFER_NAME, None)
    group_scale = getattr(module, INT8_GROUP_SCALE_BUFFER_NAME, None)
    if int8_weight is None or group_scale is None:
        return None

    return _dequantize_group_layout(
        int8_weight=int8_weight,
        group_scale=group_scale,
        group_offset=getattr(module, INT8_GROUP_OFFSET_BUFFER_NAME, None),
        input_perm=getattr(module, INT8_INPUT_PERM_BUFFER_NAME, None),
        group_lengths=getattr(module, "_int8_group_lengths", None),
        in_features=getattr(module, "in_features", None),
    )


def _write_group_buffers(
    module: nn.Module,
    *,
    int8_weight_gnk: torch.Tensor,
    group_scale: torch.Tensor,
    group_offset: Optional[torch.Tensor],
    input_perm: Optional[torch.Tensor],
    group_lengths: tuple[int, ...],
    dtype: torch.dtype,
) -> None:
    _require_native_int8(getattr(module, "bits", None))
    buffers = {
        INT8_WEIGHT_BUFFER_NAME: int8_weight_gnk.to(torch.int8).contiguous(),
        INT8_GROUP_SCALE_BUFFER_NAME: group_scale.to(torch.float32).contiguous(),
        INT8_GROUP_OFFSET_BUFFER_NAME: (
            None if group_offset is None else group_offset.to(torch.float32).contiguous()
        ),
        INT8_INPUT_PERM_BUFFER_NAME: (
            None if input_perm is None else input_perm.to(torch.int64).contiguous()
        ),
    }
    for name, tensor in buffers.items():
        if name not in module._buffers:
            module.register_buffer(name, tensor, persistent=False)
        else:
            setattr(module, name, tensor)
    module._int8_group_lengths = tuple(int(length) for length in group_lengths)
    module._int8_transform_dtype = dtype


class Int8PackedModule(torch.nn.Module):
    """CPU grouped INT8 matmul around ``aten::_weight_int8pack_mm``.

    Each quantization group keeps its own scales and zero points, so the kernel
    evaluates the checkpoint's ``scale_g * (code - zero_g)`` product instead of
    re-quantizing to one scale per output channel. Codes are stored as the
    signed residual ``code - 128``; the residual zero point is folded into a
    per-(group, output) offset, which is exactly zero for symmetric GPTQ and is
    then not stored at all. Groups with unequal lengths are padded to the widest
    group and left-multiplied by an augmented activation whose pad columns are
    zero.
    """

    def __init__(
        self,
        int8_weight_gnk: torch.Tensor,
        group_scale: torch.Tensor,
        group_offset: Optional[torch.Tensor] = None,
        input_perm: Optional[torch.Tensor] = None,
        group_lengths: Optional[tuple[int, ...]] = None,
    ):
        super().__init__()
        self.register_buffer(INT8_WEIGHT_BUFFER_NAME, int8_weight_gnk, persistent=False)
        self.register_buffer(INT8_GROUP_SCALE_BUFFER_NAME, group_scale, persistent=False)
        self.register_buffer(INT8_GROUP_OFFSET_BUFFER_NAME, group_offset, persistent=False)
        self.register_buffer(INT8_INPUT_PERM_BUFFER_NAME, input_perm, persistent=False)
        self.group_lengths = tuple(group_lengths) if group_lengths is not None else None

    @property
    def group_width(self) -> int:
        return int(self.int8_weight_gnk.shape[-1])

    @property
    def out_features(self) -> int:
        return int(self.int8_weight_gnk.shape[1])

    def _activations(self, x: torch.Tensor) -> torch.Tensor:
        """Return FP32 ``[..., G, W]`` group-ordered activations, zero padded.

        FP32 is required by the kernel (activations and scales must share a
        dtype) and it keeps tiny group scales representable until the final
        output rounding.
        """
        width = self.group_width
        num_groups = int(self.int8_weight_gnk.shape[0])
        lengths = self.group_lengths or (width,) * num_groups
        ordered = x if self.int8_input_perm is None else x.index_select(-1, self.int8_input_perm)
        ordered = ordered.float()
        if all(length == width for length in lengths):
            return ordered.reshape(ordered.shape[:-1] + (num_groups, width))
        pieces = []
        start = 0
        for length in lengths:
            piece = ordered[..., start:start + length]
            start += length
            if piece.shape[-1] != width:
                piece = F.pad(piece, (0, width - piece.shape[-1]))
            pieces.append(piece)
        return torch.stack(pieces, dim=-2)

    def forward(
        self, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        # The op requires matching activation/scale dtypes and only accepts 2D
        # activations. FP32 preserves tiny group scales and delays output
        # rounding until after the group reduction and the bias epilogue.
        activations = self._activations(x)
        groups = activations.shape[-2]
        width = activations.shape[-1]
        flat = activations.reshape(-1, groups, width)
        group_scale = self.int8_group_scale.float()
        output = None
        for group in range(groups):
            part = torch.ops.aten._weight_int8pack_mm(
                flat[:, group, :], self.int8_weight_gnk[group], group_scale[group]
            )
            output = part if output is None else output.add_(part)
        if self.int8_group_offset is not None:
            # ``scale * (code - zero) == scale * (code - 128) + scale * (128 - zero)``
            output = torch.addmm(output, flat.sum(-1), self.int8_group_offset.float())
        output = output.reshape(activations.shape[:-2] + (self.out_features,))
        if bias is not None:
            output = output + bias.float()
        return output.to(x.dtype)


class TorchInt8Linear(GPTQQuantLinear):
    SUPPORTS_BACKENDS = [BACKEND.GPTQ_TORCH_INT8]
    SUPPORTS_METHODS = [METHOD.GPTQ]
    # Keep auto-selection unchanged; this kernel is enabled via explicit backend selection.
    SUPPORTS_FORMATS = {FORMAT.GPTQ: 0, FORMAT.GPTQ_V2: 0}
    SUPPORTS_BITS = [8]
    SUPPORTS_GROUP_SIZE = [-1, 16, 32, 64, 128]
    SUPPORTS_DESC_ACT = [True, False]
    SUPPORTS_SYM = [True, False]
    SUPPORTS_SHARDS = True
    SUPPORTS_TRAINING = False
    SUPPORTS_AUTO_PADDING = True
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [1]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [1]
    SUPPORTS_DEVICES = [DEVICE.CPU]
    SUPPORTS_PLATFORM = [PLATFORM.LINUX, PLATFORM.WIN32]
    SUPPORTS_PACK_DTYPES = [torch.int32]
    SUPPORTS_ADAPTERS = [Lora]

    SUPPORTS_DTYPES = [torch.float16, torch.bfloat16, torch.float32]

    REQUIRES_FORMAT_V2 = True
    SUPPORTS_RAW_PACKING = False

    QUANT_TYPE = "torch_int8"
    GPTQ_BUFFER_NAMES = ("qzeros", "qweight", "g_idx", "scales")
    UNPACK_BUFFER_NAMES = ("wf_unsqueeze_zero", "wf_unsqueeze_neg_one")

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
            backend=kwargs.pop("backend", BACKEND.GPTQ_TORCH_INT8),
            adapter=adapter,
            register_buffers=register_buffers,
            **kwargs,
        )
        self.dequant_dtype = torch.int16 if self.bits == 8 else torch.int8
        self.int8_module: Optional[Int8PackedModule] = None
        self._int8_source_signature = None

    @classmethod
    def validate_once(cls) -> Tuple[bool, Optional[Exception]]:
        if not _has_int8_mm_op():
            return False, ImportError("aten::_weight_int8pack_mm is unavailable in this PyTorch build.")
        return True, None

    @classmethod
    def _validate(cls, bits=8, **kwargs):
        try:
            _require_native_int8(bits)
        except NotImplementedError as error:
            return False, error
        return super()._validate(bits=bits, **kwargs)

    def post_init(self):
        super().post_init()
        # The GPTQ tensors are the checkpoint. Native INT8 is a derived,
        # non-persistent runtime layout and must never replace those tensors.
        self.transform_cpu(dtype=torch.float32)
        self.optimize()

    def _source_signature(self):
        return tuple(tensor_cache_key(getattr(self, name, None)) for name in self.GPTQ_BUFFER_NAMES)

    def clear_weight_cache(self):
        """Invalidate the derived INT8 binding after a source edit or load."""
        self._int8_source_signature = None
        self.int8_module = None

    def _refresh_int8_runtime(self):
        if getattr(self, "_int8_fusion_view", False):
            return self.int8_module
        signature = self._source_signature()
        if self.int8_module is not None and signature == self._int8_source_signature:
            return self.int8_module
        # A post-load first forward can run from several free-threaded workers.
        # Serialize only this cold rebuild; steady-state forwards remain lock-free.
        with _INT8_REBUILD_LOCK:
            signature = self._source_signature()
            if self.int8_module is not None and signature == self._int8_source_signature:
                return self.int8_module
            if not self._has_all_gptq_buffers():
                raise RuntimeError("TorchInt8Linear missing GPTQ checkpoint buffers for native INT8 conversion.")
            self.transform_cpu(dtype=torch.float32)
            return self.int8_module

    def optimize(self):
        if self.optimized:
            return
        super().optimize()

    def _has_all_gptq_buffers(self) -> bool:
        return all(getattr(self, name, None) is not None for name in self.GPTQ_BUFFER_NAMES)

    def _delete_attr_if_exists(self, attr_name: str):
        if hasattr(self, attr_name):
            delattr(self, attr_name)

    def _ensure_unpack_buffers(self):
        if (
            hasattr(self, "wf_unsqueeze_zero")
            and hasattr(self, "wf_unsqueeze_neg_one")
            and self.wf_unsqueeze_zero is not None
            and self.wf_unsqueeze_neg_one is not None
        ):
            return

        if self.bits not in [2, 4, 8]:
            raise NotImplementedError("TorchInt8Linear unpack only supports bits in [2, 4, 8].")

        wf = torch.tensor(list(range(0, self.pack_dtype_bits, self.bits)), dtype=torch.int32).unsqueeze(0)
        device = self.qweight.device
        wf_zero = wf.unsqueeze(0).to(device=device)
        wf_neg_one = wf.unsqueeze(-1).to(device=device)

        if "wf_unsqueeze_zero" not in self._buffers:
            self.register_buffer("wf_unsqueeze_zero", wf_zero, persistent=False)
        else:
            self.wf_unsqueeze_zero = wf_zero

        if "wf_unsqueeze_neg_one" not in self._buffers:
            self.register_buffer("wf_unsqueeze_neg_one", wf_neg_one, persistent=False)
        else:
            self.wf_unsqueeze_neg_one = wf_neg_one

    def _drop_unpack_buffers(self):
        for name in self.UNPACK_BUFFER_NAMES:
            self._delete_attr_if_exists(name)

    def dequantize_weight(self, num_itr: int = 1):
        if num_itr != 1:
            raise NotImplementedError("TorchInt8Linear dequantize_weight only supports num_itr == 1.")
        # Explicit dequantized export still reflects the native INT8 runtime
        # when it is current; conversion always decodes the GPTQ source below.
        if self.int8_module is not None and self._int8_source_signature == self._source_signature():
            dequantized = _cached_int8_dequantize(self)
            if dequantized is not None:
                return dequantized
        return self._dequantize_gptq_source_weight()

    def _dequantize_gptq_source_weight(self):
        if not self._has_all_gptq_buffers():
            raise RuntimeError("TorchInt8Linear missing GPTQ buffers for dequantization.")

        self._ensure_unpack_buffers()

        zeros = torch.bitwise_right_shift(
            torch.unsqueeze(self.qzeros, 2).expand(-1, -1, self.pack_factor),
            self.wf_unsqueeze_zero,
        ).to(self.dequant_dtype)
        zeros = torch.bitwise_and(zeros, self.maxq).reshape(self.scales.shape)

        weight = torch.bitwise_and(
            torch.bitwise_right_shift(
                torch.unsqueeze(self.qweight, 1).expand(-1, self.pack_factor, -1),
                self.wf_unsqueeze_neg_one,
            ).to(self.dequant_dtype),
            self.maxq,
        )
        weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])

        groups = _normalize_g_idx(self.g_idx, int(self.scales.shape[0]))
        return self.scales[groups] * (weight - zeros[groups])

    @torch.inference_mode()
    def pack_block(
        self,
        linear: nn.Module,
        scales: torch.Tensor,
        zeros: torch.Tensor,
        g_idx: torch.Tensor,
        block_in: int = 8192,
        workers: int = 1,
    ):
        raise NotImplementedError(
            "TorchInt8Linear is not packable. Load GPTQ 8-bit tensors and let post_init() convert to int8."
        )

    def pack(
        self,
        linear: nn.Module,
        scales: torch.Tensor,
        zeros: torch.Tensor,
        g_idx: torch.Tensor,
        block_in: int = 8192,
        workers: int = 1,
    ):
        raise NotImplementedError(
            "TorchInt8Linear is not packable. Load GPTQ 8-bit tensors and let post_init() convert to int8."
        )

    def transform_cpu(self, dtype: torch.dtype):
        _require_native_int8(self.bits)
        if not self._has_all_gptq_buffers():
            raise RuntimeError("TorchInt8Linear missing GPTQ checkpoint buffers for native INT8 conversion.")
        self._ensure_unpack_buffers()

        zeros = torch.bitwise_right_shift(
            torch.unsqueeze(self.qzeros, 2).expand(-1, -1, self.pack_factor),
            self.wf_unsqueeze_zero,
        ).to(self.dequant_dtype)
        zeros = torch.bitwise_and(zeros, self.maxq).reshape(self.scales.shape)

        codes = torch.bitwise_and(
            torch.bitwise_right_shift(
                torch.unsqueeze(self.qweight, 1).expand(-1, self.pack_factor, -1),
                self.wf_unsqueeze_neg_one,
            ).to(self.dequant_dtype),
            self.maxq,
        ).reshape(-1, self.out_features)
        codes = codes[:self.in_features]

        num_groups = int(self.scales.shape[0])
        groups = _normalize_g_idx(self.g_idx, num_groups)
        group_lengths, _ = _group_layout(groups, num_groups)
        packed_channels = sum(group_lengths)
        if packed_channels != self.in_features:
            raise NotImplementedError(
                "TorchInt8Linear requires exactly one g_idx entry per input channel; "
                f"got {packed_channels} entries for in_features={self.in_features}."
            )

        # Channels are stored group-contiguously so one kernel launch per
        # quantization group covers every channel of that group. Stable sorting
        # keeps the relative order inside a group and makes the mapping exactly
        # invertible for dequantization and export.
        input_perm = torch.argsort(groups, stable=True)
        ordered_codes = codes.index_select(0, input_perm)
        group_width = max(group_lengths)
        if all(length == group_width for length in group_lengths):
            # [K, N] -> [G, W, N] -> [G, N, W]
            grouped_codes = ordered_codes.reshape(num_groups, group_width, self.out_features).permute(0, 2, 1)
        else:
            pieces = []
            start = 0
            for length in group_lengths:
                piece = ordered_codes[start:start + length]
                start += length
                if length != group_width:
                    piece = torch.cat(
                        (piece, piece.new_zeros(group_width - length, self.out_features)), dim=0
                    )
                pieces.append(piece)
            grouped_codes = torch.stack(pieces, dim=0).permute(0, 2, 1)

        # Store the signed residual ``code - 128`` so every 8-bit checkpoint
        # code (``0..255``) maps into the signed INT8 kernel range. The residual
        # zero point becomes a per-(group, output) offset that is exactly zero
        # for symmetric GPTQ, in which case it is not stored.
        grouped_codes = (grouped_codes - 128).to(torch.int8).contiguous()
        group_scale = self.scales.float().contiguous()
        # ``scale * (code - zero) == scale * (code - 128) + scale * (128 - zero)``
        offset = (128.0 - zeros.float()) * group_scale
        group_offset = None if bool((offset == 0).all()) else offset.contiguous()

        _write_group_buffers(
            self,
            int8_weight_gnk=grouped_codes,
            group_scale=group_scale,
            group_offset=group_offset,
            input_perm=input_perm,
            group_lengths=group_lengths,
            dtype=dtype,
        )
        self._drop_unpack_buffers()
        self._bind_int8_runtime()

    def _bind_int8_runtime(self):
        """Bind the executor to the current derived group buffers."""
        self.int8_module = Int8PackedModule(
            self.int8_weight_gnk,
            self.int8_group_scale,
            self.int8_group_offset,
            self.int8_input_perm,
            getattr(self, "_int8_group_lengths", None),
        ).eval()
        self._int8_source_signature = self._source_signature()
        return self.int8_module

    def transform(self, dtype: torch.dtype, device: str):
        if device != "cpu":
            raise NotImplementedError("TorchInt8Linear only supports CPU.")
        self.transform_cpu(dtype)

    def forward(self, x: torch.Tensor):
        _require_native_int8(self.bits)
        if self.training:
            raise NotImplementedError("TorchInt8Linear does not support training mode.")
        # Common decode path is 2D [M, K]. Skip reshape/out-shape overhead on this hot path.
        if x.dim() == 2:
            out = self._fused_op_forward(x, bias=self.bias)
            if self.adapter:
                out = self.adapter.apply(x=x, out=out)
            return out

        out_shape = x.shape[:-1] + (self.out_features,)
        x = x.reshape(-1, x.shape[-1])

        out = self._fused_op_forward(x, bias=self.bias).reshape(out_shape)
        if self.adapter:
            out = self.adapter.apply(x=x, out=out)

        return out

    @torch.no_grad
    def _fused_op_forward(
        self, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        if x.device.type != "cpu":
            raise NotImplementedError("TorchInt8Linear fused path is CPU-only.")
        int8_module = self._refresh_int8_runtime()
        if int8_module is None:
            raise RuntimeError("TorchInt8Linear int8 module is not initialized.")
        return int8_module(x.contiguous(), bias=bias)


def dequantize_model(model: PreTrainedModel):
    for name, module in model.named_modules():
        if isinstance(module, BaseQuantLinear) and not isinstance(module, TorchInt8Linear):
            raise ValueError(
                "Only models loaded using TorchInt8Linear are supported "
                "for dequantization. Please load model using backend=BACKEND.GPTQ_TORCH_INT8 or "
                "use the dequantizer for the model's packed backend."
            )

        if isinstance(module, TorchInt8Linear):
            new_module = nn.Linear(module.in_features, module.out_features, bias=module.bias is not None)
            new_module.weight = nn.Parameter(module.dequantize_weight().T.detach().to("cpu", torch.float16))
            if module.bias is not None:
                new_module.bias = torch.nn.Parameter(module.bias.detach().to("cpu", torch.float16))

            parent = model
            if "." in name:
                parent_name, module_name = name.rsplit(".", 1)
                parent = dict(model.named_modules())[parent_name]
            else:
                module_name = name

            setattr(parent, module_name, new_module)

    del model.config.quantization_config
    return model


__all__ = [
    "INT8_GROUP_OFFSET_BUFFER_NAME",
    "INT8_GROUP_SCALE_BUFFER_NAME",
    "INT8_INPUT_PERM_BUFFER_NAME",
    "INT8_SCALE_BUFFER_NAME",
    "INT8_WEIGHT_BUFFER_NAME",
    "Int8PackedModule",
    "TorchInt8Linear",
    "_cached_int8_dequantize",
    "_dequantize_group_layout",
    "_group_layout",
    "_has_int8_mm_op",
    "_normalize_g_idx",
    "_write_group_buffers",
    "dequantize_model",
]
