# SPDX-FileCopyrightText: 2024-2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
# Credit: int8 kernel sync adapted from Yintong Lu (yintong-lu), vLLM PR #35697.


import contextlib
import threading
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel

from ...adapter.adapter import Adapter, Lora
from ...models._const import DEVICE, PLATFORM
from ...nn_modules.qlinear import BaseQuantLinear, GPTQQuantLinear, unpack_qzero_lanes
from ...quantization import FORMAT, METHOD
from ...utils.backend import BACKEND
from ...utils.torch import dynamo_is_compiling
from ._compile_support import COMPILE_LEVEL
from ._numerics import autocast_active
from ._runtime_cache import tensor_cache_key


INT8_WEIGHT_BUFFER_NAME = "int8_weight_gnk"
INT8_GROUP_SCALE_BUFFER_NAME = "int8_group_scale"
INT8_GROUP_OFFSET_BUFFER_NAME = "int8_group_offset"
INT8_INPUT_PERM_BUFFER_NAME = "int8_input_perm"
# Legacy alias kept so external integrations importing the old name resolve to
# the group-scale buffer rather than silently reading different semantics.
INT8_SCALE_BUFFER_NAME = INT8_GROUP_SCALE_BUFFER_NAME
_INT8_REBUILD_LOCK = threading.Lock()
# Output channels decoded at once on the asymmetric path. The decoded tile is
# transient scratch: it is released before the next tile or group is processed.
_INT8_ASYMMETRIC_DECODE_TILE = 256
# Transient budget for one output-column slice of the checkpoint release
# verification. The inverse widens every stored code to INT32; slicing keeps
# that scratch bounded instead of materializing the whole layer at once.
_INT8_REBUILD_TILE_BYTES = 8 * 1024 * 1024

# `aten::_weight_int8pack_mm` reduces the contraction axis in 8-wide chunks and
# is only defined for `K % 8 == 0`: with a narrower or misaligned width it reads
# past the end of every weight row and silently returns wrong values (measured on
# torch 2.11: K=8/16/24/32 correct, K=4/12/15/17/33 wrong). Group codes are zero
# padded up to this alignment; the matching activation columns are already zero,
# so the extra products contribute exactly nothing.
_INT8_KERNEL_CONTRACTION_ALIGN = 8


def _align_int8_group_width(width: int) -> int:
    """Return the kernel-valid contraction width that holds ``width`` channels."""

    width = int(width)
    if width <= 0:
        raise ValueError(f"INT8 group width must be positive, got {width}")
    remainder = width % _INT8_KERNEL_CONTRACTION_ALIGN
    return width if remainder == 0 else width + _INT8_KERNEL_CONTRACTION_ALIGN - remainder


def _require_native_int8(bits: int) -> None:
    if bits != 8:
        raise NotImplementedError(
            "Torch INT8 requires 8-bit source weights; expanding lower-bit weights to persistent INT8 is disabled. "
            "Use BACKEND.AUTO to select a packed kernel, or BACKEND.TORCH for a portable fallback."
        )


def _has_int8_mm_op() -> bool:
    return hasattr(torch.ops.aten, "_weight_int8pack_mm")


def _fp32_decode_context(device_type: str):
    """Keep the FP32 decode and reduction out of an ambient autocast context.

    ``activations @ decoded.t()`` is an autocast-eligible matmul, so an outer
    ``torch.autocast`` narrows its FP32 operands to the autocast dtype. The
    result is then rounded before the bias cancels it, and assigning into an
    FP32 output only widens the already-rounded value. Disabling autocast
    locally preserves the caller's context on exit and leaves the single
    output rounding to the activation dtype to ``forward``.
    """
    if not autocast_active(device_type):
        return contextlib.nullcontext()
    return torch.autocast(device_type=device_type, enabled=False)


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
    then not stored at all. Symmetric groups take the native INT8 op; asymmetric
    groups decode a bounded tile first so the zero-point subtraction happens
    before the reduction instead of relying on two large FP32 terms to cancel.
    Groups with unequal lengths are padded to the widest group and
    left-multiplied by an augmented activation whose pad columns are zero.
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
        # The contraction axis must be kernel-aligned. Production tensors are
        # already padded by `transform_cpu`; a directly constructed module pads
        # its own copy so it cannot silently miscompute.
        self.source_group_width = int(int8_weight_gnk.shape[-1])
        aligned_width = _align_int8_group_width(self.source_group_width)
        if aligned_width != self.source_group_width:
            int8_weight_gnk = F.pad(int8_weight_gnk, (0, aligned_width - self.source_group_width))
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
        # The default group length is the *unpadded* width; `group_width` is the
        # kernel-aligned contraction width that may be larger.
        lengths = self.group_lengths or (self.source_group_width,) * num_groups
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

    def _asymmetric_group_output(self, flat: torch.Tensor, group: int) -> torch.Tensor:
        """Reduce one asymmetric group with its zero point applied first.

        The native INT8 op only accumulates ``scale * (code - 128)``; the
        caller would then add ``scale * (128 - zero)``. For an asymmetric
        checkpoint both terms are large and independently rounded, so their
        cancellation leaves a residue even when every decoded weight is exactly
        zero. Reconstructing the checkpoint's integer ``code - zero`` per
        element before the matmul keeps a zero weight exactly zero. The
        ``[tile, width]`` scratch is bounded by ``_INT8_ASYMMETRIC_DECODE_TILE``
        and dropped before the next tile is built.
        """
        weight = self.int8_weight_gnk[group]
        scale = self.int8_group_scale[group].float()
        offset = self.int8_group_offset[group].float()
        activations = flat[:, group, :]
        output = torch.empty(
            activations.shape[0], weight.shape[0], dtype=torch.float32, device=activations.device
        )
        for start in range(0, weight.shape[0], _INT8_ASYMMETRIC_DECODE_TILE):
            stop = min(start + _INT8_ASYMMETRIC_DECODE_TILE, weight.shape[0])
            decoded = weight[start:stop].float() * scale[start:stop].unsqueeze(-1)
            decoded = decoded + offset[start:stop].unsqueeze(-1)
            output[:, start:stop] = activations @ decoded.t()
        return output

    def forward(
        self, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        # The op requires matching activation/scale dtypes and only accepts 2D
        # activations. FP32 preserves tiny group scales and delays output
        # rounding until after the group reduction and the bias epilogue. An
        # outer autocast context would narrow the asymmetric group matmul to
        # the autocast dtype before that rounding, so the FP32 contract is
        # enforced locally and the caller's context is restored afterwards.
        with _fp32_decode_context(x.device.type):
            activations = self._activations(x)
            groups = activations.shape[-2]
            width = activations.shape[-1]
            flat = activations.reshape(-1, groups, width)
            output = None
            if self.int8_group_offset is None:
                group_scale = self.int8_group_scale.float()
                for group in range(groups):
                    part = torch.ops.aten._weight_int8pack_mm(
                        flat[:, group, :], self.int8_weight_gnk[group], group_scale[group]
                    )
                    output = part if output is None else output.add_(part)
            else:
                # Asymmetric zero points must be subtracted before the reduction.
                # Splitting them into a residual reduction plus an offset reduction
                # lets FP32 rounding survive the cancellation of two large terms.
                for group in range(groups):
                    part = self._asymmetric_group_output(flat, group)
                    output = part if output is None else output.add_(part)
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

    # CPU, torch 2.16.0.dev20261021+cpu: eager=True, graph_breaks=0,
    # dynamo/aot_eager/inductor all reach it. `_refresh_int8_runtime` used to
    # break dynamo on `tensor.is_inference()` in its source signature and on the
    # module `RLock` in its cold rebuild; a traced forward now reads the
    # `post_init()`-built INT8 module instead. CPU-only, so no CUDA-graph
    # capture applies.
    SUPPORTS_COMPILE_LEVEL = COMPILE_LEVEL.INDUCTOR
    SUPPORTS_COMPILE_EAGER_REASON = (
        "static-cache generate() auto-compile stays off: no CPU win was measured "
        "for this kernel in this change; the INDUCTOR level is opt-in via "
        "optimize() (#4984 policy unchanged)."
    )
    SUPPORTS_GRAPH_CAPTURE = False

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
        # ``(shape, dtype)`` of a checkpoint ``qweight`` released after
        # ``transform_cpu`` proved the INT8 group layout rebuilds it exactly.
        # ``None`` means the live buffer is authoritative (or still resident).
        self._int8_dropped_qweight = None

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
        # non-persistent runtime layout; once it proves it can rebuild the
        # checkpoint words exactly, the full-size ``qweight`` copy is released
        # (#5718). It is rebuilt for ``state_dict()``/save and before a
        # re-transform, so only one copy of the 8-bit codes stays resident.
        self.transform_cpu(dtype=torch.float32)
        self.optimize()

    def _source_signature(self):
        if self._checkpoint_qweight() is not None:
            return tuple(
                tensor_cache_key(getattr(self, name, None)) for name in self.GPTQ_BUFFER_NAMES
            )
        # The INT8 layout is the only resident copy of the codes, so it is the
        # source: an edit to the runtime buffers or to the small affine
        # metadata must invalidate the binding exactly like a qweight edit.
        return (
            "int8",
            tuple(
                tensor_cache_key(getattr(self, name, None))
                for name in (
                    INT8_WEIGHT_BUFFER_NAME,
                    INT8_GROUP_SCALE_BUFFER_NAME,
                    INT8_GROUP_OFFSET_BUFFER_NAME,
                    INT8_INPUT_PERM_BUFFER_NAME,
                    "scales",
                    "qzeros",
                    "g_idx",
                )
            ),
        )

    def clear_weight_cache(self):
        """Invalidate the derived INT8 binding after a source edit or load."""
        self._int8_source_signature = None
        self.int8_module = None

    def _refresh_int8_runtime(self):
        if getattr(self, "_int8_fusion_view", False):
            return self.int8_module
        if dynamo_is_compiling():
            # ``_source_signature`` reads `is_inference`/`_version`/`data_ptr`
            # (hard dynamo graph breaks), and the cold rebuild takes a module
            # ``RLock`` and rewrites the derived INT8 buffers, which a graph
            # must not do. ``post_init()`` already built the runtime module, so
            # a traced forward only reads it.
            if self.int8_module is not None:
                return self.int8_module
            raise RuntimeError(
                "TorchInt8Linear was compiled before its INT8 runtime layout was built; "
                "load the checkpoint (post_init) before torch.compile."
            )
        signature = self._source_signature()
        if self.int8_module is not None and signature == self._int8_source_signature:
            return self.int8_module
        # A post-load first forward can run from several free-threaded workers.
        # Serialize only this cold rebuild; steady-state forwards remain lock-free.
        with _INT8_REBUILD_LOCK:
            signature = self._source_signature()
            if self.int8_module is not None and signature == self._int8_source_signature:
                return self.int8_module
            if self._checkpoint_qweight() is None:
                self._materialize_checkpoint_qweight()
            if not self._has_all_gptq_buffers():
                raise RuntimeError("TorchInt8Linear missing GPTQ checkpoint buffers for native INT8 conversion.")
            self.transform_cpu(dtype=torch.float32)
            return self.int8_module

    def optimize(self, backend: str = "inductor", mode: str = None, fullgraph: bool = False):
        if self.optimized:
            return
        super().optimize(backend=backend, mode=mode, fullgraph=fullgraph)

    def _has_all_gptq_buffers(self, qweight: Optional[torch.Tensor] = None) -> bool:
        weight = self._checkpoint_qweight() if qweight is None else qweight
        if weight is None:
            return False
        return all(
            getattr(self, name, None) is not None
            for name in self.GPTQ_BUFFER_NAMES
            if name != "qweight"
        )

    def _checkpoint_qweight(self) -> Optional[torch.Tensor]:
        """Return the resident checkpoint ``qweight``, if one is materialized."""
        weight = self._buffers.get("qweight")
        return weight if isinstance(weight, torch.Tensor) else None

    def _int8_rebuild_plan(self) -> dict:
        """Validate the runtime INT8 layout and return shared inverse metadata."""
        int8_weight = getattr(self, INT8_WEIGHT_BUFFER_NAME, None)
        group_scale = getattr(self, INT8_GROUP_SCALE_BUFFER_NAME, None)
        if int8_weight is None or group_scale is None:
            raise RuntimeError(
                "TorchInt8Linear cannot rebuild qweight without its INT8 group layout."
            )
        if int8_weight.dim() != 3:
            raise RuntimeError(f"Expected 3D group codes, got shape={tuple(int8_weight.shape)}")

        num_groups, _, width = int8_weight.shape
        lengths = tuple(getattr(self, "_int8_group_lengths", None) or (width,) * num_groups)
        if len(lengths) != num_groups:
            raise RuntimeError(
                f"Group length metadata has {len(lengths)} entries for {num_groups} groups"
            )
        row_index = None
        if any(length != width for length in lengths[:-1]):
            # A short/empty non-trailing group leaves padding rows inside the
            # flat layout; gather each group's exact rows before un-permuting.
            row_index = torch.cat([
                torch.arange(group * width, group * width + length, device=int8_weight.device)
                for group, length in enumerate(lengths)
            ])
        keep = sum(lengths)
        if self.in_features is not None:
            keep = min(keep, self.in_features)
        input_perm = getattr(self, INT8_INPUT_PERM_BUFFER_NAME, None)
        inverse_input_perm = None if input_perm is None else torch.argsort(input_perm.long())
        return {
            "num_groups": num_groups,
            "width": width,
            "row_index": row_index,
            "keep": keep,
            "inverse_input_perm": inverse_input_perm,
        }

    def _rebuild_qweight_columns(
        self,
        column_start: int,
        column_stop: int,
        *,
        plan: dict,
        descriptor,
    ) -> torch.Tensor:
        """Rebuild one output-column slice of the checkpoint ``qweight`` words.

        The slice is a pure storage inverse, so every packed byte is compared
        exactly. Widening only this slice's codes bounds the transient memory
        of the release verification (#5835).
        """
        int8_weight = getattr(self, INT8_WEIGHT_BUFFER_NAME, None)
        shape, dtype = descriptor
        word_rows, out_features = int(shape[0]), int(shape[1])
        if int(int8_weight.shape[1]) != out_features:
            raise RuntimeError(
                f"INT8 layout has {int8_weight.shape[1]} output features, qweight expects {out_features}"
            )

        num_groups, width = plan["num_groups"], plan["width"]
        column_count = column_stop - column_start
        # Stored codes are ``code - 128``; add the residual zero back so the
        # words hold the checkpoint's unsigned 8-bit values.
        # The stored layout is ``[group, output, contraction]``, so the
        # checkpoint's output-column axis is the middle dimension.
        codes = int8_weight[:, column_start:column_stop, :].to(torch.int32).add(128)
        rows = codes.permute(0, 2, 1).reshape(num_groups * width, column_count)
        if plan["row_index"] is not None:
            rows = rows.index_select(0, plan["row_index"])
        rows = rows[: plan["keep"]]
        if plan["inverse_input_perm"] is not None:
            rows = rows.index_select(0, plan["inverse_input_perm"])

        words = torch.zeros((word_rows, column_count), dtype=dtype, device=rows.device)
        pack_factor = int(self.pack_factor)
        full_words = min(rows.shape[0] // pack_factor, word_rows)
        if full_words:
            grouped = rows[: full_words * pack_factor].reshape(
                full_words, pack_factor, column_count
            )
            for lane in range(pack_factor):
                words[:full_words].bitwise_or_(grouped[:, lane, :] << (self.bits * lane))
        tail = rows[full_words * pack_factor:]
        if tail.shape[0]:
            if full_words >= word_rows:
                raise RuntimeError(
                    "INT8 layout has more input channels than the checkpoint qweight can hold"
                )
            packed_tail = torch.zeros(column_count, dtype=dtype, device=rows.device)
            for lane in range(tail.shape[0]):
                packed_tail.bitwise_or_(tail[lane] << (self.bits * lane))
            words[full_words] = packed_tail
        return words

    def _rebuild_tile_columns(self, plan: dict, out_features: int) -> int:
        """Output columns to rebuild at once under the transient-memory budget."""
        rows = max(1, int(plan["keep"]))
        columns = _INT8_REBUILD_TILE_BYTES // (rows * 4)
        return max(1, min(int(out_features), columns))

    def _rebuild_checkpoint_qweight(self, descriptor=None) -> torch.Tensor:
        """Rebuild the checkpoint ``qweight`` words from the INT8 runtime layout.

        The group layout is a pure storage repack: each stored byte is the
        checkpoint's own code minus 128, permuted into group order and padded
        to the kernel's contraction alignment. Inverting it therefore
        reproduces the checkpoint words exactly. ``_maybe_drop_checkpoint_qweight``
        verifies that against the live tensor before releasing it, so an
        unexpected layout keeps its resident copy instead of exporting garbage.
        The inverse is assembled one bounded output-column slice at a time.
        """
        descriptor = descriptor or getattr(self, "_int8_dropped_qweight", None)
        if descriptor is None:
            raise RuntimeError(
                "TorchInt8Linear has no released qweight and no INT8 layout to rebuild it from."
            )
        plan = self._int8_rebuild_plan()
        int8_weight = getattr(self, INT8_WEIGHT_BUFFER_NAME, None)
        shape, dtype = descriptor
        word_rows, out_features = int(shape[0]), int(shape[1])
        words = torch.zeros((word_rows, out_features), dtype=dtype, device=int8_weight.device)
        tile = self._rebuild_tile_columns(plan, out_features)
        for start in range(0, out_features, tile):
            stop = min(start + tile, out_features)
            words[:, start:stop] = self._rebuild_qweight_columns(
                start, stop, plan=plan, descriptor=descriptor
            )
        return words

    def _materialize_checkpoint_qweight(self) -> bool:
        """Publish a released checkpoint ``qweight`` buffer again."""
        descriptor = getattr(self, "_int8_dropped_qweight", None)
        if descriptor is None:
            return self._checkpoint_qweight() is not None
        tensor = self._rebuild_checkpoint_qweight(descriptor=descriptor)
        # Publish the tensor before clearing the descriptor so a reader never
        # observes neither source.
        self._buffers["qweight"] = tensor
        self._int8_dropped_qweight = None
        return True

    def _maybe_drop_checkpoint_qweight(self) -> bool:
        """Release the checkpoint words once the INT8 layout rebuilds them exactly.

        Fails closed: any shape/value mismatch keeps both copies, so a layout
        this build cannot invert is never what a save emits. Verification walks
        bounded output-column slices so a wide layer never needs the whole
        widened inverse resident at once (#5835).
        """
        weight = self._checkpoint_qweight()
        if weight is None:
            return False
        descriptor = (tuple(weight.shape), weight.dtype)
        try:
            plan = self._int8_rebuild_plan()
            tile = self._rebuild_tile_columns(plan, weight.shape[1])
            for start in range(0, weight.shape[1], tile):
                stop = min(start + tile, weight.shape[1])
                rebuilt = self._rebuild_qweight_columns(
                    start, stop, plan=plan, descriptor=descriptor
                )
                if not torch.equal(rebuilt, weight[:, start:stop]):
                    return False
        except Exception:
            return False
        # Publish the descriptor before dropping the buffer so a concurrent
        # reader never sees a missing tensor without a rebuild recipe.
        self._int8_dropped_qweight = descriptor
        self._buffers["qweight"] = None
        return True

    def materialize_checkpoint_layout_for_export(self) -> bool:
        """Publish a released checkpoint ``qweight`` so an export can see it."""
        if getattr(self, "_int8_fusion_view", False):
            # A fused executor concatenates several layers' codes; its copied
            # descriptor only describes the first source, so it is not a
            # checkpoint and has no single qweight to publish.
            return False
        if getattr(self, "_int8_dropped_qweight", None) is None:
            return False
        return self._materialize_checkpoint_qweight()

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        with self._cpu_transform_lock:
            super()._save_to_state_dict(destination, prefix, keep_vars)
            descriptor = getattr(self, "_int8_dropped_qweight", None)
            if descriptor is None or getattr(self, "_int8_fusion_view", False):
                return
            tensor = self._rebuild_checkpoint_qweight(descriptor=descriptor)
            destination[prefix + "qweight"] = tensor if keep_vars else tensor.detach()

    def _load_from_state_dict(self, *args, **kwargs):
        # A strict load needs the checkpoint ``qweight`` slot to exist; rebuild
        # it before the copy so loading still works after ``post_init``
        # released it. The lock is re-entrant and the base hook clears the
        # derived runtime cache after the copy.
        with self._cpu_transform_lock:
            self._materialize_checkpoint_qweight()
            return super()._load_from_state_dict(*args, **kwargs)

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
        weight = self._checkpoint_qweight()
        device = weight.device if weight is not None else self.scales.device
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
        qweight = self._checkpoint_qweight()
        if qweight is None:
            # The checkpoint words are derived data now; rebuild them for this
            # explicit decode instead of keeping a second full-size copy.
            qweight = self._rebuild_checkpoint_qweight()
        if not self._has_all_gptq_buffers(qweight=qweight):
            raise RuntimeError("TorchInt8Linear missing GPTQ buffers for dequantization.")

        self._ensure_unpack_buffers()

        # Shared with `PackableQuantLinear.unpack_zero_points`: the packed zero
        # words hold ceil(out_features / pack_factor) lanes, so a non-aligned
        # output width decodes padding lanes that must be cropped here.
        zeros = unpack_qzero_lanes(
            self.qzeros,
            self.wf_unsqueeze_zero,
            pack_factor=self.pack_factor,
            maxq=self.maxq,
            dtype=self.dequant_dtype,
            out_features=self.out_features,
        )

        weight = torch.bitwise_and(
            torch.bitwise_right_shift(
                torch.unsqueeze(qweight, 1).expand(-1, self.pack_factor, -1),
                self.wf_unsqueeze_neg_one,
            ).to(self.dequant_dtype),
            self.maxq,
        )
        weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])

        # The packed layout pads the input-channel axis up to the pack factor
        # (K=17 stores 20 rows). Only the real channels are logical, matching
        # the crop `transform_cpu` applies when it builds the runtime view, so a
        # cold source decode must crop before indexing the per-channel groups.
        if weight.shape[0] < self.in_features:
            raise RuntimeError(
                "TorchInt8Linear GPTQ buffers decode fewer channels than "
                f"in_features: got {weight.shape[0]} rows for in_features={self.in_features}."
            )
        weight = weight[: self.in_features]

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
        if self._checkpoint_qweight() is None:
            # A cold refresh after the checkpoint copy was released rebuilds it
            # from the current INT8 codes first; it is dropped again below.
            self._materialize_checkpoint_qweight()
        if not self._has_all_gptq_buffers():
            raise RuntimeError("TorchInt8Linear missing GPTQ checkpoint buffers for native INT8 conversion.")
        self._ensure_unpack_buffers()

        # See `_dequantize_gptq_source_weight`: the shared decode drops the
        # packed lanes that pad the output axis past `out_features`.
        zeros = unpack_qzero_lanes(
            self.qzeros,
            self.wf_unsqueeze_zero,
            pack_factor=self.pack_factor,
            maxq=self.maxq,
            dtype=self.dequant_dtype,
            out_features=self.out_features,
        )

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
        # Pad the widest group up to the INT8 kernel's contraction alignment so
        # the runtime module never has to copy the codes again.
        kernel_width = _align_int8_group_width(max(group_lengths))
        pieces = None
        if all(length == kernel_width for length in group_lengths):
            # [K, N] -> [G, W, N] -> [G, N, W]
            grouped_codes = ordered_codes.reshape(num_groups, kernel_width, self.out_features).permute(0, 2, 1)
        else:
            pieces = []
            start = 0
            for length in group_lengths:
                piece = ordered_codes[start:start + length]
                start += length
                if length != kernel_width:
                    piece = torch.cat(
                        (piece, piece.new_zeros(kernel_width - length, self.out_features)), dim=0
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
        # The grouping scratch is dead once the group buffers are written;
        # release it before the inverse verification allocates its own tiles.
        del codes, ordered_codes, grouped_codes, pieces
        # Release the full-size checkpoint copy only after its inverse has been
        # proven against the live bytes (#5718). ``_bind_int8_runtime`` records
        # the post-release source signature so steady-state forwards hit.
        self._maybe_drop_checkpoint_qweight()
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
