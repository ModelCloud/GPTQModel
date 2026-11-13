# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Tenstorrent block-float inference layers.

The weights of a Tenstorrent accelerator are stored as blocks of 16 values that
share one exponent, with a fixed number of mantissa bits per value:
``bf4_b`` keeps 2 and ``bf8_b`` keeps 6. That is a storage format the hardware
unpacks natively, so a checkpoint written this way is already in the layout the
device wants and does not need a host-side requantisation step. ``ttnn`` spells
these ``bfloat4_b`` / ``bfloat8_b``; ``bf4_b``/``bf8_b`` are this repository's names
for the same thing, defined at the bit level in ``hw/dtypes.md``.

Every module here keeps that representation: packed mantissa codes plus int8
exponents are the module's state, and a dense weight only exists for the
duration of one forward pass. Nothing is cached between calls, so the
quantized-storage invariant holds for these paths exactly as it does for the
packed GPTQ kernels.

The storage is named after the buffers the module owns -- ``codes``,
``exponents``, ``code_shape`` -- so a checkpoint loads through plain
``load_state_dict`` with no name translation. That is what lets the same files
be read by this library, by a Tenstorrent kernel, or by the modeling file the
checkpoint ships for ``transformers``.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...adapter.adapter import Adapter, Lora
from ...models._const import DEVICE, PLATFORM
from ...quantization import FORMAT, METHOD
from ...utils.backend import BACKEND
from . import BaseQuantLinear


#: Values per shared exponent. The hardware format fixes this; a checkpoint
#: whose block size differs cannot be fed to a device kernel unchanged.
BLOCKFLOAT_BLOCK = 16


def packed_code_bytes(numel: int, bits: int) -> int:
    """Bytes needed to store ``numel`` codes at ``bits`` per value.

    4-bit packs two codes per byte and 8-bit one, matching
    :func:`~gptqmodel.utils.tt_blockfloat_quant._pack_codes`. That packer pads
    an odd number of codes with one spare nibble, so the byte count rounds up:
    rounding down would declare a buffer one byte smaller than the tensor the
    packer produces, and the checkpoint would fail to load for an odd
    ``out_features * in_features``.  2-bit packs four per byte and pads the same
    way.
    """

    if bits == 2:
        return (numel + 3) // 4
    if bits == 4:
        return (numel + 1) // 2
    if bits == 8:
        return numel
    raise ValueError(
        f"no Tenstorrent block-float format at {bits!r} bits; "
        "supported widths are 2, 4 and 8"
    )


def _buffer_shape(out_features: int, in_features: int, bits: int, block: int):
    """Shapes the three storage buffers must have for this geometry."""

    if in_features % block:
        raise ValueError(
            f"in_features={in_features} must be a multiple of the block size {block}"
        )
    return (
        torch.Size([packed_code_bytes(out_features * in_features, bits)]),
        torch.Size([out_features, in_features // block]),
    )


class _BlockfloatStorage(nn.Module):
    """Codes plus exponents, with a dense view computed only when needed."""

    def _init_storage(self, out_features: int, in_features: int, bits: int, block: int):
        codes_shape, exponents_shape = _buffer_shape(
            out_features, in_features, bits, block
        )
        # Sized up front, not lazily: a plain ``load_state_dict`` rejects a
        # size-0 buffer, and sizing from the geometry is what makes the
        # checkpoint loadable without any bespoke reader.
        self.register_buffer("codes", torch.zeros(codes_shape, dtype=torch.uint8))
        self.register_buffer(
            "code_shape",
            torch.tensor([out_features, in_features], dtype=torch.int64),
        )
        self.register_buffer(
            "exponents", torch.zeros(exponents_shape, dtype=torch.int8)
        )

    def hip_dispatch(self, tensor: torch.Tensor) -> bool:
        """Whether the HIP kernels can serve ``tensor`` for this storage.

        The kernels bind one exponent per 16 values, so storage with a wider
        ``block`` must stay on the Torch path even when the device and dtype
        are supported. Call sites that bind an activation flatten it to 2-D
        themselves.
        """

        from ...utils import tt_blockfloat_hip

        return (
            tensor.is_cuda
            and self.block == BLOCKFLOAT_BLOCK
            and tt_blockfloat_hip.available(tensor.device)
            and tt_blockfloat_hip.supports_bits(self.bits)
        )

    def dequantize(self, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Rebuild the exact grid weights as ``[out_features, in_features]``."""

        from ...utils import tt_blockfloat_hip
        from ...utils.tt_blockfloat_quant import _unpack_codes, recompose_blockfloat

        shape = tuple(int(v) for v in self.code_shape)
        if self.hip_dispatch(self.codes):
            # The HIP kernel expands the packed stream directly, so the host
            # never materialises the unpacked uint8 code matrix.
            return tt_blockfloat_hip.dequantize(
                self.codes, self.exponents, self.bits, shape, dtype
            )
        codes = _unpack_codes(self.codes, self.bits, shape)
        weight = recompose_blockfloat(
            codes, self.exponents, bits=self.bits, block=self.block
        )
        return weight.to(dtype)

    def set_quantized_state(self, codes: torch.Tensor, exponents: torch.Tensor) -> None:
        """Install packed codes and exponents directly (used by the writer)."""

        from ...utils.tt_blockfloat_quant import _pack_codes

        codes = codes.to(torch.int64)
        expected_codes, expected_exponents = _buffer_shape(
            self.out_features, self.in_features, self.bits, self.block
        )
        if tuple(exponents.shape) != tuple(expected_exponents):
            raise ValueError(
                f"{self.name}: exponent shape {tuple(exponents.shape)} does not "
                f"match {tuple(expected_exponents)} (one exponent per {self.block} values)"
            )
        packed = _pack_codes(codes, self.bits).contiguous()
        if tuple(packed.shape) != tuple(expected_codes):
            raise ValueError(
                f"{self.name}: code shape {tuple(codes.shape)} does not match "
                f"({self.out_features}, {self.in_features})"
            )
        self.codes = packed
        self.exponents = exponents.to(torch.int8).contiguous()

    def set_packed_state(
        self, packed_codes: torch.Tensor, exponents: torch.Tensor
    ) -> None:
        """Install codes that are already packed at the storage density."""

        expected_codes, expected_exponents = _buffer_shape(
            self.out_features, self.in_features, self.bits, self.block
        )
        packed_codes = packed_codes.to(torch.uint8).contiguous()
        if tuple(packed_codes.shape) != tuple(expected_codes):
            raise ValueError(
                f"{self.name}: packed code shape {tuple(packed_codes.shape)} "
                f"does not match {tuple(expected_codes)}"
            )
        if tuple(exponents.shape) != tuple(expected_exponents):
            raise ValueError(
                f"{self.name}: exponent shape {tuple(exponents.shape)} does not "
                f"match {tuple(expected_exponents)} (one exponent per {self.block} values)"
            )
        self.codes = packed_codes
        self.exponents = exponents.to(torch.int8).contiguous()

    def dense_weight(self) -> torch.Tensor:
        """Dense view of the weights, computed on demand and never retained.

        Deliberately not named ``weight``: a ``weight`` attribute would read as
        a dense parameter to module-tree code that tests ``module.weight.dim()``
        and would invite callers to hold the expanded tensor.
        """

        return self.dequantize()

    def _kernel_linear(self, x: torch.Tensor, bias: Optional[torch.Tensor] = None):
        """Run the block-float GEMM on this device, or return ``None``.

        The kernel reads the packed codes and exponents directly, so no dense
        weight is built.  That is the whole point of the emulation: on a CUDA
        part a ``bf4_b`` layer then costs what it will cost on Tensix,
        rather than what a dequantise-then-matmul costs.

        Returning ``None`` keeps every other device on the dense path.
        """

        from ...utils import tt_blockfloat_hip

        if self.hip_dispatch(x):
            # Fused unpack + GEMM: the packed weight is read once and expanded
            # only into shared memory, so no dense weight is created here.
            # The binding takes a 2-D activation, so flatten the leading
            # dimensions and restore them afterwards; transformers pass
            # [batch, sequence, hidden] and F.linear accepted that.
            out = tt_blockfloat_hip.linear(
                x.reshape(-1, x.shape[-1]),
                self.codes,
                self.exponents,
                self.bits,
                bias,
            )
            # Restore the caller's leading dimensions, including the 1-D case
            # where there are none and F.linear returns [out_features].
            return out.reshape(*x.shape[:-1], out.shape[-1])

        if x.device.type != "cuda" or self.codes.device != x.device:
            return None
        if self.bits not in (4, 8):
            return None
        if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            return None

        from ...utils import tt_blockfloat_cuda as bf_cuda

        if not bf_cuda.ready(x.device):
            return None

        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        out = bf_cuda.linear(flat, self.codes, self.exponents, self.bits, bias)
        return out.reshape(*shape[:-1], out.shape[-1])


class TTBlockfloatLinear(_BlockfloatStorage, BaseQuantLinear):
    """Linear over Tenstorrent block-float weights."""

    SUPPORTS_BACKENDS = [BACKEND.TT_BLOCKFLOAT]
    SUPPORTS_METHODS = [METHOD.TT_BLOCKFLOAT]
    SUPPORTS_FORMATS = {FORMAT.TT_BLOCKFLOAT: 100}
    SUPPORTS_BITS = [4, 8]
    SUPPORTS_SHARDS = True
    SUPPORTS_TRAINING = False
    SUPPORTS_AUTO_PADDING = False
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [BLOCKFLOAT_BLOCK]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [1]
    SUPPORTS_PACK_DTYPES = [torch.uint8]
    SUPPORTS_ADAPTERS = [Lora]
    SUPPORTS_DEVICES = [DEVICE.CPU, DEVICE.CUDA, DEVICE.ROCM, DEVICE.XPU, DEVICE.NPU, DEVICE.MPS]
    SUPPORTS_PLATFORM = [PLATFORM.ALL]
    SUPPORTS_DTYPES = [torch.float16, torch.bfloat16, torch.float32]

    def __init__(
        self,
        bits: int,
        in_features: int,
        out_features: int,
        bias: bool,
        backend: BACKEND,
        adapter: Optional[Adapter] = None,
        name: str = None,
        dtype: Optional[torch.dtype] = None,
        block: int = BLOCKFLOAT_BLOCK,
        pack_dtype: torch.dtype = torch.uint8,
        **kwargs,
    ):
        if block != BLOCKFLOAT_BLOCK:
            raise NotImplementedError(
                f"{type(self).__name__} stores one exponent per {BLOCKFLOAT_BLOCK} "
                f"values; block={block} is not a hardware format"
            )
        super().__init__(
            bits=bits,
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            backend=backend,
            adapter=adapter,
            name=name,
            register_buffers=False,
            dtype=dtype,
            validate_kwargs={"pack_dtype": pack_dtype},
        )
        self.block = block
        self.pack_dtype = pack_dtype
        self._init_storage(out_features, in_features, bits, block)
        if bias:
            # Bias is not quantized, so keep the precision of the model it came
            # from. Narrowing to float16 here would overflow large finite
            # values and round ordinary ones even for a float32 checkpoint.
            bias_dtype = dtype if isinstance(dtype, torch.dtype) else torch.float16
            self.register_buffer("bias", torch.zeros(out_features, dtype=bias_dtype))
        else:
            self.bias = None

    def pack(self, linear: nn.Module, scales=None, zeros=None, g_idx=None, **kwargs):
        """Quantise a dense ``linear`` onto the block-float grid."""

        from ...utils import tt_blockfloat_hip
        from ...utils.tt_blockfloat_quant import quantize_blockfloat

        weight = linear.weight.detach().to(torch.float32)
        if self.hip_dispatch(weight):
            # Quantise on the GPU: the exponent search and packing stay on the
            # device instead of round-tripping a dense fp32 copy through torch.
            codes, exponents = tt_blockfloat_hip.quantize(weight, self.bits)
            self.set_packed_state(codes, exponents)
        else:
            self.load_codes_from(
                quantize_blockfloat(weight, self.bits, block=self.block)[0]
            )
        if getattr(linear, "bias", None) is not None:
            # Preserve the source dtype exactly: this buffer is dense state,
            # not a quantized payload, so there is nothing to gain by narrowing.
            self.bias = linear.bias.detach().to(linear.bias.dtype).contiguous()
        return self

    def load_codes_from(self, grid_weight: torch.Tensor) -> None:
        """Store an already-on-grid weight as packed codes plus exponents."""

        from ...utils.tt_blockfloat_quant import decompose_blockfloat

        codes, exponents = decompose_blockfloat(
            grid_weight.detach().to(torch.float32), bits=self.bits, block=self.block
        )
        self.set_quantized_state(codes, exponents)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0:
            return self.empty_linear_output(x)

        out_dtype = x.dtype
        bias = None if self.bias is None else self.bias.to(x.dtype)
        out = self._kernel_linear(x, bias)
        if out is None:
            weight = self.dequantize(dtype=x.dtype)
            out = F.linear(x, weight, bias)
            # Release the temporary before the activation leaves this scope.
            del weight

        if self.adapter is not None:
            out = self.adapter.apply(x=x, out=out)
        return out.to(out_dtype)


class TTBlockfloatEmbedding(_BlockfloatStorage, nn.Module):
    """Embedding lookup over block-float weights.

    An embedding is a lookup table, so its cost is bytes rather than arithmetic.
    Storing it as codes keeps it at ``bf8_b`` density instead of a dense
    float16 table: 266.16 MiB against 501.00 MiB for Llama-3.2-1B, measured
    from the shard headers (250.50 MiB codes + 15.66 MiB exponents).
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        bits: int = 8,
        block: int = BLOCKFLOAT_BLOCK,
        name: str = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.name = name or type(self).__name__
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        # The table owns no float parameter, so it cannot read the activation
        # dtype off itself the way ``TTBlockfloatLinear`` does. A plain attribute
        # would also be frozen at construction -- ``.to()``, ``.float()`` and
        # ``.bfloat16()`` only convert parameters and buffers -- leaving the
        # table emitting bf16 into fp32 linears after a cast. A zero-element
        # non-persistent buffer is converted by ``Module.to``, so the table stays
        # in step without adding a key to the checkpoint.
        output_dtype = dtype if isinstance(dtype, torch.dtype) else torch.float32
        self.register_buffer(
            "_output_dtype_anchor",
            torch.zeros(0, dtype=output_dtype),
            persistent=False,
        )
        # ``_BlockfloatStorage`` reads ``out_features``/``in_features``; for a
        # table the rows are the outputs.
        self.out_features = num_embeddings
        self.in_features = embedding_dim
        self.bits = bits
        self.block = block
        self.pack_dtype = torch.uint8
        self._init_storage(num_embeddings, embedding_dim, bits, block)

    @property
    def output_dtype(self) -> torch.dtype:
        """Precision the table dequantizes into; follows ``Module.to``."""

        return self._output_dtype_anchor.dtype

    def _kernel_embedding(self, input_ids: torch.Tensor):
        """Gather-and-dequantise on the block-float kernels, or return ``None``."""

        from ...utils import tt_blockfloat_hip

        if self.hip_dispatch(input_ids):
            # F.embedding accepts int32 or int64; the binding indexes with
            # int64, so widen the narrower width instead of failing.
            if input_ids.dtype == torch.int32:
                input_ids = input_ids.to(torch.int64)
            if input_ids.numel() == 0:
                # An empty batch has null data pointers, which the launcher
                # would reject; F.embedding returned an empty table slice.
                return torch.empty(
                    *input_ids.shape,
                    self.in_features,
                    dtype=self.output_dtype,
                    device=input_ids.device,
                )
            # Dequantise only the gathered rows; the table is never expanded.
            # The output precision follows the module, as the Torch path does.
            return tt_blockfloat_hip.embedding(
                input_ids,
                self.codes,
                self.exponents,
                self.bits,
                self.out_features,
                self.in_features,
                self.output_dtype,
            )

        if input_ids.device.type != "cuda" or self.codes.device != input_ids.device:
            return None
        if self.bits not in (4, 8):
            return None
        if self.output_dtype not in (torch.float16, torch.bfloat16, torch.float32):
            return None

        from ...utils import tt_blockfloat_cuda as bf_cuda

        if not bf_cuda.ready(input_ids.device):
            return None

        shape = input_ids.shape
        flat = input_ids.reshape(-1).to(torch.int64)
        out = bf_cuda.embedding(
            flat,
            self.codes,
            self.exponents,
            self.bits,
            self.out_features,
            self.in_features,
            self.output_dtype,
        )
        return out.reshape(*shape, self.in_features)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.numel() == 0:
            # An empty gather still has a shape and a dtype to preserve; the
            # table's own precision, and a zero-block launch is an error.
            return torch.empty(
                *input_ids.shape,
                self.in_features,
                dtype=self.output_dtype,
                device=input_ids.device,
            )
        kernel = self._kernel_embedding(input_ids)
        if kernel is not None:
            return kernel
        weight = self.dequantize(dtype=self.output_dtype)
        return F.embedding(input_ids, weight)


class TTBlockfloatTiedHead(nn.Module):
    """Output projection that shares an embedding's codes instead of copying them.

    When ``tie_word_embeddings`` is set the head is the same matrix as the
    embedding. Keeping a second dense copy would defeat the point of storing the
    embedding as codes, so this reads the embedding's storage.

    The embedding is held in a list rather than as an attribute so it is not
    registered a second time: a submodule reference would duplicate every
    ``.codes`` key in the state dict.
    """

    def __init__(self, embedding: TTBlockfloatEmbedding, name: str = None):
        super().__init__()
        self.name = name or type(self).__name__
        self._embedding = [embedding]

    @property
    def embedding(self) -> TTBlockfloatEmbedding:
        return self._embedding[0]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.numel() == 0:
            return torch.empty(
                *hidden_states.shape[:-1],
                self.embedding.out_features,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
        embedding = self.embedding
        kernel = embedding._kernel_linear(hidden_states)
        if kernel is not None:
            return kernel
        weight = embedding.dequantize(dtype=hidden_states.dtype)
        return F.linear(hidden_states, weight)

    def tie_weights(self) -> None:
        """No-op: the tie is the shared storage, not a parameter copy."""

        return None


def swap_blockfloat_modules(
    model: nn.Module,
    *,
    bits_per_layer: Optional[dict] = None,
    default_bits: int = 4,
    embed_bits: int = 8,
    block: int = BLOCKFLOAT_BLOCK,
    quantized_prefixes: Optional[list] = None,
) -> dict:
    """Replace quantized modules on an already-built model with block-float ones.

    A module is quantized if its name is in ``quantized_prefixes`` or in
    ``bits_per_layer``. Returns ``{prefix: [out_features, in_features]}`` for
    everything swapped. Used by the library loader and by the modeling file the
    checkpoint ships, so both produce the same object graph.
    """

    bits_per_layer = dict(bits_per_layer or {})
    targets = list(quantized_prefixes or bits_per_layer)

    restored = {}
    for prefix in targets:
        try:
            module = model.get_submodule(prefix)
        except AttributeError:
            continue
        bits = int(bits_per_layer.get(prefix, default_bits))
        parent_name, _, attr = prefix.rpartition(".")

        if isinstance(module, nn.Embedding):
            replacement = TTBlockfloatEmbedding(
                num_embeddings=int(module.num_embeddings),
                embedding_dim=int(module.embedding_dim),
                bits=bits,
                block=block,
                name=prefix,
                # Follow the precision the model was built at, so the table does
                # not upcast the rest of the forward to float32.
                dtype=module.weight.dtype,
            )
        elif isinstance(module, nn.Linear):
            replacement = TTBlockfloatLinear(
                bits=bits,
                in_features=int(module.in_features),
                out_features=int(module.out_features),
                bias=module.bias is not None,
                backend=BACKEND.TT_BLOCKFLOAT,
                adapter=None,
                name=prefix,
                # Follow the precision the model was built at; the source bias
                # is dense state and must not be narrowed on the way in.
                dtype=module.weight.dtype,
                block=block,
            )
        else:
            continue

        if parent_name:
            setattr(model.get_submodule(parent_name), attr, replacement)
        else:
            setattr(model, attr, replacement)
        restored[prefix] = [int(replacement.out_features), int(replacement.in_features)]

    # A tied head must follow the embedding it shares, or it keeps a stale dense
    # copy of the table.
    head = getattr(model, "lm_head", None)
    embed_prefix = next(
        (prefix for prefix in restored if prefix.endswith("embed_tokens")), None
    )
    if head is not None and embed_prefix is not None:
        if getattr(model.config, "tie_word_embeddings", False):
            embedding = model.get_submodule(embed_prefix)
            model.lm_head = TTBlockfloatTiedHead(embedding, name="lm_head")

    return restored


__all__ = [
    "BLOCKFLOAT_BLOCK",
    "TTBlockfloatEmbedding",
    "TTBlockfloatLinear",
    "TTBlockfloatTiedHead",
    "packed_code_bytes",
    "swap_blockfloat_modules",
]
