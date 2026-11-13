# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Native TTNN dequant+matmul kernel for packed GPTQ weights.

``XLALinear`` reaches Tenstorrent through Torch/XLA.  This kernel bypasses
StableHLO and PJRT entirely: it owns a ``ttnn.MeshDevice`` and expresses the
whole GPTQ contract as TTNN ops, so nibble extraction, zero-point correction,
per-group scaling and the GEMM all execute on the Tensix cores.

Layout uploaded per layer (packed only, never dense weights):

- ``weight_words`` ``[out_features, in_features/8]`` uint32, transposed from the
  checkpoint so one output row is contiguous.
- ``zero_words`` ``[groups, out_features/8]`` uint32 (packed zero points).
- ``scales`` ``[groups, out_features]`` BF16 and ``g_idx`` ``[in_features]``.

Forward extracts the eight nibbles of each word with native uint32
``logical_right_shift``/``bitwise_and`` — bit-exact on Blackhole, unlike TT's
32-bit signed shift path — casts them to BF16, expands ``g_idx`` through
``ttnn.embedding`` on the scale and zero tables, and calls ``ttnn.linear``.

The GEMM uses ``MathFidelity.HiFi4`` with ``fp32_dest_acc_en``.  TT-Metal
defaults to ``LoFi`` with ``math_approx_mode``, which measured a 0.0456 maximum
absolute error on real OPT q_proj data; HiFi4 with FP32 accumulation measured
0.0093 — the same floor the dense FP32 Torch oracle itself shows — for no
measurable latency cost on this decode/prefill shape.

Tensor parallelism shards ``out_features`` across the mesh.  Each chip owns a
contiguous output slice with its matching zero-point columns and scale columns,
so every chip performs an independent, already-correct GEMM; ``all_gather``
reassembles the output features.  That path requires ``FABRIC_1D``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from ...adapter.adapter import Adapter
from ...models._const import DEVICE, PLATFORM
from ...nn_modules.qlinear import GPTQQuantLinear
from ...quantization import FORMAT, METHOD
from ...utils.backend import BACKEND
from ...utils.tt_packing import TT_GPTQ_PACK_FACTOR, _repack_zero_words, _require_4bit_gptq
from ...utils.ttnn_runtime import (
    TTNNMesh,
    TTNNUnavailable,
    close_mesh,
    current_mesh,
    import_ttnn,
    is_ttnn_available,
    open_mesh,
    ttnn_device_count,
)
from ._g_idx import validate_g_idx_bounds


# Nibble lanes inside one int32/uint32 packed GPTQ word, low nibble first.
_NIBBLE_SHIFTS = tuple(4 * lane for lane in range(TT_GPTQ_PACK_FACTOR))


@dataclass(frozen=True)
class TTNNPackedLayer:
    """Device-resident packed tensors for one GPTQ layer."""

    weight_words: object
    zero_words: object
    scales: object
    g_idx: object
    bias: Optional[object]
    in_features: int
    out_features: int
    shard_out_features: int
    device_count: int
    compute_config: object


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _ttnn_arch(ttnn):
    """Return the ``ttnn.Arch`` for this host instead of assuming one chip.

    ``ttnn`` exposes the architecture as a lowercase name (``"blackhole"``,
    ``"wormhole_b0"``); the compute-kernel config needs the enum.  Falling back
    to Blackhole would silently apply the wrong architecture's math rules on a
    Wormhole host, so an unknown name is an error.
    """

    name = str(ttnn.get_arch_name()).upper()
    arch = getattr(ttnn.Arch, name, None)
    if arch is None:
        raise TTNNUnavailable(
            f"TTNN reported architecture {ttnn.get_arch_name()!r}, which has no ttnn.Arch member"
        )
    return arch


class TTNNLinear(GPTQQuantLinear):
    """Packed 4-bit GPTQ linear executed by native TTNN ops on Tenstorrent."""

    SUPPORTS_BACKENDS = [BACKEND.GPTQ_TTNN]
    SUPPORTS_METHODS = [METHOD.GPTQ]
    SUPPORTS_FORMATS = {FORMAT.GPTQ: 15, FORMAT.GPTQ_V2: 15}
    SUPPORTS_BITS = [4]
    SUPPORTS_GROUP_SIZE = [-1, 16, 32, 64, 128, 256, 512, 1024]
    SUPPORTS_DESC_ACT = [True, False]
    SUPPORTS_SYM = [True]
    SUPPORTS_SHARDS = True
    SUPPORTS_TRAINING = False
    SUPPORTS_AUTO_PADDING = False
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [8]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [8]

    SUPPORTS_DEVICES = [DEVICE.XLA]
    SUPPORTS_PLATFORM = [PLATFORM.LINUX]
    SUPPORTS_PACK_DTYPES = [torch.int32]
    SUPPORTS_ADAPTERS = []
    SUPPORTS_DTYPES = [torch.bfloat16]

    REQUIRES_FORMAT_V2 = False
    QUANT_TYPE = "ttnn"

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
        adapter: Optional[Adapter] = None,
        register_buffers: bool = True,
        format: Optional[FORMAT] = None,
        **kwargs,
    ):
        if adapter is not None:
            raise NotImplementedError("TTNNLinear does not support adapters yet.")
        super().__init__(
            bits=bits,
            group_size=group_size,
            sym=sym,
            desc_act=desc_act,
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            pack_dtype=pack_dtype,
            backend=kwargs.pop("backend", BACKEND.GPTQ_TTNN),
            adapter=None,
            register_buffers=register_buffers,
            format=format,
            **kwargs,
        )
        self._ttnn_layer: Optional[TTNNPackedLayer] = None
        self._ttnn_key = None

    # -- capability -------------------------------------------------------

    @classmethod
    def validate_device(cls, device):
        super().validate_device(device)

    @classmethod
    def validate_once(cls):
        if not is_ttnn_available():
            return False, TTNNUnavailable(
                "TTNNLinear requires the native Tenstorrent TTNN package."
            )
        try:
            count = ttnn_device_count()
        except Exception as exc:
            error = TTNNUnavailable("TTNNLinear could not enumerate Tenstorrent devices.")
            error.__cause__ = exc
            return False, error
        if count < 1:
            return False, TTNNUnavailable("no Tenstorrent devices are visible to TTNN.")
        return True, None

    @classmethod
    def _validate(cls, **args):
        ok, err = super()._validate(**args)
        if not ok:
            return ok, err
        in_features = args.get("in_features")
        out_features = args.get("out_features")
        # The uint32 nibble ABI consumes whole 8-code words, and the packed
        # zero table indexes whole 8-code output words too.
        for name, dim in (("in_features", in_features), ("out_features", out_features)):
            if dim is not None and dim % 8:
                return False, NotImplementedError(
                    f"{cls.__name__} requires `{name}` divisible by 8: actual {name} = `{dim}`."
                )
        return True, None

    # -- placement --------------------------------------------------------

    def _mesh(self) -> TTNNMesh:
        """Return the mesh this layer is placed on, opening it if needed."""

        mesh = current_mesh()
        if mesh is not None:
            return mesh
        count = self._requested_device_count()
        return open_mesh(device_count=count, fabric=self._fabric_config())

    def _requested_device_count(self) -> int:
        requested = getattr(self, "ttnn_device_count", None)
        if requested is None:
            return 1
        return int(requested)

    def _fabric_config(self) -> str:
        # Collectives are only used when more than one chip participates.
        return "DISABLED" if self._requested_device_count() == 1 else "FABRIC_1D"

    # -- packed layout ----------------------------------------------------

    def _layout_key(self):
        from ._runtime_cache import tensor_cache_key

        tensors = (self.qweight, self.qzeros, self.scales, self.g_idx, self.bias)
        return (
            # ``tensor_cache_key`` tolerates a missing bias and inference tensors,
            # which do not expose a version counter. Reading ``._version``
            # directly crashed for both: an absent bias is ``None``, and a
            # checkpoint packed under ``InferenceMode`` raises ``RuntimeError``.
            tuple(tensor_cache_key(t) for t in tensors),
            self.qzero_format(),
            self._requested_device_count(),
        )

    def _build_layer(self) -> TTNNPackedLayer:
        key = self._layout_key()
        if self._ttnn_layer is not None and self._ttnn_key == key:
            return self._ttnn_layer

        ttnn = import_ttnn()
        _require_4bit_gptq(
            self.bits, self.pack_dtype, self.in_features, self.out_features,
            self.sym, self.planar,
        )
        groups = int(self.scales.shape[0])
        g_idx = self.g_idx.detach().to(device="cpu", dtype=torch.int64)
        validate_g_idx_bounds(g_idx, self.scales, layer_name=type(self).__name__)
        if g_idx.numel() != self.in_features:
            raise ValueError(f"g_idx length must equal {self.in_features}, got {g_idx.numel()}.")
        if int(g_idx.min()) < 0:
            g_idx = g_idx % groups

        # Packed-only host staging: view the signed checkpoint words as exact
        # bit patterns and transpose to output-major while still packed.
        weight_words = (
            self.qweight.detach().to(device="cpu", copy=True)
            .view(torch.uint32)[: _ceil_div(self.in_features, TT_GPTQ_PACK_FACTOR), : self.out_features]
            .transpose(0, 1).contiguous()
        )
        zero_words, _zero_neg_one = _repack_zero_words(
            self.qzeros, groups=groups, out_features=self.out_features,
            qzero_format=self.qzero_format(),
        )

        mesh = self._mesh()
        dev = mesh.device
        device_count = mesh.num_devices

        # Tensor parallel: every chip owns a contiguous slice of out_features.
        if self.out_features % device_count:
            raise NotImplementedError(
                f"out_features={self.out_features} must be divisible by the "
                f"{device_count} TTNN devices in the mesh"
            )
        shard_out = self.out_features // device_count
        out_words_per_device = _ceil_div(shard_out, TT_GPTQ_PACK_FACTOR)
        zero_words_per_group = _ceil_div(self.out_features, TT_GPTQ_PACK_FACTOR)

        def _upload(tensor, dtype, mesh_mapper=None):
            host = ttnn.from_torch(
                tensor,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=mesh_mapper,
            )
            return ttnn.to_device(host, dev)

        if device_count == 1:
            weight = _upload(weight_words, ttnn.uint32)
            zeros = _upload(zero_words, ttnn.uint32)
            scales = _upload(self.scales.detach().to("cpu", torch.bfloat16).contiguous(), ttnn.bfloat16)
            indices = _upload(
                g_idx.to(torch.uint32).reshape(1, 1, 1, -1).contiguous(), ttnn.uint32
            )
            bias = None
            if self.bias is not None:
                bias = _upload(
                    self.bias.detach().to("cpu", torch.bfloat16).reshape(1, -1).contiguous(),
                    ttnn.bfloat16,
                )
        else:
            shard_rows = ttnn.shard_tensor_to_mesh_mapper(dev, 0)
            shard_cols = ttnn.shard_tensor_to_mesh_mapper(dev, 1)
            replicate = ttnn.replicate_tensor_to_mesh_mapper(dev)
            weight = _upload(weight_words, ttnn.uint32, shard_rows)
            zeros = _upload(zero_words, ttnn.uint32, shard_cols)
            scales = _upload(
                self.scales.detach().to("cpu", torch.bfloat16).contiguous(),
                ttnn.bfloat16, shard_cols,
            )
            indices = _upload(
                g_idx.to(torch.uint32).reshape(1, 1, 1, -1).contiguous(), ttnn.uint32, replicate
            )
            bias = None
            if self.bias is not None:
                bias = _upload(
                    self.bias.detach().to("cpu", torch.bfloat16).reshape(1, -1).contiguous(),
                    ttnn.bfloat16, shard_cols,
                )

        # Validate the sharded packed shapes once, here, so a checkpoint whose
        # layout cannot serve this mesh fails at load rather than mid-forward.
        if device_count > 1:
            if tuple(weight.shape) != (shard_out, _ceil_div(self.in_features, TT_GPTQ_PACK_FACTOR)):
                raise ValueError(
                    f"sharded weight shape {tuple(weight.shape)} does not match "
                    f"{(shard_out, _ceil_div(self.in_features, TT_GPTQ_PACK_FACTOR))}"
                )
            if tuple(zeros.shape)[1] != out_words_per_device:
                raise ValueError(
                    f"sharded zero shape {tuple(zeros.shape)} does not match "
                    f"{out_words_per_device} packed words per group"
                )
        elif zero_words.shape[1] != zero_words_per_group:
            raise ValueError(
                f"zero words per group {zero_words.shape[1]} does not match "
                f"{zero_words_per_group}"
            )

        compute_config = ttnn.init_device_compute_kernel_config(
            _ttnn_arch(ttnn),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
        )

        self._ttnn_layer = TTNNPackedLayer(
            weight_words=weight,
            zero_words=zeros,
            scales=scales,
            g_idx=indices,
            bias=bias,
            in_features=self.in_features,
            out_features=self.out_features,
            shard_out_features=shard_out,
            device_count=device_count,
            compute_config=compute_config,
        )
        self._ttnn_key = key
        return self._ttnn_layer

    # -- lifecycle --------------------------------------------------------

    def post_init(self):
        self._build_layer()

    def clear_weight_cache(self) -> None:
        layer, self._ttnn_layer = self._ttnn_layer, None
        self._ttnn_key = None
        if layer is None:
            return
        ttnn = import_ttnn()
        for tensor in (
            layer.weight_words, layer.zero_words, layer.scales,
            layer.g_idx, layer.bias,
        ):
            if tensor is not None:
                try:
                    ttnn.deallocate(tensor)
                except Exception:  # pragma: no cover - best-effort release
                    pass

    def release_device(self) -> None:
        """Close the mesh this layer opened, if it still owns it."""

        self.clear_weight_cache()
        mesh = current_mesh()
        if mesh is not None:
            close_mesh(owner=mesh.owner)

    def pack(self, linear, scales, zeros, g_idx, **kwargs):
        self.clear_weight_cache()
        super().pack(linear, scales, zeros, g_idx, **kwargs)
        self.clear_weight_cache()

    # -- execution --------------------------------------------------------

    def _nibble_planes(self, ttnn, words):
        """Return the eight 4-bit planes of packed uint32 words, still integer.

        Planes are stacked on dim 0, not on the trailing dim. Measured on
        Blackhole p300c for ``[768, 96]`` words: stacking the eight planes on
        the trailing tile dimension costs 3.35 ms and a 4-way stack on it costs
        14.5 ms, because that dimension is not tile-aligned. Stacking on dim 0
        costs 0.03 ms and the following ``permute`` 0.13 ms, for the same bits.
        """

        planes = []
        for shift in _NIBBLE_SHIFTS:
            shifted = words if shift == 0 else ttnn.logical_right_shift(words, shift)
            planes.append(ttnn.bitwise_and(shifted, 0xF))
        return ttnn.stack(planes, 0)

    def _extract_codes(self, ttnn, words, rows):
        """Expand packed uint32 words into integer codes ``[rows, in]``."""

        planes = self._nibble_planes(ttnn, words)
        return ttnn.reshape(ttnn.permute(planes, (1, 2, 0)), (rows, self.in_features))

    def _logical_zeros(self, ttnn, words, groups, shard_out):
        """Expand packed zero points to logical BF16 codes ``[groups, shard_out]``.

        ``_repack_zero_words`` already converts GPTQ v1's serialised
        ``zero_point - 1`` into logical space while the codes are still packed
        (so the per-nibble carry cannot bleed into neighbouring fields).  The
        uploaded words are logical, and applying the ``+1`` again here would
        shift every zero point one step and halve the reconstructed weight.
        """

        packed_width = _ceil_div(shard_out, TT_GPTQ_PACK_FACTOR) * TT_GPTQ_PACK_FACTOR
        planes = self._nibble_planes(ttnn, words)
        stacked = ttnn.reshape(ttnn.permute(planes, (1, 2, 0)), (groups, packed_width))
        if packed_width != shard_out:
            stacked = stacked[:, :shard_out]
        return ttnn.typecast(stacked, ttnn.bfloat16)

    def _dequantize(self, ttnn, layer: TTNNPackedLayer):
        """Build the transient BF16 weight tile ``[in, shard_out]`` on device."""

        codes = ttnn.typecast(
            self._extract_codes(ttnn, layer.weight_words, layer.shard_out_features), ttnn.bfloat16
        )
        codes = ttnn.transpose(codes, 0, 1)
        zeros = self._logical_zeros(
            ttnn, layer.zero_words, int(self.scales.shape[0]), layer.shard_out_features
        )
        # Advance g_idx per input row so scale/zero stay aligned with codes.
        group_scales = ttnn.embedding(layer.g_idx, layer.scales, layout=ttnn.TILE_LAYOUT)
        group_zeros = ttnn.embedding(layer.g_idx, zeros, layout=ttnn.TILE_LAYOUT)
        group_scales = ttnn.reshape(group_scales, (self.in_features, layer.shard_out_features))
        group_zeros = ttnn.reshape(group_zeros, (self.in_features, layer.shard_out_features))
        weights = ttnn.multiply(ttnn.subtract(codes, group_zeros), group_scales)
        return ttnn.typecast(weights, ttnn.bfloat16)

    def _compute_on_device(self, ttnn, layer: TTNNPackedLayer, activation):
        """Run dequant + GEMM (+ all_gather) on a resident activation.

        Returns the device output. The transient dequantized weight tile is
        released here; the caller owns ``activation`` and the returned tensor.
        """

        weights = self._dequantize(ttnn, layer)
        try:
            output = ttnn.linear(
                activation, weights, bias=layer.bias,
                compute_kernel_config=layer.compute_config,
            )
        finally:
            ttnn.deallocate(weights)

        if layer.device_count > 1:
            # all_gather returns a new tensor; release the per-chip partial so
            # only one output allocation is live while it is consumed.
            gathered = ttnn.all_gather(output, 1)
            ttnn.deallocate(output)
            output = gathered
        return output

    def forward_resident(self, x):
        """Forward on an activation that is already resident on the device.

        Returns another resident activation, so a chain of these layers moves
        no activation bytes over PCIe between them. See
        :mod:`gptqmodel.utils.tt_resident`.
        """

        from ...utils.tt_resident import TTDeviceTensor, is_device_tensor

        if self.training:
            raise NotImplementedError("TTNNLinear is inference-only.")
        if not is_device_tensor(x):
            raise TypeError(
                "forward_resident expects a TTDeviceTensor from "
                "gptqmodel.utils.tt_resident; use forward() for host activations."
            )
        if x.features != self.in_features:
            raise ValueError(f"expected input width {self.in_features}, got {x.features}.")

        ttnn = import_ttnn()
        layer = self._build_layer()
        mesh = x.mesh if x.mesh is not None else current_mesh()
        if mesh is None:
            raise RuntimeError("TTNNLinear has no open device; call post_init() first.")

        output = self._compute_on_device(ttnn, layer, x.tensor)
        return TTDeviceTensor(
            tensor=output,
            mesh=mesh,
            batch_shape=tuple(x.batch_shape),
            features=self.out_features,
        )

    def forward(self, x: torch.Tensor):
        from ...utils.tt_resident import is_device_tensor

        if is_device_tensor(x):
            # Resident chaining: skip both the upload and the readback.
            return self.forward_resident(x)

        if self.training:
            raise NotImplementedError("TTNNLinear is inference-only.")
        if x.device.type != "cpu":
            raise RuntimeError(
                "TTNNLinear accepts CPU activation tensors and moves them to the "
                f"TTNN device itself, got {x.device}."
            )
        if x.dtype != torch.bfloat16:
            raise NotImplementedError(f"TTNNLinear requires BF16 activations, got {x.dtype}.")
        if x.shape[-1] != self.in_features:
            raise ValueError(f"expected input width {self.in_features}, got {x.shape[-1]}.")

        ttnn = import_ttnn()
        layer = self._build_layer()
        mesh = current_mesh()
        if mesh is None:
            raise RuntimeError("TTNNLinear has no open device; call post_init() first.")
        dev = mesh.device

        out_shape = x.shape[:-1] + (self.out_features,)
        flat = x.reshape(-1, self.in_features).contiguous()

        if layer.device_count == 1:
            activation = ttnn.to_device(
                ttnn.from_torch(flat, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), dev
            )
        else:
            activation = ttnn.to_device(
                ttnn.from_torch(
                    flat, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                    mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(dev),
                ),
                dev,
            )

        output = self._compute_on_device(ttnn, layer, activation)
        try:
            host = self._to_host(ttnn, dev, output, layer)
        finally:
            # The device activation must not outlive this call, including when
            # the readback raises.
            ttnn.deallocate(activation)
            ttnn.deallocate(output)
        return host.reshape(out_shape)

    def _to_host(self, ttnn, dev, output, layer: TTNNPackedLayer) -> torch.Tensor:
        """Read a device tensor back as BF16 CPU activations."""

        if layer.device_count == 1:
            return ttnn.to_torch(output)
        composed = ttnn.to_torch(output, mesh_composer=ttnn.ConcatMeshToTensor(dev, dim=0))
        # Replicated shards concatenate on dim 0; every copy already holds the
        # fully gathered output features after all_gather.
        return composed[: composed.shape[0] // layer.device_count]

    def dequantize_weight(self, num_itr: int = 1):
        """Explicit host-side export of the packed weights (debug only)."""

        if num_itr != 1:
            raise NotImplementedError("TTNNLinear supports only the single-group layout.")
        ttnn = import_ttnn()
        layer = self._build_layer()
        mesh = current_mesh()
        weights = self._dequantize(ttnn, layer)
        if layer.device_count == 1:
            host = ttnn.to_torch(weights)
        else:
            composed = ttnn.to_torch(
                weights, mesh_composer=ttnn.ConcatMeshToTensor(mesh.device, dim=0)
            )
            # Weight shards are disjoint along out_features, so the replicated
            # composer stacks each chip's [in, shard_out] block.
            blocks = [
                composed[i * self.in_features: (i + 1) * self.in_features]
                for i in range(layer.device_count)
            ]
            host = torch.cat(blocks, dim=1)
        ttnn.deallocate(weights)
        return host


__all__ = ["TTNNLinear", "TTNNPackedLayer"]
