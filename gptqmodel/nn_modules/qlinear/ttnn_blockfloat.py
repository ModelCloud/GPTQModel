# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""GPTQ linear stored in a Tenstorrent block-float format.

A packed int4 checkpoint has to be expanded before the matrix engine can use it,
and the TTNN op chain that does so costs about 85 % of a layer's forward. The
Blackhole unpack stage can expand a **storage** format for free in the data path
(``L1 -> unpack -> source registers -> FPU``), so re-encoding the dequantized
weights once into ``bf8_b`` or ``bf4_b`` moves that expansion into hardware and
leaves the hot path as a single ``ttnn.linear``.

Measured on real OPT-125M ``q_proj`` weights (``768x768``, batch 8):

===========  =========  ============  ==========  =========
storage      latency    max abs err   norm L1     bytes/el
===========  =========  ============  ==========  =========
bfloat16     0.101 ms   reference     -           2
bf8_b        0.071 ms   0.125         1.0 %       ~1.06
bf4_b        0.072 ms   0.914         9.8 %       ~0.56
===========  =========  ============  ==========  =========

Scope, stated precisely: this is a **storage re-encoding of an existing int4
checkpoint**, not "GPTQ for block float". GPTQ is uniform int4 with a per-group
scale and a zero point; ``bf8_b`` is symmetric with one shared exponent per 16
values, so the zero point cannot be represented and is folded into the stored
value. At 8 bits the resulting loss is small (~1 %), which is why this mode
exists at all. It costs ~2x the memory of the packed checkpoint, so it is opt-in.

``bf4_b`` is deliberately **not** offered here. Re-encoding an already 4-bit
weight into a 3-bit-mantissa block-float grid measured **9.8 %** normalized error
on a real layer -- that is degradation, not optimization. Making 4-bit
block-float viable would mean *re-quantizing the source model with GPTQ onto a
block-float grid* (a quantization-time change producing a new checkpoint
format), not converting an int4 checkpoint after the fact.

For exact inference of existing int4 checkpoints the correct path is the fused
int4 kernel: keep the checkpoint packed and expand nibbles per tile inside the
compute kernel. See ``hw/tenstorrent.md`` section 9.
"""

from __future__ import annotations

from typing import Optional

import torch

from ...adapter.adapter import Adapter
from ...models._const import DEVICE, PLATFORM
from ...nn_modules.qlinear import GPTQQuantLinear
from ...quantization import FORMAT, METHOD
from ...utils.backend import BACKEND
from ...utils.logger import setup_logger
from ...utils.resource_budget import DeviceAllocation, RuntimeAllocationUnavailable
from ...utils.tt_packing import dequantize_tt_packed_4bit, repack_gptq_for_tt
from ...utils.ttnn_runtime import (
    TTNNMesh,
    TTNNUnavailable,
    close_mesh,
    current_mesh,
    dram_allocated_bytes,
    hold_mesh_lease,
    import_ttnn,
    mesh_device_ids,
    open_mesh,
    release_mesh_lease,
    to_host_replica,
)
from ._compile_support import COMPILE_LEVEL
from ._g_idx import validate_g_idx_bounds
from .ttnn import _ttnn_arch


log = setup_logger()

#: Canonical block-float storage formats this kernel can materialise.
BLOCK_FLOAT_FORMATS = ("bf8_b",)


class _TTNNBlockFloatMixin:
    """Shared implementation; concrete kernels differ only in stored format."""

    #: ``ttnn`` attribute name for the storage format, set by each concrete
    #: kernel. This is the vendor spelling because it is passed straight to
    #: ``getattr(ttnn, ...)``; the canonical name for this format is ``bf8_b``
    #: (``bf4_b`` for the 4-bit one).
    WEIGHT_FORMAT = "bfloat8_b"
    #: Bits per stored element, used for the memory argument.
    WEIGHT_BITS = 8

    # -- placement --------------------------------------------------------

    def _mesh(self) -> TTNNMesh:
        # Routed through ``open_mesh`` even when a handle is live: the runtime is
        # what knows whether the open placement satisfies this request.
        return open_mesh(
            device_count=self._requested_device_count(),
            fabric=self._fabric_config(),
            device_id=self._requested_device_id(),
        )

    def _requested_device_count(self) -> int:
        requested = getattr(self, "ttnn_device_count", None)
        return 1 if requested is None else int(requested)

    def _requested_device_id(self) -> int:
        # Resolved by the loader from an explicit `xla:N` request; `open_mesh`
        # still rejects an ordinal combined with a multi-chip mesh.
        requested = getattr(self, "ttnn_device_id", None)
        return 0 if requested is None else int(requested)

    def _fabric_config(self) -> str:
        return "DISABLED" if self._requested_device_count() == 1 else "FABRIC_1D"

    # -- conversion -------------------------------------------------------

    def _host_weights(self) -> torch.Tensor:
        """Return dequantized GPTQ weights as ``[in_features, out_features]`` float32.

        Reuses the shared CPU oracle so this kernel and :class:`TTNNLinear`
        cannot drift apart on GPTQ semantics (v1 zero-point, ``g_idx`` order).
        """

        g_idx = self.g_idx.detach().to(device="cpu", dtype=torch.int64)
        validate_g_idx_bounds(g_idx, self.scales, layer_name=type(self).__name__)
        if int(g_idx.min()) < 0:
            g_idx = g_idx % int(self.scales.shape[0])

        layout = repack_gptq_for_tt(
            qweight=self.qweight,
            qzeros=self.qzeros,
            scales=self.scales,
            g_idx=g_idx,
            bias=self.bias,
            in_features=self.in_features,
            out_features=self.out_features,
            group_size=self.group_size,
            bits=self.bits,
            pack_dtype=self.pack_dtype,
            sym=self.sym,
            planar=self.planar,
            qzero_format=self.qzero_format(),
            device=torch.device("cpu"),
        )
        return dequantize_tt_packed_4bit(layout)

    def _build_device_weight(self):
        """Dequantize once on the host and upload in the chosen block-float format."""

        ttnn = import_ttnn()
        device_dtype = getattr(ttnn, self.WEIGHT_FORMAT)
        weights = self._host_weights().to(torch.bfloat16).contiguous()

        mesh = self._mesh()
        # Lease the shared handle for this layer; the model wrapper releases
        # only devices its own kernels hold.
        hold_mesh_lease(self)
        dev = mesh.device
        count = mesh.num_devices
        if self.out_features % count:
            raise NotImplementedError(
                f"out_features={self.out_features} must be divisible by the "
                f"{count} TTNN devices in the mesh"
            )

        # Measure the allocator before the upload so the budget can report the
        # exact per-chip storage this weight adds, including tile padding and
        # page rounding. An unreadable view is reported as unknown, not zero.
        try:
            allocated_before = dram_allocated_bytes(dev)
        except Exception:
            allocated_before = None

        host = ttnn.from_torch(
            weights,
            dtype=device_dtype,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=(
                None if count == 1 else ttnn.shard_tensor_to_mesh_mapper(dev, 1)
            ),
        )
        weight = ttnn.to_device(host, dev)

        bias = None
        if self.bias is not None:
            bias = ttnn.to_device(
                ttnn.from_torch(
                    self.bias.detach().to("cpu", torch.bfloat16).reshape(1, -1).contiguous(),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    mesh_mapper=(
                        None if count == 1 else ttnn.shard_tensor_to_mesh_mapper(dev, 1)
                    ),
                ),
                dev,
            )

        device_bytes = None
        if allocated_before is not None:
            try:
                allocated_after = dram_allocated_bytes(dev)
            except Exception:
                allocated_after = None
            if allocated_after is not None and allocated_after > allocated_before:
                device_bytes = allocated_after - allocated_before
        try:
            device_ids = mesh_device_ids(mesh)
        except Exception:
            device_ids = ()

        old_weight = getattr(self, "_ttnn_weight", None)
        old_bias = getattr(self, "_ttnn_bias", None)
        self._ttnn_weight = weight
        self._ttnn_bias = bias
        self._ttnn_devices = count
        self._ttnn_device_ids = device_ids
        self._ttnn_device_bytes = device_bytes
        self._ttnn_key = self._weight_key(mesh)
        # Publish first, then release the replaced tensors: a rebuild that fails
        # above must leave the previous, still-valid weight in place.
        for tensor in (old_weight, old_bias):
            if tensor is None:
                continue
            try:
                ttnn.deallocate(tensor)
            except Exception as exc:  # pragma: no cover - best-effort release
                log.debug(f"TTNN: block-float weight deallocate failed: {exc}")
        return weight

    def _compute_config(self, ttnn):
        """Return the HiFi4 / FP32-destination compute config for this layer.

        TT-Metal defaults to ``LoFi`` with ``math_approx_mode``. On real OPT
        ``q_proj`` data that measured 0.0456 maximum absolute error, while the
        same GEMM with HiFi4 and FP32 accumulation measured 0.0093 -- the floor
        the dense FP32 Torch oracle itself shows. ``TTNNLinear`` already uses
        this configuration; the block-float path must match it or it drops
        finite cancellation residuals that the exact result keeps.
        """

        config = getattr(self, "_ttnn_compute_config", None)
        if config is None:
            config = ttnn.init_device_compute_kernel_config(
                _ttnn_arch(ttnn),
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
            )
            self._ttnn_compute_config = config
        return config
    def _weight_key(self, mesh):
        """Identity of every input that the packed BF8 weight is built from.

        ``tensor_cache_key`` catches tensor replacement and in-place mutation;
        ``generation`` catches a closed and reopened mesh. Inference tensors do
        not expose a version counter, so for those the key falls back to storage
        identity and the explicit ``clear_weight_cache()`` contract still
        applies.
        """

        from ._runtime_cache import tensor_cache_key

        tensors = (self.qweight, self.qzeros, self.scales, self.g_idx, self.bias)
        return (
            tuple(tensor_cache_key(tensor) for tensor in tensors),
            self.qzero_format(),
            self.WEIGHT_FORMAT,
            mesh.generation,
            mesh.num_devices,
            mesh.device_id,
        )

    def _device_weight(self):
        weight = getattr(self, "_ttnn_weight", None)
        mesh = current_mesh()
        if (
            weight is not None
            and mesh is not None
            and getattr(self, "_ttnn_key", None) == self._weight_key(mesh)
        ):
            return weight
        return self._build_device_weight()

    # -- lifecycle --------------------------------------------------------

    def post_init(self):
        self._device_weight()

    def clear_weight_cache(self) -> None:
        weight, self._ttnn_weight = getattr(self, "_ttnn_weight", None), None
        bias, self._ttnn_bias = getattr(self, "_ttnn_bias", None), None
        self._ttnn_compute_config = None
        self._ttnn_key = None
        self._ttnn_device_ids = ()
        self._ttnn_device_bytes = None
        if weight is None and bias is None:
            return
        ttnn = import_ttnn()
        for tensor in (weight, bias):
            if tensor is not None:
                try:
                    ttnn.deallocate(tensor)
                except Exception as exc:  # pragma: no cover - best-effort release
                    # The reference was already dropped above, so a failed
                    # deallocate must not turn a cache clear into an error: the
                    # caller is mid-lifecycle and the device teardown that
                    # follows will reclaim it. Record it instead of hiding it.
                    log.debug(f"TTNN: block-float weight deallocate failed: {exc}")

    def iter_runtime_device_allocations(self):
        """Report the block-float device tensors this layer owns (#5810).

        The bytes are the per-chip DRAM delta measured when the weight was
        uploaded, so tile padding, allocator page rounding and replicated
        operands are included exactly as the device holds them. A layer that
        was never built owns no device storage; a built layer whose allocator
        could not be read reports its backend as unmeasured instead of zero.
        """

        if getattr(self, "_ttnn_weight", None) is None:
            return
        if self._ttnn_device_bytes is None or not self._ttnn_device_ids:
            raise RuntimeAllocationUnavailable("ttnn")
        for index in self._ttnn_device_ids:
            yield DeviceAllocation(
                backend="ttnn", index=index, weights_bytes=self._ttnn_device_bytes
            )

    def release_device(self) -> bool:
        """Drop this layer's mesh lease, closing the mesh only when last."""

        self.clear_weight_cache()
        if not release_mesh_lease(self):
            return False
        mesh = current_mesh()
        if mesh is not None:
            close_mesh(owner=mesh.owner)
        return True

    def pack(self, linear, scales, zeros, g_idx, **kwargs):
        self.clear_weight_cache()
        super().pack(linear, scales, zeros, g_idx, **kwargs)
        self.clear_weight_cache()

    # -- execution --------------------------------------------------------

    def _compute(self, ttnn, mesh, activation):
        weight = self._device_weight()
        bias = getattr(self, "_ttnn_bias", None)
        output = ttnn.linear(
            activation,
            weight,
            bias=bias,
            compute_kernel_config=self._compute_config(ttnn),
        )
        if self._ttnn_devices > 1:
            gathered = ttnn.all_gather(output, 1)
            ttnn.deallocate(output)
            output = gathered
        return output

    def forward_resident(self, x):
        """Forward on a resident activation, returning a resident activation."""

        from ...utils.tt_resident import TTDeviceTensor, is_device_tensor

        if self.training:
            raise NotImplementedError(f"{type(self).__name__} is inference-only.")
        if not is_device_tensor(x):
            raise TypeError(
                "forward_resident expects a TTDeviceTensor from "
                "gptqmodel.utils.tt_resident; use forward() for host activations."
            )
        if x.features != self.in_features:
            raise ValueError(f"expected input width {self.in_features}, got {x.features}.")

        ttnn = import_ttnn()
        mesh = x.mesh if x.mesh is not None else current_mesh()
        if mesh is None:
            raise RuntimeError(f"{type(self).__name__} has no open device; call post_init() first.")
        output = self._compute(ttnn, mesh, x.tensor)
        return TTDeviceTensor(
            tensor=output,
            mesh=mesh,
            batch_shape=tuple(x.batch_shape),
            features=self.out_features,
        )

    def forward(self, x: torch.Tensor):
        from ...utils.tt_resident import is_device_tensor

        if is_device_tensor(x):
            return self.forward_resident(x)

        if self.training:
            raise NotImplementedError(f"{type(self).__name__} is inference-only.")
        if x.device.type != "cpu":
            raise RuntimeError(
                f"{type(self).__name__} accepts CPU activation tensors and moves them "
                f"to the TTNN device itself, got {x.device}."
            )
        if x.dtype != torch.bfloat16:
            raise NotImplementedError(f"{type(self).__name__} requires BF16 activations, got {x.dtype}.")
        if x.shape[-1] != self.in_features:
            raise ValueError(f"expected input width {self.in_features}, got {x.shape[-1]}.")

        ttnn = import_ttnn()
        mesh = self._mesh()
        dev = mesh.device
        flat = x.reshape(-1, self.in_features).contiguous()
        count = mesh.num_devices

        activation = ttnn.to_device(
            ttnn.from_torch(
                flat,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=(None if count == 1 else ttnn.replicate_tensor_to_mesh_mapper(dev)),
            ),
            dev,
        )
        output = self._compute(ttnn, mesh, activation)
        try:
            host = to_host_replica(ttnn, output, mesh)
        finally:
            ttnn.deallocate(activation)
            ttnn.deallocate(output)
        return host.reshape(x.shape[:-1] + (self.out_features,))

    def dequantize_weight(self, num_itr: int = 1):
        """Host export of the stored weights (debug only)."""

        return self._host_weights().to(torch.bfloat16)

    def storage_bytes(self) -> int:
        """Device bytes retained for the weights, for the memory argument."""

        return (self.in_features * self.out_features * self.WEIGHT_BITS + 7) // 8


class TTNNBlockFloat8Linear(_TTNNBlockFloatMixin, GPTQQuantLinear):
    """GPTQ linear stored as ``bf8_b`` (hardware-unpacked, ~1 % error)."""

    # TTNN device tensors; no torch graph
    SUPPORTS_COMPILE_LEVEL = COMPILE_LEVEL.NOT_APPLICABLE
    SUPPORTS_GRAPH_CAPTURE = False

    WEIGHT_FORMAT = "bfloat8_b"
    WEIGHT_BITS = 8

    SUPPORTS_BACKENDS = [BACKEND.GPTQ_TT_BF8]
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
    QUANT_TYPE = "bf8_b"

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
            raise NotImplementedError("TTNNBlockFloat8Linear does not support adapters.")
        super().__init__(
            bits=bits,
            group_size=group_size,
            sym=sym,
            desc_act=desc_act,
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            pack_dtype=pack_dtype,
            backend=kwargs.pop("backend", BACKEND.GPTQ_TT_BF8),
            adapter=None,
            register_buffers=register_buffers,
            format=format,
            **kwargs,
        )
        self._ttnn_weight = None
        self._ttnn_bias = None
        self._ttnn_devices = 1
        # Per-chip storage measured when the block-float weight was uploaded,
        # and the chip ordinals those bytes live on (#5810).
        self._ttnn_device_ids = ()
        self._ttnn_device_bytes: Optional[int] = None

    @classmethod
    def validate_device(cls, device):
        super().validate_device(device)
        target = device if isinstance(device, torch.device) else torch.device(device)
        if target.type != "xla":
            raise NotImplementedError(f"{cls.__name__} requires an XLA-family target, got {device!r}.")

    @classmethod
    def validate_once(cls):
        from ...utils.ttnn_runtime import is_ttnn_available, ttnn_device_count

        if not is_ttnn_available():
            return False, TTNNUnavailable("the native Tenstorrent TTNN runtime is not importable.")
        try:
            count = ttnn_device_count()
        except Exception as exc:
            error = TTNNUnavailable(f"{cls.__name__} could not enumerate Tenstorrent devices.")
            error.__cause__ = exc
            return False, error
        if count < 1:
            return False, TTNNUnavailable("no Tenstorrent devices are visible to TTNN.")
        return True, None


__all__ = [
    "BLOCK_FLOAT_FORMATS",
    "TTNNBlockFloat8Linear",
]
