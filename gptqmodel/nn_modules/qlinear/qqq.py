# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

# Adapted from vllm at https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/gptq_marlin.py

from functools import lru_cache
from typing import List, Optional, Tuple

import torch

from ...adapter.adapter import Adapter, Lora
from ...models._const import CPU, DEVICE, PLATFORM
from ...nn_modules.qlinear import GroupedQuantLinear
from ...quantization import FORMAT, METHOD
from ...utils.backend import BACKEND
from ...utils.logger import setup_logger
from ...utils.qqq import qqq_gemm, qqq_runtime_available, qqq_runtime_error
from ...utils.rocm import IS_ROCM
from ._compile_support import COMPILE_LEVEL
from ._runtime_cache import tensor_cache_key


log = setup_logger()


# INT32 accumulation bound shared by every CPU INT8 GEMM path.  The most
# negative INT8 product is (-128) * (-128) = 16384, so one INT32 accumulator may
# sum at most floor((2**31 - 1) / 16384) = 131071 terms before it wraps.
# ``torch._int_mm`` and the packed CPU kernel both accumulate in INT32, and
# neither raises on overflow, so each path has to reject or split a longer
# contraction explicitly instead of relying on an exception.
INT8_INT32_MAX_REDUCTION = (2**31 - 1) // (128 * 128)

# Long contractions are summed in INT64 over bounded INT8 GEMM chunks.  65536
# is a power-of-two multiple of the 64-wide kernel alignment and keeps every
# partial sum at or below 2**30.
INT8_INT32_CHUNK_REDUCTION = 1 << 16


@lru_cache(maxsize=1)
def _qqq_gpu_quantizer():
    try:
        from .qqq_triton import dynamic_quant
    except ImportError:
        return None
    return dynamic_quant


# ``qqq_gemm`` uses its ``C`` argument only as the INT32 global-reduce scratch
# of one call.  ``global_reduce`` writes each column slice before it reads that
# slice, so ``C`` carries no state across calls: the kernel produces
# bitwise-equal output from a zeroed buffer and from one filled with garbage
# (#4621).
#
# ``C`` is therefore allocated per call and returned to the caching allocator
# when the forward drops it.  Two earlier designs cached it, first per device
# (#4387) and then per device *and* stream (#4488), and both outlived the
# model: a thread pool that created its 32 pooled streams pinned 32 x 56 MiB on
# an 8B QQQ model for the life of the process, and ``del model`` released none
# of it (#4621).  A per-call allocation is stream-safe by construction: the
# allocator only reuses a freed block on the stream that freed it, and work on
# one stream is ordered by that stream.  On the 4096x14336 shapes it measured
# 0.0-0.3 us slower per call, where one kernel call is 13.9 us.


def _allocate_reduce_scratch(
    device: torch.device,
    out_features: int,
    max_par: int = 16,
) -> torch.Tensor:
    """Allocate the INT32 global-reduce scratch for one ``qqq_gemm`` call.

    The contiguous ``(max_par * 64, out_features)`` tensor is the kernel's ``C``
    argument.  Nothing in it survives the call, so the caller may drop it as
    soon as the kernel has been enqueued.
    """

    return torch.empty(
        (max_par * 64, out_features),
        dtype=torch.int32,
        device=torch.device(device),
    )

# The Marlin-derived QQQ W4A8 kernel (``gptqmodel_ext/qqq/qqq_gemm.cu``) is
# written against CUDA inline PTX: ``cp.async``, ``mma.sync``, ``ldmatrix``,
# ``__cvta_generic_to_shared``, ``cvt.rn.f16.f32`` and ``ld.global.acquire.gpu``.
# hipify rewrites the bindings but cannot translate that PTX, so on gfx1201
# (RDNA4) hipcc rejects the build (20 errors, first at
# ``gptqmodel_ext/qqq/qqq_gemm.hip:61``).  QQQ therefore has no native ROCm
# kernel, and AUTO must select the portable ``QQQTorchLinear`` instead.
QQQ_ROCM_UNSUPPORTED = (
    "QQQ (BACKEND.QQQ) has no ROCm kernel: the Marlin-derived W4A8 extension "
    "uses CUDA-only inline PTX and cannot be built by hipcc. "
    "Use BACKEND.QQQ_TORCH, which is selected automatically for AUTO on AMD GPUs."
)


def _fake_qqq_int4(input, weight, scales, group_size):
    """FakeTensor shape for the packed QQQ CPU kernel (see ``extension.register_fake``)."""

    return input.new_empty((input.shape[0], weight.shape[0]), dtype=torch.float32)


@lru_cache(maxsize=1)
def _qqq_cpu_kernel():
    from ...extension import op as extension_op
    from ...extension import register_fake
    from ...utils.cpu_features import has_avx512_bf16

    if not has_avx512_bf16():
        return None
    try:
        kernel = extension_op("int4_qzero_group_scaled_cpu", "qqq_int4")
        register_fake("int4_qzero_group_scaled_cpu", "qqq_int4", _fake_qqq_int4)
        return kernel
    except (RuntimeError, ImportError, OSError) as exc:
        log.warning("QQQ packed CPU kernel unavailable: %s", exc)
        return None


def _workspace_numel(out_features: int, max_par: int) -> int:
    """Number of int32 slots ``qqq_gemm.cu`` indexes in the zeroed ``workspace``.

    The kernel uses ``workspace`` as its ``locks`` array and indexes it by
    output tile (``locks[slice_col]``), one tile per ``min_thread_n`` = 64
    output columns, for each of the ``max_par`` parallel batch-64 problems it
    may run. 128 is *not* the minimum tile width: a 64-column problem selects
    the ``thread_n = 64`` config, so ``n = 64`` needs one lock and ``n = 192``
    needs three.
    """

    return (out_features + 63) // 64 * max_par


def _invert_permutation(values) -> torch.Tensor:
    inverse = [0] * len(values)
    for index, value in enumerate(values):
        inverse[int(value)] = index
    # These fixed permutation tables are used after a meta-device checkpoint
    # restore, so they must contain real data even inside torch.device("meta").
    return torch.tensor(inverse, dtype=torch.long, device="cpu")


def _unpack_uint4(packed: torch.Tensor) -> torch.Tensor:
    """Unpack eight unsigned 4-bit lanes per INT32 with INT32/INT8 temporaries.

    The previous implementation widened the packed tensor to INT64 and divided
    by eight INT64 divisors, materialising several ``8 * in_features *
    out_features`` byte temporaries per forward (issue #4387 item 4).  Shifting
    and masking in INT32 needs one packed-sized INT32 temporary and a
    ``packed.numel() * 8`` UINT8 result, and returns the same codes bit for bit.
    """

    rows, columns = packed.shape
    lanes = torch.empty((rows, columns * 8), dtype=torch.uint8, device=packed.device)
    shifted = torch.empty_like(packed)
    for lane in range(8):
        shifted.copy_(packed)
        shifted.bitwise_right_shift_(4 * lane)
        shifted.bitwise_and_(0xF)
        lanes[:, lane::8] = shifted
    return lanes


def mul(
    A, B, C, D, s1, s2, s3, workspace, thread_k=-1, thread_n=-1, sms=-1, max_par=16
):
    """INT8xINT4 multiply based on Marlin kernel; can be used within `torch.compile`.
    @A: `torch.int8` input matrix of shape `(m, k)` in standard row-major layout
    @B: `torch.int32` weight matrix of original shape `(k, n)` in the specified format; see `Layer.pack()`
    @C: `torch.int32` reduce buffer of shape `(max_par * 64, n)` in standard row-major layout
    @D: `torch.float16` out matrix of shape `(m, n)` in standard row-major layout
    @s1: `torch.float32` activation per-token quantization scales of shape `(m, 1)`
    @s2: `torch.float32` weight per-channel quantization scales of shape `(1, n)`
    @s3: `torch.float16` weight per-group quantization scales of shape `(m / groupsize, n)`, it should be empty when group_size != -1
    @workspace: `torch.int32` tensor with at least `ceil(n / 64) * max_par` entries that are all zero
    @thread_k: `k` size of a thread_tile in `B` (can usually be left as auto -1)
    @thread_n: `n` size of a thread_tile in `B` (can usually be left as auto -1)
    @sms: number of SMs to use for the kernel (can usually be left as auto -1)
    @max_par: maximum number of batch 64 problems to solve in parallel for large input sizes
    """
    if not qqq_runtime_available():
        raise ModuleNotFoundError("QQQ torch.ops kernels are not properly installed. Error: " + qqq_runtime_error())
    qqq_gemm(A, B, C, D, s1, s2, s3, workspace, thread_k, thread_n, sms, max_par)


class QQQLinear(GroupedQuantLinear):
    # The AVX512 CPU forward drops these after building its packed cache, so
    # they have to be materialized before an export and required on load
    # (#5010). `s_group` is legitimately empty when `group_size` spans the whole
    # input, so only its presence is checked.
    CHECKPOINT_REQUIRED_BUFFERS = ("B", "s_group", "s_channel")
    # Per-channel layers (`group_size == in_features`) hold an empty
    # `s_group`; the kernel builds its own empty operand. Older saves dropped
    # every such key as shared storage (#5126), so accept a missing empty
    # entry. A grouped layer's non-empty `s_group` remains required.
    CHECKPOINT_OPTIONAL_EMPTY_BUFFERS = ("s_group",)

    # The native Marlin W4A8 `qqq_gemm` custom op has no fake impl and the
    # per-call reduce scratch/locks allocation is host-side; the JIT extension
    # is also unbuildable here (no CUDA toolkit). Static: EAGER.
    SUPPORTS_COMPILE_LEVEL = COMPILE_LEVEL.EAGER
    SUPPORTS_GRAPH_CAPTURE = False

    SUPPORTS_BACKENDS = [BACKEND.QQQ]
    SUPPORTS_METHODS = [METHOD.QQQ]
    SUPPORTS_FORMATS = {FORMAT.QQQ: 100}
    SUPPORTS_BITS = [4]
    SUPPORTS_GROUP_SIZE = [-1, 128]
    SUPPORTS_DESC_ACT = [False]
    SUPPORTS_SYM = [True]
    SUPPORTS_SHARDS = True
    SUPPORTS_TRAINING = False
    SUPPORTS_AUTO_PADDING = False
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [64]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [64]

    SUPPORTS_DEVICES = [DEVICE.CUDA]
    SUPPORTS_PLATFORM = [PLATFORM.LINUX]
    SUPPORTS_PACK_DTYPES = [torch.int32]
    SUPPORTS_ADAPTERS = [Lora]

    # The native Marlin-derived W4A8 kernel is FP16-only. BF16 checkpoints use
    # the portable ``QQQTorchLinear`` (BACKEND.QQQ_TORCH), which computes in
    # FP32 and is what AUTO selects for BF16.
    SUPPORTS_DTYPES = [torch.float16]

    #: Whether the native Marlin kernel's device restrictions apply: SM80 or
    #: newer on NVIDIA, and no ROCm. The portable FP32 kernel clears this.
    MARLIN_DEVICE_RESTRICTIONS = True

    REQUIRES_FORMAT_V2 = False

    # for transformers/optimum tests compat
    QUANT_TYPE = "qqq"

    IN_OUTPUT_FEATURES_DIVISIBLE_BY = [(64, 256), (128, 128), (128, 64), (64, 128)]

    def __init__(
        self, bits: int,
        group_size: int,
        desc_act: bool,
        sym: bool,
        in_features: int,
        out_features: int,
        bias: bool = False,
        pack_dtype: torch.dtype = torch.int32,
        adapter: Adapter = None,
        register_buffers: bool = True,
        **kwargs):
        self.tile = 16
        self.max_par = 16

        # self.original_in_features = in_features
        # self.original_out_features = out_features

        if desc_act:
            raise ValueError("QQQ does not support activation ordering (`desc_act=True`).")

        super().__init__(
            bits=bits,
            group_size=group_size,
            sym=sym,
            desc_act=desc_act,
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            pack_dtype=pack_dtype,
            backend=kwargs.pop("backend", BACKEND.QQQ),
            adapter=adapter,
            # QQQ does it's own buffer setup
            register_buffers=False,
            **kwargs)

        # QQQ only needs the code range, not packed GPTQ/AWQ storage metadata.
        self.maxq = (1 << self.bits) - 1

        # during quantization, we do are not loading tensors from disk so no need to preallocate buffers
        if register_buffers:
            self.register_buffer(
                "B",
                torch.empty(
                    (self.in_features // 16, self.out_features * 16 // 8), dtype=torch.int32
                ),
            )
            self.register_buffer(
                "s_channel",
                torch.empty(
                    (1, self.out_features),
                    dtype=torch.float32,
                ),
            )
            if self.group_size != self.in_features:
                self.register_buffer(
                    "s_group",
                    torch.empty(
                        (self.in_features // self.group_size, self.out_features),
                        dtype=torch.float16,
                    ),
                )
            else:
                self.register_buffer(
                    "s_group",
                    torch.tensor([], dtype=torch.float16),
                )
            self.register_buffer(
                "workspace",
                torch.zeros(
                    _workspace_numel(self.out_features, self.max_par), dtype=torch.int32
                ),
                persistent=False,
            )
        # Quantization builds this module with `register_buffers=False`, so the
        # bias attribute must be initialized outside the buffer preallocation
        # block or `pack()` cannot tell whether a bias is expected.
        if bias:
            self.register_buffer("bias", torch.zeros((self.out_features), dtype=torch.float16))
        else:
            self.bias = None

        (
            self._perm,
            self._perm_i,
            self._scale_perm,
            self._scale_perm_i,
            self._scale_perm_single,
            self._scale_perm_single_i,
        ) = self._get_perms()

        # auto-optimize on post init
        # self.optimize()

    def _get_perms(self):
        perm = []
        for i in range(32):
            perm1 = []
            col = i // 4
            for block in [0, 1]:
                for row in [
                    4 * (i % 4),
                    4 * (i % 4) + 1,
                    4 * (i % 4) + 2,
                    4 * (i % 4) + 3,
                ]:
                    perm1.append(16 * row + col + 8 * block)
            for j in range(4):
                perm.extend([p + 256 * j for p in perm1])

        if self.group_size == self.in_features:
            interleave = [4, 0, 5, 1, 6, 2, 7, 3]
        else:
            interleave = [0, 2, 4, 6, 1, 3, 5, 7]
        perm = [
            perm[row * 8 + column]
            for row in range(len(perm) // 8)
            for column in interleave
        ]
        # Checkpoint restore constructs this module under torch.device("meta").
        # Invert the Python layout before materializing it: meta tensors cannot
        # be copied back to a list.
        perm_i = _invert_permutation(perm)
        perm = torch.tensor(perm, dtype=torch.long, device="cpu")
        scale_perm = []
        for i in range(8):
            scale_perm.extend([i + 8 * j for j in range(8)])
        scale_perm_i = _invert_permutation(scale_perm)
        scale_perm_single = []
        for i in range(4):
            scale_perm_single.extend([2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
        scale_perm_single_i = _invert_permutation(scale_perm_single)
        return perm, perm_i, scale_perm, scale_perm_i, scale_perm_single, scale_perm_single_i

    # def optimize(self, backend: str = "inductor", mode: str = None, fullgraph: bool = False):
    #     if self.optimized:
    #         return
    #
    #     # compile dequantize
    #     self.forward = torch_compile(self.forward, backend=backend, mode=mode, fullgraph=fullgraph)
    #
    #     super().optimize()

    @classmethod
    def validate_once(cls) -> Tuple[bool, Optional[Exception]]:
        # Check the architecture before the runtime probe: probing the JIT
        # extension on ROCm spends ~18 s compiling before failing, and would
        # still be the wrong kernel even if a future hipify translated it.
        if IS_ROCM:
            return False, NotImplementedError(QQQ_ROCM_UNSUPPORTED)
        if not qqq_runtime_available():
            return False, ImportError(qqq_runtime_error())
        return True, None

    @classmethod
    def validate(cls, **args) -> Tuple[bool, Optional[Exception]]:
        in_features = args.get("in_features")
        out_features = args.get("out_features")
        if in_features and out_features and not any(
                in_features % thread_k == 0 and out_features % thread_n == 0
                    for thread_k, thread_n in cls.IN_OUTPUT_FEATURES_DIVISIBLE_BY
        ):
            raise ValueError(f"{cls} not supported `infeatures`: {in_features} and `outfeatures`: {out_features}.")

        # ``_validate`` alone never consults ``validate_once``, so without this
        # check AUTO selected QQQLinear on any host even when the Marlin-derived
        # JIT extension could not be built.  On ROCm that is the steady state:
        # ``qqq_gemm.cu`` relies on CUDA-only inline PTX, which hipify cannot
        # translate, so the extension fails to build for gfx1201 and the first
        # forward raises instead of the selector picking ``QQQTorchLinear``.
        # Check the runtime here so the portable kernel is chosen up front.
        ok_once, error_once = cls.cached_validate_once()
        if not ok_once:
            return False, error_once

        return cls._validate(**args)

    @classmethod
    def validate_device(cls, device: DEVICE | torch.device):
        # The native kernel runs Marlin, so it needs SM80 and has no ROCm path.
        # The portable subclass clears this flag: it computes with a plain FP32
        # matmul, so neither restriction applies to it. Keeping the difference
        # as data means one implementation and no `super()` call that skips a
        # class in the MRO.
        if not cls.MARLIN_DEVICE_RESTRICTIONS:
            super().validate_device(device)
            return
        # ROCm exposes the CUDA device namespace through torch, so the device
        # check alone would accept a cuda:N target.  Reject it explicitly here
        # instead of inferring CUDA capability from a HIP device.
        if IS_ROCM:
            raise NotImplementedError(QQQ_ROCM_UNSUPPORTED)
        super().validate_device(device)
        if (device.type if isinstance(device, torch.device) else device) in ("cuda", DEVICE.CUDA):
            targets = (
                (device,)
                if isinstance(device, torch.device)
                else tuple(torch.device(f"cuda:{index}") for index in range(torch.cuda.device_count()))
            )
            if not targets or any(torch.cuda.get_device_capability(target)[0] < 8 for target in targets):
                raise NotImplementedError("Marlin kernel only supports compute capability >= 8.0.")

    def post_init(self):
        super().post_init()

        self.s_channel = self.s_channel.to(dtype=torch.float32)
        self.s_group = self.s_group.to(dtype=torch.float16)

    def list_buffers(self) -> List:
        buf = super().list_buffers()
        if hasattr(self, "B") and self.B is not None:
            buf.append(self.B)
        if hasattr(self, "s_channel") and self.s_channel is not None:
            buf.append(self.s_channel)
        if hasattr(self, "s_group") and self.s_group is not None:
            buf.append(self.s_group)
        if hasattr(self, "workspace") and self.workspace is not None:
            buf.append(self.workspace)
        return buf

    def _reduce_scratch(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """Allocate the INT32 reduce scratch for this kernel call.

        The native kernel consumes ``C`` for the duration of one call and does
        not read its initial contents, so every call allocates its own buffer
        instead of sharing a process-wide cache that outlives the model
        (#4621).  One allocation also serves both the eager and the compiled
        path: nothing here can be captured as a graph constant.
        """

        if device is None:
            device = self.B.device if hasattr(self, "B") else torch.device("cpu")
        return _allocate_reduce_scratch(device, self.out_features, self.max_par)

    def pack(self, linear: torch.nn.Module, scales: torch.Tensor, s_extra=None):
        """Pack a fake-quantized linear layer into the Marlin representation."""
        if self.group_size != self.in_features:
            assert s_extra is not None, "s_extra is needed"
        if linear.weight.dtype != torch.float16:
            log.warn.once(
                f"""The dtype of weights is {linear.weight.dtype}, while our w4a8 GEMM's output is torch.float16.
                If you can ensure your GEMM results don't overflow torch.float16, it will still function correctly.
                Otherwise, it will yield incorrect results."""
            )

        raw_scales = scales.t()
        if self.group_size != self.in_features:
            s_extra = s_extra.reshape(1, -1).to(dtype=torch.float32)
            packed_s_group = (raw_scales / s_extra).to(dtype=torch.float16)
            packed_s_group = packed_s_group.reshape(
                (-1, len(self._scale_perm))
            )[:, self._scale_perm].reshape((-1, self.out_features)).contiguous()
            packed_s_channel = s_extra.reshape(
                (-1, len(self._scale_perm_single))
            )[:, self._scale_perm_single].reshape((-1, self.out_features)).contiguous()
        else:
            packed_s_group = None
            packed_s_channel = (
                # The stored channel scale is FP32; dividing in FP16 first
                # can round a representable FP32 scale all the way to zero.
                (raw_scales.float() / (2 ** (8 - self.bits)))
                .reshape((-1, len(self._scale_perm_single)))[:, self._scale_perm_single]
                .to(dtype=torch.float32)
                .reshape((-1, self.out_features))
                .contiguous()
            )

        input_chunk_size = min(1024, self.in_features)
        output_chunk_size = min(256, self.out_features)
        input_chunk_size -= input_chunk_size % self.tile
        output_chunk_size -= output_chunk_size % 64
        if input_chunk_size == 0 or output_chunk_size == 0:
            raise ValueError(
                f"QQQ pack requires dimensions divisible by {self.tile} and 64, "
                f"got in_features={self.in_features}, out_features={self.out_features}"
            )

        packed_weight = torch.empty(
            (self.in_features // self.tile, self.out_features * 2),
            dtype=torch.int32,
            device=CPU,
        )
        weight = linear.weight.data
        perm = self._perm.to(device=weight.device)
        for input_start in range(0, self.in_features, input_chunk_size):
            input_end = min(input_start + input_chunk_size, self.in_features)
            input_size = input_end - input_start
            if input_size % self.tile != 0:
                raise ValueError("QQQ pack input chunk is not tile aligned")

            input_indices = torch.arange(
                input_start,
                input_end,
                device=weight.device,
                dtype=torch.long,
            )
            if self.group_size != self.in_features:
                group_indices = torch.div(input_indices, self.group_size, rounding_mode="floor")
                scale_chunk = raw_scales.index_select(0, group_indices)
            else:
                scale_chunk = raw_scales.expand(input_size, -1)

            for output_start in range(0, self.out_features, output_chunk_size):
                output_end = min(output_start + output_chunk_size, self.out_features)
                output_size = output_end - output_start
                if output_size % 64 != 0:
                    raise ValueError("QQQ pack output chunk is not permutation aligned")

                weight_chunk = weight[output_start:output_end, input_start:input_end].transpose(0, 1)
                scale_chunk_view = scale_chunk[:, output_start:output_end]
                codes = torch.round(weight_chunk.float() / scale_chunk_view.float())
                if self.group_size != self.in_features:
                    codes.add_((self.maxq + 1) // 2).clamp_(0, self.maxq)
                else:
                    # Channelwise codes are signed INT4, unlike the biased
                    # unsigned codes used by the grouped representation.
                    codes.clamp_(-(1 << (self.bits - 1)), (1 << (self.bits - 1)) - 1)
                codes = codes.to(dtype=torch.int32)

                transformed = codes.reshape(
                    input_size // self.tile,
                    self.tile,
                    output_size // self.tile,
                    self.tile,
                ).permute(0, 2, 1, 3).reshape(
                    input_size // self.tile,
                    output_size * self.tile,
                )
                transformed = transformed.reshape(-1, perm.numel()).index_select(1, perm).reshape(
                    input_size // self.tile,
                    output_size * self.tile,
                )
                packed_chunk = torch.zeros(
                    (input_size // self.tile, output_size * 2),
                    dtype=torch.int32,
                    device=weight.device,
                )
                for lane in range(8):
                    packed_chunk.bitwise_or_(
                        (transformed[:, lane::8] & 0xF) << (4 * lane)
                    )

                packed_weight[
                    input_start // self.tile:input_end // self.tile,
                    output_start * 2:output_end * 2,
                ].copy_(packed_chunk.to(device=CPU))

        self.register_buffer("B", packed_weight)
        if self.group_size != self.in_features:
            self.register_buffer("s_group", packed_s_group)
            self.register_buffer("s_channel", packed_s_channel)
        else:
            if "s_group" not in self._buffers:
                self.register_buffer(
                    "s_group",
                    torch.tensor([], dtype=torch.float16),
                )
            self.register_buffer("s_channel", packed_s_channel)
            self.register_buffer("s_group", torch.empty(0, dtype=torch.float16, device=packed_s_channel.device))
        # Quantization constructs QQQLinear with register_buffers=False. Its
        # packed instance still needs the nonpersistent runtime scratch buffers
        # that a checkpoint-loaded instance receives in __init__.
        if not hasattr(self, "workspace"):
            self.register_buffer(
                "workspace",
                torch.zeros(
                    _workspace_numel(self.out_features, self.max_par),
                    dtype=torch.int32,
                    device=packed_weight.device,
                ),
                persistent=False,
            )
        if linear.bias is not None:
            bias = linear.bias.detach().to(device=packed_weight.device, dtype=torch.float16)
            if "bias" in self._buffers:
                self.bias.copy_(bias)
            else:
                self.register_buffer("bias", bias)
        elif not hasattr(self, "bias"):
            self.bias = None
        # A repack replaces ``B``/scales, so any derived CPU cache (and the
        # runtime-only layout it may have installed) is stale.
        clear = getattr(self, "clear_weight_cache", None)
        if callable(clear):
            clear()

    # activation int8 quantization
    def dynamic_quant(self, x: torch.Tensor):
        if (x.device.type == "cuda" and x.is_contiguous() and 0 < x.shape[-1] <= 32768
                and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
                and not torch.compiler.is_compiling()):
            quantizer = _qqq_gpu_quantizer()
            if quantizer is not None:
                return quantizer(x)
        x = x.float()
        quant_scale = x.abs().amax(dim=-1, keepdim=True).div(127.0)
        quant_scale = torch.where(quant_scale == 0, torch.ones_like(quant_scale), quant_scale)
        x = (x / quant_scale).round().clamp(-128, 127).to(torch.int8)
        # The element-wise fallback preserves the input's memory format, so a
        # column-strided activation (``x.t()`` of a ``[K, M]`` buffer) comes back
        # column-major. ``qqq_gemm`` reads the operand row-major with no stride
        # argument, so hand it a packed INT8 buffer (#4740). The copy is int8,
        # i.e. a quarter of the fp16 operand it replaces.
        return x.contiguous(), quant_scale

    def _split_k_locks(self, device) -> torch.Tensor:
        """A fresh zeroed split-K lock array for one forward call.

        ``qqq_gemm.cu`` spins on and resets these counters, so a single
        per-module buffer lets two streams serving the same module corrupt each
        other's counters and deadlock (#4665). EXL3 allocates its locks per
        call for the same reason; this array is a few hundred int32 slots.
        """
        return torch.zeros(
            _workspace_numel(self.out_features, self.max_par),
            dtype=torch.int32,
            device=device,
        )

    def forward(self, A):
        # TODO FIXME: parent should never call us if there is no data to process
        # check: https://github.com/ModelCloud/GPTQModel/issues/1361
        if self.input_rows(A) == 0:
            return self.empty_linear_output(A)

        A_dtype = A.dtype
        # qqq is float16 kernel only. Do not narrow BF16/FP32 here: an
        # activation above 65504 would become inf before dynamic_quant.
        if A_dtype != torch.float16:
            raise NotImplementedError(
                f"{type(self).__name__} is an FP16 kernel; use BACKEND.QQQ_TORCH "
                f"for {A_dtype} activations."
            )

        out_shape = A.shape[:-1] + (self.out_features,)
        A = A.reshape(-1, A.shape[-1])
        quant_A, s1 = self.dynamic_quant(A)
        D = torch.empty(A.shape[0], self.out_features, dtype=A.dtype, device=A.device)
        mul(
            quant_A, # A
            self.B, # B
            self._reduce_scratch(A.device), # C
            D, # D
            s1, # s1
            self.s_channel, # s2
            self.s_group, # s3
            self._split_k_locks(A.device),
            max_par=self.max_par,
        )

        # TODO: check if we should reshape at end
        D = D.reshape(out_shape)

        if self.bias is not None:
            # The kernel output is FP16; add the stored bias in FP32 so the
            # visible cast is the only rounding after the reduction.
            D = (D.float() + self.bias.float()).to(dtype=A_dtype)

        if self.adapter:
            D = self.adapter.apply(x=A, out=D)

        return D.to(dtype=A_dtype)


class QQQTorchLinear(QQQLinear):
    # Measured on GB10 (sm_121), torch 2.11.0+cu130: eager, dynamo, aot_eager,
    # inductor and reduce-overhead all pass. Explicit torch.cuda.graph capture
    # fails: forward copies the CPU `_scale_perm_single_i` table to the device.
    SUPPORTS_COMPILE_LEVEL = COMPILE_LEVEL.CUDAGRAPH
    # CPU, torch 2.16.0.dev20261021+cpu: eager=True, graph_breaks=0, and
    # dynamo/aot_eager/inductor all pass. The packed CPU kernel
    # (`qqq_int4`) now has a FakeTensor impl, and the traced forward reads the
    # kernel handle and packed cache the eager warmup published instead of
    # calling `_qqq_cpu_kernel()` (importlib) and `_packed_weight_for_cpu()`
    # (RLock + checkpoint rewrite). A compile before any eager call takes the
    # traceable `torch._int_mm` path instead.
    SUPPORTS_COMPILE_LEVEL_BY_DEVICE = {"cpu": COMPILE_LEVEL.INDUCTOR}
    SUPPORTS_GRAPH_CAPTURE = False
    # ... and a static-cache `generate()` is a win on this torch kernel:
    # Qwen2.5-0.5B-Instruct QQQ, bf16, RTX 4090 (sm_89), torch 2.15.0+cu132,
    # 4 left-padded prompts x 32 greedy tokens: 2.08 s eager -> 0.48 s compiled
    # (4.3x), 1 dynamo graph, 0 graph breaks (#5102). The 172-graph /
    # 24-115-break number this declaration used to cite came from the AUTO
    # `QQQLinear` CUDA kernel, not from this torch kernel; #5075 made the packed
    # QQQ op traceable, so the copied reason was stale.
    SUPPORTS_COMPILE = True

    SUPPORTS_BACKENDS = [BACKEND.QQQ_TORCH]
    SUPPORTS_METHODS = [METHOD.QQQ]
    SUPPORTS_FORMATS = {FORMAT.QQQ: 90}
    SUPPORTS_BITS = [4]
    SUPPORTS_GROUP_SIZE = [-1, 128]
    SUPPORTS_DESC_ACT = [False]
    SUPPORTS_SYM = [True]
    SUPPORTS_SHARDS = True
    SUPPORTS_TRAINING = False
    SUPPORTS_AUTO_PADDING = False
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [64]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [64]
    SUPPORTS_DEVICES = [DEVICE.CPU, DEVICE.CUDA, DEVICE.ROCM, DEVICE.XPU, DEVICE.NPU, DEVICE.MPS]
    SUPPORTS_PLATFORM = [PLATFORM.ALL]
    SUPPORTS_PACK_DTYPES = [torch.int32]
    SUPPORTS_ADAPTERS = [Lora]
    SUPPORTS_DTYPES = [torch.float16, torch.bfloat16]

    REQUIRES_FORMAT_V2 = False
    QUANT_TYPE = "qqq"

    #: Runtime state that ``__init__`` normally creates. ``rehome_quantized_kernel``
    #: reinterprets an already-built ``QQQLinear`` by assigning ``__class__``, which
    #: never runs this ``__init__``, so every read has to find a class-level default
    #: as well (#4940; the same class-swap path broke #4451 before it).
    _torch_packed_cache = None
    _torch_packed_cache_key = None
    _torch_dropped_checkpoint = None
    #: Resolved packed-CPU kernel handle, published by the eager warmup so a
    #: traced forward never calls ``_qqq_cpu_kernel()`` (importlib is skipped by
    #: dynamo). ``None`` means the AVX512 kernel is unavailable.
    _torch_cpu_kernel = None
    # ``None`` means "no failed shape recorded yet": a shared class-level ``set``
    # would leak one module's failures into every other module.
    _torch_int_mm_failed_shapes = None

    #: This kernel computes with a plain FP32 matmul, so it needs neither the
    #: Marlin compute-capability floor nor the CUDA-only requirement. Without
    #: this, a pre-Ampere NVIDIA card that could run the fallback is rejected at
    #: selection time.
    MARLIN_DEVICE_RESTRICTIONS = False

    @classmethod
    def _init_runtime_state(cls, module) -> None:
        """Install per-instance runtime state on a class-swapped module.

        ``_switch_quant_linear_class`` assigns ``__class__`` without running
        ``__init__``, so the target kernel has to be able to initialise the
        attributes it reads. Keeping this a hook means the next attribute added
        here is picked up by the swap instead of raising ``AttributeError`` at
        the end of ``quantize()`` (#4940).
        """

        module._torch_packed_cache = None
        module._torch_packed_cache_key = None
        module._torch_dropped_checkpoint = None
        module._torch_cpu_kernel = None
        module._torch_int_mm_failed_shapes = set()

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("backend", BACKEND.QQQ_TORCH)
        super().__init__(*args, **kwargs)
        # ``_torch_dropped_checkpoint`` is ``{name: (shape, dtype)}`` for the
        # checkpoint tensors the packed CPU cache replaced; ``None`` means the
        # checkpoint layout is still resident.
        self._init_runtime_state(self)

    def post_init(self):
        self._materialize_dropped_checkpoint()
        super().post_init()
        self.clear_weight_cache()

    def _failed_int_mm_shapes(self) -> set:
        """Per-instance ``torch._int_mm`` failure memo, created on demand."""

        failed_shapes = getattr(self, "_torch_int_mm_failed_shapes", None)
        if failed_shapes is None:
            failed_shapes = set()
            self._torch_int_mm_failed_shapes = failed_shapes
        return failed_shapes

    def clear_weight_cache(self) -> None:
        # The packed cache is the only remaining copy of ``B``/the scales once
        # it has replaced them, so restore the checkpoint layout first.
        self._materialize_dropped_checkpoint()
        self._torch_packed_cache = None
        self._torch_packed_cache_key = None
        self._failed_int_mm_shapes().clear()

    @classmethod
    def validate_once(cls) -> Tuple[bool, Optional[Exception]]:
        return True, None

    def _checkpoint_B(self) -> torch.Tensor:
        """Checkpoint-packed INT4 words for the live weight layout."""
        return self.B

    def _checkpoint_scale(self, name: str) -> torch.Tensor:
        """Checkpoint scale tensor for ``name`` for the live weight layout."""
        return getattr(self, name)

    def _unpack_weight_codes(self) -> torch.Tensor:
        unpacked = _unpack_uint4(self._checkpoint_B())
        perm_i = self._perm_i.to(device=unpacked.device)
        unpacked = unpacked.reshape(-1, self._perm_i.numel())[:, perm_i].reshape(
            self.in_features // self.tile,
            self.out_features * self.tile,
        )
        return (
            unpacked.reshape(
                self.in_features // self.tile,
                self.out_features // self.tile,
                self.tile,
                self.tile,
            )
            .permute(0, 2, 1, 3)
            .reshape(self.in_features, self.out_features)
        )

    def _unpermute_scales(self, scales: torch.Tensor, inverse_perm: torch.Tensor) -> torch.Tensor:
        inverse_perm = inverse_perm.to(device=scales.device)
        return scales.reshape(-1, inverse_perm.numel())[:, inverse_perm].reshape(scales.shape)

    def _dequantize_weight_for_torch(self) -> tuple[torch.Tensor, torch.Tensor]:
        codes = self._unpack_weight_codes()
        s_channel = self._unpermute_scales(
            self._checkpoint_scale("s_channel"), self._scale_perm_single_i
        ).to(torch.float32)

        if self.group_size != self.in_features:
            s_group = self._unpermute_scales(
                self._checkpoint_scale("s_group"), self._scale_perm_i
            ).to(torch.float32)
            weight = self._grouped_weight_fp32(codes, s_group)
        else:
            signed = codes.to(torch.int8)
            signed.bitwise_xor_(8)
            signed.sub_(8)
            weight = signed.to(torch.float32)
            weight.mul_(16.0)

        return weight, s_channel

    def _grouped_weight_fp32(self, codes: torch.Tensor, s_group: torch.Tensor) -> torch.Tensor:
        """Expand grouped QQQ codes to the FP32 INT8-valued weight plane.

        ``codes`` is the UINT8 nibble plane.  When the group size divides
        ``in_features`` the scale broadcast is a view, so no ``in_features *
        out_features`` gather temporary is created; the odd tail keeps the
        index-select fallback.
        """

        weight = codes.to(torch.float32)
        weight.sub_(8.0)
        if self.group_size > 0 and self.in_features % self.group_size == 0:
            groups = self.in_features // self.group_size
            weight = weight.reshape(groups, self.group_size, self.out_features)
            weight.mul_(s_group.reshape(groups, 1, self.out_features))
            weight = weight.reshape(self.in_features, self.out_features)
        else:
            group_idx = torch.floor_divide(
                torch.arange(self.in_features, dtype=torch.int64, device=codes.device),
                self.group_size,
            )
            weight.mul_(s_group[group_idx])
        weight.round_()
        weight.clamp_(-128.0, 127.0)
        return weight

    def _int8_weight_for_forward(self) -> torch.Tensor:
        """Build the transient INT8 weight plane for one forward.

        This is deliberately not cached: an INT4-to-INT8 expansion is an
        expanded weight cache under ``AGENTS.md``.  Channel-wise codes are
        signed nibbles scaled by 16, so they convert directly to INT8 without
        an FP32 plane; grouped codes need the rounded FP32 product.
        """

        codes = self._unpack_weight_codes()
        if self.group_size != self.in_features:
            s_group = self._unpermute_scales(
                self._checkpoint_scale("s_group"), self._scale_perm_i
            ).to(torch.float32)
            return self._grouped_weight_fp32(codes, s_group).to(torch.int8)
        signed = codes.to(torch.int8)
        signed.bitwise_xor_(8)
        signed.sub_(8)
        signed.mul_(16)
        return signed

    def _torch_weight_state_key(self):
        return (
            tensor_cache_key(self.B),
            tensor_cache_key(self.s_channel),
            tensor_cache_key(self.s_group),
            self.group_size,
            self.in_features,
            self.out_features,
        )

    def _pack_weight_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """Repack UINT8 codes into the checkpoint ``B`` word layout.

        Exact inverse of :meth:`QQQLinear._unpack_weight_codes`; the packing is
        a pure nibble permutation, so no value is rounded.
        """
        tile = self.tile
        in_features, out_features = self.in_features, self.out_features
        perm = self._perm.to(device=codes.device)
        transformed = (
            codes.reshape(in_features // tile, tile, out_features // tile, tile)
            .permute(0, 2, 1, 3)
            .reshape(in_features // tile, out_features * tile)
        )
        transformed = transformed.reshape(-1, perm.numel()).index_select(1, perm).reshape(
            in_features // tile, out_features * tile
        )
        packed = torch.zeros(
            (in_features // tile, out_features * 2),
            dtype=torch.int32,
            device=codes.device,
        )
        for lane in range(8):
            packed.bitwise_or_((transformed[:, lane::8].to(torch.int32) & 0xF) << (4 * lane))
        return packed

    def _rebuild_checkpoint_B(self) -> torch.Tensor:
        """Rebuild the checkpoint ``B`` words from the packed CPU cache."""
        cached = self._torch_packed_cache
        if cached is None:
            raise RuntimeError(
                f"{type(self).__name__} has no packed CPU cache to rebuild its checkpoint weight from"
            )
        from .torch_cpu_avx512 import unpack_hf_int4

        packed = cached[0]
        codes = unpack_hf_int4(packed, self.out_features, self.in_features).t().contiguous()
        if self.group_size == self.in_features:
            # The kernel-facing cache stores channel-wise codes biased by 8.
            codes = codes ^ 8
        return self._pack_weight_codes(codes)

    def _rebuild_checkpoint_scales(self, name: str, shape, dtype) -> torch.Tensor:
        cached = self._torch_packed_cache
        if cached is None:
            raise RuntimeError(
                f"{type(self).__name__} has no packed CPU cache to rebuild its checkpoint scales from"
            )
        if name == "s_group":
            scales, perm = cached[1], self._scale_perm
        else:
            scales, perm = cached[2], self._scale_perm_single
        if scales.numel() == 0:
            return torch.empty(shape, dtype=dtype, device=scales.device)
        return (
            self._unpermute_scales(scales, torch.tensor(perm, dtype=torch.long))
            .to(dtype=dtype)
            .reshape(shape)
            .contiguous()
        )

    def _rebuild_checkpoint_tensors(self, descriptors) -> dict[str, torch.Tensor]:
        rebuilt = {"B": self._rebuild_checkpoint_B()}
        for name in ("s_group", "s_channel"):
            shape, dtype = descriptors[name]
            rebuilt[name] = self._rebuild_checkpoint_scales(name, shape, dtype)
        return rebuilt

    def _checkpoint_B(self) -> torch.Tensor:
        """Checkpoint ``B``, rebuilt transiently when the cache replaced it."""
        if self._torch_dropped_checkpoint is None:
            return self.B
        return self._rebuild_checkpoint_B()

    def _checkpoint_scale(self, name: str) -> torch.Tensor:
        """Checkpoint scales, rebuilt transiently when the cache replaced them."""
        tensor = getattr(self, name, None)
        if tensor is not None:
            return tensor
        descriptors = self._torch_dropped_checkpoint
        if descriptors is None:
            raise RuntimeError(
                f"{type(self).__name__} has no checkpoint {name} and no packed cache"
            )
        shape, dtype = descriptors[name]
        return self._rebuild_checkpoint_scales(name, shape, dtype)

    def _materialize_dropped_checkpoint(self) -> bool:
        """Restore ``B``/scales from the cache and forget the runtime-only layout."""
        with self._cpu_transform_lock:
            descriptors = getattr(self, "_torch_dropped_checkpoint", None)
            if descriptors is None:
                return False
            for name, tensor in self._rebuild_checkpoint_tensors(descriptors).items():
                self._buffers[name] = tensor
            self._torch_dropped_checkpoint = None
            if getattr(self, "_torch_packed_cache", None) is not None:
                self._torch_packed_cache_key = self._torch_weight_state_key()
            return True

    def materialize_checkpoint_layout_for_export(self) -> bool:
        """Publish the dropped checkpoint buffers so an export can see them."""

        return self._materialize_dropped_checkpoint()

    def _maybe_drop_checkpoint_tensors(self) -> bool:
        """Release the checkpoint copy once the cache proves it is rebuildable.

        Fails closed: any shape/dtype/value mismatch keeps both layouts
        resident, so an unverified inverse can never reach a saved checkpoint.
        """
        with self._cpu_transform_lock:
            if getattr(self, "_torch_dropped_checkpoint", None) is not None:
                return False
            descriptors = {}
            for name in ("B", "s_group", "s_channel"):
                tensor = self._buffers.get(name)
                if not isinstance(tensor, torch.Tensor):
                    return False
                descriptors[name] = (tuple(tensor.shape), tensor.dtype)
            try:
                rebuilt = self._rebuild_checkpoint_tensors(descriptors)
            except Exception:
                return False
            for name, tensor in rebuilt.items():
                current = self._buffers.get(name)
                if (
                    current is None
                    or tuple(current.shape) != tuple(tensor.shape)
                    or current.dtype != tensor.dtype
                    or not torch.equal(current, tensor)
                ):
                    return False
            # Publish the descriptor before dropping the buffers so a concurrent
            # reader never sees a missing tensor without a rebuild recipe.
            self._torch_dropped_checkpoint = descriptors
            for name in rebuilt:
                self._buffers[name] = None
            return True

    def _packed_weight_for_cpu(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build the CPU kernel cache, then replace the checkpoint layout.

        The AVX512 ``qqq_int4`` kernel consumes packed INT4 codes and
        unpermuted scales, never the checkpoint ``B``/scales. Once the cache is
        built, :meth:`_maybe_drop_checkpoint_tensors` proves it can rebuild the
        checkpoint tensors bit for bit and drops them, so the packed weight is
        not resident twice (#4805). Any verification mismatch keeps both.

        The whole build runs under ``_cpu_transform_lock``, like the other CPU
        int4 kernels' ``_ensure_cpu_layout``: a concurrent cold forward would
        otherwise pass the cache check, then read ``B``/``s_channel`` after this
        thread dropped them (#4942).
        """
        from .torch_cpu_avx512 import pack_hf_int4

        with self._cpu_transform_lock:
            cached = getattr(self, "_torch_packed_cache", None)
            if cached is not None and getattr(self, "_torch_dropped_checkpoint", None) is not None:
                # The cache is the only resident weight; nothing can mutate it
                # behind the key without materializing the checkpoint first.
                return cached
            key = self._torch_weight_state_key()
            if cached is not None and getattr(self, "_torch_packed_cache_key", None) == key:
                return cached

            codes = self._unpack_weight_codes().t().to(torch.uint8)
            # Keep the scales in the checkpoint dtype: the kernel call widens the
            # group scales it needs, and an FP32 cache doubles their residency.
            s_channel = (
                self._unpermute_scales(self.s_channel, self._scale_perm_single_i)
                .to(self.s_channel.dtype)
                .contiguous()
            )
            if self.group_size == self.in_features:
                codes = codes ^ 8
                s_group = torch.empty(0, dtype=self.s_group.dtype, device=self.B.device)
            else:
                s_group = (
                    self._unpermute_scales(self.s_group, self._scale_perm_i)
                    .to(self.s_group.dtype)
                    .contiguous()
                )
            cached = (pack_hf_int4(codes), s_group, s_channel)
            self._torch_packed_cache = cached
            self._torch_packed_cache_key = key
            self._maybe_drop_checkpoint_tensors()
            return cached

    def _packed_cpu_weight_device(self):
        weight = getattr(self, "B", None)
        if weight is not None:
            return weight.device
        packed = getattr(self, "_torch_packed_cache", None)
        if getattr(self, "_torch_dropped_checkpoint", None) is not None and packed is not None:
            return packed[0].device
        return None

    def _packed_cpu_matmul(self, quant_A: torch.Tensor):
        weight_device = self._packed_cpu_weight_device()
        if (
            quant_A.device.type != "cpu"
            or weight_device is None
            or weight_device.type != "cpu"
            or self.in_features > INT8_INT32_MAX_REDUCTION
        ):
            return None
        if torch.compiler.is_compiling():
            # ``_qqq_cpu_kernel()`` resolves the extension through
            # ``importlib`` (a dynamo-skipped call) and
            # ``_packed_weight_for_cpu()`` takes the transform lock and may drop
            # the checkpoint tensors. The eager warmup publishes both, so a
            # traced graph only reads them; without a warmup it falls through
            # to the traceable ``_int8_matmul`` path.
            kernel = getattr(self, "_torch_cpu_kernel", None)
            cached = getattr(self, "_torch_packed_cache", None)
            if kernel is None or cached is None:
                return None
            packed, s_group, s_channel = cached
            if s_group.dtype != torch.float32:
                s_group = s_group.to(torch.float32)
            return kernel(quant_A.contiguous(), packed, s_group, self.group_size), s_channel
        kernel = _qqq_cpu_kernel()
        self._torch_cpu_kernel = kernel
        if kernel is None:
            return None
        packed, s_group, s_channel = self._packed_weight_for_cpu()
        if s_group.dtype != torch.float32:
            s_group = s_group.to(torch.float32)
        return kernel(quant_A.contiguous(), packed, s_group, self.group_size), s_channel

    def _apply(self, fn, recurse=True):
        # A move changes the device/dtype the cache was built for, so restore
        # the checkpoint layout before it is transformed. ``super()._apply``
        # then clears the derived cache.
        self._materialize_dropped_checkpoint()
        return super()._apply(fn, recurse=recurse)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        with self._cpu_transform_lock:
            super()._save_to_state_dict(destination, prefix, keep_vars)
            descriptors = self._torch_dropped_checkpoint
            if descriptors is None:
                return
            for name, tensor in self._rebuild_checkpoint_tensors(descriptors).items():
                destination[prefix + name] = tensor if keep_vars else tensor.detach()

    def _load_from_state_dict(self, *args, **kwargs):
        self._materialize_dropped_checkpoint()
        return super()._load_from_state_dict(*args, **kwargs)

    def _int8_matmul(self, quant_A: torch.Tensor, weight: torch.Tensor) -> Optional[torch.Tensor]:
        shape_key = (
            quant_A.device,
            int(quant_A.shape[0]),
            int(quant_A.shape[1]),
            int(weight.shape[1]),
        )
        failed_shapes = self._failed_int_mm_shapes()
        if (
            shape_key in failed_shapes
            or quant_A.device.type != "cpu"
            or quant_A.device != weight.device
            or quant_A.dtype != torch.int8
            or weight.dtype != torch.int8
            or quant_A.ndim != 2
            or weight.ndim != 2
            or not quant_A.is_contiguous()
            or not weight.is_contiguous()
            or not hasattr(torch, "_int_mm")
        ):
            return None
        try:
            reduction = int(quant_A.shape[1])
            if reduction <= INT8_INT32_MAX_REDUCTION:
                return torch._int_mm(quant_A, weight).to(torch.float32)
            # A longer contraction would wrap silently inside ``_int_mm``, so
            # the failure memo below would never see it.  Sum bounded chunks
            # in INT64 instead; each chunk stays inside the INT32 range and the
            # wider accumulator preserves the exact integer result.
            accum = None
            for start in range(0, reduction, INT8_INT32_CHUNK_REDUCTION):
                end = min(start + INT8_INT32_CHUNK_REDUCTION, reduction)
                partial = torch._int_mm(
                    quant_A[:, start:end].contiguous(),
                    weight[start:end].contiguous(),
                )
                if accum is None:
                    accum = partial.to(torch.int64)
                else:
                    accum = accum + partial.to(torch.int64)
            return accum.to(torch.float32)
        except (RuntimeError, NotImplementedError):
            # A build may not expose an integer GEMM kernel for this exact
            # shape. Remember only that shape so a later batch size/device can
            # still use the optimized path, and so a transient failure does not
            # permanently disable the module.
            failed_shapes.add(shape_key)
            return None

    def forward(self, A):
        if self.input_rows(A) == 0:
            return self.empty_linear_output(A)

        A_dtype = A.dtype
        out_shape = A.shape[:-1] + (self.out_features,)
        A_flat = A.reshape(-1, A.shape[-1])
        quant_A, s1 = self.dynamic_quant(A_flat)
        packed_result = self._packed_cpu_matmul(quant_A)
        if packed_result is not None:
            accum, s_channel = packed_result
        else:
            weight = self._int8_weight_for_forward()
            s_channel = self._unpermute_scales(
                self._checkpoint_scale("s_channel"), self._scale_perm_single_i
            ).to(torch.float32)
            accum = self._int8_matmul(quant_A, weight)
            if accum is None:
                accum = torch.matmul(
                    quant_A.to(torch.float32),
                    weight.to(device=A_flat.device, dtype=torch.float32),
                )
        accum = accum * s1
        accum = accum * s_channel.to(device=A_flat.device, dtype=torch.float32)
        if self.bias is not None:
            # The INT8 reduction and scales are still FP32 here. Add bias
            # before the output cast so a finite sum cannot overflow in FP16.
            accum = accum + self.bias.to(device=accum.device, dtype=accum.dtype)
        D = accum.to(dtype=A_dtype).reshape(out_shape)

        if self.adapter:
            D = self.adapter.apply(x=A_flat, out=D)

        return D


__all__ = ["QQQLinear", "QQQTorchLinear"]
