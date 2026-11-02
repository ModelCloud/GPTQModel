# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Native GPTQ INT4 checkpoints with per-forward FP8 weight staging."""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from ...models._const import DEVICE, PLATFORM
from ...quantization import FORMAT, METHOD
from ...utils.backend import BACKEND
from ...utils.logger import setup_logger
from . import PackableQuantLinear


log = setup_logger()


class W4AFP8Linear(PackableQuantLinear):
    SUPPORTS_BACKENDS = [BACKEND.GPTQ_W4AFP8]
    SUPPORTS_METHODS = [METHOD.GPTQ]
    # A saved activation policy selects this backend. Ordinary GPTQ loads do not.
    SUPPORTS_FORMATS = {FORMAT.GPTQ: 0, FORMAT.GPTQ_V2: 0}
    SUPPORTS_BITS = [4]
    SUPPORTS_GROUP_SIZE = [128]
    SUPPORTS_DESC_ACT = [False]
    SUPPORTS_SYM = [True]
    SUPPORTS_SHARDS = True
    SUPPORTS_TRAINING = False
    SUPPORTS_AUTO_PADDING = False
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [128]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [32]
    SUPPORTS_DEVICES = [DEVICE.CUDA]
    SUPPORTS_PLATFORM = [PLATFORM.LINUX]
    SUPPORTS_PACK_DTYPES = [torch.int32]
    SUPPORTS_ADAPTERS = []
    SUPPORTS_DTYPES = [torch.float16, torch.bfloat16]
    REQUIRES_FORMAT_V2 = True
    QUANT_TYPE = "w4afp8"
    # Native E4M3 tensor-core products start with Ada Lovelace (SM 8.9);
    # Hopper and Blackwell parts qualify as well. Subclasses with a narrower
    # contract (NVFP4) override these instead of inheriting the FP8 floor.
    MIN_COMPUTE_CAPABILITY = (8, 9)
    HARDWARE_FEATURE = "FP8"
    # This backend keeps no derived tensor between forwards: the E4M3 operand
    # is an expansion of the native INT4 codes and is rebuilt per call.
    _WEIGHT_CACHE_NAMES: Tuple[str, ...] = ()

    def __init__(self, bits: int, group_size: int, sym: bool, desc_act: bool,
                 in_features: int, out_features: int, bias: bool = False,
                 pack_dtype: torch.dtype = torch.int32, register_buffers: bool = True,
                 **kwargs):
        super().__init__(
            bits=bits, group_size=group_size, sym=sym, desc_act=desc_act,
            in_features=in_features, out_features=out_features, bias=bias,
            pack_dtype=pack_dtype, register_buffers=register_buffers,
            backend=kwargs.pop("backend", BACKEND.GPTQ_W4AFP8),
            adapter=kwargs.pop("adapter", None), **kwargs,
        )
        # Readiness marker only: no tensor is retained, so nothing to move,
        # serialize, or invalidate lazily. `pack()`/`load` clear it and
        # `post_init()` restores it.
        self._w4a_weights_ready = False
        # Version counters of qweight/qzeros/g_idx at the last validation.
        self._w4a_layout_token = None

    @classmethod
    def validate_once(cls) -> Tuple[bool, Optional[Exception]]:
        try:
            import triton  # noqa: F401

            from . import w4a_triton  # noqa: F401
        except (ImportError, OSError) as exc:
            return False, RuntimeError(f"{cls.__name__} needs Triton with FP8 support: {exc}")
        return True, None

    @classmethod
    def _validate_compute_capability(cls, ordinal: int) -> Tuple[bool, Optional[Exception]]:
        capability = torch.cuda.get_device_capability(ordinal)
        if capability < cls.MIN_COMPUTE_CAPABILITY:
            required = ".".join(str(part) for part in cls.MIN_COMPUTE_CAPABILITY)
            actual = ".".join(str(part) for part in capability)
            return False, RuntimeError(
                f"{cls.__name__} needs a CUDA device with {cls.HARDWARE_FEATURE} tensor cores "
                f"(compute capability {required}+); device {ordinal} reports {actual}."
            )
        return cls.cached_validate_once()

    @classmethod
    def validate(cls, **args) -> Tuple[bool, Optional[Exception]]:
        ok, err = cls._validate(**args)
        if not ok:
            return ok, err
        device = args.get("device")
        if not torch.cuda.is_available():
            return False, RuntimeError(
                f"{cls.__name__} needs a CUDA device with {cls.HARDWARE_FEATURE} tensor cores."
            )
        if isinstance(device, torch.device):
            ordinal = device.index if device.index is not None else torch.cuda.current_device()
        else:
            ordinal = torch.cuda.current_device()
        return cls._validate_compute_capability(ordinal)

    def pack(self, linear, scales, zeros, g_idx, **kwargs):
        self._invalidate_weight_cache()
        super().pack(linear, scales, zeros, g_idx, **kwargs)
        # pack() stores logical zero points. Saved GPTQ-v1 conversion is a
        # separate serialization step, reversed by the loader before post_init.
        self.qzero_format(2)

    def _load_from_state_dict(self, *args, **kwargs):
        self._invalidate_weight_cache()
        super()._load_from_state_dict(*args, **kwargs)

    def _invalidate_weight_cache(self):
        self._w4a_weights_ready = False
        # Version counters of qweight/qzeros/g_idx at the last validation.
        self._w4a_layout_token = None
        for name in self._WEIGHT_CACHE_NAMES:
            setattr(self, name, getattr(self, name).new_empty(0))

    def invalidate_weight_layout(self) -> None:
        """Drop the cached layout validation and require a fresh ``post_init()``.

        Inference tensors (anything built inside ``torch.inference_mode()``) have
        no version counter, so an in-place write to ``qweight``/``qzeros``/``g_idx``
        cannot be observed by ``_weight_layout_token()``. Callers that mutate
        quantized weights after ``post_init()`` must either leave inference mode
        or call this method; ``forward()`` then raises until ``post_init()`` runs
        again rather than decoding a layout that is no longer known to be valid.
        """
        self._invalidate_weight_cache()

    def _apply(self, fn, recurse=True):
        # These buffers encode hardware operands, not model-precision values.
        # Byte views let Module.to()/half()/bfloat16() move them without casting
        # FP8/FP4 codes or requiring unsupported FP4 numeric conversions. Native
        # GPTQ scales and ordinary parameters still follow the requested dtype.
        cache_dtypes = {id(self._buffers[name]): self._buffers[name].dtype
                        for name in self._WEIGHT_CACHE_NAMES}

        def preserve_weight_format(tensor):
            cache_dtype = cache_dtypes.get(id(tensor))
            if cache_dtype is not None:
                return fn(tensor.view(torch.uint8)).view(cache_dtype)
            return fn(tensor)

        result = super()._apply(preserve_weight_format, recurse=recurse)
        # A device/dtype move relocates the native tensors but does not change
        # the INT4 layout, so the cached validation stays valid; refresh the
        # identity-based token instead of forcing a revalidation.
        if self._w4a_weights_ready:
            self._w4a_layout_token = self._weight_layout_token()
        return result

    @torch.no_grad()
    def post_init(self):
        super().post_init()
        # Fail early on an unloaded or unsupported checkpoint instead of at the
        # first forward. No derived tensor is built here; see
        # `fp8_weight_operand()`.
        self._check_native_weight_layout()
        self._w4a_layout_token = self._weight_layout_token()
        self._w4a_weights_ready = True

    @classmethod
    def _group_zero_points(cls, module) -> torch.Tensor:
        """Logical GPTQ zero points, one row per 128-row group ([G, out]).

        A classmethod taking the module explicitly so the generic decoders can
        be reused on any GPTQ module (for example a ``TorchLinear`` checkpoint
        dump), not only on an instance of this class.
        """
        shifts = (torch.arange(8, device=module.qzeros.device, dtype=torch.int32) * 4)
        zeros = ((module.qzeros.unsqueeze(2) >> shifts.view(1, 1, 8)) & 15).reshape(
            module.in_features // 128, module.out_features
        )
        if module.qzero_format() == 1:
            zeros = (zeros + 1) & 15
        return zeros

    def _check_native_weight_layout(self):
        """Validate the supported GPTQ layout once, at post_init().

        Deliberately not called from `forward()`: `torch.equal` and the
        zero-point reduction both synchronize the device, so this stays off the
        decode path. Readiness is keyed on the checkpoint tensors' version
        counters (see `_weights_are_current`), so an in-place mutation or a
        replacement of ``qweight``/``qzeros``/``g_idx`` still invalidates it
        instead of letting a stale layout through.
        """
        if self.qweight.device.type == "meta":
            raise ValueError(f"{self.__class__.__name__} post_init requires loaded GPTQ weights.")
        expected_g_idx = torch.arange(self.in_features, device=self.g_idx.device, dtype=torch.int32) // 128
        if not torch.equal(self.g_idx, expected_g_idx):
            raise ValueError("W4AFP8 currently requires contiguous GPTQ groups and desc_act=False.")
        if not bool((self._group_zero_points(self) == 8).all()):
            raise ValueError("W4AFP8 first release requires symmetric GPTQ zero point 8 in every group.")

    def _weight_layout_token(self):
        """Identity + version of every tensor that defines the INT4 layout.

        Inference tensors have no version counter, so fall back to identity
        alone for those; ``pack()``/``_load_from_state_dict()`` still clear
        readiness explicitly, and an ordinary in-place write is caught by the
        version counter on the tensors that do track one.
        """
        parts = []
        for value in (self.qweight, self.qzeros, self.g_idx):
            try:
                version = value._version
            except RuntimeError:
                # Inference tensors deliberately have no version counter.
                version = None
            parts.append((value.data_ptr(), version))
        return tuple(parts)

    def _weights_are_current(self) -> bool:
        """True when the cached layout validation still describes the weights."""
        return (self._w4a_weights_ready
                and self._w4a_layout_token == self._weight_layout_token())

    def _layout_token_is_observable(self) -> bool:
        """True when every layout-defining tensor tracks in-place writes.

        Inference tensors carry no version counter, so their token entry can
        never change; a stale token would then look current forever. Such a
        module must be revalidated explicitly instead of trusted.
        """
        return all(version is not None for _, version in self._weight_layout_token())

    def _warn_unobservable_layout(self) -> None:
        """Warn once that in-place weight writes cannot be detected here.

        Revalidating on every forward would put the ``torch.equal``/all-reduce
        sync back on the decode path, so the unobservable case is reported once
        and ``invalidate_weight_layout()`` is offered as the explicit hook.
        """
        if getattr(self, "_w4a_unobservable_warned", False):
            return
        self._w4a_unobservable_warned = True
        log.warning(
            "%s holds inference tensors, which have no version counter. In-place "
            "writes to qweight/qzeros/g_idx will not be detected; call "
            "invalidate_weight_layout() after mutating quantized weights outside "
            "inference_mode building.",
            self.__class__.__name__,
        )

    def _centered_int4_codes(self) -> torch.Tensor:
        """Read native GPTQ packing without changing the saved weight format.

        This is a *decode only*: it must not validate. ``_check_native_weight_layout``
        contains ``torch.equal`` and an all-reduce, which synchronize the device,
        so calling it here would put a sync on every decode step (measured on RTX
        4090: 69-78 us for the check vs 39-47 us for the identical decode on the
        small and mid projections). Validation happens once in ``post_init()``
        and is re-armed by ``_weights_are_current`` when the checkpoint tensors
        change.

        Two layers of cost matter here, and both are paid per forward.

        *Whole-layer, not per-group.* A per-128-row-group loop issues roughly
        seven Torch dispatches per group, so a K=8192 projection queues ~448
        kernels per token; on RTX 4090 that loop was CPU-launch-bound at ~4.3 us
        per launch with the device idle. The signature is that the loop cost is
        flat in the layer size: 535 us for a 16-group layer and 2.30 ms for the
        64-group K=8192 layer, i.e. 10-14x a whole-layer decode on the smaller
        projections. Decoding the layer in one shot is byte-identical with a
        handful of launches.

        *Bytes, not int32 planes.* The packed word is 32 bits holding eight
        nibbles, so unpacking through int32 materializes a 4 B/weight plane for
        each of the shift and mask steps and moves ~337 MiB per K=8192 layer
        (4090 effective bandwidth is ~940 GB/s). That is a bandwidth cost, not a
        launch cost, so it grows with the weight count. Reading the same words
        through a ``uint8`` view keeps the intermediates at 1 B/weight instead
        (~157 MiB per K=8192 layer). Measured on a quiet RTX 4090 with both forms
        interleaved, at 2048x8192: 374.7 us -> 175.1 us (2.14x) and at
        8192x2048: 374.6 us -> 175.3 us (2.14x), byte-identical output, and the
        transient peak falls from 8 B/weight to exactly 5 B/weight.

        This Torch form is no longer the inference path on CUDA. It is the
        reference the fused kernel is pinned against and the fallback for CPU
        and meta weights: ``fp8_weight_operand`` decodes through
        ``w4a_triton.unpack_int4_to_e4m3`` when the packed words live on a CUDA
        device, and that kernel reads ``qweight`` and writes E4M3 in one pass,
        so no intermediate plane is allocated at all (1 B/weight peak instead
        of 5). Keep this decode exact and readable; it is the oracle.

        Validation only runs for instances of this class, so the checkpoint
        audits can keep calling it on a ``TorchLinear``:
        ``W4AFP8Linear._centered_int4_codes(TorchLinear)``.

        GPTQ packs row ``8*i + j`` of a group into nibble ``j`` of word ``i``
        (low nibble first), so widening the word to its four bytes and taking
        low/high nibbles of each byte in little-endian order reproduces the
        ``>> (4*j) & 15`` sequence exactly.
        """
        owned = isinstance(self, W4AFP8Linear)
        # Call the module-level unpacker, not ``self.``: this method is invoked as
        # ``W4AFP8Linear._centered_int4_codes(torch_linear_module)`` by the
        # checkpoint audits, and a ``TorchLinear`` has no ``_unpack_int4_codes``.
        codes = unpack_int4_codes(self.qweight)
        if owned:
            # post_init()/`_weights_are_current()` already proved every zero
            # point is 8. Re-checking here would be a device sync on every
            # decode step, so subtract the scalar unconditionally.
            return codes.to(torch.int16) - 8
        # A foreign GPTQ module (checkpoint audit) has no helper of ours and has
        # not been through post_init(), so decode its zero points inline.
        # ``qzeros`` packs its eight nibbles along the *output* axis
        # (``[in/128, out/8]`` -> ``[in/128, out]``), unlike ``qweight`` whose
        # nibbles run along the input axis, so it keeps the shift form. It is
        # 1/128th the size of ``qweight`` and not on any hot path.
        shifts = torch.arange(8, device=self.qzeros.device, dtype=torch.int32) * 4
        zeros = ((self.qzeros.unsqueeze(2) >> shifts.view(1, 1, 8)) & 15).reshape(
            self.in_features // 128, self.out_features
        )
        if self.qzero_format() == 1:
            zeros = (zeros + 1) & 15
        return codes.to(torch.int16) - zeros.repeat_interleave(
            128, dim=0
        ).to(torch.int16)

    @staticmethod
    def _unpack_int4_codes(packed: torch.Tensor) -> torch.Tensor:
        """See :func:`unpack_int4_codes`; kept as a class-level alias."""
        return unpack_int4_codes(packed)

    def _require_current_weights(self) -> None:
        """Revalidate a stale layout, or explain why decoding must not proceed."""
        if not self._weights_are_current():
            if self._w4a_weights_ready:
                # Weights changed in place after post_init(); revalidate rather
                # than decoding a layout that is no longer known to be valid.
                self._check_native_weight_layout()
                self._w4a_layout_token = self._weight_layout_token()
            else:
                raise RuntimeError(
                    "W4AFP8 weights are not initialized; call post_init() after loading or packing."
                )
        elif not self._layout_token_is_observable():
            self._warn_unobservable_layout()

    @torch.no_grad()
    def fp8_weight_operand(self) -> torch.Tensor:
        """Materialize the E4M3 weight operand for exactly one forward.

        Centered INT4 codes are exactly representable in E4M3, so this step is
        lossless. The returned tensor is an expansion of the native INT4 codes:
        callers must pass it straight to the kernel and drop it when that
        launch is ordered on the stream. It is never stored on the module, so
        nothing here touches module state or synchronizes the device.

        On CUDA the decode runs as one fused Triton kernel that reads packed
        words and writes E4M3 directly, so no intermediate plane exists at all.
        The Torch decode remains the CPU fallback and the reference the kernel
        is pinned against.
        """
        self._require_current_weights()
        if self.qweight.device.type == "cuda" and self.qweight.dtype == torch.int32:
            from .w4a_triton import unpack_int4_to_e4m3

            return unpack_int4_to_e4m3(
                self.qweight,
                torch.empty(
                    (self.in_features, self.out_features),
                    device=self.qweight.device,
                    dtype=torch.float8_e4m3fn,
                ),
            )
        return self._centered_int4_codes().to(torch.float8_e4m3fn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not getattr(self, "_w4a_weights_ready", False):
            raise RuntimeError("W4AFP8 weights are not initialized; call post_init() after loading or packing.")
        if not self._weights_are_current():
            raise RuntimeError(
                "W4AFP8 weights changed since post_init(); call post_init() again before inference."
            )
        if not self._layout_token_is_observable():
            self._warn_unobservable_layout()
        from .w4a_activation import W4AActivation, pack_activation
        from .w4a_triton import fp8_linear, fp8_linear_prepacked

        if isinstance(x, W4AActivation):
            if x.mode != "w4afp8" or x.shape[-1] != self.in_features:
                raise ValueError("W4AFP8 requires a matching scale-aware FP8 activation.")
            needs_rotation = bool(self.online_full_had or self.online_partial_had)
            if x.rotation_applied and not needs_rotation:
                raise ValueError("This FP8 operand was rotated for a different Linear contract.")
            if needs_rotation and not x.rotation_applied:
                rotated = self._apply_rotation_to_input(x.decode(torch.float32))
                x = pack_activation(rotated, "w4afp8", model_dtype=x.model_dtype)
            weight = self.fp8_weight_operand()
            result = fp8_linear_prepacked(
                x.codes, x.scales, weight, self.scales, self.bias, torch.float32,
            )
            result = result.reshape(x.shape[:-1] + (self.out_features,))
            return result.to(x.model_dtype)
        if getattr(self, "_require_activation_stream", False):
            raise TypeError("This W4AFP8 Linear requires encoded FP8 input with token scales.")

        x = self._apply_rotation_to_input(x)
        weight = self.fp8_weight_operand()
        return fp8_linear(x, weight, self.scales, self.bias)


def unpack_int4_codes(packed: torch.Tensor) -> torch.Tensor:
    """Expand packed GPTQ nibble words into one code per row, as uint8.

    ``packed`` is ``[in/8, out]`` for ``qweight`` and ``[in/128/8, out]``
    for ``qzeros``; both hold eight 4-bit codes per int32 word, low nibble
    first. The result is ``[rows, out]`` in ``[0, 15]``.

    Going through a ``uint8`` view instead of shifting the int32 values
    keeps every intermediate at 1 B/weight. The int32 form materializes a
    4 B/weight plane for both the shift and the mask, which dominates the
    per-forward staging cost on large projections and is pure bandwidth --
    see ``_centered_int4_codes`` for the measured split.
    """
    words, out = packed.shape
    # One int32 word is four bytes holding eight nibbles, low nibble first.
    # Little-endian byte order is exactly the nibble order of the word, so
    # byte 0 carries nibbles 0 and 1, byte 1 carries nibbles 2 and 3, ...
    viewed = packed.view(torch.uint8).reshape(words, out, 4)
    nibbles = torch.stack((viewed & 0x0F, viewed >> 4), dim=-1).reshape(
        words, out, 8
    )
    # [word, out, nibble] -> [word, nibble, out] -> [rows, out].
    return nibbles.permute(0, 2, 1).reshape(words * 8, out)
