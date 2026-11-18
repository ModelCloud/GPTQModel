# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
#
# Portions of this file are adapted from turboderp-org/exllamav3.
# Credits: TurboDerp / ExLlamaV3 contributors.

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any, Dict, Optional

import torch
import torch.nn as nn

from ..exllamav3.trellis import validate_trellis_k
from ..models._const import DEVICE, normalize_device
from . import exllamav3_bf16_kernels as _bf16_kernels


# Module-level names so a test (or a caller) can swap the fused ops for the
# eager reference; ``forward`` reads them through this module's globals.
_FP16_SAFE_AMAX = _bf16_kernels.FP16_SAFE_AMAX
_try_fused_prescale = _bf16_kernels.try_prescale
_try_fused_rescale = _bf16_kernels.try_rescale


if TYPE_CHECKING:
    from ..exllamav3.modules.quant.exl3 import LinearEXL3


_EXL3_BUFFER_NAMES = ("trellis", "suh", "svh", "su", "sv", "bias", "mcg", "mul1")

# The native EXL3 kernels accept FP16 only. A BF16 activation inside FP16's
# finite range converts exactly, so the default BF16 checkpoint can use them.
# The Hadamard rotation of a 128-wide block is bounded by sqrt(128) times the
# input maximum, so a row maximum at or below 65504 / sqrt(128) survives the
# rotation without overflowing FP16. Rows at or below the bound are passed
# through untouched, so ordinary activations are bit-exact with the plain FP16
# path, and a row that carries one massive activation is divided only by the
# power of two it needs (#4870): normalizing it to <= 1 would push its own
# small elements into FP16 subnormals. The bound itself lives with the fused
# kernel that consumes it (``exllamav3_bf16_kernels``).


def _torch_dtype(value: Any) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
    if isinstance(value, str):
        return getattr(torch, value)
    raise TypeError(f"Unsupported torch dtype value: {value!r}")


class ExllamaV3Linear(nn.Module):
    QUANT_TYPE = "exl3"
    # The trellis codes and sign vectors are the EXL3 payload. The sign
    # vectors are stored either unpacked (`suh`/`svh`) or packed (`su`/`sv`),
    # and the codebook is `mcg` on one recipe and `mul1` on another. A module
    # built for a recipe owns only one side of each pair, and the validator
    # skips the absent suffix.
    CHECKPOINT_REQUIRED_BUFFERS = (
        "trellis",
        "suh",
        "svh",
        "su",
        "sv",
        "mcg",
        "mul1",
    )
    # Every floating EXL3 buffer (the fp16 `suh`/`svh` sign vectors and the
    # bias) loads at its checkpoint dtype; the Hadamard kernels require fp16.
    MODEL_DTYPE_BUFFERS = ()
    SUPPORTS_SHARDS = True
    SUPPORTS_DEVICES = [DEVICE.CUDA]

    @classmethod
    def validate_device(cls, device) -> None:
        family = normalize_device(device)
        if family not in cls.SUPPORTS_DEVICES:
            raise NotImplementedError(
                "EXL3_EXLLAMA_V3 inference requires NVIDIA CUDA, but the module is on "
                f"`{getattr(family, 'value', family)}`; move the model to a CUDA device "
                "(or use EXL3_TORCH on ROCm or CPU)."
            )

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        name: str,
        tensor_storage: Optional[Dict[str, Any]] = None,
        tensors: Optional[Dict[str, torch.Tensor]] = None,
        out_dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.name = name
        self.out_dtype = out_dtype
        self.tensor_storage = tensor_storage or {}

        self.weight = torch.zeros((1,), dtype=torch.float16, device="meta")
        self._inner: Optional["LinearEXL3"] = None
        self._inner_signature: Optional[tuple[Any, ...]] = None
        self._torch_inner = None
        self._torch_inner_signature: Optional[tuple[Any, ...]] = None
        # Serializes handle publication and teardown. `_ensure_inner` used to
        # drop the old handle and re-read `self._inner` after building the new
        # one, so a concurrent first forward could be handed the `None` another
        # thread had just published (issue #4793). Per module, not process-wide:
        # this runs on every forward.
        self._inner_guard = threading.RLock()

        if tensors is not None:
            for buffer_name in _EXL3_BUFFER_NAMES:
                tensor = tensors.get(buffer_name)
                if tensor is None:
                    setattr(self, buffer_name, None)
                else:
                    self.register_buffer(buffer_name, tensor)
            return

        stored_tensors = (self.tensor_storage or {}).get("stored_tensors", {})
        for buffer_name in _EXL3_BUFFER_NAMES:
            metadata = stored_tensors.get(f"{name}.{buffer_name}")
            if metadata is None:
                setattr(self, buffer_name, None)
                continue

            shape = tuple(metadata["shape"])
            dtype = _torch_dtype(metadata["torch_dtype"])
            self.register_buffer(buffer_name, torch.empty(shape, dtype=dtype, device="meta"))

    @classmethod
    def from_tensors(
        cls,
        *,
        in_features: int,
        out_features: int,
        name: str,
        tensors: Dict[str, torch.Tensor],
    ) -> "ExllamaV3Linear":
        return cls(
            in_features=in_features,
            out_features=out_features,
            name=name,
            tensors=tensors,
        )

    def _bias_version(self) -> Optional[int]:
        """Version counter of the logical bias owned by this module.

        ``_ensure_inner`` hands ``LinearEXL3`` a *widened copy* of ``bias`` when
        ``out_features`` is padded, so the kernel no longer aliases the live
        parameter. In-place writes (``bias.add_``, adapter merges, scale
        folding) would then be invisible. ``Tensor._version`` bumps on every
        in-place mutation, so it cheaply distinguishes "alias" from "stale copy".
        A bias loaded under ``torch.inference_mode`` has no version counter;
        copy it once outside that mode so subsequent mutations remain visible.
        """
        bias = getattr(self, "bias", None)
        if bias is None or bias.device.type == "meta":
            return None
        if bias.is_inference():
            with torch.inference_mode(False), torch.no_grad():
                bias = bias.to(device=bias.device, copy=True)
                self.bias = bias
        return int(bias._version)

    def _normalize_inference_metadata(self) -> None:
        """Make metadata captured by the native handle observable after edits.

        ``LinearEXL3`` unpacks ``su``/``sv`` into separate sign vectors when it
        is built. Inference tensors have no version counter, so an in-place edit
        to those packed fields would otherwise keep the old vectors alive.
        Rebind only the small metadata buffers as normal, versioned tensors;
        the packed trellis remains aliased by the native handle and is never
        duplicated here.
        """
        for name in ("suh", "svh", "su", "sv", "mcg", "mul1"):
            tensor = getattr(self, name, None)
            if tensor is not None and tensor.device.type != "meta" and tensor.is_inference():
                with torch.inference_mode(False), torch.no_grad():
                    setattr(self, name, tensor.to(device=tensor.device, copy=True))

    def _current_signature(self) -> tuple[Any, ...]:
        trellis = getattr(self, "trellis", None)
        if trellis is None or trellis.device.type == "meta":
            return ("meta",)

        self._normalize_inference_metadata()
        bias_version = self._bias_version()
        signature: list[Any] = [str(trellis.device)]
        for buffer_name in _EXL3_BUFFER_NAMES:
            tensor = getattr(self, buffer_name, None)
            if tensor is None:
                signature.append(None)
                continue
            try:
                version = None if tensor.is_inference() else tensor._version
            except RuntimeError:
                version = None
            signature.append((
                tensor.data_ptr(),
                tuple(tensor.shape),
                str(tensor.dtype),
                version,
            ))
        # Only the padded path copies the bias, so only that path needs liveness.
        signature.append(bias_version)
        return tuple(signature)

    def _drop_inner(self) -> None:
        with self._inner_guard:
            self._drop_inner_locked()

    def _drop_inner_locked(self) -> None:
        if self._inner is not None:
            try:
                self._inner.unload()
            except Exception:
                # `_drop_inner` runs during teardown and `_apply`; cleanup must stay best-effort.
                pass
        self._inner = None
        self._inner_signature = None
        self._torch_inner = None
        self._torch_inner_signature = None

    def clear_weight_cache(self) -> None:
        """Release derived handles after an unversioned external tensor edit."""
        self._drop_inner()

    def _stored_features(self) -> tuple[int, int]:
        """Infer storage geometry; architecture dimensions remain logical."""
        trellis = getattr(self, "trellis", None)
        if trellis is None or trellis.ndim != 3:
            raise ValueError(f"EXL3 module `{self.name}` has invalid trellis geometry")
        try:
            validate_trellis_k(
                int(trellis.shape[-1]) // 16,
                getattr(self, "mcg", None) is not None,
                getattr(self, "mul1", None) is not None,
            )
        except ValueError as exc:
            raise ValueError(f"EXL3 module `{self.name}`: {exc}") from exc
        inputs, outputs = int(trellis.shape[0]) * 16, int(trellis.shape[1]) * 16
        padded_outputs = ((self.out_features + 127) // 128) * 128
        if (inputs != self.in_features or outputs not in (self.out_features, padded_outputs)
                or outputs < self.out_features):
            raise ValueError(f"EXL3 module `{self.name}` storage differs from logical dimensions")
        for suffix, expected in (("suh", inputs), ("svh", outputs)):
            scale = getattr(self, suffix, None)
            if scale is not None and tuple(scale.shape) != (expected,):
                raise ValueError(f"EXL3 module `{self.name}` has invalid {suffix} geometry")
        return inputs, outputs

    def _coerce_buffer_dtypes(self) -> None:
        """Restore the dtypes the EXL3 kernels require after a model-wide cast.

        ``nn.Module._apply`` converts every floating-point buffer to the target
        model dtype, so loading a bf16 checkpoint turns the unpacked sign vectors
        ``suh``/``svh`` into bf16. ``LinearEXL3`` asserts they are ``torch.half``
        because the Hadamard kernels are typed for fp16, which made a bf16 load
        fail with ``AssertionError: suh is wrong datatype``. The packed ``su``/``sv``
        are int16 and are untouched by the cast.
        """
        for buffer_name in ("suh", "svh"):
            tensor = getattr(self, buffer_name, None)
            if tensor is not None and tensor.is_floating_point() and tensor.dtype != torch.half:
                setattr(self, buffer_name, tensor.half())

    def _ensure_inner(self) -> "LinearEXL3":
        with self._inner_guard:
            return self._ensure_inner_locked()

    def _ensure_inner_locked(self) -> "LinearEXL3":
        trellis = getattr(self, "trellis", None)
        if trellis is None:
            raise RuntimeError(f"EXL3 module `{self.name}` is missing `trellis`.")
        if trellis.device.type == "meta":
            raise RuntimeError(f"EXL3 module `{self.name}` has not been materialized from checkpoint tensors yet.")
        self.validate_device(trellis.device)
        self._coerce_buffer_dtypes()

        from ..exllamav3.modules.quant.exl3 import LinearEXL3

        signature = self._current_signature()
        if self._inner is not None and signature == self._inner_signature:
            return self._inner

        self._drop_inner()
        stored_inputs, stored_outputs = self._stored_features()
        bias = getattr(self, "bias", None)
        if bias is not None and stored_outputs != self.out_features:
            if tuple(bias.shape) != (self.out_features,):
                raise ValueError(f"EXL3 module `{self.name}` has invalid logical bias geometry")
            bias = torch.nn.functional.pad(bias, (0, stored_outputs - self.out_features))
        self._inner = LinearEXL3(
            config=None,
            in_features=stored_inputs,
            out_features=stored_outputs,
            scale=None,
            su=getattr(self, "su", None),
            sv=getattr(self, "sv", None),
            suh=getattr(self, "suh", None),
            svh=getattr(self, "svh", None),
            trellis=trellis,
            mcg=getattr(self, "mcg", None),
            mul1=getattr(self, "mul1", None),
            bias=bias,
            out_dtype=self.out_dtype,
            transformers_fix=True,
            key=self.name,
        )
        inner = self._inner
        self._inner_signature = signature
        return inner

    def _ensure_torch_inner(self):
        """Build the bounded FP32 reconstruction path for wide inputs."""
        with self._inner_guard:
            return self._ensure_torch_inner_locked()

    def _needs_torch_fallback(self) -> bool:
        """Whether this module must reconstruct through the portable kernel.

        The native EXL3 kernels are compiled for CUDA only
        (``SUPPORTS_DEVICES``). A materialized trellis on any other device
        cannot run them, so the forward uses the same bounded FP32
        reconstruction that FP32 and off-CUDA BF16 already use. An absent or
        meta trellis is left to the native path, which reports that state with
        its existing message.
        """
        trellis = getattr(self, "trellis", None)
        return trellis is not None and trellis.device.type not in ("cuda", "meta")

    def _ensure_torch_inner_locked(self):
        from .exllamav3_torch import ExllamaV3TorchLinear

        signature = self._current_signature()
        if (
            self._torch_inner is not None
            and signature == self._torch_inner_signature
        ):
            return self._torch_inner

        tensors = {
            name: getattr(self, name, None)
            for name in _EXL3_BUFFER_NAMES
        }
        torch_inner = ExllamaV3TorchLinear(
            in_features=self.in_features,
            out_features=self.out_features,
            name=self.name,
            tensors=tensors,
            out_dtype=self.out_dtype,
        )
        # This helper borrows our checkpoint buffers. Registering it as a child
        # would serialize the same packed tensors twice and make strict reload
        # depend on whether a wide-input forward has run. _apply and
        # clear_weight_cache explicitly discard it before its buffers can move.
        object.__setattr__(self, "_torch_inner", torch_inner)
        self._torch_inner_signature = signature
        return torch_inner

    def __getstate__(self):
        # A lock cannot cross a copy or a process; the receiver recreates it.
        state = super().__getstate__()
        state.pop("_inner_guard", None)
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        self._inner_guard = threading.RLock()
        # The native handle belongs to the source module's process and may be
        # shared with a shallow copy, so never unload it here - just forget it.
        self._inner = None
        self._inner_signature = None
        self._torch_inner = None
        self._torch_inner_signature = None

    def post_init(self) -> None:
        self._drop_inner()
        self._coerce_buffer_dtypes()
        if getattr(self, "trellis", None) is not None and self.trellis.device.type != "meta":
            self._ensure_inner()

    def _apply(self, fn, recurse=True):
        self._drop_inner()
        result = super()._apply(fn, recurse=recurse)
        self._coerce_buffer_dtypes()
        return result

    def _native_bf16_input(
        self, x: torch.Tensor
    ) -> Optional[tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """Map a BF16 activation onto the FP16-only native kernel contract.

        BF16's 8-bit significand is a subset of FP16's 11-bit one, so every BF16
        value inside FP16's finite range converts exactly. Above ``_FP16_SAFE_AMAX`` a
        row is divided by the smallest power of two that brings it back inside
        the bound, and the kernel output row is multiplied back. The Hadamard
        rotation and the GEMM accumulation stay inside FP16's range, and
        power-of-two scaling is exact, so only the FP16 cast rounds.

        The scale is per row, not per tensor: a single outlier token (the BOS
        massive activation on a prefill, for example) would otherwise divide
        every other row of the batch by the same power of two and push them
        into FP16 subnormals, which loses most of their precision (#4870).
        Rows already inside the safe range keep a scale of 1 and stay bit-exact.

        Returns ``(fp16_input, output_scale)``, where ``output_scale`` is
        ``None`` only for an empty batch. Otherwise the kernel result must have
        the bias removed, be multiplied by ``output_scale``, and have the bias
        restored. Rows that need no scaling carry a scale of exactly 1, so the
        rescale is the identity for them. Returns ``None`` for non-finite
        activations so they keep the FP32 Torch reconstruction path.

        The whole decision is computed on the device: there is no
        ``float(amax)``/``.item()``/``bool()`` of the activation per forward,
        which used to drain the CUDA stream once per linear per decoded token
        (#4955). The one remaining host read is the ``isfinite`` warmup probe
        below, paid once per module instead of once per forward.

        Once the module is warm, a single fused kernel launch performs the
        whole mapping (#5148). The eager sequence below stays as the reference
        and as the fallback for a non-CUDA, non-contiguous or unsupported-dtype
        input and for a build without the fused ops; both produce
        bit-identical results.
        """
        if x.numel() == 0:
            # A zero-row batch has no inf-norm identity; LinearEXL3 already
            # returns an empty output for it, so no scaling is needed.
            return x.to(torch.float16), None
        if not getattr(self, "_native_bf16_inputs_finite", False):
            # Warmup probe, once per module: a per-forward ``isfinite`` is the
            # same device->host sync this method just stopped doing. A module
            # whose first BF16 activation is non-finite keeps the exact FP32
            # reconstruction and probes again on its next forward.
            row_amax = self._row_amax(x)
            if not bool(torch.isfinite(row_amax).all()):
                return None
            self._native_bf16_inputs_finite = True
            return self._prescale_from_row_amax(x, row_amax)

        fused = _try_fused_prescale(x)
        if fused is not None:
            return fused
        return self._prescale_from_row_amax(x, self._row_amax(x))

    def _row_amax(self, x: torch.Tensor) -> torch.Tensor:
        """Per-row max-abs of ``x`` as FP32, reduced on the device (#4955)."""
        # ``vector_norm(inf)`` is the max-abs reduction without materializing
        # an ``abs`` copy, and it is the cheapest form on CUDA. Reducing per row
        # keeps the decision on the device; a single tensor-wide scalar would
        # have to be read back to the host to be compared (#4955).
        return torch.linalg.vector_norm(
            x.detach(), ord=float("inf"), dim=-1, keepdim=True
        ).float()

    def _prescale_from_row_amax(
        self, x: torch.Tensor, row_amax: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Eager reference/fallback for the fused pre-scale (#5148)."""
        # ``frexp(row_amax / bound)`` yields ``row_amax / bound < 2**exponent``,
        # so the scaled maximum lies in ``(bound / 2, bound]``: still a normal
        # FP16 number, with the most headroom left for the row's small elements.
        # Rows that do not need it keep 1.0, so only the wide rows change.
        _, exponent = torch.frexp(row_amax / _FP16_SAFE_AMAX)
        # ``pow(2, exponent)`` rather than ``ldexp``: ``torch.ldexp`` launches
        # on the *current* device and faults with an illegal memory access when
        # its inputs live on another GPU, which is the normal state for a
        # ``device_map`` split or an explicit ``cuda:1`` placement (#5263).
        # ``2**exponent`` is exact for every integer exponent here, and only
        # the rows above the FP16 bound reach this value.
        output_scale = torch.where(
            row_amax > _FP16_SAFE_AMAX,
            torch.pow(2.0, exponent).to(row_amax.dtype),
            torch.ones_like(row_amax),
        )
        return (x / output_scale).to(torch.float16), output_scale

    def _rescale_native_output(
        self, result: torch.Tensor, output_scale: torch.Tensor
    ) -> torch.Tensor:
        """Eager reference/fallback re-scale (#5148).

        Undoes ``_native_bf16_input``'s pre-scale, keeping the bias unscaled.
        The fused path writes BF16 directly; this returns FP32.
        """
        bias = getattr(self, "bias", None)
        output = result.to(torch.float32)
        if bias is None:
            return output * output_scale
        bias = bias[:self.out_features].to(torch.float32)
        return (output - bias) * output_scale + bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        if self._needs_torch_fallback():
            # The native kernels are CUDA-only. After `quantize()` the packed
            # layers stay on CPU until the caller moves the model, and a
            # checkpoint loaded for a portable device also keeps them off CUDA.
            # The portable kernel reads the same trellis/suh/svh buffers, so
            # reconstruct there instead of raising from `_ensure_inner` (#4789).
            # BF16 off CUDA already used this path, so its behaviour is
            # unchanged.
            return self._ensure_torch_inner().forward(x)
        if input_dtype == torch.float16:
            inner = self._ensure_inner()
            result = inner.forward(
                x.to(dtype=torch.float16),
                {},
                out_dtype=torch.float16,
            )
            return result[..., :self.out_features].to(input_dtype)

        # BF16 is the default dtype of most EXL3 checkpoints. Keep it on the
        # native kernels instead of the ~20-200x slower Torch reconstruction
        # (#4767); the pre-scale keeps #3731's wide-input guarantee.
        if input_dtype == torch.bfloat16 and x.is_cuda:
            prepared = self._native_bf16_input(x)
            if prepared is not None:
                inner = self._ensure_inner()
                native_input, output_scale = prepared
                native_result = inner.forward(
                    native_input,
                    {},
                    out_dtype=torch.float16,
                )
                if output_scale is not None:
                    fused = _try_fused_rescale(
                        native_result,
                        output_scale,
                        getattr(self, "bias", None),
                        self.out_features,
                    )
                    if fused is not None:
                        # Already BF16: the trailing ``.to`` is a no-op.
                        return fused.to(input_dtype)
                    result = self._rescale_native_output(
                        native_result[..., :self.out_features], output_scale
                    )
                else:
                    result = native_result[..., :self.out_features]
                return result.to(input_dtype)

        # FP32 (and non-finite BF16) keep the bounded FP32 reconstruction: a
        # cast would silently drop precision, and the wide path is exact.
        return self._ensure_torch_inner().forward(x)

    def _multiplier_value(self, name: str) -> Optional[int]:
        tensor = getattr(self, name, None)
        if tensor is None:
            return None
        if tensor.device.type == "meta":
            stored = self.tensor_storage.get(f"{name}_multiplier")
            return int(stored) if stored is not None else None
        return int(tensor.view(torch.uint32).item())

    def tensor_storage_entry(self) -> Dict[str, Any]:
        stored_tensors: Dict[str, Dict[str, Any]] = {}
        for buffer_name in _EXL3_BUFFER_NAMES:
            tensor = getattr(self, buffer_name, None)
            if tensor is None:
                continue
            stored_tensors[f"{self.name}.{buffer_name}"] = {
                "shape": list(tensor.shape),
                "torch_dtype": str(tensor.dtype).split(".")[-1],
            }

        entry: Dict[str, Any] = {
            "stored_tensors": stored_tensors,
            "quant_format": "exl3",
        }
        trellis = getattr(self, "trellis", None)
        if trellis is not None:
            entry["bits_per_weight"] = int(trellis.shape[-1] // 16)

        mcg_multiplier = self._multiplier_value("mcg")
        if mcg_multiplier is not None:
            entry["mcg_multiplier"] = mcg_multiplier

        mul1_multiplier = self._multiplier_value("mul1")
        if mul1_multiplier is not None:
            entry["mul1_multiplier"] = mul1_multiplier

        return entry
