# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
#
# Portions of this file are adapted from turboderp-org/exllamav3.
# Credits: TurboDerp / ExLlamaV3 contributors.

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

import torch
import torch.nn as nn

from ..models._const import DEVICE, normalize_device


if TYPE_CHECKING:
    from ..exllamav3.modules.quant.exl3 import LinearEXL3


_EXL3_BUFFER_NAMES = ("trellis", "suh", "svh", "su", "sv", "bias", "mcg", "mul1")


def _torch_dtype(value: Any) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
    if isinstance(value, str):
        return getattr(torch, value)
    raise TypeError(f"Unsupported torch dtype value: {value!r}")


class ExllamaV3Linear(nn.Module):
    QUANT_TYPE = "exl3"
    SUPPORTS_SHARDS = True
    SUPPORTS_DEVICES = [DEVICE.CUDA]

    @classmethod
    def validate_device(cls, device) -> None:
        if normalize_device(device) not in cls.SUPPORTS_DEVICES:
            raise NotImplementedError(
                "EXL3_EXLLAMA_V3 inference requires NVIDIA CUDA; use EXL3_TORCH on ROCm or CPU."
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

    def _current_signature(self) -> tuple[Any, ...]:
        trellis = getattr(self, "trellis", None)
        if trellis is None or trellis.device.type == "meta":
            return ("meta",)

        signature: list[Any] = [str(trellis.device)]
        for buffer_name in _EXL3_BUFFER_NAMES:
            tensor = getattr(self, buffer_name, None)
            if tensor is None:
                signature.append(None)
                continue
            signature.append((tensor.data_ptr(), tuple(tensor.shape), str(tensor.dtype)))
        return tuple(signature)

    def _drop_inner(self) -> None:
        if self._inner is not None:
            try:
                self._inner.unload()
            except Exception:
                # `_drop_inner` runs during teardown and `_apply`; cleanup must stay best-effort.
                pass
        self._inner = None
        self._inner_signature = None

    def _stored_features(self) -> tuple[int, int]:
        """Infer storage geometry; architecture dimensions remain logical."""
        trellis = getattr(self, "trellis", None)
        if trellis is None or trellis.ndim != 3:
            raise ValueError(f"EXL3 module `{self.name}` has invalid trellis geometry")
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
        self._inner_signature = signature
        return self._inner

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inner = self._ensure_inner()
        input_dtype = x.dtype
        return inner.forward(x.half(), {})[..., :self.out_features].to(input_dtype)

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
