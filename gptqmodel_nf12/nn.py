# SPDX-FileCopyrightText: 2026 AlpinDale and the nf12 contributors
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Drop-in inference replacements for bf16 ``nn.Linear`` and ``nn.Embedding``.

Both hold an :class:`~nf12.NF12Weight` (``[N, K]``). Tied weights, such as an
embedding table that is also the ``lm_head``, are one ``NF12Weight`` shared by
both modules. Construct ``NF12Embedding.from_weight(w)`` and
``NF12Linear.from_weight(w)`` with the same ``w``.
"""

from __future__ import annotations

import torch
from torch import nn

from ._library import load
from .api import ESCAPES_KEY, EXP_BASE_KEY, PACKED_KEY, NF12Weight, check_weight_shape, encode, linear_fp32


# One super-atom is 1536 bytes: a 512-byte meta plane, then two low-byte planes.
_SUPER_ATOM_PLANES = (3, 512)
# Both sign bits set and zero offsets encode -0.0 without escapes at exp_base 0.
# +0.0 would hit the M == 0 escape sentinel.
_NEG_ZERO_META = 0xC0


def _check_bf16(w: torch.Tensor) -> None:
    if w.dtype != torch.bfloat16:
        raise ValueError(f"nf12: NF12 encodes bfloat16 weights losslessly; got {w.dtype} (call .bfloat16() first)")


class _NF12WeightModule(nn.Module):
    """Owns an NF12 ``[N, K]`` weight as buffers, saved under SPEC.md section 5
    names (``weight.nf12_packed``, ``weight.nf12_escapes``, ``weight.nf12_exp_base``).
    A fresh module holds an all ``-0.0`` weight."""

    _SHAPE_NAME = "(N, K)"

    def __init__(self, n: int, k: int, device=None) -> None:
        check_weight_shape(n, k)
        super().__init__()
        load()
        packed = torch.zeros((n, k * 3 // 2), dtype=torch.uint8, device=device)
        packed.view(-1, *_SUPER_ATOM_PLANES)[:, 0] = _NEG_ZERO_META
        self.register_buffer("nf12_packed", packed)
        self.register_buffer("nf12_escapes", torch.empty((0, 8), dtype=torch.int16, device=device))
        self.exp_base = 0

    @property
    def weight(self) -> NF12Weight:
        return NF12Weight(self.nf12_packed, self.nf12_escapes, self.exp_base)

    def _assign_weight(self, weight: NF12Weight) -> None:
        self.nf12_packed = weight.packed
        self.nf12_escapes = weight.escapes
        self.exp_base = weight.exp_base

    def _save_to_state_dict(self, destination, prefix, keep_vars) -> None:
        destination.update(self.weight.state_dict(prefix + "weight."))

    def _extra_state_keys(self, prefix: str) -> set[str]:
        """Non-weight state-dict keys (e.g. bias) loaded by the subclass."""
        return set()

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ) -> None:
        weight_prefix = prefix + "weight."
        weight_keys = [weight_prefix + key for key in (PACKED_KEY, ESCAPES_KEY, EXP_BASE_KEY)]
        expected = set(weight_keys) | self._extra_state_keys(prefix)

        missing = [key for key in sorted(expected) if key not in state_dict]
        missing_keys.extend(missing)
        if strict:
            unexpected_keys.extend(key for key in state_dict if key.startswith(prefix) and key not in expected)

        if any(key in missing for key in weight_keys):
            return
        try:
            weight = NF12Weight.from_state_dict(state_dict, weight_prefix)
        except ValueError as e:
            error_msgs.append(f"{weight_prefix[:-1]}: {e}")
            return
        if weight.shape != self.weight.shape:
            error_msgs.append(
                f"{weight_prefix[:-1]}: shape {weight.shape} does not match {self._SHAPE_NAME} = {self.weight.shape}"
            )
            return
        device = self.nf12_packed.device
        self._assign_weight(weight if device.type == "meta" else weight.to(device))


class NF12Linear(_NF12WeightModule):
    """``y = x @ W.T + b`` with an NF12-encoded BF16 weight.

    BF16 activations use the native NF12 kernel and return BF16. FP32
    activations use the packed FP32 kernel and return FP32.

    A freshly constructed layer holds an all ``-0.0`` weight. Fill it with
    :meth:`from_linear`, :meth:`from_weight` or ``load_state_dict``. The state
    dict stores the weight under SPEC.md section 5 names
    (``weight.nf12_packed``, ``weight.nf12_escapes``, ``weight.nf12_exp_base``).
    """

    _SHAPE_NAME = "(out_features, in_features)"

    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None) -> None:
        super().__init__(out_features, in_features, device)
        self.in_features = in_features
        self.out_features = out_features
        if bias:
            self.bias = nn.Parameter(
                torch.zeros(out_features, dtype=torch.bfloat16, device=device), requires_grad=False
            )
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_weight(cls, weight: NF12Weight, bias: torch.Tensor | None = None) -> NF12Linear:
        """A layer using ``weight`` itself (not a copy), so it can be shared, e.g. with an :class:`NF12Embedding`."""
        n, k = weight.shape
        layer = cls(k, n, bias=bias is not None, device="meta")
        layer._assign_weight(weight)
        if bias is not None:
            if bias.dtype != torch.bfloat16 or bias.shape != (n,):
                raise ValueError(f"nf12: bias must be bfloat16 [{n}], got {bias.dtype} {tuple(bias.shape)}")
            layer.bias = nn.Parameter(bias.detach().to(weight.device), requires_grad=False)
        return layer

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> NF12Linear:
        """Losslessly converts a bf16 CUDA ``nn.Linear``."""
        w = linear.weight.detach()
        _check_bf16(w)
        return cls.from_weight(encode(w), linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.float32:
            bias = self.bias.float() if self.bias is not None else None
            return linear_fp32(x, self.weight, bias)
        if x.dtype == torch.bfloat16:
            return torch.ops.nf12.linear(x, self.nf12_packed, self.nf12_escapes, self.exp_base, self.bias)
        raise TypeError(f"NF12Linear requires BF16 or FP32 activation, got {x.dtype}")

    def _save_to_state_dict(self, destination, prefix, keep_vars) -> None:
        super()._save_to_state_dict(destination, prefix, keep_vars)
        if self.bias is not None:
            destination[prefix + "bias"] = self.bias if keep_vars else self.bias.detach()

    def _extra_state_keys(self, prefix: str) -> set[str]:
        return {prefix + "bias"} if self.bias is not None else set()

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ) -> None:
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )
        bias_key = prefix + "bias"
        if self.bias is None or bias_key not in state_dict:
            return
        bias = state_dict[bias_key]
        if bias.shape != self.bias.shape:
            error_msgs.append(f"{bias_key}: shape {tuple(bias.shape)} does not match {tuple(self.bias.shape)}")
        elif self.bias.device.type == "meta":
            self.bias = nn.Parameter(bias.detach().to(torch.bfloat16), requires_grad=False)
        else:
            with torch.no_grad():
                self.bias.copy_(bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, "
            f"exp_base={self.exp_base}, escapes={self.nf12_escapes.shape[0]}"
        )


class NF12Embedding(_NF12WeightModule):
    """``out = W[ids]`` with an NF12-encoded ``[num_embeddings, embedding_dim]`` table (SPEC.md section 10).

    Lookups preserve the BF16 table values. IDs outside ``[0, num_embeddings)``
    give ``+0.0`` rows (vocab-parallel shards). ``ids`` may be int32 or int64 of
    any shape. Construction and state dict work as for :class:`NF12Linear`.
    """

    _SHAPE_NAME = "(num_embeddings, embedding_dim)"

    def __init__(self, num_embeddings: int, embedding_dim: int, device=None) -> None:
        super().__init__(num_embeddings, embedding_dim, device)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

    @classmethod
    def from_weight(cls, weight: NF12Weight) -> NF12Embedding:
        """A table using ``weight`` itself (not a copy), so it can be shared, e.g. with a tied :class:`NF12Linear`."""
        n, k = weight.shape
        layer = cls(n, k, device="meta")
        layer._assign_weight(weight)
        return layer

    @classmethod
    def from_embedding(cls, embedding: nn.Embedding) -> NF12Embedding:
        """Losslessly converts a bf16 CUDA ``nn.Embedding``. Rejects ``max_norm``, which rewrites rows."""
        if embedding.max_norm is not None:
            raise ValueError("nf12: NF12Embedding does not support max_norm")
        w = embedding.weight.detach()
        _check_bf16(w)
        return cls.from_weight(encode(w))

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return torch.ops.nf12.embedding(ids, self.nf12_packed, self.nf12_escapes, self.exp_base)

    def extra_repr(self) -> str:
        return (
            f"{self.num_embeddings}, {self.embedding_dim}, exp_base={self.exp_base}, "
            f"escapes={self.nf12_escapes.shape[0]}"
        )
