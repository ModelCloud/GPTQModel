# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Runtime quantization of delta-rule linear-attention recurrent state.

LeapQuant is not a weight format. It keeps a packed copy of the fixed-size
recurrent state that Gated DeltaNet and Kimi Delta Attention layers read and
rewrite at every decoding step. This implementation follows the published
equations in arXiv:2609.38166 and does not depend on a serving runtime, a
vendor kernel, or a checkpoint format change.
"""

from __future__ import annotations

from contextlib import nullcontext
from copy import copy
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache, LinearAttentionLayer
from transformers.generation import GenerationMixin
from transformers.generation.configuration_utils import GenerationMode


_EPS = 1e-12
_SUPPORTED_BITS = (4, 6, 8)
_SUPPORTED_BUFFER_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_FORMAT_VERSION = 2
_STATE_TENSORS = (
    "codes",
    "key_scale",
    "value_scale",
    "factors_key",
    "factors_value",
    "buffer_keys",
    "buffer_updates",
    "buffer_alpha",
)
_CODE_CONTAINERS = {4: "packed4", 6: "packed6", 8: "int8"}


def _packed_code_width(bits: int, value_dim: int) -> int:
    """Return the stored code width for one ``[B, H, K, V]`` state row."""

    if bits == 8:
        return value_dim
    if bits == 4:
        return (value_dim + 1) // 2
    return ((value_dim + 3) // 4) * 3


def _code_dtype(bits: int) -> torch.dtype:
    return torch.int8 if bits == 8 else torch.uint8


def _pack_codes(codes: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack signed codes into the persistent byte container.

    4-bit rows store two two's-complement nibbles per byte, low nibble first.
    6-bit rows store four two's-complement codes in three bytes as one
    little-endian 24-bit group. Partial groups are padded with zero codes.
    """

    if bits == 8:
        return codes.to(torch.int8)
    if codes.shape[-1] == 0:
        return torch.empty(
            (*codes.shape[:-1], _packed_code_width(bits, 0)), dtype=torch.uint8, device=codes.device,
        )
    if bits == 4:
        pad = (-codes.shape[-1]) % 2
        if pad:
            codes = F.pad(codes, (0, pad))
        unsigned = (codes.to(torch.int16) & 0xF).reshape(*codes.shape[:-1], -1, 2)
        packed = unsigned[..., 0] | (unsigned[..., 1] << 4)
        return packed.to(torch.uint8)
    if bits == 6:
        pad = (-codes.shape[-1]) % 4
        if pad:
            codes = F.pad(codes, (0, pad))
        unsigned = (codes.to(torch.int16) & 0x3F).reshape(*codes.shape[:-1], -1, 4)
        first, second, third, fourth = unsigned.unbind(dim=-1)
        byte0 = first | ((second & 0x3) << 6)
        byte1 = (second >> 2) | ((third & 0xF) << 4)
        byte2 = (third >> 4) | (fourth << 2)
        packed = torch.stack((byte0, byte1, byte2), dim=-1).reshape(*codes.shape[:-1], -1)
        return packed.to(torch.uint8)
    raise ValueError(f"unsupported LeapQuant code width: {bits}")


def _unpack_codes(packed: torch.Tensor, bits: int, value_dim: int) -> torch.Tensor:
    """Return the logical signed codes from the persistent byte container."""

    if bits == 8:
        return packed
    if bits == 4:
        unsigned = packed.to(torch.int16)
        values = torch.stack((unsigned & 0xF, (unsigned >> 4) & 0xF), dim=-1)
        values = values.reshape(*packed.shape[:-1], -1)[..., :value_dim]
        return ((values ^ 0x8) - 0x8).to(torch.int8)
    if bits == 6:
        unsigned = packed.to(torch.int16).reshape(*packed.shape[:-1], -1, 3)
        byte0, byte1, byte2 = unsigned.unbind(dim=-1)
        first = byte0 & 0x3F
        second = ((byte0 >> 6) & 0x3) | ((byte1 & 0xF) << 2)
        third = ((byte1 >> 4) & 0xF) | ((byte2 & 0x3) << 4)
        fourth = (byte2 >> 2) & 0x3F
        values = torch.stack((first, second, third, fourth), dim=-1)
        values = values.reshape(*packed.shape[:-1], -1)[..., :value_dim]
        return ((values ^ 0x20) - 0x20).to(torch.int8)
    raise ValueError(f"unsupported LeapQuant code width: {bits}")


@dataclass(frozen=True)
class LeapQuantConfig:
    """Configuration for runtime recurrent-state quantization."""

    bits: int = 8
    window: int = 16
    rank: int = 4
    smooth: bool = True
    use_triton: bool = True
    buffer_dtype: torch.dtype = torch.bfloat16
    lowrank_iters: int = 2
    lowrank_oversample: int = 4
    seed: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.bits, bool) or not isinstance(self.bits, int) or self.bits not in _SUPPORTED_BITS:
            raise ValueError(f"LeapQuantConfig.bits must be one of {_SUPPORTED_BITS}, got {self.bits!r}")
        if isinstance(self.window, bool) or not isinstance(self.window, int) or self.window < 1:
            raise ValueError(f"LeapQuantConfig.window must be a positive integer, got {self.window!r}")
        if isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank < 0:
            raise ValueError(f"LeapQuantConfig.rank must be a non-negative integer, got {self.rank!r}")
        if not isinstance(self.smooth, bool):
            raise TypeError(f"LeapQuantConfig.smooth must be bool, got {type(self.smooth).__name__}")
        if not isinstance(self.use_triton, bool):
            raise TypeError(f"LeapQuantConfig.use_triton must be bool, got {type(self.use_triton).__name__}")
        if self.buffer_dtype not in _SUPPORTED_BUFFER_DTYPES:
            raise ValueError(
                "LeapQuantConfig.buffer_dtype must be float16, bfloat16, or float32, "
                f"got {self.buffer_dtype!r}"
            )
        if (
            isinstance(self.lowrank_iters, bool)
            or not isinstance(self.lowrank_iters, int)
            or self.lowrank_iters < 0
        ):
            raise ValueError("LeapQuantConfig.lowrank_iters must be a non-negative integer")
        if (
            isinstance(self.lowrank_oversample, bool)
            or not isinstance(self.lowrank_oversample, int)
            or self.lowrank_oversample < 0
        ):
            raise ValueError("LeapQuantConfig.lowrank_oversample must be a non-negative integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("LeapQuantConfig.seed must be an integer")

    @property
    def qmax(self) -> float:
        return float(2 ** (self.bits - 1) - 1)


def _check_finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise ValueError(f"LeapQuant {name} contains non-finite values")


def _fp32_context(device: torch.device):
    if torch.amp.is_autocast_available(device.type) and torch.is_autocast_enabled(device.type):
        return torch.autocast(device_type=device.type, enabled=False)
    return nullcontext()


def _orthonormal_columns(matrix: torch.Tensor) -> torch.Tensor:
    """Orthonormalize the trailing columns through a Cholesky factor.

    A batched QR of the sketch matrices costs milliseconds on the accelerator
    linear-algebra drivers, while the two contractions below stay well under a
    tenth of that for the narrow sketches used here. A relative jitter keeps the
    Gram matrix positive definite when a sketch column is nearly dependent.
    """

    gram = matrix.transpose(-1, -2) @ matrix
    scale = gram.diagonal(dim1=-2, dim2=-1).amax(dim=-1, keepdim=True).clamp_min(_EPS)
    eye = torch.eye(gram.shape[-1], dtype=gram.dtype, device=gram.device)
    factor = torch.linalg.cholesky(gram + eye * (scale * 1e-6)[..., None])
    return torch.linalg.solve_triangular(factor, matrix.transpose(-1, -2), upper=False).transpose(-1, -2)


def _thin_singular_pairs(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the left singular vectors and values of a matrix with few rows.

    ``matrix`` has fewer rows than columns, so its Gram matrix is the small
    side and the decomposition costs a fraction of a wide SVD.
    """

    gram = matrix @ matrix.transpose(-1, -2)
    # A near-rank-deficient state produces Gram entries that underflow to
    # subnormals; CUDA ``eigh`` returns NaN for that matrix. Normalize by the
    # largest diagonal entry so the decomposition sees O(1) values, then
    # rescale the singular values.
    scale = gram.diagonal(dim1=-2, dim2=-1).amax(dim=-1, keepdim=True).clamp_min(_EPS)
    values, vectors = torch.linalg.eigh(gram / scale.unsqueeze(-1))
    return vectors.flip(-1), (values.flip(-1).clamp_min(0.0) * scale).sqrt()


def normalize_qk(query: torch.Tensor, key: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the delta-rule L2 normalization to query and key."""

    query = query.float()
    key = key.float()
    query = query / torch.sqrt(query.square().sum(dim=-1, keepdim=True) + 1e-6)
    query = query * (query.shape[-1] ** -0.5)
    key = key / torch.sqrt(key.square().sum(dim=-1, keepdim=True) + 1e-6)
    return query, key


def _repeat_grouped_qk(
    query: torch.Tensor, key: torch.Tensor, num_heads: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand grouped query/key heads to the value-head count.

    GDN permits fewer key heads than value heads; the delta-rule kernels repeat
    each key head to serve its value-head group. The packed state has one head
    per value head, so apply the same mapping before the step.
    """

    key_heads = query.shape[-2]
    if key_heads == num_heads:
        return query, key
    if key_heads < 1 or num_heads % key_heads:
        raise ValueError(
            f"LeapQuant grouped heads require num_heads={num_heads} to be a multiple "
            f"of the key-head count, got {key_heads}"
        )
    repeats = num_heads // key_heads
    return query.repeat_interleave(repeats, dim=-2), key.repeat_interleave(repeats, dim=-2)


def gdn_alpha(raw_gate: torch.Tensor, a_log: torch.Tensor, dt_bias: torch.Tensor) -> torch.Tensor:
    """Per-head GDN decay from the raw gate and discretization parameters."""

    gate = raw_gate.float()
    dt = dt_bias.float()
    a = a_log.float()
    if gate.ndim == 3:
        dt = dt.reshape(-1, 1)
        a = a.reshape(-1, 1)
    return (-torch.exp(a) * F.softplus(gate + dt)).exp()


def kda_alpha(
    raw_gate: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: Optional[float],
) -> torch.Tensor:
    """Per-key KDA decay for the softplus (Kimi) and bounded (GLM) gates."""

    gate = raw_gate.float()
    dt = dt_bias.float()
    a = a_log.float()
    if dt.ndim == 1:
        dt = dt.reshape(gate.shape[-2], -1)
    if a.ndim == 1:
        a = a[:, None]
    gate = gate + dt
    decay = torch.exp(a)
    if lower_bound is None:
        log_decay = -decay * F.softplus(gate)
    else:
        log_decay = float(lower_bound) * torch.sigmoid(decay * gate)
    return log_decay.exp()


def mask_is_unpadded(mask: Optional[torch.Tensor]) -> bool:
    """Return whether a linear-attention padding mask is equivalent to no padding."""

    return mask is None or (mask.ndim == 2 and bool((mask == 1).all()))


def validate_leapquant_inputs(attention_mask, kwargs) -> None:
    if isinstance(attention_mask, dict):
        attention_mask = attention_mask.get("linear_attention")
    if not mask_is_unpadded(attention_mask):
        raise ValueError("LeapQuant requires unpadded input; pass no attention_mask or an all-ones mask")
    if any(kwargs.get(name) is not None for name in ("cu_seqlens", "cu_seq_lens_q", "cu_seq_lens_k")):
        raise ValueError("LeapQuant does not support packed cu_seqlens")


class LeapQuantState:
    """Packed recurrent state plus the in-window update buffer for one cache layer."""

    def __init__(
        self,
        num_sequences: int,
        num_heads: int,
        key_dim: int,
        value_dim: int,
        config: LeapQuantConfig,
        *,
        gate_dim: Optional[int] = None,
        device: Optional[torch.device | str] = None,
    ) -> None:
        if not isinstance(config, LeapQuantConfig):
            raise TypeError(f"config must be LeapQuantConfig, got {type(config).__name__}")
        if min(num_sequences, num_heads, key_dim, value_dim) < 1:
            raise ValueError("LeapQuant state dimensions must all be positive")
        gate_dim = key_dim if gate_dim is None else int(gate_dim)
        if gate_dim not in (1, key_dim):
            raise ValueError(f"gate_dim must be 1 or key_dim ({key_dim}), got {gate_dim}")
        if config.rank > min(key_dim, value_dim):
            raise ValueError(
                f"LeapQuantConfig.rank must be <= min(key_dim, value_dim)="
                f"{min(key_dim, value_dim)}, got {config.rank}"
            )

        self.num_sequences = int(num_sequences)
        self.num_heads = int(num_heads)
        self.key_dim = int(key_dim)
        self.value_dim = int(value_dim)
        self.gate_dim = gate_dim
        self.config = config
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self._omega: Optional[torch.Tensor] = None
        self._omega_key: Optional[tuple[int, int, torch.device]] = None
        self._scratch: dict[str, torch.Tensor] = {}
        self._step_backend: Optional[str] = None
        self._fit_backend: Optional[str] = None
        self.reset()

    @property
    def rank(self) -> int:
        return self.factors_key.shape[-1]

    @property
    def code_container(self) -> str:
        return _CODE_CONTAINERS[self.config.bits]

    @property
    def window_count(self) -> int:
        return self._window_count

    @window_count.setter
    def window_count(self, value: int) -> None:
        self._window_count = int(value)

    def reset(self, state: Optional[torch.Tensor] = None) -> None:
        """Reset to the zero checkpoint, or quantize ``state`` as the new checkpoint."""

        if state is not None:
            self.quantize_(state)
            return

        b, h, k, v = self.num_sequences, self.num_heads, self.key_dim, self.value_dim
        r = self.config.rank
        cfg = self.config
        self.codes = torch.zeros(
            (b, h, k, _packed_code_width(cfg.bits, v)),
            dtype=_code_dtype(cfg.bits),
            device=self.device,
        )
        self.key_scale = torch.ones((b, h, k), dtype=torch.float32, device=self.device)
        self.value_scale = torch.ones((b, h, v), dtype=torch.float32, device=self.device)
        self.factors_key = torch.zeros((b, h, k, r), dtype=torch.float16, device=self.device)
        self.factors_value = torch.zeros((b, h, v, r), dtype=torch.float16, device=self.device)
        self.buffer_keys = torch.zeros((b, h, cfg.window, k), dtype=cfg.buffer_dtype, device=self.device)
        self.buffer_updates = torch.zeros((b, h, cfg.window, v), dtype=cfg.buffer_dtype, device=self.device)
        self.buffer_alpha = torch.zeros((b, h, cfg.window, self.gate_dim), dtype=torch.float32, device=self.device)
        self.window_count = 0

    def _check_state(self, state: torch.Tensor) -> torch.Tensor:
        expected = (self.num_sequences, self.num_heads, self.key_dim, self.value_dim)
        if tuple(state.shape) != expected:
            raise ValueError(f"LeapQuant state shape must be {expected}, got {tuple(state.shape)}")
        return state.to(device=self.device, dtype=torch.float32)

    def _check_token(self, name: str, tensor: torch.Tensor, width: int) -> torch.Tensor:
        expected = (self.num_sequences, self.num_heads, width)
        if tuple(tensor.shape) != expected:
            raise ValueError(f"LeapQuant {name} shape must be {expected}, got {tuple(tensor.shape)}")
        return tensor.to(device=self.device, dtype=torch.float32)

    def _check_sequence(self, name: str, tensor: torch.Tensor, width: int) -> torch.Tensor:
        if tensor.ndim != 4 or tensor.shape[0] != self.num_sequences or tensor.shape[2] != self.num_heads:
            raise ValueError(
                f"LeapQuant {name} must be [B, T, H, {width}] with B={self.num_sequences}, "
                f"H={self.num_heads}; got {tuple(tensor.shape)}"
            )
        if tensor.shape[-1] != width:
            raise ValueError(f"LeapQuant {name} last dimension must be {width}, got {tensor.shape[-1]}")
        return tensor.to(device=self.device, dtype=torch.float32)

    def _check_alpha(self, alpha: torch.Tensor) -> torch.Tensor:
        alpha = alpha.to(device=self.device, dtype=torch.float32)
        if self.gate_dim == 1:
            if alpha.shape == (self.num_sequences, self.num_heads):
                alpha = alpha[..., None]
            if alpha.shape != (self.num_sequences, self.num_heads, 1):
                raise ValueError(
                    f"LeapQuant alpha must be [B, H] or [B, H, 1], got {tuple(alpha.shape)}"
                )
            return alpha
        if alpha.shape == (self.num_sequences, self.num_heads):
            alpha = alpha[..., None].expand(-1, -1, self.key_dim)
        elif alpha.shape == (self.num_sequences, self.num_heads, 1):
            alpha = alpha.expand(-1, -1, self.key_dim)
        if alpha.shape != (self.num_sequences, self.num_heads, self.key_dim):
            raise ValueError(
                f"LeapQuant alpha must be [B, H, 1] or [B, H, {self.key_dim}], got {tuple(alpha.shape)}"
            )
        return alpha

    def _check_beta(self, beta: torch.Tensor) -> torch.Tensor:
        beta = beta.to(device=self.device, dtype=torch.float32)
        if beta.shape == (self.num_sequences, self.num_heads, 1):
            beta = beta[..., 0]
        if beta.shape != (self.num_sequences, self.num_heads):
            raise ValueError(
                f"LeapQuant beta must be [B, H] or [B, H, 1], got {tuple(beta.shape)}"
            )
        return beta

    def _check_sequence_alpha(self, alpha: torch.Tensor) -> torch.Tensor:
        alpha = alpha.to(device=self.device, dtype=torch.float32)
        if self.gate_dim == 1:
            if alpha.ndim == 4 and alpha.shape[-1] == 1:
                alpha = alpha[..., 0]
            if alpha.ndim != 3 or alpha.shape[0] != self.num_sequences or alpha.shape[2] != self.num_heads:
                raise ValueError(
                    f"LeapQuant alpha must be [B, T, H] with B={self.num_sequences}, "
                    f"H={self.num_heads}; got {tuple(alpha.shape)}"
                )
            return alpha[..., None]
        if alpha.ndim == 3 and alpha.shape[0] == self.num_sequences and alpha.shape[2] == self.num_heads:
            alpha = alpha[..., None].expand(-1, -1, -1, self.key_dim)
        if alpha.ndim != 4 or alpha.shape[0] != self.num_sequences or alpha.shape[2] != self.num_heads:
            raise ValueError(
                f"LeapQuant alpha must be [B, T, H] or [B, T, H, {self.key_dim}] with "
                f"B={self.num_sequences}, H={self.num_heads}; got {tuple(alpha.shape)}"
            )
        if alpha.shape[-1] == 1:
            alpha = alpha.expand(-1, -1, -1, self.key_dim)
        if alpha.shape[-1] != self.key_dim:
            raise ValueError(f"LeapQuant alpha last dimension must be 1 or {self.key_dim}")
        return alpha

    def _check_sequence_beta(self, beta: torch.Tensor) -> torch.Tensor:
        beta = beta.to(device=self.device, dtype=torch.float32)
        if beta.ndim == 4 and beta.shape[-1] == 1:
            beta = beta[..., 0]
        if beta.ndim != 3 or beta.shape[0] != self.num_sequences or beta.shape[2] != self.num_heads:
            raise ValueError(
                f"LeapQuant beta must be [B, T, H] with B={self.num_sequences}, H={self.num_heads}; "
                f"got {tuple(beta.shape)}"
            )
        return beta

    @torch.no_grad()
    def _orthonormalize(self, matrix: torch.Tensor) -> torch.Tensor:
        """Orthonormalize the sketch through the native kernel when available."""

        if self.config.use_triton and self.device.type == "cuda":
            from . import leapquant_triton

            basis = leapquant_triton.orthonormalize(matrix)
            if basis is not None:
                self._fit_backend = "triton"
                return basis
        self._fit_backend = "torch"
        return _orthonormal_columns(matrix)

    def _sketch(self, value_dim: int, columns: int) -> torch.Tensor:
        """Return the fixed seeded sketch, created once per state and device."""

        key = (value_dim, columns, self.device)
        if self._omega is None or self._omega_key != key:
            generator = torch.Generator(device="cpu").manual_seed(self.config.seed)
            self._omega = torch.randn(
                (value_dim, columns), generator=generator, device="cpu", dtype=torch.float32,
            ).to(self.device)
            self._omega_key = key
        return self._omega

    @torch.no_grad()
    def _fit_compensator(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Fit the rank-R additive term; exact SVD at full rank, randomized otherwise."""

        b, h, k, v = state.shape
        r = self.rank
        if r == 0:
            return (
                torch.zeros((b, h, k, 0), dtype=torch.float32, device=state.device),
                torch.zeros((b, h, v, 0), dtype=torch.float32, device=state.device),
            )
        if r >= min(k, v):
            u, s, vh = torch.linalg.svd(state, full_matrices=False)
            root = s.clamp_min(0.0).sqrt()
            return u[..., :r] * root[..., None, :], vh[..., :r, :].transpose(-1, -2) * root[..., None, :]

        columns = min(r + self.config.lowrank_oversample, min(k, v))
        omega = self._sketch(v, columns)
        q = self._orthonormalize(state @ omega)
        for _ in range(self.config.lowrank_iters):
            z = self._orthonormalize(state.transpose(-1, -2) @ q)
            q = self._orthonormalize(state @ z)
        reduced = q.transpose(-1, -2) @ state
        u, singular = _thin_singular_pairs(reduced)
        root = singular[..., :r].clamp_min(0.0).sqrt()
        # A degenerate direction must yield zero factors, never a division by
        # zero; the floor is relative to the leading singular value.
        root = root.clamp_min(root[..., :1].clamp_min(_EPS) * 1e-6)
        factors_key = q @ (u[..., :r] * root[..., None, :])
        factors_value = (reduced.transpose(-1, -2) @ u[..., :r]) / root[..., None, :]
        return factors_key, factors_value

    @torch.no_grad()
    def quantize_(self, state: torch.Tensor) -> None:
        """Replace the packed checkpoint with a quantized copy of ``state``."""

        state = self._check_state(state)
        _check_finite("state", state)
        qmax = self.config.qmax
        with _fp32_context(state.device):
            factors_key, factors_value = self._fit_compensator(state)
            factors_key = factors_key.to(torch.float16)
            factors_value = factors_value.to(torch.float16)
            approx = factors_key.float() @ factors_value.float().transpose(-1, -2)
        residual = state - approx

        if self.config.smooth:
            key_scale = residual.abs().mean(dim=-1).sqrt().clamp_min(_EPS)
        else:
            key_scale = torch.ones_like(residual[..., 0])
        scaled = residual / key_scale[..., :, None]
        value_scale = scaled.abs().amax(dim=-2).clamp_min(_EPS)
        codes = torch.round(scaled / value_scale[..., None, :] * qmax).clamp(-qmax, qmax)
        self._require_finite(
            ("key scale", key_scale),
            ("value scale", value_scale),
            ("factors key", factors_key),
            ("factors value", factors_value),
        )

        self.codes.copy_(_pack_codes(codes.to(torch.int8), self.config.bits))
        self.key_scale.copy_(key_scale)
        self.value_scale.copy_(value_scale)
        self.factors_key.copy_(factors_key)
        self.factors_value.copy_(factors_value)
        self._clear_window()

    def _codes_i8(self) -> torch.Tensor:
        """Return the logical signed codes; packed widths unpack transiently."""

        return _unpack_codes(self.codes, self.config.bits, self.value_dim)

    @torch.no_grad()
    def dequantize(self) -> torch.Tensor:
        """Materialize the checkpoint state in FP32 (transient)."""

        with _fp32_context(self.codes.device):
            approx = self.factors_key.float() @ self.factors_value.float().transpose(-1, -2)
        residual = (
            self._codes_i8().float()
            / self.config.qmax
            * self.value_scale[..., None, :]
            * self.key_scale[..., :, None]
        )
        return approx + residual

    @torch.no_grad()
    def _replay_window(self) -> torch.Tensor:
        """Return the checkpoint plus every buffered token update."""

        state = self.dequantize()
        for index in range(self.window_count):
            decay = self._decay_row_scale(self.buffer_alpha[:, :, index])
            state = (
                state * decay
                + self.buffer_keys[:, :, index].float()[..., :, None]
                * self.buffer_updates[:, :, index].float()[..., None, :]
            )
        return state

    @torch.no_grad()
    def materialize(self) -> torch.Tensor:
        """Return the current state including buffered updates (transient)."""

        return self._replay_window()

    def _require_finite(self, *items: tuple[str, torch.Tensor]) -> None:
        """Validate several tensors with a single device-to-host transfer."""

        combined = None
        for _, tensor in items:
            ok = torch.isfinite(tensor).all()
            combined = ok if combined is None else combined & ok
        if combined is None or bool(combined):
            return
        for name, tensor in items:
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"LeapQuant {name} contains non-finite values")
        raise ValueError("LeapQuant input contains non-finite values")

    def _decay_row_scale(self, decay: torch.Tensor) -> torch.Tensor:
        """Shape a per-head decay for broadcasting against a ``[B, H, K, V]`` state."""

        if self.gate_dim == self.key_dim:
            return decay[..., :, None]
        return decay[..., 0][..., None, None]

    @torch.no_grad()
    def step(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        """Consume one token per row and return the layer output."""

        query = self._check_token("query", query, self.key_dim)
        key = self._check_token("key", key, self.key_dim)
        value = self._check_token("value", value, self.value_dim)
        alpha = self._check_alpha(alpha)
        beta = self._check_beta(beta)
        if self.window_count >= self.config.window:
            raise RuntimeError("LeapQuant window is full; quantize before the next step")

        index = self.window_count
        if self.config.use_triton and self.rank > 0 and self.device.type == "cuda":
            from . import leapquant_triton

            fused = leapquant_triton.launch_step(
                codes=self.codes,
                key_scale=self.key_scale,
                value_scale=self.value_scale,
                factors_key=self.factors_key,
                factors_value=self.factors_value,
                buffer_keys=self.buffer_keys,
                buffer_updates=self.buffer_updates,
                buffer_alpha=self.buffer_alpha,
                query=query,
                key=key,
                value=value,
                alpha=alpha,
                beta=beta,
                window_count=index,
                window=self.config.window,
                rank=self.rank,
                gate_dim=self.gate_dim,
                bits=self.config.bits,
                buffer_dtype=self.config.buffer_dtype,
                device=self.device,
                scratch=self._scratch,
            )
            if fused is not None:
                output, _update, next_state, bad, staged_key, staged_update = fused
                if bool(bad.any()):
                    raise ValueError("LeapQuant input contains non-finite values")
                if next_state is not None:
                    self.quantize_(next_state)
                else:
                    self.buffer_keys[:, :, index].copy_(staged_key)
                    self.buffer_updates[:, :, index].copy_(staged_update)
                    self.buffer_alpha[:, :, index].copy_(alpha)
                    self.window_count = index + 1
                self._step_backend = "triton"
                return output

        decayed = self._replay_window() * self._decay_row_scale(alpha)
        prediction = (decayed * key[..., :, None]).sum(dim=-2)
        update = beta[..., None] * (value - prediction)
        next_state = decayed + key[..., :, None] * update[..., None, :]
        output = (next_state * query[..., :, None]).sum(dim=-2)

        buffered_key = key.to(self.config.buffer_dtype)
        buffered_update = update.to(self.config.buffer_dtype)
        # A non-finite input reaches ``output`` through the recurrence above, so
        # this single check keeps the fail-closed contract without a host round
        # trip per intermediate tensor.
        self._require_finite(
            ("output", output),
            ("buffered key", buffered_key),
            ("buffered update", buffered_update),
        )

        if index + 1 == self.config.window:
            self.quantize_(next_state)
        else:
            self.buffer_keys[:, :, index].copy_(buffered_key)
            self.buffer_updates[:, :, index].copy_(buffered_update)
            self.buffer_alpha[:, :, index].copy_(alpha)
            self.window_count = index + 1
        self._step_backend = "torch"
        return output

    @torch.no_grad()
    def prefill(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        """Consume a prompt in per-window steps and return its outputs."""

        query = self._check_sequence("query", query, self.key_dim)
        key = self._check_sequence("key", key, self.key_dim)
        value = self._check_sequence("value", value, self.value_dim)
        alpha = self._check_sequence_alpha(alpha)
        beta = self._check_sequence_beta(beta)
        for name, tensor in (("key", key), ("value", value), ("alpha", alpha), ("beta", beta)):
            if tensor.shape[1] != query.shape[1]:
                raise ValueError(
                    f"LeapQuant {name} sequence length {tensor.shape[1]} differs from query length {query.shape[1]}"
                )
        # A later token can fail even with finite inputs, e.g. when an FP16 buffer overflows.
        pending = copy(self)
        pending.__dict__.update(self._clone_tensors())
        outputs = []
        for index in range(query.shape[1]):
            outputs.append(
                pending.step(
                    query[:, index],
                    key[:, index],
                    value[:, index],
                    alpha[:, index],
                    beta[:, index],
                )
            )
        result = torch.stack(outputs, dim=1)
        self.__dict__.update(pending.__dict__)
        return result

    @torch.no_grad()
    def gdn_step(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        raw_gate: torch.Tensor,
        beta_logits: torch.Tensor,
        a_log: torch.Tensor,
        dt_bias: torch.Tensor,
    ) -> torch.Tensor:
        """GDN convenience wrapper: normalize q/k, derive alpha/beta, then step."""

        query, key = normalize_qk(query, key)
        query, key = _repeat_grouped_qk(query, key, self.num_heads)
        return self.step(query, key, value, gdn_alpha(raw_gate, a_log, dt_bias), beta_logits.sigmoid())

    @torch.no_grad()
    def kda_step(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        raw_gate: torch.Tensor,
        beta_logits: torch.Tensor,
        a_log: torch.Tensor,
        dt_bias: torch.Tensor,
        lower_bound: Optional[float] = None,
    ) -> torch.Tensor:
        """Kimi/GLM KDA convenience wrapper for raw gates."""

        query, key = normalize_qk(query, key)
        return self.step(
            query,
            key,
            value,
            kda_alpha(raw_gate, a_log, dt_bias, lower_bound),
            beta_logits.sigmoid(),
        )

    @torch.no_grad()
    def transformed_kda_step(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        """KDA step for a port that has already applied its gate transform."""

        query, key = normalize_qk(query, key)
        return self.step(query, key, value, alpha, beta)

    def _clear_window(self) -> None:
        self.window_count = 0
        if self.buffer_keys.numel():
            self.buffer_keys.zero_()
            self.buffer_updates.zero_()
            self.buffer_alpha.zero_()

    def _clone_tensors(self) -> dict[str, torch.Tensor]:
        return {name: getattr(self, name).clone() for name in _STATE_TENSORS}

    def packed_state(self) -> Dict[str, Any]:
        """Return a boundary-consistent snapshot for cache save/restore."""

        return {
            "format_version": _FORMAT_VERSION,
            "bits": self.config.bits,
            "code_container": self.code_container,
            "window": self.config.window,
            "rank": self.rank,
            "smooth": self.config.smooth,
            "buffer_dtype": str(self.config.buffer_dtype),
            "gate_dim": self.gate_dim,
            "num_sequences": self.num_sequences,
            "num_heads": self.num_heads,
            "key_dim": self.key_dim,
            "value_dim": self.value_dim,
            **self._clone_tensors(),
            "window_count": self.window_count,
        }

    @torch.no_grad()
    def load_packed_state(self, packed: Dict[str, Any]) -> None:
        """Restore a snapshot produced by :meth:`packed_state`."""

        if int(packed.get("format_version", -1)) != _FORMAT_VERSION:
            raise ValueError("unsupported LeapQuant packed-state format version")
        expected = {
            "bits": self.config.bits,
            "code_container": self.code_container,
            "window": self.config.window,
            "rank": self.rank,
            "smooth": self.config.smooth,
            "buffer_dtype": str(self.config.buffer_dtype),
            "gate_dim": self.gate_dim,
            "num_sequences": self.num_sequences,
            "num_heads": self.num_heads,
            "key_dim": self.key_dim,
            "value_dim": self.value_dim,
        }
        for name, value in expected.items():
            if packed.get(name) != value:
                raise ValueError(f"LeapQuant packed-state {name} mismatch: {packed.get(name)!r} != {value!r}")
        count = packed.get("window_count")
        if type(count) is not int or not 0 <= count < self.config.window:
            raise ValueError("LeapQuant packed-state window_count must be within the active window")

        tensors = {}
        for name in _STATE_TENSORS:
            value = packed.get(name)
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"LeapQuant packed-state {name} must be a tensor")
            previous = getattr(self, name)
            if value.shape != previous.shape or value.dtype != previous.dtype or value.layout != previous.layout:
                raise ValueError(f"LeapQuant packed-state {name} has an incompatible shape, dtype, or layout")
            value = value.to(device=self.device, copy=True).contiguous()
            if value.is_floating_point():
                _check_finite(name, value)
            if name in ("key_scale", "value_scale") and not (value > 0).all():
                raise ValueError(f"LeapQuant packed-state {name} must be positive")
            if name == "codes":
                logical = _unpack_codes(value, self.config.bits, self.value_dim)
                if not ((logical >= -self.config.qmax) & (logical <= self.config.qmax)).all():
                    raise ValueError("LeapQuant packed-state codes are outside the configured bit range")
            tensors[name] = value
        self.__dict__.update(tensors)
        self.window_count = count

    def checkpoint_bytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (
                self.codes,
                self.key_scale,
                self.value_scale,
                self.factors_key,
                self.factors_value,
            )
        )

    def buffer_bytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (self.buffer_keys, self.buffer_updates, self.buffer_alpha)
        )

    def storage_bytes(self) -> int:
        return self.checkpoint_bytes() + self.buffer_bytes()

    def to(self, device: torch.device | str) -> "LeapQuantState":
        device = torch.device(device)
        tensors = {name: getattr(self, name).to(device) for name in _STATE_TENSORS}
        self.__dict__.update(tensors)
        self.device = device
        self._omega = None
        self._omega_key = None
        self._scratch = {}
        return self

    def _check_reorder_shape(self, beam_idx: torch.Tensor) -> None:
        if beam_idx.ndim != 1 or beam_idx.numel() != self.num_sequences:
            raise ValueError("LeapQuant reorder requires one index per live sequence")

    @torch.no_grad()
    def reorder_(self, beam_idx: torch.Tensor) -> None:
        """Atomically reorder every packed component along the sequence axis."""

        self._check_reorder_shape(beam_idx)
        index = beam_idx.to(device=self.device, dtype=torch.long)
        tensors = {name: getattr(self, name).index_select(0, index) for name in _STATE_TENSORS}
        self.__dict__.update(tensors)


class LeapQuantLinearAttentionLayer(LinearAttentionLayer):
    """Transformers linear-attention cache layer with a packed LeapQuant state."""

    @property
    def is_compileable(self) -> bool:
        return self.state_quant is None and super().is_compileable

    def __init__(
        self,
        number_of_states: int = 1,
        state_quant: Optional[LeapQuantConfig] = None,
        **kwargs,
    ) -> None:
        super().__init__(number_of_states=number_of_states, **kwargs)
        self.state_quant = state_quant
        self._leapquant: Optional[LeapQuantState] = None
        self._leapquant_shape: Optional[tuple[int, ...]] = None

    def leapquant_state(
        self,
        num_sequences: int,
        num_heads: int,
        key_dim: int,
        value_dim: int,
        *,
        gate_dim: int,
        device: torch.device | str,
    ) -> LeapQuantState:
        shape = (int(num_sequences), int(num_heads), int(key_dim), int(value_dim), int(gate_dim))
        device = torch.device(device)
        if self._leapquant is None:
            if self.state_quant is None:
                raise RuntimeError("LeapQuant cache layer has no state_quant config")
            self._leapquant = LeapQuantState(
                num_sequences=shape[0],
                num_heads=shape[1],
                key_dim=shape[2],
                value_dim=shape[3],
                config=self.state_quant,
                gate_dim=shape[4],
                device=device,
            )
            self._leapquant_shape = shape
        elif self._leapquant_shape != shape:
            raise ValueError(
                f"LeapQuant cache state was created with shape {self._leapquant_shape} "
                f"and cannot be reused with {shape}; a live cache cannot change its batch "
                "size or state geometry"
            )
        elif self._leapquant.device != device:
            self._leapquant.to(device)
        return self._leapquant

    def update_recurrent_state(self, recurrent_states: torch.Tensor, state_idx: int = 0, **kwargs) -> None:
        if self.state_quant is None:
            return super().update_recurrent_state(recurrent_states, state_idx=state_idx, **kwargs)
        if self._leapquant_shape is None or self._leapquant is None:
            raise RuntimeError("LeapQuant state was not initialized before update_recurrent_state")
        self._leapquant.quantize_(recurrent_states)
        return None

    def reset(self) -> None:
        super().reset()
        if self._leapquant is not None:
            self._leapquant.reset()

    def reorder_cache(self, beam_idx: torch.Tensor) -> None:
        state = self._leapquant
        if state is None:
            super().reorder_cache(beam_idx)
            return

        state._check_reorder_shape(beam_idx)
        conv_states = {
            index: conv.index_select(0, beam_idx.to(conv.device))
            for index, conv in self.conv_states.items()
            if self.is_conv_states_initialized[index]
        }
        state.reorder_(beam_idx)
        self.conv_states.update(conv_states)

    def offload(self) -> None:
        super().offload()
        if self._leapquant is not None:
            self._leapquant.to("cpu")

    def prefetch(self) -> None:
        super().prefetch()
        if self._leapquant is not None and self.device is not None:
            self._leapquant.to(self.device)

    def crop(self, tokens_to_remove: int) -> None:
        if self.state_quant is not None:
            raise RuntimeError("LeapQuant recurrent state cannot be cropped")
        return super().crop(tokens_to_remove)

    @property
    def is_croppable(self) -> bool:
        if self.state_quant is not None:
            return False
        return super().is_croppable


class LeapQuantCacheMixin:
    """Validate batch operations before touching mixed attention cache layers."""

    state_quant: Optional[LeapQuantConfig]

    def _check_leapquant_reorder(self, beam_idx: torch.Tensor) -> None:
        if self.state_quant is None:
            return
        for layer in self.layers:
            if isinstance(layer, LeapQuantLinearAttentionLayer) and layer._leapquant is not None:
                layer._leapquant._check_reorder_shape(beam_idx)

    def batch_repeat_interleave(self, repeats: int) -> None:
        if self.state_quant is None:
            super().batch_repeat_interleave(repeats)
        elif repeats != 1:
            raise ValueError("LeapQuant does not support cache batch expansion")

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        if self.state_quant is None:
            super().batch_select_indices(indices)
        else:
            self.reorder_cache(indices)


class LeapQuantDynamicCache(LeapQuantCacheMixin, DynamicCache):
    """DynamicCache whose linear-attention layers own packed LeapQuant state."""

    def __init__(self, config=None, state_quant: Optional[LeapQuantConfig] = None, **kwargs) -> None:
        super().__init__(config=config, **kwargs)
        self.state_quant = state_quant
        if state_quant is None:
            return
        for index, layer in enumerate(self.layers):
            if type(layer) is LinearAttentionLayer:
                self.layers[index] = LeapQuantLinearAttentionLayer(
                    number_of_states=layer.number_of_states,
                    state_quant=state_quant,
                )
            elif isinstance(layer, LinearAttentionLayer):
                raise ValueError("LeapQuant does not support combined linear/full-attention cache layers")

    def reorder_cache(self, beam_idx: torch.Tensor) -> None:
        self._check_leapquant_reorder(beam_idx)
        super().reorder_cache(beam_idx)

    def crop(self, tokens_to_remove: int) -> None:
        if self.state_quant is not None:
            raise RuntimeError("LeapQuant recurrent state cannot be cropped")
        super().crop(tokens_to_remove)


def make_leapquant_cache(config, state_quant: Optional[LeapQuantConfig]):
    """Build the cache class a port should use for the requested state policy."""

    if state_quant is None:
        return DynamicCache(config=config)
    return LeapQuantDynamicCache(config=config, state_quant=state_quant)


class LeapQuantGenerationMixin(GenerationMixin):
    """Install the decoder's recurrent state policy before the generation loop."""

    def _prepare_cache_for_generation(
        self, generation_config, model_kwargs, generation_mode, batch_size, max_cache_length,
    ) -> None:
        cache = model_kwargs.get("past_key_values")
        decoder = self.get_decoder()
        state_quant = getattr(decoder, "_gptqmodel_state_quant", None)
        create_packed = cache is None and generation_config.use_cache and state_quant is not None
        reuse_packed = isinstance(cache, LeapQuantDynamicCache) and cache.state_quant is not None
        if create_packed or reuse_packed:
            if generation_config.is_assistant or generation_mode in (
                GenerationMode.ASSISTED_GENERATION,
                GenerationMode.CONTRASTIVE_SEARCH,
            ):
                raise ValueError("LeapQuant generation does not support cache rollback")
            if create_packed:
                if generation_config.cache_implementation not in (None, "dynamic"):
                    raise ValueError("LeapQuant requires a dynamic generation cache")
                model_kwargs["past_key_values"] = make_leapquant_cache(decoder.config, state_quant)
                return

        super()._prepare_cache_for_generation(
            generation_config, model_kwargs, generation_mode, batch_size, max_cache_length,
        )
