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

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache, LinearAttentionLayer


_EPS = 1e-12
_SUPPORTED_BITS = (4, 6, 8)
_SUPPORTED_BUFFER_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_FORMAT_VERSION = 1


@dataclass(frozen=True)
class LeapQuantConfig:
    """Configuration for runtime recurrent-state quantization."""

    bits: int = 8
    window: int = 16
    rank: int = 4
    smooth: bool = True
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
        dt = dt[:, None]
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

    if mask is None:
        return True
    if mask.dtype == torch.bool:
        return bool(mask.all())
    if mask.dim() > 2:
        return False
    return bool(torch.isfinite(mask).all()) and bool((mask != 0).all())


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
        self.reset()

    @property
    def rank(self) -> int:
        return self.factors_key.shape[-1]

    @property
    def window_count(self) -> int:
        return self._window_count

    @window_count.setter
    def window_count(self, value: int) -> None:
        self._window_count = int(value)

    def reset(self, state: Optional[torch.Tensor] = None) -> None:
        """Reset to the zero checkpoint, or quantize ``state`` as the new checkpoint."""

        b, h, k, v = self.num_sequences, self.num_heads, self.key_dim, self.value_dim
        r = self.config.rank
        cfg = self.config
        self.codes = torch.zeros((b, h, k, v), dtype=torch.int8, device=self.device)
        self.key_scale = torch.ones((b, h, k), dtype=torch.float32, device=self.device)
        self.value_scale = torch.ones((b, h, v), dtype=torch.float32, device=self.device)
        self.factors_key = torch.zeros((b, h, k, r), dtype=torch.float16, device=self.device)
        self.factors_value = torch.zeros((b, h, v, r), dtype=torch.float16, device=self.device)
        self.buffer_keys = torch.zeros((b, h, cfg.window, k), dtype=cfg.buffer_dtype, device=self.device)
        self.buffer_updates = torch.zeros((b, h, cfg.window, v), dtype=cfg.buffer_dtype, device=self.device)
        self.buffer_alpha = torch.zeros((b, h, cfg.window, self.gate_dim), dtype=torch.float32, device=self.device)
        self.window_count = 0
        if state is not None:
            self.quantize_(state)

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
    def _fit_compensator(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Fit the rank-R additive term; exact SVD at full rank, randomized otherwise."""

        b, h, k, v = state.shape
        r = self.rank
        if r == 0:
            return (
                torch.zeros((b, h, k, 0), dtype=torch.float32, device=state.device),
                torch.zeros((b, h, v, 0), dtype=torch.float32, device=state.device),
            )
        if not torch.any(state):
            return (
                torch.zeros((b, h, k, r), dtype=torch.float32, device=state.device),
                torch.zeros((b, h, v, r), dtype=torch.float32, device=state.device),
            )
        if r >= min(k, v):
            u, s, vh = torch.linalg.svd(state, full_matrices=False)
            root = s.clamp_min(0.0).sqrt()
            return u[..., :r] * root[..., None, :], vh[..., :r, :].transpose(-1, -2) * root[..., None, :]

        columns = min(r + self.config.lowrank_oversample, min(k, v))
        generator = torch.Generator(device="cpu").manual_seed(self.config.seed)
        omega = torch.randn((v, columns), generator=generator, dtype=torch.float32).to(state.device)
        q, _ = torch.linalg.qr(state @ omega)
        for _ in range(self.config.lowrank_iters):
            z, _ = torch.linalg.qr(state.transpose(-1, -2) @ q)
            q, _ = torch.linalg.qr(state @ z)
        reduced = q.transpose(-1, -2) @ state
        u, s, vh = torch.linalg.svd(reduced, full_matrices=False)
        root = s[..., :r].clamp_min(0.0).sqrt()
        factors_key = q @ (u[..., :r] * root[..., None, :])
        factors_value = vh[..., :r, :].transpose(-1, -2) * root[..., None, :]
        return factors_key, factors_value

    @torch.no_grad()
    def quantize_(self, state: torch.Tensor) -> None:
        """Replace the packed checkpoint with a quantized copy of ``state``."""

        state = self._check_state(state)
        _check_finite("state", state)
        qmax = self.config.qmax
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
        _check_finite("key scale", key_scale)
        _check_finite("value scale", value_scale)

        self.codes.copy_(codes.to(torch.int8))
        self.key_scale.copy_(key_scale)
        self.value_scale.copy_(value_scale)
        self.factors_key.copy_(factors_key)
        self.factors_value.copy_(factors_value)
        self._clear_window()

    @torch.no_grad()
    def dequantize(self) -> torch.Tensor:
        """Materialize the checkpoint state in FP32 (transient)."""

        approx = self.factors_key.float() @ self.factors_value.float().transpose(-1, -2)
        residual = (
            self.codes.float()
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
            alpha = self.buffer_alpha[:, :, index]
            decay = (
                alpha[..., :, None]
                if self.gate_dim == self.key_dim
                else alpha[..., 0][..., None, None]
            )
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
        _check_finite("query", query)
        _check_finite("key", key)
        _check_finite("value", value)
        _check_finite("alpha", alpha)
        _check_finite("beta", beta)

        state = self._replay_window()
        decay = (
            alpha[..., :, None]
            if self.gate_dim == self.key_dim
            else alpha[..., 0][..., None, None]
        )
        decayed = state * decay
        prediction = (decayed * key[..., :, None]).sum(dim=-2)
        update = beta[..., None] * (value - prediction)
        next_state = decayed + key[..., :, None] * update[..., None, :]
        output = (next_state * query[..., :, None]).sum(dim=-2)

        index = self.window_count
        self.buffer_keys[:, :, index].copy_(key.to(self.config.buffer_dtype))
        self.buffer_updates[:, :, index].copy_(update.to(self.config.buffer_dtype))
        self.buffer_alpha[:, :, index].copy_(alpha)
        self.window_count = index + 1
        if self.window_count == self.config.window:
            self.quantize_(next_state)
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
        outputs = []
        for index in range(query.shape[1]):
            outputs.append(
                self.step(
                    query[:, index],
                    key[:, index],
                    value[:, index],
                    alpha[:, index],
                    beta[:, index],
                )
            )
        return torch.stack(outputs, dim=1)

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

    def packed_state(self) -> Dict[str, Any]:
        """Return a boundary-consistent snapshot for cache save/restore."""

        return {
            "format_version": _FORMAT_VERSION,
            "bits": self.config.bits,
            "window": self.config.window,
            "rank": self.rank,
            "smooth": self.config.smooth,
            "buffer_dtype": str(self.config.buffer_dtype),
            "gate_dim": self.gate_dim,
            "num_sequences": self.num_sequences,
            "num_heads": self.num_heads,
            "key_dim": self.key_dim,
            "value_dim": self.value_dim,
            "codes": self.codes.clone(),
            "key_scale": self.key_scale.clone(),
            "value_scale": self.value_scale.clone(),
            "factors_key": self.factors_key.clone(),
            "factors_value": self.factors_value.clone(),
            "buffer_keys": self.buffer_keys.clone(),
            "buffer_updates": self.buffer_updates.clone(),
            "buffer_alpha": self.buffer_alpha.clone(),
            "window_count": self.window_count,
        }

    def load_packed_state(self, packed: Dict[str, Any]) -> None:
        """Restore a snapshot produced by :meth:`packed_state`."""

        if int(packed.get("format_version", -1)) != _FORMAT_VERSION:
            raise ValueError("unsupported LeapQuant packed-state format version")
        expected = {
            "bits": self.config.bits,
            "window": self.config.window,
            "rank": self.rank,
            "smooth": self.config.smooth,
            "gate_dim": self.gate_dim,
            "num_sequences": self.num_sequences,
            "num_heads": self.num_heads,
            "key_dim": self.key_dim,
            "value_dim": self.value_dim,
        }
        for name, value in expected.items():
            if packed.get(name) != value:
                raise ValueError(f"LeapQuant packed-state {name} mismatch: {packed.get(name)!r} != {value!r}")
        for name in (
            "codes",
            "key_scale",
            "value_scale",
            "factors_key",
            "factors_value",
            "buffer_keys",
            "buffer_updates",
            "buffer_alpha",
        ):
            value = packed.get(name)
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"LeapQuant packed-state {name} must be a tensor")
            setattr(self, name, value.to(device=self.device).clone())
        self.window_count = int(packed["window_count"])

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
        for name in (
            "codes",
            "key_scale",
            "value_scale",
            "factors_key",
            "factors_value",
            "buffer_keys",
            "buffer_updates",
            "buffer_alpha",
        ):
            setattr(self, name, getattr(self, name).to(device))
        self.device = device
        return self

    @torch.no_grad()
    def reorder_(self, beam_idx: torch.Tensor) -> None:
        """Atomically reorder every packed component along the sequence axis."""

        index = beam_idx.to(self.device).long()
        if index.numel() != self.num_sequences:
            raise ValueError("LeapQuant reorder requires one index per live sequence")
        for name in (
            "codes",
            "key_scale",
            "value_scale",
            "factors_key",
            "factors_value",
            "buffer_keys",
            "buffer_updates",
            "buffer_alpha",
        ):
            setattr(self, name, getattr(self, name).index_select(0, index).contiguous())


class LeapQuantLinearAttentionLayer(LinearAttentionLayer):
    """Transformers linear-attention cache layer with a packed LeapQuant state."""

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
        if self._leapquant_shape != shape or self._leapquant is None or self._leapquant.device != device:
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
        if self._leapquant is not None:
            self._leapquant.reorder_(beam_idx)
        super().reorder_cache(beam_idx)

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


class LeapQuantDynamicCache(DynamicCache):
    """DynamicCache whose linear-attention layers own packed LeapQuant state."""

    def __init__(self, config=None, state_quant: Optional[LeapQuantConfig] = None, **kwargs) -> None:
        super().__init__(config=config, **kwargs)
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


def make_leapquant_cache(config, state_quant: Optional[LeapQuantConfig]):
    """Build the cache class a port should use for the requested state policy."""

    if state_quant is None:
        return DynamicCache(config=config)
    return LeapQuantDynamicCache(config=config, state_quant=state_quant)
