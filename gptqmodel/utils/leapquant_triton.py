# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Fused Triton kernels for the LeapQuant recurrent-state path.

The portable Torch implementation of :mod:`gptqmodel.utils.leapquant` is the
correctness reference. It replays every buffered delta-rule update into a
transient dense state and launches tens of small kernels per decoded token.
The kernels here keep the same arithmetic in registers:

* ``_leapquant_step_kernel`` dequantizes one ``[K, V]`` state slice, replays
  the in-window updates, computes the prediction, the new update and the
  readout, and stages the rounded key/update record. The caller commits the
  record only after the finiteness flag passes, so a rejected token leaves
  the packed state unchanged.
* ``_mgs_kernel`` orthonormalizes the narrow randomized-fit sketch with
  modified Gram-Schmidt. It replaces the Cholesky and triangular-solve pair
  that dominates the compensator fit on accelerators.

Both kernels compute in FP32. The decode kernel reads the packed code
container directly, so no dense state is materialized or retained. On any
unsupported shape or runtime the callers return ``None`` and the Torch path
stays authoritative.
"""

from __future__ import annotations

from typing import Optional

import torch

from .logger import setup_logger


log = setup_logger()

try:  # pragma: no cover - import availability is environment dependent
    import triton
    import triton.language as tl

    from .triton_jit import jit as triton_jit
    TRITON_AVAILABLE = True
except Exception:  # pragma: no cover
    triton = None
    tl = None
    TRITON_AVAILABLE = False


_BUFFER_DTYPE_CODES = {
    torch.float32: 0,
    torch.float16: 1,
    torch.bfloat16: 2,
}
_MAX_STEP_KEY_DIM = 256
_MAX_STEP_RANK = 16
_MAX_SKETCH_KEY_DIM = 256
_MAX_SKETCH_COLUMNS = 16
_STEP_BLOCK_V = 32
_SKETCH_BLOCK = 4

_STEP_FAILED = False
_SKETCH_FAILED = False


def _device_index(device: torch.device) -> int:
    if device.index is not None:
        return device.index
    return torch.cuda.current_device()


def _available(device: torch.device) -> bool:
    """Whether Triton can target ``device``."""

    if not TRITON_AVAILABLE or device.type != "cuda":
        return False
    if not torch.cuda.is_available():
        return False
    index = _device_index(device)
    return 0 <= index < torch.cuda.device_count()


def step_supported(
    device: torch.device,
    *,
    key_dim: int,
    value_dim: int,
    window: int,
    rank: int,
    gate_dim: int,
    bits: int,
    buffer_dtype: torch.dtype,
    tensors: tuple[torch.Tensor, ...],
) -> bool:
    """Return whether the fused decode kernel accepts this state geometry."""

    if _STEP_FAILED or not _available(device):
        return False
    if key_dim < 1 or value_dim < 1 or window < 1:
        return False
    if key_dim > _MAX_STEP_KEY_DIM or rank > _MAX_STEP_RANK:
        return False
    if gate_dim not in (1, key_dim) or bits not in (4, 6, 8):
        return False
    if buffer_dtype not in _BUFFER_DTYPE_CODES:
        return False
    return all(tensor.is_contiguous() for tensor in tensors)


def sketch_supported(device: torch.device, *, key_dim: int, columns: int) -> bool:
    """Return whether the fused sketch orthonormalization is usable."""

    if _SKETCH_FAILED or not _available(device):
        return False
    return 1 <= key_dim <= _MAX_SKETCH_KEY_DIM and 1 <= columns <= _MAX_SKETCH_COLUMNS


def _compile_error(exc: Exception) -> bool:
    return type(exc).__name__ in (
        "CompilationError",
        "OutOfResources",
        "MissingTTIR",
        "PTXASError",
    ) or isinstance(exc, (ImportError, AttributeError))


def _fallback(name: str, exc: Exception) -> None:
    global _STEP_FAILED, _SKETCH_FAILED

    if name == "step":
        _STEP_FAILED = True
    else:
        _SKETCH_FAILED = True
    log.warning("LeapQuant %s Triton kernel unavailable; using the Torch path: %s", name, exc)


if TRITON_AVAILABLE:

    @triton_jit
    def _leapquant_step_kernel(
        codes_ptr,
        kscale_ptr,
        vscale_ptr,
        fk_ptr,
        fv_ptr,
        bkey_ptr,
        bupd_ptr,
        balpha_ptr,
        q_ptr,
        k_ptr,
        v_ptr,
        alpha_ptr,
        beta_ptr,
        out_ptr,
        upd_ptr,
        staged_key_ptr,
        staged_update_ptr,
        nstate_ptr,
        bad_ptr,
        n,
        K: tl.constexpr,
        BK: tl.constexpr,
        V: tl.constexpr,
        BV: tl.constexpr,
        L: tl.constexpr,
        R: tl.constexpr,
        BITS: tl.constexpr,
        GATE: tl.constexpr,
        WRITE_STATE: tl.constexpr,
        BUF: tl.constexpr,
    ):
        """One LeapQuant decode step for a ``[K, BV]`` value slice.

        ``codes`` is the persistent packed container. 4/6-bit codes are
        unpacked in registers; the logical code grid is symmetric with
        ``qmax = 2 ** (BITS - 1) - 1``. The window buffer holds the rounded
        key/update records, exactly like the Torch replay. The current record
        is written to the staging tensors, not the persistent buffer; the
        caller commits it only after the finiteness flag passes.
        """

        pid = tl.program_id(0)
        NV: tl.constexpr = (V + BV - 1) // BV
        bh = pid // NV
        vb = pid % NV
        offs_k = tl.arange(0, BK)
        offs_v = vb * BV + tl.arange(0, BV)
        mk = offs_k < K
        mv = offs_v < V

        if BITS == 8:
            W: tl.constexpr = V
            base = (bh * K + offs_k)[:, None] * W + offs_v[None, :]
            code = tl.load(codes_ptr + base, mask=mk[:, None] & mv[None, :], other=0).to(tl.float32)
        elif BITS == 4:
            W: tl.constexpr = (V + 1) // 2
            vcol = offs_v // 2
            base = (bh * K + offs_k)[:, None] * W + vcol[None, :]
            packed = tl.load(codes_ptr + base, mask=mk[:, None] & mv[None, :], other=0).to(tl.int32)
            shift = (offs_v % 2) * 4
            code = ((packed >> shift[None, :]) & 0xF).to(tl.float32)
            code = tl.where(code >= 8, code - 16, code)
        else:
            W: tl.constexpr = ((V + 3) // 4) * 3
            group = offs_v // 4
            bit = (offs_v % 4) * 6
            byte = group * 3 + bit // 8
            shift = bit % 8
            base = (bh * K + offs_k)[:, None] * W
            lo = tl.load(
                codes_ptr + base + byte[None, :],
                mask=mk[:, None] & mv[None, :],
                other=0,
            ).to(tl.int32)
            hi = tl.load(
                codes_ptr + base + byte[None, :] + 1,
                mask=mk[:, None] & mv[None, :] & ((byte + 1) < W)[None, :],
                other=0,
            ).to(tl.int32)
            raw = lo | (hi << 8)
            code = ((raw >> shift[None, :]) & 0x3F).to(tl.float32)
            code = tl.where(code >= 32, code - 64, code)

        ks = tl.load(kscale_ptr + bh * K + offs_k, mask=mk, other=0.0)
        vs = tl.load(vscale_ptr + bh * V + offs_v, mask=mv, other=0.0)
        QMAX: tl.constexpr = (1 << (BITS - 1)) - 1
        state = (code / QMAX) * ks[:, None] * vs[None, :]

        if R > 0:
            comp = tl.zeros((BK, BV), dtype=tl.float32)
            for r in tl.static_range(R):
                fk = tl.load(fk_ptr + (bh * K + offs_k) * R + r, mask=mk, other=0.0).to(tl.float32)
                fv = tl.load(fv_ptr + (bh * V + offs_v) * R + r, mask=mv, other=0.0).to(tl.float32)
                comp += fk[:, None] * fv[None, :]
            state += comp

        for i in range(n):
            if GATE == 1:
                decay = tl.load(balpha_ptr + (bh * L + i))
                state = state * decay
            else:
                decay = tl.load(balpha_ptr + (bh * L + i) * K + offs_k, mask=mk, other=0.0)
                state = state * decay[:, None]
            bkey = tl.load(bkey_ptr + (bh * L + i) * K + offs_k, mask=mk, other=0.0).to(tl.float32)
            bupd = tl.load(bupd_ptr + (bh * L + i) * V + offs_v, mask=mv, other=0.0).to(tl.float32)
            state = state + bkey[:, None] * bupd[None, :]

        qt = tl.load(q_ptr + bh * K + offs_k, mask=mk, other=0.0)
        kt = tl.load(k_ptr + bh * K + offs_k, mask=mk, other=0.0)
        vt = tl.load(v_ptr + bh * V + offs_v, mask=mv, other=0.0)
        beta = tl.load(beta_ptr + bh)
        if GATE == 1:
            at = tl.load(alpha_ptr + bh)
            decayed = state * at
        else:
            at = tl.load(alpha_ptr + bh * K + offs_k, mask=mk, other=0.0)
            decayed = state * at[:, None]

        prediction = tl.sum(kt[:, None] * decayed, axis=0)
        update = beta * (vt - prediction)
        next_state = decayed + kt[:, None] * update[None, :]
        output = tl.sum(qt[:, None] * next_state, axis=0)

        tl.store(out_ptr + bh * V + offs_v, output, mask=mv)
        tl.store(upd_ptr + bh * V + offs_v, update, mask=mv)

        if BUF == 0:
            buffered_key = kt
            buffered_update = update
        elif BUF == 1:
            buffered_key = kt.to(tl.float16)
            buffered_update = update.to(tl.float16)
        else:
            buffered_key = kt.to(tl.bfloat16)
            buffered_update = update.to(tl.bfloat16)

        # The buffer-dtype cast can turn a finite FP32 value into an inf, so
        # check the casted record as well as the FP32 output.
        bad_o = (output != output) | (tl.abs(output) == float("inf"))
        bad_u = (buffered_update != buffered_update) | (tl.abs(buffered_update) == float("inf"))
        bad_k = (buffered_key != buffered_key) | (tl.abs(buffered_key) == float("inf"))
        bad_count = tl.sum(bad_o.to(tl.int32)) + tl.sum(bad_u.to(tl.int32)) + tl.sum(bad_k.to(tl.int32))
        tl.store(bad_ptr + pid, bad_count > 0)

        tl.store(staged_key_ptr + bh * K + offs_k, buffered_key, mask=mk)
        tl.store(staged_update_ptr + bh * V + offs_v, buffered_update, mask=mv)
        if WRITE_STATE:
            base = (bh * K + offs_k)[:, None] * V + offs_v[None, :]
            tl.store(nstate_ptr + base, next_state, mask=mk[:, None] & mv[None, :])

    @triton_jit
    def _mgs_kernel(
        matrix_ptr,
        out_ptr,
        K: tl.constexpr,
        BK: tl.constexpr,
        N: tl.constexpr,
        BN: tl.constexpr,
    ):
        """Modified Gram-Schmidt for one ``[K, N]`` sketch matrix."""

        bh = tl.program_id(0)
        offs_k = tl.arange(0, BK)
        offs_n = tl.arange(0, BN)
        mk = offs_k < K
        mn = offs_n < N
        base = bh * K * N
        columns = tl.zeros((BK, BN), dtype=tl.float32)
        first = tl.load(matrix_ptr + base + offs_k * N, mask=mk, other=0.0)
        reference = tl.sqrt(tl.sum(first * first))
        for j in tl.static_range(N):
            vector = tl.load(matrix_ptr + base + offs_k * N + j, mask=mk, other=1.0)
            for i in tl.static_range(j):
                basis = tl.sum(tl.where(offs_n[None, :] == i, columns, 0.0), axis=1)
                vector = vector - tl.sum(basis * vector) * basis
            norm = tl.sqrt(tl.sum(vector * vector))
            # A rank-deficient sketch leaves only rounding noise in the
            # dependent columns. Normalizing that noise would turn it into an
            # arbitrary unit basis and inject FP16 rounding into the fitted
            # factors, so drop it relative to the leading column instead.
            keep = norm > reference * 1e-6
            vector = tl.where(keep, vector / tl.maximum(norm, 1e-12), 0.0)
            columns = tl.where(offs_n[None, :] == j, vector[:, None], columns)
        tl.store(
            out_ptr + base + offs_k[:, None] * N + offs_n[None, :],
            columns,
            mask=mk[:, None] & mn[None, :],
        )


def launch_step(
    *,
    codes: torch.Tensor,
    key_scale: torch.Tensor,
    value_scale: torch.Tensor,
    factors_key: torch.Tensor,
    factors_value: torch.Tensor,
    buffer_keys: torch.Tensor,
    buffer_updates: torch.Tensor,
    buffer_alpha: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    window_count: int,
    window: int,
    rank: int,
    gate_dim: int,
    bits: int,
    buffer_dtype: torch.dtype,
    device: torch.device,
    scratch: Optional[dict] = None,
) -> Optional[
    tuple[
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]
]:
    """Run one fused decode step, or return ``None`` for the Torch path."""

    state_tensors = (
        codes,
        key_scale,
        value_scale,
        factors_key,
        factors_value,
        buffer_keys,
        buffer_updates,
        buffer_alpha,
    )
    key_dim = key_scale.shape[-1]
    value_dim = value_scale.shape[-1]
    if not step_supported(
        device,
        key_dim=key_dim,
        value_dim=value_dim,
        window=window,
        rank=rank,
        gate_dim=gate_dim,
        bits=bits,
        buffer_dtype=buffer_dtype,
        tensors=state_tensors,
    ):
        return None

    batch, heads = codes.shape[0], codes.shape[1]
    boundary = window_count + 1 == window
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    alpha = alpha.contiguous()
    beta = beta.contiguous()

    def _scratch(name, shape, dtype):
        if scratch is None:
            return torch.empty(shape, dtype=dtype, device=device)
        tensor = scratch.get(name)
        if tensor is None or tensor.shape != shape or tensor.dtype != dtype:
            tensor = torch.empty(shape, dtype=dtype, device=device)
            scratch[name] = tensor
        return tensor

    # The caller owns the returned output; the other operands are internal and
    # can be reused between steps.
    output = torch.empty(batch, heads, value_dim, dtype=torch.float32, device=device)
    update = _scratch("update", (batch, heads, value_dim), torch.float32)
    staged_key = _scratch("staged_key", (batch, heads, key_dim), buffer_dtype)
    staged_update = _scratch("staged_update", (batch, heads, value_dim), buffer_dtype)
    next_state = (
        _scratch("next_state", (batch, heads, key_dim, value_dim), torch.float32)
        if boundary
        else _scratch("next_state_dummy", (1,), torch.float32)
    )
    block_v = _STEP_BLOCK_V
    grid = batch * heads * ((value_dim + block_v - 1) // block_v)
    bad = _scratch("bad", (grid,), torch.int32)
    block_k = triton.next_power_of_2(key_dim)

    try:
        with torch.cuda.device(device):
            _leapquant_step_kernel[(grid,)](
                codes,
                key_scale,
                value_scale,
                factors_key,
                factors_value,
                buffer_keys,
                buffer_updates,
                buffer_alpha,
                query,
                key,
                value,
                alpha,
                beta,
                output,
                update,
                staged_key,
                staged_update,
                next_state,
                bad,
                window_count,
                key_dim,
                block_k,
                value_dim,
                block_v,
                window,
                rank,
                bits,
                gate_dim,
                boundary,
                _BUFFER_DTYPE_CODES[buffer_dtype],
            )
    except Exception as exc:  # noqa: BLE001 - a compile failure falls back, runtime errors do not
        if not _compile_error(exc):
            raise
        _fallback("step", exc)
        return None

    return output, update, next_state if boundary else None, bad, staged_key, staged_update


def orthonormalize(matrix: torch.Tensor) -> Optional[torch.Tensor]:
    """Return an orthonormal basis for ``matrix``'s columns, or ``None``."""

    batch, heads, key_dim, columns = matrix.shape
    if not sketch_supported(matrix.device, key_dim=key_dim, columns=columns):
        return None
    if not matrix.is_contiguous():
        matrix = matrix.contiguous()
    out = torch.empty_like(matrix)
    block_k = triton.next_power_of_2(key_dim)
    block_n = triton.next_power_of_2(columns)
    try:
        with torch.cuda.device(matrix.device):
            _mgs_kernel[(batch * heads,)](
                matrix,
                out,
                key_dim,
                block_k,
                columns,
                block_n,
                num_warps=_SKETCH_BLOCK,
            )
    except Exception as exc:  # noqa: BLE001 - a compile failure falls back, runtime errors do not
        if not _compile_error(exc):
            raise
        _fallback("sketch", exc)
        return None
    return out
