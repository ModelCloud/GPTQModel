# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Experimental gfx950 banked Viterbi kernels, independent of NVIDIA extensions.

The trusted entry point is intentionally not a public dispatch default yet.
R[t, bank, q] = min_p F[t, bank, p * S + q] retains only the
suffix survivors; pointers retain the original bank-major/state-major tie order.
"""

import os

import torch
import triton
import triton.language as tl


def native_banked_supported(sequences, codebooks, *, bits, segment_steps, work_dtype):
    """Narrow, opt-in dispatch while full YAQA/model validation is incomplete."""
    return (
        os.environ.get("GPTQMODEL_QVQ_AMD_NATIVE_QUANTIZATION", "0") == "1"
        and torch.version.hip is not None
        and sequences.device.type == "cuda"
        and work_dtype == torch.float32
        and sequences.dtype == torch.float32
        and codebooks.dtype in (torch.float16, torch.float32)
        and sequences.is_contiguous() and codebooks.is_contiguous()
        and tuple(sequences.shape[1:]) == (128, 2) and 0 < sequences.shape[0] <= 256
        and codebooks.ndim == 3 and codebooks.shape[0] in (1, 2)
        and tuple(codebooks.shape[1:]) == (65536, 2)
        and bits in (2, 2.5, 3, 3.5) and segment_steps == 16
        and torch.cuda.get_device_properties(sequences.device).gcnArchName.split(":")[0] == "gfx950"
    )


@triton.jit(do_not_specialize=["STEP"])
def _survivor_step(X, C, W, O, Previous, Next, Pointers,
                   STEP, FIRST: tl.constexpr, MERGE: tl.constexpr, SHIFT: tl.constexpr, BANKS: tl.constexpr,
                   SEGMENT: tl.constexpr, WEIGHTED: tl.constexpr, CLOSED: tl.constexpr,
                   Q: tl.constexpr, DOT_FMA: tl.constexpr):
    batch = tl.program_id(0)
    suffix: tl.constexpr = 1 << (16 - SHIFT)
    prefixes: tl.constexpr = 1 << SHIFT
    q = tl.program_id(1) * Q + tl.arange(0, Q)
    p = tl.arange(0, prefixes)
    bank = tl.arange(0, BANKS)
    state = p[:, None] * suffix + q[None, :]
    c0 = tl.load(C + (bank[:, None, None] * 65536 + state[None, :, :]) * 2).to(tl.float32)
    c1 = tl.load(C + (bank[:, None, None] * 65536 + state[None, :, :]) * 2 + 1).to(tl.float32)
    x0 = tl.load(X + batch * 256 + STEP * 2)
    x1 = tl.load(X + batch * 256 + STEP * 2 + 1)
    if DOT_FMA:
        # Match the two-term ROCm SGEMM dot product; all other quadratic-form
        # operations retain their separate FP32 rounding (fusion disabled).
        dot = tl.fma(x1, c1, x0 * c0)
    else:
        dot = x0 * c0 + x1 * c1
    distance = tl.maximum((x0 * x0 + x1 * x1) + (c0 * c0 + c1 * c1) - 2.0 * dot, 0.0)
    if WEIGHTED:
        distance *= tl.load(W + batch * 128 + STEP)
    if FIRST:
        cost = distance
        if CLOSED:
            cost = tl.where((state[None, :, :] >> SHIFT) == tl.load(O + batch), cost, float("inf"))
    else:
        previous = tl.load(Previous + (batch * BANKS + bank[:, None, None]) * suffix
                           + (state[None, :, :] >> SHIFT))
        cost = previous + distance
    best = tl.min(cost, 1)
    prefix = tl.min(tl.where(cost == best[:, None, :], p[None, :, None], 2147483647), 1)
    pointer = bank[:, None] * prefixes + prefix
    if MERGE:
        across = tl.min(best, 0)
        winner = tl.min(tl.where(best == across[None, :], pointer, 2147483647), 0)
        best = tl.broadcast_to(across[None, :], (BANKS, Q))
        pointer = tl.broadcast_to(winner[None, :], (BANKS, Q))
    tl.store(Next + (batch * BANKS + bank[:, None]) * suffix + q[None, :], best)
    tl.store(Pointers + ((STEP * tl.num_programs(0) + batch) * BANKS + bank[:, None]) * suffix
             + q[None, :], pointer)


@triton.jit
def _traceback(C, O, Costs, Pointers, States, Values, Loss, Banks,
               SHIFT: tl.constexpr, BANKS: tl.constexpr, SEGMENT: tl.constexpr, CLOSED: tl.constexpr):
    batch = tl.program_id(0)
    suffix: tl.constexpr = 1 << (16 - SHIFT)
    prefixes: tl.constexpr = 1 << SHIFT
    ids = tl.arange(0, BANKS * suffix)
    q = ids % suffix
    bank = ids // suffix
    costs = tl.load(Costs + batch * BANKS * suffix + ids)
    ptr = tl.load(Pointers + (127 * tl.num_programs(0) + batch) * BANKS * suffix + ids).to(tl.int32)
    if CLOSED:
        costs = tl.where(q == tl.load(O + batch), costs, float("inf"))
    flat = bank * 65536 + (ptr % prefixes) * suffix + q
    minimum = tl.min(costs, 0)
    selected = tl.min(tl.where(costs == minimum, flat, 2147483647), 0)
    tl.store(Loss + batch, minimum)
    current_bank = selected // 65536
    current_state = selected % 65536
    for step in range(127, -1, -1):
        tl.store(States + batch * 128 + step, current_state.to(tl.int64))
        c0 = tl.load(C + (current_bank * 65536 + current_state) * 2)
        c1 = tl.load(C + (current_bank * 65536 + current_state) * 2 + 1)
        tl.store(Values + batch * 256 + step * 2, c0)
        tl.store(Values + batch * 256 + step * 2 + 1, c1)
        if step % SEGMENT == 0:
            tl.store(Banks + batch * (128 // SEGMENT) + step // SEGMENT, current_bank.to(tl.uint8))
        if step > 0:
            q_previous = current_state >> SHIFT
            predecessor = tl.load(Pointers + ((step - 1) * tl.num_programs(0) + batch) * BANKS * suffix
                                  + current_bank * suffix + q_previous).to(tl.int32)
            current_bank = predecessor // prefixes
            current_state = (predecessor % prefixes) * suffix + q_previous


def banked_viterbi_trusted(sequences, codebooks, *, bits, segment_steps=16,
                           overlap=None, step_weights=None, q_chunk=4):
    """Run already validated, contiguous FP32 sequences on gfx950 (experimental)."""
    if (torch.version.hip is None or sequences.device.type != "cuda"
            or torch.cuda.get_device_properties(sequences.device).gcnArchName.split(":")[0] != "gfx950"):
        raise RuntimeError("The experimental YAQA native kernel requires gfx950 ROCm")
    if (sequences.dtype != torch.float32 or codebooks.dtype not in (torch.float16, torch.float32)
            or not sequences.is_contiguous() or not codebooks.is_contiguous()):
        raise ValueError("Expected contiguous FP32 sequences and FP16/FP32 codebooks")
    if (sequences.ndim != 3 or tuple(sequences.shape[1:]) != (128, 2)
            or not 0 < sequences.shape[0] <= 256 or codebooks.ndim != 3
            or codebooks.shape[0] not in (1, 2) or tuple(codebooks.shape[1:]) != (65536, 2)
            or codebooks.device != sequences.device or bits not in (2, 2.5, 3, 3.5)
            or segment_steps != 16 or q_chunk not in (4, 8, 16, 32)):
        raise ValueError("Unsupported experimental banked recurrence geometry or rate")
    if step_weights is not None and (
        step_weights.device != sequences.device or step_weights.dtype != torch.float32
        or not step_weights.is_contiguous() or tuple(step_weights.shape) != tuple(sequences.shape[:2])
    ):
        raise ValueError("Trusted step weights must be contiguous FP32 [batch, 128] on the input device")
    if overlap is not None and (
        overlap.device != sequences.device or overlap.dtype != torch.int64 or not overlap.is_contiguous()
        or tuple(overlap.shape) != (sequences.shape[0],)
    ):
        raise ValueError("Trusted overlap must be contiguous int64 [batch] on the input device")
    batch = sequences.shape[0]
    shift = int(2 * bits)
    banks = codebooks.shape[0]
    suffix = 1 << (16 - shift)
    costs = torch.empty((2, batch, banks, suffix), device=sequences.device, dtype=torch.float32)
    pointers = torch.empty((128, batch, banks, suffix), device=sequences.device, dtype=torch.int16)
    states = torch.empty((batch, 128), device=sequences.device, dtype=torch.int64)
    values = torch.empty((batch, 128, 2), device=sequences.device, dtype=codebooks.dtype)
    loss = torch.empty((batch,), device=sequences.device, dtype=torch.float32)
    bank_ids = torch.empty((batch, 128 // segment_steps), device=sequences.device, dtype=torch.uint8)
    with torch.cuda.device(sequences.device):
        for step in range(128):
            _survivor_step[(batch, triton.cdiv(suffix, q_chunk))](
                sequences, codebooks, step_weights, overlap, costs[(step - 1) % 2], costs[step % 2], pointers,
                step, step == 0, (step + 1) % segment_steps == 0 and step != 127,
                shift, banks, segment_steps, step_weights is not None, overlap is not None, q_chunk, True,
                num_warps=4, enable_fp_fusion=False,
            )
        _traceback[(batch,)](codebooks, overlap, costs[1], pointers, states, values, loss, bank_ids,
                            shift, banks, segment_steps, overlap is not None,
                            num_warps=4, enable_fp_fusion=False)
    from ..quantization.qvq import BankedTrellisQuantizationResult

    return BankedTrellisQuantizationResult(states=states, values=values, squared_error=loss,
                                         segment_bank_ids=bank_ids)
