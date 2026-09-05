# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Experimental gfx950 banked Viterbi kernels, independent of NVIDIA extensions.

The trusted entry point is intentionally not a public dispatch default yet.
R[t, bank, q] = min_p F[t, bank, p * S + q] retains only the
suffix survivors; pointers retain the original bank-major/state-major tie order.
"""

import os
import threading
from collections import OrderedDict

import torch
import triton
import triton.language as tl

_GRAPH_CACHE_LOCK = threading.Lock()
_GRAPH_CACHE = OrderedDict()
_GRAPH_CACHE_LIMIT = 8


def _family_q_chunk(bits):
    # Measured graph-replay minima on gfx950 for four YAQA families.
    return {2: 16, 2.5: 4, 3: 2, 3.5: 4}[bits]


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
                   Q: tl.constexpr, STORE_POINTER: tl.constexpr, POINTER_OFFSET: tl.constexpr,
                   FAMILY_BATCH: tl.constexpr, SEQUENCES_PER_FAMILY: tl.constexpr):
    batch = tl.program_id(0)
    if FAMILY_BATCH:
        family = batch // SEQUENCES_PER_FAMILY
    else:
        family = 0
    suffix: tl.constexpr = 1 << (16 - SHIFT)
    prefixes: tl.constexpr = 1 << SHIFT
    q = tl.program_id(1) * Q + tl.arange(0, Q)
    p = tl.arange(0, prefixes)
    bank = tl.arange(0, BANKS)
    state = p[:, None] * suffix + q[None, :]
    codebook_index = (family * BANKS * 65536 + bank[:, None, None] * 65536 + state[None, :, :]) * 2
    c0 = tl.load(C + codebook_index).to(tl.float32)
    c1 = tl.load(C + codebook_index + 1).to(tl.float32)
    x0 = tl.load(X + batch * 256 + STEP * 2)
    x1 = tl.load(X + batch * 256 + STEP * 2 + 1)
    # Match the two-term ROCm SGEMM dot product; all other quadratic-form
    # operations retain their separate FP32 rounding (fusion disabled).
    dot = tl.fma(x1, c1, x0 * c0)
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
    if STORE_POINTER:
        tl.store(Pointers + (((STEP - POINTER_OFFSET) * tl.num_programs(0) + batch) * BANKS
                 + bank[:, None]) * suffix + q[None, :], pointer)


@triton.jit
def _traceback(C, O, Costs, Pointers, States, Values, Loss, Banks,
               SHIFT: tl.constexpr, BANKS: tl.constexpr, SEGMENT: tl.constexpr, CLOSED: tl.constexpr,
               FAMILY_BATCH: tl.constexpr, SEQUENCES_PER_FAMILY: tl.constexpr):
    batch = tl.program_id(0)
    if FAMILY_BATCH:
        family = batch // SEQUENCES_PER_FAMILY
    else:
        family = 0
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
        codebook_index = (family * BANKS * 65536 + current_bank * 65536 + current_state) * 2
        c0 = tl.load(C + codebook_index)
        c1 = tl.load(C + codebook_index + 1)
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


@triton.jit
def _midpoint_traceback(Costs, Pointers, Overlap, SHIFT: tl.constexpr, BANKS: tl.constexpr):
    batch = tl.program_id(0)
    suffix: tl.constexpr = 1 << (16 - SHIFT)
    prefixes: tl.constexpr = 1 << SHIFT
    ids = tl.arange(0, BANKS * suffix)
    q = ids % suffix
    bank = ids // suffix
    costs = tl.load(Costs + batch * BANKS * suffix + ids)
    ptr = tl.load(Pointers + ((127 - 63) * tl.num_programs(0) + batch) * BANKS * suffix + ids).to(tl.int32)
    flat = bank * 65536 + (ptr % prefixes) * suffix + q
    minimum = tl.min(costs, 0)
    selected = tl.min(tl.where(costs == minimum, flat, 2147483647), 0)
    current_bank = selected // 65536
    current_state = selected % 65536
    for step in range(127, 63, -1):
        q_previous = current_state >> SHIFT
        predecessor = tl.load(Pointers + (((step - 1 - 63) * tl.num_programs(0) + batch) * BANKS
                              + current_bank) * suffix + q_previous).to(tl.int32)
        current_bank = predecessor // prefixes
        current_state = (predecessor % prefixes) * suffix + q_previous
    tl.store(Overlap + batch, current_state % suffix)


def _banked_viterbi_launch(sequences, codebooks, *, bits, segment_steps, overlap, step_weights,
                            q_chunk, families, sequences_per_family):
    batch = sequences.shape[0]
    shift = int(2 * bits)
    banks = codebooks.shape[-3]
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
                shift, banks, segment_steps, step_weights is not None, overlap is not None, q_chunk,
                True, 0, families > 1, sequences_per_family,
                num_warps=4, enable_fp_fusion=False,
            )
        _traceback[(batch,)](codebooks, overlap, costs[1], pointers, states, values, loss, bank_ids,
                            shift, banks, segment_steps, overlap is not None,
                            families > 1, sequences_per_family,
                            num_warps=4, enable_fp_fusion=False)
    from ..quantization.qvq import BankedTrellisQuantizationResult

    return BankedTrellisQuantizationResult(states=states, values=values, squared_error=loss,
                                         segment_bank_ids=bank_ids)


def _family_midpoint_launch(sequences, codebooks, *, bits, segment_steps):
    families, count = sequences.shape[:2]
    flat_sequences = sequences.reshape(-1, 128, 2)
    batch = flat_sequences.shape[0]
    shift = int(2 * bits)
    banks = codebooks.shape[1]
    suffix = 1 << (16 - shift)
    costs = torch.empty((2, batch, banks, suffix), device=sequences.device, dtype=torch.float32)
    pointers = torch.empty((65, batch, banks, suffix), device=sequences.device, dtype=torch.int16)
    overlap = torch.empty((batch,), device=sequences.device, dtype=torch.int64)
    q_chunk = _family_q_chunk(bits)
    with torch.cuda.device(sequences.device):
        for step in range(128):
            _survivor_step[(batch, triton.cdiv(suffix, q_chunk))](
                flat_sequences, codebooks, None, None, costs[(step - 1) % 2], costs[step % 2], pointers,
                step, step == 0, (step + 1) % segment_steps == 0 and step != 127,
                shift, banks, segment_steps, False, False, q_chunk, step >= 63, 63, True, count,
                num_warps=4, enable_fp_fusion=False,
            )
        _midpoint_traceback[(batch,)](costs[1], pointers, overlap, shift, banks,
                                     num_warps=4, enable_fp_fusion=False)
    return overlap.reshape(families, count)


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
    return _banked_viterbi_launch(
        sequences, codebooks, bits=bits, segment_steps=segment_steps, overlap=overlap,
        step_weights=step_weights, q_chunk=q_chunk, families=1,
        sequences_per_family=sequences.shape[0],
    )


def family_banked_viterbi_trusted(sequences, codebooks, *, bits, segment_steps=16, overlap=None):
    """Batch independent codebook families without changing their recurrence order."""
    if (sequences.ndim != 4 or tuple(sequences.shape[2:]) != (128, 2)
            or codebooks.ndim != 4 or codebooks.shape[0] != sequences.shape[0]
            or codebooks.shape[1] != 2 or tuple(codebooks.shape[2:]) != (65536, 2)
            or sequences.dtype != torch.float32 or codebooks.dtype not in (torch.float16, torch.float32)
            or sequences.device != codebooks.device or not sequences.is_contiguous() or not codebooks.is_contiguous()
            or not 0 < sequences.numel() // 256 <= 256 or bits not in (2, 2.5, 3, 3.5)
            or segment_steps != 16):
        raise ValueError("Unsupported family-batched ROCm recurrence")
    families, count = sequences.shape[:2]
    flat_overlap = None
    if overlap is not None:
        if overlap.dtype != torch.int64 or not overlap.is_contiguous() or tuple(overlap.shape) != (families, count):
            raise ValueError("Family overlap must be contiguous int64 [family, batch]")
        flat_overlap = overlap.reshape(-1)
    result = _banked_viterbi_launch(
        sequences.reshape(-1, 128, 2), codebooks, bits=bits, segment_steps=segment_steps,
        overlap=flat_overlap, step_weights=None, q_chunk=_family_q_chunk(bits), families=families,
        sequences_per_family=count,
    )
    from ..quantization.qvq import BankedTrellisQuantizationResult

    return BankedTrellisQuantizationResult(
        states=result.states.reshape(families, count, 128),
        values=result.values.reshape(families, count, 128, 2),
        squared_error=result.squared_error.reshape(families, count),
        segment_bank_ids=result.segment_bank_ids.reshape(families, count, 8),
    )


def family_banked_viterbi_midpoint_trusted(sequences, codebooks, *, bits, segment_steps=16):
    """Return only provisional midpoint overlaps for exact two-pass tail biting."""
    if (sequences.ndim != 4 or tuple(sequences.shape[2:]) != (128, 2)
            or codebooks.ndim != 4 or codebooks.shape[0] != sequences.shape[0]
            or codebooks.shape[1] != 2 or tuple(codebooks.shape[2:]) != (65536, 2)
            or sequences.dtype != torch.float32 or codebooks.dtype not in (torch.float16, torch.float32)
            or sequences.device != codebooks.device or not sequences.is_contiguous() or not codebooks.is_contiguous()
            or not 0 < sequences.numel() // 256 <= 256 or bits not in (2, 2.5, 3, 3.5)
            or segment_steps != 16):
        raise ValueError("Unsupported family-batched ROCm midpoint recurrence")
    return _family_midpoint_launch(sequences, codebooks, bits=bits, segment_steps=segment_steps)


class _FamilyMidpointGraph:
    """Graph the provisional recurrence while retaining only exact overlaps."""

    def __init__(self, sequences, codebooks, *, bits, segment_steps):
        self.sequences = torch.empty_like(sequences)
        self.codebooks = torch.empty_like(codebooks)
        self.codebooks.copy_(codebooks)
        family_banked_viterbi_midpoint_trusted(
            self.sequences, self.codebooks, bits=bits, segment_steps=segment_steps
        )
        torch.cuda.current_stream(sequences.device).synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.output = family_banked_viterbi_midpoint_trusted(
                self.sequences, self.codebooks, bits=bits, segment_steps=segment_steps
            )

    def run(self, sequences, codebooks):
        self.sequences.copy_(sequences)
        self.codebooks.copy_(codebooks)
        self.graph.replay()
        return self.output


def family_banked_viterbi_midpoint_graph_trusted(sequences, codebooks, *, bits, segment_steps=16):
    """Replay the internal exact midpoint-only family graph."""
    stream_id = torch.cuda.current_stream(sequences.device).cuda_stream
    key = (
        "family_midpoint", sequences.device.index, stream_id, tuple(sequences.shape), tuple(codebooks.shape),
        codebooks.dtype, int(2 * bits), segment_steps,
    )
    with _GRAPH_CACHE_LOCK:
        executor = _GRAPH_CACHE.get(key)
        if executor is None:
            executor = _FamilyMidpointGraph(sequences, codebooks, bits=bits, segment_steps=segment_steps)
            _GRAPH_CACHE[key] = executor
            if len(_GRAPH_CACHE) > _GRAPH_CACHE_LIMIT:
                _GRAPH_CACHE.popitem(last=False)
        else:
            _GRAPH_CACHE.move_to_end(key)
    return executor.run(sequences, codebooks)


class _FamilyBankedViterbiGraph:
    """Graph one sequential family solve; fixed outputs are consumed before replay."""

    def __init__(self, sequences, codebooks, *, bits, segment_steps, overlap):
        self.sequences = torch.empty_like(sequences)
        self.codebooks = torch.empty_like(codebooks)
        self.overlap = torch.empty_like(overlap) if overlap is not None else None
        self.codebooks.copy_(codebooks)
        family_banked_viterbi_trusted(
            self.sequences, self.codebooks, bits=bits, segment_steps=segment_steps, overlap=self.overlap
        )
        torch.cuda.current_stream(sequences.device).synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.output = family_banked_viterbi_trusted(
                self.sequences, self.codebooks, bits=bits, segment_steps=segment_steps, overlap=self.overlap
            )

    def run(self, sequences, codebooks, *, overlap):
        self.sequences.copy_(sequences)
        self.codebooks.copy_(codebooks)
        if self.overlap is not None:
            self.overlap.copy_(overlap)
        self.graph.replay()
        return self.output


def family_banked_viterbi_graph_trusted(sequences, codebooks, *, bits, segment_steps=16, overlap=None):
    """Replay an internal graph whose output is valid until its next shape-matched call."""
    stream_id = torch.cuda.current_stream(sequences.device).cuda_stream
    key = (
        "family", sequences.device.index, stream_id, tuple(sequences.shape), tuple(codebooks.shape),
        codebooks.dtype, int(2 * bits), segment_steps, overlap is not None,
    )
    with _GRAPH_CACHE_LOCK:
        executor = _GRAPH_CACHE.get(key)
        if executor is None:
            executor = _FamilyBankedViterbiGraph(
                sequences, codebooks, bits=bits, segment_steps=segment_steps, overlap=overlap
            )
            _GRAPH_CACHE[key] = executor
            if len(_GRAPH_CACHE) > _GRAPH_CACHE_LIMIT:
                _GRAPH_CACHE.popitem(last=False)
        else:
            _GRAPH_CACHE.move_to_end(key)
    return executor.run(sequences, codebooks, overlap=overlap)


class _BankedViterbiGraph:
    """One stream/codebook/shape-specific ROCm graph with owned input buffers."""

    def __init__(self, sequences, codebooks, *, bits, segment_steps, overlap, step_weights):
        self.sequences = torch.empty_like(sequences)
        self.overlap = torch.empty_like(overlap) if overlap is not None else None
        self.step_weights = torch.empty_like(step_weights) if step_weights is not None else None
        # Compile and initialize Triton outside capture. The graph intentionally
        # reads the caller-owned codebook storage so immutable YAQA libraries do
        # not incur a 0.5 MiB copy per recurrence.
        banked_viterbi_trusted(
            self.sequences, codebooks, bits=bits, segment_steps=segment_steps,
            overlap=self.overlap, step_weights=self.step_weights,
        )
        torch.cuda.current_stream(sequences.device).synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.output = banked_viterbi_trusted(
                self.sequences, codebooks, bits=bits, segment_steps=segment_steps,
                overlap=self.overlap, step_weights=self.step_weights,
            )

    def run(self, sequences, *, overlap, step_weights):
        self.sequences.copy_(sequences)
        if self.overlap is not None:
            self.overlap.copy_(overlap)
        if self.step_weights is not None:
            self.step_weights.copy_(step_weights)
        self.graph.replay()
        # A graph owns fixed output addresses. Clone before returning so the
        # next quantized tile cannot mutate a previously returned result.
        from ..quantization.qvq import BankedTrellisQuantizationResult

        return BankedTrellisQuantizationResult(
            states=self.output.states.clone(),
            values=self.output.values.clone(),
            squared_error=self.output.squared_error.clone(),
            segment_bank_ids=self.output.segment_bank_ids.clone(),
        )


def banked_viterbi_graph(sequences, codebooks, *, bits, segment_steps=16,
                         overlap=None, step_weights=None):
    """Replay a bounded, stream-specific native graph for repeated YAQA solves."""
    stream_id = torch.cuda.current_stream(sequences.device).cuda_stream
    key = (
        "single", sequences.device.index, stream_id, codebooks.data_ptr(), tuple(sequences.shape),
        codebooks.dtype, int(2 * bits), segment_steps, overlap is not None, step_weights is not None,
    )
    with _GRAPH_CACHE_LOCK:
        executor = _GRAPH_CACHE.get(key)
        if executor is None:
            executor = _BankedViterbiGraph(
                sequences, codebooks, bits=bits, segment_steps=segment_steps,
                overlap=overlap, step_weights=step_weights,
            )
            _GRAPH_CACHE[key] = executor
            if len(_GRAPH_CACHE) > _GRAPH_CACHE_LIMIT:
                _GRAPH_CACHE.popitem(last=False)
        else:
            _GRAPH_CACHE.move_to_end(key)
    return executor.run(sequences, overlap=overlap, step_weights=step_weights)


def clear_banked_viterbi_graph_cache():
    """Release graph-owned scratch, primarily for tests and quantizer teardown."""
    with _GRAPH_CACHE_LOCK:
        _GRAPH_CACHE.clear()
