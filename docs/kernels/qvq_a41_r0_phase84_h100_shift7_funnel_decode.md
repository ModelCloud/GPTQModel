# Phase 84: H100 Shift-7 funnel decode

Phase 84 applies the useful part of the exact W3.5 quantization specialization
to packed inference. Inference has no predecessor search to prune and its
128-thread WGMMA consumer group cannot be reduced, but it also needs only the
low 16 bits of each Shift-7 circular state window. The W3.5 decoder now forms
that result directly with a 32-bit funnel shift instead of retaining the
unused high half of a 64-bit shift.

## Exact transformation

For adjacent packed words `lo`, `hi` and shift `s`, the prior decoder used

```text
uint32((uint64(hi) << 32 | lo) >> s)
```

and consumed only its low 16 bits. CUDA's `__funnelshift_r(lo, hi, s)` returns
the same low 32 bits, so the decoded state, PGC input, selected level, FP16
WGMMA operand, accumulation order, and output are unchanged. The specialization
is compile-time gated to `TransitionBits == 7`; W2--W3 retain their measured
paths.

## H100 machine-code result

Matched Nsight Compute 2026.2.1 profiles used the same W3.5 M1 grouped Llama
gate/up payload and physical 132-SM H100.

| Metric | Baseline | Shift-7 funnel | Change |
|---|---:|---:|---:|
| Kernel duration | 24.61 us | 24.48 us | 1.0053x |
| Static `SHF.R.U64` instances | 64 | 0 | removed |
| Registers/thread | 69 | 67 | -2 |
| Eligible warps/scheduler | 0.71 | 0.72 | +1.4% |
| Achieved occupancy | 15.01% | 15.03% | unchanged |

The launch remains one 128-thread WGMMA consumer group plus one 32-thread TMA
producer. This is already the minimum legal worker geometry for the selected
WGMMA instruction, so the quantizer's 1024-to-512-thread occupancy change does
not transfer literally.

## Complete Llama 3.2 1B MLP

An idle-gated candidate/control/candidate sandwich used 30 warmups, 300 CUDA
event samples, and 50 warmed CUDA Graph replays per sample at M1, M2, M4, M8,
and M16. Comparing the control with the geometric mean of the two surrounding
candidate runs improved all five cells and the complete-MLP geometric mean by
**1.00195x**. The movement is deliberately reported as a micro-optimization,
not a large inference gain.

Correctness validation passed 211 Hopper P32 tests with 24 FP8/H200-only
skips. Coverage includes W2--W3.5 dense-oracle bounds, exact grouped/plain
child parity, ordered reductions, large-M tiling, repeatability, and CUDA
Graph replay. Checkpoint bytes, selector layout, persistent VRAM, launches,
and numerical output are unchanged.

Artifacts outside Git:

- `/root/qvq-results/phase84-shift7-funnel-candidate-300a.json`
- `/root/qvq-results/phase84-shift7-baseline-300b.json`
- `/root/qvq-results/phase84-shift7-funnel-candidate-300c.json`
- `/root/qvq-profiler-artifacts/phase84-shift7-inference/`

