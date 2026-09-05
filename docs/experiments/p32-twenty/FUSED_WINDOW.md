# Fused decode and row reuse: preliminary experiment 2/3

The experimental Triton kernel reconstructs window P32 weights within each K16
MMA iteration, sharing them across a configurable 16/32/64-row activation tile.
It retains checkpoint bits and canonical FP16 codebook values, accumulates in
FP32, and optionally reduces split-K FP32 partials. It does not cache dense weights.
The harness retains planar and existing window comparators and includes SU,
Hadamards, SV, allocations, and split reduction in full-layer timings.

Layer-0 q projection, BM32/BN32/split16: **9/9 cases pass** the approved local gates.
Full-layer speed relative to the existing window kernel is approximately
0.91–1.02x across the nine M values. This configuration establishes no useful
speed gain. Raw metrics/samples are in [results/fused-window](results/fused-window).

An initial joined-pair formulation failed GEMM correctness despite reproducing
all standalone decoded weights exactly. Direct column-indexed lookup repaired
the observed failure. The compiler/layout root cause is not yet established.
Original failed debug output remains at `/root/p32-fused-window/debug.json`.
BM16/BM64 and layer-1 down follow-ups are running. Executed instruction profiling,
SASS inspection, resource accounting, full effective BPW and model evaluation
remain pending; this is an experimental, non-production implementation.
