# QVQ `<4,2,16,fused>` family-grid kernel optimisation — working status

Branch `perf/qvq-grid-kernel-opt` (base `perf/qvq-nsys-profile` @ acd959ed). Target: the
`viterbi_v2_segment_family_grid_trusted` op (61 % of GPU kernel time on Llama-3.2-1B, see
`docs/qvq_nsys_profile_llama32_1b.md`). Device: NVIDIA PG506-230 (sm_80, 124 SMs).

## Prior negative results (PRs #2, #4, #5, #6) and what this design avoids

* #2/#4: the plain Viterbi kernels are **instruction-bound (SM 62-69 %, DRAM < 1 %)**; double-buffered
  staging / unroll tweaks gave only single digits, and "batched-ILP" emission loops were already applied.
  -> this work does not re-tune unroll/barrier counts of the existing grid kernel; it changes *what work is
  done* (one emission evaluated per state for both banks, half the codebook traffic) and the launch structure.
* #4 round-3: FP32 input tiles (shared-memory pressure, -25 %) and WMMA dispatch for the GEMV were losses.
  -> no tensor-core / dtype-widening attempt here; the recurrence is min-plus, not a GEMM.
* #4: "FP16 codebook records for Viterbi L2 traffic (-33 % bytes/state)" was listed as a credible but
  quality-gated lever. -> the fused kernel reads the fp16 pair (4 B) and recomputes the norm with the exact
  reference FP32 expression instead of the 8 B packed record, so it takes the traffic win **bit-exactly** with
  no pipeline dtype change.
* #4: "Viterbi step-segmentation with overlap recomputation" (grid starvation at small batch).
  -> not needed: the real workload batches 3x128 sequences (768 CTAs); the fused kernel is persistent
  (one CTA per SM, atomic work queue) so wave quantisation disappears without changing the algorithm.
* #5/#6 are GEMV-only (inference path) and do not touch this op.

## Baseline (synthetic micro-benchmark, real shape 3 families x 128 sequences, W2, fp16 codebooks)

`qvq_v2_segment_grid_kernel<4,2,16,fused>` old path: **9.05 ms/call** (matches the 9.65 ms/call nsys number
from the real 2-layer capture, so the micro-benchmark is a faithful proxy). ncu: 768 CTAs, 3.1 waves,
L2 throughput 74 %, 32 regs, 47 % of stall cycles on L1TEX scoreboard (the 8 B packed codebook+norm record).

## Results so far

| variant | 3x128 ms/call | 3x32 ms/call | bit-exact |
|---|---:|---:|---|
| old path (8 grid + finalize + norm-pack launches) | 9.05 | 2.40 | ref |
| fused v1: persistent CTA, XOR-mask bank sharing, 128 KB smem codebook half, fp16 pairs | **4.17** | 1.38 | yes (0 mismatches over 168 configs) |
| v2: 64-bit (cost,prefix) keys on the FP64 min pipe | 6.28 | 2.06 | yes — **slower** (DSETP.MIN expands to 416 DSETP + IMAD.MOV pairs, +30 % instructions); reverted |

v1 ncu: FMA pipe 67 % utilised, issue slots 76 % busy, 50 % occupancy (64 regs x 1024 thr), L2 27 %:
the kernel is now FP32-issue-bound, not memory-bound.

## Real-workload 2-layer nsys numbers

Filled in below as captures complete (`artifacts/nsys/llama32_1b_layers2_{base,fused}_*.csv`).
