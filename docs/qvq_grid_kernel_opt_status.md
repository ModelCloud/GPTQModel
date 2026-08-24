# QVQ `<4,2,16,fused>` family-grid kernel optimisation — working status

Round 2 (branch `perf/qvq-grid-kernel-opt-r2`, base @ 351f5dba) is documented at the
end of this file; the sections below it are the round-1 record (including the rejected
segment-grid ILP trial), kept verbatim.

The pipeline-level follow-up plan is in
[`qvq_quantization_4x_roadmap.md`](qvq_quantization_4x_roadmap.md). It treats the merged
kernel work and the rejected ILP trial as the starting point rather than repeating them.

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
| v3: ping-pong frontier pairs (1 barrier/step instead of 2; 6 smem prefixes + 64 KB frontiers = 160 KB) | 4.22 | 1.38 | yes — **no gain** (barrier stalls were 7 % of stall cycles; +spills, +L2 codebook traffic cancel it); reverted |

v1 ncu: FMA pipe 67 % utilised, issue slots 76 % busy, 50 % occupancy (64 regs x 1024 thr), L2 27 %:
the kernel is now FP32-issue-bound, not memory-bound.

## Real-workload 2-layer nsys numbers

Captured with `scripts/profile_qvq_quantize_nsys.sh <run> --max-layers 2 ...` (PR #7 recipe, same datasets/config).
Baseline run from the untouched `perf/qvq-nsys-profile` worktree @ acd959ed (`artifacts/nsys/llama32_1b_layers2_base_*`):

| metric (2 layers, Llama-3.2-1B, W2 v2b2_p32 YAQA) | baseline |
|---|---:|
| `qvq_cuda.viterbi_v2_segment_family_grid_trusted` calls | 12,788 |
| range kernel time | 123.26 s |
| range kernel avg / call | **9.639 ms** |
| range share of GPU kernel time | 55.0 % |
| kernel launches inside the range | 127,880 (10 / call) |
| `qvq_v2_segment_grid_kernel<4,2,16,fused>` total (all call sites) | 135.15 s (60.3 %) |
| `qvq_quantize.main` wall | 252.4 s |
| total CUDA kernel time | 224.05 s |

After (this branch @ final commit, `artifacts/nsys/llama32_1b_layers2_fused_*`):

| metric | baseline | fused | change |
|---|---:|---:|---:|
| family-grid range kernel avg / call | 9.639 ms | **4.499 ms** | **2.14x** |
| family-grid range kernel time | 123.26 s | 57.53 s | |
| family-grid range share of GPU kernel time | 55.0 % | 37.4 % | |
| kernel launches inside the range | 127,880 (10 / call) | 53,024 (4.15 / call = 12,476 fused calls x 4 + 312 gated reference calls x 10) | |
| `viterbi_v2_segment_tail_trusted` avg / call | 3.361 ms | 2.590 ms | 1.30x |
| total CUDA kernel time | 224.05 s | 153.70 s | |
| `qvq_quantize.main` wall | 252.4 s | **181.6 s** | **-28.0 %** |
| per decoder layer | 111-117 s | 76.1 / 75.0 s | |

Other live ops unchanged (kernel time inside range): `viterbi_tail_trusted` 22.05 -> 21.96 s, `viterbi` 17.53 -> 17.48 s,
`yaqa_feedback` 12.64 -> 12.25 s, `yaqa_feedback_update` 5.64 -> 5.48 s.

## Dead ends

Three variants were implemented, verified bit-exact, measured and reverted (v2-v4); candidate pruning was evaluated
by simulation only; cross-family emission sharing was ruled out by inspection of the call sites.

| attempt | 3x128 ms/call | why it lost |
|---|---:|---|
| v2: 64-bit (cost, prefix) keys reduced with fmin(double) | 6.28 | DSETP.MIN is not one DMNMX on sm_80: +30 % instructions |
| v3: ping-pong frontiers, one barrier/step | 4.22 | barrier stalls were 7 % of stall cycles; spills + extra L2 traffic cancel it |
| v4: 8-byte packed code+norm records for the L2 half | 6.15 | doubles L2 bytes there; L1TEX scoreboard stalls return |
| exact candidate pruning (sorted predecessor G) | simulation only, not implemented | mean 9.2/16 candidates but warp-max 14.95/16: SIMT divergence |
| share emissions across the 3 families | not applicable, not implemented | only the sampled-selection call site shares sequences; the dominant block-LDLQ site does not |

## Rejected follow-up: eight-prefix ILP in the reference segment-grid recurrence

On 2026-08-23, after PR #8 was merged into `main` at `351f5dba`, the remaining non-`Shift == 7`
recurrence immediately before `g_scratch[x] = best` was tested with eight-prefix chunks.  Each chunk issued all
predecessor loads and emissions before folding candidates through `lower_pair` in ascending `h` order; `__fadd_rn`,
the fused-boundary branch, backpointers, and barriers were unchanged.  A scalar compile-time remainder handled prefix
counts not divisible by eight.  A second form kept predecessor costs and emissions in separate register arrays and
performed `__fadd_rn` during the ordered fold.  Both forms compiled and passed the complete focused CUDA subset
bit-exactly (`170 passed`, command below), but neither improved the target operation, so the kernel change was reverted.

Environment: physical GPU 0, PCI `00000000:25:00.0`, UUID
`GPU-cb9e7784-cf50-203d-4f0d-5c622a89b1f2`, NVIDIA PG506-230, `sm_80`, 124 SMs, 96 GiB;
Python 3.14.7 free-threaded, PyTorch 2.13.0+cu130, CUDA 13.0, `TORCH_CUDA_ARCH_LIST=8.0`.
The idle gate observed 0 MiB and 0% utilization for three consecutive samples.  CUDA-event timings use 20 warmups
and 100 samples with identical seeded FP16 PGC16 codebooks, constrained overlaps, and step weights.

| W2 / transition-W4 batch | `main` grid-op median ms | chunked candidate ms | separate-array candidate ms |
|---:|---:|---:|---:|
| 8 | 1.222656 | 1.227776 | 1.228800 |
| 16 | 1.241088 | 1.245184 | 1.247232 |
| 32 | 1.273856 | 1.274880 | 1.275904 |
| 64 | 2.309120 | 2.322432 | 2.317824 |
| 128 | 3.698688 | 3.694592 | 3.694592 |

The repository tail-biting Viterbi benchmark likewise regressed slightly at every covered batch: baseline/candidate
medians were 5.773/5.782, 5.765/5.788, 5.757/5.800, 5.795/5.852, and 9.800/9.849 ms for batches
8/16/32/64/128.  Paths were exact and loss deltas were `0.000e+00` throughout.  Because the predeclared gate required
the target kernel to improve before the real-model run, the Llama-3.2-1B quantization profile was not run and no
performance PR was opened.

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-cb9e7784-cf50-203d-4f0d-5c622a89b1f2 \
  PYTHONPATH=. PYTHON_GIL=0 TORCH_CUDA_ARCH_LIST=8.0 MAX_JOBS=8 NINJAFLAGS=-j8 \
  CMAKE_BUILD_PARALLEL_LEVEL=8 NVCC_THREADS=2 GPTQMODEL_QVQ_NVCC_THREADS=2 \
  pytest -q tests/test_qvq_cuda.py -k 'v2_segment or grid'

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-cb9e7784-cf50-203d-4f0d-5c622a89b1f2 \
  PYTHONPATH=. PYTHON_GIL=0 TORCH_CUDA_ARCH_LIST=8.0 GPTQMODEL_QVQ_NVCC_THREADS=2 \
  python scripts/benchmark_qvq_viterbi.py --physical-gpu 0 --bits 2 \
  --batch-sizes 8 16 32 64 128 --warmup 20 --iterations 100
```

Why 4x is out of reach for this formulation: ~10.7 thread-instructions per state-step (FP32 op order pinned by
bit-exactness), issue-bound at 76 % with 50 % occupancy (64 regs x 1024 threads; 160 KB smem per CTA blocks a second CTA).

## Phase 1 pipeline audit: no exact call reuse

The follow-up pipeline audit instrumented all three family-grid call sites. It
found no input-identical repeated solve: Block-LDLQ updates errors between
blocks, YAQA updates feedback between disjoint anti-diagonals, and sampled
selection runs once. A result cache would therefore have a 0% hit rate.

Only state 63 from each provisional 128-state path is consumed. A fused
midpoint-output specialization proved that this output sparsity does not imply
half the recurrence work: state 63 must be recovered from the globally optimal
terminal traceback. The exact storage-only specialization improved a paired
3x128 microbenchmark by just 1.008x (4.915 -> 4.875 ms) and was reverted. The
retained telemetry records this workload shape without changing results; see
[`qvq_quantization_4x_roadmap.md`](qvq_quantization_4x_roadmap.md).

## Environment gotcha

nvcc segfaults (code 139, even on untouched `qvq_hadamard_cuda.cu`) when the JIT build runs under
`nsys profile --trace=osrt`; and the harness build hash includes `--threads N`, so warm the build with the same
`GPTQMODEL_QVQ_NVCC_THREADS` value in an un-profiled run before capturing. A segfaulted build leaves a stale `lock`
in the extension dir on which the next run sleeps forever; delete the dir.

## Runtime flags

`QVQ_DISABLE_FUSED_FAMILY_GRID=1` forces the reference path; the variable is read once on first use and cached for
the process lifetime, so it cannot be toggled mid-process. `torch.ops.gptqmodel_qvq.fused_family_grid_dispatch_count()`
returns the number of fused dispatches so far (used by the tests to assert which path ran).

---

# Round 2 — relaxed rounding, decision-equivalent (branch `perf/qvq-grid-kernel-opt-r2`)

Base `agent/qvq-dual-v4` @ 351f5dba (contains the merged PR #7 profiler and PR #8 fused kernel).
The round-1 bit-exactness requirement was replaced by **decision equivalence**: floating-point op
order inside the fused kernel is free, discrete outputs must match the reference path except for
quantified, provably score-equal near-tie flips, and the real-workload per-module quantization error
must be equal within noise.

## What shipped

1. **Squared-difference emission** — `emission = (t0-c0)^2 + (t1-c1)^2` (2 FADD + FMUL + FFMA after
   the two fp16→fp32 conversions) replaces the reference `max(tn + cn − 2·dot, 0)` form
   (norm 3 ops + dot 2 + expand 2 + clip 1). The sum-of-squares form is `>= +0` by construction, so
   the clip is free and the per-step `target_norm` disappears.
2. **Weight folded into one FMA** — `candidate = fma(emission, weight, predecessor)` replaces
   `clipped*weight` + separate add.

Micro-benchmark (`scripts/benchmark_qvq_family_grid.py`, committed; 3 families x 128 sequences, W2):

| variant | 3x128 weighted ms/call | 3x128 unweighted ms/call |
|---|---:|---:|
| round-1 fused kernel (351f5dba) | 4.336 | 4.167 |
| round-2 emission + weight fold (**shipped**) | **3.479** | **3.452** |

## Verification (relaxed contract)

* **Decision equivalence**: across the round-1 comparison grid (36 configs x 3 families, 1.35M
  states) the shipped kernel flips **11 states out of 1,345,536** (8.2e-6), all in one
  scale-4.0 unweighted config, with per-sequence squared error matching the reference within
  2.1e-7 relative — genuine near-ties.  Max per-sequence squared-error deviation over the grid:
  1.2e-6 relative.
* **Stress test** (`test_qvq_cuda_fused_w2_family_grid_randomized_stress_decision_equivalence`,
  committed): exactly **11,154 fused-path sequences (1,427,712 states)** — asserted counts —
  spanning the batch gate boundary, weighted/unweighted, constrained/unconstrained, XOR-related,
  arbitrary and duplicated-code bank pairs, plus codebook-snapped adversarial targets.  Both
  kernels' returned discrete paths are independently **rescored under one common fp64
  objective**: non-flipped sequences must rescore bitwise-identically, flipped sequences must
  rescore within the derived FP32 accumulation bound (2*gamma_138 ~= 1.6e-5 rel + 1e-4 abs for
  the reference form's near-zero-distance cancellation), and each kernel's reported loss must
  match its own path's rescore (traceback/reporting consistency).  Result at the committed
  seed: **0 flipped sequences / 0 flipped states**, reported per case family.  Tolerances are
  derived in a comment above the test (FP32 forward-error bound), not chosen ad hoc.
* **Reference-path cases stay bit-exact** (small batches, disabled-flag subprocess test).
* **Quality equivalence**: see the per-module loss table below.

## Real-workload 2-layer nsys numbers (round 2)

Baseline is the committed round-1 after-capture (`artifacts/nsys/llama32_1b_layers2_fused_*`,
same box, same env, captured 2026-08-23):

| metric (2 layers, Llama-3.2-1B, W2 v2b2_p32 YAQA) | round-1 fused | round-2 | change |
|---|---:|---:|---:|
| family-grid range kernel avg / call | 4.499 ms | **3.727 ms** | **1.21x** (2.59x vs the pre-round-1 9.639 ms) |
| family-grid range kernel time | 57.53 s | 47.67 s | |
| family-grid range share of GPU kernel time | 37.4 % | 33.4 % | |
| kernel launches inside the range | 53,024 (4.15 / call) | 53,024 (4.15 / call, unchanged) | |
| `viterbi_v2_segment_tail_trusted` avg / call | 2.590 ms | 2.394 ms | 1.08x |
| total CUDA kernel time | 153.70 s | 142.87 s | |
| `qvq_quantize.main` wall | 181.6 s | **171.7 s** | **-5.5 %** |

## Per-module quantization loss (2-layer real run, old vs new kernel)

| module | round-1 loss | round-2 loss | rel change |
|---|---:|---:|---:|
| layers.0.mlp.down_proj | 2.0022759438 | 1.9972181320 | -2.53e-03 |
| layers.0.mlp.gate_proj | 2.1463098526 | 2.1462106705 | -4.62e-05 |
| layers.0.mlp.up_proj | 2.2820501328 | 2.2820501328 | +0.00e+00 |
| layers.0.self_attn.k_proj | 0.2115077376 | 0.2115077376 | +0.00e+00 |
| layers.0.self_attn.o_proj | 0.8297442198 | 0.8297442198 | +0.00e+00 |
| layers.0.self_attn.q_proj | 0.1270044148 | 0.1287282705 | +1.36e-02 |
| layers.0.self_attn.v_proj | 0.8172291517 | 0.8172291517 | +0.00e+00 |
| layers.1.mlp.down_proj | 5.6535043716 | 5.5530796051 | -1.78e-02 |
| layers.1.mlp.gate_proj | 2.0141808987 | 2.0125288963 | -8.20e-04 |
| layers.1.mlp.up_proj | 2.3970038891 | 2.3990876675 | +8.69e-04 |
| layers.1.self_attn.k_proj | 0.4231991768 | 0.4231991768 | +0.00e+00 |
| layers.1.self_attn.o_proj | 0.6808187366 | 0.6808187366 | +0.00e+00 |
| layers.1.self_attn.q_proj | 0.4412775040 | 0.4412775040 | +0.00e+00 |
| layers.1.self_attn.v_proj | 0.4954676628 | 0.4954676628 | +0.00e+00 |

8 of 14 modules have bit-identical loss; the six that differ moved through discrete YAQA
family/code re-selections triggered by near-tie flips.  Aggregate loss over the 14 modules:
20.5216 -> 20.4181 (**-0.50 %, better**); worst single-module regression +1.4 %
(`layers.0.self_attn.q_proj`), best improvement -1.8 % (`layers.1.mlp.down_proj`).
Runs: `/root/qvq_prof/r2_quality_base.log` (round-1 worktree @ ab2d34f3, warm JIT,
prepare_and_quantize 169.9 s) vs `/root/qvq_prof/r2_quality_new.log` (this branch).

## Round-2 dead ends (implemented, measured, reverted — do not retry as-is)

| attempt | 3x128 weighted ms/call | why it lost |
|---|---:|---|
| packed `(cost<<4 \| prefix)` int keys, LOP3+IMNMX argmin | 3.385 | 16-ulp tie buckets **and** masked frontier costs: the truncated winner is stored back into `g`, drift cascades and near-tie flips explode from 8e-6 to 1.4e-2 of states at input scale 4; 3 % was not worth it |
| per-step emission tables `w*(t_c - level)^2` in smem (256 levels/component, 32x lane-replicated, packed u16 rank codebook halves the codebook smem) | 3.976 | conflict-free (replication works, 165K conflicts / 277M wavefronts) but 2 LDS + PRMT extraction + address math per emission costs more issue slots than the 2 conversions + 2 subtracts it removes; LSU 41 %, instructions grew 51.2G -> 60.6G per call |
| `fminf` value-select + predicated index-select (exact, aimed at ALU-pipe rebalance) | 5.03 | breaks ptxas' predication of the compare/select pair; ~45 % slower |

The dead-end timings and their ncu counters are development measurements taken on this box
during the round; the variants were reverted, so no machine-readable profiler artifacts are
committed for them (only the shipped kernel's, see below).

## Why round 2 plateaus at ~1.2x (ncu evidence, 3x128 shape)

Measured: 1.25x on the weighted micro-benchmark, 1.21x on the real NVTX range (the range also
contains the unchanged detect/traceback launches and inter-launch gaps).  The >=1.5x round-2
target was **not reached**; the evidence below is why.

The shipped kernel executes ~54.9G thread-instructions per call (committed machine-readable
counters: `artifacts/ncu/qvq_family_grid_r2_shipped_metrics.csv`): per (h, jj) candidate pair
(one shared emission + two bank candidates) that is 2 fp16->fp32 conversions (HADD2.F32),
2 FADD subtracts, FMUL+FFMA for the square, 2 candidate FFMA/FADD, ~6 compare/select ops and
~2 addressing/loop ops.  Issue slots are 75.7 % busy, FMA pipe 55.4 %, ALU 64.1 %,
`stall_math_pipe_throttle` 2.86 per issue — the kernel is issue/math-pipe-bound with occupancy
already fixed at 50 % (64 regs x 1024 threads, 160 KB smem, 1 CTA/SM; 2048 threads/SM would cap
registers at 32 and spill).  The remaining big-ticket items are the conversions+subtracts
(killable only by an 8-byte float2 codebook — the round-1 v4 L2-traffic dead end) and the
compare/select chain (killable only by tie-widening keys — the round-2 masked-frontier dead end).
Getting beyond ~1.3x needs a different algorithm or hardware (DPX/Hopper min-plus instructions).

## Runtime flags (unchanged)

Gates and fallbacks are identical to round 1: fused path at flattened batch >= 40, sm >= 8.0,
163 KB opt-in smem, `QVQ_DISABLE_FUSED_FAMILY_GRID=1` escape hatch (now covered by a subprocess
test), dispatch counter op unchanged.
