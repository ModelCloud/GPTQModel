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

## Environment gotcha

nvcc segfaults (code 139, even on untouched `qvq_hadamard_cuda.cu`) when the JIT build runs under
`nsys profile --trace=osrt`; and the harness build hash includes `--threads N`, so warm the build with the same
`GPTQMODEL_QVQ_NVCC_THREADS` value in an un-profiled run before capturing. A segfaulted build leaves a stale `lock`
in the extension dir on which the next run sleeps forever; delete the dir.

## Runtime flags

`QVQ_DISABLE_FUSED_FAMILY_GRID=1` forces the reference path; the variable is read once on first use and cached for
the process lifetime, so it cannot be toggled mid-process. `torch.ops.gptqmodel_qvq.fused_family_grid_dispatch_count()`
returns the number of fused dispatches so far (used by the tests to assert which path ran).
