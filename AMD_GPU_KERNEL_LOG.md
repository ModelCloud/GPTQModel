# AMD GPU Kernel Log

This log contains only AMD GPU kernel work. It records successful and failed
experiments for the QVQ V2B2-P32 inference kernel targeting AMD Instinct
MI355X (`gfx950`). Performance from a contaminated GPU is retained for
engineering comparison but is never presented as an official result.

## Target and correctness contract

- GPU: AMD Instinct MI355X VF, `gfx950:sramecc+:xnack-`, 256 compute units
- PCI bus: `0000:83:00.0`
- GPU unique ID: `0x333ef6e01ec019b3`
- Driver: `7.1.3.31500000`
- Software: Python 3.14.7, PyTorch 2.13.0+rocm10.0.0, HIP 7.15.26333,
  Triton 3.8.0
- Formats: canonical QVQ V2B2-P32 W2, W2.5, W3, and W3.5
- Requested rows: M=1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048,
  and 4096
- Accuracy gate: maximum absolute error no greater than `2e-3` against an
  FP32 dense oracle reconstructed from the same packed P32 payload

## 2026-09-04 experiments

| Result | Experiment | Accuracy | Exploratory performance and decision |
| --- | --- | --- | --- |
| FAIL (environment) | Strict three-sample idle preflight | Not run | GPU utilization was 0%, but PIDs 1463574, 1463764, and 1463785 retained 254936.4 MiB (86%) VRAM. The strict benchmark correctly failed closed. The processes are outside this namespace and were not killed. |
| PASS, REJECTED | Direct planar recurrence decode, 64-column output tile | Passed focused P32 checks | W3, K=N=4096: about 0.213 ms at M1 and 0.973 ms at M4096. Rejected because continuous-window decoding reduced decode overhead. Results were exploratory because the idle gate was blocked. |
| PASS, REJECTED | Direct planar recurrence decode, 128-column output tile | Passed focused P32 checks | W3, K=N=4096: about 0.384 ms at M1, 0.284 ms at M32, and 0.994 ms at M4096. Rejected as slower than the 64-column tile. Results were exploratory. |
| FAIL, FIXED | First continuous-window decoder used signed words during funnel shifts | Maximum absolute error observed up to about 2.87, above `2e-3` | Root cause was arithmetic right shift of signed `int32` packed words. Casting both loaded words to Triton `uint32` before shifts fixed the decode. No performance result from the incorrect kernel was accepted. |
| PASS | Continuous-window decode, 64-column output tile, four M regimes | 62/62 focused tests passed; 10 bitwise-identical repeats per case; small K=N=256 sweep worst maximum absolute error about `7.15e-6` | Selected as the initial kernel. Tile regimes are 16x64 for M<=16, 32x64 for M<=64, 64x64 for M<=256, and 128x64 for larger M. |
| PASS (exploratory) | Full requested M/rate sweep, K=N=4096, 5 warmups and 20 timed iterations | All 52 cases passed; worst maximum absolute error `1.2397766e-5` | The run is explicitly invalidated by foreign VRAM residency. P32 median ranges were W2 0.142-0.786 ms, W2.5 0.160-0.850 ms, W3 0.167-0.954 ms, and W3.5 0.167-0.950 ms. Cached dense FP16 GEMM was 0.013-0.107 ms and is recorded as an ideal uncompressed ceiling, not the production fallback. |
| PASS (exploratory) | Expanded K=N=4096 sweep with production fallback, 5/20 kernel and 1/5 fallback warmup/iterations | All 52 cases passed; worst maximum absolute error `1.2397766e-5` | P32 was 3.66x-24.11x faster than reconstruct-plus-GEMM and peaked at 165.2 effective TFLOP/s. Packed storage was 7.88x, 6.32x, 5.28x, and 4.53x smaller than dense FP16 for W2 through W3.5. This run remains explicitly invalidated by the same foreign residency. |
| FAIL (test harness), FIXED | First branch-coverage invocation used a dotted `--source` module | Tests did not start | Coverage imported the side-effectful package while resolving the source and Python 3.14 then rejected a second NumPy extension import before segfaulting. Using an absolute `--include` file filter avoided the duplicate import. |
| FAIL (test harness), FIXED | Public `QVQLinear.forward()` integration assertion compared FP16 output metadata to an FP32 oracle | Numeric values reached the assertion, but 3 integration cases failed on dtype equality | Public forward intentionally restores the activation dtype after scaling. The assertion now promotes the output to FP32 before applying the `2e-3` oracle gate. This was not a kernel numeric failure. |
| PASS | Expanded correctness and contract suite | 183/183 tests passed | Covers three independent seeds for every requested M/rate pair, 10 repeat calls per pair, adversarial signs and magnitudes, all alternate bank IDs, FP16/FP32 output, public module dispatch, a non-default stream, and invalid contracts. Python branch coverage is 100% after excluding the Triton JIT body that is compiled and validated by the GPU oracle cases. |
| FAIL (pre-existing test portability) | Existing `test_qvq_v2b2_p32.py`, `test_qvq_v2b2_p32_window.py`, and `test_qvq.py` regression selection on ROCm | 853 passed, 139 skipped, 125 failed | The CUDA-marked failures are not in the new inference path. They gate only on `torch.cuda.is_available()`, which is true on ROCm, then require NVIDIA-only QVQ/diagnostic CUDA extensions or NVIDIA telemetry. The extension reports `QVQ CUDA requires NVIDIA CUDA; ROCm is not supported`. Five representative non-matrix failures were rerun separately and confirmed the same ROCm/NVIDIA capability mismatch. |
| PASS | Move M256 from 64x64/4 warps to 128x64/8 warps | All four rates passed | Median latency improved by 22.2%-33.3% in the final full sweep, depending on rate. Accepted. |
| PASS | Move M128 from 64x64/4 warps to 128x64/8 warps | All four rates passed | Median latency improved by 17.6%-22.6% in the final full sweep. Accepted. |
| PASS, REJECTED | Move M64 from 32x64/4 warps to 64x64/4 warps | All four rates passed | Median latency regressed by about 4%-6%. The 32-row tile was restored. |
| PASS, REJECTED | Move M32 from 32x64/4 warps to 64x64/4 warps | All four rates passed | Median latency regressed by about 4%-6%. The 32-row tile was restored. |
| PASS | Increase M1-M16 16x64 tile from 4 to 8 warps | All four rates at M1 and M16 passed | Median latency improved by 3.9%-7.2% in the final full sweep. Accepted. |
| PASS | Increase M32-M64 32x64 tile from 4 to 8 warps | All four rates at M32 and M64 passed | Median latency improved by 12.5%-15.7% in the final full sweep. Accepted. |
| PASS (exploratory) | Full tuned requested M/rate sweep, K=N=4096 | All 52 cases passed; worst maximum absolute error `1.2397766e-5` | Every changed requested shape improved by 3.9%-33.3%. Reconstruct-plus-GEMM speedup was 3.78x-26.64x and peak effective throughput was 165.6 TFLOP/s. The same three foreign residents keep this result explicitly invalidated for official reporting. |
| PASS | Tuned correctness and contract suite | 185/185 tests passed | Three-seed requested matrix, repeatability, adversarial inputs, alternate banks, dtypes, public dispatch, stream behavior, and contract rejection all pass. Python branch coverage remains 100% with the GPU-compiled Triton body covered by oracle tests. |
| PASS | Synchronize with `origin/main` before the second tuning pass | 185/185 post-merge tests passed | Merged `origin/main` commit `4ef2089f` as merge commit `7667bd5e`; the fixed performance baseline remains AMD kernel commit `14928c96`, not the moving main branch. |
| RESEARCH | Hopper and ROCm/AITER kernel review | Not applicable | Hopper's useful pattern was decoding one weight tile for multiple activation rows. AITER's gfx950 kernels reinforced shape-specific tiles, K-pipelining, XCD swizzling, and avoiding cache-policy assumptions. Hardware-FP4 AITER timings are not comparable to exact QVQ PGC16 lookup decoding and were not used as a baseline. |
| PASS | Pairwise adjacent-column decode | Full 52-case square matrix passed | One trellis state and bank/hash recurrence now produces both adjacent level indices before `tl.interleave`. This is the main algebraic reduction and improved the exploratory square sweep by 1.11x-1.47x on its own. |
| PASS | Shape-specialized dimensions, flattened grid/XCD swizzle, `matrix_instr_nonkdim=16`, and stage split | All focused rate/M cases passed | Static M/N remove runtime shape branches. Three compiler stages are selected through M64 and two thereafter; the flattened launch uses an eight-XCD permutation when it is bijective. |
| PASS | Dedicated scalar GEMV for tiny M | All focused square and Qwen cases passed | A one-row FP32 vector reduction avoids mostly empty 16-row MFMA tiles. It is always selected at M1 and at M2/M4 only when `M*ceil(N/64)<=256`, preventing over-dispatch on wide Qwen projections. |
| PASS | Shape-aware large-M output tile | All four affected rates passed | The 128-column tile remains selected from M2048 except for N<=1024 at M2048, where the 64-column tile restored occupancy and reduced latency from 0.240-0.291 ms to 0.153-0.204 ms. M4096 retains the 128-column tile. |
| PASS, REJECTED | 128/256-column tiles outside the selected large-M regime | Accuracy passed | Wider tiles regressed small/mid M. BN256 also lost at M1024/M2048 and offered no material M4096 advantage. |
| PASS, REJECTED | `.cg` input/weight cache policy and four compiler stages | Accuracy passed | Both changes regressed focused small-M timings and were reverted. |
| FAIL, REJECTED | Load unique packed words once and recover lanes with `tl.gather` | Did not compile | Triton rejected the gather because its lowering was not warp-local. No timing was accepted. |
| PASS (exploratory) | Qwen3.8-27B projection sweep against fixed commit `14928c96` | All 364 candidate cases passed; worst maximum absolute error `6.198883e-5` | Across seven official projection shapes, four P32 rates, and all 13 requested M values, every candidate case was faster: 1.003x minimum, 1.514x geometric mean, and 2.351x maximum. M1 geometric mean was 2.021x. The 10x target was not reached. Both baseline and candidate sweeps are invalidated by foreign GPU residency. |
| PASS (exploratory) | Final K=N=4096 sweep against fixed commit `14928c96` | All 52 cases passed; worst maximum absolute error `1.239777e-5` | Every case improved: 1.022x minimum, 1.519x geometric mean, and 2.070x maximum. M1 geometric mean was 2.039x. |
| PASS (exploratory) | ROCm Compute Profiler stages `sol` and `cu_ins` plus gfx950 ISA disassembly | Exact HSACO launches completed with the correct dynamic LDS allocation | At W3/M64/K=N=4096, pairwise stage-3 MFMA reduced static VALU 234 to 161 (31.20%), VGPRs 68 to 40 (41.18%), dynamic VALU 14.28M to 6.10M (57.31%), and dynamic VMEM 2.37M to 1.57M (33.52%); profiled median improved from 192.35 us to 85.42 us. The M1 GEMV has 208 static opcodes, 26 VGPRs, and no MFMA; its direct median improved from 189.30 us to 83.14 us (2.277x). The large-M stage-2 kernel reached 95.88% CU utilization. |
| FAIL (profiler integration), FIXED | Direct `rocprof-compute` injection into the Triton Python process | Benchmark aborted before dispatch | ROCm and Triton's LLVM libraries registered `spirv-expand-step` twice. The corrected path loads the exact cached HSACO in a minimal HIP module launcher and preserves grid, block, LDS, and argument ABI. `rocprof-compute` 3.8.0, `rocprofv3`, and `llvm-objdump` were verified; the profiler analysis dependencies were installed in `/home/ubuntu/.venvs/rocprof-compute`. |
| PASS | Final correctness and contract suite | 204/204 tests passed | Covers three seeds for every requested M/rate pair, ten repeat calls, adversarial inputs, alternate banks, FP16/FP32 output, public module dispatch, non-default stream behavior, every selected Qwen launch regime, GEMV policy, ROCm process filtering, and invalid contracts. |
| PASS | W2 cross-word load mask | Full Qwen projection sweep passed | The high packed word is skipped for W2 states whose four-bit transition does not cross a 32-bit boundary. Accepted as a strict load reduction. |
| PASS | Fold bank selection through the xor-shift recurrence | Full Qwen projection sweep passed | Algebraically rewrote `(state ^ selected*mask) ^ ((state ^ selected*mask) >> 8)` as `(state ^ (state >> 8)) ^ selected*(mask ^ (mask >> 8))`, removing a selected-dependent xor from the hot recurrence. |
| PASS | Exact gfx950 instruction selection | Full Qwen projection sweep passed | Inline `v_mad_u32_u24` replaces multiply-plus-add in the 16-bit hash. `v_alignbit_b32` replaces the variable funnel-shift only for GEMV and M32-M512, where its complete A/B operand dependency reduced latency; it was rejected for larger MFMA tiles. |
| PASS, REJECTED | Precomputed 65536-entry decode/index LUT | Accuracy passed | The extra global lookup traffic outweighed removed integer algebra across the tested Qwen shapes, so exact on-the-fly recurrence remained selected. |
| PASS, REJECTED | Extend GEMV to all wide M2/M4 shapes | Accuracy passed | Increased program count and reduction traffic regressed wide QKV/MLP projections. The existing occupancy cap of 256 programs remains selected. |
| PASS | Wider K blocks and Qwen-specific M/N tiles | Full Qwen projection sweep passed | MFMA uses BK64 from M128, BK32 for M64 with N>=10240, and BK16 otherwise. Selected row tiles range from BM32 for narrow mid-M projections through BM1024 for M1024/N10240-12288; all retain BN64 and eight warps. |
| PASS (exploratory) | Complete Qwen3.8-27B target sweep versus fixed AMD commit `14928c96` | All 364 cases passed; worst maximum absolute error about `6.2e-5` | All seven projection shapes, four P32 rates, and 13 requested M values improved. Speedup was 1.397x minimum, **2.00045x geometric mean**, and 3.228x maximum. Per-M geometric means ranged from 1.711x at M128 to 2.791x at M4096. This reaches the requested 2x exploratory target, but foreign GPU residency invalidates it as an uncontended certification result. |
| PASS (exploratory) | Final ROCm Compute Profiler coverage for every selected compiler-stage regime | Exact cached HSACO launches completed for GEMV stage 1 and MFMA stages 1, 2, and 3 | W3 Qwen representatives were M1/K5120/N12288 (GEMV stage 1), M64/K5120/N12288 (MFMA stage 3), M2048/K5120/N12288 (stage 2), and M1024/K5120/N12288 (stage 1). Median dispatches were 103.69, 114.46, 490.14, and 306.84 us. Stage-2 reached 96.81% CU utilization; dynamic MFMA counts were 0, 0.492M, 15.729M, and 7.864M respectively. Static ISA confirms `v_alignbit_b32` only in the selected small/mid regimes and `v_mad_u32_u24` in every representative. |
| PASS | Remove the resident SGLang workload before certification | Not applicable | Graceful termination was requested for launcher PID 94512 and its scheduler/detokenizer children. The processes exited and target-GPU VRAM fell from 267.84 GB to the 781.8 MiB ROCm driver floor; no SGLang process restarted. |
| PASS | Correct ROCm idle-gate target filtering and lifecycle races | 2 focused parser tests passed | KFD process records are now filtered by physical GPU and positive resident VRAM, so a GPU-1 process and a stale zero-byte GPU-0 record cannot invalidate GPU 0. All-unknown queue teardown records are retried, inter-shape cooldown prevents allocation-reclamation overlap, and valid component results can be resumed after a fail-closed interruption. The 781.8 MiB measured driver floor is explicitly covered by a 1024 MiB allowance; utilization and target-GPU process gates remain strict. |
| PASS (certified) | Uncontended Qwen3.8-27B sweep versus fixed AMD commit `14928c96` | Candidate and baseline each passed 364/364 cases; worst maximum absolute error `6.198883e-5` | Both revisions were rerun with 3x 0%-utilization idle samples before every projection, zero resident target-GPU processes, and no `--allow-busy`. Every candidate case improved: **1.4076x minimum, 2.00315x geometric mean, and 3.2191x maximum**. Per-M geometric means range from 1.716x at M128 to 2.787x at M4096. This is the uncontended confirmation of the requested 2x target. |
| PASS, REJECTED | Contiguous K-by-N FP16 runtime cache | All 364 cases passed; worst maximum absolute error `6.198883e-5` | The uncontended exploratory Qwen sweep improved 1.762x minimum and 3.721x geometric mean over `a46fdfe2`, short of the requested additional 4x. The layout was rejected in favor of direct N-by-K predecode. |
| PASS | Direct N-by-K P32 predecode plus FP32-output GEMM | 209/209 AMD tests passed, including adversarial activation, alternate-bank, mutation-invalidation, fused opt-out, and non-default-stream cases | The first complete quick probe measured 4.634x geometric mean versus `a46fdfe2`; the first strict run measured 3.9008x and exposed repeated Python dispatch setup as the remaining small-M bottleneck. The AITER-inspired transposed operand lets ROCm select faster GEMM algorithms while retaining FP32 accumulation/output. The transient cache is version-checked; `cache_weight=False` retains the storage-neutral fused kernel. |
| PASS, REJECTED | Force hipBLAS or hipBLASLt globally for the cached GEMM | Accuracy unchanged | Isolated 91-case Qwen shape/M sweeps measured only 1.0021x and 1.0047x geometric-mean gains over the default selector, respectively, with shape-dependent regressions. A process-global experimental backend setting was not justified. |
| PASS (certified) | Mutation-aware fast dispatch for the preshuffled cache | 209/209 AMD tests passed; strict sweep passed 364/364 cases with worst maximum absolute error `6.198883e-5` | Reusing the validated K-by-N transpose view and bypassing invariant device/shape reconstruction removes the host launch gap while retaining version and operand identity checks. Against `a46fdfe2`, the strict uncontended result is **1.9419x minimum, 4.27402x geometric mean, and 11.5828x maximum**. Directly against `14928c96`, it is **3.9708x minimum, 8.56151x geometric mean, and 25.2155x maximum**. Every projection passed three 0%-utilization idle samples with no resident target-GPU process. |
| PASS (profiled) | Separate predecode and steady GEMM passes, W3/M64/K5120/N12288 | Exact predecode output passed the FP32 oracle suite | After compilation warmup, predecode plus first GEMM was 0.537645 ms and steady GEMM was 0.036456 ms, estimating 0.501189 ms for predecode. Against the fixed fused median of 0.142761 ms, the cache amortizes after about 4.71 calls. The exact gfx950 predecode HSACO has 156 static opcodes, 87 VALU, 11 VMEM, 6 LDS, 16 VGPRs, no scratch, two `v_alignbit_b32`, and two `v_mad_u32_u24`. |
| FAIL (profiler integration, bounded) | Direct `rocprofv3` trace of the new Python/Triton two-pass path | Process aborted before dispatch | The same duplicate LLVM `spirv-expand-step` registry conflict reproduced. The invocation was time-bounded and left no processes or GPU residency. Pass timings use synchronized ROCm events and static math uses direct gfx950 HSACO disassembly; prior counter passes use the minimal-HSACO workaround. |
| PASS, REJECTED AS PRIMARY PATH | Full 91-case cached-GEMM dispatch/layout/split sweep | Every supported candidate matched the current FP32-output GEMM within `2e-3` | Across seven Qwen3.8-27B K/N geometries and all 13 requested M values, the independent repeat found only 1.000x minimum, 1.03801x geometric mean, and 1.27511x maximum best-per-shape speedup. Physical K-by-N caches, forced default/hipBLAS/hipBLASLt/CK selection, and 2/4/8/16-way M/N batched splits cannot provide the requested 1.5x geometric gain. |
| PASS (profiled) | Nine-regime `rocprofv3` runtime/kernel trace plus exact selected-symbol gfx950 disassembly | No numerical path changed | Layout gains come from different hipBLASLt macro-tiles: M32/full-Q changes 16x32x1024 to 64x32x256, while M128/QKV changes 64x128x128 to 192x128x64. Every selected kernel has zero scratch. The steady pass contains no P32 extraction, bank, hash, or lookup algebra; static universal hipBLASLt symbols contain guarded edge/activation paths and are not treated as dynamic instruction counts. Direct counter injection again hit the duplicate LLVM `spirv-expand-step` failure before dispatch, so no new BLAS PMCs are claimed. |
| PASS (ceiling and numeric probe) | Fold immutable QVQ axes into the persistent dense cache | Synthetic target-dimension FP16-cache probe passed `2e-3`; worst maximum absolute error was `0.001476735` at M4096/K17408/N5120 | The architecture-aware 91-case stage sweep folds `diag(SU) * H_K * W_inner * H_N * diag(SV)`, omitting disabled axes. Removing online input/output recovery has a 1.9442x minimum, 14.8003x geometric-mean, and 43.9830x maximum ceiling; all 91 cases exceed 1.5x. This is selected as the next implementation path, subject to all-rate canonical full-layer oracle testing on real Qwen3.8 payloads and cold-cache/break-even measurement. |
| PASS, PARTIALLY SELECTED | Real-payload transform-folding exploration over the complete Qwen3.8-27B matrix | 314/364 cases passed the `2e-3` FP32-oracle gate | The unrestricted path measured 1.8257x minimum, 10.1204x geometric mean, and 17.5133x maximum, but `attn_out`, `mlp_gate_up`, and `mlp_down` contained seed- or shape-sensitive FP16-folding failures up to `0.0030313`. Those three geometries were rejected as a unit rather than accepting optimistic per-M or per-seed exceptions. |
| PASS (certified) | Fail-closed full-layer folded cache versus the exact prior public forward | All 364 cases passed: 208 selected cases stayed below `2e-3` (worst `0.001853943`), and all 156 fallback cases were bitwise identical to the prior path | Four measured geometries are enabled: full Q/gate, full KV, linear QKV, and linear Z. Enabled cases measured 5.1052x minimum, **14.2855x geometric mean**, and 19.4969x maximum. Including the three exact fallbacks, the complete 364-case target matrix measured **4.53369x geometric mean**; every requested M has a 3.1168x-5.1279x geometric mean. This exceeds the requested additional 50% at every M. |
| PASS | Folded-cache construction, amortization, and residency | The folded operand reuses one contiguous allocation and the temporary predecode cache is released; 235/235 AMD tests passed | At W3/M1, JIT-warm construction was 1.88-10.56 ms across the four enabled geometries and retained 10-126 MiB per projection, the same dense-FP16 cache class as the prior fast path rather than two copies. M1 amortization is about 2.4-13.3 calls. The process-first full-Q case also records the real 3.53 s one-time Triton JIT cost; subsequent same-process construction is 10.56 ms. |

The initial exploratory sweep is stored in
`artifacts/mi355x_p32/initial_gfx950.json`. The expanded sweep is stored in
`artifacts/mi355x_p32/fallback_gfx950.json`; it also records effective
throughput, packed/dense storage, and enrolls the benchmark's host-visible ROCm
context before rejecting newly arriving PIDs ahead of every timed case. Tuning
experiments and the final full sweep are stored alongside it as
`experiment_*.json` and `tuned_gfx950.json`.

The second-pass final square sweep, Qwen3.8-27B candidate and fixed-baseline
sweeps, pairwise experiment records, comparison summaries, and counter/ISA
breakdown are stored as `final_gfx950.json`, `qwen38_27b_2x_gfx950.json`,
`qwen38_27b_2x_gfx950_shapes/`,
`qwen38_27b_2x_vs_14928c96_gfx950.json`, `experiment_*.json`, and
`isa_profile_gfx950.json` in the same directory.

The uncontended candidate, fixed-commit baseline, and comparison are stored as
`qwen38_27b_2x_certified_gfx950.json`,
`qwen38_27b_baseline_14928c96_certified_gfx950.json`, and
`qwen38_27b_2x_certified_vs_14928c96_gfx950.json`, with their per-shape files
in matching `_shapes/` directories.

The runtime-cache pass timing and exact predecode ISA breakdown are stored in
`cache_profile_gfx950.json`.

The final fast-dispatch certification, its seven per-shape components, and the
comparisons against `a46fdfe2` and `14928c96` are stored as
`qwen38_27b_4x_fastpath_certified_gfx950.json`,
`qwen38_27b_4x_fastpath_certified_gfx950_shapes/`,
`qwen38_27b_4x_fastpath_certified_vs_a46fdfe2_gfx950.json`, and
`qwen38_27b_4x_fastpath_certified_vs_14928c96_gfx950.json`.

The 91-case dispatch sweeps, transform-folding ceiling, and compact
trace/ISA/SSA decision record are stored as
`qwen38_27b_dispatch_sweep_gfx950.json`,
`qwen38_27b_dispatch_sweep_repeat_gfx950.json`,
`qwen38_27b_fold_ceiling_gfx950.json`, and
`qwen38_27b_dispatch_profile_gfx950.json`. Raw `rocprofv3` databases and CSVs
remain local profiling artifacts and are not intended for source-control
commits.

The full-layer folded-cache certification and cold-construction breakdown are
stored as `qwen38_27b_folded_full_certified_gfx950.json` and
`qwen38_27b_folded_cold_final_gfx950.json`. The unrestricted exploratory result
is retained locally but is not a production acceptance artifact.

## Implementation notes

The default gfx950 path decodes the storage-neutral continuous-window P32
layout once into a transient N-by-K FP16 cache, then uses ROCm GEMM with FP32
accumulation and output. The cache is guarded by source identities and mutation
versions and remains outside serialized checkpoints. It expands runtime weight
storage to dense FP16; callers can set `cache_weight=False` to retain the
storage-neutral fused path. That fused path applies packed binary bank
selection and the exact PGC16 mix once per adjacent output pair. A scalar GEMV path
handles occupancy-safe tiny-M shapes; the MFMA path uses 16x64 tiles through
M16 and generally 32x64 through M64. Larger shapes select row tiles from 32 to
1024 with a 64-column tile, based on the Qwen projection dimensions and M; K
blocks are 16, 32, or 64. All selected configurations use eight warps. The
exact bank/hash path folds the bank mask through the xor-shift, uses gfx950
`v_mad_u32_u24`, and selectively uses `v_alignbit_b32`. Dispatch remains
limited to ROCm `gfx950`,
inference mode, FP16 input, V2B2-P32 vector size 2, and transition widths 4
through 7. Unsupported devices and formats retain the existing reference or
CUDA paths.

For the four certified Qwen3.8-27B geometries, the gfx950 module path also folds
both enabled Hadamard axes and SU/SV into that same mutation-aware dense cache.
Steady inference becomes one FP32-output GEMM plus the final model-dtype cast
and optional bias. Unsafe geometries stay on the prior path exactly. Rebuilding
the folded entry discards the ordinary predecode intermediate, and a later
folded cache hit also clears an ordinary dense cache created through a
pretransformed caller, preventing persistent duplicate dense copies.
