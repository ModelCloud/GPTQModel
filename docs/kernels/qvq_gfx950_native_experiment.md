# Experimental native HIP window path

Status: 2026-09-07, follow-up WIP on `codex/gfx950-native-followup-20260907`,
created directly from freshly fetched `origin/main` at `e2b2333274147daa5b2b9348cfdc809f65d4fdf8`
after PR152 merged. Experimental work was restored from a retained named backup
stash, without conflicts.
Measurements below predate this branch refresh unless explicitly noted;
not production dispatch or a completed
performance certification. ZML batching/evaluation remains paused by request.

## Contract and design

The HIP decoder consumes the unchanged continuous-window payload, packed binary
bank bytes, alternate-bank ID and FP16 LUT. It writes exact decoded FP16 `[N,K]`
scratch. rocBLAS consumes that scratch and FP16 `[M,K]` inputs with FP32 compute
and output. Scales and Hadamards remain outside this inner operation. This is
not requantization, a single-bank substitution, or an FP16-output GEMM.

Hopper/Ampere source inspired paired extraction at pair offsets 0/64. A second
prototype uses a 2D launch grid to eliminate runtime tile division/remainder.
It has four rate specializations (transition bits 4..7); ordinary W4 is pending.
The experimental entrypoints are not wired into the existing exported ABI.

The installed AITER revision `7440ef72503e1c3fadc5be85a5c74eb7c9c34841`
rejects FP32 output in its public FlyDSL HGEMM wrapper. The checked upstream main
was `24a62b1c122f23645a19b9d8b0abd4750c59359b`; its API has not been adopted.
rocBLAS is a tested initial library baseline, not a universal backend selection.

## Evidence and limits

Physical gfx950: BDF `0000:83:00.0`, unique ID `0x333ef6e01ec019b3`.
Three idle samples passed with no KFD workloads, zero utilization and 285.7 MiB
driver residency (explicit 320 MiB allowance). HIP compiler 7.15, C++20 `-O3`,
`--offload-arch=gfx950`; separate host link against `/opt/rocm/lib` was required
because the compiler's default unversioned HIP runtime library was absent.

Exact decoder tests: 48 combinations of K/N, transition width and bank ID;
three changed-LUT graph replays each. Separate rocBLAS tests passed M=1,8,128
with changed-input graph replay and FP64 reference. This is leaf-level graph
evidence, not combined-operation or ZML/PJRT graph certification.

Exploratory synthetic measurements: K=5120, N=1024, transition bits=5, bank=1,
50 synchronized event samples after 10 warmups; complete decode cost included.

| Variant | M | Baseline us | Decode + GEMM us | Ratio |
|---|---:|---:|---:|---:|
| Initial flat-grid HIP | 1 | 89.646 | 28.422 | 3.15x |
| Initial flat-grid HIP | 2048 | 168.453 | 55.024 | 3.06x |
| 2D-grid HIP | 1 | 89.485 | 27.881 | 3.21x |

Both candidates and baseline passed MAE <=0.003 and max <=0.006 against the
same full-K FP64 oracle with exact independently reconstructed FP16 weights.
The ~2% incremental grid-change result is not a robust established gain.

Additional exploratory 2D-grid cases (same gates, synthetic setup and timing
protocol) expose an important prefill/decode distinction:

| M | K | N | tb | Baseline us | Decode + GEMM us | Ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 2048 | 5120 | 17408 | 6 | 4307.737 | 486.056 | 8.86x |
| 1 | 5120 | 17408 | 6 | 204.667 | 187.866 | 1.09x |
| 2048 | 17408 | 5120 | 6 | 5686.567 | 472.114 | 12.04x |

All three passed exact decoded-weight comparison and both localized error
gates. The down-projection case has MAE 7.73e-6 and max error 1.52e-4 (rounded
up), not a model-quality score. Raw JSON names are
`native-grid2-m<M>-k<K>-n<N>-t6.json` under the same artifact root. The wide
M=1 case spends ~157 us decoding; it does not inherit the prefill gains.
Its cached GEMM alone was ~46 us, but that excludes decoding and is not the
complete-operation result. Shape-specific fused decode or a fully accounted
immutable cache remains necessary for stronger small-M acceleration.
This comparison uses the pinned production AOT artifact, source revision
`66565c27ed8a42639c0c2bbe55fdb4a8e677dca0`, not the older retained AMD commit
`58c3d72a`. No improvement against that older baseline or full model is asserted.
Scratch for this shape is 10 MiB decoded weights plus 32 MiB caller-owned BLAS
workspace. A persistent full-model decoded cache has not been deployed.

Artifacts under `/home/ubuntu/qvq-gfx950-runtime/`:

- `native-first-m{1,2048}-k5120-n1024-t5.json`
- `native-grid2-m1-k5120-n1024-t5.json`
- `native-profile-m1-csv-r2/trace_kernel_trace.csv` and kernel statistics
- Preserved initial `libqvq_gfx950_decode-experimental.so` and new
  `libqvq_gfx950_decode-grid2.so`; measurement JSON records library hashes.

The initial CSV trace confirms HIP decode and gfx950 BLAS execution; it includes
setup/reference work, so total-run shares are not production operator shares.
Default SQLite report generation failed with SQLite error 10 (disk I/O); CSV
worked. A 2D-grid PMC attempt (`SQ_INSTS_VALU SQ_INSTS_SALU`, decoder filter,
iterations 12..13) aborted with `aqlprofile API table load failed`.
That initial counter failure was resolved with a process-local symlink named
`libhsa-amd-aqlprofile64.so` targeting the Python ROCm SDK's versioned library,
plus its core/libraries paths in `LD_LIBRARY_PATH` and a writable
`ROCPROF_TMPDIR`. No system libraries were modified.

Matched successful captures are `native-flat-pmc-m1/` and
`native-grid2-pmc-m1-r2/`. Both profile decoder invocations 12 and 13 for the
same M=1,K=5120,N=1024,tb=5 case and report identical counters across those two
invocations:

| Counter per dispatch | Flat grid | 2D grid |
|---|---:|---:|
| SQ_INSTS_SALU | 348160 | 184320 |
| SQ_INSTS_VALU | 1986560 | 1638400 |

Runtime trace allocation metadata is unchanged at 20 VGPR, 4 Accum VGPR and
32 SGPR, despite the smaller compiler counts below. These are measured counter
reductions, not a proportional latency prediction. PMC instrumentation timings
are excluded from the uninstrumented speed table. Memory/occupancy/stall metrics
and the full shape sweep remain pending.

Static 2D-grid ISA has no reciprocal-based division sequence; all four kernels
use 18 VGPR / 22 SGPR versus initial 19 / 24, with zero LDS and private scratch.
These resource counts do not establish executed-instruction or latency gains.

## Remaining work

### External capture-stream experiment

The ZML identity lifecycle test reached actual PJRT command-buffer recording
and observed a capture stream different from its initialization stream. The
experimental additive `qvq_gfx950_native_execute_capture` entrypoint now
requires active capture on the caller's stream and the prepared device. It
temporarily binds the already-warmed rocBLAS handle and restores the original
stream; ordinary execute retains its strict stream check. The caller must
exclusively own recording and serialize eager/replay uses of shared storage.
No device math, precision, decoder binary or tuning selection changes here.

Built with host `g++ -std=c++17 -O2 -Wall -Wextra -Werror`, reusing the profiled
2D decoder object, as `libqvq_gfx950_native-capture-stream.so` under the runtime
artifact directory. Host descriptor tests and Ruff passed. Legacy `rocm-smi`
persistently reported 2% utilization despite no KFD PIDs and 299532288 bytes
driver VRAM on BDF `0000:83:00.0`. New AMD-SMI 27.0.0 and kernel sysfs both
reported 0%; three AMD-SMI samples confirmed 0%, 285 MiB and no processes.
That corroborated idle preflight permitted the correctness run.

`tests/test_qvq_gfx950_native_plan.py` passed both tests in 2.630 seconds with
`QVQ_GFX950_NATIVE_GPU_TEST=1`, the above library and the Python ROCm SDK
core/libraries `LD_LIBRARY_PATH`. M=8,K=32,N=48, rates 4/5 used same-stream
capture; rates 6/7 used private-runtime preparation followed by capture on a
distinct stream. Three changed-payload/input replays per rate passed exact
decoded-weight comparison and FP64-reference MAE <=0.003/max <=0.006. Ordinary
execution after each replay matched graph output exactly. Wrong-stream ordinary
calls and capture-only calls outside capture were rejected. No model-quality,
native PJRT integration, concurrency, full-shape or performance claim follows
from this bounded test.

The additive `qvq_gfx950_native_prepare_owned` now provides plan-owned scratch
and an explicitly budgeted BLAS workspace for external-runtime initialization.
It warms private inputs outside capture and frees owned storage at destruction;
callers still complete all uses and destroy graphs before destroying the plan.
Built as `libqvq_gfx950_native-owned.so` with the same host compiler flags and
unchanged decoder object. ZML's native PJRT smoke passed M=8,K=32,N=48,tb=6,
bank=1 with one preparation, one separate-stream capture callback and two total
execute callbacks across three changed activation inputs. Uniform LUT=0.25
gave exact analytic outputs. This verifies the bounded external-runtime path,
not production per-lane ownership, varied-payload PJRT correctness or speed.

### Required external tuning contract

User requirement: ZML controls tuning when consuming QVQ; standalone QVQ may
autotune itself when an explicit configuration is absent. This is a required
next-ABI feature. The experimental BLAS sub-API now implements versioned MKNE
metadata, matching-solution enumeration, explicit solution preparation and
configuration readback. Combined P32 preparation now exists as described below;
standalone autotuning and ZML integration are **not yet implemented**.
Do not change the layout of the published version-1 struct in place.

`qvq_gfx950_rocblas.h` declares this additive experimental sub-API. Tests found
682 matching solutions at MKNE=1,256,256,1 and 681 at 8,5120,1024,1 and
128,5120,1024,1. Explicit solution 66575 was prepared, read back unchanged and
passed FP64-reference accuracy plus three changed-input graph replays at each
geometry. This validates that selected solution on those fixtures, not the
other enumerated candidates. Enumeration uses the installed rocBLAS beta API;
solution IDs must be tied to the exact library build. Its changed dispatch
still needs a matched profile before performance certification.

The benchmark now accepts `--solution-index` and verifies resolved configuration
equality. At M=1,K=5120,N=1024,tb=5, solution 66575 also passed three changed-input
replays of the **combined HIP decode + rocBLAS graph**, compared independently
with FP64. The baseline graph passed the same checks. Capturing initially on the
default stream was rejected; the harness now owns a non-default stream for all
preparation, capture and execution. This is still not ZML/PJRT capture evidence.
Artifact: `native-config66575-graph-m1-k5120-n1024-t5-r2.json`.
This explicit solution measured 34.840 us eager / 33.980 us graph for the complete
operation, versus 89.442 / 92.022 us baseline. It is slower than earlier heuristic
runs, not an autotune winner; solution enumeration alone does not rank candidates.

- Describe logical M (activation rows), K (reduction), N (output columns), and
  E (experts/groups). Current dense support is E=1; reject E>1 until grouped
  layouts, routing and per-expert row counts are explicitly implemented.
- Provide versioned configuration/capability enumeration for the exact geometry,
  architecture, input/output dtype, P32 rate/bank/layout and operation contract.
  Each eligible candidate reports its backend, stable configuration ID, build
  identity, scratch/workspace bytes, alignment, constraints and capture support.
- Expose applicable knobs: M/N/K tile sizes, waves/threads, pipeline stages,
  split-K and reduction policy, decode launch geometry, library solution ID,
  workspace budget and decoded-weight cache policy. Inapplicable knobs must be
  marked unsupported, not silently ignored. Do not expose arbitrary launch
  changes for a binary whose compiled specialization does not support them.
- Let ZML benchmark complete operations and choose an explicit candidate during
  compilation/preparation. QVQ validates and freezes that choice; it must not
  silently autotune, swap backends or alter precision at execution time.
- A standalone preparation entrypoint may enumerate, validate and autotune when
  no configuration is supplied. Return the resolved configuration so tuning is
  reproducible and observable. It must use the same eligibility/accuracy rules
  as external tuning, rather than a second weaker candidate set.
- Include device/build identity, MKNE, layout/rate/bank, numerical contract and
  relevant resource/cache policy in tuning cache keys. Retain caller-owned
  workspace and prepared library handles through execution and graph lifetimes.
- Verify explicit-choice round trips, invalid/unsupported selections, no hidden
  tuning during execution, cache invalidation and changed-input graph replay
  through the actual ZML integration before claiming this contract complete.

Repeat correctness and uninstrumented timing after profiling. Run the complete requested M and
model-geometry sweep, actual-weight/activation checks and combined graph tests.
Implement validated native prepared ABI integration in QVQ, with ZML only owning
execution lifetimes/calling that ABI, then resume the held-out evaluation.
One synthetic inner-operation shape exceeds 10x; a complete-sweep 10x or
full-model speedup has not been reached or claimed.

## Prepared native operation

`qvq_gfx950_native.h` provides an additive combined-operation ABI: configuration,
scratch sizing, prepare, get-config, execute and destroy. It exposes MKNE,
transition bits, bank ID and BLAS solution. The implemented decoder uses 256
threads and decodes every call; unsupported thread counts, cache policies and
E>1 are rejected. Preparation warms and synchronizes its stream. Execution
does not allocate or tune and verifies device/stream. Caller-owned scratch and
BLAS workspace must outlive plans and graphs.

`tests/test_qvq_gfx950_native_plan.py` passes host descriptor checks and combined
GPU replay at M=8,K=32,N=48 for all four P32 rates. Three random payload, bank
and activation changes per rate preserve exact decoded weights and pass the
localized FP64-reference gates. Mutating the caller configuration after prepare
does not change the plan; wrong-stream execution is rejected. This is bounded
native-ABI evidence, not full-model or ZML/PJRT certification.

Loading the combined DSO before Torch initially aborted with duplicate LLVM
`spirv-expand-step` registration. A process-local `LD_LIBRARY_PATH` containing
the Python SDK's `_rocm_sdk_core/lib` then `_rocm_sdk_libraries/lib` resolved it.
No global library changes were made. GPU tests use `QVQ_GFX950_NATIVE_GPU_TEST=1`
and `QVQ_GFX950_NATIVE_LIBRARY` pointing to the consolidated prepared DSO.
ZML's hermetic runtime/loading combination still requires its own test.
