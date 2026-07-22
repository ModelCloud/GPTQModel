# Trilin 3-bit GPTQ/AWQ kernel design and investigation journal

Last updated: 2026-07-22 UTC

## Purpose and journal rules

This document is the design source of truth and the append-only investigation journal for the Trilin hybrid native
CUDA/Triton path specialized for 3-bit GPTQ/AWQ inference. The first supported quantization point is:

- `bits=3`
- `group_size=128`
- `desc_act=False`
- `sym=True`
- CUDA `sm_80` first, with runtime capability checks and a safe existing fallback
- FP16 and BF16 activations, subject to separate numerical and performance proof

Every implementation, test, benchmark, and profiler iteration must add an entry to the investigation log. Failed
ideas remain in the log. Measured results must include their command and artifact path; hypotheses must be labeled as
hypotheses and must not be presented as measurements.

## Current status

Implemented and validated on both requested CC 8.0 devices. For both FP16 and BF16, production dispatch uses a direct
continuous-3-bit native CUDA GEMV at flattened M=1 and native CUDA WMMA at M=2..16. FP16 uses exact-value expanded
Marlin above M=16; BF16 retains fused Triton above M=16 because it is faster than the native large-M diagnostic.
Architecture, alignment, build-failure, training, and other unsupported cases retain guarded fallbacks. GPTQ/AWQ
packing, backend gates, save/reload, clock-controlled benchmarks, sanitizer checks, and Nsight Systems/Compute proof
are complete. The repository revision at the start of the investigation was `644ac5cd` on branch
`gptq-3bit-kernel`, with a clean worktree before this file was added. The dated journal below retains the complete path
from that initial design state.

## V0 capability contract

| Property | V0 contract |
| --- | --- |
| Methods | GPTQ and AWQ |
| Bits | Exactly 3 |
| Group size | Exactly 128 |
| Activation order | `desc_act=False`; natural K-to-group mapping only |
| Symmetry | Exactly `sym=True`; centered integer values are expected to use zero 4 |
| Pack word | `torch.int32` only |
| Activations | Contiguous-last-dimension FP16 or BF16 CUDA tensors |
| Accumulation | FP32 |
| Output | Same shape prefix and dtype as the activation, final dimension N |
| K constraints | Positive and divisible by 128; V0 does not silently reinterpret partial quantization groups |
| N constraints | Positive and divisible by 32 for the packed 96-bit blocks; masked Triton tile tails remain required |
| M constraints | Any positive flattened row count; separate decode and prefill launch regimes |
| Device | Runtime-probed CUDA device; optimized first for compute capability 8.0 |
| Training | Inference specialization only; existing training/dequantized path remains the fallback |
| Adapters/bias | Applied through the existing QuantLinear wrapper after the kernel unless profiling proves fusion useful |
| Unsupported cases | Must fail validation for explicit selection or fall back through existing backend selection |

The capability declarations alone cannot express conditional combinations such as “3-bit only when group size is
128.” The backend class therefore needs a per-request validation override in addition to its declarative fields.

## Packed layout contracts

### GPTQ/GPTQ_V2

The repository already packs every 32 K-axis 3-bit values into three consecutive 32-bit words. The bitstream is
continuous across the three words (96 bits total), producing the existing `10-1-10-1-10` boundary pattern:

```text
qweight: [(K / 32) * 3, N]
scales:  [K / 128, N]
qzeros:  [K / 128, (N / 32) * 3]
g_idx:   [K], with g_idx[k] == k // 128 for desc_act=False
```

For scalar index `i` in a 32-value block, `bit = 3 * i`, `word = bit // 32`, and `shift = bit % 32`.
Indices 10 and 21 straddle word boundaries and require bits from the following word. The real GPTQ packer and Torch
dequantizer are the authority for correctness.

The current Triton backend requires GPTQ V2. Source inspection confirms that the GPTQ quantizer sets a symmetric zero
to `(maxq + 1) / 2`, which is 4 for 3-bit `maxq=7`. The optimized symmetric path may therefore use constant zero 4
and avoid `qzeros` traffic. An executable pack/dequant test must still guard this source-backed contract at the backend
boundary, including GPTQ V1-to-V2 conversion.

### AWQ GEMM

The existing AWQ GEMM format and Triton kernel support only 4-bit interleaved packing. There is not yet a repository
contract for 3-bit AWQ GEMM serialization. V0 proposes a continuous 96-bit stream over each 32-value N-axis block:

```text
qweight: [K, (N / 32) * 3]
scales:  [K / 128, N]
qzeros:  [K / 128, (N / 32) * 3]
```

This proposed AWQ layout must not be exposed as compatible with external AWQ checkpoints until pack/dequant/save/load
tests establish the contract. The shared compute kernel will use a compile-time layout selector, while GPTQ and AWQ
wrappers retain their distinct storage semantics. The repository AWQ configuration accepts 3 bits and its symmetric
quantizer explicitly shifts signed values by `2 ** (bits - 1)`, producing packed zero 4. Existing AWQ pack/reorder
utilities still assume the fixed 4-bit eight-value order, so the new 3-bit pack path must not call them.

## Kernel design

### Dataflow

1. Flatten activation rows to `A[M, K]` without changing the last-dimension meaning.
2. Map one Triton program to an output tile `C[BLOCK_M, BLOCK_N]`.
3. Iterate over K in aligned chunks contained within the 128-element quantization group.
4. Load packed int32 words coalesced along N, decode the 3-bit values in registers, and center them at zero 4.
5. Convert the decoded tile to the activation dtype and issue `tl.dot` with FP32 accumulation.
6. Load one scale vector per 128-element group and reuse it across that group’s K work.
7. Store the masked M/N tail in the activation dtype on the current CUDA stream.

This follows the useful Marlin principles—fused unpack/dequantize and matrix multiply, group-scale reuse, tiled
coalesced weight access, shape-sensitive launch geometry, and no dense weight materialization—without assuming that
Marlin itself supports the target 3-bit format.

### Layout-specialized address calculation

- GPTQ: the packed-word row is `(k // 32) * 3 + word`, and N is the contiguous column.
- AWQ: K is the contiguous row selector, and the packed-word column is `(n // 32) * 3 + word`.
- The selector is a `tl.constexpr` so the unused address path is removed at compile time.
- Boundary indices 10 and 21 combine the low portion from one word with the high portion from the next word.

### Group-scale reuse options to test

Initial correctness implementation:

- Decode a K tile, subtract 4, multiply by `scales[k // 128, n]`, then call `tl.dot`.

Optimization hypothesis A:

- Accumulate the four 32-K partial dot products for one group into a group-local FP32 tile, then multiply that tile by
  one N-wide scale vector and add it to the output accumulator. This reduces scale loads and elementwise scale
  multiplies but may double accumulator register pressure.

Optimization hypothesis B:

- Use a 128-K decoded tile with broadcast scales and a single `tl.dot`. This maximizes group reuse but may reduce
  occupancy because of the larger live B tile.

Nsight Compute occupancy/register evidence and synchronized timing will decide between them.

### Launch regimes

The first candidate table is deliberately small; it will be tuned with CUDA-event measurements before profiling:

| Flattened M | Candidate `BLOCK_M` | Candidate `BLOCK_N` | K chunk | Rationale |
| ---: | ---: | ---: | ---: | --- |
| 1-4 | 16 | 32 | 32 or 64 | More N tiles to expose enough CTAs for decode |
| 5-16 | 16 | 64 | 32 or 64 | Balance Tensor Core work and output parallelism |
| 17-64 | 32 | 64 | 32 or 64 | Reuse weights across activation rows |
| 65+ | 64 | 64 or 128 | 32 or 64 | Prefill throughput and larger GEMM tiles |

`num_warps` and `num_stages` will be explicit launch choices. The kernel must not encode the observed 124-SM count;
launch decisions may consult live properties only in the Python wrapper.

### Decode alternatives

For M=1 or similarly small decode batches, a direct reduction/GEMV-style kernel may beat padded Tensor Core tiles.
That is a measured follow-up, not part of the minimum correct implementation. Split-K is also deferred until evidence
shows insufficient CTA parallelism; an atomic split-K path would require FP32 scratch initialization and a conversion
epilogue, so its launch and memory costs must be included.

### Wrapper and fallback

- Validate method, format, bits, group size, symmetry, activation ordering, pack dtype, tensor shapes/strides/dtypes,
  device equality, and runtime compute capability before launch.
- Enter a device guard derived from `qweight.device`; never launch against a fixed CUDA index.
- Use Triton’s current-stream launch behavior and add a non-default-stream test.
- Route only the exact V0 contract to the fused kernel.
- Preserve the existing dequantize-plus-matmul or Torch path for training and all unsupported configurations.
- Do not raise the automatic backend priority until matched benchmarks show a win and integration tests prove
  pack/save/reload behavior.

## Correctness proof plan

1. Generate deterministic quantized values through the real GPTQ packer; add an AWQ 3-bit packer reference only after
   its layout is explicitly accepted.
2. Decode to a dense FP32/FP16/BF16 weight with an independent Torch reference.
3. Compare `A @ W_dense` with the fused output, recording max absolute error, max relative error, and shape/dtype.
4. Include all quantized codes 0 through 7, negative/positive activations, long-K accumulation, and the straddling
   indices 10 and 21.
5. Exercise M/N tile tails, smallest legal K=128, representative K/N model dimensions, repeated calls, two devices,
   and a non-default CUDA stream.
6. Run through both raw launcher and QuantLinear/backend paths.
7. Test explicit rejection and automatic fallback for wrong bits, group size, `desc_act`, symmetry, pack dtype,
   architecture, and malformed buffers.
8. For GPTQ and AWQ integration, test quantize/pack, save, reload, and inference equivalence on a tiny module/model.

Initial numerical gates will be based on the matched dense computation with FP32 accumulation. Tolerances will be
stated only after observing the dense-vs-kernel distribution; they will not be widened merely to make a test pass.

## Benchmark and profiler plan

### Workloads

Use decode and prefill rows with representative transformer dimensions, plus tail tests. The first proposed matrix is:

```text
M: 1, 4, 16, 64, 256, 1024
K: 4096, 8192, 11008
N: 4096, 11008
dtype: fp16, bf16
```

The final matrix will include only shapes that fit both the exact packed contract and available memory. Timed scripts
belong under `scripts/`; correctness assertions belong under `tests/`.

### Baselines

- GPTQ 3-bit: existing Triton dense-dequantize plus `torch.matmul`, Torch dequantize plus `torch.matmul`, and dense
  FP16/BF16 `torch.matmul` as a numerical/performance reference.
- AWQ 3-bit: independent dense dequantized reference and any established repository fallback after its pack contract
  exists.
- Marlin: use its structure as design inspiration. If source inspection confirms that production Marlin is limited
  to 4/8 bits, label it ineligible for an apples-to-apples 3-bit timing comparison rather than comparing different
  formats as though they were equivalent.

Warm up JIT compilation and clocks, preallocate inputs/outputs, use CUDA events or `triton.testing.do_bench`, and report
median plus dispersion/tail statistics. Full ASCII result tables must include GPU identity, compute capability, dtype,
M/N/K, method/layout, latency, throughput, and speedup.

### Parallel GPU execution

GPU 0 and GPU 1 are runtime-probed independently. Matched correctness/benchmark processes will be launched concurrently
with `CUDA_VISIBLE_DEVICES=0` and `CUDA_VISIBLE_DEVICES=1`; each process will see its assigned device as local `cuda:0`
and will log the physical UUID/PCI bus ID before testing. Parallel execution is for throughput of the investigation,
not for combining results across potentially different devices.

### Nsight sequence

1. Use Nsight Systems after correctness to capture a bounded, warmed steady-state range and identify kernel share,
   launch gaps, and unexpected dequantization/materialization.
2. Extract authoritative tables with `nsys stats`; do not infer values by reading the binary report.
3. Use Nsight Compute only on the identified fused kernel, starting with `SpeedOfLight`, then add `LaunchStats` and
   `Occupancy` for latency-bound behavior or the relevant compute/memory section based on the measured classification.
4. Re-run the synchronized benchmark and correctness suite after each optimization, then capture a matched follow-up
   profile only when attribution changes.

Artifacts will go under an explicit local directory such as `artifacts/triton_3bit/<timestamp>/`. Commands, raw
summary output, and artifact paths will be recorded below.

## Investigation log

### 2026-07-22 — Initial repository and toolchain survey

Status: successful, except for the explicitly logged query failure below.

Commands/evidence:

```text
git status --short
git branch --show-current
git rev-parse --short HEAD

branch: gptq-3bit-kernel
revision: 644ac5cd
initial worktree: clean
```

```text
Python: 3.14.5 free-threading build
PyTorch: 2.13.0+cu130
PyTorch CUDA runtime: 13.0
Triton: 3.7.1
CUDA available: true
Visible CUDA devices: 8
```

```text
Nsight Systems: 2024.6.2.225-246235244400v0
Nsight Compute: 2025.3.1.0 (build 36398880)
```

Runtime PyTorch probe for the requested devices:

```text
device 0: NVIDIA PG506-230, compute capability 8.0, 124 SMs, 102191202304 bytes, memory clock 1593000 kHz,
          memory bus width 6144 bits
device 1: NVIDIA PG506-232, compute capability 8.0, 124 SMs, 102191202304 bytes, memory clock 1593000 kHz,
          memory bus width 6144 bits
```

Interpretation: both requested devices are live `sm_80` datacenter GPUs for this run. This is a host snapshot, not a
stable index-to-architecture mapping. Ampere-specific tuning is allowed only behind runtime gating.

### 2026-07-22 — Failed `nvidia-smi` inventory query

Status: failed harmlessly; no repository or GPU state changed.

Command:

```text
nvidia-smi --query-gpu=index,pci.bus_id,name,uuid,compute_cap,multiprocessor_count,memory.total,driver_version \
  --format=csv,noheader
```

Raw error:

```text
Field "multiprocessor_count" is not a valid field to query.
```

Cause: this installed `nvidia-smi` does not expose `multiprocessor_count` as a query field. SM count was obtained from
`torch.cuda.get_device_properties` instead. Next action: rerun `nvidia-smi` with only supported identity, memory,
compute-capability, and driver fields, and capture topology separately.

### 2026-07-22 — Existing packed-kernel investigation

Status: successful source inspection; no code change.

Findings:

- `gptqmodel/nn_modules/qlinear/tritonv2.py` advertises GPTQ 2/4/8-bit support and currently performs a Triton dense
  dequantization followed by `torch.matmul`.
- `gptqmodel/nn_modules/triton_utils/dequant.py` already contains an unadvertised GPTQ 3-bit decoder for the continuous
  96-bit `10-1-10-1-10` layout. It materializes the full dense weight and therefore is a correctness/layout reference,
  not the requested fused inference kernel.
- `PackableQuantLinear` has real CPU/GPU 3-bit packing and Torch dequantization logic, including both word-straddling
  values. This is the primary GPTQ layout oracle.
- The current AWQ GEMM/Triton classes and packers advertise and implement only 4-bit interleaved storage. AWQ 3-bit
  needs an explicit layout/serialization contract before external checkpoint support can be claimed.
- The target kernel should share a compute strategy while specializing GPTQ and AWQ packed addresses at compile time;
  it must not silently reinterpret one method’s tensor layout as the other’s.

### 2026-07-22 — Corrected GPU identity and topology capture

Status: successful.

Commands:

```text
nvidia-smi --query-gpu=index,pci.bus_id,name,uuid,compute_cap,memory.total,driver_version --format=csv,noheader
nvidia-smi topo -m
nvcc --version
```

Requested-device identity:

```text
GPU 0: PCI 00000000:25:00.0, NVIDIA PG506-230,
       UUID GPU-cb9e7784-cf50-203d-4f0d-5c622a89b1f2, CC 8.0, 98304 MiB, driver 610.43.02
GPU 1: PCI 00000000:2B:00.0, NVIDIA PG506-232,
       UUID GPU-20f7fde4-d88c-d6ca-e324-bd4e5e9e0855, CC 8.0, 98304 MiB, driver 610.43.02
GPU 0 <-> GPU 1 topology: NV12
NVCC: CUDA 13.0, V13.0.88
```

Interpretation: the requested devices are matched at compute capability, SM count, memory size, clock report, and bus
width, but retain distinct model strings and identities. Each parallel worker must log its assigned UUID so local
`cuda:0` results cannot be confused with the physical device index.

### 2026-07-22 — Marlin applicability check

Status: successful; 3-bit Marlin comparison rejected as invalid.

Source evidence:

```text
gptqmodel/nn_modules/qlinear/marlin.py:       SUPPORTS_BITS = [4, 8]
gptqmodel/nn_modules/qlinear/marlin_awq.py:   SUPPORTS_BITS = [4, 8]
GPTQ Marlin scalar map: (4, symmetric) and (8, symmetric)
AWQ Marlin scalar map: 4-bit and 8-bit
gptqmodel/utils/marlin.py zero-point permutations: explicit 4/8-bit branches, other widths raise
```

Decision: do not claim or manufacture a same-format Marlin 3-bit baseline. Reuse its design principles—fused
dequantization/MMA, packed-weight coalescing, scale reuse, and separate launch behavior—but compare measured 3-bit
performance against the existing GPTQ Triton/Torch dense-dequant paths and matched dense matmul. Any 4-bit Marlin
number is a separately labeled context result, never a 3-bit speedup denominator.

### 2026-07-22 — Symmetric zero and AWQ configuration proof

Status: successful source proof; executable regression tests still required.

Evidence:

- `Quantizer.find_params()` uses `(maxq + 1) / 2` for symmetric affine GPTQ. With 3 bits, `maxq=7` and zero is 4.
- `AWQProcessor.pseudo_quantize_tensor()` quantizes symmetrically in signed `[-4, 3]`, then sets packed zeros to
  `2 ** (bits - 1)`, also 4 for 3 bits.
- `BaseQuantizeConfig` accepts bit widths `[2, 3, 4, 5, 6, 8]`; `AWQConfig` does not narrow that list and defaults
  `desc_act=False`, so the requested AWQ configuration is representable.
- Existing AWQ `AWQ_ORDER`/`AWQ_REVERSE_ORDER` and Triton kernels are hard-coded for eight 4-bit values per word.
  They are not valid references for a 3-bit layout.

Decision: specialize V0 on constant zero 4 and omit `qzeros` loads inside the fused kernel, while retaining shape and
metadata validation. Implement an independent continuous-96-bit AWQ pack/dequant reference and prove round trips
before enabling its backend path.

### Next journal entry

Turn the confirmed GPTQ/AWQ packing and zero conventions into executable CPU layout-reference tests. Determine the
smallest backend integration that preserves existing 2/4/8-bit behavior, then implement only the minimum correct fused
launcher before starting tile experiments.

### 2026-07-22 — Test-first contract, expected red phase

Status: expected failure; successful test-first boundary definition.

Change:

- Added `tests/kernels/test_triton_3bit.py` with independent dense references and calls to the planned
  `gptqmodel.nn_modules.triton_utils.three_bit` API.
- Coverage includes continuous 96-bit round trips on both axes, byte-for-byte parity with the real GPTQ packer,
  conditional backend validation, raw GPTQ/AWQ layout parity for FP16/BF16 and M/N tails, current-stream behavior,
  and both QuantLinear wrappers.

Command:

```text
pytest -q tests/kernels/test_triton_3bit.py
```

Result:

```text
collected 0 items / 1 error
ModuleNotFoundError: No module named 'gptqmodel.nn_modules.triton_utils.three_bit'
exit code: 2
```

Interpretation: this is the intentional red phase. The test suite now fixes the shared API and expected behavior before
the implementation exists. Next action: implement the pure pack/unpack reference and minimum fused kernel, then use
the resulting assertion/compiler failures to refine the implementation without weakening the contract.

### 2026-07-22 — Pure layout reference and capability gate

Status: successful.

Implementation:

- Added `gptqmodel/nn_modules/triton_utils/three_bit.py` with continuous 96-bit `pack_3bit`/`unpack_3bit`, a portable
  dense dequant fallback, strict input validation, and the first untuned fused Triton matmul.
- Added conditional 3-bit validation to the existing GPTQ Triton and AWQ GEMM Triton backends.
- Added inference routing to the fused path on runtime-probed compute capability >= 8.0, retaining dense fallbacks.
- Added an AWQ 3-bit continuous-layout packer and exact registered-buffer shapes without changing the existing 4-bit
  storage path.

Command:

```text
pytest -q tests/kernels/test_triton_3bit.py -k 'pack or capability'
```

Result:

```text
5 passed, 15 deselected in 6.56s
```

Proven facts:

- Packing/unpacking is exact on both the K-packed GPTQ axis and N-packed AWQ axis.
- The new generic GPTQ qweight and qzero packing is byte-for-byte identical to `TorchLinear.pack_original` for the
  deterministic all-code pattern.
- Both backend classes accept the V0 tuple and reject wrong group size, activation order, symmetry, pack dtype, K
  divisibility, and N divisibility.

Next action: JIT-compile and run raw fused-kernel parity on physical GPU 0 before exercising wrapper integration.

### 2026-07-22 — First fused-kernel JIT attempt

Status: failed at compilation; no kernel launch or numerical result.

Command:

```text
CUDA_VISIBLE_DEVICES=0 pytest -q tests/kernels/test_triton_3bit.py -k 'raw_3bit'
```

Result:

```text
13 failed, 7 deselected in 10.16s
Triton CompilationError at: if LAYOUT == _LAYOUT_GPTQ_ID
NameError: Cannot access global variable _LAYOUT_GPTQ_ID from within @jit'ed function.
```

Cause: Triton 3.7.1 does not accept ordinary Python module globals inside `@triton.jit`, even when their values are
integers and the kernel argument being compared is `tl.constexpr`. The first failing global prevented compilation of
both layouts and every shape/dtype; no device correctness conclusion is possible from this run.

Rejected tactic: relying on Python module constants directly inside Triton JIT source or enabling the unsupported
`TRITON_ALLOW_NON_CONSTEXPR_GLOBALS` compatibility environment variable.

Next action: keep named constants in Python-side validation/packing, but use explicit compile-time numeric literals in
the JIT body and rerun one FP16 GPTQ case before expanding the matrix.

### 2026-07-22 — Second JIT attempt and test-device assertion defect

Status: kernel compiled and launched; test failed before numerical comparison because of a harness defect.

Command:

```text
CUDA_VISIBLE_DEVICES=0 pytest -q \
  'tests/kernels/test_triton_3bit.py::test_raw_3bit_matmul_matches_dense_reference[1-128-96-dtype0-gptq]'
```

Result:

```text
1 failed in 7.54s
AssertionError: device(type='cuda', index=0) == device(type='cuda')
```

Interpretation: replacing JIT-side Python globals with literals succeeded. Triton compiled for `arch=80`, launched,
and produced a finite FP16 `[1, 96]` tensor on `cuda:0`. The test created `torch.device('cuda')`, whose unresolved
index does not compare equal to the output’s resolved `torch.device('cuda:0')`. The numerical assertion was not
reached, so this is not correctness evidence.

Failed tactic: comparing a resolved output device object directly to an unresolved `torch.device('cuda')` value.
Next action: compare `actual.device` to the input tensor’s resolved device and rerun the identical case.

### 2026-07-22 — First numerical result

Status: failed correctness; kernel remains unfit for benchmarking.

Same-case result after fixing the device assertion:

```text
shape/dtype/device assertions: passed
mismatched elements: 72 / 96 (75.0%)
greatest absolute difference: 6.90234375 at output (0, 31)
greatest relative difference: 0.64501953125 at output (0, 27)
tolerance used: rtol=0.02, atol=0.25
```

Interpretation: the error is much larger and more structured than FP16 accumulation drift. Tolerance will not be
widened. The exact 75% mismatch rate is evidence for a systematic decode/addressing defect. Next action: compare the
portable unpacked dense weight to the independent reference, then print per-column fused errors to distinguish packed
word addressing from scale indexing or dot-product behavior.

### 2026-07-22 — Straddling-value root cause

Status: successful diagnosis; fix pending validation.

Diagnostic evidence:

```text
portable Torch dequantized weight vs independent reference: max_abs = 0.0
identity-input fused decode: 82 mismatches / 4096 values
mismatching K indices: 10, 21, 42, 53, 74, 85, 106, 117
```

Those indices are exactly positions 10 and 21 in each 32-value packing block. All non-straddling values decode
exactly. The packed tensors and scale expansion are therefore exonerated.

Root cause: qweight words are stored as signed int32. Arithmetic right shift of a word by 30 or 31 sign-extends the
word. The original expression merged the following word and applied `& 0x7` only afterward, allowing sign-extension
bits to occupy the straddling value’s high bit. For example, a low fragment with bit 31 set yields `111` after an
arithmetic shift even when the true high bit in the next word is zero.

Rejected tactic: masking only after combining signed low/high word fragments.

Fix: mask the low fragment to two bits at shift 30 and one bit at shift 31 before OR-ing the high fragment. Apply the
same compile-time layout logic to K-packed GPTQ and N-packed AWQ.

### 2026-07-22 — Raw fused-kernel correctness after straddling fix

Status: successful on physical GPU 0.

Command:

```text
CUDA_VISIBLE_DEVICES=0 pytest -q tests/kernels/test_triton_3bit.py -k 'raw_3bit'
```

Result:

```text
13 passed, 7 deselected in 10.79s
```

Covered:

- GPTQ K-packed and AWQ N-packed layouts.
- FP16 and BF16 activation/scale paths.
- `(M, K, N)` values `(1, 128, 96)`, `(5, 256, 160)`, and `(33, 384, 224)`.
- M and N Triton tile tails, multiple quantization groups, smallest legal K, and repeated 32-value packed blocks.
- A non-default CUDA stream with event-based dependency handoff.

Interpretation: the minimal fused kernel now meets the raw numerical contract at the stated `rtol=0.02, atol=0.25`
without dense weight materialization. This is correctness evidence only; there is no performance claim yet. Next action:
exercise GPTQ and AWQ QuantLinear wrappers, including registered buffer shapes, post-init validation, bias, reshape, and
backend routing.

### 2026-07-22 — Focused raw and QuantLinear correctness checkpoint

Status: successful on physical GPU 0.

Command:

```text
CUDA_VISIBLE_DEVICES=0 pytest -q tests/kernels/test_triton_3bit.py
```

Result:

```text
20 passed in 8.20s
```

Additional proof beyond the raw checkpoint:

- GPTQ `TritonV2Linear` 3-bit registered buffers, natural `g_idx`, post-init, fused routing, bias, and output reshape.
- AWQ `AwqGEMMTritonLinear` 3-bit registered continuous-layout buffers, fused routing, bias, and output reshape.
- Exact conditional capability validation remains green alongside both wrapper paths.

Warnings were pre-existing environment warnings for Triton under free-threaded Python, TorchScript on Python 3.14,
and a deprecated Hugging Face environment variable; no CUDA test skipped and no compile-only result is being counted.

Next action: add pack/persistence and malformed-metadata tests, run nearby 2/4/8-bit backend regressions, and only then
establish the first warmed performance baseline.

### 2026-07-22 — Pack, persistence, and malformed-metadata checkpoint

Status: successful on physical GPU 0.

Command:

```text
CUDA_VISIBLE_DEVICES=0 pytest -q tests/kernels/test_triton_3bit.py -k 'save_reload or rejects'
```

Result:

```text
4 passed, 20 deselected in 7.29s
```

Proven paths:

- GPTQ: real inherited 3-bit packer -> `state_dict` serialization -> registered TritonV2 buffers -> post-init ->
  fused inference.
- AWQ: new symmetric continuous-96-bit packer -> `state_dict` serialization -> registered AWQ Triton buffers ->
  post-init -> fused inference.
- AWQ pack rejects any zero point other than 4 under the V0 symmetric contract.
- GPTQ post-init rejects in-range but non-natural `g_idx`, because the fused `desc_act=False` path intentionally does
  not read `g_idx` per element.

Next action: run nearby backend-selection/hierarchy and established Triton regression tests before benchmarking.

### 2026-07-22 — Nearby regression checks and over-broad selection attempt

Status: focused regressions successful; one over-broad command intentionally interrupted.

Successful commands/results:

```text
CUDA_VISIBLE_DEVICES=0 pytest -q tests/kernels/test_awq_triton_accum.py
2 passed in 11.53s

pytest -q tests/kernels/test_qlinear_hierarchy.py tests/test_triton_g_idx_bounds.py
13 passed in 7.21s
```

These preserve the established 4-bit AWQ Triton FP32-accumulation behavior, QuantLinear class hierarchy, and existing
Triton out-of-bounds `g_idx` validation.

Failed/aborted tactic:

```text
pytest -q tests/kernels/test_selection.py tests/kernels/test_qlinear_hierarchy.py \
  tests/test_triton_g_idx_bounds.py
```

The unfiltered parameterized selection smoke tests began compiling unrelated ExLlamaV2 and CUDA AWQ extensions. The
run was interrupted after `2 passed, 3 skipped` at 41.12s; exit code 2 reflects the deliberate `KeyboardInterrupt`, not
a test assertion failure. Only the process started by this investigation was stopped. Future selection checks will be
targeted to the new exact contract or exclude dependency-heavy smoke cases.

Next action: add a reproducible CUDA-event benchmark/profiler driver and collect the untuned baseline on both requested
GPUs before changing launch geometry.

### 2026-07-22 — Benchmark driver first smoke attempt

Status: failed before CUDA initialization; driver path defect.

Command:

```text
CUDA_VISIBLE_DEVICES=0 python scripts/benchmark_triton_3bit.py --shape 1x128x96 \
  --warmup 2 --iterations 5 --output-json benchmark_artifacts/triton_3bit/smoke_gpu0.json
```

Result:

```text
ModuleNotFoundError: No module named 'gptqmodel'
exit code: 1
```

Cause: invoking a Python file under `scripts/` makes that directory `sys.path[0]`; this environment does not have the
worktree package installed globally. No benchmark or GPU measurement occurred.

Fix: resolve and prepend the repository root from `__file__` before importing local GPT-QModel modules. The documented
command should work without requiring an implicit `PYTHONPATH=.` shell precondition.

### 2026-07-22 — Benchmark driver smoke after path fix

Status: successful plumbing check on physical GPU 0; not accepted as performance evidence.

Result/artifact:

```text
command: identical to the previous smoke command
artifact: /root/GPT-QModel-Ultra/benchmark_artifacts/triton_3bit/smoke_gpu0.json
GPU UUID: cb9e7784-cf50-203d-4f0d-5c622a89b1f2
PCI: 0000:25:00.0
CC: 8.0, SMs: 124
```

The script successfully emitted resolved hardware/software metadata, numerical checks, CUDA-event statistics, an
ASCII table, and JSON. The tiny `(1, 128, 96)` workload used only two warmups and five samples. Its GPTQ fused samples
contained a large outlier (`median 0.1413 ms`, `mean 0.8122 ms`, `p95 3.5297 ms`), demonstrating why this smoke cannot
support a speed claim. The zero max-absolute errors at this tiny case are consistent with prior correctness tests.

Next action: collect matched 50-warmup/100-sample untuned baselines concurrently on physical GPU 0 and GPU 1 using
identical GPTQ/AWQ layouts and decode/prefill shapes.

### 2026-07-22 — Concurrent dual-GPU untuned baseline

Status: successful matched execution on physical GPU 0 and GPU 1.

Pre-run state from `nvidia-smi`:

```text
GPU 0: 0% utilization, 19655 MiB resident, 28 C, 72.15 W
GPU 1: 0% utilization,     4 MiB resident, 31 C, 64.46 W
```

GPU 0’s resident memory was pre-existing external state and was not changed or cleared. Both benchmark processes were
started concurrently with identical arguments, 50 warmups, 100 CUDA-event samples, and `PYTHON_GIL=0`.

Artifacts:

```text
/root/GPT-QModel-Ultra/benchmark_artifacts/triton_3bit/untuned_gpu0.json
/root/GPT-QModel-Ultra/benchmark_artifacts/triton_3bit/untuned_gpu1.json
```

Authoritative medians from those JSON artifacts (`ms`; speedup is fused/materialize median ratio):

```text
+------+--------------------------------------+--------+-----+----------+----------------+----------+---------+
| GPU  | physical UUID                        | layout | M   | fused ms | materialize ms | speedup  | max abs |
+------+--------------------------------------+--------+-----+----------+----------------+----------+---------+
| 0    | cb9e7784-cf50-203d-4f0d-5c622a89b1f2 | GPTQ   |   1 |   0.1193 |         0.1587 |    1.33x | 0.06250 |
| 0    | cb9e7784-cf50-203d-4f0d-5c622a89b1f2 | GPTQ   |  16 |   0.1290 |         0.1628 |    1.26x | 0.12500 |
| 0    | cb9e7784-cf50-203d-4f0d-5c622a89b1f2 | GPTQ   | 128 |   0.1731 |         0.1659 |    0.96x | 0.00000 |
| 0    | cb9e7784-cf50-203d-4f0d-5c622a89b1f2 | AWQ    |   1 |   0.1254 |         2.1898 |   17.46x | 0.06250 |
| 0    | cb9e7784-cf50-203d-4f0d-5c622a89b1f2 | AWQ    |  16 |   0.1352 |         2.1934 |   16.23x | 0.12500 |
| 0    | cb9e7784-cf50-203d-4f0d-5c622a89b1f2 | AWQ    | 128 |   0.9216 |         2.2036 |    2.39x | 0.00000 |
| 1    | 20f7fde4-d88c-d6ca-e324-bd4e5e9e0855 | GPTQ   |   1 |   0.1321 |         0.1587 |    1.20x | 0.06250 |
| 1    | 20f7fde4-d88c-d6ca-e324-bd4e5e9e0855 | GPTQ   |  16 |   0.1290 |         0.1628 |    1.26x | 0.12500 |
| 1    | 20f7fde4-d88c-d6ca-e324-bd4e5e9e0855 | GPTQ   | 128 |   0.1731 |         0.1649 |    0.95x | 0.00000 |
| 1    | 20f7fde4-d88c-d6ca-e324-bd4e5e9e0855 | AWQ    |   1 |   0.1219 |         2.0500 |   16.82x | 0.06250 |
| 1    | 20f7fde4-d88c-d6ca-e324-bd4e5e9e0855 | AWQ    |  16 |   0.1352 |         2.1837 |   16.16x | 0.12500 |
| 1    | 20f7fde4-d88c-d6ca-e324-bd4e5e9e0855 | AWQ    | 128 |   0.9144 |        16.2227 |   17.74x | 0.00000 |
+------+--------------------------------------+--------+-----+----------+----------------+----------+---------+
```

The GPU 1 AWQ M=128 materialize and dense baselines were anomalously slow (`16.2227 ms` and `0.3261 ms` versus GPU
0’s `2.2036 ms` and `0.0481 ms`) while the fused result agreed closely (`0.9144` versus `0.9216 ms`). Therefore no
cross-device claim uses that anomalous baseline. The full method/sample statistics remain in the JSON artifacts.

Findings:

1. GPTQ fused decode is 1.20-1.33x faster than the repository’s existing Triton full-dequantize-plus-matmul path at
   M=1, and 1.26x at M=16 on both devices.
2. GPTQ M=128 is slightly slower than materialize+matmul (0.95-0.96x), so the prefill launch/dataflow needs work.
3. Direct AWQ N-axis packed fused M=128 takes 0.914-0.922 ms, while the same math and shapes in GPTQ’s K-axis packed
   path take 0.1731 ms: about 5.3x slower on both devices. Repeated N lanes load the same packed AWQ word and the
   resulting access/decode layout scales poorly with M.
4. A Marlin-style one-time AWQ repack to the K-packed runtime layout is the leading tactic; it preserves serialized AWQ
   tensors while letting inference use the proven coalesced GPTQ kernel.

Next action: use Nsight Systems on the GPTQ M=128 losing case to confirm launch count and kernel attribution, then use
targeted Nsight Compute sections before changing dataflow or launch geometry.

### 2026-07-22 — Nsight Systems attribution of the untuned GPTQ M=128 case

Status: successful capture and report analysis on physical GPU 0.

Command:

```text
CUDA_VISIBLE_DEVICES=0 PYTHON_GIL=0 nsys profile -c cudaProfilerApi -t cuda,nvtx \
  --sample=none --cpuctxsw=none --force-overwrite=true \
  -o benchmark_artifacts/triton_3bit/nsys_untuned_gptq_m128 -- \
  python scripts/benchmark_triton_3bit.py --layout gptq --shape 128x4096x4096 \
  --profile --profile-iterations 20
```

Artifact:

```text
/root/GPT-QModel-Ultra/benchmark_artifacts/triton_3bit/nsys_untuned_gptq_m128.nsys-rep
```

The profiled child ended with exit code 143 after `cudaProfilerStop`, but Nsight completed and generated a readable
report. This is a capture-shutdown caveat, not a kernel failure. `nsys stats` found exactly 20 GPU operations, all 20
instances of `_matmul_3bit_kernel`; there was no hidden dequantization or auxiliary CUDA kernel in the marked range.

```text
kernel instances:       20
kernel total:            3,375,541 ns
kernel average/median:     168,777 / 168,832 ns
kernel min/max:            167,328 / 169,887 ns
kernel standard deviation:     680.6 ns
cuLaunchKernelEx calls:  20
launch API median:        9,803.5 ns
```

`cuda_kern_exec_sum` reports 19 queued launches because the benchmark intentionally enqueues the 20 profiled calls
back-to-back before synchronization. That queue time is not a CPU launch gap. The trace proves that the losing M=128
case is one stable fused launch per operation and that optimization should focus on the kernel, not launch fusion.

### 2026-07-22 — First Nsight Compute SOL triage

Status: successful three-launch metric collection on physical GPU 0; diagnostic only, with a persistent report queued
for the next deeper pass.

Command:

```text
CUDA_VISIBLE_DEVICES=0 PYTHON_GIL=0 ncu --profile-from-start off --section SpeedOfLight \
  --csv --kernel-name regex:_matmul_3bit_kernel --launch-count 3 --target-processes all -- \
  python scripts/benchmark_triton_3bit.py --layout gptq --shape 128x4096x4096 \
  --profile --profile-iterations 3
```

All launches used grid `(128, 1, 1)` and block `(128, 1, 1)`. Results were consistent:

```text
+--------+-------------+-----------+--------+---------+---------+-----------+
| launch | duration ns | compute % | memory | DRAM %  | L1/TEX %| L2 %      |
+--------+-------------+-----------+--------+---------+---------+-----------+
| 0      |     224,832 |     12.24 |  21.54 |    1.38 |   22.08 |      5.08 |
| 1      |     224,096 |     12.25 |  21.57 |    1.39 |   22.13 |      5.10 |
| 2      |     224,960 |     12.23 |  21.52 |    1.38 |   22.19 |      5.08 |
+--------+-------------+-----------+--------+---------+---------+-----------+
```

The Nsight Compute rule engine reports only `0.2` full waves across the device. Both compute and memory SOL are below
40%, so the profiling workflow classifies this as latency/underfill-bound rather than compute- or DRAM-bandwidth-bound.
The next evidence required is `LaunchStats + Occupancy`, specifically registers/thread, theoretical versus achieved
occupancy, and the limiting resource. This rules out speculative tensor-core or cache work as the first tactic.

Next action: persist a focused Nsight Compute report with `SpeedOfLight`, `LaunchStats`, and `Occupancy`, then sweep
smaller accumulator tiles/increased grid parallelism and retain only a change that improves warmed CUDA-event timing.

### 2026-07-22 — Focused Nsight Compute occupancy report

Status: successful nine-pass collection and persistent report on physical GPU 0.

Artifact:

```text
/root/GPT-QModel-Ultra/benchmark_artifacts/triton_3bit/ncu_untuned_gptq_m128.ncu-rep
```

The focused report confirms the underfill mechanism:

```text
grid / block:                    128 CTAs / 128 threads
registers per thread:           96
dynamic shared memory/block:    8.19 KiB
register-limited blocks/SM:      5
shared-memory-limited blocks/SM: 11
theoretical occupancy:          31.25% (20 warps/SM)
achieved occupancy:              6.45% (4.13 warps/SM)
waves/SM:                        0.21
duration:                        224.19 us under NCU replay
compute / memory SOL:            12.29% / 21.64%
```

There are 124 SMs but only 128 CTAs, so almost every SM receives exactly one four-warp CTA and the kernel ends before
the theoretical five resident CTAs can be used. Registers, not shared memory, cap theoretical occupancy, while the
small grid caps achieved occupancy much further. This directly supports a smaller `BLOCK_M` and/or `BLOCK_N` tactic:
it both increases CTA count and reduces accumulator register pressure. It also gives a rejection criterion: a tile
that raises occupancy but duplicates enough packed-weight traffic to increase event latency is not a win.

Next action: add an explicitly validated benchmark-only launch override, sweep candidate tile/warp configurations in
fresh processes on GPUs 0 and 1, then promote only a cross-device winner into the default selector.

### 2026-07-22 — Reproducible launch-sweep control

Status: implemented a typed `Triton3BitLaunchConfig` override used by the benchmark while runtime callers continue to
use the shape selector by default. Block dimensions must be powers of two in `[16, 128]`; warps are restricted to
Triton-supported powers of two through 8; stages are restricted to `[1, 4]`. The benchmark records the override in
JSON, avoiding source edits or undocumented environment variables during a sweep.

First lint attempt: failed with two `E402` findings on the intentionally delayed local-package imports in the
standalone script. This was exposed by the earlier robust worktree-path fix. The imports now carry narrow `noqa: E402`
annotations; no project-wide lint suppression was added.

Follow-up validation: `ruff check` on the kernel, benchmark, and focused test plus `git diff --check` all passed.

### 2026-07-22 — First concurrent launch-geometry sweep

Status: successful eight-candidate sweep split across physical GPUs 0 and 1, with one harmless orchestration retry.

The first attempt never started a subprocess because the JavaScript orchestration layer interpreted the shell artifact
name's `${cfg}` as a JavaScript interpolation and raised `ReferenceError: cfg is not defined`. Escaping the interpolation
for the shell fixed the command; there was no partial benchmark or GPU state to clean up.

Each successful candidate used `M=128, K=N=4096`, FP16, 50 warmups, 100 CUDA-event samples, and the fused numerical
guard. Every fused check had max absolute error `0.0` against the dense reference for this seeded input.

```text
+--------------------------------------+-----------------+-----------+----------------+----------+
| physical GPU UUID                    | BMxBNxBKxWxS    | fused ms  | materialize ms | speedup  |
+--------------------------------------+-----------------+-----------+----------------+----------+
| cb9e7784-cf50-203d-4f0d-5c622a89b1f2 | 32x64x32x4x1   |    0.1853 |         0.1654 |    0.89x |
| cb9e7784-cf50-203d-4f0d-5c622a89b1f2 | 32x32x32x4x1   |    0.1382 |         0.1659 |    1.20x |
| cb9e7784-cf50-203d-4f0d-5c622a89b1f2 | 64x32x32x4x1   |    0.1260 |         0.1659 |    1.32x |
| cb9e7784-cf50-203d-4f0d-5c622a89b1f2 | 16x64x32x4x1   |    0.2243 |         0.1649 |    0.74x |
| 20f7fde4-d88c-d6ca-e324-bd4e5e9e0855 | 64x64x32x8x1   |    0.1526 |         0.1649 |    1.08x |
| 20f7fde4-d88c-d6ca-e324-bd4e5e9e0855 | 32x64x32x8x1   |    0.1659 |         0.1649 |    0.99x |
| 20f7fde4-d88c-d6ca-e324-bd4e5e9e0855 | 32x64x64x4x1   |    0.1608 |         0.1659 |    1.03x |
| 20f7fde4-d88c-d6ca-e324-bd4e5e9e0855 | 64x32x32x8x1   |    0.1260 |         0.1649 |    1.31x |
+--------------------------------------+-----------------+-----------+----------------+----------+
```

Artifacts are the eight `benchmark_artifacts/triton_3bit/sweep_*_gpu[01].json` files. Narrowing `BLOCK_N` from 64 to
32 while retaining `BLOCK_M=64` is the clear tactic: it doubles the grid from 128 to 256 CTAs and reduces each CTA's
accumulator footprint without duplicating activation work across additional M tiles. Reducing only `BLOCK_M` loses,
and increasing `BLOCK_K` or warps without fixing the N grid is insufficient. The 4-warp and 8-warp `64x32` candidates
tied at 0.1260 ms but were measured on different GPUs, so neither is promoted until a matched cross-device confirmation.

Next action: repeat the default, 4-warp winner, and 8-warp winner on both devices with longer sampling, choose the
lower-variance configuration, then re-run Nsight Compute on the promoted shape.

### 2026-07-22 — Matched dual-GPU confirmation and candidate NCU comparison

Status: successful. Both devices ran the same three configurations concurrently with 100 warmups and 500 CUDA-event
samples per configuration. All numerical guards again had max absolute error `0.0`.

```text
+--------------------------------------+-----------------+-----------+----------------+----------+
| physical GPU UUID                    | BMxBNxBKxWxS    | fused ms  | materialize ms | speedup  |
+--------------------------------------+-----------------+-----------+----------------+----------+
| cb9e7784-cf50-203d-4f0d-5c622a89b1f2 | 64x64x32x4x1   |    0.1741 |         0.1644 |    0.94x |
| cb9e7784-cf50-203d-4f0d-5c622a89b1f2 | 64x32x32x4x1   |    0.1260 |         0.1659 |    1.32x |
| cb9e7784-cf50-203d-4f0d-5c622a89b1f2 | 64x32x32x8x1   |    0.1260 |         0.1659 |    1.32x |
| 20f7fde4-d88c-d6ca-e324-bd4e5e9e0855 | 64x64x32x4x1   |    0.1741 |         0.1649 |    0.95x |
| 20f7fde4-d88c-d6ca-e324-bd4e5e9e0855 | 64x32x32x4x1   |    0.1260 |         0.1649 |    1.31x |
| 20f7fde4-d88c-d6ca-e324-bd4e5e9e0855 | 64x32x32x8x1   |    0.1260 |         0.1659 |    1.32x |
+--------------------------------------+-----------------+-----------+----------------+----------+
```

The six `benchmark_artifacts/triton_3bit/confirm_*_gpu[01].json` files contain full distributions. The narrowed tile
improves the kernel median by 1.38x on both cards and changes the end-to-end comparison from a 5-6% loss to a 31-32%
win over full dequantization plus matmul.

Because event medians tied, the 4-warp candidate was profiled on physical GPU 0 and the 8-warp candidate concurrently
on physical GPU 1 with identical focused NCU sections. Persistent reports:

```text
/root/GPT-QModel-Ultra/benchmark_artifacts/triton_3bit/ncu_candidate_64x32x32x4x1_gpu0.ncu-rep
/root/GPT-QModel-Ultra/benchmark_artifacts/triton_3bit/ncu_candidate_64x32x32x8x1_gpu1.ncu-rep
```

```text
+----------------------+-------------+-------------+-------------+
| metric               | old 64x64x4| 64x32x4    | 64x32x8     |
+----------------------+-------------+-------------+-------------+
| grid / block         | 128 / 128   | 256 / 128   | 256 / 256   |
| registers/thread     | 96          | 72          | 60          |
| theoretical occupancy| 31.25%      | 43.75%      | 50.00%      |
| achieved occupancy   |  6.45%      | 12.88%      | 25.65%      |
| waves/SM             |  0.21       |  0.29       |  0.52       |
| NCU duration         | 224.19 us   | 176.67 us   | 143.65 us   |
| compute SOL          | 12.29%      | 17.43%      | 25.60%      |
| memory SOL           | 21.64%      | 33.89%      | 46.26%      |
+----------------------+-------------+-------------+-------------+
```

NCU replay duration is not substituted for CUDA-event timing, but the resource metrics break the event-time tie:
eight warps cut registers/thread further and double achieved active warps without regressing either card's event
median. Therefore `64x32x32`, 8 warps, 1 stage is promoted for `M > 64`.

Next action: apply the promoted selector and remove the remaining AWQ runtime-layout penalty by adding a Marlin-inspired,
one-time AWQ-to-K-packed repack in `post_init`; keep serialized AWQ tensors intact via a non-persistent runtime buffer.

### 2026-07-22 — Promoted M=128 tile and AWQ runtime repack

Status: implemented; verification in progress.

The default selector now uses `64x32x32`, 8 warps, 1 stage for `M > 64`. Smaller-M selectors remain unchanged because
the profile and sweep targeted only the M=128 regime.

For 3-bit AWQ, `post_init` now validates the packed checkpoint zero points and creates a K-packed runtime qweight from
the serialized N-packed qweight. The runtime tensor is a non-persistent registered buffer: it follows module device
moves, but `state_dict()` retains the original AWQ checkpoint representation and does not serialize backend cache
state. Evaluation on CC >= 8.0 uses the coalesced GPTQ-layout kernel; training, low-capability fallback, and a module
whose lifecycle omitted `post_init` retain the original portable/direct-layout paths. This mirrors Marlin's one-time
runtime repack principle without overwriting the checkpoint tensor.

Tests now assert the repacked bytes, non-persistence, malformed checkpoint zero rejection, and the promoted M=128 path
for both FP16/BF16 and both source layouts. Focused Ruff and whitespace checks pass.

Next action: run the complete focused test file concurrently on physical GPUs 0 and 1 before benchmarking the repacked
AWQ path.

### 2026-07-22 — Concurrent full focused validation after AWQ repack

Status: successful on both requested devices.

Command shape on each process:

```text
CUDA_VISIBLE_DEVICES=<0|1> PYTHON_GIL=0 pytest -q tests/kernels/test_triton_3bit.py
```

Both processes ran concurrently and independently collected the same 29 tests:

```text
physical GPU 0 / cb9e7784-cf50-203d-4f0d-5c622a89b1f2: 29 passed in 11.93s
physical GPU 1 / 20f7fde4-d88c-d6ca-e324-bd4e5e9e0855: 29 passed in 11.86s
```

Coverage includes exact GPTQ pack-byte parity, generic pack/unpack, strict backend capability rejection, both packing
layouts, FP16 and BF16, boundary-straddling 3-bit codes, M=1/5/33/128, current-stream execution, bias/wrapper behavior,
pack-save-reload-inference, AWQ runtime repack/non-persistence, malformed AWQ zeros, and malformed GPTQ `g_idx`.

Next action: collect matched optimized GPTQ/AWQ inference tables on both GPUs, including the serialized AWQ layout as a
diagnostic method but treating the post-init K-packed buffer as the production fused path.

### 2026-07-22 — First optimized table exposed an AWQ layout crossover

Status: successful benchmark, but the always-repacked AWQ dispatch tactic was only partially successful and is not the
final selector.

Artifacts from matched 100-warmup/500-sample runs:

```text
/root/GPT-QModel-Ultra/benchmark_artifacts/triton_3bit/optimized_gpu0.json
/root/GPT-QModel-Ultra/benchmark_artifacts/triton_3bit/optimized_gpu1.json
```

GPTQ production medians were 0.1188-0.1382 ms at M=1, 0.1290-0.1341 ms at M=16, and 0.1270 ms at M=128. The M=128
result is a stable 1.31x faster than full dequantization plus matmul on both GPUs.

AWQ's K-packed runtime buffer gave 0.1249 ms at M=128 versus 0.1475-0.1485 ms for direct serialized-layout access, a
1.18-1.19x runtime-layout win. At M=16, however, direct N-packed access was 0.1188-0.1208 ms versus 0.1290 ms for the
repacked layout. Always selecting the runtime repack would therefore regress the small-token regime even though it
fixed prefill. This tactic is retained only as a runtime option, not unconditional dispatch.

### 2026-07-22 — AWQ layout-crossover sweep

Status: successful and consistent on both GPUs, using 50 warmups and 200 CUDA-event samples per shape.

Artifacts:

```text
/root/GPT-QModel-Ultra/benchmark_artifacts/triton_3bit/awq_layout_crossover_gpu0.json
/root/GPT-QModel-Ultra/benchmark_artifacts/triton_3bit/awq_layout_crossover_gpu1.json
```

```text
+-----+----------------------+----------------------+----------------------+
| M   | K-packed ms GPU0/1  | N-packed ms GPU0/1  | selected layout      |
+-----+----------------------+----------------------+----------------------+
| 32  | 0.1423 / 0.1423      | 0.1188 / 0.1198      | serialized N-packed  |
| 64  | 0.1597 / 0.1587      | 0.1311 / 0.1321      | serialized N-packed  |
| 65  | 0.1198 / 0.1198      | 0.1393 / 0.1393      | runtime K-packed     |
| 96  | 0.1208 / 0.1260      | 0.1423 / 0.1423      | runtime K-packed     |
| 128 | 0.1260 / 0.1260      | 0.1475 / 0.1485      | runtime K-packed     |
| 256 | 0.2386 / 0.2386      | 0.2888 / 0.2888      | runtime K-packed     |
+-----+----------------------+----------------------+----------------------+
```

The crossover is sharp and identical to the selector boundary where `M > 64` switches to the profiled 64x32 tile.
Final AWQ dispatch therefore uses the serialized N-packed kernel for `M <= 64` and its post-init K-packed runtime buffer
for `M >= 65`. The benchmark's `fused` row now models this production hybrid; its diagnostic row is renamed
`alternate-layout-fused` so it always means the non-selected layout.

Next action: validate both sides of the hybrid boundary on both GPUs, rerun the final matched table, then capture final
Nsight Systems evidence for GPTQ and AWQ production dispatch.

Implementation check caught one failed edit before execution: the first benchmark-selector patch matched the earlier
`return {` in `_device_metadata` and placed the hybrid-selection block in the wrong function. Ruff reported the ten
resulting undefined/unused names. The block was moved into `_make_case` immediately before its return; no benchmark ran
with the malformed script.

### 2026-07-22 — Rejected iteration-only warmup for light-kernel comparisons

Status: measurement tactic rejected and corrected.

The first post-hybrid matched run produced physically inconsistent method-order results: AWQ M=1 direct-layout latency
jumped to 0.9298-0.9779 ms on both GPUs while its mathematically equivalent alternate packed path remained about
0.118 ms. GPU 1 also produced unrelated M=128 dense/materialize outliers. An immediate isolated matched rerun restored
AWQ direct M=1 to 0.1178 ms on GPU 0 and 0.1219 ms on GPU 1, and a GPTQ-then-AWQ single-process rerun remained normal
at 0.1239 ms. The anomalous artifacts are retained as `final_gpu[01].json`; they are not accepted as final evidence.

The likely confounder is workload/order-dependent clock state: the cards idle at a reported 210 MHz, and 100 warmups of
a 0.12 ms kernel provide only about 12 ms of light GPU work. Swapping which layout was timed first also swapped which
one appeared faster at M=16. The benchmark now performs the same unmeasured 128-iteration 4096x4096 FP16 GEMM clock
warmup before every method, followed by the method's own warmups. This common precondition models a linear layer inside
an already-active inference workload and is recorded in JSON. Iteration-only warmups remain in place for cache/JIT
stabilization.

Next action: repeat the layout crossover and final dual-GPU tables with common clock preconditioning; discard the
hybrid threshold if the order-controlled evidence no longer supports it.

### 2026-07-22 — Clock-controlled AWQ layout decision

Status: successful matched rerun; provisional hybrid threshold rejected.

Artifacts:

```text
/root/GPT-QModel-Ultra/benchmark_artifacts/triton_3bit/awq_layout_clocked_gpu0.json
/root/GPT-QModel-Ultra/benchmark_artifacts/triton_3bit/awq_layout_clocked_gpu1.json
```

With identical heavy preconditioning before every method, the apparent small-M advantage of serialized N packing
mostly reversed:

```text
+-----+----------------------+----------------------+----------------------+
| M   | K-packed ms GPU0/1  | N-packed ms GPU0/1  | finding              |
+-----+----------------------+----------------------+----------------------+
| 1   | 0.1198 / 0.1178      | 0.1239 / 0.1198      | K slight win         |
| 16  | 0.1229 / 0.1219      | 0.1352 / 0.1362      | K 1.10-1.12x faster  |
| 32  | 0.1229 / 0.1239      | 0.1403 / 0.1413      | K 1.14x faster       |
| 64  | 0.1618 / 0.1505      | 0.1516 / 0.1526      | mixed/tied           |
| 65  | 0.1239 / 0.1167      | 0.1628 / 0.1403      | K 1.20-1.31x faster  |
| 96  | 0.1239 / 0.1188      | 0.1741 / 0.1577      | K 1.33-1.41x faster  |
| 128 | 0.1249 / 0.1219      | 0.1628 / 0.1628      | K 1.30-1.34x faster  |
+-----+----------------------+----------------------+----------------------+
```

Only M=64 is mixed across devices, and the difference is small relative to the clear K-packed wins elsewhere. The
provisional `M >= 65` hybrid is removed. Final inference always uses the post-init K-packed buffer when available; the
serialized AWQ layout remains only the fallback when `post_init` was omitted or fused execution is unavailable. This
reduces dispatch complexity and is the evidence-backed Marlin-style design.

Next action: rerun the final GPTQ/AWQ table with clock control and then capture matched production Nsight Systems traces.

### 2026-07-22 — Final clock-controlled dual-GPU benchmark

Status: successful matched run with 128 heavy clock-warmup GEMMs per method, 100 method warmups, and 500 CUDA-event
samples. These are the authoritative performance artifacts:

```text
/root/GPT-QModel-Ultra/benchmark_artifacts/triton_3bit/final_clocked_gpu0.json
/root/GPT-QModel-Ultra/benchmark_artifacts/triton_3bit/final_clocked_gpu1.json
```

FP16, `K=N=4096`, exact `bits=3/group_size=128/desc_act=False/sym=True` medians:

```text
+------+--------------------------------------+--------+-----+----------+----------------+----------+---------+
| GPU  | physical UUID                        | source | M   | fused ms | materialize ms | speedup  | max abs |
+------+--------------------------------------+--------+-----+----------+----------------+----------+---------+
| 0    | cb9e7784-cf50-203d-4f0d-5c622a89b1f2 | GPTQ   |   1 |   0.1239 |         0.1454 |    1.17x | 0.06250 |
| 0    | cb9e7784-cf50-203d-4f0d-5c622a89b1f2 | GPTQ   |  16 |   0.1249 |         0.1485 |    1.19x | 0.12500 |
| 0    | cb9e7784-cf50-203d-4f0d-5c622a89b1f2 | GPTQ   | 128 |   0.1270 |         0.1516 |    1.19x | 0.00000 |
| 0    | cb9e7784-cf50-203d-4f0d-5c622a89b1f2 | AWQ    |   1 |   0.1244 |         2.0634 |   16.58x | 0.06250 |
| 0    | cb9e7784-cf50-203d-4f0d-5c622a89b1f2 | AWQ    |  16 |   0.1290 |         2.1453 |   16.63x | 0.12500 |
| 0    | cb9e7784-cf50-203d-4f0d-5c622a89b1f2 | AWQ    | 128 |   0.1249 |         2.1729 |   17.39x | 0.00000 |
| 1    | 20f7fde4-d88c-d6ca-e324-bd4e5e9e0855 | GPTQ   |   1 |   0.1203 |         0.1587 |    1.32x | 0.06250 |
| 1    | 20f7fde4-d88c-d6ca-e324-bd4e5e9e0855 | GPTQ   |  16 |   0.1229 |         0.1587 |    1.29x | 0.12500 |
| 1    | 20f7fde4-d88c-d6ca-e324-bd4e5e9e0855 | GPTQ   | 128 |   0.1260 |         0.1516 |    1.20x | 0.00000 |
| 1    | 20f7fde4-d88c-d6ca-e324-bd4e5e9e0855 | AWQ    |   1 |   0.1249 |         2.1238 |   17.00x | 0.06250 |
| 1    | 20f7fde4-d88c-d6ca-e324-bd4e5e9e0855 | AWQ    |  16 |   0.1290 |         2.0608 |   15.97x | 0.12500 |
| 1    | 20f7fde4-d88c-d6ca-e324-bd4e5e9e0855 | AWQ    | 128 |   0.1290 |         2.0490 |   15.88x | 0.00000 |
+------+--------------------------------------+--------+-----+----------+----------------+----------+---------+
```

AWQ's materialize baseline is the portable Torch unpack/dequantize plus matmul fallback because the repository's
established AWQ Triton kernels are hard-coded for 4-bit packing. Its large speedup therefore means "versus the valid
3-bit fallback," not versus a pre-existing optimized AWQ-3 kernel. Full mean/std/min/p95/max and dense-mm rows are in
the JSON artifacts.

### 2026-07-22 — Final paired Nsight Systems proof

Status: successful concurrent captures: production GPTQ M=128 on physical GPU 0 and production AWQ M=128 on physical
GPU 1. Both `nsys` wrapper processes returned 143 only after the application closed its CUDA-profiler capture range;
both reports completed and parse successfully.

Artifacts:

```text
/root/GPT-QModel-Ultra/benchmark_artifacts/triton_3bit/nsys_final_gptq_m128_gpu0.nsys-rep
/root/GPT-QModel-Ultra/benchmark_artifacts/triton_3bit/nsys_final_awq_m128_gpu1.nsys-rep
```

```text
+--------------------------+----------------------+----------------------+
| nsys metric              | GPTQ / physical GPU0 | AWQ / physical GPU1  |
+--------------------------+----------------------+----------------------+
| marked operations        | 20                   | 20                   |
| `_matmul_3bit_kernel`    | 20                   | 20                   |
| any other GPU kernel     | 0                    | 0                    |
| kernel median            | 120.2405 us          | 120.6870 us          |
| kernel average           | 120.2529 us          | 120.8454 us          |
| kernel min / max         | 118.752 / 122.976 us | 119.615 / 123.999 us |
| launch API median        | 9.844 us             | 10.004 us            |
+--------------------------+----------------------+----------------------+
```

This proves that both source formats reach the same single-launch runtime dataflow and that AWQ repacking is not in the
hot path.

### 2026-07-22 — Final Nsight Compute proof and remaining headroom

Status: successful focused report on the promoted GPTQ M=128 kernel:

```text
/root/GPT-QModel-Ultra/benchmark_artifacts/triton_3bit/ncu_final_gptq_m128_gpu0.ncu-rep
```

Compared with the untuned report, the final tile changes grid/block from `128/128` to `256/256`, registers/thread from
96 to 60, theoretical occupancy from 31.25% to 50.00%, and achieved occupancy from 6.45% to 25.84%. NCU replay duration
falls from 224.19 us to 144.61 us; compute SOL rises from 12.29% to 25.66% and memory SOL from 21.64% to 46.37%.
CUDA-event latency, not replay duration, remains the performance authority.

NCU still reports only 0.52 waves/SM and register-limited 50% theoretical occupancy. That is explicit future headroom,
not evidence for another unmeasured change: further splitting could duplicate packed-weight or activation traffic, so a
future iteration should sweep larger production shapes and inspect warp stalls before adding split-K or another tile.

Next action: run the final dual-GPU regression suite, established neighboring backend tests, Ruff, and final diff review.

### 2026-07-22 — Final source review and verification

Status: V0 complete for the exact requested contract.

The final source review caught three boundary issues before handoff:

1. Adding BF16 for 3-bit AWQ had broadened the class-level dtype declaration. A conditional validator now explicitly
   keeps 4-bit AWQ BF16 unsupported, preserving the previous 4-bit selection behavior while permitting the tested
   3-bit BF16 kernel path.
2. An empty dynamic map is semantically "no per-layer override" and is now accepted; non-empty dynamic maps remain
   rejected for the V0 3-bit specialization.
3. GPTQ inference centers codes at constant zero 4 and therefore must validate checkpoint data, not only metadata.
   `post_init` now unpacks and checks GPTQ qzeros just as the AWQ path does. A malformed symmetric checkpoint fails
   before launch instead of silently computing with inconsistent zero points.

These are load/selection checks and do not alter the profiled hot kernel. New regression cases cover all three.

Final dual-GPU command:

```text
CUDA_VISIBLE_DEVICES=<0|1> PYTHON_GIL=0 pytest -q tests/kernels/test_triton_3bit.py
```

Results:

```text
physical GPU 0 / cb9e7784-cf50-203d-4f0d-5c622a89b1f2: 33 passed in 10.83s
physical GPU 1 / 20f7fde4-d88c-d6ca-e324-bd4e5e9e0855: 33 passed in 10.54s
CUDA tests executed; none were skipped.
```

Established neighboring regressions on physical GPU 0:

```text
pytest -q tests/kernels/test_awq_triton_accum.py \
  tests/kernels/test_qlinear_hierarchy.py tests/test_triton_g_idx_bounds.py
result: 15 passed in 7.49s
```

Final static checks:

```text
ruff check gptqmodel/nn_modules/triton_utils/three_bit.py \
  gptqmodel/nn_modules/qlinear/gemm_awq_triton.py \
  gptqmodel/nn_modules/qlinear/tritonv2.py \
  scripts/benchmark_triton_3bit.py tests/kernels/test_triton_3bit.py
result: All checks passed

git diff --check
result: passed with no output
```

Final implementation files:

```text
design_trilin_kernel_3bit.md
gptqmodel/nn_modules/triton_utils/three_bit.py
gptqmodel/nn_modules/qlinear/tritonv2.py
gptqmodel/nn_modules/qlinear/gemm_awq_triton.py
tests/kernels/test_triton_3bit.py
scripts/benchmark_triton_3bit.py
```

Known V0 boundaries remain deliberate: only 3-bit/group-128/natural-order/symmetric-zero-4/int32 packing; K divisible
by 128; N divisible by 32; fused CC >= 8.0; and a repository-defined continuous AWQ-3 serialization contract rather
than an unverified claim of compatibility with an external 3-bit AWQ checkpoint format. Existing 2/4/8-bit paths and
unsupported-device fallbacks remain intact.

### 2026-07-22 — V0 publication and occupancy iteration start

Status: V0 was committed as `cb71f4f2` (`Add fused Triton 3-bit GPTQ and AWQ kernel`), pushed to
`gptq-3bit-kernel`, and published as draft PR <https://github.com/ModelCloud/GPT-QModel-Ultra/pull/29>.

The next optimization target is achieved occupancy at the production `M=128, K=N=4096` shape. The current
`BM=64, BN=32, BK=32, warps=8, stages=1` launch has 256 CTAs on a 124-SM GPU. Nsight Compute reports 50.00%
theoretical occupancy but 25.84% achieved occupancy and 0.52 waves/SM. A concrete first hypothesis is therefore
grid-limited residency: 256 CTAs provide only about 2.06 CTAs/SM, while the 60-register kernel can theoretically
host four 8-warp CTAs/SM. Before changing the promoted selector, the iteration will:

1. capture scheduler and warp-state metrics for the committed control;
2. sweep `BN=16` and smaller-M tiles to create 512 or more CTAs without split-K atomics;
3. test `BK=16` as a register-pressure alternative;
4. compare CUDA-event latency on both physical GPUs, then profile only credible winners; and
5. reject occupancy gains that regress end-to-end kernel latency.

This explicitly treats occupancy as a means to hide latency, not as the optimization objective by itself.

### 2026-07-22 — Deeper occupancy control profile

Status: successful 13-pass Nsight Compute capture on physical GPU 0.

Artifact:

```text
/root/GPT-QModel-Ultra/benchmark_artifacts/triton_3bit/ncu_occupancy_control_gptq_m128_gpu0.ncu-rep
```

The committed control reproduces the V0 profile at `BM64/BN32/BK32/W8/S1`: 256 CTAs, 256 threads/CTA,
60 registers/thread, 6.14 KiB dynamic shared memory/CTA, 50.00% theoretical occupancy, 25.71% achieved occupancy,
0.52 waves/SM, and 144.67 us replay duration. The new scheduler and warp-state evidence is:

```text
active warps/scheduler:                4.08
eligible warps/scheduler:              0.41
scheduler cycles with eligible warp: 27.21%
scheduler cycles with no eligible:   72.79%
cycles per issued instruction:       15.01
long-scoreboard stall:                 7.92 cycles/instruction (52.73%)
```

This supports the grid-residency hypothesis rather than disproving it: roughly two CTAs/SM expose only half of the
register-limited theoretical warps, and the exposed warps frequently wait for L1TEX dependencies. The first measured
tactic will double the grid to 512 CTAs through `BN=16` or `BM=32`, allowing up to about four CTAs/SM without changing
the K reduction or adding synchronization. CUDA-event latency on both GPUs remains the acceptance criterion.

### 2026-07-22 — Occupancy-oriented tile sweep

Status: successful matched nine-configuration sweep run concurrently on both physical GPUs. Each configuration used
FP16 GPTQ, `M=128, K=N=4096`, 64 clock-warmup GEMMs, 50 kernel warmups, 300 CUDA-event samples, and the fused numerical
guard. All valid candidates matched the seeded dense reference with max absolute error `0.0`.

```text
+-----------------+------------------+------------------+-------------------------------------------+
| BMxBNxBKxWxS    | GPU0 median ms   | GPU1 median ms   | result                                    |
+-----------------+------------------+------------------+-------------------------------------------+
| 64x32x32x8x1    |           0.1260 |           0.1260 | committed control                         |
| 64x16x32x4x1    |           0.1280 |           0.1290 | close, but no occupancy increase expected |
| 64x16x32x8x1    |           0.1505 |           0.1516 | higher occupancy, 19-20% slower           |
| 64x16x16x4x1    |           0.8704 |           0.8735 | clock-contaminated; all methods slowed    |
| 64x16x16x8x1    |           0.2253 |           0.2263 | rejected                                  |
| 32x32x32x8x1    |           0.1751 |           0.1751 | rejected                                  |
| 32x32x16x8x1    |           0.2458 |           0.2478 | rejected                                  |
| 32x16x32x4x1    |           0.1597 |           0.1597 | rejected                                  |
| 32x16x32x8x1    |           0.2324 |           0.2324 | rejected                                  |
+-----------------+------------------+------------------+-------------------------------------------+
```

Artifacts are `benchmark_artifacts/triton_3bit/occupancy_sweep_gpu[01]_<config>.json`. The `BK=16/W4` row is not
treated as a valid absolute timing: both GPUs simultaneously showed a roughly 6x slowdown in the fused, dense, and
materialization methods during that pair of processes. Its normalized behavior and every other `BK=16` result still
fail to identify a credible winner, so it is not promoted. This is logged as a measurement failure rather than hidden.

The main conclusion is that merely splitting the output tile does not improve latency. `BN=16/W4` keeps the same
total launched warp count as the control, while `BN=16/W8` doubles the launched warps but assigns less useful work to
each warp. Focused profiles are required to distinguish whether the latter at least achieves the intended residency.

### 2026-07-22 — High-occupancy candidate profile

Status: successful concurrent Nsight Compute profiles of `BN=16/W8` on physical GPU 0 and `BN=16/W4` on physical
GPU 1. Persistent reports:

```text
/root/GPT-QModel-Ultra/benchmark_artifacts/triton_3bit/ncu_occupancy_candidate_64x16x32x8x1_gpu0.ncu-rep
/root/GPT-QModel-Ultra/benchmark_artifacts/triton_3bit/ncu_occupancy_candidate_64x16x32x4x1_gpu1.ncu-rep
```

```text
+-----------------------------+------------------+------------------+------------------+
| metric                      | BN32/W8 control | BN16/W4          | BN16/W8          |
+-----------------------------+------------------+------------------+------------------+
| grid / threads              | 256 / 256       | 512 / 128       | 512 / 256       |
| registers/thread            | 60              | 64              | 48              |
| theoretical occupancy       | 50.00%          | 50.00%          | 62.50%          |
| achieved occupancy          | 25.71%          | 25.68%          | 51.18%          |
| active warps/scheduler      | 4.08            | 4.11            | 8.16            |
| scheduler eligible cycles   | 27.21%          | 32.75%          | 41.10%          |
| long-scoreboard cycles/inst | 7.92            | 6.8             | 9.3             |
| L1/TEX throughput           | 49.23%          | 70.18%          | 78.43%          |
| NCU duration                | 144.67 us       | 139.87 us       | 154.43 us       |
| CUDA-event median           | 0.1260 ms       | 0.128-0.129 ms  | 0.151-0.152 ms  |
+-----------------------------+------------------+------------------+------------------+
```

The tactic successfully doubles achieved occupancy and active warps, so the original grid-residency diagnosis was
correct. It is nevertheless rejected for promotion: L1/TEX becomes the dominant resource, the absolute
long-scoreboard delay grows, and authoritative CUDA-event latency regresses on both GPUs. Occupancy is not accepted
when it increases redundant activation-tile work enough to slow the operation.

Next tactic: reduce the packed-weight load/decode dependency chain and FP32 dequantization temporaries, then retest
both the efficient `BN=32/W8` control geometry and the now-proven `BN=16/W8` high-occupancy geometry.

### 2026-07-22 — FP16 dequantization-temporary tactic

Status: correctness success, performance failure, reverted.

The experiment changed the dequantization intermediate from explicit FP32 subtract/multiply followed by an FP16 cast
to integer subtract followed by an FP16 multiply. This is mathematically compatible with the dense FP16 reference and
all 33 focused tests passed independently on each physical GPU. A matched 100-warmup/500-sample benchmark nevertheless
showed a stable regression:

```text
+------------------+----------------------+----------------------+----------------------+
| geometry         | V0 GPU0/GPU1 ms     | FP16 temp GPU0/GPU1  | outcome              |
+------------------+----------------------+----------------------+----------------------+
| 64x32x32x8x1    | 0.1260 / 0.1260      | 0.1372 / 0.1382      | 8.9-9.7% slower      |
| 64x16x32x8x1    | 0.1505 / 0.1516      | 0.1495 / 0.1505      | noise-level change   |
+------------------+----------------------+----------------------+----------------------+
```

Artifacts are `benchmark_artifacts/triton_3bit/fp16_dequant_gpu[01]_<config>.json`. The change was reverted immediately;
there is no reason to spend an Nsight replay on a tactic that already loses authoritative event timing on both GPUs.
This also shows that reducing source-level precision conversions is not equivalent to reducing the measured L1TEX
dependency chain.

Next tactic: load the three unique int32 words for each aligned 32-code K block once and select/broadcast them in
registers, instead of expressing repeated packed-word addresses across all 32 logical K lanes.

### 2026-07-22 — Unique packed-word load/broadcast tactic

Status: initial correctness failure fixed; corrected implementation was a performance failure and was reverted.

The first prototype loaded three int32 words per aligned 32-code block and selected them with register broadcasts.
It compiled, and all AWQ direct-layout cases remained correct, but all nine GPTQ-focused cases failed identically on
both GPUs. Root cause: the original kernel's high-word load is masked to the two codes that cross a 32-bit boundary;
the prototype selected a neighboring word for every code and ORed it after a wrapped shift. Reintroducing
`shift > 29` around the high fragment fixed the bug. The immediate rerun passed all 17 raw matmul/stream tests on both
physical GPUs. No failing implementation was benchmarked.

The corrected version then ran 100 warmups and 500 CUDA-event samples on both GPUs:

```text
+------------------+----------------------+----------------------+----------------------+
| geometry         | V0 GPU0/GPU1 ms     | unique-load GPU0/GPU1| outcome              |
+------------------+----------------------+----------------------+----------------------+
| 64x32x32x8x1    | 0.1260 / 0.1260      | 0.1546 / 0.1597      | 22.7-26.8% slower    |
| 64x16x32x8x1    | 0.1505 / 0.1516      | 0.1638 / 0.1638      | 8.1-8.8% slower      |
+------------------+----------------------+----------------------+----------------------+
```

Artifacts are `benchmark_artifacts/triton_3bit/unique_word_load_gpu[01]_<config>.json`. Explicit broadcasting and
selection is substantially more expensive than the compiler's lowering of repeated-address loads, so the apparent
source-level redundancy was not a valid optimization proxy. The tactic was reverted completely.

Next tactic: sweep Triton's software-pipeline stages on the known efficient and known high-occupancy geometries. This
targets the measured long-scoreboard waits without rewriting the compiler-friendly decode expression.

### 2026-07-22 — Triton software-pipeline stage sweep

Status: successful matched dual-GPU sweep; stages above one were rejected.

Each geometry ran stages 1 through 4 with 50 warmups and 300 CUDA-event samples on both physical GPUs. Median results:

```text
+------------------+----------------------+----------------------+----------------------+
| geometry         | stages              | GPU0 ms              | GPU1 ms              |
+------------------+----------------------+----------------------+----------------------+
| 64x32x32x8      | 1 / 2 / 3 / 4       | .1249/.1649/.1669/.1679 | .1260/.1659/.1669/.1679 |
| 64x16x32x8      | 1 / 2 / 3 / 4       | .1505/.2468/.2330/.2294 | .1505/.2488/.2342/.2304 |
+------------------+----------------------+----------------------+----------------------+
```

Artifacts are `benchmark_artifacts/triton_3bit/stage_sweep_gpu[01]_<config>.json`. Extra stages are a large and
consistent regression for both the efficient and high-occupancy tiles. Triton's added pipeline state does not hide
enough of this decode loop's dependency latency to repay its resource and scheduling cost. Stage 1 remains promoted.

### 2026-07-22 — Optimization objective redirected to latency and throughput

The user explicitly redirected the work away from occupancy as a target. From this point, occupancy is diagnostic
metadata only. A tactic is accepted only when warmed CUDA-event latency and throughput improve on both physical GPUs;
native CUDA is in scope and should replace the Triton hot path when it provides a validated win, with Triton retained
as the portable fallback. The next investigation audits the repository's Marlin CUDA dataflow, packing assumptions,
JIT/extension registration, architecture gates, and test infrastructure for the smallest credible 3-bit native path.

### 2026-07-22 — Native CUDA route audit

Status: repository audit complete; one direct route rejected and one exact-value compatibility route selected for a
measured prototype.

The generated Marlin integer kernels and their repackers instantiate 4-bit and 8-bit scalar types. Although the scalar
type registry contains a `uint3b4` descriptor, the ordinary GPTQ/Marlin packing path assumes `32 / bits` integral
packing and the generated CUDA dequantization schedules do not implement continuous 96-bit blocks. Passing the V0
3-bit checkpoint buffer directly to Marlin would therefore be a format mismatch, not a valid 3-bit implementation.

ExLlamaV2 also contains native 3-bit dequantization machinery, but its GPTQ `QMatrix` constructor derives K from a
4-bit-shaped `qweight` (`packed_rows * 8`). Its general EXL2 3-bit format is not the requested continuous GPTQ/AWQ
format, and adapting that matrix lifecycle would be a larger, format-specific port. It was rejected as the first
native tactic.

The selected compatibility route preserves the checkpoint and its numerical interpretation exactly:

```text
checkpoint code:                 q3 in [0, 7]
requested symmetric value:       q3 - 4
runtime Marlin uint4b8 nibble:    q4 = q3 + 4
Marlin signed interpretation:     q4 - 8 = q3 - 4
```

Thus, a one-time post-initialization conversion can expand each 3-bit code into a 4-bit nibble and reuse the native
Marlin `uint4b8` Tensor Core kernel without changing scales, zero-point semantics, or outputs. Serialized weights stay
3-bit; only the non-persistent runtime cache grows from 3 to 4 bits/weight (33.3%). That memory tradeoff must remain
explicit. The native cache will be optional and guarded, with the fused Triton kernel retained for BF16, N not
divisible by 64, unavailable native builds, and other unsupported conditions.

The existing FP16 Marlin JIT extension compiled successfully on this host in 139 seconds and was cached under:

```text
/root/.cache/gptqmodel/torch_extensions/marlin_fp16/1c97e271f1ebddbd
```

An initial GPU-0 `M=1, K=N=4096` smoke measurement produced `0.0625 ms` for expanded Marlin versus `0.1249 ms`
for fused Triton, with the same `0.0625` maximum absolute error. This justified a complete dual-GPU sweep before
integration.

### 2026-07-22 — Expanded-native Marlin benchmark

Status: successful on both physical GPUs and accepted for integration as the FP16 fast path.

Both processes ran concurrently with 128 large-GEMM clock warmups, 100 operation warmups, and 500 CUDA-event samples
per method. The benchmark covers both serialized layouts and all target M regimes. Persistent artifacts:

```text
benchmark_artifacts/triton_3bit/marlin_expanded_full_gpu0.json
benchmark_artifacts/triton_3bit/marlin_expanded_full_gpu1.json
```

GPTQ medians:

```text
+------+-------------------+-------------------+-------------------+-------------------+----------------+
| M    | GPU0 Triton ms    | GPU0 native ms   | GPU1 Triton ms    | GPU1 native ms   | native TFLOP/s |
+------+-------------------+-------------------+-------------------+-------------------+----------------+
| 1    |            0.1249 |            0.0645 |            0.1239 |            0.0614 |    0.520-0.546 |
| 16   |            0.1290 |            0.0696 |            0.1321 |            0.0635 |    7.710-8.456 |
| 128  |            0.1260 |            0.0727 |            0.1393 |            0.0819 |  52.429-59.075 |
+------+-------------------+-------------------+-------------------+-------------------+----------------+
```

AWQ medians after its one-time N-packed-to-K-packed conversion:

```text
+------+-------------------+-------------------+-------------------+-------------------+----------------+
| M    | GPU0 Triton ms    | GPU0 native ms   | GPU1 Triton ms    | GPU1 native ms   | native TFLOP/s |
+------+-------------------+-------------------+-------------------+-------------------+----------------+
| 1    |            0.1224 |            0.0625 |            0.1193 |            0.0614 |    0.537-0.546 |
| 16   |            0.1229 |            0.0655 |            0.1208 |            0.0625 |    8.192-8.595 |
| 128  |            0.1183 |            0.0727 |            0.1219 |            0.0727 |         59.075 |
+------+-------------------+-------------------+-------------------+-------------------+----------------+
```

The native route reduces median latency by approximately 1.5x to 2.1x across every device/layout/M combination. Its
numerical guard exactly matches the existing fused path: maximum absolute error is `0.0625` at M=1, `0.125` at M=16,
and `0.0` for the seeded M=128 case. This is a real hot-path win and is accepted. Integration will construct the
expanded and Marlin-repacked tensors once in `post_init`, register every derived tensor as non-persistent, and route
eligible FP16 inference through native CUDA. Checkpoint state and the Triton fallback remain unchanged.

Memory accounting correction: the native cache itself is 4 bits/weight, 33.3% larger than the serialized 3-bit
tensor. Because the initial integration retains that original tensor on the same device for state-dict correctness and
BF16/unsupported-shape fallback, total resident qweight storage is temporarily 7 bits/weight (a 133.3% increase over
3-bit-only residency), plus scales and small workspace. `GPTQMODEL_TRILIN_NATIVE=0` disables this cache. A future true
3-bit native kernel or a lifecycle-safe source-weight offload is required to remove that residency cost; it must not be
described as a 4-bit-total implementation while both copies are resident.

### 2026-07-22 — Native reduction and packed-prefill control sweep

Status: all alternate controls rejected; ordinary Marlin with FP32 reduction remains selected.

The integrated production helper was swept concurrently on both GPUs at `K=N=4096`, M=1/16/128, with 64 clock
warmups, 100 operation warmups, and 300 event samples. The controls were ordinary Marlin with reduced-precision
reduction and each of the four repository packed-prefill specializations.

Reduced-precision reduction moved M=1 from about `0.0809` to `0.0778 ms` on both GPUs and gave a similarly small
M=16 change. It increased seeded M=1 mean absolute error from `0.000078` to `0.009107` and M=16 mean error from
`0.007670` to `0.011858`. At M=128 it was slower on GPU 0 (`0.0922` to `0.0963 ms`) even though GPU 1 improved.
It is rejected because the speedup is not universal and its accuracy cost is measurable.

Packed configs 1-4 are inactive below M=16 by native dispatch. At M=16 every config regressed on both GPUs: ordinary
was `0.0819/0.0788 ms` on GPU0/GPU1 while packed variants ranged `0.0870-0.0922/0.0881-0.0901 ms`. At M=128,
GPU 0 regressed from `0.0922` to `0.1024-0.1075 ms`; GPU 1 improved to approximately `0.0881-0.0891 ms`, but a
device-specific win is insufficient for promotion. The seeded outputs remained within the established error guard.
The production helper therefore keeps `use_fp32_reduce=True`, `use_packed_prefill=False`, and config zero.

The sweep's absolute clock state differed from the earlier full benchmark, so it is used only as a matched within-run
control comparison. The earlier 500-sample dual-GPU sweep remains the authoritative native-versus-Triton result.

### 2026-07-22 — True continuous-3-bit native CUDA kernel design

Status: implemented, correctness-proven, and selected for FP16 decode/small-token inference.

The expanded Marlin route proved that native Tensor Core execution was worthwhile, but retaining a 4-bit runtime
qweight cache costs memory. The next implementation therefore consumes the original GPTQ continuous 96-bit blocks
directly. AWQ uses its already-required non-persistent N-packed-to-K-packed 3-bit runtime tensor; neither method
creates a dense weight for this path.

The native source is `gptqmodel_ext/trilin/trilin_3bit_wmma.cu`; its Python JIT wrapper is
`gptqmodel/utils/trilin.py`. The `gptqmodel.extension` registry exposes the concrete name `trilin` and alias
`trilin_3bit`. The current kernel contract is:

```text
activation:       FP16 contiguous [M, K]
qweight:          int32 contiguous [(K / 32) * 3, N]
scales:           FP16 contiguous [K / 128, N]
centered weight:  M=1 uses float(code - 4) * float(scale); M=2..16 rounds the product to FP16 for WMMA
output:           FP16 [M, N], with optional FP16 bias fused into the final write
device:           runtime-probed CUDA compute capability >= 8.0
shape:            K % 128 == 0, N % 64 == 0
production M:     1 through 16
```

The promoted WMMA geometry for M=2..16 is `BlockM=16`, `BlockN=64`, `TileK=32`, 128 threads, and four warps. Each warp
owns one 16x16 N tile and uses `wmma::mma_sync` with FP16 A/B fragments and FP32 accumulators. For each K tile,
threads 0-63 load one output column's three packed words, extract all 32 codes including the two word-straddling
codes, center them at four, scale them, and stage the resulting FP16 B tile in shared memory. The four WMMA warps
then consume the shared tile. This retains the core Marlin ideas—fused packed decode, shared staging, Tensor Core
compute, shape-specific scheduling, and no hot-path materialization—while implementing the actual continuous 3-bit
format instead of pretending it is Marlin's 4-bit format.

Small M needs more N/K parallel work than the 64 N tiles alone provide. The kernel therefore supports power-of-two
split-K values from 1 through 32. Each split writes an FP32 partial to `[split, M, N]`; a second CUDA kernel performs
the deterministic FP32 sum, adds bias, and converts to FP16. Production keeps at least 128 K values in every slice.
The M>16 implementation was retained as a correctness/debug path but is not selected: a no-split `BlockM=32` WMMA
launch was about `0.397 ms` at M=128, far slower than expanded Marlin.

Production dispatch is intentionally hybrid:

1. FP16, CC >= 8.0, K/N aligned, M=1, native JIT available: true 3-bit CUDA GEMV.
2. The same contract at M=2..16: true 3-bit CUDA WMMA.
3. FP16 and an initialized exact-value Marlin cache: expanded signed Marlin, including M>16.
4. BF16, N divisible by 32 but not 64, disabled/unavailable native JIT, or another native ineligibility: fused Triton.
5. Training and contracts outside the 3-bit specialization: the established backend fallback.

All native derived tensors are non-persistent. `GPTQMODEL_TRILIN_NATIVE=0` disables both native preparations. Runtime
device guards and the current PyTorch CUDA stream come from the tensors; no kernel decision assumes a fixed device
index.

The final sm_80 JIT cache used for profiling is:

```text
/root/.cache/gptqmodel/torch_extensions/trilin/724b2b7672378cb6
```

Its exact NVCC flags from `build.ninja` were:

```text
-gencode=arch=compute_80,code=compute_80
-gencode=arch=compute_80,code=sm_80
-static-global-template-stub=false
-O3 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=1
--threads 8 --optimize=3 -lineinfo
-Xfatbin -compress-all -diag-suppress=179,39,177 --use_fast_math
```

The first clean compile took 19-23 seconds depending on the source revision. A cached load is sub-second. This is in
addition to the one-time Marlin JIT cost when the large-M fallback cache is enabled.

### 2026-07-22 — Native JIT compile failures

Status: two implementation failures diagnosed and fixed before benchmarking.

The first native build used `torch::Tensor` while intentionally including only the lightweight ATen/library headers.
That alias was unavailable, so NVCC failed during host compilation. The binding now uses `at::Tensor`, avoiding the
large `torch/extension.h` include and its compile-time cost.

The second build then failed because `at::empty` was not declared by the selected headers. Adding the narrow
`<ATen/ops/empty.h>` include fixed the binding without broadening to the heavyweight extension header. Both failures
were compile-time only; no invalid binary was launched or timed.

### 2026-07-22 — Initial native CUDA correctness and split-K result

Status: successful; enough evidence to integrate the native small-M route.

The first `TileK=32` implementation passed deterministic packed-word boundaries, multiple K/N/M boundaries, bias,
and current-stream checks. At `K=N=4096`, a dual-GPU split sweep found approximately `0.0481 ms` for M=1 with
split 8 and `0.0573 ms` for M=16 with split 8 on both physical devices. The maximum errors remained `0.0625` and
`0.125`, respectively, matching the Triton and expanded-Marlin paths. This beat the then-current expanded Marlin
small-M result of roughly `0.076-0.080 ms` while reading the original 3-bit qweight.

Split 16 improved the initial M=1/M=16 controls to approximately `0.0502/0.055-0.056 ms`, and split 32 improved M=1
again to about `0.048-0.049 ms`. The first selector accidentally returned split 16 for M=128, which the native binding
correctly rejected because split-K is supported only for M<=16. The selector was fixed to return one for M>16 and a
regression assertion was added. No M=128 benchmark result from that failed invocation was used.

### 2026-07-22 — Group-sized `TileK=128` staging experiment

Status: initial illegal shared-memory access fixed; corrected implementation slower and reverted.

The hypothesis was that staging an entire 128-value quantization group would reuse each scale more directly and reduce
barriers. The first prototype accidentally iterated 128 local K positions for each of four 32-value chunks, writing
past the shared B allocation. The asynchronous fault later surfaced as a cuBLAS allocation error, which was a
misleading symptom. `compute-sanitizer` located the invalid shared write at the dequantization line. Restricting the
per-chunk loop to 32 values removed the fault; the corrected kernel completed sanitizer with zero errors and passed
the numerical checks.

The valid `TileK=128` result was still slower: about `0.0502 ms` at M=1, `0.0604 ms` at M=16, and `0.426 ms` at
M=128, versus approximately `0.0481`, `0.0573`, and `0.397 ms` for `TileK=32`. The larger staging window was reverted.
This entry is important because the deferred cuBLAS error was not an allocator problem; it was a prior asynchronous
kernel fault.

### 2026-07-22 — Initial hybrid production integration

Status: successful and retained.

Both 3-bit QuantLinear implementations now prepare the native state in `post_init`. GPTQ passes its serialized
K-packed tensor directly. AWQ first creates its non-persistent K-packed 3-bit runtime tensor, then shares that tensor
between native Trilin and the Marlin expansion. Forward dispatch fuses bias in both native paths. Tests monkeypatch the
Triton or alternate native function to raise, proving that M=1 actually selects true 3-bit CUDA and M=17 selects
expanded Marlin. State-dict tests prove that all runtime caches remain non-persistent.

The first complete hybrid artifacts were:

```text
benchmark_artifacts/triton_3bit/native_hybrid_full_gpu0.json
benchmark_artifacts/triton_3bit/native_hybrid_full_gpu1.json
```

Before the later vector-staging optimization, representative production-hybrid medians were:

```text
+------+---------------------+---------------------+---------------------+---------------------+
| M    | GPU0 GPTQ ms        | GPU1 GPTQ ms        | GPU0 AWQ ms         | GPU1 AWQ ms         |
+------+---------------------+---------------------+---------------------+---------------------+
| 1    |              0.0666 |              0.0686 |              0.0686 |              0.0696 |
| 16   |              0.0696 |              0.0686 |              0.0686 |              0.0696 |
| 128  |              0.0891 |              0.0870 |              0.0870 |              0.0891 |
+------+---------------------+---------------------+---------------------+---------------------+
```

M=1/16 use true 3-bit CUDA; M=128 uses expanded Marlin. Triton medians in the same run were approximately
`0.117-0.139 ms` at small M and `0.120-0.128 ms` at M=128. The native paths retained the established seeded maximum
errors of `0.0625`, `0.125`, and `0.0`.

### 2026-07-22 — First native Nsight attribution

Status: successful; identified the separate reduction launch as the next latency target.

Artifacts:

```text
benchmark_artifacts/triton_3bit/ncu_native_trilin_gptq_m1_gpu0.ncu-rep
benchmark_artifacts/triton_3bit/nsys_native_trilin_gptq_m1_gpu1.nsys-rep
```

Nsight Systems recorded exactly two kernels per operation. Across five operations, the main kernel averaged
`37.8556 us` with a `37.760 us` median; the reduction averaged `5.3184 us` with a `5.376 us` median. Ten
`cudaLaunchKernel` calls therefore correspond exactly to the five two-kernel operations. There is no hidden dense
dequantization or repack in the marked range.

Nsight Compute full replay measured the main kernel at `40.16 us` with grid `(64,1,32)`, block 128, 40 registers per
thread, 5.12 KiB dynamic shared memory, 1.38 waves/SM, `38.66%` compute throughput, `71.17%` memory throughput,
`79.08%` L1/TEX throughput, and no spills. The reduction measured `8.80 us` under replay with only 16 blocks. These
replay durations are diagnostic and are not substituted for CUDA-event or Nsight Systems timing.

### 2026-07-22 — Cooperative single-launch reduction experiment

Status: correctness success, latency failure, reverted.

The split-16 grid has 1024 blocks and fits within the device-wide cooperative residency limit reported by the CUDA
occupancy API. A prototype launched the main kernel cooperatively, used `grid.sync()` after workspace writes, and let
the split-zero CTAs perform the final reduction and bias write. This removed the second kernel launch without relying
on a fixed SM count and preserved the existing error exactly.

Matched dual-GPU medians showed that the grid-wide synchronization cost exceeded the saved reduction launch:

```text
+------+---------+------------------+------------------+------------------------------+
| M    | split   | GPU0 median ms   | GPU1 median ms   | implementation               |
+------+---------+------------------+------------------+------------------------------+
| 1    | 8       |           0.0563 |           0.0614 | cooperative                  |
| 1    | 16      |           0.0696 |           0.0625 | cooperative                  |
| 1    | 32      |           0.0492 |           0.0502 | ordinary main + reduction    |
| 16   | 8       |           0.0799 |           0.0819 | cooperative                  |
| 16   | 16      |           0.0860 |           0.0809 | cooperative                  |
| 16   | 32      |           0.0543 |           0.0502 | ordinary main + reduction    |
+------+---------+------------------+------------------+------------------------------+
```

The experiment is rejected even though it reduces launch count. A future one-launch design needs a different
dataflow—such as a safe last-CTA protocol or an in-block K reduction—not a device-wide barrier.

### 2026-07-22 — Two-thread-per-column dequantization experiment

Status: correctness success, performance failure, reverted.

The original decoder uses 64 threads, one per N column, and extracts 32 values serially from three packed words. A
prototype used all 128 threads by assigning two threads to each N column and 16 K values to each thread. This halved
the source-level decode loop but duplicated the three global packed-word loads and scale load.

The duplication was decisively slower. GPU0 medians moved to approximately `0.0604 ms` at M=1 and
`0.0748-0.0758 ms` at M=16; GPU1 moved to `0.0655-0.0666 ms` and `0.0727-0.0737 ms`. The same maximum errors proved
that this was a valid performance rejection, not a correctness failure. The one-thread-per-column decoder was restored.

### 2026-07-22 — Vector activation staging and native FP16 decode arithmetic

Status: successful on both GPUs and retained.

The main kernel originally staged `BlockM * 32` activation halves with scalar loads/stores. At M=1, 480 of the 512
staged values are padded zeros, so scalar staging spends instructions on data that does not exist. The retained change
uses aligned `int4` loads/stores: four 16-byte vectors copy each valid 32-half row, while one 16-byte zero vector fills
each padded segment. Alignment is guaranteed by PyTorch allocation, K divisible by 128, 32-aligned K tiles, and the
explicitly 16-byte-aligned shared buffer.

The first matched vector-only run produced approximately `0.0481 ms` at M=1 on both cards. At M=16 it produced
`0.0512 ms` on GPU0 and `0.0471 ms` on GPU1 with split 16, compared with the prior roughly `0.055-0.056 ms` control.
The optimization preserves output bits under the test vectors.

The retained decoder also converts the centered integer directly to half and uses `__hmul` with the FP16 scale,
instead of converting scale/code to FP32, multiplying, and rounding back to FP16. Since the staged WMMA B operand is
FP16 in both cases and integers -4 through 3 are exactly representable, the seeded output errors are unchanged.
Six alternating-order 1000-sample M=1 repetitions put split-32 medians mostly at `0.044-0.047 ms` on GPU0 and
`0.044-0.048 ms` on GPU1 after these changes.

### 2026-07-22 — Final small-M split selector

Status: split 32 selected for every M from 1 through 16 at K=4096.

A matched sweep covered M=`2,4,5,8,12,16` and split=`8,16,32`, using 96 clock-warmup GEMMs, 100 operation warmups,
and 500 samples per point on both GPUs. Split 32 was fastest at nearly every point and never showed a durable median
loss. A separate six-repetition, alternating-order, 1000-sample M=16 check found split 16 and 32 mostly tied on GPU0,
while split 32 was typically 1-3 us faster on GPU1. Production therefore begins at split 32 for all M<=16 and halves
it only until K is divisible into 32-value tiles with at least 128 values per slice. M>16 always returns split one.

This is an explicit memory-for-latency trade. The FP32 workspace is `split * M * N * 4` bytes; at M=16, N=4096,
split=32 it is 8 MiB, versus 4 MiB for split 16. The tensor is temporary and PyTorch's allocator can reuse it, but
the cost must be revisited for very large N or highly concurrent streams.

### 2026-07-22 — Final native-hybrid benchmark

Status: successful concurrent 500-sample run on both requested GPUs; accepted.

Commands used both serialized layouts, M=`1,16,128`, K=N=4096, FP16, 128 large-GEMM clock warmups before every method,
100 operation warmups, and 500 CUDA-event samples. Persistent artifacts:

```text
benchmark_artifacts/triton_3bit/native_hybrid_vectorized_gpu0.json
benchmark_artifacts/triton_3bit/native_hybrid_vectorized_gpu1.json
```

Production medians and throughput, compared with the fused Triton fallback in the same run:

```text
+------+--------+-----+-----------+-----------+---------+--------------------+
| GPU  | layout | M   | Triton ms | hybrid ms | speedup | hybrid TFLOP/s     |
+------+--------+-----+-----------+-----------+---------+--------------------+
| 0    | GPTQ   |   1 |    0.1188 |    0.0655 |   1.81x |              0.512 |
| 0    | GPTQ   |  16 |    0.1229 |    0.0676 |   1.82x |              7.944 |
| 0    | GPTQ   | 128 |    0.1413 |    0.0845 |   1.67x |             50.840 |
| 1    | GPTQ   |   1 |    0.1229 |    0.0707 |   1.74x |              0.475 |
| 1    | GPTQ   |  16 |    0.1239 |    0.0696 |   1.78x |              7.710 |
| 1    | GPTQ   | 128 |    0.1249 |    0.0993 |   1.26x |             43.240 |
| 0    | AWQ    |   1 |    0.1178 |    0.0676 |   1.74x |              0.496 |
| 0    | AWQ    |  16 |    0.1219 |    0.0614 |   1.99x |              8.738 |
| 0    | AWQ    | 128 |    0.1219 |    0.0891 |   1.37x |             48.210 |
| 1    | AWQ    |   1 |    0.1239 |    0.0696 |   1.78x |              0.482 |
| 1    | AWQ    |  16 |    0.1219 |    0.0696 |   1.75x |              7.710 |
| 1    | AWQ    | 128 |    0.1249 |    0.0911 |   1.37x |             47.127 |
+------+--------+-----+-----------+-----------+---------+--------------------+
```

True 3-bit CUDA serves M=1/16; expanded Marlin serves M=128. The benchmark deliberately also times the unselected
native WMMA M=128 diagnostic, which remains around `0.296-0.303 ms`; production does not route to it. Numerical
guards remain maximum absolute error `0.0625` at M=1, `0.125` at M=16, and `0.0` for the seeded M=128 case.

The full sweep runs many clock warmups and methods back-to-back, so absolute small-M medians are higher than the
isolated split experiments. Claims use matched methods from the same run; the JSON artifacts retain mean, standard
deviation, minimum, p95, maximum, and every baseline.

### 2026-07-22 — Final native Nsight proof

Status: successful concurrent NCU on physical GPU0 and NSYS on physical GPU1.

Artifacts:

```text
benchmark_artifacts/triton_3bit/ncu_native_trilin_vectorized_gptq_m1_gpu0.ncu-rep
benchmark_artifacts/triton_3bit/nsys_native_trilin_vectorized_gptq_m1_gpu1.nsys-rep
benchmark_artifacts/triton_3bit/nsys_native_trilin_vectorized_gptq_m1_gpu1.sqlite
```

Nsight Systems recorded five operations and exactly ten kernels. The final main-kernel average/median is
`35.910/35.744 us`, down from the pre-vector `37.856/37.760 us`. The reduction average/median is `5.318/5.280 us`,
essentially unchanged. The main kernel is 87.1% and reduction 12.9% of device kernel time in the marked range.

Nsight Compute full replay comparison:

```text
+----------------------------+-------------------+-------------------+
| metric                     | before staging    | final staging     |
+----------------------------+-------------------+-------------------+
| main duration              |          40.16 us |          39.33 us |
| registers/thread           |                40 |                40 |
| achieved occupancy         |            58.14% |            59.27% |
| compute throughput         |            38.66% |            30.58% |
| memory throughput          |            71.17% |            71.74% |
| L1/TEX throughput          |            79.08% |            82.61% |
| no eligible warp cycles    |            56.53% |            64.73% |
| reduction replay duration  |           8.80 us |           9.02 us |
+----------------------------+-------------------+-------------------+
```

The final kernel is faster even though compute SOL and eligible-warp metrics look worse, consistent with removing
scalar staging/conversion instructions rather than increasing occupancy. This is direct evidence for the user's
latency/throughput objective: occupancy is useful diagnostic metadata, not the acceptance criterion. The remaining
separate reduction is still roughly 13% of kernel time, but the cooperative fusion experiment proved that launch
removal alone is insufficient when it introduces a global barrier.

### 2026-07-22 — Final sanitizer and regression verification

Status: successful.

Final `compute-sanitizer --tool memcheck` ran the warmed M=1, K=N=4096 native path on physical GPU1 and reported:

```text
ERROR SUMMARY: 0 errors
```

The complete focused suite then ran concurrently and independently on physical GPUs 0 and 1:

```text
GPU0: 40 passed, 0 skipped, 11.17 s
GPU1: 40 passed, 0 skipped, 11.28 s
```

Coverage includes real GPTQ packing, both layouts, FP16/BF16 Triton fallback, exact 3-to-4-bit conversion, native
split selection, native current-stream execution, GPTQ/AWQ module dispatch at M=1 and M=17, non-persistent caches,
bias, save/reload, malformed zero points, and natural group-index validation.

An initial aggregate extension-loader run was aborted because the fake registry did not include the newly registered
`trilin` extension, causing the test to build real Trilin and then unrelated optional EoRA/Grasshopper extensions on
this CUDA host. The deterministic fix added a fake Trilin extension and a `trilin-3bit` alias test. With CUDA hidden
so unrelated host-conditional extensions remain out of scope, the extension/JIT utility suites passed `41` tests.
The neighboring AWQ FP32-accumulation suite passed `2` tests on GPU0.

### 2026-07-22 — FP16/BF16 quality gates against eager Torch 3-bit reference

Status: successful on both requested GPUs and retained.

An accuracy audit found that the raw fused kernel already had FP16 and BF16 dense-reference coverage for both
serialized layouts, but the production `QuantLinear` dispatch test used only FP16 and a hand-expanded dense weight.
The retained test adds an independent `TorchLinear` reference for the exact v0 contract: 3 bits, group size 128,
`desc_act=False`, `sym=True`, natural group indices, real packed zero points, and bias. The reference receives canonical
GPTQ K-packed codes and explicitly disables both `torch.compile` optimization and Triton dequantization, forcing the
repository's eager Torch 3-bit bit-shift/unpack, dequantization, and `torch.matmul` path. The candidate receives the
real GPTQ or AWQ serialized layout and uses normal production post-init and dispatch.

The matrix covers GPTQ and AWQ, FP16 and BF16, and M=`1,16,33` at K=512 and N=256. This intentionally crosses native
true-3-bit FP16 small-M dispatch, expanded-Marlin FP16 dispatch, and BF16 fused-Triton fallback. Every case checks
shape, output dtype, finite values, maximum and mean absolute error, relative RMSE, and cosine similarity.

The measured worst case on each physical GPU was identical:

```text
+-------+---------+----------+---------------+-------------------+------------------+
| dtype | max abs | mean abs | relative RMSE | minimum cosine    | accepted gates   |
+-------+---------+----------+---------------+-------------------+------------------+
| FP16  | 0.06250 | 0.014552 |      0.001050 |       0.999999404 | <=0.125 max      |
|       |         |          |               |                   | <=0.020 mean     |
|       |         |          |               |                   | <=0.0015 relRMSE |
|       |         |          |               |                   | >=0.999995 cos   |
| BF16  | 0.12500 | 0.000488 |      0.000259 |       0.999999881 | <=0.500 max      |
|       |         |          |               |                   | <=0.010 mean     |
|       |         |          |               |                   | <=0.0020 relRMSE |
|       |         |          |               |                   | >=0.999990 cos   |
+-------+---------+----------+---------------+-------------------+------------------+
```

After fixing the gates from these observations, the complete focused suite ran concurrently and independently:

```text
GPU0: 52 passed, 0 skipped, 21.50 s
GPU1: 52 passed, 0 skipped, 18.79 s
```

`ruff check tests/kernels/test_triton_3bit.py` and `git diff --check` also passed.

### 2026-07-22 — Interpreting FP16 versus BF16 Torch-reference error

Status: reported metric confirmed; it is backend disagreement, not a cross-dtype accuracy ranking.

The FP16 worst-case mean absolute error (`0.014552` at M=33) is much larger than the BF16 value (`0.000344`) when
each production output is compared with the eager Torch output of the same dtype. This initially looks inverted, but
the two production cases take different paths and their output formats have different rounding granularity:

- M=33 FP16 dispatches to the expanded-Marlin path. Only 29.14% of its output elements are bit-identical to eager
  Torch; the remaining fine-grained accumulation-order differences remain visible in FP16.
- M=33 BF16 dispatches to the fused Triton path. Its output is exactly the direct fused-Triton output and 98.05% of
  elements equal eager Torch after BF16 rounding. A few one-ULP disagreements produce the larger `0.125` maximum,
  but their sparsity keeps the mean low.
- Direct FP16 fused Triton matches eager Torch bit-for-bit for this seeded M=33 case. The `0.014552` production delta
  is therefore attributable to Marlin's accumulator/reduction ordering, not 3-bit packing or dequantization.

An FP32 dense oracle using the same quantized codes, FP16 scales, dtype-specific input, and bias confirms the expected
accuracy ordering:

```text
+-----+-------+--------------------+---------------+
| M   | dtype | mean abs vs FP32   | relative RMSE |
+-----+-------+--------------------+---------------+
|   1 | FP16  |           0.003372 |      0.000222 |
|   1 | BF16  |           0.030042 |      0.002071 |
|  16 | FP16  |           0.002344 |      0.000206 |
|  16 | BF16  |           0.024909 |      0.002374 |
|  33 | FP16  |           0.015503 |      0.001054 |
|  33 | BF16  |           0.025811 |      0.002380 |
+-----+-------+--------------------+---------------+
```

FP16 remains more accurate than BF16 against the FP32 oracle at every tested M. The existing Torch-reference gates
are still useful for catching backend drift, but their values must not be compared across dtypes as model-accuracy
scores. The M=33 Marlin-versus-FP32 delta is a valid future numerical-quality tuning target even though it remains
small in relative terms and within the accepted gate.

### 2026-07-22 — Last-CTA single-launch reduction mega-kernel

Status: correctness success, stable latency/throughput failure, rejected and reverted.

The existing small-M dependency chain is:

```text
input/qweight/scales -> split-K WMMA CTAs -> FP32 [split,M,N] workspace
                     -> kernel boundary -> FP32 reduction + bias -> FP16 output
```

The candidate kept the same split CTAs and accumulation order, but every writer issued a device fence and then one
thread atomically incremented a per-output-tile completion counter. The last CTA for each tile read all published
workspace slices, reduced splits in the original 0..31 order, added bias, and wrote FP16 output. This used no
cooperative launch, no grid barrier, and no fixed SM count. Counters occupied the tail of the existing workspace
allocation and were reset with a small stream-ordered `cudaMemsetAsync` before the launch.

Exact-bit A/B comparison passed on both physical GPUs for split 1, 2, and 32; M=1/16; K=128/256/4096; N=64/256/4096;
100 repeated small cases, 20 repeated full cases, bias, and non-default streams. An initial sequential 1000-sample
run misleadingly favored the candidate by roughly 3-8% in median, so it was not accepted without alternating-order
repetition.

Six alternating-order, clock-warmed rounds of 1000 samples per method gave the stable result:

```text
+------+-----+-----------+-----------+-----------+-----------+----------+----------+
| GPU  | M   | base p50  | fused p50 | base mean | fused mean| base p95 | fused p95|
+------+-----+-----------+-----------+-----------+-----------+----------+----------+
| 0    |   1 |  0.065536 |  0.066560 |  0.068311 |  0.070464 | 0.076800 | 0.078848 |
| 0    |  16 |  0.063488 |  0.064512 |  0.065149 |  0.067961 | 0.072704 | 0.074752 |
| 1    |   1 |  0.062464 |  0.063488 |  0.064962 |  0.069018 | 0.070656 | 0.071680 |
| 1    |  16 |  0.064512 |  0.065536 |  0.075351 |  0.075113 | 0.073728 | 0.077824 |
+------+-----+-----------+-----------+-----------+-----------+----------+----------+
```

Units are milliseconds. The candidate is 1.6% slower in every aggregate median and regresses p95 by 1.4-5.6%.
GPU1 M=16 mean is nominally 0.3% better only because the control has a larger extreme outlier; median and p95 both
reject the candidate.

Bounded 200-operation Nsight Systems captures explain the loss:

```text
+------+----------------+----------+----------+-----------+-----------+-------------+
| GPU  | path           | main med | reduce   | memset med| device sum| GPU op/call |
+------+----------------+----------+----------+-----------+-----------+-------------+
| 0    | two-launch     | 35.904 us| 5.312 us |         - | 41.216 us |           2 |
| 0    | last-CTA       | 41.712 us|        - |  1.312 us | 43.024 us |           2 |
| 1    | two-launch     | 35.872 us| 5.312 us |         - | 41.184 us |           2 |
| 1    | last-CTA       | 41.664 us|        - |  1.312 us | 42.976 us |           2 |
+------+----------------+----------+----------+-----------+-----------+-------------+
```

The last-CTA kernel's fences, atomics, and less-parallel tile-local reduction add about 5.8 us to the main kernel,
slightly more than the removed 5.3 us reduction. Counter initialization then adds another 1.3 us device memory op.
Nsight's projected NVTX span incorrectly looks favorable because tracing overhead applies to 400 baseline kernel
launch API calls versus 200 candidate launch calls plus 200 memsets. The uninstrumented CUDA-event distributions are
the acceptance evidence, consistent with the profiling workflow's warning not to substitute trace timing for final
performance.

Artifacts:

```text
benchmark_artifacts/triton_3bit/last_cta_ab_gpu0.json
benchmark_artifacts/triton_3bit/last_cta_ab_gpu1.json
benchmark_artifacts/triton_3bit/nsys_lastcta_baseline_gpu0.nsys-rep
benchmark_artifacts/triton_3bit/nsys_lastcta_candidate_gpu0.nsys-rep
benchmark_artifacts/triton_3bit/nsys_lastcta_baseline_gpu1.nsys-rep
benchmark_artifacts/triton_3bit/nsys_lastcta_candidate_gpu1.nsys-rep
```

This rejects another launch-count-only tactic. A future fusion attempt must avoid both per-call counter initialization
and serializing the highly parallel reduction onto the final producer CTA.

### 2026-07-22 — Per-group scale register-cache experiment

Status: correctness success, inconsistent device/shape performance, rejected and reverted.

The retained `TileK=32` kernel reloads one FP16 scale per output column on every tile even though group size 128 keeps
that scale constant for four consecutive tiles. A compile-time A/B specialization cached the current group and scale
in registers, loading once at the beginning of each split and only reloading when `tile_k / 128` changed. It did not
change weight decoding, WMMA geometry, split selection, reduction order, or outputs.

Exact-bit comparison passed at K=128, K=4096, and K=640. The last case is important because a split can begin inside
a quantization group and later cross a group boundary; it proves that the cache-update path is not relying on the
4096x4096 benchmark's convenient 128-value slices.

Six alternating-order, clock-warmed rounds of 1000 samples per method produced:

```text
+------+-----+-----------+-----------+-----------+-----------+----------+----------+
| GPU  | M   | base p50  | cache p50 | base mean | cache mean| base p95 | cache p95|
+------+-----+-----------+-----------+-----------+-----------+----------+----------+
| 0    |   1 |  0.063488 |  0.062464 |  0.068330 |  0.064283 | 0.071680 | 0.070656 |
| 0    |  16 |  0.064512 |  0.064512 |  0.071113 |  0.071914 | 0.073728 | 0.073728 |
| 1    |   1 |  0.063488 |  0.064512 |  0.071026 |  0.069384 | 0.072704 | 0.071680 |
| 1    |  16 |  0.065536 |  0.066560 |  0.069483 |  0.070676 | 0.074752 | 0.074752 |
+------+-----+-----------+-----------+-----------+-----------+----------+----------+
```

Units are milliseconds. M=1 improves 1.6% in median on GPU0 but regresses 1.6% on GPU1. M=16 is tied on GPU0 and
1.6% slower on GPU1, with no p95 improvement. The saved scale traffic is only a few cache-friendly half loads per
CTA, while the cached value/group and update branch extend live state. Without a repeatable cross-device gain, the
tactic is rejected before heavy Nsight Compute replay.

### 2026-07-22 — Dedicated activation-staging warp roles

Status: exact correctness, sub-microsecond raw movement, no stable operator win; rejected and reverted.

For `BlockM=16`, activation staging comprises exactly 64 aligned `int4` vectors. The retained mapping assigns those
loads to threads 0-63, which then also perform the serial packed-weight decode while threads 64-127 wait at the same
barrier. A compile-time candidate instead assigned the A vectors to threads 64-127 and kept B decode on threads 0-63,
allowing independent warps to issue activation loads and decode work before the existing CTA barrier. It changed no
loads, arithmetic, shared layout, WMMA ownership, or reduction order and matched the baseline bit-for-bit at all tested
shapes.

Six alternating-order, clock-warmed 1000-sample rounds gave mixed results. Aggregate M=1 median was tied on both
GPU0 and GPU1; aggregate M=16 was one 1.024-us event bucket slower on GPU0 and tied on GPU1. Mean/p95 sometimes
favored the candidate on GPU1, but GPU0 was flat or worse, so normal timing did not establish a cross-device win.

Bounded 200-operation Nsight Systems captures isolated the main kernel:

```text
+------+-----------+----------------+----------------+--------------+--------------+
| GPU  | variant   | main avg us    | main median us | main std us  | reduce med us|
+------+-----------+----------------+----------------+--------------+--------------+
| 0    | baseline  |         36.081 |         35.904 |        0.454 |        5.152 |
| 0    | dedicated |         35.938 |         35.904 |        0.163 |        5.216 |
| 1    | baseline  |         36.261 |         36.048 |        0.497 |        5.152 |
| 1    | dedicated |         36.010 |         36.000 |        0.160 |        5.152 |
+------+-----------+----------------+----------------+--------------+--------------+
```

The candidate lowers raw main-kernel average by only 0.14-0.25 us; GPU0 median is identical and GPU1 median moves
0.048 us, below a defensible end-to-end acceptance threshold. Traced total spans also disagreed by device, reinforcing
that this is profiler-resolution movement rather than a retained latency/throughput improvement. Full NCU replay was
not run for a candidate that failed normal timing.

The first temporary profiler-script invocation failed before CUDA initialization because `/tmp` removed the repository
from Python's import path. Adding the explicit repository `PYTHONPATH` fixed collection; no result from the failed
invocation was used.

Artifacts:

```text
benchmark_artifacts/triton_3bit/nsys_dedicated_baseline_gpu0.nsys-rep
benchmark_artifacts/triton_3bit/nsys_dedicated_candidate_gpu0.nsys-rep
benchmark_artifacts/triton_3bit/nsys_dedicated_baseline_gpu1.nsys-rep
benchmark_artifacts/triton_3bit/nsys_dedicated_candidate_gpu1.nsys-rep
```

### 2026-07-22 — Four-output vectorized split-K reduction

Status: exact correctness, no stable operator improvement; rejected and reverted.

The separate FP32 split reducer is only about 5.15 us, but it still contributes roughly 12% of the two-kernel device
time. This candidate gave each reduction thread four contiguous output values, loaded every split slice with aligned
`float4` operations, and stored two packed `half2` pairs. Because N is contractually divisible by 64, each vector is
contained within one row and every split-workspace offset remains 16-byte aligned. The producer kernel, split count,
summation order, bias placement, and FP16 rounding were unchanged.

Both physical GPUs passed bit-identical A/B checks for split 1/2/4/32, M=1/3/16, K=128/256/640/4096,
N=64/256/4096, bias and no bias, 100 repeated launches, and a non-default stream. The candidate also passed the
existing FP16 Torch-reference tolerance. The test ran on the PCI-ordered GPU 0 and GPU 1 concurrently after a single
successful sm_80 JIT build with CUDA 13.0.

Six alternating-order, clock-warmed rounds of 1000 samples per method produced:

```text
+------+-----+-----------+-----------+-----------+-----------+----------+----------+
| GPU  | M   | base p50  | vec4 p50  | base mean | vec4 mean | base p95 | vec4 p95 |
+------+-----+-----------+-----------+-----------+-----------+----------+----------+
| 0    |   1 |  0.062464 |  0.062464 |  0.065123 |  0.065262 | 0.072704 | 0.072704 |
| 0    |  16 |  0.063488 |  0.063488 |  0.071631 |  0.069332 | 0.080896 | 0.081920 |
| 1    |   1 |  0.067584 |  0.068608 |  0.069266 |  0.070457 | 0.076800 | 0.074752 |
| 1    |  16 |  0.068608 |  0.069632 |  0.069496 |  0.070216 | 0.077824 | 0.077824 |
+------+-----+-----------+-----------+-----------+-----------+----------+----------+
```

Units are milliseconds. GPU0 is tied at the median and M=16 p95 is one 1.024-us timing bucket worse. GPU1 regresses
1.5% at the median for both shapes. The GPU0 M=16 mean is not acceptance evidence because both methods contain rare
multi-millisecond external outliers; median and p95 do not support it. Reducing CTA count and scalar instruction count
inside a launch-dominated reducer does not improve end-to-end latency. Nsight Compute replay was skipped because the
candidate failed the normal timing gate.

Artifacts:

```text
benchmark_artifacts/triton_3bit/vector4_correctness_gpu0.json
benchmark_artifacts/triton_3bit/vector4_correctness_gpu1.json
benchmark_artifacts/triton_3bit/vector4_ab_gpu0.json
benchmark_artifacts/triton_3bit/vector4_ab_gpu1.json
```

### 2026-07-22 — 128-column producer CTA

Status: exact correctness, conflicting operator timing, slower isolated producer; rejected and reverted.

The retained producer uses 64 output columns, 128 threads, and four WMMA warps per CTA. The candidate doubled the
column tile to 128 with 256 threads and eight WMMA warps. At N=4096 this halves producer CTA count from 2048 to 1024
for split-K 32, halves duplicate activation staging, and leaves total decoded weights and total WMMA work unchanged.
It used the retained 64-column kernel as a fallback whenever N was not divisible by 128 or M exceeded 16.

Both physical GPUs matched the baseline bit-for-bit for split 1/2/4/32, M=1/3/16, K=128/256/640/4096,
N=64/128/256/4096, bias/no-bias, 100 repeated launches, and a non-default stream. Six concurrent, alternating-order,
clock-warmed rounds of 1000 samples initially looked promising but disagreed by device:

```text
+------+-----+-----------+-----------+-----------+-----------+----------+----------+
| GPU  | M   | base p50  | wide p50  | base mean | wide mean | base p95 | wide p95 |
+------+-----+-----------+-----------+-----------+-----------+----------+----------+
| 0    |   1 |  0.070656 |  0.068608 |  0.071793 |  0.069616 | 0.078848 | 0.076800 |
| 0    |  16 |  0.070656 |  0.068608 |  0.072178 |  0.070635 | 0.077824 | 0.078848 |
| 1    |   1 |  0.065536 |  0.065536 |  0.069360 |  0.068660 | 0.073728 | 0.074752 |
| 1    |  16 |  0.068608 |  0.068608 |  0.072401 |  0.071819 | 0.076800 | 0.077824 |
+------+-----+-----------+-----------+-----------+-----------+----------+----------+
```

Units are milliseconds. GPU0 median improves 2.9%, but GPU1 ties and candidate p95 is one 1.024-us event bucket worse
for both M values. Bounded 200-operation Nsight Systems captures resolve the contradiction against retention:

```text
+------+-----------+----------------+----------------+----------------+----------------+
| GPU  | variant   | main avg us    | main median us | reduce avg us  | reduce med us  |
+------+-----------+----------------+----------------+----------------+----------------+
| 0    | N64       |         38.634 |         38.784 |          6.730 |          6.720 |
| 0    | N128      |         39.990 |         40.000 |          6.711 |          6.720 |
| 1    | N64       |         38.516 |         38.496 |          6.685 |          6.688 |
| 1    | N128      |         39.961 |         39.936 |          6.670 |          6.656 |
+------+-----------+----------------+----------------+----------------+----------------+
```

The isolated wide producer is 1.2-1.4 us slower on both cards. Halving CTA count does not compensate for the larger
CTA's scheduling/resource costs. The GPU0 end-to-end result is therefore not a defensible kernel improvement, and the
candidate is rejected without Nsight Compute replay.

Artifacts:

```text
benchmark_artifacts/triton_3bit/wide_n_correctness_gpu0.json
benchmark_artifacts/triton_3bit/wide_n_correctness_gpu1.json
benchmark_artifacts/triton_3bit/wide_n_ab_gpu0.json
benchmark_artifacts/triton_3bit/wide_n_ab_gpu1.json
benchmark_artifacts/triton_3bit/nsys_wide_baseline_gpu0.nsys-rep
benchmark_artifacts/triton_3bit/nsys_wide_candidate_gpu0.nsys-rep
benchmark_artifacts/triton_3bit/nsys_wide_baseline_gpu1.nsys-rep
benchmark_artifacts/triton_3bit/nsys_wide_candidate_gpu1.nsys-rep
```

### 2026-07-22 — Direct native CUDA M=1 GEMV specialization

Status: successful latency and numerical-quality improvement; retained in production.

The WMMA path pads every decode token to a 16-row Tensor Core tile. For M=1, this performs 16 times the useful
matrix work and stages mostly zero activations. A direct GEMV is a better decode mapping: one CUDA thread owns one
output column, adjacent threads read adjacent packed-weight columns, every activation load is warp-uniform, and each
thread decodes one continuous 96-bit block into 32 signed 3-bit values. The existing split-K workspace and ordered
FP32 reducer are retained, so there is no new synchronization protocol or persistent state.

The promoted M=1 producer has this contract:

```text
block:          128 threads
grid:           ceil(N / 128) by split_k
split_k:        existing selector; 32 at K=4096
weight loads:   3 contiguous uint32 words per 32 K values and output column
dequant:        float(code - 4) * float(FP16 scale)
multiply/add:   FP32 FMA with FP16 input converted to FP32
partial output: FP32 [split_k, N]
final output:   existing ordered reducer, optional bias, FP16 rounding
fallback:       unchanged WMMA/Marlin/Triton production routes
```

This specializes only M=1. M=2..16 continues to use the retained WMMA kernel, so the decode optimization does not
perturb the validated multi-token route.

#### Launch and split-K sweep

The first direct-GEMV sweep compared 64, 128, and 256 threads against the prior WMMA producer in six alternating-order
rounds with a 4096x4096 FP16 problem, clock warmup, 100 operation warmups, and 1000 CUDA-event samples per method.
All three launch sizes produced similar medians and improved both devices:

```text
+------+--------------+------------+-------------+-------------+-------------+
| GPU  | WMMA p50 ms  | T64 p50 ms | T128 p50 ms| T256 p50 ms| T128 speedup|
+------+--------------+------------+-------------+-------------+-------------+
| 0    |     0.067584 |   0.057344 |    0.057344 |    0.057344 |       1.18x |
| 1    |     0.068608 |   0.058368 |    0.058368 |    0.058368 |       1.18x |
+------+--------------+------------+-------------+-------------+-------------+
```

A separate four-round sweep tested split-K 4, 8, 16, and 32 at 64 and 128 threads. Split 4 was clearly worse. Splits
8, 16, and 32 clustered at `0.055296 ms`, except 128-thread split 32 reached `0.054272 ms` on GPU1. There was no
cross-device evidence to replace the existing split selector, so production retains split 32 for K=4096. The exact
commands used the temporary A/B harnesses with one process per physical device:

```text
CUDA_VISIBLE_DEVICES=0 python /tmp/trilin_gemv_ab.py --output \
  benchmark_artifacts/triton_3bit/gemv_launch_sweep_gpu0.json
CUDA_VISIBLE_DEVICES=1 python /tmp/trilin_gemv_ab.py --output \
  benchmark_artifacts/triton_3bit/gemv_launch_sweep_gpu1.json
CUDA_VISIBLE_DEVICES=0 python /tmp/trilin_gemv_split_sweep.py --output \
  benchmark_artifacts/triton_3bit/gemv_split_sweep_gpu0.json
CUDA_VISIBLE_DEVICES=1 python /tmp/trilin_gemv_split_sweep.py --output \
  benchmark_artifacts/triton_3bit/gemv_split_sweep_gpu1.json
```

Those device commands were run concurrently. The JSON records UUID, PCI bus ID, compute capability, SM count,
memory, PyTorch/CUDA versions, sample counts, method order, and complete latency distributions.

#### FP32 scale multiplication improves the oracle result

The first GEMV exactly reproduced the WMMA path's FP16-rounded dequantized weight. That made the performance
comparison controlled, but it also preserved an avoidable rounding step. Multiplying the signed code and stored FP16
scale in FP32 before FMA has no measurable latency cost and is much closer to the FP32 dense oracle. At M=1,
K=N=4096 with bias, both devices produced the same quality result:

```text
+----------------------+----------+--------------+---------------+
| producer arithmetic  | max abs  | mean abs     | relative RMSE |
+----------------------+----------+--------------+---------------+
| padded WMMA           | 0.062500 | 0.005329410  |   0.000248720 |
| GEMV, FP16 dequant    | 0.062500 | 0.005320650  |   0.000248505 |
| GEMV, FP32 dequant    | 0.015625 | 0.000019501  |   0.000009899 |
+----------------------+----------+--------------+---------------+
```

The oracle constructs the dense weight as `float(code - 4) * float(scale)` and performs FP32 matmul and bias before
the final FP16 conversion. The retained unit test uses the same independent formula at M=1, K=4096, N=512 and gates
maximum absolute error at `0.0625`, mean absolute error at `0.0005`, relative RMSE at `0.0001`, and cosine similarity
at `0.999999` or better.

The final general benchmark reports `native_trilin_max_abs=0.125` because that legacy field compares against its
FP16-materialized reference, not this FP32 oracle. The tighter test and table above are the relevant quality proof:
the larger legacy-reference delta comes from removing FP16 weight-product rounding and represents improved agreement
with FP32 truth, not a regression.

The controlled FP32-dequant timing run retained the performance win:

```text
+------+--------------+----------------+----------+----------+
| GPU  | WMMA p50 ms  | GEMV p50 ms    | WMMA p95| GEMV p95 |
+------+--------------+----------------+----------+----------+
| 0    |     0.066560 |       0.056320 | 0.074752 | 0.066560 |
| 1    |     0.065536 |       0.056320 | 0.074752 | 0.065536 |
+------+--------------+----------------+----------+----------+
```

Artifacts:

```text
benchmark_artifacts/triton_3bit/gemv_float_correctness_gpu0.json
benchmark_artifacts/triton_3bit/gemv_float_correctness_gpu1.json
benchmark_artifacts/triton_3bit/gemv_float_ab_gpu0.json
benchmark_artifacts/triton_3bit/gemv_float_ab_gpu1.json
```

#### Rejected GEMV scale cache

An aligned-group candidate loaded one scale before the K loop and refreshed it only at 128-value group boundaries.
It matched the retained FP32 GEMV exactly, including K=640 boundary-crossing cases, but failed the timing gate. On
GPU0, its 64-thread median regressed from `0.057344` to `0.058368 ms` and its 128-thread median regressed from
`0.056320` to `0.057344 ms`. GPU1 tied at 64 threads and had a worse 128-thread p95 (`0.068608` versus
`0.065536 ms`). The cached scale extends live state and adds a loop branch while the original loads have favorable
cache behavior. The candidate was rejected and removed.

```text
benchmark_artifacts/triton_3bit/gemv_cached_correctness_gpu0.json
benchmark_artifacts/triton_3bit/gemv_cached_correctness_gpu1.json
benchmark_artifacts/triton_3bit/gemv_cached_ab_gpu0.json
benchmark_artifacts/triton_3bit/gemv_cached_ab_gpu1.json
```

#### Nsight Systems and Nsight Compute proof

A bounded full Nsight Systems trace on GPU0 used:

```text
CUDA_VISIBLE_DEVICES=0 nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  --force-overwrite=true -o benchmark_artifacts/triton_3bit/nsys_gemv_full_gpu0 \
  python /tmp/trilin_profile_gemv.py --variant all --iterations 200 --full-trace
```

The valid raw kernel medians were:

```text
+------------------------+----------------+
| kernel                 | GPU0 median us |
+------------------------+----------------+
| padded WMMA producer   |         35.968 |
| direct GEMV, 64 threads|         14.976 |
| direct GEMV, 128 threads|        14.688 |
| split-K reducer        |          5.728 |
+------------------------+----------------+
```

The 128-thread producer removes about 59% of the old producer latency and is marginally faster than 64 threads in the
trace. A focused GPU1 Nsight Compute duration check independently measured `18.208 us` at 64 threads and `17.500 us`
at 128 threads, selecting 128 threads even though normal end-to-end medians were tied.

Full Nsight Compute section replay was then run once on each device with `--set full`, a demangled kernel-name filter,
one profiled launch after six warmups, and the `float128` harness. The reports show nearly identical behavior:

```text
+------+-------------+-------------+------------+----------+--------+---------+----------+
| GPU  | duration us | compute SOL | memory SOL | L1/TEX % | L1 hit | L2 hit  | regs/thd |
+------+-------------+-------------+------------+----------+--------+---------+----------+
| 0    |       17.09 |      48.38% |     23.38% |   30.31% | 70.58% |  12.76% |       31 |
| 1    |       16.83 |      49.71% |     24.02% |   30.05% | 70.55% |  11.28% |       31 |
+------+-------------+-------------+------------+----------+--------+---------+----------+
```

Both launches use block size 128, grid size 1024, zero spills, and 0.52 waves per SM. The dominant warp issue delay is
the L1/TEX scoreboard dependency at about 3.4 cycles per issued instruction, roughly one third of sampled stalls.
Occupancy metadata is recorded for reproducibility, but it is not the optimization objective. The next useful tactic
must reduce dependent packed-weight/scale load latency or expose more decode/FMA instruction-level parallelism while
preserving coalescing and output quality.

Artifacts:

```text
benchmark_artifacts/triton_3bit/nsys_gemv_full_gpu0.nsys-rep
benchmark_artifacts/triton_3bit/ncu_gemv_float64_gpu1.ncu-rep
benchmark_artifacts/triton_3bit/ncu_gemv_float128_gpu1.ncu-rep
benchmark_artifacts/triton_3bit/ncu_gemv_float128_full_gpu0.ncu-rep
benchmark_artifacts/triton_3bit/ncu_gemv_float128_full_gpu1.ncu-rep
```

The first profiler attempt used `cudaProfilerStart/Stop` capture-range control. Nsight Systems 2024.6.2 with CUDA
13.0 printed capture messages but produced empty traces for both devices. After moving synchronization inside the
capture range, it still produced no CUDA events. Full-trace collection recovered GPU0. The first GPU1 full-trace run
was interrupted while importing Triton and its retry left an incomplete raw stream that the importer rejected, so no
GPU1 Nsight Systems number from either failure is used. The successful independent GPU1 NCU report supplies the
second-device kernel-level proof.

#### Final production benchmark and regression matrix

The experimental entry points and compile-time comparison branches were removed before the production build. The
public `trilin::matmul` operation now dispatches M=1 directly to the 128-thread FP32-dequant GEMV. M=2..16 and all
fallback behavior are unchanged. The final benchmark command, run concurrently with physical-device isolation, was:

```text
CUDA_VISIBLE_DEVICES=<0|1> python scripts/benchmark_triton_3bit.py --layout both --dtype fp16 \
  --shape 1x4096x4096 --shape 16x4096x4096 --warmup 100 --clock-warmup-iterations 64 \
  --iterations 1000 --include-native-trilin --include-marlin-expanded \
  --output-json benchmark_artifacts/triton_3bit/final_gemv_gpu<0|1>.json
```

The production M=1 medians are:

```text
+------+--------+----------------+----------------+----------------+--------------------------+
| GPU  | layout | native GEMV ms | fused input ms | speedup fused | expanded Marlin ms       |
+------+--------+----------------+----------------+----------------+--------------------------+
| 0    | GPTQ   |       0.062144 |       0.124256 |          2.00x |                 0.080800 |
| 0    | AWQ    |       0.061440 |       0.180224 |          2.93x |                 0.079872 |
| 1    | GPTQ   |       0.056736 |       0.120352 |          2.12x |                 0.076160 |
| 1    | AWQ    |       0.057344 |       0.138240 |          2.41x |                 0.076800 |
+------+--------+----------------+----------------+----------------+--------------------------+
```

For AWQ, `fused input` is the serialized N-packed checkpoint layout. The already-available K-packed alternate fused
layout measured `0.120832/0.118784 ms` on GPU0/GPU1, so native GEMV is still `1.97x/2.07x` faster when layout cost is
removed. M=16 remains on WMMA and is intentionally unchanged; its medians were `0.070656-0.074752 ms` on GPU0 and
`0.067584-0.091136 ms` on GPU1 in this noisy concurrent run.

The complete kernel test file ran concurrently on both requested devices after the final JIT build:

```text
CUDA_VISIBLE_DEVICES=0 pytest -q tests/kernels/test_triton_3bit.py
CUDA_VISIBLE_DEVICES=1 pytest -q tests/kernels/test_triton_3bit.py

GPU0: 54 passed, 16 warnings, 18.80 s
GPU1: 54 passed, 16 warnings, 18.82 s
```

Coverage includes the independent GPTQ/AWQ FP16 and BF16 Torch 3-bit quality matrix, the new strict M=1 FP32-oracle
gate, current-stream execution at M=1 and M=3, production dispatch, packed-layout boundaries, bias, and fallback
behavior. Final benchmark artifacts are:

```text
benchmark_artifacts/triton_3bit/final_gemv_gpu0.json
benchmark_artifacts/triton_3bit/final_gemv_gpu1.json
```

Final production memcheck used the exact public dispatch at M=1, K=N=4096 with bias on GPU0:

```text
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/root/GPT-QModel-Ultra compute-sanitizer --tool memcheck \
  --error-exitcode=99 python /tmp/trilin_sanitize_final.py

TRILIN_FINAL_SANITIZER_OK
ERROR SUMMARY: 0 errors
```

`ruff check gptqmodel/nn_modules/triton_utils/three_bit.py tests/kernels/test_triton_3bit.py` also passed. The final
hygiene pass additionally covers `gptqmodel/utils/trilin.py` and `git diff --check`.

After renaming the runtime display text from WMMA-specific to native-CUDA-specific wording, the extension rebuilt and
the strict M=1 oracle plus M=1/M=3 stream tests passed independently on both GPUs (`4 passed` per device). A broad
`tests/test_extension_load_api.py` smoke was intentionally interrupted after its first pass when the `all` case began
building unrelated optional Grasshopper kernels; that partial run is not counted as validation evidence.

### 2026-07-22 — `backend=AUTO` GPTQ 3-bit selection and runtime-route audit

Status: successful for the target contract on both requested GPUs; no implementation change required.

The quantized-model loader defaults `backend` to `BACKEND.AUTO`, passes it through `make_quant`, and repeats the final
selection after checkpoint dispatch. Automatic selection walks the declared format-priority order and returns the
first available class whose capability validation succeeds. For GPTQ/GPTQ_V2 on CUDA, the higher-priority classes
support only 4 or 8 bits. `TritonV2Linear` is the first class that accepts 3 bits:

```text
+----------+----------------------+--------------------------+---------------+
| priority | backend              | class                    | supported bits|
+----------+----------------------+--------------------------+---------------+
|      110 | gptq_torch_aten      | TorchAtenLinear          | 4             |
|      100 | gptq_machete         | MacheteLinear            | 4, 8          |
|       90 | gptq_marlin          | MarlinLinear             | 4, 8          |
|       80 | gptq_exllama_v2      | ExllamaV2Linear          | 4             |
|       50 | gptq_torch_fused     | TorchFusedLinear         | 4             |
|       40 | gptq_triton          | TritonV2Linear           | 2, 3, 4, 8    |
+----------+----------------------+--------------------------+---------------+
```

The GPTQ and GPTQ_V2 calls were checked with the exact first contract on physical GPU0 and GPU1, independently and
concurrently:

```text
select_quant_linear(
    bits=3,
    group_size=128,
    desc_act=False,
    sym=True,
    device=DEVICE.CUDA,
    backend=BACKEND.AUTO,
    format=FORMAT.GPTQ,  # repeated with FORMAT.GPTQ_V2
    quant_method=METHOD.GPTQ,
    pack_dtype=torch.int32,
    dtype=torch.float16, # repeated with torch.bfloat16
)
```

Every device/format/dtype selection returned `TritonV2Linear`. A second live harness instantiated the AUTO-selected
class with real continuous 3-bit weights at K=4096, N=512, ran `post_init`, wrapped each production matmul entry point
to record which one executed, and asserted shape, output dtype, finite values, and native-cache readiness. Both cards
returned the same route matrix:

```text
+-------------+-----------------------------+
| activation  | production route            |
+-------------+-----------------------------+
| FP16 M=1    | Trilin native CUDA GEMV      |
| FP16 M=16   | Trilin native CUDA WMMA      |
| FP16 M=17   | exact expanded Marlin        |
| BF16 M=1    | fused Triton 3-bit           |
+-------------+-----------------------------+
```

Device evidence:

```text
GPU0: NVIDIA PG506-230, UUID cb9e7784-cf50-203d-4f0d-5c622a89b1f2, CC 8.0,
      selected=TritonV2Linear, native_ready=true, marlin_ready=true
GPU1: NVIDIA PG506-232, UUID 20f7fde4-d88c-d6ca-e324-bd4e5e9e0855, CC 8.0,
      selected=TritonV2Linear, native_ready=true, marlin_ready=true
```

Thus, `backend=AUTO` does select the measured fast 3-bit hybrid automatically on this branch and hardware. Trilin is
not a separate public backend: it is the eligible small-M FP16 route inside the AUTO-selected `TritonV2Linear`.
"Fastest" is a measured static routing policy, not a load-time autotune across every kernel. The native route still
requires group size 128, `desc_act=False`, `sym=True`, int32 packing, natural group indices, aligned K/N, FP16 input,
CC >= 8.0, an available JIT extension, and `GPTQMODEL_TRILIN_NATIVE` not set to zero. If a native condition fails, the
same container safely falls through to expanded Marlin or fused Triton; unsupported 3-bit quantization metadata can
fall through to the generic Torch backend instead.

One load-time issue was exposed while extending the probe to AWQ: `BaseQuantLinear.validate` calls a candidate's
`validate_once` dependency check before `_validate` rejects unsupported bit widths. AUTO can therefore begin loading
or compiling a higher-priority 4-bit-only extension while resolving a 3-bit request. The AWQ portion began compiling
the unrelated 4-bit AWQ extension and was intentionally interrupted on both devices; no AWQ result from that failed
probe is used here. This can increase cold model-load latency, but it does not alter the verified GPTQ selection or
inference route. A separate selection-layer improvement should prefilter declarative bit/device/dtype capabilities
before optional dependency initialization.

### 2026-07-22 — Native BF16 Trilin GEMV and WMMA

Status: implemented, performance-retained, accuracy-proven, profiled, and selected for BF16 M=1..16 on both requested
GPUs. This supersedes the earlier AUTO audit's BF16 M=1 fused-Triton row; AUTO still selects `TritonV2Linear`, but its
internal production route now selects native Trilin for eligible small-M BF16 inputs.

#### Reproducible environment

The hardware and toolchain were reprobed rather than inferred from CUDA indices:

```text
+------+-------------------+-------------------+------+-----+-----------+
| GPU  | PCI bus           | model             | CC   | SMs | memory MiB|
+------+-------------------+-------------------+------+-----+-----------+
| 0    | 0000:25:00.0      | NVIDIA PG506-230 | 8.0  | 124 |     98304 |
| 1    | 0000:2b:00.0      | NVIDIA PG506-232 | 8.0  | 124 |     98304 |
+------+-------------------+-------------------+------+-----+-----------+
```

GPU0 UUID is `cb9e7784-cf50-203d-4f0d-5c622a89b1f2`; GPU1 UUID is
`20f7fde4-d88c-d6ca-e324-bd4e5e9e0855`. The run used Python 3.14.5t, PyTorch `2.13.0+cu130`, CUDA runtime 13.0,
NVCC 13.0 build `36424714_0`, Triton 3.7.1, Nsight Compute 2025.3.1, and Nsight Systems 2024.6.2. The successful JIT
build targeted `compute_80,sm_80` and used `-O3`, `-DENABLE_BF16`, `--use_fast_math`, and `-lineinfo`. It completed in
18 seconds in the content-addressed build directory
`/root/.cache/gptqmodel/torch_extensions/trilin/beddb7dddc0be2e1`.

#### Retained dtype contract and implementation

The model storage contract is intentionally mixed dtype:

```text
activation: BF16 contiguous [M, K]
qweight:    int32 continuous 3-bit GPTQ layout [(K / 32) * 3, N]
scales:     FP16 contiguous [K / 128, N]
bias:       optional FP16 or BF16 contiguous [N]
workspace:  FP32 [split_k, M, N] when split_k > 1
output:     BF16 [M, N]
```

Keeping scales FP16 matches the real GPTQ/AWQ module buffers and avoids a load-time model-state conversion. The CUDA
source now templates activation staging, output conversion, GEMV, WMMA fragments, and split reduction over `half`
and `__nv_bfloat16`. Bias has a small uniform runtime dtype tag so production's FP16 bias and an explicit BF16 bias
both fuse into the final FP32 epilogue.

M=1 retains the direct 128-thread GEMV: BF16 activation values and FP16 scales convert explicitly to FP32, centered
3-bit values multiply the scale in FP32, and `__fmaf_rn` accumulates before the existing ordered split reducer rounds
once to BF16. M=2..16 stages BF16 A/B tiles and uses Ampere BF16 WMMA with FP32 accumulator fragments. The centered
integer times FP16 scale is rounded to BF16 only when constructing the WMMA B operand. The FP16 specialization keeps
its existing half multiply and output behavior.

Both `TritonV2Linear` and `AwqGEMMTritonLinear` now admit FP16 or BF16 activations to the native M<=16 branch. Expanded
Marlin remains FP16-only. The measured routing policy is:

```text
+----------------+-------------------------------+
| activation/M   | production route              |
+----------------+-------------------------------+
| FP16 M=1       | native Trilin CUDA GEMV        |
| FP16 M=2..16   | native Trilin CUDA WMMA        |
| FP16 M>16      | exact expanded Marlin          |
| BF16 M=1       | native Trilin CUDA GEMV        |
| BF16 M=2..16   | native Trilin BF16 CUDA WMMA   |
| BF16 M>16      | fused Triton 3-bit             |
+----------------+-------------------------------+
```

The raw native binding can still launch its M>16 diagnostic, but production does not select it. At GPTQ M=33,
K=N=4096, it measured approximately 0.260-0.261 ms versus 0.149-0.151 ms for fused Triton, so retaining the fallback is
required for latency and throughput.

#### Baseline and benchmark tactic corrections

The first dual-GPU baseline command failed before CUDA work because it used the nonexistent `--json-output` option.
The accepted spelling is `--output-json`; the earlier FP16 command transcription in this journal was corrected. The
benchmark fixture was also corrected to keep scales FP16 for BF16 activation tests, matching production instead of
creating synthetic BF16 scale buffers.

The corrected pre-native control ran concurrently on both devices with 100 operation warmups, 64 large-BF16-GEMM
clock warmups, and 500 CUDA-event samples:

```text
CUDA_VISIBLE_DEVICES=<0|1> python scripts/benchmark_triton_3bit.py --layout both --dtype bf16 \
  --shape 1x4096x4096 --shape 16x4096x4096 --shape 33x4096x4096 \
  --warmup 100 --clock-warmup-iterations 64 --iterations 500 \
  --output-json benchmark_artifacts/triton_3bit/bf16_fused_baseline_gpu<0|1>.json
```

```text
+------+--------+----------+--------------------+
| GPU  | layout | M        | fused median ms    |
+------+--------+----------+--------------------+
| 0    | GPTQ   | 1 / 16   | 0.1157 / 0.1290    |
| 0    | AWQ    | 1 / 16   | 0.1178 / 0.1290    |
| 1    | GPTQ   | 1 / 16   | 0.1208 / 0.1290    |
| 1    | AWQ    | 1 / 16   | 0.1249 / 0.1290    |
+------+--------+----------+--------------------+
```

The retained native comparison used the same setup plus `--include-native-trilin`:

```text
+------+--------+----+------------------+-----------------+---------+
| GPU  | layout | M  | native median ms | fused median ms | speedup |
+------+--------+----+------------------+-----------------+---------+
| 0    | GPTQ   |  1 |         0.059392 |        0.120832 |   2.03x |
| 0    | GPTQ   | 16 |         0.069632 |        0.129024 |   1.85x |
| 0    | AWQ    |  1 |         0.066560 |        0.122880 |   1.85x |
| 0    | AWQ    | 16 |         0.072704 |        0.124928 |   1.72x |
| 1    | GPTQ   |  1 |         0.060416 |        0.120832 |   2.00x |
| 1    | GPTQ   | 16 |         0.071680 |        0.129024 |   1.80x |
| 1    | AWQ    |  1 |         0.063488 |        0.124928 |   1.97x |
| 1    | AWQ    | 16 |         0.072704 |        0.122880 |   1.69x |
+------+--------+----+------------------+-----------------+---------+
```

Artifacts are `bf16_native_gpu0.json` and `bf16_native_gpu1.json` under
`benchmark_artifacts/triton_3bit/`. The late GPU0 AWQ M=33 diagnostic samples experienced system interference: dense,
alternate fused, native, and production timings all jumped together after the stable small-M samples. Those noisy
large-M numbers are not used for a routing claim; the clean GPTQ M=33 measurements on both GPUs independently prove
that native large-M is slower.

#### BF16 accuracy investigation

The first focused production-quality run produced four failures on each GPU: GPTQ/AWQ at BF16 M=1 and M=16. All
native boundary, bias, route, and stream tests passed. The failed Torch-reference metrics were identical across
layouts and devices:

```text
+----+---------+-----------+---------------+-------------+
| M  | max abs | mean abs  | relative RMSE | cosine      |
+----+---------+-----------+---------------+-------------+
|  1 | 0.25000 | 0.022400  |      0.002426 | 0.999997020 |
| 16 | 0.50000 | 0.019000  |      0.002892 | 0.999995828 |
+----+---------+-----------+---------------+-------------+
```

Changing split-K from 4 to 2 or 1 did not change the BF16-rounded output, ruling out the reducer order as the source.
A three-way comparison then used an independent FP32 dequantized-weight oracle. For the seeded M=1 case, eager
Torch/fused Triton differed from the oracle by mean absolute `0.022156` and relative RMSE `0.002419`, while native
Trilin was bit-identical to the BF16-rounded oracle. At M=16, eager Torch/fused Triton differed by mean absolute
`0.015686` and relative RMSE `0.002789`; native differed only by approximately `5.8e-11` mean from one sub-BF16 FP32
intermediate and was bit-identical after BF16 interpretation.

Therefore the initial failures were not a native quality regression: the old limits described how closely fused
Triton reproduced eager Torch's BF16 GEMM order, not error against the higher-precision mathematical reference. The
Torch-reference BF16 gate remains tight at maximum 0.5, mean 0.025, relative RMSE 0.003, and cosine 0.99999. It now
also constructs the independent FP32 oracle and requires native M<=16 mean error and relative RMSE to be no worse
than eager Torch. The strict native M=1 test covers FP16 input/FP16 bias, BF16 input/FP16 production bias, and BF16
input/BF16 bias. This preserves both requested reference types instead of weakening the test around a better result.

A final long-K test replaced the exactly BF16-representable scale pattern with random FP16 scales at K=4096, N=512.
At M=1, native remained bit-identical to the FP32 oracle while Torch BF16 had mean absolute error `0.069395` and
relative RMSE `0.002787`. At M=16, native had mean absolute error `0.062494` and relative RMSE `0.002592` against the
oracle, better than Torch BF16's `0.087409` and `0.003183`. The test also keeps a direct native-versus-Torch gate of
maximum 1.0, mean 0.1, relative RMSE 0.004, and cosine 0.99999 for this longer accumulation regime.

#### Nsight proof and failed profiler tactics

Nsight Compute ran on physical GPU0 while Nsight Systems was assigned physical GPU1. NCU full replay captured exactly
the BF16 GEMV producer and FP32 ordered reducer:

```text
+--------------------------+-------------+------------+
| metric                   | BF16 GEMV   | reducer    |
+--------------------------+-------------+------------+
| duration                 |    17.92 us |    8.13 us |
| block / grid             | 128 / 1024  | 256 / 16   |
| registers per thread     |          32 |         34 |
| local spills             |           0 |          0 |
| compute throughput       |      46.83% |      0.87% |
| memory throughput        |      22.63% |      2.70% |
| achieved occupancy       |      39.18% |     12.27% |
+--------------------------+-------------+------------+
```

Occupancy is recorded only as diagnostic metadata. The acceptance evidence is the matched CUDA-event latency plus
the kernel durations and absence of spills/errors.

A second full NCU replay at M=16 captured the BF16 WMMA producer and reducer. The producer measured `42.08 us`, 40
registers/thread, zero spills, 75.25% memory throughput, 84.06% L1/TEX throughput, 35.72% compute throughput, and
58.61% achieved occupancy; the reducer measured `10.56 us`. The tensor pipe was active, and the correlated SASS view
explicitly contains `HMMA.16816.F32.BF16`, proving that this route executes Ampere BF16 Tensor Core instructions with
FP32 destinations rather than an FP16 or scalar emulation path.

Two NSYS full-trace attempts with `--trace=cuda,nvtx,osrt` deadlocked during `logbar`'s background-thread startup on
this Python 3.14t environment, once with the default free-threading state and once with `PYTHON_GIL=1`. Both were
interrupted before CUDA initialization and their import-only reports were rejected. Removing `osrt` tracing while
retaining `cuda,nvtx` resolved the tooling interaction without changing the workload. The successful GPU1 report
recorded producer median `15.104 us`, reducer median `5.280 us`, and one five-operation NVTX range of `299.842 us`, or
`59.968 us` per public Trilin call. This agrees with the 0.060416 ms CUDA-event median on GPU1.

Profiler artifacts:

```text
benchmark_artifacts/triton_3bit/ncu_bf16_native_gptq_m1_gpu0.ncu-rep
benchmark_artifacts/triton_3bit/ncu_bf16_native_gptq_m16_gpu0.ncu-rep
benchmark_artifacts/triton_3bit/nsys_bf16_native_gptq_m1_gpu1.nsys-rep
benchmark_artifacts/triton_3bit/nsys_bf16_native_gptq_m1_gpu1.sqlite
```

#### Final regression and memory-safety proof

The complete focused kernel file ran concurrently and independently:

```text
CUDA_VISIBLE_DEVICES=0 pytest -q tests/kernels/test_triton_3bit.py
CUDA_VISIBLE_DEVICES=1 pytest -q tests/kernels/test_triton_3bit.py

GPU0: 64 passed, 16 warnings, 27.24 s
GPU1: 64 passed, 16 warnings, 27.76 s
```

Coverage added by this iteration includes BF16 native route proof at M=1 and M=16 for GPTQ/AWQ, FP32-oracle quality,
FP16 and BF16 bias loads, BF16 output dtype, and current-stream execution for GEMV and WMMA. Existing FP16, packed
layout, save/reload, invalid-contract, and M>16 fallback coverage remains green.

Final `compute-sanitizer --tool memcheck` ran public BF16 Trilin calls at M=1 with FP16 bias and M=16 with BF16 bias,
K=N=4096, on physical GPU0:

```text
TRILIN_BF16_SANITIZER_OK
ERROR SUMMARY: 0 errors
```

`ruff check` over every changed Python path and `git diff --check` both passed before publication.

### 2026-07-22 — CTA-local M=1 split reduction experiment

Status: in progress; fresh control captured, candidate not yet accepted.

The next iteration started from commit `eab4d28a` with a clean worktree. Runtime probing, rather than a fixed CUDA
index assumption, found eight visible compute-capability 8.0 devices. The two assigned test devices are physical
GPU0 (`PG506-230`, UUID `GPU-cb9e7784-cf50-203d-4f0d-5c622a89b1f2`, PCI `25:00.0`) and physical GPU1
(`PG506-232`, UUID `GPU-20f7fde4-d88c-d6ca-e324-bd4e5e9e0855`, PCI `2b:00.0`); both expose 124 SMs and
98304 MiB. The software stack is PyTorch `2.13.0+cu130`, CUDA toolkit/runtime 13.0, driver `610.43.02`, Nsight
Compute `2025.3.1`, and Nsight Systems `2024.6.2`.

Before selecting another fusion tactic, the previous journal was audited. Cooperative grid synchronization, a
last-CTA counter/fence protocol, a vectorized reducer, 128-column producer CTAs, scale caching, and dedicated staging
warps have all already failed the cross-device latency gate. None is being repeated.

Matched 200-operation Nsight Systems controls used `--trace=cuda,nvtx` after JIT, allocator, and 64 large-GEMM clock
warmups. FP16 ran on physical GPU0 while BF16 ran concurrently on physical GPU1. Each public M=1 call still launches
one split-K producer followed by one ordered reducer:

```text
+------+-------+------------+------------+----------------+----------------+-------------+
| GPU  | dtype | producer   | reducer    | projected/call | host range/call| GPU ops/call|
+------+-------+------------+------------+----------------+----------------+-------------+
| 0    | FP16  | 14.784 us  | 5.216 us   | 50.507 us      | 50.851 us      |           2 |
| 1    | BF16  | 15.072 us  | 5.184 us   | 49.784 us      | 50.175 us      |           2 |
+------+-------+------------+------------+----------------+----------------+-------------+
```

The kernel values are `nvtx_kern_sum` medians; projected and host values are the `nvtx_gpu_proj_sum` totals divided
by 200. The roughly 30 us between the summed kernel medians and projected per-call span is combined launch/queue/gap
cost, not a claim that the reducer alone costs 30 us. Profiler timings are attribution evidence; warmed CUDA-event
distributions remain the acceptance measurement.

Artifacts:

```text
benchmark_artifacts/triton_3bit/nsys_current_fp16_gptq_m1_gpu0.nsys-rep
benchmark_artifacts/triton_3bit/nsys_current_fp16_gptq_m1_gpu0.sqlite
benchmark_artifacts/triton_3bit/nsys_current_bf16_gptq_m1_gpu1.nsys-rep
benchmark_artifacts/triton_3bit/nsys_current_bf16_gptq_m1_gpu1.sqlite
```

The bounded candidate changes the dataflow rather than adding global synchronization. One CTA owns 32 adjacent
output columns. Warp `s` computes exactly the same contiguous K slice previously assigned to split-K CTA `s`, so
lanes still load adjacent packed-weight columns and each lane's FMA order is unchanged. The warps publish their
partials to a bank-conflict-free shared array `[split_k, 32]`; warp zero then sums split indices in the existing
`0..split_k-1` order, adds bias, and writes the final FP16 or BF16 value.

For the primary `K=4096, split_k=32` shape this folds 32 global producer CTAs plus the separate reducer dependency
into one 1024-thread CTA per 32 output columns. Total producer threads and FMA work are unchanged. The candidate
removes the FP32 global workspace write/read, its allocation, and the second kernel launch without a grid barrier,
atomics, counters, or a fixed-SM assumption. Because the split boundaries and final reduction order are preserved,
the expected numerical result is bit-identical; that expectation will be tested rather than assumed.

#### First CTA-local implementation and exactness

The first implementation used 32 output columns per CTA. For the primary shape, `split_k=32` therefore launched
128 blocks of 1024 threads. Each block stored 32 by 32 FP32 partials in 4 KiB of shared memory, synchronized once,
and let split zero perform the existing ordered reduction. The public output allocation remained, but M=1 no longer
allocated the `[split_k, M, N]` FP32 workspace.

An experimental A/B entry point was used only during tuning and removed from the final source. Both physical GPUs
matched the old producer-plus-reducer result bit-for-bit for FP16 and BF16, split counts 1/2/4/32, K values
128/256/640/4096, N values 64/256/512/4096, FP16 and BF16 bias, and the full 4096 by 4096 problem. Because every
warp retained one original split boundary and the final shared-memory loop retained split order, this exact result
also confirms that the compiler did not change the numerical contract.

Six alternating-order, clock-warmed rounds with 1000 CUDA-event samples per method established an end-to-end win for
the first 32-column version on both devices and dtypes:

```text
+------+-------+------------+------------+----------+----------+---------+
| GPU  | dtype | old p50 ms | CTA p50 ms | old p95  | CTA p95  | speedup |
+------+-------+------------+------------+----------+----------+---------+
| 0    | FP16  |   0.038912 |   0.036864 | 0.044032 | 0.039936 |   1.06x |
| 0    | BF16  |   0.035840 |   0.034816 | 0.049152 | 0.044032 |   1.03x |
| 1    | FP16  |   0.036864 |   0.034816 | 0.043008 | 0.039936 |   1.06x |
| 1    | BF16  |   0.036864 |   0.034816 | 0.044032 | 0.039936 |   1.06x |
+------+-------+------------+------------+----------+----------+---------+
```

The corresponding public-style A/B, which performed split selection and extension-op lookup on every call, improved
median latency by 6-8% on both cards and dtypes. This mattered because a raw-kernel-only comparison initially looked
unfavorable: Nsight Systems measured the 32-column CTA kernel at 22.080 us FP16 and 22.976 us BF16, versus about
20.0-20.3 us for the old producer and reducer combined. The single launch nevertheless shortened the 200-call GPU
projection from 50.507 to 32.376 us/call for FP16 and from 49.784 to 24.215 us/call for BF16. The normal CUDA-event
distribution, not the profiled span, was the retention gate.

Full NCU replay of the 32-column control reported 23.84 us, 32 registers/thread, 4.10 KiB static shared memory, zero
spills, 35.70% compute throughput, 17.11% memory throughput, and a 128-block by 1024-thread launch. The report also
showed an SM active-cycle imbalance: the maximum SM was 37.21% above the mean because 128 blocks leave a four-block
tail on 124 SMs. This motivated a bounded output-column sweep while keeping total threads and arithmetic unchanged.

#### Output-column sweep and retained 16-column CTA

Experimental 8-, 16-, and 32-column variants all remained bit-identical to the old path. Six-round timing of the
first full sweep was invalidated because every method on both GPUs simultaneously jumped from about 35-40 us to
270-320 us during later rounds. The process table showed no competing process on GPU0 or GPU1 after the event. The
aggregate numbers from that contaminated run were rejected rather than filtered into an optimization claim.

A clean repeat compared only the old two-launch path, 32 columns, and 16 columns in eight alternating-order rounds
of 500 samples after 64 clock-warmup GEMMs per round. The table reports the median of the eight per-round medians and
the median of the eight per-round p95 values:

```text
+------+-------+------------+------------+------------+----------+----------+---------+
| GPU  | dtype | old p50 ms | C32 p50 ms | C16 p50 ms | old p95  | C16 p95  | speedup |
+------+-------+------------+------------+------------+----------+----------+---------+
| 0    | FP16  |   0.039936 |   0.036352 |   0.034816 | 0.051712 | 0.045568 |   1.15x |
| 0    | BF16  |   0.038912 |   0.035328 |   0.034816 | 0.050688 | 0.046080 |   1.12x |
| 1    | FP16  |   0.037888 |   0.035840 |   0.033792 | 0.049664 | 0.043008 |   1.12x |
| 1    | BF16  |   0.037376 |   0.034816 |   0.032768 | 0.046592 | 0.039936 |   1.14x |
+------+-------+------------+------------+------------+----------+----------+---------+
```

Sixteen columns folds two original split warps into each hardware warp. Each half-warp still reads 16 adjacent N
columns, so packed-weight and scale accesses remain coalesced; the two half-warps use different K slices. At
`split_k=32`, the retained launch is 256 blocks by 512 threads with 2 KiB of shared partials. Eight columns was slower
than 16 in the uncontaminated intervals and was rejected. Thirty-two columns was consistently slower than 16 in the
clean repeat and was superseded.

Nsight Systems confirmed that the 16-column dataflow improves the device work as well as removing the launch:

```text
+------+-------+-----------------+-----------------+--------------------+-------------+
| GPU  | dtype | old producer    | old reducer     | retained CTA median| projected/call|
+------+-------+-----------------+-----------------+--------------------+-------------+
| 0    | FP16  |       14.784 us |        5.216 us |          18.208 us |   23.615 us |
| 1    | BF16  |       15.072 us |        5.184 us |          18.720 us |   20.129 us |
+------+-------+-----------------+-----------------+--------------------+-------------+
```

The retained NCU full replay on physical GPU0 measured 21.22 us, 32 registers/thread, 2.05 KiB static shared memory,
zero spills, 40.21% compute throughput, 38.18% memory throughput, 58.93% L1/TEX throughput, 81.47% L1 hit rate,
and 47.03% achieved occupancy. Occupancy is diagnostic metadata, not the optimization goal. Relative to the
32-column NCU control, duration fell 11.0% while registers stayed constant and shared memory halved. Nsight Systems
normal-trace medians are lower than NCU replay durations as expected; only matched timings from the same tool are
compared.

Profiler artifacts:

```text
benchmark_artifacts/triton_3bit/nsys_cta_reduce_fp16_gptq_m1_gpu0.nsys-rep
benchmark_artifacts/triton_3bit/nsys_cta_reduce_bf16_gptq_m1_gpu1.nsys-rep
benchmark_artifacts/triton_3bit/ncu_cta_reduce_fp16_gptq_m1_gpu0.ncu-rep
benchmark_artifacts/triton_3bit/nsys_cta16_fp16_gptq_m1_gpu0.nsys-rep
benchmark_artifacts/triton_3bit/nsys_cta16_bf16_gptq_m1_gpu1.nsys-rep
benchmark_artifacts/triton_3bit/ncu_cta16_fp16_gptq_m1_gpu0.ncu-rep
```

#### Final production latency, quality, and safety

The final source contains only the 16-column production specialization; every experimental operator and alternate
column instantiation was removed. M=1 with `split_k>1` uses the CTA-local reduction. M=1 with `split_k=1` keeps the
old direct write, M=2..16 keeps split-K WMMA plus its ordered reducer, and all M>16 and unsupported-contract fallbacks
are unchanged. The final JIT fingerprint is:

```text
/root/.cache/gptqmodel/torch_extensions/trilin/737859811d44ec98
```

The standard production benchmark ran 100 warmups, 64 large-GEMM clock warmups, and 500 CUDA-event samples for each
method. M=1 absolute native medians and throughput are:

```text
+------+-------+--------+-----------+--------------+-----------+----------------+
| GPU  | dtype | layout | native ms | native op/s  | fused ms  | native speedup |
+------+-------+--------+-----------+--------------+-----------+----------------+
| 0    | FP16  | GPTQ   |    0.0584 |       17,123 |    0.1382 |          2.37x |
| 0    | FP16  | AWQ    |    0.0543 |       18,416 |    0.1198 |          2.21x |
| 0    | BF16  | GPTQ   |    0.0543 |       18,416 |    0.1219 |          2.25x |
| 0    | BF16  | AWQ    |    0.0553 |       18,083 |    0.1270 |          2.30x |
| 1    | FP16  | GPTQ   |    0.0573 |       17,452 |    0.1249 |          2.18x |
| 1    | FP16  | AWQ    |    0.0573 |       17,452 |    0.1229 |          2.14x |
| 1    | BF16  | GPTQ   |    0.0573 |       17,452 |    0.1219 |          2.13x |
| 1    | BF16  | AWQ    |    0.0594 |       16,835 |    0.1280 |          2.15x |
+------+-------+--------+-----------+--------------+-----------+----------------+
```

The final benchmark also covered M=16 to prove that the existing WMMA route remains active and faster than the fused
Triton fallback. Complete tables and distributions are in:

```text
benchmark_artifacts/triton_3bit/cta16_final_gpu0_fp16.json
benchmark_artifacts/triton_3bit/cta16_final_gpu0_bf16.json
benchmark_artifacts/triton_3bit/cta16_final_gpu1_fp16.json
benchmark_artifacts/triton_3bit/cta16_final_gpu1_bf16.json
```

The complete focused suite passed independently and concurrently on both cards:

```text
GPU0: 66 passed, 16 warnings in 35.11 s
GPU1: 66 passed, 16 warnings in 35.62 s
```

Those tests include GPTQ and AWQ, FP16 and BF16 eager-Torch references, the independent FP32 dequantization oracle,
random FP16 scales at long K, FP16/BF16 bias, output dtype/shape/finite checks, current-stream execution, repeated
calls, serialization, backend routing, and the M>16 fallbacks. The dedicated A/B also proved the new reduction
bit-identical to the old native result across all tested split counts and dimensions.

The two added cases capture and replay the M=1 specialization in a CUDA Graph for FP16 and BF16 at K=4096, compare
the replay output bit-for-bit with eager native Trilin, and passed on both physical GPUs. This closes the graph-capture
part of the mega-kernel deployment contract rather than relying on the successful eager path alone.

Compute Sanitizer then ran public M=1 FP16 and BF16 calls at K=N=4096. Memcheck on physical GPU0 and synccheck on
physical GPU1 each reported `ERROR SUMMARY: 0 errors`; racecheck on physical GPU0 reported `0 hazards displayed
(0 errors, 0 warnings)`. The shared handoff is therefore covered by memory, synchronization, and race analysis.

The first hardware query in this iteration requested the unsupported `nvidia-smi` CSV field
`multiprocessor_count`; it failed before returning partial metadata. The accepted probe used supported identity,
memory, compute-capability, and driver fields plus PyTorch's runtime device properties for SM count. No hardware
claim uses the failed query.

### 2026-07-22 — Paired activation loads and rejected wider mega-kernel variants

Status: paired FP16/BF16 activation loads retained; all experimental entry points and losing variants removed.

This pass started from commit `ab5a4477`, the retained 16-column CTA-local M=1 reduction. Runtime probing again
identified physical GPU0 as `PG506-230` (UUID `GPU-cb9e7784-cf50-203d-4f0d-5c622a89b1f2`, PCI `25:00.0`) and
physical GPU1 as `PG506-232` (UUID `GPU-20f7fde4-d88c-d6ca-e324-bd4e5e9e0855`, PCI `2b:00.0`). Both are CC 8.0,
124-SM, 98304 MiB devices. The run used PyTorch `2.13.0+cu130`, CUDA runtime/toolkit 13.0, `nvcc` 13.0.88,
Nsight Systems 2024.6.2, and Nsight Compute 2025.3.1. Physical GPU0 and GPU1 were always selected by a fresh
visibility mapping; no kernel behavior depends on either fixed index.

The retained control is one native launch for M=1: 256 CTAs, 512 threads/CTA, 16 output columns/CTA, and 32
split-K slices at K=4096. Each thread decodes one 32-value K tile, accumulates in FP32, publishes one split partial
to a 2 KiB shared array, and split zero performs the ordered reduction. The pre-change focused NCU report measured
21.216 us replay duration, 32 registers/thread, zero spills, 2.048 KiB static shared memory, and 0.52 waves/SM.
Its dominant actionable signal was 4.076 long-scoreboard stalled warps per issued instruction, with 589824 global
load requests and 1277952 sectors. Occupancy remained diagnostic metadata, not the optimization objective.

#### Rejected tactics

All candidates below were compiled into temporary comparison entry points, checked for FP16 and BF16 correctness,
timed in alternating order, and then removed. Results from a contaminated interval were rejected in full rather
than selectively filtered.

1. **Two or four adjacent output columns per thread.** C2 and C4 used `uint2`/`uint4` packed-weight loads,
   paired scale loads, and a shared input tile. Both were exact. Across eight alternating 500-sample rounds, C2
   was mostly flat with mixed mean/tail behavior and BF16 regressions; C4 was generally slower. The extra live
   accumulators and decode state did not repay the activation reuse.
2. **Next-tile register prefetch.** Prefetching packed words and scale for the next K tile was exact but increased
   the steady launch batch on both devices. GPU0 FP16 moved from 18.650 to 19.310 us/call and GPU1 BF16 from
   17.357 to 17.902 us/call. It was removed; the added live state outweighed any latency hiding.
3. **One-CTA M=16 WMMA mega-kernel.** This removed the 8 MiB split workspace and ordered reducer by assigning a
   complete 16x16 output tile to each CTA, loading A directly, staging per-warp B, and reducing local FP32 split
   partials. It was exact but slower:

```text
+------+-------+-------+-------------------+----------------+-------------------+----------------+
| GPU  | dtype | warps | old batch us/call | CTA us/call    | old event p50 us  | CTA p50 us     |
+------+-------+-------+-------------------+----------------+-------------------+----------------+
| 0    | FP16  |     8 |            43.853 |         45.796 |            48.128 |         49.664 |
| 1    | BF16  |     8 |            42.187 |         47.972 |            47.104 |         51.200 |
| 0    | FP16  |    16 |            44.394 |         46.368 |            46.592 |         49.152 |
| 1    | BF16  |    16 |            42.285 |         47.273 |            46.080 |         51.200 |
+------+-------+-------+-------------------+----------------+-------------------+----------------+
```

   A 32-warp build was also attempted. `ptxas` rejected its 65536-byte static shared allocation because it exceeds
   this launch's default 49152-byte per-block limit. Dynamic shared-memory opt-in was not pursued: the smaller
   eight- and 16-warp versions already failed the latency gate.
4. **Paired split slices per thread.** Halving the M=1 thread count and having each thread accumulate two adjacent
   split-K slices was exact, but GPU0 FP16 regressed from 18.239 to 19.748 us/call and GPU1 BF16 from 17.525 to
   18.357 us/call. A 32-column version was worse again in clean early intervals: roughly 17.9-18.9 us controls
   versus 21.0-22.3 us for the candidate.
5. **Contaminated split-pair aggregate.** During the later all-method sweep, every method on both devices jumped
   together to 0.25-0.30 ms. Because the disturbance affected controls and candidates, no value from that interval
   was used. The clean early comparison above was sufficient to reject the tactic.

The first matched Nsight Systems attempt also failed before profiling because its temporary helper was invoked
without the repository on `PYTHONPATH`. The rerun explicitly set `PYTHONPATH=/root/GPT-QModel-Ultra`; no number
from the failed setup appears in a result table.

#### Retained paired activation load

NCU showed that the remaining M=1 kernel was instruction and dependency limited around its activation reads. The
retained change leaves the quant decode, FP32 FMA sequence, split boundaries, reduction order, launch geometry, and
fallback routing unchanged. It only reads adjacent activation values with one aligned 32-bit `half2` or
`__nv_bfloat162` load, converts that pair to `float2`, and executes the original two FMAs in the original order.
`kTileK=32`, an even pair offset, and PyTorch's aligned tensor storage satisfy the vector-load alignment contract.

Temporary scalar-versus-paired entry points proved bit-identical output for FP16 and BF16 across K=128, 256, 640,
and 4096, including the primary K=N=4096 case. They were removed after acceptance. The final source has one public
implementation and JIT fingerprint:

```text
/root/.cache/gptqmodel/torch_extensions/trilin/12ce05ded59bcbd5
```

Alternating-order continuous-launch A/B timing showed a stable batch improvement in all four device/dtype
directions. The primary concurrent pairing improved GPU0 FP16 from 18.750 to 17.562 us/call (6.34%) and GPU1 BF16
from 17.479 to 16.615 us/call (4.95%). The reverse pairing improved GPU0 BF16 from 17.579 to 16.760 us/call (4.66%)
and GPU1 FP16 from 18.146 to 17.435 us/call (3.92%). Per-event tails were noisier, so the matched profiler
distribution below is the tail-latency evidence.

The experimental matched A/B Nsight Systems capture used 200 launches per range and reversed method order on
physical GPU1. Every kernel statistic improved:

```text
+------+-------+----------+----------+----------+----------+----------+----------+------------+
| GPU  | dtype | old mean | new mean | old p50  | new p50  | old p95  | new p95  | mean gain  |
+------+-------+----------+----------+----------+----------+----------+----------+------------+
| 0    | FP16  | 18.408 us| 17.373 us| 18.400 us| 17.376 us| 18.560 us| 17.504 us|      5.62% |
| 1    | BF16  | 18.875 us| 17.897 us| 18.880 us| 17.888 us| 18.976 us| 18.016 us|      5.18% |
+------+-------+----------+----------+----------+----------+----------+----------+------------+
```

An exact-final-source capture then measured 200 retained kernels at 17.224 us mean, 17.216 us median, and 17.344 us
p95 on physical GPU0 FP16. Physical GPU1 BF16 measured 17.781 us mean, 17.760 us median, and 17.888 us p95. Each
NVTX range contained exactly 200 GPU operations, confirming that the M=1 mega-kernel stayed a one-launch path.

Focused Nsight Compute reports tie the gain to reduced work and dependencies:

```text
+----------------------------------+-------------+-------------+-------------+
| metric                           | scalar load | paired load | change      |
+----------------------------------+-------------+-------------+-------------+
| replay duration                  |   21.216 us |   19.168 us |      -9.65% |
| executed instructions            |     5001216 |     4739072 |      -5.24% |
| global-load requests             |      589824 |      327680 |     -44.44% |
| global-load sectors              |     1277952 |      753664 |     -41.03% |
| long-scoreboard / issue          |    4.076320 |    2.865758 |     -29.70% |
| shared-memory bank conflicts     |         260 |         113 |     -56.54% |
| registers/thread                 |          32 |          32 |   unchanged |
| static shared memory             |   2.048 KiB |   2.048 KiB |   unchanged |
| spills                           |           0 |           0 |   unchanged |
| grid x threads                   |   256 x 512 |   256 x 512 |   unchanged |
+----------------------------------+-------------+-------------+-------------+
```

DRAM read traffic was effectively unchanged (6.571648 versus 6.571392 MB), as expected because the same packed
weights and activations are consumed. The reported L1 hit-rate percentage fell because vectorization changes the
request population and denominator; the absolute request/sector counts, instruction count, dependency stalls, and
matched latency all improved. This is an instruction-coalescing win, not an occupancy change.

Profiler artifacts:

```text
benchmark_artifacts/triton_3bit/nsys_input2_fp16_gpu0.nsys-rep
benchmark_artifacts/triton_3bit/nsys_input2_bf16_gpu1.nsys-rep
benchmark_artifacts/triton_3bit/nsys_input2_final_fp16_gpu0.nsys-rep
benchmark_artifacts/triton_3bit/nsys_input2_final_fp16_gpu0.sqlite
benchmark_artifacts/triton_3bit/nsys_input2_final_bf16_gpu1.nsys-rep
benchmark_artifacts/triton_3bit/nsys_input2_final_bf16_gpu1.sqlite
benchmark_artifacts/triton_3bit/ncu_cta16_fp16_gptq_m1_gpu0.ncu-rep
benchmark_artifacts/triton_3bit/ncu_input2_final_fp16_gpu0.ncu-rep
benchmark_artifacts/triton_3bit/ncu_input2_final_mem_fp16_gpu0.ncu-rep
```

#### Final public-path latency, quality, and safety

The production benchmark used 100 method warmups, 64 large-GEMM clock warmups, and 500 CUDA-event samples. The
table gives absolute M=1 native Trilin medians; the production hybrid selected the same native route within timing
noise for every row:

```text
+------+-------+--------+-----------+--------------+-----------+----------------+
| GPU  | dtype | layout | native ms | native op/s  | fused ms  | native speedup |
+------+-------+--------+-----------+--------------+-----------+----------------+
| 0    | FP16  | GPTQ   |    0.0532 |       18,780 |    0.1188 |          2.23x |
| 0    | FP16  | AWQ    |    0.0563 |       17,756 |    0.1280 |          2.27x |
| 0    | BF16  | GPTQ   |    0.0548 |       18,253 |    0.1249 |          2.28x |
| 0    | BF16  | AWQ    |    0.0553 |       18,084 |    0.1229 |          2.22x |
| 1    | FP16  | GPTQ   |    0.0532 |       18,780 |    0.1208 |          2.27x |
| 1    | FP16  | AWQ    |    0.0553 |       18,084 |    0.1239 |          2.24x |
| 1    | BF16  | GPTQ   |    0.0563 |       17,756 |    0.1270 |          2.25x |
| 1    | BF16  | AWQ    |    0.0543 |       18,426 |    0.1219 |          2.25x |
+------+-------+--------+-----------+--------------+-----------+----------------+
```

The latter half of the concurrent physical-GPU1 BF16 M=16 AWQ benchmark was contaminated (native roughly 0.53 ms
while adjacent methods moved inconsistently), so it is not used for an M=16 claim. M=1 completed before that
interval, stayed in its normal range, and is backed by the matched A/B and exact-source profiler captures. Complete
standard result payloads are:

```text
benchmark_artifacts/triton_3bit/input2_final_gpu0_fp16.json
benchmark_artifacts/triton_3bit/input2_final_gpu0_bf16.json
benchmark_artifacts/triton_3bit/input2_final_gpu1_fp16.json
benchmark_artifacts/triton_3bit/input2_final_gpu1_bf16.json
```

The paired load does not alter arithmetic, and scalar-versus-paired comparison was bit-identical. The complete
focused test file then passed independently and concurrently on both assigned devices:

```text
GPU0: 66 passed, 16 warnings in 37.40 s
GPU1: 66 passed, 16 warnings in 38.29 s
```

Coverage includes eager Torch 3-bit references for GPTQ and AWQ in FP16 and BF16, a dense FP32 dequantization
oracle, random FP16 scales, FP16/BF16 bias combinations, M=1/M=16/M=33 quality cases, output shape and dtype,
current-stream behavior, CUDA Graph capture/replay, save/reload inference, backend selection, and fallback routing.
The final primary-shape benchmark reported the same quality values as the scalar path: FP16 fused-reference mean
absolute error `0.0001707` and BF16 `0.0282301`; vectorization introduced no additional error.

Finally, Compute Sanitizer memcheck ran all three K=4096 M=1 FP16/BF16 bias combinations selected by
`test_native_trilin_3bit_m1_matches_fp32_dequant_reference` on physical GPU0. All three tests passed and sanitizer
reported `ERROR SUMMARY: 0 errors`. `git diff --check` is required again after this journal update before commit.

### 2026-07-22 — Group-scale factoring and warp-paired primary-shape reduction

Status: retained for the exact M=1, K=N=4096, split-K=32 path; every other shape remains on the preceding native
kernel or established fallback.

This pass started from commit `5d933978`, the paired-activation-load kernel. A fresh runtime probe identified physical
GPU0 as `PG506-230` (UUID `GPU-cb9e7784-cf50-203d-4f0d-5c622a89b1f2`, PCI `25:00.0`) and physical GPU1 as
`PG506-232` (UUID `GPU-20f7fde4-d88c-d6ca-e324-bd4e5e9e0855`, PCI `2b:00.0`). Both devices report CC 8.0, 124 SMs,
and 98304 MiB. The software stack was PyTorch `2.13.0+cu130`, CUDA runtime/toolkit 13.0, `nvcc` 13.0.88, Nsight
Systems 2024.6.2, and Nsight Compute 2025.3.1. Physical-device selection was isolated with fresh
`CUDA_VISIBLE_DEVICES` mappings; the implementation contains no fixed-device-index assumptions.

The committed control launches one 512-thread CTA per 16 output columns. Its 32 half-warps independently compute the
32 K splits, and the first 16 threads then add all 32 FP32 split partials from a 2 KiB shared array. At K=4096 and
split-K=32, every split is exactly one 128-value quantization group. The control nevertheless converts and multiplies
the same group scale into every one of the 128 element products, or 524288 scale multiplies over the full operation.
The fresh control benchmark and profiler artifacts were:

```text
benchmark_artifacts/triton_3bit/group_accum_baseline_gpu0_fp16.json
benchmark_artifacts/triton_3bit/group_accum_baseline_gpu1_bf16.json
benchmark_artifacts/triton_3bit/ncu_group_accum_control_fp16_gpu0.ncu-rep
```

The public control medians were 0.055296 ms on GPU0 FP16 and 0.053248 ms on GPU1 BF16. NCU replay measured
19.65 us, 4739072 executed instructions, 32 registers/thread, 2.05 KiB static shared memory, zero spills, and a
256-CTA by 512-thread launch. It was instruction/dependency limited rather than bandwidth saturated.

#### Candidate progression and rejected tactics

All stages below were temporary native CUDA entry points. Each was checked against the existing kernel before its
timing was accepted, and every temporary operator and benchmark helper was removed from the production diff.

1. **Factor one group scale outside the 128-value accumulation.** Each split first accumulates signed 3-bit codes
   times FP16/BF16 activations into FP32, then applies its single FP16 scale. This is algebraically equivalent but
   changes FP32 association, so it was treated as a numerical change and tested rather than assumed exact. One
   accumulation chain improved matched raw latency by 6.37% on GPU0 FP16 and 3.95% on GPU1 BF16. NCU reduced
   instructions by 13.9%, but exposed more dependent-load stalls.
2. **Sweep one, two, and four FP32 accumulation chains.** Two chains removed some serial FMA dependency. Four chains
   produced no meaningful incremental gain: GPU0 FP16 means were 16.0862 us for two chains and 16.0601 us for four;
   GPU1 BF16 means were 16.9434 and 16.9427 us. The four-chain variant was rejected because its extra state and
   complexity did not clear a useful timing margin.
3. **Pair adjacent group splits inside each warp.** Lower and upper half-warps compute adjacent 128-value groups, a
   width-16 shuffle combines the two scaled partials, and only one 16-column row per warp enters shared memory. Warp
   zero reduces eight pair rows in each half and merges those halves with one final shuffle. A shared stride of 18
   keeps the two simultaneous reduction halves on disjoint banks. This reduced the runtime-N candidate to 15.0266 us
   on GPU0 FP16 and 15.6763 us on GPU1 BF16.
4. **Specialize N=4096 as well as K=4096.** Making the packed-weight, scale, and output strides compile-time constants
   removed residual dynamic address arithmetic. Matched alternating-order Nsight Systems means were 14.7742 us
   versus 17.2360 us control on GPU0 FP16 (14.28% faster), and 15.2977 us versus 17.7812 us on GPU1 BF16 (13.97%
   faster). The generic runtime-N form was rejected; only the measured exact shape is dispatched to the new kernel.
5. **Reject coarse whole-call CUDA-event A/B as relative proof.** An early alternating harness suggested implausible
   40-50% gains because allocator/host gaps and surrounding work dominated the tiny kernel. No claim uses those
   values. Matched raw Nsight kernel distributions and the standard public benchmark are the accepted evidence.

The main matched progression was:

```text
+----------------------+-----------------------------+-----------+-----------------------------+-----------+
| candidate            | GPU0 FP16 candidate/control | mean gain | GPU1 BF16 candidate/control | mean gain |
+----------------------+-----------------------------+-----------+-----------------------------+-----------+
| scale once, 1 chain  | 16.1500 / 17.2491 us       |     6.37% | 17.0976 / 17.8016 us       |     3.95% |
| scale once, 2 chains | 16.0862 / 17.2282 us       |     6.63% | 16.9434 / 17.6940 us       |     4.24% |
| warp-paired runtime N| 15.0266 / 17.2503 us       |    12.89% | 15.6763 / 17.6571 us       |    11.22% |
| warp-paired fixed N  | 14.7742 / 17.2360 us       |    14.28% | 15.2977 / 17.7812 us       |    13.97% |
+----------------------+-----------------------------+-----------+-----------------------------+-----------+
```

Experimental profiler artifacts retained for reproducibility include:

```text
benchmark_artifacts/triton_3bit/nsys_group_accum_ab_fp16_gpu0.nsys-rep
benchmark_artifacts/triton_3bit/nsys_group_accum_ab_bf16_gpu1.nsys-rep
benchmark_artifacts/triton_3bit/nsys_group_accum_c2_fp16_gpu0.nsys-rep
benchmark_artifacts/triton_3bit/nsys_group_accum_c2_bf16_gpu1.nsys-rep
benchmark_artifacts/triton_3bit/nsys_group_accum_c4_fp16_gpu0.nsys-rep
benchmark_artifacts/triton_3bit/nsys_group_accum_c4_bf16_gpu1.nsys-rep
benchmark_artifacts/triton_3bit/nsys_group_warp_reduce_fp16_gpu0.nsys-rep
benchmark_artifacts/triton_3bit/nsys_group_warp_reduce_bf16_gpu1.nsys-rep
benchmark_artifacts/triton_3bit/nsys_group_warp_fixed_n_fp16_gpu0.nsys-rep
benchmark_artifacts/triton_3bit/nsys_group_warp_fixed_n_bf16_gpu1.nsys-rep
benchmark_artifacts/triton_3bit/ncu_group_accum_candidate_fp16_gpu0.ncu-rep
benchmark_artifacts/triton_3bit/ncu_group_warp_reduce_candidate_fp16_gpu0.ncu-rep
```

#### Retained kernel and exact-production profiling

The retained kernel is still one launch. A 512-thread CTA owns 16 columns and 32 group splits. Every thread uses two
FP32 chains for the even/odd activation-code products in its group, then applies the group scale once. Sixteen warps
publish 16 paired-split rows to an 18-float-stride shared array. After one CTA barrier, warp zero performs the final
reduction, adds FP16 or BF16 bias, and stores the requested output dtype. Static shared memory falls from 2048 to
1152 bytes. The launch is selected only when `M=1`, `K=4096`, `N=4096`, and `split_k=32`; existing checks and paths
handle every other valid shape.

Exact-production full-trace captures used one NVTX range with 200 calls. `nvtx_kern_sum` found exactly 200 native
kernels in each range, proving that no helper launch was introduced:

```text
+------+-------+-----------+-----------+-----------+-----------+----------+
| GPU  | dtype | mean us   | median us | min us    | max us    | std ns   |
+------+-------+-----------+-----------+-----------+-----------+----------+
| 0    | FP16  |   15.2329 |    15.232 |    15.104 |    15.392 |     45.9 |
| 1    | BF16  |   15.3541 |    15.360 |    15.232 |    15.488 |     41.2 |
+------+-------+-----------+-----------+-----------+-----------+----------+
```

Exact-final-source NCU on physical GPU0 FP16 ties the gain to deleted work:

```text
+----------------------------------+----------------+----------------+-------------+
| metric                           | paired control | retained final | change      |
+----------------------------------+----------------+----------------+-------------+
| replay duration                  |       19.65 us |       17.44 us |     -11.25% |
| executed instructions            |        4739072 |        3661824 |     -22.73% |
| compute SOL                       |         41.67% |         40.75% |       lower |
| memory SOL                        |         23.45% |         25.13% |      higher |
| no eligible warp                 |         37.68% |         42.25% |       worse |
| long scoreboard / issue          |           2.89 |           3.62 |       worse |
| registers/thread                 |             32 |             32 |   unchanged |
| static shared memory             |       2.05 KiB |       1.15 KiB |     -43.90% |
| achieved occupancy               |         44.47% |         45.97% |  incidental |
| spills                           |              0 |              0 |   unchanged |
| grid x threads                   |      256 x 512 |      256 x 512 |   unchanged |
+----------------------------------+----------------+----------------+-------------+
```

The final kernel issued 315392 global-load requests and 729088 L1 sectors. Its padded final reduction recorded zero
shared-load bank conflicts; 198 sampled conflicts were on shared stores and represent only 0.58% shared-pipe
throughput. The increase in no-eligible and long-scoreboard stalls is important: occupancy did not cause this gain.
Removing arithmetic exposed the remaining packed-weight/load dependency, which is the next optimization signal.

Final profiler artifacts:

```text
benchmark_artifacts/triton_3bit/nsys_group_warp_final_fp16_gpu0.nsys-rep
benchmark_artifacts/triton_3bit/nsys_group_warp_final_fp16_gpu0.sqlite
benchmark_artifacts/triton_3bit/nsys_group_warp_final_bf16_gpu1.nsys-rep
benchmark_artifacts/triton_3bit/nsys_group_warp_final_bf16_gpu1.sqlite
benchmark_artifacts/triton_3bit/ncu_group_warp_final_fp16_gpu0.ncu-rep
```

#### Public-path performance, numerical quality, and safety

The standard benchmark ran 100 method warmups, 64 clock warmups, and 500 CUDA-event samples. GPTQ calls the native
kernel directly; AWQ reaches the same kernel after its existing exact one-time runtime repack. Absolute native
medians and throughput are:

```text
+------+-------+--------+-----------+-------------+--------+
| GPU  | dtype | layout | median ms | operations/s| p95 ms |
+------+-------+--------+-----------+-------------+--------+
| 0    | FP16  | GPTQ   |  0.051200 |      19531  | 0.0584 |
| 0    | FP16  | AWQ    |  0.053248 |      18780  | 0.0604 |
| 0    | BF16  | GPTQ   |  0.058368 |      17133  | 0.0727 |
| 0    | BF16  | AWQ    |  0.057344 |      17439  | 0.0686 |
| 1    | FP16  | GPTQ   |  0.056320 |      17756  | 0.0686 |
| 1    | FP16  | AWQ    |  0.054272 |      18426  | 0.0645 |
| 1    | BF16  | GPTQ   |  0.055296 |      18084  | 0.0635 |
| 1    | BF16  | AWQ    |  0.056320 |      17756  | 0.0645 |
+------+-------+--------+-----------+-------------+--------+
```

Complete standard result payloads are:

```text
benchmark_artifacts/triton_3bit/group_warp_final_gpu0_fp16.json
benchmark_artifacts/triton_3bit/group_warp_final_gpu0_bf16.json
benchmark_artifacts/triton_3bit/group_warp_final_gpu1_fp16.json
benchmark_artifacts/triton_3bit/group_warp_final_gpu1_bf16.json
```

A new exact-shape test uses random 3-bit codes, random FP16 group scales, and bias. It compares both FP16 and BF16
against (1) normal Torch matmul with dtype-rounded dequantized weights and (2) an independent FP32-dequantized,
FP32-accumulated oracle rounded only at output. Measured fixed-seed errors were:

```text
+-------+----------------+---------+------------+---------------+------------+
| dtype | reference      | max abs | mean abs   | relative RMSE | cosine     |
+-------+----------------+---------+------------+---------------+------------+
| FP16  | dtype Torch    | 0.125   | 0.00908189 | 0.000390731   | 1.00000000 |
| FP16  | FP32 oracle    | 0.015625| 0.00000736 | 0.000006045   | 0.99999988 |
| BF16  | dtype Torch    | 1.0     | 0.08621514 | 0.003360020   | 0.99999440 |
| BF16  | FP32 oracle    | 0.0     | 0.0        | 0.0           | 0.99999988 |
+-------+----------------+---------+------------+---------------+------------+
```

The dtype-Torch comparison is intentionally noisier because that reference rounds the dequantized weight before the
matmul. The kernel accumulates into FP32 and is substantially closer to the FP32 oracle. This also explains why a
BF16 mean can appear lower than FP16 for a particular seed: BF16's coarser output grid can round both results to the
same value; it is not evidence that BF16 has intrinsically more precise arithmetic.

The CUDA Graph test now uses the exact K=N=4096 specialization and passed bit-for-bit eager-versus-replay checks for
both dtypes. The complete focused file passed independently and concurrently on both physical devices:

```text
GPU0: 68 passed, 16 warnings in 26.80 s
GPU1: 68 passed, 16 warnings in 27.78 s
```

Compute Sanitizer then ran both primary FP16/BF16 reference cases. Memcheck on physical GPU0 and synccheck on
physical GPU1 each reported `ERROR SUMMARY: 0 errors`; racecheck on physical GPU0 reported
`0 hazards displayed (0 errors, 0 warnings)`. `ruff check tests/kernels/test_triton_3bit.py` and
`git diff --check` also pass.

Next signal: retain the exact-shape specialization and investigate packed-weight load/decode dependency scheduling.
The final NCU report says long scoreboard is now the largest stall component, while neither occupancy nor peak DRAM
bandwidth is the limiting target.

### 2026-07-22 — Packed-word software prefetch

Status: retained. The exact `M=1, K=N=4096, split_k=32` kernel now loads all twelve packed 32-bit words for one
group before starting its four 32-value decode/accumulate tiles. This is a scheduling change only: the 3-bit decode,
two FP32 accumulation chains, scale application, reduction order, bias, output dtype, public dispatch, and every
fallback remain unchanged.

The hardware inventory was probed again instead of assuming fixed CUDA indices. Physical GPU0 was an
`NVIDIA PG506-230` at PCI `25:00.0` (UUID `cb9e7784-cf50-203d-4f0d-5c622a89b1f2`); physical GPU1 was an
`NVIDIA PG506-232` at PCI `2b:00.0` (UUID `20f7fde4-d88c-d6ca-e324-bd4e5e9e0855`). Both exposed compute
capability 8.0, 124 SMs, and 96 GiB. The software stack was PyTorch 2.13.0+cu130, CUDA 13.0, NVCC 13.0.88,
Nsight Systems 2024.6.2, and Nsight Compute 2025.3.1.

#### Investigation and rejected tactics

Source-correlated NCU inspection accounted for the previous kernel's 315392 global-load requests as 77 requests per
warp: 64 activation pairs, 12 packed words, and one scale. Long-scoreboard samples were concentrated at activation
consumers and one packed-word decode dependency. The compiler had already unrolled and interleaved all four tiles,
so the next experiments targeted latency exposure rather than occupancy.

1. **Stage the complete 4096-element activation in shared memory — rejected.** Every 512-thread CTA cooperatively
   copied one aligned `int4` per thread into an 8 KiB shared array, synchronized once, and reused the staged values.
   The candidate was bit-identical to production, but the barrier and shared reads cost more than broadcast global
   loads. Matched Nsight Systems distributions were:

```text
+------+-------+------------+--------------+------------+
| GPU  | dtype | control us | candidate us | regression |
+------+-------+------------+--------------+------------+
| 0    | FP16  |    14.9311 |      15.0435 |      0.75% |
| 1    | BF16  |    15.4907 |      15.5526 |      0.40% |
+------+-------+------------+--------------+------------+
```

   A first launch of the temporary benchmark failed before measurement because the repository root was absent from
   `sys.path`; adding the path fixed the harness. No result from that setup failure was used. Because the corrected
   candidate lost on both devices, the shared-staging code and temporary operator were removed without heavy NCU
   replay. This also rejects a `cp.async` variant for this one-copy-per-thread shape: it cannot remove the required
   consumer barrier or the extra shared reads, and the synchronous staging family was already behind.

2. **Prefetch only the twelve packed words into registers — retained.** This leaves activation broadcasts in global
   memory and moves the four groups of three coalesced packed-word loads ahead of decode. The temporary A/B operator
   produced bit-for-bit identical FP16 and BF16 outputs. Reversed range ordering and 400 raw launches per method gave:

```text
+------+-------+------------+--------------+----------+
| GPU  | dtype | control us | candidate us | mean gain|
+------+-------+------------+--------------+----------+
| 0    | FP16  |    15.5590 |      15.4857 |    0.47% |
| 1    | BF16  |    15.5689 |      15.5429 |    0.17% |
+------+-------+------------+--------------+----------+
```

   The BF16 median was tied at the profiler's 32 ns display quantum, so the much larger NCU dependency movement was
   required before promotion. Focused 14-pass NCU reports compared control and candidate on each physical device:

```text
+--------------------------------+-------------------+-------------------+-------------------+-------------------+
| metric                         | GPU0 FP16 control | GPU0 FP16 prefetch| GPU1 BF16 control | GPU1 BF16 prefetch|
+--------------------------------+-------------------+-------------------+-------------------+-------------------+
| replay duration us             |            18.112 |            17.280 |            18.080 |            17.856 |
| executed instructions          |           3661568 |           3608320 |           3662592 |           3609344 |
| long scoreboard / issue        |            3.4811 |            3.0758 |            3.3159 |            2.9715 |
| registers/thread               |                32 |                32 |                32 |                32 |
| static shared memory KiB       |             1.152 |             1.152 |             1.152 |             1.152 |
| local spilling requests        |                 0 |                 0 |                 0 |                 0 |
+--------------------------------+-------------------+-------------------+-------------------+-------------------+
```

   Thus the candidate deleted about 1.45% of executed instructions and reduced long-scoreboard issue delay by 11.6%
   on GPU0 and 10.4% on GPU1 without spending registers or shared memory. This is dependency scheduling evidence,
   not an occupancy claim.

3. **Profiler command setup failure — corrected.** The first exact-final Nsys command used the unsupported
   `--stop-on-range-end=true` spelling and exited before launching the workload. Nsight Systems 2024.6.2 uses
   `--capture-range-end=stop`; the corrected CUDA Profiler API capture succeeded. No measurement was taken from the
   failed invocation.

#### Exact-production proof

After the experimental operator was removed, the prefetch was folded directly into the existing production kernel.
Exact-final-source Nsys captured exactly 400 kernels inside each NVTX range:

```text
+------+-------+-----------+-----------+-----------+-----------+--------+
| GPU  | dtype | mean us   | median us | min us    | max us    | std ns |
+------+-------+-----------+-----------+-----------+-----------+--------+
| 0    | FP16  |   14.7976 |    14.784 |    14.656 |    15.008 |   43.9 |
| 1    | BF16  |   15.3303 |    15.328 |    15.200 |    15.552 |   38.0 |
+------+-------+-----------+-----------+-----------+-----------+--------+
```

The earlier exact-production checkpoint recorded 15.2329 us FP16 and 15.3541 us BF16. Those absolute captures
corroborate the direction, but the matched temporary A/B table above remains the causal speedup evidence because GPU
clock state can differ between profiler sessions.

Fresh exact-final NCU reports measured 17.056 us and 3608576 instructions on GPU0 FP16, and 17.568 us and 3608576
instructions on GPU1 BF16. Both retained 32 registers/thread, 1.152 KiB static shared memory, and zero local spilling.
Long-scoreboard delay was 3.0696 and 3.1388 instructions per issued instruction respectively.

Profiler artifacts:

```text
benchmark_artifacts/triton_3bit/nsys_shared_input_fp16_gpu0.nsys-rep
benchmark_artifacts/triton_3bit/nsys_shared_input_bf16_gpu1.nsys-rep
benchmark_artifacts/triton_3bit/nsys_prefetch_words_fp16_gpu0.nsys-rep
benchmark_artifacts/triton_3bit/nsys_prefetch_words_bf16_gpu1.nsys-rep
benchmark_artifacts/triton_3bit/ncu_prefetch_control_fp16_gpu0.ncu-rep
benchmark_artifacts/triton_3bit/ncu_prefetch_candidate_fp16_gpu0.ncu-rep
benchmark_artifacts/triton_3bit/ncu_prefetch_control_bf16_gpu1.ncu-rep
benchmark_artifacts/triton_3bit/ncu_prefetch_candidate_bf16_gpu1.ncu-rep
benchmark_artifacts/triton_3bit/nsys_prefetch_final_fp16_gpu0.nsys-rep
benchmark_artifacts/triton_3bit/nsys_prefetch_final_bf16_gpu1.nsys-rep
benchmark_artifacts/triton_3bit/ncu_prefetch_final_fp16_gpu0.ncu-rep
benchmark_artifacts/triton_3bit/ncu_prefetch_final_bf16_gpu1.ncu-rep
```

#### Public-path performance, quality, and safety

The standard public benchmark used 64 clock warmups, 100 method warmups, and 500 CUDA-event samples. GPTQ and the
existing one-time AWQ repack both reached the same native kernel through `production-hybrid`:

```text
+------+-------+----------+----------+----------------+
| GPU  | dtype | GPTQ ms  | AWQ ms   | GPTQ/AWQ op/s  |
+------+-------+----------+----------+----------------+
| 0    | FP16  | 0.056320 | 0.057344 | 17756 / 17439  |
| 0    | BF16  | 0.058368 | 0.054272 | 17133 / 18426  |
| 1    | FP16  | 0.055296 | 0.056320 | 18084 / 17756  |
| 1    | BF16  | 0.053248 | 0.056320 | 18780 / 17756  |
+------+-------+----------+----------+----------------+
```

Complete payloads are `prefetch_final_gpu{0,1}_{fp16,bf16}.json` under
`benchmark_artifacts/triton_3bit/`.

The complete focused test file ran concurrently and independently on both devices: GPU0 passed all 68 tests in
53.10 s and GPU1 passed all 68 in 53.44 s. This includes FP16 and BF16 comparisons against the dtype-rounded Torch
3-bit reference and the independent FP32-dequantized/FP32-accumulated oracle, plus GPTQ/AWQ, CUDA Graph, stream,
save/reload, capability, and fallback coverage. The change only reorders loads, and all exact-specialization outputs
remained bit-identical to the prior kernel.

Compute Sanitizer then ran both primary FP16/BF16 reference cases. Memcheck on physical GPU0 and synccheck on
physical GPU1 each passed both cases and reported `ERROR SUMMARY: 0 errors`. `git diff --check` also passes.

Next signal: the register-only word prefetch is retained. Remaining NCU latency is still dominated by activation and
decode dependency stalls. A bounded early scale-load experiment is reasonable only if it preserves zero spills and
wins matched raw latency; whole-input shared staging must not be repeated.

### 2026-07-22 — Early group-scale load scheduling

Status: retained. The exact `M=1, K=N=4096, split_k=32` production kernel now issues its one FP16 group-scale load
after the twelve packed-word loads and before the 128-value decode/accumulate region. The scale stays live until the
two FP32 chains are combined. This moves one existing load without changing the number of loads, arithmetic,
accumulation/reduction order, launch geometry, shared-memory handoff, output dtype, dispatch gate, or fallbacks.

#### Candidate screening and profiler corrections

The temporary control instantiated the packed-word-prefetch kernel with its scale load after accumulation; the
candidate instantiated the same source with the load before accumulation. Both FP16 and BF16 candidates were
bit-for-bit identical to control. Eight alternating CUDA-event batches initially showed consistent movement:

```text
+-------+-----------------+-------------------+-----------------+-------------------+
| dtype | control mean us | candidate mean us | control p50 us  | candidate p50 us  |
+-------+-----------------+-------------------+-----------------+-------------------+
| FP16  |         15.7066 |           15.6273 |         15.6406 |           15.5853 |
| BF16  |         16.2875 |           16.2043 |         16.2826 |           16.2028 |
+-------+-----------------+-------------------+-----------------+-------------------+
```

These batches were only a screen. The first Nsys experiment put all 400 control launches in one range and all 400
candidate launches in another. GPU0 FP16 changed clock state between ranges and misleadingly reported 13.8241 us
control followed by 14.4058 us candidate, while the reverse-ordered GPU1 BF16 run reported 13.2035 us candidate and
13.2328 us control. Neither separated-range comparison is used as performance evidence.

The corrected experiment bracketed only the measurement with the CUDA Profiler API and alternated adjacent AB/BA
pairs inside one range. Exactly 500 launches of each variant were captured for every GPU/dtype combination. P95 was
queried from `CUPTI_ACTIVITY_KIND_KERNEL` durations in the Nsys-generated SQLite database using nearest-rank
`ceil(0.95 * 500)`:

```text
+------+-------+-------------------+---------------------+------------------+--------------------+------------------+--------------------+
| GPU  | dtype | control mean us   | candidate mean us   | control p50 us   | candidate p50 us   | control p95 us   | candidate p95 us   |
+------+-------+-------------------+---------------------+------------------+--------------------+------------------+--------------------+
| 0    | FP16  |           14.9657 |             14.8851 |           14.976 |             14.880 |           15.040 |             14.976 |
| 0    | BF16  |           15.5587 |             15.5287 |           15.552 |             15.520 |           15.648 |             15.616 |
| 1    | FP16  |           14.9102 |             14.8254 |           14.912 |             14.816 |           14.976 |             14.911 |
| 1    | BF16  |           15.4647 |             15.4187 |           15.456 |             15.424 |           15.552 |             15.488 |
+------+-------+-------------------+---------------------+------------------+--------------------+------------------+--------------------+
```

The mean gain is 0.54% and 0.19% on GPU0 FP16/BF16, and 0.57% and 0.30% on GPU1 FP16/BF16. More importantly for a
sub-percent candidate, mean, median, and p95 all improve in all four independent captures. The candidate temporary
operator and benchmark were removed after promotion.

The first targeted NCU attempt used the default `--kernel-name-base function`, which strips boolean template
arguments. Both variant filters therefore matched no kernel and NCU explicitly reported `No kernels were profiled`.
The reports were overwritten by a corrected run with `--kernel-name-base demangled --rename-kernels 0`; no metric
from the failed filter is used.

NCU confirms resource safety but does not corroborate the real-time latency direction:

```text
+-----------------------------+-------------------+-------------------+-------------------+-------------------+
| metric                      | GPU0 FP16 control | GPU0 FP16 early   | GPU1 BF16 control | GPU1 BF16 early   |
+-----------------------------+-------------------+-------------------+-------------------+-------------------+
| replay duration us          |            17.248 |            17.536 |            18.080 |            18.304 |
| executed instructions       |           3608320 |           3608320 |           3609344 |           3609344 |
| long scoreboard / issue     |            3.0890 |            3.1992 |            3.0298 |            3.3985 |
| registers/thread            |                32 |                32 |                32 |                32 |
| static shared memory KiB    |             1.152 |             1.152 |             1.152 |             1.152 |
| local spilling requests     |                 0 |                 0 |                 0 |                 0 |
+-----------------------------+-------------------+-------------------+-------------------+-------------------+
```

NCU replays a counter-heavy, cache-perturbed kernel and must not replace the adjacent-launch Nsys distribution.
The early load is retained because real mean/p50/p95 improve on both devices and dtypes, outputs are identical, and
the resource footprint is unchanged. It is not claimed to reduce long-scoreboard stalls; the replay counters moved
in the opposite direction and are recorded explicitly.

#### Exact-production proof

After removing the template switch and experimental API, exact-final-source Nsys captured one production launch per
call and exactly 400 calls per range:

```text
+------+-------+-----------+-----------+-----------+-----------+--------+
| GPU  | dtype | mean us   | median us | min us    | max us    | std ns |
+------+-------+-----------+-----------+-----------+-----------+--------+
| 0    | FP16  |   14.7116 |    14.720 |    14.592 |    14.848 |   47.5 |
| 1    | BF16  |   15.3123 |    15.296 |    15.200 |    15.584 |   41.3 |
+------+-------+-----------+-----------+-----------+-----------+--------+
```

The preceding packed-word-only exact-source capture was 14.7976 us FP16 and 15.3303 us BF16. This independent
absolute comparison follows the balanced A/B direction, while the four exactly balanced traces above remain the
causal evidence.

Fresh exact-final NCU reports measured 17.088 us on GPU0 FP16 and 17.728 us on GPU1 BF16. Both executed 3608576
instructions, used 32 registers/thread and 1.152 KiB static shared memory, and recorded zero local spilling requests.
The measured long-scoreboard ratios were 3.1667 and 3.3231 respectively.

Profiler artifacts:

```text
benchmark_artifacts/triton_3bit/nsys_prefetch_scale_balanced_fp16_gpu0.nsys-rep
benchmark_artifacts/triton_3bit/nsys_prefetch_scale_balanced_bf16_gpu0.nsys-rep
benchmark_artifacts/triton_3bit/nsys_prefetch_scale_balanced_fp16_gpu1.nsys-rep
benchmark_artifacts/triton_3bit/nsys_prefetch_scale_balanced_bf16_gpu1.nsys-rep
benchmark_artifacts/triton_3bit/ncu_prefetch_scale_control_fp16_gpu0.ncu-rep
benchmark_artifacts/triton_3bit/ncu_prefetch_scale_candidate_fp16_gpu0.ncu-rep
benchmark_artifacts/triton_3bit/ncu_prefetch_scale_control_bf16_gpu1.ncu-rep
benchmark_artifacts/triton_3bit/ncu_prefetch_scale_candidate_bf16_gpu1.ncu-rep
benchmark_artifacts/triton_3bit/nsys_scale_final_fp16_gpu0.nsys-rep
benchmark_artifacts/triton_3bit/nsys_scale_final_bf16_gpu1.nsys-rep
benchmark_artifacts/triton_3bit/ncu_scale_final_fp16_gpu0.ncu-rep
benchmark_artifacts/triton_3bit/ncu_scale_final_bf16_gpu1.ncu-rep
```

#### Public path, quality, and safety

The standard 64-clock-warmup, 100-method-warmup, 500-sample public benchmark confirmed that GPTQ and the exact
one-time AWQ repack both still reach `production-hybrid`:

```text
+------+-------+----------+----------+----------+----------+
| GPU  | dtype | GPTQ ms  | GPTQ p95 | AWQ ms   | AWQ p95  |
+------+-------+----------+----------+----------+----------+
| 0    | FP16  | 0.054272 | 0.066560 | 0.057344 | 0.076800 |
| 0    | BF16  | 0.053248 | 0.060416 | 0.055296 | 0.065536 |
| 1    | FP16  | 0.052224 | 0.066560 | 0.054272 | 0.069632 |
| 1    | BF16  | 0.058368 | 0.073728 | 0.057344 | 0.080896 |
+------+-------+----------+----------+----------+----------+
```

Complete payloads are `scale_final_gpu{0,1}_{fp16,bf16}.json` under
`benchmark_artifacts/triton_3bit/`.

The complete focused suite passed independently and concurrently after promotion: GPU0 reported 68 passed in
58.01 s and GPU1 reported 68 passed in 57.45 s. This revalidates FP16/BF16 against the dtype-rounded Torch 3-bit
reference and independent FP32 oracle, GPTQ/AWQ packing and routing, CUDA Graphs, streams, serialization, and all
fallback shapes. The temporary A/B outputs were bit-identical for both dtypes.

An initial post-promotion Compute Sanitizer invocation was externally interrupted immediately after printing only
the `COMPUTE-SANITIZER` header. No process remained and no partial result was used. The clean rerun passed both
primary FP16/BF16 oracle cases under GPU0 memcheck and GPU1 synccheck; each reported `ERROR SUMMARY: 0 errors`.
`git diff --check` also passes.

Next signal: retain both packed-word and scale scheduling. NCU still points to activation/decode dependencies, but
the failed whole-input shared staging rules out paying a CTA barrier for activation reuse. Any next candidate should
change decode instruction scheduling or bounded per-tile activation ILP, preserve the 32-register/no-spill envelope,
and clear the same adjacent-launch mean/p50/p95 standard on both dtypes.

### 2026-07-22 — Dtype-specialized adjacent-pair decode

Status: retained for FP16 only. BF16 remains on the preceding independent-code extraction because applying the same
source transformation to BF16 caused a repeatable regression on both physical devices. All experimental operators
and the temporary profiling script were removed before the final build.

The runtime probe again identified GPU0 as PG506-230 UUID `cb9e7784-cf50-203d-4f0d-5c622a89b1f2`, PCI
`0000:25:00.0`, and GPU1 as PG506-232 UUID `20f7fde4-d88c-d6ca-e324-bd4e5e9e0855`, PCI `0000:2b:00.0`.
Both devices reported compute capability 8.0, 124 SMs, and 96 GiB. The software remained PyTorch 2.13.0+cu130,
CUDA/nvcc 13.0 (`V13.0.88`), Nsight Systems 2024.6.2, and Nsight Compute 2025.3.1. The JIT build used
`compute_80` plus `sm_80`, `-O3`, C++17, line info, fast math, BF16 support, and fatbin compression.

#### Candidate and correctness

The original unrolled loop extracts the first and second 3-bit codes of a pair independently. The candidate instead
extracts the six contiguous bits beginning at `pair_k * 6`, joins the next packed word only for pair fields that
cross a 32-bit boundary, and obtains the second code by shifting the six-bit field by three. Packed-word loads,
activation loads, integer-to-float conversions, two FP32 FMA chains, group-scale multiplication, reduction order,
launch geometry, and dispatch gates are unchanged.

A temporary `matmul_pair_decode` operator instantiated control and candidate kernels in one extension. Random
`M=1, K=N=4096` outputs were bit-for-bit identical for FP16 and BF16 on both GPUs. With the same deterministic input,
candidate quality was:

```text
+-------+---------------+----------------+------------------+-------------+
| dtype | reference     | max abs        | mean abs         | relative RMSE|
+-------+---------------+----------------+------------------+-------------+
| FP16  | Torch FP16    | 0.125000       | 0.0090818852     | 0.000390731 |
| FP16  | FP32 oracle   | 0.015625       | 0.0000073612     | 0.000006045 |
| BF16  | Torch BF16    | 1.000000       | 0.0854774714     | 0.003285421 |
| BF16  | FP32 oracle   | 0.000030518    | 0.0000000075     | 0.000000010 |
+-------+---------------+----------------+------------------+-------------+
```

FP16/BF16 cosine similarity versus the FP32 oracle was `0.99999988` and `1.0`. The apparently larger BF16 error is
against the dtype-rounded Torch matmul reference; the native kernel remains much closer to the independent
FP32-dequantized, FP32-accumulated oracle. The pair transformation itself adds no error because candidate and control
are bit-identical.

An eight-batch adjacent CUDA-event screen was intentionally not used for promotion. Per-operation windows measured
roughly 32-34 us while Nsight Systems measured the actual kernel near 15 us, showing that the event windows contained
stream idle time while Python dispatched each operation. Outliers also changed the aggregate direction on GPU1.
This is a useful failure of the harness for a sub-percent raw-kernel change, not evidence about the candidate.

#### Balanced Nsight Systems decision

Each valid trace used 64 large-GEMM clock warmups and 100 warmups per variant, then captured 500 adjacent control and
candidate launches. Pair order alternated AB/BA, so every variant appeared exactly 500 times in the same range.
Nearest-rank p95 came from the raw `CUPTI_ACTIVITY_KIND_KERNEL` durations in the exported SQLite reports:

```text
+------+-------+-----------------+-------------------+---------------+-----------------+---------------+-----------------+-----------+
| GPU  | dtype | control mean us | candidate mean us | control p50 us| candidate p50 us| control p95 us| candidate p95 us| movement  |
+------+-------+-----------------+-------------------+---------------+-----------------+---------------+-----------------+-----------+
| 0    | FP16  |         14.8644 |           14.8211 |        14.848 |          14.816 |        14.944 |          14.880 | +0.2923%  |
| 1    | FP16  |         14.7914 |           14.7395 |        14.784 |          14.720 |        14.848 |          14.816 | +0.3521%  |
| 0    | BF16  |         15.5161 |           15.5709 |        15.520 |          15.584 |        15.616 |          15.648 | -0.3517%  |
| 1    | BF16  |         15.3888 |           15.4503 |        15.392 |          15.456 |        15.456 |          15.520 | -0.3981%  |
+------+-------+-----------------+-------------------+---------------+-----------------+---------------+-----------------+-----------+
```

The direction is symmetric across devices: every FP16 distribution improves in mean, median, and p95, while every
BF16 distribution regresses. Production therefore uses a compile-time scalar trait: FP16 extracts adjacent pairs as
one six-bit field, and BF16 retains two independent three-bit extractions. `if constexpr` removes the unused branch,
so there is no runtime dtype check inside the kernel.

#### Nsight Compute interpretation and failed profiler setup

Three NCU setup failures were corrected and excluded. Escaping `(bool)` in the first regex matched no kernels. The
next command requested `derived__local_spilling_requests` as a raw metric, which NCU rejected before collection.
Finally, a permissive `bool.*1` regex also matched control because `T1` appears later in the demangled signature.
The successful reports used `[(]bool[)]0` / `[(]bool[)]1` and direct local-load/store byte counters.

Focused FP16 reports showed that the source simplification changes scheduling, not dynamic instruction count:

```text
+-----------------------------+-------------------+-------------------+-------------------+-------------------+
| metric                      | GPU0 control      | GPU0 pair decode  | GPU1 control      | GPU1 pair decode  |
+-----------------------------+-------------------+-------------------+-------------------+-------------------+
| replay duration us          |            18.496 |            18.016 |            18.112 |            18.336 |
| executed instructions       |           3608320 |           3608320 |           3608320 |           3608320 |
| long scoreboard / issue     |            6.2560 |            3.9655 |            3.8724 |            4.0946 |
| registers/thread            |                32 |                32 |                32 |                32 |
| static shared memory KiB    |             1.152 |             1.152 |             1.152 |             1.152 |
| local load/store bytes      |               0/0 |               0/0 |               0/0 |               0/0 |
+-----------------------------+-------------------+-------------------+-------------------+-------------------+
```

NCU replay is mixed across GPUs, just as it was for early scale scheduling. It proves the resource envelope and lack
of spills but does not replace the adjacent-launch Nsys distributions. No claim is made that the candidate deletes
instructions or universally lowers long-scoreboard delay.

#### Exact-production proof

After removing the temporary API and profiling script, production-only Nsys captured one native kernel per call and
exactly 400 calls per range:

```text
+------+-------+----------+----------+----------+----------+--------+
| GPU  | dtype | mean us  | p50 us   | p95 us   | min us   | std ns |
+------+-------+----------+----------+----------+----------+--------+
| 0    | FP16  |  14.6643 |   14.656 |   14.720 |   14.528 |   42.2 |
| 1    | BF16  |  15.3644 |   15.360 |   15.424 |   15.231 |   39.1 |
+------+-------+----------+----------+----------+----------+--------+
```

Exact-final NCU measured 17.632 us on GPU0 FP16 and 18.464 us on GPU1 BF16. Both executed 3,608,576 instructions,
used 32 registers/thread and 1.152 KiB static shared memory, and recorded zero local load/store bytes. Long-scoreboard
ratios were 3.8719 and 3.9162 respectively.

The public 64-clock-warmup, 100-method-warmup, 500-sample benchmark confirmed GPTQ and the one-time AWQ repack still
route through `production-hybrid`:

```text
+------+-------+----------------+----------+---------------+---------+
| GPU  | dtype | GPTQ median ms | GPTQ p95 | AWQ median ms | AWQ p95 |
+------+-------+----------------+----------+---------------+---------+
| 0    | FP16  |       0.055296 | 0.069632 |      0.055296 | 0.075776|
| 1    | FP16  |       0.053248 | 0.061440 |      0.053248 | 0.065536|
| 0    | BF16  |       0.056320 | 0.069632 |      0.056320 | 0.065536|
| 1    | BF16  |       0.053248 | 0.064512 |      0.054272 | 0.064512|
+------+-------+----------------+----------+---------------+---------+
```

The complete focused suite passed independently and concurrently: GPU0 reported 68 passed in 30.27 s and GPU1
reported 68 passed in 29.54 s. This covers FP16/BF16 Torch 3-bit references and the independent FP32 oracle,
GPTQ/AWQ routing, current streams, CUDA Graph capture, save/reload, capability validation, and fallback shapes.
Compute Sanitizer then ran both primary FP16/BF16 oracle cases: GPU0 memcheck and GPU1 synccheck each passed both
tests and reported `ERROR SUMMARY: 0 errors`.

Profiler and benchmark artifacts:

```text
benchmark_artifacts/triton_3bit/nsys_pair_decode_balanced_fp16_gpu0.nsys-rep
benchmark_artifacts/triton_3bit/nsys_pair_decode_balanced_bf16_gpu0.nsys-rep
benchmark_artifacts/triton_3bit/nsys_pair_decode_balanced_fp16_gpu1.nsys-rep
benchmark_artifacts/triton_3bit/nsys_pair_decode_balanced_bf16_gpu1.nsys-rep
benchmark_artifacts/triton_3bit/ncu_pair_decode_control_fp16_gpu0.ncu-rep
benchmark_artifacts/triton_3bit/ncu_pair_decode_candidate_fp16_gpu0.ncu-rep
benchmark_artifacts/triton_3bit/ncu_pair_decode_control_fp16_gpu1.ncu-rep
benchmark_artifacts/triton_3bit/ncu_pair_decode_candidate_fp16_gpu1.ncu-rep
benchmark_artifacts/triton_3bit/nsys_pair_decode_final_fp16_gpu0.nsys-rep
benchmark_artifacts/triton_3bit/nsys_pair_decode_final_bf16_gpu1.nsys-rep
benchmark_artifacts/triton_3bit/ncu_pair_decode_final_fp16_gpu0.ncu-rep
benchmark_artifacts/triton_3bit/ncu_pair_decode_final_bf16_gpu1.ncu-rep
benchmark_artifacts/triton_3bit/pair_decode_final_gpu0_fp16.json
benchmark_artifacts/triton_3bit/pair_decode_final_gpu0_bf16.json
benchmark_artifacts/triton_3bit/pair_decode_final_gpu1_fp16.json
benchmark_artifacts/triton_3bit/pair_decode_final_gpu1_bf16.json
```

Next signal: the adjacent-pair extraction is closed as a dtype-specific tactic. A BF16 follow-up should preserve its
independent decode and test bounded activation-load scheduling within one 32-value tile; do not repeat whole-input
shared staging or apply the six-bit extraction globally.

### 2026-07-22 — Compile-time projection-width specializations

Status: retained for `M=1, K=4096, split_k=32` at `N=1024`, `4096`, `11008`, and `14336`. The `N=4096`
instantiation is the existing production fast path expressed through the new dispatch helper; the other three widths
are new. Every other shape continues to use the existing generic CTA, direct GEMV, or WMMA fallback.

The runtime probe identified physical GPU0 as PG506-230 UUID `cb9e7784-cf50-203d-4f0d-5c622a89b1f2`, PCI
`0000:25:00.0`, and GPU1 as PG506-232 UUID `20f7fde4-d88c-d6ca-e324-bd4e5e9e0855`, PCI `0000:2b:00.0`.
Both reported compute capability 8.0, 124 SMs, and 96 GiB. The software was PyTorch 2.13.0+cu130, CUDA/nvcc 13.0
(`V13.0.88`), driver 610.43.02, Triton 3.7.1, Nsight Systems 2024.6.2, and Nsight Compute 2025.3.1. The JIT build
used the repository defaults for `compute_80` plus `sm_80`, `-O3`, C++17, line info, BF16 support, fatbin
compression, and `--use_fast_math`.

#### Shape investigation and design

No local continuous-3-bit `K=4096` model was available to profile end to end. The local model configs instead
included hidden sizes 896, 5120, and 12288, so no claim is made that the selected widths came from a locally loaded
3-bit checkpoint. The experiment used representative attention/GQA and MLP projection widths and retained an exact
generic fallback for every unlisted `N`.

For `K=4096` with the measured `split_k=32`, each split consumes exactly 128 input values: one complete quantization
group. The already-retained `N=4096` mega-kernel therefore does not depend on that output width for arithmetic or
reduction. Only qweight/scale row strides and the number of 16-column CTAs depend on `N`. A `SizeN` template now
turns those strides and the grid size into compile-time constants. Dispatch attempts the four fixed instantiations
only under the exact `M=1, K=4096, split_k=32` gate and otherwise falls through unchanged.

A temporary `matmul_control` operator forced the old generic CTA kernel while production selected the candidate.
A temporary profiler issued adjacent AB/BA pairs in one capture range. Both were removed before the final build, so
neither the experimental schema nor a runtime control branch remains in production.

#### Baseline and timing-harness caveat

The pre-change public benchmark established that non-4096 widths reached native Trilin but used the generic CTA
reducer. Its Python CUDA-event windows were suitable for route and broad throughput checks, not raw-kernel proof.
One concurrently collected GPU0 FP16 `N=1024` run reported roughly 0.463 ms while the later same-device BF16 run and
the independent GPU1 FP16 run were near 0.055 ms. That clock/host-gap outlier is excluded. The baseline artifacts are:

```text
benchmark_artifacts/triton_3bit/projection_width_baseline_gpu0_fp16.json
benchmark_artifacts/triton_3bit/projection_width_baseline_gpu0_bf16.json
benchmark_artifacts/triton_3bit/projection_width_baseline_gpu1_fp16.json
benchmark_artifacts/triton_3bit/projection_width_baseline_gpu1_bf16.json
```

#### Correctness and the initial BF16 threshold failure

The first expanded test used the old `N=4096`-specific bounds at all widths. Two BF16 cases failed: `N=1024` had
mean absolute difference 0.11172 versus BF16 Torch matmul, just above the old 0.1 bound, while `N=14336` had one
0.5-absolute-error element versus the dtype-rounded FP32 oracle, above the old 0.25 max bound. This was investigated
before changing a tolerance.

Direct candidate/control and independent reference measurements showed no indexing or packing error. The candidate
remained far closer to the FP32-dequantized, FP32-accumulated oracle than BF16 Torch matmul in mean error and relative
RMSE. The isolated 0.5 value was a BF16 rounding-boundary event; its mean error was only `4.8522e-5`. The test now
covers all four widths for both dtypes, permits BF16's observed dtype-boundary max, and keeps strict mean, relative
RMSE, cosine, and “no worse than Torch versus the FP32 oracle” assertions.

```text
+-------+-------+----------------+------------------+---------------+
| dtype | N     | oracle max abs | oracle mean abs  | relative RMSE |
+-------+-------+----------------+------------------+---------------+
| FP16  | 1024  |     0.00000095 | 0.00000000093    | 0.00000000063|
| FP16  | 4096  |     0.01562500 | 0.00000736117    | 0.00000604505|
| FP16  | 11008 |     0.03125000 | 0.00001420982    | 0.00000935354|
| FP16  | 14336 |     0.06250000 | 0.00003261305    | 0.00002399512|
| BF16  | 1024  |     0.00000000 | 0.00000000000    | 0.00000000000|
| BF16  | 4096  |     0.00000000 | 0.00000000000    | 0.00000000000|
| BF16  | 11008 |     0.25000000 | 0.00003127719    | 0.00005149012|
| BF16  | 14336 |     0.50000000 | 0.00004852244    | 0.00009262392|
+-------+-------+----------------+------------------+---------------+
```

Across the same cases, FP16 mean difference versus dtype Torch matmul was `0.00905-0.01185` with relative RMSE at
most `4.47e-4`; BF16 mean difference was `0.08422-0.11172` with relative RMSE at most `3.78e-3`. Candidate and
control are not expected to be bit-identical: the specialized kernel sums one group before applying its scale while
the generic control distributes that scale through the per-value FMA sequence. Their largest observed difference was
0.0625 for FP16 and 0.25 for BF16, while the candidate's oracle metrics above remained substantially tighter.

#### Balanced Nsight Systems decision

Every trace used 64 large-GEMM clock warmups and 100 warmups of each variant. The measured range then contained 500
control and 500 candidate launches in alternating AB/BA order. P95 is nearest-rank p95 from raw CUPTI kernel
durations. All new widths improve mean, median, and p95 on both devices and dtypes:

```text
+------+-------+-------+------------------+--------------------+-----------------+-------------------+-----------+
| GPU  | dtype | N     | control mean/p50 | candidate mean/p50 | control/cand p95| mean speedup      | decision  |
+------+-------+-------+------------------+--------------------+-----------------+-------------------+-----------+
| 0    | FP16  | 1024  |  8.2974 /  8.288|   6.9408 /  6.944 |  8.352 /  7.008|            19.54% | retain    |
| 1    | FP16  | 1024  |  8.2588 /  8.256|   6.9034 /  6.912 |  8.320 /  6.944|            19.63% | retain    |
| 0    | BF16  | 1024  |  8.4760 /  8.480|   7.1603 /  7.168 |  8.513 /  7.200|            18.37% | retain    |
| 1    | BF16  | 1024  |  8.4485 /  8.448|   7.1191 /  7.104 |  8.512 /  7.168|            18.67% | retain    |
| 0    | FP16  | 11008 | 31.0701 / 31.072|  25.8368 / 25.824 | 31.200 / 25.952|            20.26% | retain    |
| 1    | FP16  | 11008 | 30.9635 / 30.944|  25.7207 / 25.728 | 31.072 / 25.824|            20.38% | retain    |
| 0    | BF16  | 11008 | 32.4462 / 32.448|  27.3294 / 27.328 | 32.576 / 27.424|            18.72% | retain    |
| 1    | BF16  | 11008 | 32.3512 / 32.351|  27.2391 / 27.232 | 32.448 / 27.328|            18.77% | retain    |
| 0    | FP16  | 14336 | 39.8831 / 39.872|  32.7953 / 32.800 | 40.032 / 32.896|            21.61% | retain    |
| 1    | FP16  | 14336 | 40.0405 / 40.032|  32.9198 / 32.927 | 40.192 / 33.024|            21.63% | retain    |
| 0    | BF16  | 14336 | 41.7425 / 41.728|  34.9169 / 34.912 | 41.888 / 35.008|            19.55% | retain    |
| 1    | BF16  | 14336 | 41.8787 / 41.856|  35.0391 / 35.040 | 42.048 / 35.136|            19.52% | retain    |
+------+-------+-------+------------------+--------------------+-----------------+-------------------+-----------+
```

Times in that table are microseconds. As a harness sanity check, forcing the generic control at `N=4096` reproduced
the existing fast path's advantage: 17.68-17.80% for FP16 and 15.90-16.04% for BF16. Those are not claimed as a new
`N=4096` gain.

#### Nsight Compute interpretation

One-pass focused NCU captures compared `N=11008` on GPU0 FP16 and GPU1 BF16. They corroborate the Nsys direction and
show why the group-specialized dataflow wins:

```text
+-----------------------------+-------------------+-------------------+-------------------+-------------------+
| metric                      | GPU0 FP16 control | GPU0 FP16 fixed   | GPU1 BF16 control | GPU1 BF16 fixed   |
+-----------------------------+-------------------+-------------------+-------------------+-------------------+
| replay duration us          |            34.848 |            29.248 |            36.480 |            30.816 |
| executed instructions       |          12734880 |           9645072 |          12736944 |           9647824 |
| long scoreboard / issue     |          2.703696 |          3.097590 |          2.616051 |          2.957885 |
| registers/thread            |                32 |                32 |                32 |                32 |
| static shared memory KiB    |             2.048 |             1.152 |             2.048 |             1.152 |
| local load/store bytes      |               0/0 |               0/0 |               0/0 |               0/0 |
| grid x threads              |         688 x 512 |         688 x 512 |         688 x 512 |         688 x 512 |
+-----------------------------+-------------------+-------------------+-------------------+-------------------+
```

Replay latency improves 16.07% for FP16 and 15.53% for BF16, while executed instructions fall about 24.3%. The
long-scoreboard-per-issued-instruction ratio rises 13-15%; the candidate issues much less total work, so this ratio's
denominator changes. This tactic is retained on latency, instruction count, lower shared state, and zero spilling,
not on a claim that long-scoreboard stalls fell.

#### Exact production, public routing, and safety

After deleting the temporary operator and profiler, exact-final-source Nsys captured exactly one native kernel per
call and 400 calls per range:

```text
+------+-------+-------+----------+----------+----------+----------+
| GPU  | dtype | N     | mean us  | p50 us   | p95 us   | min us   |
+------+-------+-------+----------+----------+----------+----------+
| 0    | FP16  | 1024  |   6.7877 |    6.784 |    6.848 |    6.720 |
| 0    | FP16  | 11008 |  25.8290 |   25.824 |   25.920 |   25.632 |
| 0    | FP16  | 14336 |  32.7746 |   32.768 |   32.864 |   32.576 |
| 1    | BF16  | 1024  |   7.0139 |    7.008 |    7.072 |    6.944 |
| 1    | BF16  | 11008 |  27.3880 |   27.392 |   27.488 |   27.136 |
| 1    | BF16  | 14336 |  34.8451 |   34.848 |   34.944 |   34.624 |
+------+-------+-------+----------+----------+----------+----------+
```

Every final launch used 32 registers/thread, 1.152 KiB static shared memory, and zero local memory. The public
64-clock-warmup, 100-method-warmup, 500-sample benchmark covered GPTQ and AWQ at all four widths, both dtypes, and
both devices. All 32 `production-hybrid` cases reached native Trilin. FP16 production medians were 0.0543-0.0563 ms
at `N=1024/4096`, 0.0563-0.0584 ms at `N=11008`, and 0.0625-0.0666 ms at `N=14336`. BF16 medians were
0.0543-0.0666, 0.0543-0.0614, 0.0573-0.0584, and 0.0625-0.0635 ms respectively. The exact Nsys distributions above
remain the kernel-latency evidence because the public per-operation event windows include Python dispatch gaps.

The complete focused suite then passed independently and concurrently against the exact final source: GPU0 reported
74 passed in 47.02 s and GPU1 reported 74 passed in 48.68 s. This includes FP16/BF16 Torch references and the
independent FP32 oracle at every fixed width, GPTQ/AWQ routing, streams, CUDA Graph capture, serialization, and
fallback shapes. Compute Sanitizer
covered both dtypes and narrow/wide specializations: GPU0 memcheck and GPU1 synccheck each passed two cases and
reported `ERROR SUMMARY: 0 errors`. Ruff and `git diff --check` also passed.

Profiler and benchmark artifacts:

```text
benchmark_artifacts/triton_3bit/nsys_projection_width_ab_{fp16,bf16}_n{1024,4096,11008,14336}_gpu{0,1}.nsys-rep
benchmark_artifacts/triton_3bit/ncu_projection_width_{control,candidate}_fp16_n11008_gpu0.ncu-rep
benchmark_artifacts/triton_3bit/ncu_projection_width_{control,candidate}_bf16_n11008_gpu1.ncu-rep
benchmark_artifacts/triton_3bit/nsys_projection_width_final_fp16_n{1024,11008,14336}_gpu0.nsys-rep
benchmark_artifacts/triton_3bit/nsys_projection_width_final_bf16_n{1024,11008,14336}_gpu1.nsys-rep
benchmark_artifacts/triton_3bit/projection_width_final_gpu{0,1}_{fp16,bf16}.json
```

Next signal: fixed-width reuse is now proven. The next high-leverage experiment is a launch-only Q/K/V or gate/up
mega-launch over independent projection descriptors, initially without cooperative launch, persistence, or a grid
barrier. It should retain the per-projection specialized CTA dataflow and be promoted only if a real model or a
representative adjacent-projection trace shows that removing launches beats the added descriptor/dispatch cost.

### 2026-07-22 — Fused gate/up-to-SwiGLU mega-kernel

Status: retained and installed for exact-shape Llama and Mistral decode. One native CUDA launch now replaces the two
Trilin 3-bit projection launches, eager SiLU launch, and eager multiply launch for `M=1, K=4096`,
`N in {11008,14336}`, FP16 or BF16, 3-bit group-size 128, `desc_act=False`, and `sym=True`. Every non-target
case retains the original MLP forward.

The runtime inventory was re-probed rather than inferred from a CUDA index. Physical GPU0 is PG506-230 UUID
`cb9e7784-cf50-203d-4f0d-5c622a89b1f2`, PCI `0000:25:00.0`; physical GPU1 is PG506-232 UUID
`20f7fde4-d88c-d6ca-e324-bd4e5e9e0855`, PCI `0000:2b:00.0`. Both are compute capability 8.0 with 124 SMs and
96 GiB. The stack is PyTorch 2.13.0+cu130, CUDA/nvcc 13.0 (`V13.0.88`), driver 610.43.02, Triton 3.7.1,
Nsight Systems 2024.6.2, and Nsight Compute 2025.3.1. The JIT command targeted `compute_80` and `sm_80` with
`-O3`, C++17, line info, BF16 support, fatbin compression, and `--use_fast_math`.

#### Kernel and integration design

The retained kernel keeps the proven fixed-width group-warp reducer and fuses only work with a direct data dependency:

1. A 512-thread CTA owns 16 output columns. Its 16 warps cover the 32 complete K=128 quantization groups.
2. Every lane preloads 12 packed gate words and 12 packed up words for its group and output column.
3. Each activation pair is loaded once and feeds four independent FP32 accumulators: gate even/odd and up even/odd.
4. Adjacent group splits are paired with warp shuffles. A `2 x 16 x 18` FP32 shared array holds gate and up partials;
   warp zero performs the ordered final reduction for both projections.
5. Gate and up are rounded to the activation dtype. SiLU is evaluated in FP32, rounded to the activation dtype, then
   multiplied by the rounded up value and rounded once more. This reproduces the eager dtype boundaries rather than
   silently changing the MLP's numerical contract.
6. Compile-time `SizeN` instantiations cover 11008 and 14336. The registered `gptqmodel_trilin::silu_mul` operator
   validates every dimension, dtype, layout, contiguity, and device boundary before launch and uses the current CUDA
   stream and device guard.

The production installer runs after quantized post-initialization and `model.eval()`. It accepts only LlamaMLP or
MistralMLP with `hidden_act=silu`, exact projection sizes, matching GPTQ Triton or AWQ Triton 3-bit projection types,
native Trilin readiness, no bias or adapter, same-device contiguous runtime tensors, and exact compute capability 8.0.
GPTQ consumes its serialized K-packed qweight; AWQ consumes its nonpersistent K-packed runtime repack. Runtime checks
require one row and FP16/BF16 on the installed device. Training, prefill, other dtypes, moved tensors, other model
families, other shapes, unsupported hardware, and unavailable native extensions all call the saved original forward.

The loader invokes this installer for the same exact 3-bit quantization contract regardless of whether the user
requested an explicit backend or `backend=auto`. Selection still chooses the existing validated GPTQ/AWQ Triton
QuantLinear; the hook only activates after those layers successfully prepare native Trilin tensors.

#### Investigation outcomes and harness corrections

The direct two-projection design compiled and passed its first numerical check. Nsight Compute later confirmed that
the 24 explicitly prefetched weight words did not cause the feared register spill: both profiled instantiations use
32 registers/thread and zero local spilling. The tactic was therefore retained rather than split into staged loads.

Three measurement caveats were investigated rather than treated as kernel results:

- Invoking the new profiler as `python scripts/profile_trilin_3bit_swiglu.py` failed before CUDA work because this
  checkout was not installed on that interpreter's script search path. All valid runs use
  `python -m scripts.profile_trilin_3bit_swiglu`; the failed import produced no timing sample.
- Nsight Systems exits the child with code 143 after `cudaProfilerStop` on this CUDA/Nsys combination, as in earlier
  captures. Both reports completed import and contain the exact requested operations, so the shutdown status is a
  capture-range caveat rather than a CUDA failure.
- The busy host produced isolated CUDA-event control maxima as high as 2-3 ms. Full distributions are reported, but
  acceptance uses the stable median/p95 direction plus raw Nsight kernel and projected-range attribution. One-launch
  NCU timing printed by the harness during replay is intentionally excluded because profiler replay perturbs it.

#### FP16/BF16 quality against both requested references

The profiler constructs the eager Torch 3-bit reference by unpacking both projections, multiplying in the requested
activation dtype, and applying eager SiLU/multiply. It separately constructs an FP32-dequantized, FP32-accumulated
oracle and rounds only the final output. Representative seeded results are:

```text
+-------+-------+-------------------------+-------------------------+-------------------------+-------------------------+
| dtype | N     | fused/control max,mean  | fused/Torch max,mean    | fused/oracle max,mean   | fused/oracle rrms       |
+-------+-------+-------------------------+-------------------------+-------------------------+-------------------------+
| FP16  | 11008 | 0.000488, 0.0000000444  |  2.0, 0.044092          |  2.0, 0.034440          | 0.00037696              |
| BF16  | 11008 | 0.000000, 0.0000000000  | 16.0, 0.475394          | 16.0, 0.298693          | 0.00317218              |
| FP16  | 14336 | 0.007812, 0.0000006131  |  2.0, 0.035174          |  2.0, 0.029149          | 0.00041707              |
| BF16  | 14336 | ~0.00000, ~0.0000000000 | 16.0, 0.369858          | 16.0, 0.226227          | 0.00324590              |
+-------+-------+-------------------------+-------------------------+-------------------------+-------------------------+
```

For the same rows, Torch-versus-oracle mean error was 0.053417, 0.516241, 0.043654, and 0.418770 respectively;
its relative RMSE was 0.00044421, 0.00387723, 0.00047368, and 0.00411153. The fused path is effectively identical
to the existing two-Trilin-projection control and is closer than the dtype Torch reference to the independent oracle
in every measured mean-error and relative-RMSE comparison. The tests retain direct fused-versus-Torch gates rather
than relying only on this better oracle result.

#### Balanced end-to-end CUDA-event matrix

Each final row uses 64 large GEMM clock warmups, 100 method warmups, four alternating control/candidate rounds, and
500 samples per method per round. Values are microseconds; speedup uses the stable median. The control is two public
Trilin calls plus eager SiLU and multiply, while the candidate is the public fused operator.

```text
+------+-------+-------+-----------------------+-----------------------+--------------+
| GPU  | dtype | N     | control mean/p50/p95  | fused mean/p50/p95    | p50 speedup  |
+------+-------+-------+-----------------------+-----------------------+--------------+
| 0    | FP16  | 11008 | 109.886/102.912/121.600| 54.159/54.016/56.064 | 1.905x       |
| 1    | FP16  | 11008 | 111.500/107.520/131.328| 53.054/53.248/54.528 | 2.019x       |
| 0    | BF16  | 11008 | 123.998/120.064/145.664| 51.430/50.944/52.480 | 2.356x       |
| 1    | BF16  | 11008 | 115.445/111.872/137.984| 51.385/50.944/54.016 | 2.196x       |
| 0    | FP16  | 14336 | 114.250/112.384/125.184| 67.273/66.560/70.912 | 1.688x       |
| 1    | FP16  | 14336 | 116.667/109.312/130.048| 66.898/66.816/69.632 | 1.636x       |
| 0    | BF16  | 14336 | 132.478/109.056/120.832| 66.811/66.560/67.840 | 1.638x       |
| 1    | BF16  | 14336 | 114.301/111.616/119.296| 65.632/65.792/67.328 | 1.696x       |
+------+-------+-------+-----------------------+-----------------------+--------------+
```

Every mean, median, and p95 improved. The GPU0 BF16 N=14336 control mean is visibly contaminated by its isolated
3.037 ms maximum; its p50, p95, and independent Nsight trace still agree on the retained direction.

#### Nsight Systems launch proof

Matched 300-call ranges show the exact operation removal:

```text
+------+-------+-------+-------------------------------+------------------+----------------------+---------+
| GPU  | dtype | N     | control kernel medians us     | fused kernel us  | projected us/call    | GPU ops |
+------+-------+-------+-------------------------------+------------------+----------------------+---------+
| 0    | FP16  | 11008 | 2x27.040 + 3.520 + 2.560     | 49.664           | 110.328 -> 51.032    | 4 -> 1  |
| 1    | BF16  | 14336 | 2x36.368 + 3.680 + 2.496     | 65.152           | 112.338 -> 66.526    | 4 -> 1  |
+------+-------+-------+-------------------------------+------------------+----------------------+---------+
```

The fused kernel is faster than the sum of the four control kernel medians itself, and removing three launches plus
two intermediate tensors widens the end-to-end gain. `nvtx_gpu_proj_sum` supplies the projected spans; the candidate
host NVTX push/pop range is shorter than its asynchronous GPU work and is not used as a latency measurement.

#### Nsight Compute resource and bottleneck evidence

Full 51-pass NCU replay captured one launch on each physical device:

```text
+----------------------------+-----------------------+------------------------+
| metric                     | GPU0 FP16 N=11008     | GPU1 BF16 N=14336      |
+----------------------------+-----------------------+------------------------+
| replay duration            | 53.22 us              | 66.08 us               |
| compute / memory SOL       | 70.98% / 43.34%       | 74.42% / 30.27%        |
| L1/TEX / L2 throughput     | 29.05% / 28.28%       | 30.86% / 30.49%        |
| registers/thread           | 32                    | 32                     |
| static shared memory       | 2.30 KiB              | 2.30 KiB               |
| local spill requests       | 0                     | 0                      |
| grid x block               | 688 x 512             | 896 x 512              |
| waves/SM                   | 1.39                  | 1.81                   |
| achieved occupancy         | 74.65%                | 82.15%                 |
| executed instructions      | 16,957,136            | 22,083,712             |
| long-scoreboard issue gap  | 4.17 inst             | 4.08 inst              |
+----------------------------+-----------------------+------------------------+
```

Occupancy is recorded as diagnostic metadata, not an optimization objective. The important result is that the fused
working set stays at 32 registers/thread with no spills while sustaining 71-74% compute SOL. Future work should target
instruction/dataflow reduction or a wider fusion boundary, not inflate launch resources to chase occupancy.

#### Final regression, graphs, streams, and sanitizer

The combined existing and new suites ran independently and concurrently against exact final source:

```text
GPU0: 83 passed, 16 warnings, 40.58 s
GPU1: 83 passed, 16 warnings, 40.66 s
```

New coverage includes both widths and dtypes against eager Torch 3-bit and FP32-oracle references, exact output dtype
and shape, current-stream execution, CUDA Graph capture, GPTQ/AWQ production routing, Llama/Mistral installation, and
the multi-row prefill fallback. Existing backend selection, native matmul, save/reload, packing, bias, BF16, and
fallback coverage remains green.

Compute Sanitizer ran the exact fused public op in parallel. GPU0 memcheck covered FP16 N=11008 and GPU1 memcheck
covered BF16 N=14336; GPU0 synccheck covered BF16 N=14336 and GPU1 synccheck covered FP16 N=11008. All four runs
reported `ERROR SUMMARY: 0 errors`. Ruff over every changed Python path and `git diff --check` also passed.

Profiler artifacts:

```text
benchmark_artifacts/triton_3bit/nsys_swiglu_fp16_n11008_gpu0.nsys-rep
benchmark_artifacts/triton_3bit/nsys_swiglu_fp16_n11008_gpu0.sqlite
benchmark_artifacts/triton_3bit/nsys_swiglu_bf16_n14336_gpu1.nsys-rep
benchmark_artifacts/triton_3bit/nsys_swiglu_bf16_n14336_gpu1.sqlite
benchmark_artifacts/triton_3bit/ncu_swiglu_fp16_n11008_gpu0.ncu-rep
benchmark_artifacts/triton_3bit/ncu_swiglu_bf16_n14336_gpu1.ncu-rep
```

Next signal: this completes the first useful cross-operator Trilin mega-kernel. A fully fused gate/up/down MLP would
require a cross-CTA dependency after SwiGLU and should not start with a device-wide barrier. The safer next experiment
is an exact-shape Q/K/V single launch that assigns independent CTA ranges to the three projections, retains the proven
per-projection reducer, and uses tensor views to avoid copying its combined output.

### 2026-07-22 — Fused Q/K/V projection mega-launch

Status: retained and installed for exact-shape Llama and Mistral decode. One native CUDA launch now replaces three
independent Trilin launches for `M=1`, `K=4096`, `Q=4096`, and `K/V in {1024,4096}`. The path covers FP16 and BF16,
GPTQ and AWQ, 3-bit group-size 128, `desc_act=False`, and `sym=True`. It uses one combined allocation and returns
zero-copy Q/K/V views. Every non-target contract executes the saved projection forwards.

The hardware and toolchain were probed again rather than inferred from fixed CUDA indices. Physical GPU0 was
PG506-230 at PCI `0000:25:00.0`; physical GPU1 was PG506-232 at PCI `0000:2b:00.0`. Both reported compute capability
8.0, 124 SMs, and 96 GiB. PyTorch was 2.13.0+cu130. The source compiled with nvcc 13.0.88 for `compute_80` and
`sm_80`, `-O3`, C++17, line info, BF16 support, fatbin compression, and `--use_fast_math`.

#### Baseline and fusion contract

The initial warmed CUDA-event baselines measured three public Trilin calls. These values include Python dispatch,
three output allocations, and queue gaps, so they established the end-to-end opportunity rather than kernel time:

```text
+------+-------+------+----------+----------+----------+----------+----------+
| GPU  | dtype | K/V  | mean us  | p50 us   | p95 us   | min us   | max us   |
+------+-------+------+----------+----------+----------+----------+----------+
| 0    | FP16  | 1024 |  112.359 |  108.032 |  124.160 |   92.672 |  609.024 |
| 1    | BF16  | 1024 |  125.933 |  109.568 |  120.832 |   88.064 | 3460.864 |
| 0    | BF16  | 4096 |  111.840 |  109.312 |  122.880 |  104.192 |  462.848 |
| 1    | FP16  | 4096 |  132.407 |  104.960 |  120.320 |   97.280 | 3541.760 |
+------+-------+------+----------+----------+----------+----------+----------+
```

The first formal Nsys baselines separated that range from device execution. GPU0 FP16 GQA used one 14.592 us Q
kernel and two 6.784 us K/V kernels, only 28.160 us of summed kernel medians inside a 115.606 us projected range.
GPU1 BF16 MHA used three 15.392 us kernels, 46.176 us summed, inside a 118.737 us projected range. Each logical call
created three GPU operations. This proved that a launch-only fusion could matter without introducing a cross-CTA
dependency.

The retained design deliberately does not use a grid barrier, cooperative launch, persistence, or global scratch:

1. The fixed-width projection reducer was factored into an inline device body without changing its arithmetic order.
2. A 512-thread CTA still owns 16 output columns. Its 16 warps cover all 32 complete `K=128` quantization groups,
   preload the same 12 packed words per lane, accumulate in FP32, pair adjacent splits with shuffles, and finish the
   ordered reduction through the same `16 x 18` FP32 shared array.
3. `blockIdx.x` makes a CTA-uniform selection of Q, K, or V tensors and maps the CTA to a disjoint segment of one
   combined output. The MHA specialization shares one `SizeN=4096` body after pointer selection; GQA compiles Q as
   4096 and K/V as 1024.
4. The launch grids are 384 CTAs for GQA and 768 CTAs for MHA. Q, K, and V are independent consumers of the same
   activation, so no phase can observe another phase and no synchronization is needed between CTA ranges.
5. The registered operator validates the exact input, qweight, scale, dtype, layout, contiguity, device, and width
   contract, guards the current device, launches on the current stream, and returns shape `(1, Q + K + V)`.

The first four-sample smoke run compiled successfully and produced the expected slices. It showed about 54 us versus
154 us median for FP16 GQA, but this sample was treated only as bring-up evidence. A separate two-sample BF16 GQA run
contained a 608 us candidate outlier and reported a false 0.41x mean speedup; the stable sample was 66.56 us. That
result was rejected as an under-sampled host outlier, and all acceptance decisions below use four alternating rounds,
500 samples per method, full distributions, and independent Nsight attribution.

#### FP16/BF16 quality against both requested references

The fused slices were bit-identical to three independent native Trilin calls for every Q/K/V projection, dtype, and
attention shape tested. The profiler additionally unpacked the actual continuous 3-bit qweights, formed
`(code - 4) * FP16 scale`, and evaluated both an activation-dtype Torch matmul and an independent FP32-dequantized,
FP32-accumulated oracle. The table reports the worst metric across Q, K, and V for each seeded case:

```text
+-------+------+-------------------------+---------------------------+---------------------------+
| dtype | K/V  | fused/control max,mean  | fused/Torch max,mean,rrms | fused/oracle max,mean,rrms |
+-------+------+-------------------------+---------------------------+---------------------------+
| FP16  | 1024 | 0, 0                    | .0625, .004761, .0003926  | .007813, 4.19e-6, 6.72e-6 |
| FP16  | 4096 | 0, 0                    | .0625, .002621, .0002787  | .015625, 1.25e-5, 1.62e-5 |
| BF16  | 1024 | 0, 0                    | .5000, .040163, .0031630  | .031250, 7.70e-6, 2.18e-5 |
| BF16  | 4096 | 0, 0                    | .5000, .029654, .0026775  | .000977, 2.38e-7, 7.25e-7 |
+-------+------+-------------------------+---------------------------+---------------------------+
```

As in the established projection tests, the BF16 Torch comparison is looser because BF16 matmul rounds internal
operands/accumulation differently. The independent FP32 oracle shows that the native FP32 accumulation remains much
closer than the BF16 Torch path. Unit tests use qweights produced by the repository's real 3-bit packer, retain direct
Torch gates, and separately require the native result to be at least as close to the oracle in mean error and relative
RMSE.

#### Balanced end-to-end latency matrix

Every row below used 64 large-GEMM clock warmups, 100 method warmups, four alternating control/candidate rounds, and
500 CUDA-event samples per method per round. Values are microseconds. The candidate includes the public operator and
three Python tensor views; the control includes three public native calls. Speedup uses the stable median.

```text
+------+-------+------+-------------------------+-------------------------+-------------+
| GPU  | dtype | K/V  | control mean/p50/p95    | fused mean/p50/p95      | p50 speedup |
+------+-------+------+-------------------------+-------------------------+-------------+
| 0    | FP16  | 1024 | 110.209/108.032/119.296 | 50.343/48.896/58.880   | 2.209x      |
| 1    | FP16  | 1024 | 119.863/111.104/126.976 | 54.644/49.152/59.648   | 2.260x      |
| 0    | BF16  | 1024 | 114.100/111.872/126.976 | 52.057/50.688/61.952   | 2.207x      |
| 1    | BF16  | 1024 | 109.517/107.520/122.368 | 49.516/48.384/57.088   | 2.222x      |
| 0    | FP16  | 4096 | 124.233/105.472/239.616 | 51.756/47.360/58.112   | 2.227x      |
| 1    | FP16  | 4096 | 107.844/105.472/119.552 | 49.607/48.128/55.808   | 2.191x      |
| 0    | BF16  | 4096 | 125.286/105.984/131.072 | 54.063/48.640/55.808   | 2.179x      |
| 1    | BF16  | 4096 | 118.506/107.008/121.344 | 50.807/48.896/57.856   | 2.188x      |
+------+-------+------+-------------------------+-------------------------+-------------+
```

All 24 mean, median, and p95 comparisons improve. The busy host still produced isolated control outliers, most visibly
in GPU0 FP16 MHA p95, but every stable distribution and both independent Nsys traces agree on the retained direction.

#### Nsight Systems launch and tail proof

Matched 300-call control/candidate ranges used the exact retained source:

```text
+------+----------+-----------------------------+------------------+-------------------------+---------+
| case | method   | kernel medians per call us  | projected us/call| logical GPU ops per call| decision|
+------+----------+-----------------------------+------------------+-------------------------+---------+
| GQA  | control  | 14.624 + 2 x 6.816 = 28.256| 111.717          | 3                       | baseline|
| GQA  | fused    | 18.432                      |  30.334          | 1                       | retain  |
| MHA  | control  | 3 x 15.392 = 46.176        | 113.355          | 3                       | baseline|
| MHA  | fused    | 30.975                      |  32.541          | 1                       | retain  |
+------+----------+-----------------------------+------------------+-------------------------+---------+
```

GQA was GPU0 FP16 and MHA was GPU1 BF16. The combined kernel is 1.53x and 1.49x faster than the sum of the separate
kernel medians themselves, in addition to removing two launches and two allocations. This is consistent with merging
three independently underfilled projection grids into one larger grid and paying the tail wave once. Nsys again
terminated the profiled child with status 143 after `cudaProfilerStop`; both reports completed import and contain the
requested 900-control/300-candidate operations, so the known shutdown behavior did not invalidate either capture.

#### Targeted Nsight Compute result

The initial SpeedOfLight classification was followed by ComputeWorkloadAnalysis only for the compute-heavy MHA case:

```text
+----------------------------+--------------------+---------------------+
| metric                     | GPU0 FP16 GQA      | GPU1 BF16 MHA       |
+----------------------------+--------------------+---------------------+
| replay duration us         | 20.45              | 33.28               |
| compute / memory SOL       | 51.75% / 31.91%    | 65.07% / 39.65%     |
| DRAM / L1 / L2 throughput  | 19.77/43.76/21.43% | 24.32/47.07/25.18%  |
| registers/thread           | 32                 | 32                  |
| static shared memory       | 1.152 KiB          | 1.152 KiB           |
| grid x block               | 384 x 512          | 768 x 512           |
| waves/SM                   | 0.77               | 1.55                |
| achieved occupancy         | 62.28%             | 78.81%              |
+----------------------------+--------------------+---------------------+
```

Occupancy is diagnostic metadata, not the optimization target. The material signal is that fusion retains the proven
32-register/1.152-KiB projection body and cuts actual kernel latency through better whole-grid scheduling. For MHA,
ComputeWorkloadAnalysis reports 64.34% ALU utilization, 56.04% issue slots busy, and 2.65 executed IPC while identifying
integer/logic work as the highest-utilized pipeline. The next inner-kernel experiment should therefore reduce packed
3-bit decode/index instructions; it should not inflate resources merely to raise occupancy.

#### Production routing, AUTO selection, and fallbacks

The loader installs the hook after quantized post-initialization and `model.eval()` whenever the quantization contract
is exactly 3-bit, group-size 128, non-act-order, symmetric. A live selector audit returned `TritonV2Linear` for GPTQ
and `AwqGEMMTritonLinear` for AWQ under `backend=AUTO`, for both FP16 and BF16. Explicit backend behavior is unchanged;
Trilin QKV is a post-initialization fast path inside those existing QuantLinear implementations, not a new public
backend or a priority change.

The installer accepts only exact `LlamaAttention`/`MistralAttention` modules, one consistent GPTQ or AWQ projection
class across Q/K/V, exact dimensions, no bias or adapter, native Trilin-ready state, contiguous same-device runtime
tensors, and runtime compute capability 8.0. GPTQ uses serialized K-packed qweights; AWQ uses its nonpersistent
K-packed runtime repack. Q projection execution launches the combined operator and attaches the K/V views to the
input tensor with a per-attention cache key. The normal Transformers 5.14.1 Llama/Mistral order is Q then K then V;
K retrieves its cached view and V retrieves then clears the cache. A changed call order remains correct because a
cache miss invokes the saved projection forward.

Training, prefill/multiple rows, noncontiguous input, other dtypes, wrong shapes, mixed projection classes, moved or
noncontiguous runtime tensors, unsupported model families, other architectures, bias/adapters, and missing native
state all preserve the original forwards. The native C++ operator itself allows compute capability >=8 for direct
testing, while production remains gated to the two profiled sm80 devices.

An extra broad generic selector sweep reported 64 passed, 17 skipped, and three failures in untouched baseline paths:
its independent bit/group pickers chose the invalid conditional AWQ combination `bits=3, group_size=16`, and two
`TorchQuantEmbeddings` cases resolved the shared Torch backend to `TorchLinear`. The implicated selection test and
QuantLinear sources have no diff in this checkpoint. The exact requested AUTO matrix above passed all four rows, and
the focused 3-bit capability, routing, save/reload, and inference tests passed. The generic sweep is logged rather than
misrepresented as a Trilin QKV regression.

#### Final regression and artifacts

The new QKV file passed nine tests independently on each physical GPU. The combined existing projection, SwiGLU, and
QKV suites then passed concurrently against the final source:

```text
GPU0: 92 passed, 16 warnings, 50.93 s
GPU1: 92 passed, 16 warnings, 50.67 s
```

Coverage includes real 3-bit packing, FP16/BF16 Torch and FP32-oracle quality, bit-exact three-launch equivalence,
zero-copy view offsets, current-stream execution, CUDA Graph capture, GPTQ/AWQ runtime tensors, GQA/MHA widths,
Llama/Mistral installation, cache cleanup, mixed-backend rejection, prefill fallback, existing bias paths, and
pack/save/reload. Compute Sanitizer ran in parallel: GPU0 memcheck FP16 GQA and synccheck BF16 MHA, plus GPU1 memcheck
BF16 MHA and synccheck FP16 GQA, all reported `ERROR SUMMARY: 0 errors`. Ruff and `git diff --check` passed.

Profiler artifacts:

```text
benchmark_artifacts/triton_3bit/qkv_ab_gpu{0,1}_{fp16,bf16}_kv{1024,4096}.log
benchmark_artifacts/triton_3bit/nsys_qkv_fp16_gqa_gpu0.{nsys-rep,sqlite}
benchmark_artifacts/triton_3bit/nsys_qkv_bf16_mha_gpu1.{nsys-rep,sqlite}
benchmark_artifacts/triton_3bit/ncu_qkv_fp16_gqa_gpu0.ncu-rep
benchmark_artifacts/triton_3bit/ncu_qkv_bf16_mha_gpu1.ncu-rep
benchmark_artifacts/triton_3bit/ncu_qkv_compute_bf16_mha_gpu1.ncu-rep
```

Next signal: preserve this launch fusion and investigate instruction/dataflow reduction inside the 3-bit decoder.
The already-rejected BF16 adjacent-pair tactic should not be repeated unchanged. A separate higher-level candidate is
Q/K rotary fusion, but only after profiling the actual reshape/rotary boundary and preserving cache and graph fallbacks.

## 2026-07-22: generalized group-size support with measured degradation

### Goal and retained fast contract

This iteration checked whether the production GPTQ/AWQ 3-bit path can accept the other group sizes already declared by
the corresponding quantization formats without weakening the optimized `group_size=128`, `desc_act=False`, `sym=True`
contract. It can. Group 128 remains the only native continuous-3-bit Trilin and mega-kernel contract; the other sizes
use correctness-preserving, slower routes selected at post-initialization and forward time.

```text
+----------------------------+----------------------+-------------------------------+
| quantization contract      | FP16 production path | BF16 production path          |
+----------------------------+----------------------+-------------------------------+
| group 128, M <= 16         | native Trilin        | native Trilin                  |
| group 128, M > 16          | expanded Marlin      | fused Triton                   |
| group 32 or 64             | expanded Marlin      | generalized fused Triton       |
| channelwise (-1 / K)       | expanded Marlin      | generalized fused Triton       |
| group 16                   | generalized Triton   | generalized fused Triton       |
| group 256/512/1024 (GPTQ)  | generalized Triton   | generalized fused Triton       |
+----------------------------+----------------------+-------------------------------+
```

GPTQ now admits its complete declared set `-1,16,32,64,128,256,512,1024`; AWQ admits its complete declared set
`-1,16,32,64,128`. `backend=AUTO` exact-selector tests choose `TritonV2Linear` for every GPTQ row and
`AwqGEMMTritonLinear` for every AWQ row. The group-128 QKV/SwiGLU installers retain their exact group guard and do not
silently attach fixed-group mega-kernels to a generalized layer.

### Investigation and implementation record

Successful tactics:

- Separated the 32-value packing alignment from the 128-value optimized scale contract. Continuous 3-bit packing only
  requires K divisible by 32, so AWQ-to-GPTQ repacking and the fused input validator can safely accept smaller groups.
- Made group size a Triton compile-time specialization and replaced the fixed `offsets_k // 128` scale lookup with
  `offsets_k // GROUP_SIZE`. The emitted kernel remains one launch and supports both continuous GPTQ and AWQ layouts.
- Normalized public `group_size=-1` to K at runtime. This also fixed AWQ buffer allocation for channelwise checkpoints;
  the constructor must use normalized `self.group_size`, not the raw `-1` argument, for qzero/scale shapes.
- Passed the real group size through Torch dequantization, native-cache preparation, benchmark construction, and
  production forward dispatch. Scale-shape validation now reports the effective group contract explicitly.
- Probed expanded Marlin instead of assuming its advertised 4-bit group matrix applies to expanded 3-bit values.
  On GPU0 at M=1,K=N=4096, groups 32 and 64 and channelwise K=4096 were correct; the group-16 and group-256 launches
  rejected their configurations. The guarded cache therefore allows only 32/64/128/channelwise and falls back cleanly.

Rejected or failed tactics:

- Directly sending group 16 or group 256 through expanded Marlin failed its runtime configuration checks. These sizes
  are not marked native-eligible even though the 3-bit-to-uint4b8 value expansion itself is exact.
- Generalizing the native Trilin reducers or the QKV/SwiGLU mega-kernels in this iteration was rejected: those kernels
  embed group-128 scale scheduling, split-K reduction assumptions, and measured launch geometry. Reinterpreting their
  metadata would risk silent wrong results and could regress the primary path. They remain explicitly group-128-only.
- Initial FP16 quality gates of mean absolute error 0.02 versus eager Torch and 0.01 versus the independent FP32
  dequant/accumulate oracle were too strict for expanded Marlin's different reduction order at K=2048. Observed means
  were about 0.027-0.031 while relative RMSE remained 0.00146-0.00163 and cosine exceeded 0.999998. The final tests use
  mean 0.04 plus max, relative-RMSE, and cosine gates; this is tighter and more informative than accepting by max alone.

### Correctness, safety, and compatibility proof

The raw generalized Triton matrix covered GPTQ groups `16,32,64,128,256,512,1024,-1` on GPU0 and AWQ groups
`16,32,64,128,-1` on GPU1 in both FP16 and BF16. Every case matched an independently expanded FP32 weight/reference;
the smoke matrix observed FP16 max error no greater than 0.015625 and BF16 max error no greater than 0.5.

Production QuantLinear tests use K=2048 and compare every newly admitted group against both an eager Torch 3-bit
unpack/dequant path and a separate FP32 dequantization/accumulation oracle. They check shape, dtype, finiteness, max and
mean absolute error, relative RMSE, and cosine similarity. The 14 GPTQ rows and eight AWQ rows passed across FP16/BF16.
Additional tests cover current-stream execution, CUDA Graph replay, and pack/save/reload for group 64, group 128, and
channelwise. The existing group-128 QKV/SwiGLU mega-kernel suite also remained green.

Compute Sanitizer memcheck ran three boundary routes on GPU0: group-16 FP16 Triton, group-64 BF16 AWQ/Triton, and
group-64 FP16 expanded Marlin. Every run reported `ERROR SUMMARY: 0 errors`. AUTO selection's 13 exact group rows passed.
The broader selector file still has three unrelated baseline failures: one generic picker constructs the incompatible
AWQ tuple `bits=3, group_size=16, desc_act=True`, and two `TorchQuantEmbeddings` cases resolve a shared Torch backend to
`TorchLinear`. These are recorded rather than presented as generalized-Trilin regressions.

### Latency and throughput degradation

Hardware was re-probed rather than inferred from device indices. Physical GPU0 is a PG506-230 at PCI `0000:25:00.0`;
physical GPU1 is a PG506-232 at PCI `0000:2b:00.0`. Both report sm80, 124 SMs, and 96 GiB. The software stack was
PyTorch 2.13.0+cu130, CUDA 13.0, Triton 3.7.1, and Python 3.14.5t. Benchmarks used M=1,K=N=4096, eight clock-warmup
GEMMs, 20 kernel warmups, and 40 CUDA-event samples. Values below are production-hybrid median latency/TFLOP/s;
`slowdown` is relative to the same GPU/layout/dtype group-128 row.

```text
+------+--------+-------+------------+----------+----------+
| GPU  | layout | dtype | group      | p50 ms   | slowdown |
+------+--------+-------+------------+----------+----------+
| 0    | GPTQ   | FP16  | 128        | 0.057344 | 1.00x    |
| 0    | GPTQ   | FP16  | 32 / 64    | 0.086016 / 0.080896 | 1.50x / 1.41x |
| 0    | GPTQ   | FP16  | -1         | 0.078848 | 1.38x    |
| 0    | GPTQ   | FP16  | 16         | 0.124928 | 2.18x    |
| 0    | GPTQ   | FP16  | 256/512/1024 | 0.122880/0.124416/0.124928 | 2.14x/2.17x/2.18x |
| 1    | AWQ    | FP16  | 128        | 0.061440 | 1.00x    |
| 1    | AWQ    | FP16  | 32 / 64 / -1 | 0.078848/0.079872/0.079872 | 1.28x/1.30x/1.30x |
| 1    | AWQ    | FP16  | 16         | 0.124928 | 2.03x    |
| 0    | GPTQ   | BF16  | 128        | 0.054272 | 1.00x    |
| 0    | GPTQ   | BF16  | all non-128| 0.119808-0.124928 | 2.21x-2.30x |
| 1    | AWQ    | BF16  | 128        | 0.056320 | 1.00x    |
| 1    | AWQ    | BF16  | all non-128| 0.120832-0.140288 | 2.15x-2.49x |
+------+--------+-------+------------+----------+----------+
```

At larger M, the penalty narrows because both paths converge on throughput kernels. GPU0 GPTQ FP16 group 64 versus
128 was 0.079872 versus 0.070656 ms at M=16 (1.13x) and exactly 0.089088 ms for both at M=128. GPU1 AWQ BF16 was
0.124928 versus 0.078848 ms at M=16 (1.58x), but 0.122880 versus 0.128000 ms at M=128 (effective parity/no regression).

### Nsight Systems launch proof

Nsight Systems 2024.6.2 capture-range profiles used ten production calls per route at M=1,K=N=4096. Every case remains
one logical GPU operation per call; the degradation is kernel work/routing, not an accidental materialize-plus-GEMM or
extra-launch fallback.

```text
+------+-------+-----------------+------------------------+-------------------+
| GPU  | group | production path | median kernel time us  | projected us/call |
+------+-------+-----------------+------------------------+-------------------+
| 0    | 64    | FP16 Marlin     | 14.560                 | 59.43             |
| 1    | 128   | FP16 Trilin     | 14.720                 | 65.41             |
| 0    | 16    | FP16 Triton     | 69.184                 | 89.65             |
| 1    | 64    | BF16 Triton     | 69.279                 | 95.55             |
+------+-------+-----------------+------------------------+-------------------+
```

The CUDA-event distributions are the decision source because they include synchronized end-to-end operator timing;
the Nsys captures prove kernel identity and launch count. Busy-host p95 outliers occurred for GPTQ FP16 groups 32 and
256 and BF16 group 1024, so no routing choice was made from those tails.

Artifacts:

```text
benchmark_artifacts/triton_3bit/groups_{fp16,bf16}_{gptq_gpu0,awq_gpu1}.json
benchmark_artifacts/triton_3bit/groups_prefill_{fp16_gpu0,bf16_gpu1}.json
benchmark_artifacts/triton_3bit/nsys_group64_fp16_marlin_gpu0.{nsys-rep,sqlite}
benchmark_artifacts/triton_3bit/nsys_group128_fp16_native_gpu1.{nsys-rep,sqlite}
benchmark_artifacts/triton_3bit/nsys_group16_fp16_triton_gpu0.{nsys-rep,sqlite}
benchmark_artifacts/triton_3bit/nsys_group64_bf16_triton_gpu1.{nsys-rep,sqlite}
```

The final focused source passed `tests/kernels/test_triton_3bit.py` independently and concurrently on both devices:
GPU0 reported 151 passed in 60.74 seconds and GPU1 reported 151 passed in 61.53 seconds (16 dependency warnings each).

Decision: ship generalized correctness with explicit, measured degradation. A future optimization should specialize
native continuous-3-bit scale scheduling for group 32/64 first; those sizes are common and already show that native
routing reduces decode latency. Group 16 and the large GPTQ groups remain valid but lower-priority Triton fallbacks.

## 2026-07-22 — Native CUDA specialization for every positive group size

Status: implemented, benchmarked, memory-checked, and profiled on both physical sm80 devices. The retained production
set is now `16,32,64,96,128,192,256,384,512,1024`; channelwise `-1` deliberately remains on the existing fallback.

### Main merge and public contract

`origin/main` advanced to `a00fb759` (`Add extended GPTQ and AWQ group sizes (#33)`). It was merged into this branch as
`342b6b21`. Conflicts in `tritonv2.py`, `gemm_awq_triton.py`, and `test_selection.py` were resolved by retaining the
3-bit Trilin route while taking main's expanded group declarations and both selector test families. The merge commit
was pushed and PR #29 was updated before this optimization began.

Main added GPTQ group 384 and AWQ groups 256/384/512. This iteration additionally makes group 96 and group 192 public
for GPTQ and AWQ configuration, Torch fallback, Triton selection, and AWQ's 4-bit Triton helper. Config serialization
round trips preserve both values, and `backend=AUTO` selects the intended Triton QuantLinear for every declared 3-bit
group. Explicit groups must divide K; incompatible shapes continue to reject or fall back rather than reinterpret
scale metadata.

`prepare_trilin_3bit` now emits `log.warn.once(...)` for every requested positive group other than 128. The message
contains the group value, so repeated layers of the same group warn once while different non-primary groups each get
one useful warning. Group `-1` is exempt because it does not enter the native positive-group route. The warning states
that group 128 remains the fully optimized contract and that other groups can have higher latency.

### Retained CUDA design

The Torch operation now receives an explicit `group_size`, with a schema default of 128 for raw-call compatibility.
Python forwards the original requested value so `-1` is never confused with a normalized channel count. A dedicated
native eligibility check is separate from expanded Marlin's narrower `32/64/128/channelwise` contract.

The native kernels compile group size into each CUDA specialization:

- WMMA and generic GEMV select scale row `tile_k / GroupSize` for every 32-value packed tile.
- Group 16 loads two scales per packed tile and accumulates each 16-value half independently. Applying one scale to
  all 32 decoded values would be numerically wrong and was not used.
- Groups 32 and 64 keep the K=4096 decode warp reducer but apply four or two scales, respectively, inside each
  128-value split.
- Groups 128/256/384/512/1024 reuse unscaled 128-value partials when group boundaries align. The scale is applied after
  the partial, avoiding per-value scale multiplication.
- Groups 96 and 192 do not divide, or form a multiple of, the reducer's 128-value split. Their template instantiations
  therefore compile out the warp-reducer call and use the generic native CTA reducer. At K=3072 its 192-value split is
  group-aligned for both values.
- The production wrapper requires CUDA capability >= 8.0, K divisible by 128, N divisible by 64, a supported group
  that divides K, FP16 scales, and FP16/BF16 activation. Other shapes and channelwise `-1` retain Marlin/Triton/Torch
  fallbacks. QKV and SwiGLU mega-kernels remain explicitly group-128-only.

The final JIT cache key was `87a080291724d341`. The detected-device build used NVCC 13.0.88 with
`-gencode=arch=compute_80,code=compute_80`, `-gencode=arch=compute_80,code=sm_80`, `-O3`, `--use_fast_math`,
`-lineinfo`, `--threads 8`, BF16 enabled, and C++17. No fixed CUDA device index is used by runtime capability checks.

### Performance decision

Hardware was re-probed immediately before testing. Physical GPU0 was NVIDIA PG506-230, PCI `0000:25:00.0`, sm80,
124 SMs, 96 GiB; physical GPU1 was NVIDIA PG506-232, PCI `0000:2b:00.0`, sm80, 124 SMs, 96 GiB. The stack was
PyTorch 2.13.0+cu130, CUDA 13.0, Triton 3.7.1, Python 3.14.5t, Nsight Systems 2024.6.2, and Compute Sanitizer
2025.3.1. Timings used eight clock-warmup GEMMs, 20 operator warmups, 40 CUDA-event samples, and the median as the
routing decision statistic.

Groups 32 and 64 were measured against an exact pre-change checkpoint. FP16 previously selected expanded Marlin;
BF16 previously selected Triton.

```text
+------+--------+-------+-------+----------------+---------------+---------+
| GPU  | layout | dtype | group | old p50 ms     | new p50 ms    | speedup |
+------+--------+-------+-------+----------------+---------------+---------+
| 0    | GPTQ   | FP16  | 32    | 0.079872       | 0.058368      | 1.37x   |
| 0    | GPTQ   | FP16  | 64    | 0.080384       | 0.056320      | 1.43x   |
| 0    | GPTQ   | BF16  | 32    | 0.119808       | 0.058368      | 2.05x   |
| 0    | GPTQ   | BF16  | 64    | 0.123904       | 0.058368      | 2.12x   |
| 1    | AWQ    | FP16  | 32    | 0.079872       | 0.061440      | 1.30x   |
| 1    | AWQ    | FP16  | 64    | 0.080896       | 0.057344      | 1.41x   |
| 1    | AWQ    | BF16  | 32    | 0.124928       | 0.057344      | 2.18x   |
| 1    | AWQ    | BF16  | 64    | 0.124416       | 0.059392      | 2.09x   |
+------+--------+-------+-------+----------------+---------------+---------+
```

The remaining positive groups were compared with their same-run fused Triton fallback. K=4096 was used for groups
16/256/512/1024; K=3072 was used for groups 96/192/384. All retained routes are material wins rather than merely
correct alternatives.

```text
+------+--------+-------+-------------------------+--------------------+--------------------+---------------+
| GPU  | layout | dtype | groups                  | Triton p50 range ms| native p50 range ms| speedup range |
+------+--------+-------+-------------------------+--------------------+--------------------+---------------+
| 0    | GPTQ   | FP16  | 16,96,192,256,384,512,1024 | 0.1213-0.1275 | 0.0553-0.0594 | 2.04-2.25x |
| 0    | GPTQ   | BF16  | 16,96,192,256,384,512,1024 | 0.1208-0.1326 | 0.0543-0.0594 | 2.09-2.25x |
| 1    | AWQ    | FP16  | 16,96,192,256,384,512  | 0.1219-0.1285      | 0.0553-0.0614      | 2.05-2.24x   |
| 1    | AWQ    | BF16  | 16,96,192,256,384,512  | 0.1234-0.1295      | 0.0573-0.0625      | 2.02-2.25x   |
+------+--------+-------+-------------------------+--------------------+--------------------+---------------+
```

Busy-host mean/p95 outliers appeared in isolated rows, including GPTQ FP16 group 192 and AWQ FP16 group 96. They did
not move the routing decision: medians were stable across the matched runs, and every promoted group had approximately
a 2x fallback gap except the already faster expanded-Marlin FP16 group-32/64 baselines.

### Numerical, lifecycle, and safety proof

The final native matrix covers every positive group for M=1 and M=3, FP16 and BF16, on both GPUs. It compares native
output plus FP16 bias with an independently decoded FP32 dequantization/accumulation oracle. Passing gates were:

```text
+-------+---------+----------+---------------+----------+
| dtype | max abs | mean abs | relative RMSE | cosine   |
+-------+---------+----------+---------------+----------+
| FP16  | <=0.125 | <=0.002  | <=0.0002      | >=0.999999 |
| BF16  | <=0.5   | <=0.01   | <=0.001       | >=0.99999  |
+-------+---------+----------+---------------+----------+
```

The benchmark's dense-dtype comparison observed FP16 max absolute error at most 0.0625 and BF16 at most 1.0; the
larger BF16 display-unit difference is expected from BF16 rounding and is bounded independently by the FP32-oracle
tests above. FP16 is not less accurate than BF16 under the independent metrics.

Final validation:

- `tests/kernels/test_triton_3bit.py`: 229 passed independently on GPU0 in 144.47 s and GPU1 in 143.99 s.
- GPTQ Torch/Triton 4-bit positive-group quality: 12 passed, including 96/192 in FP16 and BF16.
- AWQ Torch/Triton 4-bit parity and accumulation: 17 passed, including 96/192 at sequence lengths 1/8/129.
- Config round trip and relevant AUTO selector matrix: 41 passed, including exact 3-bit selection for every group.
- Existing group-128 QKV and SwiGLU mega-kernel suites: 9 passed each on separate GPUs.
- Compute Sanitizer memcheck: group-16 FP16 warp reducer on GPU0 and group-96 BF16 CTA reducer on GPU1 both reported
  `ERROR SUMMARY: 0 errors`.

The broad selector smoke suite still reports three unrelated baseline failures: its generic picker combines AWQ
3-bit with `desc_act=True`, and two shared Torch backend cases expect `TorchQuantEmbeddings` where selection returns
`TorchLinear`. The focused selector rows affected by this change all pass; these baseline mismatches were not hidden
or reclassified as group-size regressions.

### Nsight launch proof

Nsight Systems used CUDA Profiler API capture ranges around 20 warmed native calls on each device. The reports prove
the intended kernel identity and exactly one GPU operation per logical matmul call.

```text
+------+-------+-------+------------+--------------------------------------+-----------+-----------+
| GPU  | group | dtype | shape      | native kernel                        | instances | median us |
+------+-------+-------+------------+--------------------------------------+-----------+-----------+
| 0    | 64    | FP16  | 1x4096x4096 | group_warp_reduce<half,4096,64>     | 20        | 14.784    |
| 1    | 96    | BF16  | 1x3072x4096 | gemv_cta_reduce<bfloat16,96>        | 20        | 14.720    |
+------+-------+-------+------------+--------------------------------------+-----------+-----------+
```

The CUDA-event numbers remain the end-to-end routing evidence; the approximately 14.7 us Nsight kernel durations
prove that neither path accidentally materializes weights, launches an auxiliary reduction, or falls back to Triton.

Artifacts:

```text
benchmark_artifacts/triton_3bit/trilin-{pre,post}-native-g32-g64-*.json
benchmark_artifacts/triton_3bit/trilin-{g16,g384,g96-g192,large}-*.json
benchmark_artifacts/triton_3bit/nsys_positive_groups_g64_fp16_gpu0.{nsys-rep,sqlite}
benchmark_artifacts/triton_3bit/nsys_positive_groups_g96_bf16_gpu1.{nsys-rep,sqlite}
```

### Investigation record

Successful tactics:

- Passed the requested group through the Python wrapper and Torch schema instead of deriving it from scale shape.
- Split native eligibility from expanded Marlin eligibility; reusing the latter silently excluded groups above 128.
- Used compile-time group templates so inner-loop division is constant-folded and invalid schedules cannot be selected
  at runtime.
- Specialized group 16 at half-tile granularity and groups 32/64 inside the existing 128-value warp split.
- Compile-time-disabled the 128-split warp reducer for 96/192 and retained the generic one-launch CTA reducer.
- Required both independent FP32 numerical gates and warmed latency wins before enabling production routing.

Failed, corrected, or rejected tactics:

- The first post-change benchmark inherited `TORCH_CUDA_ARCH_LIST=8.0`, changing Marlin's cache fingerprint and
  starting an unrelated multi-minute rebuild. Both benchmark children were interrupted before measurement and rerun
  with the normal environment. No result from the interrupted runs was used.
- The first large-group eligibility draft called `marlin_3bit_eligible`, whose correct Marlin-specific group filter
  rejected 256/384/512/1024. A separate continuous-native eligibility contract fixed the routing boundary.
- Applying the 128-value group-warp schedule to group 96/192 was rejected at design time because its split boundaries
  cross scale groups. The compile-time guard prevents those template calls from being instantiated.
- Broad selector smoke testing was useful for transparency but is not a clean regression gate because of the three
  baseline mismatches described above. Focused config/selector matrices are the decision evidence.

Decision: retain native CUDA routing for every declared positive group size. Keep group 128 as the primary optimized
contract and warn once per non-128 group; keep channelwise `-1` and incompatible shapes on the safe existing fallback.
