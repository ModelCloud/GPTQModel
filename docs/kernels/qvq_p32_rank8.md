# P32 window rank-8 reference integration

Baseline: `0cbb57c7` (main including PR #137). The window representation and
all CUDA decoders remain unchanged. This change adds rank-8 state to
`QVQLinear`, its quantization result, and the grouped Hopper consumer.
Existing output "recovery" still means Hadamard/SV/bias; new tensor names are
`rank8_A`, `rank8_B`, and `rank8_metadata`.

For row-major activations the contract is:

```
x_prime = existing_input_transform(x)
inner = existing_window_dispatch(x_prime)          # existing FP32 result
hidden = fp16(fp32(x_prime) @ fp32(rank8_A))
corrected = fp32(inner) + fp32(hidden) @ fp32(rank8_B)
y = existing_output_hadamard_sv_bias(corrected)
```

The separate-reference correction requires TF32 disabled. It has explicit
FP16 hidden rounding and FP32 expansion/addition; it does not fold A/B into
weights. Recovery-off does not inspect correction tensors in forward and
retains existing dispatch and arithmetic. No A100 M512 crossover is installed.

## Quantization and artifact ownership

`quantize_qvq_linear(..., rank8_calibration=Rank8Calibration(...))` retains the
original FP weight, completes normal P32 quantization, fits the completed
runtime module, and returns optional rank-8 buffers in `serialized_tensors()`.
The calibration processor also accepts explicit per-module calibration through
`set_rank8_calibration(name, calibration)`. Collection of original activations
and document provenance remains the caller's responsibility; there is no
implicit reuse of propagation gates or evaluation benchmarks. Automatic
whole-model document capture is not implemented. Atomic SwiGLU selection and
output alignment must finish before fitting and are currently rejected by the
processor attachment.

The initial fitter is bounded-calibration CPU FP64 reduced-rank regression
with `gelsd`, rcond 1e-5. It fits the output predicted by least squares, not a
weight-space SVD. The two objectives are ordinary output L2 and output L2
weighted by calibration residual row energy. B is fit in final-output
coordinates then transformed by inverse SV/Hadamard into inner coordinates.
This matters for nonuniform SV and nonsymmetric composite Hadamards.

Both candidates are scored with FP16 serialized factors, the deployed inner
operator and its output transform on training and disjoint held-out documents.
Validation requires MSE improvement and no p99 absolute-error regression on
both splits. Balanced selection additionally requires the configured minimum
relative MSE improvement (default 1%). Quality accepts every validated module.
No synthetic test establishes real-model quality or promotion eligibility.

The ordinary checkpoint remains canonical planar P32 with optional rank-8
buffers; old checkpoints require no additional storage. The explicit unified
window exporter stores window words instead of planar words (never both),
actual codebook levels, selectors, transforms, bias, factors and fit metadata.
Loading reconstructs canonical planar ownership for existing QVQLinear and
uses its existing transient window cache. This is the same reversible layout,
not another rank-8 weight format. The package and standard state_dict both
reload with correction off until explicitly prepared.

SHA256 binds factors to exact window bytes, levels, bank state, dimensions,
rate, codebook version, transform flags, SU/SV and bias. Separate hashes bind
factors and original teacher parameters. Hashing happens before capture;
identity/version guards reject state mutation during eager execution. Do not
mutate captured graph buffers or tensor `.data` behind the ownership boundary.

`save_window_package` reports actual file bytes including the Torch container.
`window_package_storage` separately reports tensor bytes, per-module window
and recovered BPW, weight-count-weighted average BPW and the analytic FP16
rank-8 increment `16*8*(K+N)/(K*N)`. These are supplied-module totals, not a
whole-model inventory unless every model tensor has been accounted for.

## Runtime and external policy

```python
from gptqmodel.quantization.qvq_rank8 import P32WindowConfig, prepare_rank8

prepare_rank8(layer, P32WindowConfig(recovery_mode="auto", quality_mode="balanced"))
y = layer(x)  # same module, same input/output transform boundaries
```

`qvq_p32_window_linear(layer, x, config)` is the Python entry point over the
module-owned operands. Configure outside graph capture. Policy is immutable
per capture; separate fast/balanced/quality captures must be owned by the
execution graph caller. Tests retain and replay off/on graphs after changing
the eager policy. There is no graph manager that silently changes modes.

`auto`/`production_window` preserve existing dispatch. Explicit `hopper_m16`
and `hopper_direct_decode_mma` expose existing M16 and row-reuse WGMMA
consumers, split count and M range for external correctness/timing comparisons.
Unsupported devices, activation contracts and shapes fail explicitly.
`window_tuning_key` includes physical product name/UUID/memory/SM count,
shape, rate, M, TP world/rank, transforms, quality/recovery state and caller
build identity. It is a key builder, not a ZML autotuner or StableHLO lowering.
Grouped dispatch retains its existing geometry policy.

Grouped QKV and gate/up apply child corrections to completed FP32 outputs using
exactly the shared padded activation that fed WGMMA, then run existing child
or paired output transforms. Each child may independently enable correction.
Folded alternatives without that activation boundary are bypassed. Whole-MLP
fusion falls back to grouped gate/up plus ordinary down forward while rank-8
is active; raw split partials cannot accept correction before reduction.

## Remaining work and promotion boundary

The reference and fused output epilogue are implemented. Concurrent WGMMA input projection,
complete native pipeline fusion, FP8 factors, rank-16 Tensor Core sweeps,
BN32/additional stages, native ABI v3/StableHLO lowering, actual ZML
compiler integration, automatic graph construction, real-model quality evaluation,
TP1/2/4/8 and H100/H200 performance promotion remain unimplemented/unvalidated.
`fused_epilogue` runs expansion/add/Hadamard/SV/bias in one kernel.
`fully_fused` remains unsupported; it does not alias the partial fusion.
No <=3–5% overhead or model speed/quality claim follows from these tests.

## Validation of this WIP

- CPU regression: `CUDA_VISIBLE_DEVICES='' python -m pytest -q
  tests/test_qvq_window_recovery.py tests/test_qvq_v2b2_p32_window.py
  tests/test_qvq_grouped_runtime.py --disable-warnings --maxfail=1`:
  79 passed, 179 skipped (before the final quantization smoke was added).
- Final H200 run: `CUDA_DEVICE_ORDER=PCI_BUS_ID
  CUDA_VISIBLE_DEVICES=GPU-0c667065-5c47-38ce-0b0a-d211392ce9ea
  python -m pytest -q tests/test_qvq_window_recovery.py
  tests/test_qvq_grouped_runtime.py --disable-warnings --maxfail=1`:
  **50 passed, 86 skipped**. The new rank-8 file itself passes all 26 tests.
  Skips include H100-specific and opt-in real-model tests.
- H200 single-operator matrix: M1/16/64/512, K256/N256, FP16 activations,
  W3, auto/M16/large-M consumers, correction off/on, ten graph replays.
  The independent FP32-inner correction composition agrees exactly at the
  returned FP16 boundary. Retained off/on graphs replay correctly after eager
  mode changes. Grouped QKV and gate/up at M1/64 pass both local absolute-error
  limits and exact eager/graph replay with independent child flags.
- CPU smoke runs actual P32 quantization, rank-8 fitting, serialization and
  reload. Synthetic low-rank-teacher fixtures validate both Hadamard-axis
  settings and nonuniform SV; they are algebra tests, not model-quality data.
- New Python files pass Ruff. Comparing Ruff results for changed existing
  files with `HEAD` found no additional findings; their pre-existing lint
  findings remain. `git diff --check` passes.

Hardware: physical GPU0, PCI `00000000:1C:00.0`, NVIDIA H200,
UUID `GPU-0c667065-5c47-38ce-0b0a-d211392ce9ea`, driver 610.57.04,
143771 MiB; Torch `2.15.0.dev20260901+cu130`, Triton `3.8.0+gitc01b6774`.
This was correctness-only, without performance-isolation or Nsight promotion
claims. No CUDA/C++/Triton kernel source or compiler flags were changed.
Logs: `/tmp/qvq-rank8-final.log`, `/tmp/qvq-rank8-regression.log`,
`/tmp/qvq-rank8-quantize.log`. The initial graph-policy rejection is retained
in `/tmp/qvq-rank8-h200.log`; explicit FP16 dispatch now preserves the existing
BF16 overflow-rescue consumer during capture.


## Fused output epilogue (00a71f4b)

`recovery_kernel="fused_epilogue"` retains the reference FP32 projection/FP16
hidden boundary and fuses expansion, FP32 base addition, existing rounded
output Hadamard, SV and bias into one Triton kernel. QVQLinear and grouped
QKV/gate-up consume it through the same operator. One CTA owns an output row;
no persistent scratch or grid barrier is introduced. Supported output widths
are powers of two from16 through16384 on SM90, with optional Hadamard/bias and
strided grouped row storage. Unsupported widths fail explicitly in this
opt-in implementation; automatic dispatch remains unchanged.

The butterfly keeps each historical FP16 rounding point and rescues an
individual overflowing conversion in FP32, matching modes3/4 in
`qvq_hadamard_cuda.cu`. FP32 addition follows the completed eight-product rank
reduction. `enable_fp_fusion=False` prevents contracting that addition into
rank products. Source-correlated SASS confirms separate FMUL/FADD rank
instructions; later FFMA instructions implement correctly rounded division.
Triton's half2 folding preserves the tested half rounding boundaries.

At matched M16/K2048/N2048/W3, Nsight counts13 baseline kernels versus9 fused,
and 2,977,948 versus2,486,932 executed warp instructions (16.5% fewer).
The fused output kernel executes135,616 instructions, uses47 registers/thread,
8KiB shared memory,128 threads/CTA, and no spills. The old Hadamard alone
uses27 registers/thread,8.25KiB shared memory and1024 threads/CTA.
The fused kernel has substantial shared exchange work: the NCU shared-conflict
metric is1568 versus6 for the padded old Hadamard, and measured active-warp
occupancy is6.24% versus49.59% on this small M16 grid. These are optimization
opportunities, not evidence that fewer instructions guarantees lower latency.
The expanded correction and its casts/add/global-memory boundary disappear;
the input projection and its casts/reduction remain separate.

Post-profile tests: **87 passed,86 skipped**, including output widths through
16384, per-operation overflow rescue, no-bias/strided outputs, independent
single/grouped composition, and ten graph replays per case. Skips are mostly
H100 and opt-in real-model tests. Performance remains synthetic operator
scope; no real-model or H100 promotion is implied.

Artifacts: `/tmp/p32-rank8-{reference,fused}-nsys.nsys-rep`,
`/tmp/p32-rank8-{reference,fused}-ncu.ncu-rep`, corresponding `-sass.csv`,
`-metrics.csv`, and `-opcode-summary.json`; correctness log
`/tmp/p32-rank8-fused-postprofile-tests.log`. The full remaining requirements
are tracked in [the phase ledger](qvq_p32_rank8_plan.md).

Post-profile graph medians, matched K=N2048/W3/FP16 on the H200:

| M | Window off us | Separate rank8 us | Fused output rank8 us | Rank8 implementation speedup | Remaining recovery overhead |
|---:|---:|---:|---:|---:|---:|
|1|33.115|47.560|41.808|1.138x|26.25%|
|16|31.586|50.392|43.881|1.148x|38.93%|
|128|35.019|56.571|48.428|1.168x|38.29%|
|512|65.843|99.847|87.404|1.142x|32.75%|
|2048|238.922|296.029|265.462|1.115x|11.11%|

Mean and p95 improve as well; the machine-readable
[summary](results/p32_rank8_h200_epilogue.json) retains those values. All
preflight and pre-timing exclusivity checks passed, with no foreign compute
processes. This meets the local opt-in implementation gates at the measured
scope but does **not** meet the final <=3–5% recovery overhead target.

## Shared input producer candidate

`P32WindowConfig(recovery_mode="on", recovery_projection="input_fused",
recovery_kernel="fused_epilogue")` also combines SU, input Hadamard, and
FP32 rank projection in one Triton producer. It publishes the exact rounded
FP16 activation for the existing window consumer and FP16 hidden factors for
the output epilogue. This changes the projection's FP32 reduction order;
it does not change the FP16 hidden boundary or permit TF32 accumulation.
The producer retains the existing CUDA mode2 overflow rescue before input
normalization. Grouped QKV/gate-up pass up to three independently enabled
factor pointers through the same producer without concatenating checkpoint
state. Disabled children do not supply factors. BF16 execution retains the
reference projection path.

This explicit candidate requires SM90, FP16 operands and a power-of-two K
between 16 and 16384. The output epilogue has its own N eligibility gate.
It is not `fully_fused`: the window consumer and epilogue are separate
launches, and transformed activations still cross global memory.

The [H200 producer measurements](results/p32_rank8_h200_producer.json) show
116 focused tests passing after profiling and a four-launch M16 operator,
down from nine. K=N=2048 W3 latency improves at M16/128 but regresses at
M1/512/2048. No automatic dispatch or model-quality promotion is enabled.

## Direct geometry controls and candidate enumeration

The unified policy now exposes existing unsplit Hopper consumers with
`algorithm="hopper_direct_decode_mma"`, `block_m=32|64|128`, and
`block_n=64|128`. BM is the activation-row tile and BN the output-channel
tile. BK is 256, the pipeline has two stages, and BN64/128 uses one/two
consumer warp groups. Requests for unimplemented combinations fail at the
configuration boundary. BM/BN zero preserves existing dispatch. Explicit
geometry pads M to BM and slices the result back to the original rows.

The native `gptqmodel_qvq_wgmma::p32_window_tuned` entry point selects the
existing row-reuse specializations and can force BN128 independently of
the previous H100 shape rules. It adds 12 host dispatch sites (four rates
times three BM values) and zero new device kernel specializations; BN
selects between the existing one/two-consumer kernels. Existing callers
retain their dispatch. BN32, more pipeline stages and independent warp-group
variants are still required work.

Call `prepare_rank8(layer, quality_policy)` and then
`window_kernel_candidates(layer, m=M)` to enumerate implementations for
that quality policy. Every supported geometry remains available at each M,
including geometries that lost at other shapes. The enumeration does not
change correction eligibility or choose a winner. Non-Hopper devices retain
the production candidate until their backend exposes additional controls.
`config.to_backend_config()` and `P32WindowConfig.from_backend_config(...)`
provide strict versioned dictionaries for external tuning, including ZML;
they do not implement a StableHLO/XLA FFI lowering. The shared tuner below
consumes these controls and persists measured choices.

The benchmark accepts `--block-m` and `--block-n`, measures the full operator
with correction off and on, and retains source hashes, geometry, local drift,
graph agreement and latency samples. Selection must remain device-, shape-,
rate-, M-, TP- and correction-specific, with correctness filtering before
latency ranking. H200 measurements cannot supply H100 tuning entries.

The initial [H200 geometry scorecard](results/p32_window_h200_geometry.json)
contains all six BM/BN candidates and existing direct dispatch, each measured
with correction off/on at M1/16/128/512/2048 for K=N=2048 W3. Post-profile
tests pass 331 cases; 86 cases are skipped. BM128/BN128 improves M2048 but
regresses smaller M, so no universal dispatch threshold is installed.
At M2048 the matched committed baseline and candidate execute the same
2,097,152 WGMMA instructions. TMA-load instructions fall from 139,264 to
69,632, while registers increase 147 to 153 and dynamic shared memory
160,512 to 172,800 bytes. Neither kernel spills. The producer shares staged
activations across two consumers without changing the accumulation order
within an output element. The generic window decode remains unchanged.
Both extension binaries contain exactly the same 155 demangled device
kernel specializations; the new geometry controls add host dispatch only.

## Correctness-gated native and external tuning

`qvq_window_tuning.tune_window_kernel` operates on a prepared quality policy.
It enumerates that policy's eligible kernels and compares each executable
against production window plus the same correction state on every supplied
activation case. Failed candidates retain their errors and timing but cannot
win. A failed candidate more than 1.25x faster than the fastest passing one
gets an explicit human-review flag; this does not relax the accuracy gates.
The benchmark CLI prints such flags when they occur.

The caller supplies `benchmark(executable, activation)`, returning positive
microsecond samples under its device-exclusivity/timing contract. The optional
`compile_candidate(backend_config)` receives direct geometry and correction
controls and returns the executable that will actually be validated and timed.
This is the integration point for external compilers such as ZML. A native
StableHLO/XLA FFI lowering is still required; a Python callback is not evidence
that ZML itself has been run.

An optional `cache_dir` stores atomic JSON entries binding exact activation
cases and strides, deployment payload and transforms, enabled correction
factors, device identity, TP, software/driver API, compiler build and candidate
set. The CLI adds the actual JIT fingerprints, Triton version and driver
release to its build identity. Cache hits revalidate the selected executable
before use. Exceptions restore the original policy; `apply=False` also keeps
the original policy after successful tuning. The default applies the selected
static M policy, which must be prepared/captured outside request execution.
Existing captured graphs do not change when a new policy is prepared.

Run the integrated benchmark with `--autotune --m 128 2048 --tuning-cache DIR`.
It validates on two independent synthetic activation tensors per M, measures
all eligible off/on candidates, then separately checks and times the winner.
These inputs test implementation preservation and do not fit rank8, select
quality, or establish full-model accuracy. Non-Hopper backends currently
expose their production candidate; additional backend-specific candidates
and actual ZML compiler integration remain open.

The retained [H200 autotuning result](results/p32_window_h200_autotuning.json)
covers 90 candidate evaluations across M128/M2048 and correction off/on.
All candidates passed both supplied cases. M128 selected production window,
with the shared producer/fused epilogue when corrected; M2048 selected
BM128/BN128, with separate projection/fused epilogue when corrected. A fresh
process hit all four persistent entries and revalidated both cases for each.
The broader regression run passed 334 tests with 86 skips; the final focused
tuner tests also passed after cache-identity changes.

## Full-size and chunked prefill

Explicit FP16 Hopper policies now accept M through 8192. `chunk_m=0` uses
one native window launch; `chunk_m=4096` splits only the window computation,
retaining the shared full-size input transform and rank8/output stages.
Both are exposed by candidate enumeration for M>4096. FP8 and the specialized
H100 row-reuse-11 bounds remain unchanged.

The [M8192 H200 record](results/p32_window_h200_m8192.json) contains 16 off
and 64 on candidates, checked on two independent activation cases each.
Post-profile BM128/BN128 full-size versus chunked medians are 847.162 versus
888.707 us off and 944.509 versus 987.054 us on. Both have zero drift in this
matched fixture. Full-size removes a second window launch and concatenation:
nine versus eleven launches and 498,543,996 versus 500,764,567 source-correlated
executed instructions. Window resources remain 153 registers, 172,800 bytes
dynamic shared memory and no spills. These results select neither a universal
crossover nor a production default.

`recovery_projection="tensor_core"` is an additional explicit SM90 candidate:
FP16 X'/A, rank padded to 16, FP32 accumulation and FP16 hidden output. It
reads the already transformed activation without repeating SU/H. BF16 rescue
retains reference projection. Whole-operator timing and correctness gates apply
to this candidate just as they do to `input_fused` and `separate_reference`.

The [padded Tensor Core scorecard](results/p32_rank8_h200_tensor_core.json)
records matched BM128/BN128 full-operator medians on H200:

| M | Correction off (us) | Reference projection on (us) | Padded TC on (us) | TC marginal cost |
|---:|---:|---:|---:|---:|
| 512 | 77.359 | 98.891 | 89.840 | 16.1% |
| 2048 | 219.627 | 249.145 | 235.984 | 7.4% |
| 8192 | 846.720 | 944.952 | 852.085 | 0.6% |

All use K=N=2048, W3, A16, synthetic activation/factor fixtures and the fused
output epilogue. Maximum drift is 0.001953125 against production window plus
reference projection. These are local kernel results, not model quality or
general recovery-overhead guarantees. At M2048, the recovered operator drops
from nine to five launches and 120,822,714 to 107,700,862 source-correlated
executed instructions. The projection executes 32,768 HMMA instructions,
uses 40 registers and 6,144 bytes of dynamic shared memory, and has no spills.
It still performs shared staging and barrier work and reads published X'
separately from the window consumer.

With this candidate included, the tuner passes 126 candidate evaluations at
M128/M2048 against two independent cases each. M128 selects the shared input
producer; M2048 selects padded Tensor Core projection with BM128/BN128. All
supported choices remain available at other shapes and to external compilers.
The post-profile regression suite passes 494 tests with 86 skips, including
mixed projection implementations within grouped siblings. A fresh process
revalidates all four selected entries from the persistent tuning cache.
