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

Kernel tuning follows the same distinction. When `rank8_enabled=0`, every
finite locally-correct launch row remains eligible, including rows carrying an
unverified producer signature; an optional recovery-overhead budget is ignored
because no correction executes. Once rank8 is enabled, balanced and quality
policies continue to require their certified arithmetic signatures, and an
explicit overhead budget applies only to candidates with a matched off/on
measurement. The 3--6% scorecard target is report-only unless a caller
supplies a promotion cap.

## Quantization and artifact ownership

`quantize_qvq_linear(..., rank8_calibration=Rank8Calibration(...))` retains the
original FP weight, completes normal P32 quantization, fits the completed
runtime module, and returns optional rank-8 buffers in `serialized_tensors()`.
The calibration processor also accepts explicit per-module calibration through
`set_rank8_calibration(name, calibration)`. Collection of original activations
and document provenance remains the caller's responsibility; there is no
implicit reuse of propagation gates or evaluation benchmarks. Automatic
whole-model document capture is not implemented. Atomic SwiGLU selection now
defers fitting until its complete triplet is selected; output-aligned modules
defer fitting until the final aligned layer pass, using the aligned SU/SV
payload and the immutable dense teacher snapshot.

`Rank8Calibration` may carry a third `audit_inputs` fold with
`audit_document_ids` and `audit_row_counts`. When present, the quantizer
evaluates every audit document after fitting and promotes factors only when the
independent gate accepts them. A calibration with no audit remains a fit result
for research/inspection; `export_window_package` and the native ZML loader
reject its rank-8 sidecar until `audit_validated=true` and
`audit_acceptance.accepted=true` are recorded.

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

Promotion has a separate hard confirmation boundary: the independent audit
must be finite on every document, improve MSE by the requested threshold, and
not worsen p99 tail error. A failed audit clears A/B/metadata and marks the
module rejected; only accepted audit evidence is serialized in the package.

The bounded full-model H200 fitting run is recorded in
`results/p32_rank8_llama_full_model.json`. It covers the 94 eligible P32
modules across all 16 Llama-3.2-1B layers, with disjoint train/selection/audit
documents and hard activation/solver memory limits. It reports the actual
window-plus-rank8 bytes and weighted BPW, while leaving W4 output modules
outside the contract. The result establishes broad fitting provenance only;
propagated perplexity, task accuracy, multi-device performance, and promotion
gates remain separate validations.

Quantization jobs may request `rank_candidates=(2, 4, 6, 8, 12)` on
`fit_rank8`. The report then records the same predictable-residual solver,
serialized-FP16 output scores, effective rank and solver mode for every
candidate under both fitting objectives. Only the rank-8 candidate is attached
to the current deployment buffers; lower or higher ranks are comparison data
until the runtime kernel ABI and graph scorecard support their shapes.

The ordinary checkpoint remains canonical planar P32 with optional rank-8
buffers; old checkpoints require no additional storage. The explicit unified
window exporter stores window words instead of planar words (never both),
actual codebook levels, selectors, transforms, bias, factors and fit metadata.
CUDA package loading makes the window payload the sole live weight storage and
releases the temporary CPU planar reconstruction; a legacy/debug caller can
request `retain_planar=True`. This is the same reversible layout, not another
rank-8 weight format. The package and standard state_dict both reload with
correction off until explicitly prepared.

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

Automatic tuning and automatic activation are separate decisions. A prepared
`fast` rank8 sweep measures every eligible projection, including Tensor Core,
and applies the fastest numerically accepted candidate without requiring a
3--5% overhead result. The package default remains recovery-off because rank8
factors are optional per module and Tensor Core currently has an unverified
arithmetic signature; changing that default would change the established
window output for models that did not request correction. Run the sweep and
prepare the resulting policy before capture to make Tensor Core automatic for
that exact device, shape and M bucket. Balanced/quality still require the
validated audit and a certified/reference arithmetic signature.

`auto`/`production_window` preserve existing dispatch. Explicit `hopper_m16`
and `hopper_direct_decode_mma` expose existing M16 and row-reuse WGMMA
consumers, split count and M range for external correctness/timing comparisons.
Unsupported devices, activation contracts and shapes fail explicitly.
`window_tuning_key` includes physical product name/UUID/memory/SM count,
shape, rate, M, TP world/rank, transforms, quality/recovery state and caller
build identity. It is a key builder, not a ZML autotuner or StableHLO lowering.
Grouped Hopper dispatch accepts a complete tuple of `hopper_m16` child
policies and consumes each child’s selected split count when building the
segmented payload. Single-child BM/BN controls remain rejected for grouped
execution until a grouped kernel exposes those axes directly.

When tuning is applied to a module, the unified package also carries an
optional versioned `kernel_tuning` record. It stores the selected backend
configuration, quality/correction state, candidate identity and the P32/factor
semantic hashes used for that selection. Export validates those hashes against
the current module; package and artifact loaders validate them again before
accepting the hint. This is advisory deployment metadata, not a portable
performance claim: a loader must not apply it to a different device, shape, TP
layout or build. ZML and other external consumers must enumerate and benchmark
their own eligible executables, then cache the result under the full
device/shape/rate/M/TP/correction key.

The ZML candidate enumerator binds each returned launch policy to the exact
measured `M` (`min_m=max_m=M`). A BM/BN winner therefore cannot be reused for a
different request shape without a fresh enumeration and measurement.

The non-Hopper SM80 consumer exposes the same rule through
`qvq_p32_window_ampere_kernel_candidates((M, K), out_features=N, bits=...)`.
It returns the measured shape-specific split first, followed by a bounded
probe set, without CUDA allocation or timing. A kernel tuner or ZML can time
those explicit `split_count` values before capture and pass the winner to
`qvq_p32_window_ampere`; a cold graph never launches the event-based tuner.

Grouped SM80 consumers expose the same control per child through
`qvq_p32_window_ampere_grouped_kernel_candidates((M, K), out_features=(N_0,
N_1[, N_2]), bits=...)`. The first tuple is the independent child-policy
baseline; later tuples vary child split waves without treating the concatenated
width as one shape. ZML publishes the matching pure `p32GroupedCandidateSet`
and accepts a selected geometry through
`p32WindowMatmulGroupedWithConfig`. Enumeration and selection happen before
capture; replay contains only fixed launch attributes.

Grouped QKV and gate/up apply child corrections to completed FP32 outputs using
exactly the shared padded activation that fed WGMMA, then run existing child
or paired output transforms. Each child may independently enable correction.
Grouped runtime preparation accepts `separate_reference` and `input_fused`
rank8 projections only; `concurrent_reference`, Tensor Core and other
single-child projection choices fail closed until grouped producers implement
their own graph-safe shared scheduling.
Folded alternatives without that activation boundary are bypassed. Fused MLP
execution now carries rank-8 through gate/up, SiLU/product, and the down
projection: the down correction is added after its completed FP32 inner output
and before the existing output transform/store. Raw split partials cannot
accept correction before reduction, so those specialized reductions are
disabled for a rank-8 down child while the graph-safe completed-output path is
used instead.

## Remaining work and promotion boundary

The reference/fused output epilogues, shared input producer, padded Tensor Core
projection, request-owned model graphs, typed native artifact loading and
initial native/ZML reference bridge are implemented. Allocation-free ZML
candidate selection and direct executable timing hooks are also available for
pre-capture tuning. Concurrent WGMMA input projection, complete native pipeline
fusion, FP8 factors, BN32/additional stages, explicit external user capture,
KV/TP graph ownership, the full real-model scorecard and H100/H200 promotion
remain open.
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

`recovery_projection="project_output_fused"` is an additional fast-mode
candidate that removes that separate projection launch as well. It computes
`X' @ A` inside the row-owned output epilogue, rounds the hidden state to the
declared FP16 boundary, and then applies `B`, the output Hadamard, SV and bias.
The candidate is graph-safe after shape-specific Triton warming and is marked
`unverified_project_output_fused`, so balanced and quality policies reject it
 until an arithmetic signature is certified. On the current accepted Q
 projection (H200, K=N=2048), the candidate is shape-dependent: it measured
 41.7 us at M128 versus 45.9 us for reference projection/fused epilogue, but
 262.2 us at M2048 versus 249.4 us and 1,016.2 us at M8192 versus 951.2 us.
 The tuner therefore selects it only for the M128 shape and rejects it for the
 larger shapes. The complete matrix is recorded in
 `results/p32_rank8_qproj_h200_project_output_fused.json`; this is an exposed
 experiment, not a claim of universal fusion benefit.

`recovery_projection="concurrent_reference"` is the first shared-producer
candidate for the full pipeline. It keeps the exact reference FP32 projection
and FP16 hidden rounding, but schedules that projection on a prepared auxiliary
CUDA stream while the current stream runs the window decoder. The stream and
events are created before capture; the exact `(device, M, K)` shape must be
warmed before capture, and replay contains only captured dependencies. On the
accepted Q H200 replay, the candidate measured 58.2/302.6/1,106.7 us at
M128/2048/8192 versus 31.7/214.2/814.6 us with correction disabled. The
resulting 83.7%/41.3%/35.8% marginal costs show that this naive overlap is
resource-contended on this shape. It remains exposed to the shape tuner and is
never a default quality choice; the measurement is recorded in
`results/p32_rank8_qproj_h200_concurrent_reference.json`.

Reference rank8 projections now retain prepared contiguous FP32 A/B factor
buffers per module and invalidate them on factor version changes. This removes
the repeated factor-conversion allocations from eager and captured replay while
leaving the FP32 projection, FP16 hidden boundary, and FP32 expansion ordering
unchanged. The refreshed accepted-Q H200 replay measures fused-epilogue
marginal costs of 38.5%/15.5%/16.7% at M128/2048/8192; the complete rows are in
`results/p32_rank8_qproj_h200_factor_cache.json`. Factor residency is an exact
optimization, but it does not by itself meet the 3--5% promotion target.

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

For input Hadamard modules this candidate requires SM90, FP16 operands and a
power-of-two K between 16 and 16384. Folded-input modules with
`input_hadamard=false` also admit composite K widths through a masked producer
(for example K=5120), while retaining the same FP16 transform and hidden
boundaries. The output epilogue has a corresponding folded-width path when
`output_hadamard=false`; Hadamard composite widths remain outside this
candidate. It is not `fully_fused`: the window consumer and epilogue are
separate launches, and transformed activations still cross global memory.

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
This is the integration point for external compilers such as ZML. The callback
tests alone are not ZML evidence. The native/ZML section below separately
records actual StableHLO lowering and execution; connecting that executable
to the latency tuner and its cache remains open.

Use `measure_rank8_overhead(layer, inputs, benchmark=...)` after selecting a
candidate to record a matched correction-off/on pair. It restores the original
policy even when timing fails and reports both medians, absolute cost and
`overhead_percent`; it never promotes an arithmetic implementation or changes
quality eligibility. `tune_window_kernel(..., measure_recovery=True)` attaches
the same record to its result. For non-Hopper consumers the Ampere module
provides the allocation-free `qvq_p32_window_ampere_kernel_candidates` and
`qvq_p32_window_ampere_grouped_kernel_candidates` APIs; callers benchmark the
returned split waves before capture and pass the selected split explicitly.
ZML has the corresponding `enumerateCandidates`, `benchmarkExecutable` and
`selectFastest` controls. Each backend keeps its own shape/device/rate cache.
The native adapter also exposes `benchmarkRecoveryPair` and
`selectFastestWithRecoveryGate`; a supplied nonnegative budget requires a
matched complete-executable correction-off/on pair for every winning
candidate before graph compilation.

Grouped gate/up and QKV callers use `tune_grouped_window_kernel` with a tuple of
child policies. The tuner validates and times each complete tuple, including
matched correction-off/on samples when an overhead budget is requested, then
caches the tuple identity with every child payload and shape. This preserves
independent child split choices on SM80 and exposes the same explicit policy
tuple to native callers and ZML; a policy is prepared before capture and is
restored on failure.

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
quality, or establish full-model accuracy. Non-Hopper backend candidates are
only eligible when their backend enumerator and executable are explicitly
provided; no Hopper geometry is silently reused. ZML executable tuning/cache
integration through this Python helper remains open even though the direct Zig
controls are available.

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

## Capture during public model quantization

`model.quantize(..., rank8_capture=Rank8Capture(...))` captures selected dense
Linear inputs before quantization and hands each module's document-separated
activations to its existing P32 finalizer. For example:

```python
from gptqmodel.quantization.qvq_rank8_capture import Rank8Capture, Rank8Document

capture = Rank8Capture(
    module_names=("model.layers.0.self_attn.q_proj",),
    train=(Rank8Document("calibration/train/document-id", train_token_tensors),),
    heldout=(Rank8Document("calibration/heldout/document-id", heldout_token_tensors),),
    audit=(Rank8Document("calibration/audit/document-id", audit_token_tensors),),
    rows_per_document=128,
    max_bytes=512 * 1024 * 1024,
    max_solver_bytes=256 * 1024 * 1024,
)
model.quantize(calibration, rank8_capture=capture)
```

Each tensor dictionary contains one tokenized document (`input_ids[1,T]` and
optional binary `attention_mask[1,T]`) on the dense model's execution device.
Use original calibration documents with distinct provenance, never evaluation
benchmarks. Capture rejects intersecting IDs or identical unmasked token
content across folds, skips padding, selects evenly spaced token positions,
and bounds retained CPU activation storage including concatenation scratch.
The bound does not cover the model or its forward-pass working memory.
Hooks are removed on success and failure. Teacher hashes bind capture to the
weights used by the finalizer; preprocessing that changes that teacher fails
explicitly. Unconsumed module requests also fail rather than silently losing
the requested fit.

The public path requires calibration-based P32 A16 and explicit target module
names. A lazy dense teacher can be supplied through
`rank8_teacher_materializer=callable` on `model.quantize`; the callback
materializes requested meta-device Linear modules in place once, before hooks
or any graph capture. Atomic replay/output alignment and weight-only jobs
remain unsupported for this capture path. The FP64 fitter uses its exact
least-squares/SVD reference while the estimated workspace fits
`max_solver_bytes`, then switches to a fixed-seed rank-8 output-range solver
for larger K×N projections. That path range-finds the predictable residual
`P_X R` with four oversampling columns, then derives output directions from
`Q_Z^T R`, rather than spending rank on unpredictable residual components. The
selected solver and cap are recorded in the fit metadata. Unverified
fused/input-fused/Tensor-Core arithmetic signatures remain available for fast
experiments but are excluded from balanced/quality latency selection until
independent equivalence certification exists.

The complete tiny-Llama integration test exercises public quantize, accepted
rank8 fitting, normal model save/reload, exact factor preservation, payload
validation, and exact corrected module output. Its synthetic documents test
pipeline plumbing, not held-out model quality. The loader now allocates
optional buffers from safetensors headers before Accelerate loads shards,
validating complete A/B/metadata triples and their shapes/dtypes first.
For native/ZML deployment, `save_window_artifact(layer, directory)` writes the
same tensors as immutable `.bin` files plus a versioned `manifest.json`.
`integrations/zml/qvq_window.zig::loadArtifact` validates the manifest's
descriptor-level `payload_sha256` binding, exact file names, dtypes, shapes,
byte counts, and per-file SHA-256 hashes before device upload. It accepts the
checkpoint's FP32 SU/SV tensors while the native ABI performs its declared FP16
transform narrowing. The Python package loader also checks the recovery
base/factor semantic digests; native loading retains those fields for policy
inspection while the descriptor binding protects the exact bytes consumed by
the native operator.
The focused capture/checkpoint/lifecycle regression passes 75 tests. A separate
[real Llama capture record](results/p32_rank8_llama_capture.json) contains
first-layer Q/gate activation and teacher hashes for four distinct calibration
documents on H200, with 64 training and 64 held-out rows per module. It proves
capture execution only; the fitting audit below supplies separate evidence.

## First real module fits and audit

`scripts/evaluate_qvq_window_rank8.py` now fits selected modules of an existing
P32 checkpoint against its original dense teacher. Its initial document policy
uses eight calibration documents for fitting, four for candidate selection,
and four additional audit documents; documents are capped at 512 tokens with
64 evenly spaced activation rows retained each. The CLI exposes the retained-row
count and hard capture/solver byte bounds for broader module sets while keeping
the same bounded, disjoint contract. IDs and token-content overlap are checked
across the folds. It saves fitted unified packages, captured
activations, hashes, candidate scores, and per-document audit errors. Fitting
now explicitly scores the deployed final output dtype, including its final
conversion, rather than requesting an FP32 final output for half activations.

The [first-layer Llama result](results/p32_rank8_llama_first_layer.json) uses
the F6 P32 checkpoint and original Llama-3.2-1B-Instruct teacher on H200.
Both modules selected output-aware L2 over the tail-weighted candidate:

| Module | K × N / rate | Audit MSE improvement | Window BPW | With rank8 BPW | Serialized package bytes |
|---|---|---:|---:|---:|---:|
| layer 0 Q | 2048 × 2048 / W2 | 11.16% | 2.0635 | 2.1885 | 1,153,629 |
| layer 0 gate | 2048 × 8192 / W3 | 5.23% | 3.0510 | 3.1292 | 6,568,499 |

MSE, MAE and p99 error improved on each audit document. Maximum teacher error
increased on one Q document and two gate documents. These are four-document
module results, not evidence of full-model non-regression or task gains.
No evaluation benchmark entered fitting or candidate selection.

`benchmark_qvq_window_rank8.py --package PACKAGE --activation-file CAPTURE`
can replay the saved `audit_1`/`audit_2` activation matrices cyclically to the
requested M. It records the source file hash and expansion policy; this is
complete-operator timing, not a full-model prefill measurement. With BM128/BN128,
padded Tensor Core projection and the fused output epilogue:

| Module | M | Off (us) | On (us) | Marginal correction |
|---|---:|---:|---:|---:|
| Q | 2048 | 220.512 | 237.038 | 7.49% |
| Q | 8192 | 849.207 | 854.370 | 0.61% |
| gate | 2048 | 626.542 | 679.112 | 8.39% |
| gate | 8192 | 2408.204 | 2546.246 | 5.73% |

Both real-factor/activation cases pass the unchanged local kernel gates and
eager/graph equality. Maximum implementation drift is 0.00390625 for Q and
0.0009765625 for gate. The gate result exceeds the desired fused recovery
overhead target and reinforces the need for shape-specific tuning and further
epilogue/producer fusion. The factor packages and activation captures are
archived under `/root/qvq-results/window-rank8-first-layer-h200`, with hashes
in the result record. Full-model propagation, more documents/layers, C4,
ARC/GSM8K, H100 and TP validation remain pending.

## Model propagation and C4 subset

`scripts/evaluate_qvq_rank8_propagation.py` validates fixed package/base and
teacher hashes, runs identical token IDs through the teacher and both model
modes, and reports token-weighted teacher cross-entropy/PPL, temperature-one
and low-temperature KL, top-1/top-5/top-10/top-32 agreement, logit margins and
paired document-bootstrap intervals. It performs no fitting. Older result
files contain the original metric subset; rerunning the script writes the full
metric record without changing the fitting or package inputs. The
[initial propagation diagnostic](results/p32_rank8_llama_propagation.json)
uses 16 additional calibration documents excluded from fitting, selection,
and the earlier module audit.

When `--verify-graphs` is enabled, `--max-graph-resident N` bounds the number
of captured input signatures retained during the run. The evaluator records
per-document and final `graph_residency` snapshots, including the configured
bound and retired-key count, so graph memory behavior is visible alongside the
quality metrics.

The [C4 reference-correction subset](results/p32_rank8_llama_c4_subset.json)
uses `allenai/c4` revision `1588ec454efa1a09f29cd18ddd04fe05fc8653a2`, English
validation shard 0, first 128 documents, BOS/default tokenizer special tokens,
no chat template, up to 2048 tokens per document. All modes score the same
47,550 next-token predictions. Only the first-layer Q/gate corrections are
enabled. This is a specified subset diagnostic, not full C4 validation.

| Mode | PPL | Mean teacher KL | Top-1 teacher agreement |
|---|---:|---:|---:|
| Dense teacher | 18.85942 | — | — |
| Existing quantized model, correction off | 20.03599 | 0.07837847 | 84.8833% |
| Separate reference correction | 20.03124 | 0.07835802 | 84.8749% |
| Reference projection + fused epilogue/store | 20.03123 | 0.07835780 | 84.8728% |
| Padded Tensor Core projection + fused epilogue/store | 20.03359 | 0.07835008 | 84.8749% |

The reference correction reduces mean NLL by 0.0002372 nats/token; its paired
95% document-bootstrap interval is [-0.0003944, -0.0001040]. KL and top-1
changes are not resolved by their intervals. The
[Tensor Core result](results/p32_rank8_llama_c4_fused_subset.json) has a small
NLL increase against the reference correction: +0.0001173, interval
[+0.0000397, +0.0002029]. The
[epilogue-only ablation](results/p32_rank8_llama_c4_epilogue_subset.json)
does not show that increase. Projection remains externally selectable;
these results do not establish full quality equivalence or promote a default.

## Direct FP16 output store

Eligible fused epilogues now store the final FP16 result directly, preserving
all FP32 expansion/addition and existing output-transform rounding. Single
and grouped paths pass the requested output dtype. Paths whose surrounding
overflow retry needs FP32 range retain FP32 output; the independent reference
API also defaults to FP32. Direct-store and cast-afterward tests include range
edges and CUDA Graph replay. Post-profile regression passes 405 tests with
86 skips, including grouped K2048 cases.

The [H200 store-fusion record](results/p32_rank8_h200_half_store.json) retains
the executed instruction audit and matched real-activation timings. At
K=N=M2048 in the profiling fixture, launches fall from five to four and
source-correlated executed instructions from 107,700,862 to 107,202,330.
The epilogue uses 40 rather than 47 registers, with 8,192 bytes of shared
memory and no spills. Post-profile recovered medians change as follows:

| Module | M | FP32 output + cast (us) | Direct FP16 store (us) |
|---|---:|---:|---:|
| Q | 2048 | 237.038 | 224.895 |
| Q | 8192 | 854.370 | 802.722 |
| gate | 2048 | 679.112 | 635.615 |
| gate | 8192 | 2546.246 | 2373.063 |

Those store-fusion timings used different output implementations for off and
on. They are superseded for marginal rank8 cost by the shared-epilogue
comparison below.

## Shared correction-off/on output epilogue

The same fused output Hadamard/SV/bias/store implementation now serves both
correction states in single and grouped paths. A compile-time flag removes
all rank8 projection, factor loads and expansion when disabled. Tests pass
invalid/poisoned factors to disabled paths and check eager and graph output.
The epilogue remains explicitly selectable and is included in native/external
autotuning candidates for both states.

The [shared-epilogue record](results/p32_window_h200_shared_epilogue.json)
contains the executed instruction audit, post-profile timing, expanded tuning
run and C4 off-equivalence check. In the M=K=N=2048 W3 profiling fixture,
original off uses four launches and 148,026,013 source-correlated instructions;
shared off uses three and 103,378,821; shared on uses four and 107,202,547.
Both shared epilogues use 40 registers and 8,192 shared bytes, with no spills.
Post-profile tests pass 414 cases with 86 skips. Switching the two target
modules to fused-off preserves full-model logits bit for bit on all 128 C4
subset documents (47,550 predictions).

Matched real-factor/captured-activation timings with BM128/BN128, shared
output epilogue and Tensor Core rank8 projection are:

| Module | M | Off (us) | On (us) | Marginal correction |
|---|---:|---:|---:|---:|
| Q | 128 | 50.524 | 64.813 | 28.28% |
| Q | 2048 | 198.969 | 225.063 | 13.11% |
| Q | 8192 | 757.233 | 803.166 | 6.07% |
| gate | 128 | 63.004 | 78.672 | 24.87% |
| gate | 2048 | 599.291 | 635.514 | 6.04% |
| gate | 8192 | 2299.251 | 2374.356 | 3.27% |

These compare equally optimized off/on output paths and supersede earlier
near-zero marginal-cost estimates. They are module replay measurements,
not whole-model latency. Wide gate meets the 3–5% large-M overhead target;
square Q does not. Forced BM128/BN128 is not the best small-M policy.

The real-Q tuning run evaluates 18 off and 54 on candidates at each of M128
and M2048, each against two independent captured activation cases. At M128
it selects production window with fused output and, when enabled, shared
input projection (31.721 us off, 43.393 us on). At M2048 it selects direct
BM128/BN128 with fused output and Tensor Core projection (199.165 us off,
225.153 us on). These are per-shape latency selections under the local
numerical gate, not model-quality promotion of Tensor Core projection.
External capture and KV/TP graph ownership, ZML executable tuning, concurrent
native WGMMA/rank8 fusion and the full model/device/TP scorecard remain open.

## Request-owned quality graphs

`gptqmodel.quantization.qvq_window_graphs.P32WindowGraphs` captures three
separate model graphs per input signature. It calls the existing window
operator; rank8 remains module state and is never a separate model correction
wrapper. `fast` disables correction, `balanced` uses quantizer-selected
modules, and `quality` uses all validated corrections. Per-module configs
supply geometry/projection/epilogue choices; capture resolves their correction
state from the requested quality mode instead of trusting a tuner's off/on
field.

The managed CUDA extension, Hopper WGMMA bridge, Ampere provider, rank8 Triton
producer/epilogue, YAQA Triton projector and ROCm YAQA trusted recurrence all
reject cold JIT/operator registration during capture. Warm and validate each
shape, dtype, device and correction state before creating a graph; replay does
not tune, allocate kernel caches or change policy. The managed CUDA path also
requires the PGC16 level table and vector-size-four bank selectors to be
prepared and validated before capture, so those first-use allocations and
device-side validation synchronizations cannot enter a graph.

```python
from gptqmodel.quantization.qvq_window_graphs import P32WindowGraphs

owner = P32WindowGraphs(model.eval(), max_graphs=8)
try:
    owner.capture(
        "prefill128",
        {"input_ids": ids, "attention_mask": causal_mask},
        configs=per_mode_module_configs,
        static_kwargs={"use_cache": False, "return_dict": False},
    )
    logits = owner.replay(
        "prefill128", "balanced", input_ids=next_ids,
        attention_mask=causal_mask,
    )[0]
finally:
    owner.close()
```

`max_graphs` bounds retained input signatures; each resident signature owns up
to one graph for each quality mode. Least-recently-used retirement synchronizes
the latest replay event before releasing its CUDA graph/pool references. Use
`owner.residency_stats()` while idle to record resident keys and retirements in
the graph scorecard.

Input tensors must have the captured shape, dtype and CUDA device. Static
keywords are immutable scalar values; outputs are tensor pytrees. This API
captures a stateless model call. KV state must be supplied explicitly as tensor
inputs rather than hidden Python state; integration with a generation server's
KV cache and TP request scheduler remains open. Transformers eager attention
may create CPU scalars in its mask builder, so the validation script supplies
an equivalent additive causal mask before capture.

The owner prevents overlapping host requests and multiple owners for the same
model. CUDA events order successive requests across streams. Each request
returns independent output storage. Captured graphs retain cached tensor and
grouped payload storage even after eager policy restoration. Failed capture
restores all eager policies and never installs a partial three-mode entry.
Tracked weight/buffer mutations, module replacement, transform changes and
training mode invalidate replay; untracked `.data`/external-pointer mutation
is unsupported. `invalidate()` retires all graphs and permits recapture;
`close()` additionally releases ownership. The caller must not execute or
mutate the model concurrently outside the owner.

Tests cover all three policies with independently selected child corrections,
request/output lifetime, cross-stream ordering, changed input signatures,
failed-capture rollback, poisoned correction state in fast mode, mutation
rejection, grouped gate/up payload lifetime and a full tiny Llama forward.
The real Llama-3.2-1B checkpoint graph check uses fixed first-layer Q/gate
corrections on four C4 documents capped at 128 tokens. Both reference-output and fused-output runs match eager full logits bit for
bit in all three modes on all four documents (508 predictions per run).
See `results/p32_window_llama_graphs.json`. This is execution validation, not
an expanded quality scorecard or a graph-manager latency claim. Replay still
checks model state on the host and copies returned outputs; request overhead,
large-context graph residency and model-wide generation performance require
measurement before serving promotion.


## Native ABI and actual ZML execution

The [ZML adapter](../../integrations/zml/README.md) now lowers
`stablehlo.custom_call @qvq_p32_window_linear` to a native C ABI on the
PJRT-provided CUDA stream. It executes without Python and reuses existing
Hopper window/Hadamard operators. Geometry and correction state are individual
compiler attributes. The initial native contract supports explicit M16 or
BM32/64/128 with BN64/128, BK256/stages2/split1 and reference rank8, on SM90
with 256-wide K2048..16384/N256..17408 and M1..8192. Hadamard transforms
remain power-of-two only; transform-free composite shapes such as
K5120/N17408 are now admitted and have a real H200 bit-exact ABI check.

The [native/ZML evidence](results/p32_window_native_zml.json) retains 16
bit-exact real-Q native/reference cases at M1/33/128/2048, plus successful ZML
CUDA execution with correction off/on at M33, K=N2048, W2, BM64/BN64. The
transient fixture is hash checked; it is not another deployment format.
Post-profile validation passes the native/graph/tuning cases and extension
registry checks. Native tests cover all four legal rates and six BM/BN pairs,
M8192 boundaries, poisoned disabled pointers and capture rejection. The
standalone loader now validates descriptor and per-file hashes, tensor shapes,
and Python-compatible base/factor semantic hashes before upload.

The instruction audit shows 15 Python-reference launches versus 16 native
launches, with 5,076,434 versus 5,080,407 source-correlated instructions. The
bridge adds one SV FP16-to-FP32 conversion (3,312 instructions); common kernel
resources are unchanged. No device kernels or precision boundaries changed.
Post-profile M33 prepared-call medians are 207.90 us Python eager, 78.85 us
native C and 61.41 us captured Python. These are one-module execution timings,
not ZML or full-model latency. The C bridge is not promoted over graph serving.

The low-level raw C entry still rejects CUDA capture because it allocates
temporary tensors. The ZML handler advertises
`command_buffer_compatible=true` only for its prepared native-graph path:
each exact executable buffer set is warmed outside capture, retained on its
own stream, and replayed or inserted as a child graph. Capture before warmup
fails closed. Replicated sharding semantics remain unchanged. Caller-owned
workspace, native grouped/fused correction, TP-aware lowering and broader
model validation remain required.

Window package loading now has an explicit `window_only` ownership mode. CUDA
loads retain continuous `window_words` on the execution device and keep the
canonical planar reconstruction CPU-side for legacy/debug access; Hopper and
other window consumers never repack or retain a device planar source. The
window-owned module fails closed if a shape selects an unsupported planar
fallback, so legacy callers must request a normal planar module explicitly.

On the non-unified gfx950 path, `qvq_p32_amd_kernel_candidates(M, N, K)`
publishes the bounded `QVQAMDLaunchConfig` sweep (including the measured
shape heuristic). The unified `window_kernel_candidates` API maps those exact
launch choices to `P32WindowConfig(algorithm="amd_gfx950")`, and
`explicit_window_inner` forwards the selected BM/BK/warp/stage values to
`qvq_p32_amd(..., cache_weight=False, launch_config=...)`. A caller can warm,
correctness-check and benchmark the same configs before capture. Cold
decoder/cache preparation is rejected during CUDA capture, so the selected
launch is fixed before graph replay.

For atomic SwiGLU module replay, rank8 fitting is deferred until the complete
gate/up/down triplet is selected. The fitter receives the selected candidate's
serialized P32 tensors plus the immutable dense teacher snapshot, and accepted
factors are appended to that same payload before host staging. This preserves
the base-payload binding when the selector chooses a nonzero candidate arm.
When output alignment is enabled, both ordinary and atomic modules defer the
fit one step further, until alignment has finished revising SU/SV.

The native ABI now exposes `recovery_projection=1` as the graph-safe
`concurrent_reference` producer. A prepared graph owns its auxiliary CUDA
stream and ready/done events; the producer consumes the same transformed
activation `X'` as the window decoder and joins before rank expansion. Raw ABI
calls retain the synchronous reference behavior, while graph replay performs
no stream/event allocation. The H200 K=N=2048 BM64/BN64 spot sweep measured
52.42%, 35.60%, and 34.76% marginal overhead at M=128, 2048, and 8192,
respectively. These measurements keep the policy opt-in and below no default
promotion budget; large-M rank8 remains shape- and kernel-dependent.

The native ABI also exposes `recovery_projection=2` as the concurrent FP16
Tensor Core producer. It is tagged `unverified_tensor_core` and is available
only to fast-mode tuning until every shape passes the local MAE/max-error
contract and independent model-quality confirmation. The reference projection
modes and correction-off path are unchanged.

An H200 graph-replay spot sweep for K=N=2048, BM64/BN64 measured the
Tensor Core producer's marginal overhead at 35.86% (M=128), 18.64% (M=2048),
and 16.61% (M=8192). This is an improvement over the reference concurrent
producer but remains above the 3--5% promotion target, so the candidate stays
fast-only and requires wider shape/device validation. The target is a scorecard
goal rather than an automatic discard criterion; explicit overhead budgets may
still reject a candidate when requested.

The concurrent producer now launches immediately after `X'` and before the
base WGMMA, with the existing event join retained before expansion. A sequential
H200 spot rerun at K=N=2048, BM64/BN64 measured reference-concurrent overhead of
30.06% at M=8192 and Tensor Core overhead of 15.89% at M=8192. These are
measured improvements and remain available to explicit tuning; the 3--5% value
is a target, not an automatic discard gate.
