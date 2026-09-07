# P32 window custom call for ZML

This adapter lowers the unified window operator to a real StableHLO custom call
and invokes `qvq_p32_window_linear` on the PJRT-provided CUDA stream. It uses the
existing lossless window payload and Hopper operators. Rank8 is optional input
state of the same call. It does not introduce another checkpoint format.

The initial native reference bridge supports:

- SM90, contiguous FP16 activation/scales/levels/factors/output;
- M=1..8192, 256-wide K=2048..16384 and N=256..17408. Input/output
  Hadamard remains available for power-of-two widths; transform-free
  composite projections such as K=5120/N=17408 are accepted by the same
  window ABI.
- P32 W2/W2.5/W3/W3.5, the existing unpacked per-tile bank selectors;
- existing M16 or direct BM32/64/128, BN64/128, BK256, stages2, split1;
- optional input/output Hadamard and bias;
- rank8 off, or FP32 projection → FP16 hidden → FP32 expansion/addition. The
  prepared graph ABI also exposes `concurrent_reference` and an explicitly
  unverified Tensor Core producer on owned auxiliary streams; both join before
  the epilogue.

`Config` exposes the geometry, M range, transform flags and correction flag
as individual compiler attributes. Resolve the correction flag from validated
quantizer metadata and requested fast/balanced/quality mode before tuning.
There is no hard-coded A100-derived M crossover. An external tuner must measure
whole executable candidates under the existing numerical gates and distinguish
device product, module shape/rate, M bucket, TP configuration and correction
state in its cache. A latency result cannot establish model-quality eligibility.
Packages may include a versioned `kernel_tuning` hint produced by the Python
quantizer. That hint is bound to the exact P32 state and optional rank8 factors
and is revalidated by the Python package/artifact loaders. It is advisory only:
ZML must still enumerate and benchmark candidates for its own device, shape,
rate, M bucket, TP layout and correction state rather than applying a cached
choice from another environment.

Automatic candidate selection is intentionally distinct from enabling rank8 by
default. Once a module is prepared with rank8 enabled in `fast` mode, ZML
selects the fastest locally accepted candidate, including an unverified Tensor
Core producer, even when its measured overhead is above the 3--5% scorecard
target unless an explicit overhead cap is supplied. The package default keeps
correction off because factors are optional and the Tensor Core arithmetic
signature is not yet certified; enabling it globally would alter the existing
window output. The selected policy is resolved and captured outside graph
capture for the exact device/shape/M bucket, then replayed without a dynamic
branch. Balanced and quality modes retain their stricter audit and arithmetic
requirements.
`enumerateCandidates` exposes the M16 and all six direct BM/BN candidates as
ordinary Zig data, so a ZML autotune pass can compile the exact same `linear`
call for each geometry and retain the winner in its shape/device cache. It
also emits `recovery_projection=0` (separate reference),
`recovery_projection=1` (concurrent reference), and
`recovery_projection=2` (concurrent Tensor Core, unverified/fast-only) for
every geometry. For
transform-free output modules each projection is paired with
`recovery_kernel=0` (separate epilogue) and `recovery_kernel=1` (fused
epilogue), yielding 42 candidates for transform-free and power-of-two
output-Hadamard modules; composite output-Hadamard modules yield 21. The
off-state enumeration retains the same indices while ignoring A/B, so every
on-state projection has a matched correction-off baseline. The list is
deliberately not pruned globally: a geometry that wins one M or projection
shape remains available to the tuner for other shapes. Use
`enumerateCandidatesForShape` when a shape-aware pass should measure tile
compatible candidates first; it preserves the complete candidate set and is
only an ordering hint. After each candidate has been warmed and measured
outside capture, pass its correctness and median timings to `selectFastest`;
the selector applies the same MAE/max error gate as the Python tuner and uses
stable enumeration order for ties.
The companion `candidateShapeScoreForShape` API exposes the deterministic
ordering score for telemetry and cache records; it does not change eligibility.

For transform-free outputs, the native ABI uses the fused FP32-add/FP16-store
epilogue for both separate-reference and concurrent rank8 projections. The
same graph-safe epilogue now folds the power-of-two output Hadamard path;
composite output widths retain the reference epilogue until a native fused
transform is certified. The Tensor Core projection remains an explicitly
unverified candidate. ZML enumerates the fused output-Hadamard candidate only
when its native power-of-two contract is valid.

Grouped P32 tuning follows the same contract through
`p32GroupedCandidateSetForShape`: child split tuples matching the exact
`(M,K,{N_i},transition_bits)` policy are measured first, while alternate
per-child split and launch candidates remain available to the selector.
`p32GroupedShapeScoreForShape` exposes the corresponding score for grouped
telemetry without requiring callers to duplicate the split policy.

The Tensor Core producer remains unverified and is excluded from balanced and
quality selection. An H200 K=N=2048 BM64/BN64 spot sweep measured its matched
correction overhead at 35.86% / 18.64% / 16.61% for M=128 / 2048 / 8192;
these results improve on the reference concurrent producer. The 3--5% value is
a scorecard target, not an automatic discard criterion; only an explicitly
requested recovery budget rejects a candidate for overhead.
`benchmarkExecutable` provides the common warmup/result-readiness timing loop
for an already-compiled candidate. Compile the serving executable only after
selection, then warm its prepared native graph before any enclosing ZML/CUDA
graph capture.

For recovery-aware tuning, compile matching correction-off and correction-on
executables for each eligible geometry and call `benchmarkRecoveryPair`. It
returns both complete-executable medians and the signed marginal overhead;
`RecoveryPairMeasurement.meetsTarget` is a reporting predicate, while
`selectFastestWithRecoveryGate` can enforce an explicit nonnegative budget on
every candidate. A supplied budget rejects candidates without a measured pair
or above the limit; omitting it preserves report-only tuning. The
quantizer's audit and arithmetic-signature gates still decide whether recovery
is eligible for `balanced` or `quality`, and the pair result is keyed with the
same device, shape, rate, TP and artifact identity as the geometry winner.
The checked-in H200 reports include unbudgeted winners above 5%; the ZML test
suite keeps those rows selectable and verifies that only the explicitly
budgeted run removes them from winner selection while retaining them in the
report.

Use `selectFastestWithPolicy` when the serving graph has an explicit quality
mode. `quality` admits only `reference_fp32_v1`, `balanced` admits that
signature plus `certified_tensor_core_v1`, and `fast` is the permissive
correction-off policy. Alternative signatures must be certified by the
producer before they can enter a balanced graph.
The verifier's version-2 tuning report records the selected policy for each
correction state and the arithmetic signature on every candidate row.
Its correction-on policy comes from the fixture's `quality_mode` by default;
pass `--quality-mode fast|balanced|quality` to override it for an explicit
experiment. The chosen mode is written back to the report so the tuning
decision is reproducible.

The fused epilogue is an explicit tuning candidate and is not promoted by
default: initial H200 measurements for K=N=2048 showed substantial marginal
cost, so rank8 overhead must be measured per M/shape/device before selection.
The bridge links LibTorch and the existing QVQ CUDA/WGMMA operator libraries,
but execution does not require Python. `loadArtifact` is the standalone native
handoff: it validates the versioned manifest, the descriptor-level
`payload_sha256` binding, exact file names, dtypes, shapes, byte counts and
per-file SHA-256 hashes before uploading any payload. It accepts the
FP32 SU/SV precision used by QVQ checkpoints (and FP16 bias where present); the
native ABI applies its declared FP16 transform boundary. Recovery metadata still
has to be accepted by the deployment policy; the loader recomputes and checks
the Python fit's semantic base/factor digests before uploading any payload, so
the correction cannot be attached to a different base or factor tensor set.
Callers retain all input/output storage until the submitted stream work
completes.

The low-level `qvq_p32_window_linear` entry allocates ATen temporaries and
rejects CUDA capture. The ZML adapter therefore never calls it directly. Its
first eager invocation prepares a retained native graph for the exact
executable buffer addresses; subsequent invocations submit that graph, and a
capture invocation inserts it as a child node. The FFI registration advertises
`command_buffer_compatible=true` only for this prepared path. A graph capture
before the eager warmup fails explicitly instead of allocating or timing inside
capture. No TP-aware attribute is emitted: operands retain ordinary replicated
custom-call semantics. Native grouped consumers, fused rank8 implementations,
broader shapes and TP lowering remain open. This bridge is not promoted as a
performance improvement.

### Prepared native graphs

The native library also exposes `qvq_p32_window_graph_create`,
`qvq_p32_window_graph_run` and `qvq_p32_window_graph_destroy`. Creation takes
the ten linear buffers in argument order and a fixed config, warms the same
operator, then captures it into a retained ATen private pool. Preparation
requires a non-default stream outside capture and synchronizes that stream.
Run either replays the prepared graph or inserts a child graph into an active
capture, preserving incoming dependencies and subsequent stream work.

The caller retains all input/output/artifact buffers at their original addresses
and keeps artifact values immutable. Keep the owning CUDA stream and loaded
operator libraries alive until handle destruction. Input contents may change between requests.
Use one handle per execution lane; submit on its owning stream. Enclosing
graphs must also replay on that stream, and must be destroyed before releasing
the handle. Destroy waits for stream work before releasing the private pool.
This is an explicit low-level ownership contract; arbitrary external pointer
mutation or overlapping use is not automatically tracked by the C ABI.

The Zig adapter owns one bounded (256-entry) handle registry for its loaded
runtime. It destroys every prepared handle after the owning stream is
synchronized during `Runtime.deinit`, and LRU eviction destroys older handles
before inserting new executable/buffer sets. Each entry is keyed by all
input/output addresses, byte sizes, element dtypes, stream and static config;
per-entry locks serialize replay and prevent eviction during an in-flight call,
while unrelated keys remain concurrent. Callers must destroy ZML executables
and enclosing graphs before deinitializing the adapter. A capture attempted
before warmup is a reported precondition error; it is never silently downgraded
to an allocating raw call. This keeps preparation and allocator activity out
of capture and connects native handle lifetime to the ZML runtime owner.

## Build and verify

Tested against QvQ `6df72173` and ZML-Ultra `4d8ce52`, Zig 0.16,
Bazel 9.1.1, Torch 2.15.0.dev20260901+cu130 and an H200. Copy this directory
into a ZML checkout as `integrations/qvq_window`; the included Bazel targets
use that checkout's `//zml` dependency. No changes to ZML itself are required.

This adapter is shipped as the `//integrations/qvq_window` target in ZML-Ultra.
The Python wheel includes the native ABI and existing CUDA sources needed for
JIT compilation; the Zig target supplies the graph-safe StableHLO/PJRT bridge.
Artifact hashes are recorded in QvQ's
`docs/kernels/results/p32_window_native_zml.json`.

From QvQ, create the native libraries and a disposable correctness fixture from
an already-bound real window package and its disjoint captured audit activations:

```bash
export CUDA_HOME=/usr/local/cuda
export CUDA_PATH=/usr/local/cuda
export CUDACXX=/usr/local/cuda/bin/nvcc
python scripts/verify_qvq_window_abi.py \
  --package /path/to/module.pt \
  --activations /path/to/module.activations.pt \
  --fixture /tmp/p32-window-zml-fixture \
  --output /tmp/p32-window-native-real.json
```

Keep `CUDA_DEVICE_ORDER=PCI_BUS_ID` and restrict `CUDA_VISIBLE_DEVICES` to the
verified target UUID. Set bounded compiler parallelism from the host quota for
JIT builds. A safe default is half the available CPU cores (with at least one
worker), reduced further if memory or swap pressure increases:

```bash
BUILD_CORES="$(nproc)"
BUILD_JOBS="$((BUILD_CORES / 2))"
if [ "$BUILD_JOBS" -lt 1 ]; then BUILD_JOBS=1; fi
export MAX_JOBS="$BUILD_JOBS"
export NINJAFLAGS="-j$BUILD_JOBS"
export CMAKE_BUILD_PARALLEL_LEVEL="$BUILD_JOBS"
export NVCC_THREADS=2
```
The script runs an idle-device preflight, validates native/reference outputs,
and saves hashes for each fixture buffer. These binary files are test fixtures,
not an alternative deployment format.

From the ZML checkout:

```bash
bazel test //integrations/qvq_window:test --jobs="$BUILD_JOBS"
bazel build //integrations/qvq_window:verify --jobs="$BUILD_JOBS" --@zml//platforms:cuda=true
bazel run //integrations/qvq_window:verify --jobs="$BUILD_JOBS" --@zml//platforms:cuda=true -- \
  --fixture=/tmp/p32-window-zml-fixture \
  --max-recovery-overhead-percent=5 \
  --tuning-output=/tmp/p32-window-zml-tuning.json
```

The fixture exporter can add a hard recovery budget with
`--max-recovery-overhead-percent=5`. The verifier accepts the same option as a direct override (otherwise it uses the manifest value), then pairs each correction-on
candidate with its correction-off measurement and rejects any geometry above
that budget before selecting the serving executable. Without the option, the
paired overhead is retained as tuning telemetry and selection remains
correctness-gated only.

`--tuning-output` writes the complete off/on candidate sweep, correctness gates,
matched recovery overheads, and selected candidate indices as versioned JSON.
When the fixture manifest carries one, it also records the exact
`artifact_payload_sha256` binding.
The report is produced after tuning and before the executable is handed to an
enclosing graph capture; it is a cache/audit artifact and is never read during
captured execution. Persistent consumers must still key any cache by device,
shape, artifact identity, correction state and configuration.

Set `LD_LIBRARY_PATH` to the matching Torch library directory and CUDA dependency
directory before running the executable. The manifest records the three exact
operator-library paths. The verifier checks input file hashes, compiles distinct
off/on calls and compares returned output bytes with the Python operator's
reference. On the real Q fixture (M33, K=N2048, W2, BM64/BN64), both were bit-exact.
