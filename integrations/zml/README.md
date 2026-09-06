# P32 window custom call for ZML

This adapter lowers the unified window operator to a real StableHLO custom call
and invokes `qvq_p32_window_linear` on the PJRT-provided CUDA stream. It uses the
existing lossless window payload and Hopper operators. Rank8 is optional input
state of the same call. It does not introduce another checkpoint format.

The initial native reference bridge supports:

- SM90, contiguous FP16 activation/scales/levels/factors/output;
- M=1..8192, power-of-two K=2048..16384 and N=256..16384;
- P32 W2/W2.5/W3/W3.5, the existing unpacked per-tile bank selectors;
- existing M16 or direct BM32/64/128, BN64/128, BK256, stages2, split1;
- optional input/output Hadamard and bias;
- rank8 off, or FP32 projection → FP16 hidden → FP32 expansion/addition.

`Config` exposes the geometry, M range, transform flags and correction flag
as individual compiler attributes. Resolve the correction flag from validated
quantizer metadata and requested fast/balanced/quality mode before tuning.
There is no hard-coded A100-derived M crossover. An external tuner must measure
whole executable candidates under the existing numerical gates and distinguish
device product, module shape/rate, M bucket, TP configuration and correction
state in its cache. A latency result cannot establish model-quality eligibility.
`enumerateCandidates` exposes the M16 and all six direct BM/BN candidates as
ordinary Zig data, so a ZML autotune pass can compile the exact same `linear`
call for each geometry and retain the winner in its shape/device cache. The
list is deliberately not pruned globally: a geometry that wins one M or
projection shape remains available to the tuner for other shapes. After each
candidate has been warmed and measured outside capture, pass its correctness
and median timings to `selectFastest`; the selector applies the same MAE/max
error gate as the Python tuner and uses stable enumeration order for ties.
`benchmarkExecutable` provides the common warmup/result-readiness timing loop
for an already-compiled candidate. Compile the serving executable only after
selection, then warm its prepared native graph before any enclosing ZML/CUDA
graph capture.

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

The Zig adapter owns one handle registry for its loaded runtime. It destroys
every prepared handle after the owning stream is synchronized during
`Runtime.deinit`; callers must destroy ZML executables and enclosing graphs
before deinitializing the adapter. Each handle is keyed by all input/output
buffer addresses, stream and static config, so a different executable or graph
lane cannot reuse a stale native graph. A capture attempted before warmup is a
reported precondition error; it is never silently downgraded to an allocating
raw call. This keeps preparation and allocator activity out of capture and
connects the native handle lifetime to the ZML runtime lifetime.

## Build and verify

Tested against ZML `567434be798db31ad888c586293eecedff10b526`, Zig 0.16,
Bazel 9.1.1, Torch 2.15.0.dev20260901+cu130 and an H200. Copy this directory
into a ZML checkout as `integrations/qvq_window`; the included Bazel targets
use that checkout's `//zml` dependency. No changes to ZML itself are required.

The source distribution includes this adapter directory. The Python wheel
includes the native ABI and existing CUDA sources needed for JIT compilation;
copy the Zig adapter from the source distribution or repository. Both package
builds were checked against the source files byte for byte; artifact hashes are
recorded in `docs/kernels/results/p32_window_native_zml.json`.

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
verified target UUID. Set the documented compiler limits (`MAX_JOBS=8`,
`NINJAFLAGS=-j8`, `CMAKE_BUILD_PARALLEL_LEVEL=8`, `NVCC_THREADS=2`) for JIT builds.
The script runs an idle-device preflight, validates native/reference outputs,
and saves hashes for each fixture buffer. These binary files are test fixtures,
not an alternative deployment format.

From the ZML checkout:

```bash
bazel test //integrations/qvq_window:test --jobs=8
bazel build //integrations/qvq_window:verify --jobs=8 --@zml//platforms:cuda=true
bazel run //integrations/qvq_window:verify --jobs=8 --@zml//platforms:cuda=true -- \
  --fixture=/tmp/p32-window-zml-fixture
```

Set `LD_LIBRARY_PATH` to the matching Torch library directory and CUDA dependency
directory before running the executable. The manifest records the three exact
operator-library paths. The verifier checks input file hashes, compiles distinct
off/on calls and compares returned output bytes with the Python operator's
reference. On the real Q fixture (M33, K=N2048, W2, BM64/BN64), both were bit-exact.
