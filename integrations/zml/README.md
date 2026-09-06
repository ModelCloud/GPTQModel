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

The bridge links LibTorch and the existing QVQ CUDA/WGMMA operator libraries,
but execution does not require Python. The caller must validate the unified
artifact's cryptographic base/factor binding at load and retain all input/output
storage until the submitted stream work completes. The Python helper performs
that validation through `prepare_rank8`; a complete native artifact loader is
still required for standalone deployment.

The reference ABI allocates ATen temporaries. The FFI registration deliberately
sets `command_buffer_compatible=false`; the native entry rejects CUDA capture.
No TP-aware attribute is emitted: operands retain ordinary replicated custom
call semantics. External capture workspace ownership, native grouped consumers,
fused rank8 implementations, broader shapes and TP lowering remain open. This
bridge is not promoted as a performance improvement.

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

The Zig adapter still uses the ordinary reference entry and keeps
`command_buffer_compatible=false`. Connecting prepared-handle ownership to the
ZML executable lifecycle remains required before enabling external command
buffers there. A native child-graph test is not proof of ZML capture safety.

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
