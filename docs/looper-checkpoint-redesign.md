# Looper checkpoint/resume

## Usage

```python
from gptqmodel import CheckpointConfig, CheckpointStopped

try:
    model.quantize(
        calibration,
        checkpoint=CheckpointConfig(
            path="./quant-checkpoint",
            resume="auto",  # also "required" or "never"
            every_layers=1,
            keep_last=2,
        ),
    )
except CheckpointStopped:
    # Reload the original model and repeat with the same calibration/config/path.
    pass
```

The adapter supports the GPTQ (including RTN), AWQ, QQQ, ParoQuant, EXL3,
FP8, GGUF and bitsandbytes processor paths for Llama and Qwen3 MoE, with
`offload_to_disk=True`. EoRA continuation is supported for GPTQ, AWQ, QQQ
and ParoQuant, the methods that support EoRA generation without checkpointing.
Packed formats use explicit constructor schemas and the same kernel/config
validation as ordinary quantization; checkpointing does not enable unavailable
kernels or deprecated export formats. Other model families, dynamic exclusions,
rotation, embedding/lm_head quantization and GPTAQ/FOEM remain unsupported.
Checkpointed quantization runs on the main thread so it can temporarily install
and restore SIGINT/SIGTERM handlers. Normal quantization is unchanged.

`auto` starts anew only when CURRENT is absent. It does not silently start over
when a published checkpoint is incompatible or all retained generations are
corrupt. `required` rejects a missing checkpoint; `never` requires a new
directory. Legacy `GPTQMODEL_RESUME` is retired: old marker/offload files cannot
provide a complete continuation and are not migrated or replayed.

## Responsibilities

| Component | Responsibility |
| --- | --- |
| LoopPlan, LoopContext | Stable step identities and startup context |
| LoopExtensions.start() | Install continuation and validate next-step cursor |
| LoopBoundary.quiesce() | Drain preceding finalizers before capture |
| CheckpointExtension | Frequency, safe-stop policy, capture/publish coordination |
| QuantizationCheckpointAdapter | Compose packed modules, dense weights, processor continuations, scheduler and RNG |
| Processor continuation protocol | Own caches, counters, results, statistics and method-specific state |
| Packed-module schemas | Explicit constructors for standard, QQQ, ParoQuant, EXL3 and BNB modules |
| ContinuationCodec | Versioned tagged values and tensors; no pickle |
| CheckpointStore | Writer lease, immutable objects, publication and integrity |

Both the calibration looper and weight-only looper expose the same extension
boundaries and shared execution-placement interface. The layer stage only skips work before the restored cursor and publishes its
existing execution boundaries. It does not know checkpoint filenames, signal
policy, model serialization or restoration details. The old original-weight
replay helpers and marker branches have been removed.

`on_start(context)` is optional for observer extensions. It may return a cursor;
conflicting or out-of-range cursors are rejected. `on_boundary(boundary)` runs
on the orchestration thread. A boundary is valid only within that callback.
Quiescence waits for every outstanding finalizer before surfacing a failure.
Without a checkpoint extension, normal asynchronous finalization is preserved.

## Transaction and recovery

1. Reach a completed layer boundary and quiesce finalizers.
2. Freeze the complete InputCache (masks, positions, kwargs and source inputs),
   shared state, every processor's continuation, Python/NumPy/Torch CPU/CUDA RNG
   state, scheduler/device assignments and quantized-module specifications.
   Preserve materialized nonpacked weights/buffers as well: AWQ/ParoQuant can
   modify normalization weights. Preserve tied parameter identity. EoRA stores
   completed LoRA tensors and rank/path metadata through an explicit schema;
   ParoQuant stores clean replay inputs and AWQ stores calibration counters.
3. Stream new finalized artifacts into content-addressed objects. Earlier
   artifacts are reused by immutable reference, not recopied into RAM.
4. Persist a manifest naming the continuation and **all** completed artifacts.
5. Atomically replace CURRENT with the new and retained manifest references.
6. Collect objects unreachable from retained manifests. Cleanup failure is
   logged separately and does not invalidate a successful commit.

Object files, manifest objects, CURRENT and their containing directories are
flushed. This requires a local filesystem with working atomic rename, locking
and directory fsync; it is not an object-store/NFS durability guarantee.
Unsupported flushing fails rather than silently weakening durability.

Recovery considers only generations referenced by CURRENT, never arbitrary
files discovered by scanning. All referenced objects are SHA-256 verified.
A corrupt newest generation falls back to the preceding complete published
generation; malformed CURRENT or no complete generation fails closed.

Identity covers the execution plan, exact source config/shard contents,
complete prepared calibration cache, serialized quantization settings
(excluding disposable offload location and telemetry), processor chain, selected
packed kernel, GPU topology, and Torch/Transformers versions.
Mismatch reports differing top-level fields and never triggers replay.

GPU topology is strict by default: the GPU count, physical GPU UUIDs and available
serial numbers at each logical CUDA index, visible ordering, and ordered quantization/dense/MoE/forward device pools
must match. Multi-GPU-to-single-GPU resume, reordered/replaced GPUs, and changed
device pools are rejected before decoding or restoring tensors. Forward-visible
GPUs are included even when a quantization-device filter excludes them, since
forward work and CUDA RNG state can still depend on them. No device remapping
or topology migration is supported.

Serials are queried through `nvidia-smi` and joined by GPU UUID, never by the
physical ordering returned by that tool (which may differ from CUDA visibility).
When serials are unavailable, UUID matching remains mandatory. If a serial was
recorded but cannot be verified on resume, strict resume fails closed.
Topology schema version 2 includes serial availability; older topology schemas
are rejected rather than silently upgraded.

`CheckpointConfig(path, skip_strict_gpu_check=False)` is the default.
The explicit opt-in `skip_strict_gpu_check=True` bypasses only UUID/serial
comparison. GPU counts, logical indices, ordered pools, GPU model/capability,
source model, calibration, algorithm and runtime identity still must match.
An unconditional warning identifies this override. It cannot enable a
multi-GPU-to-single-GPU resume, and does not guarantee bit-exact future arithmetic
on different physical hardware.

The looper exposes its execution-placement state through the extension context.
Its round-robin device-assignment cursor and module-device map are checkpointed,
so skipping completed layers does not restart GPU assignment from device zero.
Continuation tensors and device-valued metadata retain explicit device indices;
CPU tensors stay on CPU and active CUDA tensors return to their recorded GPUs.
Completed packed modules intentionally remain offloaded/meta until saving,
except EXL3, whose ordinary completed state uses CPU buffers read by its writer.
The composed adapter schema is version 3; topology and tensor codec schemas
remain version 2. Older adapter checkpoints are rejected rather than guessed
or migrated.

Restore verifies every continuation tensor's exact dtype, shape, raw element bits
(including signed zeros and NaN payloads), and `device:index` after transfer.
It also reads back and verifies scheduler state and CPU/CUDA/Python/NumPy RNG
state. Mismatch raises before the layer loop resumes, even if telemetry is off.
GPU tensor verification copies the restored tensor back to CPU for bytewise
comparison; this deliberately adds resume-time transfer cost for safety.

Enable lifecycle telemetry with `QuantizeConfig(telemetry=TelemetryConfig(device=True))`
(`TelemetryConfig` is exported from `gptqmodel`). The default is disabled.
The old environment switch is no longer read. This nested configuration is
serialized in quantization metadata, but excluded from checkpoint algorithm
identity so diagnostics can be enabled only on resume. A per-invocation context
propagates the setting explicitly to device workers, async offloaders and
finalizer watchers without changing the environment or leaking into reused workers.
`checkpoint_prepare` / `checkpoint_topology_validated` show the physical identity
gate; capture/tensor/commit events identify what was durably published.
`checkpoint_tensor_restored` reports expected/actual devices and separate exact
state/placement checks. Device-valued metadata, scheduler state, and CUDA RNG
indices have explicit restore events. `checkpoint_adapter_restored` distinguishes
disk-backed packed `meta` tensors from live CUDA continuation tensors.
`checkpoint_restore_complete` is emitted only after all restore checks succeed;
rejection/failure events never masquerade as successful restore. An identity
override is reported as `strict_gpu_check=False`, with its actual physical
identity match result, rather than claiming the GPUs matched.
Telemetry contains placement/shape/dtype metadata, not tensor contents, and uses
the existing thread-safe in-memory device telemetry collector and logger.

Restoration validates tensor schemas and creates quantized modules on `meta`.
Per-attempt save indexes point directly to immutable checkpoint objects.
Completed layers are not forwarded and their original weights are not loaded
to reconstruct calibration. The writer reads packed tensors lazily from those
references; the next layer receives the saved continuation.

SIGINT/SIGTERM handlers only set a stop request. The next boundary checkpoints
and raises CheckpointStopped; workers drain before the lease is released.
SIGKILL requires no handler and loses work after the last committed cursor.

Partial Hessians and unfinished subsets are deliberately not committed. On a
restart, the entire uncommitted layer is run again from its saved input cache:
fresh GPTQ tasks start with zero sample/forward counts and empty Hessian partials.
All calibration batches and earlier subsets within that interrupted layer are
re-executed. Only fully committed layers are skipped.
Cross-device Hessian partials are reduced in stable device-index order, not
worker-arrival order, to avoid restart-dependent floating-point rounding.
ParoQuant activation batches are ordered by calibration index before optimizer
sample selection. EXL3 uses private, seeded random generators rather than
resetting global RNG state from concurrent workers. CUDA events order EXL3's
shared writable scratch-buffer use across streams without allocating a large
workspace per thread. These guarantees apply to
uninterrupted execution too; a checkpoint cannot repair nondeterministic inputs
or shared random-state races in the underlying algorithm.

## Scope and operational limits

- Initial calibration capture is still reconstructed for identity validation
  on restart; completed transformer layers are not replayed.
- Source hashing and integrity checks favor correctness over startup speed.
  Continuation serialization copies activation tensors to CPU.
- Each attempt gets a private offload directory, retained because live model
  save references may depend on it. After saving and discarding live
  quantization objects, users may remove the checkpoint directory.
  Retention bounds checkpoint generations, not attempt directories.
- Exclusions and endpoint continuation need additional adapter/plan coverage
  before support is enabled; the public adapter rejects them today.
- Checkpoints and source weights must not be edited externally during a run.
  The lease excludes competing checkpoint runs, not arbitrary external tools.

## Validation

Storage tests cover publication errors, corruption/fallback, missing artifacts,
unpublished orphans, retention, leases, thread ownership and SIGTERM/SIGKILL
immediately before/after CURRENT publication. Codec tests cover complete caches,
typed nested containers, noncontiguous tensors and unsupported-state rejection.

Real two-layer Llama and Qwen3 MoE tests compare every saved tensor against an
uninterrupted run using raw-byte equality, including EoRA adapter tensors.
The method/format matrix is in `tests/test_checkpoint_method_compatibility.py`.
Guarded subprocess tests exercise SIGINT, SIGKILL before and
after commits, injected publication failures, and restart without completed-layer
execution. The same driver accepts /monster/data/model/Llama-3.2-1B for larger
local validation. With PYTHON_GIL=0 it asserts the GIL remains disabled after
model-stack imports. Tiny calibration is a recovery test, not a quality benchmark.

Run the opt-in full-size test with
`GPTQMODEL_CHECKPOINT_TEST_MODEL=/monster/data/model/Llama-3.2-1B`,
`CUDA_VISIBLE_DEVICES=0,1`, and `PYTHON_GIL=0`:
`python -m pytest tests/test_checkpoint_local_model.py`.
It tests SIGTERM and a mid-Hessian SIGKILL, then compares all saved tensors,
GPU assignment telemetry and (for SIGKILL) regenerated Hessian audits.

Partial-Hessian fault injection kills both early in a layer and during a later
MLP subset, after earlier subsets have already been quantized. Audit checks
compare fresh-task initialization, every calibration-batch input hash, final
sample/forward counts, pre-quantization Hessian hashes and every saved tensor
against an uninterrupted run. Device tests cover strict topology rejection and
scheduler continuation. Real two-GPU tensor-placement and dense/MoE recovery
tests are included but skip explicitly when fewer than two GPUs are available;
simulated topology tests are not substitutes for those E2E results.

### Export and dependency boundaries

GPTQ BitBLAS is an inference repacker, not a direct quantization packer in this
repository; use a supported GPTQ export and the existing inference conversion.
The deprecated Marlin export is likewise not enabled by checkpointing. AWQ
BitBLAS has its own packer and needs the optional BitBLAS runtime/compiler and
a working CUDA toolchain. Optional-dependency skips are not resume passes.

All method E2E tests compare the saved packed model, and EoRA cases separately
compare saved adapter tensors. They validate recovery correctness, not inference
quality or every possible bit width, group size, model architecture or optional
kernel combination. Unsupported processor chains fail explicitly instead of
silently losing their state.

### Host validation record (2026-09-06)

Validation used two RTX 4090 GPUs, Python 3.14.7 free-threaded,
`PYTHON_GIL=0`, and Torch `2.15.0.dev20260828+cu130`. The host's installed
Transformers was 5.5.4; temporary, test-only import shims supplied missing vision
and recurrent-mask imports. These text-model tests do not exercise those shims'
functionality and do not replace CI against the project's supported dependency
versions. Optional packages were installed in temporary test directories, not
added as unconditional runtime dependencies.

Completed validation includes:

| Suite | Result |
| --- | --- |
| Checkpoint store, codec, topology/serial gates, processor state, telemetry, module schemas and focused race tests | 138 passed |
| Dual-GPU recovery, looper and thread-pool regression suite | 121 passed, 2 skipped, 1 xpassed |
| CPU dense/MoE recovery, including partial-Hessian kills | 16 passed |
| Full local Llama-3.2-1B, SIGTERM and mid-Hessian SIGKILL | 2 passed, exact saved tensors and GPU placement |
| QQQ/ParoQuant with EoRA, dense/MoE signal and kill recovery | 8 passed |
| AWQ with EoRA, dense/MoE signal, kill and publication-error recovery | 8 passed |
| GPTQ with EoRA, dense/MoE seven fault modes | All 14 cases passed, including the repeated timeout case noted below |
| AWQ without EoRA, dense/MoE four fault modes | 8 passed |
| RTN/FP8/GGUF/BNB dense four fault modes | 16 passed |
| RTN/FP8/GGUF/BNB/QQQ/ParoQuant/EXL3 MoE signal and kill recovery | All 14 cases passed after the EXL3 workspace fix |
| Additional GPTQ v2/P and AWQ GEMV/GEMV_FAST/LLM-AWQ layouts with EoRA | 5 passed |
| QQQ/ParoQuant/EXL3 dense faults plus ParoQuant eager-mode recovery | 13 passed |
| Export-layout tests and unsupported-export guards | 11 passed; optional BitBLAS skipped in the default environment |
| AWQ BitBLAS, explicitly installed with CUDA toolchain configured | 1 passed, exact SIGKILL/resume comparison |
| Explicit packed schemas: GPTQ widths/layouts, every GGUF alias, FP8 scales, BNB, QQQ, ParoQuant, EXL3 | 33 passed (also included above) |

The method matrix is executed in separate selections, not as the entire
repository test suite. GPTQ/GPTQ+EoRA device tests assert that both GPU indices
actually performed Hessian work; other method tests expose the same two-GPU
topology. A publication-error GPTQ+EoRA case timed out once during overlapping
validation; its isolated rerun and three further consecutive repetitions passed.
The subprocess guard now preserves timeout diagnostics and cleans up its own
child process group. No comparison tolerance was relaxed to obtain a pass.

BitBLAS validation additionally used `bitblas==0.1.0.post1`, its CUDA 12 runtime
libraries, and `CUDA_HOME=/usr/local/cuda-13.3` with the compiler directory on
`PATH`. `TVM_IMPORT_PYTHON_PATH` pointed at BitBLAS's bundled TVM so its import
setup did not discard the test subprocess's `PYTHONPATH`. The successful E2E
run follows fixes for AWQ method selection, compute dtype and zero-buffer schema;
the earlier dependency/compiler and schema failures are not counted as passes.
