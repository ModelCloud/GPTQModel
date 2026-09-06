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

The initial adapter supports sequential GPTQ for Llama and Qwen3 MoE, with
`offload_to_disk=True`. Other families, dynamic exclusions, rotation,
embedding/lm_head quantization, GPTAQ/FOEM and adapters are rejected explicitly.
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
| GPTQCheckpointAdapter | Module schemas, full cache, shared state, log and RNG |
| ContinuationCodec | Versioned tagged values and tensors; no pickle |
| CheckpointStore | Writer lease, immutable objects, publication and integrity |

The layer stage only skips work before the restored cursor and publishes its
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
   shared state, processor log, Python/NumPy/Torch CPU/CUDA RNG state,
   and quantized-module construction specifications.
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
(excluding disposable offload location), and Torch/Transformers versions.
Mismatch reports differing top-level fields and never triggers replay.

Restoration validates tensor schemas and creates quantized modules on `meta`.
Per-attempt save indexes point directly to immutable checkpoint objects.
Completed layers are not forwarded and their original weights are not loaded
to reconstruct calibration. The writer reads packed tensors lazily from those
references; the next layer receives the saved continuation.

SIGINT/SIGTERM handlers only set a stop request. The next boundary checkpoints
and raises CheckpointStopped; workers drain before the lease is released.
SIGKILL requires no handler and loses work after the last committed cursor.

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
uninterrupted run. Guarded subprocess tests exercise SIGINT, SIGKILL before and
after commits, injected publication failures, and restart without completed-layer
execution. The same driver accepts /monster/data/model/Llama-3.2-1B for larger
local validation. With PYTHON_GIL=0 it asserts the GIL remains disabled after
model-stack imports. Tiny calibration is a recovery test, not a quality benchmark.
