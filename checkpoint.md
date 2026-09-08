# Quantization checkpointing and resume

GPT-QModel can checkpoint long quantization jobs at completed transformer-layer
boundaries. If a process is stopped, restarted, or killed, a later invocation
can restore the latest complete checkpoint and continue without quantizing the
completed layers again.

Checkpointing is designed for recovery, not as the final model export:

1. Load the original, unquantized model.
2. Run `model.quantize(..., checkpoint=CheckpointConfig(...))`.
3. If the run stops, repeat steps 1 and 2 with the same inputs and checkpoint
   path.
4. After quantization completes, call `model.save(...)`.

## Quick start

Use an explicit checkpoint path for any checkpoint that must survive a process
restart:

```python
from pathlib import Path

from datasets import load_dataset

from gptqmodel import CheckpointConfig, CheckpointStopped, GPTQConfig, GPTQModel

model_id = "meta-llama/Llama-3.2-1B-Instruct"
output_path = Path("Llama-3.2-1B-Instruct-gptqmodel-4bit")
checkpoint_path = Path("checkpoints/Llama-3.2-1B-Instruct-gptqmodel-4bit")

calibration_dataset = load_dataset(
    "allenai/c4",
    data_files="en/c4-train.00001-of-01024.json.gz",
    split="train",
).select(range(1024))["text"]

quant_config = GPTQConfig(
    bits=4,
    group_size=128,
    offload_to_disk=True,
)

model = GPTQModel.load(model_id, quant_config)

try:
    model.quantize(
        calibration_dataset,
        batch_size=1,
        checkpoint=CheckpointConfig(
            path=checkpoint_path,
            resume="auto",
            interval="layer:1",
            keep_last=2,
        ),
    )
except CheckpointStopped:
    print(f"Safe checkpoint committed to {checkpoint_path}; run this script again.")
    raise SystemExit(75)

model.save(output_path)
```

Run the same script again after a stop or crash. `resume="auto"` detects the
published checkpoint, validates it against the new invocation, restores its
state, and starts at the next unfinished layer.

Do not call `model.save(...)` from the `CheckpointStopped` handler. Reload the
original model and resume first.

## Stopping safely

Checkpointed quantization temporarily handles `SIGINT` and `SIGTERM` on the
main thread:

- **Ctrl+C / `SIGINT`** and **`SIGTERM`** request a safe stop. Quantization
  finishes the current layer, waits for its finalizers, commits a checkpoint,
  and raises `CheckpointStopped`.
- **`SIGKILL`**, a process crash, or a host failure cannot run a handler.
  Restarting resumes from the last checkpoint that had already been published.
  Work performed after that checkpoint is repeated.

Only fully completed layers are checkpointed. If interruption occurs while a
layer or one of its subsets is being quantized, the whole uncommitted layer is
re-run. Partial Hessians are not restored.

The stop is intentionally not immediate. For a large layer, Ctrl+C may take
time to return because the layer must reach a safe boundary first.

Checkpointed quantization must run on the Python main thread so GPT-QModel can
install and restore the signal handlers.

## `CheckpointConfig` reference

```python
CheckpointConfig(
    path="auto",
    resume="auto",
    interval="layer:1",
    keep_last=2,
    strict_device_check=True,
)
```

| Field | Default | Behavior |
| --- | --- | --- |
| `path` | `"auto"` | Checkpoint root. An explicit durable path is recommended. `"auto"` resolves to `quantize_config.offload_to_disk_path`. |
| `resume` | `"auto"` | Controls whether an existing published checkpoint is loaded. See [Resume policies](#resume-policies). |
| `interval` | `"layer:1"` | Publish after every positive number of completed layers, for example `"layer:2"`. The final layer always publishes a checkpoint. |
| `keep_last` | `2` | Number of published generations retained for fallback. Must be at least `2`. |
| `strict_device_check` | `True` | Require the physical GPU UUID/serial identity recorded by the checkpoint to match. |

Checkpointing requires `quantize_config.offload_to_disk=True`.

### Resume policies

| Policy | No published checkpoint | Compatible checkpoint exists | Incompatible or corrupt checkpoint |
| --- | --- | --- | --- |
| `"auto"` | Start a new run | Resume | Raise an error; never silently start over |
| `"required"` | Raise an error | Resume | Raise an error |
| `"never"` | Start a new run | Raise an error and require a path with no published checkpoint | Raise an error |

Use `"auto"` for a script that may either start or resume. Use `"required"`
when a recovery job must not accidentally start from the beginning. Use
`"never"` when creating a deliberately fresh run.

## Choosing the checkpoint path

An explicit path is the safest choice:

```python
checkpoint = CheckpointConfig(
    path="/fast-local-disk/checkpoints/my-quantization",
)
```

`path="auto"` uses `quantize_config.offload_to_disk_path`. If that offload path
was generated automatically, it can be a temporary directory that changes when
the model is loaded in a new process. A new process would then not find the old
checkpoint. For cross-process or host-restart recovery, either:

- pass an explicit `CheckpointConfig.path`, or
- configure a stable `quantize_config.offload_to_disk_path`.

Keep the checkpoint path separate from the final `model.save(...)` output path
and from the source model directory.

Each invocation creates a private `attempt-*` offload directory below the
checkpoint root. Published checkpoint manifests can reference files retained
from earlier attempts, so do not delete attempt directories while quantization
or final model saving is in progress.

## What must match when resuming

Before restoring tensors, GPT-QModel verifies that the new invocation describes
the same quantization job. Keep all of the following stable:

- source model configuration and safetensors shard contents;
- calibration samples, order, tokenization, and dataset-preparation arguments;
- quantization method, format, bits, group size, damping, fallback, MoE, and
  other quantization settings;
- `batch_size`, calibration concatenation/sorting, and processor chain whenever
  they affect the prepared calibration cache or execution plan;
- selected packing kernel and backend;
- EoRA adapter rank and path, when used;
- PyTorch and Transformers versions;
- visible GPU count, logical ordering, device pools, GPU model/capability, and,
  by default, physical GPU UUIDs and serial numbers.

The disposable offload location and telemetry setting are excluded from the
algorithm identity. Telemetry may therefore be enabled for a recovery attempt
without invalidating an otherwise compatible checkpoint.

An identity mismatch fails closed and reports the top-level fields that differ.
`resume="auto"` does not erase or replace the checkpoint. Restore the original
inputs/environment or intentionally start a new run at a different path.

### Resuming on different GPUs

`strict_device_check=True` is the default and is recommended for reproducible
recovery. It rejects replaced or reordered physical GPUs.

```python
checkpoint = CheckpointConfig(
    path="checkpoints/my-run",
    strict_device_check=False,
)
```

Setting `strict_device_check=False` bypasses only the physical UUID/serial
comparison. It does **not** permit:

- changing the GPU count;
- changing logical CUDA indices or ordered device pools;
- moving a multi-GPU run to one GPU;
- changing GPU model/capability, source data, algorithm, or runtime identity.

Recovery on different physical hardware may not be bit-exact even when this
override is accepted.

## Supported configurations

The current checkpoint adapter supports:

- model `model_type` values `llama` and `qwen3_moe`;
- GPTQ, RTN, AWQ, QQQ, ParoQuant, EXL3, FP8, GGUF, and bitsandbytes
  quantization paths;
- EoRA continuation with GPTQ, AWQ, QQQ, and ParoQuant;
- the same normally supported export formats and packing kernels as ordinary
  quantization.

Checkpointing does not make an otherwise unsupported method, format, kernel, or
optional dependency available.

The following are currently unsupported with checkpointing:

- other model families;
- `true_sequential=False`;
- dynamic per-module quantization/exclusions;
- rotation;
- `lm_head` or embedding quantization;
- GPTAQ or FOEM;
- quantization preprocessor chains;
- adapters other than the supported EoRA `Lora` continuation;
- source models without local safetensors weights.

GPTQ BitBLAS remains an inference repack path rather than a direct quantization
export. The deprecated Marlin export format is also not enabled by
checkpointing.

Unsupported combinations are rejected before the layer loop starts rather than
being partially checkpointed.

## Checkpoint storage and durability

A checkpoint root contains:

- `CURRENT`, which atomically publishes the retained checkpoint generations;
- `objects/`, containing content-addressed continuation and tensor objects;
- `writer.lock`, preventing two quantization jobs from writing the same root;
- per-invocation `attempt-*` offload directories.

Objects referenced by a published generation are SHA-256 verified during
recovery. If the newest generation is damaged but an older retained generation
is complete, GPT-QModel falls back to that older generation. It never scans
unpublished files and guesses that they are recoverable.

Use a local filesystem that supports file locking, atomic rename, file and
directory `fsync`. Durability is not guaranteed for object stores or network
filesystems that do not provide those semantics.

Only one writer may use a checkpoint root at a time. Separate concurrent jobs
must use separate checkpoint paths.

`keep_last` limits published generations, not the number of `attempt-*`
directories. Checkpoint storage may therefore remain large after retries.
After `model.save(...)` has completed and the live quantization objects are no
longer needed, the checkpoint root can be removed.

Never edit `CURRENT`, manifests, objects, attempt files, or source weights
during a run.

## Recovery recipes

### Resume only if progress exists

Use this for a scheduler retry that should fail rather than repeat the whole
job:

```python
checkpoint = CheckpointConfig(
    path="checkpoints/my-run",
    resume="required",
)
```

### Force a clean run

Use a new checkpoint path with `resume="never"`:

```python
checkpoint = CheckpointConfig(
    path="checkpoints/my-run-clean",
    resume="never",
)
```

Do not point `"never"` at a root that already contains a published checkpoint.

### Recover after a hard kill

1. Leave the checkpoint directory unchanged.
2. Restore the same software environment and GPU visibility/order.
3. Reload the original model with the same quantization configuration.
4. Recreate the same calibration data in the same order.
5. Call `model.quantize(...)` with the same path and `resume="auto"` or
   `resume="required"`.
6. Save the model only after quantization returns normally.

### Recover from an incompatible checkpoint error

The error lists identity areas that changed, such as `source`,
`quantization`, `calibration`, `packed_kernel`, `device_topology`, `runtime`, or
`execution_plan`.

- If the change was accidental, restore the original setting and retry.
- If the change was intentional, use a new checkpoint path.
- Do not delete the old checkpoint until you are certain it is no longer
  needed.

### Recover from corruption

GPT-QModel automatically tries retained generations from newest to oldest.
If no complete published generation remains, recovery fails rather than
silently restarting. Restore the checkpoint directory from storage backup or
start a new run at a different path.

## Operational gotchas

- **A checkpoint is not a saved model.** Always run `model.save(...)` after a
  successful quantization return.
- **Resume starts from the original model.** Do not load a partially quantized
  attempt directory as the source model.
- **Initial calibration preparation still runs on resume.** It is required to
  validate identity, although completed transformer layers are not replayed.
- **Resume can have noticeable startup cost.** Source files are hashed,
  checkpoint objects are verified, calibration is prepared, and restored GPU
  tensors are checked byte-for-byte.
- **The checkpoint directory can be large.** It stores completed packed
  modules, continuation data, calibration state, and private offload attempts.
- **Intervals trade I/O for repeated work.** A larger interval reduces
  checkpoint writes but a hard failure may repeat more completed layers since
  the last publication.
- **Ordinary Python exceptions do not create a partial checkpoint.** Recovery
  uses the last fully published generation.
- **Legacy `GPTQMODEL_RESUME=1` is retired.** Old marker/offload files do not
  contain a complete continuation and cannot be migrated. Use
  `CheckpointConfig` with a new checkpoint directory.

## Optional checkpoint telemetry

Device telemetry can expose checkpoint preparation, validation, capture,
commit, restoration, and failure events:

```python
from gptqmodel import GPTQConfig, TelemetryConfig

quant_config = GPTQConfig(
    bits=4,
    group_size=128,
    offload_to_disk=True,
    telemetry=TelemetryConfig(device=True),
)
```

Telemetry records placement, shape, dtype, and lifecycle metadata. It does not
record tensor contents. Because telemetry is excluded from checkpoint algorithm
identity, it may be enabled only for a resume attempt when diagnosing a failure.
