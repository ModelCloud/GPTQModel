# Looper checkpoint redesign

Checkpointing requires an explicit execution boundary: completed module
artifacts and the continuation needed to start the next step must be committed
together. Replaying original weights after quantizing earlier layers is not
an equivalent continuation for GPTQ.

## Boundary foundation (implemented)

`ModuleLooper(..., extensions=(extension,))` accepts internal extensions with
an `on_boundary(boundary)` method. This is not yet a public `quantize()` option.
The layer stage invokes extensions synchronously on the orchestration thread
after scheduling finalization and before handing off to the next step.

`LoopStep` separates the step kind, model-layer index, and name. It is the
foundation for a stable execution plan, not yet a serialized resume cursor.
The current boundary reports executed steps only; explicit skipped-step plan
entries are part of the plan work below.

`boundary.quiesce()` waits for all retained finalization futures through the
current step, then returns their results or raises their error. It waits for
all workers even when one fails. Extensions that only observe progress need
not wait; successful completed futures are released between boundaries.
Boundaries cannot be retained after the callback or used from worker threads.
Extension exceptions propagate to the caller. With no extensions installed,
existing asynchronous scheduling is unchanged.

Legacy resume replay and extensions cannot currently be combined. The guard
prevents extensions from observing an incomplete sequence of boundaries while
the new restoration protocol is under construction.

## Next implementation stages

1. Build a stable execution plan containing endpoint, layer, and skipped steps;
   add a cursor pointing to the next step. This eliminates endpoint index
   ambiguity and makes exclusions explicit.
2. Add continuation capture/restore adapters for the full InputCache,
   supported processor and model state, and RNG state where applicable.
   Capture runs only after quiescence. Unsupported state is rejected explicitly;
   arbitrary Python objects are not pickled.
3. Extend finalization results with immutable artifact references and module
   quantization specifications. Committed artifacts cannot be overwritten by
   speculative work or ordinary offload cleanup.
4. Implement CheckpointExtension and CheckpointStore. Write and validate all
   artifacts and continuation data, then publish a versioned manifest and
   atomically advance CURRENT. A run-level lease rejects duplicate writers.
   Retain at least two complete generations and collect unreferenced artifacts.
5. Restore completed modules lazily through a model adapter, install the saved
   continuation, and begin at its cursor without replaying completed layers.
   Replace the legacy resume branches only once this path is validated.
6. Expose CheckpointConfig with path, resume policy, checkpoint frequency, and
   retention. Provide inspection and field-level compatibility diagnostics.

SIGINT/SIGTERM should request a checkpoint at a boundary; signal handlers only
set a flag. SIGKILL relies on the last committed manifest. Host failure requires
explicit file/directory flushing according to the store's durability contract.

Canonical run identity must cover the plan, source checkpoint, prepared
calibration, and algorithm-affecting settings. Storage and compatibility logic
belong outside the looper.

## Validation

The foundation has tests for asynchronous default behavior, outstanding work
across steps, draining failures, callback lifetime, thread ownership, and an
extension attached to real tiny MoE quantize/save/reload execution.

The eventual checkpoint path requires guarded subprocess tests comparing
uninterrupted and resumed dense/MoE outputs. Cover normal stops, SIGKILL at each
publication boundary, storage failures, corrupt generations, exclusions,
repeated resumes, and concurrent-writer rejection. Verify the GIL is disabled
after imports. A saved output file alone is not evidence of resume equivalence.
