<!-- SPDX-FileCopyrightText: 2026 ModelCloud.ai -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Concurrency guards

GPT-QModel runs quantization and inference from several threads, including on
free-threaded CPython (`PYTHON_GIL=0`). Five process-global resources are
shared by those threads: the ROCm BLAS backend selection, the TF32 precision
setting, the CPU oneDNN BF16x3 matmul precision, libtriton's JIT compiler, and
the transformers model-construction state (default dtype plus the meta-device
and no-init contexts). This page documents how the guards for those resources
behave, and why they are reference-counted, serialized, or classified as
compiler failures.

The shapes are deliberate: a guard that saves and restores a process-global
value per entry is wrong as soon as two callers overlap, and a compiler that is
assumed to be thread-safe is wrong as soon as two threads miss the same kernel
cache entry.

## 1. BLAS, TF32 and CPU BF16x3 guards are reference-counted

`quant_blas_backend` (`gptqmodel/utils/rocm.py`) and
`tf32_high_precision_guard` (`gptqmodel/utils/torch.py`) protect a
process-global setting. They now keep one depth counter and one saved value
under a module-level `RLock`. The first entry captures the user's setting and
applies the override; every later entry only increments the depth. Only the
last exit restores the captured value.

`cpu_bf16x3_scope` (`gptqmodel/utils/cpu_features.py`) is the same shape for
`torch.backends.mkldnn.matmul.fp32_precision`. It is shared by every CPU caller
that runs FP32 matmuls as BF16x3: GGUF CPU and bitsandbytes CPU go through
`cpu_bf16x3_matmul()`, and the EXL3 torch kernel enters the scope directly after
its own shape and environment gate. The lock is an `RLock` because the scope is
re-entrant; a plain `Lock` would deadlock a nested entry.

```mermaid
flowchart TD
    A["quant_blas_backend, tf32_high_precision_guard or cpu_bf16x3_scope"] --> B{"depth == 0"}
    B -- yes --> C["snapshot user setting"]
    C --> D["apply override"]
    D --> E["depth = 1"]
    B -- no --> F["depth += 1, keep current override"]
    E --> G["yield to caller"]
    F --> G
    G --> H["finally: depth -= 1"]
    H --> I{"depth == 0"}
    I -- yes --> J["restore snapshot"]
    I -- no --> K["another caller still inside"]
```

The same flow serves a single caller, a nested caller on one thread, and two
concurrent `quantize()` calls. The first caller to *enter* owns the snapshot;
the first caller to *exit* does not restore anything unless it is also the
last one inside.

Before the change, each entry snapshotted the current value and each exit
restored it. Quantize A entered first and set rocBLAS; quantize B entered
second, saw rocBLAS, and skipped the override; A returned and restored
hipBLASLt while B was still quantizing. B's remaining layers then ran on the
backend whose per-process kernel selection #4075/#4079 removed.

| Mechanism | Before | After |
| --- | --- | --- |
| BLAS depth | none; per-entry save/restore (`utils/rocm.py:240-283` at `718256fd7`) | `_QUANT_BLAS_GUARD_LOCK`, `_QUANT_BLAS_GUARD_DEPTH`, `_QUANT_BLAS_GUARD_PREVIOUS`, `_QUANT_BLAS_GUARD_TARGET` |
| TF32 depth | none; per-entry save/restore (`utils/torch.py:1021` at `718256fd7`) | `_TF32_GUARD_LOCK`, `_TF32_GUARD_DEPTH`, `_TF32_GUARD_PREVIOUS` |
| BF16x3 depth | none; per-entry save/restore in two copies (`utils/cpu_features.py:88` and `nn_modules/exllamav3_torch.py:74` at `fe302047d`) | `_CPU_BF16X3_LOCK` (`RLock`), `_CPU_BF16X3_DEPTH`, `_CPU_BF16X3_RESTORE` in `cpu_features.py`; the EXL3 copy is deleted |
| Restore rule | every exit restores | only the exit that reaches depth 0 restores |

The BLAS guard also carries a *target*, because a quantization run may need
rocBLAS (it accumulates a Hessian) or hipBLASLt (it does not). Runs that need
the same target share the refcount. Runs that need different targets cannot
share one process-wide setting, so the later run waits for the active one to
finish rather than flipping the backend under it; without that, a concurrent
AWQ run quantized part of its layers on each backend and its checkpoint
depended on when the other thread exited (issue #4123). See
[ROCm default process environment](rocm-default-environment.md#quantization-blas-scope).

The BF16x3 guard used to exist twice. `nn_modules/exllamav3_torch.py` kept its
own `_cpu_bf16x3_matmul` copy of the save/restore pair next to the one in
`utils/cpu_features.py`, so the EXL3 forward and the GGUF/bitsandbytes forward
could restore each other's precision. There is now one implementation; the EXL3
module keeps only its environment-variable and shape gates.

The failure was the same shape as the BLAS one: caller A entered first and
selected BF16x3, caller B entered while A held it (B's copy saw BF16x3 and
skipped its own switch), then A exited and restored `none`, so B's remaining
matmuls ran in IEEE FP32. Measured on a 2048-wide `ExllamaV3TorchLinear` CPU
forward with the GGUF/bitsandbytes scope as A: chunk precisions
`bf16, bf16, none, none` and 5/5 forwards differing from the all-BF16x3
reference. After the change all four chunks stay `bf16` and 0/5 differ.

The guards stay scoped where they are entered. #4092 narrows
`quant_blas_backend` to the Hessian/linalg phase; the counting works for that
placement too, because it depends only on entry and exit, not on which phase
wraps the context.

## 2. Triton JIT compilation is single-flight and globally serialized

`TritonPatch.apply()` now installs `patch_triton_jit_compile()` in addition to
the autotuner patch. The wrapper around `JITFunction._do_compile` takes
`_TRITON_COMPILE_LOCK`, rechecks the per-device kernel cache for the requested
specialization key, and compiles only if the key is still missing.

```mermaid
flowchart TD
    A["JITFunction.run misses kernel cache"] --> B["patched_do_compile"]
    B --> C["acquire _TRITON_COMPILE_LOCK"]
    C --> D{"kernel cache has key"}
    D -- yes --> E["return cached kernel, no compile"]
    D -- no --> F["call libtriton compile"]
    F --> G["store kernel in cache"]
    G --> H["release lock"]
    E --> H
```

Two threads that miss the same specialization are collapsed into one compile.
Different specializations are serialized too, because the unsafe nanobind
state lives in libtriton as a whole, not in one `JITFunction`. The lock is only
taken on a cache miss; steady-state launches take the existing fast path and
never touch it.

Before the change, `utils/nogil_patcher.py` patched only the autotuner. With an
empty `TRITON_CACHE_DIR` and `PYTHON_GIL=0`, 24 threads launching
`awq_gemm_kernel` together aborted the process with
`Critical nanobind error: inst_dealloc ... attempted to delete an unknown
instance` and exit code 134. The same wrapper covers the MACaQ Triton kernels,
which launch through the same `JITFunction.run` path.

| Mechanism | Before | After |
| --- | --- | --- |
| Autotuner state | thread-local `nargs`/`best_config`, locked cache | unchanged |
| JIT compile | unsynchronized, one compile per missing thread | `_single_flight_triton_compile` under `_TRITON_COMPILE_LOCK` |
| Cold-start cost | concurrent compiles race | cold compiles run one at a time; the first caller pays the compile, the rest wait and reuse it |

## 3. Dynamo guard failures fall back to eager

`_is_compiler_failure` (`gptqmodel/utils/torch.py`) classifies an exception by
its innermost traceback frame. Any exception raised inside `torch/_dynamo`,
`torch/_inductor`, or `torch/_functorch` is a compiler failure and the eager
implementation serves the call. User code raising the same exception type has
its own frame at the origin, so it still propagates and is never retried.

```mermaid
flowchart TD
    A["compiled callable raises"] --> B["_is_compiler_failure"]
    B --> C{"control flow exception"}
    C -- yes --> D["propagate, never fall back"]
    C -- no --> E{"BackendCompilerFailed or TorchRuntimeError"}
    E -- yes --> F["fall back to eager"]
    E -- no --> G{"fullgraph"}
    G -- yes --> H["propagate, fullgraph is a contract"]
    G -- no --> I{"cause chain has TorchDynamoException"}
    I -- yes --> F
    I -- no --> J{"innermost frame in torch compiler package"}
    J -- yes --> F
    J -- no --> K["propagate user error"]
```

The dynamo guard `AssertionError` (`Guard failed on the same frame it was
created ... GLOBAL_STATE changed: default_dtype`) is raised from
`torch/_dynamo/guards.py`, so the last decision sends it to the eager path.
Before the change the frame check ran only for `RuntimeError`; the guard
assertion escaped and aborted a forward that raced another thread's
`torch.set_default_dtype`.

| Mechanism | Before | After |
| --- | --- | --- |
| Frame-origin check | only `RuntimeError` | every non-control-flow exception |
| Guard `AssertionError` | propagated, forward failed | eager fallback, one warning |
| User `AssertionError` | propagated | propagated, no retry |

## 4. Model loads are serialized

`ModelLoader.from_pretrained` and `ModelLoader.from_quantized` are wrapped with
`_serialized_model_load` (`gptqmodel/models/loader.py`), which holds
`_MODEL_LOAD_LOCK` (an `RLock`) for the whole construction. transformers'
`from_pretrained` saves, sets and restores `torch.set_default_dtype`
(`transformers/modeling_utils.py`) and builds parameters through meta-device and
no-init contexts; `from_config` does the same for the quantized shell. Those are
process-global, so two overlapping loads interleave them and hand back models
with mixed dtypes, a meta tied weight, or untied embeddings.

```mermaid
flowchart TD
    A["thread 1: GPTQModel.load"] --> B["acquire _MODEL_LOAD_LOCK"]
    C["thread 2: GPTQModel.load"] --> D{"lock free"}
    D -- no --> E["wait for the other load"]
    B --> F["transformers from_pretrained or from_config"]
    F --> G["assert_loaded_model_state"]
    G -- "invariant violated" --> H["RuntimeError: load fails loudly"]
    G -- "consistent" --> I["release lock, return model"]
    E --> B
    I --> D
```

`quantize()` itself is not serialized: once a model is loaded, overlapping
quantization stays supported (section 1).

### Safety net

Every build ends with `assert_loaded_model_state` (`gptqmodel/utils/hf.py`),
which raises `RuntimeError` when

- `torch.get_default_dtype()` differs from the value captured before the load;
- a floating parameter is still on the meta device, which is how
  `lm_head.weight` stayed meta and the embeddings untied;
- the requested dtype is explicit and a floating parameter has another dtype;
- or, when no dtype was requested, the floating parameters disagree with each
  other.

The `offload_to_disk` meta shell is exempt from the dtype and meta checks
because it is intentionally meta until `LazyTurtle` fills each layer.

### Init suspension is reference-counted

`suspend_hf_weight_init` (`gptqmodel/utils/hf.py`) patches process-wide torch
initializers and `PreTrainedModel.init_weights`. It keeps
`_HF_INIT_LOCK`/`_HF_INIT_DEPTH`/`_HF_INIT_SUSPENSION`: the first entry installs
the patches, overlapping entries only increment the depth, and the last exit
restores the originals. The old per-entry restore let the first caller to leave
re-enable init for a thread still building a model, and the second exit then
restored the no-op stubs for the rest of the process.

### No cross-run AWQ scratch size

`quantize()` used to write `AWQ_BATCH_SIZE` into `os.environ`, and
`gptqmodel_post_init` read it to size the AWQ ExLlamaV2 scratch on a later load,
so one run could size another run's scratch. The variable is gone from both
sides: `gptqmodel_post_init(..., awq_batch_size=...)` takes the value explicitly
and defaults to 1.

## Failure and ambiguity contract

| Situation | Result |
| --- | --- |
| Platform refuses the BLAS switch | guard yields without counting; caller keeps its backend |
| User already selected a non-hipBLASLt backend | guard yields without counting |
| Caller already selected BF16x3, or oneDNN refuses the switch | `cpu_bf16x3_scope` yields without counting; nothing to restore |
| Restore raises | swallowed; the quantization result is already computed |
| Triton compile raises | lock is released; the next miss retries the compile |
| Triton cache already populated | no lock, no compile |
| Dynamo guard fails on a user frame | exception propagates; no double execution |
| `fullgraph=True` capture failure | exception propagates; no silent fallback |
| Two `GPTQModel.load()` calls overlap | the second waits, then builds; both models are consistent |
| A load leaves a mixed dtype, a meta tied weight, or a changed default dtype | `RuntimeError`; the corrupted model is never returned |
| Overlapping `suspend_hf_weight_init` scopes | patches stay installed until the last exit |

## Validation

The regressions are covered by `tests/test_concurrency_guards.py`,
`tests/test_triton_patch_api.py`, `tests/test_awq_triton_cold_jit.py`, and
`tests/test_torch_compile_fallback.py`. The cold-JIT test runs a child process
with `PYTHON_GIL=0`, an empty `TRITON_CACHE_DIR`, and 24 threads, and asserts a
clean exit; it is skipped on GIL builds and when the AWQ checkpoint is absent.

Measured on 2026-11-15 with the ROCm 10.1 R9700 (`gfx1201`) venv and the CUDA
2.16.0.dev+cu130 venv:

| Check | Before | After |
| --- | --- | --- |
| Overlapping quantize, tensors differing from serial | 268 of 504, 2/2 runs | 0 of 504, 3/3 runs |
| BLAS seen by the second quantize after the first guard exits | 76 hessian calls on Cublaslt | 76 hessian calls on Cublas |
| TF32 setting after two concurrent guards | first exit restores while the second is inside | restored only by the last exit |
| Cold concurrent AWQ Triton JIT, 24 threads | exit 134, 4/4 runs at 24 threads (8/8 at 24-32) | exit 0, 4/4 runs |
| Dynamo dtype race, `PYTHON_GIL=0`, 40 iterations | 18 ok / 22 failed | 40 ok / 0 failed |
| Dynamo dtype race, `PYTHON_GIL=1`, 40 iterations | 2 failed in the initial 40-run probe | 40 ok / 0 failed |
| Natural 3-thread load+forward, TorchLinear | intermittent; deterministic race used for the regression | 3 ok / 0 failed |
| BF16x3 guard order A-in/B-in/A-out, both former call sites | B sees `none` after A exits | B sees `bf16` after A exits |
| BF16x3 chunks in a 2048-wide EXL3 CPU forward while the other call site exits | `bf16, bf16, none, none`; 5/5 forwards differ | `bf16, bf16, bf16, bf16`; 0/5 differ |
| BF16x3 race, 30 trials, `PYTHON_GIL=0` | 11/30 forwards differ | 0/30 forwards differ |
| BF16x3 race, 30 trials, `PYTHON_GIL=1` | 2/30 forwards differ | 0/30 forwards differ |
| Concurrent dense loads of Qwen2.5-0.5B and Qwen3-0.6B at fp16, 6 rounds, `PYTHON_GIL=0` | 6/6 rounds corrupted (an earlier run: 5/6) | 0/6 rounds corrupted |
| Same loads, 6 rounds, `PYTHON_GIL=1` | 5/6 rounds corrupted (an earlier run: 6/6) | 0/6 rounds corrupted |

The 268/504 count is this host's reproduction; the issue measured 177-210 of
794 on its checkpoint layout. Both counts are the same defect: the second
quantize's later layers ran on the backend selected by the first quantize's
restore.

The BF16x3 numbers use a CPU host with AVX512-BF16 and the
`Llama-3.2-1B-Instruct-EXL3-W4-ALL` checkpoint: a 2048-wide
`ExllamaV3TorchLinear` forward (`q_proj`) against a shorter EXL3 forward on
another thread, 16-row activations, reference computed inside the scope. The
issue measured 5/30 (`PYTHON_GIL=0`) and 1/30 (GIL on) on its own host; the hit
rate depends on how much of the guarded window is matmul rather than weight
decode, so the ordering probe above is the stable regression.
