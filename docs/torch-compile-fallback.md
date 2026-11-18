<!-- SPDX-FileCopyrightText: 2026 ModelCloud.ai -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Torch compile fallback

Kernel-level compile and graph-capture capability is declared per kernel and
tabulated in [the kernel compile matrix](kernel-compile-matrix.md); this page
covers what happens when a compile fails at runtime.

`TorchLinear.optimize()` compiles its dequantize path by default, and
`BaseQModel.optimize()` compiles the whole model wrapper. Both go through
`gptqmodel.utils.torch.torch_compile`. That helper used to guard only the
`torch.compile(...)` constructor. Inductor does not compile there: it compiles on
the first invocation, after `post_init()` has already returned. A failing
compile worker therefore escaped from a normal forward and aborted the inference
request.

## First-use compile shares the model-load lock (#5127)

Model construction serializes the process-global `torch.set_default_dtype`
through `_MODEL_LOAD_LOCK` (`gptqmodel/models/loader.py:1449`): transformers'
`from_pretrained`/`from_config` saves, sets and restores it. Dynamo's
`preserve_global_state` does the same save/restore around every trace
(`torch/_dynamo/convert_frame.py`), including the lazy first trace of
`TorchLinear.dequantize_weight`. Before this change that trace ran outside the
lock, so a compile could capture a load's bf16 window and restore bf16 after the
load restored fp32, leaving the process default dtype wrong for the rest of the
process; or it could restore bf16 while the load was still asserting its
invariant, failing a valid load.

`model_global_state_lock()` (`gptqmodel/utils/torch.py:432`) resolves the same
re-entrant loader lock. The callable and module compile proxies hold it only
for the first (compiling) invocation; `TorchLinear.optimize()`
(`gptqmodel/nn_modules/qlinear/torch.py:467`) holds it while it constructs the
dynamo wrapper. Once the first call succeeds, `state["compiled"]` is set and
steady-state forwards never take the lock.

```mermaid
flowchart TD
    A["thread A: GPTQModel.load"] --> B["acquire _MODEL_LOAD_LOCK"]
    B --> C["transformers sets default dtype bf16"]
    C --> D["construct model and TorchLinear kernels"]
    D --> E["restore fp32; assert_loaded_model_state"]
    E --> F["release lock; load returns"]

    G["thread B: first TorchLinear forward"] --> H["_install_callable_compile_fallback.call"]
    H --> I{"state.compiled?"}
    I -- yes --> J["compiled call, lock-free steady state"]
    I -- no --> K["acquire _MODEL_LOAD_LOCK"]
    K --> L{"state.compiled? (re-check)"}
    L -- yes --> J
    L -- no --> M["compiled call: dynamo preserve_global_state saves/restores dtype"]
    M --> N{"first call succeeded?"}
    N -- yes --> O["state.compiled = true; return result"]
    N -- compiler failure --> P["state.fallback = true; warn once; eager result"]
    N -- user error --> Q["re-raise; no retry"]

    B -. "thread B waits here" .-> K
    style Q fill:#f8d7da,stroke:#b02a37
    style P fill:#fff3cd,stroke:#997404
```

The lock is an `RLock` and the loader already holds it during
`TorchLinear.post_init()`, so the nested `optimize()` entry is re-entrant and
cannot deadlock. A compile that is reached from an ordinary forward waits for
the in-flight load instead of mutating the dtype under it.

### Failure path without the lock

```mermaid
flowchart TD
    A["thread A: load sets default dtype bf16"] --> B["thread B: compile starts"]
    B --> C["preserve_global_state captures bf16"]
    A --> D["load restores fp32"]
    C --> E["compile restores bf16"]
    D --> F{"load asserts before or after the restore?"}
    F -- before --> G["RuntimeError: model load left a corrupted state"]
    F -- after --> H["process default dtype stays bf16"]
    H --> I["later torch.ones / nn.Linear silently use bf16"]
    G --> J["valid load fails"]
    style G fill:#f8d7da,stroke:#b02a37
    style H fill:#f8d7da,stroke:#b02a37
    style J fill:#f8d7da,stroke:#b02a37
```

| Mechanism | Before | After |
| --- | --- | --- |
| First `TorchLinear` trace vs. load | no shared lock; compile ran from the forward thread while `_MODEL_LOAD_LOCK` was held by a load | `model_global_state_lock()` in `_install_callable_compile_fallback.call` (`utils/torch.py:448`) and `_install_module_compile_fallback.forward` (`utils/torch.py:495`) |
| `optimize()` wrapper construction | unlocked (`nn_modules/qlinear/torch.py:435` at `e2f9676c6`) | lock held around `torch_compile` and the adapter (`nn_modules/qlinear/torch.py:467`) |
| Dynamo-internal lookup failure | `KeyError` from `get_guard_fail_reason` escaped a plain forward | `LookupError` whose innermost frame is under `torch/_dynamo/` is a compiler failure (`utils/torch.py:338`) |
| Steady-state forwards | lock-free | still lock-free: `state["compiled"]` short-circuits the lock |

## Old flow

```mermaid
flowchart TD
    A["TorchLinear.post_init"] --> B["optimize()"]
    B --> C["torch_compile(dequantize_weight)"]
    C --> D{"torch.compile() constructor raises?"}
    D -->|"yes"| E["log.warn.once; return eager"]
    D -->|"no"| F["return torch.compile wrapper"]
    F --> G["first forward: dequantize_weight()"]
    G --> H{"inductor compiles successfully?"}
    H -->|"yes"| I["compiled kernel result"]
    H -->|"no"| J["BackendCompilerFailed escapes the forward"]
    J --> K["inference request aborted"]
    style J fill:#f8d7da,stroke:#b02a37
    style K fill:#f8d7da,stroke:#b02a37
```

| Removed mechanism | Location |
| --- | --- |
| Construction-only `try/except` around `torch.compile` | `gptqmodel/utils/torch.py:249-253` (before this change) |
| Error classification limited to "did `torch.compile` raise?" | `gptqmodel/utils/torch.py:249-253` |
| No caller-facing way to disable compilation | `gptqmodel/nn_modules/qlinear/torch.py:327-344` |

## New flow

```mermaid
flowchart TD
    A["TorchLinear.__init__(compile=bool)"] --> B["optimize()"]
    B --> C{"_compile_enabled?"}
    C -->|"False"| D["optimized=True; eager dequantize_weight kept"]
    C -->|"True"| E["torch_compile(eager)"]
    E --> F{"TORCHDYNAMO_DISABLE=1?"}
    F -->|"yes"| D
    F -->|"no"| G["torch.compile(eager)"]
    G --> H{"constructor raises?"}
    H -->|"yes"| I["log.warn.once; eager kept"]
    H -->|"no"| J["_install_callable_compile_fallback(eager, compiled)"]
    J --> K["first forward"]
    K --> L{"compiled call raises?"}
    L -->|"no"| M["compiled result"]
    L -->|"_is_compiler_failure()"| N["log.warn.once once; lazy fallback=True"]
    N --> O["eager result for this and every later call"]
    L -->|"user code or control flow"| P["re-raise unchanged"]
    style N fill:#fff3cd,stroke:#997404
    style O fill:#d1e7dd,stroke:#146c43
    style P fill:#f8d7da,stroke:#b02a37
```

`_is_compiler_failure()` decides which failures may degrade to eager:

| Exception | Falls back to eager | Reason |
| --- | --- | --- |
| `torch._dynamo.exc.BackendCompilerFailed` | yes | Inductor/backend failed; includes `RuntimeError: Attempting to use a closed pool` |
| `torch._dynamo.exc.TorchRuntimeError` | yes | A backend runtime failure wrapped by dynamo |
| other `TorchDynamoException` (e.g. `Unsupported`) | only when `fullgraph=False` | A capture failure is soft unless the caller demanded one graph |
| any exception under `fullgraph=True` | no | The single-graph contract must fail closed |
| `RuntimeError` whose innermost frame is inside `torch/_dynamo/`, `torch/_inductor/`, or `torch/_functorch/` | yes | Compile workers raise outside the traced frame |
| `LookupError` (`KeyError`/`IndexError`) whose innermost frame is inside those packages | yes | Dynamo's guard bookkeeping raises `KeyError` from `get_guard_fail_reason` when a concurrent compile changes global state (#5127) |
| `RuntimeError` from traced user code | no | Its innermost frame is the user body; retrying would run it twice and hide the real error |
| `ValueError` / `TypeError` from traced user code | no | Not a compiler failure |
| `KeyboardInterrupt` / `SystemExit` | no | Control flow must always propagate |

Dynamo's `eval_frame.compile_wrapper` frame sits between the caller and the
traced body, so "the traceback contains a compiler frame" would also match user
code raising `RuntimeError`. Only the innermost frame identifies the origin.

The fallback is armed lazily and permanently: the first compiler failure drops
the compiled callable, logs once, and runs `dequantize_weight` / the module
forward eagerly. Later calls never re-enter the broken compiler.

`BaseQModel.optimize()` compiles the model's `forward` in place rather than
replacing `self.model`, because `generate()` calls the original module's
`generate` and would otherwise never enter the compiled wrapper (#4669). The
compiled proxy keeps `self.model` an `nn.Module`; when `torch.compile` returns
an unusable object the eager forward is kept and the replacement is reported
once instead of being stored where a callable is required.

## Opting out

| Control | Scope | Notes |
| --- | --- | --- |
| `TorchLinear(..., compile=False)` | one kernel | Skips `torch.compile`; the module is still marked `optimized` so no later `optimize()` retries it |
| `QuantizeConfig(compile=False)` | every kernel built from that config | Propagated through `quant_linear_init_kwargs()`; runtime-only and never serialized into a checkpoint |
| `TORCHDYNAMO_DISABLE=1` | the whole process | PyTorch's own kill switch; `torch_compile` returns the eager callable without importing dynamo |

`compile` is deliberately not persisted. A checkpoint compiled on one host can
be unusable on another (a dead inductor worker pool, a different Triton
toolchain), so the loading host decides.

## Tests

`tests/test_torch_compile_fallback.py` pins the contract: a first-call
`BackendCompilerFailed` yields the eager result and is not retried, a
dynamo-internal `KeyError` falls back to eager, user-code and control-flow
exceptions still propagate, `fullgraph=True` still fails closed, a compiled
module falls back to its own `forward` and stays an `nn.Module`, `compile=False`
skips compilation, and the config field stays runtime-only.
`tests/test_model_load_lock_4204.py` pins the cross-thread invariant: a load is
held in its bf16 window while another thread performs the first `TorchLinear`
forward, and the process default dtype must still be float32 afterwards.
