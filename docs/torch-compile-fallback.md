<!-- SPDX-FileCopyrightText: 2026 ModelCloud.ai -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Torch compile fallback

`TorchLinear.optimize()` compiles its dequantize path by default, and
`BaseQModel.optimize()` compiles the whole model wrapper. Both go through
`gptqmodel.utils.torch.torch_compile`. That helper used to guard only the
`torch.compile(...)` constructor. Inductor does not compile there: it compiles on
the first invocation, after `post_init()` has already returned. A failing
compile worker therefore escaped from a normal forward and aborted the inference
request.

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
| `RuntimeError` from traced user code | no | Its innermost frame is the user body; retrying would run it twice and hide the real error |
| `ValueError` / `TypeError` from traced user code | no | Not a compiler failure |
| `KeyboardInterrupt` / `SystemExit` | no | Control flow must always propagate |

Dynamo's `eval_frame.compile_wrapper` frame sits between the caller and the
traced body, so "the traceback contains a compiler frame" would also match user
code raising `RuntimeError`. Only the innermost frame identifies the origin.

The fallback is armed lazily and permanently: the first compiler failure drops
the compiled callable, logs once, and runs `dequantize_weight` / the module
forward eagerly. Later calls never re-enter the broken compiler.

`BaseQModel.optimize()` assigns the result to `self.model`, so a module input
must stay an `nn.Module`. When `torch.compile` returns anything else, the eager
module is kept and the unusable replacement is reported once instead of being
stored where a child module is required.

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
`BackendCompilerFailed` yields the eager result and is not retried, user-code and
control-flow exceptions still propagate, `fullgraph=True` still fails closed, a
compiled module falls back to its own `forward` and stays an `nn.Module`,
`compile=False` skips compilation, and the config field stays runtime-only.
