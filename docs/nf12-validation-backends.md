# NF12 validation backend selection

`scripts/validate_nf12_ad.py` compares native NF12 encode, decode, and linear
results against independent Torch references. An explicit `--backend cuda` or
`--backend hip` must load that native extension. `--backend auto` may use the
Torch codec when neither extension is available; the report labels it `torch`.
Performance measurements require a native backend and are skipped in that case.

```mermaid
flowchart TD
    A["main parses --backend"] --> B["run_accuracy or run_perf calls backend_module"]
    B --> C{"backend is hip?"}
    C -- Yes --> D{"HIP prewarm succeeds?"}
    D -- Yes --> H["Use HIP native extension"]
    D -- No --> X["Raise HIP extension error"]
    C -- No --> E{"CUDA load succeeds?"}
    E -- Yes --> I["Use CUDA native extension"]
    E -- No --> F{"backend is cuda?"}
    F -- Yes --> Y["Raise CUDA extension error"]
    F -- No --> G{"HIP prewarm succeeds on ROCm?"}
    G -- Yes --> H
    G -- No --> J["Use Torch codec; report backend torch"]
    X --> K["main records failure; exits 1"]
    Y --> K
    J --> L{"Performance requested?"}
    L -- Yes --> M["run_perf skips native benchmarks"]
    L -- No --> N["run_accuracy checks Torch codec"]
    classDef failure fill:#ffe8e8,stroke:#b30000,color:#111;
    class X,Y,K failure;
```

`run_perf` resolves the backend before querying GPU properties or measuring
bandwidth. A performance exception sets `ok=False` in `main`, including with
`--perf-only`, so the command exits with status 1. A successful performance-only
report records the actual selected backend.

| Mechanism | Before | Current |
| --- | --- | --- |
| Explicit CUDA request when extension load failed | Fell through to the `torch` result (`b98ae70:scripts/validate_nf12_ad.py:167-174`) | `backend_module` raises with `nf12_ad.last_error()` (`scripts/validate_nf12_ad.py:167-171`) |
| Performance-only error | `main` recorded the exception but left `ok=True` (`b98ae70:scripts/validate_nf12_ad.py:593-606`) | `main` sets `ok=False` and exits 1 (`scripts/validate_nf12_ad.py:597-608`) |
| Performance probe with no native backend | Queried GPU properties and measured bandwidth before discovering the missing extension (`b98ae70:scripts/validate_nf12_ad.py:382-406`) | `run_perf` resolves the backend first and skips measurements when `auto` selects Torch (`scripts/validate_nf12_ad.py:385-391`) |

`tests/test_validate_nf12_ad.py` covers the explicit CUDA error and the
performance-only exit status without requiring a CUDA compiler or HIP device.
