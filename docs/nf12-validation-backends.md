# NF12 validation backend and performance flow

`scripts/validate_nf12_ad.py` compares native NF12 encode, decode, and linear
results against independent Torch references. The requested `--device` governs
which backend may be selected and which GPU the performance timers wait for.
Explicit CUDA/HIP requests require a compatible GPU and a loadable native
extension. Automatic mode can use the Torch codec when no native backend is
available; the report labels that path `torch` and skips native benchmarks.

This flow was introduced in #3636 and refined in #3639, #3640, and #3642.

```mermaid
flowchart TD
    A["main parses --device and --backend"] --> B["run_accuracy or run_perf calls backend_module"]
    B --> C{"torch.device(device).type is cuda?"}
    C -- No --> D{"backend is auto?"}
    D -- Yes --> T["Use Torch codec; label backend torch"]
    D -- No --> X["Raise: GPU backend requires CUDA/HIP device"]
    C -- Yes --> E{"backend is hip?"}
    E -- Yes --> F{"HIP prewarm succeeds?"}
    F -- Yes --> H["Use HIP native runtime"]
    F -- No --> Y["Raise HIP extension error"]
    E -- No --> G{"CUDA load succeeds?"}
    G -- Yes --> I["Use CUDA native runtime"]
    G -- No --> J{"backend is cuda?"}
    J -- Yes --> Z["Raise CUDA extension error"]
    J -- No --> K{"ROCm HIP prewarm succeeds?"}
    K -- Yes --> H
    K -- No --> T
    H --> U["run_accuracy compares native output with Torch oracle"]
    I --> U
    H --> N["run_perf packs weights with rt.encode on selected GPU"]
    I --> N
    N --> R{"rt.encode succeeds?"}
    R -- Yes --> O["rt.decode and rt.linear consume native packed tensors"]
    R -- No --> W["main records performance failure and exits 1"]
    O --> P["timed synchronizes selected device"]
    T --> V["run_accuracy checks Torch codec; run_perf skips native benchmarks"]
    X --> Q["main records failure and exits 1"]
    Y --> Q
    Z --> Q
    classDef failure fill:#ffe8e8,stroke:#b30000,color:#111;
    classDef fallback fill:#fff3cf,stroke:#ad7700,color:#111;
    class X,Y,Z,Q,W failure;
    class T,V fallback;
```

`run_perf` resolves the backend before querying GPU properties or measuring
bandwidth. With a native runtime, its decode and linear inputs come from that
runtime's encoder. The encode throughput row measures the same encoder. Its
timing, bandwidth, and cache helpers synchronize the selected `--device`, even
when another GPU is the process's current device. A native encode failure
propagates to `main`, which records the performance failure and exits 1.

| Mechanism | Before | Current |
| --- | --- | --- |
| Explicit CUDA request when extension load failed | Fell through to the Torch result (`b98ae70:scripts/validate_nf12_ad.py:167`) | `backend_module` raises with `nf12_ad.last_error()` (`scripts/validate_nf12_ad.py:176`) |
| CPU target with automatic selection | Probed native CUDA first (`2f22621:scripts/validate_nf12_ad.py:167`) | Returns the Torch codec before probing an extension (`scripts/validate_nf12_ad.py:160`) |
| CPU target with explicit GPU backend | Tried the requested extension (`2f22621:scripts/validate_nf12_ad.py:160`, `2f22621:scripts/validate_nf12_ad.py:167`) | Raises a device/backend error (`scripts/validate_nf12_ad.py:161`) |
| Performance-only exception | Recorded the exception but left `ok=True` (`b98ae70:scripts/validate_nf12_ad.py:598`) | Sets `ok=False` and exits 1 (`scripts/validate_nf12_ad.py:608`) |
| No native backend during performance validation | Queried GPU properties before checking availability (`b98ae70:scripts/validate_nf12_ad.py:383`) | Resolves the backend first and skips native benchmarks (`scripts/validate_nf12_ad.py:393`) |
| Timing on a non-current GPU | `torch.cuda.synchronize()` waited on the current GPU (`61b2ae5:scripts/validate_nf12_ad.py:321`) | All helpers synchronize `device` (`scripts/validate_nf12_ad.py:321`) |
| HIP performance encoding | Moved the weight to CPU, encoded with `nf12_codec`, and copied bytes back (`2b468d4:scripts/validate_nf12_ad.py:417`) | `run_perf` uses `rt.encode(w)` for CUDA and HIP (`scripts/validate_nf12_ad.py:420`) |

`tests/test_validate_nf12_ad.py` checks backend errors, CPU fallback, selected
GPU synchronization, and native HIP performance routing without requiring an
AMD GPU. Kernel throughput and numerical validation still require the target
hardware.
