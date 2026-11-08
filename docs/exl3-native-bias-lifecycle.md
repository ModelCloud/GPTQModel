# EXL3 native bias lifecycle

The native EXL3 module holds quantized trellis storage and a logical output
bias. When the stored output width is padded, `LinearEXL3` receives a widened
bias copy. `ExllamaV3Linear._current_signature()` uses the logical bias version
to rebuild that copy after an in-place edit.

A bias created while `torch.inference_mode()` is active has no version counter.
`_bias_version()` clones it once with inference mode disabled and rebinds the
logical buffer before `_current_signature()` records its pointer and version.
Later forwards reuse the native inner module until the bias changes. This
normalization keeps the logical bias shape and the packed trellis unchanged.

```mermaid
flowchart TD
    A["ExllamaV3Linear._ensure_inner"] --> B{"Trellis present and materialized?"}
    B -- No --> X["RuntimeError: missing or meta trellis"]
    B -- Yes --> C{"validate_device accepts NVIDIA CUDA?"}
    C -- No --> Y["NotImplementedError: unsupported native device"]
    C -- Yes --> D["_coerce_buffer_dtypes then _current_signature"]
    D --> E["_bias_version checks logical bias"]
    E --> F{"Bias is an inference tensor?"}
    F -- Yes --> G["Clone once under torch.inference_mode(False) and rebind self.bias"]
    F -- No --> H["Read bias version or None"]
    G --> H
    H --> I["Record current bias pointer and version in signature"]
    I --> J{"Signature matches _inner_signature?"}
    J -- Yes --> K["Reuse LinearEXL3"]
    J -- No --> L["_drop_inner then _stored_features"]
    L --> M["Pad logical bias if stored width is wider"]
    M --> N["Build LinearEXL3 and save _inner_signature"]
    K --> O["forward crops to logical out_features"]
    N --> O
    classDef failure fill:#ffe8e8,stroke:#b30000,color:#111;
    class X,Y failure;
```

| Condition | Before | Current behavior |
| --- | --- | --- |
| Inference-created bias | `_bias_version()` read `bias._version` directly and raised `RuntimeError` (`gptqmodel/nn_modules/exllamav3.py:115` before this change). | The bias is cloned once outside inference mode and rebound (`gptqmodel/nn_modules/exllamav3.py:114-121`). |
| Signature after normalization | A direct `_version` read needed no buffer rebind. | `_current_signature()` normalizes before collecting buffer pointers, so its first signature matches the inner module it builds (`gptqmodel/nn_modules/exllamav3.py:128-137`). |
| In-place bias edit | A normal bias already triggered a native rebuild; an inference-created bias could not reach its first forward. | The versioned logical bias triggers a rebuild after mutation, including for padded outputs. Repeated unchanged forwards reuse the inner module. |

The native inner keeps the packed trellis and scale vectors. Its padded bias is
a small output vector, not a dense reconstructed weight matrix. The original
inference-created bias is no longer retained by the module after rebinding.

Introduced in [PR #3625](https://github.com/ModelCloud/GPTQModel/pull/3625).
