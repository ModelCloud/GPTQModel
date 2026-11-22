<!-- SPDX-FileCopyrightText: 2026 ModelCloud.ai
SPDX-License-Identifier: Apache-2.0 -->

# Paro affine fitting and optimization controls

GPT-QModel fits the transformed-domain affine grid once for quantizer
initialization and uses the same parameters for pseudo quantization and export.
Including zero prevents constant and one-sided groups from collapsing when the
runtime integer zero point is clamped. Fitting in at least FP32 prevents FP16
range overflow and FP16/BF16 zero-point rounding errors.

Introduced in [#4160](https://github.com/ModelCloud/GPTQModel/pull/4160).

## Affine grid

```mermaid
flowchart TD
    A["GroupLinearQuantizer.__init__ or pseudo_quantize_dequant"] --> B["_calc_affine_qparams: amin / amax per group"]
    B --> C["Promote FP16/BF16 extrema to FP32"]
    C --> D["low = min(min_val, 0); high = max(max_val, 0)"]
    D --> E["span = clamp(high - low, min=1e-5)"]
    E --> F["zero_point = clamp(round(-low / span * qmax), 0, qmax)"]
    F --> G["Refit each side with a nonzero code budget"]
    G --> H["scale = clamp(max(negative_scale, positive_scale), 1e-5, 1e5)"]
    H --> I["pseudo_quantize_dequant: round, clamp codes, dequantize"]
    H --> J["GroupLinearQuantizer.pack_params: clamp learned scales to the same bounds; export integer zeros"]
    I --> K["Restore weight dtype at output"]
    F -.-> L["Boundary: zero_point 0 or qmax can discard a small opposite-sign endpoint"]
    H -.-> M["Boundary: scale limits permit saturation"]
    style L fill:#ffe4e6,stroke:#be123c
    style M fill:#ffe4e6,stroke:#be123c
```

The grid represents `(code - zero_point) * scale`, with `code` in
`[0, 2**bits - 1]`. Zero points use ties-to-even rounding. Normalizing by the
span before multiplying by `qmax` avoids a rounded intermediate scale moving a
balanced range off its zero-point tie. FP16/BF16 inputs receive FP32 scale and
zero-point parameters; FP32 and FP64 retain their parameter dtype.

The scale refit also changes mixed-sign groups. This is a quantization fit,
not a lossless transform or an unconditional endpoint-containment guarantee.
The existing scale floor and cap remain `1e-5` and `1e5`. When the integer zero
point lands at an extreme, the opposite side has no available codes. Export
clamps learned scales to the same bounds as pseudo quantization, so optimizer
updates outside those bounds cannot change the reconstructed packing grid.

| Before | After | Source evidence |
| --- | --- | --- |
| Fit raw min/max and later clamp an unrepresentable zero point | Include zero before fitting; refit around the integer zero point | Base `ad21f0db:optimization.py:977-982`; `_calc_affine_qparams` |
| Subtract FP16 extrema and divide BF16 values in the model dtype | Promote only reduced extrema to FP32 before arithmetic | Initial PR `801653830:optimization.py:997-1009`; `_calc_affine_qparams` |
| Divide by an already-rounded base scale to select the zero point | Normalize the range first, then multiply by `qmax` | Initial PR `801653830:optimization.py:1003-1004`; `_calc_affine_qparams` |
| Export raw learned scales while pseudo quantization clamps them | Export the same effective bounded scale | Initial PR `801653830:optimization.py:1074`; `GroupLinearQuantizer.pack_params` |

All `optimization.py` references above refer to
`gptqmodel/quantization/paro/optimization.py` at the named commit.

## Explicit optimization controls

Runtime choices come from configuration or an explicit call argument. An
omitted `fused_rotation` or `stage_cudagraph` resolves to `True`. Compilation is
controlled by `_maybe_compile_large_train_quant(..., compile_quant=...)` and
keeps its existing size, training, runtime and failure gates.

```mermaid
flowchart TD
    A["ParoConfig.opt_fused_rotation / explicit fused_rotation"] --> B["_apply_rotation"]
    B --> C{"Fused enabled?"}
    C -->|yes| D["apply_paro_rotation or apply_paro_rotation_autograd"]
    C -->|no| E["Torch reference rotation"]
    F["ParoConfig.opt_stage_cudagraph / explicit stage_cudagraph"] --> G["_should_use_paro_stage_cudagraph"]
    G --> H{"Enabled and existing runtime / shape gates pass?"}
    H -->|yes| I["_run_stage_gptqmodel_cudagraph"]
    H -->|no| J["_run_stage_gptqmodel_impl"]
    K["_maybe_compile_large_train_quant: compile_quant"] --> L{"Enabled and existing training / size gates pass?"}
    L -->|no| M["pseudo_quantize_dequant"]
    L -->|yes| N["_get_large_train_quant_compile"]
    N --> O{"Compiled callable available?"}
    O -->|no| M
    O -->|yes| P["Call compiled pseudo quantizer"]
    P --> Q{"Call succeeded?"}
    Q -->|yes| R["Return quantized tensor"]
    Q -->|no| S["Warn; set _LARGE_TRAIN_QUANT_COMPILE_DISABLED"]
    S --> M
    I -.-> T["Capture failure: restore model state and use eager stage"]
    T --> J
    style S fill:#ffe4e6,stroke:#be123c
    style T fill:#ffe4e6,stroke:#be123c
```

| Initial PR mechanism | Replacement | Source at `801653830` |
| --- | --- | --- |
| Environment fallback for fused rotation | `fused_rotation`, supplied by `ParoConfig.opt_fused_rotation` | `optimization.py:883-886` |
| Environment gate for large training quantizer compilation | Explicit `compile_quant` argument on the owning helper | `optimization.py:1185` |
| Environment fallback for graph capture | `stage_cudagraph`, supplied by `ParoConfig.opt_stage_cudagraph` | `optimization.py:1869-1870` |

The ROCm capture exclusion and existing compiled-call fallback are preserved.
Removing environment decisions does not add a kernel or change kernel selection.

## Validation contracts

`tests/test_paro.py` compares independent FP64 scales, integer zero points,
and every quantization code. It covers constant and one-sided groups, mixed
signs, FP16 range overflow, small-magnitude groups at the scale floor, exact rounding ties
and their adjacent representable values. Arithmetic error is checked separately
from parameter-storage and visible-output rounding. The fallback export and
explicit optimization arguments have direct coverage. Export is also checked after
learned scales cross the floor or cap.

`scripts/gil0_api_probe.py --only 'Paro affine fit'` exercises the composed
fit and pseudo quantizer concurrently; its measured result is recorded in
[`pytorch_gil_0_compat.md`](../pytorch_gil_0_compat.md).
