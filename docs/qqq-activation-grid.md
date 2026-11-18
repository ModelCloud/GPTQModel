# QQQ activation INT8 grid contract

QQQ is a W4A8 method: packed INT4 weights multiply an INT8 activation and a
per-row FP32 scale. That activation grid is defined once, in
`gptqmodel/quantization/qqq_activation.py::qqq_activation_quantize`, and both
the portable inference kernel and the GSQ calibration statistics call it. This
page records the arithmetic contract, why the calibration used to disagree with
it, and how the two consumers stay pinned together.

## The deployed grid

For one activation row `x` with `a = max_j |x_j|`:

```text
scale = a / 127            (FP32)
code_j = clamp(round(x_j / scale), -128, 127)    (INT8)
dequantized_j = code_j * scale                   (FP32)
```

The maximum and the division happen in FP32, and the scale is FP32. A row whose
maximum is exactly zero receives `scale = 1`, so it decodes to exactly zero.
`127`, not `128`, is the divisor; `-128` is reachable only through rounding.

`QQQLinear.dynamic_quant` uses the frozen Torch path when the Triton quantizer is
unavailable, and the Triton (`qqq_triton.py`) and Metal (`mlx_qqq.py`) kernels
evaluate the same expression with FP32 operands. The Torch path is the
reference those kernels are compared against.

## Before and after

The GSQ calibration built its own grid instead of reusing the kernel's:

| Step | Before (`gsq_qqq.py:85-95`) | After |
| --- | --- | --- |
| Operand | `inputs.half()` | `inputs.float()` |
| Row maximum | FP16 `amax` | FP32 `amax` |
| Scale | `(maximum / 127).float()` — division in FP16 | `maximum / 127` in FP32 |
| Zero row | `scale > 0` mask, codes forced through a unit divisor | helper's unit scale, codes exactly zero |
| Tiny row | raised `runtime token scale underflows` when FP16 divided to zero | same FP32 scale the kernel deploys |
| Code/scale source | private copy of the kernel's expression | shared helper |

`[2**-20, 2**-22]` is the sharpest case: the FP16 scale `2**-20 / 127` is a
subnormal that rounds to zero, so the old calibration raised while the deployed
kernel used the representable FP32 scale `7.5092466e-9` and codes `[127, 32]`.
For ordinary rows the FP16 division also rounded the scale away from the FP32
value, so every moment carried a small, silent error:

| Input (FP16) | Old Hessian relative error | Old outcome |
| --- | ---: | --- |
| `[1, 0.25]` | `1.220694e-4` | accepted, silently wrong |
| `[0.001, 0.00025]` | `2.381857e-3` | accepted, silently wrong |
| `[2**-20, 2**-22]` | n/a | raised, though the kernel deploys a finite grid |

Both columns are gone after the change: the same helper produces the codes, the
FP32 scale and the moments, and the FP64 oracle in
`tests/test_gsq_qqq_activation_contract.py` agrees to `1e-6` normalized error.

## Flow

```mermaid
flowchart TD
    X["activation rows, last axis = in_features"] --> H{"QQQLinear.dynamic_quant"}
    H -- "CUDA, contiguous, Triton built" --> T["qqq_triton.dynamic_quant<br/>FP32 codes + FP32 scale"]
    H -- "otherwise" --> S["qqq_activation_quantize<br/>FP32 codes + FP32 scale"]
    X --> C["qqq_calibration_moments"]
    C --> S
    S --> M["Hq = Xq.T @ Xq<br/>D = (Xteacher - Xq).T @ Xq"]
    T -.->|"pinned by kernel tests"| M
    M --> R["refine_qqq_codes"]
    style T fill:#d1e7dd,stroke:#198754,stroke-width:2px
    style S fill:#d1e7dd,stroke:#198754,stroke-width:2px
```

The Triton and Metal kernels are separate implementations of the same grid, so
they are qualified by their own tests rather than by routing through the Torch
helper. `tests/test_gsq_qqq_activation_contract.py::test_triton_quantizer_matches_the_torch_grid`
compares their codes and scales for supported dtypes; `tests/test_nan_activation_quantizers_4220.py`
covers nonfinite rows, where the scale is poisoned instead of folded into a code.

## Failure contract

- A non-floating or empty operand is rejected by the helper.
- A nonfinite activation is rejected by the calibration entry point, not
  silently clamped.
- No activation int8 row may have a nonpositive or nonfinite scale.
- The calibration never substitutes a different grid when the kernel's own
  contract rejects an input; it fails instead of fitting statistics for an
  operand the runtime will not produce.

## Consumers and tests

| Consumer | Location | Guarantee |
| --- | --- | --- |
| Portable kernel | `gptqmodel/nn_modules/qlinear/qqq.py::QQQLinear.dynamic_quant` | uses the shared helper |
| Triton kernel | `gptqmodel/nn_modules/qlinear/qqq_triton.py::dynamic_quant` | matches codes/scale |
| GSQ moments | `gptqmodel/quantization/gsq_qqq.py::qqq_calibration_moments` | uses the shared helper |
| GSQ fitter | `gptqmodel/quantization/gsq_qqq.py::refine_qqq_codes` | consumes the moments unchanged |
