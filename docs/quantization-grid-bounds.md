# Weight-grid fit limits: what a fit may select and reconstruct

A weight grid is fitted on an FP32 working copy of the weight and then narrowed
back to the dtype the module replays in. A symmetric INT4 grid stores
`zero = 8` and `scale = 2 * amax / 15`, so the negative endpoint of a group
rounds to code `0` and reconstructs `-16 * amax / 15` - strictly larger than
`amax`. At `amax = 61440` that is `-65536`, finite as an input and as a scale
but outside FP16's range, so the narrowed replay weight becomes `-inf` and the
packer loses the code it would have recovered. The grid and its rounding are
correct; the missing constraint is that the *selected codes* have to stay
representable in the dtype the reconstruction narrows to (#6001, #6012, #6020).

## The contract

```python
from gptqmodel.quantization.quantizer import FitLimits

limits = FitLimits.for_module(module)      # or FitLimits.for_dtype(torch.float16)
limits.max_abs_value   # finfo(dtype).max, or None when the dtype imposes none
limits.scale_floor     # the storage floor for the fitted scale

grid = quantizer.fit_weight(block, limits)  # -> WeightGrid(scale, zero, code_lo, code_hi)
```

| field | meaning |
| --- | --- |
| `dtype` | the dtype the reconstruction is narrowed to; required |
| `max_abs_value` | finite magnitude cap; `None` for FP32 as a working dtype and for integer weights |
| `scale_floor` | storage floor for the fitted scale |

`for_module` / `for_dtype` derive the two limits once, in the working domain.
An optional `restore_factor` for smoothed rows divides **both**: the fit runs
on `weight / factor`, so a bound on the final weight and the storage floor both
carry that factor. Keeping the division here is what stops the two halves from
drifting apart, and per-row `[rows, 1]` factors are normalized to the fitted
lane axis, so they cannot broadcast the floor into `[rows, rows]`.

RTN scores codes directly rather than through `find_params` and bounds FP32 by
its own finite maximum; `bound_fp32=True` keeps that solver rule explicit.
GPTQ leaves FP32 unconstrained.

## The single entry

```python
quantizer.fit_weight(block, limits)   # required: the dtype that will replay it
```

`fit_weight` is the only way a weight fit may reach `find_params`: it passes
`weight=True`, `scale_min=limits.scale_floor` and `value_limit=limits.max_abs_value`
together, normalizes per-row limits to the fitted lane axis, and refuses a
caller that cannot name its replay dtype. Activation fits keep
`find_params(weight=False)`.

The returned `WeightGrid` carries scale, zero and the inclusive code range of
that one fit. The fused consumer reads exactly that range through
`_fused_code_bounds`; the packers recover codes from the replay weight produced
with the same scale and zero.

## Consumers

| caller | through |
| --- | --- |
| `RTN` plain branch | `fit_weight(block, FitLimits.for_module(module, restore_factor=..., bound_fp32=True))` |
| `RTN` MSE branch | `mse_optimal_quant(value_limit=limits.max_abs_value)`, scored on the same cap |
| `GPTQ._fit_weight_params` | `fit_weight(weight, FitLimits.for_module(module))`; GPTAQ and FOEM inherit it |
| GPTQ's RTN fallback | `fit_weight(block, FitLimits.for_module(module, restore_factor=scale_factor))` |

## Adding a new fit

1. Build the limits from the module or the replay dtype - never invent a cap.
2. Pass them through `fit_weight` (or `mse_optimal_quant`'s `value_limit` when the
   fit scores codes itself).
3. Pin the endpoint fixture: an FP16 layer with rows `[-a, a, 0, ...]` must
   replay `-65504` at `a = 61408` and `-57344` at `a = 61440`, and unpacking
   codes `0` and `1` must recover those values. `tests/test_fit_limits_contract.py`
   is the model for that test.
