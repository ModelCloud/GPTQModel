# Weight-grid bounds: what a fit may reconstruct

A weight grid is fitted on an FP32 working copy of the weight and then narrowed
back to the dtype the module replays in. A symmetric INT4 grid stores
`zero = 8` and `scale = 2 * amax / 15`, so the negative endpoint of a group
rounds to code `0` and reconstructs `-16 * amax / 15` - strictly larger than
`amax`. At `amax = 61440` that is `-65536`, which is finite as an input and as a
scale but outside FP16's range, so the narrowed replay weight becomes `-inf`
and the packer loses the code it would have recovered. The grid and its
rounding are correct; the missing constraint is that the *selected codes* have
to stay representable in the dtype the reconstruction narrows to (#6001,
#6012, #6020).

## The contract

```python
from gptqmodel.quantization.quantizer import ReplayBounds, Quantizer

bounds = ReplayBounds.for_module(module)          # reads module.weight.dtype
bounds.value_limit   # finfo(dtype).max, or None for FP32 / integer dtypes
bounds.scale_min     # the storage floor for the fitted scale
```

| field | meaning |
| --- | --- |
| `replay_dtype` | the dtype the reconstruction is narrowed to; required |
| `scale_floor` | storage floor for the scale; defaults to `SCALE_STORAGE_FLOOR` |
| `restore_factor` | optional per-row smoothing factor the caller multiplies back |

`restore_factor` exists because a smoothed fit runs on `weight / factor`: a
bound on the final weight, and the storage floor, both divide by that factor
before they reach the fit. The type does that division once, so a caller cannot
forget one half of it.

## The single entry

```python
quantizer.fit_weight(block, bounds)   # required: the dtype that will replay it
```

`fit_weight` is the only way a weight fit may reach `find_params`: it passes
`weight=True`, `scale_min=bounds.scale_min` and `value_limit=bounds.value_limit`
together. A fit that cannot name its replay dtype is refused with a `TypeError`,
because the unconstrained grid is exactly the failure above. Activation fits
keep `find_params(weight=False)`.

## Consumers

| caller | through |
| --- | --- |
| `RTN` plain branch | `fit_weight(block, ReplayBounds.for_module(module, restore_factor=...))` |
| `RTN` MSE branch | `mse_optimal_quant(value_limit=bounds.value_limit)`, scored on the same bound |
| `GPTQ._fit_weight_params` | `fit_weight(weight, ReplayBounds.for_module(module))`; GPTAQ and FOEM inherit it |
| GPTQ's RTN fallback | `fit_weight(block, ReplayBounds.for_module(module, restore_factor=scale_factor))` |

## Adding a new fit

1. Build the bounds from the module or the replay dtype - never invent a limit.
2. Pass them through `fit_weight` (or `mse_optimal_quant`'s `value_limit` if the
   fit scores codes itself).
3. Pin the endpoint fixture: an FP16 layer with rows `[-a, a, 0, ...]` must
   replay `-65504` at `a = 61408` and `-57344` at `a = 61440`, and unpacking
   codes `0` and `1` must recover those values. `tests/test_replay_bounds_contract.py`
   is the model for that test.
