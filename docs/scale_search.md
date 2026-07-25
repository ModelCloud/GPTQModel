# ScaleSearch

ScaleSearch optimizes the quantization scale/zero for each group by scanning a grid of candidate clipping ranges and selecting the one that minimizes a reconstruction objective. It is controlled through `QuantizeConfig` and applies to GPTQ-style quantization in GPT-QModel.

## When ScaleSearch runs

ScaleSearch is active when `mse > 0` **or** when `scale_search` is explicitly set to a non-`None` value. The default `mse` is `0.0`, but the default `scale_search` is `ScaleSearchConfig.ACTIVATION`; the config normalization automatically sets `mse = 2.0` for any non-`None` scale-search mode, so a plain `QuantizeConfig(bits=4, group_size=128)` will use **activation** scale search.

To disable scale search and fall back to a simple min/max scale, use:

```python
from gptqmodel.quantization import QuantizeConfig, ScaleSearchConfig

qcfg = QuantizeConfig(bits=4, group_size=128, mse=0.0, scale_search=None)
```

## Algorithms

| Mode | How it scores candidates | Speed | Best for |
|------|--------------------------|-------|----------|
| `ScaleSearchConfig.MSE` (legacy) | Minimizes uniform squared reconstruction error: `(dequant - weight)^2` | Fastest | Baseline; ignores activation/Hessian structure. |
| `ScaleSearchConfig.ACTIVATION` | Weights the squared error by the Hessian diagonal / activation importance per group | Fast | Default. Usually better than plain MSE with small extra cost; Triton optimized. |
| `ScaleSearchConfig.HESSIAN` | Minimizes the full quadratic form `error^T @ H_group @ error` using per-group Hessian blocks | Slower | When per-group correlations matter and `group_size` is 32/64/128. |
| `ScaleSearchConfig.HYBRID` | Averages the Hessian quadratic form with the uniform MSE term (`0.5 * Hessian + 0.5 * MSE`) | Slowest | Trade-off that keeps some correlation awareness while penalizing uniform error. |

### Effect on speed

- `MSE` and `ACTIVATION` are roughly 2-3x faster than `HESSIAN`/`HYBRID` because they avoid the per-group matrix multiply in the loss.
- `HESSIAN`/`HYBRID` are accelerated by a Triton fast path when `group_size <= 128`, `bits >= 4`, and the input is contiguous CUDA FP16/BF16/FP32. `bits=2` falls back to the exact Python loop because the coarse `maxq=3` grid creates near-tie candidates that are unsafe with the Triton top-k shortlist.
- Disable all Triton scale-search fast paths with `GPTQMODEL_SCALE_SEARCH_TRITON=0`.

### Effect on quality

- `MSE` is the simplest reference; it may miss outliers that the calibration activations reveal.
- `ACTIVATION` up-weights columns that carry larger activation energy, which usually improves perplexity over `MSE` without a large slowdown.
- `HESSIAN` can improve quality further when weights within a group are strongly correlated, because the Hessian captures second-order structure. It is most useful for small `group_size` where the local Hessian is well-conditioned.
- `HYBRID` can be more robust than pure Hessian when the Hessian is noisy (few calibration samples, nearly singular blocks), but it adds the MSE term and is the slowest mode.

## Usage

```python
from gptqmodel.quantization import QuantizeConfig, ScaleSearchConfig

# Default: activation scale search with mse=2.0
qcfg = QuantizeConfig(bits=4, group_size=128)

# Plain MSE grid search (legacy behavior)
qcfg = QuantizeConfig(bits=4, group_size=128, mse=2.0, scale_search=ScaleSearchConfig.MSE)

# Hessian-aware scale search
qcfg = QuantizeConfig(bits=4, group_size=128, mse=2.0, scale_search=ScaleSearchConfig.HESSIAN)

# Hybrid objective
qcfg = QuantizeConfig(bits=4, group_size=128, mse=2.0, scale_search=ScaleSearchConfig.HYBRID)
```

### Per-module override

Use the standard `dynamic` config map to select a different scale-search mode for specific layers:

```python
qcfg = QuantizeConfig(
    bits=4,
    group_size=128,
    mse=2.0,
    scale_search=ScaleSearchConfig.ACTIVATION,
    dynamic={
        r".*\.self_attn\.(q_proj|k_proj|v_proj)$": {"scale_search": ScaleSearchConfig.HESSIAN},
    },
)
```

## Tuning knobs

- `mse` (float, default `0.0`): controls the scale-search grid and objective. For `ACTIVATION`/`HESSIAN`/`HYBRID` it is normalized to `2.0` if unset. For `MSE` it is the exponent used in the loss (`error.abs().pow(mse)`); `2.0` gives standard squared error.
- `grid` (int, default `100`) and `maxshrink` (float, default `0.8`): the number of candidate shrink ratios is `int(grid * maxshrink)`, defaulting to 80. Larger values search more finely but cost more compute. These are `Quantizer` parameters set through `quantizer.configure(grid=..., maxshrink=...)`.
- `group_size`: the size of each quantization group. ScaleSearch is optimized for `group_size` 32, 64, and 128. `group_size=1` is not tested; `group_size=-1` disables grouping and uses a single scale per column.

## Accuracy guarantees

The Triton fast paths are validated against the exact per-group `Quantizer.find_params` reference with a `<1e-6` scale/zero tolerance. You can rerun this check with:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=5 PYTHON_GIL=0 \
  python scripts/validate_find_params_batched_strict.py
```

The script must report `STRICT CHECK PASSED` before any speed optimization is accepted.

## Environment variables

- `GPTQMODEL_SCALE_SEARCH_TRITON=0`: force the exact Python chunk loop instead of the Triton fast paths.
- `PYTHON_GIL=0`: recommended for free-threaded runs on multi-GPU setups.
