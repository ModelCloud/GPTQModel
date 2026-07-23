# ParoQuant

## Paper calibration regime

The paper-reproduction path uses 2,048 full 2,048-token training sequences mixed evenly from WikiText2, C4, and
RedPajama, plus an independent 64-sequence Pile validation set. Both streams use seed 0 and a batch size of 16.

GPT-QModel exposes the official corpus recipe without downloading data implicitly:

```python
from gptqmodel.quantization.config import ParoConfig
from gptqmodel.quantization.paroquant import build_paroquant_calibration_datasets

paper_calibration = build_paroquant_calibration_datasets(model.tokenizer)
model.quantize_config = ParoConfig(
    bits=4,
    group_size=128,
    opt_scope="layer",
    opt_stage_impl="reference",
    opt_train_samples=2048,
    opt_validation_samples=64,
    opt_batch_size=16,
)
model.quantize(
    calibration=paper_calibration.train,
    validation_calibration=paper_calibration.validation,
    calibration_sort=None,
    batch_size=16,
)
```

`validation_calibration` is ParoQuant-only and is prepared independently from `calibration`. If it is omitted,
ParoQuant retains the older single-stream prefix/suffix split for compatibility. That fallback is not an exact
reproduction of the paper and can overlap when too few sequences are supplied.

`ParoConfig.opt_batch_size` retains GPT-QModel's tuned default of 64 and controls the activation-row minibatch in
module scope. The independent `quantize(batch_size=...)` argument controls calibration capture/replay and retains its
default of 1. The exact reproduction call above sets both to 16 explicitly; either value may be tuned independently
for the selected scope and available memory. The paper uses a capture batch of 8 for its 70B run.

Whole-layer `opt_scope="layer"` is required for the paper's layer-wise recovery objective. The lighter
`opt_scope="module"` path still bounds each linear's optimization to sampled activation rows after separating the
training and validation streams. The paper-reproduction path also selects `opt_stage_impl="reference"` so each
optimization stage evaluates and snapshots its initial validation state, retaining it when no epoch improves the
validation loss. The tuned `opt_stage_impl="fast"` path defers its first snapshot until after epoch one.

## Activation ordering

`ParoConfig` does not expose or serialize `desc_act`. That setting controls GPTQ activation-order packing and is not
part of the ParoQuant paper, optimizer, packed-weight format, or inference path. Generic quantized-linear plumbing
still receives an internal `False` compatibility value.

Older ParoQuant checkpoint configs that contain global or dynamic `desc_act` entries remain loadable; the loader
discards those ineffective entries. New ParoQuant configurations should omit the setting.

## Activation Checkpointing

`ParoConfig.opt_gradient_checkpointing` controls activation checkpointing during ParoQuant's train-style optimization stages.

- `opt_scope="layer"` defaults to `opt_gradient_checkpointing=True`
- `opt_scope="module"` defaults to `opt_gradient_checkpointing=False`
- `opt_scope="compute_block"` defaults to `opt_gradient_checkpointing=False`

Current internal benchmarks have only shown a clear resource-usage benefit for `layer` scope. `module` and `compute_block` support the toggle, but they are not enabled by default because we have not yet measured a consistent memory win there.
