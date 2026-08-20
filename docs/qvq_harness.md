# Unified QVQ quantization and evaluation harnesses

QVQ has two production-facing, model-agnostic command-line harnesses:

- `scripts/qvq_quantize.py` loads a dense model, runs every required preparation stage through the normal
  GPTQModel lifecycle, quantizes, saves the checkpoint, and writes a reproducibility manifest.
- `scripts/qvq_evaluate.py` loads an already saved checkpoint. Its `diagnostics` command compares final logits
  with the dense source; its `tasks` command runs Evalution suites.

Preparation and evaluation are deliberately separated. Sketch-B, activation Hessians, output alignment, and
module-granular replay can affect quantization and therefore belong in the quantization process. Held-out final-logit
or task evaluation must not mutate the checkpoint or feed data back into preparation.

## Quantize

The following example uses three disjoint 512-row streams. Add separate replay search and confirmation streams when
module-granular replay is enabled.

```bash
python scripts/qvq_quantize.py \
  --model /private/monster/data/model/DenseModel \
  --output /private/monster/data/model/qvq-model \
  --bits 2 \
  --format qvq_v2b2_p32 \
  --rounding yaqa \
  --calibration-dataset neuralmagic/calibration --calibration-row-start 0 --calibration-rows 512 \
  --yaqa-dataset neuralmagic/calibration --yaqa-row-start 512 --yaqa-rows 512 \
  --validation-dataset neuralmagic/calibration --validation-row-start 1024 --validation-rows 512
```

The script does not contain architecture-specific module names. `GPTQModel.load()` selects the registered model
adapter and `model.quantize()` owns preparation ordering, module discovery, shell/turtle materialization, GPU
delegation, and quantization. Omitting `--layers` quantizes every supported layer and module.

For exact production configuration, pass `--quant-config config.json`. The resulting checkpoint contains
`qvq_quantize_run.json`, including the Git revision, resolved QVQ configuration, exact dataset slices, timing, and
layer scope. Existing output directories are never overwritten.

## Evaluate final logits

Use a disjoint slice. When the quantization manifest is available, accidental overlap with any preparation stream
fails closed.

```bash
python scripts/qvq_evaluate.py diagnostics \
  --dense-model /private/monster/data/model/DenseModel \
  --checkpoint /private/monster/data/model/qvq-model \
  --dataset neuralmagic/calibration --row-start 1536 --rows 512 \
  --device cuda:0 \
  --output artifacts/qvq-model-diagnostics.json
```

Diagnostics preserve full row lengths by default and report token-weighted final KL, Top-1 agreement, and the three
Divergent-300 statistics. `--include-topn` additionally reports legacy Top-5 and Top-10 set overlap.

## Run Evalution tasks

```bash
python scripts/qvq_evaluate.py tasks \
  --checkpoint /private/monster/data/model/qvq-model \
  --task arc_challenge --task gsm8k_platinum_cot \
  --output artifacts/qvq-model-tasks.json
```

The former model-specific comparison and lifecycle-validation scripts remain research and compatibility tools. New
production sweeps should invoke these two unified harnesses so quantization and post-quant evaluation cannot silently
use different lifecycle implementations.
