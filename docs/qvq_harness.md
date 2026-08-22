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

## Experiment ledger: Qwen3-8B QVQ W2 acceptance harness (2026-08-22)

This implementation adds a stricter architecture-locked authority in `scripts/accept_qwen3_8b_qvq.py` without
changing the general harness. The target is Qwen/Qwen3-8B revision
`b968826d9c46dd6066d109eabc6255188de91218`; the frozen source is `neuralmagic/calibration` revision
`fb6bc2f8c66543876fb31613f5872b9030220e15`, config `LLM`, split `train`. The branch is
`polly/qwen3-acceptance`, based on `5f1183c0`.

The split plan is calibration rows 0--511, YAQA/tuning 512--1023, validation 1024--1535, held-out diagnostics
1536--2047, and a disjoint diverse pool 2048--2559. Diverse-32 sorts the pool by canonical UTF-8 content length and
stable row identity, partitions it into 32 bins of 16, and selects rank 8 per bin. Manifests contain source identities
and SHA-256 content hashes; all ten pairwise comparisons must be empty.

```bash
python scripts/accept_qwen3_8b_qvq.py export-frozen-splits \
  --dataset neuralmagic/calibration --dataset-config LLM --dataset-split train \
  --dataset-revision fb6bc2f8c66543876fb31613f5872b9030220e15 \
  --output-dir artifacts/qwen3_8b_qvq_w2/splits

huggingface-cli download Qwen/Qwen3-8B \
  --revision b968826d9c46dd6066d109eabc6255188de91218 \
  --local-dir artifacts/qwen3_8b_qvq_w2/dense-model

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<verified-idle-GPU-UUIDs> python scripts/qvq_quantize.py \
  --model artifacts/qwen3_8b_qvq_w2/dense-model --output artifacts/qwen3_8b_qvq_w2/checkpoint \
  --quant-config configs/qwen3_8b_qvq_w2_acceptance.json \
  --calibration-dataset artifacts/qwen3_8b_qvq_w2/splits/calibration.jsonl --calibration-rows 512 \
  --yaqa-dataset artifacts/qwen3_8b_qvq_w2/splits/yaqa_tuning.jsonl --yaqa-rows 512 \
  --validation-dataset artifacts/qwen3_8b_qvq_w2/splits/validation.jsonl --validation-rows 512

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<same-verified-idle-GPU-UUIDs> \
python scripts/accept_qwen3_8b_qvq.py evaluate \
  --dense-model artifacts/qwen3_8b_qvq_w2/dense-model \
  --revision b968826d9c46dd6066d109eabc6255188de91218 \
  --checkpoint artifacts/qwen3_8b_qvq_w2/checkpoint --manifest-dir artifacts/qwen3_8b_qvq_w2/splits \
  --validation-jsonl artifacts/qwen3_8b_qvq_w2/splits/validation.jsonl \
  --diverse-jsonl artifacts/qwen3_8b_qvq_w2/splits/diverse_32.jsonl \
  --maximum-bpw 2.1 --score-min 0.85 --final-kl-max-nats 0.10 \
  --output artifacts/qwen3_8b_qvq_w2/acceptance.json

python scripts/accept_qwen3_8b_qvq.py gate --report artifacts/qwen3_8b_qvq_w2/acceptance.json
sha256sum artifacts/qwen3_8b_qvq_w2/acceptance.json artifacts/qwen3_8b_qvq_w2/checkpoint/*.safetensors
```

Evaluation performs a fresh `GPTQModel.load`, requires 252 exact-type `QVQLinear` modules, and accounts every tensor
below each requested projection prefix: trellis, packed selectors/bank metadata, FP32 SU/SV, bias, explicit outliers,
and future auxiliaries. Non-target tensors are separate. Per-cell final-KL is direct evidence: one dense projection
output is replaced with the reloaded QVQ module output on the identical dense input before observing final logits.

Focused tests cover leakage, diverse cardinality, missing modules, dense/higher-precision fallback, auxiliary/BPW
accounting, thresholds, missing cells, coverage, and report schema. The implementation gate passed 28 focused and
existing unified-harness tests, Ruff on all changed Python paths, `compileall`, config construction, CLI imports, and
`git diff --check`. After safely installing missing declared dependencies (`accelerate`, `threadpoolctl`, `device-smi`,
`defuser`, and `pillow`), the exact `export-frozen-splits` command above succeeded against the pinned real dataset:
four 512-row manifests and one 32-row manifest were emitted and all pairwise identity/content checks passed.

No Qwen3-8B quantization or accuracy run has been performed for this ledger entry, so no model artifact or accuracy
result exists. The host exposes one idle NVIDIA PG506-232 (`GPU-20f7fde4-d88c-d6ca-e324-bd4e5e9e0855`, PCI
`00000000:2B:00.0`, 98,304 MiB, 0 MiB used and 0% utilization at inspection), but the pinned 8B model snapshot is not
materialized. Still required are model download, the full quantize/save/reload run, BPW <=2.1, all 252 interventions,
global metrics, and artifact hashes. Missing evidence is a hard rejection and is not represented as a passing score.
