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
`polly/qwen3-acceptance`, based on `5f1183c0`; the initial harness implementation commit is
`3793fa9d05cf6f78365d407ec03494ce73842cd4`.

The split plan is calibration rows 0--511, YAQA/tuning 512--1023, validation 1024--1535, held-out diagnostics
1536--2047, and a disjoint diverse pool 2048--2559. Diverse-32 sorts the pool by canonical UTF-8 content length and
stable row identity, partitions it into 32 bins of 16, and selects rank 8 per bin. Manifests contain source identities
and SHA-256 content hashes; all ten pairwise comparisons must be empty.

```bash
python scripts/accept_qwen3_8b_qvq.py export-frozen-splits \
  --dataset neuralmagic/calibration --dataset-config LLM --dataset-split train \
  --dataset-revision fb6bc2f8c66543876fb31613f5872b9030220e15 \
  --output-dir artifacts/qwen3_8b_qvq_w2/splits

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<verified-idle-GPU-UUIDs> python scripts/qvq_quantize.py \
  --model /monster/data/model/Qwen3-8B --output artifacts/qwen3_8b_qvq_w2/checkpoint \
  --quant-config configs/qwen3_8b_qvq_w2_acceptance.json \
  --calibration-dataset artifacts/qwen3_8b_qvq_w2/splits/calibration.jsonl --calibration-rows 512 \
  --yaqa-dataset artifacts/qwen3_8b_qvq_w2/splits/yaqa_tuning.jsonl --yaqa-rows 512 \
  --validation-dataset artifacts/qwen3_8b_qvq_w2/splits/validation.jsonl --validation-rows 512 \
  --verify-qwen3-acceptance-payload-parity

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<same-verified-idle-GPU-UUIDs> \
python scripts/accept_qwen3_8b_qvq.py evaluate \
  --dense-model /monster/data/model/Qwen3-8B \
  --revision b968826d9c46dd6066d109eabc6255188de91218 \
  --checkpoint artifacts/qwen3_8b_qvq_w2/checkpoint --manifest-dir artifacts/qwen3_8b_qvq_w2/splits \
  --validation-jsonl artifacts/qwen3_8b_qvq_w2/splits/validation.jsonl \
  --held-out-diagnostics-jsonl artifacts/qwen3_8b_qvq_w2/splits/held_out_diagnostics.jsonl \
  --diverse-jsonl artifacts/qwen3_8b_qvq_w2/splits/diverse_32.jsonl \
  --maximum-bpw 2.1 --score-min 0.85 --final-kl-max-nats 0.10 \
  --output artifacts/qwen3_8b_qvq_w2/acceptance.json

python scripts/accept_qwen3_8b_qvq.py gate \
  --dense-model /monster/data/model/Qwen3-8B \
  --revision b968826d9c46dd6066d109eabc6255188de91218 \
  --checkpoint artifacts/qwen3_8b_qvq_w2/checkpoint --manifest-dir artifacts/qwen3_8b_qvq_w2/splits \
  --validation-jsonl artifacts/qwen3_8b_qvq_w2/splits/validation.jsonl \
  --held-out-diagnostics-jsonl artifacts/qwen3_8b_qvq_w2/splits/held_out_diagnostics.jsonl \
  --diverse-jsonl artifacts/qwen3_8b_qvq_w2/splits/diverse_32.jsonl \
  --maximum-bpw 2.1 --score-min 0.85 --final-kl-max-nats 0.10 \
  --report artifacts/qwen3_8b_qvq_w2/acceptance.json
sha256sum artifacts/qwen3_8b_qvq_w2/acceptance.json artifacts/qwen3_8b_qvq_w2/checkpoint/*.safetensors
```

Evaluation performs a fresh `GPTQModel.load`, requires 252 exact-type W2 `QVQLinear` modules, and parses the saved
safetensors headers/data offsets to account every tensor below each requested projection prefix: trellis, packed
selectors/bank metadata, FP32 SU/SV, bias, explicit outliers, and future auxiliaries. Container overhead and non-target
tensors are separate. Global and per-cell metrics cover all 512 validation, 512 held-out diagnostic, and 32 diverse
records. Per-cell final-KL is direct evidence: one dense projection output is replaced with the reloaded QVQ module
output on the identical dense input before observing final logits.

The committed fixture directory `tests/data/qwen3_8b_qvq_acceptance` is the content authority produced by the export
command: it includes `diverse_pool_512.jsonl` and its manifest in addition to the selected `diverse_32` files. The
selection is recomputed from canonical UTF-8 content length and stable identity (32 bins of 16, rank 8), and the gate
proves that the full pool is identity/content-disjoint from calibration, tuning, validation, and held-out diagnostics.

For every canonical projection, dimensions are fixed by role: q/o 4096x4096, k/v 4096x1024, gate/up 4096x12288,
and down 12288x4096 (`in_features x out_features`). Runtime drift is rejected before the fixed denominator is used.
The dense authority is the exact five-shard/index SHA-256 set, not Hub metadata alone. Quantization hashes the actual
direct parameter/buffer bytes before save, frees the producer model, launches a fresh process and fresh checkpoint
load, and requires canonical per-module and aggregate hashes to match. Evaluation hashes its independent reload again.

Focused tests cover leakage, diverse cardinality, missing modules, dense/higher-precision fallback, auxiliary/BPW
accounting, thresholds, missing cells, coverage, and report schema. The implementation gate passed 31 focused and
existing unified-harness tests, Ruff on all changed Python paths, `compileall`, config construction, CLI imports, and
`git diff --check`. After safely installing missing declared dependencies (`accelerate`, `threadpoolctl`, `device-smi`,
`defuser`, and `pillow`), the exact `export-frozen-splits` command above succeeded against the pinned real dataset:
four 512-row manifests and one 32-row manifest were emitted and all pairwise identity/content checks passed.
`pyright` was not installed in this worktree environment, so no static-typecheck result is claimed; Python bytecode
compilation and the executable config/CLI import checks passed.

No Qwen3-8B quantization or accuracy run has been performed for this ledger entry, so no model artifact or accuracy
result exists. The host exposes one idle NVIDIA PG506-232 (`GPU-20f7fde4-d88c-d6ca-e324-bd4e5e9e0855`, PCI
`00000000:2B:00.0`, 98,304 MiB, 0 MiB used and 0% utilization at inspection), but the pinned 8B model snapshot is not
materialized. Still required are model download, the full quantize/save/reload run, BPW <=2.1, all 252 interventions,
global metrics, and artifact hashes. Missing evidence is a hard rejection and is not represented as a passing score.

## Rejected review and blocking fixes (2026-08-23)

Review rejected commits `3793fa9d` and `297197bf` for eight blocking issues: AND instead of OR score semantics;
report-only acceptance trusting caller booleans/hashes/counts; weak `config.json` and shard-index authority; no exact
local model identity; quantization inputs bound only to paths; incomplete canonical accounting consistency; weak
pairwise-manifest schema validation; and missing explicit unified-harness regression coverage.

This follow-up corrects the score contract at every global and layer-role scope to `(top1 >= 0.85) OR
(diverse_32 >= 0.85)`. Final KL remains a separate mandatory maximum in nats. Regression cases cover top-1-only pass,
diverse-32-only pass, and both-fail rejection. The standalone `gate` command no longer has a report-only invocation:
it reloads the model, reparses configs and safetensors, rehashes dense/checkpoint files and every manifest/JSONL,
revalidates the quantization-run content bindings, reruns global and all 252 direct intervention scopes, and requires
the submitted report to exactly equal that artifact-recomputed evidence.

The exact local target is `/monster/data/model/Qwen3-8B`; no model download was performed or added to the workflow.
Its local Hugging Face metadata pins revision `b968826d9c46dd6066d109eabc6255188de91218`. Its known `config.json` SHA-256
is `f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30`; its shard-index SHA-256 is
`f9fdbcb91c23971c13ec5d5f2573d2349e8f61f2f049371ec699281748fdb1bc`. The validated identity is
`model_type=qwen3`, architecture `Qwen3ForCausalLM`, hidden/intermediate 4096/12288, 36 layers exactly, 32 query and
8 KV heads, head dimension 128, vocab 151936, and untied embeddings. The index contains exactly layers 0--35 and no
extra decoder layer. The model card names this post-trained, instruction-following, switchable-thinking model
**Qwen3-8B**, not **Qwen3-8B-Instruct**; the acceptance objective's “Instruct” label refers to its post-trained
instruction behavior rather than a distinct local repository name.

The earlier 2026-08-22 statement that the snapshot was not materialized was incorrect: inspection on 2026-08-23
confirmed the complete five-shard 16,381,470,720-byte indexed snapshot was already local. This correction preserves
the earlier entry while superseding that blocker. No quantization or accuracy run has been performed, and no >=85%
result is claimed. The actual model-run evidence remains unresolved until the frozen quantize/save/reload and complete
artifact-aware gate commands finish.

Implementation-gate ledger for this rejected-review repair (run from
`/root/qvq/.worktrees/qwen3-acceptance` on branch `polly/qwen3-acceptance`; parent commit
`297197bfd3bad4bc71ecf446ae011b0d8c3dfd13`):

- `pytest -q tests/test_qvq_qwen3_acceptance.py tests/test_qvq_unified_harness.py`: 41 passed, 14 upstream
  `torch.jit` deprecation warnings.
- `pytest --collect-only -q tests/test_qvq_qwen3_acceptance.py tests/test_qvq_unified_harness.py`: exactly 41 tests
  collected, matching the executed focused-plus-unified files.
- `ruff check gptqmodel/utils/qvq_acceptance.py scripts/accept_qwen3_8b_qvq.py scripts/qvq_quantize.py
  tests/test_qvq_qwen3_acceptance.py tests/test_qvq_unified_harness.py`: all checks passed.
- `python -m compileall -q gptqmodel/utils/qvq_acceptance.py scripts/accept_qwen3_8b_qvq.py
  scripts/qvq_quantize.py tests/test_qvq_qwen3_acceptance.py tests/test_qvq_unified_harness.py`: passed.
- `python scripts/accept_qwen3_8b_qvq.py --help`, `python scripts/accept_qwen3_8b_qvq.py gate --help`, and
  `python scripts/qvq_quantize.py --help`: passed; the gate help requires all authoritative artifact/evaluation
  inputs and exposes no accepting report-only path.
- `validate_qwen3_model_artifact(Path('/monster/data/model/Qwen3-8B'), require_pinned_dense=True)`: passed with the
  pinned revision and config/index hashes recorded above, exact layers 0--35, and all seven critical local Hub
  content identities.
- `git diff --check`: passed. Worktree cleanliness is checked again after the repair commit. The resulting commit is
  identified by `git rev-parse HEAD`; artifact identity remains the hashes above until a real quantized checkpoint
  exists.

The implementation gates validate fail-closed behavior and schemas only. Effective BPW and all score/KL gates still
have no real Qwen3-8B quantized artifact to measure, so they remain unresolved rather than recorded as successes.
