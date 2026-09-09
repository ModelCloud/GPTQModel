---
name: quantized-model-provenance
description: Require self-contained run and evaluation records for every quantized model artifact.
---

# Quantized model and evaluation provenance

Use this skill for every model-affecting quantization, conversion, export, or
post-quantization evaluation. A quantized model is not publishable, reusable,
or a valid comparison artifact until its provenance record is complete.

## Required model record

Create `model_run.md` at the root of the stored model artifact. If one run
produces multiple format directories, put one record at the common snapshot
root and repeat or link it from each independently reusable format directory.
The record must contain the values inline; a pointer to a log, shell history,
or an ephemeral command file is not sufficient.

The record must include all of the following:

- `run_id`: stable unique identifier for the experiment/run.
- `arm_id`: stable identifier for this exact comparison arm. Use a separate
  arm for each engine, kernel, quantizer, rate, seed, or other changed factor.
- UTC start/end timestamps and status (`completed`, `failed`, or `aborted`).
- The absolute stored artifact path, plus the relative format path when there
  is a common snapshot root. Do not use only a URI, symlink, or shortened path.
- Source model ID and exact source model path; tokenizer path, revision, and
  chat-template binding.
- The complete quantization/conversion configuration inline, including every
  non-default option, dynamic module/rate override, seed, calibration limits,
  kernel/runtime switches, and output format. Also list the paths to the
  exact config files copied into the artifact.
- Every dataset used by the run, with an absolute path or an explicit remote
  dataset ID and revision, split, row selection/order, and SHA-256 when the
  source is a local file. This includes calibration, held-out/audit, and any
  dataset used to derive a scale, imatrix, recipe, or plan.
- The full 40-character QVQ commit SHA that produced or loaded the artifact.
  If QVQ was not involved, write `N/A (not a QVQ artifact)` explicitly. When
  ZML consumed or built the artifact, also record the full 40-character ZML
  commit SHA and the exact dependency pin used.
- Full environment and hardware identity: Python/package versions, compiler
  and CUDA/ROCm versions, GPU model and device identity, and relevant build
  flags.
- The exact output files, byte sizes, and SHA-256 hashes; include every model
  shard and loader/config/tokenizer file needed to load the artifact.
- The complete quantization stdout/stderr log, stored as a durable file beside
  the artifact (for example `logs/quantization.log`), with its absolute path,
  byte size, and SHA-256. Do not rely on terminal scrollback or a CI URL that
  may expire.
- Links or absolute paths to every post-quant evaluation record and raw result
  produced for this run/arm.

Do not replace an exact value with `default`, `latest`, `current`, a short
commit, `...`, or a shell variable whose value is not recorded. If a field is
not applicable, record `N/A` and explain why.

## Required post-quant evaluation record

For every post-quant task or test invocation, create a unique Markdown record
next to the raw result, for example:

```text
<artifact>/evaluations/<task>__<run_id>__<arm_id>.md
<artifact>/evaluations/<task>__<run_id>__<arm_id>.json
```

Never overwrite a result from another arm. If a suite writes multiple raw
files, the Markdown record must enumerate all of them. The record must include:

- `run_id` and `arm_id`, matching `model_run.md`.
- The absolute model artifact path and the absolute evaluation-output paths.
- The complete copy-pasteable CLI used to run the test, including all flags,
  environment assignments, working directory, task names, row limits,
  batch/context limits, backend, device, dtype, attention/paging/graph/cache
  settings, and generation parameters.
- The complete effective evaluation configuration inline, not only the CLI
  wrapper name.
- The exact evaluation dataset path or remote dataset ID/revision, split,
  sample count, row selection, ordering, prompt/few-shot/chat-template
  settings, tokenizer path, and scoring/extraction rules.
- Full 40-character QVQ and ZML commit SHAs, or explicit `N/A` values, plus
  all relevant dependency/runtime versions (including Evalution).
- Full results: every reported metric and numerator/denominator, invalid or
  incomplete counts, per-task results, timings/throughput, and the path and
  SHA-256 of the raw per-row output. Preserve the raw JSON/text output rather
  than summarizing it away.
- The complete evaluation stdout/stderr log, stored as a durable file beside
  the raw result (for example `logs/evaluation.log`), with its absolute path,
  byte size, and SHA-256. Store failed and partial logs as well as successful
  logs.
- A comparison interpretation that separates model-quality changes from
  tokenizer, prompt, scheduling, kernel, backend, or runtime changes.
- The exact command(s) used for validation and whether each passed, failed,
  skipped, or was not run.

For the F6/seed7 Transformers comparison contract, use and record
`--attn-implementation 'paged|flash_attention_2'` with an explicit
`--cuda-graph-mode` value. The intended production comparison is paged
continuous batching with decode CUDA-graph replay (`--cuda-graph-mode decode`;
varlen remains eager unless `both` is deliberately selected). Confirm the
effective attention backend, `paged_attention`, continuous-refill mode, and
graph booleans in the durable log and raw result; a requested flag alone is
not evidence that the runtime used it.

“Full results” means the raw per-sample predictions remain available. A score
alone, a console excerpt, or a benchmark screenshot is not a post-quant
evaluation record.

## Run procedure

1. Allocate `run_id` and all arm IDs before starting. Record the intended
   baseline and changed factors.
2. Resolve and record both repository SHAs before quantization or evaluation:

   ```bash
   git -C /root/qvq rev-parse HEAD
   git -C /root/zml-ultra rev-parse HEAD
   ```

   Record the actual checkout paths when they differ from these examples.
3. Capture the exact command and effective configuration before launching.
4. Resolve every dataset/config/model path and record its revision and hash.
5. Redirect or tee complete quantization and evaluation stdout/stderr to
   durable log files in the artifact/result area. Keep logs for failed runs.
6. Run the smallest targeted check first, then the requested full evaluation.
7. Write `model_run.md` and the evaluation Markdown record before reporting
   success. Validate that the stored artifact loads from its stored path and
   that every referenced raw result and log exists and hashes as recorded.
8. Run `git diff --check` on repository documentation changes. Do not claim a
   comparison is controlled when its run/arm, model, dataset, tokenizer,
   engine, or relevant commits differ.

## Minimal record template

Use this outline, filling every field with an exact value:

```markdown
# Quantized model run

- run_id: `<full stable id>`
- arm_id: `<exact arm id>`
- status: `completed|failed|aborted`
- started_utc: `<timestamp>`
- finished_utc: `<timestamp>`
- stored_artifact_path: `/absolute/path/to/model`
- source_model_path: `/absolute/path/or/model@revision`
- tokenizer_path_and_revision: `/absolute/path`, `<revision>`
- qvq_commit: `<40-char SHA or N/A>`
- zml_commit: `<40-char SHA or N/A>`

## Exact CLI

```bash
<complete command, with no omitted flags>
```

## Effective configuration

```yaml
<complete effective quantization/conversion configuration>
```

Config files:

- `/absolute/path/to/config`: `<copied/hash>`

## Datasets

| Role | Absolute path or remote ID/revision | Split/rows/order | SHA-256 |
|---|---|---|---|
| calibration | ... | ... | ... |

## Environment and hardware

`<full versions, GPU identity, driver, compiler, and build flags>`

## Outputs and hashes

| Relative file | Bytes | SHA-256 |
|---|---:|---|
| ... | ... | ... |

## Post-quant evaluations

- `/absolute/path/to/evaluations/<task>__<run_id>__<arm_id>.md`
- `/absolute/path/to/evaluations/<task>__<run_id>__<arm_id>.json`
- `/absolute/path/to/logs/quantization.log`
- `/absolute/path/to/evaluations/logs/evaluation.log`
```

Use the same required fields for each evaluation record and include its full
raw result rather than relying on this model-level summary.
