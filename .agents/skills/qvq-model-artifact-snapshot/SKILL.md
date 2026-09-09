---
name: qvq-model-artifact-snapshot
description: Persist QvQ quantization and conversion experiment snapshots under /monster/data/model/qvq with complete model outputs, exact calibration inputs and derived calibration artifacts, and reproducibility metadata. Use after any QvQ model-affecting experiment or when publishing a quantized model snapshot; do not use for source-only changes.
---

# QvQ model artifact snapshots

Only full-model quantizations and full-model dynamic/mixed-rate quantizations in this repository may leave a complete,
portable snapshot under `/monster/data/model/qvq`. A result report, a single tensor shard, or a pointer to a
calibration file is not a model snapshot.

Do not use this path for single-layer or partial-layer quantization, selected-module experiments, kernel-only outputs,
partial checkpoints, tensor probes, calibration-only runs, or other piece-meal artifacts. Keep those outputs in their
experiment workspace or the repository's existing results area. Before copying, verify that the output contains the
complete model tensor set and loader metadata; if it does not, stop and do not create a snapshot directory.

## Snapshot layout and naming

Use one unique, directly addressable directory per experiment snapshot below `/monster/data/model/qvq`. Do not rely on
opaque run numbers when this directory may contain thousands of snapshots. Use this fixed, sortable naming pattern:

```text
<repo>__<model>__<experiment>__<formats>__<calibration>__seed<seed>__<YYYYMMDD>__commit<git12>__<content12>
```

For example:

```text
/monster/data/model/qvq/modelcloud-qvq__llama-3.2-1b-instruct__f6-p32__qvq-p32-gguf-exl3__yaqa125x__seed7__20260904__commitabc123def456__88b3f5bbcc4
├── qvq-p32/                  # native QvQ checkpoint, when produced
├── gguf/                     # GGUF output, when produced
├── exl3/                     # EXL3 output, when produced
├── calibration/
│   ├── source/               # exact source datasets used by the run
│   └── derived/              # rendered text, imatrix, packed tensors, etc.
├── metadata/                 # copied run configs, recipes, plans, and reports
├── why.md                    # why this full-model snapshot was made
└── snapshot_manifest.json    # source paths, destination paths, sizes, and SHA-256 hashes
```

Use lowercase ASCII and hyphens within fields; use `__` only as the field separator. Normalize the repository remote,
model IDs, and experiment labels by replacing slashes and spaces with hyphens. The `repo` field is the normalized
repository name (for this repository, `modelcloud-qvq`); the `experiment` field should include the quantizer/rate and
packing family when material (for example `f6-p32`); `formats` is a sorted hyphen-separated list of every output
format in the snapshot; `calibration` is the short stable dataset/plan identifier; `seedna` is required when no seed
exists; the date is UTC; `git12` is the first 12 lowercase hexadecimal characters of the source repository commit;
and `content12` is the first 12 lowercase hexadecimal characters of a hash over the final model/config manifests.
Record the full remote URL, branch, and commit in `snapshot_manifest.json`. Never use spaces, timestamps without a
date, random temporary names, or an uninformative name such as `run1`.

Keep only the format directories that the experiment actually produced. If a local deployment convention requires
model grouping, the same fully informative snapshot name must still be used as the leaf directory. The snapshot must
remain self-describing and must stay below `/monster/data/model/qvq`.

## Lookup routing

When a request asks to find, reuse, compare, or load an existing quantized-model snapshot, search
`/monster/data/model/qvq` first. Inspect the informative snapshot directory names, then read `why.md` and
`snapshot_manifest.json` to identify the model, repository commit, formats, calibration data, and completeness. Search
`/root/qvq-results`, experiment workspaces, caches, or other locations only after the published snapshot path has been
checked or when the requested snapshot is not present there. Treat an artifact found only in a work directory as an
unpublished experiment output, not as the canonical model snapshot.

## Required contents

- Copy the complete final quantized output for every format produced. Include every tensor shard, shard index, config,
  quantization config, tokenizer, special-token files, chat template, and generation config needed to load it.
- A full-model dynamic or mixed-rate quantization qualifies even when different modules use different bit rates. The
  deciding criterion is complete model coverage, not uniform bits.
- For native QvQ output, retain the full checkpoint in its declared format (for example `qvq_v2b2_p32`), not only
  evaluation JSON or selected layers.
- For GGUF, retain the final `.gguf` and any required imatrix or conversion metadata. For EXL3, retain the complete
  loadable checkpoint directory, including all shards and its EXL3 quantization metadata.
- Copy every exact calibration source used by the run into `calibration/source/`, even when it lives outside this
  checkout. Copy derived calibration artifacts used by a backend into `calibration/derived/<backend>/` (for example
  packed EXL3 calibration tensors, rendered GGUF text, imatrix GGUF, recipes, or tensor plans).
- Copy the run metadata that proves what was used: `qvq_quantize_run.json`, `quantize_config.json`, `args.json`,
  `manifest.json`, backend reports, recipes/plans, and relevant quantization logs. Preserve the original path in the
  snapshot manifest rather than rewriting provenance away.
- Write a human-readable `why.md` at the snapshot root. State whether the snapshot was created for a named experiment
  or an explicit user request, the concrete goal or test it supports, the model/format/rate/seed scope, and the date.
  Do not write a generic explanation such as “saved for backup”; future agents must be able to tell why this full model
  was worth retaining and which comparison or deployment question it answers.
- Write `model_run.md` at the stored artifact root by following
  [quantized-model-provenance](../quantized-model-provenance/SKILL.md). It must contain the absolute stored path,
  `run_id`, `arm_id`, the full CLI, complete effective quantization configuration, every calibration/evaluation
  dataset path and revision, the full relevant QVQ and ZML commit SHAs, environment/hardware, output hashes, and
  links to all evaluation records. Store the complete quantization stdout/stderr log with the artifact and record
  its path and hash. A snapshot without this record and log is incomplete, even if all model shards are present.
- Record the calibration dataset split, row selection, row count, model/tokenizer binding when available, source size,
  and SHA-256. If a run used more than one calibration source, record each one separately.

Do not copy unrelated evaluation corpora merely because they are nearby. Evaluation data may be copied as metadata or
as a separate evaluation bundle only when it is needed to reproduce the reported result. Do not publish temporary
working state such as optimizer caches or resumable quantizer checkpoints unless the user explicitly asks for a
resumable experiment snapshot; label that state clearly when it is included.

## Locate provenance before copying

Trace calibration and output provenance from the run itself; do not infer it from a directory name. Inspect the nearest
available `qvq_quantize_run.json`, `quantize_config.json`, `args.json`, backend `report.json`, input `manifest.json`,
recipe/plan, and quantization log. Resolve every referenced local path and confirm that it exists. For QvQ experiments,
check the declared format, seed, calibration bindings, selected rows, and source-weight configuration. For GGUF/EXL3,
also check the backend-specific input corpus and derived calibration file.

If an exact calibration source cannot be found or its hash does not match the run metadata, stop and report the missing
provenance. Do not mark an artifact as a complete snapshot with an unverified or substituted dataset.

## Safe copy procedure

1. Inspect the active experiment outputs and repository status. Preserve unrelated tracked and untracked files.
2. Check available space with `df -h /monster` and estimate every source with `du -sh`. Account for source datasets and
   derived calibration files in addition to model tensors.
3. Create a unique staging directory ending in `.partial` below the final snapshot directory's parent. Never use
   `--delete` and never overwrite a different experiment under an existing snapshot name.
4. Copy with a metadata-preserving, resumable operation such as `rsync -a --partial --info=progress2`; use `cp -a` only
   when `rsync` is unavailable. Copy real files, not absolute-path symlinks, so the snapshot is portable.
5. Generate `snapshot_manifest.json` with relative destination paths, original source paths, byte sizes, and SHA-256
   hashes. Include the experiment identity, model identity, formats, seed/rate, calibration bindings, creation time,
   source repository remote, branch, and full commit.
6. Write and review `why.md` before publishing the snapshot.
7. Validate that all expected shards and loader metadata are present, compare file counts and sizes, and hash the
   manifest plus critical model and calibration files. A completed model directory must load from the snapshot path,
   not from the original source path.
8. Only after validation, rename the `.partial` staging directory to its final name. Leave an incomplete copy marked
   `.partial` and report the error if the copy or validation fails.

For repeated runs, prefer a deterministic experiment identifier plus a content hash or timestamp suffix. If a snapshot
already exists, compare its manifest before reusing it; do not silently replace a model or its calibration dataset.

## Post-quant evaluation records

Every post-quant evaluation associated with a snapshot must have a unique
Markdown record and raw per-sample result beside the artifact. The record must
contain the run/arm IDs, absolute model/result paths, full copy-pasteable CLI,
effective configuration, exact dataset paths/revisions and prompt settings,
full QVQ/ZML SHAs, dependency versions, all metrics and counts, timings, raw
result hash, and the complete evaluation stdout/stderr log with path and hash.
Keep failed and partial logs. Use the
[quantized-model-provenance](../quantized-model-provenance/SKILL.md) template;
do not treat a score-only JSON or console excerpt as complete evidence.

## Completion report

Report the final snapshot path, formats and model sizes, the `why.md` rationale, calibration sources and derived
artifacts included, manifest hash, validation performed, and any intentionally excluded temporary state. If the
operation was blocked by missing files, insufficient space, or provenance mismatch, report that explicitly and do not
claim completion.
