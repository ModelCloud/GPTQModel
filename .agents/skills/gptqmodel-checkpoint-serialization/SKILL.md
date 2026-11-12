---
name: gptqmodel-checkpoint-serialization
description: Choose the on-disk container for converted or quantized model checkpoints in GPT-QModel. Use when writing a new checkpoint writer, converter, or export path; when adding a payload file to a published artifact; when changing how existing weights, scales, zero points, codebooks, or embeddings are stored; or when reviewing a save path that calls torch.save, pickle, or writes .pt/.bin/.pth files.
---

# GPT-QModel checkpoint serialization

## Rule

Every tensor payload that a converted or quantized checkpoint publishes is
written as **safetensors**, unless the user explicitly asked for a different
container.

This is the default, not a per-feature preference. A new writer does not need to
justify safetensors; a writer that emits anything else must point at the user
instruction that asked for it.

Applies to:

- Quantized weight payloads and their metadata (scales, zero points, codebooks,
  exponents, packed codes, group indices).
- Non-quantized tensors shipped alongside them: embeddings, lm-head, norms,
  biases, buffers.
- Any new quantization method, backend, or conversion tool that produces a
  loadable model directory or a single payload file.
- Payloads uploaded to a Hugging Face repository, including experiment
  artifacts.

Existing writers that already follow this: `gptqmodel/models/writer.py`,
`gptqmodel/utils/model_dequant.py`, `gptqmodel/looper/input_cache.py`, and the
Tenstorrent `save_blockfloat_checkpoint` / `save_blockfloat_model` pair in
`gptqmodel/utils/tt_blockfloat_quant.py`.

## What the rule does not cover

Transient scratch that is written and consumed inside a single run is out of
scope: calibration replay buffers, captured hidden states, GSQ training
payloads, RCO intermediates. Those are not checkpoints and are frequently
non-tensor objects, which safetensors cannot represent. Do not "convert" them
as a drive-by change; if one becomes a published artifact, it becomes a
checkpoint and this rule applies.

## Why

- `torch.save`/`torch.load` is pickle. Loading an untrusted checkpoint is
  arbitrary code execution, and every Hub artifact is untrusted input.
- safetensors is memory-mappable and lazily readable. Loaders can open one
  tensor without materialising the whole payload, and `safe_open` exposes a
  header for inspection.
- Resharding and shard selection read headers only
  (`gptqmodel/reshard.py`, `docs/reshard-source-selection.md`). A pickle
  payload cannot participate in that path at all.
- A mixed checkpoint (`blockfloat.pt` next to `embed_tokens.bf8.safetensors`)
  forces every consumer to implement two code paths for no benefit. One
  container, one reader.

## Implementing a writer

1. Use `safetensors.torch.save_file`. Contiguous tensors only; call
   `.contiguous()` on views and on anything produced by slicing, transposing, or
   packing.
2. Record the format in the file's `metadata` dict, not only in a sidecar
   JSON: at minimum `format`, plus the fields a loader must dispatch on (`bits`,
   `block`, `version`, or the equivalent for the method). safetensors requires
   string values.
3. Keep the dispatch fields authoritative. When merging caller-supplied
   metadata, write the format fields last so a caller cannot silently overwrite
   `bits` or `version` and corrupt the file.
4. Sharded checkpoints get a `model.safetensors.index.json` weight map, the
   same shape Transformers emits. Do not invent a second index schema.
5. Store a per-tensor dtype that matches what inference will load. Do not rely
   on the loader to upcast a payload that was written wide for convenience.
6. Write to a temporary path and rename into place when the payload is
   published or resumed from, so a partial write is never mistaken for a
   complete checkpoint.

## Keeping legacy checkpoints readable

Read old formats, never write them.

- Route every reader through one payload resolver rather than duplicating a
  `torch.load` fallback per call site. `_load_blockfloat_payload` in
  `tt_blockfloat_quant.py` is the reference: prefer safetensors, fall back to
  the legacy name, raise a single clear error if neither exists.
- Legacy `.pt`/`.bin` support is read-only and is not a reason to keep emitting
  them. Do not add a `format="pt"` writer option to preserve symmetry.
- Do not delete legacy support in the same change that adds the new writer
  unless the user asked for the removal; existing published checkpoints must
  keep loading.

## Explicit override

If a user asks for a specific container, honour it and nothing more:

1. Add one named option on the writer (for example `format="safetensors"` as
   the default). Do not add a `GPTQMODEL_*` environment variable; runtime
   behavior belongs in the config object.
2. Validate the value and fail closed on anything unrecognised. Do not fall
   back silently to pickle.
3. Keep the override on the writer only. Readers sniff the payload; they must
   not require the caller to restate the format.
4. Document the override where the writer is documented, and state that it is
   not the recommended container.

## Checklist before opening the pull request

- [ ] Every tensor file the change produces is `.safetensors`.
- [ ] No new `torch.save` call writes something a downstream loader consumes as
      a checkpoint.
- [ ] Each payload carries `format` and its dispatch fields in `metadata`.
- [ ] Readers go through one resolver and still load pre-change checkpoints.
- [ ] Round-trip test asserts tensor equality after save and reload, and covers
      the legacy-read path separately.
- [ ] A published artifact lists only safetensors payloads, and its README and
      checksum file name those files.
