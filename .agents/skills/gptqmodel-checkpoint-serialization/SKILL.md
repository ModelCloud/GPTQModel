---
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
name: gptqmodel-checkpoint-serialization
description: Choose the on-disk container for converted or quantized model checkpoints in GPT-QModel. Use when writing a new checkpoint writer, converter, or export path; when adding a payload file to a published artifact; when changing how existing weights, scales, zero points, codebooks, or embeddings are stored; or when reviewing a save path that calls torch.save, pickle, or writes .pt/.bin/.pth files.
---

# GPT-QModel checkpoint serialization

## At a glance

Publish **all tensor payloads as safetensors** unless the user explicitly asks
for another container. This covers converters, exports, single files, sharded
models and Hub experiments: packed weights and metadata (scales, zeros,
codebooks, exponents, group indices), embeddings, lm-head, biases, norms and
buffers. A mixed-format model still requires two readers. Pickle can execute
code on load; safetensors supports lazy mapped reads and header-only inspection
for [resharding](../../../docs/reshard-source-selection.md).

Transient scratch consumed in one run (calibration replay, hidden states, GSQ
training payloads, RCO intermediates) is not a checkpoint. Do not convert it
incidentally; the rule applies if it becomes a published artifact.

## Write and validate

1. Use `safetensors.torch.save_file` with contiguous tensors; apply
   `.contiguous()` to views, including sliced, transposed or packed data.
   Store the dtype inference actually loads; do not rely on loader-side casts.
2. Write `format` and method dispatch fields (`bits`, `block`, `version`, etc.)
   as **strings in the payload metadata**, not just sidecar JSON. Merge
   caller metadata first so it cannot overwrite dispatch fields.
3. For shards use the Transformers `model.safetensors.index.json` weight map.
   Publish or resume via temporary path and atomic rename; partial writes
   must not appear complete.
4. Assert exact tensor equality after save/reload and separately test legacy
   reads. Check the published file list, README and checksum file name only
   the intended safetensors payloads. Do not add a `torch.save` checkpoint
   write.

Writer examples: `gptqmodel/models/writer.py`, `gptqmodel/utils/model_dequant.py`,
`gptqmodel/looper/input_cache.py`, and `save_blockfloat_checkpoint` /
`save_blockfloat_model` in `gptqmodel/utils/tt_blockfloat_quant.py`.

## Legacy reads and explicit overrides

- Route readers through one resolver: prefer safetensors, fall back to the
  legacy name, raise a clear error if neither exists. See
  `_load_blockfloat_payload` in `tt_blockfloat_quant.py`. Keep old
  `.pt`/`.bin`/`.pth` checkpoints readable; do not remove a reader without
  user direction or add a pickle writer for symmetry.
- **Only on explicit user request** add a named writer option defaulting to
  `format="safetensors"`, never an environment switch. Cite the request,
  reject unknown formats without fallback, let readers detect the payload,
  and document the alternative as non-recommended.
