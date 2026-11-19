# LocateAnything clean-room scope

Port directory: `gptqmodel/models/ports/locateanything/`.
Definition: `gptqmodel/models/definitions/locateanything.py`.

## Governing license evidence

- `ports/locateanything/LICENSE`: the full **NVIDIA License** of the pinned source
  (HF `nvidia/LocateAnything-3B`, revision `c32291ca5e996f5a7a485845b4f57a233936bba0`;
  project `NVlabs/Eagle/Embodied`). Section 3.3 limits the Work **and any derivative works** to
  non-commercial (research/evaluation) use. Section 3.2 says the license keeps applying to
  derivatives whatever other terms are attached to them.
- `ports/locateanything/NOTICE`: says the repository license applies to every file adapted from the
  source repository, and that the per-file MIT and Apache headers **do not override** it. It also
  lists which upstream header each file kept.
- Four files carry the NVIDIA proprietary-rights notice ("any use ... without an express license
  agreement from NVIDIA CORPORATION is strictly prohibited"): `modeling_vit.py`,
  `image_processing_locateanything.py`, `generate_utils.py` and `mask_sdpa_utils.py`.

Rule applied: any file that came from the NVIDIA source repository, or is an adaptation of one, is
**restricted**, whatever header it carries. Only code written by ModelCloud with no upstream
lineage, or code under a separate Apache-2.0 lineage that the port only imports, counts as
permissive.

## Per-file classification (port directory)

| File | Header found | Origin | Class | Action |
|---|---|---|---|---|
| `configuration_locateanything.py` | InternVL / OpenGVLab MIT + "NVIDIA LICENSE applies in addition" | NVIDIA repo config (adapted). `MoonViTConfig` and `LocateAnythingConfig` are both here. | **RESTRICTED** | Replace (config field names and defaults are interface facts; see SPEC §2) |
| `modeling_locateanything.py` | "NVIDIA, Copyright (c) 2025 NVIDIA, MIT" | NVIDIA repo top-level model and localization `generate` loop | **RESTRICTED** | Replace whole file |
| `modeling_qwen2.py` | HF/Qwen Apache-2.0 + "NVIDIA LICENSE also applies; see NOTICE" | Adapted from the NVIDIA repo's block-diffusion Qwen2 file. The ModelCloud rewrite sits on the vendored native Qwen2/Qwen3 classes, but the model-specific logic (image injection, block-diffusion mask selection, cache policy, `prepare_inputs_for_generation`) is upstream-derived | **RESTRICTED** | Replace whole file. The imported `ports/native/qwen2` and `ports/native/qwen3` stay (Apache) |
| `modeling_vit.py` | NVIDIA proprietary notice | NVIDIA's MoonViT copy (patch embed, 2D RoPE, packed attention, patch merger) | **RESTRICTED** | Replace whole file. Note: it contains a second, unused `MoonViTConfig`. The loader uses the one in the configuration module |
| `image_processing_locateanything.py` | NVIDIA proprietary notice | NVIDIA | **RESTRICTED** | Replace whole file |
| `generate_utils.py` | NVIDIA proprietary notice | NVIDIA (token-id table, logit processors, box/ref decoding, pattern handling) | **RESTRICTED** | Replace whole file. `decode_bbox_avg` and `handle_pattern` are imported by tests, so their names and signatures are interface |
| `mask_sdpa_utils.py` | NVIDIA proprietary notice | NVIDIA (block-diffusion visibility builders) | **RESTRICTED** | Replace whole file |
| `processing_locateanything.py` | HF Apache-2.0 (2024) | NVIDIA repo processor (adapted). ModelCloud added the bounded remote fetch wiring, pinned `from_pretrained` and `save_pretrained`, but these sit inside an upstream-derived file | **RESTRICTED** | Replace whole file. The helpers it calls in `gptqmodel/utils/image.py` and `gptqmodel/utils/hf.py` are GPT-QModel Apache code and stay |
| `__init__.py` | "Copyright 2026 ModelCloud.ai; derivatives subject to NVIDIA LICENSE" | ModelCloud. A docstring only, no upstream code | Permissive content (ModelCloud) | Keep. Update the license line once the replacement lands |
| `.gitattributes` | none | ModelCloud. It only protects LICENSE whitespace | Permissive | Remove together with LICENSE |
| `LICENSE` | NVIDIA License text | NVIDIA | Legal text | Keep **until** the clean-room code replaces every restricted file, then remove |
| `NOTICE` | ModelCloud attribution and handoff notes | ModelCloud, describing NVIDIA terms | Legal text | Same as LICENSE: remove or rewrite after replacement |

Summary: 8 of the 8 Python source modules other than `__init__.py` are restricted. Nothing in the
port directory holds model logic that can be kept.

## Outside the port directory (permissive, keep)

| Path | Evidence | Notes |
|---|---|---|
| `gptqmodel/models/definitions/locateanything.py` | `SPDX-License-Identifier: Apache-2.0`, ModelCloud | Keep. It refers to the port only by dotted class-path strings and attribute names (SPEC §9). |
| `gptqmodel/models/ports/native/qwen2/*`, `native/qwen3/*` | Apache-2.0 vendored Transformers | Building blocks for the decoder |
| `tests/test_locateanything_local_port.py`, `tests/test_locateanything_support.py`, `tests/module_tree/test_locateanything_local_port.py`, `tests/models/test_locateanything.py` | `SPDX ... Apache-2.0` (ModelCloud). The support test has no header but is ModelCloud-authored | Keep. These are the acceptance criteria (SPEC §10). |
| `tests/fixtures/ports/locateanything/checkpoint_config.json` | Config of the released checkpoint | Factual interface metadata (field names and values) used for the 770-tensor inventory test. Flag for legal review: it is copied from the NVIDIA HF repo. |
| `gptqmodel/utils/image.py`, `gptqmodel/utils/hf.py` | GPT-QModel Apache | Provide the bounded URL fetch and tokenizer loading the processor uses |
| docs rows (`docs/transformers5-model-ports.md` line 142, `docs/supported-models.md`, README row) | ModelCloud | Update the license column after replacement |

## Replacement surface (what phase 2 must provide)

New files under `gptqmodel/models/ports/locateanything/` that expose, at minimum:

- `configuration_locateanything.LocateAnythingConfig`, `configuration_locateanything.MoonViTConfig`
- `modeling_locateanything.LocateAnythingForConditionalGeneration` (with `LocateAnythingPreTrainedModel`)
- `processing_locateanything.LocateAnythingProcessor`, plus the module-level names tests patch:
  `load_image_from_url`, `load_bytes_from_url`, `VIDEO_READER_BACKENDS`, `REMOTE_MEDIA_MAX_BYTES`,
  `fetch_image`, `fetch_video`
- `image_processing_locateanything.LocateAnythingImageProcessor`
- `generate_utils.decode_bbox_avg`, `generate_utils.handle_pattern`

Internal module layout beyond these names is free.
