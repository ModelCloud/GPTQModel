<!-- SPDX-FileCopyrightText: 2026 ModelCloud.ai
SPDX-License-Identifier: Apache-2.0 -->

# Regression fixtures from the Nov 20 audit

283 PRs merged to main from Nov 18 01:00 to Nov 20 01:00 UTC (2026).
This overlaps the [Nov 19 audit](2026-11-19-regressions.md); it is not a
count of distinct bugs. These four families had separate landed fixes for
the same assumption.

| Merged fixes | What a plausible passing test missed | Existing CPU fixtures to extend |
| --- | --- | --- |
| Checkpoint selection: [wrapper](https://github.com/ModelCloud/GPTQModel/pull/5461), [TT block-float](https://github.com/ModelCloud/GPTQModel/pull/5476), [legacy sidecars](https://github.com/ModelCloud/GPTQModel/pull/5494) | Correct weight lookup does not prove that the earlier format probe, processor, or later re-save reads the same directory. Give root and child distinct markers. For a Hub lookup, distinguish the pinned revision from the default; on an offline cache miss forbid remote listing. | [Root vs. child](../../../../tests/test_checkpoint_subfolder_resolution.py), [sidecar presence/absence](../../../../tests/test_legacy_sidecar_selection_5379.py), [public re-save](../../../../tests/test_nested_checkpoint_resave_5380.py) |
| Resumed export: [identity](https://github.com/ModelCloud/GPTQModel/pull/5481) → [layout cleanup](https://github.com/ModelCloud/GPTQModel/pull/5540) | Rejecting the old manifest and writing the correct new shards was not enough: an old single-file checkpoint could win discovery over the new index. Opening only files named by the new manifest cannot catch that. | [Source/dtype changes and compatible resume](../../../../tests/test_dequant_resume_identity_5396.py), [both layout directions and unrelated files](../../../../tests/test_dequant_resume_layout_5502.py) |
| Calibration masks: [columnar mapping](https://github.com/ModelCloud/GPTQModel/pull/5486) → [HF Dataset](https://github.com/ModelCloud/GPTQModel/pull/5523) | Identical columns can take different iterators. Looking at one all-zero float row after splitting loses the evidence in its sibling that these are 0/1 keep masks, not additive masks. | [Container parity and additive controls](../../../../tests/test_calibration_padding_mask.py) |
| LoRA scale folding: [storage range](https://github.com/ModelCloud/GPTQModel/pull/5431) → [compute range](https://github.com/ModelCloud/GPTQModel/pull/5524) | With rank one, A=1, B=32768 and scale 4, the folded B fits FP32 but not FP16. For FP16 input 0.125 the unfused result is exactly 16384. Matching storage and activation dtypes never exercises the later narrowing. | [Mixed-dtype overflow, underflow and safe-fold controls](../../../../tests/test_adapter_scaling.py) |

**Detection evidence.** On the parent of the HF Dataset fix
([`3294d931`](https://github.com/ModelCloud/GPTQModel/commit/3294d931b)),
`test_float_0_1_mask_container_parity[dataset]` kept an extra row:
`[4, 5, 6]` with mask `[1, 1, 1]`. The same CPU test on the
[audited main](https://github.com/ModelCloud/GPTQModel/commit/1b2aa37208cdb477fed48b688ea2da5b4b6fbc23)
kept only `[1, 2]`. The other fix chains are evidence from merged changes,
not claims of current breakage.

**Design follow-up:** a decoy catches one consumer choosing the wrong root;
it does not remove the need for every new consumer to choose. The
[checkpoint selection proposal](https://github.com/ModelCloud/GPTQModel/issues/5586)
would give detection, loading and re-save two views of the same selection:
Hub arguments or a resolved local directory. It is not a shipped API.
