<!-- SPDX-FileCopyrightText: 2026 ModelCloud.ai
SPDX-License-Identifier: Apache-2.0 -->

# Regression fixtures from the Nov 21–23 audit

29 PRs merged to main between 2026-11-21 01:00 and 2026-11-23 01:00 UTC.
The examples below trace landed fixes. [Audited revision](https://github.com/ModelCloud/GPTQModel/commit/f4cc45255c3ef89171ce38eb55b3570ec6c491bb)

| Fix history | Fixture that distinguishes the failure | Family to extend |
| --- | --- | --- |
| Measurement coverage and pooling ([#5989](https://github.com/ModelCloud/GPTQModel/pull/5989), [#5992](https://github.com/ModelCloud/GPTQModel/pull/5992)) | Enumerate shifted targets across window seams. For batching, use rows of lengths 2, 5 and 5 with a single top-1 disagreement on the short row. Batch sizes 1, 2 and 3 must each yield nine positions and top-1 overlap 8/9. A constant loss or equal-length batch hides incorrect weighting. | [Window coverage](../../../../tests/test_measure_windows.py) · [sample counts and pooled results](../../../../tests/test_measure_model.py) |
| Shared logit capture ([#5994](https://github.com/ModelCloud/GPTQModel/pull/5994)) | Swap logits 16 and 16 + 1/256 between FP32 candidate and reference. Public top-1 overlap must be 0 and RMSE 1/256. Identical finite BF16 logits containing 65536 must give zero KLD and RMSE. Testing the metric on tensors supplied directly bypasses the lossy capture. | [Model API tests](../../../../tests/test_measure_model.py) |
| Inkling expert representations ([#5971](https://github.com/ModelCloud/GPTQModel/pull/5971), [#5973](https://github.com/ModelCloud/GPTQModel/pull/5973)) | Seed packed rows as interleaved gate/up pairs and dense rows in their released layout. Use a BF16 input and FP32 routing weight 0.4731. Compare packed, dense and defused execution with the routed activation rounded at the checkpoint model's boundary. A split and join can restore the original bytes even when both read the wrong rows. | [Packed row markers](../../../../tests/test_inkling2_loop_packed_requant.py) · [dense route after Defuser](../../../../tests/module_tree/test_inkling2_loop_plan.py) |
| Inkling vision order ([#5974](https://github.com/ModelCloud/GPTQModel/pull/5974), [#5976](https://github.com/ModelCloud/GPTQModel/pull/5976)) | Compare packed attention to separate per-image SDPA with explicit token/head axes. Zero the pixel projection and label the learned position table. On a 2×4 grid the two 2×2 merge blocks consume indices 0,1,4,5 then 2,3,6,7. A single block hides the permutation. | [Vision attention and positions](../../../../tests/test_inkling2_loop_vision.py) |
| CPU packing under inference mode ([original fix #5833](https://github.com/ModelCloud/GPTQModel/pull/5833), [packer split #5919](https://github.com/ModelCloud/GPTQModel/pull/5919), [repair #5990](https://github.com/ModelCloud/GPTQModel/pull/5990)) | The ambient-mode buffer check already existed when the original packer lost its override. Re-run both block and original packers on refactors. Leave inference mode, copy different packed values into the destination with `assign=False` and inspect the stored words. | [Mutable packed buffers](../../../../tests/test_packed_buffer_state_dict_5814.py) |

## AWQ fold ownership

LFM3 attention and convolution layers permit different norm folds. PanGu's
final query reads position embeddings, and two generic Inkling attention folds
change the layer output. The merged fixes filter groups by the actual forward.
Pin the retained edges as well as the unsafe controls so blanket suppression
also fails the test. [LFM3 fix #5978](https://github.com/ModelCloud/GPTQModel/pull/5978) · [Variant fixes #5980](https://github.com/ModelCloud/GPTQModel/pull/5980) · [PanGu](../../../../tests/test_pangu_alpha_local_port.py) · [Inkling](../../../../tests/module_tree/test_inkling2_loop_plan.py)

Tree order and compatible widths can suggest a source on the wrong branch.
The design proposal gives each variant ownership of its qualified fold edges
and reuses the existing search and replay nodes. [Proposal #5995](https://github.com/ModelCloud/GPTQModel/issues/5995)
