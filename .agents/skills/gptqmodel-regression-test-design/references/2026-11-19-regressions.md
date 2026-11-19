<!-- SPDX-FileCopyrightText: 2026 ModelCloud.ai
SPDX-License-Identifier: Apache-2.0 -->

# Regression fixtures from the Nov 19 audit

428 PRs merged into main between 2026-11-17 01:01 and 2026-11-19 01:01 UTC.
The window overlaps the [Nov 18 audit](2026-11-18-regressions.md). These
examples trace selected follow-ups through the merged code and tests at
[`661f17a`](https://github.com/ModelCloud/GPTQModel/commit/661f17af629b7b1d7832021d40878fdf0125e062).

| Merged fixes | Why a passing test could miss the failure | Test family to extend |
| --- | --- | --- |
| Bounded AWQ capture [#5175](https://github.com/ModelCloud/GPTQModel/pull/5175), sampling [#5212](https://github.com/ModelCloud/GPTQModel/pull/5212), in-place merge [#5236](https://github.com/ModelCloud/GPTQModel/pull/5236), mixed capture [#5273](https://github.com/ModelCloud/GPTQModel/pull/5273) | A full buffer could overrepresent recent batches. Testing planned and routed batches on separate experts also missed a planned batch arriving after the *same* expert acquired a reservoir. Its appended rows vanished at replay. Tag each row with its batch and local index. Check exact survivors below capacity and representation across the stream at capacity. | [Sampling across quarters](../../../../tests/test_awq_token_rows_bounded_capture_5137.py), [mixed capture and local indices](../../../../tests/test_awq_reservoir_incremental_5233.py) |
| AWQ selection [#5282](https://github.com/ModelCloud/GPTQModel/pull/5282), [#5289](https://github.com/ModelCloud/GPTQModel/pull/5289), [#5315](https://github.com/ModelCloud/GPTQModel/pull/5315) | A mask with one true bit in eight slots must remove one row. Separately, with sinks eligible, 64 single-row quotas could select only first positions or only centres. Both extremes satisfied the budget. Under the audited policy, 64 batches of length 40 with budget 64 select eight first positions and 56 centres. | [Sink exclusion](../../../../tests/test_awq_sink_token_rows_5275.py), [single-row distribution](../../../../tests/test_awq_token_row_position_5265.py) |
| NF12 grad-mode forward [#5230](https://github.com/ModelCloud/GPTQModel/pull/5230), trace safety [#5309](https://github.com/ModelCloud/GPTQModel/pull/5309), fused dispatch [#5308](https://github.com/ModelCloud/GPTQModel/pull/5308) | The CPU backward exercised the decoded-weight path. The fused GPU op could return a tensor with a `grad_fn` and still leave `x.grad` unset. The eager inference-tensor probe also caused a graph break. Check the gradient value, the inference dispatch, and execution under the advertised compiler separately. | [Cold pack and backward](../../../../tests/test_nf12_grad_mode_forward_5211.py), [forward probe guard](../../../../tests/test_nf12_forward_no_inference_probe_5299.py), [GPU input gradient](../../../../tests/test_nf12_native_grad_5305.py) |
| Required payloads [#5239](https://github.com/ModelCloud/GPTQModel/pull/5239) and post-init [#5276](https://github.com/ModelCloud/GPTQModel/pull/5276) | A missing-key rejection exits before initialization. Given a complete checkpoint, the residency check still assumed every family had `qweight`. Exercise both sides of the existing per-kernel payload contract through the loader. | [Truncated payload](../../../../tests/test_fail_closed_checkpoint_5218.py), [resident and offloaded post-init](../../../../tests/test_post_init_payload_5261.py) |

The [AWQ capture ownership proposal](https://github.com/ModelCloud/GPTQModel/issues/5319)
would give each module one owner for retained rows and their source indices
across plan matches, reservoir merges and replay.
