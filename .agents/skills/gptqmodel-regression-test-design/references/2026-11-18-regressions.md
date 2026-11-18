<!-- SPDX-FileCopyrightText: 2026 ModelCloud.ai
SPDX-License-Identifier: Apache-2.0 -->

# Follow-ups that escaped the first regression

Window: 2026-11-16 01:01 to 2026-11-18 01:01 UTC, ending at
[`d6b4934`](https://github.com/ModelCloud/GPTQModel/commit/d6b4934ea689d65e58df14aedeb1d7c6ba226002).
477 PRs merged into main, not 477 distinct bugs. These are selected follow-ups
from that inventory, checked against the merged changes and their tests. The
window overlaps the [previous audit](2026-11-17-regressions.md); the counts
are not additive. No historical hardware failures were rerun for this audit.

| Landed changes | What the earlier check could miss | Existing test to extend |
| --- | --- | --- |
| AWQ [#4996](https://github.com/ModelCloud/GPTQModel/pull/4996) → [#5030](https://github.com/ModelCloud/GPTQModel/pull/5030) | Packing one token ID per hidden row fixed the length mismatch, but indexing the padded grid by kept-row number still paired equal-length arrays incorrectly. Padding or excluding a sink must change at least one expected ID. | [Kept-row replay](../../../../tests/test_awq_hash_moe_kept_rows_5015.py) |
| Placement [#4915](https://github.com/ModelCloud/GPTQModel/pull/4915) → [#4983](https://github.com/ModelCloud/GPTQModel/pull/4983) | An integer ordinal avoided creating an unwanted GPU context, but a downstream truthiness check treated zero as missing. A nonzero ordinal passed; forward could silently copy CPU-resident packed weights. | [Packed-weight placement](../../../../tests/test_fp8_device_map_int_zero_4964.py) |
| Native load [#4890](https://github.com/ModelCloud/GPTQModel/pull/4890) and bridge [#5046](https://github.com/ModelCloud/GPTQModel/pull/5046) | Sibling paths, not a revert: native offload dispatch worked, but a Transformers load of the same map could choose a kernel that could not initialize on the offloaded layers. Test both public consumers, not only the map parser. | [Bridge/native parity](../../../../tests/test_hf_bridge_cpu_offload_5022.py) |
| Bridge guard [#5046](https://github.com/ModelCloud/GPTQModel/pull/5046) → [#5051](https://github.com/ModelCloud/GPTQModel/pull/5051) → [#5052](https://github.com/ModelCloud/GPTQModel/pull/5052) | An optional Optimum import at fixture setup blocked CPU assertions; changing only the bridge test didn't trigger its workflow. Fixing a test and arranging for it to execute are separate obligations. | [CPU job triggers and node selection](../../../../.github/workflows/awq_cpu.yml) |
| Run exit [#5031](https://github.com/ModelCloud/GPTQModel/pull/5031), [#5039](https://github.com/ModelCloud/GPTQModel/pull/5039), [#5049](https://github.com/ModelCloud/GPTQModel/pull/5049) | Different holes in one caller guarantee: post-forward stop left hooks, a restore-tail failure escaped recovery, and the weight-only path missed the in-flight layer's return home. Success or a single looper's abort test could not cover the other exits. | [Grad-mode forward](../../../../tests/test_failed_run_model_state_5009.py), [tail fault](../../../../tests/test_post_loop_tail_failure_5023.py), [weight-only placement](../../../../tests/test_weight_only_failed_run_device_5011.py) |

**After the audit:** [#5066](https://github.com/ModelCloud/GPTQModel/pull/5066)
implemented the [run-exit proposal](https://github.com/ModelCloud/GPTQModel/issues/5060).
Both loopers now share one recovery owner; extend the
[shared conformance suite](../../../../tests/test_run_exit_recovery_contract_5060.py)
when the [caller guarantee](../../../../docs/cancelled-run-contract.md) changes.
