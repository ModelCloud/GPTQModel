<!-- SPDX-FileCopyrightText: 2026 ModelCloud.ai
SPDX-License-Identifier: Apache-2.0 -->

# Regression fixtures from the Nov 19–21 audit

Window: 2026-11-19 01:00 to 2026-11-21 01:00 UTC, ending at
[audited main](https://github.com/ModelCloud/GPTQModel/commit/9c22da019d93d247ccd49297bee936b91d55d91b).
These are landed fixes and tests to extend, not reports of current failures.

| Fix chain | What a passing test missed | Discriminating fixture and control |
| --- | --- | --- |
| Dead columns: [GPTQ/QQQ](https://github.com/ModelCloud/GPTQModel/pull/5664) → [GPTAQ/FOEM](https://github.com/ModelCloud/GPTQModel/pull/5766) | A base-class test never ran the subclasses’ grid fits. The discarded weight could set every live column’s scale even though the dead column ended at zero. | Calibrate a 32-column projection on identity rows 1–31. Change only column 0 from 0 to +128 or −128; compare scales, zeros and live weights within each solver. For the zero-correction GPTAQ/FOEM case also compare to GPTQ. [Family suite](../../../../tests/test_dead_hessian_column_5611.py) |
| Integer export: [GPTQ/RTN](https://github.com/ModelCloud/GPTQModel/pull/5794) → [GPTAQ/FOEM](https://github.com/ModelCloud/GPTQModel/pull/5872) | Matching the replay weight does not establish what the downstream packer wrote. | Use correlated inputs and 8-bit asymmetric BF16 weights. Prove the FP64 reference code is 207 but replay alone recovers 206; extract the byte from the packed word after finalization. A fresh FP16 control recovers 207 both ways. [Base](../../../../tests/test_gptq_bf16_pack_codes_5721.py) · [variant fixture with corrections](../../../../tests/test_gptaq_foem_bf16_pack_codes_5819.py) |
| GSQ candidate budget: [row decisions](https://github.com/ModelCloud/GPTQModel/pull/5702) and [score aggregation](https://github.com/ModelCloud/GPTQModel/pull/5703) | Homogeneous rows hid a scale trial gated by its neighbor; even a correct export could report the wrong score when a chunk had zero teacher energy. | With two columns and four FP32 candidates, compare 32 and 64 bytes (one vs two rows). Pair a scale-only row with a code-changing row; require scales 1.25 and 1.125. Separately isolate a zero-energy row and compute the objective from **whole-layer** FP64 sums (baseline 5; all-zero teacher control 2). [Scales](../../../../tests/test_gsq_scale_chunk_budget_5692.py) · [scores](../../../../tests/test_gsq_baseline_chunk_energy_5693.py) |
| MACaQ wide decode: [autocast guard](https://github.com/ModelCloud/GPTQModel/pull/5895) → [removed in dtype-lock change](https://github.com/ModelCloud/GPTQModel/pull/5897) → [restored](https://github.com/ModelCloud/GPTQModel/pull/5938) | The guard **and its tests** disappeared together. Fixing the public input dtype did not constrain the matmul inside autocast. | Use the same FP16 input and module with autocast off/on: a decoded weight of 65536 times 0.25 must yield 16384. Check the underflow case too, and that the caller’s autocast remains enabled. When retiring a dtype-switch test, retain any valid same-dtype arithmetic case. [MACaQ fixture](../../../../tests/test_macaq_weight_narrow_5631.py) |

**Why propose a design change only for the solver handoff?** Each solver
still owns the order of grid preparation and must remember to provide an exact
pack source when replay loses codes. [Design issue #5960](https://github.com/ModelCloud/GPTQModel/issues/5960)
proposes shared boundaries around the method-specific update. The GSQ and
autocast examples instead have defined behavior that small fixtures can check.
