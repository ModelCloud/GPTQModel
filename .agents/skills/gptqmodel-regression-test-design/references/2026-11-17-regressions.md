<!-- SPDX-FileCopyrightText: 2026 ModelCloud.ai
SPDX-License-Identifier: Apache-2.0 -->

# Evidence behind the regression-test design gate

Audit window: **2026-11-15 02:28 UTC through 2026-11-17 02:28 UTC**.
The main-branch commit inventory contains 279 commits (not 279 bugs), ending at
[`5c53ee1118`](https://github.com/ModelCloud/GPTQModel/commit/5c53ee1118b138da82222d8cb64fdb42a7d3c272).
Commit subjects were screened across the window; representative implementation
and test diffs were reviewed for the sequences below. Issue/PR references in a
single squash commit were not counted as separate regressions. This is a test
design audit, not a claim that every historical fix was reproduced on hardware.

## Demonstrated regressions or incomplete fixes

| Sequence | Observed mechanism and test gap | Required regression construction |
| --- | --- | --- |
| [#4409](https://github.com/ModelCloud/GPTQModel/pull/4409) -> [#4416](https://github.com/ModelCloud/GPTQModel/pull/4416) -> [#4498](https://github.com/ModelCloud/GPTQModel/pull/4498) | Cleanup after save removed offload files still owned by live meta modules. Keeping the directory fixed reuse; later loop-exit restoration removed live disk hooks. Checking only directory deletion or only save success missed the next consumer. | Save then forward/save again; failed save then retry; move/use the restored model; assert eventual cleanup only after the last owner releases storage. Do not freeze the intermediate keep-on-disk implementation as the contract. |
| [#4442](https://github.com/ModelCloud/GPTQModel/pull/4442) -> [#4492](https://github.com/ModelCloud/GPTQModel/pull/4492) | The original OOM regression failed the first W-shaped allocation and used a diagonal Hessian. Off-block error propagation was zero, hiding a retry that started from already mutated W. Static-group state also had to move with the fallback. | Correlated Hessian, multiple blocks, failure after completed updates, independent fresh-run oracle, dead columns/activation order/static groups; assert the fault fired and non-OOM errors propagate. |
| [#4436](https://github.com/ModelCloud/GPTQModel/pull/4436) -> [#4524](https://github.com/ModelCloud/GPTQModel/pull/4524) | QQQ's per-device scratch sharing reduced memory but aliased writable reduction storage across streams. Same-device reuse tests asserted sharing, not exclusive ownership during overlapping use. | Same-stream reuse plus distinct concurrent-stream storage/leases and actual overlapping native numerical checks. Allocation locking alone is insufficient. |
| [#4434](https://github.com/ModelCloud/GPTQModel/pull/4434) -> [#4441](https://github.com/ModelCloud/GPTQModel/pull/4441) | AWQ's schedule mode was fed into a plan slot; the forward fell back to the process policy instead of the module's setting. | Non-default mode/explicit plan through actual forward dispatch; change process policy after construction to expose accidental global reads. |
| [#4445](https://github.com/ModelCloud/GPTQModel/pull/4445) -> [#4576](https://github.com/ModelCloud/GPTQModel/pull/4576) | Auto eval-batch handling still re-resolved a per-task `auto` override on every recursive OOM retry. A correct chooser alone did not prove bounded retry behavior. | Inject repeated OOMs through the consumer; assert effective sizes shrink, retries terminate, non-OOM errors propagate, and the next top-level call starts fresh. |

## Recurring families, not proof of a single reverted fix

These fixes share an escaped invariant. Their proximity does **not** establish
that each later bug was introduced by the earlier PR.

| Family and representative fixes | Missing dimension to include in tests |
| --- | --- |
| Checkpoint layout: [#4235](https://github.com/ModelCloud/GPTQModel/pull/4235), [#4330](https://github.com/ModelCloud/GPTQModel/pull/4330), [#4414](https://github.com/ModelCloud/GPTQModel/pull/4414), [#4425](https://github.com/ModelCloud/GPTQModel/pull/4425) | Disk load versus `assign=True`, fresh-kernel state copies, repeated conversion, and live offload layout. Check words and forward output, not just format labels. |
| Storage dtype and shell coverage: [#4334](https://github.com/ModelCloud/GPTQModel/pull/4334), [#4417](https://github.com/ModelCloud/GPTQModel/pull/4417), [#4507](https://github.com/ModelCloud/GPTQModel/pull/4507), [#4570](https://github.com/ModelCloud/GPTQModel/pull/4570), [#4608](https://github.com/ModelCloud/GPTQModel/pull/4608) | Model dtype versus stored dtype; buffers versus Parameters; mixed dense/quantized manifests; real fused-expert shells. Re-save and reload must preserve every owned payload. |
| Mutable config and policy: [#4328](https://github.com/ModelCloud/GPTQModel/pull/4328), [#4407](https://github.com/ModelCloud/GPTQModel/pull/4407), [#4497](https://github.com/ModelCloud/GPTQModel/pull/4497), [#4553](https://github.com/ModelCloud/GPTQModel/pull/4553), [#4585](https://github.com/ModelCloud/GPTQModel/pull/4585), [#4606](https://github.com/ModelCloud/GPTQModel/pull/4606) | Reused config, non-default round trip, warm cache plus in-place typed-leaf mutation, old immutable snapshots, and public load-to-quantize inheritance. |
| Calibration/replay: [#4291](https://github.com/ModelCloud/GPTQModel/pull/4291), [#4293](https://github.com/ModelCloud/GPTQModel/pull/4293), [#4329](https://github.com/ModelCloud/GPTQModel/pull/4329), [#4332](https://github.com/ModelCloud/GPTQModel/pull/4332), [#4352](https://github.com/ModelCloud/GPTQModel/pull/4352), [#4526](https://github.com/ModelCloud/GPTQModel/pull/4526) | Accepting one iterator pass is not replayability; plain dict/list fixtures miss real Mapping/column types and VLM consumers; uniform masks and always-BOS fixtures hide semantic differences. |
| Failure ownership: [#4361](https://github.com/ModelCloud/GPTQModel/pull/4361), [#4515](https://github.com/ModelCloud/GPTQModel/pull/4515), [#4527](https://github.com/ModelCloud/GPTQModel/pull/4527), [#4580](https://github.com/ModelCloud/GPTQModel/pull/4580), [#4600](https://github.com/ModelCloud/GPTQModel/pull/4600) | Cold concurrent construction, cancel/error teardown, and failure after future publication; assert usability/retry and bounded waiter completion, not only success-path cleanup. |
| Precision and geometry: [#4264](https://github.com/ModelCloud/GPTQModel/pull/4264), [#4365](https://github.com/ModelCloud/GPTQModel/pull/4365), [#4533](https://github.com/ModelCloud/GPTQModel/pull/4533), [#4535](https://github.com/ModelCloud/GPTQModel/pull/4535), [#4539](https://github.com/ModelCloud/GPTQModel/pull/4539), [#4603](https://github.com/ModelCloud/GPTQModel/pull/4603), [#4605](https://github.com/ModelCloud/GPTQModel/pull/4605) | Fused versus portable arithmetic, outer autocast, cancellation before the final cast, and real strided/batched inputs rather than contiguous random matrices only. |

## Existing tests to extend

Paths refer to the audited main revision above. These are starting points, not
a claim that every stronger requirement in the skill is already implemented.

| Invariant | Test reference |
| --- | --- |
| Partial-progress OOM retry | `tests/test_gptq_post_inverse_fallback_4400.py::test_block_retry_restarts_from_the_pre_pass_weight` |
| Live offload ownership and loop-exit restoration | `tests/test_offload_run_cleanup_4368.py`; `tests/test_offload_quantized_modules_restored_4450.py` |
| Stream-exclusive scratch | `tests/test_qqq_kernel_memory_4387.py::test_qqq_reduce_scratch_is_exclusive_per_cuda_stream` (ownership evidence; its mocked GEMM companion is not native numerical qualification) |
| Future failure and same-key retry | `tests/test_triton_patch_api.py::test_patched_autotuner_pruning_exception_resolves_future` |
| Checkpoint layout and dtype | `tests/test_qzeros_format_load_4377.py`; `tests/test_qzeros_assign_load_4304.py`; `tests/test_resave_preserves_checkpoint_dtypes_4379.py` |
| MoE re-save retains experts | `tests/test_moe_resave_preserves_experts_4589.py::test_resave_quantized_olmoe_keeps_every_expert` |
| Warm typed-leaf cache and immutable snapshots | `tests/test_dynamic_config_invalidation_4475.py::test_mse_config_leaf_edit_invalidates_fresh_resolution` |
| Policy at the consumer | `tests/test_backend_config_reuse_4446.py`; `tests/test_kernel_policy_snapshot_4555.py`; `tests/kernels/test_awq_triton_default_schedule.py` |
| One-shot multi-pass calibration | `tests/test_generator_calibration_4307.py`; `tests/test_calibration_input_gaps_4311.py`; `tests/test_vlm_mapping_calibration_4308.py` |
| Wide arithmetic and stride boundaries | `tests/test_int8_bias_cancellation.py`; `tests/test_paroquant_single_rounding_4317.py`; `tests/kernels/test_marlin_strided_activation_4563.py` |
| Test-harness retry termination | `tests/test_model_test_eval_oom_retry_4536.py` |

Existing AWQ/model-plan and numerical skills already require native oracles,
relationship evidence, and target-device qualification. The new gate adds
cross-stage regression construction and explicit detection evidence; it does
not weaken those requirements or claim documentation prevents regressions by
itself.

## Structural follow-up

[Design issue #4615](https://github.com/ModelCloud/GPTQModel/issues/4615) proposes
explicit persistence, attempt/commit, asynchronous workspace, and resolved-policy
contracts where test coverage alone leaves responsibility spread across consumers.
It distinguishes existing fixes from architectural proposals and related open
reports; this documentation change does not implement those designs.
