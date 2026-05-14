# Failed Experiments

This file records Cannoe/Komodo optimization probes that were rejected or left
inconclusive. Keep metric evidence here because `/tmp` benchmark artifacts are
ephemeral and repeated failed probes waste NPU time.

For new entries, include:

- Date and kernel path.
- Exact change or environment knob tested.
- Benchmark artifacts or command shape.
- NPU devices used. Current policy for routine tests is NPU 0 and NPU 1 only.
- Speed, accuracy, and memory data when available.
- Decision and what would justify re-testing.

## 2026-05-14: AWQ BF16 Fuse Bias For All Group-32 Shapes

Status: failed accuracy gate, replaced by selective fusion.

Tested change: pass the FP32 fused bias through the BF16 native CANN path for
all Qwen3 27B AWQ group-size 32 projections.

Artifacts:

- `/tmp/cannoe_awq_qwen27_bf16_fused_bias_npu0.json`
- `/tmp/cannoe_awq_qwen27_bf16_fused_bias_npu1.json`
- `/tmp/cannoe_awq_qwen27_bf16_no_bias_fuse_control_npu0.json`
- `/tmp/cannoe_awq_qwen27_bf16_no_bias_fuse_control_npu1.json`
- `/tmp/cannoe_awq_qwen27_bf16_selective_bias_final_npu0.json`
- `/tmp/cannoe_awq_qwen27_bf16_selective_bias_final_npu1.json`

| Variant | NPU0 total ms | NPU1 total ms | max_abs | max_rel | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| No fused bias control | 1.301638 | 1.321499 | 8.0 | 0.0076336 | Accurate, slower |
| Fuse all group-32 bias | 1.159320 | 1.216283 | 16.0 | 0.1171875 | Reject |
| Selective bias fusion | 1.264040 | 1.274098 | 8.0 | 0.0077519 | Keep |

Reason: all-group32 bias fusion was faster but widened drift too much,
especially q/gate/up shapes. Keep the current selective rule that only fuses
narrow group-32 shapes and group-128 shapes.

Re-test only if the fused-bias accumulation order can keep `max_rel` under
about 0.01 and `max_abs` no worse than the no-fuse path, or if a user explicitly
requests a high-drift speed mode.

## 2026-05-14: GPTQ Group16 Disable Grouped Matmul

Status: failed speed and accuracy gates.

Tested change: disable the group16 grouped matmul path and fall back to the
per-group loop path.

Artifacts:

- `/tmp/cannoe_gptq_groups_grouped_off_probe_npu1.json`
- `/tmp/cannoe_gptq_groups_default_probe_npu0.json`
- `/tmp/cannoe_gptq_groups_tuning_rows_npu1.json`

| Variant | Device | Total ms | Suite speedup vs Torch | max_abs | max_rel |
| --- | --- | ---: | ---: | ---: | ---: |
| Default grouped path | npu:0 | 0.945393 | 5.6228x | 0.03125 | 34.71875 |
| Group16 tuning rows probe | npu:1 | 0.941582 | 5.6110x | 0.03125 | 34.71875 |
| Grouped-off loop path | npu:1 | 10.719219 | 0.4936x | 30.5703125 | 50469.5703125 |

Per-case evidence from the grouped-off run:

| Case | Path | Candidate ms | Speedup vs Torch | max_abs | max_rel |
| --- | --- | ---: | ---: | ---: | ---: |
| `gptq_gs16` | `native_int4_group16_loop` | 5.095682 | 0.1291x | 28.814453 | 12318.2568 |

Reason: group16 loop fallback is catastrophically slower and also produces bad
drift in this benchmark configuration. Do not disable grouped group16 as a
default optimization.

Re-test only if the loop implementation changes materially, for example a true
fused Ascend C group16 tile kernel that avoids grouped output materialization
and the Python-side reduction pattern.

## 2026-05-14: GPTQ Group16 Unfused Bias

Status: failed speed gate.

Tested change: avoid group16 bias fusion to see if the grouped path became
faster or more stable.

Artifacts:

- `/tmp/cannoe_gptq_groups_group16_unfused_bias_npu0.json`
- `/tmp/cannoe_gptq_groups_group16_unfused_bias_npu1.json`
- `/tmp/cannoe_gptq_groups_default_probe_npu0.json`
- `/tmp/cannoe_gptq_groups_tuning_rows_npu1.json`

| Variant | Device | Total ms | max_abs | max_rel | Decision |
| --- | --- | ---: | ---: | ---: | --- |
| Default grouped path | npu:0 | 0.945393 | 0.03125 | 34.71875 | Keep |
| Group16 unfused bias | npu:0 | 0.979822 | 0.03125 | 34.71875 | Reject |
| Group16 tuning rows probe | npu:1 | 0.941582 | 0.03125 | 34.71875 | Reference |
| Group16 unfused bias | npu:1 | 0.937853 | 0.03125 | 34.71875 | Inconclusive |

Reason: unfused bias did not produce a stable two-device win and gave a clear
slowdown on NPU0. Keep the current group16 bias behavior.

Re-test only with a same-run A/B harness that repeats both variants on NPU0 and
NPU1, or if the grouped matmul call structure changes.

## 2026-05-14: GPTQ Group16 `tuning_config=[rows]`

Status: inconclusive, not retained.

Tested change: pass a row-count tuning config into the group16 grouped matmul
call.

Artifacts:

- `/tmp/cannoe_gptq_groups_tuning_rows_npu0.json`
- `/tmp/cannoe_gptq_groups_tuning_rows_npu1.json`
- `/tmp/cannoe_gptq_groups_default_probe_npu0.json`

| Variant | Device | Total ms | max_abs | max_rel |
| --- | --- | ---: | ---: | ---: |
| Default grouped path | npu:0 | 0.945393 | 0.03125 | 34.71875 |
| `tuning_config=[rows]` | npu:0 | 0.958994 | 0.03125 | 34.71875 |
| `tuning_config=[rows]` | npu:1 | 0.941582 | 0.03125 | 34.71875 |

Reason: NPU0 was slower than default and there was no same-run two-device win.
The change adds API surface without a proven benefit.

Re-test only if CANN 9 grouped matmul tuning docs expose shape-specific values
for W4A16 group16 or if a profiler shows tiling selection is the active
bottleneck.

## 2026-05-14: GPTQ Wide BF16 Forced Native Direct Path

Status: failed speed gate.

Tested change: force GPTQ Qwen3 27B BF16 shapes through the native direct CANN
path with `GPTQMODEL_CANNOE_BF16_NATIVE=1`.

Artifacts:

- `/tmp/cannoe_gptq_bf16_forced_direct_qwen27_npu0.json`
- `/tmp/cannoe_gptq_bf16_forced_direct_qwen27_npu1.json`
- `/tmp/cannoe_gptq_bf16_default_rerun_qwen27_npu0.json`
- `/tmp/cannoe_gptq_bf16_default_rerun_qwen27_npu1.json`

| Variant | NPU0 total ms | NPU1 total ms | max_abs | max_rel | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| Default BF16 policy | 1.017812 | 0.961641 | 0.5 | 85.8000 | Keep |
| Forced direct BF16 | 1.035402 | 1.014040 | 0.5 | 0.09375 | Reject |

Reason: forced direct was slower on both NPU0 and NPU1 in this run. The lower
reported `max_rel` did not compensate for the speed regression because
`max_abs` stayed the same and relative error is noisy around near-zero outputs.

Re-test only for new shape classes or if CANN exposes a lower-overhead direct
BF16 W4A16 path.

## 2026-05-14: `npu_convert_weight_to_int4pack(inner_k_tiles=...)`

Status: inconclusive, not retained.

Tested change: pass explicit `inner_k_tiles` values into
`npu_convert_weight_to_int4pack` for Qwen3 35B-style GPTQ shapes.

Artifacts:

- `/tmp/cannoe_gptq_qwen35_innerk0_npu0.json`
- `/tmp/cannoe_gptq_qwen35_innerk1_npu0.json`
- `/tmp/cannoe_gptq_qwen35_innerk2_npu1.json`
- `/tmp/cannoe_gptq_qwen35_innerk4_npu1.json`

| Variant | Device | Total ms | max_abs | max_rel |
| --- | --- | ---: | ---: | ---: |
| `inner_k_tiles=0` | npu:0 | 0.469934 | 0.25 | 237.0 |
| `inner_k_tiles=1` | npu:0 | 0.453026 | 0.25 | 237.0 |
| `inner_k_tiles=2` | npu:1 | 0.478567 | 0.25 | 237.0 |
| `inner_k_tiles=4` | npu:1 | 0.470621 | 0.25 | 237.0 |

Reason: the best value was not validated as a stable two-device win in a
same-run matrix, and the path change would affect packed weight layout broadly.
Leave the default CANN packing behavior in place.

Re-test with a matrix that runs `inner_k_tiles` values 0, 1, 2, and 4 on both
NPU0 and NPU1 in the same benchmark batch.

## 2026-05-14: AWQ BF16 `inner_k_tiles` Style Probe

Status: inconclusive, not retained.

Tested change: apply similar packing/tile exploration to Qwen3 27B AWQ BF16.

Artifacts:

- `/tmp/cannoe_awq_qwen27_innerk1_npu0.json`
- `/tmp/cannoe_awq_qwen27_innerk2_npu1.json`
- `/tmp/cannoe_awq_qwen27_bf16_no_bias_fuse_control_npu0.json`
- `/tmp/cannoe_awq_qwen27_bf16_no_bias_fuse_control_npu1.json`

| Variant | Device | Total ms | max_abs | max_rel |
| --- | --- | ---: | ---: | ---: |
| No-fuse control | npu:0 | 1.301638 | 8.0 | 0.0076336 |
| Tile probe | npu:0 | 1.279285 | 8.0 | 0.0076336 |
| No-fuse control | npu:1 | 1.321499 | 8.0 | 0.0076336 |
| Tile probe | npu:1 | 1.324850 | 8.0 | 0.0076336 |

Reason: NPU0 was slightly faster but NPU1 was slightly slower. Not a stable
two-device win.

Re-test only if the packing implementation can specialize by shape/device or a
larger repeated run shows the NPU1 result was measurement noise.

## 2026-05-14: GPTQ Group16 int32 `group_list`

Status: invalid CANN API input.

Tested change: change the cached group16 grouped-matmul `group_list` tensor
from `torch.int64` to `torch.int32`.

Commands:

- Physical NPU0: `ASCEND_RT_VISIBLE_DEVICES=0 ... python scripts/benchmark_komodo_npu_ab.py --device 0 --cases gptq_group_sizes --dtype fp16 --warmup 8 --iters 80 --komodo-native-int4 --komodo-drop-source-weights --cannoe --json-output /tmp/cannoe_gptq_group_list_int32_phys0.json`
- Physical NPU1: `ASCEND_RT_VISIBLE_DEVICES=1 ... python scripts/benchmark_komodo_npu_ab.py --device 0 --cases gptq_group_sizes --dtype fp16 --warmup 8 --iters 80 --komodo-native-int4 --komodo-drop-source-weights --cannoe --json-output /tmp/cannoe_gptq_group_list_int32_phys1.json`

| Device | Result | Error |
| --- | --- | --- |
| physical NPU0 | Failed before timing JSON | `aclnnGroupedMatmulV5 failed, error code 161002`; `Only int64 is supported for groupList` |
| physical NPU1 | Failed before timing JSON | `aclnnGroupedMatmulV5 failed, error code 161002`; `Only int64 is supported for groupList` |

Reason: despite some grouped-matmul docs implying int32 group-list support in
other modes, the CANN path used here is split-M, single-x, single-weight,
single-y W4A16 grouped matmul, and it requires int64 `group_list`.

Re-test only if we switch to a different grouped matmul API or CANN release
notes explicitly state int32 `group_list` support for this exact grouped W4A16
mode.

## 2026-05-14: GPTQ Group16 `npu_grouped_matmul_finalize_routing`

Status: invalid CANN API input for current W4A16 grouped layout.

Tested change: replace group16 `npu_grouped_matmul(...)[0]` plus
`reshape(...).sum(0)` with `npu_grouped_matmul_finalize_routing`, hoping CANN
could do grouped matmul and row scatter/reduction internally.

Probe shape:

- `x`: `[512, 16]` FP16, representing 64 groups x 8 rows.
- `weight`: `[64, 16, 128]` packed INT4.
- `scale`: `[64, 1, 1024]`.
- `offset`: `[64, 1, 1024]`.
- `group_list`: `[64]` cumulative int64, `group_list_type=0`.
- `row_index`: `[512]`, tested both int64 and int32.
- `output_bs=8`.

Commands:

- Physical NPU0: `ASCEND_RT_VISIBLE_DEVICES=0 ... python - <<'PY'` inline probe using `BenchCase('gptq_gs16', 'gptq', 'fp16', 8, 1024, 1024, 16)`.
- Physical NPU1: same probe with `ASCEND_RT_VISIBLE_DEVICES=1`.

| Device | row_index dtype | Bias | Result |
| --- | --- | --- | --- |
| physical NPU0 | int64 | false | `aclnnGroupedMatmulFinalizeRoutingV3 failed`, error code `161001` |
| physical NPU0 | int64 | true | `aclnnGroupedMatmulFinalizeRoutingV3 failed`, error code `161001` |
| physical NPU0 | int32 | false | `aclnnGroupedMatmulFinalizeRoutingV3 failed`, error code `161001` |
| physical NPU0 | int32 | true | `aclnnGroupedMatmulFinalizeRoutingV3 failed`, error code `161001` |
| physical NPU1 | int64 | false | `aclnnGroupedMatmulFinalizeRoutingV3 failed`, error code `161001` |
| physical NPU1 | int64 | true | `aclnnGroupedMatmulFinalizeRoutingV3 failed`, error code `161001` |
| physical NPU1 | int32 | false | `aclnnGroupedMatmulFinalizeRoutingV3 failed`, error code `161001` |
| physical NPU1 | int32 | true | `aclnnGroupedMatmulFinalizeRoutingV3 failed`, error code `161001` |

Reason: the finalize-routing API exists in torch-npu/CANN 9, but this W4A16
packed weight plus antiquant scale/offset layout is not accepted by
`aclnnGroupedMatmulFinalizeRoutingV3`.

Re-test only if CANN exposes W4A16 antiquant support for finalize-routing, or
if we change group16 to a different packed format that the finalize-routing API
explicitly supports.
