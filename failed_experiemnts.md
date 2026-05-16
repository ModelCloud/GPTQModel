# Failed Experiments

This file records Cannoe/Komodo optimization probes that were rejected or left
inconclusive. Keep metric evidence here because `/tmp` benchmark artifacts are
ephemeral and repeated failed probes waste NPU time.

For new entries, include:

- Date and kernel path.
- Exact change or environment knob tested.
- Benchmark artifacts or command shape.
- NPU devices used. Current policy for broad speed-discovery sweeps is one
  worker per physical NPU across NPU 0-7 when the user asks for all devices;
  use narrower NPU 0/1 checks only for quick confirmation.
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

## 2026-05-14: GPTQ Qwen3 27B wide BF16 force-direct

Status: failed stable speed gate; keep the moderate-feature guard for GPTQ
BF16 direct.

Tested change: force `GPTQMODEL_CANNOE_BF16_NATIVE=1` for all Qwen3 27B GPTQ
BF16 projection shapes, including the wide `gate_proj`, `up_proj`, and
`down_proj` cases that are intentionally blocked by the default
`in_features/out_features <= 8192` guard.

Commands:

- Physical NPU0: `ASCEND_RT_VISIBLE_DEVICES=0 GPTQMODEL_CANNOE_BF16_NATIVE=1 ... python scripts/benchmark_komodo_npu_ab.py --device 0 --cases qwen3_6_27b_gptq --warmup 6 --iters 50 --komodo-native-int4 --komodo-drop-source-weights --cannoe --json-output /tmp/cannoe_qwen27b_gptq_forced_direct_phys0.json`
- Physical NPU1: `ASCEND_RT_VISIBLE_DEVICES=1 GPTQMODEL_CANNOE_BF16_NATIVE=1 ... python scripts/benchmark_komodo_npu_ab.py --device 0 --cases qwen3_6_27b_gptq --warmup 6 --iters 50 --komodo-native-int4 --komodo-drop-source-weights --cannoe --json-output /tmp/cannoe_qwen27b_gptq_forced_direct_phys1.json`

| Variant | Device | Total ms | max_abs | max_rel |
| --- | --- | ---: | ---: | ---: |
| Default guarded policy | physical NPU0 | 0.9684 | 0.5 | 85.8 |
| Forced direct | physical NPU0 | 0.9670 | 0.5 | 0.09375 |
| Default guarded policy | physical NPU1 | 0.9974 | 0.5 | 85.8 |
| Forced direct | physical NPU1 | 1.0374 | 0.5 | 0.09375 |

Reason: forced direct greatly improves relative drift on the wide GPTQ shapes,
but the speed result is not a two-device win. Physical NPU0 is effectively tied,
while physical NPU1 regresses by about 4%.

Re-test only if a shape-specific policy is introduced for the wide projections,
or if a larger repeated run shows the NPU1 regression was measurement noise.

## 2026-05-14: GPTQ Qwen3 27B BF16 direct `inner_precise=1`

Status: failed speed gate.

Tested change: pass CANN `inner_precise=1` through the GPTQ BF16 native-direct
path for group-32 Qwen3 27B q/k/v projection shapes.

Commands:

- Physical NPU0: `ASCEND_RT_VISIBLE_DEVICES=0 ... python scripts/benchmark_komodo_npu_ab.py --device 0 --cases qwen3_6_27b_gptq --warmup 6 --iters 50 --komodo-native-int4 --komodo-drop-source-weights --cannoe`
- Physical NPU1: `ASCEND_RT_VISIBLE_DEVICES=1 ... python scripts/benchmark_komodo_npu_ab.py --device 0 --cases qwen3_6_27b_gptq --warmup 6 --iters 50 --komodo-native-int4 --komodo-drop-source-weights --cannoe`

| Variant | Device | Total ms | max_abs | max_rel |
| --- | --- | ---: | ---: | ---: |
| Default BF16 direct | physical NPU0 | 0.9684 | 0.5 | 85.8 |
| GPTQ direct `inner_precise=1` | physical NPU0 | 1.0066 | 0.5 | 85.8 |
| Default BF16 direct | physical NPU1 | 0.9974 | 0.5 | 85.8 |
| GPTQ direct `inner_precise=1` | physical NPU1 | 1.0099 | 0.5 | 85.8 |

Reason: isolated q/k/v microprobes looked mixed-to-positive, but the complete
Qwen3 27B GPTQ projection set regressed on both physical NPUs with no accuracy
improvement. Keep the default GPTQ BF16 direct call shape.

Re-test only if CANN changes the `inner_precise` implementation or if we add a
per-projection policy with full-model two-device benchmark evidence.

## 2026-05-15: Cannoe Ascend C VecOut Local-A fused handoff

Status: correctness scaffold validated after logical-block cap, but failed the
speed gate and remains experimental only.

Accepted change: cap the Ascend C host tiler and Python staged planner to 8
logical blocks. Before the cap, scalar output ownership was sparse/wrong and
the VecOut local-A path could hang. After the cap:

| Path | Shape | Tile attrs | Result |
| --- | --- | --- | --- |
| Scalar baseline | rows=1,K=384,N=256,group=32 | base_m=16,base_n=256,base_k=-128 | pass, max_abs=0, mean_abs=0, first custom_ms=509.012 |
| VecOut local-A | rows=1,K=384,N=256,group=32 | base_m=16,base_n=-256,base_k=-128 | pass, max_abs=0.00390625, mean_abs=0.0005016, first custom_ms=513.256 |

Rejected/not promoted:

| Shape | Tile attrs | Result |
| --- | --- | --- |
| rows=1,K=5120,N=6144,group=32 | base_m=16,base_n=-256,base_k=-128 | pass, warmed custom_ms=76.118, max_abs=0.0625, mean_abs=0.006378 |
| rows=1,K=5120,N=6144,group=32 | base_m=16,base_n=-128,base_k=-128 | pass, warmed custom_ms=76.332, max_abs=0.0625, mean_abs=0.006317 |
| rows=1,K=5120,N=256,group=32 | base_m=16,base_n=-128,base_k=-128 | pass only with relaxed tolerance, warmed custom_ms=12.816, max_abs=0.046875, mean_abs=0.006222 |
| rows=1,K=5120,N=256,group=32 | base_m=16,base_n=-256,base_k=-128 | pass only with relaxed tolerance, warmed custom_ms=25.439, max_abs=0.0625, mean_abs=0.005989 |
| rows=1,K=5120,N=1024,group=32 | base_m=16,base_n=-256,base_k=-128 | pass only with relaxed tolerance, warmed custom_ms=25.479, max_abs=0.0390625, mean_abs=0.006622 |
| rows=1,K=5120,N=1024,group=32 | base_m=16,base_n=-128,base_k=-128 | AICore 507015 illegal instruction, likely unaligned UUB access |
| rows=1,K=5120,N=256,group=32 | base_m=16,base_n=-64,base_k=-128 | AICore 507015 illegal instruction, likely unaligned UUB access |
| rows=1,K=5120,N=6144,group=32 | base_m=16,base_n=-512,base_k=-128 | AICore 507015 load3d out-of-range |
| rows=1,K=5120,N=256,group=32 | base_m=16,base_n=-256,base_k=-256 | AICore 507015 load3d out-of-range |

Comparison point: the native Cannoe Qwen3 27B fp16 GPTQ NPU0 total was
previously 0.9179 ms, while A100 Marlin reported 0.3835 ms. A fused q-proj
tile taking 76 ms is not a candidate default path.

Decision: keep VecOut local-A as an opt-in Ascend C correctness scaffold that
avoids full FP16 weight materialization through GM/L2, but do not route
production Cannoe through it until the local-B INT4 decode path stops using
scalar SetValue-style expansion and can feed Cube with vectorized UB/TSCM
staging at native-kernel speed.

## 2026-05-15: Cannoe CANN9 `basic_api/reg_compute` INT4 dequant probe on 910B

Status: failed compile gate; do not retry on 910B unless Huawei expands
`basic_api/reg_compute` to `__NPU_ARCH__ == 2201`.

Tested change: an opt-in VecOut local-A strategy that attempted to use CANN9
`basic_api/reg_compute/kernel_reg_compute_intf.h` for 128-lane signed INT4 to
FP16 conversion in UB, followed by vector scale multiply and vector store into
the Cube-facing local B tile.

Command:

- `python scripts/build_cannoe_ascendc.py --strategy vecout-reg-dequant --output /tmp/cannoe_w4a16_vecout_reg_dequant --clean`

Result:

| Probe | Result |
| --- | --- |
| Include gated by `ASC_DEVKIT_MAJOR >= 9` | Failed because `ASC_DEVKIT_MAJOR` was not defined at include time, so the reg-compute definitions were skipped. |
| Include gated only by `CANNOE_EXPERIMENTAL_VECOUT_REG_DEQUANT` | Still failed; CANN9 `kernel_reg_compute_intf.h` defines `AscendC::Reg` only for `__NPU_ARCH__` 3510/5102/3003/3113 or host debug, not 910B `__NPU_ARCH__ == 2201`. |

Representative compiler errors:

- `no type named 'CastTrait' in namespace 'AscendC::Reg'`
- `no type named 'MaskReg' in namespace 'AscendC::Reg'`
- `no member named 'RegTensor' in namespace 'AscendC::Reg'`

Reason: the local CANN9 headers expose 910B-compatible C API vector/cube
primitives under `asc/include/c_api`, but the C++ register-compute API used by
the public antiquant examples is not compiled for 910B. Keeping an opt-in build
flag for this path would leave a known-noncompiling strategy in the tree, so the
probe was reverted after logging.

Next viable target: continue with 2201-compatible C API primitives:
`asc_int42half_sync` for local vector dequant, `asc_mul_sync` or related vector
compute for scaling when the tile layout is contiguous, and C API Cube data
movement/compute for a direct L1/L0B handoff if high-level Matmul cannot consume
the staged local tile efficiently.

## 2026-05-15: Cannoe VecOut local-B 2201 vector pair scale/store

Status: failed runtime smoke; reverted.

Tested change: in the symmetric/zero-offset VecOut local-A path, replace two
scalar 8-lane B-tile writes with a 16-lane 2201 C API sequence:

1. pack two INT4 words into a 32 B UB scratch buffer;
2. call `asc_int42half_sync(..., count=16)`;
3. copy 16 FP16 scales from GM to UB once per packed-column pair;
4. call `asc_mul_sync` with the destination pointer set directly to the local B
   tile at the 32 B-aligned column pair.

Build result:

- `python scripts/build_cannoe_ascendc.py --strategy vecout-local-a --output /tmp/cannoe_w4a16_vecout_pair_scale --clean`
- Result: compiled and packaged successfully.

Runtime smoke:

- Installed package with `custom_opp_ubuntu_aarch64.run --quiet --install-path=/tmp/cannoe_vecout_pair_scale_install`.
- Single physical NPU0 worker:
  `ASCEND_RT_VISIBLE_DEVICES=0 ... python scripts/validate_cannoe_ascendc_raw.py --worker --device 0 --case-json '{"rows":1,"k":384,"n":256,"group":32,"seed":2200,"base_m":16,"base_n":-256,"base_k":-128}' --warmup 1 --iters 3 --max-abs 0.08 --mean-abs 0.008`
- Result: no JSON result after 90 seconds; worker was killed. This is worse
  than the previous scalar local-A smoke, which returned first-call timings near
  513 ms for the same small shape.

Likely cause: direct `asc_mul_sync` vector store into the high-level Matmul
`TPosition::VECOUT` local B tensor is not a safe/forward-progress path on this
code shape, even when the destination offset is arranged as a 32 B pair. It may
need explicit C API L1/L0B movement instead of writing through the Matmul
LocalTensor abstraction.

Decision: do not use vector compute to write directly into Matmul's local B
`LocalTensor`. Revisit only inside a lower-level C API Cube pipeline where the
destination memory space and synchronization are controlled explicitly.

## 2026-05-15: Cannoe high-level TSCM direct handoff on CANN 9.0.0 all-NPU sweep

Status: failed runtime gate; keep the high-level `Matmul`/`TPosition::TSCM`
route experimental and do not promote it as the default fused handoff on the
current CANN 9.0.0-beta.2 stack.

Tested package:

- Build: `python scripts/build_cannoe_ascendc.py --strategy tscm-direct-multik --output /tmp/cannoe_w4a16_tscm_direct_current --clean`
- Install: `/tmp/cannoe_tscm_direct_current_install`
- Runtime harness: `scripts/validate_cannoe_ascendc_raw.py` with physical
  devices `0,1,2,3,4,5,6,7`, one worker per NPU.

Results:

| Probe | Devices | Shape / tile | Result |
| --- | --- | --- | --- |
| Current direct TSCM, Qwen-like wide N tile | 0-7 | `rows=8,K=128,N=8192,group=32,base_m=16,base_n=256,base_k=128` | All 8 workers failed before timing JSON with `507015`; CANN reported AICore illegal instruction, usually caused by unaligned UUB addresses. |
| TSCM-specific logical block cap raised from 8 to 24 | 0-7 | same as above | All 8 workers still failed with the same `507015` / unaligned-UUB class error. Temporary source change was reverted. |
| Older known-good family retry | 0-7 | `rows=8,K=64,N=8192,group=32,base_m=8,base_n=256,base_k=64` | Run hit the outer 180 s timeout before worker JSON. Several workers exited early while others remained stuck. |

Representative error text:

- `npuSynchronizeDevice ... error code is 507015`
- `The aicore execution is abnormal`
- `errorStr: Illegal instruction, which is usually caused by unaligned UUB addresses`

Interpretation: the public high-level Matmul TSCM local-B route is currently
not a stable production path on this CANN 9.0.0-beta.2 environment, even when
the dequantized tile is bounded and never written as a full dense FP16 matrix.
The next true fused attempt should move below the high-level Matmul local tensor
handoff into explicit C API Cube movement/compute primitives where L1/L0B
destination alignment and synchronization are controlled directly. The existing
optimized Cannoe/VecOut path remains the runnable baseline while this lower
level handoff is developed.

## 2026-05-15: Qwen3 27B Ascend C VecOut local-A as a production route

Status: failed speed gate; keep as an opt-in fused-kernel scaffold only.

Tested change: rebuild the current accepted `vecout-local-a` Ascend C package
and force the benchmark through `GPTQMODEL_CANNOE_ASCENDC=1`,
`GPTQMODEL_CANNOE_STAGED_DEQUANT=1`, and
`GPTQMODEL_CANNOE_CUBE_CONSUMER=1` for Qwen3 27B GPTQ fp16 group-32 shapes.

Artifacts:

- `/tmp/cannoe_vecout_local_a_current_install`
- `/tmp/cannoe_qwen27_debug_npu0.json`
- `/tmp/cannoe_layer_k_npu1.json`
- `/tmp/cannoe_layer_gate_npu3.json`
- `/tmp/cannoe_layer_up_npu4.json`
- `/tmp/cannoe_layer_down_bk64_npu7.json`

| Probe | Device | Shape | Mean ms | Peak MB | Result |
| --- | --- | --- | ---: | ---: | --- |
| q default fused | NPU0 | `M=1,K=5120,N=6144` | 76.1650 | 214.4 | Reject |
| k default fused | NPU1 | `M=1,K=5120,N=1024` | 25.5160 | 92.9 | Reject |
| gate default fused | NPU3 | `M=1,K=5120,N=17408` | 228.2138 | 271.8 | Reject |
| up default fused | NPU4 | `M=1,K=5120,N=17408` | 228.0849 | 271.8 | Reject |
| down fused, `base_k=64` | NPU7 | `M=1,K=17408,N=5120` | 259.5352 | 703.9 | Reject |

Additional all-NPU probes:

| Probe | Devices | Result |
| --- | --- | --- |
| Full six-layer Qwen pass through forced Ascend C fused path | NPU0 | Timed out at 300 s before table output. |
| Isolated q/v/down/default and gate `base_n=128` while eight workers were active | NPU0/NPU2/NPU5/NPU6 | Timed out at 300 s. |
| Forced fused `base_n=512` and `base_k=256` | NPU2/NPU4 | Exited nonzero before JSON, consistent with unsupported oversize tile buffers. |

Reason: the local-A VecOut path avoids full dense FP16 weight materialization,
but for Qwen-scale K/N it is orders of magnitude slower than the production
native CANN path. It is still useful for correctness and handoff experiments,
not as a default inference route.

Re-test only after the device kernel stops scalar-expanding the whole B tile
and uses a true vectorized INT4 dequant-to-Cube handoff with explicit
L1/L0B synchronization.

## 2026-05-15: Qwen3 27B native Cannoe prepack/tuning sweep

Status: no production change; default plain-native path remains fastest in the
all-layer total.

Tested change: run the production native Cannoe path on all physical NPUs
concurrently, one process per device, with different prepack/tuning knobs.

Artifacts:

- `/tmp/cannoe_native_qwen27_default_npu0.json`
- `/tmp/cannoe_native_qwen27_komodo_tile512_npu1.json`
- `/tmp/cannoe_native_qwen27_komodo_tile2048_npu2.json`
- `/tmp/cannoe_native_qwen27_komodo_tile4096_npu3.json`
- `/tmp/cannoe_native_qwen27_komodo_tile8192_npu4.json`
- `/tmp/cannoe_native_qwen27_cannoe_tile2048_npu5.json`
- `/tmp/cannoe_native_qwen27_inner1_npu6.json`
- `/tmp/cannoe_native_qwen27_desc_act_npu7.json`

| Variant | Device | Total mean ms | q | k | v | gate | up | down | Peak MB max |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Default plain native | NPU0 | 0.8980 | 0.0776 | 0.0686 | 0.0723 | 0.2124 | 0.1905 | 0.2766 | 513.2 |
| `GPTQMODEL_KOMODO_PREPACK_TILE_N=512` | NPU1 | 0.9201 | 0.0775 | 0.0728 | 0.0756 | 0.2215 | 0.1928 | 0.2799 | 360.6 |
| `GPTQMODEL_KOMODO_PREPACK_TILE_N=2048` | NPU2 | 1.0230 | 0.0778 | 0.0708 | 0.0740 | 0.2491 | 0.2505 | 0.3007 | 819.4 |
| `GPTQMODEL_KOMODO_PREPACK_TILE_N=4096` | NPU3 | 1.0035 | 0.0772 | 0.0612 | 0.0716 | 0.2482 | 0.2457 | 0.2997 | 1389.2 |
| `GPTQMODEL_KOMODO_PREPACK_TILE_N=8192` | NPU4 | 0.9694 | 0.0756 | 0.0699 | 0.0732 | 0.2446 | 0.2332 | 0.2727 | 1695.6 |
| `GPTQMODEL_CANNOE_PREPACK_TILE_N=2048` | NPU5 | 1.1047 | 0.1114 | 0.1144 | 0.1146 | 0.2467 | 0.2498 | 0.2678 | 819.6 |
| `GPTQMODEL_CANNOE_INNER_PRECISE=1` | NPU6 | 1.1198 | 0.1091 | 0.1096 | 0.1107 | 0.2420 | 0.2475 | 0.3008 | 819.6 |
| Default desc-act | NPU7 | 1.1441 | 0.1048 | 0.1093 | 0.1107 | 0.2467 | 0.2601 | 0.3124 | 512.8 |

Reason: larger prepack tiles can help isolated q/down timings slightly, but
they slow gate/up enough to lose the full Qwen layer total and can inflate peak
allocation substantially. Forced Cannoe tuning disables the plain-native fast
binding and is slower for this workload. Forced `inner_precise=1` remains
shape-dependent and should not be broadened beyond the existing auto rule.

Re-test if CANN changes the packed W4A16 layout or if we add shape-local packing
selection that can choose a different tile per projection without increasing
the persistent packed-weight footprint.

## 2026-05-16: Cannoe VecOut `asc_int42half_sync` dequant-to-local-B probes

Status: failed correctness and forward-progress gates; reverted.

Tested change: keep the fused Ascend C VecOut/local-A scaffold, but replace
part of the scalar INT4 unpack with CANN 9 public vector INT4-to-FP16 device
API before feeding the dequantized B values into the high-level Matmul local-B
tile. All probes used physical devices `0,1,2,3,4,5,6,7`, one worker per NPU.

Artifacts:

- `/tmp/cannoe_vecout_word_vector_summary.json`
- `/tmp/cannoe_vecout_word_vector_barrier_summary.json`
- `/tmp/cannoe_vecout_tile_vector_summary.json`

| Probe | Devices | Returned timings | Accuracy / progress | Result |
| --- | --- | ---: | --- | --- |
| Per-packed-word `asc_int42half_sync` into scratch UB, then scalar placement | 0-7 | 2.8988-4.3048 ms on returned workers | `max_abs=3392`, `NaN`, or timeout on 5/8 workers | Reject |
| Same word-vector path with `PipeBarrier<PIPE_V>()` after vector cast | 0-7 | 2.2079-5.1139 ms on returned workers | `max_abs=12.75`, `NaN`, `Infinity`, or timeout on 5/8 workers | Reject |
| Whole packed B tile copied to VECCALC UB, tile-wide `asc_int42half_sync`, scalar scale/place into local B | 0-7 | 1.6115-3.1559 ms on returned workers | `max_abs=3.09-3.66`, `mean_abs=0.73-0.83`, or timeout on 5/8 workers | Reject |

Representative failing cases:

| Probe | Device | Shape | Mean ms | Max abs | Mean abs |
| --- | --- | --- | ---: | ---: | ---: |
| Word vector | NPU5 | `rows=3,K=768,N=256,group=96` | 4.3048 | 3392.0 | 10.5938 |
| Word vector | NPU1 | `rows=2,K=512,N=256,group=64` | 2.8988 | NaN | NaN |
| Word vector + barrier | NPU0 | `rows=1,K=384,N=256,group=32` | 2.2079 | 12.75 | 3.0566 |
| Word vector + barrier | NPU6 | `rows=7,K=896,N=256,group=128` | 5.1139 | Infinity | Infinity |
| Tile vector | NPU0 | `rows=1,K=384,N=256,group=32` | 1.6115 | 3.2422 | 0.7349 |
| Tile vector | NPU2 | `rows=4,K=512,N=256,group=32` | 2.1496 | 3.6562 | 0.8296 |
| Tile vector | NPU5 | `rows=3,K=768,N=256,group=96` | 3.1559 | 3.0938 | 0.8188 |

Interpretation: `asc_int42half_sync` is not a drop-in replacement for the
current torch-npu INT4 packed-word decode order. The per-word granularity is
unstable even with a V-pipe barrier, while the whole-tile granularity is faster
for workers that return but still decodes the wrong values and can hang. This
points to a packed-lane/layout mismatch between `npu_convert_weight_to_int4pack`
and CANN's public `int4b_t` vector cast semantics, or to an unsafe source tensor
interpretation for the current local-B handoff.

Decision: do not keep either vector path in production or behind a normal
runtime knob. Re-test only after adding a small lane-order diagnostic that
compares known packed words against CANN `int4b_t` vector decode, or after
switching to the public `AscendAntiQuant` local-tile API with explicit layout
handling. The production baseline remains native Cannoe while the true fused
kernel target moves toward a verified local INT4 tile layout plus explicit
Cube handoff.
