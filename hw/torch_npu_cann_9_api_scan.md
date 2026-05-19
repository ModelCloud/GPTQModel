# Torch-NPU and CANN 9 API Scan

Date: 2026-05-01

This is a local scan of the current Ascend environment after sourcing
`/etc/profile.d/ascend.sh`. It records the public and private API surface that
matters for GPTQModel/Komodo work, and compares the current Torch-NPU operator
namespace with the older `hw/ascend_npu.md` Torch-NPU 2.9.0 snapshot.

## Environment

| Item | Value |
| --- | --- |
| Python | 3.11.15 |
| PyTorch | 2.11.0+cpu |
| torch-npu wheel | 2.11.0rc1 |
| torch-npu git version | ad99d6c22fe33f0143cce471b7b0be9e16749b23 |
| CANN runtime | 9.0.0.beta2 |
| Active CANN path | /usr/local/Ascend/cann-9.0.0-beta.2 |
| Active OPP path | /usr/local/Ascend/cann-9.0.0-beta.2/opp |
| SoC version from torch-npu | 220 |
| NPU count | 8 |

The HiAscend 26.0.0 release notes describe the product release as
`Ascend Extension for PyTorch 26.0.0` and map CANN 9.0.0 to public package
versions for PyTorch 2.7.1, 2.8.0, 2.9.0, and 2.10.0. The local wheel reports
`torch_npu.__version__ == 2.11.0rc1`, so treat this host as a newer local wheel
running against CANN 9.0.0-beta.2 rather than a byte-for-byte match to the
public 26.0.0 matrix.

Sources:

- Local package: `/root/ascend910b-py311-torch211/lib/python3.11/site-packages/torch_npu`
- Local CANN 9: `/usr/local/Ascend/cann-9.0.0-beta.2`
- Local CANN 8 baseline: `/usr/local/Ascend/cann-8.5.1`
- HiAscend 26.0.0 release notes:
  https://www.hiascend.com/document/detail/zh/Pytorch/2600/releasenote/docs/zh/release_notes/release_notes.md

## Public Torch-NPU Additions

`torch.ops.npu` currently exposes 340 public operator names after excluding
namespace metadata and private names. The older Torch-NPU 2.9.0 snapshot in
`hw/ascend_npu.md` listed 291 names. This local scan found 49 additions and no
removals versus that snapshot.

High-level grouping of the 49 added ops:

| Area | Count | Notes |
| --- | ---: | --- |
| Quantized matmul, collective matmul, MX quant | 27 | Newest surface for weight-only, collective, all-to-all, reduce-scatter, and MX quant paths. |
| Attention / NSA / block sparse | 7 | Includes CANN 9 attention additions and v3 attention APIs. |
| Norm and activation fusion | 4 | RMSNorm v2, Gelu quant, and Swiglu quant style additions. |
| KV/cache helpers | 3 | Page-attention cache scatter/gather helpers. |
| MoE/routing | 1 | New top-k softmax v2 API. |
| Other runtime/graph hooks | 7 | NPUGraph tensor save and super-kernel scope hooks are visible. |

Full added `torch.ops.npu` names:

```text
npu_add_quant_gmm
npu_add_quant_gmm_
npu_add_quant_matmul
npu_add_quant_matmul_
npu_add_rms_norm_v2
npu_add_rms_norm_v2_functional
npu_all_gather_quant_mm
npu_all_to_all_matmul
npu_all_to_all_quant_matmul
npu_alltoallv_quant_gmm
npu_attention_to_ffn
npu_block_sparse_attention
npu_dense_lightning_indexer_grad_kl_loss
npu_dense_lightning_indexer_softmax_lse
npu_dual_level_quant_matmul
npu_dynamic_dual_level_mx_quant
npu_dynamic_mx_quant
npu_dynamic_mx_quant_with_dual_axis
npu_ffn_to_attention
npu_fused_floyd_attention
npu_fused_floyd_attention_backward
npu_fused_matmul
npu_fusion_attention_grad_v3
npu_fusion_attention_v3
npu_gather_pa_kv_cache
npu_gather_pa_kv_cache_functional
npu_gelu_quant
npu_grouped_dynamic_block_quant
npu_grouped_dynamic_mx_quant
npu_matmul_all_to_all
npu_matmul_compress_dequant
npu_mhc_post
npu_mhc_sinkhorn
npu_moe_gating_top_k_softmax_v2
npu_qkv_rms_norm_rope_cache
npu_qkv_rms_norm_rope_cache_functional
npu_quant_all_reduce
npu_quant_fusion_attention
npu_quant_fusion_attention_backward
npu_quant_gmm_alltoallv
npu_quant_matmul_all_to_all
npu_quant_matmul_gelu
npu_quant_mm_reduce_scatter
npu_quant_reduce_scatter
npu_scatter_pa_cache
npu_transpose_quant_batchmatmul
save_npugraph_tensor
super_kernel_scope_begin
super_kernel_scope_end
```

Komodo-relevant existing and new public ops are present both under
`torch.ops.npu` and as `torch_npu.*` exports:

```text
npu_convert_weight_to_int4pack
npu_weight_quant_batchmatmul
npu_grouped_matmul
npu_prompt_flash_attention
npu_incre_flash_attention
npu_quant_matmul_gelu
npu_fusion_attention_v3
npu_block_sparse_attention
npu_quant_mm_reduce_scatter
npu_all_to_all_quant_matmul
npu_add_quant_matmul
npu_dynamic_mx_quant
npu_swiglu_quant
save_npugraph_tensor
super_kernel_scope_begin
super_kernel_scope_end
```

## Official 26.0.0 API Change Notes

The public HiAscend 26.0.0 release notes list no deleted features and define API
change categories for added, modified, deprecated, and deleted APIs. The listed
custom API additions for the v2.7.1 branch also apply to the public v2.8.0,
v2.9.0, and v2.10.0 branches in that document.

The release-note additions are:

```text
torch_npu.npu.NpuGraphOpHandler
torch_npu.npu_block_sparse_attention
torch_npu.npu_dense_lightning_indexer_grad_kl_loss
torch_npu.npu_dense_lightning_indexer_softmax_lse
torch_npu.npu_fused_floyd_attention
torch_npu.npu_fusion_attention_v3
torch_npu.npu_quant_matmul_gelu
torch_npu.save_npugraph_tensor
torch_npu.npu_attention_to_ffn
torch_npu.npu_attention_update
torch_npu.npu_add_rms_norm
torch_npu.npu_recurrent_gated_delta_rule
torch_npu.npu.matmul.cube_math_type
torch_npu.npu_swiglu_quant
torch_npu.npu_rms_norm_quant
torch_npu.npu_add_rms_norm_quant
torch_npu.npu_clipped_swiglu
```

The release-note modification is:

```text
torch_npu.npu_fusion_attention
```

That API gained optional `dropout_mask`, `seed`, and `offset` parameters in the
26.0.0 notes.

## Local Deprecation and Compatibility Warnings

The local wheel has explicit deprecation or compatibility warnings in these
areas:

| API or option | Local status | Replacement or note |
| --- | --- | --- |
| `torch_npu.npu.amp.custom_fwd` | Deprecated | Use `torch.amp.custom_fwd(..., device_type="npu")`. |
| `torch_npu.npu.amp.custom_bwd` | Deprecated | Use `torch.amp.custom_bwd(..., device_type="npu")`. |
| `ACL_OP_SELECT_IMPL_MODE` | Marked for future deprecation | Avoid setting through `torch_npu.npu.set_option`. |
| `ACL_OPTYPELIST_FOR_IMPLMODE` | Marked for future deprecation | Avoid setting through `torch_npu.npu.set_option`. |
| `torch.nn.DropoutWithByteMask` patch | Deprecated wrapper | Use `torch_npu.contrib.module.DropoutWithByteMask`. |
| `torch.nn.functional.dropout_with_byte_mask` patch | Deprecated wrapper | Use `torch_npu.contrib.function.dropout_with_byte_mask`. |
| Profiler `msprof_tx` | Marked for future deprecation | Use `mstx`. |
| TorchAir `reduce-overhead` mode | Marked for future deprecation | Use `npugraph_ex` backend. |
| `torch.npu.aclnn.version` | Not implemented | Do not rely on it for version probing. |
| `torch.npu.preferred_linalg_library` | Not implemented | Compatibility stub only. |

The wheel also installs `TORCH_NPU_USE_COMPATIBLE_IMPL`; the release notes list
it as a new environment variable. Locally it backs
`torch_npu.npu.use_compatible_impl()` and
`torch_npu.npu.are_compatible_impl_enabled()`. Distributed gather paths inspect
that setting and may choose a compatibility implementation.

## Private Torch-NPU Surface

Private means leading underscore or internal extension binding. Do not build
production integrations against these without a fallback and version gate.

Local counts:

| Namespace | Public count | Private/internal count | Notes |
| --- | ---: | ---: | --- |
| `torch_npu` | 410 | 83 | Includes public op exports and many patch modules. |
| `torch_npu.npu` | 181 | 40 | Public PyTorch-like device API plus allocator, graph, memory, stream, and option APIs. |
| `torch_npu._C` | 5 | 162 | Mostly internal allocator, stream, graph, profiler, dump, recovery, and device-control bindings. |

Important private/internal bindings observed in `torch_npu._C`:

```text
_aclnn_reselect_static_kernel
_get_cann_version
_get_silent_check_version
_is_gte_cann_version
_npu_get_soc_version
_npu_getOption
_npu_setOption
_npu_fused_infer_attention_score_out_graph
_graph_task_group_begin
_graph_task_group_end
_graph_task_update_begin
_graph_task_update_end
_super_kernel_scope_begin
_super_kernel_scope_end
_npu_getDeviceProperties
_npu_getDeviceCount
_npu_getMemoryFraction
_npu_memoryStats
_npu_memorySnapshot
_npu_emptyCache
_npu_synchronize
_npu_restart_device
_npu_shutdown
```

Use `torch_npu._C._get_cann_version("CANN")` only as a diagnostic probe. The
supported application-facing version checks remain `torch_npu.__version__`,
`python -m torch_npu.utils.collect_env`, and CANN's own package metadata.

## CANN 9 Public Header Delta

Compared with local CANN 8.5.1:

| Scope | CANN 8.5.1 | CANN 9.0.0-beta.2 | Added | Removed |
| --- | ---: | ---: | ---: | ---: |
| `aarch64-linux/include/acl` runtime headers | 72 | 72 | 0 | 0 |
| `aarch64-linux/include/aclnnop` ACLNN headers, including `level2` duplicates | 1645 | 1651 | 6 | 0 |
| `aarch64-linux/asc/include` Ascend C public headers | 600 | 726 | 126 | 0 |
| `aarch64-linux/asc/include/adv_api/hccl` public HCCL headers | 6 | 6 | 0 | 0 |
| `aarch64-linux/asc/include/adv_api/hccl/internal` internal HCCL headers | 2 | 2 | 0 | 0 |
| `aarch64-linux/pkg_inc/hccl` package/internal HCCL headers | 9 | 9 | 0 | 0 |

Top-level ACLNN additions:

```text
aclnn_hbm_stress_test_pi.h
aclnn_silent_check.h
aclnn_silent_check_v2.h
```

The corresponding new ACLNN functions are:

```text
aclnnHbmStressTestPi
aclnnHbmStressTestPiGetWorkspaceSize
aclnnSilentCheck
aclnnSilentCheckGetWorkspaceSize
aclnnSilentCheckV2
aclnnSilentCheckV2GetWorkspaceSize
```

Ascend C public header additions are much broader. Category counts:

| Category | Added headers | Examples |
| --- | ---: | --- |
| `adv_api/math` | 38 | Bitwise, logical, FMA, hypot, finite/inf/nan, Philox, rint, sincos, where. |
| `basic_api` | 26 | Basic/cube/vector interfaces plus register-compute interfaces. |
| `simt_api` | 24 | SIMT scalar/vector, math, warp, atomic, device type, and FP format headers. |
| `c_api` | 18 | C API for SIMD, atomic, cache, cube/vector compute and data movement, sync, sys var. |
| `adv_api/quantization` | 7 | Anti-quantize, dequantize, quantize, and quantize utils. |
| `interface` | 7 | Basic/cube/vector interfaces and common system constants/types. |
| `utils` | 6 | System macros/constants and debug print/dump/time/assert helpers. |

Practical implication: CANN 9 did not remove the ACLNN or HCCL header surface
we were using, but it expands the lower-level Ascend C and SIMT/C API surface.
Custom kernels should still include stable public headers from `asc/include`
and avoid `asc/impl`, `pkg_inc`, and `internal` paths unless the code is gated
as an experiment.

## 2026-05-01 Cannoe Rescan

The current rescan confirms the same runtime versions as the first CANN 9 pass:

```text
torch              2.11.0+cpu
torch_npu          2.11.0rc1
torch_npu git      ad99d6c22fe33f0143cce471b7b0be9e16749b23
CANN               9.0.0.beta2
SoC                220
NPU count          8
torch.ops.npu      340 public names
```

The important delta versus CANN 8.5.1 is still the public Ascend C device API,
not the top-level ACLNN API. The rescan found no removed public ACL, ACLNN, or
Ascend C headers. It did find that CANN 9 adds the entire public `asc/include/c_api`
tree locally; CANN 8.5.1 did not have this tree on the host.

Public CANN 9 APIs worth using or probing for Cannoe:

| API area | Local header | 910B relevance | Cannoe action |
| --- | --- | --- | --- |
| Vector int4 to FP16 conversion | `asc/include/c_api/vector_compute/vector_compute.h`, `asc/include/c_api/reg_compute/reg_convert.h` | `asc_int42half` and `asc_int4x22half` replace scalar nibble unpack in the AIV dequant producer. | Implemented as a guarded CANN 9 staged producer with `--experimental-cann9-vector-dequant`. The validated 910B route is `asc_int42half_sync`; `asc_int4x22half` remains register/SIMT-adjacent and is not the first 2201 target. |
| Matmul `VECOUT`/`TSCM` inputs | `asc/include/adv_api/matmul/*` | Official Matmul docs list A/B inputs from `TPosition::VECOUT` and `TPosition::TSCM` on A2, with the per-core tile fully resident in UB/L1. Local headers show non-TSCM local B copies through Matmul workspace, while TSCM local B passes a TSCM physical address. | `--experimental-vecout-consumer` remains an API probe. `--experimental-tscm-consumer` is now the structural target: `B_TYPE=TPosition::TSCM`, `CubeFormat::NZ`, and a staged GM-to-TSCM tile load plus `SetTensorB(LocalTensor<half>)` probe. |
| C API Cube data movement | `asc/include/c_api/cube_datamove/cube_datamove.h` | Adds `asc_copy_l12l0a` and `asc_copy_l12l0b` overloads for `int4b_t`, plus explicit GM/L1/L0 movement primitives. | Useful if high-level Matmul cannot consume the staged tile shape directly. Keep behind a CANN 9 experimental build flag. |
| C API Cube compute | `asc/include/c_api/cube_compute/cube_compute.h` | Adds `asc_mmad_s4` for `int4b_t x int4b_t -> int32_t` on `__NPU_ARCH__ == 2201`. | Not directly suitable for W4A16 because activations are FP16, not INT4. Only probe if we add a separate W4A4/W4A8 or quantized-activation path and accept a larger accuracy contract. |
| Device cache and sync controls | `asc/include/c_api/cache_ctrl/cache_ctrl.h`, `asc/include/c_api/sync/sync.h` | Exposes data-cache preload, DCCI variants, MTE sync, block-arrive/wait, and data barriers for 2201. | Secondary. Use after the VECOUT/TSCM producer-consumer path exists, to pipeline GM copy, vector dequant, L1/UB handoff, Cube compute, and copy-out. |
| `aclnnWeightQuantBatchMatmulNz` | `aarch64-linux/include/aclnnop/aclnn_weight_quant_batch_matmul_nz.h` | Exported by `libopapi.so`; accepts NZ weights with `int32`, `float`, `float4_e2m1`, and `int4`. This header already existed in local CANN 8.5.1. | Probe as a native fallback only. It does not remove the generic ACLNN boundary and has no torch-npu Python binding in this wheel. |
| `aclnnTransMatmulWeight` / `aclnnCalculateMatmulWeightSizeV2` | `aarch64-linux/include/aclnnop/aclnn_trans_matmul_weight.h` | Exported by `libopapi.so`; documented for INT8/FP16/BF16 weight transforms, and mentions V2/V3 matmul weight sizing. | Useful for native fallback layout probes, but not the fused custom-kernel target. Verify INT4 behavior at runtime before relying on it. |
| `aclnnMatmulCompressDequant`, `aclnnQuantMatmulDequant` | `aarch64-linux/include/aclnnop/*compress_dequant*.h`, `*quant_matmul_dequant*.h` | Public and exported; torch-npu exposes matching `npu_matmul_compress_dequant` and `npu_quant_matmul_dequant`. | Not a direct GPTQ W4A16 path. These are INT8/compressed-weight style APIs, so use only for exploratory native baselines. |
| SIMT API | `asc/include/simt_api/*` | Local C++ SIMT headers are guarded for `__NPU_ARCH__ == 3510 || 5102`; public docs currently target Atlas 350 for many Reg/SIMT entries. | Do not target 910B Cannoe with SIMT first. Prefer 2201 C API vector/cube primitives. |

Private/internal API observations:

- `aarch64-linux/asc/impl` grew from 1025 files in CANN 8.5.1 to 1897 files in
  CANN 9.0.0-beta.2. The 873 additions include private implementations for the
  new C API, register-compute conversion, quantization, Matmul tiling, and
  C310/L300 specializations.
- Private Matmul tiling code exposes useful intent such as `enableQuantVector`,
  `TSCM` scale positions, `iterateOrder`, `scheduleType`, and
  `isEnableChannelSplit`, but these are implementation details. Do not include
  `asc/impl` paths directly from Cannoe.
- The public CANN 9 headers themselves sometimes include private implementation
  headers after defining internal include guards. That is acceptable when the
  include originates from `asc/include/...`; it is not a license to include the
  private file directly.
- torch-npu private bindings such as `_aclnn_reselect_static_kernel`,
  `_super_kernel_scope_begin`, `_get_cann_version`, `_npu_setOption`, and
  `_npu_getOption` remain diagnostics or host/runtime controls. They should not
  be part of the custom fused-kernel ABI.

Validated 2026-05-01 local checks:

- `--experimental-staged-dequant --experimental-cann9-vector-dequant` builds
  under `/usr/local/Ascend/cann-9.0.0-beta.2` and launches through the
  repository Ascend C bridge.
- The scalar fused path needed an explicit signed INT4 decode
  `raw < 8 ? raw : raw - 16`; the older xor form produced non-finite lane-0
  output on CANN 9. After this fix, a controlled
  `M=8,K=1024,N=1024,group_size=32` check stayed finite with
  `max_abs=7.62939453125e-06`.
- An 8-NPU one-shard-per-device `gptq_group_sizes` sweep with staged dequant
  enabled kept `max_abs=0.015625` on group sizes 32/64/128/full and act-order
  32/128. Group-size 16 cases still route through the native group16 CANN path.
- `--experimental-vecout-consumer` also builds with the same CANN 9 package,
  validating the public Matmul B-type and `SetTensorB(LocalTensor<half>)` probe
  but not yet feeding the staged tile to Cube at runtime.
- Header inspection of `matmul_client.h` shows VECOUT local B uses Matmul
  workspace copy, while TSCM local B uses `GetTscmAddr`. The next build target is
  therefore `--experimental-tscm-consumer`, not VECOUT, for the no-full-FP16-GM
  Cube handoff.
- `--experimental-tscm-runtime-handoff` builds and runs a narrow live handoff
  probe. On `M=8,K=64,N=8192,group_size=32` it matched a CPU reference with
  `max_abs=0.0`; timing was flat versus the non-runtime staged probe
  (`3.646628 ms` versus `3.651168 ms`) because this version still stages the
  dequantized tile through GM before copying it to TSCM.
- `--experimental-tscm-direct-dequant` validates the public UB-to-TSCM route:
  CANN 9 vector INT4 decode fills a UB tile, then `DataCopy(LocalTensor TSCM,
  LocalTensor UB, Nd2NzParams)` performs the NZ handoff to Cube without writing
  the FP16 B tile through GM/L2. The same NPU0 probe matched with `max_abs=0.0`;
  timing was `3.655767 ms`, which is still flat on one K tile.
- `--experimental-tscm-direct-multik` extends the same direct handoff to
  `K % base_k == 0` by iterating B tiles through TSCM/NZ and accumulating later
  K tiles in Cube Matmul. On NPU0, `M=8,K=128,N=8192,group_size=32,base_k=64`
  matched the CPU reference with finite output and `max_abs=0.0`. Timing was
  flat versus a fresh current-source scalar package (`7.132362 ms` versus
  `7.130773 ms`), which confirms correctness but not overlap yet.
- A follow-up direct multi-K scheduling pass allocates two TSCM B slots and
  hoists scale/offset loads for `base_k >= 128` or `k_tiles >= 4`. This is the
  first live double-buffered shape of the handoff: stage tile `i + 1` while Cube
  is working on tile `i`, then wait before the dependent accumulation. NPU0
  checks stayed finite with `max_abs=0.0` on symmetric
  `M=8,K=512,N=8192,base_k=128`, and the nonzero-offset
  `M=8,K=128,N=8192,base_k=128` probe had `max_abs=0.00390625`, mean drift
  `5.9e-7`. Timing is still effectively flat, around `27.89-27.94 ms` for the
  sampled K512/baseK128 shape.
- Python staged Cube-consumer plans now pick `base_k=128` automatically when
  `K % 128 == 0`, with `GPTQMODEL_CANNOE_BASE_K` as an explicit
  positive-multiple-of-64 override. Producer-only staged plans remain at
  `base_k=64`. A plan-shaped local-A sweep validated the broader default across
  all eight NPUs for rows `8/17/32/48/64/96/129/160` with worst drift
  `max_abs=0.015625` and `mean_abs=0.00142669677734375`.

Concrete next implementation order:

1. Add real AIV/AIC overlap to the direct TSCM path: double-buffer B tiles,
   start dequant for K tile `i + 1` while Cube consumes tile `i`, and keep all
   staging bounded and per-core resident. Never allocate a full dense
   dequantized weight matrix.
2. If high-level Matmul cannot express the required B tile, move one layer down
   to the new public `c_api` movement primitives: GM packed INT4 to L1/UB,
   vector dequant in UB, L1/L0B movement, Cube matmul, then Fixpipe/copy-out.
3. Keep `aclnnWeightQuantBatchMatmulV3` and `aclnnWeightQuantBatchMatmulNz` as
   native CANN fallback probes only. They are useful correctness and layout
   references, but the measured V3 bridge remains slower than the native
   torch-npu op and still crosses the generic ACLNN executor boundary.
4. Delay cache-control tuning until the producer-consumer path is real. Use
   `asc_datacache_preload`, `asc_sync_mte2`, `asc_sync_mte3`,
   `asc_sync_block_arrive`, and `asc_sync_block_wait` only inside a measured
   ping-pong pipeline.

## 2026-05-19 CANN 9.1.0-beta.1 910B Sweep

The active Ascend SDK now resolves to
`/usr/local/Ascend/cann-9.1.0-beta.1`. The local CANN 9.0.0-beta.2 tree on this
host is a thin package and does not expose the full public ACLNN, Ascend C, or
OPP source tree under `aarch64-linux`. Treat this section as the practical
local delta from the 9.0 beta2 install to the new 9.1 beta1 install, not as a
complete upstream-vs-upstream release diff.

Local package inventory:

| Tree | 9.0.0-beta.2 local files | 9.1.0-beta.1 local files | Cannoe relevance |
| --- | ---: | ---: | --- |
| `aarch64-linux/include/aclnnop` | missing | 1675 | Public ACLNN headers are now available locally, including W4A16, NZ, MX, and compress/dequant matmul variants. |
| `aarch64-linux/asc/include` | missing | 521 | Public Ascend C headers are available for custom kernel builds and C API probes. |
| `aarch64-linux/asc/impl` | missing | 2142 | Private implementation headers are useful for reading intent only; do not include them directly from production code. |
| `opp/built-in/op_impl` | missing | 22188 | Reference Ascend C/TBE sources now include CMCT matmul, antiquant prologues, quant batch matmul, and compress/dequant kernels. |
| `opp/built-in/data/tiling` | missing | 575 | Includes 910B matmul repository and cost-model files for multiple AIC counts. |
| `opp/built-in/data/op` | missing | 37 | Includes 910B unified-bank metadata for `Mc2MatMulV3` and `Mc2QuantBatchMatmulV3`. |

Newly exposed or newly actionable public ACLNN headers for GPTQ/Cannoe:

| Surface | Header | Status | Cannoe read |
| --- | --- | --- | --- |
| W4A16 weight-only matmul V3 | `aclnn_weight_quant_batch_matmul_v3.h` | Exported by `libopapi.so`; public signature includes `innerPrecise`. | Keep the current V3 bridge as a shape-policy probe. It can call the same CANN op family directly and select `innerPrecise` from our measured table, but it is not a fused-kernel replacement. |
| W4A16 right-weight NZ matmul | `aclnn_weight_quant_batch_matmul_nz.h` | Exported by `libopapi.so`; documents FP16/BF16 `x`, INT4 NZ `weight`, FP16/BF16 output, group size, optional bias, scale, and offset. | Best next native fallback probe. If we can store GPTQ packed weights in CANN's NZ INT4 layout without dense FP16 caching, this may cut internal transpose/prepack overhead on large `down_proj`. |
| Public INT4 pack converter | `aclnn_convert_weight_to_int4_pack.h` | Exported header for CANN INT4 pack conversion. | Compare its layout with `torch.ops.npu.npu_convert_weight_to_int4pack` before building an NZ path. |
| Quant matmul V5 | `aclnn_quant_matmul_v5.h` | Supports both inputs as `float4_e2m1`, INT8, INT4, FP8, or HiFloat8 with output INT8/FP16/BF16/FP32. | Not direct GPTQ W4A16 because activations are also quantized, but useful for future A4W4/A8W4 experiments. |
| Fused quant matmul weight NZ | `aclnn_fused_quant_matmul_weight_nz.h` | Supports INT4/INT8/INT32 inputs, NZ weight, optional bias, GELU fusion, FP16/BF16 output. | Not a decode GPTQ path unless activations are quantized. Good reference for NZ weight contracts and epilogue fusion. |
| Dual-level quant matmul NZ | `aclnn_dual_level_quant_matmul_nz.h` | Targets MxFP4/MxFP4 with dual-level scales and NZ right matrix. | Not GPTQ-compatible today, but confirms 9.1 has first-class FP4/NZ matmul work. |
| Quant matmul dequant | `aclnn_quant_matmul_dequant.h` and grouped/NZ variants | Document INT8 weight dequant paths, including grouped and NZ variants. | Not direct GPTQ INT4, but useful reference for host ABI and weight-scale layout. |
| Matmul compress/dequant | `aclnn_matmul_compress_dequant.h` | Public compressed-weight dequant API. | Exploratory only; no clear GPTQ INT4 mapping yet. |
| Matmul weight transform | `aclnn_trans_matmul_weight.h` | Weight-size and transform helpers mention V3 matmul weight sizing, but documented dtypes are INT8/FP16/BF16. | Do not assume INT4 support; runtime-probe before using. |

The 9.1 `libopapi.so` exports the key symbols:

```text
aclnnWeightQuantBatchMatmulV3
aclnnWeightQuantBatchMatmulV3GetWorkspaceSize
aclnnWeightQuantBatchMatmulNz
aclnnWeightQuantBatchMatmulNzGetWorkspaceSize
aclnnQuantMatmulV5
aclnnFusedQuantMatmulWeightNz
aclnnMatmulCompressDequant
aclnnQuantGroupedMatmulDequantWeightNZ
```

Torch-NPU still exposes the high-level Python binding:

```text
torch.ops.npu.npu_weight_quant_batchmatmul(
    x, weight, antiquant_scale, antiquant_offset=None,
    quant_scale=None, quant_offset=None, bias=None,
    antiquant_group_size=0, inner_precise=0)
```

The installed torch-npu packages do not expose a Python binding for
`aclnnWeightQuantBatchMatmulNz`, so NZ probing needs a C++/ACLNN bridge or a
future torch-npu update.

9.1 OPP sources now include a useful CMCT reference shape for the true Cannoe
fused kernel:

```text
opp/built-in/op_impl/ai_core/tbe/impl/ops_nn/ascendc/common/cmct/kernel/kernel_matmul_a_prefetch_b_antiquant.h
opp/built-in/op_impl/ai_core/tbe/impl/ops_nn/ascendc/common/cmct/prologue/block_prologue_b_antiquant_scmc_nd_kn.h
opp/built-in/op_impl/ai_core/tbe/impl/ops_nn/ascendc/common/cmct/prologue/block_prologue_b_antiquant_scmc_nd_nk_nz_kn.h
opp/built-in/op_impl/ai_core/tbe/impl/ops_nn/ascendc/common/cmct/prologue/tile/antiquant_nd_kn.h
opp/built-in/op_impl/ai_core/tbe/impl/ops_nn/ascendc/common/cmct/prologue/tile/antiquant_nd_nk.h
opp/built-in/op_impl/ai_core/tbe/impl/ops_nn/ascendc/common/cmct/prologue/tile/antiquant_zn.h
opp/built-in/op_impl/ai_core/tbe/impl/ops_nn/ascendc/common/cmct/prologue/tile/tile_antiquant.h
```

The important design pattern in those sources is a split AIV/AIC operator:
AIV walks B-weight tiles, scale tiles, and offset tiles and runs the antiquant
prologue; AIC preloads A and runs the matmul block scheduler over the same
N/M tile space. This is the closest shipped reference to Cannoe's target
kernel. It should guide scheduling, tile ownership, and ND/NZ antiquant
prologue structure, but production Cannoe should still include public Ascend C
headers rather than private `opp` implementation headers.

910B-specific OPP metadata is now visible for multiple B variants:

```text
Ascend910B1_24_AiCore_Mc2MatMulV3_runtime_kb.json
Ascend910B1_24_AiCore_Mc2QuantBatchMatmulV3_runtime_kb.json
Ascend910B1_25_AiCore_Mc2QuantBatchMatmulV3_runtime_kb.json
Ascend910B2*_Mc2QuantBatchMatmulV3_runtime_kb.json
Ascend910B3*_Mc2QuantBatchMatmulV3_runtime_kb.json
Ascend910B4*_Mc2QuantBatchMatmulV3_runtime_kb.json
```

`Mc2*` is primarily a communication/distributed matmul family, so it is not the
single-card GPTQ decode path by itself. The useful part for Cannoe is the
metadata shape: CANN is now shipping 910B-specific quant matmul bank data and
CMCT matmul sources that can be used as scheduling references.

Actionable order after this sweep:

1. Add a small ACLNN NZ bridge probe for `aclnnWeightQuantBatchMatmulNz`, using
   a persistent packed INT4 NZ weight layout. The acceptance rule is no dense
   FP16 weight cache and a full Qwen q/k/v/gate/up/down benchmark versus current
   Cannoe and Komodo.
2. Keep the existing V3 bridge for `innerPrecise` shape-policy expansion only.
   It is correct but slower than the native torch-npu call boundary in the
   current measurements.
3. Use the CMCT `A-prefetch/B-antiquant` sources as the implementation map for
   the true fused Cannoe kernel: AIV antiquant prologue, AIC matmul consumer,
   bounded per-tile storage, and no full dequantized weight materialization.
4. Do not spend time on `aclnnQuantMatmulV5`, fused quant matmul weight NZ, or
   dual-level MX/FP4 paths for GPTQ FP16 decode until there is an activation
   quantization mode to justify them.

## CANN 9 Operator Metadata Check

The CANN 9.0.0-beta.2 910B OPP package is installed and has the metadata needed
for simple tensor creation and `OnesLike`/`InplaceOne` paths:

```text
ascend-cann-910b-ops 9.0.0-beta.2
Ascend910B kernel config JSON count: 863
Ascend910B kernel binary count: 18828
OnesLike/InplaceOne metadata hits: 134
```

`torch.ones(..., device="npu")` passed after installing the 910B OPP package.
Any future `aclnn* failed` error for a basic op should first check that
`ASCEND_OPP_PATH` points at the CANN 9 OPP and that the target SoC has matching
metadata under that OPP tree.

## Guidance for GPTQModel

- Keep the existing Komodo int4 path on stable public ops:
  `npu_convert_weight_to_int4pack`, `npu_weight_quant_batchmatmul`, and
  `npu_grouped_matmul`.
- Gate any new CANN 9-only acceleration path on both CANN version and operator
  presence. Useful first probes are `npu_quant_matmul_gelu`,
  `npu_add_quant_matmul`, `npu_all_to_all_quant_matmul`,
  `npu_quant_mm_reduce_scatter`, `npu_dynamic_mx_quant`, and
  `npu_fusion_attention_v3`.
- Treat `torch_npu._C` bindings and private `torch_npu._*` functions as
  diagnostics or experiments. They are not compatibility contracts.
- Do not infer public support from the presence of `aclnnop/level2` duplicates
  alone; prefer the top-level `aclnnop/*.h` header plus matching exported
  library symbol and a runtime probe.
- The local wheel is a 2.11.0 release candidate. If a compatibility bug is
  traced to the wheel rather than CANN/OPP, retest against the public 26.0.0
  wheel family for PyTorch 2.10.0 or the matching public wheel once Huawei
  publishes a 2.11 build.

## Rescan References

- Local public Ascend C headers:
  `/usr/local/Ascend/cann-9.0.0-beta.2/aarch64-linux/asc/include`
- Local private Ascend C implementation headers:
  `/usr/local/Ascend/cann-9.0.0-beta.2/aarch64-linux/asc/impl`
- Local torch-npu package:
  `/root/ascend910b-py311-torch211/lib/python3.11/site-packages/torch_npu`
- Official CANN 9 Matmul API notes:
  https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta2/API/ascendcopapi/atlasascendc_api_07_0614.html
- Official CANN 9 ACLNN API overview:
  https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta2/API/aolapi/atlasascendc_api_07_1042.html
- Official TSCM Matmul scenario note:
  https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1/opdevg/Ascendcopdevg/atlas_ascendc_10_10024.html
