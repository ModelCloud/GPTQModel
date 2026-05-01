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
