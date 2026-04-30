# Ascend NPU CUDA API Equivalents

This file tracks the Huawei Ascend Extension for PyTorch 7.3.0 support status for
PyTorch 2.9.0 `torch.cuda` APIs that have `torch_npu.npu` or `torch.npu`
equivalents.

Source:

- Huawei overview: https://www.hiascend.com/document/detail/zh/Pytorch/730/apiref/PyTorchNativeapi/docs/zh/native_apis/pytorch_2-9-0/overview.md
- Huawei CUDA mapping page: https://www.hiascend.com/document/detail/zh/Pytorch/730/apiref/PyTorchNativeapi/docs/zh/native_apis/pytorch_2-9-0/torch-cuda.md
- Local `torch.ops.npu` scan:
  `/root/ascend910b-py311/bin/python`, `torch 2.9.0+cpu`,
  `torch_npu 2.9.0`.

Snapshot date: 2026-04-30.

Huawei's overview defines four cases: supported with no restrictions, supported
with restrictions, unsupported, and present in upstream PyTorch but not listed in
Huawei's support table. Treat unlisted APIs as unverified on NPU.

For supported CUDA namespace APIs, replace `torch.cuda.*` with the listed NPU
form. Huawei documents `torch_npu.npu.*` and `torch.npu.*` as functionally
equivalent forms for these APIs.

## Public `torch.ops.npu` Operator Snapshot

Importing `torch_npu` registers the `torch.ops.npu` namespace. The local
2.9.0 scan exposes 291 non-private operator names after excluding the namespace
metadata attribute `name`.

GPTQModel currently probes or calls this subset directly:

| Operator | Use |
| --- | --- |
| `npu_convert_weight_to_int4pack` | Komodo GPTQ/AWQ native int4 prepack. |
| `npu_weight_quant_batchmatmul` | Komodo native int4 inference matmul. |
| `npu_prompt_flash_attention` | NPU `flash_attention_2` prompt availability gate. |
| `npu_incre_flash_attention` | NPU `flash_attention_2` incremental decode availability gate. |

The scanned public operator names are:

```text
attention_worker_scheduler
attention_worker_scheduler_
batch_norm_gather_stats_update
batch_norm_reduce
copy_memory_
dropout_with_byte_mask
empty_with_format
empty_with_swapped_memory
fast_gelu
ffn_worker_scheduler
ffn_worker_scheduler_
fft_c2r_backward
fft_r2c_backward
fused_cross_entropy_loss_with_max_sum
fused_linear_cross_entropy_loss_with_max_sum_grad
fused_linear_online_max_sum
get_npu_format
get_storage_size
kl_div_backward
l1_loss_backward
matmul_double_backward
npu_add_layer_norm
npu_add_layer_norm_backward
npu_add_rms_norm
npu_add_rms_norm_cast
npu_add_rms_norm_dynamic_quant
npu_add_rms_norm_quant
npu_advance_step_flashattn
npu_all_gather_base_mm
npu_alloc_float_status
npu_alltoallv_gmm
npu_anchor_response_flags
npu_anti_quant
npu_apply_adam
npu_apply_adam_w
npu_apply_rotary_pos_emb
npu_attention_update
npu_attention_worker_combine
npu_attn_softmax_
npu_attn_softmax_backward_
npu_batch_gather_matmul
npu_batch_gather_matmul_
npu_batch_nms
npu_bert_apply_adam
npu_binary_cross_entropy_with_logits_backward
npu_bmmV2
npu_bmm_v2_mat1_backward
npu_bmm_v2_mat2_backward
npu_bounding_box_decode
npu_bounding_box_encode
npu_broadcast
npu_change_data_ptr
npu_ciou
npu_ciou_backward
npu_clear_float_status
npu_clipped_swiglu
npu_confusion_transpose
npu_confusion_transpose_backward
npu_conv2d
npu_conv2d_backward
npu_conv3d
npu_conv3d_backward
npu_conv_transpose2d
npu_conv_transpose2d_backward
npu_conv_transpose3d_backward
npu_convert_weight_to_int4pack
npu_convolution
npu_convolution_backward
npu_convolution_transpose
npu_convolution_transpose_backward
npu_cross_entropy_loss
npu_cross_entropy_loss_backward
npu_deep_norm
npu_deep_norm_backward
npu_deformable_conv2d
npu_deformable_conv2dbk
npu_dequant_bias
npu_dequant_rope_quant_kvcache
npu_dequant_swiglu_quant
npu_diou
npu_diou_backward
npu_dropout_backward
npu_dropout_do_mask
npu_dropout_gen_mask
npu_dropout_with_add_softmax
npu_dropout_with_add_softmax_backward
npu_dtype_cast
npu_dtype_cast_
npu_dtype_cast_backward
npu_dynamic_block_quant
npu_dynamic_quant
npu_dynamic_quant_asymmetric
npu_fast_gelu
npu_fast_gelu_backward
npu_ffn
npu_ffn_worker_batching
npu_format_cast
npu_format_cast_
npu_fused_attention_layernorm_qkv_fwd
npu_fused_attention_qkv_grad
npu_fused_attention_score
npu_fused_attention_score_backward
npu_fused_attention_score_fwd
npu_fused_attention_score_grad
npu_fused_infer_attention_score
npu_fused_infer_attention_score_v2
npu_fusion_attention
npu_fusion_attention_grad
npu_fusion_attention_grad_v2
npu_fusion_attention_v2
npu_gather_backward
npu_gather_sparse_index
npu_gather_sparse_index_backward
npu_geglu
npu_geglu_grad
npu_gelu
npu_gelu_backward
npu_gelu_mul
npu_gemma_rms_norm
npu_get_float_status
npu_giou
npu_giou_backward
npu_gmm_alltoallv
npu_grid_assign_positive
npu_group_norm_silu
npu_group_norm_swish
npu_group_norm_swish_grad
npu_group_quant
npu_grouped_matmul
npu_grouped_matmul_add
npu_grouped_matmul_add_
npu_grouped_matmul_finalize_routing
npu_grouped_matmul_swiglu_quant
npu_grouped_matmul_swiglu_quant_v2
npu_gru
npu_gru_backward
npu_hans_decode
npu_hans_encode
npu_ifmr
npu_incre_flash_attention
npu_indexing
npu_interleave_rope
npu_iou
npu_kronecker_quant
npu_kv_quant_sparse_flash_attention
npu_kv_rmsnorm_rope_cache
npu_kv_rmsnorm_rope_cache_v2
npu_kv_rmsnorm_rope_cache_v2_functional
npu_layer_norm_eval
npu_layernorm_grad
npu_lightning_indexer
npu_lightning_indexer_grad
npu_linear
npu_linear_backward
npu_lstm
npu_lstm_backward
npu_lstm_cell
npu_lstm_cell_backward
npu_lstm_data
npu_lstm_data_backward
npu_masked_fill_range
npu_masked_softmax_with_rel_pos_bias
npu_max
npu_max_backward
npu_min
npu_min_backward
npu_mish
npu_mish_backward
npu_mla_prolog
npu_mla_prolog_v2
npu_mla_prolog_v3
npu_mla_prolog_v3_functional
npu_mm_all_reduce_base
npu_mm_reduce_scatter_base
npu_moe_compute_expert_tokens
npu_moe_distribute_combine
npu_moe_distribute_combine_add_rms_norm
npu_moe_distribute_combine_v2
npu_moe_distribute_dispatch
npu_moe_distribute_dispatch_v2
npu_moe_finalize_routing
npu_moe_gating_top_k
npu_moe_gating_top_k_softmax
npu_moe_init_routing
npu_moe_init_routing_quant
npu_moe_init_routing_v2
npu_moe_re_routing
npu_moe_token_permute
npu_moe_token_permute_grad
npu_moe_token_permute_with_routing_map
npu_moe_token_permute_with_routing_map_grad
npu_moe_token_unpermute
npu_moe_token_unpermute_grad
npu_moe_token_unpermute_with_routing_map
npu_moe_token_unpermute_with_routing_map_grad
npu_moe_update_expert
npu_mrope
npu_multi_head_attention
npu_multi_head_attention_backward
npu_nms_rotated
npu_nms_v4
npu_nms_with_mask
npu_normalize_batch
npu_nsa_compress
npu_nsa_compress_attention
npu_nsa_compress_attention_infer
npu_nsa_compress_grad
npu_nsa_compress_infer
npu_nsa_select_attention
npu_nsa_select_attention_grad
npu_nsa_select_attention_infer
npu_one_hot
npu_pad
npu_prefetch
npu_prompt_flash_attention
npu_ps_roi_pooling
npu_ps_roi_pooling_backward
npu_ptiou
npu_quant_conv2d
npu_quant_grouped_matmul_dequant
npu_quant_lightning_indexer
npu_quant_matmul
npu_quant_matmul_dequant
npu_quant_matmul_reduce_sum
npu_quant_scatter
npu_quant_scatter_
npu_quantize
npu_random_choice_with_mask
npu_recurrent_gated_delta_rule
npu_recurrent_gated_delta_rule_functional
npu_reshape
npu_rms_norm
npu_rms_norm_backward
npu_rms_norm_quant
npu_roi_align
npu_roi_alignbk
npu_rope_quant_kvcache
npu_rotary_mul
npu_rotary_mul_backward
npu_rotated_box_decode
npu_rotated_box_encode
npu_rotated_iou
npu_rotated_overlaps
npu_scaled_masked_softmax
npu_scaled_masked_softmax_backward
npu_scatter
npu_scatter_list
npu_scatter_list_
npu_scatter_nd_update
npu_scatter_nd_update_
npu_scatter_pa_kv_cache
npu_scatter_pa_kv_cache_functional
npu_sign_bits_pack
npu_sign_bits_unpack
npu_silu
npu_silu_
npu_silu_backward
npu_sim_exponential_
npu_slice
npu_softmax_cross_entropy_with_logits
npu_softmax_cross_entropy_with_logits_backward
npu_sort_v2
npu_sparse_flash_attention
npu_sparse_flash_attention_grad
npu_sparse_lightning_indexer_grad_kl_loss
npu_stride_add
npu_stride_copy
npu_sub_sample
npu_swiglu
npu_swiglu_backward
npu_swiglu_quant
npu_top_k_top_p
npu_top_k_top_p_sample
npu_trans_quant_param
npu_transpose
npu_transpose_batchmatmul
npu_view_copy
npu_weight_quant_batchmatmul
npu_yolo_boxes_encode
obfuscation_calculate
obfuscation_finalize
obfuscation_initialize
one_
repeat_interleave_backward_int
repeat_interleave_backward_tensor
scatter_update
scatter_update_
slow_conv_dilated2d_backward
slow_conv_transpose2d_backward
stft_backward
unsafe_empty_with_format
```

Schemas for the direct GPTQModel probes/calls:

```text
npu::npu_convert_weight_to_int4pack(Tensor weight, int inner_k_tiles=0) -> Tensor

npu::npu_weight_quant_batchmatmul(Tensor x, Tensor weight, Tensor antiquant_scale,
    Tensor? antiquant_offset=None, Tensor? quant_scale=None,
    Tensor? quant_offset=None, Tensor? bias=None,
    int antiquant_group_size=0, int inner_precise=0) -> Tensor

npu::npu_prompt_flash_attention(Tensor query, Tensor key, Tensor value, *,
    Tensor? padding_mask=None, Tensor? atten_mask=None, Tensor? pse_shift=None,
    int[]? actual_seq_lengths=None, Tensor? deq_scale1=None,
    Tensor? quant_scale1=None, Tensor? deq_scale2=None,
    Tensor? quant_scale2=None, Tensor? quant_offset2=None, int num_heads=1,
    float scale_value=1., int pre_tokens=2147483647, int next_tokens=0,
    str input_layout="BSH", int num_key_value_heads=0,
    int[]? actual_seq_lengths_kv=None, int sparse_mode=0) -> Tensor

npu::npu_incre_flash_attention(Tensor query, Tensor key, Tensor value, *,
    Tensor? padding_mask=None, Tensor? atten_mask=None, Tensor? pse_shift=None,
    SymInt[]? actual_seq_lengths=None, Tensor? antiquant_scale=None,
    Tensor? antiquant_offset=None, Tensor? block_table=None,
    Tensor? dequant_scale1=None, Tensor? quant_scale1=None,
    Tensor? dequant_scale2=None, Tensor? quant_scale2=None,
    Tensor? quant_offset2=None, Tensor? kv_padding_size=None, int num_heads=1,
    float scale_value=1., str input_layout="BSH", int num_key_value_heads=0,
    int block_size=0, int inner_precise=1) -> Tensor
```

## Unsupported CUDA APIs

| CUDA API | NPU equivalent | Notes |
| --- | --- | --- |
| `torch.cuda.comm.gather` | - | Unsupported. |
| `torch.cuda.comm.scatter` | - | Unsupported. |
| `torch.cuda.get_device_capability` | - | Unsupported because NPU has no matching CUDA capability concept. |
| `torch.cuda.memory_usage` | - | Unsupported. |

## Supported CUDA API Equivalents

| CUDA API | NPU equivalent | Notes |
| --- | --- | --- |
| `torch.cuda.StreamContext` | `torch.npu.StreamContext` | - |
| `torch.cuda.can_device_access_peer` | `torch_npu.npu.can_device_access_peer` | - |
| `torch.cuda.current_blas_handle` | `torch_npu.npu.current_blas_handle` | - |
| `torch.cuda.current_device` | `torch_npu.npu.current_device` | - |
| `torch.cuda.current_stream` | `torch_npu.npu.current_stream` | If no device is set, this may implicitly initialize device 0. |
| `torch.cuda.default_stream` | `torch_npu.npu.default_stream` | If no device is set, this may implicitly initialize device 0. |
| `torch.cuda.device` | `torch_npu.npu.device` | - |
| `torch.cuda.device_count` | `torch_npu.npu.device_count` | - |
| `torch.cuda.device_of` | `torch_npu.npu.device_of` | - |
| `torch.cuda.get_device_name` | `torch_npu.npu.get_device_name` | - |
| `torch.cuda.get_device_properties` | `torch_npu.npu.get_device_properties` | Only `name`, `total_memory`, `L2_cache_size`, `cube_core_num`, and `vector_core_num` are populated; other CUDA properties are empty. |
| `torch.cuda.get_sync_debug_mode` | `torch_npu.npu.get_sync_debug_mode` | - |
| `torch.cuda.init` | `torch_npu.npu.init` | - |
| `torch.cuda.ipc_collect` | `torch_npu.npu.ipc_collect` | - |
| `torch.cuda.is_available` | `torch_npu.npu.is_available` | - |
| `torch.cuda.is_initialized` | `torch_npu.npu.is_initialized` | - |
| `torch.cuda.set_device` | `torch_npu.npu.set_device` | - |
| `torch.cuda.set_stream` | `torch_npu.npu.set_stream` | - |
| `torch.cuda.set_sync_debug_mode` | `torch_npu.npu.set_sync_debug_mode` | - |
| `torch.cuda.stream` | `torch_npu.npu.stream` | - |
| `torch.cuda.synchronize` | `torch_npu.npu.synchronize` | - |
| `torch.cuda.utilization` | `torch_npu.npu.utilization` | - |
| `torch.cuda.get_rng_state` | `torch_npu.npu.get_rng_state` | - |
| `torch.cuda.set_rng_state` | `torch_npu.npu.set_rng_state` | - |
| `torch.cuda.set_rng_state_all` | `torch_npu.npu.set_rng_state_all` | - |
| `torch.cuda.manual_seed` | `torch_npu.npu.manual_seed` | - |
| `torch.cuda.manual_seed_all` | `torch_npu.npu.manual_seed_all` | - |
| `torch.cuda.seed` | `torch_npu.npu.seed` | - |
| `torch.cuda.seed_all` | `torch_npu.npu.seed_all` | - |
| `torch.cuda.initial_seed` | `torch_npu.npu.initial_seed` | - |
| `torch.cuda.Stream` | `torch_npu.npu.Stream` | - |
| `torch.cuda.Stream.wait_stream` | `torch_npu.npu.Stream.wait_stream` | - |
| `torch.cuda.Event` | `torch_npu.npu.Event` | - |
| `torch.cuda.Event.elapsed_time` | `torch_npu.npu.Event.elapsed_time` | - |
| `torch.cuda.Event.query` | `torch_npu.npu.Event.query` | - |
| `torch.cuda.Event.wait` | `torch_npu.npu.Event.wait` | - |
| `torch.cuda.is_current_stream_capturing` | `torch.npu.is_current_stream_capturing` | - |
| `torch.cuda.graph_pool_handle` | `torch.npu.graph_pool_handle` | Inference only; training is unsupported. |
| `torch.cuda.CUDAGraph` | `torch.npu.NPUGraph` | Inference only; training is unsupported. |
| `torch.cuda.CUDAGraph.capture_begin` | `torch.npu.NPUGraph.capture_begin` | Inference only; training is unsupported. |
| `torch.cuda.CUDAGraph.capture_end` | `torch.npu.NPUGraph.capture_end` | Inference only; training is unsupported. |
| `torch.cuda.CUDAGraph.debug_dump` | `torch.npu.NPUGraph.debug_dump` | Inference only; training is unsupported. Dump output is JSON. |
| `torch.cuda.CUDAGraph.pool` | `torch.npu.NPUGraph.pool` | Inference only; training is unsupported. |
| `torch.cuda.CUDAGraph.replay` | `torch.npu.NPUGraph.replay` | Inference only; training is unsupported. |
| `torch.cuda.CUDAGraph.reset` | `torch.npu.NPUGraph.reset` | Inference only; training is unsupported. |
| `torch.cuda.graph` | `torch.npu.graph` | Inference only; training is unsupported. |
| `torch.cuda.make_graphed_callables` | `torch.npu.make_graphed_callables` | Inference only; training is unsupported. |
| `torch.cuda.empty_cache` | `torch_npu.npu.empty_cache` | - |
| `torch.cuda.mem_get_info` | `torch_npu.npu.mem_get_info` | - |
| `torch.cuda.memory_stats` | `torch_npu.npu.memory_stats` | - |
| `torch.cuda.memory_summary` | `torch_npu.npu.memory_summary` | - |
| `torch.cuda.memory_allocated` | `torch_npu.npu.memory_allocated` | - |
| `torch.cuda.max_memory_allocated` | `torch_npu.npu.max_memory_allocated` | - |
| `torch.cuda.reset_max_memory_allocated` | `torch_npu.npu.reset_max_memory_allocated` | - |
| `torch.cuda.memory_reserved` | `torch_npu.npu.memory_reserved` | - |
| `torch.cuda.max_memory_reserved` | `torch_npu.npu.max_memory_reserved` | - |
| `torch.cuda.set_per_process_memory_fraction` | `torch_npu.npu.set_per_process_memory_fraction` | - |
| `torch.cuda.memory_cached` | `torch_npu.npu.memory_cached` | - |
| `torch.cuda.max_memory_cached` | `torch_npu.npu.max_memory_cached` | - |
| `torch.cuda.reset_max_memory_cached` | `torch_npu.npu.reset_max_memory_cached` | - |
| `torch.cuda.reset_peak_memory_stats` | `torch_npu.npu.reset_peak_memory_stats` | - |
| `torch.cuda.caching_allocator_alloc` | `torch_npu.npu.caching_allocator_alloc` | - |
| `torch.cuda.caching_allocator_delete` | `torch_npu.npu.caching_allocator_delete` | - |
| `torch.cuda.get_allocator_backend` | `torch_npu.npu.get_allocator_backend` | - |
| `torch.cuda.CUDAPluggableAllocator` | `torch_npu.npu.NPUPluggableAllocator` | High-risk allocator API; see Huawei's custom NPU allocator API docs before use. |
| `torch.cuda.change_current_allocator` | `torch_npu.npu.change_current_allocator` | High-risk allocator API; see Huawei's custom current-allocator API docs before use. |
| `torch.cuda._sanitizer.enable_cuda_sanitizer` | `torch_npu.npu._sanitizer.enable_npu_sanitizer` | - |
