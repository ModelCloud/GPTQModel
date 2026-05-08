# Ascend NPU CUDA API Equivalents

This file tracks the Huawei Ascend Extension for PyTorch 7.3.0 support status for
PyTorch 2.9.0 `torch.cuda` APIs that have `torch_npu.npu` or `torch.npu`
equivalents.

Source:

- Huawei overview: https://www.hiascend.com/document/detail/zh/Pytorch/730/apiref/PyTorchNativeapi/docs/zh/native_apis/pytorch_2-9-0/overview.md
- Huawei CUDA mapping page: https://www.hiascend.com/document/detail/zh/Pytorch/730/apiref/PyTorchNativeapi/docs/zh/native_apis/pytorch_2-9-0/torch-cuda.md

Snapshot date: 2026-04-28.

Huawei's overview defines four cases: supported with no restrictions, supported
with restrictions, unsupported, and present in upstream PyTorch but not listed in
Huawei's support table. Treat unlisted APIs as unverified on NPU.

For supported CUDA namespace APIs, replace `torch.cuda.*` with the listed NPU
form. Huawei documents `torch_npu.npu.*` and `torch.npu.*` as functionally
equivalent forms for these APIs.

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
