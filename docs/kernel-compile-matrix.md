<!-- SPDX-FileCopyrightText: 2026 ModelCloud.ai -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Kernel compile and graph-capture matrix

Every quantized kernel declares two capabilities in its own class body, so
callers never guess from a class name:

- `SUPPORTS_COMPILE_LEVEL` - the highest `torch.compile` level its `forward`
  reaches, measured with `tests/kernels/compile_probe.py`:

  | Level | Probe |
  | --- | --- |
  | `EAGER` | eager forward only; dynamo breaks the graph (or compile is a measured pessimization) |
  | `DYNAMO` | `torch.compile(fullgraph=True, backend="eager")` traces with zero graph breaks |
  | `AOT_EAGER` | `backend="aot_eager"` (meta/fake tensors and `aot_autograd`) |
  | `INDUCTOR` | `backend="inductor"` lowers the forward |
  | `CUDAGRAPH` | `mode="reduce-overhead"` captures the compiled forward as a CUDA graph |

- `SUPPORTS_GRAPH_CAPTURE` - whether the **eager** `forward` is safe inside an
  explicit `torch.cuda.graph(...)` capture: no host synchronisation, no
  tensor-to-host transfer, no Python-side allocation or device decision, no
  data-dependent branch. Independent of dynamo: a custom op without a fake
  implementation can be capture-safe yet untraceable.

`SUPPORTS_COMPILE` remains the separate #4669 policy answer to "is compiling
this forward not a pessimization for static-cache `generate()`". A traceable
kernel can still keep it `False` when eager is faster.

`tests/kernels/test_kernel_compile_protocol.py` enforces that every registered
kernel declares both attributes, that a capture claim has no host-sync
construct in its own `forward`, and that this table matches the code.
Regenerate with:

```bash
python -m tests.kernels.gen_compile_matrix
```

| Kernel | Module | Level | Graph capture | Compiled generation |
| --- | --- | --- | --- | --- |
| `AWQBitBLASMlxQuantLinear` | `mlx` | `EAGER` | no | no |
| `AWQBitBlasKernel` | `bitblas_awq` | `EAGER` | no | no |
| `AWQMarlinMlxQuantLinear` | `mlx` | `EAGER` | no | yes |
| `AllSparkAwqLinear` | `allspark_awq` | `EAGER` | no | no |
| `AllSparkLinear` | `allspark` | `EAGER` | no | no |
| `AwqExllamaV2Linear` | `exllamav2_awq` | `EAGER` | no | no |
| `AwqGEMMLinear` | `gemm_awq` | `EAGER` | no | no |
| `AwqGEMMTritonLinear` | `gemm_awq_triton` | `EAGER` | no | no |
| `AwqGEMVFastLinear` | `gemv_fast_awq` | `EAGER` | no | no |
| `AwqGEMVLinear` | `gemv_awq` | `EAGER` | no | no |
| `AwqGEMVTorchLinear` | `gemv_awq` | `EAGER` | no | no |
| `AwqGEMVTritonLinear` | `gemv_awq` | `EAGER` | no | no |
| `AwqGemvFastMlxQuantLinear` | `mlx` | `EAGER` | no | no |
| `AwqGemvMlxQuantLinear` | `mlx` | `EAGER` | no | no |
| `AwqMacheteLinear` | `machete_awq` | `EAGER` | no | no |
| `AwqMarlinLinear` | `marlin_awq` | `EAGER` | no | no |
| `AwqMlxQuantLinear` | `mlx` | `EAGER` | no | yes |
| `AwqSwordfishLinear` | `swordfish` | `EAGER` | no | no |
| `AwqTorchLinear` | `torch_awq` | `EAGER` | no | yes |
| `BitblasLinear` | `bitblas` | `EAGER` | no | no |
| `BitsAndBytesLinear` | `bitsandbytes` | `EAGER` | no | no |
| `BitsAndBytesMlxQuantLinear` | `mlx` | `EAGER` | no | no |
| `ExllamaV2Linear` | `exllamav2` | `EAGER` | no | no |
| `FP8MlxQuantLinear` | `mlx` | `EAGER` | no | no |
| `GGUFCppKernel` | `gguf_cpp` | `EAGER` | no | no |
| `GGUFCudaKernel` | `gguf_cpp` | `EAGER` | no | no |
| `GGUFMlxQuantLinear` | `mlx` | `EAGER` | no | no |
| `GGUFTorchLinear` | `gguf` | `EAGER` | no | no |
| `GGUFTritonKernel` | `gguf_triton` | `EAGER` | no | no |
| `GPTQBitBLASMlxQuantLinear` | `mlx` | `EAGER` | no | no |
| `GPTQMarlinMlxQuantLinear` | `mlx` | `EAGER` | no | no |
| `LLMAwqLinear` | `gemv_fast_awq` | `EAGER` | no | no |
| `LLMAwqMlxQuantLinear` | `mlx` | `EAGER` | no | no |
| `MacaqLinear` | `macaq` | `EAGER` | no | no |
| `MacaqMlxQuantLinear` | `mlx` | `EAGER` | no | no |
| `MacaqTritonLinear` | `macaq` | `EAGER` | no | no |
| `MacheteLinear` | `machete` | `EAGER` | no | no |
| `MarlinLinear` | `marlin` | `EAGER` | no | no |
| `MlxQuantLinear` | `mlx` | `EAGER` | no | no |
| `NF12Linear` | `nf12` | `EAGER` | no | no |
| `NF12MlxQuantLinear` | `mlx` | `EAGER` | no | no |
| `ParoLinear` | `paroquant` | `EAGER` | no | no |
| `ParoMlxQuantLinear` | `mlx` | `EAGER` | no | no |
| `ParoQuantTritonLinear` | `paroquant_triton` | `EAGER` | no | no |
| `QQQLinear` | `qqq` | `EAGER` | no | no |
| `QQQMlxQuantLinear` | `mlx` | `EAGER` | no | no |
| `QQQTorchLinear` | `qqq` | `EAGER` | no | no |
| `SwordfishLinear` | `swordfish` | `EAGER` | no | no |
| `TTBlockfloatLinear` | `tt_blockfloat` | `EAGER` | no | no |
| `TTNNBlockFloat8Linear` | `ttnn_blockfloat` | `EAGER` | no | no |
| `TTNNLinear` | `ttnn` | `EAGER` | no | no |
| `TorchAtenAwqLinear` | `torch_aten_kernel_awq` | `EAGER` | no | no |
| `TorchAtenLinear` | `torch_aten_kernel` | `EAGER` | no | no |
| `TorchCpuAvx512AwqLinear` | `torch_cpu_avx512_awq` | `EAGER` | no | no |
| `TorchCpuAvx512Linear` | `torch_cpu_avx512` | `EAGER` | no | no |
| `TorchFP8Linear` | `fp8` | `EAGER` | no | no |
| `TorchFusedAwqLinear` | `torch_fused_awq` | `EAGER` | no | no |
| `TorchFusedLinear` | `torch_fused` | `EAGER` | no | no |
| `TorchInt8AwqLinear` | `torch_int8_awq` | `EAGER` | no | no |
| `TorchInt8Linear` | `torch_int8` | `EAGER` | no | no |
| `TorchLinear` | `torch` | `EAGER` | no | no |
| `TritonV2Linear` | `tritonv2` | `EAGER` | no | no |
| `VPTQCudaLinear` | `vptq_cuda` | `EAGER` | no | no |
| `VPTQLinear` | `vptq_torch` | `EAGER` | no | no |
| `W4AFP8Linear` | `w4a_floatx` | `EAGER` | no | no |
| `W4ANVFP4Linear` | `w4a_nvfp4` | `EAGER` | no | no |
| `XLALinear` | `tt_xla` | `EAGER` | no | no |
