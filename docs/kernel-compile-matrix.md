<!-- SPDX-FileCopyrightText: 2026 ModelCloud.ai -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Kernel compile and graph-capture matrix

Every quantized kernel declares two capabilities in its own class body, so
callers never guess from a class name:

- `SUPPORTS_COMPILE_LEVEL` - the highest `torch.compile` level its `forward`
  reaches, measured with `tests/kernels/compile_probe.py`:

  | Level | Probe |
  | --- | --- |
  | `NOT_APPLICABLE` | not a torch module (MLX/TTNN/XLA); its own compiler owns the graph |
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

The levels are the `torch.compile` API surface of torch 2.13+ (the project's
floor is 2.13 for Python < 3.11 and 2.15 otherwise): `backend="eager"`,
`backend="aot_eager"`, `backend="inductor"`, and `mode="reduce-overhead"`.
`max-autotune` is a tuning mode of `INDUCTOR`, not a separate capability, so it
is not a level. A declaration records the torch version it was measured on.

`SUPPORTS_COMPILE` remains the separate #4669 policy answer to "is compiling
this forward not a pessimization for static-cache `generate()`". A traceable
kernel can still keep it `False` when eager is faster.

Two MLX kernels (`AwqMlxQuantLinear`, `AWQMarlinMlxQuantLinear`) inherit
`SUPPORTS_COMPILE = True` from `AwqTorchLinear`. That flag is not consulted on
the MLX runtime (the model is not a `torch.nn.Module`), so their
`torch.compile` level is still `NOT_APPLICABLE`; the column shows the inherited
policy flag for completeness.

`tests/kernels/test_kernel_compile_protocol.py` enforces that every registered
kernel declares both attributes, that a capture claim has no host-sync
construct in its own `forward`, and that this table matches the code.
Regenerate with:

```bash
python -m tests.kernels.gen_compile_matrix
```

| Kernel | Module | Level | Graph capture | Compiled generation |
| --- | --- | --- | --- | --- |
| `AWQBitBLASMlxQuantLinear` | `mlx` | `NOT_APPLICABLE` | no | no |
| `AWQBitBlasKernel` | `bitblas_awq` | `EAGER` | no | no |
| `AWQMarlinMlxQuantLinear` | `mlx` | `NOT_APPLICABLE` | no | yes |
| `AllSparkAwqLinear` | `allspark_awq` | `EAGER` | yes | no |
| `AllSparkLinear` | `allspark` | `EAGER` | yes | no |
| `AwqExllamaV2Linear` | `exllamav2_awq` | `EAGER` | yes | no |
| `AwqGEMMLinear` | `gemm_awq` | `EAGER` | yes | no |
| `AwqGEMMTritonLinear` | `gemm_awq_triton` | `EAGER` | yes | no |
| `AwqGEMVFastLinear` | `gemv_fast_awq` | `EAGER` | yes | no |
| `AwqGEMVFastTorchLinear` | `gemv_fast_awq` | `AOT_EAGER` | no | no |
| `AwqGEMVLinear` | `gemv_awq` | `EAGER` | yes | no |
| `AwqGEMVTorchLinear` | `gemv_awq` | `CUDAGRAPH` | yes | no |
| `AwqGEMVTritonLinear` | `gemv_awq` | `CUDAGRAPH` | yes | no |
| `AwqGemvFastMlxQuantLinear` | `mlx` | `NOT_APPLICABLE` | no | no |
| `AwqGemvMlxQuantLinear` | `mlx` | `NOT_APPLICABLE` | no | no |
| `AwqMacheteLinear` | `machete_awq` | `EAGER` | no | no |
| `AwqMarlinLinear` | `marlin_awq` | `EAGER` | yes | no |
| `AwqMlxQuantLinear` | `mlx` | `NOT_APPLICABLE` | no | yes |
| `AwqSwordfishLinear` | `swordfish` | `EAGER` | no | no |
| `AwqTorchLinear` | `torch_awq` | `CUDAGRAPH` | no | yes |
| `BitblasLinear` | `bitblas` | `EAGER` | no | no |
| `BitsAndBytesLinear` | `bitsandbytes` | `EAGER` | no | no |
| `BitsAndBytesMlxQuantLinear` | `mlx` | `NOT_APPLICABLE` | no | no |
| `ExllamaV2Linear` | `exllamav2` | `EAGER` | yes | no |
| `FP8MlxQuantLinear` | `mlx` | `NOT_APPLICABLE` | no | no |
| `GGUFCppKernel` | `gguf_cpp` | `EAGER` | no | no |
| `GGUFCudaKernel` | `gguf_cpp` | `EAGER` | no | no |
| `GGUFMlxQuantLinear` | `mlx` | `NOT_APPLICABLE` | no | no |
| `GGUFTorchLinear` | `gguf` | `EAGER` | yes | no |
| `GGUFTritonKernel` | `gguf_triton` | `EAGER` | yes | no |
| `GPTQBitBLASMlxQuantLinear` | `mlx` | `NOT_APPLICABLE` | no | no |
| `GPTQMarlinMlxQuantLinear` | `mlx` | `NOT_APPLICABLE` | no | no |
| `LLMAwqLinear` | `gemv_fast_awq` | `EAGER` | yes | no |
| `LLMAwqMlxQuantLinear` | `mlx` | `NOT_APPLICABLE` | no | no |
| `LLMAwqTorchLinear` | `gemv_fast_awq` | `AOT_EAGER` | no | no |
| `MacaqLinear` | `macaq` | `CUDAGRAPH` | no | no |
| `MacaqMlxQuantLinear` | `mlx` | `NOT_APPLICABLE` | no | no |
| `MacaqTritonLinear` | `macaq` | `EAGER` | yes | no |
| `MacheteLinear` | `machete` | `EAGER` | no | no |
| `MarlinLinear` | `marlin` | `EAGER` | yes | no |
| `MlxQuantLinear` | `mlx` | `NOT_APPLICABLE` | no | no |
| `NF12Linear` | `nf12` | `EAGER` | no | no |
| `NF12MlxQuantLinear` | `mlx` | `NOT_APPLICABLE` | no | no |
| `ParoLinear` | `paro` | `EAGER` | yes | no |
| `ParoMlxQuantLinear` | `mlx` | `NOT_APPLICABLE` | no | no |
| `ParoTritonLinear` | `paro_triton` | `CUDAGRAPH` | yes | no |
| `QQQLinear` | `qqq` | `EAGER` | no | no |
| `QQQMlxQuantLinear` | `mlx` | `NOT_APPLICABLE` | no | no |
| `QQQTorchLinear` | `qqq` | `CUDAGRAPH` | no | no |
| `SwordfishLinear` | `swordfish` | `EAGER` | no | no |
| `TTBlockfloatLinear` | `tt_blockfloat` | `EAGER` | no | no |
| `TTNNBlockFloat8Linear` | `ttnn_blockfloat` | `NOT_APPLICABLE` | no | no |
| `TTNNLinear` | `ttnn` | `NOT_APPLICABLE` | no | no |
| `TorchAtenAwqLinear` | `torch_aten_kernel_awq` | `EAGER` | no | no |
| `TorchAtenLinear` | `torch_aten_kernel` | `EAGER` | no | no |
| `TorchCpuAvx512AwqLinear` | `torch_cpu_avx512_awq` | `EAGER` | no | no |
| `TorchCpuAvx512Linear` | `torch_cpu_avx512` | `EAGER` | no | no |
| `TorchFP8Linear` | `fp8` | `CUDAGRAPH` | yes | yes |
| `TorchFusedAwqLinear` | `torch_fused_awq` | `EAGER` | no | no |
| `TorchFusedLinear` | `torch_fused` | `EAGER` | no | no |
| `TorchInt8AwqLinear` | `torch_int8_awq` | `EAGER` | no | no |
| `TorchInt8Linear` | `torch_int8` | `EAGER` | no | no |
| `TorchLinear` | `torch` | `EAGER` | yes | no |
| `TorchQuantEmbeddings` | `torch` | `EAGER` | no | no |
| `TritonV2Linear` | `tritonv2` | `EAGER` | yes | no |
| `VPTQCudaLinear` | `vptq_cuda` | `EAGER` | yes | no |
| `VPTQLinear` | `vptq_torch` | `CUDAGRAPH` | yes | no |
| `W4AFP8Linear` | `w4a_floatx` | `EAGER` | yes | no |
| `W4ANVFP4Linear` | `w4a_nvfp4` | `EAGER` | yes | no |
| `XLALinear` | `tt_xla` | `NOT_APPLICABLE` | no | no |
