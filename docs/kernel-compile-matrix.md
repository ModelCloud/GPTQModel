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

- `SUPPORTS_COMPILE_LEVEL_BY_DEVICE` - optional per-device override, keyed by
  `torch.device.type`. The "Level" column is the default-device declaration;
  the "CPU level" column is what a CPU-resident instance declares. A kernel
  that runs on both devices records the CPU ceiling here when it differs (for
  example `TorchLinear`: CUDA `EAGER` because its streaming path allocates a
  `torch.cuda.Stream`, CPU `INDUCTOR`). Only the dict in the kernel's own class
  body applies, so an MLX or Triton subclass never inherits its torch parent's
  CPU answer.

A class declaration answers for the kernel's default runtime state. Where one
kernel class serves several stored formats, an instance may refine it through
`supports_graph_capture()`, and `graph_capture_safe(instance)` returns that
answer. `GGUFTorchLinear` is the one case today: its device-side dequantizers
(`Q4_0`, `Q8_0`, `Q4_K`, `Q5_K`, `Q6_K` - including the default `q_k_m`) are
capturable, while `Q2_K`, `Q3_K` and the remaining codebook types dequantize
through numpy on the host and cannot be captured (#5014).

The levels are the `torch.compile` API surface of torch 2.13+ (the project's
floor is 2.13 for Python < 3.11 and 2.15 otherwise): `backend="eager"`,
`backend="aot_eager"`, `backend="inductor"`, and `mode="reduce-overhead"`.
`max-autotune` is a tuning mode of `INDUCTOR`, not a separate capability, so it
is not a level. A declaration records the torch version it was measured on.
The CPU levels in this table were re-measured on
`torch 2.16.0.dev20261021+cpu`; the default-device levels keep their recorded
GPU measurements.

`SUPPORTS_COMPILE` remains the separate #4669 policy answer to "is compiling
this forward not a pessimization for static-cache `generate()`". A traceable
kernel can still keep it `False` when eager is faster, but then it has to
record the measured loss in `SUPPORTS_COMPILE_EAGER_REASON` (the "Policy
reason" column): a level at or above `INDUCTOR` is the documented bar for a
compiled win, so the two declarations cannot drift apart silently (#4984).

Two MLX kernels (`AwqMlxQuantLinear`, `AWQMarlinMlxQuantLinear`) inherit
`SUPPORTS_COMPILE = True` from `AwqTorchLinear`. That flag is not consulted on
the MLX runtime (the model is not a `torch.nn.Module`), so their
`torch.compile` level is still `NOT_APPLICABLE`; the column shows the inherited
policy flag for completeness.

`tests/kernels/test_kernel_compile_protocol.py` enforces that every registered
kernel declares both attributes, that a capture claim has no host-sync
construct in its own `forward`, that a compile-capable opt-out records its
reason, and that this table matches the code.
Regenerate with:

```bash
python -m tests.kernels.gen_compile_matrix
```

| Kernel | Module | Level | CPU level | Graph capture | Compiled generation | Policy reason |
| --- | --- | --- | --- | --- | --- | --- |
| `AWQBitBLASMlxQuantLinear` | `mlx` | `NOT_APPLICABLE` | `NOT_APPLICABLE` | no | no | - |
| `AWQBitBlasKernel` | `bitblas_awq` | `EAGER` | `EAGER` | no | no | - |
| `AWQMarlinMlxQuantLinear` | `mlx` | `NOT_APPLICABLE` | `NOT_APPLICABLE` | no | yes | - |
| `AllSparkAwqLinear` | `allspark_awq` | `EAGER` | `EAGER` | yes | no | - |
| `AllSparkLinear` | `allspark` | `EAGER` | `EAGER` | yes | no | - |
| `AwqExllamaV2Linear` | `exllamav2_awq` | `EAGER` | `EAGER` | yes | no | - |
| `AwqGEMMLinear` | `gemm_awq` | `EAGER` | `EAGER` | yes | no | - |
| `AwqGEMMTritonLinear` | `gemm_awq_triton` | `EAGER` | `EAGER` | yes | no | - |
| `AwqGEMVFastLinear` | `gemv_fast_awq` | `EAGER` | `EAGER` | yes | no | - |
| `AwqGEMVFastTorchLinear` | `gemv_fast_awq` | `INDUCTOR` | `INDUCTOR` | no | no | CPU-only portable GEMV_FAST fallback; no measured static-cache win (#4984) |
| `AwqGEMVLinear` | `gemv_awq` | `EAGER` | `EAGER` | yes | no | - |
| `AwqGEMVTorchLinear` | `gemv_awq` | `INDUCTOR` | `INDUCTOR` | yes | no | CPU-only portable GEMV fallback; no measured static-cache win (#4984) |
| `AwqGEMVTritonLinear` | `gemv_awq` | `CUDAGRAPH` | `CUDAGRAPH` | yes | no | explicit-selection-only GEMV decode kernel; no static-cache generate win measured (#4984) |
| `AwqGemvFastMlxQuantLinear` | `mlx` | `NOT_APPLICABLE` | `NOT_APPLICABLE` | no | no | - |
| `AwqGemvMlxQuantLinear` | `mlx` | `NOT_APPLICABLE` | `NOT_APPLICABLE` | no | no | - |
| `AwqMacheteLinear` | `machete_awq` | `EAGER` | `EAGER` | no | no | - |
| `AwqMarlinLinear` | `marlin_awq` | `EAGER` | `EAGER` | yes | no | - |
| `AwqMlxQuantLinear` | `mlx` | `NOT_APPLICABLE` | `NOT_APPLICABLE` | no | yes | - |
| `AwqSwordfishLinear` | `swordfish` | `EAGER` | `EAGER` | no | no | - |
| `AwqTorchLinear` | `torch_awq` | `CUDAGRAPH` | `INDUCTOR` | no | yes | - |
| `BitblasLinear` | `bitblas` | `EAGER` | `EAGER` | no | no | - |
| `BitsAndBytesLinear` | `bitsandbytes` | `CUDAGRAPH` | `INDUCTOR` | yes | yes | - |
| `BitsAndBytesMlxQuantLinear` | `mlx` | `NOT_APPLICABLE` | `NOT_APPLICABLE` | no | no | - |
| `ExllamaV2Linear` | `exllamav2` | `EAGER` | `EAGER` | yes | no | - |
| `FP8MlxQuantLinear` | `mlx` | `NOT_APPLICABLE` | `NOT_APPLICABLE` | no | no | - |
| `GGUFCppKernel` | `gguf_cpp` | `EAGER` | `EAGER` | no | no | static-cache generate() auto-compile stays off: no CPU win was measured for this kernel in this change; the INDUCTOR level is opt-in via optimize() (#4984 policy unchanged). |
| `GGUFCudaKernel` | `gguf_cpp` | `EAGER` | `EAGER` | no | no | static-cache generate() auto-compile stays off: no CPU win was measured for this kernel in this change; the INDUCTOR level is opt-in via optimize() (#4984 policy unchanged). |
| `GGUFMlxQuantLinear` | `mlx` | `NOT_APPLICABLE` | `NOT_APPLICABLE` | no | no | - |
| `GGUFTorchLinear` | `gguf` | `INDUCTOR` | `INDUCTOR` | yes | no | static-cache generate() auto-compile stays off: no CPU win was measured for this kernel in this change; the INDUCTOR level is opt-in via optimize() (#4984 policy unchanged). |
| `GGUFTritonKernel` | `gguf_triton` | `EAGER` | `EAGER` | yes | no | static-cache generate() auto-compile stays off: no CPU win was measured for this kernel in this change; the INDUCTOR level is opt-in via optimize() (#4984 policy unchanged). |
| `GPTQBitBLASMlxQuantLinear` | `mlx` | `NOT_APPLICABLE` | `NOT_APPLICABLE` | no | no | - |
| `GPTQMarlinMlxQuantLinear` | `mlx` | `NOT_APPLICABLE` | `NOT_APPLICABLE` | no | no | - |
| `LLMAwqLinear` | `gemv_fast_awq` | `EAGER` | `EAGER` | yes | no | - |
| `LLMAwqMlxQuantLinear` | `mlx` | `NOT_APPLICABLE` | `NOT_APPLICABLE` | no | no | - |
| `LLMAwqTorchLinear` | `gemv_fast_awq` | `INDUCTOR` | `INDUCTOR` | no | no | CPU-only portable GEMV_FAST fallback; no measured static-cache win (#4984) |
| `MacaqLinear` | `macaq` | `CUDAGRAPH` | `CUDAGRAPH` | no | yes | - |
| `MacaqMlxQuantLinear` | `mlx` | `NOT_APPLICABLE` | `NOT_APPLICABLE` | no | no | - |
| `MacaqTritonLinear` | `macaq` | `EAGER` | `EAGER` | yes | no | - |
| `MacheteLinear` | `machete` | `EAGER` | `EAGER` | no | no | - |
| `MarlinLinear` | `marlin` | `EAGER` | `EAGER` | yes | no | - |
| `MlxQuantLinear` | `mlx` | `NOT_APPLICABLE` | `NOT_APPLICABLE` | no | no | - |
| `NF12Linear` | `nf12` | `EAGER` | `INDUCTOR` | no | no | static-cache generate() auto-compile stays off: no CPU win was measured for this kernel in this change; the INDUCTOR level is opt-in via optimize() (#4984 policy unchanged). |
| `NF12MlxQuantLinear` | `mlx` | `NOT_APPLICABLE` | `NOT_APPLICABLE` | no | no | - |
| `ParoLinear` | `paro` | `EAGER` | `INDUCTOR` | yes | no | static-cache generate() auto-compile stays off: no CPU win was measured for this kernel in this change; the INDUCTOR level is opt-in via optimize() (#4984 policy unchanged). |
| `ParoMlxQuantLinear` | `mlx` | `NOT_APPLICABLE` | `NOT_APPLICABLE` | no | no | - |
| `ParoTritonLinear` | `paro_triton` | `CUDAGRAPH` | `CUDAGRAPH` | yes | no | single-layer trace is clean, but static-cache generate recompiles per step and is 1.3-8x slower than eager (#4984) |
| `QQQLinear` | `qqq` | `EAGER` | `EAGER` | no | no | - |
| `QQQMlxQuantLinear` | `mlx` | `NOT_APPLICABLE` | `NOT_APPLICABLE` | no | no | - |
| `QQQTorchLinear` | `qqq` | `CUDAGRAPH` | `INDUCTOR` | no | no | static-cache generate recompiles per step (172 graphs, 24-115 breaks) and is 1.3-8x slower than eager (#4984) |
| `SwordfishLinear` | `swordfish` | `EAGER` | `EAGER` | no | no | - |
| `TTBlockfloatLinear` | `tt_blockfloat` | `EAGER` | `INDUCTOR` | no | no | static-cache generate() auto-compile stays off: the portable CPU block-float decode has no measured static-cache win (#4984). |
| `TTNNBlockFloat8Linear` | `ttnn_blockfloat` | `NOT_APPLICABLE` | `NOT_APPLICABLE` | no | no | - |
| `TTNNLinear` | `ttnn` | `NOT_APPLICABLE` | `NOT_APPLICABLE` | no | no | - |
| `TorchAtenAwqLinear` | `torch_aten_kernel_awq` | `INDUCTOR` | `INDUCTOR` | no | no | static-cache generate() auto-compile stays off: no CPU win was measured for this kernel in this change; the INDUCTOR level is opt-in via optimize() (#4984 policy unchanged). |
| `TorchAtenLinear` | `torch_aten_kernel` | `INDUCTOR` | `INDUCTOR` | no | no | static-cache generate() auto-compile stays off: no CPU win was measured for this kernel in this change; the INDUCTOR level is opt-in via optimize() (#4984 policy unchanged). |
| `TorchCpuAvx512AwqLinear` | `torch_cpu_avx512_awq` | `INDUCTOR` | `INDUCTOR` | no | no | static-cache generate() auto-compile stays off: no CPU win was measured for this kernel in this change; the INDUCTOR level is opt-in via optimize() (#4984 policy unchanged). |
| `TorchCpuAvx512Linear` | `torch_cpu_avx512` | `INDUCTOR` | `INDUCTOR` | no | no | static-cache generate() auto-compile stays off: no CPU win was measured for this kernel in this change; the INDUCTOR level is opt-in via optimize() (#4984 policy unchanged). |
| `TorchFP8Linear` | `fp8` | `CUDAGRAPH` | `INDUCTOR` | yes | yes | - |
| `TorchFusedAwqLinear` | `torch_fused_awq` | `INDUCTOR` | `INDUCTOR` | no | no | static-cache generate() auto-compile stays off: no CPU win was measured for this kernel in this change; the INDUCTOR level is opt-in via optimize() (#4984 policy unchanged). |
| `TorchFusedLinear` | `torch_fused` | `INDUCTOR` | `INDUCTOR` | no | no | static-cache generate() auto-compile stays off: no CPU win was measured for this kernel in this change; the INDUCTOR level is opt-in via optimize() (#4984 policy unchanged). |
| `TorchInt8AwqLinear` | `torch_int8_awq` | `EAGER` | `EAGER` | no | no | - |
| `TorchInt8Linear` | `torch_int8` | `INDUCTOR` | `INDUCTOR` | no | no | static-cache generate() auto-compile stays off: no CPU win was measured for this kernel in this change; the INDUCTOR level is opt-in via optimize() (#4984 policy unchanged). |
| `TorchLinear` | `torch` | `EAGER` | `INDUCTOR` | yes | no | static-cache generate() auto-compile stays off: no CPU win was measured for the portable Torch decode in this change; the INDUCTOR level is opt-in via optimize() (#4984 policy unchanged). |
| `TorchQuantEmbeddings` | `torch` | `EAGER` | `INDUCTOR` | no | no | static-cache generate() auto-compile stays off: no CPU win was measured for this kernel in this change; the INDUCTOR level is opt-in via optimize() (#4984 policy unchanged). |
| `TritonV2Linear` | `tritonv2` | `EAGER` | `EAGER` | yes | no | static-cache generate() auto-compile stays off: no CPU win was measured for the portable Torch decode in this change; the INDUCTOR level is opt-in via optimize() (#4984 policy unchanged). |
| `VPTQCudaLinear` | `vptq_cuda` | `EAGER` | `EAGER` | yes | no | no measured static-cache generate win for the portable torch path; re-measure before opting in (#4984) |
| `VPTQLinear` | `vptq_torch` | `CUDAGRAPH` | `INDUCTOR` | yes | no | no measured static-cache generate win for the portable torch path; re-measure before opting in (#4984) |
| `W4AFP8Linear` | `w4a_floatx` | `EAGER` | `EAGER` | yes | no | - |
| `W4ANVFP4Linear` | `w4a_nvfp4` | `EAGER` | `EAGER` | yes | no | - |
| `XLALinear` | `tt_xla` | `NOT_APPLICABLE` | `NOT_APPLICABLE` | no | no | - |
