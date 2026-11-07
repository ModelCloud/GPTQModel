# Fusion kernel inventory

The native fusion adapters qualify 23 exact GPTQ/AWQ kernel classes on this host.
This inventory accounts for all 57 registry classes and the three specialized
EXL3/FP4 loaders. **Other quantization formats remain ineligible for projection
fusion**; their presence in this inventory is not a numerical qualification.

CPU is AMD Zen 5 with AVX512-BF16; CUDA is RTX 4090 (SM89); ROCm is Radeon AI PRO R9700 (gfx1201).
The numerical and checkpoint evidence is in [device validation](fusion-device-validation.md).
[Raw contracts and dependency/architecture reasons](benchmarks/fusion-kernel-inventory.json)
include every dtype and native request. Unavailable and other-family cases are
not counted as validated. A registry declaration alone is not an executed test.

| Exact class | CPU | RTX 4090 | R9700 |
| --- | --- | --- | --- |
| `AWQBitBLASMlxQuantLinear` | — | — | — |
| `AWQBitBlasKernel` | — | Unavailable | — |
| `AWQMarlinMlxQuantLinear` | — | — | — |
| `AllSparkAwqLinear` | — | Fusion tested | Fusion tested |
| `AllSparkLinear` | — | Fusion tested | Fusion tested |
| `AwqExllamaV2Linear` | — | Fusion tested | — |
| `AwqGEMMLinear` | — | Fusion tested | — |
| `AwqGEMMTritonLinear` | — | Fusion tested | Fusion tested |
| `AwqGEMVFastLinear` | — | Fusion tested | — |
| `AwqGEMVLinear` | — | Fusion tested | — |
| `AwqGEMVTritonLinear` | — | Fusion tested | Fusion tested |
| `AwqGemvFastMlxQuantLinear` | — | — | — |
| `AwqGemvMlxQuantLinear` | — | — | — |
| `AwqMacheteLinear` | — | Unavailable | — |
| `AwqMarlinLinear` | — | Fusion tested | — |
| `AwqMlxQuantLinear` | — | — | — |
| `AwqSwordfishLinear` | — | Unavailable | — |
| `AwqTorchLinear` | Fusion tested | Fusion tested | Fusion tested |
| `BitblasLinear` | — | Unavailable | — |
| `BitsAndBytesLinear` | Other format | Other format | Other format |
| `BitsAndBytesMlxQuantLinear` | — | — | — |
| `ExllamaV2Linear` | — | Fusion tested | — |
| `FP8MlxQuantLinear` | — | — | — |
| `GGUFCppKernel` | Other format | — | — |
| `GGUFCudaKernel` | — | Other format | — |
| `GGUFMlxQuantLinear` | — | — | — |
| `GGUFTorchLinear` | Other format | Other format | Other format |
| `GGUFTritonKernel` | — | Other format | Other format |
| `GPTQBitBLASMlxQuantLinear` | — | — | — |
| `GPTQMarlinMlxQuantLinear` | — | — | — |
| `LLMAwqLinear` | — | Fusion tested | — |
| `LLMAwqMlxQuantLinear` | — | — | — |
| `MacheteLinear` | — | Unavailable | — |
| `MarlinLinear` | — | Fusion tested | — |
| `MlxQuantLinear` | — | — | — |
| `ParoLinear` | Other format | Other format | Other format |
| `ParoMlxQuantLinear` | — | — | — |
| `ParoQuantTritonLinear` | — | Other format | Other format |
| `QQQLinear` | — | Other format | — |
| `QQQMlxQuantLinear` | — | — | — |
| `QQQTorchLinear` | Other format | Other format | Other format |
| `SwordfishLinear` | — | Unavailable | — |
| `TorchAtenAwqLinear` | Fusion tested | — | — |
| `TorchAtenLinear` | Fusion tested | — | — |
| `TorchCpuAvx512AwqLinear` | Fusion tested | — | — |
| `TorchCpuAvx512Linear` | Fusion tested | — | — |
| `TorchFP8Linear` | Other format | Other format | Other format |
| `TorchFusedAwqLinear` | Fusion tested | — | — |
| `TorchFusedLinear` | Fusion tested | — | — |
| `TorchInt8AwqLinear` | Unavailable | — | — |
| `TorchInt8Linear` | Fusion tested | — | — |
| `TorchLinear` | Fusion tested | Fusion tested | Fusion tested |
| `TritonV2Linear` | — | Fusion tested | — |
| `VPTQCudaLinear` | — | Other format | — |
| `VPTQLinear` | Other format | Other format | Other format |
| `W4AFP8Linear` | — | Fusion tested | — |
| `W4ANVFP4Linear` | — | Unavailable | — |

“Fusion tested” refers to the finalized native contract, not every combination
the backend may support. INT4 cases are symmetric, group 128, no activation
order. AllSpark is native eight-bit/per-channel; CPU INT8 is native eight-bit.
Fast AWQ uses its native int16 packing. FP16/BF16 run where declared; AWQ
Triton and fast kernels are FP16-only, CPU AVX512 is BF16-only. Portable AWQ
GQA QKV is explicitly rejected; its gate/up and MHA QKV are tested.

## Explicit exclusions

- BitBLAS GPTQ/AWQ: the installed 0.1.0.post1 wheel needs CUDA 12 libraries in this CUDA 13.2 environment. An isolated CUDA 12.9 runtime/NVRTC probe imports it but construction fails with “could not build a runnable matmul” for FP16 INT4, group 128, FP32 accumulation. Neither adapter is qualified.
- Machete: requires SM90+; Swordfish: Blackwell; W4ANVFP4: SM121+. None is an RTX 4090 path.
- Metal classes require Apple hardware. Retired AWQ INT8 expands lower-bit storage and remains unavailable.
- Bitsandbytes, GGUF, FP8, ParoQuant, QQQ and VPTQ registry kernels have no projection-fusion adapter. Their standalone inference support does not imply fusion eligibility.
- `ExllamaV3Linear`, `ExllamaV3TorchLinear`, and `TorchFP4Linear` use specialized format loaders outside the qlinear registry and have no projection-fusion adapter.
- W4AFP8 qualifies ordinary tensor inputs only. Encoded W4A activation streams and ExLlamaV2 activation ordering require separate executors and are rejected.

## Reproduce the inventory

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python tests/benchmark_fusion_kernel_inventory.py \
  --output-json /tmp/fusion-kernel-inventory-cuda.json

HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 PYTHON_GIL=0 PYTHONPATH=. \
  /opt/venvs/rocm101t/bin/python tests/benchmark_fusion_kernel_inventory.py \
  --output-json /tmp/fusion-kernel-inventory-rocm.json
```

`test_every_registered_gptq_awq_kernel_has_an_explicit_fusion_contract` fails
when a newly registered CPU/CUDA/ROCm GPTQ/AWQ class has neither a native
adapter nor an explicit qualification reason. Positive execution tests use
the exact class and never replace it with a portable implementation.
