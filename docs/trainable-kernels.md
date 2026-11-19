<!-- SPDX-FileCopyrightText: 2026 ModelCloud.ai -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Trainable quant kernels: the supported set

Decision record for #5551. A quant kernel opts into training; it never gets it
by inheritance. The supported trainable set is deliberately small, and every
other kernel is inference-only: it rejects `.train()`, rejects a `trainable=True`
selection, and keeps its forward free of autograd, graph-capture and GIL
bookkeeping that only training would need.

This page is the single source of truth. `tests/test_training_support_contract_5551.py`
fails when a kernel's `SUPPORTS_TRAINING` and this table disagree, so adding or
dropping training support is an explicit edit here, not an accident of
inheritance.

## The trainable set

| Kernel | Method | Devices (`SUPPORTS_DEVICES`) | Why it is in the set |
| --- | --- | --- | --- |
| `TorchLinear` (`gptq_torch`) | GPTQ | CPU, CUDA, ROCm, XPU, NPU, MPS | Portable GPTQ reference; the supported fine-tuning path |
| `TritonV2Linear` (`gptq_triton_v2`) | GPTQ | CUDA | Triton GPTQ with the tiled transpose backward |
| `AwqTorchLinear` (`awq_torch`, HF alias `torch_awq`) | AWQ | CPU, CUDA, ROCm, XPU, NPU, MPS | Portable AWQ GEMM reader; PEFT target names depend on it |
| `AwqGEMMLinear` | AWQ | CUDA | AWQ GEMM with the packed decode backward |
| `AwqGEMMTritonLinear` | AWQ | CUDA, ROCm | Triton AWQ GEMM |
| `AwqGEMVTritonLinear` | AWQ | CUDA, ROCm | Triton AWQ GEMV (`_AwqGemvTritonFn.backward`) |
| `TorchFusedLinear` (`torch_fused`) | GPTQ | CPU, XPU | Intel fused GPTQ |
| `TorchFusedAwqLinear` (`torch_fused`) | AWQ | CPU, XPU | Intel fused AWQ |
| `TorchAtenLinear` (`torch_aten`) | GPTQ | CPU | ATen int4pack GPTQ |
| `TorchAtenAwqLinear` (`torch_aten`) | AWQ | CPU | ATen int4pack AWQ |
| `TorchCpuAvx512Linear` (`gptq_torch_cpu_avx512`) | GPTQ | CPU | AVX512-BF16 GPTQ; PEFT training fix #3480 |
| `TorchCpuAvx512AwqLinear` (`awq_torch_cpu_avx512`) | AWQ | CPU | AVX512-BF16 AWQ |
| `TorchFP8Linear` | FP8 | CPU, CUDA, ROCm, XPU, MPS | FP8 PEFT (W8A16 decode with an input-gradient path) |
| `GGUFTorchLinear` | GGUF | CPU, CUDA, ROCm, XPU, NPU, MPS | Torch GGUF reader used by PEFT recipes |
| `ParoLinear` | Paro | CPU, CUDA, ROCm, XPU, NPU, MPS | Paro keeps its own differentiable rotation parameters |
| `ParoTritonLinear` | Paro | CUDA, ROCm | Triton Paro |

Everything else — Marlin, Machete, ExLlamaV2, ExLlamaV3, QQQ, MACaQ, NF12,
VPTQ, BitBLAS, bitsandbytes, Swordfish, AllSpark, GGUF C++/Triton, MLX, TNN,
TT-XLA, `torch_int8`, `w4a_floatx`, `w4a_nvfp4` and the GEMV-fast family — is
inference-only.

## The rule

1. `BaseQuantLinear.SUPPORTS_TRAINING` is `None`, not `False`. A kernel that
   never states the decision is a contract violation; `None` is not "quietly
   inference-only".
2. A kernel that declares `True` must be in the table above and must be covered
   by a backward test (`tests/test_training_kernel_inplace_4974.py` runs every
   buildable trainable kernel, `tests/test_train_mode_atomicity.py` pins the
   whole-model switch).
3. A kernel that declares `False` must not spend forward-time work on autograd,
   graph-capture or GIL handling. Eval-mode gradients through a frozen operand
   are not part of the contract; `BACKEND.AUTO_TRAINABLE` and the CPU `AUTO`
   trainable preference pick a kernel from the table instead.

## Enforcement points

| Boundary | Behaviour |
| --- | --- |
| `BaseQuantLinear.validate(..., trainable=True)` | Rejects an inference-only kernel with `NotImplementedError` |
| `BaseQuantLinear.train(True)` | Rejects an inference-only kernel and points at `auto_trainable` |
| `BaseQModel.train(True)` | Preflights the whole tree so a rejected switch leaves no partial state |
| `BACKEND.AUTO_TRAINABLE` | Selects only kernels from the table |
| CPU `BACKEND.AUTO` (GPTQ/AWQ) | Prefers a trainable CPU kernel when one is eligible (#3408) |

## Changing the set

Edit the table above, `TRAINABLE_KERNELS` in
`tests/test_training_support_contract_5551.py`, and the kernel's
`SUPPORTS_TRAINING` in the same commit. Dropping a kernel from the set is a
breaking change for its users: announce it and keep a migration note on the
kernel's class docstring.
