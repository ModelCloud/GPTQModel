# Native fusion inference on CPU, CUDA and ROCm

Fusion preserves the optimized backend selected by AUTO. This report qualifies
**23 exact native GPTQ/AWQ kernel classes** on Zen 5, RTX 4090 and Radeon AI PRO
R9700. Fusion remains opt-in. [All 57 registry classes and the EXL3/FP4 loaders](fusion-kernel-inventory.md)
are inventoried; unavailable kernels and other quantization formats are not
reported as validated fusion implementations.

[Raw per-case results](benchmarks/fusion-device-validation.json) include dtypes,
native layouts, oracle errors, baseline variation, storage assertions, and
checkpoint results. [Current Llama latency and drift tables](fusion-planner.md#measured-llama-32-1b-results)
are separate synchronized wall-clock measurements.

## AUTO selection on this host

| Family | GPTQ FP16 | GPTQ BF16 | AWQ FP16 | AWQ BF16 |
| --- | --- | --- | --- | --- |
| Zen 5 CPU | ATen int4pack | AVX512-BF16 | ATen int4pack | AVX512-BF16 |
| RTX 4090 CUDA | Marlin | Marlin | Marlin | Marlin |
| R9700 ROCm | GPTQ Torch | GPTQ Torch | AWQ Triton GEMM | AWQ Torch |

`TorchLinear` on RDNA4 includes architecture-specific packed Triton execution;
the class name does not imply an unaccelerated fallback. Every full-checkpoint
case asserts the selected exact kernel class before and after fusion. No
backend is changed to obtain an eligible fusion group.

CPU/CUDA runtime: Torch 2.15.0+cu132, Python 3.14.4. ROCm runtime: Torch
2.16.0.dev20261025+rocm10.1, HIP 7.16.26385, Python 3.14.10 free-threaded.
CPU is AMD x86_64 Eng Sample 100-000001458-01 with AVX512-BF16; NVIDIA is SM89
and AMD is gfx1201. CPU cases are also executed in the ROCm build.

## Native projection evidence

The actual Llama parent modules use K=512, gate/up widths 1024+1024, GQA widths
512+256+256, and MHA widths 512+512+512. Tests cover 0/1/8/33/257 rows, with
1025 rows for gate/up to cross large-M/dequantization/reduction thresholds.
Each projection starts from independently generated integer codes, logical
zero points, and group/output-varying dyadic scales. FP32 matrix multiplication
forms the oracle; no fused output is reused as an expectation. All elementwise
comparisons use `rtol=atol=2e-3` and preserve the supported activation dtype.

W4AFP8 uses the specified per-token E4M3 quantize/dequantize rule, then FP32
accumulation and FP32 bias before output rounding. A separate uniform-scale
1025-row regression pins that bias-rounding order for both dtypes. CPU INT8
is decoded independently from its finalized native eight-bit codes/scales;
no INT4-to-INT8 expansion is admitted. AllSpark is tested as native eight-bit
per-channel storage, and fast AWQ uses native int16 packing. These are separate
native contracts, not substitutes for symmetric INT4/g128.

| Family | Exact kernel | Tested dtypes | Max projection/oracle difference | Max native-parent off/on difference |
| --- | --- | --- | ---: | ---: |
| cpu | `AwqTorchLinear` | bfloat16/float16 | 0.000244140625 | 0 |
| cpu | `TorchAtenAwqLinear` | bfloat16/float16 | 0.000244140625 | 0 |
| cpu | `TorchAtenLinear` | bfloat16/float16 | 0.000244140625 | 0 |
| cpu | `TorchCpuAvx512AwqLinear` | bfloat16 | 0.001953125 | 0 |
| cpu | `TorchCpuAvx512Linear` | bfloat16 | 0.001953125 | 0 |
| cpu | `TorchFusedAwqLinear` | bfloat16/float16 | 0.000244140625 | 0 |
| cpu | `TorchFusedLinear` | bfloat16/float16 | 0.000244140625 | 0 |
| cpu | `TorchInt8Linear` | bfloat16/float16 | 0.0009765625 | 0 |
| cpu | `TorchLinear` | bfloat16/float16 | 0.000244140625 | 0 |
| cuda | `AllSparkAwqLinear` | bfloat16/float16 | 0.0009765625 | 0 |
| cuda | `AllSparkLinear` | bfloat16/float16 | 0.0009765625 | 0 |
| cuda | `AwqExllamaV2Linear` | bfloat16/float16 | 0.000122070312 | 3.05175781e-05 |
| cuda | `AwqGEMMLinear` | bfloat16/float16 | 0.00048828125 | 0 |
| cuda | `AwqGEMMTritonLinear` | float16 | 0.000122070312 | 1.52587891e-05 |
| cuda | `AwqGEMVFastLinear` | float16 | 0.000122070312 | 0 |
| cuda | `AwqGEMVLinear` | float16 | 0.00048828125 | 0 |
| cuda | `AwqGEMVTritonLinear` | float16 | 0.000122070312 | 0 |
| cuda | `AwqMarlinLinear` | bfloat16/float16 | 0.000244140625 | 7.62939453e-06 |
| cuda | `AwqTorchLinear` | bfloat16/float16 | 0.00048828125 | 7.62939453e-06 |
| cuda | `ExllamaV2Linear` | bfloat16/float16 | 0.000122070312 | 3.05175781e-05 |
| cuda | `LLMAwqLinear` | float16 | 0.000122070312 | 0 |
| cuda | `MarlinLinear` | bfloat16/float16 | 0.000244140625 | 7.62939453e-06 |
| cuda | `TorchLinear` | bfloat16/float16 | 0.00048828125 | 7.62939453e-06 |
| cuda | `TritonV2Linear` | bfloat16/float16 | 0.00048828125 | 7.62939453e-06 |
| cuda | `W4AFP8Linear` | bfloat16/float16 | 0.001953125 | 0 |
| rocm | `AllSparkAwqLinear` | bfloat16/float16 | 0.00048828125 | 6.10351562e-05 |
| rocm | `AllSparkLinear` | bfloat16/float16 | 0.00048828125 | 6.10351562e-05 |
| rocm | `AwqGEMMTritonLinear` | float16 | 0.000122070312 | 7.62939453e-06 |
| rocm | `AwqGEMVTritonLinear` | float16 | 0.000122070312 | 0 |
| rocm | `AwqTorchLinear` | bfloat16/float16 | 0.00048828125 | 1.52587891e-05 |
| rocm | `TorchLinear` | bfloat16/float16 | 0.00048828125 | 1.52587891e-05 |

Portable AWQ GQA QKV is an explicit rejection case; gate/up and MHA QKV are
qualified. AWQ Triton/fast kernels declare FP16 only, CPU AVX512 declares BF16
only. ExLlamaV2 requires canonical input order; W4AFP8 encoded activation
streams remain ineligible. Those limits are not skipped positive qualifications.

## Shape, workspace, descriptor and storage contracts

- Every source must match its finalized native tiles, and the kernel validates the combined output width before installation and cold refresh. A malformed output width is rejected before launch.
- Marlin owns zero-initialized device-sized scratch. Replacement, resizing and dtype changes invalidate its cached executor and rebuild correctly sized zeroed workspace. Scratch contents are not version-guarded because kernels write them.
- AllSpark, ExLlamaV2 and other backends allocate temporary scratch for the consuming launch as their row/output sizes change. ExLlamaV2 descriptors bind finalized shuffled words without a second destructive shuffle; transient direct-call descriptors are freed.
- Group/output-varying scale metadata, in-place mutation through the final sibling, explicit clear, direct source calls while fused, repeated calls, empty inputs, and unfusing are executed for every admitted backend/dtype/profile.
- Fused operands remain contiguous. Source views share those allocations, original native weight objects are released, and no dense/expanded weight cache may remain in owned modules/containers. Unfusing restores exact packed bytes and independent contiguous source allocations.
- CPU affine metadata and legacy/INT8 child operators are rebuilt from current buffers. No separate source weight copies persist after fusion. Native source fallback compaction is temporary.
- Guard invalidation explicitly retires generated namespaces, including on free-threaded Python. Completed compiled-source trace cycles are collected at cold storage boundaries. Native construction failures roll back installed groups and restore a previous plan.

## Real Llama 3.2 1B checkpoints

All **12** AUTO cases pass: GPTQ/AWQ × FP16/BF16 × CPU/CUDA/ROCm. The prompt
is “The capital of France is”; checks compare prefill plus three cached decode
steps and four greedy new tokens at the same dtype in both modes. All cases
retain zero additional packed-weight bytes and stay inside a dtype-derived ULP
envelope of the unfused logits (8 ULP of the peak magnitude). Greedy output is
required to diverge only where the unfused top-2 margin is inside that same
envelope. These are execution regressions, not dataset accuracy or perplexity
measurements.

Identical greedy IDs are **not** asserted, and should not be: fusion emits one
wider GEMM per group, so a backend may legitimately choose a different reduction
tree (split-K/slice count, tile shape) for the new N. Bit-identical output only
ever held for profiles whose widest fused N happened to keep the unfused tile
configuration. On a Qwen2.5-0.5B GPTQ w4a16 g128 sm_120 profile, fusing qkv to
N=1152 is bit-identical to 896+128+128, while fusing gate/up to N=9728 changes
split-K from 2 to 4 slices and drifts 2.1 ULP (bf16) / 2.8 ULP (fp16) — about
1-6% of the error the checkpoint already carries against its fp32 base model.
The same 0.5B checkpoint's *unfused* Marlin kernel disagrees with the
independent `TorchLinear` kernel on 6/10 prompts, i.e. as often as it disagrees
with itself fused, so exact greedy equality measures a reduction order rather
than correctness.

| Family | Method | Dtype | Exact AUTO kernel | Groups | Max logit difference | Relative L2 | Baseline repeat max | Unfuse max |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| cpu | AWQ | bfloat16 | `TorchCpuAvx512AwqLinear` | 32 | 0 | 0 | 0 | 0 |
| cpu | GPTQ | bfloat16 | `TorchCpuAvx512Linear` | 32 | 0 | 0 | 0 | 0 |
| cpu | AWQ | float16 | `TorchAtenAwqLinear` | 32 | 0 | 0 | 0 | 0 |
| cpu | GPTQ | float16 | `TorchAtenLinear` | 32 | 0 | 0 | 0 | 0 |
| cuda | AWQ | bfloat16 | `AwqMarlinLinear` | 32 | 0.34375 | 0.0210424997 | 0 | 0 |
| cuda | GPTQ | bfloat16 | `MarlinLinear` | 32 | 0.265625 | 0.0173433311 | 0 | 0 |
| cuda | AWQ | float16 | `AwqMarlinLinear` | 32 | 0.0361328125 | 0.00252606976 | 0 | 0 |
| cuda | GPTQ | float16 | `MarlinLinear` | 32 | 0.02734375 | 0.00217597047 | 0 | 0 |
| rocm | AWQ | bfloat16 | `AwqTorchLinear` | 16 | 0 | 0 | 0 | 0 |
| rocm | GPTQ | bfloat16 | `TorchLinear` | 32 | 0 | 0 | 0 | 0 |
| rocm | AWQ | float16 | `AwqGEMMTritonLinear` | 32 | 0.056640625 | 0.00340095116 | 0.0375976562 | 0.044921875 |
| rocm | GPTQ | float16 | `TorchLinear` | 32 | 0 | 0 | 0 | 0 |

BF16 Marlin accumulates more whole-model logit drift than FP16 in this
fixture; both stay inside the ULP envelope, and any greedy divergence is
confined to sub-envelope margins. The `0` entries above are single-shape
reductions on those devices and are genuinely bit-exact, not suppressed
tolerances. Projection oracle tolerances remain separate from compounded
full-model differences. ROCm AWQ Triton
split-K atomic reductions also vary between repeated unfused calls. The
table reports that measured baseline variation separately; exact packed-byte
restoration is checked independently, and restored logits must stay within
the measured repeat-noise envelope with exact top-1 agreement. Every other
checkpoint case restores bitwise-identical baseline logits.

## Commands and results

| Run | Passed | Skipped |
| --- | ---: | ---: |
| Native backend matrix, cuda runtime | 146 | 34 |
| Consumer/kernel regressions, cuda runtime | 2032 | 173 |
| Llama AUTO checkpoints, cuda runtime | 8 | 4 |
| Native backend matrix, rocm runtime | 87 | 93 |
| Consumer/kernel regressions, rocm runtime | 583 | 13 |
| Llama AUTO checkpoints, rocm runtime | 4 | 0 |
| Final core/config contracts, CUDA runtime | 101 | 0 |

Matrix skips are the inactive vendor runtime; optional/model/hardware skips
in the wider consumer regression suite are retained with precise pytest reasons.
Other quantization formats are explicitly ineligible rather than counted as
passed or numerically exercised. The [inventory](fusion-kernel-inventory.md)
records BitBLAS construction/dependency gaps and architecture exclusions.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCHDYNAMO_DISABLE=1 \
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. /root/venv/bin/python -m pytest \
  tests/module_tree/test_fusion_kernel_backends.py -q

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCHDYNAMO_DISABLE=1 \
HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 PYTHON_GIL=0 PYTHONPATH=. \
  /opt/venvs/rocm101t/bin/python -m pytest \
  tests/module_tree/test_fusion_kernel_backends.py -q

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCHDYNAMO_DISABLE=1 \
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
GPTQMODEL_LLAMA32_FUSION_AWQ=/tmp/llama32-symmetric-awq \
  /root/venv/bin/python -m pytest tests/models/test_llama3_2_fusion_devices.py -q

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCHDYNAMO_DISABLE=1 \
HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 PYTHON_GIL=0 PYTHONPATH=. \
GPTQMODEL_LLAMA32_FUSION_AWQ=/tmp/llama32-symmetric-awq \
  /opt/venvs/rocm101t/bin/python -m pytest \
  tests/models/test_llama3_2_fusion_devices.py -q -k rocm
```

The backend matrix and checkpoint acceptance use eager execution and one CPU
thread. Compiler fallback/source lifetime is checked separately with compilation
enabled; source graphs cannot be installed as the fused native parent. Local
checkpoints are not downloaded by these tests. Full hardware qualification of
every model architecture is separate from this kernel/Llama report.

## Configuration composition and tree-owned qualification

`FusableConfig("target").merge(...)` composes immutable role profiles and policy.
The target identity stays local. Multiple sources and diamond contributions
are supported; conflicting profiles require `replace_profile()` and inherited
policy conflicts require `override()`. Scoped rules select profile families and
layer indices. Incomplete/unresolved configurations cannot change an installed
fusion plan. Printed/extracted decisions include origins and winning policy.

The tree resolves exact projection paths and actual live parents. Declarations
contain no HF parent-class import paths or class allowlists. Forward structure,
quantization settings and native layouts remain validated. Indexer/cache
preambles execute once, including when execution uses the original continuation.

| Validation after composition and tree-owned parent resolution | Passed | Skipped |
| --- | ---: | ---: |
| configuration | 58 | 0 |
| module tree CUDA | 1,593 | 142 |
| core config device ROCm | 138 | 29 |
| save reload and fresh quantization | 4 | 0 |
| native kernels CUDA | 146 | 34 |
| native kernels ROCm | 87 | 93 |

The full module-tree run includes all 198 registry aliases, 101 real-model
topology cases and 464 native-parent oracle cases. Skips are retained with their
pytest reasons; an inactive vendor runtime is not counted as native validation.
Renamed runtime-parent/weight regressions cover GPTQ and AWQ, including a
side-effecting preamble and hooked fallback. Llama checkpoint tests verify
Marlin selection, the ULP/tie-break envelope above, zero extra packed-weight
storage and exact save/reload behavior. Ruff, compileall and diff-whitespace checks pass.

`plan = model.fuse()` applies fusion and returns the `FusionPlanner` directly.
Inspect it with `plan.print()`, `plan.render()` and `plan.to_dict()`;
`model.fuse(False)` returns `None`. Benchmark callers retain that object for
storage/diagnostics. Updated Llama latency samples and retained-storage
measurements are in [the benchmark report](benchmarks/llama32-fusion-native-backends.json).

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
  python -m pytest tests/module_tree \
  --ignore=tests/module_tree/test_fusion_kernel_backends.py -q

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 \
PYTHON_GIL=0 PYTHONPATH=. /opt/venvs/rocm101t/bin/python -m pytest \
  tests/module_tree/test_fusable_merge.py tests/module_tree/test_fusable_config.py \
  tests/module_tree/test_llama_fusion.py tests/module_tree/test_fusion_device_inference.py -q
```

## Review regression validation

Native operand/mapping collections retain their variable-length layout contract;
zero-free kernels never receive placeholder zero-point fields. A fast contract
regression covers that invariant. A forward-analysis regression rejects dynamic
metadata arguments explicitly. Guard variable names and test imports were
clarified, and unused non-compacting descriptor-owner bindings use the discard
name; actual ExLlamaV2 descriptor lifetime remains covered by the native matrix.

| Review regression run | Passed | Skipped |
| --- | ---: | ---: |
| module_tree | 1,594 | 142 |
| native_cuda | 147 | 34 |
| native_rocm | 88 | 93 |

Ruff, compileall and diff-whitespace checks pass.

## Additional CPU BF16 precision check on ARM

On an aarch64 Cortex-X925/A725 host with Torch 2.15.0+cu132 and no
AVX512-BF16, the native CPU int4pack operation accepted FP32 activations and
affine metadata while keeping weights packed. The BF16-input path now uses
those FP32 operands for the current call, then returns BF16. Temporary
activation and metadata conversions are released after the call; no dense or
expanded weight copy is retained. FP16 keeps its existing path.

The independent oracle computes the FP32 projection and bias, then rounds once
to the output dtype. These measurements use K=512, GQA Q/K/V widths
512/256/256, group size 128, and 1/8/33/257 rows. The gate remains
`rtol=atol=2e-3`. [Raw results](benchmarks/cpu-bf16-int4pack-arm-regression.json)
record the same four classes before and after the change.

| Native class | Before: max oracle difference | After: max oracle difference | Parent off/on difference | Additional packed-weight bytes |
| --- | ---: | ---: | ---: | ---: |
| `TorchAtenAwqLinear` | 0.0029296875 | 0.001953125 | 0 | 0 |
| `TorchAtenLinear` | 0.0029296875 | 0.001953125 | 0 | 0 |
| `TorchFusedAwqLinear` | 0.0029296875 | 0.001953125 | 0 | 0 |
| `TorchFusedLinear` | 0.0029296875 | 0.001953125 | 0 | 0 |

The CPU op-only timing probe used eight Torch threads, four warmups and twenty
timed calls per shape. BF16 operands took 0.052/0.526/3.565 ms for 1/8/257
rows; FP32 operands including per-call conversions took 0.055/0.548/3.070 ms.
These are local kernel timings, not full-model latency claims. The combined
fusion matrix passed 165 tests and skipped 126 unavailable native cases on this
host; mutation, exact packed-weight restoration and persistent-storage checks
are part of that matrix.
For the K=512, N=512, M=257 projection, explicit temporary FP32 input,
affine metadata and output tensors total 1,069,056 bytes
(526,336 + 16,384 + 526,336); this excludes any internal ATen scratch.
