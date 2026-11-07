# Fusion inference across CPU, CUDA and ROCm

Fusion has been exercised on actual CPU, NVIDIA CUDA and AMD ROCm devices with
GPTQ and AWQ. The tests use symmetric INT4, group size 128, `desc_act=False`
and explicit model-definition profiles. Fusion remains off by default.

[Raw results](benchmarks/fusion-device-validation.json) include every case,
backend, dtype, output error, group count and retained-storage measurement.
These are correctness and lifecycle checks, not latency measurements. The
[existing Llama latency tables](fusion-planner.md#measured-llama-32-1b-results)
remain separate from this report.

## Hardware and scope

| Family | Actual hardware | Runtime | Validated fusion executor |
| --- | --- | --- | --- |
| CPU | AMD x86_64 Eng Sample 100-000001458-01 | Torch 2.15.0+cu132 and the ROCm build below | `TorchLinear`, `AwqTorchLinear` |
| CUDA | NVIDIA RTX 4090, SM 8.9 | Torch 2.15.0+cu132 | `MarlinLinear`, `AwqMarlinLinear`, plus portable Torch |
| ROCm | Radeon AI PRO R9700, gfx1201 | Torch 2.16.0.dev20261025+rocm10.1, HIP 7.16.26385, Python 3.14.10 free-threaded | `TorchLinear`, `AwqTorchLinear` |

The 32 unique small-model cases cover two methods, two activation dtypes,
GQA/MHA layouts and four executor/device combinations. CPU cases were also
repeated in the ROCm Python build. They instantiate real tiny Llama models and
consume their declared module plans. The six full-checkpoint cases load local
Llama 3.2 1B models. This does not establish hardware qualification for every
architecture in the [model matrix](fusion-model-coverage.md).

CPU and ROCm use explicit `BACKEND.GPTQ_TORCH` / `BACKEND.AWQ_TORCH` for these
fusion checks. `BACKEND.AUTO` on CUDA must retain Marlin in both modes. CPU or
ROCm AUTO can select a different backend class that is outside the current
fusion executor; such groups are skipped with reasons. Fusion never switches
the chosen backend to make a group eligible. This report does not claim fusion
support for optimized CPU AVX512/ATen or other untested backend classes.

Portable AWQ GQA currently fuses gate/up only: 2 groups in the tiny two-layer
model, 16 in the 1B model. Its QKV decisions record the explicit numerical
qualification restriction. AWQ MHA, GPTQ GQA/MHA and Marlin GQA/MHA exercise both
QKV and gate/up: 4 groups in the tiny model, 32 in the 1B model. Routed expert
fusion remains excluded.

## Independent projection oracles and whole-model behavior

Each projection oracle starts from independently constructed integer codes,
logical zero points and scales, performs its own FP32 matrix multiplication,
rounds to the activation dtype and adds bias. All elementwise comparisons use
`rtol=atol=2e-3`. Inputs have 1, 8 and 32 rows; FP16 and BF16 must be preserved.
Eligible GPTQ Torch paths also assert packed GEMV dispatch on supported GPU
architectures. The actual R9700 and RTX 4090 runs exercised packed GEMV and
FP16 small-M GEMM; BF16 paths retain the existing architecture gates.

The model checks compare a four-token prefill plus three cached decode steps,
three greedy generated tokens, repeated forwards, scale mutation/invalidation,
restoration and `fuse(False)`. The following maxima span both methods, dtypes
and attention layouts:

| Executor | Cases | Maximum projection error | Maximum prefill/decode logit difference | Extra packed-weight bytes |
| --- | ---: | ---: | ---: | ---: |
| `cpu_torch` | 8 | 0.0001220703125 | 0 | 0 |
| `cuda_torch` | 8 | 0.0001220703125 | 0 | 0 |
| `cuda_marlin` | 8 | 6.103515625e-05 | 0 | 0 |
| `rocm_torch` | 8 | 3.051757812e-05 | 0 | 0 |

Greedy token IDs match in all cases. Source leaves share the contiguous fused
packed allocations. The tests inspect owned tensors and containers after
repeated forwards and reject persistent dense/expanded weight caches. Model
storage accounting distinguishes packed weights from runtime metadata and
Marlin workspace: the tiny model adds at most 128 metadata bytes for Torch and
2,048 workspace bytes for Marlin, with zero extra checkpoint packed-weight
bytes. Disabling fusion restores independent contiguous source tensors.

## Real Llama 3.2 1B checkpoints

The full-checkpoint checks compare prefill, three cached decode steps and four
greedy generated tokens. CPU uses BF16; CUDA and ROCm use FP16. FP16 and BF16
are both covered by the smaller CPU model tests above. Full-model reductions
can compound rounding differences; projection-level oracle bounds remain
separate from the following measured logit drift.

| Family | Method | Dtype | Kernel | Fused groups | Max absolute logit difference | Relative L2 | Top-1 agreement | Extra packed bytes |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| cpu | AWQ | bfloat16 | `AwqTorchLinear` | 16 | 0 | 0 | 100.0% | 0 |
| cpu | GPTQ | bfloat16 | `TorchLinear` | 32 | 0 | 0 | 100.0% | 0 |
| cuda | AWQ | float16 | `AwqMarlinLinear` | 32 | 0.036132812 | 0.0025260698 | 100.0% | 0 |
| cuda | GPTQ | float16 | `MarlinLinear` | 32 | 0.02734375 | 0.0021759705 | 100.0% | 0 |
| rocm | AWQ | float16 | `AwqTorchLinear` | 16 | 0 | 0 | 100.0% | 0 |
| rocm | GPTQ | float16 | `TorchLinear` | 32 | 0 | 0 | 100.0% | 0 |

All six cases produce identical greedy token IDs and restore bitwise-identical
baseline logits after disabling fusion. Whole-model comparisons use identical
loaded weights and dtype in both modes; this is not a comparison between
CPU BF16 and GPU FP16 outputs. Checkpoints are local and are not downloaded by
the tests.

## Explicit guard retirement

The generated mutation guard owns references to packed operands in a private
global namespace. Invalidation retires its code to `_retired_guard`, clears
those bindings, then releases the executor's buffers and descriptor. A retained
guard returns `False` and cannot retain weights or reactivate fusion. This
works with free-threaded Python's deferred function destruction and does not
add work to the token loop. The
[operand-lifetime diagram](fusion-planner.md#operand-and-activation-lifetime)
shows the same invalidation flow.

`test_retired_guard_releases_weight_bindings_even_when_function_is_retained`
keeps an external reference to the old guard and verifies the packed tensor's
weak reference is dead without `gc.collect()`. The existing ownership and
unfuse tests also pass in the ROCm free-threaded runtime.

## Commands and results

| Run | Result |
| --- | --- |
| Device matrix, CUDA build | 25 passed, 8 other-vendor cases skipped |
| Device matrix, ROCm build | 17 passed, 16 NVIDIA cases skipped |
| Core fusion/config/device contracts, CUDA build | 478 passed |
| Core fusion suite, ROCm build | 56 passed, 13 NVIDIA-only cases skipped |
| Full checkpoints, CPU and CUDA | 4 passed, 2 ROCm cases skipped |
| Full checkpoints, ROCm | 2 passed, 4 other-family cases deselected |

Vendor skips are intentional: an inactive Torch runtime cannot address the
other vendor's devices. Every requested family has actual numerical execution;
no mocked ROCm result is reported as hardware validation.

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. /root/venv/bin/python -m pytest \
  tests/module_tree/test_fusion_device_inference.py -q

HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 PYTHON_GIL=0 PYTHONPATH=. \
  /opt/venvs/rocm101t/bin/python -m pytest \
  tests/module_tree/test_fusion_device_inference.py -q

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCHDYNAMO_DISABLE=1 \
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. \
  GPTQMODEL_LLAMA32_FUSION_AWQ=/tmp/llama32-symmetric-awq \
  /root/venv/bin/python -m pytest \
  tests/models/test_llama3_2_fusion_devices.py -q

HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 PYTHON_GIL=0 PYTHONPATH=. \
  GPTQMODEL_LLAMA32_FUSION_AWQ=/tmp/llama32-symmetric-awq \
  /opt/venvs/rocm101t/bin/python -m pytest \
  tests/models/test_llama3_2_fusion_devices.py -q -k rocm
```

CPU/CUDA checkpoint inference uses eager execution with one CPU thread to
avoid compilation startup dominating a numerical acceptance test. The compiler
fallback contract is exercised separately by the core fusion suite.
