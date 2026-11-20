<!-- SPDX-FileCopyrightText: 2026 ModelCloud.ai
SPDX-License-Identifier: Apache-2.0 -->

# Linear bias regression examples

Choose cases for the paths the patch affects and extend the closest test family.
Verify which backend served the public call. For native ABI changes, exercise
the wrapper and any direct launcher whose bias contract changes.

## Reduction and dispatch

- Use opposing terms with a small nonzero residue. With FP16 output, an exact
  dot product of `65536` must return `1` with FP32 bias `-65535` and `32` with
  FP16 bias `-65504`. Under BF16 autocast, a dot product of `512 + 1` with bias
  `-512` must return `1`. Compute the oracle from independent decoded weights
  and the original bias with autocast off, then round the sum to the declared
  output dtype. Assert that dtype and shape, and exercise ambient autocast on
  and off where supported. Converting a rounded GEMM output to FP32 preserves
  its rounding error.
  [GPTQ wide-bias fallback](../../../../docs/low-precision-bias-epilogue.md#portable-gptq-fallback),
  [INT8 cancellation tests](../../../../tests/test_int8_bias_cancellation.py),
  [MLX scaled reduction fix #4539](https://github.com/ModelCloud/GPTQModel/pull/4539),
  [AWQ autocast report #5613](https://github.com/ModelCloud/GPTQModel/issues/5613).
- Cross the affected decode, prefill, and portable fallback paths with absent
  bias, representable bias, and a genuinely inexact wide bias. FP16 and BF16
  are incomparable, so use a value that exercises the gate for each dtype.
  FP32 `0.1` is inexact in both, while `65536` fits in BF16 only. Assert that
  eligible controls still use the intended native path. A policy test can
  pin eligibility, and an actual launch is needed for native accuracy.
  [AllSpark cast tests](../../../../tests/test_allspark_bias_width.py),
  [FP8 native GEMM fix #4716](https://github.com/ModelCloud/GPTQModel/pull/4716).
- For split-K, force multiple nonempty partitions and use distinct nonzero
  bias values by output column. Check the final bias contribution and cover
  padded output columns where supported. When changing the native bias width,
  reconcile the launcher dtype, vector alignment, and shared-memory byte
  budget with the kernel pointer type and stage offsets. Check for overlap
  with live reduction scratch and repeat the target-device forward against
  the oracle to expose corrupted tiles.
  [Marlin stage fix #4819](https://github.com/ModelCloud/GPTQModel/pull/4819),
  [Marlin layout and determinism tests](../../../../tests/kernels/test_marlin_tile_determinism_4735.py).

## Packing, loading, and fusion

- Use a checkpoint bias dtype that differs from the model dtype. Include a
  value that would overflow, such as FP32 `65520` when narrowed to FP16, and
  a tiny nonzero value such as `2**-25` that would round to FP16 zero. Check
  bias keys, exact values, dtype, and shape after the affected transitions.
  Compare in-session and restored forwards and inspect the tensor after
  re-save where applicable. When changing runtime bias preparation, check
  storage after switching to another supported activation dtype. Check
  `MODEL_DTYPE_BUFFERS` and restore destination dtypes when the loader is
  involved. Include offload or resume when that is the changed entry point.
  [AWQ pack/load tests](../../../../tests/test_awq_pack_bias_representability.py),
  [FP8 in-session/reload test](../../../../tests/test_fp8_bias_dtype_4575.py),
  [Checkpoint resume tests](../../../../tests/test_checkpoint_bias_dtype_4324.py),
  [QQQ packing report #5603](https://github.com/ModelCloud/GPTQModel/issues/5603).
- Use different bias values across output columns to detect permutations.
  For changed fusion, compare each output slice to its unfused oracle, include
  padded widths and bias-free slices where supported, and verify the source
  checkpoint values and shapes after fusion is removed.
  [Marlin layout and fusion](../../../../docs/marlin-bias-epilogue.md#fusion),
  [Marlin bias tests](../../../../tests/kernels/test_marlin_bias_epilogue.py).

## Compilation and training

- A value-dependent narrowing check can synchronize the device with the host
  or break a graph. Use a traceable wide branch when the check would break
  declared capture or compile support. Replay with a changed live bias where
  supported. For declared full-graph support, run `torch.compile(fullgraph=True)`
  after an eager warmup that builds the runtime layout. If caching a converted
  bias or its eligibility, test live updates through the supported mutation
  paths, including `.data` writes where supported because they bypass version
  counters.
  [VPTQ compile and replay tests](../../../../tests/test_vptq.py),
  [Paro warmup fix #5350](https://github.com/ModelCloud/GPTQModel/pull/5350).
- When training is supported and the epilogue or its gate changes, freeze the
  other operands, make only the bias trainable, and compare its gradient
  with an independent sum of nonuniform upstream gradients over the broadcast
  axes. Check that frozen bias and `no_grad` retain eligible fused paths.
  If a custom autograd forward feeds an in-place adapter add, exercise that
  consumer and backward for affected 2-D and batched outputs.
  [FP8 bias-gradient fix #5497](https://github.com/ModelCloud/GPTQModel/pull/5497),
  [In-place training tests](../../../../tests/test_training_kernel_inplace_4974.py).
