# Failed Experiments

Date: 2026-05-22

This file records optimization attempts that did not produce a usable runtime
result, including enough metric/error data to avoid repeating the same probe
without a changed hypothesis.

## Cannoe Zero-Offset Source Patch Runtime Guards

Context:

- Source change: symmetric `zero_offsets != 0` branches skip offset tensor reads
  in direct CANN9 vector-dequant tile fill, single-word direct fallback, staged
  producer, and M=1 scalar decode fallback.
- Build/test environment: CANN `/usr/local/Ascend/cann-9.1.0-beta.1`, NPU7
  excluded with `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6`.
- Source/build regression test passed:
  `pytest -q tests/test_cannoe_ascendc_build.py` -> `52 passed, 14 warnings`.

Failed runtime guards:

| Probe | Device | Result | Metric data |
|---|---:|---|---|
| `scripts/benchmark_qwen3_27b_gptq_fp16.py --path cannoe --device npu:0 --tokens 1 --layers q,k,v,gate,up,down --warmup 2 --iters 5` | visible NPU0 | Failed before timings in native `torch_npu.npu_weight_quant_batchmatmul` | `aclnnWeightQuantBatchMatmulV2` status `161002`; CANN log: `Parse dynamic kernel config fail`; no speed/accuracy data |
| Same benchmark after sourcing `/usr/local/Ascend/cann-9.1.0-beta.1/set_env.sh` | visible NPU0 | Same failure | `aclnnWeightQuantBatchMatmulV2` status `161002`; no speed/accuracy data |
| Fresh direct multi-K build: `scripts/build_cannoe_ascendc.py --output /tmp/cannoe_zero_offset_skip_ws --clean --strategy tscm-direct-multik --target install` | build only | Build succeeded | package SHA256 from CPack: `b3764fa04db3f56024a7a43c109e486db156f1a94b5af3790575f0b6410656a7` |
| Raw custom-op Qwen down one-hot validation using fresh package and CPU reference | physical NPU0 | Failed before launch in workspace sizing | all 8 one-hot cases failed with `aclnnCannoeW4A16MatmulGetWorkspaceSize` status `161001`; `custom_ms_*`, `max_abs_max`, `mean_abs_max` all `null` |
| Raw custom-op default validation using fresh package and CPU reference | physical NPU0 | Failed before launch in workspace sizing | all 8 default cases failed with `aclnnCannoeW4A16MatmulGetWorkspaceSize` status `161001`; `custom_ms_*`, `max_abs_max`, `mean_abs_max` all `null` |

Interpretation:

- The source patch is compile/source-test clean, but the current host cannot use
  the native public ACLNN W4A16 path for full Qwen benchmarking after the 9.1
  beta update.
- The fresh direct multi-K custom-op package also fails before device execution,
  so it cannot validate or invalidate the zero-offset source patch at runtime.
- Do not treat these failed runtime probes as performance regressions. Re-test
  after fixing the CANN 9.1 beta custom-op workspace registration or after
  rebuilding the bridge/package pair in a known-good custom-op install layout.
