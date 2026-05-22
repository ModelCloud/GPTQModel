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
- Follow-up on the native benchmark failure: direct module calls and direct
  native op calls for all six Qwen projection shapes succeeded on visible NPU0.
  The failure was caused by the benchmark redirecting process stdout/stderr file
  descriptors to `/dev/null` before CANN dynamic-kernel parsing. Removing FD
  redirection and using only env-level CANN log quieting fixed the native Qwen
  benchmark.

Failed runtime guards:

| Probe | Device | Result | Metric data |
|---|---:|---|---|
| `scripts/benchmark_qwen3_27b_gptq_fp16.py --path cannoe --device npu:0 --tokens 1 --layers q,k,v,gate,up,down --warmup 2 --iters 5` before the FD-redirection fix | visible NPU0 | Failed before timings in native `torch_npu.npu_weight_quant_batchmatmul` | `aclnnWeightQuantBatchMatmulV2` status `161002`; CANN log: `Parse dynamic kernel config fail`; no speed/accuracy data |
| Same benchmark after sourcing `/usr/local/Ascend/cann-9.1.0-beta.1/set_env.sh`, before the FD-redirection fix | visible NPU0 | Same failure | `aclnnWeightQuantBatchMatmulV2` status `161002`; no speed/accuracy data |
| Fresh direct multi-K build: `scripts/build_cannoe_ascendc.py --output /tmp/cannoe_zero_offset_skip_ws --clean --strategy tscm-direct-multik --target install` | build only | Build succeeded | package SHA256 from CPack: `b3764fa04db3f56024a7a43c109e486db156f1a94b5af3790575f0b6410656a7` |
| Raw custom-op Qwen down one-hot validation using fresh package and CPU reference | physical NPU0 | Failed before launch in workspace sizing | all 8 one-hot cases failed with `aclnnCannoeW4A16MatmulGetWorkspaceSize` status `161001`; `custom_ms_*`, `max_abs_max`, `mean_abs_max` all `null` |
| Raw custom-op default validation using fresh package and CPU reference | physical NPU0 | Failed before launch in workspace sizing | all 8 default cases failed with `aclnnCannoeW4A16MatmulGetWorkspaceSize` status `161001`; `custom_ms_*`, `max_abs_max`, `mean_abs_max` all `null` |

Interpretation:

- The source patch is compile/source-test clean, and the native public ACLNN
  W4A16 path can run full Qwen benchmarking after avoiding process FD
  redirection during CANN initialization.
- The fresh direct multi-K custom-op package also fails before device execution,
  so it cannot validate or invalidate the zero-offset source patch at runtime.
- Do not treat the pre-fix native benchmark failure as a performance regression.
  Re-test the custom-op package after fixing CANN 9.1 beta custom-op workspace
  registration or after rebuilding the bridge/package pair in a known-good
  custom-op install layout.

## Cannoe Native Symmetric Offset Elision

Context:

- Hypothesis: for symmetric GPTQ, the public CANN
  `npu_weight_quant_batchmatmul` path might accept `offsets=None` or a
  broadcast-shaped constant offset tensor, allowing Cannoe to avoid retaining
  full per-group/per-output offset memory.
- Environment: CANN `/usr/local/Ascend/cann-9.1.0-beta.1`, visible devices
  limited to NPU0-6, probe executed on visible NPU0.

Failed probes:

| Probe | Shape | Result | Metric data |
|---|---|---|---|
| `offsets=None` with `K=32,N=8,group=32` | one group | Rejected | CANN error: `antiquant_group_size can be either 0 or a multiple of 32 within the range 32 to weight_k_dim - 1` |
| `offsets=None` with `K=64,N=64,group=32` | two groups | Accepted but wrong | full-offset mean abs output `5.8015`, no-offset mean abs output `0.6635`, max diff `9.6484` |
| Broadcast offsets `(1,N)`, `(G,1)`, scalar `(1,)`, and empty `(0,)` with `K=64,N=64,group=32` | two groups | Rejected | each failed in `aclnnWeightQuantBatchMatmulV2` with status `161002` |
| Full logical-shape offsets backed by one scalar with zero strides | `K=64,N=64,group=32` | Rejected | contiguous full offsets passed; `expand_as()` and `torch.as_strided(..., stride=(0,0))` both failed in `aclnnWeightQuantBatchMatmulV2` with status `161002` |

Interpretation:

- The public native CANN op requires a full offset tensor for correct symmetric
  GPTQ semantics. Do not pass `None`, do not attempt offset broadcasting, and
  do not use zero-stride logical full-shape offsets.
- The viable optimization is narrower: skip reading and unpacking `qzeros`
  during Cannoe symmetric prepack, then generate the required full constant
  `offsets=8` tensor directly.

## Cannoe Plain-Native Inner-Precise Auto

Context:

- Hypothesis: reuse the existing q-like `inner_precise=1` shape policy in the
  bound plain-native fp16 path, so q-proj could get the optional CANN precision
  mode without routing through the heavier planned path.
- Environment: CANN `/usr/local/Ascend/cann-9.1.0-beta.1`, visible devices
  limited to NPU0-6, probes executed on visible NPU0.

Failed probe:

| Probe | Result | Metric data |
|---|---|---|
| Qwen q-proj same-module forced `inner_precise=0` vs auto | Correct but not useful | `max_abs=0.0`, `mean_abs=0.0`, path `plain_native_bound` |
| Full Qwen3 27B Cannoe gate with plain-native auto `inner_precise` | Regressed speed | total mean `1.1885 ms`; q `0.1253`, k `0.1060`, v `0.1018`, gate `0.2525`, up `0.2810`, down `0.3219`; all `live_src=0.0` |

Interpretation:

- Do not pass optional `inner_precise` from the bound plain-native fp16 path by
  default. Even the q-like zero-drift shape slowed in the full module gate on
  CANN 9.1 beta. Keep the existing planned-path/direct probes as diagnostics
  only unless a same-gate retest shows a clear win.

## Cannoe SVDQuant Raw Launcher And Global Tile Probes

Context:

- Hypothesis: SVDQuant's 910B launcher style could let Cannoe bypass custom-op
  ACLNN workspace validation and call a generated `aclrtlaunch_*` entry point
  directly, while a broader `tile_n=768` policy might improve all Qwen shapes.
- Environment: CANN `/usr/local/Ascend/cann-9.1.0-beta.1`, visible devices
  limited to NPU0-6, probes executed on visible NPU0.

Failed or non-actionable probes:

| Probe | Result | Metric data |
|---|---|---|
| Search local Cannoe generated package for raw launch symbols | No runtime hook to use | `libcust_opapi.so` exports only `aclnnCannoeW4A16Matmul` and `aclnnCannoeW4A16MatmulGetWorkspaceSize`; generated artifacts exposed no `aclrtlaunch_*` symbol or header |
| Full Qwen3 27B Cannoe gate with `GPTQMODEL_KOMODO_PREPACK_TILE_N=768` | Too noisy for a global default | initial sweep total `1.0076 ms`, but paired confirms were baseline `1.0981 ms` vs tile768 `1.0908 ms`, then baseline `1.1477 ms` vs tile768 `1.1209 ms` |
| Large down-only sweep with `tile_n=4096` | Memory regression | down `0.2773 ms`, similar to `1024`/`1536`, but peak rose to `355.4 MB` versus `239.2 MB` |

Interpretation:

- SVDQuant still points in the right architectural direction: raw launch,
  explicit AIC/AIV ownership, and bounded device-side rings. Cannoe cannot adopt
  the raw-launch part without changing the Ascend C build to retain/export the
  generated launch stub.
- Do not set a global `768` prepack tile policy. For CANN 9.1 beta, the useful
  production adjustment is narrower: let the bound plain-native Qwen down shape
  inherit parent `1024`, and keep non-plain planned/fused experiments isolated.

## Cannoe Down-Projection SVDQuant Follow-Up Probes

Context:

- Hypothesis: after the successful common LoRA `addmm(out=...)` epilogue,
  decode-only `M=1` LoRA could be faster with a vector-shaped `addmv` epilogue,
  and down-projection bias might be faster outside the public CANN W4A16 op.
- Environment: CANN `/usr/local/Ascend/cann-9.1.0-beta.1`, visible devices
  limited to NPU0-6, probes executed on visible NPU0.

Failed or non-actionable probes:

| Probe | Result | Metric data |
|---|---|---|
| Monkey-patched rank-16 LoRA decode epilogue using `addmv` when `M=1` | Regressed | Current `addmm` path total `1.3145 ms`, down `0.3209 ms`; `addmv` total `1.5600 ms`, down `0.3641 ms`; drift unchanged enough to evaluate speed |
| `GPTQMODEL_KOMODO_FUSE_BIAS=0` in Cannoe AB Qwen3 27B GPTQ fp16 | Not a stable default win | No-fused-bias total `1.0403 ms`, down `0.2789 ms`, max_abs `0.015625`; paired default fused-bias total `0.9809 ms`, down `0.2916 ms`, max_abs `0.0625` |
| Down-only retest of inherited plain-native tile | Confirms current policy | `tile_n=1024` down `0.2733 ms`; `1536` down `0.2856 ms`; `2048` down `0.2788 ms`; all `239.2 MB` peak |

Interpretation:

- Keep the common LoRA `addmm(out=...)` epilogue. Do not add a decode-only
  `addmv` branch on Ascend 910B; the smaller mathematical shape loses to the
  existing GEMM-shaped runtime path.
- Keep fused bias enabled by default. Disabling it can reduce one down-only
  number and drift in the AB harness, but the paired full-shape total favored
  the default fused-bias path.
- Keep the plain-native large down projection on the inherited `1024` prepack
  tile for CANN 9.1 beta.
