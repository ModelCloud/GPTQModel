# Komodo-CANN Design and Optimization Log

Date: 2026-04-30

This file records the Komodo-CANN hotspot work and the design contract for the
next real speed step: a fused Ascend C W4A16 matmul path that consumes INT4
weight tiles inside the device kernel instead of materializing full FP16 dense
weights in GM/L2.

The filename keeps the requested `knodo-cann` spelling. The backend and code use
the canonical `komodo_cann` name.

## Current State

Komodo-CANN is a separate backend, not a mode inside plain Komodo.

- GPTQ backend: `BACKEND.GPTQ_KOMODO_CANN`
- AWQ backend: `BACKEND.AWQ_KOMODO_CANN`
- Generic alias: `BACKEND.KOMODO_CANN`
- Runtime classes: `KomodoCannLinear`, `AwqKomodoCannLinear`

The default runtime is still the profiled native CANN baseline:
`torch.ops.npu.npu_weight_quant_batchmatmul`, which maps to
`aclnnWeightQuantBatchMatmulV2` in the local profiler traces.

As of this pass, the Python runtime also has a real fused-op dispatch boundary:

```text
torch.ops.gptqmodel_komodo_cann.w4a16_matmul
torch.ops.gptqmodel_komodo_cann.komodo_cann_w4a16_matmul
torch.ops.gptqmodel_komodo_cann.komodo_cann_w4_a16_matmul
torch.ops.npu.gptqmodel_komodo_cann_w4a16_matmul
torch.ops.npu.komodo_cann_w4a16_matmul
torch.ops.npu.komodo_cann_w4_a16_matmul
```

If one of those ops is registered, Komodo-CANN uses it for supported W4A16 GPTQ
shapes before falling back to native CANN. If no op is registered, the default
behavior is unchanged. During kernel bring-up:

```bash
GPTQMODEL_KOMODO_CANN_FUSED=1
GPTQMODEL_KOMODO_CANN_FUSED_REQUIRE=1
GPTQMODEL_KOMODO_CANN_FUSED_OP=gptqmodel_komodo_cann.w4a16_matmul
```

`REQUIRE=1` fails fast if the fused operator is missing, which prevents benchmark
runs from accidentally measuring the native fallback.

## Historical Passes

Plain Komodo baseline:

- Moved the default NPU path to native int4 packed execution.
- Kept dense dequantized weight caching opt-in only. Default inference should not
  persist dense FP16 weight copies.
- Added source-weight drop after native packing to reduce resident memory.
- Added lookahead native prepack so the next layer can pack while the current
  layer computes.
- Tuned group-16 handling with grouped matmul where it helps and loop fallbacks
  where grouped launch overhead dominates.
- Fused bias when the CANN path can absorb it without increasing drift.

Komodo-CANN separation:

- Added backend names and explicit selection without changing plain Komodo auto
  selection.
- Added per-shape 910B tiling records: active cube/vector cores, Split-K, base
  tiles, L0/UB/L2 byte estimates, dequant task counts, and prefetch/fused status.
- Added Qwen3.6 27B projection-shaped benchmarks and 8-NPU matrix runners using
  one NPU per task.

910B planning:

- Use AI Core, AIC/Cube, and AIV/Vector terminology. Do not model this as CUDA
  SM/shared-memory behavior.
- Query device caps at runtime. Local 910B1 reports 24 cube cores through
  `torch.npu.get_device_properties()` and 192 MiB L2.
- Keep INT4 K tiles aligned to C0=64.
- Start from `(baseM, baseN, baseK) = (128, 256, 64)` for full tiles, then use
  `baseM=16` for decode-like tiny-M cases.
- Split-K only for decode-like shapes: `rows <= 16`, `K >= 4096`, and
  `K >= 2N`, unless overridden by env.

Prefetch pass:

- Host-issued `npu_prefetch` was made opt-in. It added host calls without a
  useful device-side win in the tested steady-state loops.
- The default no-prefetch path now bypasses prefetch probing entirely.

CANN profiler pass:

- `msprof` is available under `/usr/local/Ascend/cann-8.5.1/bin/msprof`.
- `torch_npu.profiler` is available and now used by
  `scripts/profile_komodo_cann_npu.py`.
- The profiler helper parses `operator_details.csv` and `kernel_details.csv`.
- Level1 traces expose AIC/AIV counters directly in JSON summaries.

Key Qwen3.6 down-proj trace:

- Native op: `aclnnWeightQuantBatchMatmulV2`.
- Level1 duration: about `856.3us` across 3 measured iterations.
- Cube utilization: about `87.7%`.
- AIC MAC ratio: about `5.9%`.
- AIV MTE2 ratio: about `75.9%`.

Interpretation: the next bottleneck is vector/dequant and memory movement, not
host prefetch or Python dispatch. A custom op must reduce GM/L2 traffic between
AIV dequant and AIC matmul.

aclnn V3 probe pass:

- Added `gptqmodel_ext/komodo_cann/wq_bmm_v3_probe.cpp`, a raw ACLNN bridge for
  `aclnnWeightQuantBatchMatmulV3`.
- The probe registers `torch.ops.gptqmodel_komodo_cann.w4a16_matmul`, so the
  existing Komodo-CANN fused-op hook can select it when preloaded.
- Added the managed `komodo_cann_v3` torch.ops JIT extension and
  `GPTQMODEL_KOMODO_CANN_V3=1` runtime autoload gate. Default Komodo-CANN still
  falls back to native CANN unless this gate or another registered fused op is
  present.
- Avoided the Torch-NPU `EXEC_NPU_CMD` macro because its inline logging path
  references hidden Torch-NPU RTTI symbols in this environment.
- The raw bridge creates ACL tensor descriptors directly; packed Komodo INT4
  weights are exposed to ACLNN as logical `ACL_INT4 [K, N]` tensors over the
  existing `int32 [K, N / 8]` storage.
- Do not explicitly destroy the returned `aclOpExecutor` in this probe. The
  local CANN runtime double-freed when the probe destroyed it manually.

V3 correctness checks on 2026-04-30:

- `M=8,K=256,N=256`, no bias: exact match vs native for group sizes `0`, `32`,
  `64`, `128`.
- `M=8,K=256,N=256`, bias: exact match vs native for group sizes `0`, `32`,
  `64`, `128`.
- `M=1,K=4096,N=4096,group_size=128`: exact match vs native in the final single
  probe run.
- Through `KomodoCannLinear` with the probe preloaded and
  `GPTQMODEL_KOMODO_CANN_FUSED_REQUIRE=1`, the runtime path was
  `fused_w4a16_matmul` and max drift vs the torch baseline was `0.0009765625`.

Inner-precise auto policy:

- Environment control: `GPTQMODEL_KOMODO_CANN_INNER_PRECISE=auto|0|1`.
- Default `auto` selects `inner_precise=1` only for q-like group-32 decode
  projections: `rows <= 16`, `group_size == 32`, `K >= 4096`, and
  `K <= N <= 2K`.
- All other sampled shapes stay on `inner_precise=0`.

Same-NPU direct native CANN checks, 50 iterations, no dense baseline:

| Shape | group | inner=0 ms | inner=1 ms | 0-vs-1 drift | Auto |
|---|---:|---:|---:|---:|---|
| `M=1,K=5120,N=6144` q-proj | 32 | 0.07889 | 0.07687 | 0 | `1` |
| `M=1,K=5120,N=1024` k/v-proj | 32 | 0.04674 | 0.04687 | 0 | `0` |
| `M=1,K=5120,N=17408` gate/up | 32 | 0.23563 | 0.23547 | 0 | `0` |
| `M=1,K=17408,N=5120` down | 32 | 0.31400 | 0.31388 | 0 | `0` |
| `M=1,K=4096,N=4096` balanced | 128 | 0.03383 | 0.03472 | max `0.015625` | `0` |

The same direct V3 bridge check also kept all sampled group-32 shapes exact and
showed the same group-128 drift. The q-like allowlist is intentionally narrow:
it captures the only clear speed win from the sampled native CANN path and
avoids group-128 drift.

V3 workspace-cache pass:

- Environment control:
  `GPTQMODEL_KOMODO_CANN_V3_WORKSPACE_CACHE=0|1`; default is `1`.
- The cache keeps the ACLNN workspace tensor in the executor-cache entry and
  reuses it for the hot shape instead of allocating a byte tensor on every call.

| Shape | group | workspace cache off ms | workspace cache on ms | cache drift |
|---|---:|---:|---:|---:|
| `M=1,K=5120,N=6144` q-proj | 32 | 0.07620 | 0.07614 | 0 |
| `M=1,K=5120,N=1024` k/v-proj | 32 | 0.05209 | 0.04629 | 0 |
| `M=1,K=17408,N=5120` down | 32 | 0.30607 | 0.29884 | 0 |
| `M=1,K=4096,N=4096` balanced | 128 | 0.05072 | 0.04080 | max `1.53e-05` |

8-NPU q-like auto validation, V3 bridge and workspace cache enabled, tile
`1024`, warmup `2`, iters `5`, passed 8/8:

| Case set | Kernel | Source drop | Komodo total ms | Max abs drift | Auto inner=1 |
|---|---|---:|---:|---:|---|
| GPTQ group sizes + act-order | Komodo | 0 | 1.7750 | 0.03125 | none |
| GPTQ group sizes + act-order | Komodo-CANN V3 bridge | 0 | 2.4404 | 0.03125 | none |
| GPTQ group sizes + act-order | Komodo | 1 | 1.6649 | 0.03125 | none |
| GPTQ group sizes + act-order | Komodo-CANN V3 bridge | 1 | 2.4540 | 0.03125 | none |
| Qwen3.6 27B GPTQ projections | Komodo | 0 | 1.2569 | 0.0625 | n/a |
| Qwen3.6 27B GPTQ projections | Komodo-CANN V3 bridge | 0 | 1.6005 | 0.0625 | q-proj |
| Qwen3.6 27B GPTQ projections | Komodo | 1 | 1.2814 | 0.0625 | n/a |
| Qwen3.6 27B GPTQ projections | Komodo-CANN V3 bridge | 1 | 1.7097 | 0.0625 | q-proj |

Earlier V3 8-NPU A/B read:

| Case set | Kernel | Source drop | Komodo total ms | Max abs drift |
|---|---|---:|---:|---:|
| GPTQ group sizes + act-order | Komodo | 0 | 1.7615 | 0.03125 |
| GPTQ group sizes + act-order | Komodo-CANN V3 bridge | 0 | 2.5008 | 0.03125 |
| GPTQ group sizes + act-order | Komodo | 1 | 1.6408 | 0.03125 |
| GPTQ group sizes + act-order | Komodo-CANN V3 bridge | 1 | 2.5408 | 0.03125 |
| Qwen3.6 27B GPTQ projections | Komodo | 0 | 1.2548 | 0.0625 |
| Qwen3.6 27B GPTQ projections | Komodo-CANN V3 bridge | 0 | 1.6629 | 0.0625 |
| Qwen3.6 27B GPTQ projections | Komodo | 1 | 1.2696 | 0.0625 |
| Qwen3.6 27B GPTQ projections | Komodo-CANN V3 bridge | 1 | 1.9257 | 0.0625 |

This V3 bridge is still a native CANN baseline entry point. It does not solve
the final speed problem by itself. Even after descriptor, executor, and workspace
reuse, it is slower than the current native
`torch.ops.npu.npu_weight_quant_batchmatmul` path. The real target remains an
Ascend C device kernel that stages INT4 dequant tiles without materializing full
FP16 weights through GM/L2.

Ascend C custom-op bring-up:

- Added `gptqmodel_ext/komodo_cann/ascendc/` as the repo-owned overlay for the
  msopgen-generated custom operator project.
- Added `scripts/build_komodo_cann_ascendc.py` to generate the CANN project from
  `op_ir/komodo_cann_w4a16_matmul.json`, overlay the custom host/kernel sources,
  and optionally build it.
- The build helper now patches generated `CMakePresets.json` to
  `ASCEND_COMPUTE_UNIT=ascend910b`, and the op definition registers
  `ascend910b`. This makes the generated ACLNN support list advertise
  `SOC_VERSION_ASCEND910B`; the msopgen default emitted a 910A support list and
  failed executor creation on the local 910B host.
- Added `w4a16_ascendc_bridge.cpp`, a managed torch.ops bridge that dlopens the
  generated `libcust_opapi.so` and calls
  `aclnnKomodoCannW4A16MatmulGetWorkspaceSize` /
  `aclnnKomodoCannW4A16Matmul`. The bridge registers
  `torch.ops.gptqmodel_komodo_cann.komodo_cann_w4_a16_matmul`.
- The Python runtime can auto-load that bridge behind
  `GPTQMODEL_KOMODO_CANN_ASCENDC=1`. It keeps plain Komodo-CANN fallback
  behavior unchanged unless the gate or another fused op is enabled.
- The first device kernel was a correctness baseline: the writing AI Core read
  packed INT4 words directly, staged a 64-value INT4-to-FP16 dequant tile in UB,
  accumulated FP32, and wrote only FP16 output.
- The next validated kernel removed that intermediate FP16 dequant tile and
  restructured the loop around Komodo's `int32 [K, N / 8]` packing. Each packed
  word is loaded once and dequantized across all eight output lanes while the
  activation value stays live, cutting repeated GM packed-weight reads.
- The first scalar baseline host tiler pinned `blockDim=1`. Earlier dynamic AIV
  block counts were unsafe because CANN did not always expose a contiguous
  logical block-index set, which could leave stale output when packed output
  words were owned directly by `GetBlockIdx()`.
- The torch bridge now dispatches the ACLNN run through
  `at_npu::native::OpCommand::RunOpApiV2`. Direct ACLNN launch could race
  Torch-NPU queued producers for freshly-created NPU tensors; the OpCommand path
  fixed the mixed no-bias/bias/group-size correctness sweep.
- Raw-op timing on NPU0 for `M=8,K=256,N=256,group_size=32,bias=True` improved
  from `63.67 ms` on the UB-tile baseline at commit `2e12b84f` to `10.33 ms`
  with the 8-lane packed-word loop. Correctness sweeps remained below `0.003`
  max absolute drift against the CPU dequant reference.
- Hoisting the eight scale/offset loads out of the inner K loop and reusing them
  for each quant group reduced the same raw-op timing from `10.33 ms` to
  `9.22 ms`. This keeps the no-full-dense-materialization contract and reduces
  repeated GM scale/offset traffic; observed max drift in the mixed smoke was
  `3.0517578125e-05`.
- A two-row scalar micro-tile then reused each packed INT4 word and its
  dequantized eight-lane result across two activation rows. The same timing
  dropped from `9.22 ms` to `5.56 ms` with zero observed drift in the mixed
  correctness sweep. Odd row counts fall back to the single-row path.
- Replacing the per-lane signed INT4 branch with branchless two's-complement
  sign extension dropped the same raw-op timing from `5.56 ms` to `4.60 ms`.
  The mixed correctness sweep passed with max observed drift
  `0.00048828125`, and the 8-NPU deterministic smoke reported zero error on
  all devices.
- A four-row micro-tile then reused each packed word and dequantized lane group
  across four activation rows. This dropped the same raw-op timing from
  `4.60 ms` to `3.66 ms`, with max observed drift `0.000244140625`. An 8-NPU
  smoke covered row counts 1-8 and group sizes 0, 32, 64, 96, and 128 with zero
  deterministic error.
- A CANN profiler sample on `qwen3_6_27b_gptq_down_proj` with `tokens=8`
  reported the custom kernel as AIV-only with `cube_utilization(%) = 0.0`,
  confirming that the scalar path is still a staging baseline, not the target
  Cube-consuming fused design.
- An eight-row micro-tile for decode-like `rows=8` reduced the raw-op timing
  from `3.66 ms` to `3.42 ms`, with max observed drift `0.00048828125`. The
  follow-up 8-NPU smoke covered row counts 1, 2, 3, 4, 5, 7, 8, and 9 plus
  group sizes 0, 32, 64, 96, and 128 with zero deterministic error.
- A single-row two-packed-word micro-tile then reused each FP16 activation load
  across sixteen adjacent output channels. Decode timings on NPU0 improved from
  `1.03 ms` to `0.72 ms` for `M=1,K=256,N=256,group_size=32`, and from
  `16.73 ms` to `11.42 ms` for `M=1,K=1024,N=1024,group_size=32`. The existing
  row-oct path stayed essentially unchanged at `3.37 ms` for
  `M=8,K=256,N=256,group_size=32`, and the 8-NPU smoke passed on all devices
  with zero error for the deterministic packed-one validation.
- A two-row two-packed-word micro-tile extended the same adjacent-output reuse
  to `M=2` tails without materializing dequantized weights. Rechecked median
  timings on NPU0 improved from `1.239 ms` to `1.191 ms` for
  `M=2,K=256,N=256,group_size=32`, and from `18.734 ms` to `17.749 ms` for
  `M=2,K=1024,N=1024,group_size=32`. `M=1` medians stayed unchanged within
  noise, and random CPU-reference checks over group sizes 0, 32, 64, and 128
  stayed below `0.008` max error for the covered small shapes.
- A four-row two-packed-word micro-tile extended adjacent-output reuse to
  `M=4..7` decode tails. Relative to the prior two-row commit, NPU0 medians
  improved from `3.445 ms` to `3.348 ms` for
  `M=7,K=256,N=256,group_size=64`, from `53.206 ms` to `51.678 ms` for
  `M=7,K=1024,N=1024,group_size=64`, and from `36.833 ms` to `36.688 ms` for
  `M=5,K=1024,N=1024,group_size=32`. The small
  `M=4,K=256,N=256,group_size=32` median stayed flat within timing noise
  (`1.665 ms` to `1.678 ms`). Deterministic and random CPU-reference checks
  covered group sizes 0, 32, 64, and 128 with max observed drift below `0.008`,
  and the 8-NPU smoke passed on all devices.
- The AscendC torch bridge now preserves optional bias as a real null optional
  input instead of materializing an all-zero FP16 bias and synchronizing the
  stream to keep that temporary alive. On NPU0 this reduced no-bias medians from
  `0.827 ms` to `0.802 ms` for `M=1,K=256,N=256,group_size=32`, from
  `1.118 ms` to `1.086 ms` for `M=2,K=256,N=256,group_size=32`, and from
  `3.248 ms` to `3.213 ms` for `M=8,K=256,N=256,group_size=32`. Bias-present
  timings were unchanged within noise. This removes host-side artificial
  materialization; it does not change the device kernel's scalar AIV-only
  status.
- Symmetric GPTQ weights have all-zero offsets, so the planner now marks
  `rows >= 8` symmetric calls with a zero-offset flag and encodes that flag as a
  negative `base_k` attribute for the Ascend C tiler. The host tiler restores
  the absolute `base_k` value for tile sizing and stores a separate tiling flag;
  only the row-oct path consumes it. This skips eight FP16 offset GM reads per
  quant group without touching AWQ or smaller GPTQ tails. NPU0 medians improved
  from `3.214 ms` to `2.964 ms` for `M=8,K=256,N=256,group_size=32`, from
  `3.931 ms` to `3.685 ms` for `M=9,K=256,N=256,group_size=32`, from
  `6.353 ms` to `5.833 ms` for `M=16,K=256,N=256,group_size=32`, from
  `50.033 ms` to `45.781 ms` for `M=8,K=1024,N=1024,group_size=32`, and from
  `99.956 ms` to `91.477 ms` for `M=16,K=1024,N=1024,group_size=32`.
  M1/M2/M4 cases continue to use positive `base_k` and normal offset loads;
  they measured flat within noise. CPU-reference checks covered zero-offset and
  nonzero-offset paths with max observed error below `0.001`.
- The current scalar custom op parallelizes packed output-column ownership
  across up to eight logical AIV owners. The host tiler caps `blockDim <= 8`
  using the available AIV core count and `N / 8` packed-word count. The device
  kernel maps sparse physical IDs with `GetBlockIdx() % tiling.block_dim`, then
  assigns each owner a contiguous packed-word range so adjacent-output
  micro-tiles remain intact. Attempts at `blockDim=32` left unwritten chunks on
  this 910B runtime; capping at eight passed random CPU-reference sweeps over
  single-row, two-row, four-row, row-oct, group-size 64, and K=1024 cases with
  max observed error below `0.0005`.
- The 8-owner raw-op A/B sweep used all eight NPUs with one visible NPU per
  subprocess and compared against the prior pushed zero-offset baseline. Timings
  improved from `0.713 ms` to `0.099 ms` for `M=1,K=256,N=256`, from
  `1.002 ms` to `0.146 ms` for `M=2`, from `1.556 ms` to `0.234 ms` for
  `M=4`, from `2.895 ms` to `0.376 ms` for `M=8` nonzero offsets, from
  `2.868 ms` to `0.374 ms` for `M=8` zero offsets, from `5.750 ms` to
  `0.743 ms` for `M=16` zero offsets, from `46.037 ms` to `5.887 ms` for
  `M=8,K=1024,N=1024` nonzero offsets, and from `45.673 ms` to `5.824 ms` for
  the matching zero-offset case.
- The single-row path now hoists the offset term out of the per-K inner loop.
  For each quant group it accumulates `sum(x)` once, accumulates only the signed
  INT4 lane products in the K loop, then applies `sum(x) * offset * scale` once
  per output lane. This keeps the no-full-dense-materialization contract and
  removes one offset add from every lane/K product at the cost of FP32
  reassociation drift on asymmetric offsets. The 8-NPU raw-op A/B sweep improved
  `M=1,K=256,N=256,group_size=32` from `0.101 ms` to `0.093 ms`,
  `M=1,K=1024,N=1024,group_size=32` from `1.493 ms` to `1.355 ms`, and
  `M=1,K=1024,N=1024,group_size=64` from `1.457 ms` to `1.312 ms`. Symmetric
  zero-offset M1 cases stayed exact. The Qwen down-proj-shaped
  `M=1,K=17408,N=5120,group_size=32` check improved from `127.73 ms` to
  `116.32 ms` with max drift `0.0625` and mean drift about `1.0e-4`.
- The row-pair path now uses the same quant-group offset hoist, with a separate
  activation sum for each of the two rows. Against the M1-hoist baseline, the
  8-NPU raw-op A/B sweep improved `M=2,K=256,N=256,group_size=32` from
  `0.155 ms` to `0.121 ms`, `M=2,K=1024,N=1024,group_size=32` from
  `2.354 ms` to `1.809 ms`, and `M=2,K=1024,N=1024,group_size=64` from
  `2.318 ms` to `1.764 ms`. Mixed tails improved from `0.242 ms` to
  `0.207 ms` for `M=3,K=256,N=256` and from `0.411 ms` to `0.381 ms` for
  `M=6,K=256,N=256`; the M1 guard stayed flat. Max observed drift was
  `0.00390625`, and symmetric zero-offset M2 stayed exact.
- The row-oct path now hoists nonzero GPTQ offset contributions out of the K
  loop without touching the existing zero-offset fast path or the smaller-row
  paths. The AIV fused op still avoids dense dequantized-weight materialization.
  An 8-NPU one-shard-per-device A/B sweep over `gptq_group_sizes` improved
  fused group-size cases from `6.8177 ms` to `6.1858 ms` for group 32, from
  `6.7423 ms` to `6.0609 ms` for group 64, from `6.7174 ms` to `5.9748 ms`
  for group 128, and from `6.6647 ms` to `5.9162 ms` for full-group. Act-order
  cases improved from `6.8434 ms` to `6.1966 ms` for group 32 and from
  `6.7026 ms` to `5.9862 ms` for group 128. The fused geomean speedup was
  `1.115x`; max abs drift stayed unchanged at `0.015625`. Group-16 cases
  correctly remained on the native grouped path.
- Rejected follow-up variants:
  - Extending the offset hoist to the row-quad path improved direct `M=4`
    timings by about `5-6%`, but consistently slowed the existing row-pair path
    by about `2%`. Marking the row-quad helper `noinline` made the row-pair
    regression much worse at about `17%`, so the variant was reverted.
  - A branch-heavy small-row zero-offset specialization improved row-quad
    timings but regressed M1 nonzero by about `1%` or M2 by more than `6%`
    depending on how the branch was placed. Avoid that separate small-row branch
    shape; use the generic `OffsetValue(..., zero_offsets)` side band instead.
- A direct CANN `Matmul<fp16, int4, fp16>` probe was rejected for now. In the
  default msopgen package the kernel still compiled as `VectorCore`, so the
  sentinel Cube path returned zeros because no AIC side was scheduled. Forcing
  the generated project to `MIX_AIC` produced the expected `taskRation=1:2`
  package metadata, but the existing scalar fallback then raised device-side
  `SUSPECT REMOTE ERROR` at synchronize even with a minimal cross-core flag
  handshake. A no-op MIX package also failed on the AIC entry with an MTE DDR
  range fault, so this is a launch/ABI issue rather than scalar math. Do not
  force the current scalar op to mixed launch; the next mixed attempt needs a
  built-in-style AIC/AIV state machine from the start.
- The scalar kernel now has a neutral mixed-launch guard for the next
  built-in-style attempt: AIV block IDs are normalized by `GetTaskRation()` when
  present, and accidental AIC entry returns before touching GM state. The default
  generated package still launches as AIV-only. An 8-NPU raw-op A/B against the
  row-pair baseline was flat across `M=1/2/4/8/16` and `K=256/1024`, with the
  largest measured delta `+0.22%` on `M=2,K=256,N=256`.
- Added the first explicit staged-dequant control plane for the real fused
  AIV/Cube target. `GPTQMODEL_KOMODO_CANN_STAGED_DEQUANT=1` marks the tiling
  plan and passes a negative `base_n` only to the fused Ascend C op; the normal
  planner and scalar package remain unchanged. The host tiler requests custom-op
  workspace only when the bounded ping-pong FP16 tile store is smaller than a
  full dense dequantized `K x N` matrix, so the default path still has zero
  workspace and never caches dequantized dense weights. This commit is the
  workspace/ABI hook for the future AIV producer / AIC Cube consumer state
  machine; it does not make the scalar baseline a mixed kernel.
- Validation for that staged-dequant control plane built both the default and
  `--experimental-staged-dequant` OPP packages, ran the focused planner tests,
  and ran two 8-NPU raw-op smokes with one visible NPU per process. The default
  positive-`base_n` package and the experimental negative-`base_n` package both
  covered rows `1/2/3/4/6/8/16`, group sizes `0/32/64/128`, bias/no-bias, and
  `K` values `256/384/512`; max observed drift against native CANN was
  `0.0078125`.
- The experimental staged path now has a real AIV producer loop. When compiled
  with `--experimental-staged-dequant` and triggered by negative `base_n`, AIV
  owners unpack all assigned `(baseK, baseN)` GPTQ INT4 weight tiles into their
  bounded ping-pong FP16 staging slots, reusing the same scale/offset formula as
  the scalar path. The scalar accumulator still produces the user-visible output
  while the Cube consumer is being wired, so this is a producer validation step,
  not the final fused kernel. Both default and experimental packages compiled,
  and an 8-NPU negative-`base_n` smoke with producer writes enabled passed on
  rows `1/2/3/4/6/8/16`, group sizes `0/32/64/128`, and max native-CANN drift
  `0.0078125`.
- The first Cube-consumer bring-up now compiles behind
  `--experimental-cube-consumer`. The failed approach was to embed CANN's
  `TCubeTiling` as a nested dynamic tiling-data struct; CANN's build-time
  tiling parser first required a `TCubeTilingOp` registration, then generated a
  kernel-side `TCubeTiling` class that conflicted with Ascend C's own
  `TCubeTiling` alias. The working scaffold keeps Komodo-CANN's existing
  primitive tiling fields as the ABI and constructs an Ascend C `TCubeTiling`
  locally inside the experimental kernel before `REGIST_MATMUL_OBJ`.
- Default and staged producer packages still build without the Cube consumer
  flag. The Cube scaffold is compile-only at this point: it registers a
  `Matmul<GM/ND fp16, GM/ND fp16, GM/ND fp16>` consumer object with a locally
  populated Cube tiling record, but it does not yet feed staged INT4-dequant
  tiles into Cube or replace the scalar visible-output path.
- The next workspace-layout pass separates CANN Matmul/KFC system workspace
  from the staged FP16 tile ring. `GPTQMODEL_KOMODO_CANN_CUBE_CONSUMER=1`
  now marks the fused call with a negative `split_k` attribute. The host tiler
  decodes the absolute Split-K value and reports the 16 MiB CANN
  Matmul/KFC system-reserved workspace separately from the staged tile ring.
  The kernel now calls `GetUserWorkspace(workspace)` and stages FP16 tiles at
  user-workspace offset `0`, so AIV producer writes do not overlap CANN's
  message queues once the Cube consumer starts running.
- Validation for the workspace split built default,
  `--experimental-staged-dequant --experimental-cube-consumer`, and
  `--experimental-staged-dequant --experimental-mixed-launch` packages. The
  non-mixed staged+Cube 8-NPU fused-op smoke used one visible NPU per process on
  the q-proj shape with `tokens=1`; all devices reported user-workspace offset
  `0`, cube system workspace `16777216`, and custom user workspace `524288`.
  The earlier Cube-reserved ABI reported offset `12582912`, cube workspace
  `12582912`, and total custom workspace `13107200`; this was corrected because
  CANN already reserves system workspace ahead of the user workspace on 910B.
- The next mixed-launch gate is now explicit and remains opt-in:
  `--experimental-mixed-launch` implies the Cube-consumer compile define unless
  the new `--experimental-mixed-aiv-baseline` probe is used. Both modes switch
  kernel metadata from default `KERNEL_TYPE_AIV_ONLY` to
  `KERNEL_TYPE_MIX_AIC_1_2`. The build helper must coalesce all experimental
  defines into one `add_ops_compile_options` line; separate lines made CANN's
  dynamic compile script keep only the last define. Validation built a default
  control package with binary metadata `coreType=VectorCore`, `core_type=AIV`,
  and mixed packages with `coreType=MIX`, `taskRation=tilingKey`,
  `intercoreSync=1`, and binary config `coreType=0`.
- The mixed kernel currently keeps the visible output path on the AIV scalar
  worker. The `--experimental-mixed-aiv-baseline` package skips KFC/Matmul
  registration and returns immediately on AIC; an 8-NPU smoke on the q-proj
  shape completed on every device with fused path `fused_w4a16_matmul`, strategy
  `fused_w4a16_staged_dequant_aic_matmul`, offset `0`, cube workspace `0`, and
  custom workspace `524288`. This proves MIX launch plus AIV scalar staging is
  viable.
- The full mixed Cube-consumer package now calls `REGIST_MATMUL_OBJ` from both
  AIC and AIV sides because CANN's macro creates both the KFC server and client.
  The runtime hang was the vector-side macro waiting for `WORKSPACE_SYNC_ID`;
  explicitly calling `clearWorkspace(workspace)` in the mixed Cube path lets AIC
  clear the KFC workspace and notify that event before registration. An 8-NPU
  smoke with `GPTQMODEL_KOMODO_CANN_CUBE_CONSUMER=1` now completes on every
  device with offset `0`, cube system workspace `16777216`, and custom user
  workspace `524288`.
- The full mixed path is still a registration/runtime topology milestone. The
  visible output remains the AIV scalar baseline while the AIC side registers
  the Cube consumer and exits through the KFC lifecycle. The next target is to
  replace the scalar visible-output loop with actual AIV-produced staged tiles
  consumed by Cube, then wire the Cube result back without full FP16 weight
  materialization.
- The next pass added `--experimental-vecout-runtime-handoff`, a mixed-launch
  runtime path that fills a bounded local FP16 B tile from packed INT4 and gives
  that tile directly to CANN Matmul as `TPosition::VECOUT`. This is the first
  live Cube path in this tree that avoids writing the dequantized B tile through
  GM/L2 before Cube consumes it. It is still guarded because the current
  correctness implementation uses scalar direct INT4 decode for the live tile.
  Direct `asc_int42half_sync` produced invalid lane values in this context, so
  the vectorized INT4-to-FP16 mapping remains a follow-up rather than an enabled
  fast path.
- VecOut handoff validation on 2026-05-01 used all eight NPUs with one process
  per NPU and covered rows `1/4/8`, K `384/512/768/1024`, N `256/512`, and group
  sizes `32/64/128`. The package matched native CANN within
  `max_abs=0.015625` and `mean_abs<=0.002598`. The important scheduler
  discoveries were that `WaitIterateAll` must follow an `IterateAll` call with
  `waitIterateAll=true`, and that explicit multi-row `SetOrgShape` caused
  device error `507057`. Until the A stride issue is solved, the VecOut path
  launches Cube one row at a time for `M>1`.
- CANN 9 generated projects now need template tiling-key metadata copied into
  both host and kernel overlays. The build helper recognizes generated
  `npu_op_kernel_sources(...)` CMake and coalesces all experimental defines into
  one `npu_op_kernel_options(... ALL OPTIONS ...)` line. Mixed-launch builds add
  the corresponding host define so the host tiler selects the mixed key, and
  staged/Cube workspace accounting adds CANN's system workspace to Komodo-CANN's
  bounded user workspace before requesting memory from ACLNN.
- The next guarded attempt to recover multi-row Cube work is
  `--experimental-vecout-local-a`. Instead of asking Matmul to interpret a
  strided GM A tile, each AIV owner stages a contiguous `M x baseK` FP16
  activation tile in VECOUT and gives both A and B to Cube as local operands.
  This should reduce the current one-Cube-call-per-row behavior for `M>1`; it
  remains a compile/runtime probe until validated against the row-wise VecOut
  baseline.
- The first installed local-A runtime smoke passed on NPU0 for
  `M=4,K=512,N=256,group_size=64` with
  `base_m=16,base_n=-256,base_k=-128`, `max_abs=0.0078125`, and
  `mean_abs=0.0014190673828125` versus native CANN. This confirms the local A+B
  VecOut handoff can execute correctly for one multi-row shape; promotion still
  requires a broader shape sweep and timing comparison.
- The follow-up validation used one custom/native call per process. A
  single-device sweep passed five shapes covering rows `1/4/8`, group sizes
  `32/64/128`, K `384/512/1024`, and N `256/512`; the all-8-NPU sweep passed
  rows `1/2/3/4/7/8`, group sizes `32/64/96/128`, K
  `384/512/768/896/1024`, and N `256/512`. Worst observed drift was
  `max_abs=0.015625` and `mean_abs=0.0024566650390625`.
- A direct GM-to-VECOUT `DataCopy` replacement for the scalar A-tile staging
  loop compiled, but the installed package timed out before the first runtime
  result. Keep scalar local-A staging as the validated path unless a future
  probe isolates the required queue/barrier semantics for that copy.
- A zero-offset-only direct B-fill specialization for VecOut compiled and passed
  the first three NPU0 shapes, then timed out on
  `M=4,K=512,N=256,group_size=128`. Do not enable that branch without a smaller
  device-side repro; the generic offset-aware fill remains the stable path.
- Added `scripts/validate_komodo_cann_ascendc_raw.py` to make raw Ascend C
  package validation reproducible. The first run against the local-A install
  launched one worker per NPU across all eight devices and reproduced the manual
  sweep: all eight cases passed with worst drift `max_abs=0.015625` and
  `mean_abs=0.0024566650390625`.
- Extended the same raw harness with `--warmup`/`--iters` timing. Worker JSON now
  includes `custom_ms`, and the parent summary records min/mean/max custom-op
  latency plus the timing parameters, so future package A/B checks can use the
  reproducible eight-worker sweep instead of one-off timing scripts. With
  `--warmup 1 --iters 2`, the current local-A package passed the established
  all-8-NPU local-A case file with custom latency min/mean/max
  `2.284225/4.648382/8.693375 ms` and worst drift `max_abs=0.015625`,
  `mean_abs=0.001491546630859375`.
- A repeated-call timing run over the default tiny case set timed out on
  `rows=3,K=768,N=256,group_size=96` after seven other workers passed. The
  one-call default correctness path still passed all eight NPUs, so keep repeated
  timing on the known-good A/B case files until the shape-specific runtime
  sensitivity is isolated.
- The same harness also passed all eight local-A cases with positive
  `base_k=128`. With both side-band forms validated, the GPTQ Komodo-CANN
  planner now marks every symmetric fused call as zero-offset, including small
  `M<8` calls, while keeping the generic direct B fill rather than the rejected
  zero-offset-only branch.
- The scalar visible-output fallback now threads the same zero-offset side band
  into M1/M2/M4 tails instead of only M8. A fresh CANN 9 non-mixed staged/vector
  package passed NPU0 scalar checks at the standard `base_n=256` tile width with
  positive `base_k=128` and negative `base_k=-128`; the negative-base run
  covered rows `1/2/4/8` with worst drift `max_abs=0.015625`.
- The local-A VecOut runtime path now has a larger-row scheduling fix: for
  `rows > base_m`, the loop fills each direct B tile once per `(K tile, N tile)`
  and reuses it across all M tiles. This avoids the old repeated large B dequant
  inside the M-tile loop while leaving the already-validated single-M-tile decode
  order intact. A fresh local-A package passed an all-8-NPU raw sweep over rows
  `1/8/16/17/24/31/32/48`, group sizes `32/64/96/128`, K up to `1024`, and
  N `256/512`, with worst drift `max_abs=0.015625` and
  `mean_abs=0.001491546630859375`. The old local-A package timed out at 120 s on
  `rows=17,K=512,N=512,group_size=64`; the new package completed the same timing
  probe in `3.060030 ms` with `max_abs=0.0078125`.
- The local-A activation tile fill now uses row-wise GM-to-VECOUT `DataCopy` only
  for `rows >= 32`. Unconditional `DataCopy` was correct but slightly slower on
  tiny decode tiles; the shape-gated variant passed both all-8-NPU sweeps and
  improved the planner-shaped warmed mean from `4.369839 ms` to `4.066544 ms`
  while preserving worst drift `max_abs=0.015625`.
- A single 2D `DataCopyParams` activation copy was also tested. It was accurate,
  but slower than the per-row `DataCopy` gate on both all-8-NPU sweeps
  (`4.808621 ms` legacy mean, `4.145872 ms` planner-shaped mean), so the kernel
  keeps row-wise copies for the gated path.
- The next accepted local-B pass added a narrow zero-offset split inside the
  existing direct B-tile fill loop. It does not change the VecOut scheduler shape
  that timed out in the earlier zero-offset-only specialization; it only avoids
  per-lane offset GM reads and offset adds when the tiler marks symmetric GPTQ
  as zero-offset. The CANN 9 package passed the legacy all-8-NPU warmed sweep
  with custom latency min/mean/max `2.278330/4.510223/8.395375 ms`, and the
  planner-shaped sweep improved from `4.066544 ms` mean to `4.026535 ms` mean.
  Worst drift stayed `max_abs=0.015625`.
- A follow-up packed-B staging pass now copies the packed INT4 B tile into a
  bounded UB scratch tile with one contiguous `DataCopy` per K row, then
  dequantizes from that local int32 tile into the local FP16 B tile. This keeps
  the no-full-FP16-materialization contract while replacing thousands of scalar
  packed-weight GM reads per B tile. The first planner-shaped all-8 run had one
  raw-worker timeout on `rows=96,base_m=96`, but the isolated shape passed on the
  same NPU in `2.910235 ms` and the full retry passed. Final warmed means:
  legacy `4.052616 ms`, planner-shaped `3.651770 ms`, worst drift
  `max_abs=0.015625`.
- Rejected adjacent probes in the same pass: `base_k=256` timed out before
  emitting worker output in the full planner sweep, `rows >= 16` local-A
  `DataCopy` timed out on the legacy `rows=24,K=768,N=512,group_size=96` case,
  and a broader `base_m >= 32 || rows >= 31` local-A gate later timed out on
  `rows=16`. Keep the validated `rows >= 32` local-A copy gate.
- The Python staged Cube-consumer planner now uses `base_k=128` for every
  compatible `K % 128 == 0` shape instead of only `rows <= 16`. A plan-shaped
  local-A raw sweep passed all eight NPUs with rows
  `8/17/32/48/64/96/129/160`, per-case planner `base_m`, K `512/1024`, N
  `256/512`, and groups `32/64/128`; worst drift stayed at
  `max_abs=0.015625`, `mean_abs=0.00142669677734375`.
- This baseline intentionally avoids writing full dequantized FP16 weights
  through GM/L2. It is slower than the target design, but it creates the real
  custom-op registration, tiling, shape inference, optional bias handling, and
  packed-weight decode path needed before vector/Cube fusion.
- The initial multi-core strided writer was accepted by CANN but produced sparse
  output writes on the local 910B runtime. The validated follow-up limits
  ownership to eight logical chunks and maps sparse physical block IDs back into
  that logical range.
- Local validation after installing the generated custom OPP:
  - deterministic all-ones `M=2,K=64,N=16` matched exactly with and without bias.
  - randomized sweeps over `(M,K,N)=(8,64,64),(3,96,32),(1,128,128)`,
    `group_size` in `{0,32,64}` where divisible, and bias/no-bias all passed
    against a CPU dequant reference with observed max error below `0.003` after
    rerunning the tail case that had one transient high reading.

## Fused W4A16 Operator Contract

Runtime call shape:

```python
op(
    x,              # fp16 [M, K]
    packed_weight,  # Komodo/CANN packed int4 tensor, logical [K, N]
    scales,         # fp16 antiquant scales
    offsets,        # fp16 antiquant offsets
    bias,           # optional fp16 [N]
    group_size,     # 0, 32, 64, 128
    split_k,        # planner split count
    base_m,
    base_n,
    base_k,
) -> fp16 [M, N]
```

Supported first target:

- GPTQ W4A16, FP16 activations and output.
- `group_size` in `{0, 32, 64, 128}`.
- Decode and small-batch rows first (`M <= 16`), then broader shapes.
- Bias optional.
- Act-order is handled before the fused call by applying the existing input
  permutation to `x`.

Group-16 remains on the native fallback path until there is a dedicated grouped
fused design; it has a different pack/decomposition path and needs separate
tiling.

## Device-Kernel Design

Goal:

- Load INT4 packed B tiles from GM.
- Dequantize only the current K/N tile.
- Consume the tile immediately by Cube matmul.
- Avoid writing a full FP16 dequantized weight matrix to GM/L2.

910B constraints:

- AIV and AIC exchange through GM/L2 on 220x-class devices. A fused kernel still
  needs careful staging; it does not get CUDA-like shared memory between vector
  and cube phases.
- UB is 192 KiB and 32 B aligned.
- L0A/L0B are 512 B aligned; L0C is 64 B aligned.
- L0C is 128 KiB, so accumulator tiles must fit before considering double
  buffering.
- Keep GM moves at least 16 KiB where possible.

Implementation plan:

1. AIC-only wrapper baseline:
   - Register the custom operator and produce identical output.
   - Measure launch overhead and profiler naming.
2. AIV dequant stage:
   - Unpack vectorized INT4 values from `int32` lanes.
   - Apply scale/offset per group.
   - Write only one staged B tile, not a dense full weight.
3. AIC matmul stage:
   - Consume staged B tile with `baseK=64`.
   - Use L0A/L0B double buffering.
   - Keep `baseN=256` unless L0/UB pressure says otherwise.
   - Keep `TCubeTiling` construction device-local; do not add it as generated
     nested tiling data unless the CANN parser/name-conflict issue is solved.
   - Keep the KFC system workspace and staged tile ring in separate GM ranges.
4. Split-K decode:
   - Use 24 cube cores for large `K >> N`.
   - Reduce partials on vector cores or through an atomic/fixpipe path.
5. Tune with CANN profiler:
   - `PipeUtilization` first.
   - `Memory`, `MemoryUB`, `MemoryL0`, and `L2Cache` next.
   - `ResourceConflictRatio` if AIV vector time remains low but stalls persist.

## Benchmark Snapshot

Fresh 8-NPU A/B after profiler-guided host-path cleanup:

| Case set | Kernel | Source drop | Komodo total ms | Max abs drift |
|---|---|---:|---:|---:|
| GPTQ group sizes + act-order | Komodo | 0 | 1.8125 | 0.03125 |
| GPTQ group sizes + act-order | Komodo-CANN | 0 | 1.9142 | 0.03125 |
| GPTQ group sizes + act-order | Komodo | 1 | 1.6394 | 0.03125 |
| GPTQ group sizes + act-order | Komodo-CANN | 1 | 1.7173 | 0.03125 |
| Qwen3.6 27B GPTQ projections | Komodo | 0 | 1.2378 | 0.0625 |
| Qwen3.6 27B GPTQ projections | Komodo-CANN | 0 | 1.3255 | 0.0625 |
| Qwen3.6 27B GPTQ projections | Komodo | 1 | 1.2979 | 0.0625 |
| Qwen3.6 27B GPTQ projections | Komodo-CANN | 1 | 1.2861 | 0.0625 |

Observed CANN plans for Qwen projection cases used Split-K values `{1, 2, 8}`
and vector dequant task counts `{320, 1920, 5440}`.

## Commands

Focused profiler:

```bash
ASCEND_RT_VISIBLE_DEVICES=0 \
python scripts/profile_komodo_cann_npu.py \
  --mode cann \
  --case qwen3_6_27b_gptq_down_proj \
  --profiler-level level1 \
  --aic-metrics PipeUtilization
```

8-NPU A/B matrix:

```bash
python scripts/benchmark_komodo_npu_matrix.py \
  --devices all \
  --max-active 8 \
  --skip-unit \
  --skip-loop \
  --skip-quick \
  --include-komodo-cann-ab \
  --tiles 1024 \
  --warmup 2 \
  --iters 5 \
  --limit-tasks 8
```

Fused-op bring-up smoke:

```bash
GPTQMODEL_KOMODO_CANN_FUSED=1 \
GPTQMODEL_KOMODO_CANN_FUSED_REQUIRE=1 \
python scripts/profile_komodo_cann_npu.py --mode cann --iters 3 --warmup 1
```

If the fused op is not registered, that command must fail before measuring the
native fallback.

V3 API probe:

```bash
GPTQMODEL_KOMODO_CANN_V3=1 \
python scripts/probe_komodo_cann_v3.py \
  --device 0 \
  --rows 8 \
  --in-features 256 \
  --out-features 256 \
  --bias
```

8-NPU V3 bridge A/B:

```bash
GPTQMODEL_KOMODO_CANN_V3=1 \
GPTQMODEL_KOMODO_CANN_FUSED_REQUIRE=1 \
GPTQMODEL_KOMODO_CANN_FUSED_OP=gptqmodel_komodo_cann.w4a16_matmul \
python scripts/benchmark_komodo_npu_matrix.py \
  --devices all \
  --max-active 8 \
  --skip-unit \
  --skip-loop \
  --skip-quick \
  --include-komodo-cann-ab \
  --tiles 1024 \
  --warmup 2 \
  --iters 5 \
  --limit-tasks 8
```
