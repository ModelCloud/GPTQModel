# Cannoe Kernel Notes

Date: 2026-05-02

Cannoe is the Ascend CANN kernel experiment. It remains separate from the plain
Komodo backend and runtime path.

## Selection

- GPTQ backend: `BACKEND.GPTQ_CANNOE` / `gptq_cannoe`
- AWQ backend: `BACKEND.AWQ_CANNOE` / `awq_cannoe`
- Generic selector: `BACKEND.CANNOE` / `cannoe`
- Benchmark flag: `scripts/benchmark_komodo_npu_ab.py --cannoe`

The current implementation subclasses the plain Komodo packed int4 plan only as
a baseline plan format. Runtime dispatch is through `CannoeLinear` and
`AwqCannoeLinear`, not through `KomodoLinear` or `AwqKomodoLinear`.

## CANN Prefetch Policy

Cannoe can issue CANN `torch.ops.npu.npu_prefetch` hints for eligible
activation, packed int4 weight, scale, offset, and fused-bias tensors before the
native quantized matmul. This is off by default because host-issued prefetch
probes regressed the steady-state microbenchmarks.

Environment controls:

```bash
GPTQMODEL_CANNOE_PREFETCH=1          # enable Cannoe prefetch probe
GPTQMODEL_CANNOE_PREFETCH_MAX_BYTES  # override max bytes per tensor
GPTQMODEL_CANNOE_PREFETCH_MIN_BYTES  # default: 4MiB
```

Only `GPTQMODEL_CANNOE_*` environment names are accepted for Cannoe-specific
controls.

On Ascend 910B, the default max prefetch window is derived from reported L2
cache size and cube-core count:

```text
max(4MiB, min(L2_cache_size / cube_core_num * 2, 32MiB))
```

On the local 910B1 host this resolves to `16MiB` from `192MiB` L2 and `24`
cube cores. Small tensors below the minimum byte threshold are skipped because
the prefetch call overhead dominated the early microbenchmarks.

## 910B Split-K Plan

`hw/ascend_910b.md` records the hardware constraints this path should target:

- Use AI Core/Cube and Vector terminology, not NVIDIA SM terminology.
- Query memory sizes at runtime; do not hardcode L1/L0/L2/HBM sizes.
- Start custom matmul tiling from `(baseM, baseN, baseK) = (128, 256, 64)`,
  with INT4 K aligned to C0=`64`.
- UB is 192 KB and 32 B aligned; L0C is 128 KB; L0A/L0B are 512 B aligned.
- AIV and AIC exchange through GM/L2 on 910B-class 220x devices, so a vector
  dequant stage that writes FP16 weights to a workspace must be reused enough
  to justify that extra round trip.

The arXiv W4A16 Ascend paper uses exactly that high-level structure: AIV
dequantizes INT4 weights to FP16, AIC/Cube performs GEMM, Split-K helps when
`K >> N`, and vector cores reduce Split-K partials. The paper also reports that
the main bottleneck is the extra GM movement of dequantized weights, not the
INT4-to-FP16 conversion itself.

Current Python-side Cannoe records this plan per shape:

- `split_k=1` for balanced shapes and group-size/act-order sweeps.
- `split_k>1` for decode-like Qwen projections where `rows <= 16`, `K >= 4096`,
  and `K >= 2N`.
- `base_k=64`, matching the INT4 C0 alignment requirement for producer-only
  staged plans.
- Staged Cube-consumer experiments now select `base_k=128` automatically when
  `K` is divisible by 128. This reduces direct TSCM/VecOut handoff count while
  keeping the FP16 B tile bounded; set `GPTQMODEL_CANNOE_BASE_K=64` to
  force the original C0-sized tile.
  A plan-shaped local-A package sweep passed on all eight NPUs with rows
  `8/17/32/48/64/96/129/160`, per-case planner `base_m`, K `512/1024`, N
  `256/512`, and groups `32/64/128`; worst drift was `max_abs=0.015625` and
  `mean_abs=0.00142669677734375`.
- The staged Cube-consumer planner uses `base_n=128` for `N=256`, for validated
  `N=512`/`N=768` shapes where `K` is `512`, `768`, or `896`, and for
  validated `N=640` shapes where `K` is `512` or `768`.
  The N768 planner-shaped all-8 sweep improved mean latency from
  `3.414122 ms` at `base_n=256` to `1.773176 ms` at `base_n=128`, with worst
  drift still `max_abs=0.015625`. The N640 K512/K768 sweep passed at
  `1.700607 ms` mean; K896/N640 and several N896 probes faulted, so they stay
  outside the gate.
- The plan records per-tile packed INT4 bytes, FP16 dequant workspace bytes,
  L0A/L0B/L0C tile bytes, K tiles per Split-K shard, and the number of vector
  dequant tasks. These are the checks the future Ascend C op must satisfy before
  it writes any dequantized FP16 tile to GM/L2.
- The Python planner and Ascend C host tiler no longer cap staged tile owners at
  eight. Public CANN W4A16 source uses core-grid ownership over the N/M block
  space rather than a fixed eight-owner staging limit, so large-N staged/Cube
  experiments can now use up to the runtime vector-core count while still
  bounding workspace below dense FP16 dequantization.
- `strategy=planned_split_k_aiv_dequant_aic_matmul` means the shape is a
  candidate for the future Ascend C op. Runtime still uses the native quantized
  matmul fallback until that custom op lands.
- The plan is cached by hot shape/device so benchmark loops do not re-query
  hardware or environment variables every forward.

## Current Benchmark Read

The first steady-state prefetch implementation was slower than plain Komodo on
the tested synthetic shapes, including Qwen3.6 27B projection-sized cases. That
is why it now lives behind a separate Cannoe backend instead of an option
inside plain Komodo.

Current 8-NPU A/B after the profiler-guided host-path patch on 2026-04-30
with tile `1024`, FP16, warmup `2`, iters `5`:

| Case set | Kernel | Source drop | Komodo total ms | Max abs drift |
|---|---|---:|---:|---:|
| GPTQ group sizes + act-order | Komodo | 0 | 1.8125 | 0.03125 |
| GPTQ group sizes + act-order | Cannoe | 0 | 1.9142 | 0.03125 |
| GPTQ group sizes + act-order | Komodo | 1 | 1.6394 | 0.03125 |
| GPTQ group sizes + act-order | Cannoe | 1 | 1.7173 | 0.03125 |
| Qwen3.6 27B GPTQ projections | Komodo | 0 | 1.2378 | 0.0625 |
| Qwen3.6 27B GPTQ projections | Cannoe | 0 | 1.3255 | 0.0625 |
| Qwen3.6 27B GPTQ projections | Komodo | 1 | 1.2979 | 0.0625 |
| Qwen3.6 27B GPTQ projections | Cannoe | 1 | 1.2861 | 0.0625 |

For Qwen3.6 27B GPTQ projection shapes, the recorded CANN plans used
`split_k` values `{1, 2, 8}` and vector dequant task counts `{320, 1920, 5440}`
across the six projection cases.

Current 8-NPU quick sweep:

- `quick_cannoe_keep_tile1024`: `0.3650ms`, no prefetch, `split_k=1`.
- `quick_cannoe_prefetch_keep_tile1024`: `0.3804ms`, prefetch enabled.
- `quick_native_drop_tile1024`: `0.3085ms`.
- `quick_prefetch_keep_tile1024`: `0.3231ms`.

Next real kernel work should move beyond host-issued prefetch hints and toward
an Ascend C custom op that can pipeline copy-in, int4 dequant, cube matmul, and
copy-out inside the device kernel.

## CANN 9 Rescan Impact

The 2026-05-01 torch-npu/CANN rescan did not change the native ACLNN conclusion:
`aclnnWeightQuantBatchMatmulV3` is useful for ABI and correctness probes, but it
does not remove the generic executor boundary or the device-side dequant
materialization problem. The useful new surface for the custom kernel is the
CANN 9 Ascend C public device API.

The next Cannoe implementation should prioritize these public CANN 9 paths:

- Replace scalar nibble unpack in the staged AIV producer with C API vector
  conversion from packed INT4 to FP16 in UB. Local headers expose
  `asc_int42half` and register-level `asc_int4x22half`.
- Feed the dequantized B tile to Cube through Matmul `TPosition::TSCM` with
  `CubeFormat::NZ`. The local CANN 9 headers show that non-TSCM local B operands
  are copied through Matmul workspace, while TSCM local B is passed by physical
  TSCM address to the Cube client.
- Keep the tile contract per-core resident and bounded. A VECOUT/TSCM tile is a
  producer-consumer handoff, not a persistent dense dequant cache.
- Use lower-level `asc/include/c_api/cube_datamove` and
  `asc/include/c_api/cube_compute` only if high-level Matmul cannot consume the
  staged tile. `asc_mmad_s4` is not a direct W4A16 solution because it computes
  `int4b_t x int4b_t -> int32_t`; using it would require quantized activations
  and a different accuracy contract.
- Defer `asc_datacache_preload` and explicit MTE/block sync tuning until the
  UB/L1 producer-consumer path exists and profiler counters show the next
  bottleneck.

2026-05-01 implementation update:

- Added a guarded CANN 9 staged-dequant producer build flag,
  `--experimental-cann9-vector-dequant`, which includes public
  `asc/include/c_api/asc_simd.h` and uses `asc_int42half_sync` to convert packed
  INT4 to FP16 lanes in UB before applying Komodo scale/offset into the bounded
  staging tile.
- Added `--experimental-vecout-consumer` as a Matmul template probe with
  `B_TYPE` at `TPosition::VECOUT`. This validates that the public CANN 9 Matmul
  surface accepts the intended UB/VECOUT B operand type and
  `SetTensorB(LocalTensor<half>)` call on the local 910B toolchain. Header
  inspection then showed that this path copies through Matmul workspace, so it
  is a compile/API probe rather than the final local handoff route.
- Added `--experimental-tscm-consumer` as the next Matmul template and data-move
  probe. It switches B to `TPosition::TSCM`/`CubeFormat::NZ` and compiles a
  staged GM-to-TSCM tile load plus a `SetTensorB(LocalTensor<half>)` probe. This
  is the correct structural route for the AIV dequant tile to AIC Cube handoff.
- Added `--experimental-tscm-runtime-handoff` as a narrow mixed-launch runtime
  probe. It handles the conservative full-N-tile/full-`base_k` K-tile subset by
  staging the B tile, copying it to TSCM/NZ, and invoking Cube Matmul for the
  visible output; unsupported shapes fall back to the scalar path. Fused bias is
  supported with the same split-K pattern used by CANN Matmul kernels:
  `SetBias` on the first K tile and `ClearBias` on later accumulating K tiles.
  The broad target remains direct dequant into the TSCM tile with multi-K
  producer/consumer scheduling.
- Validated the runtime probe on NPU0 for `M=8,K=64,N=8192,group_size=32`:
  finite output, `max_abs=0.0` versus CPU reference. The quick A/B timing was
  effectively flat (`3.646628 ms` TSCM runtime versus `3.651168 ms` non-runtime
  staged probe), which confirms the handoff is structurally live but not yet a
  speed path while it still stages through GM.
- Added `--experimental-tscm-direct-dequant`, which fills the B tile in UB with
  CANN 9 `asc_int42half_sync` vector INT4 decode plus Komodo scale/offset, then
  copies UB directly to TSCM/NZ with `DataCopy(..., Nd2NzParams)`. This removes
  the staged FP16 GM tile from the live handoff. The same NPU0 shape matched the
  CPU reference with `max_abs=0.0`; timing remained flat at `3.655767 ms`, so the
  speed-relevant follow-up is multi-K direct handoff and overlap.
- Added `--experimental-tscm-direct-multik`, which keeps the direct UB-to-TSCM
  path but iterates `base_k` B tiles and uses Cube Matmul accumulation for later
  K tiles. On NPU0, `M=8,K=128,N=8192,group_size=32,base_k=64` matched the CPU
  reference with finite output and `max_abs=0.0`. It was timing-neutral against
  a fresh current-source scalar package for the same shape (`7.132362 ms` versus
  `7.130773 ms`), so the next performance step is still overlapping the vector
  producer with Cube consumption rather than just broadening K coverage.
- Added a guarded direct multi-K scheduling pass that uses two TSCM B slots and
  hoisted scale/offset loads for larger direct tiles (`base_k >= 128`) or at
  least four K tiles. This lets AIV stage the next B tile after launching Cube
  on the current tile, while preserving the dependency before the next
  accumulation. NPU0 validation stayed finite: symmetric `M=8,K=512,N=8192` with
  `base_k=128` matched with `max_abs=0.0`; a nonzero-offset
  `M=8,K=128,N=8192,base_k=128` probe had `max_abs=0.00390625` and mean drift
  `5.9e-7`. Timings remained close to run noise, with `K=512,base_k=128` around
  `27.89-27.94 ms`.
- SVDQuant's useful transferable lesson is to keep scale/offset metadata
  lifetimes block-local and branch symmetric INT4 paths away from offset
  tensors entirely. Cannoe now keeps the direct CANN9 vector-dequant tile fill,
  single-word direct fallback, staged producer, and M=1 decode scalar fallback
  on explicit no-offset branches when `zero_offsets != 0`. That removes offset
  GM reads and the `x_sum * offset` accumulation from symmetric Qwen-style GPTQ
  decode while leaving asymmetric/nonzero-offset GPTQ on the previous hoisted
  offset path.
- Added fused-bias support to the direct TSCM runtime path. The kernel now uses
  CANN Matmul's split-K bias convention: `SetBias` before the first K tile for
  each output tile and `ClearBias` before later accumulating K tiles. NPU0
  validation for `M=8,K=128,N=8192,group_size=32,base_k=128` passed with
  nonzero offsets and FP16 bias (`max_abs=0.00012207`,
  `mean_abs=0.00000381`, `mean_ms=7.897985`); the paired no-bias probe was
  finite with `max_abs=0.00003052`, and a zero-offset group-32 bias probe stayed
  within `max_abs=0.00006104`.
- Fixed the scalar fused path's INT4 signed-nibble decode from xor-based
  sign extension to an explicit `raw < 8 ? raw : raw - 16` decode. The previous
  expression miscompiled lane 0 on the local CANN 9 package and produced
  `inf` outputs in every first lane of an 8-output pack. This blocker made the
  vector-dequant runtime validation ambiguous until corrected.
- Validation after the fix:
  `--experimental-staged-dequant --experimental-cann9-vector-dequant` built on
  CANN 9.0.0-beta.2 and produced finite controlled output for
  `M=8,K=1024,N=1024,group_size=32` with `max_abs=7.62939453125e-06`.
  An 8-NPU one-shard-per-device `gptq_group_sizes` staged sweep kept
  `max_abs=0.015625` for group-size 32/64/128/full and act-order 32/128; group
  size 16 still routes through the native group16 CANN path.
- Added `--experimental-vecout-runtime-handoff` as the first mixed-launch
  runtime path that feeds a local FP16 B tile directly to Cube Matmul instead of
  writing a staged FP16 weight tile through GM/L2. The flag implies staged
  metadata, CANN 9 device headers, VecOut B, and mixed AIC/AIV launch. It also
  copies the new template tiling-key header into both generated host and kernel
  trees so CANN 9 emits the AIV-only and `MIX_AIC_1_2` binaries from one op.
- Fixed the host workspace accounting for staged/Cube packages: CANN Matmul
  system workspace and Cannoe user tile workspace are now added together
  before `SetWorkspaceSizes`. The fused decision still uses only the bounded
  user staging bytes, so Cannoe does not make a full dense dequantized
  weight cache the default.
- VecOut runtime validation on 2026-05-01 passed an 8-NPU smoke with one shape
  per NPU: rows `1/4/8`, K `384/512/768/1024`, N `256/512`, and group sizes
  `32/64/128`. Drift against native CANN stayed within `max_abs=0.015625` and
  `mean_abs<=0.002598`.
- Two CANN 9 runtime lessons are now captured in code. First,
  `Matmul::IterateAll<false>(..., waitIterateAll=true)` must be paired with
  `WaitIterateAll`; using `waitIterateAll=false` either returned before the
  output was complete or hung on the later wait. Second, explicit multi-row
  `SetOrgShape` on this mixed VecOut handoff raised device error `507057`, so
  the current correctness path launches one Cube row at a time for `M>1`.
- The live VecOut direct tile fill intentionally falls back to scalar INT4 lane
  decode for now. The direct `asc_int42half_sync` fill compiled but produced
  wildly incorrect values, which means its input packing, element count, or lane
  order is still wrong for this use. Keep the CANN 9 vector conversion active in
  staged probes, but do not enable it for the live fused tile until that mapping
  is isolated.
- Added a compile-guarded `--experimental-vecout-local-a` batching probe. It
  implies VecOut runtime handoff, switches Matmul A to `TPosition::VECOUT`, and
  stages each `M x baseK` activation tile into contiguous UB before reusing it
  across the output-N tiles owned by that core. This is the current route around
  the failing GM-stride `SetOrgShape` experiment; keep it behind its own flag
  until runtime validation proves correctness and speed for `M>1`.
- The first local-A runtime smoke installed the generated package and passed on
  NPU0 for `M=4,K=512,N=256,group_size=64` with
  `base_m=16,base_n=-256,base_k=-128`. Drift versus native CANN was
  `max_abs=0.0078125` and `mean_abs=0.0014190673828125`. This is a proof that
  local A+B VecOut handoff can produce correct multi-row output, not yet enough
  evidence to make it default.
- The broader local-A validation kept each worker to one custom/native call.
  NPU0 passed five shapes over rows `1/4/8`, group sizes `32/64/128`, and K up
  to `1024`; the follow-up all-8-NPU sweep passed rows `1/2/3/4/7/8`, group
  sizes `32/64/96/128`, K `384/512/768/896/1024`, and N `256/512`. Worst drift
  was `max_abs=0.015625` and `mean_abs=0.0024566650390625`.
- Do not replace the local-A scalar activation staging loop with direct
  GM-to-VECOUT `DataCopy` unconditionally. The original all-row `DataCopy` probe
  compiled with CANN 9 but timed out before emitting a result, and a later
  unconditional retry regressed tiny decode tiles. The validated version now gates
  row-wise `DataCopy` to `rows >= 32`, where medium/larger local-A tiles benefit.
- A VecOut-only zero-offset B-fill specialization also failed validation: it
  passed the first three NPU0 shapes, then timed out on
  `M=4,K=512,N=256,group_size=128`. Keep the generic offset-aware direct B fill
  until the exact code-generation or barrier sensitivity is isolated.
- The local-A VecOut path now reuses each direct B tile across M tiles when
  `rows > base_m`. The previous loop filled the large dequantized B tile once per
  M tile; the new loop keeps the single-M-tile order unchanged but switches
  larger batches to `(K tile, N tile) -> M tile`, trading repeated B dequant for
  the much smaller local-A copy. A fresh all-8-NPU raw sweep covered rows
  `1/8/16/17/24/31/32/48`, group sizes `32/64/96/128`, K up to `1024`, and
  N `256/512`, with worst drift `max_abs=0.015625` and
  `mean_abs=0.001491546630859375`. The prior local-A package timed out at
  120 s on `rows=17,K=512,N=512,group_size=64`; the new package ran the same
  timing probe in `3.060030 ms`.
- With the `rows >= 32` local-A `DataCopy` gate, the current all-8-NPU
  planner-shaped warmed timing sweep improved from `4.369839 ms` mean to
  `4.066544 ms` mean. The largest gains were the medium/larger local-A cases
  `rows=32/96/129/160`; worst drift stayed `max_abs=0.015625`.
- A follow-up single-call 2D `DataCopyParams` activation copy was correct but
  slower than the per-row `DataCopy` gate: the legacy all-8-NPU mean regressed to
  `4.808621 ms`, and the planner-shaped mean was `4.145872 ms`. Keep the row-wise
  copy path for now.
- The accepted zero-offset B-fill follow-up keeps the generic direct local-B
  scheduler but splits the inner tile fill when the tiler has already marked the
  shape as symmetric zero-offset. It skips offset GM reads and offset adds in
  that case without enabling the earlier rejected VecOut-only specialization.
  The CANN 9 package passed the legacy all-8-NPU warmed sweep with
  `custom_ms_mean=4.510223` and the planner-shaped warmed sweep with
  `custom_ms_mean=4.026535`; worst drift stayed `max_abs=0.015625`.
- The accepted packed-B staging pass adds a bounded UB scratch tile for the
  packed INT4 B tile and fills it with row-wise `DataCopy` before scalar
  dequantization into the local FP16 B tile. This cuts scalar packed-weight GM
  reads without materializing dequantized FP16 weights in GM/L2. The legacy
  all-8-NPU warmed mean improved to `4.052616 ms`, and a retry planner-shaped
  all-8-NPU warmed sweep improved to `3.651770 ms`; worst drift stayed
  `max_abs=0.015625`. Do not promote `base_k=256` yet: the full planner probe
  timed out before worker output.
- The next accepted B-dequant fusion processes two adjacent packed columns per
  staged INT4 row in the zero-offset path, filling 16 local FP16 B lanes per
  inner-loop iteration. This reduced warmed means to `3.749311 ms` on the
  legacy sweep and `3.368754 ms` on the planner-shaped sweep with unchanged
  `max_abs=0.015625`. A strided 2D packed-B `DataCopyParams` probe was accurate
  but slower (`4.081596 ms` legacy mean), so keep row-wise packed copies.
- The local-B handoff barrier after staged dequant is now `PIPE_V` instead of
  `PIPE_ALL`; the packed GM-to-UB staging copy keeps the full pipe barrier
  before vector decode reads. This was effectively neutral on the legacy sweep
  (`3.749492 ms`) and improved the planner-shaped sweep to `3.364377 ms` with
  unchanged `max_abs=0.015625`.
- The zero-offset staged-B inner loop now unrolls to four adjacent packed
  columns when available, then falls back to the pair/single paths for tails and
  nonzero-offset shapes. This improved warmed means to `3.645413 ms` on the
  legacy sweep and `3.305584 ms` on the planner-shaped sweep with unchanged
  `max_abs=0.015625`.
- A CANN `msprof` PipeUtilization profile of the accepted quartet path on
  `rows=160,K=512,N=256,group_size=32,base_m=128,base_k=128` reported roughly
  `2.54 ms` per custom-op launch, `aic_scalar_ratio~=95%`,
  `aiv_scalar_ratio~=97%`, and Cube utilization around `4%`. The bottleneck is
  still scalar tile decode/control, not dense FP16 GM materialization. Rejected
  adjacent probes: row-offset hoisting regressed the legacy mean to
  `3.666554 ms`, scale UB staging regressed it to `4.125134 ms`, and live
  zero-offset `asc_int42half_sync` still produced invalid output
  (`max_abs=Infinity`) while slowing the planner smoke.

The full rescan and public/private API notes are in
`hw/torch_npu_cann_9_api_scan.md`.

The Python JIT bridge now resolves CANN roots from `ASCEND_HOME_PATH`, then
`ASCEND_TOOLKIT_HOME`, then the installed latest-toolkit symlinks before falling
back to older versioned paths. This avoids silently building the Cannoe V3
or Ascend C bridge against the stale CANN 8.5.1 tree when a stripped shell omits
the normal Ascend environment variables.

Use `scripts/validate_cannoe_ascendc_raw.py` for raw Ascend C package
smokes. It takes `--bridge-lib`, `--opp-install`, and `--devices`, launches one
worker per device, and keeps each worker to one custom/native call. The first
run against `/tmp/cannoe_vecout_local_a_install` reproduced the manual
8-NPU local-A sweep with all eight cases passing and worst drift
`max_abs=0.015625`, `mean_abs=0.0024566650390625`.
Set `--warmup` and `--iters` for repeatable timing while preserving the same
custom/native accuracy check; worker rows include `custom_ms`, and summaries add
`custom_ms_min`, `custom_ms_mean`, `custom_ms_max`, `warmup`, and `iters`.
Use `warmup >= 1` for steady-state timing because the first custom-op call
includes runtime setup. On the current local-A package, an all-8-NPU warmed sweep
over `/tmp/cannoe_local_a_b_reuse_cases.json` passed with custom latency
min/mean/max `2.284225/4.648382/8.693375 ms` and worst drift
`max_abs=0.015625`, `mean_abs=0.001491546630859375`.
The same harness passed with positive `base_k=128`, so the module planner can
mark all symmetric GPTQ fused calls as zero-offset while retaining a validated
positive-`base_k` fallback shape.

2026-05-18 mixed-launch lifecycle update:

- A no-Matmul `MIX_AIC_1_2` entry diagnostic passed on all eight NPUs with
  `warmup=10,iters=50`, averaging `0.041555 ms` and returning marker `911`.
  This isolates custom-op registration, tiling-key selection, and the basic
  mixed AIC/AIV launch as repeat-safe.
- A Matmul-registration diagnostic then showed the generic mixed Cube path must
  use the same KFC workspace choreography as the known-good mixed-entry and
  AIC/TSCM branches: AIC clears system workspace, AIV waits on
  `WORKSPACE_SYNC_ID`, and only then both sides register the Matmul object.
  The old unconditional `clearWorkspace` path timed out on all eight NPUs.
- After that fix, the Matmul-registration diagnostic passed on all eight NPUs
  with marker `915` and mean `0.041603 ms`, proving `REGIST_MATMUL_OBJ` itself
  is repeat-safe when the workspace event is owned by AIC.
- The real VecOut/local-A runtime package rebuilt with the same fix passed the
  default all-8 raw validation sweep (`warmup=2,iters=6`) with min/mean/max
  `1.936433/3.515399/5.107260 ms`, worst `max_abs=0.015625`, and worst
  `mean_abs=0.0024566650390625`. This restores repeatable fused local-A
  execution; the remaining target is replacing the VecOut local-B path with a
  true TSCM/NZ dequant-to-Cube handoff.
- `--strategy tscm-direct-local-a` is the first all-8 passing direct
  dequant-to-TSCM/NZ-to-Cube path. The key fixes were using a non-transposed
  TSCM B handoff, staging the live A tile into VECOUT so row stride is `base_k`
  instead of full `K`, and using scalar A fill for the TSCM-local-A variant.
  Validation passed with min/mean/max `2.246633/4.456605/8.655868 ms`, worst
  `max_abs=0.015625`, and worst `mean_abs=0.0024566650390625`. This is
  structurally closer to the final kernel than VecOut local-B, but not yet
  faster.
- The existing planner-shaped `base_n` policy is better for TSCM-local-A than
  a fixed `base_n=256`: use `base_n=128` for the six validated smaller cases
  and keep `base_n=256` for the two `rows=8,K=1024,N=512` cases. The hybrid
  all-8 sweep passed with min/mean/max `1.144248/3.161435/8.643608 ms`, making
  the true TSCM/NZ handoff faster than the fixed-256 VecOut/local-A sweep
  (`3.515399 ms` mean) on this raw case set while preserving the same drift.
- The direct TSCM/local-A path now enables CANN Matmul sequential GM writes only
  when the visible C tile is contiguous (`m_len <= 1` or the N tile spans the
  full row). This keeps multi-row strided output safe while improving decode-like
  tiles. The warmed all-8 fixed-256 sweep improved from
  `2.160392/4.317659/8.454348 ms` min/mean/max to
  `2.059953/4.107422/8.158103 ms`; worst drift stayed `max_abs=0.015625` and
  `mean_abs=0.0024566650390625`.
- The same contiguous-output write guard is now shared by VecOut/local-A,
  direct TSCM/local-A, and the staged TSCM fallback. The VecOut/local-A package
  passed the all-8 fixed-256 raw sweep with min/mean/max
  `1.919938/3.478979/5.051962 ms`, improving the documented repeat-safe
  VecOut/local-A reference mean of `3.515399 ms`; worst drift stayed
  `max_abs=0.015625` and `mean_abs=0.0024566650390625`. The rebuilt direct
  TSCM/local-A package also passed all eight cases with
  `2.062128/4.108074/8.156297 ms`, confirming the helper did not regress the
  true TSCM/NZ handoff path.
- `scripts/validate_cannoe_ascendc_raw.py --planner-tiles` now applies the
  runtime planner's validated fused tile policy to raw package sweeps. On the
  same rebuilt direct TSCM/local-A package, this planner-shaped all-8 run used
  `base_n=128` for the six validated fast cases and `base_n=256` for the two
  unsafe `rows=8,K=1024,N=512` cases. It passed with min/mean/max
  `1.050940/2.927955/8.144283 ms`, improving the fixed-256 mean while
  preserving worst drift at `max_abs=0.015625` and
  `mean_abs=0.0024566650390625`.
- The direct TSCM/local-A zero-offset B-tile fill now processes four adjacent
  packed INT4 columns per inner K sweep, matching the already validated
  VecOut packed-tile unroll but reading directly from GM into the live UB tile.
  A fresh `--strategy tscm-direct-local-a` package passed the all-8
  `--planner-tiles` raw sweep with min/mean/max
  `1.072683/2.744680/6.195503 ms`; worst drift remained
  `max_abs=0.015625` and `mean_abs=0.0024566650390625`.
- After every Cannoe kernel change, including micro-optimizations that only
  target raw fused probes, run the full Qwen3 27B GPTQ FP16 projection gate with
  all default layers (`q,k,v,gate,up,down`) to catch module-level regressions.
  The post-unroll NPU0 gate passed with total repeat `0.9371 ms`:
  `q=0.0772`, `k=0.0730`, `v=0.0709`, `gate=0.2214`, `up=0.2188`,
  `down=0.2757`; JSON artifact
  `/tmp/cannoe_qwen3_27b_full_gate_after_tscm_unroll.json`.
- The large GPTQ down-projection default prepack tile is now narrowed further to
  `tile_n=320` for `group_size=32`, `K>=16384`, and `4096<=N<=8192`. This keeps
  the plain native CANN fast path while reducing cold prepack workspace for the
  Qwen3 27B down shape. The NPU0 down-only sweep found `tile_n=320` at
  `0.2568 ms`, `241.8 MB` peak versus the previous `tile_n=512` at
  `0.2765 ms`, `244.9 MB` peak in the same sweep. Full Qwen3 27B FP16 gate
  repeats remained noisy but favored the new tile: `tile_n=320` totals
  `0.9288 ms` and `0.9161 ms`, while `GPTQMODEL_KOMODO_PREPACK_TILE_N=512`
  totals were `0.9718 ms` and `1.0140 ms`. Count this as a validated native
  CANN fast-path retune and cold/prepack memory improvement, not a substitute
  for the true fused Ascend C dequant-to-Cube kernel target.
- A planned-path `inner_precise` probe for the same Qwen3 27B down shape did not
  validate. Down-only NPU0 measurements were: default plain-native `0.2625 ms`,
  forced planned `inner_precise=0` `0.2756 ms`, and forced planned
  `inner_precise=1` `0.2901 ms`. Keep the default down projection on the bound
  plain-native path; do not route it through the heavier Cannoe plan just to
  pass the optional `inner_precise` argument.
- A public `Matmul::IterateBatch` partial-sum replacement for the same TSCM
  direct path compiled but timed out on all eight raw workers at `120s`. Keep
  the validated `IterateAll` accumulation route until a smaller
  `IterateBatch` lifecycle diagnostic passes.
- The raw Ascend C validator now has a `--case-preset qwen3_27b_down` target
  for the actual large decode projection shape (`M=1,K=17408,N=5120,group=32`).
  This is the current fused-kernel target guard. A fresh CANN 9
  `tscm-direct-local-a` package still passed the eight default small raw cases
  (`max_abs=0.015625`, `mean_abs=0.0024566650390625`) but failed the Qwen down
  target at `base_n=256,base_k=128`: `custom_ms=851.9068`,
  `max_abs=0.21875`, `mean_abs=0.0218505859375`. The matching tile sweep also
  failed for `base_n in {128,256}` and `base_k in {64,128}` with best observed
  latency `801.015 ms` and worst `max_abs=0.34375`. A later narrow down-only
  sweep found `base_n=128,base_k=256` is finite and directionally better
  (`custom_ms=745.4321`, `max_abs=0.15625`,
  `mean_abs=0.0155792236328125`), so the raw validator planner now uses this as
  the Qwen down diagnostic tile. The planner replay selected
  `base_n=-128,base_k=-256` and reproduced the same drift at
  `custom_ms=763.5723`; after rebasing onto a newer `main`, repeat replays
  ranged from `723.0416` to `833.5908` with the same drift. Do not widen further
  without a new resource proof: `base_k in {512,1024}` and
  `base_n=256,base_k>=256` faulted with runtime `507015` AIC/AIV memory-access
  errors. This confirms the remaining fused work is not ordinary tile retuning;
  large-K accumulation must stop writing partial C through GM atomics and must
  keep accuracy within the established raw envelope before runtime enablement.
- The validator also has `--case-preset qwen3_27b_down_onehot`, which keeps the
  same Qwen3 27B down shape but activates only one K lane at group and
  `base_k` boundaries. The first all-8 NPU run passed exactly with
  `max_abs_max=0`, `mean_abs_max=0`, and `diff_nonfinite_count=0`; timing stayed
  slow (`custom_ms_mean=903.9163`) because it still launches the full fused
  tile loop. A rebased all-8 repeat later saw one NPU0 `one_hot_k=0` worker
  report `diff_nonfinite_count=256`, and an isolated rerun of that exact case
  passed with `diff_nonfinite_count=0`. Treat this as a boundary diagnostic
  rather than a runtime enablement gate: it isolates the main deterministic
  Qwen down drift to multi-K accumulation/partial-C lifecycle, while still
  keeping non-finite output visible in future runs.
- Added fixed-seed `qwen3_27b_down_pairwise`, `qwen3_27b_down_sparse`, and
  `qwen3_27b_down_random_scale` presets to bracket the large-down failure. The
  pairwise guard passed exact for same-tile, adjacent-tile, middle, and
  far-apart K pairs. Sparse ramped activations passed up to 128 active K lanes,
  reproduced tile-local non-finites at 256 active K lanes, and passed again at
  512 lanes in the sampled run. Dense random activations with the same weight
  seed passed through `input_scale=0.1`, then failed at `0.25`, `0.5`, and
  `1.0` with max drift scaling roughly linearly up to `0.21875`. This points at
  dense accumulation magnitude / partial-C stability, not simple K-tile
  coverage, as the next fused-kernel target.

## aclnn V3 Probe

`scripts/probe_cannoe_v3.py` builds
`gptqmodel_ext/cannoe/wq_bmm_v3_probe.cpp`, registers
`torch.ops.gptqmodel_cannoe.w4a16_matmul`, and calls
`aclnnWeightQuantBatchMatmulV3` directly with Komodo's packed INT4 weights. The
probe exposes packed `int32 [K, N / 8]` storage as a logical `ACL_INT4 [K, N]`
tensor.

Runtime opt-in:

```bash
GPTQMODEL_CANNOE_V3=1
GPTQMODEL_CANNOE_FUSED_REQUIRE=1
GPTQMODEL_CANNOE_FUSED_OP=gptqmodel_cannoe.w4a16_matmul
```

The bridge caches repeatable ACL executors and CANN workspace tensors by
default. Disable these only for isolation:

```bash
GPTQMODEL_CANNOE_V3_EXECUTOR_CACHE=0
GPTQMODEL_CANNOE_V3_WORKSPACE_CACHE=0
```

Latest local checks:

- `M=8,K=256,N=256`, group sizes `0`, `32`, `64`, `128`: exact match against
  native `torch.ops.npu.npu_weight_quant_batchmatmul`.
- Same shape with FP16 bias: exact match.
- `M=1,K=4096,N=4096,group_size=128`: exact match in the final single probe.
- With the probe preloaded, `CannoeLinear` selected `fused_w4a16_matmul`
  and matched the torch baseline with `max_abs=0.0009765625`.

## Inner-Precise Shape Policy

`GPTQMODEL_CANNOE_INNER_PRECISE` accepts `auto`, `0`, or `1`. The default
is `auto`.

The current auto rule enables `inner_precise=1` only for q-like group-32 decode
projections:

```text
rows <= 16
group_size == 32
K >= 4096
K <= N <= 2K
```

Same-NPU direct CANN measurements on 2026-04-30 showed the reason for the narrow
gate:

| Shape | group | inner=0 ms | inner=1 ms | 0-vs-1 drift | Auto |
|---|---:|---:|---:|---:|---|
| `M=1,K=5120,N=6144` q-proj | 32 | 0.07889 | 0.07687 | 0 | `1` |
| `M=1,K=5120,N=1024` k/v-proj | 32 | 0.04674 | 0.04687 | 0 | `0` |
| `M=1,K=5120,N=17408` gate/up | 32 | 0.23563 | 0.23547 | 0 | `0` |
| `M=1,K=17408,N=5120` down | 32 | 0.31400 | 0.31388 | 0 | `0` |
| `M=1,K=4096,N=4096` balanced | 128 | 0.03383 | 0.03472 | max `0.015625` | `0` |

Group-32 produced exact outputs in these checks, while the sampled group-128
balanced shape drifted when forced to `inner_precise=1`. Keep forced `1` as a
profiling override, not a default, until a shape is added to the table.

V3 workspace-cache pass:

| Shape | group | workspace cache off ms | workspace cache on ms | cache drift |
|---|---:|---:|---:|---:|
| `M=1,K=5120,N=6144` q-proj | 32 | 0.07620 | 0.07614 | 0 |
| `M=1,K=5120,N=1024` k/v-proj | 32 | 0.05209 | 0.04629 | 0 |
| `M=1,K=17408,N=5120` down | 32 | 0.30607 | 0.29884 | 0 |
| `M=1,K=4096,N=4096` balanced | 128 | 0.05072 | 0.04080 | max `1.53e-05` |

8-NPU validation with the q-like auto rule, V3 bridge, and workspace cache
enabled passed 8/8:

| Case set | Kernel | Source drop | Komodo total ms | Max abs drift | Auto inner=1 |
|---|---|---:|---:|---:|---|
| GPTQ group sizes + act-order | Komodo | 0 | 1.7750 | 0.03125 | none |
| GPTQ group sizes + act-order | Cannoe V3 bridge | 0 | 2.4404 | 0.03125 | none |
| GPTQ group sizes + act-order | Komodo | 1 | 1.6649 | 0.03125 | none |
| GPTQ group sizes + act-order | Cannoe V3 bridge | 1 | 2.4540 | 0.03125 | none |
| Qwen3.6 27B GPTQ projections | Komodo | 0 | 1.2569 | 0.0625 | n/a |
| Qwen3.6 27B GPTQ projections | Cannoe V3 bridge | 0 | 1.6005 | 0.0625 | q-proj |
| Qwen3.6 27B GPTQ projections | Komodo | 1 | 1.2814 | 0.0625 | n/a |
| Qwen3.6 27B GPTQ projections | Cannoe V3 bridge | 1 | 1.7097 | 0.0625 | q-proj |

Earlier 8-NPU A/B with `GPTQMODEL_CANNOE_V3=1` passed 8/8. The V3 bridge
was correct but slower than the current native CANN path:

| Case set | Kernel | Source drop | Komodo total ms | Max abs drift |
|---|---|---:|---:|---:|
| GPTQ group sizes + act-order | Cannoe V3 bridge | 0 | 2.5008 | 0.03125 |
| GPTQ group sizes + act-order | Cannoe V3 bridge | 1 | 2.5408 | 0.03125 |
| Qwen3.6 27B GPTQ projections | Cannoe V3 bridge | 0 | 1.6629 | 0.0625 |
| Qwen3.6 27B GPTQ projections | Cannoe V3 bridge | 1 | 1.9257 | 0.0625 |

This confirms V3 API access, but it is still a raw CANN op call boundary. Even
with descriptor, executor, and workspace reuse, the remaining speed work is the
true Ascend C fused operator that avoids full dequantized FP16 weight
materialization through GM/L2 and removes the generic ACLNN call boundary.

## Ascend C Raw Validation Notes

The raw Ascend C validation harness now tolerates CANN stdout noise by scanning
worker output for the last JSON object. This is needed on CANN 9.0.0-beta.2
because tiling-struct warnings can be emitted on the same line as the worker
result.

The zero-offset side band is now threaded through the scalar M1/M2/M4 row
fallbacks, matching the existing M8 zero-offset row path. Validation used a
fresh CANN 9 non-mixed staged/vector package at the standard `base_n=256` tile
width: positive `base_k=128` passed rows `1/4/8`, and negative `base_k=-128`
passed rows `1/2/4/8` with worst `max_abs=0.015625`.

## Native CANN Bias Fusion

The 2026-05-03 two-NPU pass used only physical NPUs `0,1` after the device
limit changed. A forced prepack-tile sweep did not justify a broader tile
policy change: Qwen3.6-35B-A3B GPTQ tile `256` and Qwen3.6-27B AWQ tile `1536`
both looked promising in one sweep, but paired repeats across NPU0/NPU1 lost on
average. Keep the current auto tile rules.

AWQ Cannoe can fuse FP16 bias into `npu_weight_quant_batchmatmul`, but the
benefit is shape-sensitive. Group-32 Qwen3.6-27B AWQ got faster with fused bias
but widened the drift envelope (`max_abs` moved from `1.0` to `2.0` on the
synthetic projection benchmark), so Cannoe keeps group-32 AWQ bias unfused by
default.

Group-128 Qwen3.6-35B-A3B AWQ validated cleanly and is now fused by default:

| Case set | Mode | Runs | Mean Cannoe total ms | Paired new/old | Max abs drift |
|---|---|---:|---:|---:|---:|
| Qwen3.6-35B-A3B AWQ | old unfused bias | 6 | 1.011527 | 1.0000 | 1.0 |
| Qwen3.6-35B-A3B AWQ | group-128 fused bias | 6 | 0.842399 | 0.8338 | 1.0 |
| Qwen3.6-27B AWQ | old unfused bias | 2 | 1.214885 | 1.0000 | 1.0 |
| Qwen3.6-27B AWQ | group-32 gated unfused | 2 | 1.236357 | 1.0189 | 1.0 |

Dense dequantized weight caching remained disabled throughout
(`GPTQ_CACHE_DEQUANTIZED_WEIGHTS=0`).

## GPTQ BF16 Decode Support

The 2026-05-04 two-NPU pass enabled BF16 inputs for GPTQ Cannoe only. CANN
9.0.0 `npu_weight_quant_batchmatmul` still rejects BF16 activations/scales for
the int4 path, so Cannoe casts BF16 activations to FP16 for the native CANN
call and casts the result back to BF16. This keeps the fast native packed-int4
path available for BF16 decode-shaped GPTQ layers without enabling dense
dequantized weight caching.

Validated with physical NPUs `0,1` and `GPTQ_CACHE_DEQUANTIZED_WEIGHTS=0`:

| Case set | Dtype | NPU | Cannoe total ms | Speedup vs Torch reference | Max abs drift | Min cosine |
|---|---|---:|---:|---:|---:|---:|
| Qwen3.6-27B GPTQ | BF16 | 0 | 1.475314 | 91.65x | 0.5 | 0.999995589 |
| Qwen3.6-35B-A3B GPTQ | BF16 | 1 | 1.480676 | 4.77x | 0.25 | 0.999994993 |
| GPTQ group sizes + act-order | FP16 | 0/1 | 1.527064 mean | 3.50x mean | 0.03125 | unchanged |

AWQ BF16 was deliberately not enabled in this pass. It ran fast, but synthetic
Qwen3.6 AWQ BF16 checks widened max-abs drift to `8.0-16.0`, so AWQ Cannoe
remains FP16-only until there is a tighter BF16 path.

Follow-up bias replacement removed the repeated BF16-bias to FP16 conversion
without retaining a second resident bias tensor. The first native CANN BF16
call promotes the registered bias buffer itself to FP16 on the active NPU; the
output is still cast back to BF16. A direct state probe confirmed
`before_bias_dtype=torch.bfloat16`, `after_bias_dtype=torch.float16`,
`output_dtype=torch.bfloat16`, and no `_cann_bias_cache` attribute.

Validation kept the same drift envelope and improved BF16 GPTQ totals:

| Case set | Before bias replacement ms | Replaced-bias ms | Change | Max abs drift |
|---|---:|---:|---:|---:|
| Qwen3.6-27B GPTQ BF16 | 1.475314 | 1.360772 | 7.8% faster | 0.5 |
| Qwen3.6-35B-A3B GPTQ BF16 | 1.480676 | 1.240929 | 16.2% faster | 0.25 |

The FP16 GPTQ group-size/act-order regression stayed clean on NPU1:
`1.458037ms`, max abs drift `0.03125`.

SVDQuant's 910B path keeps scale metadata short-lived: each K-block loads scale
rows into UB, consumes them, and reuses the local storage. Cannoe now applies a
safe version of the same memory rule to its BF16 native direct plan. When
`GPTQMODEL_CANNOE_BF16_NATIVE=1` explicitly forces direct BF16 native
execution for BF16-resident Cannoe modules, eager prepack builds the BF16
scale/offset plan directly instead of first creating an FP16 native plan and
then retaining BF16-converted scale/offset copies. If source weights were
already dropped and only an FP16 packed plan exists, the forced BF16 path
converts that metadata once and evicts the stale FP16 scale/offset backing. In
the default shape-auto policy, Cannoe keeps the FP16 backing plan so larger
non-direct BF16 shapes can still fall back to the FP16 native CANN call.

## CANN Profiling Read

## CANN 9.1-beta1 Kernel Surface

The 2026-05-19 SDK sweep found the active toolkit at
`/usr/local/Ascend/cann-9.1.0-beta.1`. The most useful new local surface for
Cannoe is not a replacement one-call torch op; it is a fuller 910B CANN source
and header package that exposes the pieces needed to finish the fused design.

Immediate candidates:

- `aclnnWeightQuantBatchMatmulV3` is public and exported, and its C API exposes
  `innerPrecise` directly. Keep using the existing V3 probe to expand the
  shape table, but do not promote it as the fast path without a full Qwen
  q/k/v/gate/up/down win.
- `aclnnWeightQuantBatchMatmulNz` is public and exported for FP16/BF16
  activation, INT4 NZ right weight, group scale/offset, optional bias, and
  FP16/BF16 output. There is no installed torch-npu Python binding for it, so
  the next native fallback experiment needs a small C++ ACLNN bridge and a
  persistent packed INT4 NZ weight layout. This must not create or cache a
  dense FP16 weight tensor.
- `aclnnConvertWeightToINT4Pack` and `npu_convert_weight_to_int4pack` should be
  compared before implementing the NZ bridge, because the current Cannoe and
  Komodo path already depends on CANN's packed INT4 layout.
- `aclnnQuantMatmulV5`, `aclnnFusedQuantMatmulWeightNz`, and dual-level MX/FP4
  NZ matmuls are not direct GPTQ FP16 decode paths because they quantize the
  activation side too. Keep them as future A4W4/A8W4 or FP4 references, not the
  current speed target.

The most valuable 9.1 reference source is CANN's CMCT AIV/AIC antiquant matmul
pattern:

```text
opp/built-in/op_impl/ai_core/tbe/impl/ops_nn/ascendc/common/cmct/kernel/kernel_matmul_a_prefetch_b_antiquant.h
opp/built-in/op_impl/ai_core/tbe/impl/ops_nn/ascendc/common/cmct/prologue/tile/tile_antiquant.h
```

That reference splits the operator structurally the way Cannoe needs to:
AIV iterates B-weight tiles plus scale/offset tiles and runs antiquant
prologues, while AIC preloads A and runs matmul over the same scheduled N/M
tiles. The next fused-kernel pass should use this pattern for scheduling and
tile ownership, while keeping Cannoe code on public Ascend C headers. Do not
include private `opp` implementation headers directly.

Use the profiling helper for single-shape CANN traces:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID ASCEND_RT_VISIBLE_DEVICES=6 \
python scripts/profile_cannoe_npu.py \
  --mode cannoe \
  --case qwen3_6_27b_gptq_down_proj \
  --profiler-level level1 \
  --aic-metrics PipeUtilization
```

The helper writes the raw CANN profiler output and a compact JSON summary. Level
1 summaries include AIC/AIV counters from `kernel_details.csv`.

On the local 910B1 host, the Qwen3.6 27B GPTQ down-projection trace reported:

- Native CANN op: `aclnnWeightQuantBatchMatmulV2`.
- Level-1 kernel duration: `856.3us` across 3 measured iterations.
- Cube utilization: about `87.7%`.
- AIC MAC ratio: about `5.9%`.
- AIV MTE2 ratio: about `75.9%`.

That profile points at vector/dequant and memory-movement pressure, not missing
host prefetch. The explicit prefetch trace emitted 36 `npu_prefetch` host calls
over 12 iterations and no separate device kernel, so prefetch remains opt-in and
the default no-prefetch path bypasses that helper completely.

## Ascend C Fused Prototype Status

The TSCM direct local-A fused prototype now has a correctness fix for one
critical failure mode: direct multi-K paths must not use the per-packed-word
`asc_int42half_sync` helper. With that helper disabled, the raw asymmetric
offset probe at `rows=8,K=1024,N=512,group=32,base_n=256,base_k=128` is finite
with `max_abs=0.015625`, and the standard raw validator passes on all 8 NPUs.

That is not yet the target runtime path for Qwen-sized GPTQ projections. A
forced Qwen3.6-27B GPTQ FP16 run on NPU0 is finite but still slow:

| Path | Total repeat ms | Worst max abs | Status |
|---|---:|---:|---|
| Cannoe native CANN baseline | 0.940 | 0.0625 | production path |
| Komodo | 1.102 | 0.0625 | comparison baseline |
| Cannoe fused TSCM local-A prototype | 997.459 | 1.203125 | disabled prototype |

The current fused prototype still calls high-level CANN Matmul once per
`base_k` tile and writes/reads FP16 C through GM between K tiles. That defeats
the intended dequant-to-Cube handoff for large reductions. A deferred
`GetTensorC` attempt timed out on all eight raw validator cases, so the next
real speed step should move accumulation to a lower-level Cube path that can
keep partial sums in Cube-local storage instead of materializing partial C in
GM/L2 after every K tile.

One important API detail: `Matmul::IterateAll(gm, ...)` does not take a
partial-sum flag. Its first argument after `gm` is `enAtomic`, so using
`k_tile != 0` there still writes partial C through GM rather than accumulating
in CO1. The guarded public `tscm-iterate-getc-diagnostic` strategy now keeps a
reproducible probe for synchronous `Iterate(enPartialSum)` plus final
`GetTensorC`; a fresh CANN 9 build loaded successfully, but all eight default
raw validator workers timed out at 180 seconds with no stderr. That keeps
lower-level Cube/L0C control, or a different CANN Matmul lifecycle, as the
active target.

Two TSCM lifetime probes further narrow the Qwen down failure:

- Adding an explicit `PIPE_MTE2` barrier after direct UB-to-TSCM B-tile copies
  kept the standard raw guard green on NPU0 (`8/8` pass,
  `max_abs=0.015625`, `mean_abs=0.0024566650390625`) but the Qwen3 27B down
  target still failed with `custom_ms=799.7961`, `max_abs=0.21875`,
  `mean_abs=0.0218505859375`.
- The guarded `tscm-direct-local-a-serial-k` strategy waits for each Cube K-tile
  compute before filling/loading the next B tile. It also kept the standard raw
  guard green on NPU0 (`8/8` pass, `max_abs=0.015625`,
  `mean_abs=0.0024566650390625`) but the Qwen down target still failed with
  `custom_ms=812.3234`, `max_abs=0.21875`, `mean_abs=0.0218505859375`.

These probes rule out async UB-to-TSCM copy completion and K-overlap/TSCM-slot
lifetime as the primary large-down drift source. The remaining target is still
the multi-K partial-C accumulation path itself.

The next down-specific planner probe keeps `base_n=128` and raises only
`base_k` to `256`, cutting the Qwen down K-loop count from `136` to `68`.
On NPU0 this improved the fused diagnostic to `custom_ms=745.4321`,
`max_abs=0.15625`, and `mean_abs=0.0155792236328125`, still outside the raw
acceptance envelope but better than the `base_k=128` TSCM lifetime probes. The
automatic planner path replayed between `custom_ms=723.0416` and `833.5908`
with the same drift across two rebases. Larger down tiles crossed the current
local-memory/lifecycle boundary and faulted with runtime `507015`, so the next
correctness step remains true partial-C accumulation control rather than larger
staged B tiles.

The guarded `aic-staged-gm-visibility-diagnostic` strategy now isolates another
handoff boundary. The marker-only build has AIV write two FP16 markers into the
bounded custom-op user workspace after AIC-owned workspace clear and AIV
`WORKSPACE_SYNC_ID` wait, then uses `SyncAll<false>()` before AIC readback. It
still fails on all eight default raw cases with runtime `507015`,
`fftsplus aivector error`, and D-cache-to-UB bus-response errors before any
marker reaches host JSON. That means the blocker is not the INT4 dequant loop
or `GetUserWorkspace()` timing alone: standalone AIV writes to the generated
ACLNN workspace are not a safe AIC handoff mechanism in this mixed launch. The
next fused attempt should use CANN/Matmul-managed KFC/SCM buffer ownership or a
lower-level Cube API path for the producer-consumer buffer.

## 2026-05-21: SVDQuant 910B Kernel Lessons

I inspected `Qubitium/svdquant-kernels` at local commit `eeed047` after cloning
from https://github.com/Qubitium/svdquant-kernels. The relevant Ascend source is
`csrc/kernels/gemm_w4a4/ascend/`: it is a W4A4 SVDQuant kernel, not a GPTQ
W4A16 kernel, so the raw math path is not directly portable to Cannoe without
activation quantization and a different accuracy contract.

Useful structural pieces to carry into Cannoe:

- Use explicit AIC/AIV mixed launch ownership. SVDQuant uses a cube-side INT4
  main path and vector-side scale/finalization work instead of relying on a
  high-level matmul call per K tile.
- Use a bounded ring handoff between Cube and Vector work. SVDQuant stages
  int32 partials in a small ring, then AIV applies per-block scales and casts to
  FP16. That maps to our desired design better than writing full dequantized
  FP16 weights through GM/L2.
- Keep scale lifetime tied to K blocks. Their per-64-K block scales force a
  drain after each K block, which is a useful model for GPTQ group-scale
  scheduling even though the data type contract differs.
- Avoid manual user-workspace AIV-to-AIC sharing for the fused path. Our marker
  and GM-visibility probes keep showing that generated ACLNN workspace is not a
  safe producer/consumer buffer for mixed AIC/AIV handoff in this lifecycle.

Applied follow-up:

- The staged dequant ring depth is now explicit through
  `GPTQMODEL_CANNOE_STAGING_SLOTS`. Default runtime behavior remains a two-slot
  ring for large multi-wave shapes, while both the Python planner and Ascend C
  host tiler cap allocation to the number of waves that can actually be used.
  Dense-equivalent one-wave staging remains disabled by the existing guard, so
  Cannoe does not allocate a temporary that defeats the memory premise of GPTQ.
- The knob accepts `1..8`, matching the current Cannoe logical block cap. Set
  `GPTQMODEL_CANNOE_STAGING_SLOTS=6` for SVDQuant-style ring-depth experiments
  on large shapes without changing the custom-op ABI or staging tensor layout.
  The workspace formula is now `aligned(base_k * base_n * sizeof(fp16)) *
  staging_blocks * min(requested_slots, staging_waves)`.
- The staged/Cube dense guard now counts the full allocated workspace:
  `staging_workspace + cube_system_workspace`. This matches the Ascend C host
  tiler allocation and prevents small Cube-consumer probes from passing the
  memory gate while hiding CANN's 16 MiB Matmul system workspace.
- The Ascend C kernel now resets AIV vector masks with `set_mask_norm()` and
  `set_vector_mask(-1, -1)` at vector-side entry points before INT4 dequant,
  VecOut/TSCM handoff work, and marker diagnostics. This follows SVDQuant's
  AIV hygiene pattern and removes a plausible source of stale-mask lane
  corruption while fused AIC/AIV handoff work continues.

- The Python plan cache now includes `GPTQMODEL_CANNOE_STAGING_SLOTS` in both
  the normal and fast hot-cache keys. This keeps ring-depth trials honest:
  changing the requested ring depth rebuilds the plan and workspace estimate
  instead of reusing stale two-slot metadata.
- CANN9 vector INT4 dequant helpers now fence `PIPE_V` after
  `asc_int42half_sync()` before scalar code consumes the dequantized UB lanes.
  SVDQuant's vector path is strict about V-pipe ordering when a UB region is
  reused or read immediately after vector work; Cannoe now follows the same
  rule in the vectorized dequant staging helpers.
- Native GPTQ/AWQ forwards now use the env-aware `_cann_plan()` hot cache
  directly instead of a second `_cann_native_hot_plan` bypass. This keeps
  staging slots, base tile sizes, prefetch, and fused-path toggles coherent
  during ring-depth and workspace sweeps.
- Plan-affecting environment names are centralized in
  `_CANNOE_PLAN_ENV_NAMES`, and the native tuning detector extends that same
  list for non-tiling native-pack knobs. Future SVDQuant-style sweeps for ring
  depth, base tile sizes, prefetch windows, fused-op selection, and
  inner-precise mode now share one cache-key source of truth, which avoids
  stale plan reuse when a new performance or VRAM knob is added.

Parallel validation used one experiment per NPU with CANN 9.1.0-beta.1. The
raw validator now avoids unrelated public ACLNN parser failures by creating test
tensors on CPU, delaying custom OPP exposure until the custom op is called, and
using a CPU reference by default. `--reference npu` remains available when the
native `aclnnWeightQuantBatchMatmulV3` path is healthy.

| NPU | Experiment | Result | Metric |
| ---: | --- | --- | --- |
| 0 | `tscm-direct-local-a`, Qwen down | Rejected | `507015`; AICore illegal instruction / unaligned UUB |
| 1 | `tscm-direct-local-a-serial-k`, Qwen down | Rejected | `507015`; AICore illegal instruction / unaligned UUB |
| 2 | `tscm-iterate-getc-diagnostic`, default cases | Rejected | hung on a small default case after earlier children; killed manually |
| 3 | `aic-tscm-syncall-diagnostic`, Qwen down | Passed marker diagnostic only | `349.2921 ms`, markers `[914, 1, 1, 16, 128, 256, 8, 8, ...]` |
| 4 | `aic-tscm-index-diagnostic`, Qwen down | Passed marker diagnostic only | `365.6911 ms`, markers `[913, 1, 1, 16, 128, 256, 8, 8, ...]` |
| 5 | `aic-staged-gm-visibility-diagnostic`, Qwen down | Rejected | `507015`; D-cache-to-UB bus response error |
| 6 | `aic-tscm-ping-diagnostic`, Qwen down | Rejected | timeout at 60 s |
| 7 | `aic-tscm-zero-b-diagnostic`, Qwen down | Rejected | `507015`; MPU invalid access |

The actionable result is not a production speed win yet. It is a design
constraint: the next real fused attempt should emulate SVDQuant's low-level
Cube plus vector lifecycle and ring ownership, not keep pushing high-level
Matmul/TSCM or ACLNN user-workspace handoffs. The currently passing marker
diagnostics prove mixed launch registration and tiling metadata can work, but
they do not validate data movement or matmul correctness.

## Sources

- Local 910B notes: `hw/ascend_910b.md`
- Torch-NPU and CANN 9 API scan: `hw/torch_npu_cann_9_api_scan.md`
- arXiv 2601.16536, "W4A16 Mixed-Precision Matrix Multiplication on Decoupled
  Architecture": https://arxiv.org/abs/2601.16536
- SVDQuant kernels for Ascend/NVIDIA: https://github.com/Qubitium/svdquant-kernels
