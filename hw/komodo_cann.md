# Komodo-CANN Kernel Notes

Date: 2026-04-30

Komodo-CANN is a separate Ascend CANN kernel experiment. It does not change the
plain Komodo backend or its runtime path.

## Selection

- GPTQ backend: `BACKEND.GPTQ_KOMODO_CANN` / `gptq_komodo_cann`
- AWQ backend: `BACKEND.AWQ_KOMODO_CANN` / `awq_komodo_cann`
- Generic alias: `BACKEND.KOMODO_CANN` / `komodo_cann`
- Benchmark flag: `scripts/benchmark_komodo_npu_ab.py --komodo-cann`

The current implementation subclasses the plain Komodo packed int4 plan only as
a baseline plan format. Runtime dispatch is through `KomodoCannLinear` and
`AwqKomodoCannLinear`, not through `KomodoLinear` or `AwqKomodoLinear`.

## CANN Prefetch Policy

Komodo-CANN can issue CANN `torch.ops.npu.npu_prefetch` hints for eligible
activation, packed int4 weight, scale, offset, and fused-bias tensors before the
native quantized matmul. This is off by default because host-issued prefetch
probes regressed the steady-state microbenchmarks.

Environment controls:

```bash
GPTQMODEL_KOMODO_CANN_PREFETCH=1          # enable Komodo-CANN prefetch probe
GPTQMODEL_KOMODO_CANN_PREFETCH_MAX_BYTES  # override max bytes per tensor
GPTQMODEL_KOMODO_CANN_PREFETCH_MIN_BYTES  # default: 4MiB
```

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

Current Python-side Komodo-CANN records this plan per shape:

- `split_k=1` for balanced shapes and group-size/act-order sweeps.
- `split_k>1` for decode-like Qwen projections where `rows <= 16`, `K >= 4096`,
  and `K >= 2N`.
- `base_k=64`, matching the INT4 C0 alignment requirement.
- The plan records per-tile packed INT4 bytes, FP16 dequant workspace bytes,
  L0A/L0B/L0C tile bytes, K tiles per Split-K shard, and the number of vector
  dequant tasks. These are the checks the future Ascend C op must satisfy before
  it writes any dequantized FP16 tile to GM/L2.
- `strategy=planned_split_k_aiv_dequant_aic_matmul` means the shape is a
  candidate for the future Ascend C op. Runtime still uses the native quantized
  matmul fallback until that custom op lands.
- The plan is cached by hot shape/device so benchmark loops do not re-query
  hardware or environment variables every forward.

## Current Benchmark Read

The first steady-state prefetch implementation was slower than plain Komodo on
the tested synthetic shapes, including Qwen3.6 27B projection-sized cases. That
is why it now lives behind a separate Komodo-CANN backend instead of an option
inside plain Komodo.

Current 8-NPU A/B after the profiler-guided host-path patch on 2026-04-30
with tile `1024`, FP16, warmup `2`, iters `5`:

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

For Qwen3.6 27B GPTQ projection shapes, the recorded CANN plans used
`split_k` values `{1, 2, 8}` and vector dequant task counts `{320, 1920, 5440}`
across the six projection cases.

Current 8-NPU quick sweep:

- `quick_komodo_cann_keep_tile1024`: `0.3650ms`, no prefetch, `split_k=1`.
- `quick_komodo_cann_prefetch_keep_tile1024`: `0.3804ms`, prefetch enabled.
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

The next Komodo-CANN implementation should prioritize these public CANN 9 paths:

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
  probe. It handles the conservative one-full-K-tile/no-bias/full-N-tile subset
  by staging the B tile, copying it to TSCM/NZ, and invoking Cube Matmul for the
  visible output; unsupported shapes fall back to the scalar path. The broad
  target remains direct dequant into the TSCM tile with multi-K producer/consumer
  scheduling.
- Validated the runtime probe on NPU0 for `M=8,K=64,N=8192,group_size=32`:
  finite output, `max_abs=0.0` versus CPU reference. The quick A/B timing was
  effectively flat (`3.646628 ms` TSCM runtime versus `3.651168 ms` non-runtime
  staged probe), which confirms the handoff is structurally live but not yet a
  speed path while it still stages through GM.
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

The full rescan and public/private API notes are in
`hw/torch_npu_cann_9_api_scan.md`.

## aclnn V3 Probe

`scripts/probe_komodo_cann_v3.py` builds
`gptqmodel_ext/komodo_cann/wq_bmm_v3_probe.cpp`, registers
`torch.ops.gptqmodel_komodo_cann.w4a16_matmul`, and calls
`aclnnWeightQuantBatchMatmulV3` directly with Komodo's packed INT4 weights. The
probe exposes packed `int32 [K, N / 8]` storage as a logical `ACL_INT4 [K, N]`
tensor.

Runtime opt-in:

```bash
GPTQMODEL_KOMODO_CANN_V3=1
GPTQMODEL_KOMODO_CANN_FUSED_REQUIRE=1
GPTQMODEL_KOMODO_CANN_FUSED_OP=gptqmodel_komodo_cann.w4a16_matmul
```

The bridge caches repeatable ACL executors and CANN workspace tensors by
default. Disable these only for isolation:

```bash
GPTQMODEL_KOMODO_CANN_V3_EXECUTOR_CACHE=0
GPTQMODEL_KOMODO_CANN_V3_WORKSPACE_CACHE=0
```

Latest local checks:

- `M=8,K=256,N=256`, group sizes `0`, `32`, `64`, `128`: exact match against
  native `torch.ops.npu.npu_weight_quant_batchmatmul`.
- Same shape with FP16 bias: exact match.
- `M=1,K=4096,N=4096,group_size=128`: exact match in the final single probe.
- With the probe preloaded, `KomodoCannLinear` selected `fused_w4a16_matmul`
  and matched the torch baseline with `max_abs=0.0009765625`.

## Inner-Precise Shape Policy

`GPTQMODEL_KOMODO_CANN_INNER_PRECISE` accepts `auto`, `0`, or `1`. The default
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
| GPTQ group sizes + act-order | Komodo-CANN V3 bridge | 0 | 2.4404 | 0.03125 | none |
| GPTQ group sizes + act-order | Komodo | 1 | 1.6649 | 0.03125 | none |
| GPTQ group sizes + act-order | Komodo-CANN V3 bridge | 1 | 2.4540 | 0.03125 | none |
| Qwen3.6 27B GPTQ projections | Komodo | 0 | 1.2569 | 0.0625 | n/a |
| Qwen3.6 27B GPTQ projections | Komodo-CANN V3 bridge | 0 | 1.6005 | 0.0625 | q-proj |
| Qwen3.6 27B GPTQ projections | Komodo | 1 | 1.2814 | 0.0625 | n/a |
| Qwen3.6 27B GPTQ projections | Komodo-CANN V3 bridge | 1 | 1.7097 | 0.0625 | q-proj |

Earlier 8-NPU A/B with `GPTQMODEL_KOMODO_CANN_V3=1` passed 8/8. The V3 bridge
was correct but slower than the current native CANN path:

| Case set | Kernel | Source drop | Komodo total ms | Max abs drift |
|---|---|---:|---:|---:|
| GPTQ group sizes + act-order | Komodo-CANN V3 bridge | 0 | 2.5008 | 0.03125 |
| GPTQ group sizes + act-order | Komodo-CANN V3 bridge | 1 | 2.5408 | 0.03125 |
| Qwen3.6 27B GPTQ projections | Komodo-CANN V3 bridge | 0 | 1.6629 | 0.0625 |
| Qwen3.6 27B GPTQ projections | Komodo-CANN V3 bridge | 1 | 1.9257 | 0.0625 |

This confirms V3 API access, but it is still a raw CANN op call boundary. Even
with descriptor, executor, and workspace reuse, the remaining speed work is the
true Ascend C fused operator that avoids full dequantized FP16 weight
materialization through GM/L2 and removes the generic ACLNN call boundary.

## CANN Profiling Read

Use the profiling helper for single-shape CANN traces:

```bash
ASCEND_RT_VISIBLE_DEVICES=0 \
python scripts/profile_komodo_cann_npu.py \
  --mode cann \
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

## Sources

- Local 910B notes: `hw/ascend_910b.md`
- Torch-NPU and CANN 9 API scan: `hw/torch_npu_cann_9_api_scan.md`
- arXiv 2601.16536, "W4A16 Mixed-Precision Matrix Multiplication on Decoupled
  Architecture": https://arxiv.org/abs/2601.16536
