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
- The scalar baseline host tiler now pins `blockDim=1`. The earlier dynamic AIV
  block count was unsafe because CANN did not always launch a contiguous set
  including block index 0, which could leave stale output when the kernel used a
  single writer.
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
- This baseline intentionally avoids writing full dequantized FP16 weights
  through GM/L2. It is slower than the target design, but it creates the real
  custom-op registration, tiling, shape inference, optional bias handling, and
  packed-weight decode path needed before vector/Cube fusion.
- The initial multi-core strided writer was accepted by CANN but produced sparse
  output writes on the local 910B runtime. The committed correctness baseline
  therefore uses one writing AI Core while keeping the host tiling metadata for
  the next pass.
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
