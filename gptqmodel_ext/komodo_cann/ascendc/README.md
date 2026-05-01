# Komodo-CANN Ascend C Kernel

This directory contains the repository-owned Ascend C implementation for the
optional Komodo-CANN W4A16 custom operator. It is intentionally separate from
plain Komodo and from the `aclnnWeightQuantBatchMatmulV3` probe.

The current device kernel is a bring-up baseline for GPTQ W4A16:

- Input activations are FP16 `[M, K]`.
- Packed weights use Komodo's CANN INT4 pack, physically `int32 [K, N / 8]`.
- Scales and offsets are FP16 `[groups, N]`.
- Output is FP16 `[M, N]`.
- Supported group sizes are `0`, `32`, `64`, and `128`.

The kernel deliberately does not materialize a full dense FP16 weight matrix in
GM or L2. The current validated baseline uses up to eight logical AIV owners
over disjoint packed output-column ranges. Each owner reuses one packed INT4
word across its eight output lanes while the matching FP16 activation value
stays live, and it writes only the final output columns. This is not yet the
final high-throughput Cube-tiled design; it is the custom-op baseline needed
before replacing the scalar accumulation loop with Cube tile consumption.

The host tiler caps scalar launch ownership at `blockDim <= 8`. Earlier
multi-block experiments exposed sparse/non-contiguous physical AIV block IDs on
the local 910B runtime, so the device kernel maps `GetBlockIdx()` through
`physical_id % tiling.block_dim` and then owns a contiguous packed-column range.
Wider logical ownership such as 32 chunks left unwritten columns on this host.
If a future generated package enters a mixed AIC/AIV task layout, the scalar
fallback first normalizes AIV block IDs by `GetTaskRation()` and returns
immediately on AIC entry. The checked-in package remains AIV-only; this guard is
only preparation for the staged vector/Cube kernel.

The fused-kernel bring-up path now has an explicit staged-dequant mode. It is
off by default. Python enables it only with
`GPTQMODEL_KOMODO_CANN_STAGED_DEQUANT=1`, and the fused-call ABI encodes that
request with a negative `base_n` attribute so existing generated packages keep
their behavior. The host tiler then requests workspace only when the planned
ping-pong FP16 tile storage is strictly smaller than a dense `K x N` dequantized
weight matrix. The current default scalar package therefore still reports zero
custom-op workspace; an experimental package can be built with:

```bash
python scripts/build_komodo_cann_ascendc.py \
  --output /tmp/komodo_cann_w4a16_staged_op \
  --experimental-staged-dequant
```

This is a bounded tile-staging hook for the eventual AIV producer / AIC Cube
consumer state machine. It is not a default cache and it never requests a full
dense dequantized-weight buffer.

In an experimental build, the staged path now runs the AIV producer loop: each
logical staging owner unpacks its assigned GPTQ INT4 `(baseK, baseN)` tiles into
its ping-pong FP16 workspace slots. The scalar path still writes the visible
output while the AIC consumer is under construction, so this is a bring-up probe
rather than the final fused kernel.

CANN 9 public vector dequant can be compiled into that staged producer with:

```bash
python scripts/build_komodo_cann_ascendc.py \
  --output /tmp/komodo_cann_w4a16_cann9_vector \
  --experimental-staged-dequant \
  --experimental-cann9-vector-dequant
```

This path includes `asc/include/c_api/asc_simd.h` and uses
`asc_int42half_sync` in UB to convert one packed INT4 word into eight FP16 lanes
before applying Komodo scales and offsets into the bounded staging tile. It is
guarded because CANN 8 does not expose the public `asc/include/c_api` tree. The
2026-05-01 CANN 9.0.0-beta.2 rebuild and runtime checks validated that this path
compiles, launches, and preserves the scalar visible-output accuracy envelope
after fixing scalar lane-0 signed nibble decode. A controlled
`M=8,K=1024,N=1024,group_size=32` finite-input check produced finite output with
`max_abs=7.62939453125e-06`; the 8-NPU `gptq_group_sizes` staged sweep stayed at
`max_abs=0.015625` for group sizes 32/64/128/full and act-order 32/128.

The CANN 9 Matmul consumer type probe can also switch B to `TPosition::VECOUT`:

```bash
python scripts/build_komodo_cann_ascendc.py \
  --output /tmp/komodo_cann_w4a16_vecout_probe \
  --experimental-staged-dequant \
  --experimental-cann9-vector-dequant \
  --experimental-vecout-consumer
```

This validates the public Matmul template surface and instantiates
`SetTensorB(LocalTensor<half>)` for a UB/VECOUT B operand, but it is still a
compile-time consumer probe. The CANN 9 Matmul client maps non-TSCM local B
operands through Matmul workspace, so VECOUT is not the zero-GM/L2 handoff path.

The current local-B handoff target is TSCM/NZ:

```bash
python scripts/build_komodo_cann_ascendc.py \
  --output /tmp/komodo_cann_w4a16_tscm_probe \
  --experimental-staged-dequant \
  --experimental-cann9-vector-dequant \
  --experimental-tscm-consumer
```

This build switches the Matmul B operand to
`MatmulType<TPosition::TSCM, CubeFormat::NZ, half>` and compiles a staged
GM-to-TSCM tile load plus a `SetTensorB(LocalTensor<half>)` probe.
It is still guarded because the scheduler must next connect the AIV producer,
TSCM/NZ tile lifetime, AIC Cube `IterateAll`, and output ownership without
falling back to a full dequantized FP16 weight materialization.

There is also a narrower runtime handoff probe:

```bash
python scripts/build_komodo_cann_ascendc.py \
  --output /tmp/komodo_cann_w4a16_tscm_runtime \
  --experimental-staged-dequant \
  --experimental-cann9-vector-dequant \
  --experimental-tscm-runtime-handoff
```

This implies mixed launch and TSCM consumer. It only handles the deliberately
safe scheduler subset where the request has full `base_n` output tiles and the K
dimension is covered by full `base_k` tiles; unsupported shapes fall back to the
scalar visible-output path. Fused bias is handled in Cube with CANN's split-K
pattern: `SetBias` on the first K tile and `ClearBias` on accumulating K tiles.
The purpose is to validate the live AIV staged tile -> TSCM/NZ -> AIC Matmul
handoff before broadening it to deeper overlap and direct dequant into TSCM.

Local validation on 2026-05-01 built this path with CANN 9.0.0-beta.2, installed
it into `/tmp/komodo_cann_tscm_runtime_install`, and ran
`M=8,K=64,N=8192,group_size=32` on NPU0. The output was finite with
`max_abs=0.0` versus a CPU reference. Timing for that narrow shape was flat
against the non-runtime staged probe (`3.646628 ms` versus `3.651168 ms`), so
this is a correctness/scheduler proof, not a speed win yet.

The direct-dequant variant removes that staged FP16 GM tile from the same narrow
handoff:

```bash
python scripts/build_komodo_cann_ascendc.py \
  --output /tmp/komodo_cann_w4a16_tscm_direct \
  --experimental-tscm-direct-dequant
```

This implies staged-dequant metadata, CANN 9 vector dequant, mixed launch, and
the TSCM runtime handoff. It fills a bounded UB B tile with `asc_int42half_sync`
plus Komodo scale/offset, then uses `DataCopy(LocalTensor TSCM, LocalTensor UB,
Nd2NzParams)` to hand the NZ tile to Cube without writing the FP16 tile through
GM/L2. Local validation on the same `M=8,K=64,N=8192,group_size=32` shape
matched the CPU reference with finite output and `max_abs=0.0`; timing was still
flat at `3.655767 ms`, so the next speed-relevant step is multi-K direct
handoff and overlap rather than more single-tile tuning.

The guarded multi-K direct handoff probe broadens that path to shapes where
`K` is an integer multiple of `base_k`:

```bash
python scripts/build_komodo_cann_ascendc.py \
  --output /tmp/komodo_cann_w4a16_tscm_direct_multik \
  --experimental-tscm-direct-multik
```

This implies direct dequant and iterates `base_k` TSCM/NZ B tiles while asking
Cube Matmul to accumulate later K tiles into the existing output tile. Local
validation on NPU0 with `M=8,K=128,N=8192,group_size=32,base_k=64` matched the
CPU reference with `max_abs=0.0` and finite output. Timing was effectively tied
with a fresh current-source scalar package for the same shape (`7.132362 ms`
versus `7.130773 ms`), so this is a correctness and scheduling milestone, not a
speed path until producer/consumer overlap is added.

The next guarded scheduling pass keeps the small two-tile `base_k=64` case on
that sequential path, but uses two TSCM B slots plus hoisted scale/offset loads
when `base_k >= 128` or the shape has at least four K tiles. The second TSCM
slot lets the AIV side stage tile `i + 1` after launching Cube on tile `i`,
then waits before issuing the dependent accumulation. The scale/offset hoist
loads GPTQ group parameters once per group/output pack while filling the direct
UB tile. Local NPU0 checks stayed finite: symmetric `M=8,K=512,N=8192` with
`base_k=128` matched the CPU reference with `max_abs=0.0`, and a nonzero-offset
`M=8,K=128,N=8192,base_k=128` probe had `max_abs=0.00390625` and mean drift
`5.9e-7`. Timing remains close to noise (`K=512,base_k=128` sampled around
`27.89-27.94 ms`), so this is still a structural overlap step rather than the
final throughput target.

The direct TSCM runtime path now supports fused FP16 bias. It uses CANN's
standard split-K Matmul pattern: `SetBias` is emitted before the first K tile
for each output tile, and `ClearBias` is emitted before later accumulating K
tiles. Local NPU0 validation on 2026-05-01 for
`M=8,K=128,N=8192,group_size=32,base_k=128` stayed finite with nonzero offsets
and bias (`max_abs=0.00012207`, `mean_abs=0.00000381`, `mean_ms=7.897985`).
The paired no-bias nonzero-offset probe stayed finite with `max_abs=0.00003052`
and `mean_ms=7.899011`. A zero-offset group-32 bias probe matched within
`max_abs=0.00006104`.

Python-side staged Cube-consumer plans now expose the sampled larger tile by
default: when `K` is divisible by 128, `base_k` becomes 128 instead of 64. The
original C0-sized tile remains available with `GPTQMODEL_KOMODO_CANN_BASE_K=64`,
and any override must be a positive multiple of 64 that divides `K`.
A plan-shaped local-A sweep validated this broader default on all eight NPUs
with `base_k=-128`, rows `8/17/32/48/64/96/129/160`, per-case `base_m` values
matching the Python planner, K `512/1024`, N `256/512`, and groups
`32/64/128`; worst drift was `max_abs=0.015625` and
`mean_abs=0.00142669677734375`.

The next guarded bring-up layer is the Cube consumer scaffold:

```bash
python scripts/build_komodo_cann_ascendc.py \
  --output /tmp/komodo_cann_w4a16_cube_probe \
  --experimental-staged-dequant \
  --experimental-cube-consumer
```

This compiles Ascend C `Matmul` registration with a device-local `TCubeTiling`
constructed from Komodo-CANN's primitive tiling fields. Keep that layout: adding
`TCubeTiling` directly as nested generated tiling data collides with CANN's
kernel-side `TCubeTiling` alias. The scaffold is not the runtime default and
does not yet consume staged INT4-dequant tiles through Cube.

Runtime Cube bring-up uses a second opt-in gate,
`GPTQMODEL_KOMODO_CANN_CUBE_CONSUMER=1`. When it is combined with staged
dequant, Python passes a negative `split_k` attribute and the host tiler
records CANN's 16 MiB Matmul/KFC system-reserved workspace separately from the
user FP16 staged tiles. The kernel calls `GetUserWorkspace(workspace)` and the
AIV producer writes staged tiles at user-workspace offset `0`, avoiding
collision with CANN's message queues and UB map. The user tile ring remains
bounded by the same "smaller than dense dequant" check. The non-mixed
staged+Cube path is the validated runtime baseline; mixed AIC/AIV launch remains
experimental and is kept behind `--experimental-mixed-launch`. Mixed Cube
registration must call `clearWorkspace(workspace)` before `REGIST_MATMUL_OBJ` so
the vector-side KFC client sees `WORKSPACE_SYNC_ID`.

To isolate MIX launch from CANN Matmul/KFC registration, build:

```bash
python scripts/build_komodo_cann_ascendc.py \
  --output /tmp/komodo_cann_w4a16_mixed_aiv_baseline \
  --experimental-staged-dequant \
  --experimental-mixed-aiv-baseline
```

This keeps the visible output on the AIV scalar path and returns immediately on
AIC. It is not the target fused kernel; it is the validated scheduler baseline
for later AIC/Cube handoff work.

The full mixed Cube-consumer build now clears and signals the KFC workspace, then
registers the Matmul object on both AIC and AIV. It still keeps visible output on
the AIV scalar baseline; the next target is consuming staged dequant tiles through
Cube instead of only registering the consumer.

The first runtime local-B Cube handoff that avoids a staged FP16 GM weight tile
is the guarded VecOut path:

```bash
python scripts/build_komodo_cann_ascendc.py \
  --output /tmp/komodo_cann_w4a16_vecout_runtime \
  --experimental-vecout-runtime-handoff
```

This flag implies staged dequant metadata, CANN 9 vector-dequant headers,
`TPosition::VECOUT` B, and mixed AIC/AIV launch. The host selects the generated
mixed-launch tiling key, requests CANN Matmul system workspace plus the bounded
user workspace, and the kernel fills one local FP16 B tile from packed INT4
before passing that local tile to `Matmul::SetTensorB`. It therefore does not
write a full dequantized FP16 weight matrix through GM/L2. The path currently
uses scalar direct INT4 decode for the live VecOut tile because the first direct
`asc_int42half_sync` tile fill produced incorrect lane values; the CANN 9 vector
conversion remains compiled for staged producer probes and needs a separate
packing/count/order fix before it can be used in the live fused handoff.

Local validation on 2026-05-01 installed the VecOut runtime package and passed an
8-NPU one-shape-per-device smoke across group sizes 32/64/128 and rows 1/4/8.
Observed drift versus native CANN stayed within `max_abs=0.015625` and
`mean_abs<=0.002598`. Two scheduler details are important: `IterateAll` must be
called with `waitIterateAll=true` before `WaitIterateAll`, and explicit
multi-row `SetOrgShape` on this mixed VecOut path caused device error `507057`.
The current safe implementation therefore launches Cube one row at a time for
`M>1`; batching rows again is the next performance target after fixing the
direct vector dequant lane mapping.

The next compile-guarded batching probe stages A locally as well:

```bash
python scripts/build_komodo_cann_ascendc.py \
  --output /tmp/komodo_cann_w4a16_vecout_local_a \
  --experimental-vecout-local-a
```

This implies the VecOut runtime handoff and changes the Matmul A operand to
`TPosition::VECOUT`. Each active AIV owner copies the current `M x baseK`
activation tile into a contiguous UB tile, then reuses it across the output-N
tiles owned by that core. The goal is to restore multi-row Cube calls without
the failing GM-stride `SetOrgShape` path. It is intentionally separate from the
validated row-wise VecOut handoff until runtime checks prove that local-A
batching is correct and faster for `M>1`.

First local-A runtime smoke on 2026-05-01 installed the generated custom OPP and
completed a single-shot NPU0 correctness check for
`M=4,K=512,N=256,group_size=64` with `base_m=16,base_n=-256,base_k=-128`.
Drift versus native CANN was `max_abs=0.0078125` and
`mean_abs=0.0014190673828125`. This proves the local-A handoff can execute and
match native for one multi-row case, but it is still not promoted to the
validated path until a broader shape sweep and timing comparison pass.

The broader follow-up kept one call per process to avoid the earlier benchmark
hang pattern. A single-device sweep passed five shapes covering rows `1/4/8`,
group sizes `32/64/128`, K `384/512/1024`, and N `256/512`. An all-8-NPU sweep
then passed one shape per NPU, including rows `1/2/3/4/7/8`, group sizes
`32/64/96/128`, K `384/512/768/896/1024`, and N `256/512`, with worst drift
`max_abs=0.015625` and `mean_abs=0.0024566650390625`. The first A-tile
optimization attempt replaced scalar A staging with GM-to-VECOUT `DataCopy`;
that package compiled, but the runtime smoke timed out before the first result,
so scalar A staging remains the validated local-A implementation.
A later VecOut-only zero-offset B-fill specialization also compiled and passed
the first three NPU0 shapes, then timed out on the fourth shape
(`M=4,K=512,N=256,group_size=128`). Keep the generic offset-aware B fill for
now; the runtime appears sensitive to this code shape even when the math is
equivalent.

The reusable raw-op harness for this style of package validation is:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python scripts/validate_komodo_cann_ascendc_raw.py \
  --bridge-lib /tmp/gptqmodel_komodo_cann_ascendc_vecout_row/dae610bed1a54586/gptqmodel_komodo_cann_ascendc_ops.so \
  --opp-install /tmp/komodo_cann_vecout_local_a_install \
  --devices 0,1,2,3,4,5,6,7
```

It launches one process per listed NPU, applies the custom OPP env, runs one
custom/native comparison per worker, and emits a JSON summary. The first harness
run reproduced the manual 8-NPU sweep with all eight cases passing and the same
worst drift: `max_abs=0.015625`, `mean_abs=0.0024566650390625`.
Pass `--warmup` and `--iters` to turn the same correctness sweep into a compact
A/B timing probe; each worker reports `custom_ms`, and the parent summary records
the min/mean/max custom-op latency plus the timing parameters. Use at least one
warmup for steady-state timing; `warmup=0,iters=1` includes first-call setup.
The same harness also passed all eight cases with positive `base_k=128`, which
covers the offset-aware call shape used when the module does not encode the
zero-offset side band.
The harness parser now scans worker stdout for the last JSON object instead of
assuming the result is on a clean final line; CANN 9 can splice tiling warnings
around the JSON result on stdout.
The scalar visible-output path now receives the same zero-offset side band for
M1/M2/M4 tails as the existing M8 path. A non-mixed staged/vector package built
from this source passed NPU0 scalar checks at the validated tile width
(`base_n=256`) for positive `base_k=128` and negative `base_k=-128`; the latter
covered rows `1/2/4/8` with worst drift `max_abs=0.015625` and
`mean_abs=0.0018758773803710938`.

The local-A VecOut runtime path now avoids refilling the large direct B tile for
every M tile when `rows > base_m`. It keeps the previous A-outer loop for
single-M-tile decode, but for larger row batches fills each `(K tile, N tile)` B
tile once and reuses it across all M tiles, paying only the much smaller local-A
copy per M tile. A CANN 9 local-A package passed an all-8-NPU raw sweep covering
rows `1/8/16/17/24/31/32/48`, K `384/512/768/896/1024`, N `256/512`, and group
sizes `32/64/96/128` with worst drift `max_abs=0.015625` and
`mean_abs=0.001491546630859375`. The pre-change local-A package timed out at
120 s on the `rows=17,K=512,N=512,group_size=64` timing probe; the new package
completed the same probe in `3.060030 ms` with `max_abs=0.0078125`.
The local-A activation fill now switches from scalar `GetValue`/`SetValue` to
row-wise GM-to-VECOUT `DataCopy` only for `rows >= 48`. Unconditional `DataCopy`
regressed tiny decode tiles, but the shape gate improved the planner-shaped
all-8-NPU warmed timing sweep from `4.369839 ms` mean to `4.109257 ms` mean while
keeping worst drift at `max_abs=0.015625`.

Validated raw-op timing on NPU0 for `M=8,K=256,N=256,group_size=32,bias=True`
improved from `63.67 ms` on the initial UB dequant-tile baseline to `10.33 ms`
with the 8-lane packed-word loop, then to `9.22 ms` after hoisting scale/offset
loads to quant-group scope, and to `5.56 ms` after adding a two-row micro-tile
that reuses each dequantized packed word across two activation rows. Branchless
INT4 sign extension then reduced the same raw-op timing to `4.60 ms`, and a
four-row micro-tile reduced it further to `3.66 ms`. The current eight-row
micro-tile reaches `3.42 ms` for the same shape.

For single-row decode, the scalar baseline also has a two-packed-word micro-tile
that reuses each FP16 activation load across sixteen adjacent output channels.
On NPU0 this reduced `M=1,K=256,N=256,group_size=32` from `1.03 ms` to
`0.72 ms`, and `M=1,K=1024,N=1024,group_size=32` from `16.73 ms` to `11.42 ms`.
The two-row tail now has the same two-packed-word reuse. In a rechecked NPU0
median run it reduced `M=2,K=256,N=256,group_size=32` from `1.239 ms` to
`1.191 ms`, and `M=2,K=1024,N=1024,group_size=32` from `18.734 ms` to
`17.749 ms` while keeping `M=1` effectively unchanged.
The four-row path also reuses each FP16 activation load across two adjacent
packed weight words. Compared with the two-row commit, NPU0 medians improved
from `53.206 ms` to `51.678 ms` for `M=7,K=1024,N=1024,group_size=64` and from
`36.833 ms` to `36.688 ms` for `M=5,K=1024,N=1024,group_size=32`; the small
`M=4,K=256,N=256` median was effectively flat (`1.665 ms` to `1.678 ms`) while
the 8-NPU smoke passed on all devices.
The torch bridge now passes a true null optional bias into ACLNN instead of
allocating a synthetic zero-bias tensor and synchronizing the stream to preserve
that temporary. No-bias NPU0 medians improved from `0.827 ms` to `0.802 ms` for
`M=1,K=256,N=256,group_size=32`, from `1.118 ms` to `1.086 ms` for `M=2`, and
from `3.248 ms` to `3.213 ms` for `M=8`; bias-present timings stayed within
noise.
For symmetric GPTQ cases with at least eight rows, the planner marks the
eight-row path as zero-offset and the host tiler decodes that internal flag from
a negative `base_k` attribute while preserving the absolute tile size. This lets
the row-oct loop skip eight GM offset loads per group. NPU0 medians improved
from `3.214 ms` to `2.964 ms` for `M=8,K=256,N=256,group_size=32`, from
`6.353 ms` to `5.833 ms` for `M=16,K=256,N=256,group_size=32`, and from
`50.033 ms` to `45.781 ms` for `M=8,K=1024,N=1024,group_size=32`; M1/M2/M4
nonzero-offset paths stayed within timing noise.
For the fused Ascend C path, the planner now marks every symmetric GPTQ call as
zero-offset, not only `M>=8`. That broader planner side band is validated for
the local-A VecOut package by the raw 8-NPU sweeps above; it uses the generic
offset-aware B fill with `OffsetValue(..., zero_offsets)` rather than the
rejected zero-offset-only specialization.
With capped 8-owner AIV packed-column parallelism, the same no-dense baseline
improved by about 85-87% in an 8-NPU raw-op A/B sweep: `M=1,K=256,N=256`
from `0.713 ms` to `0.099 ms`, `M=8,K=256,N=256` from `2.895 ms` to
`0.376 ms`, symmetric `M=16,K=256,N=256` from `5.750 ms` to `0.743 ms`, and
`M=8,K=1024,N=1024` from `46.037 ms` to `5.887 ms`. Max observed error in that
sweep stayed below `0.0005`.
The M1 path now hoists the constant offset contribution out of each quant-group
K loop and applies it from a per-group activation sum. In an 8-NPU raw-op A/B
sweep this reduced `M=1,K=256,N=256,group_size=32` from `0.101 ms` to
`0.093 ms`, `M=1,K=1024,N=1024,group_size=32` from `1.493 ms` to `1.355 ms`,
and `M=1,K=1024,N=1024,group_size=64` from `1.457 ms` to `1.312 ms`.
The large down-proj check `M=1,K=17408,N=5120,group_size=32` improved from
`127.73 ms` to `116.32 ms`; max drift was `0.0625` with mean drift about
`1.0e-4`.
The same offset-hoist pattern is now applied to the two-row path. Against the
M1-hoist baseline, an 8-NPU raw-op A/B sweep improved
`M=2,K=256,N=256,group_size=32` from `0.155 ms` to `0.121 ms`,
`M=2,K=1024,N=1024,group_size=32` from `2.354 ms` to `1.809 ms`, and
`M=2,K=1024,N=1024,group_size=64` from `2.318 ms` to `1.764 ms`. Mixed tails
also improved: `M=3,K=256,N=256` from `0.242 ms` to `0.207 ms` and
`M=6,K=256,N=256` from `0.411 ms` to `0.381 ms`. Max observed drift in this
sweep was `0.00390625`; symmetric zero-offset M2 stayed exact and M1 stayed
flat.
The eight-row path now applies the same offset-hoist idea only for nonzero GPTQ
offsets, leaving the existing zero-offset fast path and smaller-row paths
untouched. In an 8-NPU one-shard-per-device `gptq_group_sizes` fused-op A/B
sweep, group-size 32/64/128/full improved from `6.8177/6.7423/6.7174/6.6647 ms`
to `6.1858/6.0609/5.9748/5.9162 ms`; act-order group-size 32/128 improved from
`6.8434/6.7026 ms` to `6.1966/5.9862 ms`. Max abs drift stayed unchanged at
`0.015625`.

Build from the repo root:

```bash
python scripts/build_komodo_cann_ascendc.py \
  --output /tmp/komodo_cann_w4a16_op
```

Use `--no-build` to generate and overlay the project without invoking CMake.
The helper calls `msopgen` with
`gptqmodel_ext/komodo_cann/op_ir/komodo_cann_w4a16_matmul.json`, overlays the
files in this directory, and then runs the generated `build.sh`.
