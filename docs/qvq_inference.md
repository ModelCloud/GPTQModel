# QVQ V4 inference kernel design for CUDA SM80

Status: native V4 CUDA inference paths exist behind capability/shape dispatch; the Torch reconstruction remains the
independent oracle and fallback. Full-model quality and reload gates remain required before default promotion.

This document defines the native Ampere `sm_80` inference plan for L16/V4 at W1--W4. It complements
[qvq.md](qvq.md), [qvq_quant.md](qvq_quant.md), and the measured V2 CUDA history in [qvq_cuda.md](qvq_cuda.md).
The target is lower latency than the production V2 kernel with identical packed BPW and exact reconstruction.
Apple MPS and MLX kernels are documented separately in [qvq_inference_mlx_mps.md](qvq_inference_mlx_mps.md).

## Work and storage model

Both codecs store exactly `8R` int32 words per 16 by 16 tile, or `R` raw bits per weight:

```text
V2: 128 transitions * 2R bits = 256R bits/tile
V4:  64 transitions * 4R bits = 256R bits/tile
```

Per four decoded weights, the implicit work is:

```text
V2: 2 state reconstructions + 2 PGC16 mixes + 4 level lookups + 4 MACs
V4: 1 state reconstruction  + 2 PGC16 mixes + 4 level lookups + 4 MACs
```

V4 therefore saves one state reconstruction while retaining the same payload traffic, mixer count, lookup count,
and matrix multiply work. It can beat V2 only if the kernel consumes the four-value state as a unit; calling the V2
decoder twice would discard the advantage.

## Kernel architecture

### 1. Fuse one V4 state directly into four MACs

Compile edge-width specializations for `E in {4, 6, 8, 10, 12, 14, 16}`. A decoded state produces:

```text
p0 = pgc16_mix(state)
p1 = pgc16_mix(state ^ 0xA5A5)
w  = {G[p0 >> 8], G[p0 & 255], G[p1 >> 8], G[p1 & 255]}.
```

Interleave the two independent integer IMAD/xor chains to hide dependency latency, then issue four FP32-accumulating
MACs. Keep the values in registers; never write a dense weight or decoded tile to global memory. Map each V4 state to
four consecutive K coordinates so input loads and accumulator ownership remain coalesced.

The canonical scalar table remains FP16 even for BF16 input. Converting it to BF16 changes the format-defined values.
The FP16 path may use FP16 Tensor Core operands for prefill, but the BF16 input path must preserve the FP16 levels and
use the proven FP32 scalar-FMA route unless a mixed-operand kernel is demonstrated bit-compatible.

### 2. Decode planar edges once per warp slab

The planar payload for 64 transitions naturally divides into two 32-transition slabs. Each slab contains exactly `E`
int32 words holding the format's aligned low-to-high 8-, 4-, 2-, and 1-bit subplanes. Cooperatively stage those words,
then let warp lane `t` extract field `t` from every applicable subplane and combine the fields into one edge. Rebuild
neighboring 16-bit states with warp shuffles of the edge codes instead of rereading the subplane words for every
scalar weight.

The state requires `ceil(16/E)` recent edges: four at W1, three at W1.5, two at W2--W3.5, and one at W4. Specializing
this fixed window removes the generic loop and modulo arithmetic. The circular wrap state for the first lanes should
come from the second slab via shuffle/shared handoff, preserving exact tail-biting semantics.

Use shared staging only for the packed tile/slab, not reconstructed states. Benchmark register pressure carefully:
excessive live edge/state vectors can reduce occupancy or spill to local memory and erase the saved decode work.

### 3. Provide two exact decoder modes

The default implicit decoder uses only the canonical 512-byte FP16 scalar table and the two PGC16 mixes. Also test an
optional process/device-wide expanded V4 decode table:

```text
65,536 states * 4 values * 2 bytes = 512 KiB/device.
```

Store each entry as two aligned `half2` values or one aligned 64-bit vector. It converts two integer mixer chains and
four scattered level lookups into one 8-byte vector load. On A100-class devices it is small relative to L2, but cache
residency must be measured under complete-model traffic rather than assumed.

The expanded table is derived exactly from frozen FP16 bit patterns during `post_init`, shared across all V4 modules
on a device, and never serialized. It adds 512 KiB per device, not per module, and does not alter raw or effective
checkpoint BPW. Validate every one of its 65,536 entries bit-for-bit against the implicit decoder. Dispatch between
implicit and expanded modes by measured `(M, K, N, rate)` latency; do not make the cache an ABI requirement.

For the 512-byte scalar table, compare three placements: the current per-CTA shared copy, read-only/L1 loads, and a
replicated/padded shared layout. SM80 has 32 shared-memory banks, so four data-dependent indices can conflict.
Constant memory is not automatically faster because divergent warp addresses serialize; it must earn selection in a
profile.

### 4. Optimize decode/GEMV for M=1--8

Small-M inference is primarily launch, payload, and decode bound. Start from the production row-specialized V2
kernel but change ownership to one four-weight state per lane/group:

- Stage multiple packed K tiles per barrier, as V2 already does, while respecting SM80 shared-memory occupancy.
- Load a V4 state once, decode four values, and reuse the associated activation values across output accumulators.
- Use deterministic warp reductions and retain FP32 accumulators through the range-safe output transform.
- Preserve split-K for narrow-N projections, but retune split count from runtime `K`, `N`, row count, device SM count,
  workspace cost, and measured launch overhead.
- Keep split-K partials FP32 and reduce them in a fixed order; do not introduce floating-point atomics.

For very small N, evaluate a persistent CTA that consumes several output tiles before exit. Enable it only if the
register footprint leaves enough active warps and it beats deterministic split-K on representative attention, MLP,
and MoE shapes.

### 5. Reuse one decoded tile for M >= 16

For prefill, decode the 16 by 16 weight tile once into a bank-safe shared-memory layout, then reuse it for multiple
activation rows. On FP16:

- use `cp.async` to double-buffer packed trellis words and activation tiles;
- store decoded weights directly in an `ldmatrix`/`mma.sync`-friendly swizzle;
- use SM80 FP16 Tensor Cores for the matrix multiply;
- overlap decode of tile `k+1` with MMA on tile `k` when register/shared-memory limits permit.

V4 halves state-reconstruction work, but Tensor Core compute dominates as M grows, so the expected gain over V2 is
smaller than in decode. Verify that dimensions meet Tensor Core alignment; odd/tail shapes need an explicit scalar
fallback and must not be counted as Tensor Core results.

### 6. Dispatch by shape and decoder mode

Use a small, deterministic dispatch table keyed by:

```text
(format=qvq_v4, sm capability, input dtype, rate, M bucket, K tiles, N tiles)
```

Candidate paths are implicit GEMV, expanded-table GEMV, deterministic split-K, FP16 Tensor Core prefill, and the
reference fallback. Populate thresholds from warmed CUDA-event benchmarks over square Llama shapes and real narrow
attention/MoE projections. Do not infer hardware from a fixed CUDA ordinal; query the active device.

Cache the selected policy, not module-specific decoded weights. CUDA Graph capture must observe stable workspace and
table addresses after `post_init`.

### 7. Defer cross-transform mega-fusion until profiling proves it

QVQ also applies `SU`, input Hadamard, output Hadamard, `SV`, and bias. The existing range-safe fused transforms are
correctness-critical. First land a fast native V4 inner kernel behind the same transform lifecycle. Only then profile
full module execution to determine whether launch overhead justifies a persistent or cooperative fusion.

Any fusion must preserve early Hadamard normalization, FP32 overflow rescue, historical finite FP16 rounding points,
CUDA Graph compatibility, and the non-SM80 fallback. A lower standalone GEMV latency that slows the complete
`QVQLinear` call is not a win.

## ABI and lifecycle contract

- `format="qvq_v4"` selects `vector_size=4`; it must never enter a V2 ABI.
- Pass the exact integer transition width `E=4R`, not a rounded floating-point bit rate.
- Validate trellis shape as `8R` int32 words per tile and exactly 64 transitions per tile.
- Prepare the scalar or optional expanded table during `QVQLinear.post_init`, before the first forward.
- Cache derived tables per `(device, codec_version, dtype/layout)`, not per module.
- Reject unsupported codebook versions, rates above W4, malformed payloads, and incompatible devices before launch.
- Preserve the Torch reconstruction path on CPU, non-SM80 CUDA, unsupported dtypes/shapes, and during training.

## Expected performance targets

These are design gates, not measured claims:

| Regime | First native target versus V2 | Stretch target | Expected limiter |
|:---|:---|:---|:---|
| M=1--2, square | `<= 0.85x` latency | `<= 0.70x` | Decode, payload, and launch overhead. |
| M=1--8, narrow N | `<= 0.85x` latency | `<= 0.65x` | Grid occupancy and split-K reduction. |
| M=8--16 | `<= 0.90x` latency | `<= 0.75x` | Activation reuse and reduction work. |
| M=32+ FP16 | `<= 0.95x` latency | `<= 0.85x` | Tensor Core compute increasingly dominates. |

No V4 path may be enabled if it is more than 5% slower than V2 for the same shape/rate without an explicit user
override. Report complete `QVQLinear` latency as the primary result and inner-kernel latency as attribution.

## Correctness and benchmark gates

Before native V4 inference becomes loadable by default:

1. Match reference reconstruction for all 65,536 states and planar tile reconstruction at every half-step W1--W4.
2. Match decoded-weight FP32 matmul across FP16/BF16 inputs, M=1/2/4/8/16/32+, square and real attention/MLP/MoE
   shapes, structured payloads, accumulation-heavy K, and tail dimensions supported by the ABI.
3. Report MAE, RMSE, maximum absolute error, relative L2, final KLD/JSD, top-1, ordered top-5, and exact shape/dtype.
4. Validate range-safe SU/Hadamard/GEMV/Hadamard/SV/bias execution, non-default streams, deterministic replay,
   concurrent devices, save/load, first-forward behavior, and CUDA Graph replay.
5. Verify no persistent dense weight exists and account for the optional 512 KiB derived table and all workspaces.
6. Benchmark with CUDA events after warmup. Record hardware properties, CUDA/PyTorch/NVCC/driver versions, build flags,
   generated `sm_80` code, JIT state, launch count, registers, shared memory, occupancy, and memory throughput.
7. Retain V2 and the Torch V4 decoder as independent oracles until native V4 passes unit, lifecycle, full-model,
   quality, and performance gates.

## Implementation order

1. Add an explicit V4 CUDA ABI and exact implicit small-M GEMV.
2. Add warp-slab planar reconstruction and rate specializations.
3. Add the optional 512 KiB expanded decoder and select it only where measured faster.
4. Retune deterministic split-K for narrow projections.
5. Add FP16 `cp.async`/Tensor Core prefill and BF16-correct fallback.
6. Profile complete `QVQLinear`; consider transform fusion only after the dominant remaining cost is proven.


## 2026-08-13 CUDA V4 implementation progress

- The Torch reference remains the accuracy oracle: `reconstruct_qvq_inner_weight(..., vector_size=4)` followed by dense `x @ weight`. Quantization payload/state generation is unchanged; quantization parity is therefore required to be exact (100%).
- Added an opt-in `gemv_v4` CUDA ABI for SM80+ and selected it from `QVQLinear` only when `vector_size=4`; the established V2 `gemv` ABI is unchanged.
- V4 packing uses `E=rate*4`, two planar words per E-bit plane, and 64 transitions per 16x16 tile. The device decoder applies the exact two PGC16 mixer chains (`state` and `state ^ 0xA5A5`) and canonical FP16 level table.
- Initial random-trellis parity on GPU 0 for W1/W2/W3/W4 produced max absolute error 0.0 and bitwise equality against the Torch reference. This validates reconstruction/inference math before performance work.
- Failed experiment: first compile attempt omitted the default `VectorSize=2` template argument for legacy WMMA call sites; compilation failed only, with no runtime result. Fixed before parity testing.
- No Nsight Compute measurements have been accepted yet. GPU 0 and GPU 1 were idle; busy GPUs were not used. Next profiling must compare V4 against the Torch reference and capture kernel metrics on an uncontended GPU.


### Audit update: V4 is not release-ready yet

- Corrected a V4 CUDA staging defect: the tile-load index divisor must be `2 * E`, not the V2 `4 * E`; both normal and split-K paths were affected. A multi-K-tile CUDA test (`K=64`) now passes exact FP16 output parity after this fix.
- The earlier one-tile parity result was insufficient and is explicitly superseded.
- Native V4 CUDA quantization is still **not implemented**: CUDA Viterbi dispatch intentionally remains restricted to `vector_size=2`, so V4 quantization uses the Torch reference. This preserves 100% quantization correctness but provides no CUDA quantization speedup yet.
- V4 prefill WMMA and state reuse remain unimplemented; no speed claim or Nsight result is accepted until quantization and the full V4 CUDA coverage matrix are complete.

## 2026-08-13 inference-only sweep update

### Forward

- The inference-only optimization is committed as `7c6fb5d0` on PR 244's `agent/qtip-integration-foundation` branch.
- The V4 scalar GEMV and split-K GEMV paths now decode one V4 state per four-column group and broadcast the state,
  mixer outputs, and four FP16 level values with warp shuffles. The V2 ABI and non-CUDA/reference paths are unchanged.
- No persistent dense weights or persistent dequantized weight cache was added. The only derived data used by the
  kernel remains the per-launch shared 256-entry FP16 level table and transient tile/workspace storage.
- The worktree was synchronized to the PR tip and pushed through GitHub authenticated Git. The remote head is
  `7c6fb5d0b91e024cf21045a1b92736579f9b27e7`.

### Sweep failure / pending validation

- The requested physical GPU set was `{4, 5, 6, 7}` under PCI ordering. GPUs 4, 5, and 7 were idle; GPU 6 was not
  available, with approximately 22 GiB resident. The allocator therefore did not grant the requested exclusive set,
  and no partial or unmanaged sweep was started.
- The local Python 3.14.6 environment is `/root/vm314-codex-one`; it contains PyTorch
  `2.14.0.dev20260804+cu132` and Triton `3.8.0+git10f6be36`. It is ready for the sweep once all requested GPUs are
  available.
- No performance result, Nsight result, or new accuracy claim is recorded from this attempt. The existing exact V4
  parity and multi-K parity notes above remain the only accepted correctness evidence.
- The existing `scripts/benchmark_qvq_cuda.py` benchmark currently exercises the V2 call signature and does not yet
  expose a V4-only sweep configuration. A V4 inference benchmark must be added or invoked through the `vector_size=4`
  wrapper before accepting speed claims.

### ExLlama v3 reference notes

The local ExLlama v3 implementation was consulted in `gptqmodel_ext/exllamav3/quant/exl3_gemm_inner.cuh` and
`gptqmodel_ext/exllamav3/quant/exl3_dq.cuh`. The useful patterns for the next V4 pass are:

- stage packed trellis payloads with `cp.async` into shared memory, then decode directly into Tensor Core fragments;
- specialize packed-bit extraction by rate and use funnel shifts/BFE instead of generic per-value bit loops;
- pipeline global loads, shared-memory staging, fragment loads, and MMA while keeping the decoded values transient;
- treat `su/sv` as separate transform operands around the quantized GEMM, preserving their FP16 lifecycle and avoiding
  a dense reconstructed weight;
- retain shape/rate-specific kernel maps and runtime SM-aware launch selection.

QVQ V4 differs in its PGC16 state-history decoder and four-value state mapping, so ExLlama v3's codebook/decode
routine is inspiration only. Its packed-bit extraction, asynchronous staging, fragment scheduling, and transform
ownership are the portions to benchmark against the current V4 GEMV path. Persistent decoded-weight caching remains
out of scope for QVQ.

### Rate coverage contract

The inference benchmark and CUDA tests cover the complete public rate ladder for V2: W1, W1.5, W2, W2.5, W3, W3.5,
W4, W4.5, W5, W5.5, W6, W6.5, W7, W7.5, and W8. V4 is intentionally covered at W1 through W4, including every
half-rate; rates above W4 are rejected for `vector_size=4` because the V4 format contract does not support them.
The benchmark defaults to all V2 rates, accepts `--vector-size 4` for the V4 subset, and gates maximum absolute
inference error at `2e-3` unless overridden for diagnostic runs.

### 2026-08-13 decode-pass update

After reviewing the QVQ CUDA history and the ExLlama v3 trellis kernels, the V4 decode path now performs both PGC16
mixer broadcasts as one packed 32-bit warp shuffle. It also removes the former V4 weight shuffle, whose source lane
was always the current lane and therefore added synchronization overhead without moving data. This preserves the
exact mixer bits, FP16 level lookup, and FP32 accumulation; it adds no persistent decoded-weight or activation cache.

The updated source compiled successfully with the Python 3.14.6 environment's ATen headers using
`nvcc -std=c++20 -O3 -arch=sm_80`. Runtime timing was not accepted from the first corrected sweep: an installed
editable package initially resolved the benchmark to `/root/repos/GPT-QModel-Ultra`, and the subsequent current-checkout
JIT build was blocked by stale concurrent compiler jobs. The benchmark launcher now forces the repository through
`PYTHONPATH`; a clean runtime sweep remains required before claiming the 4x target.

### 2026-08-13 W4 state fast path (validated)

The inference GEMV decoder now specializes the one-transition W4 state (`TransitionBits=16`) and replaces the
generic wrapped recurrence with a power-of-two edge mask. The non-W4 recurrence is unchanged, and the transformation
is algebraically identical for both vector layouts; no persistent dequantized data is introduced.

The optimized translation unit compiled successfully with the host Ampere `sm_80` target using Ninja `-j8`. The
V4 FP32-accumulator matrix passed W1 through W4, including M=1/4/16/32 and both FP16 and BF16 inputs, with the
`2e-3` maximum-absolute-error gate. This is a correctness progression; a matched A/B timing result is still
required before claiming a speedup.

Matched A/B timing at K=64, N=48, M=32, W2, 30 CUDA-event iterations per side:

| Path | Previous commit | Current worktree | Change | Accuracy gate |
| --- | ---: | ---: | ---: | --- |
| V2 FP16 output | 0.0809 ms | 0.0788 ms | 1.027x faster | max abs 3.90625e-3 FP16 rounding |
| V4 FP16 output | 0.0819 ms | 0.0799 ms | 1.025x faster | max abs 0 (reference path) |

The current-worktree measurements are uncommitted candidate results; they are not a 4x/10x claim. The full CLI
sweeps cover V2 W1 through W8 and V4 W1 through W4, with FP32-output validation below `2e-3`.

### 2026-08-13 V4 large-batch WMMA sweep (rejected)

The V4 inference dispatcher was evaluated with a Tensor Core candidate for FP16 `M >= 32`. It reused the existing transient
16x16 WMMA path, but decodes V4's two-word tile layout and maps each four-column group through both PGC16 mixer
chains. V2 dispatch remains unchanged, BF16 retains the scalar fallback, and no persistent dequantized weights are
introduced. The candidate was rejected: W1.5/M32 produced max absolute error `0.00390625`, exceeding the current
`2e-3` inference threshold. V4 therefore remains on the scalar path until a Tensor Core implementation meets the
selected accuracy contract across the full matrix. The V4 all-rate CUDA test matrix includes `M=32` for future
candidate validation.

The candidate compiled, but was rejected: at W1.5/M32 its maximum absolute error was `0.00390625`, above the
current `2e-3` inference gate. V4 therefore remains on the validated scalar path; no WMMA speedup is claimed.
