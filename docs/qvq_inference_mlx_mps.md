# QVQ V4 inference kernels for Apple MPS and MLX

Status: native and validated on Apple M4 Max. MPS is integrated into `QVQLinear`; MLX checkpoint loading replaces
Torch `QVQLinear` modules with `QVQMLXLinear` while preserving their planar payload and codec metadata. This
document covers the Apple-specific L16/V4 inference implementation for W1--W4. CUDA design and measurements remain
in [qvq_inference.md](qvq_inference.md).

## Format and reconstruction contract

The MPS and MLX kernels consume the same planar `qvq_v4` checkpoint payload as the Torch reference:

```text
state width:          16 bits
vector size:          4 weights/state
transition width:     E = 4R bits
transitions/tile:     64
words/16x16 tile:     2E = 8R int32 words
raw payload:          R bits/weight
supported V4 rates:   W1, W1.5, W2, W2.5, W3, W3.5, W4
```

The experimental `qvq_v4_l18` format is narrower: it supports W1--W2.5 and currently has a native MLX reference
kernel only. Its 18-bit state uses the high two history bits as an implicit rate-keyed bank selector and the low
16 bits as the canonical PGC state. It carries no `bank_ids` tensor and preserves the exact planar bytes per weight.
MPS deliberately falls back to Torch reconstruction until a dedicated L18 shader is implemented and validated.

`qvq_dual_v2` is also native in MLX. It interleaves two independent L16/V2 edge streams in the ordinary planar
payload and reconstructs each state from same-parity edges. It adds no tensors or bytes and retains V2's two mixes
and four scalar lookups per four weights. The current MPS path intentionally falls back to Torch for this topology.

One state reconstructs four values with two implicit PGC16 permutations and four canonical FP16 scalar-table
lookups:

```text
p0 = pgc16_mix(state)
p1 = pgc16_mix(state ^ 0xA5A5)
w  = {G[p0 >> 8], G[p0 & 255], G[p1 >> 8], G[p1 & 255]}
```

The kernels never materialize or cache a dense weight. The only persistent derived object is the process/backend
canonical 256-entry FP16 level table. MPS and MLX therefore retain the checkpoint's exact payload BPW and add only
512 bytes for the shared scalar table, not 512 bytes per module.

The opt-in four-bank V4 variant is also native on both Apple paths. Its packed
`bank_ids` tensor is consumed directly as four two-bit selectors per byte. The
kernel derives the logical 16 by 16 tile ID from `(k, n)`, loads its selector,
and applies the corresponding rate-keyed second PGC16 mixer mask. There is no
expanded selector tensor and no per-forward host unpack. Canonical single-bank
V4 retains its selector-free kernels. The only banked-format payload overhead
is two bits per 16 by 16 tile, or exactly 0.0078125 bits per weight.

MPS preserves the E4, generic, multi-row, K64, and K128 launch families for
banked decoding. `QVQLinear.post_init()` packs a dense research selector once;
ordinary loaded checkpoints already contain the canonical packed tensor. MLX
uses the same packed ABI and supports Metal's constant- and device-address-
space treatment of small and large selector buffers.

The derived MPS selector cache is keyed by the source tensor identity and its
PyTorch mutation version. Replacing or mutating `bank_ids` during quantization
or recovery therefore rebuilds the packed selector before the next Metal
launch; a concurrent free-threaded mutation fails closed instead of decoding
with stale bank assignments.

## Native Apple kernel paths

Both backends have two execution regimes.

### Independent-row kernel, `M < 4`

MLX now exploits an exact property of the planar V4 stream: the four V4 states for one K row and one 16-column
tile are consecutive states of the same recurrence. One 32-lane SIMD group reconstructs the first state from its
history, advances the next three states with one stored edge each, and produces all 16 columns. This replaces four
full state-history reconstructions with one without changing the state, weight, or FP32 accumulation order. For
short-K projections, a guarded N32 kernel processes two tiles per SIMD group; the old N4 form remains selected for
the small square buckets where its extra grid parallelism wins.

W1 uses a separate E4 kernel. E4 has exactly one packed 4-bit plane and exactly four predecessor edges, so its state
is reconstructed directly:

```text
edge_j = (word >> (4 * lane_in_word)) & 0xF
state  = (edge_0 << 12) | (edge_1 << 8) | (edge_2 << 4) | edge_3
```

This removes the generic plane-width loop, divisions, and dynamic edge-window calculation. It is isolated in a
separate compiled kernel so W1 specialization cannot perturb W1.5--W4 register allocation or code generation.

### Shared-decode multi-row kernel, `M >= 4`

MLX SIMD group 0 reconstructs 32 complete 16-column rows into four 32-entry threadgroup `float4` buffers. Each
remaining SIMD group owns one activation row and reuses those values, accumulating 16 independent FP32 outputs.
The 16-column chain therefore amortizes both trellis history and a bank-selector load across four consecutive V4
states. No reconstructed values survive the kernel launch.

For `M >= 8`, or `M >= 5` on wide-N projections, MLX instead maps each 8x8 output tile to an Apple SIMD-group matrix fragment. The lanes reconstruct
the exact V4 values directly into an FP16 B fragment, load FP16 activations into A, and use
`simdgroup_multiply_accumulate` with an FP32 C fragment. This keeps the public FP16-input/FP32-accumulator contract
while moving the dot-product reduction onto Apple matrix hardware. Accumulation order changes relative to the
scalar reference, so this path is gated by dense-reference MSE, relative L2, cosine, KLD, top-1, and ordered top-5
tests at both complete and partial 8-row tiles. W1 retains its direct E4 state reconstruction inside this path;
paired measurements improved MMA latency by another 1.08--1.26x on the tested decoder shapes.

The original fixed policy used a 16-row tile through M=16 and 32 rows afterward. M4 Max measurements showed large
occupancy and tail-utilization cliffs, especially at M=4, M=17, and wide N. MLX's chained decoder uses:

| M range | `N <= 2048` | `N > 2048` |
|:---|---:|---:|
| `M <= 4` | 4 rows; 8 only for W1 K=N=2048 | 4 rows |
| `5 <= M <= 8` | 8 rows | 8 rows |
| `9 <= M <= 16` | 8 rows | 8 rows |
| `17 <= M < 20` | 4 rows | 4 rows |
| `M >= 20` | 8 rows; 4 for W1 K,N>=8192 | 8 rows; 4 for W1 K,N>=8192 |

This table deliberately uses smaller tiles around partial-tile boundaries. A larger threadgroup amortizes decode
over more rows only when those rows exist; otherwise idle SIMD groups still participate in both threadgroup barriers
and consume scheduling resources.

The older MLX K64/K128 slabs were rejected for the chained decoder: paired measurements were 3--8% slower than its
K32 staging. MPS retains its independently measured K64/K128 policy.

## MPS implementation

`gptqmodel/utils/qvq_mps.py` compiles one Metal library with explicit E4/E6/E8/E10/E12/E14/E16 V4 entry points.
PyTorch tensors must be contiguous FP16 activations and contiguous int32 planar words on the same MPS device. Launch
geometry is derived from runtime M/K/N and the row policy above; no fixed device index or inferred GPU capability is
used.

The direct MPS API writes FP16 by default and retains FP32 products, accumulators, and SIMD reductions. Production
`QVQLinear` requests the native FP32-output entry point so a finite decoded inner product cannot overflow while being
rounded to FP16 before the normalized output Hadamard and SV epilogue. FP16 MPS Hadamards likewise normalize before
the butterfly at every width; delaying normalization can overflow a narrow transform even when its normalized result
is finite. The completed linear output is rounded to the model dtype only after Hadamard, SV, and bias. The E4 fast
path changes integer unpacking only; it does not change floating-point accumulation order.

## MLX implementation

`gptqmodel/utils/qvq_mlx.py` uses `mx.fast.metal_kernel` with the same reconstruction and row policy. `EdgeBits` is a
compile-time template argument. MLX dimension arrays are cached, and the canonical compander is prepared once and
reused. `ensure_row_contiguous=True` prevents a hidden strided-input slow path.

The standalone MLX API also supports a range-preserving FP32-output entry point for a future full QVQ epilogue. Its
default remains FP16 for direct inner-GEMV compatibility. Both Apple APIs reject non-integral output dimensions
instead of silently truncating them.

The production benchmark is `scripts/benchmark_qvq_v4_mlx.py`; it requests performance QoS, warms every compiled
shape/rate, evaluates each lazy MLX result, and reports median latency and raw-payload bandwidth. On the development
M4 Max, paired old/new kernels were bit-exact for independent rows. Representative medians improved by 2.47x for
W1 M1 2048x2048, 1.75--1.89x for M4 8192x2048, 1.29--1.34x for M24 8192x2048, and 1.30--1.36x for M32
8192x8192. The subsequent MMA path adds 1.25--2.16x over that chained decoder for tested `M >= 8` shapes. Some
decode/prefill buckets remain near parity, so this is not a blanket 10x result.

W1 has its own `gptqmodel_qvq_planar_v4_e4` Metal kernel. W1.5--W4 retain the generic V4 kernel, while all multi-row
rates use the shared-decode kernel. Separating E4 avoids a measured cross-rate compiler regression from placing an
E4 conditional inside the common kernel.

## M4 Max performance evidence

Measurements used warmed kernels, explicit `torch.mps.synchronize()` or `mx.eval()`, median wall latency, and the
same randomized planar payload for each paired old/new comparison. These are inner-GEMV results on this M4 Max, not
full-model throughput claims.

The accepted shape-aware multi-row policy produced representative improvements:

| Shape `(K,N)` | M | Observed MPS speedup | Observed MLX speedup |
|:---|---:|---:|---:|
| `(2048,2048)` | 4 | 1.12--1.13x | 1.07--1.13x |
| `(8192,2048)` | 4 | 1.26--1.31x | 1.19--1.32x |
| `(8192,2048)` | 8 | 1.19--1.21x | 1.19--1.24x |
| `(8192,2048)` | 17 | 1.23--1.27x | 1.17--1.28x |
| `(2048,8192)` | 4 | 1.50--1.62x | 1.52--1.55x |
| `(2048,8192)` | 17 | 1.30--1.55x | 1.28--1.51x |
| `(2048,8192)` | 32 | 1.14--1.15x | 1.14--1.18x |

The isolated W1 E4 path at `(M,K,N)=(1,2048,2048)` measured 1.14x on MPS and 1.16x on MLX. W2--W4 retain their
separate common-kernel code generation and showed no accepted regression from this specialization.

Latency is small enough that host synchronization and system activity can move individual readings. Dispatch
changes must therefore use paired, warmed sweeps across rates and real projection aspect ratios rather than one
microbenchmark point.

## Rejected or deferred experiments

- The earlier N8 experiment reconstructed both states independently and regressed. The accepted N16 path instead
  reconstructs one state and follows the exact transition recurrence for the next three states.
- A process-wide expanded `65,536 x 4` FP16 decode table removed mixer work but added 512 KiB and regressed large
  K/N shapes by roughly 15--60%; the implicit 256-level table remains production.
- Shared decode at M=2--3 was usually slower, particularly for K=8192, so the multi-row threshold remains M=4.
- Applying compile-time transition specialization to every V4 multi-row rate was neutral or slower in important
  shapes. Only the independent-row E4 closed form was retained.
- An E4-specialized multi-row decoder was neutral and sometimes slower because decode is already amortized across
  rows.
- On MPS, a guarded K64 shared-decode kernel stages two 32-value slabs per barrier pair. It is selected for `M >= 16` with
  `N >= 8192`, or `M >= 24` with `K >= 8192`; smaller and narrow-N boundary shapes retain K32. Across the accepted
  sweep region it measured approximately 1.05x geometric-mean speedup on both MPS and MLX, with larger individual
  wins up to 1.14x MPS and 1.18x MLX. K32 remains the exact fallback.
- On MPS, a further guarded K128 path stages four slabs. It is limited to non-square large projections: narrow N with
  `M >= 24, K >= 8192`, or wide N with `M >= 16, K < 8192`. Marginal W4 wide-N buckets remain on K64. The accepted
  regions measured roughly 1.02x geometric-mean improvement over K64 on both Apple backends; square 8192 shapes
  remain K64 because K128 regressed at M=16.

## Accuracy and lifecycle gates

The Apple implementation is validated against reconstructed-weight FP32 matmul. Tests cover the default FP16 and
range-preserving FP32 inner-output contracts across every supported rate, all V4 half-step rates, M=1/4/17, lifecycle
preservation of `vector_size=4`, repeated launch determinism, malformed inputs, and the dispatch table. An adversarial
full-layer MPS gate forces the decoded inner result above FP16's 65,504 limit while keeping the completed linear
result finite. Accuracy assertions include MSE, relative L2, cosine similarity, forward KLD, top-1 agreement, and
ordered top-5 agreement.

`QVQLinear` dispatches both V2 and V4 payloads to the native MPS kernel when Metal shaders are available. The V4
lifecycle test spies on the kernel entry point and asserts that `vector_size=4` reaches it; dense-reference agreement
alone is insufficient because it can also pass when an obsolete format guard silently selects reconstruction.

The chained MLX decoder is covered across every V4 half-step rate, single-bank and four-bank payloads, M1/M4/M17,
partial K slabs, deterministic replay, dispatch boundaries, and dense-reference metrics. MPS, CPU, V2, CUDA, and
unsupported-format fallbacks are unchanged.

## Next measured work

1. Extend the guarded K64 shared-decode sweep to additional real Q/K/V/O/MLP and MoE shapes before widening dispatch.
2. Benchmark complete `QVQLinear` calls including SU, Hadamard transforms, SV, and bias; inner GEMV is only one stage.
3. Profile Metal occupancy, threadgroup-barrier cost, register pressure, and memory traffic before adding more
   decoded values per synchronization.
4. Add shape buckets only when both MPS and MLX show repeatable wins or maintain separate backend policies when their
   compilers demonstrably diverge.
5. Preserve the implicit decoder, exact planar payload, FP32 accumulation, and tested fallback for every change.

## Backend deduplication decision (2026-08-17)

MLX is now the fused Metal implementation for quantization-time V2 YAQA and V2B2-P32/V2B4-P64 YAQA recurrence
paths. The standard V2 MLX recurrence is explicitly available through `qvq_mlx_viterbi` and is limited to W1--W3.5
for exact-path parity; W4+ uses the native MPS recurrence or the Torch reference path.

We measured matched warmed kernels on the Apple M4 Max at `(M,K,N)=(1,2048,2048)`, W2. Inner GEMV was 0.151 ms
on MPS versus 0.422 ms on MLX (MLX 2.80x slower). Standard V2 YAQA Viterbi on one 128-pair tile measured:

| Rate | MPS Viterbi | MLX Viterbi | MLX/MPS |
|---:|---:|---:|---:|
| W1 | 2.693 ms | 3.773 ms | 1.40x |
| W2 | 2.959 ms | 3.028 ms | 1.02x |
| W2.5 | 4.654 ms | 6.483 ms | 1.39x |
| W3 | 3.119 ms | 3.353 ms | 1.08x |
| W3.5 | 4.580 ms | 6.554 ms | 1.43x |

Therefore MPS kernels are retained where they are measurably faster, rather than deleting a faster implementation merely
to deduplicate source. The MLX V2 recurrence remains the canonical implementation for MLX-native callers and for
banked YAQA quantization, while MPS Viterbi/GEMV remain the faster Torch-MPS paths. All comparisons used FP32
accumulation and dense-reference parity gates; a backend is not removed based on source-count preference alone.
