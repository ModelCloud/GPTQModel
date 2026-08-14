# QVQ V4 quantization kernel design for CUDA SM80

Status: native V4 CUDA Viterbi quantization is implemented behind the existing CUDA capability gate and is under
accuracy/performance validation. The Torch recurrence remains the independent oracle and fallback.

This document defines the CUDA/Ampere `sm_80` plan for quantizing `format="qvq_v4"`. It complements the codec and
quality design in [qvq.md](qvq.md). The objective is to make L16/V4 faster than the production L16/V2 quantizer at
W1--W4 while preserving the reference Torch path, state sequence, squared error, and lowest-state tie breaking.
Nothing here changes the checkpoint payload, QVQ objective, or production format.

## Why V4 can be faster

For a 16 by 16 weight tile, V2 emits 128 two-value transitions and V4 emits 64 four-value transitions. At rate `R`,
their transition widths are `E2=2R` and `E4=4R`. Both evaluate 256 scalar coordinates per state:

```text
V2 emission coordinates: 128 steps * 2 values * 65,536 states = 256S
V4 emission coordinates:  64 steps * 4 values * 65,536 states = 256S
```

The emission FLOP count is therefore similar, but V4 performs half as many recurrence and traceback steps. It also
shrinks the retained suffix and tail-overlap spaces substantially:

| Rate | V2 edge bits | V2 steps | V2 suffix states | V4 edge bits | V4 steps | V4 suffix states |
|---:|---:|---:|---:|---:|---:|---:|
| W1 | 2 | 128 | 16,384 | 4 | 64 | 4,096 |
| W1.5 | 3 | 128 | 8,192 | 6 | 64 | 1,024 |
| W2 | 4 | 128 | 4,096 | 8 | 64 | 256 |
| W2.5 | 5 | 128 | 2,048 | 10 | 64 | 64 |
| W3 | 6 | 128 | 1,024 | 12 | 64 | 16 |
| W3.5 | 7 | 128 | 512 | 14 | 64 | 4 |
| W4 | 8 | 128 | 256 | 16 | 64 | 1 |

This table describes exact dynamic-programming geometry, not an expected quality ranking. V4 consumes state history
more quickly, especially at W3--W4, and must pass propagated model-quality gates independently of kernel speed.

## Kernel architecture

### 1. Use a native four-coordinate emission path

Store the quantization-only codebook as contiguous `float4` rows and load one row per state. Interleave four FP32
products so SM80 can hide dependent FMA latency, and cache the exact FP32 codebook norm once per device/codebook:

```text
d(x, c_s) = max(||x||^2 + ||c_s||^2 - 2 x^T c_s, 0) * step_weight.
```

V4 reads twice as many bytes per codebook row as V2 but executes half as many steps, so total codebook bytes per tile
remain equal. The aligned `float4` access is more efficient than issuing two independent `float2` emissions. Do not
materialize a `[step, state]` distance matrix; fuse emission and suffix reduction so each codebook row is consumed
once per step.

The codebook is 1 MiB in FP32, 512 KiB larger than V2, and exists only during quantization. The 256 KiB norm vector
is process/device cached and shared by modules. These tensors are not checkpoint data and do not affect EBPW.

### 2. Compile rate-family recurrence kernels

One generic reduction is unlikely to win across edge widths 4--16. Dispatch on the exact integer edge width:

| V4 rates | Edge bits | Suffixes | SM80 reduction plan |
|:---|---:|---:|:---|
| W1 | 4 | 4,096 | One-thread local 16-way minimum; two FP32 suffix buffers in shared memory. |
| W1.5 | 6 | 1,024 | One-thread local 64-way minimum; enough CTAs/batches to cover all live SMs. |
| W2--W2.5 | 8--10 | 256--64 | Warp-cooperative 256/1,024-way segmented minima with fixed reduction order. |
| W3--W3.5 | 12--14 | 16--4 | Multiple warps per suffix, then a short deterministic shared-memory merge. |
| W4 | 16 | 1 | Dedicated memoryless nearest-codeword kernel; no recurrent suffix state. |

The current V2 CUDA implementation uses one 1,024-thread CTA per sequence. V4 should sweep 256, 512, and 1,024
threads per rate family rather than inheriting that launch shape. Register count, shared-memory use, active warps,
and measured occupancy decide the launch. Device SM count and shared-memory limits must be queried at runtime; no
fixed GPU index or audit-host inventory belongs in dispatch logic.

W1 needs two 4,096-float suffix buffers, or 32 KiB, before reduction scratch. W1.5 and above need much less. This is
comfortably below the A100-class SM80 opt-in block limit, but the launcher must still query and validate the actual
device limit and fail to the Torch reference when unsupported.

### 3. Shrink backpointers without changing their meaning

The current native implementation stores predecessor prefixes in `int32`. V4 needs fewer entries, and the prefix
range is known from `E`:

- `uint8` is exact for `E <= 8` (W1--W2).
- `uint16` is exact for `8 < E <= 16` (W2.5--W4).
- W4 needs no recurrent backpointer; the selected state is the emitted state.

Use byte-addressable arrays rather than sub-byte packing; sub-byte read/modify/write usually costs more than it saves.
Traceback widens values to `uint32` before shifts. Keep an `int32` debug path until byte/halfword storage is bit-exact
for constrained and unconstrained searches.

At W2, V4 has about 32 times fewer backpointer entries than V2. At W2.5, it has about 64 times fewer entries; using
`uint16` instead of the current `int32` doubles that byte reduction. This permits larger tile batches without a large
workspace increase.

### 4. Fuse tail-biting work and batch candidate reruns

Do not launch a new host-controlled pipeline for every candidate. The provisional search should retain the overlap
scores needed for ranking. Constrained reruns should batch `(tile, candidate)` pairs in one launch and reuse the
device codebook, norm table, and workspace. V4's overlap count is only 4,096 at W1, 256 at W2, 64 at W2.5, and one at
W4, so candidate widening is materially cheaper than V2.

Candidate batching changes throughput only. Candidate order, exact FP32 score, and deterministic final selection
must match the reference. Never choose a candidate using trellis SSE alone when the caller requests a downstream
selection metric; that selection remains outside the Viterbi kernel.

### 5. Special-case memoryless W4

At V4 W4, `E=L=16`. Each transition directly chooses one of all 65,536 states and retains no prior state bits. The
exact objective separates by step, so recurrence, tail constraint, backpointer allocation, and traceback are all
unnecessary. Use a streaming fused nearest-codeword kernel that emits one state and exact loss per four-value target.
It should still use the same `float4` codebook and tie-break toward the lowest state index.

This special case is an optimization of the same math, not an approximate search.

### 6. Reuse workspaces and make batching occupancy-aware

Maintain per-device reusable workspaces for suffix costs, temporary minima, backpointers, and output states. Grow
them monotonically to the required batch, but do not retain module-sized weight or activation tensors. Choose batch
size from runtime device properties, current free memory, rate family, and tile count. More quantization workspace is
acceptable, but the implementation must avoid making assumptions from CUDA device ordinal.

The primary batching goal is enough independent CTAs to occupy the live SMs. A single trellis cannot expose enough
parallel CTAs because recurrence steps are sequential, so module tiles and tail candidates are the natural batch
dimensions.

### 7. Keep exact math as the production path

Production Viterbi uses FP32 emissions, FP32 costs, explicit round-to-nearest operations where the reference requires
them, and a fixed `(cost, state_index)` reduction order. TF32 or FP16 Tensor Core distance evaluation can perturb
near ties and select a different circular path, so it must not silently replace the exact path.

An experimental two-stage search may use Tensor Cores to screen candidates, followed by exact FP32 rescoring and an
exact baseline candidate. It is not equivalent to exhaustive Viterbi because the true winner can be screened out.
It therefore requires a separate opt-in flag, path-churn reporting, and full propagated quality validation; it is
not part of the initial SM80 implementation.

## Expected performance targets

These are acceptance targets, not measured results:

| Rate family | First native target versus V2 | Stretch target | Main source of gain |
|:---|:---|:---|:---|
| W1--W1.5 | `<= 0.75x` wall time | `<= 0.60x` | Half the recurrence plus smaller suffix workspace. |
| W2--W2.5 | `<= 0.67x` wall time | `<= 0.50x` | Much smaller suffix/backpointer state and batched tail search. |
| W3--W3.5 | `<= 0.75x` wall time | `<= 0.55x` | Few suffixes; specialized cooperative reductions. |
| W4 | `<= 0.50x` wall time | `<= 0.35x` | Exact memoryless nearest-codeword specialization. |

Compare the same number of tiles, tail candidates, objective mode, and Viterbi batch. Report emission, recurrence,
tail ranking/reruns, traceback, packing, and total wall time separately. A faster V4 reference fallback is not a
native-kernel result.

## Correctness and benchmark gates

Before enabling CUDA dispatch for `format="qvq_v4"`:

1. Match Torch states and squared error bit-for-bit at every half-step W1--W4 for ordinary and weighted searches.
2. Cover exact ties, all-zero/extreme inputs, non-finite rejection, tail constraints, candidates 1/4/8/16, one-step
   and short sequences, odd batch counts, and maximum supported batches.
3. Verify planar pack/reconstruct parity and quantize-save-load-reconstruct parity.
4. Test deterministic replay, non-default streams, concurrent devices, CUDA graphs where supported, and explicit
   fallback/rejection on non-SM80 hardware.
5. Record device properties, PyTorch/CUDA/NVCC/driver versions, build flags, generated `sm_80` code, dtype, shapes,
   warmups, CUDA-event samples, workspace peak, register count, shared memory, and achieved occupancy.
6. Compare propagated two-layer and full-model metrics against the Torch V4 artifact. Kernel optimization must not
   change decoded weights, final KLD/JSD, top-k agreement, answer margins, or benchmark outputs.

## Implementation order

1. Add an explicit V4 native ABI carrying `vector_size=4` and `transition_bits=4R`; never reinterpret V4 as V2.
2. Implement exact `float4` emission and W1/W1.5 recurrence, retaining `int32` backpointers for initial parity.
3. Add W2--W3.5 rate-family reductions and the W4 memoryless kernel.
4. Add `uint8`/`uint16` backpointers, batched tail candidates, and reusable workspaces.
5. Tune CTA size and batch policy on actual SM80 profiles, then enable dispatch only for measured wins.


## 2026-08-13 CUDA V4 implementation progress

- The Torch reference remains the accuracy oracle: `reconstruct_qvq_inner_weight(..., vector_size=4)` followed by dense `x @ weight`. Quantization payload/state generation is unchanged; quantization parity is therefore required to be exact (100%).
- Added an opt-in `gemv_v4` CUDA ABI for SM80+ and selected it from `QVQLinear` only when `vector_size=4`; the established V2 `gemv` ABI is unchanged.
- V4 packing uses `E=rate*4`, two planar words per E-bit plane, and 64 transitions per 16x16 tile. The device decoder applies the exact two PGC16 mixer chains (`state` and `state ^ 0xA5A5`) and canonical FP16 level table.
- Initial random-trellis parity on GPU 0 for W1/W2/W3/W4 produced max absolute error 0.0 and bitwise equality against the Torch reference. This validates reconstruction/inference math before performance work.
- Failed experiment: first compile attempt omitted the default `VectorSize=2` template argument for legacy WMMA call sites; compilation failed only, with no runtime result. Fixed before parity testing.
- No Nsight Compute measurements have been accepted yet. GPU 0 and GPU 1 were idle; busy GPUs were not used. Next profiling must compare V4 against the Torch reference and capture kernel metrics on an uncontended GPU.

### Native V4 CUDA quantizer and half-step rate gate (2026-08-13)

The native Viterbi ABI now carries `vector_size=4` explicitly as `viterbi_v4`; V4 is never reinterpreted through the
legacy V2 operator. Emission, codebook norms, target strides, and tail/weighted searches are specialized for four
coordinates. The quantizer dispatches CUDA V4 for all supported half-step rates W1, W1.5, W2, W2.5, W3, W3.5, and W4.
The first implementation accidentally advanced V4 targets by two scalars after step zero; the error was found by the
multi-step oracle test and corrected to a `VectorSize` stride before accepting any benchmark.

The recurrence now computes the immutable 65,536-entry codebook-norm table once per launch and shares it across all
batch CTAs. The prior implementation rebuilt that table independently for every batch item. This is an offline
quantization-only workspace optimization: it changes neither the decoded format nor the objective, and does not retain
dequantized weights.

The exact validation command (Python 3.14.6 free-threaded, `PYTHON_GIL=0`) is:

```text
PYTHON_GIL=0 CUDA_VISIBLE_DEVICES=0 python -m pytest tests/test_qvq_cuda.py -k 'v4_cuda_viterbi' -q
```

The state sequence matched the independent eager recurrence for every half-step rate and for one-, three-, and
17-step sequences. The loss scalar is compared with `rtol=1e-6, atol=1e-5`: CUDA uses a fixed FP32 FMA reduction while
the host eager oracle uses its BLAS reduction, so the paths are exact while the reported accumulated loss can differ by
only a few ulps. Weighted and overlap-constrained W2/V4 cases also pass exact state parity. A direct standalone SM80
extension load exercised the same production kernel when the full multi-source JIT build was contended by unrelated
compilations.

The first full-extension build attempt failed at compile time because the launch macro still referenced the old
single-template kernel (`qvq_viterbi_kernel<BITS>`). It was corrected to include `VectorSize`; there was no runtime
result or checkpoint produced by that failed build. No illegal-address, non-finite, or state-mismatch failure occurred
after the correction.

Synchronized CUDA-event microbenchmarks used the same `batch=16, steps=64, vector_size=4` workload and one PG506-230/232
SM80 GPU per process. The A/B baseline is the immediately preceding V4 kernel at commit `7c6fb5d0`; current results are
after the shared codebook-norm change (same source, build flags, and workload). Lower milliseconds are better.

```text
+------+---------------+---------------+-------------+-------------+----------+---------+
| Rate | Before ms GPU | After ms GPU  | Speedup     | State exact | Finite   | GPU     |
+------+---------------+---------------+-------------+-------------+----------+---------+
| W1   | 2.084352 (0)  | 1.953126 (0)  | 1.067x      | yes         | yes      | PG506-230 |
| W1.5 | 2.089011 (0)  | 2.212864 (0)  | 0.944x      | yes         | yes      | PG506-230 |
| W2   | 2.541926 (1)  | 2.218701 (1)  | 1.146x      | yes         | yes      | PG506-232 |
| W2.5 | 3.001037 (1)  | 2.212506 (1)  | 1.356x      | yes         | yes      | PG506-232 |
| W3   | 2.427341 (2)  | 2.201446 (2)  | 1.103x      | yes         | yes      | PG506-230 |
| W3.5 | 2.462822 (2)  | 2.377830 (2)  | 1.036x      | yes         | yes      | PG506-230 |
| W4   | 2.897203 (3)  | 2.881024 (3)  | 1.006x      | yes         | yes      | PG506-230 |
+------+---------------+---------------+-------------+-------------+----------+---------+
```

The W1.5 slowdown is within the expected small-kernel timing noise and is not promoted as a regression or a speed
claim; repeated uncontended samples and complete quantization wall time are still required. W2.5 is the largest current
gain because the shared norm construction was a larger fraction of its recurrence work. These numbers are inner Viterbi
kernel time, not end-to-end model quantization time; future reports must separately measure emission, recurrence,
traceback, packing, and total layer wall time.

### Exact W4 memoryless specialization (2026-08-13)

W4 has `transition_bits=16`, so every 16-bit state is a complete four-value transition and `suffix_count=1`. The
generic kernel previously serialized all 64 steps inside one CTA per batch item. The W4 path now launches one exact
nearest-codeword CTA per `(batch, step)`, applies optional step weights, preserves FP32 emission and lowest-state
tie-breaking, and reduces the per-step losses in a fixed order. This is algebraically identical to the generic
recurrence; it is not an approximate or screened search. A new weighted W4 unit test covers zero/nonzero weights.

The same CUDA-event workload as the preceding table (`batch=16`, `steps=64`, V4, SM80) measured:

```text
+------+---------------+--------------+----------+-------------+
| Rate | Before (ms)   | After (ms)   | Speedup  | State exact |
+------+---------------+--------------+----------+-------------+
| W4   |      2.881024 |     0.407757 |   7.066x | yes         |
+------+---------------+--------------+----------+-------------+
```

The loss remains finite and agrees with the eager FP32 oracle within the existing strict tolerance; decoded states are
exact. This is an inner-kernel result and does not imply a 7x end-to-end model quantization gain. The next target for
the 2x objective is rate-family recurrence parallelism at W1--W3.5, where predecessor dependencies still serialize
steps.

### Production-batch throughput check (2026-08-13)

The default CUDA trellis batch sizes are intentionally much larger than the small parity benchmark (`496` for W1/W1.5,
`512` for W2, and `3968` for W2.5--W6.5) so independent tiles occupy the SM80 device. A CUDA-event run on GPU 0 with
`steps=64`, V4, and `batch=496` measured the following. The per-trellis column is the synchronized kernel time divided
by the number of independent trellises; it is a throughput attribution, not a separate launch measurement.

```text
+------+------------+---------------+----------------+
| Rate | Batch      | Total ms      | ms / trellis   |
+------+------------+---------------+----------------+
| W1   | 496        | 7.230310      | 0.014577       |
| W1.5 | 496        | 7.599923      | 0.015323       |
| W2   | 496        | 7.311974      | 0.014742       |
| W2.5 | 496        | 7.271577      | 0.014660       |
| W3   | 496        | 7.149158      | 0.014414       |
| W3.5 | 496        | 7.543245      | 0.015208       |
+------+------------+---------------+----------------+
```

All outputs were finite and retained the exact eager-selected paths in the preceding half-step gate. The combination
of occupancy-sized batching and the exact W4 memoryless path is the current route to a broad 2x+ quantization-speed
improvement; small-batch recurrence timings should not be used to reject the production configuration.


### Audit update: V4 is not release-ready yet

- Corrected a V4 CUDA staging defect: the tile-load index divisor must be `2 * E`, not the V2 `4 * E`; both normal and split-K paths were affected. A multi-K-tile CUDA test (`K=64`) now passes exact FP16 output parity after this fix.
- The earlier one-tile parity result was insufficient and is explicitly superseded.
- Native V4 CUDA quantization is implemented through the explicit `viterbi_v4` ABI. The wrapper passes
  `qvq_transition_bits(bits, vector_size=4)` and the Torch recurrence remains the independent accuracy oracle.
  Full lifecycle propagation and V2-vs-V4 quality gates are still required before promotion.
- V4 prefill WMMA and state reuse remain unimplemented; no speed claim or Nsight result is accepted until quantization and the full V4 CUDA coverage matrix are complete.


## Experimental investigation: four implicit banks per V4 rate

Status: opt-in banked lifecycle implemented; canonical `format="qvq_v4"`
remains single-bank by default. `bank_count=4` selects the explicit banked
variant and stores packed selectors; it is not a reinterpretation of existing
single-bank checkpoints.
Current V4 has one fixed decoder mapping shared by every model, module, and tile. The banked variant adds four
rate-keyed implicit mappings and stores a two-bit winning-bank selector per 16 by 16 tile.

### Goal and non-goals

The goal is to improve the unstable W1.5--W2.5 region by giving each tile several complementary four-dimensional
reconstruction geometries while retaining:

- the L16/V4 state and 64-transition tile;
- the exact `R`-bpw planar path payload;
- the canonical 256-entry FP16 scalar compander;
- two PGC-style mixes and four scalar lookups per state;
- no serialized dense codebook;
- a fixed, activation-independent weight at inference.

This does **not** increase the selected trellis's legal successors. At rate `R`, every selected bank still has
`2**(4R)` outgoing edges per state and exactly 65,536 state-addressed four-vectors. The quantizer chooses one of four
manifolds for the whole tile before storing its path:

```text
four-bank union available during quantization
    |
    +-- bank 0: 65,536 four-vectors
    +-- bank 1: 65,536 four-vectors
    +-- bank 2: 65,536 four-vectors
    +-- bank 3: 65,536 four-vectors
    |
    v
one bank selected and stored for this 16x16 tile
    |
    v
ordinary 64-state-path V4 inference under that fixed bank
```

The reference codec now defines four rate-keyed bank masks, validates packed
selectors, reconstructs banked tiles in Torch, and accepts an optional
`bank_ids` auxiliary tensor in `QVQLinear`. Bank zero is bit-for-bit identical
to canonical V4. CUDA V4 inference consumes dense unpacked selectors per tile;
checkpoint storage remains packed. Banked Block-LDLQ and YAQA select banks
sequentially while feeding committed errors into later tiles, and the full
mixed result is gated against canonical bank 0 under the active Hessian proxy.

`select_banked_tiles_by_output_error` is the corresponding held-out workflow
primitive. It starts from bank 0, scores each tile through real held-out
module inputs, updates the output residual after every tile, and resolves ties
to bank 0. This makes bank decisions sensitive to propagated output error
within a projection rather than only local weight SSE. Four-bank V4
Block-LDLQ enables this module-boundary refinement by default in
`QVQProcessor`: the dense capture hook reserves the final one-eighth of valid
token rows from each captured batch, up to 512 rows, excludes those rows from
the input Hessian, and uses them only for proposal acceptance. It never uses
final benchmark rows implicitly. Direct `quantize_qvq_linear` callers still
have to supply an explicit gate.

The gain, if any, comes from choosing a better state-to-vector geometry for a tile. It is not equivalent to expanding
the L16 state to 18 bits, and it does not expose 4x more branches at every transition.

### Decoder definition

For rate `r`, bank `b`, and 16-bit state `s`, define

```text
D[r,b](s) = {
    G[high(P[r,b,0](s))],
    G[ low(P[r,b,0](s))],
    G[high(P[r,b,1](s))],
    G[ low(P[r,b,1](s))]
}.
```

`G` remains the canonical versioned FP16 table. Each `P` is a cheap versioned implicit permutation constructed from
xor shifts, an odd 16-bit multiply, an add, and a fixed xor seed. The first pair permutation must be bijective, which
keeps all 65,536 four-vectors unique regardless of the second pair. Bank 0 must be exactly the current decoder:

```text
P[r,0,0](s) = pgc16_mix(s)
P[r,0,1](s) = pgc16_mix(s ^ 0xA5A5)
```

Keeping bank 0 unchanged provides an exact fallback candidate and makes every banked experiment comparable with the
current V4 artifact. Banks 1--3 must change geometry relative to the fixed transition graph. A mere global relabeling
that is a trellis automorphism may expose exactly the same path set and provide no new reconstruction capacity.

### Why four banks, and why one family per rate

Four banks are the first experimental point, not a theorem that four is universally optimal:

- Four choices need exactly two selector bits and map naturally to one integer/warp-uniform lookup.
- They provide three complementary alternatives while always retaining current V4 as bank 0.
- Four bank/path searches are straightforward to batch on CUDA and are still small enough for exact enumeration.
- Two banks may not provide enough geometric diversity; eight banks double candidate-search work again and increase
  the decoder-constant selection space. Both remain required comparison controls.

The banks should be **per rate**, not one four-bank family reused from W1 through W4. The state graph seen by the
decoder changes materially with `E=4R`:

| Rate | Edge bits | Successors per state | Retained state bits | Primary bank-design pressure |
|---:|---:|---:|---:|:---|
| W1 | 4 | 16 | 12 | Orthant and magnitude coverage in very small local neighborhoods. |
| W1.5 | 6 | 64 | 10 | Crosses the minimum 64-successor target; balance local coverage and memory. |
| W2 | 8 | 256 | 8 | Improve four-coordinate shaping under severe propagated-error sensitivity. |
| W2.5 | 10 | 1,024 | 6 | Stabilize path geometry near the observed task-quality frontier. |
| W3 | 12 | 4,096 | 4 | Geometry is less constrained; avoid sacrificing useful short history. |
| W3.5 | 14 | 16,384 | 2 | Likely smaller benefit; retain mainly as a consistency/control arm. |
| W4 | 16 | 65,536 | 0 | Memoryless nearest-vector problem; optimize global covering geometry. |

A permutation optimized for contiguous 16-successor neighborhoods at W1 is not generally optimal for 1,024-way
neighborhoods at W2.5 or the global memoryless search at W4. Rate-specific families can optimize the actual block
geometry, occupancy, and retained-history regime. For the seven half-step rates W1--W4, the codec defines 28 small
implicit mappings. A module's existing rate selects its family, so no additional family ID is stored.

The first experiment should use codec-defined universal families shared by every model and module. If they help but
leave systematic role-dependent error, evaluate model-level or role-level learned constants later. Do not start with
per-module bank definitions: their raw bytes are small, but decoder variability, overfitting, kernel specialization,
and validation costs grow with every module.

### Quantization-time bank selection

The selector is an exact discrete choice, not an inference-time router and not initially a gradient-trained tensor.
For corrected tile target `T_t`, rate `r`, and bank `b`, run the same configured tail-biting search under that bank:

```text
feedback-corrected tile T_t
    |
    +-- bank 0 -> exact Viterbi/path candidates -> Q[t,0]
    +-- bank 1 -> exact Viterbi/path candidates -> Q[t,1]
    +-- bank 2 -> exact Viterbi/path candidates -> Q[t,2]
    +-- bank 3 -> exact Viterbi/path candidates -> Q[t,3]
    |
    v
score with the active Block-LDLQ or YAQA acceptance objective
    |
    v
store winning path plus bank_id; propagate only the winner's error
```

Conceptually,

```text
(b_t*, p_t*) = argmin over b,p of D_t(T_t, Decode[r,b](p)).
```

`D_t` must be the objective configured for the active rounding pipeline. Plain local MSE is an insufficient final
selector. Under input-Hessian weighting a candidate error `E_b=T_t-Q[t,b]` has proxy

```text
D_BlockLDLQ(b) = trace(E_b H_I E_b^T),
```

and a YAQA candidate uses the configured two-sided proxy

```text
D_YAQA(b) = trace(E_b H_I E_b^T H_O).
```

The candidate must be generated from the current sequentially corrected target, and only the winning error may enter
later LDLQ/YAQA feedback. Choosing every tile independently from the original dense weight would break the sequential
objective. Ties resolve first to lower loss, then lower bank ID, then the reference's lower-state/path ordering; this
keeps bank 0 as the deterministic fallback.

The CUDA quantizer should batch `(tile, bank, tail_candidate)` rather than issue four serial host pipelines. Codebook
norms and immutable bank definitions are shared. With the whole-module bank-0 rollback oracle enabled, naive arithmetic
is approximately 5x the Viterbi candidate work (four bank candidates plus one canonical reference pass), but wall time
can grow much less when additional banks fill otherwise idle SM80 CTAs. Quantization time and workspace must be reported
separately; neither is an inference-format cost.

The production module-boundary propagation stage retains bank alternatives,
replays held-out module inputs, and accepts a banked proposal only on its
reserved output objective. It uses a locked selection set and cannot select
directly on final benchmark test rows. A separate deferred full-model pass can
replace this module-output gate with final-logit or task-like replay.

At the low-level API, provide `propagated_inputs` and
`propagated_target_output` to `quantize_qvq_linear`, or register them with
`QVQProcessor.set_propagation_gate`. The `propagated_acceptance` callback
receives the proposed dense weight and the current accepted baseline (which
may already be mixed-bank or Hessian-selected) and returns `True` to commit.
The callback remains mandatory at that boundary; there is no veto-only
default.

At the model lifecycle boundary, effective behavior is default-on whenever a
module uses `format="qvq_v4"`, `bank_count=4`, and
`rounding="block_ldlq"`. `QVQProcessor.preprocess` enables the per-module
switch even though the raw `QVQConfig.propagated_bank_selection` field defaults
to `False` so single-bank, V2, and YAQA configurations remain valid. The dense
hook automatically constructs a held-out module-output callback. A caller-set
gate takes precedence over that automatic gate. Consequently, explicitly
setting the public field to `False` does not disable propagation for the
four-bank Block-LDLQ lifecycle; use a non-banked configuration or a different
rounding pipeline to omit it.

The selector's internally reported module-output loss remains diagnostic. The
acceptance callback is authoritative. The automatic lifecycle callback uses
the reserved module-output MSE; an explicitly registered callback may instead
use next-module, final-logit, task-margin, or another downstream objective and
may accept a locally worse finite proposal when it improves that objective.

For a deferred full-model pass, `QVQCandidate` stores the exact trellis, bank
selectors, transforms, and decoded diagnostic weight. `QVQPropagationRefiner`
installs each candidate through its serialized tensors, evaluates the complete
model, and atomically restores rejected candidates. The refiner is an explicit
opt-in attachment; ordinary `QVQProcessor` quantization does not run a
full-model replay implicitly. Default-on module-boundary propagation must not
be described as this full-model refiner.

The propagated implementation scores the exact inner-space residual in
held-out-row chunks (currently 256 rows per workspace pass). It no longer has a
logical tile-count cap and does not materialize a full dense reconstruction for
each proposal. Module-boundary selection is enabled by default for four-bank
V4 Block-LDLQ; the stronger deferred full-model refiner remains opt-in pending
full-model propagated/task gates. Callers should budget the four-bank candidate
tensors and acceptance replay during quantization.

CUDA smoke validation on an NVIDIA PG506-230 (Python 3.14.6, ``PYTHON_GIL=0``)
confirmed the opt-in path across the first large projection boundaries:

| Shape | Held-out rows | Baseline sec | Propagated sec | Nonzero selectors | Pack/decode max error |
|---|---:|---:|---:|---:|---:|
| 2048x2048 | 257 | 3.478 | 6.125 | 26 | 0.0 |
| 4096x2048 | 257 | 5.333 | 8.793 | 38 | 0.0 |

The 257-row cases cross the 256-row workspace boundary. These are correctness
and overhead smoke tests, not quality-promotion results. Four-bank Block-LDLQ
now runs the module-boundary gate by default; these measurements do not promote
the separate deferred full-model refiner.

### Serialization and effective BPW

Pack one two-bit selector per 256-weight tile, 16 selectors per int32 word:

```text
selector overhead = 2 bits / 256 weights = 0.0078125 bpw.
```

The planar path remains exactly `R` bpw, so the banked raw rate is `R + 0.0078125` bpw before existing `SU`, `SV`,
and metadata. The selector adds about 0.39% to a W2 path payload. Over two trillion quantized weights it is about
1.953 GB decimal, compared with a 500 GB decimal W2 path payload.

Universal bank definitions are frozen into a new codec version and add no checkpoint tensor. If later experiments
learn model-level implicit constants, serialize their exact integer bit patterns once per rate family; do not add a
dense codebook tensor or silently change a codec-defined bank. Mixed-rate modules use their already-serialized rate
to select the correct bank family.

### Inference behavior and cost

At inference, each tile always uses the selector stored during quantization. Activations do not choose or train it:

```text
tile_id -> load 2-bit bank_id -> select fixed mixer constants -> decode 64 V4 states -> GEMV/GEMM
```

All threads consuming one tile observe the same bank, so a switch is warp-uniform. The preferred CUDA path loads the
selector once per tile/CTA, places the selected constants in registers, and executes the ordinary fused V4 decoder.
The CUDA path keeps the selector as transient tile metadata and does not retain an expanded dequantized cache. Do not
launch separate kernels per bank unless profiling proves that sorting/indirection is cheaper than the uniform selector.
`QVQLinear` caches only the unpacked selector metadata between forwards, invalidating it on device moves or selector
mutation; this removes repeated host-side selector expansion without caching decoded weights.

The production gate is no more than 5% inference slowdown versus single-bank V4 for every supported target shape and
dtype. Report selector traffic, registers, occupancy, L1/L2 behavior, decode latency, complete `QVQLinear` latency,
and CUDA Graph behavior. A tiny selector payload does not prove negligible runtime cost.

### Who should benefit—and who may not

The strongest expected beneficiaries are:

- W1.5, W2, and W2.5 tiles whose current local successor sets have poor sign/orthant, radius, or covariance coverage;
- heterogeneous attention, MLP, and MoE modules that prefer different standardized reconstruction geometries;
- tiles near downstream answer-margin thresholds, where changing correlated error direction can prevent propagated
  decision flips;
- very large models, because bank definitions remain shared while billions of tiles choose independently.

W1 may benefit from better geometry but still has only 16 outgoing V4 edges, so banks do not solve its fan-out limit.
W3.5/W4 may see little benefit because their local/global choice sets are already large. Banks also cannot repair a
poor compander magnitude distribution, inaccurate Hessian/Fisher factors, cross-layer error interactions, or an
inferior V4 topology. A local proxy gain is not evidence of end-to-end recovery.

### Designing the four banks

Construct banks offline from a diverse training corpus of transformed, standardized tiles. Keep bank 0 fixed and
search only cheap bijective mixer constants/seeds for banks 1--3. Optimize a multi-objective rate-specific score:

- worst-case and occupancy-weighted local successor covering radius;
- all 16 four-dimensional sign-orthant occupancy and balance;
- centroid norm and covariance condition of each legal successor neighborhood;
- radial/tail and coordinate-correlation diversity;
- low duplication and low nearest-vector agreement between banks;
- exact Viterbi proxy improvement on held-out modules;
- propagated next-module, next-layer, and final-logit behavior on disjoint prompts.

Do not select banks solely from synthetic Gaussian nearest-neighbor MSE. RHT makes weights more Gaussian-like, but
the observed low-rate failures are downstream- and module-sensitive. Freeze the corpus split, rate-specific mixer
constants, compander bits, and tie rules in the codec version before model evaluation.

If constrained implicit banks show value, an EM-style model-level experiment may alternate between assigning each
tile to a bank/path and updating a small constrained set of integer mixer constants. Every update must preserve
bijection and be followed by exact re-encoding. Arbitrary learned 65,536 by 4 codebooks are outside this design
because they add storage, cache traffic, and a different decoder.

### Required A/B matrix and acceptance gates

Compare at least:

| Arm | Selector cost | Purpose |
|:---|---:|:---|
| Current one-bank V4 | 0 bpw | Exact production-format control. |
| Two banks per rate | 0.00390625 bpw | Test whether one complementary mapping is sufficient. |
| Four banks per rate | 0.0078125 bpw | Proposed balance of diversity, search cost, and selector width. |
| Eight banks per rate | 0.01171875 bpw | Measure marginal benefit versus doubled candidate work. |
| Four banks shared across rates | 0.0078125 bpw | Test the necessity of rate-specific families. |
| Random legal banks | Same as candidate | Control for gains from deliberate geometry design. |
| Matched extra-bit/mixed-rate rescue | Matched actual EBPW | Determine whether selectors beat simply spending bits. |

The four-bank design advances only if all of these hold:

1. Bank 0 reconstructs every state and tile bit-for-bit identically to current V4.
2. Selector packing is bit-exact across Torch and every supported backend, including save/reload.
3. The chosen bank/path never has worse configured quantization proxy than bank 0 for the same corrected target.
4. Held-out propagated KLD/JSD, answer margins, paired top-k flips, and task behavior improve at W1.5--W2.5 across
   multiple calibration/Fisher seeds; local MSE alone cannot promote it.
5. Compare actual EBPW against uniform and mixed-rate controls, including all selector and metadata bytes.
6. CUDA quantization reports the separate 4x candidate cost and demonstrates useful batched-SM80 scaling.
7. Torch, CUDA, MPS, and MLX reconstruction agree exactly on bank selection semantics and decoded FP16 values.
8. CUDA/MPS/MLX inference is no more than 5% slower than single-bank V4 at every gated W1--W4 shape.
9. New banks receive material selector occupancy; collapsed occupancy is reported and the unnecessary banks are not
   retained merely because aggregate error moved within noise.
10. The feature remains experimental and default-off until complete two-layer, full-model, save/load, and external
    benchmark validation passes.

### Audit disposition (2026-08-13)

- **Fixed:** the native wrapper preserves V4's `E=4R` transition width. A regression test exercises the wrapper
  through tail-biting, not only the raw extension ABI.
- **Fixed:** `format=qvq_v4` dynamic module overrides reject rates above W4 at configuration time.
- **Covered:** lifecycle tests run both V2 and V4 through masked calibration, Hessian finalization, RHT/QVQ
  installation, replay, and runtime-module output parity on a CPU reference fixture. Full-model save/reload and
  propagated V2-vs-V4 quality gates remain integration benchmarks, not unit-test substitutes.
- **Not promoted:** the fixed second mapping (`state ^ 0xA5A5`) has not demonstrated better four-dimensional local
  geometry or propagated quality than V2. More successors alone are not an accuracy argument; any replacement
  permutation must be searched and evaluated at identical EBPW with held-out KLD/JSD/top-k and task gates.
- **Format limitation:** `format=qvq_v4` is currently a model-wide codec choice. A V4-low/V2-high hybrid requires an
  explicit per-module codec/vector-size metadata extension and is not silently represented by dynamic bit overrides.

### Offline V4 candidate screening (2026-08-13)

`gptqmodel.quantization.qvq_v4_candidates` and
`scripts/benchmark_qvq_v4_candidates.py` provide an opt-in pruning stage for
alternative second-pair XOR mappings and V4-specific normalized scales. The
utility is deliberately not wired into `QVQConfig`, checkpoint packing, or
`QVQLinear`: candidate mappings have no serialized decoder metadata and cannot
be promoted from a local score alone. Every survivor must pass live-prefix and
held-out final-logit gates at matched EBPW.

The first real-weight screen used 4,096 normalized four-weight vectors from
`Llama-3.2-1B-Instruct/model.layers.0.self_attn.q_proj.weight` on an SM80 GPU:

| Candidate | XOR mask | Scale | Local MSE | P95 | Orthants |
|:--|--:|--:|--:|--:|--:|
| canonical | `0xA5A5` | 1.00 | 0.164530 | 0.533624 | 16 |
| mask-5A5A | `0x5A5A` | 1.00 | 0.130450 | 0.362871 | 16 |
| mask-3C3C | `0x3C3C` | 1.00 | **0.127504** | **0.347040** | 16 |
| mask-C3C3 | `0xC3C3` | 1.00 | 0.164421 | 0.474364 | 16 |
| mask-9696 | `0x9696` | 1.00 | 0.220662 | 0.736108 | 16 |
| canonical | `0xA5A5` | 0.95 | 0.174607 | 0.578692 | 16 |
| canonical | `0xA5A5` | 1.05 | 0.157304 | 0.502967 | 16 |

This is evidence that the fixed mapping is not locally optimal for this
weight slice, not evidence of an end-to-end improvement. The result is saved
only as `/tmp/qvq_v4_candidates_l0.json`; no checkpoint or production default
was changed.

The candidate codebooks were also passed through the native V4 BlockLDLQ
selector on one real 16x16 Llama Q-projection tile (W2, identity Hessian,
CUDA, free-threaded Python):

| Candidate | Selected-tile MSE |
|:--|--:|
| canonical `0xA5A5` | 0.08712156 |
| mask `0x5A5A` | 0.08076598 |
| mask `0x3C3C` | **0.07661094** |
| canonical, scale 1.05 | 0.08712156 |

The scale-only control did not change this tile because the selector operates
on the normalized target; scale search must therefore be evaluated in the full
weight/Hessian path. The mapping improvement surviving native Viterbi is
useful pruning evidence, but it still has no live activation or final-logit
validation and remains non-promotable.

A held-out output proxy rejected the apparent local winner. Using the first
128x128 slice of the same Q projection, independent 16x16 W2 tiles, identity
tile Hessians, and 64 fixed random probe vectors:

| Candidate | Weight relative-L2 | Output relative-L2 | Gate |
|:--|--:|--:|:--|
| canonical `0xA5A5` | 0.316255 | **0.300047** | baseline |
| `0x5A5A` | 0.313782 | 0.301432 | reject |
| `0x3C3C` | **0.310328** | 0.304177 | **reject** |

This demonstrates a local-MSE/output-error inversion on real weights. The
candidate framework therefore correctly refuses promotion; a future winner
must pass activation-derived prefix and final-logit KLD/JSD/top-k gates, not
only weight or tile diagnostics.

#### Experimental-codebook serialization safety

Candidate results are evaluation-only. `QVQLinearQuantizationResult` marks
results created with an experimental codebook as non-serializable, and its
`serialized_tensors()` method fails closed before any tensor is staged for a
checkpoint. The QVQ lifecycle uses this method rather than reconstructing a
payload dictionary, so a candidate cannot silently enter a production save or
reload path. Canonical A5A5 results remain serializable. Experimental tables
also bypass the canonical device cache; the cache is populated only from the
official versioned PGC16 levels, so prior candidate data cannot pollute a later
production quantization.

### V4 W2 held-out mapping gate and quantization telemetry (2026-08-13)

The research-only dense reconstruction gate quantized every Q/K/V/O and MLP
projection in the first two Llama-3.2-1B-Instruct decoder layers. It used 128
unpacked `neuralmagic/calibration:LLM` rows (49,725 valid tokens) for Hessians
and the disjoint rows `[128,256)` (15,635 valid tokens, 749 padding tokens
excluded) for output metrics. Three independent RHT seeds were run on SM80
with Python 3.14.6 free-threading:

| Seed | Mapping | KLD | JSD | Top-1 | Top-5 overlap |
|--:|:--|--:|--:|--:|--:|
| 0 | canonical `0xA5A5` | 0.158207 | 0.030246 | 0.903742 | 0.815555 |
| 0 | `0x3C3C` | **0.142420** | **0.027697** | **0.909178** | **0.824253** |
| 1 | canonical `0xA5A5` | 0.157868 | 0.030412 | 0.905469 | 0.814659 |
| 1 | `0x3C3C` | **0.144818** | **0.028192** | **0.907579** | **0.821107** |
| 2 | canonical `0xA5A5` | 0.154955 | 0.029858 | 0.904893 | **0.816681** |
| 2 | `0x3C3C` | **0.150322** | **0.028967** | **0.907771** | 0.816118 |

`0x3C3C` improves KLD, JSD, and top-1 in all three seeds. Its seed-2 top-5
movement is -0.056 percentage points, which is noise-sized and does not agree
with the other strong signals. This candidate advances to broader held-out and
task gates but remains non-serializable and default-off.

A second held-out smoke gate used the local mixed-calibration parquet (128 rows
for Hessian capture and disjoint rows `[128,256)` for evaluation) on Llama-3.2-1B,
quantizing all 14 projections in two layers. The result again favored the
research mapping, but top-5 was effectively unchanged:

| Mapping | KLD | JSD | Top-1 | Top-5 overlap |
|:--|--:|--:|--:|--:|
| canonical `0xA5A5` | 0.177447 | 0.034735 | 0.904806 | 0.816223 |
| `0x5A5A` | 0.170102 | 0.033398 | 0.909147 | 0.814983 |
| `0x3C3C` | **0.147797** | **0.029684** | **0.912372** | 0.816124 |

This is additional research evidence, not a production promotion gate: it is a
single calibration/evaluation split, has no matched V2 control, and the
candidate remains non-serializable. The diagnostic loader now accepts local
message-based parquet calibration files without changing quantization math.

Opt-in `QVQQuantizationTelemetry` now records host-dispatch and CUDA-event time
without synchronizing each phase. Production lifecycle logging enables it only
with `GPTQMODEL_QVQ_TELEMETRY=1`. A warmed 2048x2048 W2/V4 Llama Q projection
on the local SM80 device showed:

| Phase | Warm GPU ms | Calls |
|:--|--:|--:|
| native tail-biting Viterbi | 1013.36 | 256 native recurrences |
| Block-LDL factorization | 17.91 | 1 |
| device-local trellis packing | 1.70 | 1 |
| total module | 1054.31 | 1 |

The previous GPU-to-CPU pack, CPU planar pack, and CPU-to-GPU copy took 325.51
ms on the same real trellis shape. Device-local tensor packing takes 1.66--1.70
ms after warmup (about 192x faster), is bit-exact across V4 W1--W4, and changes
no selected state or reconstructed weight. Native Viterbi now accounts for
about 96% of steady-state module time; the next optimization target is a fused
native provisional-plus-constrained tail-biting operator.

The first native V4 follow-up replaces four scalar target/codebook loads with
one aligned `float4` load while retaining the exact `x, y, z, w` FP32 FMA
sequence. CUDA allocations and contiguous `[batch, steps, 4]` / `[65536, 4]`
rows provide the required 16-byte alignment. A controlled source A/B rebuilt
the scalar and vectorized extensions in turn, then quantized the same real
Llama-3.2-1B-Instruct layer-0 2048x2048 `q_proj` at W2/V4 with an identity
Hessian. Medians exclude the first cold run and use the following 11 warm runs:

| Warm median | Scalar CUDA | V4 `float4` | Speedup | Reduction |
|:--|--:|--:|--:|--:|
| full module wall time | 1060.702 ms | 745.472 ms | 1.423x | 29.719% |
| native Viterbi GPU time | 1018.324 ms | 702.677 ms | 1.449x | 30.997% |

All 12 runs within each arm selected identical trellises. Native CUDA versus
eager-reference path parity passes W1, W1.5, W2, W2.5, W3, W3.5, and W4 at 1,
3, and 17 steps, plus weighted/overlap and W4 memoryless edge cases (31 tests).
The half-step cases previously stopped before launch because their deterministic
test seed was a float; the seed is now normalized to an integer so those rates
exercise the actual native kernel under Python 3.14.6 free-threading.

Production PGC16 levels are frozen FP16 bit patterns. Keeping the immutable
CUDA Viterbi codebook in FP16 and promoting each `half2` lane to FP32 in registers
therefore changes no value or arithmetic, while halving codebook load traffic.
Arbitrary research codebooks retain the FP32 path. The same 11-run warm-median
method produced:

| Warm median | Scalar FP32 | `float4` FP32 | `half2` FP16 | FP16 vs scalar |
|:--|--:|--:|--:|--:|
| full module wall time | 1060.702 ms | 745.472 ms | 632.858 ms | 1.676x |
| native Viterbi GPU time | 1018.324 ms | 702.677 ms | 591.533 ms | 1.721x |

Both FP16 and FP32 codebook-storage paths pass the full W1--W4 half-step CUDA
parity matrix, including weighted overlap and W4 memoryless cases (47 tests).
The shared V2 implementation also passes an FP16-storage path gate at every
W1--W8 half-step rate (15 additional tests).
Computing the codebook norm again from the four promoted values was also tested:
it remained exact, but regressed the warm module median to about 755 ms because
four extra FP32 square/add pairs cost more than the cached norm load. That
experiment was rejected and the cached FP32 norm table remains in production.

An Nsight Compute run on the W2/V4 `half2` recurrence reported 99.98% L2 hit,
0.02% DRAM throughput, 50.77% achieved occupancy, and 55.29% of issue cycles
stalled on an L1TEX scoreboard dependency. The 128-sequence Llama projection
provides only one 1024-thread block per local SM even though two can reside.
Direct recurrence measurements using actual DeepSeek-V4-Flash-0731 projection
shapes confirm that wider independent-sequence batches improve utilization:

| DeepSeek projection shape | Sequences | Median | Throughput |
|:--|--:|--:|--:|
| expert gate/up, `[2048,4096]` | 128 | 2.032 ms | 4.032 M vectors/s |
| expert down, `[4096,2048]` | 256 | 3.124 ms | 5.245 M vectors/s |
| dense `q_b`, `[32768,1024]` | 2048 | 18.172 ms | 7.213 M vectors/s |

This makes grouped, same-device expert dispatch the next MoE optimization:
combine independent experts inside the one ThreadX CUDA-device owner while
retaining each expert's Hessian, Block-LDL feedback order, and outputs. It must
not introduce another host thread for the device or reorder work within an
expert.

The production table lifetime is now shared at the Python quantizer boundary:
canonical PGC16 tables are cached by `(device, vector_size, codebook version,
dtype)`. Repeated expert modules therefore pass one stable table pointer into
the native norm cache, which retains one FP16 table plus one FP32 norm vector
per device/format instead of one pair per module. A CUDA test verifies pointer
identity for both V2 and V4 tables; experimental codebooks remain uncached.
On a cold real 2048x2048 W2/V4 quantization, canonical table construction took
47.095 ms; the second and third module-equivalent runs measured 0.084 ms and
0.063 ms for that phase, with exact trellis reuse and 626.204/627.688 ms total
wall time versus 1030.550 ms cold.

Output alignment keeps the trellis and RHT `SU` transform fixed. Only the
output-side `SV` scales and the intentionally supported later-dense sibling
compensation remain trainable during temporary replay; `SU` is never silently
converted into a learned dense transform.

## CUDA V4 banked lifecycle gate (ecf396e9)

On the SM80 PG506-230/232 cards, the native V4 Viterbi/GEMV path passed the
CUDA reference parity suite (639 tests before the loader regression fix, plus
the focused selector test afterward). A real Llama-3.2-1B two-layer,
128-calibration-row sweep used `PYTHON_GIL=0`, FP16 inference, four banks, and
one GPU per arm:

| Rate | Quant s | Reload s | Warm forward s | Final-logit KLD | Top-1 | Top-5 |
|:--|--:|--:|--:|--:|--:|--:|
| W1.5 | 107.31 | 4.55 | 0.0781 | 0.10697 | 0.8919 | 0.7784 |
| W2 | 151.24 | 4.65 | 0.0636 | 0.03093 | 0.9459 | 0.8919 |
| W2.5 | 158.26 | 4.42 | 0.0645 | 0.02995 | 0.9189 | 0.9243 |
| W3 | 88.03 | 4.26 | 0.0576 | 0.00917 | 0.9730 | 0.9243 |

The banked checkpoint saved and reloaded successfully through Accelerate after
deferring selector value checks for meta shells. Native CUDA inference produced
finite logits from all 14 banked modules. These are two-layer promotion probes,
not full-model quality evidence. `bank_count=4` remains explicit opt-in; once
selected with Block-LDLQ, module-boundary propagation is automatic. The
deferred full-model refiner remains a separate opt-in gate.
