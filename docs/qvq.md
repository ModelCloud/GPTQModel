# QVQ (QTIP-derived) integration design

Status as of 2026-08-12: **QVQ** names this repository's QTIP-derived quantizer plus the planar PGC16 codec, half-step
W1, W1.5, ..., W8 coverage, and optimized inference backends. Production quantization, configuration, checkpoint
loading, and Torch/MPS/MLX/CUDA inference accept only `pgc16-v1` with fixed Gaussian levels. Learned `pgc16-v2` is
retired because repeated held-out tests showed that its lower local reconstruction proxy did not reliably improve
final model-output KLD. Its design/results remain in this document, and its research math lives only under
`qvq_codecs/deprecated`; it cannot be selected by a production model loader. Configuration and `QVQLinear` reject legacy codebook names and serialized
codebook tensors. HYB-Q9 and uniform product decoders remain explicit regression, benchmark, and debugging references;
neither is a loadable checkpoint format. YAQA v3 two-sided rounding, Sketch-B reference math, and a full-model
real-Fisher collector and guarded dedicated processor lifecycle are implemented. Dense-model replacement and
save/load/reload lifecycle tests pass; broader model-quality promotion gates remain pending.

The executable CUDA/model-quality/lifecycle handoff is tracked in [qvq_todos.md](qvq_todos.md). Slow Apple-host
model runs are stopped; the interim model-quality gate is now two decoder layers on a CUDA quantization host.

This document is the design record. Any change to the selected format, decoder, quantizer, kernel strategy, or measured
acceptance evidence must update the decision log and implementation-status table below in the same change.

## Decision summary

1. Keep the QTIP bitshift state-transition graph at `L=16`, `V=2`, with half-step rates W1 through W8.
2. Store one whole transition label for every two weights in a Pangolin-style planar stream. The integer transition
   width is `E = 2 * rate`, so W1 through W8 map exactly to `E=2` through `E=16`. Backward compatibility with earlier
   scalar-split QTIP artifacts is not a goal, and legacy streams must fail closed.
3. Replace the current 512-entry two-dimensional HYB-Q9 LUT with PGC16: a bijective 16-bit state mixer followed by two
   lookups into one canonical 256-entry scalar Gaussian-compander table.
4. Freeze the mixer constants and fixed Gaussian FP16 table as the only production codec, `pgc16-v1`. The historical
   `pgc16-v2` experiment kept the mixer but learned 256 model-level FP16 values; it is now research-only.
5. Do not serialize a codebook per module. The canonical table is process-wide or compiled into a kernel; a fitted
   module amplitude is folded into the existing transform scale, not a new checkpoint tensor.
6. Decode adjacent output values as a pair for M=1--3. At M>=4, decode into threadgroup memory once and reuse the
   values across row tiles. MPS normally emits four adjacent columns and uses eight only for measured large W8/M15--16
   projections. MLX retains pair, N4, and N8 kernels and selects among them by measured rate, row, and projection shape.
7. Keep `L=16`, `V=2` for low rates. Improve offline search with a non-regressing tail-biting overlap list and use
   YAQA v3's two-sided input/output Hessian feedback when a valid full-model Hessian sketch is available. Neither
   control changes checkpoint bytes or inference kernels.
8. Keep `hyb_reference`, `uniform_reference`, and learned-v2 implementations for research, but expose only PGC16-v1 through production
   configuration, protocol compilation, checkpoint loading, and the codec package's default exports.
9. Do not claim broad dequantization superiority from one machine. All three declared Apple matrix shapes pass, and
   the separate W8 native-int8 comparison remains visible.

### Plain-language terminology

“Trellis” is the coding-theory name for the graph of legal state changes over time. In user-facing QVQ text, prefer
the less technical terms below:

```text
state-transition graph (the trellis: every legal route)
    |
    +-- search chooses one state path (the winning route)
            |
            +-- checkpoint stores path codes (the turn taken at each step)
```

The 16-bit `state` is the rolling history needed to interpret those turns. The checkpoint tensor keeps only the path
codes; it does not serialize the graph or all visited candidate states. The Python tensor remains named `trellis` for
the established internal API, but documentation should call its contents “planar path codes” or a “transition stream.”

## Is PGC16 better than the current LUT?

PGC16 is decisively better for W2--W8 capacity and metadata memory, and it has lower distortion in the current
synthetic Gaussian experiment at every rate. Its paired MPS/MLX kernels pass the latency gate at the first declared
representative shape; that is evidence for this implementation and shape, not a universal compute claim.

| Property | Current HYB-Q9 | Selected PGC16 | Conclusion |
|---|---:|---:|---|
| Distinct two-value vectors | At most 1,024 | Exactly 65,536 | PGC16 supports the full W8 state space |
| Effective codebook ceiling | 5 bits/weight at `V=2` | 8 bits/weight at `V=2` | PGC16 covers W1--W8 without a format split |
| Serialized FP16 codebook/module | 2,048 bytes | 0 bytes | PGC16 is smaller on disk |
| Runtime table | 2,048 bytes/module | 512 bytes shared, or kernel constants | PGC16 has the smaller cache footprint |
| Planar stream | Exactly `rate / 8` bytes/weight | Exactly `rate / 8` bytes/weight | Equal; transform metadata is common |
| Scalar decoder | Hash, vector lookup, conditional sign | Two xor-shifts, MAD, scalar lookup | PGC16 is branch-free but has extra ALU |
| Pair decoder | Retained scalar reference | Rate-specialized pair decode plus tiled row reuse | Implemented in MPS and MLX |
| Measured dequant/GEMV speed | Reference baseline | 0.014--0.582x HYB on MPS and 0.014--0.520x on MLX, W2--W5 | All three declared shapes pass |
| Synthetic Gaussian distortion | Saturates after W5 | Decreases through W8 | PGC16 is lower error in this proxy |
| Four-layer KLD/top-1 | Not yet a valid QVQ endpoint | Not measured | Model-quality conclusion is still pending |

The smaller table and pair decoder are favorable for MPS/MLX cache behavior, but the codebook change alone does not
guarantee a compute win. HYB computes `s*s+s`, selects a two-value LUT entry, and conditionally flips the second sign.
PGC16 performs two xor-shifts and one multiply-add before selecting two scalar entries. The production kernels recover
one state and accumulate both adjacent outputs together. The retained HYB reference remains deliberately scalar: it is
the exact pre-PGC behavior used as a regression oracle and historical benchmark baseline, not a production competitor
that will receive new format or kernel work.

## Why the current HYB LUT saturates at W5

For a vector size of two, a true `b`-bit/weight code needs at least `2 ** (2*b)` distinct reconstruction vectors:

| Rate | Required vectors |
|---:|---:|
| W2 | 16 |
| W3 | 64 |
| W4 | 256 |
| W5 | 1,024 |
| W6 | 4,096 |
| W7 | 16,384 |
| W8 | 65,536 |

HYB-Q9 selects one of 512 two-dimensional entries and adds one sign choice, so it exposes at most 1,024 distinct
vectors. That is exactly 10 bits/vector, or 5 bits/weight for `V=2`. Increasing the trellis rate after W5 adds path
choices but no new reconstruction values, so distortion reaches a codebook floor. Enlarging the existing
`x * (x + 1)` hash is also insufficient: its low bit is always zero, so it cannot become a bijection over all 16-bit
states merely by selecting more output bits.

## PGC16 specification

PGC16 means **permuted Gaussian compander with a 16-bit state**. It is a GPT-QModel extension to the QTIP decoder, not
a codebook described by the QTIP paper. Given trellis state `s` as an unsigned 16-bit integer:

```text
p = s ^ (s >> 8)
p = (p * 40503 + 17011) mod 65536
p = p ^ (p >> 7)
value_0 = G[p >> 8]
value_1 = G[p & 255]
```

`G` is a strictly increasing 256-entry FP16 scalar-compander table. In `pgc16-v1`, it is the canonical symmetric
Gaussian table. In `pgc16-v2`, it is a learned model-level table frozen to exact FP16 bit patterns. Both xor-shifts are
invertible, and `40503` is odd, so multiplication modulo `2**16` is invertible. Their composition is therefore a
permutation of all 65,536 states. Each byte selects one of 256 distinct scalar levels, giving exactly
`256 * 256 = 65,536` distinct two-value vectors for either version.

The mixer is required. Mapping the high and low state bytes directly to the two scalar levels is capacity-correct but
orders nearby trellis states along highly correlated code vectors; the measured adjacent-vector cosine was `0.992753`
and W2 proxy MSE rose to `0.50748074`. The selected xor-MAD-xor mixer retained all 65,536 vectors and reduced the W2
screening MSE to approximately `0.06844`. A one-MAD affine permutation reached approximately `0.07813`, so the two
xor diffusion steps earn their small decode cost.

The codebook version owns all of the following and changes if any one changes:

- trellis window and vector size (`L16`, `V2`);
- mixer operation order, integer widths, and constants;
- the exact 256 FP16 level bit patterns and level ordering;
- planar symbol order and tile layout;
- scale normalization applied during quantization and reconstruction.

The versioned Gaussian scale multipliers for W1, W1.5, ..., W8 are
`1.0, 1.0, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.4, 1.5, 1.5, 1.5, 1.5, 1.5`.
The multiplier is folded into `SV`; it does not add a tensor or byte to the checkpoint. W1 through W2 retain the
control scale, while higher rates receive more range so their additional reconstruction levels are not wasted by
tail clipping.
Both table versions use the exact canonical-v1 FP32 codebook RMS (`0.9975093603134155`) as their normalization
reference. A learned table changes only `G`; it must not silently change the coordinate system used to fit samples,
select states, or fold the module scale into `SV`.

### W1 and half-step rates

QVQ uses the same `L16/V2` state machine and decoder at every supported rate. The public rate `R` is an integer or
half-integer from 1 through 8. One transition reconstructs two weights and appends the exact integer width `E=2R` to
the 16-bit state:

```text
state_next = ((state << E) & 0xffff) | path_code
```

The serialized `trellis` tensor stores these winning path codes, not the 65,536-node search graph and not a sequence
of complete states. A 16x16 tile contains 128 codes. Packing each code in `E` low-to-high bit planes therefore uses
`128 * E / 32 = 4E = 8R` int32 words and exactly `R/8` payload bytes per weight. W1 uses `E=2` and eight words per
tile; W1.5 uses `E=3` and twelve; W2.5 uses `E=5` and twenty; W7.5 uses `E=15` and sixty. The internal integer `E`
keeps packing and kernel dispatch exact; the checkpoint configuration stores the JSON-safe rate value. No new scale,
codebook, or runtime tensor is required, and `gptq_p` remains restricted to integer W2--W8.

CUDA W1 uses four predecessor prefixes, 16,384 suffixes, 64 KiB of opt-in dynamic shared memory, and a conservative
16-tile Viterbi batch.

The production CUDA gate covers W1 at M=1/2/4/8/16/32, FP16/BF16, random and structured streams, extreme finite
inputs, deterministic and non-default-stream launches, dense reconstruction, KLD, top-1/top-5, and concurrent
free-threaded calls on two physical GPUs. The full results, including real DeepSeek V4 Flash projection shapes and
two-layer Llama held-out W1 v1/v2/EXL3 quality, are recorded in `docs/qvq_cuda.md` and `docs/qvq_todos.md`. Fixed v1
stays the default: learned v2 and EXL3 reduce W1 weight relative L2 but materially regress held-out forward KLD.

### L16/V1 W1 research branch

Keep this design and its negative first result available for later reconsideration. It is not a production codec and
must not be accepted by checkpoint loading. The motivation was to spend the same exact W1 payload on one scalar at a
time instead of reconstructing a pair from each state:

```text
Current L16/V2 W1                    Proposed L16/V1 W1

16-bit predecessor                  16-bit predecessor
        |                                    |
        | append 2 bits                      | append 1 bit
        v                                    v
4 successor states                  2 successor states
        |                                    |
        | PGC16 -> two values                | low bit selects sign
        v                                    | high 15 bits select a(context)
2 reconstructed weights                     v
                                     one value: -a(context) or +a(context)

128 transitions * 2 bits            256 transitions * 1 bit
= 256 bits per 16x16 tile            = 256 bits per 16x16 tile
= exactly 1.0 bit/weight             = exactly 1.0 bit/weight
```

The research decoder maps the high 15 state bits through a fixed implicit permutation into a shared positive FP16
amplitude table. The low state bit is the sign. For a one-bit transition, both successors therefore form the exact
centered pair `{-a(context), +a(context)}` while retaining 15 state-history bits. This was attractive because the
current W1 `L16/V2` graph retains only 14 history bits and its four local two-dimensional successors have substantial
centroid and anisotropy error. The scalar formulation guarantees local centering without increasing payload size.

W1.5 uses the same scalar decoder with the periodic transition schedule `1, 2, 1, 2, ...`. A 16x16 tile contains 128
one-bit and 128 two-bit transitions, so it stores exactly 384 bits, 48 bytes/tile, or 1.5 bits/weight. On each two-bit
step the four successors provide two centered magnitude pairs. No fractional bit metadata or padding is required.
The historical prototype tested shared 1,024- and 4,096-entry amplitude tables (2 KiB and 8 KiB respectively). Their levels are
fitted from disjoint calibration modules and frozen to FP16. The variable-transition Viterbi recurrence is checked
against exhaustive circular-path search on a small state graph before any accuracy measurements run.

#### First post-quant accuracy screen

The 2026-08-12 P-core-only screen uses 12 disjoint held-out synthetic modules, each 64x64, with 1,024 calibration
activation rows and 4,096 held-out rows. It preserves the production randomized Hadamard transform, damping, and
BlockLDLQ feedback. Four separate calibration modules supply 16 sampled tiles for the shared tables; PGC16-v2 runs
three accepted Hessian-weighted Lloyd iterations. The activation distribution contains correlated columns, lognormal
channel scales, heterogeneous weight rows, and deterministic weight-column outliers. These results are a fast module
proxy, not a substitute for the two-layer Llama gate.

```text
+------+----------------+----------+----------------+---------+----------+----------+--------+--------+
| Rate | Arm            | W rel-L2 | Out rel-L2     | SQNR dB | KLD      | JSD      | Top-1  | Top-5  |
+------+----------------+----------+----------------+---------+----------+----------+--------+--------+
| W1   | PGC16-v1       | 0.604621 | 0.448541       |   6.988 | 0.170831 | 0.041691 | 67.90% | 68.37% |
| W1   | PGC16-v2       | 0.593486 | 0.444448       |   7.067 | 0.174790 | 0.042695 | 67.92% | 68.83% |
| W1   | L16/V1, 1,024 | 0.620673 | 0.463639       |   6.700 | 0.187563 | 0.045678 | 66.01% | 67.47% |
| W1   | L16/V1, 4,096 | 0.620963 | 0.464148       |   6.691 | 0.187696 | 0.045701 | 65.96% | 67.33% |
| W1.5 | PGC16-v1       | 0.432058 | 0.310516       |  10.207 | 0.081047 | 0.020141 | 77.68% | 78.31% |
| W1.5 | PGC16-v2       | 0.432880 | 0.309096       |  10.242 | 0.081097 | 0.020175 | 77.60% | 78.39% |
| W1.5 | L16/V1, 1,024 | 0.448911 | 0.323502       |   9.854 | 0.089529 | 0.022222 | 76.94% | 77.18% |
| W1.5 | L16/V1, 4,096 | 0.449085 | 0.323575       |   9.852 | 0.089788 | 0.022274 | 76.87% | 77.16% |
+------+----------------+----------+----------------+---------+----------+----------+--------+--------+
```

Against fixed PGC16-v1, the 1,024-level scalar arm regresses W1 KLD by `+0.016732` (`+9.79%`), top-1 by `-1.89`
percentage points, and top-5 by `-0.90` points. At W1.5 it regresses KLD by `+0.008482` (`+10.47%`), top-1 by
`-0.74` points, and top-5 by `-1.13` points. Expanding the table from 1,024 to 4,096 levels does not recover quality;
it is slightly worse in both KLD and top-k agreement. PGC16-v2 illustrates why weight error is not the promotion
metric at W1: it lowers weight relative-L2, but its held-out KLD is worse than fixed v1.

The likely first-design weakness is local controllability rather than table capacity. At W1 the newly appended bit
chooses only the sign; the amplitude is inherited from the predecessor context. The path can plan future amplitudes,
but it cannot choose both sign and magnitude for the current scalar. A four-times-larger amplitude table does not add
successors and therefore cannot fix this limitation. The current table fit is distribution-aware but not yet an
assignment-aware scalar Lloyd/PAVA loop, and the implicit context permutation has not been optimized for transition
geometry. Those are the first two changes to test before abandoning the general `L16/V1` idea. A polar/centered
`L16/V2` W1 relabeling remains the fallback if scalar local controllability proves fundamental.

#### Inference and checkpoint VRAM

The payload is exactly unchanged: 32 bytes/tile at W1 and 48 bytes/tile at W1.5. `SU` and `SV` remain FP32 and have
the same shapes in every arm, so they add the same `4 * (in_features + out_features)` bytes per module. Excluding a
common bias, the effective inference rate for a module is:

```text
effective bpw = rate
              + 32 * (in_features + out_features) / (in_features * out_features)
              + 8 * shared_table_bytes / model_quantized_weight_count
```

The current v1 table is implicit in the checkpoint; v2 carries 512 logical bytes of frozen table metadata. Both
prepare a 512-byte FP16 runtime table. The scalar candidates require one model-shared 2 KiB or 8 KiB runtime table.
For a one-billion-quantized-weight normalization, payload plus shared runtime table is:

```text
+------+----------------+---------------+-------------+---------------+
| Rate | Arm            | Payload bytes | Table bytes | Effective bpw |
+------+----------------+---------------+-------------+---------------+
| W1   | PGC16-v1/v2    |   125,000,000 |         512 |   1.000004096 |
| W1   | L16/V1, 1,024 |   125,000,000 |       2,048 |   1.000016384 |
| W1   | L16/V1, 4,096 |   125,000,000 |       8,192 |   1.000065536 |
| W1.5 | PGC16-v1/v2    |   187,500,000 |         512 |   1.500004096 |
| W1.5 | L16/V1, 1,024 |   187,500,000 |       2,048 |   1.500016384 |
| W1.5 | L16/V1, 4,096 |   187,500,000 |       8,192 |   1.500065536 |
+------+----------------+---------------+-------------+---------------+
```

Thus the 1,024-level option costs only 1,536 additional runtime bytes per model over PGC16; the 4,096-level option
costs 7,680 additional bytes. An optimized implicit scalar decoder would not materialize its 65,536-state codebook.
The readable FP32 reference materializes 256 KiB versus 512 KiB for the readable PGC16 pair codebook, but neither
temporary belongs in an optimized inference kernel or checkpoint.

#### Decision, revisit gate, and rollback point

Do not promote this first `L16/V1` implementation: it fails the post-quant accuracy objective at both W1 and W1.5.
The benchmark implementation and its test-only imports were removed from the active tree because QVQ must not ship an
unsupported, accuracy-regressing quantizer. The exact prototype remains recoverable from historical commit
`e788eb6d`; the complete design, measurements, and failure analysis remain here. Production checkpoints stay on
`L16/V2` PGC16-v1. Quantization latency and backpointer workspace were recorded for engineering visibility but are
explicitly **not** acceptance gates at this stage. Accuracy recovery comes first.

Reopen the design after assignment-aware scalar Lloyd fitting and transition-label optimization. Promotion requires
a paired held-out KLD improvement whose 95% confidence interval is below zero versus fixed PGC16-v1, with no paired
top-1 or top-5 regression, followed by the same result in the two-layer Llama W1/W1.5 diagnostic. Weight relative-L2,
output relative-L2, SQNR, and JSD remain required supporting metrics; lower weight error alone is insufficient. The
rollback point is clean: the research codec has no production config value, loader path, packed checkpoint, or kernel
dispatch, so retaining PGC16-v1 requires no checkpoint migration.

Two exact-payload `L16/V2` centered successor codecs were also screened after the scalar result. The first mapped the
low two state bits to four sign quadrants and the high 14 bits through a bijective context mixer to two positive
magnitudes; every W1 successor quartet was an exactly centered rectangle. The second used two independently mixed
antipodal magnitude pairs, preserving exact zero centroid while allowing the two sign diagonals different magnitudes.
Both retained 65,536 unique vectors, two lookups, a 256-byte positive table, and the existing 1.0/1.5-bpw stream. Both
regressed the small held-out screen. Exact local centering was insufficient because it constrained magnitude choice
inside each quartet more than PGC16's uncentered but jointly sign/magnitude-varying successors. Do not revisit pure
centering without adding a demonstrated local magnitude-control mechanism.

The existing per-output `SV` scale optimizer remains an experimental, explicitly default-disabled direction. At
W1/W1.5 its opt-in path shrinks each damped full-input-Hessian closed-form correction halfway toward one before
original-Hessian acceptance. This changes only values already stored in `SV`; it adds zero payload bytes, runtime
tensors, lookups, branches, or inference FLOPs. The two-layer 128-prompt gate showed why local acceptance is
insufficient: local/live/layer errors improved, but every output-scale arm regressed final KLD and JSD, with W1 also
regressing top-1 significantly. Higher W3--W8 rates remain a separate research hypothesis because past higher-rate
results have transferred local improvements more reliably, but no such rate may be enabled without its own held-out
model-output gate.

Fixed-trellis output alignment is enabled by default for QVQ and is explicitly disabled with
`output_alignment=None`. Its optimizer is configurable as `adam` or `adamw`, with `optimizer="adam"` and
`weight_decay=0.0` as the defaults. These defaults match the public QTIP/YAQA recovery implementations, which
construct Adam without decoupled weight decay; AdamW and nonzero decay are opt-in tuning controls for calibration
experiments and do not change the serialized payload or inference kernel. Promotion was based on real Llama 3.2 1B
four-layer Q/K/V/O evaluation: two further-disjoint holdouts reduced final KL/JSD by about 13--15% and increased
Top-5/Top-10 by about 0.5 points, while Top-1 moved down only 0.03--0.04 points, within noise.

YAQA can also use a separate calibration stream through `GPTQModel.quantize(..., yaqa_calibration=...)`. This
stream is prepared independently and is used only for the full-model Fisher/Sketch-B factors. The normal
`calibration` argument continues to drive activation-Hessian capture and quantization replay. When
`yaqa_calibration` is omitted, it defaults to the prepared ordinary calibration stream.

For the Llama-3.2-1B YAQA row-count sweep started on 2026-08-13, strict runs reserve ordinary quantization rows
`[0,512)` and use disjoint YAQA learning slices: YAQA512 uses `[512,1024)`, YAQA1024 uses `[512,1536)`, and
YAQA2048 uses `[512,2560)`. Earlier completed snapshots are overlap controls, not strict-split results:

- W2.5/W3/W3.5 YAQA512 reused the ordinary `[0,512)` stream because no separate `yaqa_calibration` was supplied.
- W2.5/W3 YAQA1024 used `[0,1024)` for YAQA learning, overlapping all 512 ordinary quantization rows.
- W2 YAQA512/1024/2048 and W2.5 YAQA2048 had already started with row zero when the strict convention was adopted;
  retain and label their results as overlap controls.

All YAQA quants that had not started when this convention was adopted use the strict disjoint slices above. Do not
pool overlap-control and strict-split results into one row-count trend: their calibration populations differ.

### Calibration-data scaling policy

QVQ does not have one generic "calibration data" knob. Ordinary BlockLDLQ and YAQA estimate different geometry, so
their data populations and budgets must be controlled independently. The default policy is **more distinct,
representative data until held-out convergence**, not either the smallest possible set or the largest available set.
Raw row count is not a promotion criterion.

For ordinary QVQ, concatenate the mask-selected module inputs as `X[tokens, in]`. The input Hessian estimate is

```text
H_X = (2 / N) * X.T @ X.
```

The factor `2 / N` makes this a normalized token-weighted second moment; damping also scales with its mean diagonal.
Adding rows therefore does not merely make the objective numerically stronger--it changes the estimated covariance
geometry. Under matched independent sampling, more valid tokens normally reduce sampling variance, improve rank, and
cover rare activation directions. Exact duplication adds no new geometry and only reweights the duplicated
population.

More rows can still reduce held-out quality when they change the domain or context-length mixture, dilute rare but
task-important directions, or estimate a local quadratic proxy that is poorly aligned with downstream behavior. The
trellis assignment is discrete, so a small covariance change can also switch paths discontinuously. This is more
precisely **distribution/proxy mismatch plus discrete decision instability**, not classical finite-sample overfitting:
if new samples are independent draws from the actual target distribution, the covariance estimator itself becomes
less overfit as `N` grows. QVQ's randomized transform and fixed PGC16 reconstruction manifold reduce dependence on a
learned scalar grid, but they do not remove dependence on the calibration distribution.

The existing matched two-layer BlockLDLQ control demonstrates the non-monotonic result:

| Rate | Rows | Final KLD | JSD | Top-1 | Top-5 |
|---:|---:|---:|---:|---:|---:|
| W1 | 512 | 0.465712 | 0.094010 | 0.7023 | 0.6934 |
| W1 | 1,024 | 0.454664 | 0.095325 | 0.6913 | 0.6890 |
| W1.5 | 512 | 0.204621 | 0.043625 | 0.7895 | 0.7945 |
| W1.5 | 1,024 | 0.177956 | 0.040593 | 0.7880 | 0.8015 |

At W1, doubling rows improved KLD by 2.37% while JSD, top-1, and top-5 regressed. At W1.5, it improved KLD by
13.03%, JSD by 6.95%, and top-5 by 0.70 percentage points, while top-1 fell by 0.15 points. More ordinary Hessian
data is therefore neither uniformly beneficial nor uniformly harmful.

YAQA is a separate and generally more data-hungry estimator. For each independent sequence `s`, Sketch-B forms one
sampled full-model score-gradient matrix `G_s[out, in]` and estimates

```text
H_I = sum_s(G_s.T @ G_s) / (S * out_features)
H_O = sum_s(G_s @ G_s.T) / (S * in_features).
```

The current collector uses one Monte Carlo categorical sample per valid output position and sums token scores within
each sequence. More independent sequences reduce both data-sampling and Monte Carlo variance and should stabilize the
normalized Kronecker geometry. They do not make YAQA exact: a single `H_O ⊗ H_I` remains a misspecified
approximation to a full-model Hessian block, cross-module terms remain absent, long or domain-heavy sequences can
dominate the sequence-gradient Gram, and small factor changes can select different trellis paths. Repeating the same
text with a new Monte Carlo draw can reduce conditional sampling variance, but distinct representative sequences add
the coverage YAQA needs.

In the current lifecycle, `rounding="yaqa"` passes the YAQA input and output factors to the quantization core instead
of the ordinary activation Hessian. The ordinary calibration stream still drives input capture, sequential replay,
and any separately enabled lifecycle features. Consequently, increasing ordinary `calibration` rows while holding
the YAQA factors fixed is not an increase in YAQA Hessian evidence. `YaqaConfig.minimum_sequences` is only a
fail-closed lower-bound check; the actual treatment is the population supplied through `yaqa_calibration`. Report the
observed `independent_sequences` and valid-output-token count, not only the configured minimum.

The completed Llama-3.2-1B overlap controls held ordinary calibration at `[0,512)` and expanded only the nested YAQA
population from `[0,512)` to `[0,1024)`. Their changes are mixed:

| Rate | ARC acc | ARC norm | GSM8K | STEM | History | Macro dense-relative score |
|---:|---:|---:|---:|---:|---:|---:|
| W2.5, YAQA1024 - YAQA512 | +0.512 pp | -0.256 pp | +2.564 pp | +0.190 pp | -2.688 pp | +0.066 pp |
| W3, YAQA1024 - YAQA512 | -0.341 pp | +0.256 pp | +1.406 pp | -0.159 pp | -0.645 pp | +0.531 pp |

The five task columns are raw benchmark accuracy-point changes. The macro column is the change in the unweighted
dense-relative score over ARC norm, GSM8K, STEM, and History; ARC acc is not counted twice. Thus 512 to 1,024 YAQA
sequences was essentially flat in aggregate at W2.5 and modestly positive at W3, but neither result was task-uniform.
These are overlapping development controls, not independent-data evidence. The incomplete W3.5 YAQA512-overlap and
YAQA1024-strict arms cannot measure a row-count effect because both sample count and population provenance changed.

Use this two-axis protocol before changing a default:

1. Sweep ordinary BlockLDLQ Hessian data with YAQA disabled or its factors frozen. Use nested, matched valid-token
   budgets and hold domain and context-length composition fixed.
2. Hold ordinary calibration at `[0,512)` and sweep YAQA independently. First compare matched 512-sequence overlap
   and disjoint arms to isolate overlap; then use nested strict-disjoint slices `[512,1024)`, `[512,1536)`, and
   `[512,2560)` for 512, 1,024, and 2,048 YAQA sequences.
3. Repeat YAQA Monte Carlo seeds. Record row/token hashes, independent sequences, valid tokens, and the context-length
   and domain distributions. A natural short row is not evidence-equivalent to a fixed 2,048-token paper sequence.
4. Compare normalized Kronecker products, eigenspaces, and held-out quadratic proxies rather than raw factor norms:
   individual factor scales are not identifiable because reciprocal rescaling leaves `H_O ⊗ H_I` unchanged.
5. Record trellis-assignment change rates, propagated next-module/layer/final-logit metrics, and paired task flips.
   Use paired-bootstrap intervals for raw score deltas and an exact McNemar test from the discordant counts.

Choose the smallest population after which the held-out geometry and propagated behavior are stable across seeds.
Favor additional diversity over redundant rows, and never select a calibration size from training-proxy or local KLD
alone. The smallest 2,000-sequence YAQA paper ablation is a useful reference point, not a universal minimum or proof
that the next sequence improves QVQ.

### W2 cliff, W2.5 instability, and the 64-successor constraint

The latest results show two related effects: a real low-rate transition-geometry threshold and a larger nonlinear
task threshold. Do not describe the entire W2 task cliff as a packing, decoder, or local rate-distortion defect. On
the matched two-layer diagnostics, the raw error curve is close to the expected Gaussian high-rate law

```text
D(R) proportional to 2**(-2R).
```

Every additional half bit should therefore halve error energy, or reduce RMSE by approximately `1/sqrt(2)`. The
reported QVQ-v1 weight relative-L2 values `0.336747` at W2, `0.241287` at W2.5, and `0.173289` at W3 are consistent
with that smooth progression: their squared successive ratios are approximately `1.95` and `1.94`. Final KLD also
falls smoothly by factors of approximately `2.11` and `2.25`. These ratios combine same-protocol documented sweeps,
not one bit-identical artifact, so they are strong consistency evidence rather than an exact aggregate-MSE identity.

Task behavior is thresholded rather than smooth. In the matched YAQA overlap controls, W2 to W2.5 changes GSM8K by
`+17.70` raw accuracy points at 512 YAQA sequences and `+16.87` points at 1,024, while W2.5 to W3 changes it by only
`+5.46` and `+4.30` points. The W2 STEM/History rows are incomplete, so these are task-specific deltas, not complete
macro comparisons. The GSM8K changes correspond to hundreds of examples and are too large to dismiss as ordinary
aggregate noise.

For a dense answer margin `m` and quantization-induced margin error `delta_m`, correctness is the threshold event

```text
m + delta_m > 0.
```

W3 appears below most critical error margins, W2 beyond many of them, and W2.5 near the transition band. Around that
band, a small change in the direction of correlated error can flip many answers. Autoregressive exact-match reasoning
compounds the effect across critical token decisions. The encoder is also discontinuous:

```text
path*(theta) = argmin_path cost(path, theta).
```

Small changes in Fisher samples, Hessians, scale, randomized signs, or tail-boundary candidates can switch a nearly
tied path. Two paths can have almost equal local cost while their residuals point in very different downstream
directions. This is the leading explanation for the large W2.5 configuration swings. Confirm it by measuring
best-versus-second path gaps, state/path Hamming churn, propagated answer margins, and paired task flips across seeds.

There is also an exact structural threshold in the current `L16/V2` graph. A rate `R` transition appends `E=2R` bits
and therefore has exactly `2**E` successors from a fixed predecessor:

| Rate | Edge bits `E` | Successors/context | Retained state bits | Tail-overlap states |
|---:|---:|---:|---:|---:|
| W2 | 4 | 16 | 12 | 4,096 |
| W2.5 | 5 | 32 | 11 | 2,048 |
| W3 | 6 | 64 | 10 | 1,024 |

An exhaustive enumeration of the fixed PGC16-v1 local successor blocks, before applying the rate scale, gives:

| Rate | Contexts missing at least one sign quadrant | p95 centroid norm | p95 covariance condition |
|---:|---:|---:|---:|
| W2 | 594 / 4,096 (14.50%) | 0.7694 | 4.480 |
| W2.5 | 0 / 2,048 | 0.3763 | 2.162 |
| W3 | 0 / 1,024 | 0.1935 | 1.428 |

Thus W2 crosses a concrete local-coverage boundary: some contexts cannot immediately emit a pair in every sign
quadrant. W2.5 repairs sign-quadrant coverage but still has substantially more biased and anisotropic local fans than
W3. This is a strong causal hypothesis for low-rate sensitivity, not yet proof that successor geometry alone causes
the task cliff; weight-error energy and nonlinear propagation remain material.

#### Information limit for the current two-weight graph

For a deterministic decoder with a fixed predecessor state and an `E`-bit path code,

```text
number of independently selectable successors <= 2**E.
```

Consequently, `L16/V2` cannot expose 64 independent choices at W2 or W2.5 while remaining below W3. W2.5 needs one
additional bit per two-weight transition, which costs `0.5` bpw and becomes W3. W2 needs two additional bits per
transition, which costs `1.0` bpw and also becomes W3. A larger state window, more PGC levels, more tail-biting
candidates, or a learned compander cannot change this bound. Tail-candidate widening searches circular boundary
states; it does not change per-state out-degree.

A per-tile codec-bank selector can cheaply offer multiple alternative 16/32-way graphs, but after the tile's bank is
known each state still has only 16/32 outgoing edges. It must be reported as banked graph selection, not as 64
independent successors per transition. A nondeterministic graph with duplicate path labels would require expensive
sequence decoding at inference or would cease to be uniquely decodable; it is not a production escape from the bound.

#### Selected first prototype: low-rate `L16/V4`

Change the low-rate super-symbol from two weights to four while retaining the 16-bit state. Then `E=4R`:

| Rate | `L16/V4` edge bits | Direct successors | Retained state bits |
|---:|---:|---:|---:|
| W1 | 4 | 16 | 12 |
| W1.5 | 6 | 64 | 10 |
| W2 | 8 | 256 | 8 |
| W2.5 | 10 | 1,024 | 6 |

This satisfies a 64-outgoing-edge requirement **per four-weight transition** for W1.5--W2.5 without increasing raw
payload. W1 gains 16 joint successors versus four under `V2`, but cannot reach 64 at exactly 1 bpw with `V4`. The
layout does not provide 64 independently combinable choices for each constituent pair: at W2, 64 choices for each
pair would require `64 * 64 = 4,096` joint edges, while the unchanged-rate transition has only 256. The four-
dimensional codebook must distribute those 256 joint choices so that both pair marginals and all 16 four-dimensional
sign orthants are well covered, accepting that the pair choices are correlated.

A 16x16 tile has 64 four-weight transitions, so it still stores

```text
64 * (4R) = 256R bits = exactly R bits/weight,
```

the same `8R` planar int32 words as `L16/V2`. A four-value implicit decoder can use two PGC-style mixes and four
scalar-table lookups per transition. The current decoder uses one mix and two lookups per two weights, so mixer and
lookup counts per weight can remain unchanged. The 256-entry FP16 scalar table, `SU`, `SV`, and zero-serialized-
codebook contract also remain unchanged. Viterbi performs half as many transition steps; total emission-coordinate
work is similar, and traceback storage should remain in the same order. These are design expectations, not latency
claims; Torch, CUDA, MPS, and MLX must be benchmarked.

This topology does **not** create free information. Two ordinary `V2` transitions over four weights and one `V4`
transition have the same `2**(4R)` edge-sequence entropy. The intended gain is better joint four-dimensional shaping,
no forced 16/32-choice intermediate reconstruction boundary, a smaller tail-overlap space, and a low-rate codec whose
local orthant/radius geometry can be optimized directly. It may still regress if the new implicit four-vector
manifold or loss of the intermediate decode boundary is worse. `L16/V4` has only 65,536 global four-vectors, a
4-bpw codebook-capacity ceiling. Evaluate it at W1--W2.5, but retain `L16/V2` at W3 and above unless propagated
quality and kernel benchmarks demonstrate a reason to widen that dispatch.

The minimum same-payload construction at W2 is `L16/V3`: `E=2*3=6`, exactly 64 successors, and ten retained state
bits. At W2.5, `V3` needs an alternating seven/eight-bit schedule, giving 128/256 successors while averaging exactly
2.5 bpw. These are useful reference arms, but three-value vectors do not divide a 16x16 tile, and the alternating-
width planar stream, tile mapping, and kernels are more complicated. `L16/V4` is the first production prototype
because four values align with the 256-weight tile and every rate retains one fixed edge width. If `V4` regresses
because it consumes too much history at once, `V3` is the lower-fan-out, longer-memory fallback.

The first reference codec should derive two distinct bijective PGC permutations from the state and decode four
Gaussian levels. Because the first pair mapping is itself bijective, all 65,536 four-vectors remain unique. Search
the second permutation and pair ordering for worst-context four-dimensional orthant coverage, covering radius,
centroid, covariance condition, and magnitude diversity. Then select only on occupancy-weighted live paths and the
complete propagation gate--not on synthetic nearest-neighbor MSE alone.

#### Experimental retained-history arm: `L18/V4`

`format="qvq_v4_l18"` is a separate, fail-closed W1--W2.5 research format. It keeps the V4 transition width
`E=4R` and exact `R`-bpw planar payload, but expands the state from 16 to 18 bits. The low 16 bits remain the
canonical PGC16 state. The high two bits select one of the four existing rate-keyed V4 mixer masks:

```text
18-bit state
  |-- high 2 bits --> implicit reconstruction bank (0..3)
  `-- low 16 bits --> canonical PGC16 state
                         |
                         `--> four reconstructed scalars
```

The selector is therefore part of trellis history, not a serialized per-tile tensor. Raw and effective BPW are
unchanged relative to L16 at the same rate. The extra state increases retained history by exactly two bits while
leaving direct successor count unchanged:

| Rate | Edge bits | Successors | L16/V4 history | L18/V4 history |
|---:|---:|---:|---:|---:|
| W1 | 4 | 16 | 12 | 14 |
| W1.5 | 6 | 64 | 10 | 12 |
| W2 | 8 | 256 | 8 | 10 |
| W2.5 | 10 | 1,024 | 6 | 8 |

Exact Viterbi quantization has four times as many states and codebook rows (`262,144 x 4`), so its dominant state
costs and workspace should be budgeted at roughly 4x L16/V4 before backend-specific optimization. Inference does not
materialize that table: it reconstructs the 18-bit state from the planar stream, extracts the high two bits, and
performs the same two PGC mixes and four scalar lookups. Depending on `E`, reconstructing 18 rather than 16 history
bits needs zero or one additional edge extraction; measure latency rather than inferring it from the offline table.

The initial implementation provides Torch quantization/reference inference and a native MLX decoder. MPS and CUDA
fall back to the Torch reference for this format until dedicated L18 kernels pass parity. It rejects explicit
`bank_ids`, rates above W2.5, and any vector size other than four.

A deterministic 16x16 Block-LDLQ synchronization probe (128 Hessian rows, 1,024 held-out rows, seed 18240) found:

| Rate | Arm | Weight MSE | Output KL | Top-1 | Top-5 overlap |
|---:|:---|---:|---:|---:|---:|
| W1 | V2 | 0.279328 | 0.807800 | 54.39% | 72.64% |
| W1 | V4 | 0.290393 | 0.897001 | 53.32% | 71.99% |
| W1 | L18/V4 | 0.276493 | 0.954248 | 53.03% | 74.49% |
| W1.5 | V2 | 0.134872 | 0.400912 | 67.38% | 81.54% |
| W1.5 | V4 | 0.154858 | 0.555226 | 63.77% | 80.66% |
| W1.5 | L18/V4 | 0.144913 | 0.405387 | 67.09% | 80.12% |
| W2 | V2 | 0.067028 | 0.206508 | 75.10% | 85.80% |
| W2 | V4 | 0.089202 | 0.279160 | 74.61% | 85.02% |
| W2 | L18/V4 | 0.082399 | 0.242434 | 75.68% | 85.16% |
| W2.5 | V2 | 0.035471 | 0.109628 | 84.18% | 90.23% |
| W2.5 | V4 | 0.060060 | 0.170244 | 78.61% | 87.71% |
| W2.5 | L18/V4 | 0.048411 | 0.148707 | 81.05% | 88.79% |

This probe verifies synchronization and shows that retained history recovers part of L16/V4's loss, especially at
W2.5. It does **not** establish model-level recovery: L18/V4 remains worse than V2 on most local metrics, and the W1
case again demonstrates that lower weight MSE need not lower output KL. Promotion requires propagated two-layer and
full-model held-out gates.

#### Experimental factored arm: Dual-V2

`format="qvq_dual_v2"` preserves two independent L16/V2 chains inside each 16x16 tile. Four consecutive scalar
weights are decoded as one pair from chain A and one pair from chain B:

```text
planar pair edges:  A0 B0 A1 B1 ... A63 B63
                       |             |
                       v             v
                  L16/V2 chain A  L16/V2 chain B
                       |             |
                       `---- 4 reconstructed weights
```

The conceptual joint state has 32 bits, but the transition and emission objective factor into two 65,536-state
problems. The implementation therefore never constructs a `2**32` codebook or cost vector. Block-LDLQ and YAQA
still compute their corrected 16x16 tile first; only the tile's Euclidean trellis rounding is split into its even
and odd pair chains. Each chain is tail-bitten independently, and their states are interleaved before planar packing.

At rate `R`, each chain appends `2R` bits. The joint successor count is `2**(4R)`, while retained history is the sum
of the two independent chain histories, `2*(16-2R)`:

| Rate | Joint successors | Retained history, total | Retained history per chain |
|---:|---:|---:|---:|
| W1 | 16 | 28 | 14 |
| W1.5 | 64 | 26 | 13 |
| W2 | 256 | 24 | 12 |
| W2.5 | 1,024 | 22 | 11 |
| W3 | 4,096 | 20 | 10 |
| W3.5 | 16,384 | 18 | 9 |
| W4 | 65,536 | 16 | 8 |
| W4.5 | 262,144 | 14 | 7 |
| W5--W8 | `2**(4R)` | `32-4R` | `16-2R` |

This is a joint count across two factored pair decisions, not 1,024 mutually coupled four-dimensional reconstruction
vectors at W2.5. That distinction is precisely why exact search remains practical. Quantization performs the same
total 128 pair emissions per tile as V2, split over two 64-step paths; workspace per path remains L16-sized and the
two paths may be batched. Inference performs the same two PGC mixes and four scalar lookups per four weights as V2.
Its only topology change is that state reconstruction walks same-parity planar edges. Raw payload, SU/SV storage,
and effective BPW are unchanged.

The Torch quantizer/reference decoder and native MLX decoder support W1--W8. CUDA and MPS deliberately use the Torch
reference until dedicated same-parity state reconstruction passes backend parity.

Adding Dual-V2 to the same deterministic 16x16 synchronization probe gives:

| Rate | Arm | Weight MSE | Output KL | Top-1 | Top-5 overlap |
|---:|:---|---:|---:|---:|---:|
| W1 | V2 | 0.279328 | 0.807800 | 54.39% | 72.64% |
| W1 | Dual-V2 | 0.281140 | 0.936393 | 55.57% | 72.60% |
| W1 | L16/V4 | 0.290393 | 0.897001 | 53.32% | 71.99% |
| W1 | L18/V4 | 0.276493 | 0.954248 | 53.03% | 74.49% |
| W1.5 | V2 | 0.134872 | 0.400912 | 67.38% | 81.54% |
| W1.5 | Dual-V2 | 0.145535 | 0.393389 | 68.95% | 79.51% |
| W1.5 | L16/V4 | 0.154858 | 0.555226 | 63.77% | 80.66% |
| W1.5 | L18/V4 | 0.144913 | 0.405387 | 67.09% | 80.12% |
| W2 | V2 | 0.067028 | 0.206508 | 75.10% | 85.80% |
| W2 | Dual-V2 | 0.069705 | 0.234060 | 74.61% | 86.31% |
| W2 | L16/V4 | 0.089202 | 0.279160 | 74.61% | 85.02% |
| W2 | L18/V4 | 0.082399 | 0.242434 | 75.68% | 85.16% |
| W2.5 | V2 | 0.035471 | 0.109628 | 84.18% | 90.23% |
| W2.5 | Dual-V2 | 0.034796 | 0.101282 | 82.62% | 90.96% |
| W2.5 | L16/V4 | 0.060060 | 0.170244 | 78.61% | 87.71% |
| W2.5 | L18/V4 | 0.048411 | 0.148707 | 81.05% | 88.79% |

Dual-V2 is the strongest of the two history-preserving prototypes at W1.5 and W2.5 in this one local probe, and
both history-preserving arms recover substantial error relative to L16/V4. Dual-V2 is not uniformly better than
V2, however, and its W2.5 Top-1 moves opposite to MSE/KL/Top-5. This is screening evidence only. The format must be
selected by propagated held-out recovery, not by this local table.

##### Lesson learned: capacity must be usable by the quantizer

Decoder state or codebook cardinality is not, by itself, effective quantization capacity. Let

```text
C_format = every reconstruction representable by the serialized format
C_search(x) = reconstructions the quantizer actually compares for source x
C_useful(x) = candidates compared under an objective aligned with propagated model loss
```

Only `C_useful` can improve post-quantization recovery. A larger `C_format` does not help when the quantizer cannot
reach, compare, or correctly score the additional reconstructions.

Dual-V2 is the concrete warning. Over four weights, ordinary V2 already appends two `2R`-bit edges and therefore
has `2**(4R)` two-edge choices. Dual-V2 redistributes those same two edges across independent parity chains; it
does not add edge entropy over the fair four-weight span. Even under the favorable interpretation that its two
16-bit histories provide a larger conceptual joint context, the implemented objective factorizes:

```text
argmin_(path_A, path_B) [loss_A(path_A) + loss_B(path_B)]
  = (argmin_path_A loss_A(path_A), argmin_path_B loss_B(path_B))
```

No survivor, emission, or loss term lets an A-chain choice alter the B-chain decision. The split therefore removes
ordinary V2's consecutive A-to-B conditioning instead of giving the optimizer a jointly usable four-weight
correction space. Cross-device four-layer measurements confirmed the consequence: the regression exists in the
dense reconstructed weight before packing or backend inference.

Future low-rate format proposals must pass all of these gates before implementation:

1. Compare capacity over the same number of weights and payload bits.
2. Prove that the new representable set contains the canonical V2 baseline or serialize an exact baseline selector.
3. Show that quantization actually enumerates or searches the added degrees of freedom; do not sum independent
   state sizes and call the result jointly searchable capacity.
4. Include the coupling terms needed to exploit joint capacity, or explicitly classify the design as a factored
   product code with no cross-component correction.
5. Preserve a byte-exact V2 fallback and select alternatives using propagated search plus independent confirmation;
   local MSE/KL may shortlist candidates but cannot accept them below W3.
6. Report both theoretical format cardinality and measured effective search diversity: unique candidates evaluated,
   path churn, winner distribution, and downstream gains per added byte.

The resulting principle is: **increase usable, propagation-scored capacity—not merely decoder cardinality or the
sum of independent histories.**

#### V2B2-P32 first implementation slice

`format="qvq_v2b2_p32"` is a banked-V2 prototype at W1--W3.5. It preserves the canonical L16/V2
recurrence and spends one selector bit every 32 weights:

```text
one 16x16 tile (256 weights)
  |
  +-- eight contiguous 32-weight segments
  +-- one binary selector per segment
  +-- selector 0: exact canonical V2 decoder
  `-- selector 1: one module-selected complementary decoder family
```

The eight selectors occupy exactly one byte per tile, so the selector cost is `1/32 = 0.03125` bpw: the same
selector entropy as V2B4-P64, at twice the switching resolution and with two rather than four active bank paths.
Each module also stores one `uint8` alternative-family ID. That byte chooses one of the three rate-keyed graph
families already defined by the four-bank library; it is module metadata, not a per-weight selector.

The initial reference implementation deliberately separates codec validation from downstream optimization:

1. Quantize canonical V2 independently as the exact rollback oracle.
2. For each alternative family, run the coupled two-bank P32 recurrence and complete Block-LDLQ reconstruction.
3. Score the complete serialized reconstruction with `tr(E H_x E^T)`.
4. Accept only a finite strict improvement; otherwise emit the independent V2 path, all-zero selectors, and a
   deterministic alternative-family ID.

YAQA level 1 now feeds each two-sided corrected tile through the same exact segmented recurrence. It evaluates all
three complementary families as complete YAQA artifacts, selects one family per module under the complete Kronecker
proxy, and retains an independently encoded canonical V2+YAQA oracle for atomic rollback. On Apple, those corrected
tiles automatically use the native MLX segmented-V2 recurrence. This still does **not** justify low-rate quality
promotion: local and Kronecker-proxy improvements below W3 have repeatedly regressed live downstream metrics.
Live-prefix propagation-aware search and disjoint confirmation remain the promotion gate.

B2-P32 YAQA has two explicit experimental modes because its selection is hierarchical:

```text
module
  +-- choose one complementary family from IDs 1, 2, 3
  `-- choose canonical bank 0 or that family every 32 weights
```

- `yaqa.v2b2_family_mode="fixed_block_ldlq"` freezes the family chosen by a matched Block-LDLQ pass and lets YAQA
  change only the V2 path and binary P32 schedule. The family control uses Block-LDLQ's ordinary `0.01` input-Hessian
  damping rather than YAQA's `1e-4` damping, so it reproduces the matched baseline selection geometry. This isolates
  YAQA on an unchanged codec candidate space.
- `yaqa.v2b2_family_mode="reselect"` (default) evaluates all three family IDs as complete YAQA module artifacts and
  retains the strict best full-Kronecker result. This measures the combined ceiling because an input-Hessian winner
  need not be the winner under the two-sided YAQA objective.
- `yaqa.v2b2_family_mode="sampled_proxy"` is an experimental fast mode. It scores every family on 64 evenly spaced
  real module tiles with the corresponding diagonal input/output Hessian blocks, then runs a complete YAQA pass only
  for the selected family. The final candidate is still compared with an independently encoded canonical V2+YAQA
  oracle under the complete Kronecker proxy. The sampling step changes candidate generation, not YAQA arithmetic or
  the serialized format, so it needs propagated held-out confirmation rather than an algebraic equivalence claim.

Both modes independently encode canonical V2+YAQA. A non-finite, tied, or worse banked candidate restores that exact
artifact and all-zero selectors. The result reports fallback, selector churn, Block-LDLQ family ID, and family change;
selector entropy and selected family are derivable from the serialized selectors and `bank_alt_id`.

The matched factorial is V2, B2-P32, and B4-P64 under both Block-LDLQ and YAQA. For a lower-is-better loss `L`, report
the interaction

```text
I_B = (L_V2,YAQA - L_B,YAQA) - (L_V2,Block - L_B,Block).
```

Positive interaction means YAQA exploits the added bank space especially well; zero means approximately additive;
negative means overlap or ineffective candidate ranking. Promotion still depends on `L_B,YAQA < L_V2,YAQA` under
propagated held-out metrics, not on the sign of the interaction alone.

Current status:

- complete: exact P32 dynamic program, binary selector packing, module-selected alternative ID, independent V2
  rollback, format/config/processor/QVQLinear lifecycle, Torch reconstruction, strict reload, and focused math tests;
- complete: YAQA corrected-target integration, full Kronecker rollback, native CUDA quantization and packed
  inference, and native MLX corrected-tile quantization/inference through W3.5;
- partial: localized propagated selection now supports explicit disjoint search rows and an independent
  accept/rollback callback; native MPS decode and four-layer final-KL/Top-N/task evidence remain pending;
  four-layer final-KL/Top-N/task evidence;
- baseline comparison: `scripts/compare_qvq_codecs_llama_qkvo.py` defaults to matched `v2` and `v2b2-p32` arms.

#### Eigenspace-boosted YAQA P1

The default-off `yaqa.spectral_refinement` experiment applies the residual-spectrum idea from
[EoRA](https://arxiv.org/abs/2410.21271) without retaining an adapter. It first encodes the exact V2B2-P32+YAQA
rollback artifact `Q0`, then analyzes its actual transformed residual `E0=W-Q0`. For inner-orientation weights and
stabilized factors `H_I=C_I C_I.T`, `H_O=C_O C_O.T`, it computes

```text
M0 = C_I.T @ E0 @ C_O
M0 = U @ diag(sigma) @ V.T.
```

The right singular vectors are output-space modes. For each configured rank, P1 builds the PSD penalty

```text
H_O,spec = C_O @ V_r @ V_r.T @ C_O.T
trace(H_O,spec) = trace(H_O)
H_O' = H_O + lambda * H_O,spec.
```

Ranks default to `{8,16,32}` and strengths to `{0.1,0.25,0.5,1.0}`. Each boosted factor is only a candidate
generator: V2B2-P32 reruns complete YAQA family reselection, but every resulting artifact is scored under the
original unboosted `H_I,H_O`. A non-finite, tied, or worse candidate restores `Q0` bit-for-bit. The SVD, boosted
factors, and EoRA-style low-rank modes are discarded; serialized tensors and inference are unchanged.

P1 reports rank energy concentration, selected rank/strength, absorption efficiency, selector churn, and family
change. These are diagnostics rather than promotion evidence. Final-KL, Top-N, and task recovery on disjoint live
replay remain required because a lower module Kronecker proxy can still regress downstream behavior below W3.

> Shih-Yang Liu et al. “EoRA: Training-free Compensation for Compressed LLM with Eigenspace Low-Rank
> Approximation.” arXiv:2410.21271. P1 borrows the post-compression residual-spectrum principle; it does not retain
> EoRA `(A,B)` factors or add a runtime LoRA branch.

#### Eigenspace spectral push P3

The default-off `yaqa.spectral_push` experiment addresses P1/P2's measured failure mode: output-factor boosting
substantially reduced local/live KL while leaving final KL effectively unchanged, and its median discrete absorption
was zero. P3 therefore uses the continuous low-rank correction to cross rounding boundaries directly instead of
changing the objective used by YAQA.

For baseline `Q0`, residual `E0=W-Q0`, Cholesky roots `H_I=C_I C_I.T`, `H_O=C_O C_O.T`, and truncated SVD

```text
C_I.T @ E0 @ C_O = U @ diag(sigma) @ V.T,
```

P3 recovers the rank-`r` correction in QVQ's inner orientation without forming matrix inverses:

```text
C_I.T @ L_r @ C_O = U_r @ diag(sigma_r) @ V_r.T.
```

The implementation uses two triangular solves, then proposes each configured `alpha` through

```text
T_round = T_YAQA + alpha * L_r.
```

This changes only the tile target supplied to the segmented V2 Viterbi search. YAQA's committed error remains
`W-Q`, its two-sided feedback remains relative to the original dense `W`, and every complete candidate is rescored
under the original unmodified `H_I,H_O` Kronecker objective. A non-finite, tied, or worse candidate restores `Q0`
bit-for-bit. SVD factors and `L_r` are discarded, so checkpoint tensors, effective BPW, and inference are unchanged.

P3 reports the continuous-oracle loss for each rank, selected rank/alpha, rank-energy concentration, discrete
absorption efficiency, selector churn, and family change. Fixed-family and family-reselection arms remain separate:
the fixed arm is the first causal gate, while reselection is justified only if the fixed candidate shows propagated
benefit. Local or Kronecker improvement alone cannot promote P3; disjoint live-prefix/final-logit confirmation is
still required below W3.

The strict W2 experiment rejects this global-push form. Across 16 real Llama 3.2 1B Q/K/V/O modules and all
`rank={8,16,32}` by `alpha={0.5,1,2,4}` combinations, all 192 candidates changed selectors and almost every V2
state, but none improved the original YAQA objective. Selector churn averaged about 50% and state churn exceeded
99.6%; increasing either rank or alpha monotonically worsened the original proxy. Exact rollback therefore retained
the baseline artifact and final logits. The failure is a path avalanche, not insufficient proposal movement. Keep P3
default-off as a diagnostic; future spectral proposals must localize the correction or impose an explicit trust
region before they warrant another propagated gate.

#### Fixed-boundary localized propagation P4

P4 keeps the accepted V2B2-P32+YAQA artifact as an immutable rollback oracle. By default it changes at most one
32-weight segment per module. The optional `spectral_localized_max_changes` control can compose several disjoint
segments for experiments, but it does not weaken rollback or alter the serialized format. For segment `j`, the
search fixes both the predecessor state entering the segment and the final state leaving it. The search therefore
retains the complete V2 history before and after the segment:

```text
accepted YAQA path
  |
  +-- fixed entry state -> search 16 V2 transitions / two banks -> fixed exit state
  |
  `-- every state and selector outside this P32 segment remains bit-identical
```

Candidate generation ranks segments using the first-order YAQA term. A disjoint search set chooses among the
resulting exact fixed-boundary candidates using module-output squared error. The candidate is then packed and
decoded before an independent propagation callback sees it; that callback, rather than local MSE or Kronecker loss,
is the final acceptance authority. A rejection, non-finite proposal, serialization mismatch, or callback error
restores the exact original trellis, selectors, family ID, `SV`, and dense reconstruction.

This is default-off as `yaqa.spectral_localized`. V2B2 propagation must be explicitly enabled and supplied through
`QVQProcessor.set_propagation_gate`; it never derives search rows by splitting ordinary Hessian calibration data.
The checkpoint format, `0.03125`-bpw selector overhead, and inference kernels are unchanged. Multi-change mode uses
greedy conditional composition: after each accepted fixed-boundary replacement, every remaining candidate is
rescored against the current live residual, and a `(tile, segment)` can be selected only once. It therefore avoids
the invalid assumption that standalone candidate gains add. The default remains one change. Broader composition is
experimental and must pass a disjoint full-model confirmation horizon; module output or short-prefix improvement is
not sufficient evidence below W3.

An optional full-horizon shortlist closes the remaining selection mismatch without making every spectral proposal
expensive. Cheap conditional module loss first orders the legal fixed-boundary candidates. At each greedy step only
the best `spectral_localized_replay_candidates` candidates are serialized exactly and scored by a caller-supplied
full-model search loss:

```text
all legal fixed-boundary candidates
  -> conditional module-loss ordering
  -> bounded top-K shortlist
  -> exact pack/decode/reconstruct
  -> full-model search-split score
  -> conditional winner or no change
  -> independent confirmation split
```

The search scorer is lower-is-better and is evaluated first on the immutable baseline. Non-finite baseline scores or
callback failures retain the exact baseline. Every later greedy step scores candidates on top of the already selected
serialized path, so the procedure does not assume that standalone downstream gains add. This mode is default-off
(`spectral_localized_replay_candidates=0`) because it adds up to `1 + K * max_changes` full-model search forwards per
module. It changes neither checkpoint bytes nor inference operations. Independent confirmation also requires a
minimum relative KL improvement (0.1% in the validation driver by default) in addition to the Top-N guard. Merely
changing the sign of KL by an amount below that threshold is treated as no demonstrated improvement.

When the full-horizon scorer is present, local/module loss is only a cheap ordering heuristic for the top-`K`
shortlist. It is not an eligibility gate and it is not the final rollback gate. Requiring local improvement before
replay would remove exactly the locally unfavorable but propagation-cancelling directions this low-rate mode is
intended to find. The immutable full-horizon baseline score decides whether a replayed candidate advances. When no
full-horizon scorer is present, the original strict local-improvement requirement remains in force.

An optional direct-path portfolio adds a candidate family that is independent of the EoRA rank/alpha direction.
For the highest-residual P32 segments it re-encodes the dense transformed weights while fixing the accepted entry and
exit states. Thus every proposal is a legal replacement inside the same V2 chain, cannot disturb neighboring
segments, and changes no serialized layout. If `D` of the `K` replay slots are reserved for direct candidates, the
search cost remains exactly `1 + K * max_changes` full-model forwards; only portfolio composition changes. Exact
duplicates of spectral candidates are removed before replay. Direct candidates are prioritized by segment residual
energy and then local loss, but only the full-horizon score can select one. The default is `D=0` so historical P6
behavior is preserved.

An optional P9 ranking stage moves downstream information into candidate generation. Let the full-model search loss
at the exact serialized baseline be teacher forward KL and define

```text
G_W = d KL_teacher / d W
G_I = R*(G_W)
```

where `R` is QVQ's complete inner-to-dense RHT reconstruction, including `SU`, `SV`, and the selected module scale,
and `R*` is its adjoint. For a legal fixed-boundary inner-weight delta `D`, the first-order downstream prediction is

```text
Delta KL = <G_I, D> + O(||D||^2).
```

P9 ranks the bounded portfolio by this term rather than residual energy or module-output loss. Negative values are
predicted improvements, but they are proposals only: exact serialized full-model replay, fold minimax scoring, and
independent confirmation remain the selection authorities. The implementation obtains `R*` from the production RHT
with autograd and unit-tests the adjoint identity `<G_W,R(D)> = <R*(G_W),D>`, avoiding a duplicated sign/transpose
formula. The teacher-KL gradient is accumulated in FP32 one full sequence at a time through the live quantized
prefix and suffix. This adds one backward sweep over the search rows, but no checkpoint bytes or inference work.
Gradient callback failure, non-finite output, wrong geometry, or serialization mismatch fails closed to the immutable
baseline. The gradient uses the search split and therefore is not confirmation evidence.

The validation driver can also divide the fixed search rows into `F` round-robin folds without adding prompts or
forwards. Let `K_f(Q)` be full-model teacher KL on fold `f` and let `Q_0` be the immutable serialized baseline. It
ranks a candidate with the minimax relative score

```text
S(Q) = max_f K_f(Q) / K_f(Q_0).
```

The baseline has score one, so a candidate can win only when it improves every search fold. Round-robin assignment
avoids making a contiguous prompt region the entire fold. This is a conservative robustness screen, not independent
confirmation: the folds still participate in candidate selection, and the winner must pass the separate confirmation
set and its minimum-effect/Top-N gates. `F=1` preserves the pooled-search behavior. Larger `F` reduces tokens per
fold and can amplify estimator noise, so fold count must remain small and predeclared.

#### V2B4-P64 implementation slice

`format="qvq_v2b4_p64"` is the baseline-safe replacement for Dual-V2 at W1--W3.5. It keeps the canonical L16/V2
state recurrence and adds a two-bit decoder-bank selector for each contiguous 64-weight segment:

```text
one 16x16 tile (256 weights)
  |
  +-- 64 weights -- bank selector b0 -- 32 coupled V2 transitions
  +-- 64 weights -- bank selector b1 -- 32 coupled V2 transitions
  +-- 64 weights -- bank selector b2 -- 32 coupled V2 transitions
  `-- 64 weights -- bank selector b3 -- 32 coupled V2 transitions

bank 0: canonical V2 decoder (frozen)
banks 1--3: rate-keyed bijective state relabellings
```

The selector cost is exactly `2/64 = 0.03125` bpw. Four selectors fit in one byte per 16x16 tile. The trellis
payload itself is unchanged. Unlike Dual-V2, bank selection does not split the path into independent parity chains:
every transition remains conditioned on the immediately preceding V2 state.

The reference dynamic program keeps four bank costs inside a segment. At a P64 boundary it minimizes over the prior
bank and the legal V2 predecessor prefixes, then exposes all four current-bank emissions:

```text
C_t(b, s') = d_b(x_t, s')
             + min over prior bank a and legal predecessor s -> s' of C_(t-1)(a, s)
```

Inside a segment `a=b`, so the selected bank cannot change early. At a boundary the merge retains the best prior
history for each next state before opening the next four-bank choice. This is an exact coupled recurrence for the
declared P64 search space; it does not enumerate `4**4` complete schedules and does not factor the V2 path.

Bank zero is a complete independently quantized V2 artifact. After Block-LDLQ, the mixed-bank artifact is compared
with it using the full transformed input-Hessian objective

```text
D(E) = tr(E H_x E^T).
```

Non-finite, tied, or worse mixed results atomically restore the bank-zero states and all-zero selectors. Therefore
the initial implementation has an algebraic no-regression guarantee for this proxy, not an end-to-end quality
guarantee. Low-rate promotion still requires propagated held-out replay and independent confirmation because this
local quadratic objective is not sufficient below W3.

Current implementation status:

- complete: format/config lifecycle, rate-keyed graph banks, exact Torch P64 Viterbi, Block-LDLQ integration,
  planar selector serialization, save/load-compatible `QVQLinear`, dense Torch inference, exhaustive bank-zero/V2
  parity, and full-proxy rollback tests;
- complete: YAQA corrected-target P64 search, independent V2+YAQA oracle, complete Kronecker rollback, native CUDA
  quantization and packed inference, and native MLX corrected-tile quantization/inference through W3.5;
- deliberately pending: learned table portfolios, propagation-aware bank refinement, native MPS decode, and
  model-level KLD/Top-N/task recovery gates;
- rejected as a claim: four graph banks alone are not yet evidence of a 50% recovery improvement. They are the
  baseline-safe search substrate on which propagation-aware selection and learned portfolios can be tested.

#### All-rate storage and compute consequences

Pure `L16/V4` is defined only through W4. Its transition width is `E=4R`, and a bitshift trellis requires `E <= L`.
At W4, `E=16`: the transition consumes the entire state, leaves no history, and directly selects one of 65,536
four-value vectors. At W4.5--W8, `E=18--32` exceeds the 16-bit state. Masking those extra bits would waste payload
and leave reconstruction capped at 4 bpw. Calling that layout `L16/V4` would be incorrect.

| Rate | Edge bits `E=4R` | Direct successors | Retained history bits | Pure `L16/V4` status |
|---:|---:|---:|---:|:---|
| W1 | 4 | 16 | 12 | valid; below 64-successor target |
| W1.5 | 6 | 64 | 10 | valid |
| W2 | 8 | 256 | 8 | valid |
| W2.5 | 10 | 1,024 | 6 | valid |
| W3 | 12 | 4,096 | 4 | valid, but V2 is the recommended dispatch |
| W3.5 | 14 | 16,384 | 2 | valid, but V2 is the recommended dispatch |
| W4 | 16 | 65,536 | 0 | valid but memoryless |
| W4.5--W8 | 18--32 | not representable | not representable | invalid for L16 |

For every valid rate W1--W4, raw payload is unchanged from `L16/V2`:

```text
L16/V2: (256 / 2) transitions * (2R) bits = 256R bits/tile
L16/V4: (256 / 4) transitions * (4R) bits = 256R bits/tile
         256R / 256 weights = R bpw.
```

Both layouts therefore use exactly `8R` planar int32 words per 16x16 tile and `R/8` bytes per weight. Reusing the
current `SU`, `SV`, and implicit 256-entry PGC table leaves effective BPW unchanged. For the measured Llama-3.2-1B
artifacts, whose common auxiliary overhead is `0.023168` bpw, the predicted values remain:

| Rate | Raw BPW | Predicted effective BPW | Change from `L16/V2` |
|---:|---:|---:|---:|
| W1 | 1.000000 | 1.023168 | 0 |
| W1.5 | 1.500000 | 1.523168 | 0 |
| W2 | 2.000000 | 2.023168 | 0 |
| W2.5 | 2.500000 | 2.523168 | 0 |
| W3 | 3.000000 | 3.023168 | 0 |
| W3.5 | 3.500000 | 3.523168 | 0 |
| W4 | 4.000000 | 4.023168 | 0 |

A second implicit mixer adds constants, not checkpoint tensors. Even a second model-level 256-entry FP16 table would
add only 512 bytes, or about `0.0000041` bpw over one billion weights. A one- or two-bit bank selector per 256-weight
tile adds exactly `0.00390625` or `0.0078125` bpw (about 0.48 or 0.95 MB over 972.8 million weights). State version
metadata is negligible; report fixed-shard filesystem padding separately from logical EBPW.

The optimized inference operation count should also be approximately unchanged. Per four weights:

```text
current V2: 2 state reconstructions + 2 PGC mixes + 4 scalar lookups
planned V4: 1 state reconstruction  + 2 PGC mixes + 4 scalar lookups.
```

Planar bytes read and weight multiply-accumulates are identical. The wider edge doubles bits per transition but
halves transitions. There is no fundamental extra inference FLOP or bandwidth term; an optimized fused kernel should
target `0.95--1.05x` current latency. Budget `1.05--1.20x` for the first native implementation while its four-value
register layout and bit extraction are tuned. A Python/eager prototype is not a useful latency predictor. Banked
mixers add one small selector load per tile and should target another `0--3%`; measure rather than assume branch-free
specialization.

Exact Viterbi quantization does not have a corresponding compute increase. With `S=65,536` states and 256 weights per
tile, the emission dot-product work is

```text
V2: 128 steps * 2 coordinates * S = 256S
V4:  64 steps * 4 coordinates * S = 256S.
```

Transition reduction is approximately `S` comparisons per step, so V4 performs about half as many comparisons and
traceback steps. At W2 the tail-overlap space falls from 4,096 to 256 states; at W2.5 it falls from 2,048 to 64.
Backpointer entry counts fall from `127*4,096` to `63*256` at W2 and from `127*2,048` to `63*64` at W2.5--about
32x fewer bytes after accounting for wider prefixes. The quantization-only FP32 codebook grows from 65,536x2 to
65,536x4, adding only 512 KiB. Total cost-array storage is unchanged.

The expected optimized quantization wall-time range is therefore `0.6--1.1x` current `L16/V2`, not a required
slowdown: emission work is equal, transition/traceback work is lower, but 256/1,024-way reductions may have lower
hardware efficiency than current 16/32-way reductions. The existing native CUDA, MPS, and MLX encoders are specialized
for `V2`; a generic `V4` fallback may be several times slower and is not a useful estimate of the final design. Record
emission, reduction, overlap ranking, constrained reruns, and traceback separately before making a wall-time claim.

There are three ways to cover W4.5--W8, none better than the current format by default:

1. **Recommended hybrid:** evaluate `L16/V4` at W1 and use it for the unstable W1.5--W2.5 band when it passes the
   propagation gates; retain `L16/V2` for W3--W8. W1/V4 has only 16 direct four-weight successors, so it does not
   satisfy the 64-successor goal and must earn its inclusion on quality. Raw and effective BPW stay exactly
   as above/current at every rate. W1.5--W2.5 gain 64--1,024 direct joint successors; stable and high rates retain
   the validated codec and kernels. Per-module rate metadata already exists.
2. **Hierarchical V4 refinement:** keep a 4-bpw `L16/V4` base and store `R-4` residual bits/weight. Raw payload can
   remain exactly `R` bpw, but the decoder needs residual unpack/decode/add work and usually another scale. A FP16
   scale per 256-weight tile adds `0.0625` bpw; a fixed scale adds no metadata but is less adaptive. Greedy staged
   quantization is estimated at `1.2--1.8x` current high-rate encoding and inference at `1.1--1.6x`, depending on
   residual complexity. Exact joint optimization would be combinatorial. This is a different hierarchical codec,
   not pure `L16/V4`, and must beat native V2/int8 at matched EBPW to justify itself.
3. **`L32/V4`: reject for exact search.** It permits `E=4R <= 32` through W8 and keeps raw BPW at `R`, but expands
   the state space from `2**16` to `2**32`: 65,536x more DP states. One FP32 cost vector alone is about 16 GiB per
   trellis, before emissions, codebook values, or backpointers. Quantization time and workspace are unreasonable even
   if implicit inference decode were cheap.

The all-rate implementation decision is therefore a versioned hybrid dispatch, not forcing V4 above its 4-bpw
capacity ceiling. It gives the requested low-rate successor expansion with zero logical VRAM increase and leaves the
already stable W3--W8 storage and compute paths untouched.

The implementation toggle is `QVQConfig(format="qvq_v4")`. It selects `vector_size=4` automatically for W1--W4,
serializes that vector size in the QVQ configuration/lifecycle metadata, and rejects W4.5--W8. The default remains
`format="qvq"`/`L16/V2`, so A/B artifacts can be produced from the same quantization configuration. V4 currently uses
the reference Torch reconstruction on inference backends; native CUDA/MPS/MLX V4 kernels are a separate follow-up and
must pass reconstruction parity before being enabled.

#### High-rate V1 alternative (design check)

For W4.5--W8, a scalar `L16/V1` transition would append `R` bits and emit one scalar per step. It is mathematically
valid for integer rates W1--W8: the direct out-degree is `2**R`, the retained history is `16-R` bits, and a 256-
weight tile has `256/R` scalar steps when `R` divides 256. It does not reduce the serialized payload: the tile still
contains `(256/R)*R = 256R` bits, exactly `R` bpw. It also does not increase the number of jointly selectable
vectors per four weights; it trades V2's paired shaping for a one-dimensional scalar codebook and shorter edges.

At W4.5, W5, and other half-step rates, `R` is fractional and a scalar edge cannot carry a non-integer number of
bits. Supporting those rates requires alternating edge widths or a mixed V1/V2 schedule, which complicates planar
mapping, circular-state validation, and every decoder. At high rates, V2 already has abundant joint choices and the
PGC16 two-scalar manifold is a stronger shaping primitive, so V1 should be a measured research arm rather than the
default high-rate dispatch. A valid implementation would need a new explicit format (for example `qvq_v1`), a scalar
PGC16-derived decoder, integer/half-rate schedule metadata, and matched quality/latency tests; it cannot be inferred
from `format="qvq"` or silently reinterpret existing V2 payloads.

#### Small-overhead companion controls

Run these after or alongside the `L16/V4` reference:

1. **Banked `L16/V2` graph control.** Use two complementary implicit mixer banks at W2.5 and four at W2. A one- or
   two-bit bank id per 256-weight tile costs only `0.00390625` or `0.0078125` bpw. Also test one bank id per 32
   weights, costing `0.03125` or `0.0625` bpw, to measure the adaptation/granularity tradeoff. The runtime loads one
   tiny selector per tile/subtile and uses bank-specific constants; no serialized scalar table is added. This is a
   cheap geometry repair, but not a strict 64-outgoing-edge graph.
2. **Fixed-trellis `SU`/`SV` alignment.** Apply blockwise clean-target alignment followed by short full-model
   soft-target alignment with disjoint validation and rollback. It adds no inference byte or operation. Existing W2
   diagnostics lowered KLD/JSD by about 19.66%/18.14% and raised top-1/top-5 by 4.42/3.15 points, making this the
   strongest demonstrated zero-runtime-cost recovery control. A corrected W2 V2B2-P32 layer-0 gate nevertheless
   produced a mixed 64-row propagated result: Top-1/Top-5 rose by 0.175/0.045 points while final KLD/JSD regressed by
   0.80%/0.07%. A separate strict post-quant SU/SV-only arm improved all four 64-row evaluation metrics, but failed
   its independent validation gate because Top-5 fell 0.068 points. Keep it explicit and transactional; default
   promotion requires a larger disjoint B2-P32+YAQA factorial rather than clean layer-MSE or report-only acceptance.
   The full audit is recorded in `docs/qvq_todos.md`.
3. **Fused low-rank residual fallback.** If joint shaping and existing-scale alignment leave a diffuse residual, fit
   a propagation-aware rank-`r` correction and fuse its two small matmuls. FP16 factors add

   ```text
   delta_bpw = 16 * r * (in_features + out_features) / (in_features * out_features).
   ```

   For a square 4,096-wide matrix, rank 4 and rank 8 cost `0.03125` and `0.0625` bpw. This corrects residual error but
   does not increase trellis successor count; report it separately and guard against low-rank-factor overfitting.
4. **Sparse escape control.** An extra branch bit on a fraction `p` of W2.5 pairs costs at least `0.5p` bpw before
   encoding the locations. It can target pathological pairs while staying below W3, but cannot guarantee 64 choices
   for every pair and adds irregular decode work. Prefer `L16/V4` unless measured error is highly concentrated.
5. **Two-edge lookahead control.** Keep `L16/V2`, but make reconstruction at step `t` depend on both its state and the
   next path code. A two-step path then exposes 256 W2 or 1,024 W2.5 candidate labels without adding payload. The
   choices are correlated with the following pair and graph out-degree remains 16/32. Quantization needs a
   transition-dependent emission recurrence, potentially multiplying encoder work by 16/32; inference needs an
   additional edge read and keyed mix. This is a useful discriminator if `V4` loses too much intermediate context.
6. **Propagation-aware mixed rate.** If the requirement is high recovery below 3 average bpw rather than 64 branches
   in every module, keep most weights at W2.5 and promote only causal high-gain modules to W3. Promoting a weight
   fraction `f` gives payload `2.5 + 0.5f` bpw, which remains below W3 for every `f < 1`. Compare against uniform
   W2.5 and random promotions at identical serialized EBPW.
7. **Propagation-reranked path list.** Retain several near-optimal encodings generated by overlap candidates, RHT
   seeds, mixer banks, or an explicit `K`-best recurrence, then select on a disjoint live-replay set. This attacks
   the discontinuous single-`argmin` instability directly. It can use `K` times more offline workspace/compute while
   leaving the selected checkpoint and inference path unchanged. It does not increase graph out-degree and must not
   be selected by trellis SSE, local KLD, or the same rows used to form its Hessian/Fisher factors.

Variable-length entropy coding is the only way to expose 64 pair symbols while averaging fewer than six stored bits
per pair without correlating choices, and only when their measured entropy is below six bits. It cannot guarantee a
sub-W3 worst-case payload, adds block indexes and serial decode dependencies, and is unlikely to compress a well-
mixed near-uniform path stream enough to justify the MPS/MLX/CUDA cost. Do not prioritize it without first measuring
edge histograms and blockwise entropy on real W2/W2.5 paths.

The recommended low-rate stack is therefore: establish `L16/V4` as the strict 64-plus-successor reference; add a
small bank selector only if it improves occupancy-weighted live geometry; use propagation-reranked candidates to
stabilize quantization; then apply fixed-trellis `SU`/`SV` alignment. If task loss remains concentrated, spend bits on
causal W3 promotions; if it remains diffuse, compare the low-rank residual at the same actual EBPW. This ordering
separates graph capacity, path-selection variance, downstream alignment, and residual correction instead of changing
all four at once.

Current lifecycle policy makes the module-boundary propagation stage
default-on after a caller opts into four-bank V4 Block-LDLQ. The processor
reserves held-out token rows from Hessian capture and accepts candidates on
their dense module outputs. This is not the deferred full-model
`QVQPropagationRefiner`: final-logit/task replay remains a separate opt-in
selection stage and the stronger promotion gate.

Acceptance requires exact `R/8` base payload for `L16/V4`, 65,536 unique implicit vectors, exhaustive small-graph
path correctness, deterministic packing, save/reload parity, and Torch/CUDA/MPS/MLX reconstruction parity. Compare
W2/W2.5 path churn across quantization and Fisher seeds, held-out propagation, final KLD/JSD, margins, top-1/top-5,
paired task flips, and task scores. Require the selected low-rate arm to beat current `L16/V2` across multiple seeds
and remain within 5% of its inference latency. Compare any residual or banked variant at its actual serialized EBPW.

## Accuracy-sensitive approximations and improvement plan

The following mechanisms are mathematically intentional approximations. They are not algebraic correctness bugs, but
their approximation error can dominate post-quantization quality. KLD/MSE improvements on an isolated module are not
promotion criteria; every proposal below requires held-out replay, final-logit ranking, and task-level validation.

### Required QVQ propagation protocol

Every QVQ change must be evaluated with the real quantized dataflow, not independent dense-input replays:

```text
dense h_l
  -> QVQ module l
  -> actual quantized output
  -> next QVQ/dense module
  -> ... remaining layer
  -> subsequent layers
  -> final logits / generated tokens
```

For each boundary record both the incremental error introduced by the current module and the total error arriving from
all earlier modules. Report activation norm/cosine, covariance drift, per-channel outliers, logit margins, top-1/top-5,
KL/JSD, and task behavior. The key acceptance quantity is not just `||e_l||`, but its downstream gain:

```text
G_l = || J_{L<-l} e_l || / (||e_l|| + ε)
```

where `J_{L<-l}` is measured or approximated on the held-out replay path. A lower local error with materially higher
`G_l` is a regression. Companders, scale searches, rounding, YAQA, and candidate widening must all use this protocol.

| Area | Current approximation | Possible improvement | Required validation gate |
|---|---|---|---|
| YAQA Fisher factors | Sampled full-model score gradients are compressed into Kronecker factors; quality depends on sequence/token count and diversity. | First increase independent sequences and valid tokens, stratify by context length/domain, and compare Sketch-A/B with deterministic held-out seeds. Compare normalized Kronecker products, eigenspaces, and held-out quadratic proxies because the individual factor scales are not identifiable. Consider models beyond one Kronecker product only after convergence and causal rescues show diffuse residual error; that may require a new solver, not just a new tensor. | Normalized-factor/product convergence versus sequence/token count, held-out factor proxy, final-logit KL/JSD, top-1/top-5, perplexity and task scores. |
| Block-LDLQ | Tiled input-side feedback approximates the unrestricted full-Hessian optimum. | Tune tile geometry; use overlapping/block-diagonal-plus-low-rank Hessians; selectively solve larger blocks for sensitive modules. | Compare against dense small-matrix optimum and measure weighted error on replayed activations; reject any end-to-end regression. |
| Tail biting | Candidate widening evaluates selected overlap states, not every circular path. | Use exact circular dynamic programming for small overlap spaces; widen/rerank candidates with full proxy cost; add branch-and-bound bounds for larger spaces. | Exhaustive circular oracle on small fixtures, then held-out final-logit and task gates at each rate. |
| Inference precision | FP16 Hadamard/epilogue paths can round or drift despite mathematically correct decoded weights. | Keep FP32 accumulation where supported; use stage-normalized butterflies; add BF16/FP32 fallback only on non-finite or high-error paths; compare fused kernels to reference. | Bitwise/ULP reconstruction parity, finite-output stress tests, latency budget, and full-model logits by dtype/backend. |
| PGC16 table and scale | Fixed Gaussian scalar levels and empirical per-rate multipliers are not optimal for every layer. | Learn global/role-specific companders; fit layer statistics with frozen runtime layout; optimize scale jointly with trellis assignments. | **Do not accept on local error alone:** replay the quantized module into the next module, then through the remaining layer/model; require propagated activation and final-logit/task gates. |
| Damping | Mean diagonal damping is robust but not a fully adaptive Hessian regularizer. | Test diagonal/eigenvalue-aware damping, per-block damping, condition-number targets, and YAQA-factor-aware regularization. | Cholesky stability, condition-number diagnostics, weighted reconstruction, and held-out behavior; never select on local loss alone. |
| Output-scale optimization | Closed-form channel/module scale fixes a decoded matrix and cannot change trellis assignments. | Alternate scale proposals with re-encoding; jointly search scale and trellis candidates; use downstream-output sensitivity for acceptance. | Exact checkpoint-dtype reconstruction, original-Hessian acceptance, replayed layer outputs, final-logit margins, and task benchmarks. |

### Interpretation rule

Local MSE/KLD is a localization signal, not a quality verdict—and a local improvement can be actively misleading.
Changing a compander changes the *direction* and correlation structure of the residual, not only its norm. Let
`e_l = q_l(h_l) - f_l(h_l)` be the error of module `l`. The next module receives `h_{l+1}+e_l`, so the propagated
perturbation is approximately

```text
e_{l+1} = (J_{l+1} - J^q_{l+1}) e_l + q_{l+1}(h_{l+1}) - f_{l+1}(h_{l+1})
```

and after many modules

```text
e_L ≈ Σ_l (Π_{j>l} J_j) e_l + cross terms.
```

Thus a compander may reduce `||e_l||` or local KLD while rotating `e_l` into a high-gain downstream direction,
increasing the next-module error, residual-stream drift, attention-logit error, or autoregressive trajectory divergence.
The correct experiment is therefore a *propagation chain*: quantize one module, run its actual output into the next
module; then quantize the next module and repeat, measuring each boundary. A compander is promoted only when its
benefit survives the next module, the remainder of the decoder layer, subsequent layers, final logits, and held-out
tasks. Full teacher-forced KLD remains only one downstream signal: it can still miss decision-boundary flips, rare
answer tokens, and self-conditioned generation drift.

### Propagation-aware precision allocation

The strongest next *hypothesis* for improving low-rate full-model recovery is conditional precision allocation on top
of a validated YAQA baseline. It is not yet a production conclusion. The completed YAQA512 arms are undersampled
overlap controls, and a local or one-at-a-time rescue is not sufficient evidence for a final mixed-rate policy.

Within this analysis, use the same dense-minus-quantized error convention as the YAQA fixed-point section. For dense
and reconstructed weights `W_l*` and `W_l^q`, define

```text
E_l = W_l* - W_l^q
delta_l = vec(E_l)
delta = concat_l(delta_l)
```

For the teacher-model KL `K(theta) = E_x[KL(p_theta* || p_theta)]`, the dense model is a zero-valued minimum, so the
linear term vanishes. Around the dense model,

```text
K(theta* + delta)
  = 1/2 * sum_i sum_j delta_i.T @ H_ij @ delta_j + o(||delta||^2).
```

Benchmark accuracy is discrete and does not itself have this Taylor expansion. For a different differentiable task
loss whose dense checkpoint is not stationary, a linear gradient term may remain; do not mix that case into the KL
derivation.

Unweighted reconstruction sees only `||E_l||_F^2`. GPTQ/BlockLDLQ uses the local input-activation proxy

```text
D_local,l = trace(E_l @ H_x,l @ E_l.T),  H_x,l = E[X_l.T @ X_l].
```

Here `X_l[tokens, in]`, so `H_x,l[in, in]`; reversing the product would produce a token-by-token matrix and is not
dimensionally compatible with `E_l[out, in]`.

YAQA replaces that with a Kronecker approximation to the diagonal full-model Hessian block `H_ll`:

```text
D_YAQA,l = trace(E_l @ H_I,l @ E_l.T @ H_O,l).
```

This is the corresponding Kronecker quadratic form under YAQA's vectorization convention. YAQA therefore includes
sampled downstream sensitivity *inside one module/layer block*. Independent module rounding still omits `H_ij` for
`i != j`; finite-error live-input drift and nonlinear/autoregressive trajectory changes are omitted by the local
quadratic expansion as well. Cross-module curvature is consequently a plausible explanation for interacting errors,
not something established by a better or worse aggregate benchmark score.

Precision selection is a budgeted, path-dependent intervention. Let `A` be the current set of accepted actions and
let `a = (semantic group, new rate/configuration)` be a proposed action. Define one predeclared, normalized,
lower-is-better held-out selection loss `L`; retain every component metric alongside it. For actions with positive
serialized size cost, the conditional loss reduction per byte is

```text
S_a(A) = [L(A) - L(A union {a})] / [B(A union {a}) - B(A)].
```

`B` is the exact logical serialized size of codes, scales, transforms, tables, and required metadata. Also report
fixed-sharding on-disk bytes and effective BPW. A same-BPW candidate has a zero denominator: evaluate zero-byte
actions first and retain only held-out Pareto improvements with no guardrail regression; never divide by zero.
Because `S_a(A)` changes after every accepted action, never build a policy from a static list of isolated module
scores. Preserve the full quality-versus-BPW Pareto frontier; the scalar loss is only for search ordering, with metric
normalization, weights, and direction fixed before seeing intervention results.

Five non-overlapping data roles are required:

1. ordinary QVQ calibration for activation Hessians and sequential replay;
2. independent YAQA Fisher sequences, at or above a validated sequence-count gate;
3. an allocation-search set for propagated logits, margins, and task-like behavior;
4. a locked allocation-confirmation set, or cross-fitting folds, for confirming searched actions;
5. a preregistered external benchmark suite or split not previously inspected, touched only after the precision
   policy is frozen.

The existing ARC, GSM8K, STEM, and History aggregates have already influenced the research priority. Treat them as
development evidence. Re-running them after policy selection is useful but descriptive, not an unbiased final test.

The rescue program is hierarchical so the expensive in-loop search is spent where it can change the decision:

1. Establish matched uniform W2.5/W3 YAQA baselines, a dense reference, and a healthy W4-or-higher reference using
   identical runtime and prompt contracts.
2. On the allocation-search set, screen dense or higher-rate replacements under full live execution. Start with
   decoder bands and module-tree semantic groups, then descend to exact modules inside positive groups. Do not infer
   roles from path substrings. Preregister band boundaries, candidate rates, byte budgets, stopping rules, and
   numerical guardrail margins.
3. Save per-example outputs. Report correct-to-wrong, wrong-to-correct, and net flips; token/sequence KL; top-k;
   teacher-forced margins; and deterministic generated-answer changes. A single answer-token margin is insufficient
   for autoregressive GSM8K.
4. Recompute conditional gains after each accepted promotion. On the final shortlist, test pair interactions or use a
   small beam/coordinate search; this is where omitted cross-module terms can reverse a standalone ranking.
5. For propagation-sensitive groups, also compare same-rate QVQ candidates generated by controlled rounding,
   tail-biting, scale, or seed choices and reranked on the allocation-search set. This may recover quality at
   zero additional BPW; it is not a license to accept the candidates on local loss.
6. Re-quantize from the dense checkpoint with the selected candidate choices and dynamic rate overrides using the
   same fixed data/seeds.
   Post-hoc replacement is a sensitivity screen only: it cannot undo sequential quantization and recovery errors, and
   modules from independently recovered checkpoints must not be transplanted into one claimed result.
7. Confirm the selected actions on the locked confirmation split. Report a 95% paired-bootstrap interval for each raw
   score delta and an exact McNemar test from correct-to-wrong and wrong-to-correct counts; search-set statistics are
   not confirmatory evidence after testing many candidates.
8. Compare with uniform-rate and random-promotion controls at matched actual BPW. Freeze the policy, verify save/reload
   and every supported target backend, then run the preregistered external benchmark once. Repeat calibration/Fisher
   seeds before enabling a default.

For reporting only, the fraction of a dense gap rescued on task `t` is

```text
R_t = (score_promoted,t - score_base,t) / (score_dense,t - score_base,t).
```

Use this ratio only when the dense score is strictly better and the denominator is material; otherwise report the raw
paired delta. Do not call `score_quantized / score_dense` "fraction of lost quality recovered"--it is only a
dense-relative score. Do not tune the allocation on already-observed ARC/GSM8K/MMLU rows and then report those same
rows as an unbiased evaluation.

## Learned PGC16 compander (`pgc16-v2`)

QVQ-v2 learns the scalar reconstruction manifold while preserving the complete PGC16 runtime contract:

```text
16-bit state -> xor-MAD-xor permutation -> G[p >> 8], G[p & 255]
```

Only `G[256]` changes. The trellis state, vector size `V=2`, planar payload, mixer constants, two lookups, branch count,
kernel launch count, and checkpoint tensor shapes are identical to v1. A loaded model has one shared 512-byte FP16
table per runtime/device, replacing the v1 table rather than supplementing it. There is no `levels`, `lut`, `tlut`, or
`codebook` tensor in any module state dict.

The implemented fitter starts from the canonical Gaussian table and alternates:

1. tail-biting Viterbi assignment using the current PGC16 table;
2. a Hessian-importance-weighted Lloyd centroid update for the two scalar coordinates;
3. weighted PAVA projection to a monotone table;
4. exact FP16 freezing while preserving 256 distinct ordered values;
5. reassignment and fail-closed rejection if weighted error increases.

The centroid objective uses a broadcastable nonnegative importance tensor, normally the diagonal of the calibration
Hessian. Full-Hessian coupling remains represented by QTIP's BlockLDL error-feedback correction before vectors reach
the local trellis search. This is a deterministic Lloyd implementation; AdamW/SGD/LBFGS remain unselected research
alternatives because they add optimizer nondeterminism without changing inference.

The selected scope is one table per model. A per-module table is rejected because it would multiply metadata and cache
entries while undermining the simple shared decoder. Functional attention/MLP/embedding tables are not implemented;
they remain an evidence-gated option only if held-out two-/four-layer KLD shows a meaningful gain over the model-level
table. The current core API can fit a shared sequence population, but production collection of all eligible modules is
blocked on the dedicated QVQ processor lifecycle.

Frozen levels are stored in `quantization_config.compander_bits` as 256 unsigned FP16 bit patterns and are owned by
`codebook=pgc16-v2`. V1 rejects this metadata; v2 requires it. This adds no checkpoint tensor and no inference table
bytes, although the JSON metadata itself is larger than the implicit v1 identifier.

Accuracy and corner-case coverage includes deterministic fitting, weighted-error non-regression, asymmetric and
symmetric projections, tied/unobserved bins, all-zero importance, noncontiguous samples, invalid shapes/dtypes/devices,
nonfinite/negative importance, FP16 overflow, exact 65,536-vector capacity, exact metadata round-trip, missing/misowned
metadata, and learned-table quantize/reconstruct parity. Torch, MPS, and MLX learned-table inference passes W2/W5/W8,
M=1/4, three seeds, ten repeated launches, forward KL `< 2e-5`, exact top-1, and exact ordered top-5. Before the
whole-edge half-step format change, CUDA passed the corresponding integer-rate contract across W2--W8, M=1/8/32,
FP16/BF16, three seeds, and ten repeated launches on a PG506-230 `sm_80` GPU. The CUDA suite now covers E2--E16 in
source and test collection, but those changed kernels require fresh runtime validation on CUDA hardware.

The value-independent latency check uses a valid non-Gaussian FP16 table with the same unchanged kernels. On Apple M4
Max, P-core QoS, 2048x2048, M=1/4, W2--W5, both MPS and MLX, ten warmups, 15 samples, and 20 synchronized inner
iterations, v2/v1 median latency ratio was `0.995`, with range `0.921--1.029`; all 16 rows passed the no-more-than-5%
slower gate. Raw distributions are generated artifacts and are intentionally not stored in `docs/`; publish them as CI
artifacts when rerunning the benchmark. The table values cannot alter shader control flow or memory traffic, so this
benchmark validates dispatch/cache plumbing rather than claiming a new kernel speedup.

### Learned v2 versus fixed v1 model-quality A/B

The matched two-layer CUDA diagnostic now covers every rate W2--W8. Each arm quantized the same 14 projections from
the first two Llama 3.2 1B decoder layers. Calibration used the first 128 `nm-calibration/LLM` rows through the
production preparation path: 49,725 valid tokens contributed to Hessians, while 1,475 padding positions were excluded.
The held-out comparison used the same dense model and 44 valid tokens; four padding positions were excluded from every
metric. Each rate's v2 table was fitted from 896 deterministically sampled transformed tiles (64 per module), weighted
by the diagonal Hessian, and frozen to exact FP16 bits. Tests ran concurrently on physical GPUs 6 and 7, both PG506-230
`sm_80`, with Python 3.14.6 free-threaded, PyTorch 2.13.0+cu130, CUDA 13.0, and `PYTHON_GIL=0`.

```text
+---+----------+----------+----------+----------+----------+----------+----------+----------+
| W | v1 wRel  | v2 wRel  | v1 KLD   | v2 KLD   | v1 JSD   | v2 JSD   | v1 RMSE  | v2 RMSE  |
+---+----------+----------+----------+----------+----------+----------+----------+----------+
| 2 | 0.336754 | 0.336724 | 0.104966 | 0.060530 | 0.021898 | 0.013271 | 0.725617 | 0.704611 |
| 3 | 0.173287 | 0.173042 | 0.015648 | 0.016310 | 0.003637 | 0.003897 | 0.370942 | 0.375119 |
| 4 | 0.090464 | 0.090238 | 0.003754 | 0.004149 | 0.000931 | 0.001026 | 0.192320 | 0.189186 |
| 5 | 0.048687 | 0.048451 | 0.001204 | 0.000861 | 0.000302 | 0.000216 | 0.105578 | 0.104471 |
| 6 | 0.027454 | 0.027375 | 0.000360 | 0.000421 | 0.000090 | 0.000105 | 0.059077 | 0.057531 |
| 7 | 0.015485 | 0.015469 | 0.000114 | 0.000118 | 0.000028 | 0.000029 | 0.031894 | 0.032677 |
| 8 | 0.009165 | 0.009184 | 0.000040 | 0.000031 | 0.000010 | 0.000008 | 0.019016 | 0.018873 |
+---+----------+----------+----------+----------+----------+----------+----------+----------+
```

```text
+---+--------+--------+--------+--------+---------+-----------+-----------+------------+
| W | v1 T1  | v2 T1  | v1 T5  | v2 T5  | v2/v1 s | Fit start | Fit final | Fit change |
+---+--------+--------+--------+--------+---------+-----------+-----------+------------+
| 2 | 1.0000 | 1.0000 | 0.8636 | 0.8955 | 1.068x  | 1263.2833 | 1256.6553 | -0.525%    |
| 3 | 1.0000 | 1.0000 | 0.9409 | 0.9182 | 1.033x  |  273.0662 |  270.4325 | -0.965%    |
| 4 | 1.0000 | 1.0000 | 0.9727 | 0.9545 | 1.041x  |   62.5136 |   61.6364 | -1.403%    |
| 5 | 1.0000 | 1.0000 | 0.9864 | 0.9909 | 0.939x  |   15.5799 |   15.1193 | -2.957%    |
| 6 | 1.0000 | 1.0000 | 0.9909 | 0.9955 | 0.976x  |    3.6777 |    3.6271 | -1.377%    |
| 7 | 1.0000 | 1.0000 | 1.0000 | 0.9955 | 0.948x  |    1.1816 |    1.1714 | -0.858%    |
| 8 | 1.0000 | 1.0000 | 0.9955 | 1.0000 | 0.930x  |    0.4275 |    0.4259 | -0.395%    |
+---+--------+--------+--------+--------+---------+-----------+-----------+------------+
```

The fitter lowers its own weighted transformed-tile objective at every rate, and v2 lowers mean weight relative-L2
at W2--W7. That proxy improvement does not reliably transfer to the held-out model: v2 worsens final-logit KLD at
W3, W4, W6, and W7, worsens RMSE at W3 and W7, and lowers top-5 overlap at W3, W4, and W7. W2 and W5 are clean
accuracy wins in this sample; W8 improves the output metrics while slightly worsening mean weight relative-L2. The
single concurrent timing samples span `0.930x--1.068x` and are not sufficient performance evidence. Therefore this
experiment does **not** justify making `pgc16-v2` the default or claiming a general quality improvement. Fixed
`pgc16-v1` remains the accuracy-safe default while v2 stays gated for further investigation.
Exact local/live/layer KLD and full-precision summary values are in
[`qvq_compander_two_layer_comparison.csv`](qvq_compander_two_layer_comparison.csv).

### `QVQLinear` lifecycle and prepared inference tables

The production quantized module is `QVQLinear`, a proper `BaseQuantLinear` implementation registered as
`BACKEND.QVQ` for `METHOD.QVQ`/`FORMAT.QVQ`. It declares the canonical QVQ contract (half-step W1--W8, group size `-1`,
`desc_act=False`, `sym=True`, int32 planar words, 16-aligned input/output dimensions), supports standard checkpoint
buffer preallocation, and is selected by both explicit `BACKEND.QVQ` and `BACKEND.AUTO`. `QVQReferenceLinear` is a
non-selectable dense-reconstruction oracle used by accuracy tests and the differentiable training fallback.

The MPS table is prepared after weights have reached their runtime device:

```text
checkpoint load / device placement
                |
                v
      QVQLinear.post_init()
                |
                +-- BaseQuantLinear.post_init()
                |
                +-- validate trellis/SU/SV/bias shapes, dtypes, device
                |
                +-- resolve (device, codec version, exact 256 FP16 bits)
                |          once
                v
       ephemeral prepared table handle
                |
                v
             forward
                |
                +-- PGC16 state mixer
                +-- G[mixed >> 8], G[mixed & 255]
```

The prepared handle is a plain runtime attribute, not a parameter or buffer, so it adds nothing to `state_dict` and
cannot become a checkpoint tensor. `Module._apply()` invalidates it before a device/dtype move; the next `post_init`
or inference rebuilds it for the new runtime device. Direct MPS/MLX kernel callers retain a safe fallback cache keyed
by the normalized codec version and exact table bits (plus device on MPS), while stable tuple metadata has a one-entry
identity hot path. Production `QVQLinear` bypasses that metadata work entirely after `post_init`. The MPS shader binds
the 512-byte table in Metal's constant address space; MLX passes the same prepared immutable array to its unchanged
pair/N4/N8 kernels.

On the 2048x2048 W2--W8, M=1/4/16/32 synchronized matrix, prepared QVQ-v2 versus v1 had median latency ratios of
`0.997` on MPS and `0.997` on MLX relative to the preceding hot-cache implementation. Because individual M=1 launches
are only about 24--52 microseconds, the independent 15-sample run had noisy outliers. A 50-sample, 100-inner-iteration
repeat of the affected W5/W7 rows measured v2/v1 ratios of `1.001`/`0.993` on MPS and `0.997`/`1.003` on MLX. Thus the
prepared learned table is latency-neutral within the 5% gate; its benefit is removing metadata normalization and table
resolution from every production forward and eliminating first-forward table materialization when `post_init` runs.

## Measured distortion evidence

The following is a synthetic codebook experiment, not a model-quality result. It used 16 batches of 128 two-value
vectors drawn from a seeded Gaussian distribution, `L=16`, `V=2`, two-pass tail-biting, and the scale grid 0.8--1.6.
Quantization ran on MPS with the host requesting performance-core QoS. The validation script can emit the complete
machine-readable result to a CI artifact or temporary results directory.

| Bits | HYB-Q9 MSE | PGC16 MSE | PGC/HYB | PGC rel-L2 | PGC SQNR | Payload B/w |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0.07005474 | 0.06852494 | 0.9782 | 0.262603 | 11.61 dB | 0.250 |
| 3 | 0.01926404 | 0.01798953 | 0.9338 | 0.134550 | 17.42 dB | 0.375 |
| 4 | 0.00557561 | 0.00500298 | 0.8973 | 0.070956 | 22.98 dB | 0.500 |
| 5 | 0.00212569 | 0.00148333 | 0.6978 | 0.038636 | 28.26 dB | 0.625 |
| 6 | 0.00212541 | 0.00044549 | 0.2096 | 0.021174 | 33.48 dB | 0.750 |
| 7 | 0.00212541 | 0.00013218 | 0.0622 | 0.011533 | 38.76 dB | 0.875 |
| 8 | 0.00212541 | 0.00004277 | 0.0201 | 0.006560 | 43.66 dB | 1.000 |

This passes the exact 65,536-vector, exact payload, W2 non-regression, and W5--W8 non-plateau gates. The report also
records MAE, RMSE, cosine similarity, maximum and percentile absolute errors, and bias. It does not establish KLD,
top-1/top-5 agreement, perplexity, or task accuracy; those require the four-layer endpoint below.

A zero-table uniform product decoder was also viable but less accurate: its measured MSE was `0.084017` at W2,
`0.002816` at W5, `0.000932` at W6, `0.000309` at W7, and `0.000080` at W8. The shared 512-byte Gaussian table is
therefore retained; it roughly halves the high-rate proxy error without adding checkpoint bytes.

## Alternatives considered

The ranking balances exact stream size, W2--W8 capacity, reconstruction quality, decoder cost, and implementation risk.

| Rank | Strategy | Metadata and decode trade-off | Decision |
|---:|---|---|---|
| 1 | PGC16 global scalar compander | 512-byte shared table; branch-free pair decode; full capacity | Selected |
| 2 | Two learned additive 2D codebooks | About 2 KiB; two vector loads and an add; full capacity | Best fallback for quality |
| 3 | HYB base plus computed residual digits | About 2 KiB plus constants; base lookup and residual add | Fallback if PGC W2 regresses |
| 4 | Uniform product quantizer | No table and cheap decode; measurably higher distortion | Keep as diagnostic baseline |
| 5 | Companded `A2` or structured lattice | Tiny constants and good shaping potential; complex decoder | Research candidate |
| 6 | Fully computed bijective code | No table; lowest metadata; quality and uniqueness need proof | Research candidate |
| 7 | `V=1` scalar trellis for W6--W8 | Small table but twice the trellis steps and a split format | Rejected |
| 8 | W5 QVQ base plus explicit residual planes | Small metadata but two streams and two-stage decode | Rejected |
| 9 | Rate-grown sign-folded 2D LUT | W6/W7/W8 tables are 8/32/128 KiB per module | Rejected for cache/memory |
| 10 | Full 65,536 by 2 FP16 LUT | 256 KiB per module and one lookup | Quality oracle only |

The additive-codebook option follows the broad lesson from
[AQLM](https://arxiv.org/abs/2401.06118) that several small codebooks can span a much larger reconstruction set. The
structured-codebook alternatives are informed by [QuIP#](https://arxiv.org/abs/2402.04396). Kernel evaluation should
also compare the cache and lookup lessons in [FLUTE](https://arxiv.org/abs/2407.10960). These references motivate the
alternatives; none makes PGC16 part of the cited methods.

## Paper, credit, and clean-room boundary

QTIP was introduced by Albert Tseng, Qingyao Sun, David Hou, and Christopher De Sa in
[QTIP: Quantization with Trellises and Incoherence Processing](https://arxiv.org/abs/2406.11235), a NeurIPS 2024
Spotlight paper. The paper is the source for the trellis-coded quantization, bitshift trellis, HYB codebook,
tail-biting approximation, randomized Hadamard incoherence processing, and BlockLDLQ algorithms described here.

> Albert Tseng, Qingyao Sun, David Hou, and Christopher De Sa. “QTIP: Quantization with Trellises and Incoherence
> Processing.” Advances in Neural Information Processing Systems 37 (NeurIPS 2024), Spotlight. arXiv:2406.11235.
> [Paper](https://arxiv.org/abs/2406.11235) · [DOI](https://doi.org/10.48550/arXiv.2406.11235)

YAQA was introduced by Albert Tseng, Zhaofeng Sun, and Christopher De Sa in
[Model-Preserving Adaptive Rounding](https://arxiv.org/abs/2505.22988), an ICML 2026 paper. Version 3 is the source
for the Kronecker-factored full-model Hessian objective, Sketch A/Sketch B, structural-nilpotence analysis, and
symmetric input/output adaptive-rounding equation used by QVQ's optional `rounding="yaqa"` path.

> Albert Tseng, Zhaofeng Sun, and Christopher De Sa. “Model-Preserving Adaptive Rounding.” International Conference
> on Machine Learning (ICML 2026). arXiv:2505.22988v3.
> [Paper](https://arxiv.org/abs/2505.22988) ·
> [Official repository](https://github.com/Cornell-RelaxML/yaqa-quantization)

GPT-QModel uses a planar-only checkpoint contract rather than the official repository's packed representation. PGC16,
the planar tensor layout, and the MPS/MLX kernels are GPT-QModel implementation choices; they are not formats,
codebooks, or kernels claimed by the QTIP paper.

The official QTIP and YAQA reference implementations are GPL-3.0 and GPT-QModel is Apache-2.0, so implementation code
must not be copied. New codec, Viterbi, transform, rounding, and kernel code must be derived from the published
equations/specification and validated against independent mathematical oracles and a dense decoder.

## Selected checkpoint contract

QTIP is not a GPTQ packing variant. GPTQ stores scalar affine integer codes with group scales and optional zero points;
QVQ stores the winning path through a state-transition search as a stream of transition codes. QVQ must never produce
or consume GPTQ `qweight`, `qzeros`, `scales`, or `g_idx` tensors.

| Tensor | Meaning |
|---|---|
| `trellis` (`int32`) | Planar path codes (winning transition labels), tiled over input and output dimensions |
| `SU` | Input-channel sign/scale vector used by the right incoherence transform |
| `SV` | Output-channel sign/scale vector used by the left incoherence transform |
| `bias` | Optional dense bias, unchanged by trellis coding |

`SV` is multiplied after the runtime's left Hadamard, so its existing per-output entries may carry independent signed
output-channel scales without changing tensor shapes or adding an inference operation. That is distinct from an
input-column group scale: `SU` is multiplied before the right Hadamard, and a scale for each input group cannot commute
through that transform or be folded into `SU` or `SV`. Exact input-group scaling therefore requires a separate QVQ
checkpoint-format revision and decoded-GEMM contract; it is not a `pgc16-v2` table feature.

The default-off `module_scale_search` control uses the existing common magnitude in `SV`; it adds no tensor and no
inference operation. For fixed reconstruction `Q`, it proposes the full-input-Hessian optimum
`alpha = trace(Q H W^T) / trace(Q H Q^T)`, rounds the resulting `SV` to its exact checkpoint dtype, accepts only a
strict improvement under the original undamped Hessian, and performs at most one re-encoding at the proposed scale.
The exact original encoding is retained if neither candidate improves. This input-Hessian-only selector is not allowed
with YAQA, whose correct acceptance objective also contains the output-Hessian factor.

There is no serialized `tlut` in the selected contract. The format metadata identifies `pgc16-v1` or `pgc16-v2`; v2
also stores the exact 256 unsigned FP16 bit patterns in `compander_bits`. The learned table is model-level metadata,
not a module tensor. For a 16 by 16 tile, `transition_bits = 2 * rate` is an integer from 2 through 16 and the planar
path-code shape is `[tile_count, 4 * transition_bits]`, equivalently `[tile_count, 8 * rate]`. This is exactly
`rate / 8` bytes/weight before the common transform vectors. Every transition width uses the same low-to-high Pangolin
plane decomposition with aligned 8-, 4-, 2-, and 1-bit subplanes. The former scalar-symbol split is neither emitted
nor accepted.

## Quantization math and processor boundary

For a weight matrix `W` and calibration Hessian proxy `H = X^T X / N`, QTIP first applies randomized Hadamard
incoherence processing:

```text
W_r = V_m S_m W S_n V_n^T
H_r = V_n S_n H S_n V_n^T
```

`S_m` and `S_n` are random sign diagonals and `V_m` and `V_n` are normalized Hadamard transforms, including the paper's
factorization for non-power-of-two dimensions. The processor performs a block LDL decomposition of `H_r`, walks input
blocks backwards, adds already-quantized block error feedback, reshapes each corrected tile into a sequence, and uses
Viterbi search to find its minimum-distortion trellis path. This is the BlockLDLQ rounding boundary; PGC16 replaces
only the state-to-vector reconstruction rule and the corresponding Viterbi codebook values.

This differs from GPTQ in two important ways:

1. The local quantizer is a high-dimensional, stateful TCQ search, not nearest scalar rounding on an affine grid.
2. Incoherence processing makes transformed weights approximately Gaussian before trellis coding. Adding only a QTIP
   codebook after the existing GPTQ column loop would omit a central part of the method.

Quantization still has `2**L` Viterbi states. PGC16 does not increase that search order relative to HYB-Q9; it only
materializes a richer state codebook once and reuses it. A small sampled scale search is allowed, but it must not add a
per-module codebook tensor.

### YAQA v3 rounding in QVQ

BlockLDLQ minimizes the local activation proxy with one input-side factor. YAQA instead approximates the Hessian of
the full-model KL for a dense-orientation weight `W[out, in]` as `H_O ⊗ H_I`. With block decompositions
`H_I = L_I D_I L_Iᵀ`, `H_O = L_O D_O L_Oᵀ`, feedback factors `L_I' = L_I - I`, `L_O' = L_O - I`, and
`Delta = W - W_hat`, the paper's fixed point is:

```text
W_hat = Q(W + L_O'.T @ Delta @ L_I'
                  + L_O'.T @ Delta
                  + Delta @ L_I')

proxy = trace(Delta @ H_I @ Delta.T @ H_O)
```

QVQ stores the transformed inner weight as `A = W.T`, so the implemented equation is the exact transpose:

```text
A_hat = Q(A + L_I'.T @ E @ L_O'
                  + L_I'.T @ E
                  + E @ L_O')
```

where `E = A - A_hat`. Strictly block-lower dependencies allow one bottom-right-to-top-left anti-diagonal schedule;
tiles on the same anti-diagonal are independent and may be Viterbi-quantized together. With `H_O = I`, output
feedback is zero and the implementation is state-for-state identical to BlockLDLQ. Both Hessian factors pass through
the corresponding randomized Hadamard basis before factorization.

The implemented Sketch-B reference consumes one weight-gradient matrix `G_s[out, in]` for each independent sequence:

```text
H_I = E_s[G_s.T @ G_s] / out_features
H_O = E_s[G_s @ G_s.T] / in_features
```

These must be gradients of the full-model KL/Fisher construction described by YAQA, including the paper's independent
sequence and Monte-Carlo output sampling contract. A minibatch-averaged gradient is invalid because its outer product
introduces cross-sequence terms. Ordinary forward activation observations provide `H_I` only and cannot synthesize
`H_O`; `rounding="yaqa"` therefore fails closed when no output Hessian is supplied.

The low-bit diagnostic now implements that contract. For every calibration batch it samples one categorical outcome
from every mask-selected model-output distribution, sums token score losses within each sequence, sums those sequence
losses, runs one backward
pass through the complete source model, and forms each target projection's per-sequence gradient as
`G_s = grad_output_s.T @ input_s`. The batch dimension is retained through both Gram products, so there are no
cross-sequence terms. Gram computation and storage use FP32 with CUDA TF32 disabled, matching the paper's numerical
guidance. Labels from the calibration dataset are deliberately ignored because they would produce the empirical
Fisher rather than the required real Fisher.

The full-model score-gradient pass has training-like activation memory even though no optimizer step or parameter
gradient is produced. Production therefore applies non-reentrant activation checkpointing to every declared decoder
layer, like the official collector: forward activations are recomputed during backward while the target module
identities and FP32 Sketch-B Gram math remain unchanged. The checkpoint context distinguishes recomputation from the
original target hooks, restores every module forward transactionally, preserves RNG state, and records the enabled
state and checkpointed-layer count in telemetry.

Paper-faithful YAQA uses diagonal regularization `1e-4 * trace(H) / n` for each factor. The default safety floor is
2,000 independent sequences because that is the smallest population reported in the paper's Qwen3-8B W2 ablation;
it is an empirical ablation point, not a theoretical or universal minimum. The main paper population is 65,536
independent 2,048-token RedPajama document crops. The official public repository was released two hours after arXiv v1
and has
no later algorithm-code commits, but its quantization CLI defaults to `1e-2` after diagonal-mean normalization. That
100x-stronger value is retained only as the explicit `yaqa_regularization=0.01` author-code control. Future sweeps must
label and test both controls rather than conflating the repository default with the current v3 paper. Summed sequence
scores implement the paper's sequence-level real Fisher and remain invariant to padding and batch grouping. For the
paper's fixed-length contexts, they differ from the official global-token-mean implementation only by a common scalar.

When `--qvq-rounding` includes `yaqa`, `--layers 2` means “quantize the first two layers,” not “truncate the model to
two layers.” The harness loads every source decoder layer so the gradient includes all downstream transformations,
but it stores Sketch-B factors only for the 14 selected projections. Run matched local/full-model rounding arms on the
CUDA host with:

```bash
python scripts/analyze_gptq_low_bit_grid.py \
  --model /path/to/Llama-3.2-1B \
  --layers 2 \
  --method qvq \
  --bits 2 3 4 \
  --qvq-device cuda \
  --qvq-codebooks pgc16-v1 \
  --qvq-rounding block_ldlq yaqa \
  --qvq-yaqa-regularization 1e-4 \
  --calibration-rows 256 \
  --calibration-concat-size 0 \
  --calibration-batch-size 1 \
  --qvq-yaqa-minimum-sequences 256 \
  --json-out docs/qvq_yaqa_two_layer.json \
  --csv-out docs/qvq_yaqa_two_layer.csv
```

This 256-row diagnostic is deliberately below the paper's smallest published ablation and must be labeled as such.
It preserves every source row as one variable-length independent sequence instead of concatenating unrelated rows.
The JSON records source/loaded/target layer counts, independent sequence count, valid Monte-Carlo output count, seed,
factor dtype, TF32 policy, checkpoint policy, loss reduction, sequence floor, and regularization. This makes an accidentally truncated
model, undersampled/averaged-gradient sketch, or mixed paper/author-code control visible in the artifact. The long
Llama run remains assigned to the CUDA host; the Apple host only runs the mathematical and tiny
full-model integration oracles.

YAQA is quantization-only. It selects different states in the existing PGC16 space and adds no inference operation,
checkpoint tensor, or payload bit. Independent per-output `SV` scale optimization and the BlockLDLQ-only diagonal
emission experiment are rejected in YAQA mode because neither is the two-sided coupled objective. The default remains
`rounding="block_ldlq"`. The Ultra lifecycle now collects valid Sketch-B factors in a one-time full-model real-Fisher
prepass before replacing any dense layer, then releases each module's factors after use. YAQA remains opt-in until
held-out two-/four-layer KLD/top-1/top-5 gates demonstrate a model-level win. Exact backward currently requires
`offload_to_disk=False`, an entirely dense single-device source model, and a non-MoE architecture; unsupported shell,
incremental, and conditionally routed cases fail closed instead of substituting activation covariance or
truncated-model gradients.

The deterministic W2 unit fixture below is a small coupled linear/downstream system with a 64-state test codebook,
not model-promotion evidence. It exists to ensure that the implemented two-sided feedback improves more than MSE:

| Rounding | Kronecker proxy | Output KLD | Top-1 agreement | Top-5 overlap |
|---|---:|---:|---:|---:|
| BlockLDLQ | 5.26296234 | 0.62530845 | 0.703125 | 0.834375 |
| YAQA v3 | 4.80313635 | 0.58639497 | 0.718750 | 0.846875 |

The production-layout unit matrix covers every half-step rate from W1 through W8 and requires exact pack/unpack and
Torch reconstruction, exact forward parity, exact top-1 and ordered top-5, bounded forward KLD, and exactly `rate / 8`
payload bytes/weight. It also covers the smallest and largest transition labels, circular state recovery, invalid
quarter-step/nonfinite/bool rates, configuration round-trip, lifecycle preallocation, and deterministic native
dispatch. MPS and MLX execute the full half-step reconstruction matrix on the Apple validation host. CUDA source and
test collection cover the same transition widths; runtime validation must be repeated on a CUDA host after this
format-breaking change.

### W2 tail-biting candidate list

The canonical `L=16`, `V=2` format is unchanged. A W2 transition emits four bits, so every state has exactly 16 legal
successors; no deterministic fixed-rate decoder can provide more outgoing edges without increasing `V`, increasing
the payload rate, or storing another branch selector. The 16-way out-degree is not a 16-solution limit: ordinary
Viterbi retains one survivor for each of 65,536 states and represents exponentially many complete paths.

The production reference previously inherited QTIP Algorithm 4's approximate tail-biting boundary choice: an
unconstrained pass over a half-rotated sequence selected one overlap, followed by one constrained circular pass. QVQ
now exposes `tail_biting_candidates`. Candidate 1 is exactly that historical overlap. When the count is greater than
one, a forward/backward min-sum recurrence ranks additional overlap states at the rotated boundary, each candidate is
run through the exact constrained recurrence, and the path with the lowest reported distortion is retained. Strict
`<` selection makes candidate 1 an exact tie-preserving fallback, so widening the list cannot regress the encoder's
additive objective.

Widening the candidate list changes quantization time only. For any selected rate, it leaves the 16-bit state,
two-value PGC16 decoder, planar path-code stream, `rate / 8` payload bytes per weight, checkpoint tensors, and inference
kernels unchanged. The default candidate count remains 1 until held-out model evidence justifies a wider production
policy; W2 research runs should compare 1, 4, and 8 candidates and report quantization time with KLD/top-1/top-5.

The readable MPS quantizer uses conservative rate-aware search batches for all half steps: 96 tiles at W1/W1.5,
32 at W2--W3.5, 64 at W4--W6.5, 256 at W7/W7.5, and 512 at W8. The integer-rate anchors were measured on the Apple
M4 Max screening workload; half-step values inherit the conservative neighboring regime until separately benchmarked.
This is a quantizer scheduling default, not part of `pgc16-v1`, and an explicit diagnostic option can override it.

The CUDA quantizer uses a persistent native Viterbi operator: one CUDA block owns one sequence for the complete
128-step recurrence, retains only compressed backpointers, and performs traceback in the same launch. This replaces
hundreds of eager PyTorch launches per tail-biting pass. Native dispatch receives the integer transition width rather
than a floating-point rate. CUDA half steps now use independently measured production defaults.

The A/B below was measured on physical GPUs 6 and 7, both runtime-reported PG506-230 `sm_80` devices with 124 SMs
and 96 GiB, using FP32 search, Python 3.14.6 free-threaded, PyTorch 2.13.0+cu130, CUDA 13.0, and 128 vectors per tile.
The eager baseline was freshly forced on the same rebased tip through non-contiguous inputs; the native rows use the
new production batch map. Each timing has warmups and at least ten CUDA-event samples. Speedup is end-to-end
tail-biting tile throughput, including both passes where the rate requires them.

```text
+---+-------------+---------------+--------------+-----------+----------+----------------+------------------+-------------------+
| W | Eager batch | Eager tiles/s | Native batch | Median ms | Tiles/s  | Net speedup    | Peak alloc MiB | State/value/error |
+---+-------------+---------------+--------------+-----------+----------+----------------+------------------+-------------------+
| 2 | 128         | 1588.1        | 512          | 43.158    | 11863.5  | 7.47x          | 1147.0           | exact             |
| 3 | 256         | 1877.6        | 496          | 36.696    | 13516.4  | 7.20x          | 373.0            | exact             |
| 4 | 256         | 1934.8        | 496          | 37.068    | 13380.9  | 6.92x          | 188.9            | exact             |
| 5 | 512         | 2082.8        | 992          | 75.941    | 13062.8  | 6.27x          | 284.6            | exact             |
| 6 | 512         | 2043.7        | 1024         | 85.695    | 11949.3  | 5.85x          | 270.0            | exact             |
| 7 | 256         | 1600.0        | 1488         | 111.275   | 13372.3  | 8.36x          | 383.6            | exact             |
| 8 | 512         | 4240.4        | 2976         | 31.200    | 95383.8  | 22.49x         | 11.6             | exact             |
+---+-------------+---------------+--------------+-----------+----------+----------------+------------------+-------------------+
```

The native emission uses explicit round-to-nearest square/sum operations and the eager cuBLAS K=2 accumulation order.
This is required: a contracted expression changed two W8 states in one adversarial production-length seed. The
regression seed is now permanent. Across W2--W8, three production-length seeds, constrained/unconstrained searches,
and ten repeated launches, native states, reconstructed values, and `squared_error` are bitwise identical to a
same-device eager oracle. The free-threaded two-device and non-default-stream tests also pass. The combined QTIP/QVQ
CUDA suite reports 471 passed and 115 platform skips.

#### W1/W1.5 low-rate batch saturation

The 2026-08-12 follow-up measured the previously conservative W1/W1.5 batch-16 defaults on two isolated PG506-230
`sm_80` GPUs. Python 3.14.6 ran free-threaded, search arithmetic remained FP32, every sequence contained the production
128 vectors, and timings used four warmups plus 20 CUDA-event samples near the selected knees. Increasing only the
number of independent trellises in a launch produced:

```text
+------+-----------+-----------+---------+----------+----------------+-------------------+
| Rate | Batch old | Batch new | Tiles/s | Speedup  | Peak alloc MiB | State/value/error |
+------+-----------+-----------+---------+----------+----------------+-------------------+
| W1   |        16 |       496 | 11527.7 |    7.15x |         1235.2 | bitwise exact     |
| W1.5 |        16 |       496 | 11902.3 |   10.58x |          619.0 | bitwise exact     |
+------+-----------+-----------+---------+----------+----------------+-------------------+
```

W1 first replaced the 64-KiB shared suffix array with two ping-pong FP32 cost vectors. Each thread still visits
prefixes 0--3 in order, preserves strict tie-breaking, and executes the same explicit round-to-nearest emission. W1
and W1.5 then stored their transient 2/3-bit predecessor indices losslessly in `uint8` instead of `int32`. At batch
496 this cuts W1/W1.5 peak allocation from approximately 4.1/2.0 GiB to 1.2/0.6 GiB while also reducing backpointer
write traffic. W1.5 batch 992 is only 1.1% faster than 496 but doubles memory, so both rates select 496. A regression
compares every state, reconstructed value, and reported error from each selected batch against batch-16 chunks. The
two-device free-threaded test runs W1 and W1.5 concurrently.

#### W2.5--W8 rate-specific saturation and W8 recurrence collapse

The 2026-08-12 follow-up benchmarked every half-step with production 128-vector trellises on physical GPUs 6 and 7.
The only changed scheduling dimension was the number of independent trellises per launch. Each selected batch retained
exact PGC16 states and zero measured loss delta versus the corresponding smaller-batch control:

```text
+------+-----------+-------------+-----------+-------------+----------+----------------+
| Rate | Batch old | Tiles/s old | Batch new | Tiles/s new | Speedup  | Peak alloc MiB |
+------+-----------+-------------+-----------+-------------+----------+----------------+
| W2.5 |       512 |     12728.5 |       496 |     13356.2 |    1.05x |          250.9 |
| W3   |      1024 |     13284.1 |       496 |     13516.4 |    1.02x |          373.0 |
| W3.5 |      1024 |     13225.2 |       496 |     13447.0 |    1.02x |          250.9 |
| W4   |      1024 |     13131.3 |       496 |     13380.9 |    1.02x |          188.9 |
| W4.5 |       512 |     11965.2 |       992 |     13048.0 |    1.09x |          315.8 |
| W5   |       512 |     11984.1 |       992 |     13062.8 |    1.09x |          284.6 |
| W5.5 |       512 |     12619.6 |      1488 |     13365.8 |    1.06x |          404.7 |
| W6   |       512 |     11510.9 |      1024 |     11949.3 |    1.04x |          270.0 |
| W6.5 |       512 |     12637.4 |      1488 |     13384.6 |    1.06x |          386.5 |
| W7   |       512 |     12636.2 |      1488 |     13372.3 |    1.06x |          383.6 |
| W7.5 |       512 |     12618.9 |      1488 |     13378.9 |    1.06x |          382.2 |
| W8   |      1024 |     24700.5 |      2976 |     95383.8 |    3.86x |           11.6 |
+------+-----------+-------------+-----------+-------------+----------+----------------+
```

W2.5 now stores its exact five-bit predecessor in `uint8`, reducing peak transient storage at batch 496 from about
619 MiB to 251 MiB. This first pass left the W3--W7.5 recurrence unchanged and changed only their measured batches.
At W8 all 16 state bits shift out at each vector, so the next step depends on one scalar minimum rather than a
65,536-entry survivor vector. The specialized kernel keeps that scalar FP32 recurrence and the selected state for each
step, preserving the prior CUDA kernel's FP32 addition order and lowest-state tie rule while eliminating all cost and
backpointer allocations. A compiled old/new A/B was bitwise exact for every state and reported error. Long 128-step
tests using the production PGC16 table also match eager paths exactly for constrained and unconstrained searches,
weighted costs, two seeds, non-default streams, and Python 3.14.6 with `PYTHON_GIL=0`.

The next increment fused emission, transition, and suffix reduction for W2.5--W7.5. An ordinary transition needs only
one minimum per surviving suffix; materializing all 65,536 state costs in global memory was redundant. The fused
kernel ping-pongs the exact suffix minima in shared memory and retains the same predecessor backpointers, FP32
addition order, and lowest-index tie rules. A separately compiled old/new A/B over all 11 rates, weighted 128-step
paths, and constrained/unconstrained modes was bitwise exact for every state and reported error. The larger throughput
knee is batch 3,968:

```text
+------+---------------+---------------+---------+----------+----------------+
| Rate | Before tile/s | Fused tile/s  | Speedup | vs eager | Peak alloc MiB |
+------+---------------+---------------+---------+----------+----------------+
| W2.5 |       13356.2 |       47612.0 |   3.56x |        - |         1008.5 |
| W3   |       13516.4 |       44248.8 |   3.27x |   23.57x |         1991.8 |
| W3.5 |       13447.0 |       47915.8 |   3.56x |        - |         1007.6 |
| W4   |       13380.9 |       48059.6 |   3.59x |   24.84x |          515.5 |
| W4.5 |       13048.0 |       44901.0 |   3.44x |        - |          269.4 |
| W5   |       13062.8 |       44880.2 |   3.44x |   21.55x |          146.4 |
| W5.5 |       13365.8 |       41774.5 |   3.13x |        - |           86.3 |
| W6   |       11949.3 |       43290.8 |   3.62x |   21.18x |           54.1 |
| W6.5 |       13384.6 |       42650.4 |   3.19x |        - |           38.7 |
| W7   |       13372.3 |       44737.2 |   3.35x |   27.96x |           31.0 |
| W7.5 |       13378.9 |       44284.2 |   3.31x |        - |           27.2 |
+------+---------------+---------------+---------+----------+----------------+
```

The production PGC16 long-path oracle additionally covers every half-step fused shape against eager with exact paths
and bounded FP32 loss (`2e-5` absolute/relative). This changes offline scheduling and transient workspace only; the
selected trellis, checkpoint payload, decoder, effective BPW, and inference path remain unchanged.

The same tip then ran one complete Llama-3.2-1B-Instruct decoder layer (all seven projections) per rate. Both arms used
the first 128 `nm-calibration/LLM` rows, 49,725 non-padding calibration tokens, batch one, and a disjoint held-out row.
The CUDA extension was warm, and all timing boundaries synchronized the device:

```text
+------+---------+------------+------------+----------+---------+----------+--------+
| Rate | Quant s | Validate s | Replay s   | Weight L2 | Fwd KLD | JSD      | Top-1  |
+------+---------+------------+------------+----------+---------+----------+--------+
| W1   | 46.6845 |     0.1354 |     0.0143 | 0.670083 | 0.22330 | 0.034523 | 0.8750 |
| W1.5 | 46.7440 |     0.1487 |     0.0154 | 0.487081 | 0.04223 | 0.008505 | 0.9375 |
+------+---------+------------+------------+----------+---------+----------+--------+
```

This small held-out row is a lifecycle smoke gate, not a replacement for the existing 128-row quality matrix. Since
the optimized encoder is bitwise identical to the prior encoder, the performance change itself cannot alter model
accuracy; the model run verifies transforms, feedback, reconstruction, validation, and replay still compose correctly.

### Viterbi quantization bottleneck profile

A 2026-08-11 Apple M4 Max profile isolated the offline PGC16 quantizer from model inference. The process requested
user-interactive performance-core QoS, capped host math at the 12 performance cores, and used PyTorch 2.5.1 MPS.
Each sample used the production 16x16 tile, 128 two-value vectors, 65,536 states, two-pass approximate tail biting,
the rate-selected MPS tile batch, one excluded warmup, and three timed iterations. CUDA and full-model execution were
not part of this profile.

```text
+---+------------+-----------+---------+---------+----------------------+
| W | Tile batch | Median ms | Tiles/s | ms/tile | Projected 2048x2048 |
+---+------------+-----------+---------+---------+----------------------+
| 2 |         32 |    175.73 |   182.1 |    5.49 |               90.0 s |
| 3 |         32 |    121.28 |   263.9 |    3.79 |               62.1 s |
| 4 |         64 |    281.33 |   227.5 |    4.40 |               72.0 s |
| 5 |         64 |    278.11 |   230.1 |    4.35 |               71.2 s |
| 6 |         64 |    282.53 |   226.5 |    4.41 |               72.3 s |
| 7 |        256 |   1119.58 |   228.7 |    4.37 |               71.7 s |
| 8 |        512 |   2342.01 |   218.6 |    4.57 |               74.9 s |
+---+------------+-----------+---------+---------+----------------------+
```

At these rates, the 237,568 tiles in the seven projections of one Llama 3.2 1B-sized decoder layer require an
estimated 15.0--21.7 minutes of Viterbi work. Two layers require approximately 30.0--43.5 minutes before calibration,
transforms, packing, or evaluation.

The implementation issues 256 emission calculations, 254 predecessor reductions, 254 transition gather/adds, and
254 backpointer writes per tail-biting call. A synchronized operator drill-down is intentionally diagnostic rather
than an additive latency measurement because each inserted synchronization reduces normal MPS overlap:

```text
+----------------------+-------+-------+-------+
| Operation            | W2    | W4    | W8    |
+----------------------+-------+-------+-------+
| Emission distance    | 46.3% | 40.6% | 33.0% |
| Predecessor reduction | 22.4% | 12.9% | 11.2% |
| Transition gather/add | 20.8% | 36.4% | 53.2% |
| Backpointer writes   |  7.3% |  7.2% |  1.6% |
| Traceback            |  2.6% |  2.3% |  0.5% |
+----------------------+-------+-------+-------+
```

Thus 97--99% of the measured device work belongs to the forward dynamic-programming recurrence. Traceback, setup,
and final value materialization are not first-order targets. The main FP32 state tensor is 8 MiB at the selected W2
batch, 16 MiB at W4, and 128 MiB at W8. Retained int32 backpointers consume approximately 63.5 MiB per W2 pass,
7.9 MiB per W4 pass, and 0.25 MiB per W8 pass.

A warmed 128x128 end-to-end CPU diagnostic on the same 12 performance cores attributed 99.73%, 99.72%, and 99.69%
of W2, W4, and W8 quantization time to Viterbi. In contrast, well-conditioned CPU block-LDL factorization took
12.8 ms at width 2,048, 89.2 ms at 4,096, and 447.9 ms at 8,192. An 8192x2048 projection contains 65,536 trellis
tiles and projects to 248--360 seconds of MPS Viterbi work, so eliminating the factorization entirely would save
less than 0.2% in this regime.

W8 admits two specializations because `bits * V = L = 16` and therefore has no overlap state:

1. The provisional tail-biting pass is unused. Returning the unconstrained pass directly is path-exact. The
   implemented fast path reduced the 512-tile MPS median from 2322.38 ms to 1169.33 ms, a 1.986x speedup, with
   bitwise-identical states, reconstructed values, and reported loss.
2. PGC16 spans every pair of the 256 scalar levels exactly once. A direct two-coordinate selection followed by the
   inverse PGC16 permutation measured 0.860 ms versus 2324.6 ms for the original two-pass/gather 512-tile call. It
   changed 6 of 8,192 random states (0.073%) because the recurrence accumulates FP32 rounding, while reducing true
   SSE from 12.98417282 to 12.98417091. This path is a quality-gated research candidate, not yet an exact replacement.

The transition originally loaded `best_cost[:, predecessor_suffix]` through a 65,536-entry int64 gather map even
though state numbering repeats every suffix cost in one contiguous run. The MPS path now expresses that operation as
`repeat_interleave(2**shift)`. The selected-batch matrix retained exact state/value/loss parity and measured the
following median improvements against the preceding gather path:

```text
+---+-----------+-----------+---------+
| W | Gather ms | Repeat ms | Speedup |
+---+-----------+-----------+---------+
| 2 |    174.07 |    135.96 |   1.28x |
| 3 |    120.57 |     83.79 |   1.44x |
| 4 |    277.18 |    155.86 |   1.78x |
| 5 |    273.76 |    151.90 |   1.80x |
| 6 |    278.51 |    156.61 |   1.78x |
| 7 |   1120.54 |    604.71 |   1.85x |
| 8 |   1159.38 |    640.81 |   1.81x |
+---+-----------+-----------+---------+
```

Together with the exact W8 single-pass shortcut, the current W8 path is about 3.66x faster than the original
two-pass gather implementation. CPU and CUDA retain the previously validated gather expression until they are
benchmarked independently.

The selected optimization order is therefore:

1. retain the implemented W8 single-pass shortcut and its exact state/value/loss parity gate;
2. evaluate direct W8 PGC16 selection with reconstruction, KLD, top-1, and top-5 gates;
3. retain the implemented contiguous MPS suffix expansion and its W2--W8 exact-parity gate;
4. fuse PGC16 decoding, emission, transition, and backpointer output in a rate-specialized Metal Viterbi kernel for
   W2--W7;
5. store W2--W4 backpointers in 8 bits and W5--W7 backpointers in 16 bits inside the native kernel;
6. optimize or share Hessian collection and factorization after the recurrence is no longer dominant.

Removing the host-visible transition assertion between the two passes changed measured latency by only -0.1% to
+0.36%, so it is not an optimization target. PyTorch 2.5.1 also lacks native MPS Cholesky, and its automatic CPU
fallback produced non-finite off-diagonal block factors in this environment. Until a native path is available, the
processor must route factorization explicitly to CPU rather than rely on automatic MPS fallback.

## Kernel design and performance gate

The Torch decoder is the numerical reference. MPS and MLX decode adjacent output columns together:

1. read the planar edges and recover one 16-bit state for the output pair;
2. apply the xor-MAD-xor mixer once;
3. load `G[p >> 8]` and `G[p & 255]`;
4. accumulate both values into two output accumulators;
5. apply the inverse transform and existing scale path without materializing dense weights.

Every W1, W1.5, ..., W8 production entry point is compile-time transition-width-specialized, so the planar layout
branches and state-loop trip count fold out of the hot path. For M>=4, one SIMD group decodes a K slice into
256--512 bytes of threadgroup memory;
the other SIMD groups reuse it for up to 16 or 32 input rows. The N8 layout follows the verified Pangolin MPS/MLX
structure: two shared `float4` decode planes, a 256-thread group, and two rows per SIMD group. This bounds accumulator
pressure while halving output groups where the extra decode work amortizes.

Adjacent QVQ states are recovered incrementally in the optimized N4 kernels: after reconstructing state `p`, state
`p+1` is exactly `((state[p] << transition_bits) | path_code[p+1]) & 0xffff`. MLX applies the same recurrence across
all four pairs in N8. This mirrors Pangolin's adjacent planar decode strategy while preserving QTIP's state-transition
math. Synchronized
head-to-head measurements retained bitwise-identical reconstruction and improved all 63 N4 rows on each backend:
median new/old latency was 0.856 on MPS and 0.848 on MLX. MLX rolling N8 improved all 42 measured rows with median
0.719 and range 0.618--0.787.

MPS normally emits four columns. It selects N8 only for W8, M=15--16, and a projection with `max(K, N) >= 8192`, where
both orientations repeatedly improved by about 9%; lower rates, smaller matrices, and M>16 remained on N4 after N8
regressed them. MLX retains pair, N4, and N8 kernels. At M=16 it uses N8 for W4/W8, N4 for W2/W5/W6 and measured
long-K W3/W7 cases, and otherwise keeps the pair path. At M=32 it uses N8 except for W3 and marginal long-K W2/W5--W7
cases. Other row counts keep the preceding pair/N4 policy. All three kernels reconstruct bitwise-identical outputs.

Before calling PGC16 faster than HYB, benchmark both with paired kernels, warmed execution, and identical compiler
settings. Cover M values 1, 4, 16, and 32; representative matrices 2048x2048, 2048x8192, and 8192x2048; and every rate
W2--W8. Report latency, tokens/second, effective weight bandwidth, and speedup. Compare W8 honestly with Pangolin/int8
rather than only with a deliberately unpaired HYB baseline.

The accepted Apple M4 Max matrix used five warmups, seven samples, five synchronized inner iterations, and
performance-core QoS. It covers all 168 combinations of MPS/MLX, W2--W8, M=1/4/16/32, and 2048x2048, 2048x8192, or
8192x2048. Across W2--W5, PGC16/HYB median-latency ratios span 0.014--0.582 on MPS and 0.014--0.520 on MLX, so every
gate row passes. Complete latency distributions and payload bandwidth are generated as CI artifacts rather than
checked into the design-docs tree. HYB remains a deliberately scalar regression oracle, so these ratios demonstrate
the production-kernel optimization rather than an intrinsic codec-only speedup.

In synchronized head-to-head dispatch measurements, the 32 changed MLX matrix rows improved with a median
new/previous latency ratio of 0.909 and a range of 0.712--0.992. The two selected MPS W8/M16 large-projection rows
improved by about 9%. Candidate N8 regimes that did not beat the existing kernel stayed disabled.

Against the pre-optimization 2048x2048 PGC16 kernel, median speedup across W2--W8 was 3.53x/5.67x/15.82x/18.91x on
MPS and 3.40x/5.43x/16.66x/18.76x on MLX for M=1/4/16/32. Rate specialization drives the decode gain; tiled row reuse
drives the larger prefill gains.

W8 is reported separately because Pangolin uses a native int8 path and QVQ should not be presumed faster. In the
current three-shape matrix, PGC16/Pangolin median-latency ratios span 0.308--0.953 on MPS and 0.171--0.877 on MLX;
QVQ won all 24 measured rows. At 2048x2048 the ratios for M=1/4/16/32 were 0.894/0.424/0.582/0.356 on MPS and
0.877/0.592/0.500/0.204 on MLX. This is evidence for these kernels, shapes, and Apple M4 Max—not a presumption that
QVQ generally beats native int8.

Acceptance requires:

- exact Torch/MPS/MLX decoder parity for every rate and adversarial boundary states;
- PGC16 MPS and MLX no more than 5% slower than an equivalently paired HYB kernel at W2--W5;
- no dense materialization or per-token allocation in the optimized path;
- exact `rate / 8` stream bytes/weight and zero serialized per-module LUT bytes;
- monotonic synthetic distortion through W8 and no material W2 regression;
- interim two-layer KLD/top-1/top-5 quality gates passing before enabling model save/load, followed by the four-layer
  gate when faster kernels or a CUDA host make it practical.

If PGC16 misses the performance gate, first tune state reuse, table placement, vector loads, and accumulator layout. If
it still misses, evaluate the zero-table uniform decoder and the HYB-plus-residual fallback with the same benchmark.

## Current implementation and planned changes

| Area | Current implementation | Selected change | Status |
|---|---|---|---|
| Config | `pgc16-v1` fixed table or `pgc16-v2` with exact model-level `compander_bits`; rejects serialized codebook tensors | Keep HYB/uniform out of the loadable contract | Implemented |
| Codec boundary | Production exports PGC16 v1/v2; HYB/uniform live in explicit reference modules | Preserve references without compatibility loaders | Implemented |
| Decoder | Versioned 256-level table plus unchanged xor-MAD-xor state permutation | Require exact 65,536-vector parity | Implemented |
| Learned compander | Hessian-weighted Lloyd/PAVA fit, Gaussian initialization, exact monotone FP16 freeze, degrading-update rejection | Collect one bounded shared model population in a pre-quantization lifecycle pass | Implemented |
| Quantizer core | PGC16 Viterbi, non-regressing tail-biting overlap list, exact W8 single pass, block LDL/BlockLDLQ, default-off guarded module-scale search, YAQA v3 two-sided anti-diagonal rounding, Sketch-B Gram reference, and lifecycle full-model real-Fisher collection | Complete W1/W1.5 real-model BlockLDLQ/YAQA A/B | Lifecycle implemented; model gate pending |
| Stream codec | Whole-edge planar W1, W1.5, ..., W8 path codes, 16x16 tiles, exact `rate/8` B/weight | Reject legacy scalar-split layouts | Implemented |
| QuantLinear | `QVQLinear(BaseQuantLinear)` with standard preallocation, validation, post-init, device-move invalidation, CPU reference, MPS, and CUDA dispatch | Wire model replacement/save/load around this module | Backend lifecycle implemented |
| MPS kernel | Transition-width-specialized E2--E16 pair/rolling-N4 plus narrow large-W8 N8 dispatch; post-init prepared 512-byte table in constant address space | Benchmark half-step latency without weakening numerical gates | Half-step reconstruction parity passes |
| MLX kernel | Transition-width-specialized E2--E16 pair/rolling-N4/rolling-N8 dispatch with tiled reuse and a prepared shared 512-byte level table | Benchmark half-step latency without weakening numerical gates | Half-step reconstruction parity passes |
| Diagnostic | Uses the first 128 `nm-calibration/LLM` rows, excludes masked outputs, accepts half-step W1--W8 for QVQ-only runs, can collect exact per-sequence full-model YAQA-B factors, and reports guarded module-scale decisions while targeting only the first N layers | Run two-layer W1/W2 module-scale and BlockLDLQ/YAQA gates on CUDA, then four layers after acceleration | Large synthetic module-scale gate captured; real-model evidence pending |
| Lifecycle | BlockLDLQ fixed-v1 quantization, module replacement, save/load, and reload inference are enabled | Keep learned v2 outside production lifecycle/configuration | Implemented; v2 retired |
| CUDA | Direct E2--E16 PGC16 planar GEMV/Viterbi source, M>=32 Ampere WMMA, deterministic split-K, and process-cached versioned levels | Run half-step correctness/performance gates on CUDA hardware | Source and test collection complete; runtime pending |

The intended source ownership is:

```text
qvq_codecs/
├── pgc16.py             # production fixed-v1 codec and exact FP16 registry
├── deprecated/
│   ├── pgc16_v2.py      # historical learned-table metadata helpers
│   └── learned_compander.py # historical model-level Lloyd/PAVA fitter
├── hyb_reference.py     # regression oracle, historical benchmark baseline, debugger
└── uniform_reference.py # zero-table capacity/error diagnostic
```

Do not delete the HYB or uniform reference decoder. They are test and research tools, not compatibility formats.
Only `pgc16-v1` is accepted by configuration, protocol compilation, checkpoint loading, and production quantized
linear modules. The following v2 lifecycle description is retained as a historical design record; it is not active.

For `pgc16-v2`, an omitted `compander_bits` requests an accuracy-first prepass. The lifecycle collects the same
masked input Hessians and exact randomized-Hadamard search basis used by the quantizer, samples every eligible module
with a balanced deterministic quota, fits one shared table, freezes it to exact FP16 bit patterns, and only then starts
module encoding. `compander_fit_tiles_per_module`, `compander_fit_max_tiles`, and `compander_fit_iterations` bound
host memory and offline work. `compander_fit_damp_percent` controls the explicit nonnegative diagonal damping used
only while conditioning each captured Hessian for population weighting; its default remains `0.01`. Dynamic mixed
QVQ rates require an explicitly pre-fitted table because one learned population cannot silently represent multiple
rate-specific normalization distributions.

## Staged model-quality gate

The interim gate uses two decoder layers of Llama 3.2 1B and identical held-out prompts for dense and W2--W8 arms.
Four layers remain the later confirmation gate once faster quantization kernels or a suitable CUDA host are available.
Record at every projection and decoder-layer boundary:

- relative weight L2 and Hessian-weighted reconstruction error;
- local module KL with the same dense input and live module KL with propagated quantized input;
- cosine similarity, MSE, SQNR, max error, percentile errors, and activation/outlier drift;
- layer-output KL and final raw-vocabulary-logit KL;
- top-1/top-5 agreement and token-level cross entropy against dense logits.

The reference codec must prove pack/unpack and dense reconstruction parity first. Only then can a KL change be
attributed to QVQ math rather than serialization or a backend. Full-model perplexity and task evaluation are later
acceptance gates, not the inner development loop.

The first W2--W8 matrix collected on 2026-08-11 is **superseded and is not acceptance evidence**. It used four
synthetic calibration prompts and allowed padded token positions into both activation Hessians and output metrics.
The historical summary below is retained only to make that invalidation auditable; its quality numbers must not be
used in comparisons or release decisions. The underlying JSON and CSV files have been overwritten with corrected
measurements.

The replacement contract uses the first 128 rows of `/monster/data/model/dataset/nm-calibration`, configuration
`LLM`, split `train`, prepared by the production calibration path with chat templating, descending length sort, and
concatenation length 2,048. That preparation produces 25 batches, 49,725 valid tokens, and 1,475 padded slots. Only
the 49,725 attention-mask-selected tokens contribute to `X.T @ X`; the denominator is exactly 49,725. The diagnostic
runs the decoder backbone during calibration and therefore does not collect calibration logits. Evaluation inputs,
module outputs, layer outputs, and final logits are likewise filtered by their attention mask before metrics.

The corrected flat summaries are stored in the corresponding W2--W8 CSV files. Per-module, per-layer, and raw-logit
JSON is intentionally kept as a generated CI artifact because it is several megabytes and is not design documentation.

```text
HISTORICAL INVALID DATA — SYNTHETIC CALIBRATION AND PADDED TOKEN CONTAMINATION
+------------------------+----------+------------+-----------+-----------+--------+--------+----------+
| Arm                    | Seconds  | Weight Rel | Logit KLD | Logit JSD | Top-1  | Top-5  | SQNR dB  |
+------------------------+----------+------------+-----------+-----------+--------+--------+----------+
| W2 symmetric GPTQ      |    51.09 |   0.413184 |  6.024426 |  0.473461 | 0.3333 | 0.2000 |   -2.238 |
| W2 adjacent-asymmetric |    78.02 |   0.393448 |  1.031807 |  0.131441 | 0.8333 | 0.7667 |    4.178 |
| W2 PGC16 QVQ           |   370.17 |   0.281709 |  0.165474 |  0.034111 | 0.9375 | 0.8292 |    8.177 |
| W3 symmetric GPTQ      |    58.81 |   0.217964 |  0.203345 |  0.036645 | 0.9375 | 0.8583 |    9.431 |
| W3 adjacent-asymmetric |    59.67 |   0.205965 |  0.196215 |  0.036380 | 0.9375 | 0.8583 |   10.020 |
| W3 PGC16 QVQ           |   344.98 |   0.143587 |  0.025016 |  0.005835 | 0.9375 | 0.9208 |   14.417 |
| W4 symmetric GPTQ      |    69.60 |   0.115346 |  0.025279 |  0.006045 | 0.9792 | 0.9375 |   15.118 |
| W4 adjacent-asymmetric |    69.73 |   0.107011 |  0.021703 |  0.005201 | 0.9792 | 0.9292 |   15.857 |
| W4 PGC16 QVQ           |   333.15 |   0.074804 |  0.005674 |  0.001416 | 0.9792 | 0.9375 |   19.964 |
| W5 symmetric GPTQ      |    84.60 |   0.060856 |  0.006822 |  0.001695 | 1.0000 | 0.9333 |   20.346 |
| W5 adjacent-asymmetric |    78.47 |   0.055790 |  0.007035 |  0.001730 | 1.0000 | 0.9375 |   21.044 |
| W5 PGC16 QVQ           |   336.00 |   0.040229 |  0.001657 |  0.000413 | 0.9792 | 0.9833 |   25.450 |
| W6 symmetric GPTQ      |    94.37 |   0.032666 |  0.003820 |  0.000967 | 0.9792 | 0.9375 |   24.952 |
| W6 adjacent-asymmetric |    86.58 |   0.029993 |  0.002765 |  0.000695 | 0.9792 | 0.9500 |   25.255 |
| W6 PGC16 QVQ           |   342.23 |   0.022687 |  0.000705 |  0.000175 | 0.9792 | 0.9792 |   30.595 |
| W7 symmetric GPTQ      |    92.40 |   0.017774 |  0.001359 |  0.000345 | 1.0000 | 0.9708 |   29.310 |
| W7 adjacent-asymmetric |    84.55 |   0.016497 |  0.000898 |  0.000225 | 0.9792 | 0.9583 |   29.295 |
| W7 PGC16 QVQ           |   392.28 |   0.012792 |  0.000163 |  0.000041 | 1.0000 | 0.9958 |   35.578 |
| W8 symmetric GPTQ      |    94.17 |   0.009771 |  0.000284 |  0.000071 | 1.0000 | 0.9750 |   33.485 |
| W8 adjacent-asymmetric |    83.83 |   0.009142 |  0.000262 |  0.000066 | 1.0000 | 0.9958 |   33.684 |
| W8 PGC16 QVQ           |   315.43 |   0.007570 |  0.000059 |  0.000015 | 1.0000 | 0.9958 |   40.027 |
+------------------------+----------+------------+-----------+-----------+--------+--------+----------+
```

No speedup or quality conclusion may be drawn from the superseded table. The corrected W2--W8 artifacts below use
the 128-row, padding-excluded contract and replace it as acceptance evidence.

Corrected matrix on the same 2026-08-11 CUDA/Python environment:

```text
+------------------------+---------+-----------+--------+--------+
| Arm                    | Seconds | Logit KLD | Top-1  | Top-5  |
+------------------------+---------+-----------+--------+--------+
| W2 symmetric GPTQ      |   57.42 |  7.326883 | 0.2273 | 0.1364 |
| W2 adjacent-asymmetric |   89.10 |  1.056947 | 0.9091 | 0.7955 |
| W2 PGC16 QVQ           |  377.78 |  0.104966 | 1.0000 | 0.8636 |
| W3 symmetric GPTQ      |   64.90 |  0.205386 | 0.9545 | 0.8591 |
| W3 adjacent-asymmetric |   66.98 |  0.199719 | 0.9773 | 0.8773 |
| W3 PGC16 QVQ           |  354.56 |  0.015648 | 1.0000 | 0.9409 |
| W4 symmetric GPTQ      |   70.88 |  0.024610 | 0.9773 | 0.9227 |
| W4 adjacent-asymmetric |   71.21 |  0.022460 | 1.0000 | 0.9318 |
| W4 PGC16 QVQ           |  334.04 |  0.003754 | 1.0000 | 0.9727 |
| W5 symmetric GPTQ      |   80.96 |  0.005283 | 1.0000 | 0.9500 |
| W5 adjacent-asymmetric |   84.63 |  0.005315 | 0.9773 | 0.9545 |
| W5 PGC16 QVQ           |  344.92 |  0.001204 | 1.0000 | 0.9864 |
| W6 symmetric GPTQ      |   86.09 |  0.006068 | 1.0000 | 0.9500 |
| W6 adjacent-asymmetric |   92.22 |  0.001835 | 1.0000 | 0.9591 |
| W6 PGC16 QVQ           |  351.37 |  0.000360 | 1.0000 | 0.9909 |
| W7 symmetric GPTQ      |   89.03 |  0.001391 | 1.0000 | 0.9682 |
| W7 adjacent-asymmetric |   92.39 |  0.000757 | 1.0000 | 0.9591 |
| W7 PGC16 QVQ           |  399.06 |  0.000114 | 1.0000 | 1.0000 |
| W8 symmetric GPTQ      |   88.51 |  0.000292 | 1.0000 | 0.9773 |
| W8 adjacent-asymmetric |   84.11 |  0.000227 | 1.0000 | 0.9955 |
| W8 PGC16 QVQ           |  319.30 |  0.000040 | 1.0000 | 0.9955 |
+------------------------+---------+-----------+--------+--------+
```

The regenerated W2--W8 artifacts record 128 source rows, 25 prepared 2,048-token batches, 49,725 valid calibration
tokens, 1,475 excluded padding slots, 44 valid evaluation tokens, and four excluded evaluation padding slots. All
21 arms are finite. Against the better GPTQ KLD at each rate, PGC16 improves W2--W8 by 10.07x, 12.76x, 5.98x,
4.39x, 5.10x, 6.64x, and 5.61x respectively. PGC16 retains top-1 `1.0` at every rate.

The GPTQ W2--W8 symmetric/asymmetric measurements and the reason valid QVQ endpoint rows remain blocked are recorded
in [the four-layer diagnostic report](gptq_qvq_four_layer_diagnostic.md). Scalar and nearest-codebook proxy results
must never be labeled as QVQ.

## Decision and implementation log

| Date | Change | Evidence or reason |
|---|---|---|
| 2026-08-11 | Selected one planar format for all W2--W8; no backward compatibility | Simplifies storage and kernels |
| 2026-08-11 | Identified HYB-Q9's 1,024-vector/5-bit ceiling | Capacity proof and measured W6--W8 MSE plateau |
| 2026-08-11 | Selected `pgc16-v1` over nine alternatives | Full capacity, 512-byte shared table, lowest tested proxy MSE |
| 2026-08-11 | Kept HYB and uniform as explicit reference codecs; only PGC16 is loadable | Preserves regression/debug value without format compatibility |
| 2026-08-11 | Implemented frozen PGC16 levels, versioned rate scales, and zero-LUT Torch reconstruction | Exact capacity and zero serialized codebook gates pass |
| 2026-08-11 | Implemented paired PGC16 MPS and MLX kernels | Torch/MPS/MLX reconstruction parity passes W2--W8 |
| 2026-08-11 | Measured 2048x2048 PGC16 versus HYB and Pangolin W8 | W2--W5 gate passes; Pangolin wins six of eight W8 rows |
| 2026-08-11 | Added rich model diagnostic columns including KLD and top-1/top-5 | Four-layer model run remains pending |
| 2026-08-11 | Ported CUDA GEMV/WMMA from serialized HYB to process-cached PGC16 levels | Restores the one-codec production contract; W2--W8 FP16/BF16 validation passes on `sm_80` |
| 2026-08-11 | Added measured rate-aware MPS trellis batch defaults and per-module progress | Reduces W3--W8 diagnostic time without changing quantization math |
| 2026-08-11 | Stopped slow Apple model runs and moved the interim gate to two layers on CUDA | Preserves matched metrics while deferring four-layer cost until acceleration |
| 2026-08-11 | Added deterministic split-K CUDA inference for narrow real projection shapes | 48 DeepSeek V4 Flash W4 rows pass MSE/KLD/top-1 gates; see `qvq_cuda.md` |
| 2026-08-11 | Enabled the CUDA reference quantizer with measured rate-aware batches | 6.61--8.75x throughput over batch 16 with exact trellis paths and at most 1.43 GiB reserved |
| 2026-08-11 | Specialized W2--W8 Apple kernels and shared decoded slices across row tiles | 3.40--18.91x median speedup by row regime versus the pre-optimization PGC16 kernels |
| 2026-08-11 | Widened MPS multi-row output and added measured pair/N4 MLX dispatch | Exact reconstruction retained; all three Apple matrix shapes pass |
| 2026-08-11 | Borrowed Pangolin's bounded-register N8 layout for measured QVQ regimes | MPS large W8/M16 improves about 9%; 32 changed MLX rows improve 0.8--28.8% with exact reconstruction |
| 2026-08-11 | Rolled adjacent trellis states instead of rescanning each planar window | MPS/MLX N4 median latency improves 14.4%/15.2%; MLX N8 improves 28.1%; all measured rows win with bitwise parity |
| 2026-08-11 | Removed W8's unused provisional tail-biting pass | MPS batch-512 median improves 2322.38 to 1169.33 ms (1.986x) with exact state/value/loss parity |
| 2026-08-11 | Replaced MPS Viterbi's int64 suffix gather with contiguous expansion | W2--W8 improve 1.28--1.85x with exact parity; combined original-to-current W8 gain is about 3.66x |
| 2026-08-11 | Invalidated the initial two-layer W2--W8 quality reports | They used four synthetic prompts and included masked padding in Hessians and logit metrics; replacement uses 128 `nm-calibration/LLM` rows and mask-selected tokens only |
| 2026-08-11 | Superseded the earlier W2--W8 acceptance entries | Their synthetic calibration and padding-contaminated metrics are retained only as historical records and confer no acceptance status |
| 2026-08-11 | Named the QTIP-derived planar PGC16 system QVQ and added `pgc16-v2` | Exact learned FP16 model-level metadata preserves the v1 mixer, stream, and decoder operations |
| 2026-08-11 | Implemented Hessian-weighted Lloyd/PAVA compander fitting | Deterministic updates never accept higher weighted error and preserve 256 unique levels/65,536 vectors |
| 2026-08-11 | Routed learned tables through Torch, MPS, MLX, and CUDA inference | No module table tensor; Torch/MPS/MLX KL/top-k/reconstruction and repeated-launch tests pass |
| 2026-08-11 | Benchmarked QVQ-v2 versus fixed PGC16-v1 on Apple | Median v2/v1 is 0.995 across 16 W2--W5 rows; maximum is 1.029 and every row passes the 5% gate |
| 2026-08-11 | Replaced eager CUDA Viterbi scheduling with one persistent block per tile | W2--W8 gain 5.50--7.94x over current-tip eager; states, values, and error are bitwise exact |
| 2026-08-11 | Replaced backend-specific module classes with selectable `QVQLinear(BaseQuantLinear)` | Explicit/AUTO selection, checkpoint preallocation, post-init, CPU/MPS dispatch, state-dict, and device-move tests pass |
| 2026-08-11 | Prepared the v2 table in `QVQLinear.post_init` and bound MPS levels through constant address space | No per-forward metadata lookup; W2--W8 MPS/MLX parity passes and high-sample W5/W7 v2/v1 ratios remain 0.993--1.003 |
| 2026-08-11 | Completed matched two-layer CUDA v1/v2 model-quality A/B for W2--W8 | V2's lower fitting objective does not transfer consistently; v1 remains the accuracy-safe default |
| 2026-08-11 | Added format-preserving W2 accuracy controls | Fixed-trellis per-output `SV` scales and conditioned-Hessian diagonal Viterbi are default-off; the Euclidean path remains an exact full-Hessian acceptance control, all changed Python lines are covered, and W2 model A/B is pending |
| 2026-08-12 | Added guarded module-scale search and a large paired W1/W2 gate | One existing-`SV` scalar plus at most one re-encoding leaves inference storage/math unchanged; 20-seed synthetic evidence rejects W1 promotion and supports only a default-off W2 real-model hypothesis |
| 2026-08-12 | Added a non-regressing tail-biting overlap candidate list without changing `L16/V2` | Full-list small-trellis search matches an exhaustive circular oracle; a fixed W2 case improves while candidate 1 remains the exact historical control |
| 2026-08-12 | Implemented clean-room YAQA v3 rounding and Sketch-B reference math inside QVQ | Independent Gram/proxy oracles pass; `H_O=I` is exactly BlockLDLQ; the two-sided fixed point, planar reconstruction, KLD, top-1, and top-5 fixture gates pass without inference changes |
| 2026-08-12 | Enabled valid YAQA-B collection in the two-layer diagnostic | The full source model stays loaded, one real-Fisher score gradient is retained per independent sequence, both factors match an independent autograd oracle, and only the first two layers are quantization targets |
| 2026-08-12 | Completed dense-model YAQA-v3 lifecycle wiring for W1/W1.5 testing | One prepass collects exact per-sequence full-model Sketch-B factors before replacement, factors are consumed and released per module, and shell/MoE/incremental cases fail closed; default remains BlockLDLQ |
| 2026-08-12 | Extended the unchanged QVQ codec and native backends to W1 | W1 packing, quantization, reconstruction, YAQA, CUDA inference/Viterbi, two-GPU free-threaded safety, real DeepSeek shapes, and two-layer held-out quality pass; fixed v1 remains default |
| 2026-08-12 | Replaced scalar-split symbols with one whole-edge planar path code and enabled every half-step W1--W8 | Exact integer transition widths E2--E16 give 8--64 words per tile and exact `rate/8` B/weight; CPU, MPS, and MLX reconstruction parity pass, while CUDA runtime validation is pending |
| 2026-08-12 | Extended the model diagnostic to QVQ half-step rates | `--method qvq --bits 1 1.5 ... 8` is canonicalized exactly; mixed GPTQ/EXL3 arms reject fractional rates rather than misreporting them |
| 2026-08-12 | Completed the QVQ-only software identity migration | Config/checkpoint values, Python and native symbols, filenames, scripts, tests, and environment controls now use QVQ; QTIP remains only for paper attribution and the HYB reference oracle |
| 2026-08-12 | Screened, documented, then removed exact-payload scalar `L16/V1` at W1 and alternating 1/2-bit transitions at W1.5 | Same 1.0/1.5 bpw payload and negligible shared-table VRAM, but both 1,024- and 4,096-level arms regress held-out KLD/top-k versus fixed PGC16-v1; notes stay here and prototype commit `e788eb6d` is recoverable, while unsupported quantizer code is absent from active tip |
| 2026-08-12 | Rejected centered-rectangle and dual-antipodal `L16/V2` successor remaps | Zero local centroid and unchanged payload/table/decoder complexity still regressed held-out screens because centering reduced current-transition magnitude control |
| 2026-08-12 | Added W1/W1.5 shrinkage to the opt-in per-output `SV` optimizer | Half-strength damped input-Hessian corrections use only existing `SV`; 12/12 held-out KLD wins at both rates and both 8/1,024-row calibration, with zero inference storage/math change; remains default-off pending two-layer model evidence |


## 2026-08-12 — Fused CUDA Hadamard transform for QVQ inference (decode 216 -> ~55-65 ms/step)

### Problem
QVQLinear applies the randomized Hadamard (RHT) incoherence transform twice per
matmul via `matmul_hadU`/`matmul_hadU_stable`, which are Python-loop butterflies
(11-13 stages x ~4 tensor ops each). On Llama-3.2-1B W4 decode this launched
~5,000 tiny kernels per step: 216 ms/step total, of which the transform overhead
was ~200 ms (kernel-level QVQ GEMV itself is only ~1 ms/step).

### Change
New `qvq_hadamard_cuda.cu` in the QVQ CUDA extension:
- one kernel launch per transform (row-per-block, dynamic shared memory);
- butterfly stages in ascending bit order (1, 2, ..., n/2) matching
  `matmul_hadU`'s view recursion exactly; float add/sub with output rounding
  (torch elementwise semantics);
- two normalization modes mirroring the Python references bitwise:
  - mode 0 (width >= 2048, `matmul_hadU_stable`): divide by fp16(sqrtf(n)) with
    IEEE fp32 division (on-device divisor tensor semantics);
  - mode 1 (narrow widths, `matmul_hadU`): multiply by the fp32-rounded
    reciprocal 1/sqrtf(n), rounding the result to fp16 before the epilogue
    (CPU-scalar divisor path semantics);
- fused SU pre-scale, SV post-scale, and bias epilogues with the same rounding
  sequence as the Python forward (removes 3 elementwise launches per matmul);
- gated on `not self.training` so the differentiable Python butterfly remains in
  the training/reference path; non-pow2 / CPU / bf16 fall back to Python.

### Validation (GPU 0/1, PG506-230, W4 Llama-3.2-1B)
- kernel vs Python butterfly bitwise-identical for modes 0/1 x pre/post/bias x
  n in {256, 512, 1024, 2048, 8192} x rows {1, 64} plus a 10M-element sweep;
- model logits bitwise identical to the original Python path (max diff 0.0);
- tests/test_qvq_cuda.py: 492 passed, 2 skipped (incl. training differentiable
  reference path);
- decode/step: 216 ms -> 53.7 ms min / 64.9 ms median (3.3-4.0x);
  prefill 128 tok: 224 ms -> 79.8 ms (2.8x). Host is multi-tenant/noisy.

### Next
- M=1 GEMV decode kernel rework (register decode + k-split FMA; current kernel
  uses 16/256 threads for the dot product at M=1);
- model-level CUDA graph / launch-overhead reduction for the remaining ~55-65
  ms/step.


## 2026-08-12 — CUDA-graph decode + capture-safe QVQLinear forward

### Change
- QVQLinear.forward skips the `torch.isfinite(output).all()` device sync while the
  current CUDA stream is capturing (`torch.cuda.is_current_stream_capturing()`),
  making one decode step capturable into a CUDA graph.
- `scripts/benchmark_qvq_decode.py`: reproducible decode benchmark (eager vs
  CUDA-graph replay at an identical cache state, logits bitwise-equality check;
  the eager attention mask is made capture-safe inside the script).

### Speedup table (W4 Llama-3.2-1B, PG506-230, GPU 0)

| stage | change | decode ms/step (min / median) | vs baseline |
|---|---|---|---|
| baseline (b01cadfd) | Python-loop butterfly transforms | 216 / ~216 | 1.0x |
| c98e6c6a | fused CUDA Hadamard kernel | 51.6 / 52.3 | 4.1x |
| this commit | + CUDA-graph decode replay | 14.8 / 14.8 | 14.6x |

The 216 ms baseline: ~200 ms was the Python butterfly transform launch
overhead (~5000 tiny kernels/step); the QVQ GEMV kernels themselves are only
~1 ms/step. The graph replay removes the remaining ~1850 launches/step
(12.5 ms CPU) and stream syncs; GPU work is ~15 ms/step.

### Validation
- graph replay logits bitwise identical to eager forward at the same cache
  state (torch.equal on int16 views, True);
- tests/test_qvq_cuda.py: 492 passed, 2 skipped;
- no VRAM-destructive caching: graphs capture kernel launch sequences only;
  weights stay packed 4-bit on device.


## 2026-08-12 — Row-specialized GEMV decode kernel (register decode, fewer barriers)

### Change
qvq_gemv_kernel / qvq_gemv_splitk_kernel now take a compile-time ROWS
specialization ({1, 8, 16, 32}, dispatch rounds M up). Each thread decodes the
weight it needs into a register (no shared decoded_weight round trip) and
accumulates ROWS row dot products per k-tile, so all 256 threads stay busy at
M=1 and each k-tile needs one __syncthreads instead of three. ncu showed the
old kernel was 65% barrier-stall bound. The k-slot partials reduce
deterministically (kslot 0..15 order). Double-buffered trellis prefetch was
tried and reverted (regressed the split-K path; the loads were not the
bottleneck).

### A/B (PG506-230, same-condition, min ms per qvq_cuda_gemv call, W4)

| shape | old (b01cadfd) | new | speedup |
|---|---|---|---|
| 2048x2048 M1 | 0.08 | 0.07 | 1.14x |
| 2048x8192 M1 | 0.17 | 0.15 | 1.13x |
| 8192x2048 M1 | 0.15 | 0.13 | 1.15x |
| 2048x2048 M16 | 0.10 | 0.10 | 1.00x |
| 2048x8192 M16 | 0.23 | 0.22 | 1.05x |

Model-level (W4 Llama-3.2-1B, graph replay): 14.8 -> 13.6 ms/step (1.09x),
logits bitwise equal, 492/492 tests. Accuracy vs dense reference: MSE <= 1e-5
(inference tolerance 1e-3).


## 2026-08-12 — Batched k-loop (U=8) for the GEMV decode kernels

### Change
Both qvq_gemv_kernel and qvq_gemv_splitk_kernel batch kUnroll=8 k-tiles per
sync pair: all 8 tiles' trellis words + input rows are loaded before one
barrier, then all 8 are decoded (register decode) and FMA'd before the next
barrier. The batched loads overlap with 8x the per-tile decode work, hiding the
global load latency that dominated the per-tile barrier waits (ncu had shown
65% barrier stalls with one tile per sync pair).

### A/B (PG506-230, same-condition, min ms per qvq_cuda_gemv call, W4)

| shape        | old (cabc2a49) | U=8 | speedup |
|--------------|----------------|-----|---------|
| 2048x2048 M1 | 0.07           | 0.07| 1.00x   |
| 2048x8192 M1 | 0.15           | 0.10| 1.50x   |
| 8192x2048 M1 | 0.13           | 0.10| 1.30x   |
| 2048x8192 M16| 0.22           | 0.21| 1.05x   |

Model-level (W4 Llama-3.2-1B, CUDA-graph decode): 14.5 -> 10.6 ms/step
(1.37x); logits bitwise equal; 502/502 tests. Cumulative vs the 216 ms
transient-decode baseline: 20.4x.


## 2026-08-12 — Session summary: W4 QVQ decode 216 ms -> 10.6 ms/step (20.4x)

Cumulative changes on PR #244 (all bitwise-exact vs the Python reference; the
1e-3 inference tolerance is met with margin — differences are 0):

| commit | change | decode ms/step (graph) | vs 216 ms baseline |
|---|---|---|---|
| c98e6c6a | fused CUDA Hadamard kernel (Python butterfly -> 1 launch) | 14.8 | 14.6x |
| c12e9827 | CUDA-graph capture-safe decode | 14.8 | 14.6x |
| cabc2a49 | row-specialized GEMV (register decode, 1-2 syncs/k-tile) | 13.6 | 15.9x |
| e5921d2a | batched k-loop U=8 (hide trellis load latency) | 10.6 | 20.4x |

Validation: tests/test_qvq_cuda.py 502 passed / 2 skipped; graph replay logits
bitwise equal to eager; per-call MSE vs dense reference <= 1e-5. Tried and
reverted: double-buffered trellis prefetch (regressed split-K), EPP-register
Hadamard (no gain, M=1 latency-bound). Remaining headroom: hadamard fusion
into the GEMV kernel (~4.8 ms/step) and attention/norm (~1.9 ms/step).

## 2026-08-12 — Hadamard latency fix: padded shared + 1024-thread block

ncu showed the hadamard kernel was 1-block/8-warp latency-bound (12.5%
occupancy, 36% wait stalls, 2-way shared bank conflicts). Two changes:

| change | pure kernel duration (n=2048, M=1) |
|---|---|
| baseline (256 threads, unpadded) | 18.9 us |
| + padded shared layout (i + i/32, conflict-free) | 17.9 us |
| + 1024-thread block (50% occupancy) | 9.4 us (2.0x) |

Model decode (CUDA-graph, W4 Llama-3.2-1B, PG506-230):

| config | decode ms/step | vs 216 ms baseline |
|---|---|---|
| before (e5921d2a) | 10.59 | 20.4x |
| after (padded + 1024t) | 8.06 | 26.8x |

Bitwise-validated modes 0/1 x n in {256..8192} x rows {1,4}; 502/502 tests;
graph logits bitwise equal to eager.

## 2026-08-12 — Cache QVQLinear SU/SV/bias dtype conversions

Per-forward `.to(compute_dtype)` of the constant codec tensors (SU, SV, bias)
launched ~500 tiny copy kernels per decode step (~1.4 ms). A lazy cache keyed
by (buffer name, dtype chain) with identity + _version invalidation converts
each constant once (the fp16->fp32 chains keep the exact rounding of the
previous per-forward `.to(a).to(b)` sequence).

| config | graph decode ms/step | vs 216 ms baseline |
|---|---|---|
| before (5207b085) | 8.06 | 26.8x |
| after (cached casts) | 7.50 | 28.8x |

Eager decode 43-48 -> 39 ms. 502/502 tests; graph logits bitwise equal.

## 2026-08-12 — GEMV k-loop U=8 -> U=16

Doubling the batched k-tiles per barrier pair halves the barrier count again
(loads stay overlapped with 2x the decode work). Standalone A/B (min ms,
same conditions):

| shape | U=8 | U=16 |
|---|---|---|
| 2048x2048 M1 split6 | 0.072 | 0.070 |
| 2048x8192 M1 | 0.105 | 0.100 |
| 8192x2048 M1 split6 | 0.108 | 0.103 |
| 8192x8192 M1 split6 | 0.267 | 0.253 |

Graph decode 7.50 -> 7.41 ms/step (29.2x vs 216 ms baseline). 502/502 tests;
graph logits bitwise equal.

## 2026-08-12 — GEMV k-loop U=16 -> U=24

U=32 overflows static shared for ROWS=32 (52.5 KB > 48 KB); U=24 fits
(44 KB) and is the practical max. Standalone A/B (min ms):

| shape | U=16 | U=24 |
|---|---|---|
| 2048x2048 M1 split6 | 0.070 | 0.070 |
| 2048x8192 M1 | 0.100 | 0.099 |
| 8192x2048 M1 split6 | 0.103 | 0.102 |
| 8192x8192 M1 split6 | 0.253 | 0.242 |

Graph decode 7.41 -> 7.22 ms/step (29.9x vs 216 ms baseline). 502/502 tests;
graph logits bitwise equal.

## 2026-08-12 — Fused RMSNorm in the decode harness

LlamaRMSNorm chained 7 kernels per norm (fp32 copy, pow, mean, add, rsqrt,
two muls, fp16 copy) - ~0.5 ms/step total. torch.nn.functional.rms_norm is a
single fused kernel, bitwise identical on [1,1,2048] fp16 (max abs diff 0.0)
and ~4x faster; capture-safe single graph node.

| config | graph decode ms/step | vs 216 ms baseline |
|---|---|---|
| before (7647ea9c) | 7.22 | 29.9x |
| after (fused rmsnorm) | 6.83 | 31.6x |

Eager 37.1 ms. 502/502 tests; graph logits bitwise equal.
