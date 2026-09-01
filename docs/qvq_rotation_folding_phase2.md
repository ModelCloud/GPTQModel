# QVQ rotation folding phase 2: target material decode gains

Date: 2026-09-01. This note continues the A0--A30 search documented in
`docs/qvq_rotation_folding_mlx.md`.

## Why phase 2 exists

A25 validates that exact offline folding can survive QVQ quantization and reach
production packed inference. It removes the head-local V-output and O-input
Hadamards, reducing 14 to 12 online Hadamards per decoder block at identical
`2.054419` W2 effective BPW. Across three disjoint validation streams (5,630
tokens), packed A25 and A0 KL are statistically tied: `0.916587` versus
`0.917579`, with paired bootstrap KL delta median `-0.001236` and 95% interval
`[-0.028447, +0.026979]`. Containment metrics currently trend toward A0, so A25
is a quality/runtime Pareto candidate rather than an accuracy win.

The important systems result is that removing only two Hadamards already
produces a measurable packed decode signal on SM80: A25 is 1.90% faster at B1
and 4.1--4.4% faster at B2--B8 median decode. This validates the premise that
removing transform work can matter end to end.

A 14 -> 12 reduction is useful validation, but it is not the final target.
Phase 2 should aim for a materially larger reduction while retaining A0-class
post-quant quality.

## Phase-2 target

Primary target:

- reduce 14 online full Hadamards per block to **8 or fewer**, with a stretch
  target of **5 or fewer**;
- obtain at least **10% packed B1/B2 decode improvement** on the same SM80
  measurement protocol, or an equivalent gain on the Apple M4 Max once the
  transformed graph is integrated end to end;
- keep effective BPW unchanged unless a metadata increase is explicitly
  reported and wins on an equal-EBPW comparison;
- preserve exact dense-model equivalence for every operation described as a
  fold;
- require multi-stream and independent-seed post-quant evidence before
  promotion.

The objective is not simply to minimize transform count. The winning point is
the fastest architecture whose propagated/full-model QVQ quality remains
statistically compatible with A0.

## Main lesson from A0--A30

The failed aggressive arms do **not** show that large transform reduction is
impossible. They show that replacing QVQ's independent RHT coordinate systems
with fixed graph-level bases without re-optimizing those bases damages the
structured PGC/trellis quantization manifold.

In particular:

- A1 proved that a fixed global residual fold is exact in dense arithmetic but
  loses post-quant quality.
- A3/A20/A21 proved that simple fixed RoPE-compatible Q/K maps do not replace
  the recovery of the full Q/K output Hadamards.
- A23 proved that deleting the down-input transform is unacceptable.
- A27/A29 proved that naive identity/permutation replacements around SwiGLU do
  not retain propagated quality.
- A25 proved that a transform pair can be folded when the graph symmetry and
  the quantization basis are both favorable.

Phase 2 therefore needs to optimize **where transforms are shared and what
basis QVQ quantizes in**, rather than merely deleting Hadamards.

## A31: sibling-shared input RHT + A25

This is the highest-priority low-risk arm.

Q, K, and V consume the same attention input activation. Today each QVQLinear
independently computes an input RHT. Instead, compute one layer-local shared
orthogonal transform:

```
z_attn = x_attn U_attn
```

and quantize all three projections in that common input basis:

```
Wq_tilde = U_attn^-1 Wq Tq
Wk_tilde = U_attn^-1 Wk Tk
Wv_tilde = U_attn^-1 Wv Tv
```

The original residual activation remains available for the skip path; only the
side buffer is transformed. The same idea applies to the MLP input shared by
`gate_proj` and `up_proj`:

```
z_mlp = x_mlp U_mlp
```

The existing input recovery is ordered as `(x * SU) H`, so a single shared
input transform requires bit-identical stored `SU` within each sibling group.
The current fixed-seed fitter already produces that state because `SU` is a
deterministic random sign vector of the input width; inference itself uses no
RNG. The runtime must verify equality and fail closed rather than assuming it.
Trellis paths, `SV`, and output transforms remain module-local. Future learned
module-local input recovery would require an explicitly supported post-shared-H
diagonal (`z_shared * D_alpha_i`); an arbitrary pre-H `SU_i` cannot commute
through the shared Hadamard.

Transform count with the already-validated A25 V/O fold:

- Q/K/V input: 3 -> 1, saving 2;
- gate/up input: 2 -> 1, saving 1;
- V output + O input: 2 -> 0, saving 2;
- total: **14 -> 9 full Hadamard applications/block**.

This is less restrictive than A1 because the basis is layer-local and
branch-local rather than one fixed residual basis for the entire model. It is
also exact before quantization.

### A31 SM80 result

A31 is implemented and validated on the full 16-layer Llama 3.2 1B W2 P32
model. The runtime uses plan-declared, role-agnostic sibling groups and keeps
the original packed `QVQLinear` payloads unchanged. Installation validates
width, input-Hadamard state, and bit-identical stored `SU`; execution fails
closed on duplicate, out-of-order, or interleaved consumers.

The result is exact relative to A25 at quantization and packed inference:

| Metric | A25 | A31 |
| --- | ---: | ---: |
| Online full H/block | 12 | **9** |
| Shared groups/model | 0 | **32** |
| Effective BPW | `2.054419` | `2.054419` |
| Dense logits rel-L2 | `1.1171e-6` | `1.1171e-6` |
| Packed final KL | `0.916587` | `0.916587` |
| Packed logits rel-L2 | `0.494783` | `0.494783` |
| Packed Top-1/5/10 | `.54885/.81812/.88082` | `.54885/.81812/.88082` |

All 112 module payload-hash dictionaries are identical, and the 2,000-sample
paired bootstrap gives exactly zero A31-minus-A25 KL and Top-1 deltas. A31
therefore inherits A25's A0 comparison: packed KL is statistically tied to A0
(`-0.001236`, 95% CI `[-0.028447, +0.026979]`), while containment metrics
continue to trend toward A0.

Five alternating idle-host timing cycles give:

| Batch | A25 decode median (p95) ms | A31 decode median (p95) ms | Median delta | p95 delta |
| ---: | ---: | ---: | ---: | ---: |
| 1 | `36.9556 (40.5217)` | `34.3101 (37.2716)` | **-7.16%** | **-8.02%** |
| 2 | `36.8906 (40.9569)` | `34.7965 (38.1399)` | **-5.68%** | **-6.88%** |
| 4 | `36.9229 (40.9856)` | `34.3107 (36.1984)` | **-7.07%** | **-11.68%** |
| 8 | `37.7492 (40.8422)` | `34.5523 (36.9797)` | **-8.47%** | **-9.46%** |

The isolated 112-projection packed suite improves `8.51--9.34%` at median
and `6.43--8.99%` at p95. Prefill changes by less than `0.32%`, as expected
because GEMM work dominates there. The full-model A31 result is also
`8.78%/9.24%/12.85%/10.01%` faster than the separately measured A0 medians at
B1/B2/B4/B8, but that cross-artifact comparison is contextual rather than a
fresh paired timing run.

A31 is a new Pareto point: it preserves A25 quality and storage while producing
a reproducible decode win. It does **not** fully clear the phase-2 runtime gate,
because paired B1/B2 gains versus A25 remain below 10%. The next runtime step
should be A41-style grouped-kernel fusion for the already-valid A31 topology;
A32 learned bases are unnecessary unless future fitting makes sibling `SU`
module-local and breaks exact sharing.

Artifacts:

- `artifacts/qvq_rotation_a25_a31_w2_packed_cuda_sm80_stage1.json`
- `artifacts/qvq_rotation_a25_a31_w2_packed_cuda_sm80_full16.json`

Validation on the SM80 host is green: `2,157 passed, 136 skipped` across the
broad QVQ, CUDA, P32, folded-axis, planner, and shared-runtime matrix. The
skips are unavailable MPS/MLX, multi-GPU, and free-threaded cases. Ruff and
`git diff --check` also pass.

## A32: QVQ-trained sibling-shared basis

Keep A31's 14 -> 9 topology, but fit `U_attn` and `U_mlp` against the actual QVQ
objective instead of choosing them randomly.

Use a structured orthogonal parameterization initialized exactly from the
current RHT. A HARP-style staged/butterfly parameterization is a useful design
reference, but the objective must be QVQ-native:

```
L_attn(U) = L_qvq(Wq | U) + lambda_k L_qvq(Wk | U)
          + lambda_v L_qvq(Wv | U)

L_mlp(U)  = L_qvq(Wgate | U) + lambda_u L_qvq(Wup | U)
```

The practical loss should use the existing Hessian/YAQA/output-alignment
signals and held-out propagation, not weight MSE alone. Fit on sampled tiles or
cropped matrices first; only refresh complete P32 trellises for promoted
candidates.

A31/A32 are important because they can remove five of fourteen transforms
without requiring any transform to cross RoPE, residual addition, or SwiGLU.

## A33: learned persistent residual basis + A25

Revisit A1's 14 -> 5 topology, but do **not** reuse A1's fixed random basis.
Learn the residual basis jointly for QVQ.

For orthogonal residual basis `R`, all compatible projections use the usual
exact rewrite:

```
x_tilde = x R
W_tilde = R^T W R_out
```

and embeddings, norms, residual-output projections, and the LM head are
rewritten consistently. A25 supplies the folded V/O head-local pair.

A1 showed the topology is exact and fast. Its failure is evidence against its
chosen fixed basis, not evidence against the topology itself. The phase-2
search should optimize a shared structured `R` against an aggregate objective
over all residual-connected QVQ projections.

This arm targets **14 -> 5** online full Hadamards/block:

- Q output;
- K output;
- gate output;
- up output;
- down input.

If A33 reaches A0-class quality, it is the first architecture likely to produce
a double-digit packed decode gain.

## A34: layer-adapted residual bases with a common core

A single globally fixed `R` may be too restrictive. A per-layer basis normally
requires a runtime basis conversion across the residual identity edge, but a
structured family can make that bridge cheap.

Choose

```
R_l = H S_l
```

where `H` is one common orthogonal Hadamard core and `S_l` is a cheap
layer-specific orthogonal map such as a signed permutation or sparse block
rotation. Then the exact bridge is

```
R_l^-1 R_(l+1)
  = S_l^-1 H^T H S_(l+1)
  = S_l^-1 S_(l+1)
```

so the expensive common Hadamard cancels. If `S_l` is a signed permutation,
the layer-to-layer bridge is only a signed permutation and can potentially be
fused into the next layer's loads.

This reopens the earlier per-layer-basis idea under a different cost model:
allow QVQ to adapt its residual coordinate system by layer **without paying a
full FHT at every layer boundary**.

Search increasingly expressive `S_l` families:

1. signs only;
2. signed permutation;
3. 2x2/4x4 block orthogonal maps;
4. a few independent Givens/butterfly stages.

The dense rewrite must remain exactly invertible. Scaling beyond signs requires
separate treatment because arbitrary anisotropic scale is not an RMSNorm
orthogonal symmetry.

## A35: stage-shared residual bases

If A34's common-core family is not expressive enough, group layers into stages
with independently fitted residual bases, for example 4x4-layer groups.

A basis conversion is then paid only at the three group boundaries. Even if a
bridge costs one or two full FHTs, that cost is amortized over the full model
and remains far below restoring nine per-block transforms.

Test 2-, 4-, and 8-layer sharing intervals. Rank by complete-model latency, not
raw transform count.

## A36: learned RoPE-commuting Q/K folding

Only attempt this after a 9-H or 5-H parent passes quality.

The fixed A3/A20/A21 maps were too weak. Expand the exact RoPE-commuting family
and optimize it specifically for QVQ:

- learned SO(2) rotation per rotary pair;
- reciprocal pair scale where exact;
- pair permutation with the corresponding RoPE frequency ordering rewritten;
- signed pair swaps where algebraically valid;
- GQA-aware sharing between each KV head and its query-head group.

Optimize the transform jointly for Q and K quantization plus held-out attention
output error. Do not select by Q/K local weight error alone.

A successful A36 removes Q and K output transforms from a 5-H parent:

**5 -> 3 H/block**.

## A37: Q/K head-local transform + RoPE fusion fallback

If exact offline Q/K folding cannot retain quality, stop trying to force it.
Keep the transform mathematically online but make it cheaper.

Compare full-width H with per-head H64 / block-H transforms and fuse the chosen
transform with RoPE preparation so there is no standalone transform launch or
intermediate global-memory round trip. This is a speed arm rather than an
offline-fold arm, but it may preserve far more of A0's Q/K recovery than the
restricted RoPE-commuting family.

## A38: learned exact SwiGLU channel symmetry

A27 tested a simple permutation and failed propagation. That does not exhaust
the exact SwiGLU symmetry family or a quantization-aware channel assignment.

Search a common gate/up permutation `P` and an up/down diagonal `D` jointly:

```
W_gate' = W_gate P
W_up'   = W_up P D
W_down' = D^-1 P^T W_down
```

with the exact identity

```
SiLU(g P) * (u P D) = (SiLU(g) * u) P D.
```

Choose `P` by an assignment/search objective based on the **joint** gate, up,
and down QVQ error and downstream replay. Do not reuse a random or heuristic
permutation and conclude that the topology fails.

A successful exact SwiGLU arm removes gate/up output Hadamards. From a 5-H
parent this gives **5 -> 3**. Combining it with a successful A36 gives a
one-H architecture while retaining the down-input transform that A23 proved is
important.

## A39: make the unavoidable down transform cheaper

A23 proves that deleting the down-input transform is destructive. The next
question is how much transform strength is actually necessary.

Test, independently from the failed A4 parent:

- block-H16;
- block-H32;
- block-H64;
- one/two/three pairwise Givens stages;
- structured butterfly stages aligned to K16/K32 QVQ tile geometry;
- learned sparse orthogonal maps optimized against down-projection QVQ error.

Fuse the winner into the down QVQ input-load path where possible. This arm may
remain mathematically online, but it can reduce both arithmetic and launch/
memory overhead while preserving the recovery that zero-H lost.

## A40: tile/codebook-aware transform absorption

Longer-term, stop treating the RHT as a separate activation operator at all.
Search transforms aligned with QVQ's P32 tile/reconstruction geometry so the
transform can be precomposed with decode/accumulation.

Candidate direction:

- block-diagonal transforms on 16/32-channel QVQ tile boundaries;
- quantize directly in that local basis;
- modify the PGC reconstruction/accumulator so decoded values contribute in
  the transformed basis without materializing a transformed activation;
- preserve the exact P32 payload where possible;
- account explicitly for any extra selector/codebook metadata.

This does not necessarily reduce the abstract number of additions to zero, but
it can remove standalone FHT launches and intermediate memory traffic, which is
what matters for decode latency.

## A41: sibling/boundary transform fusion

Track kernel-launch savings separately from mathematical transform savings.
Even when a transform must remain:

- batch/fuse Q and K output transforms when their chosen basis permits it;
- batch gate and up output transforms in one dispatch;
- fuse Q/K transform + SV + RoPE preparation;
- fuse down input transform into the QVQ GEMV input stage;
- retain current SU/SV/bias fusion and avoid materialized intermediate buffers.

A phase-2 winner may combine offline folding, shared transforms, and fusion.
Do not reject a topology because its mathematical H count is unchanged if its
full packed decode time is materially better.

## Search order

Use the following promotion sequence rather than a Cartesian sweep:

1. A31 fixed sibling-shared basis. **Complete: quality-exact to A25 and Pareto.**
2. Apply A41 grouped-kernel fusion to A31 to target the remaining B1/B2 gap.
3. Reserve A32 for a future fitter whose module-local input recovery breaks
   exact sibling sharing.
4. In parallel, pursue A33 learned global residual basis.
5. If A33 is too constrained, A34 common-core layer-adapted basis, then A35
   stage-shared basis.
6. Only after a <=9-H parent passes propagation, try A36 Q/K and A38 SwiGLU.
7. Independently characterize A39 as a cheaper replacement for the one
   demonstrably important down transform.
8. Apply A41 fusion to every promoted topology.
9. Treat A40 as the deeper codec/kernel co-design track.

## Screening and promotion gates

For each new arm:

### Dense gate

- full-model dense logits relative L2 <= `2e-5`;
- 100% dense Top-1 identity on the existing parity set;
- no claimed fold may rely on an approximate nonlinear commutation.

### Stage-1 QVQ gate

Start with W2 and the existing seven-role layer-0 screen. Also run W1.5/W2.5
for any candidate within 10% of A0 on W2. Reject only when the failure is
clearly structural; otherwise refine the basis before discarding a promising
topology.

### Propagation gate

Promote candidates through:

1. one complete decoder layer;
2. 4-layer progressive propagation;
3. all 16 layers / 112 projections.

Every later layer must see the actual previously quantized model state.

### Quality gate

Final promotion requires multiple disjoint held-out streams and at least two
independent fitting/transform seeds. Report paired bootstrap intervals for KL
and containment deltas. A candidate does not need to beat A0 statistically,
but its quality must be statistically compatible with A0 unless a deliberate
quality/speed tradeoff is being reported.

### Runtime gate

Measure actual packed-model execution, not only summed module microbenchmarks.
At minimum report B1/B2/B4/B8 median and p95 decode, prefill, and raw samples.
Alternate arm order and repeat idle-host cycles.

## Expected payoff

A25 removes 2/14 transforms and already produces a 1.9--4.4% packed median
decode signal. Phase 2 should focus first on topologies that remove or share
another three to nine transforms without crossing hard nonlinear boundaries.

The most promising near-term points are therefore:

| Candidate | Full-H/block target | Main idea | Risk |
| --- | ---: | --- | --- |
| A31 | 9 | share QKV and gate/up input RHTs + A25 | low/medium |
| A32 | 9 | learn those shared bases for QVQ | medium |
| A33 | 5 | learned persistent residual basis + A25 | medium/high |
| A34 | 5 + cheap bridges | common H core, layer-adapted sparse basis | medium/high |
| A36 | 3 | learned exact RoPE-compatible Q/K fold | high |
| A38 | 3 | learned exact SwiGLU symmetry | high |
| A36+A38 | 1 | combine successful Q/K and SwiGLU folds | very high |
| A39/A41 | variable | cheaper/fused unavoidable transforms | implementation risk |

The next milestone should not be another two-transform reduction. It should be
a **quality-valid <=9-H architecture with a reproducible >=10% packed decode
speedup**, followed by a push toward the 5-H topology.
