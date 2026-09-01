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

Validation at the A31 checkpoint was green: `2,157 passed, 136 skipped` across the
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
A25 is the only new offline fold in this phase: it removes two Hadamards.
A31 removes three further executions by sharing two sibling transforms, not by
folding them offline. A41 leaves the nine-Hadamard mathematical topology
unchanged and removes three P32 launches by grouping execution. Thus the exact
accounting is `14 -> 12` by offline folding, `12 -> 9` by sharing, and
independently `7 -> 4` P32 inner launches.

A41 keeps A31's quality-valid nine-Hadamard topology but fuses the sibling
inner decodes that consume each shared activation. The first Q/K/V consumer
runs one combined P32 GEMV and caches all three recovered outputs; the first
gate/up consumer does the same for those two outputs. Each projection retains
its own output Hadamard, `SV`, and bias.

P32 fitting chooses `bank_alt_id` independently per projection, so forcing a
common alternative bank would alter the quantizer. The new gated CUDA entry
point interleaves the original trellis words and packed selector bytes in the
decoder's K-major/N-major layout. Child payload copies are then released, so
the trellis and selector payload remains storage-neutral.

The first implementation expanded each projection's scalar `bank_alt_id` to
one uint8 per N16 output tile. That was correct but redundant: it cost 19,376
bytes across 32 groups, or `+0.000159297` BPW. The promoted compact revision
retains only the original two or three alternative-bank bytes per group and
passes cumulative N16 output boundaries as launch scalars. Those boundaries
are derivable from the existing child output widths, so A41 now has exactly
zero incremental stored metadata and the same `2.054419024` EBPW as A31.

The initial implementation exposed an important rejected variant. Splitting a
row-major grouped output produced non-contiguous module slices for M > 1,
which made the output-recovery path about 188% slower in the seven-projection
suite. Materializing each small slice contiguously removed that regression:
the corrected stage-1 suite is 10.7--14.4% faster at M=1/2/4/8 and prefill is
0.2--0.7% faster. The failed and corrected raw artifacts are both retained.

### A41 full-depth SM80 result

A31 was fit progressively across all 112 projections. A41 then reused those
exact fitted tensors rather than refitting a runtime-only arm. All module
payload-hash dictionaries are identical, reconstructed quality is identical,
and both dense rewrites have `1.1171e-6` logits relative L2, `9.2506e-5` max
absolute logit delta, and 100% Top-1/5/10 identity.

| Metric | A31 | A41 grouped P32 | A41 delta |
| --- | ---: | ---: | ---: |
| Online full H/block | 9 | 9 | 0 |
| P32 inner launches/block | 7 | **4** | **-3** |
| Effective BPW | `2.054419024` | `2.054419024` | **0** |
| Packed final KL | `.916587496` | **`.916420329`** | `-0.0182%` |
| Packed logits rel-L2 | `.494783025` | `.494856902` | `+0.0149%` |
| Packed Top-1 | `.548845` | `.548313` | `-0.0533` point |
| Packed Top-5 | `.818117` | `.818117` | tie |
| Packed Top-10 | `.880817` | `.880639` | `-0.0178` point |

The 2,000-resample paired held-out-text bootstrap gives A41-minus-A31 KL
median `-0.000160`, 95% CI `[-0.000403, +0.000062]`, and Top-1 median
`-0.000533`, CI `[-0.002116, +0.001255]`. Both intervals cross zero. Against
the separately measured but sample-aligned A0 model, A41 KL is `.916420`
versus `.917579`; the paired KL CI is `[-0.028652, +0.026789]` and the Top-1
CI is `[-0.021935, +0.004124]`. A41 therefore remains statistically
compatible with A0 under the present evidence.

Five alternating idle-host cycles with the compact metadata give the promoted
paired A31/A41 runtime result:

| Batch | A31 decode median (p95) ms | A41 decode median (p95) ms | Median delta | p95 delta |
| ---: | ---: | ---: | ---: | ---: |
| 1 | `34.4877 (36.4676)` | **`30.2130 (31.6513)`** | **-12.40%** | **-13.21%** |
| 2 | `34.3100 (36.2937)` | **`31.5958 (32.5810)`** | **-7.91%** | **-10.23%** |
| 4 | `34.1264 (35.3610)` | **`31.6802 (32.4189)`** | **-7.17%** | **-8.32%** |
| 8 | `34.2510 (35.8650)` | **`32.1640 (33.6889)`** | **-6.09%** | **-6.07%** |

The compact isolated 112-projection suite improves `14.61%` at M1 and
`8.29--8.58%` at M2--M8 median; p95 improves `5.21--11.86%`. Prefill medians
improve `0.19--1.28%`.

The requested fresh five-cycle A0/A41 control was also run, with both arms
independently fitted from the same calibration and fitting seed:

| Batch | A0 decode median (p95) ms | A41 decode median (p95) ms | Median delta | p95 delta |
| ---: | ---: | ---: | ---: | ---: |
| 1 | `38.5312 (42.1340)` | **`30.0659 (30.9259)`** | **-21.97%** | **-26.60%** |
| 2 | `38.6982 (43.9698)` | **`31.6813 (32.5845)`** | **-18.13%** | **-25.89%** |
| 4 | `39.0107 (42.4347)` | **`31.8814 (33.5868)`** | **-18.28%** | **-20.85%** |
| 8 | `38.4161 (40.2852)` | **`31.7058 (34.9701)`** | **-17.47%** | **-13.19%** |

B1 throughput moves from `25.95` to `33.26` token/s. This direct control used
the correct but expanded alternative-bank vector immediately before the
compact revision. Compact A41 then measured `30.2130` ms B1, within `0.49%`
of that A41 result across runs, while its paired A31 comparison strengthened
and all p95 deltas became favorable. The compact representation therefore
removes storage without giving back the runtime gain.

A41 is now the fastest quality-compatible Pareto arm and clears the primary
M=1 >=10% phase-2 runtime gate while retaining the <=9-H topology. The fresh
A0 control also clears 10% at B2/B4/B8; the incremental compact A31-to-A41 B2
gain is 7.91%. Productionization still needs a checkpoint/load-time grouped
payload format; the experimental runtime deliberately fails closed on
serialization after releasing child payload copies.

Artifacts:

- `artifacts/qvq_rotation_a31_a41_w2_packed_cuda_sm80_stage1.json` (rejected
  strided-output implementation);
- `artifacts/qvq_rotation_a31_a41_w2_packed_cuda_sm80_stage1_contiguous.json`;
- `artifacts/qvq_rotation_a31_a41_w2_packed_cuda_sm80_full16.json` (expanded
  alternative-bank metadata);
- `artifacts/qvq_rotation_a0_a41_w2_packed_cuda_sm80_full16.json` (fresh direct
  A0/A41 control);
- `artifacts/qvq_rotation_a31_a41_w2_packed_cuda_sm80_full16_compact_alt.json`
  (promoted storage-neutral compact metadata).

Post-compact-A41 validation on the SM80 host is green: `2,165 passed, 136 skipped`
across the broad QVQ, CUDA, P32, folded-axis, planner, shared-runtime, and
grouped-runtime matrix. Both generic CUDA JIT variants were rebuilt from
source. The focused compact grouped test covers two- and three-consumer layouts
at M=1/8/17 plus invalid IDs/boundaries and end-to-end runtime parity. Ruff,
Python compilation, artifact JSON validation, and `git diff --check` also
pass. After merging `origin/main` at `6b3cea54`, the freshly rebuilt
latest-main Ampere extension also passes its 28 exactness and dispatch tests.

### Deterministic harness reproduction on latest main

The deterministic validation wrapper was then executed on the SM80 host after
merging `origin/main` at `90c4fa5f`. It enforces the 16-layer/112-projection W2
contract, exact source revision, clean tree, fixed model/data identities,
dense parity, reconstructed and packed quality, exclusive GPU startup, five
alternating timing cycles, 150 raw decode samples per batch/arm, and paired
bootstrap output.

The same-payload `a31-a41-runtime` profile at `2711059c` fit A31 once and reused
all 112 payloads as A41. The complete per-module hash dictionaries and EBPW
were identical. It produced a more mixed incremental runtime result than the
earlier compact run:

| Batch | A31 decode median (p95) ms | A41 decode median (p95) ms | Median delta | p95 delta |
| ---: | ---: | ---: | ---: | ---: |
| 1 | `35.5327 (42.6884)` | `32.0832 (37.7244)` | **-9.71%** | **-11.63%** |
| 2 | `35.8228 (83.2631)` | `32.9124 (39.2551)` | **-8.12%** | **-52.85%** |
| 4 | `35.2472 (38.6627)` | `33.3702 (38.2424)` | **-5.33%** | **-1.09%** |
| 8 | `35.0527 (38.6027)` | `35.2336 (40.9179)` | **+0.52%** | **+6.00%** |

The B2 A31 p95 contains a large outlier and is not evidence for a 52.85% tail
gain. The isolated suite improves `12.63%` at M1 and `5.61--7.68%` at M2--M8,
but the B8 full-model result shows that grouped execution is not uniformly
faster under every latest-main run. The profile still passes its correctness
purpose: A31/A41 payload identity is exact; reconstructed quality is reused;
packed KL and Top-1 bootstrap intervals cross zero; both arms have
`2.054419024` EBPW; and dense rel-L2 is `1.1171e-6` with 100% Top-1 identity.

The independently fitted `a0-a41-production` profile at `1b368547` gives a
stronger same-run production-control result:

| Batch | A0 decode median (p95) ms | A41 decode median (p95) ms | Median delta | p95 delta |
| ---: | ---: | ---: | ---: | ---: |
| 1 | `39.2876 (40.8555)` | `31.3608 (31.9233)` | **-20.18%** | **-21.86%** |
| 2 | `39.5296 (41.1352)` | `32.8206 (34.1593)` | **-16.97%** | **-16.96%** |
| 4 | `39.5008 (41.4623)` | `32.9159 (34.7605)` | **-16.67%** | **-16.16%** |
| 8 | `39.4781 (41.4087)` | `33.1755 (34.9786)` | **-15.96%** | **-15.53%** |

The isolated 112-projection suite improves `26.48%` at M1 and
`20.47--21.29%` at M2--M8. Prefill improves only `0.43--1.43%`, consistent
with transform and launch removal primarily benefiting decode. Packed A0/A41
KL is `0.917579/0.916420`, logits rel-L2 is `0.493024/0.494857`, and Top-1 is
`0.556838/0.548313`. The A41-minus-A0 KL interval
`[-0.028652, +0.026789]` and Top-1 interval
`[-0.021935, +0.004124]` both cross zero; this remains compatibility evidence,
not an accuracy-win claim.

The first harness execution also exposed that `--local-files-only` previously
covered model/tokenizer loading but not dataset metadata lookup. The wrapper
now launches the complete engine with Hugging Face Hub and datasets offline
mode enabled. The production profile confirms that both train and validation
data resolve from the cached `b08601e0` dataset revision without network
requests. Both profiles use model revision `bb3d2111`, identical calibration
rows, and identical validation row/hash sets.

The post-harness latest-main SM80 matrix is green:
`2,172 passed, 136 skipped, 0 failed` across QVQ, CUDA, P32, folded axes,
planner, shared/grouped runtime, and the harness contracts. The seven harness
tests include the positive same-payload case plus rejection of payload drift,
dense-parity regression, validation overlap, missing timing samples,
source-revision mismatch, and failure to propagate offline mode to the
complete engine process.

Harness artifacts:

- `artifacts/qvq_validation/a31-a41-runtime_seed20260831_2711059c.json`;
- `artifacts/qvq_validation/summary_a31-a41-runtime_2711059c.json`;
- `artifacts/qvq_validation/a0-a41-production_seed20260831_1b368547.json`;
- `artifacts/qvq_validation/summary_a0-a41-production_1b368547.json`.

### Canonical P32 runtime and recovery oracles

Two stricter no-requantization profiles were added after merging
`origin/main` at `03144a22`. These are runtime oracles, not additional
quantization arms. Both use the A31/A25-compatible transformed-weight plan and
materialize one canonical per-module P32 payload:

- `P0` executes those bytes through seven ordinary per-module `QVQLinear`
  calls. No shared-transform or grouped-runtime hook is installed, so the
  A25 axes execute 12 full Hadamards/block.
- `R0` loads the exact same bytes and compiles compatible Q/K/V and gate/up
  sibling groups. It executes nine Hadamards and four P32 inner launches/block.
- `P0+C` and `R0+C` repeat the same control after the repository's production
  fixed-trellis output-channel correction. That correction freezes trellis,
  selectors, alternative-bank choice, bias, and `SU`; only serialized `SV`
  may change. Inference uses no random seed.

The refactored compiler is recovery-safe rather than optimistically strict.
It groups a sibling set only when its packed geometry and stored `SU` are
compatible. An incompatible set remains as ordinary per-module P32. A unit
test with unequal module-local `SU` proves that fallback is exact. This matters
for any future recovery algorithm that learns input scales independently;
the current production alignment explicitly freezes `SU` and is SV-only.

The clean-tree `p0-r0-runtime` profile at `02b20151` passed all harness gates:

- 112/112 complete projection records and byte-identical payload hashes;
- one canonical payload materialization and no candidate fit;
- 32/32 grouped sibling sets, zero fallback;
- identical `2.054419024` effective BPW;
- direct R0-versus-P0 packed logits KL `7.4977e-5`, relative L2 `0.004626`,
  maximum absolute delta `0.232422`, Top-1 identity `99.183%`, and Top-5/10
  identity `100%` across 5,630 held-out tokens.

Five alternating timing cycles give a material runtime win against plain P32:

| Batch | P0 decode median (p95) ms | R0 decode median (p95) ms | Median delta | p95 delta |
| ---: | ---: | ---: | ---: | ---: |
| 1 | `37.1427 (39.9424)` | **`30.8287 (32.2134)`** | **-17.00%** | **-19.35%** |
| 2 | `37.1377 (41.5199)` | **`32.6202 (35.4383)`** | **-12.16%** | **-14.65%** |
| 4 | `36.9230 (40.2394)` | **`32.0682 (34.2309)`** | **-13.15%** | **-14.93%** |
| 8 | `36.8466 (40.2223)` | **`31.8748 (33.2723)`** | **-13.49%** | **-17.28%** |

B1 throughput rises from `26.92` to `32.44` token/s (`+20.48%`). The isolated
112-projection suite improves `21.61%` at M1 and `15.67--17.56%` at M2--M8;
all suite p95 deltas are favorable.

The `p0c-r0c-correction` profile at `f495e745` also passed. Correction accepted
374,732 output-channel updates across all 112 modules; every module changed
only `SV`. The summed local Hessian proxy fell from `474.7300` to `468.7766`
(`-1.25%`). R0+C still compiled 32/32 groups with zero fallback and retained
identical EBPW and payload hashes. Direct R0+C-versus-P0+C packed equivalence
is KL `6.8878e-5`, relative L2 `0.004352`, maximum delta `0.210938`, Top-1
identity `99.325%`, and Top-5/10 identity `100%`.

| Batch | P0+C decode median (p95) ms | R0+C decode median (p95) ms | Median delta | p95 delta |
| ---: | ---: | ---: | ---: | ---: |
| 1 | `38.5307 (45.8208)` | **`32.6092 (38.6828)`** | **-15.37%** | **-15.58%** |
| 2 | `37.8954 (43.6457)` | **`34.4376 (41.0248)`** | **-9.12%** | **-6.00%** |
| 4 | `38.3092 (44.9101)` | **`33.9781 (40.6990)`** | **-11.31%** | **-9.38%** |
| 8 | `39.8481 (45.3532)` | **`33.5493 (275.4560)`** | **-15.81%** | `+507.36%` outlier |

The corrected B8 p95 contains one extreme R0+C sample and is explicitly not
tail-latency evidence. Its median and the independent isolated suite remain
favorable; the corrected suite improves `19.16--25.98%` at M1--M8 median.

The correction itself is a negative propagated-quality result. P0+C versus
uncorrected P0 moves full-model packed KL from `0.916587` to `0.958022`
(`+4.52%`) and logits relative L2 from `0.494783` to `0.504663` (`+2.00%`),
although Top-1/5/10 each improve by `0.18--0.30` percentage point. Improving
every local fixed-trellis Hessian proxy therefore does not establish a
full-model quality improvement. The oracle proves that the refactored runtime
preserves the repository's existing post-quant recovery representation; it
does not promote this particular local SV correction as the production
quality default.

Oracle artifacts:

- `artifacts/qvq_validation/p0-r0-runtime_seed20260831_02b20151.json`;
- `artifacts/qvq_validation/summary_p0-r0-runtime_02b20151.json`;
- `artifacts/qvq_validation/p0c-r0c-correction_seed20260831_f495e745.json`;
- `artifacts/qvq_validation/summary_p0c-r0c-correction_f495e745.json`.

Post-oracle validation on the merged latest-main SM80 head is green:
`2,179 passed, 136 skipped, 0 failed`. This is the previous 2,308-test matrix
plus the seven new runtime-oracle and semantic-fallback cases. Ruff, Python
compilation, artifact JSON parsing, and `git diff --check` also pass.

Further fusion opportunities remain even when a transform must remain:

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
2. A41 grouped-kernel fusion. **Complete: M1 gate passed and new Pareto point.**
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
| A31 | 9 | share QKV and gate/up input RHTs + A25 | validated Pareto |
| A41 | 9 | A31 plus grouped QKV and gate/up P32 decode | **validated winner** |
| A32 | 9 | learn those shared bases for QVQ | medium |
| A33 | 5 | learned persistent residual basis + A25 | medium/high |
| A34 | 5 + cheap bridges | common H core, layer-adapted sparse basis | medium/high |
| A36 | 3 | learned exact RoPE-compatible Q/K fold | high |
| A38 | 3 | learned exact SwiGLU symmetry | high |
| A36+A38 | 1 | combine successful Q/K and SwiGLU folds | very high |
| A39 | variable | cheaper unavoidable down transform | implementation risk |

The <=9-H, >=10% M1 milestone is met by A41 against the independently fitted
A0 production control. The latest same-payload A31/A41 profile measures a
9.71% B1 gain and a 0.52% B8 regression, so the incremental grouped-runtime
claim remains workload- and run-sensitive. The next milestone is to repeat the
production profile with an independent fitting seed, stabilize tail latency,
productionize grouped checkpoint loading, add the plain-P32/refactored-P32
same-payload oracle, and only then push toward the learned 5-H A33/A34
topology.
