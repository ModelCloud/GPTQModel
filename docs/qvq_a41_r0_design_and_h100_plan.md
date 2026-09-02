# A41 / R0 design, validation state, and H100 implementation plan

Status: design and validation record for PR #84

Last updated: 2026-09-02

## Purpose

This document freezes the current understanding of the A41/R0 work so the research branch can be merged without losing the reasoning, rejected paths, accuracy constraints, or the exact contract that the later H100 kernel implementation must satisfy.

It is intentionally broader than a kernel note. It records:

- what A41 and R0 mean;
- what is already proven;
- what is only experimentally promising;
- which transforms are truly folded into weights and which are only shared at runtime;
- the canonical checkpoint/runtime boundary;
- the Torch oracle design;
- the accuracy and recovery constraints;
- the implementation handoff to `perf/qvq-p32-h100`;
- the current H100 kernel constraints that the grouped implementation must respect;
- rejected five-H research and why it should not block productionization;
- the ordered TODO list and merge criteria.

The goal is to stop treating A41/R0 as another arm in an open-ended rotation search. The semantics are now sufficiently well understood that the next step is to freeze the abstraction and implement it on the H100 kernel branch.

---

## Branch authority policy

From this point forward there are two explicit authorities.

### PR #84 / `feat/qvq-v2b2-p32-lr-mlx`

This branch is the **experiment, oracle, and design authority** for A41/R0.

It may establish:

- transform-planner semantics;
- graph rewrites;
- exact folding identities;
- A0-A41 experiment results;
- P0/R0 same-payload runtime oracles;
- correction/recovery oracles;
- grouped runtime legality rules;
- checkpoint metadata concepts;
- Torch reference behavior;
- rejected rotation/topology families;
- acceptance criteria for a future native implementation.

It is **not** the implementation authority for the final H100 kernel.

### `perf/qvq-p32-h100`

This branch is the **only implementation authority** for the final H100 P32 realization.

Kernel recommendations must be grounded in what exists on this branch: its P32 layout, TMA/WGMMA implementation, split-K scheduling, reduction behavior, H100 measurements, CUDA Graph behavior, and current source structure.

Do not use `origin/main`, H200-only branches, or another backend as evidence for what the H100 implementation currently does. They may be external context only when explicitly requested.

The intended flow is therefore:

```text
PR84 experiments / proofs / Torch oracle
                |
                v
       frozen A41/R0 contract
                |
                v
   perf/qvq-p32-h100 implementation
                |
                v
        H100 exactness + timing
```

---

# 1. What problem are we solving?

Standard P32 inference around one decoder block contains several randomized-Hadamard-transform operations and several independent P32 projection launches.

The original safe A0 topology executes 14 full Hadamards per block.

Research found three conceptually different ways to reduce cost:

1. **True offline folding**: remove a transform from runtime by changing the dense weight basis before quantization.
2. **Runtime sharing**: compute an identical input transform once and reuse the transformed activation for sibling projections.
3. **Grouped decode**: execute sibling P32 inner products inside one launch while preserving each child's independent quantization state and arithmetic contract.

These must not be conflated.

The current proven progression is:

```text
A0   : 14 H/block, 7 independent P32 projection launches
A25  : 12 H/block via true offline V/O folding
A31  :  9 executed H/block via shared QKV and gate/up input transforms
A41  :  9 executed H/block, 7 -> 4 inner P32 launches via grouped sibling decode
```

Only A25 is an additional weight-basis fold.

A31/A41 do not require changing the already-quantized child payload when the stored transforms are compatible. They are execution-schedule optimizations.

---

# 2. Terminology: A41, P0, R0, C0

## A41

A41 is the experimental performance target that combines:

- the A25 offline V/O fold;
- A31 shared input-transform execution for Q/K/V and gate/up;
- one grouped P32 decode for Q/K/V;
- one grouped P32 decode for gate/up;
- independent O and down decodes.

Thus:

```text
Q/K/V input transform: compute once
Q/K/V P32 inner decode : one grouped launch

gate/up input transform: compute once
gate/up P32 inner decode : one grouped launch

O    : one launch
down : one launch
```

That changes the inner-launch count from seven to four while retaining nine executed Hadamards per block.

A41 is a performance arm. It is not the final safety abstraction.

## P0

P0 is the boring canonical execution oracle for the A25-compatible payload.

It uses the exact same canonical per-module P32 tensors as R0, but runs each projection independently. No shared-transform hook and no grouped P32 runtime is installed.

P0 exists to answer one question:

> Does the refactored runtime change the meaning of an already-quantized P32 payload?

## R0

R0 is the production semantic contract.

R0 takes canonical per-module P32 modules plus a graph plan and compiles only those groups whose assumptions are actually true.

If a group is legal, R0 may share its input transform and group its P32 decode.

If a group is not legal, R0 retains ordinary per-module P32 execution.

R0 therefore means:

```text
optimize when exact
fall back when not exact
never modify the quantization solution to satisfy a runtime optimization
```

This is the abstraction that should survive into the H100 implementation.

## C0

C0 is the checkpoint/load-time form of the same idea.

The checkpoint remains canonical per-module P32. A small manifest identifies candidate groups. At load time the runtime validates the actual modules and compiles compatible groups into a transient backend-specific representation.

C0 is useful architecture work, but the current H100 implementation target remains R0 semantics first. Checkpoint compilation should follow once the H100 grouped kernel contract is stable.

---

# 3. Canonical P32 representation must remain independent of execution schedule

The central architecture rule is:

```text
quantization representation != execution schedule
```

Each module remains a canonical P32 module with its own state, including at least:

- trellis words;
- bank selectors;
- alternative bank ID;
- SU;
- SV;
- optional bias;
- input/output transform semantics;
- codec/rate metadata.

The grouped runtime is a compilation of those bytes, not a new quantizer.

The canonical checkpoint must remain backend independent.

Do not serialize a CUDA-only grouped/interleaved layout as the authoritative weight format.

A backend may construct an ephemeral grouped layout after load, but it must be possible to reconstruct every child's canonical payload exactly.

This gives us three useful invariants:

```text
canonical checkpoint -> plain P32
canonical checkpoint -> R0 grouped P32
canonical checkpoint -> future backend
```

all represent the same quantized model.

---

# 4. What requires modifying weights?

This distinction is important for existing snapshots.

## A31/A41/R0 sharing and grouping

No dense weight fold is required.

If existing sibling modules have the same legal pre-H transform, R0 can compute that transform once and reuse it without changing the P32 payload.

For example, if Q/K/V store the same SU and input-H state:

```text
t = H(x * SU)

q = P32_Q(t)
k = P32_K(t)
v = P32_V(t)
```

is the same operator as redundantly computing the same `H(x * SU)` three times.

Likewise for gate/up.

## A25 V/O offline fold

This is different.

A25 changes the dense weight basis before P32 quantization so a V-output/O-input transform pair can be removed from runtime.

A dense orthogonal rewrite can be exact, but in general:

```text
Q(W R) != Q(W) R
```

for the P32 quantizer.

Therefore an old already-quantized A0 snapshot cannot safely be converted into the A25 payload by rotating compressed trellis state after the fact.

The safe A25 flow is:

```text
dense model
   -> exact V/O basis fold
   -> quantize the folded dense weights to P32
   -> canonical A25-compatible P32 snapshot
   -> R0 runtime sharing/grouping
```

R0 itself must not requantize.

---

# 5. Why A41/R0 is now the main target

The research question used to be whether more rotations should be folded offline.

The data changed the answer.

A41/R0 produced a large decode gain while preserving the same P32 representation.

The independently fitted A0/A41 production comparison measured approximately:

| Batch | A0 -> A41 median latency reduction |
|---:|---:|
| 1 | 20.18% |
| 2 | 16.97% |
| 4 | 16.67% |
| 8 | 15.96% |

The quality intervals crossed zero. This is compatibility evidence, not an accuracy-win claim.

More importantly, the same-payload P0/R0 oracle eventually became exact after preserving each child segment's original split-K arithmetic.

Across 5,630 held-out tokens the accepted P0/R0 run had:

- logits relative L2 = 0;
- maximum absolute logit delta = 0;
- exact-logit fraction = 100%;
- Top-1/5/10 identity = 100%;
- 112/112 canonical payload hash dictionaries identical;
- 32/32 groups compiled;
- zero fallback.

That makes R0 fundamentally different from another quality experiment.

R0 is now a semantics-preserving runtime/compiler optimization.

The accepted same-payload R0 timing result versus P0 was:

| Batch | P0 -> R0 median latency reduction |
|---:|---:|
| 1 | 17.93% |
| 2 | 15.88% |
| 4 | 15.05% |
| 8 | 11.48% |

This is the point where architecture implementation has higher expected value than continuing an unbounded transform search.

---

# 6. What the rotation experiments taught us

## Dense exactness is not enough

Many aggressive rewrites are algebraically exact in the dense model yet quantize worse under P32.

The correct mental model is:

```text
Can the dense network be rewritten exactly?
```

and

```text
Does P32 still represent the rewritten weights well?
```

are different questions.

P32 has preferred coordinate systems determined by its trellis/codebook/selectors and SU/SV degrees of freedom.

An exact dense basis change can move a matrix into a region that is materially harder for P32.

## V/O folding is the strong positive result

A25 showed a true offline fold that remains P32-compatible and removes two runtime Hadamards per block.

This proves offline folding is useful when it follows an exact graph symmetry that P32 tolerates.

## Down-input H is currently important

Removing the down-input transform caused a large local error increase in prior experiments, roughly from 0.137 to 0.420 relative error in the cited down-projection screen.

The current working assumption is:

```text
retain down-input H unless a new quantizer/fused implementation proves otherwise
```

## Q/K output folding is difficult

RoPE restricts the exact transform family. General H does not commute with position-dependent RoPE rotations.

RoPE-compatible structured transforms exist, but the tested families did not preserve P32 quality well enough.

## SwiGLU is another hard boundary

A general H cannot commute through:

```text
SiLU(g) * u
```

Only restricted coordinatewise symmetries are exact. Tested simple variants did not produce a promoted P32 topology.

---

# 7. Five-H A33/A34 result

A five-H topology remains mathematically interesting, but the tested A33/A34 families are not promoted.

The `a33-a34-stage-screen_seed20260902.json` screen tested fixed/global and common-core signed-permutation families.

No candidate cleared the staged promotion gate.

Important observations:

- A1 fixed global five-H remained worse than A0 on P32 quality.
- A33 core/adaptation seeds either failed the exact dense Top-1 gate or the quality gate.
- A34 common-core layer adaptation remained dense-exact in the tested case but failed propagated quality.
- No fresh acceptance stream, W1.5/W2.5 sweep, or 16-layer run was consumed because stage-1/early propagation did not justify it.

Decision:

```text
Reject the tested five-H global/common-core signed-permutation families.
```

This does not prove that five-H is impossible.

It does make a simple persistent global/common-core basis a lower-priority path than A41/R0 productionization.

The current optimization philosophy is therefore:

```text
offline what P32 naturally tolerates
share what cannot be offlined
fuse/group what must remain online
```

rather than minimizing mathematical H count at any cost.

---

# 8. Post-quant recovery constraints

A grouped runtime must never remove useful quantization/recovery degrees of freedom merely to keep a fast path legal.

The historical fixed-trellis correction path permits module-local output correction through SV while freezing trellis/selectors/SU/bank choice/bias.

The P0+C/R0+C same-payload oracle proved that R0 can preserve that representation exactly.

However, the correction objective itself is not promoted: it improved its local Hessian proxy while full-model propagated KL regressed by about 4.52%.

This teaches two separate lessons:

1. R0 must preserve a correction representation exactly.
2. A locally improving correction objective is not automatically a model-quality improvement.

Future correction methods may want module-local SU as well.

Therefore R0 must not permanently require sibling SU equality as a model constraint.

Instead:

```text
if sibling SU is equal -> sharing may be legal
if sibling SU diverges -> de-group / plain P32 fallback
```

A future grouped-compatible correction parameterization can be researched separately, but it may not silently replace independent module-local correction freedom.

---

# 9. R0 legality contract

A candidate shared-input/grouped-P32 set must be validated from the actual loaded modules.

For the first implementation, a group should require:

- same source activation;
- same execution cycle/order;
- same device;
- same input width K;
- same input-H state;
- bit-identical stored SU for the complete shared pre-H transform;
- standard V2B2-P32;
- vector size 2;
- trellis window 16;
- compatible codebook/rate geometry;
- output widths supported by the backend grouped kernel.

Each child retains independent:

- trellis payload;
- bank selectors;
- alternative-bank choice;
- output width;
- output transform state;
- SV;
- bias;
- backend schedule such as split-K and reduction ordering.

If a group cannot satisfy the backend requirements, R0 keeps plain per-module P32.

Semantic corruption should fail closed. Optional optimization incompatibility should fall back.

---

# 10. Grouped P32 physical representation

For child `i`, canonical P32 trellis can be viewed as:

```text
[K_tiles * N_tiles_i, words_per_tile]
```

Reshape it logically to:

```text
T_i = [K_tiles, N_tiles_i, words_per_tile]
```

and selectors to:

```text
B_i = [K_tiles, N_tiles_i]
```

A grouped transient representation can concatenate on the N-tile dimension:

```text
T_G = cat_N(T_0, T_1, ...)
B_G = cat_N(B_0, B_1, ...)
```

with compact metadata:

- ordered child list;
- N/output boundaries;
- child bank-alt IDs;
- child scheduling descriptors.

The fundamental payload round-trip invariant is:

```text
canonical_child(group(canonical_children), i) == original_child_i
```

byte-for-byte for trellis/selectors/bank-alt metadata.

SU/SV/bias are not merged into the grouped inner payload.

---

# 11. Phase 1: Torch oracle — accuracy before speed

The first implementation target is a pure Torch semantic oracle on PR84.

No CUDA optimization should be required to prove A41/R0 correctness.

The Torch oracle must be deliberately boring and should use canonical P32 reconstruction rather than production dispatch.

## 11.1 Existing canonical accuracy contract

The P32 accuracy reference is conceptually:

```text
reconstruct canonical P32 inner matrix in FP32
apply the exact QVQ transform sequence in FP32
perform dense FP32 matmul
apply output transform/SV/bias in the original order
```

The grouped oracle should preserve that same operation order.

## 11.2 Stateless group truth function

Build a stateless semantic function first:

```text
torch_grouped_p32_oracle(x, group_spec, child_payloads) -> tuple[child_outputs]
```

For a legal group:

```text
t = H(x * SU_shared)

z_0 = t @ Q_0
z_1 = t @ Q_1
...

y_i = child_output_recovery_i(z_i)
```

The shared input transform is computed once.

The individual child inner matmuls should remain separate in the promotion oracle. Do not define correctness using one giant concatenated dense GEMM because GEMM tiling/reduction changes would add an irrelevant floating-point confound.

A concatenated GEMM can be a diagnostic only.

## 11.3 Stateful model bridge second

After the stateless function is proven, add a coordinator for existing Hugging Face call order.

The first sibling invocation may compute/cache all group outputs; later siblings consume their entries.

The coordinator must fail closed on:

- duplicate consumer;
- out-of-order consumer;
- different source tensor object;
- source mutation when a version counter is available;
- incomplete/aborted cycle without reset.

This adapter is model-integration plumbing, not the mathematical oracle.

## 11.4 Torch correctness sequence

Run in this order:

1. P32 child payload group/ungroup byte-exact tests.
2. QKV group with same SU, distinct SV, distinct bank-alt IDs, and V output-H folded/off.
3. gate/up group with independent child output recovery.
4. unequal-SU fallback test.
5. incompatible format/rate/geometry fallback tests.
6. single real decoder-layer equality.
7. four-layer progressive equality.
8. full 16-layer / 112-projection equality.
9. final-logit equality.

For same-payload Torch P0/R0, require exact tensor equality wherever the operation ordering is intentionally identical.

Do not start native H100 grouped-kernel work until the Torch semantic oracle is green.

---

# 12. Permanent same-payload runtime gates

The PR84 runtime harness permanently tightened the P0/R0 and P0+C/R0+C gates after an earlier split-K ordering bug was diagnosed.

Permanent full-model gates are:

- absolute final KL <= 1e-6;
- logits relative L2 <= 1e-8;
- maximum absolute logit delta <= 1e-7;
- exact-logit value fraction exactly 100%;
- Top-1 identity exactly 100%;
- Top-5 identity exactly 100%;
- Top-10 identity exactly 100%;
- complete payload hashes identical where same-payload identity is required.

The earlier loose result with ~0.0046 logits relative L2 is diagnostic history only and must never be accepted again.

Its root cause was important: grouped QKV selected a different split-K/reduction arithmetic than the ordinary child kernels. Very small FP32 inner differences crossed later BF16 rounding boundaries and propagated through the full model.

This is the strongest design requirement for the native implementation:

```text
A grouped child must preserve the arithmetic schedule of the plain child.
```

---

# 13. H100 implementation authority and current state

The final native realization will be implemented against `perf/qvq-p32-h100`.

As of this design freeze, the branch head is `29f9a3e5`.

Relevant branch characteristics include:

- Hopper SM90 exact standard-P32 window/TMA/WGMMA implementation;
- canonical P32 payload semantics retained;
- FP32 output accumulation;
- explicit split-K support;
- rate/shape/device-aware split scheduling for existing measured shapes;
- H100-specific split decisions in the Python wrapper;
- deterministic partial-plane reduction machinery for ordinary split counts;
- a special atomic fused-reduction experiment/path for a selected split count;
- logical M<=16 work represented through the branch's current kernel strategy.

The exact implementation source on that branch, not another branch, defines what must be extended.

The branch's current public grouped-kernel gap is that a P32 WGMMA invocation still fundamentally describes one projection:

```text
one input
one P32 payload
one N/out_features
one bank-alt ID
one split count
```

A41 needs a segment-aware grouped form.

---

# 14. H100 native A41/R0 target

The native H100 goal is one shared transformed activation feeding one grouped P32 launch with multiple independent child segments.

Conceptually:

```text
x
|
+-- shared complete input transform --------------------------+
|                                                            |
v                                                            |
t = H(x * SU_shared)                                         |
|                                                            |
+---------------- one grouped P32 launch ---------------------+
|                    |                    |
Q segment            K segment            V segment
own payload           own payload           own payload
own N                 own N                 own N
own bank-alt          own bank-alt          own bank-alt
own split schedule    own split schedule    own split schedule
own reduction order   own reduction order   own reduction order
|                    |                    |
Q recovery            K recovery            V recovery
```

and similarly for gate/up.

The grouped implementation must not collapse the segments into one synthetic projection for scheduling purposes.

---

# 15. Segment-specific scheduling is mandatory

The biggest native-kernel lesson from the PR84 exactness work is that a grouped QKV kernel cannot select one split-K count from `N_total` and assume it is equivalent.

Each child must first resolve the same schedule it would use independently on the H100 branch.

The grouped launch then maps work to something equivalent to:

```text
SegmentPlan {
    n_begin
    n_width
    bank_alt_id
    split_count
    reduction_mode
}
```

The work scheduler must preserve child-specific partition boundaries and reduction ordering.

The conceptual invariant is:

```text
GroupedH100(segment_i) == PlainH100(child_i)
```

at the native child-output level under the backend's exactness contract.

Only after that passes do we measure launch savings.

---

# 16. What the H100 split-K experiments teach the grouped design

The H100 branch already demonstrates that split policy can depend on device, rate, K/N shape, and workload.

That means R0 should not encode split heuristics itself.

Instead:

```text
R0 graph compiler
    asks H100 backend for child launch plan
    preserves that plan inside grouped execution
```

The backend remains responsible for choosing a child schedule.

The group compiler only combines compatible work without changing the schedule semantics.

This cleanly separates:

- graph legality;
- quantization semantics;
- backend scheduling.

---

# 17. Reduction strategy for initial H100 implementation

Correctness comes before clever reduction fusion.

For any grouped child that requires split-K, the first native grouped implementation should preserve the H100 branch's deterministic reduction behavior wherever possible.

Do not begin by replacing all split reduction with unordered atomics or a new cluster scheme.

Recommended progression:

1. reproduce plain-child schedule exactly inside the grouped work grid;
2. write child partials using the same logical partition order;
3. use the same deterministic reduction order as the plain H100 child;
4. prove grouped child output matches plain H100 child;
5. only then optimize the reduction path.

This mirrors the PR84 lesson: a tiny inner FP32 arithmetic difference is enough to fail full-model exactness later.

---

# 18. H100 Llama down-projection occupancy work is related but separate

The H100 branch is also the home of the Llama down-projection occupancy experiments.

For a Llama `8192 -> 2048` down projection with an N64 CTA mapping, the unsplit N dimension provides only 32 base CTAs. A split-4 K schedule would expose 128 decode CTAs, close to a 132-SM H100 wave.

This is valuable scheduler research and should improve the backend launch planner.

However, it is not itself A41/R0.

The relationship is:

```text
H100 child scheduler improves plain child schedule
             |
             v
A41/R0 grouped kernel must preserve that improved schedule per segment
```

Do not tie A41 correctness to one hard-coded Llama split value.

---

# 19. Initial H100 grouped API direction

The exact C++/Python interface can evolve, but the semantic shape should resemble:

```text
qvq_p32_grouped_h100(
    transformed_input,
    grouped_payload,
    segment_descriptors,
)
```

where each descriptor carries enough information to reproduce the plain child's backend plan.

Candidate descriptor information:

```text
n_begin / output offset
n_width
bank_alt_id
split_count
reduction mode / reduction plan ID
possibly route/specialization ID if the H100 branch has multiple kernel paths
```

Do not put Q/K/V or gate/up role names inside the low-level grouped kernel.

The graph planner provides generic groups. The backend receives ordered segments.

---

# 20. Shared-input transform implementation boundary

A41 includes both transform sharing and grouped P32 decode.

The first native implementation should keep those as two semantic steps:

```text
one existing exact H100-compatible input transform
        -> one grouped P32 inner launch
```

Do not initially fuse the input Hadamard into the WGMMA decoder.

Likewise, do not initially fuse:

- Q/K output transforms;
- RoPE;
- gate/up output transforms;
- SwiGLU;
- down input transform;
- child SV/bias recovery.

Those are later performance passes after grouped P32 exactness is established.

This keeps the H100 implementation close to the proven Torch/R0 contract and makes failures attributable.

---

# 21. Checkpoint / load-time target

The desired checkpoint architecture remains:

```text
on disk:
    canonical per-module P32 payloads
    + small versioned grouping manifest

load time:
    validate manifest
    validate actual modules
    resolve H100 child schedules
    build transient grouped payload/layout
    install grouped delegate/compiled runtime

save again:
    emit canonical per-module P32 tensors
    never make the transient H100 layout authoritative
```

The manifest should record graph/group identity, not hardware scheduling details that can be recomputed by the H100 backend.

Unknown schema or corrupted semantic metadata should fail closed.

A legacy checkpoint without the manifest should remain ordinary P32.

A manifest group that is semantically valid but not optimizable on the current device should remain plain P32.

---

# 22. Production immutability / invalidation contract

Once a grouped runtime has compiled child payload state, the model must have an explicit rule for later mutation.

Two acceptable designs are:

1. **Inference payload immutable after compilation**.
2. Mutating SU/trellis/selectors/etc. invalidates and recompiles the group.

Do not leave this implicit.

In particular, a future correction method that changes module-local SU after group compilation must not keep using a stale shared-transform assumption.

The simplest initial production contract is likely:

```text
compile only after all quantization/correction is complete
then treat inference payload as immutable
```

with explicit runtime reinstall required after mutation.

---

# 23. Validation hierarchy

A41/R0 must be validated at multiple layers.

## Level 1: payload identity

- group/ungroup canonical tensors byte-exact;
- no requantization;
- no selector/bank winner changes;
- no SU/SV change caused by runtime compilation.

## Level 2: Torch semantic oracle

- plain child vs shared/grouped Torch semantics;
- exact legal-group output;
- exact fallback output for incompatible groups.

## Level 3: native H100 child oracle

For a fixed already-quantized payload:

```text
plain H100 child output
vs
grouped H100 child segment output
```

must satisfy the intentionally frozen native arithmetic contract.

If a child schedule can be made bit-exact, require bit-exactness.

If a backend specialization intentionally changes floating accumulation order, create an explicit ordered split reference and require the optimized version to match that reference exactly while remaining inside the dense P32 error gate. Do not silently weaken the full-model runtime oracle.

## Level 4: full-model same-payload runtime oracle

The PR84 permanent gates remain authoritative.

## Level 5: independently fitted A0/A41 quality control

This is a separate statistical question from runtime exactness.

Do not use a same-payload oracle to claim an A41 quantization-quality win, and do not use a quality-compatible A0/A41 run to excuse a runtime semantic mismatch.

---

# 24. Benchmark discipline

Every native performance claim should retain the existing style of deterministic measurement:

- exact source revision;
- exact H100 identity;
- exclusive/idle GPU gate;
- CUDA Graph safety where relevant;
- raw timing samples;
- alternating comparison order where whole-model timings are compared;
- exact payload identity for runtime comparisons;
- dense P32 accuracy check;
- no hidden persistent VRAM increase;
- separate median and tail-latency reporting.

A41 should be evaluated at both projection-suite and full-model levels.

A microkernel launch reduction is not automatically a full-model win.

---

# 25. What we should not do

Do not:

- make grouped H100 payloads the canonical checkpoint format;
- hard-code Q/K/V role logic into generic P32 kernel code;
- force sibling SU equality to preserve a fast path;
- requantize during runtime compilation;
- choose one split count from concatenated QKV N;
- use unordered atomics merely because they remove a reducer if they violate the exact runtime contract;
- infer quantized quality from dense exactness;
- infer full-model quality from local weight/output MSE alone;
- resume large Cartesian rotation sweeps before A41/R0 implementation is stable;
- treat H200/main implementation details as H100 implementation authority.

---

# 26. Current state of mind on feasibility

## A41/R0

High confidence.

Reasons:

- large measured decode gain;
- same-payload full-model exactness achieved after schedule preservation;
- no new quantization format required;
- canonical per-module payload remains intact;
- fallback semantics protect future recovery freedom;
- checkpoint/runtime separation has already been prototyped;
- remaining work is mainly backend integration rather than mathematical discovery.

## More aggressive offline folding toward five-H

Possible but lower confidence.

The tested global/common-core families did not clear P32 quality gates.

Further work likely needs more expressive, quantization-aware bases and therefore carries more complexity and uncertainty.

It should not block A41/R0.

## Zero-H / fold-everything objective

Not the current goal.

The data supports minimizing latency subject to P32 quality, not minimizing H count at all costs.

---

# 27. Ordered execution plan

## Phase 1 — freeze Torch oracle on PR84

Goal: prove semantics, not speed.

- isolate generic grouped P32 spec;
- byte-exact group/ungroup helpers;
- stateless grouped Torch oracle;
- R0 legality/fallback compiler;
- stateful sibling bridge only after stateless proof;
- one-layer, four-layer, full-16-layer same-payload equality;
- retain permanent full-model oracle thresholds.

Exit condition:

```text
Torch P0 == Torch R0 for the same payload
```

and illegal groups fall back exactly.

## Phase 2 — H100 backend segment plan

Goal: map every child to the exact schedule the current H100 branch would use independently.

- define generic segment descriptor;
- query/resolve child H100 split/reduction route;
- group only compatible P32 payload geometry;
- keep child schedules independent.

Exit condition:

```text
one grouped launch plan can describe Q/K/V and gate/up
without changing any child's plain schedule
```

## Phase 3 — H100 grouped inner kernel

Goal: one grouped P32 launch, no transform fusion yet.

- transient grouped trellis/selector layout;
- segment-aware N mapping;
- segment-aware bank-alt IDs;
- segment-aware split-K;
- preserve deterministic child reduction order;
- split grouped output into child FP32 inner results.

Exit condition:

```text
grouped H100 child segment == plain H100 child
```

under the frozen native exactness contract.

## Phase 4 — shared H100 input-transform execution

Goal: complete A31/A41 runtime behavior.

- compute one legal complete input transform per sibling group;
- feed it into grouped inner decode;
- independent child output recovery remains unchanged.

Exit condition:

```text
native same-payload P0/R0 full-model oracle passes
```

## Phase 5 — H100 performance promotion

- QKV grouped decode;
- gate/up grouped decode;
- B1/B2/B4/B8 full-model measurements;
- projection-suite measurements;
- p95/tail checks;
- no persistent memory regression.

## Phase 6 — checkpoint/load-time compilation

- canonical payload stays on disk;
- grouping manifest authored by quantization/save lifecycle;
- fresh-process save -> load -> compile -> execute test;
- save compiled model -> canonical state round-trip.

## Phase 7 — optional post-promotion fusion

Only after exact grouped H100 is stable, investigate:

- shared-transform fusion into grouped decode;
- Q/K output transform + SV + RoPE preparation;
- gate/up output transform batching;
- down-input transform fusion into input stage;
- reduction optimization/cluster schemes when measurements justify them.

---

# 28. Immediate TODO

### PR84 before merge

- [x] Preserve A0-A41 experiment history.
- [x] Preserve exact P0/R0 runtime oracle result.
- [x] Preserve corrected P0+C/R0+C oracle result.
- [x] Record rejected A33/A34 five-H screen.
- [x] Record canonical checkpoint/runtime design.
- [x] Freeze branch authority policy in this document.
- [ ] Make sure the Torch-only A41/R0 oracle is clearly separable from CUDA performance code.
- [ ] Verify tests/docs still describe A0 as the safe default and A41/R0 as opt-in/experimental where native production support is incomplete.
- [ ] Rebase/merge latest target main as required by normal PR hygiene before final merge.
- [ ] Run the focused PR84 validation set after the final main synchronization.

### H100 implementation branch

- [ ] Rebase/synchronize `perf/qvq-p32-h100` onto the intended implementation base without importing an H200-only implementation contract.
- [ ] Add generic child segment descriptor.
- [ ] Expose the H100 plain-child schedule resolver to grouped compilation.
- [ ] Implement grouped transient P32 payload layout.
- [ ] Implement grouped QKV inner launch preserving per-child split/reduction plan.
- [ ] Implement grouped gate/up inner launch preserving per-child split/reduction plan.
- [ ] Prove native grouped child vs plain child exactness.
- [ ] Add shared complete input transform execution.
- [ ] Run full same-payload P0/R0 H100 model oracle.
- [ ] Benchmark and promote only after exactness.
- [ ] Integrate canonical checkpoint/load-time grouping after runtime promotion.

### Research deferred until after A41/R0

- [ ] More expressive learned persistent bases beyond rejected A33/A34 families.
- [ ] RoPE-compatible learned Q/K transforms.
- [ ] Exact/structured SwiGLU symmetry search.
- [ ] Cheaper replacement or fusion for the important down-input transform.
- [ ] New propagated post-quant correction objective.

---

# 29. Merge-to-main contract for PR84

PR84 should be mergeable as a **research/oracle/planner foundation**, not as a claim that every backend now has production A41 support.

The safe merge posture is:

- A0 remains the safe/default execution path unless an explicitly supported runtime compiler is installed.
- A41/R0 semantics and exactness tests are retained.
- Experimental optimization paths remain guarded/fail-closed.
- Canonical P32 storage semantics are not replaced.
- Rejected A33/A34 paths remain research history, not default behavior.
- No H100 production claim is made until implementation and validation occur on `perf/qvq-p32-h100`.

After merge, the main purpose of PR84 is to provide the stable contract that the H100 branch implements.

---

# 30. Final target

The near-term target is not five-H and not zero-H.

The target is:

```text
A25-compatible canonical P32 snapshot
        |
        v
R0 legality compiler
        |
        +-- incompatible group -> plain P32
        |
        +-- compatible QKV -> one shared input transform + one grouped H100 P32 launch
        |
        +-- compatible gate/up -> one shared input transform + one grouped H100 P32 launch
        |
        v
independent child output recovery
```

with these non-negotiable properties:

1. no requantization;
2. canonical child payload remains reconstructable exactly;
3. child-specific H100 scheduling is preserved;
4. post-quant recovery freedom is not sacrificed for grouping;
5. same-payload full-model runtime semantics remain exact;
6. unsupported/incompatible groups fall back cleanly;
7. speed is measured only after accuracy is proven.

That is A41/R0.

It is now the primary systems target.