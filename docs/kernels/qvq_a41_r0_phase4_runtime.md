# QVQ A41/R0 Phase 4: production Hopper runtime

Phase 4 connects the exact grouped P32 operator from Phases 1–3 to the
production model graph.  It does not change the quantized checkpoint or CUDA
kernel.  Architecture-declared QKV and gate/up siblings keep their existing
`QVQLinear` modules and canonical payloads; inference replaces only their
forward methods with an exact, fail-closed coordinator.

## Production operator

For child (i), ordinary inference evaluates

\[
T_i = H(X \odot SU_i),
\qquad
Z_i = T_i Q_i,
\qquad
Y_i = H(Z_i) \odot SV_i + b_i.
\]

R0 promotes a group only after proving bit-for-bit that

\[
SU_0 = SU_1 = \cdots = SU_{G-1}
\]

and that every other input-side codec property agrees.  Production grouped
execution is therefore

\[
T = H(X \odot SU_{shared})
\]

once, followed by the Phase-3 segmented Hopper kernel

\[
(Z_0,\ldots,Z_{G-1}) =
\operatorname{GroupedP32Hopper}(T,P_G,B_G,A_0,\ldots,A_{G-1}),
\]

and the unchanged child-local recovery

\[
Y_i = H(Z_i) \odot SV_i + b_i.
\]

`QVQLinear._qvq_prepare_inference_input()` and
`QVQLinear._qvq_recover_inference_output()` are the single implementation of
those two transforms.  Both ordinary and grouped execution call these helpers,
so the coordinator cannot silently reorder a scale, Hadamard, bias, cast, or
FP32 accumulation boundary.

Phase 4 preserves the current P32 checkpoint semantics in which every
`QVQLinear` has an output Hadamard.  Architecture-level output-axis folding is
a separate quantization transformation and is not inferred by this runtime.

## Exact R0 promotion gate

An installed group must contain two or three evaluation-mode `QVQLinear`
children and satisfy all of the following:

- canonical V2B2-P32, vector size 2, window 16, and two banks;
- W2, W2.5, W3, or W3.5 with one codebook version;
- identical K, device, and bit-identical `SU` dtype, shape, device, and values;
- K and every child N divisible by 256;
- concrete co-located trellis, selector, and alternative-bank tensors;
- no training, adapter, or pre-existing fusion owner;
- each child's existing Hopper split policy resolves to split 1.

Split-K remains outside the exact grouped path because the existing atomic
reducer does not define a stable cross-CTA addition order.  An ineligible group
is not an error and keeps ordinary per-child execution.

`BaseQModel.fuse()` obtains ordered candidates from the model definition's
`module_tree`, installs QVQ groups first, and then lets the existing GPTQ
fusion scanner handle unrelated modules.  The public low-level installer also
supports the established QKV and gate/up naming conventions.

## Sibling lifecycle

Hugging Face Llama invokes Q, K, and V as separate Python module calls.  The
coordinator implements one strict cycle:

```text
first sibling
    validate runtime input
    shared SU/H transform
    one grouped Hopper launch
    all child-local recoveries
    return child 0 and retain later outputs

ordered later siblings
    require the same tensor object and mutation version
    return exactly one retained output

last sibling
    clear the cycle immediately
```

A missing primary, duplicate, out-of-order consumer, changed tensor, mutated
tensor, training input, unsupported dtype, non-SM90 device, or M outside 1–16
clears the cycle and invokes that child's original forward.  Before any plain
fallback, the runtime drops its grouped payload so a later child window cannot
coexist with it and inflate persistent VRAM.

The warmed path performs no tensor-to-host condition.  Payload construction,
selector validation, alternative-bank extraction, source equality checks, and
extension compilation occur before CUDA Graph capture.  Capture records the
shared transform, grouped kernel, and child recoveries; graph replay performs
no Python lifecycle work.

## Storage and invalidation

For transition width (R), the children would independently cache

\[
\sum_i K_t N_{t,i}(4R)
\]

32-bit window words.  Phase 4 stores exactly

\[
P_G:[K_t,\sum_i N_{t,i},4R]
\]

with the same word count.  It also stores one grouped selector byte per tile,
equal to the sum of the child selector bytes.  There is no trailer, padding,
split workspace, or dequantized weight cache.

For Llama 3.2 1B, persistent grouped-window sizes are:

| Rate | QKV window | gate/up window | Child windows avoided | Extra bytes versus child windows |
|---:|---:|---:|---:|---:|
| W2 | 1,572,864 | 8,388,608 | 9,961,472 | 0 |
| W2.5 | 1,966,080 | 10,485,760 | 12,451,840 | 0 |
| W3 | 2,359,296 | 12,582,912 | 14,942,208 | 0 |
| W3.5 | 2,752,512 | 14,680,064 | 17,432,576 | 0 |

Canonical checkpoint payloads remain resident because exact fallback and
`state_dict()` still require them.  This matches ordinary Hopper P32, which
also retains canonical storage plus one equal-sized window cache.

The grouped cache key covers tensor identity and mutation version for trellis,
selectors, alternative-bank metadata, and `SU`, plus every static codec field.
Source replacement/mutation rebuilds or rejects the group.  `Module._apply`
invalidates the external cache before a device or dtype move, releasing the
old device allocation promptly.

## Device dispatch

The existing plain P32 Hopper dispatcher now admits both H100 and H200 SM90
devices.  This makes the production parity gate meaningful on H100: grouped
children are compared against the same TMA/register-sourced WGMMA arithmetic,
not against the older planar CUDA fallback.  No CUDA source changed in Phase
4.

## Correctness gates

`tests/test_qvq_grouped_runtime.py` covers:

1. exact R0 acceptance and unequal-`SU` rejection;
2. one launch per ordered sibling cycle and stale-cache prevention;
3. bit-exact production output at Llama 3.2 1B QKV and gate/up shapes for
   M1/M2/M4/M8/M16;
4. grouped-window bytes equal to the removed child-window bytes;
5. warmed CUDA Graph capture and replay on H100;
6. a real Transformers Llama decoder layer with exact logits and exact cached
   greedy generation before and after installation;
7. uninstall and ordinary-forward restoration.

The H100 full-operation benchmark is recorded in
[`qvq_a41_r0_phase4_h100_benchmark.md`](qvq_a41_r0_phase4_h100_benchmark.md).
