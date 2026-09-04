# QVQ A41/R0 Phase 2: exact grouped P32 on Ampere

This document defines the SM80 realization of the A41/R0 semantics in
[`qvq_a41_r0_phase1.md`](../qvq_a41_r0_phase1.md).  Checkpoints remain ordinary
per-module planar P32.  Grouping is a transient, lossless execution layout; it
does not change the quantization solution.

## Operator math

For legal children (i=0,ldots,G-1), Phase 1 proves that the input scale and
input Hadamard are identical.  Phase 2 therefore receives one already-shared
FP16 activation

\[
T = H(X \odot SU_{shared})
\]

and computes each inner projection independently:

\[
Z_i = T Q_i,
\qquad
Q_i = \operatorname{P32Decode}(P_i, B_i, A_i).
\]

(P_i) is the continuous-window P32 payload, (B_i) is its selector-byte
matrix, and (A_i\in[0,3]) selects the module-local alternative bank.  The
grouped kernel returns FP32 (Z_i).  Output Hadamard, (SV_i), and bias remain
child-local and are outside this kernel:

\[
Y_i = H_i(Z_i) \odot SV_i + b_i.
\]

No output epilogue, rotary embedding, or activation is fused in Phase 2.

## Lossless grouped payload

For (K_t=K/16), child (i) has

\[
P_i:[K_t,N_{t,i},4R],
\qquad
B_i:[K_t,N_{t,i}],
\]

where (R\in\{4,5,6,7\}) is the transition width for W2 through W3.5.
The cached execution payload is concatenated only along output tiles:

\[
P_G=\operatorname{cat}_{N_t}(P_0,\ldots,P_{G-1}),
\qquad
B_G=\operatorname{cat}_{N_t}(B_0,\ldots,B_{G-1}).
\]

This has exactly the sum of child P32 payload bytes.  It adds no trailer,
padding, duplicated trellis state, or persistent split workspace.

`qvq_pack_p32_window_ampere_group()` performs this lossless packing once.  A
runtime must cache the resulting `QVQAmpereGroupedP32Payload`; using the
convenience pack-and-run API on every token would copy the weights every call.

## Per-child launch policy

The group is never treated as a synthetic `(K, sum(N_i))` GEMM.  The Python
planner invokes the ordinary child split resolver independently and records:

```text
segment i
    output_tile_start
    output_tile_count
    out_features
    bank_alt_id
    split_count
```

Thus Q, K, and V may retain different split counts and alternative banks.  A
fixed descriptor for at most three A41 siblings is passed to CUDA by value;
there is no descriptor allocation or global-memory descriptor fetch.

The fused main grid is

```text
blockIdx.x = child-local N tile block
blockIdx.y = segment
blockIdx.z = child-local split
```

Blocks outside a smaller child's N or split range exit uniformly.  Active
blocks call the same device body as the ordinary P32 kernel with the child's
own K interval:

\[
k_{begin}=\left\lfloor\frac{K_t s}{S_i}\right\rfloor,
\qquad
k_{end}=\left\lfloor\frac{K_t(s+1)}{S_i}\right\rfloor.
\]

The row route is also unchanged:

| Rows | Phase-2 route |
| ---: | --- |
| M1-M4 | scalar decode/FMA, with the ordinary M2/M4 K-stage depths |
| M5-M15 | partial-row WMMA |
| M16 | full-row WMMA |

The main operation is one CUDA launch for all siblings.  When any child uses
split K, one grouped reducer launch follows, just as an ordinary split-K child
requires a reducer.

The output allocation is segment-major rather than a strided row-major slice:

\[
Z_G=[\operatorname{flat}(Z_0),\ldots,\operatorname{flat}(Z_{G-1})].
\]

Each returned child is therefore a contiguous `[M, N_i]` view.  Child output
Hadamard kernels do not pay a hidden materialization or non-contiguous fallback.

## Exact split reduction

Partial storage is compact and segment-local:

\[
\operatorname{offset}_{i+1}=
\operatorname{offset}_i + S_i M N_i
\]

for children with (S_i>1).  A child with (S_i=1) writes directly to its
slice of the concatenated output and consumes no partial workspace.

For split children, each output element is reduced in exactly increasing split
order:

\[
Z_i[m,n] = (((0 + P_{i,0}[m,n]) + P_{i,1}[m,n]) + \cdots)
            + P_{i,S_i-1}[m,n].
\]

This matches the ordinary scalar FP32 reducer.  The two existing KV cases that
use a warp-tree reducer (`N=1024`, M1-M4/split64 or M8/split48) deliberately
fall back to the exact per-child native dispatcher because their addition
tree is different.  Long-K groups and groups larger than three children also
fail closed to that dispatcher.  R0 never silently changes floating-point
order to force fusion.

## Public Phase-2 API

The implementation lives in:

```text
gptqmodel/utils/qvq_ampere_cuda.py
gptqmodel_ext/qvq/qvq_ampere_cuda.cu
```

The intended lifecycle is:

```python
plan = qvq_p32_window_ampere_group_plan(...)
payload = qvq_pack_p32_window_ampere_group(trellises, bank_ids, plan)

# Per inference call, after exactly one shared SU + input-Hadamard launch:
z_q, z_k, z_v = qvq_p32_window_ampere_grouped_packed(shared, payload, levels)
```

`qvq_p32_window_ampere_grouped()` is a correctness/convenience wrapper that
builds the plan and payload in one call.  Production integration must use the
cached packed form.

## Promotion gates

The Phase-2 tests require all of the following:

1. each child width is passed to the split resolver independently;
2. the grouped payload preserves ordered tile boundaries and bank metadata;
3. W2/W2.5/W3/W3.5 at M1/M2/M4/M8/M16 are bit-exact to ordinary child
   launches;
4. both scalar and WMMA routes remain within `atol=2e-3, rtol=0` of the dense
   FP32 reconstructed-P32 oracle;
5. realistic Llama 3.2 1B Q/K/V (`2048/512/512`) and gate/up
   (`8192/8192`) shapes are bit-exact to ordinary child launches;
6. the dynamic segmented body is bit-exact to the ordinary static-N routes at
   `N=5120/1024` for every supported rate and row count.

The development host has no SM80 GPU.  Native validation runs only on its H100
by JIT-compiling the extension's embedded compute-80 PTX; this is opt-in via
`QVQ_AMPERE_ALLOW_SM90_VALIDATION=1` and does not enable the Ampere path in
normal Hopper routing.  Final SM80 promotion still requires the same tests on
an A100.  The H100 compatibility-path benchmark is reported separately in
[`qvq_a41_r0_phase2_h100_benchmark.md`](qvq_a41_r0_phase2_h100_benchmark.md).
Its same-device comparison isolates the Phase-2 launch change, but its absolute
latencies must not be presented as an A100 result or as native Hopper-kernel
performance.

The native SM90a implementation of the same execution contract is documented
in [`qvq_a41_r0_phase3_hopper.md`](qvq_a41_r0_phase3_hopper.md).
