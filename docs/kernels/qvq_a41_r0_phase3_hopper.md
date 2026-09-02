# QVQ A41/R0 Phase 3: exact grouped P32 on Hopper

Phase 3 realizes the grouped semantics from
[`qvq_a41_r0_phase1.md`](../qvq_a41_r0_phase1.md) in the native SM90a
TMA/register-sourced WGMMA path.  Canonical checkpoints and the Torch oracle do
not change.  The implementation consumes one shared transformed activation and
a transient, losslessly grouped continuous-window P32 payload.

## Operator math

R0 has already proved that every legal child has the same input scale and
input Hadamard.  Phase 3 receives the shared FP16 activation

\[
T = H(X \odot SU_{shared})
\]

and evaluates each child independently:

\[
Z_i = TQ_i,
\qquad
Q_i = \operatorname{P32Decode}(P_i, B_i, A_i).
\]

Here \(P_i\) is the child's continuous-window trellis, \(B_i\) is its packed
selector-byte tensor, and \(A_i\) is its alternative-bank ID.  The grouped
kernel returns FP32 \(Z_i\).  Output Hadamard, \(SV_i\), bias, rotary embedding,
and SwiGLU remain child-local and outside this kernel:

\[
Y_i = H_i(Z_i) \odot SV_i + b_i.
\]

## Storage-neutral TMA payload

For \(K_t=K/16\), transition width \(R\), and child output tiles
\(N_{t,i}=N_i/16\), each physical child window is

\[
P_i:[K_t,N_{t,i},4R],
\qquad
B_i:[K_t,N_{t,i}].
\]

The cached grouped payload concatenates only the output-tile dimension:

\[
P_G=\operatorname{cat}_{N_t}(P_0,\ldots,P_{G-1}),
\qquad
B_G=\operatorname{cat}_{N_t}(B_0,\ldots,B_{G-1}).
\]

This is a permutation-free concatenation of the existing child window format.
It adds no trailer, per-tile padding, selector duplication, or persistent
workspace.  `qvq_pack_p32_window_hopper_group()` performs the copy once; a
runtime must cache its result rather than repacking during inference.

One TMA tensor map describes each grouped tensor:

```text
trellis: [4R, sum(Nt_i), Kt]
bank:    [sum(Nt_i), Kt]
input:   [16, K]
```

Every child starts on a 256-column boundary, so both its trellis and selector
coordinates remain aligned to the existing 16-output-tile TMA transaction.

## Segmented Hopper grid

A fixed by-value descriptor supports the A41 maximum of three siblings:

```text
segment_count
n_tile_start[3]
n_tiles[3]
bank_alt_id[3]
split_count[3]
output_offset[3]
```

The one main-kernel grid is

```text
blockIdx.x = child-local 64-column output block
blockIdx.y = child segment
blockIdx.z = child-local K split
```

The grid bounds use the largest child width and split count.  Blocks outside a
smaller child's rectangle exit uniformly before constructing the TMA pipeline.
For an active block in child \(i\), the physical output-tile coordinate is

\[
n^{global}_{64} = N^{start}_{t,i}/4 + n^{local}_{64}.
\]

That coordinate feeds the grouped trellis and selector TMA maps.  Input TMA,
the two-stage producer/consumer pipeline, P32 state decode, shared PGC level
table, and register-sourced WGMMA instruction are otherwise identical to the
plain specialization.

Output is segment-major:

\[
Z_G=[\operatorname{flat}(Z_0),\ldots,\operatorname{flat}(Z_{G-1})].
\]

Therefore every returned `[16, N_i]` child is contiguous and needs no
materialization before its output recovery.

## Reduction-order boundary

Llama 3.2 1B's `K=2048` QKV and gate/up children resolve to split 1 in the
current Hopper policy.  Their grouped kernel executes the same K stages and
WGMMA accumulation sequence as each plain child and is required to be bit
exact.

The existing split-K Hopper kernel accumulates different CTA partials with
`atomicAdd`.  Moving those CTAs into a segmented grid can change their arrival
order even when every K interval is retained.  The public grouped packed API
therefore rejects split-K plans, while the pack-and-run convenience API fails
closed to ordinary child launches.  Phase 3 never claims exactness by treating
atomic addition as associative.  A future grouped split-K promotion requires a
deterministic child-local workspace and an explicitly matched reducer.

The plain kernel is compiled with `Grouped=false`; `if constexpr` removes all
segment lookup and coordinate-remap code.  Thus adding Phase 3 does not place a
dynamic segment branch in the existing single-child hot path.

## Public API

The intended cached lifecycle is:

```python
plan = qvq_p32_window_wgmma_group_plan(...)

# Only promote a fused group when every segment has split_count == 1.
payload = qvq_pack_p32_window_hopper_group(trellises, bank_ids, plan)

z_q, z_k, z_v = qvq_p32_window_wgmma_grouped_packed(
    padded_shared_m16,
    payload,
    levels,
)
```

The implementation lives in:

```text
gptqmodel/utils/qvq_wgmma_cuda.py
gptqmodel_ext/qvq/qvq_wgmma_cuda.cu
```

## Correctness gates

`tests/test_qvq_p32_hopper_grouped.py` requires:

1. child order, output boundaries, bank IDs, and independently resolved split
   policies survive planning;
2. grouped payload storage equals the sum of child storage and child slices are
   byte-exact;
3. W2/W2.5/W3/W3.5 grouped outputs at logical M1/M2/M4/M8/M16 are bit-exact to
   plain Hopper children;
4. the same outputs remain within `atol=2e-3, rtol=0` of the dense reconstructed
   P32 oracle;
5. real Llama 3.2 1B QKV (`K=2048`, `N=2048/512/512`) and gate/up
   (`K=2048`, `N=8192/8192`) groups are bit-exact at every target M;
6. split-K plans cannot enter the fused public path and fail closed to plain
   child execution.

Native development validation and performance measurement use only the
exclusive H100.  H100 and H200 both execute this same SM90a kernel, but results
must retain the device name and memory system because the two products have
different HBM capacity and bandwidth.

The H100 performance comparison against the byte-identical pre-Phase-3
`origin/main` Hopper sources is in
[`qvq_a41_r0_phase3_h100_benchmark.md`](qvq_a41_r0_phase3_h100_benchmark.md).
