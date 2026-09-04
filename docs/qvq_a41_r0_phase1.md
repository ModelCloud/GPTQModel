# QVQ A41/R0 Phase 1: grouped P32 Torch oracle

This document defines the first A41/R0 implementation.  It is intentionally a
pure Torch semantic layer.  It does not change checkpoint serialization,
quantization, or any Ampere/Hopper kernel.

## Goal and boundary

Ordinary QVQ executes each projection independently:

\[
t_i = H(x \odot SU_i),\qquad
z_i = t_i Q_i,\qquad
y_i = H_i(z_i) \odot SV_i + bias_i.
\]

Here \(Q_i\) is the exact FP32 inner matrix reconstructed from the child’s
canonical P32 trellis and selector payload.  A legal A41 group evaluates the
common input transform once:

\[
t = H(x \odot SU_{shared}),\qquad
y_i = H_i(t Q_i) \odot SV_i + bias_i.
\]

The inner matrix and output recovery remain child-local.  In particular, a
member may set `output_hadamard=False` when an architecture has folded that
output transform into a later operation; its `SV` and bias are still never
shared.

The implementation lives in
[`gptqmodel/quantization/qvq_grouped.py`](../gptqmodel/quantization/qvq_grouped.py).
The native CUDA paths are deliberately not imported by this module.

For the A25/A31/A41 basis used by Llama-family blocks, the intended groups are:

| Projection | Input basis | Input H | Output H | Group |
|---|---|---:|---:|---|
| `q_proj` | attention input | yes | yes | Q/K/V |
| `k_proj` | attention input | yes | yes | Q/K/V |
| `v_proj` | attention input | yes | architecture-dependent | Q/K/V |
| `o_proj` | independent | architecture-dependent | yes | none |
| `gate_proj` | MLP input | yes | yes | gate/up |
| `up_proj` | MLP input | yes | yes | gate/up |
| `down_proj` | independent | architecture-dependent | yes | none |

The ungrouped basis repeats the input transform for all three Q/K/V children
and both gate/up children.  A legal A41 execution performs those five input
transforms as two shared transforms, reducing the block’s transform count from
12 to 9 while leaving every child’s output recovery independent.

## Descriptors and groups

An architecture emits one `QVQExecutionDescriptor` per child:

```python
QVQExecutionDescriptor(
    module_name="layer.0.self_attn.q_proj",
    input_basis_id="layer.0.self_attn.input",
    input_hadamard=True,
    output_hadamard=True,
)
```

`QVQGroupedP32Spec` contains the basis ID and the ordered member names.  The
order is part of the contract because it fixes output-tile boundaries and
keeps a future grouped kernel’s metadata deterministic.

R0 accepts a group only when all of these conditions hold:

* every child is a concrete `QVQLinear` in V2B2-P32 format;
* `vector_size == 2`, `trellis_window == 16`, and `bank_count == 2`;
* all children have the same `K`, rate, codebook version, and trellis geometry;
* all descriptors use one `input_basis_id` and agree on `input_hadamard`;
* every child’s `SU` has the same shape, dtype, device, and values bit-for-bit;
* each output width is a positive multiple of 16;
* each child has valid two-bank selectors and a scalar `bank_alt_id` in `[1, 3]`.

Different `SV`, bias, output widths, output-Hadamard flags, selector bytes, and
alternative-bank IDs are legal.  Any failed rule is a normal fail-closed
fallback condition: the caller should execute ordinary per-module P32.

## Transient grouped payload

Checkpoint storage remains unchanged.  For a child with

```text
Ktiles = K / 16
Ntiles = N / 16
W      = canonical P32 words per 16x16 tile
```

the canonical payload is represented as:

```text
trellis  [Ktiles * Ntiles, W]   int32
bank_ids [Ktiles] x [Ntiles]   uint8, one packed byte per tile
```

The grouping helper reshapes each tensor to `[Ktiles, Ntiles, ...]` and
concatenates along `Ntiles`:

```text
grouped.trellis  [Ktiles * sum(Ntiles_i), W]
grouped.bank_ids [Ktiles * sum(Ntiles_i)]
```

Each `QVQGroupedP32Segment` records `output_tile_start`,
`output_tile_count`, the original output width, and that child’s
`bank_alt_id`.  `ungroup_canonical_p32_payload()` slices the grouped tensors
back to the original shape.  The Phase 1 invariant is byte equality:

```python
torch.equal(original.trellis, recovered.trellis)
torch.equal(original.bank_ids, recovered.bank_ids)
torch.equal(original.bank_alt_id, recovered.bank_alt_id)
```

No grouped tensor is installed on a child and no quantization state is
rewritten.

## Torch oracle

`qvq_torch_child_oracle()` is the explicit FP32 child reference.  It accepts a
descriptor so output recovery can be represented without changing
`QVQLinear`.

`qvq_torch_group_oracle()` first validates and snapshots the grouped payload,
then performs exactly one shared input scale/Hadamard transform.  It
reconstructs each child inner matrix separately and computes each child GEMM,
output transform, `SV`, and bias separately.  It intentionally does **not** use
one concatenated GEMM: a larger GEMM could select a different reduction tree
and obscure an execution-semantic error.

Both functions:

* run under `torch.inference_mode()`;
* return detached FP32 tensors with the input’s leading dimensions preserved;
* reject gradient-tracking inputs before reconstructing any payload;
* do not populate `QVQLinear` caches or mutate child buffers.

For descriptors with both Hadamard flags enabled, the grouped result must be
`torch.equal()` to independent calls of the existing
`qvq_dense_oracle_forward()`.  That is stronger than the native-kernel
`2e-3` accuracy gate because this phase has no low-precision execution.

## Tests

[`tests/test_qvq_grouped_oracle.py`](../tests/test_qvq_grouped_oracle.py) covers:

1. two- and three-member payload round trips with distinct selectors and bank IDs;
2. exact grouped-vs-independent dense oracle equality;
3. child-local output recovery with `output_hadamard=False`;
4. fail-closed behavior for different `SU`, basis IDs, Hadamard flags, order, and formats;
5. gradient rejection and cache/state non-mutation.

Phase 2 may implement the same descriptor and segment contract in the SM80
path.  Phase 3 may implement it in the SM90a TMA/RS-WGMMA path.  Neither phase
may change this Torch definition of A41 semantics.

The Phase-2 implementation and its exact split/reduction contract are
documented in
[`kernels/qvq_a41_r0_phase2_ampere.md`](kernels/qvq_a41_r0_phase2_ampere.md).
