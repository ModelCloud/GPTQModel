# QVQ A41/R0 Phase 5: exact fused Hopper MLP recovery

Phase 5 removes launch and intermediate-memory overhead around the grouped P32
Hopper kernel introduced in Phases 3 and 4.  It does not change the checkpoint,
P32 decoder, WGMMA arithmetic, quantization rate, or model activation.  The new
operators are exact replacements for existing FP16/FP32 operation sequences
and retain the ordinary path for every unsupported runtime state.

The implementation has two stages:

1. recover equal-width gate and up outputs in one CUDA grid and write FP16
   directly; and
2. combine the already-activated FP16 gate with the FP16 up output while
   applying the down projection's input scale and Hadamard transform.

## Exact operator math

Let the grouped P32 kernel return FP32 inner products

\[
Z_g = TQ_g, \qquad Z_u = TQ_u.
\]

Ordinary Phase-4 recovery executes two serial calls:

\[
G = \operatorname{fp16}\left(H_{fp16}(Z_g) \odot SV_g + b_g\right),
\]

\[
U = \operatorname{fp16}\left(H_{fp16}(Z_u) \odot SV_u + b_u\right).
\]

Here, \(H_{fp16}\) denotes the existing range-safe transform: finite values
are rounded at the same FP16 arithmetic boundaries, while overflowing values
remain FP32 until they return to the finite FP16 range.  Phase 5 schedules both
row sets in one grid and performs the former final casts at the output stores:

\[
(G,U)=\operatorname{PairRecover}(Z_g,Z_u,SV_g,SV_u,b_g,b_u).
\]

Each block still owns exactly one child row.  There is no cross-child
arithmetic, and every child retains its own scale and optional bias.  The
change is scheduling and storage, not algebra.

The original Llama MLP then computes its activation unchanged:

\[
A = \operatorname{SiLU}(G).
\]

PyTorch materializes the ordinary FP16 product

\[
P = \operatorname{fp16}(A \odot U)
\]

before the down projection applies

\[
D = H_{fp16}(P \odot SU_d).
\]

The Phase-5 precondition kernel evaluates the same sequence within one block:

\[
D = H_{fp16}\left(
    \operatorname{fp16}(A \odot U) \odot SU_d
\right).
\]

The product is explicitly rounded to FP16 before multiplication by \(SU_d\),
and the existing transform's normalization and butterfly rounding points are
preserved.  The down P32/WGMMA kernel and child-local output recovery are then
unchanged:

\[
Z_d=DQ_d, \qquad
Y=\operatorname{fp16}\left(H_{fp16}(Z_d)\odot SV_d+b_d\right).
\]

## Why SiLU remains separate

The runtime invokes the model's original activation function.  Replacing it
with a custom CUDA approximation changed a small number of FP16 values in
direct experiments and therefore cannot satisfy the same-payload bit-exact
contract.  Phase 5 only fuses operations after that established activation
boundary.  This leaves one activation launch, but preserves exact layer
outputs and full-model logits.

## Launch and dependency graph

```text
shared input scale + input Hadamard
    |
grouped gate/up P32 decode + WGMMA
    | produces two FP32 inner tensors
paired output recovery                         one launch
    | produces FP16 gate and FP16 up
original model activation                      unchanged
    | produces activated FP16 gate
rounded gate*up + down scale + input Hadamard  one launch
    | produces preconditioned FP16 down input
ordinary down P32 decode + WGMMA
    |
ordinary down output recovery
```

The paired-recovery blocks are independent and may run gate and up rows
concurrently.  The activation remains a real dependency between recovery and
the precondition kernel.  The precondition output is a real dependency of the
down projection; no stream or host synchronization is introduced.

## Specialization budget

Phase 5 adds exactly two fixed device kernels:

| Kernel | Compile-time variants | Runtime dimensions |
|---|---:|---|
| paired FP32-to-FP16 output recovery | 1 | rows, width, bias presence, recovery mode |
| FP16 SwiGLU/down precondition | 1 | rows, width |

Rates W2, W2.5, W3, and W3.5 share both operators.  M1/M2/M4/M8/M16 share
both operators.  Width and optional bias do not create template variants.
This keeps build cost bounded and avoids duplicating the P32 kernel matrix.

## Promotion and fallback gates

Paired recovery requires:

- exactly two grouped gate/up children;
- equal power-of-two output widths no greater than 16,384;
- FP32 contiguous inner outputs and child-local FP32 scales/biases; and
- the existing exact grouped Hopper runtime eligibility gate.

The complete MLP wrapper additionally requires:

- a structurally recognized safe MLP parent and its original activation;
- a direct QVQ down projection with matching gate/up geometry;
- evaluation mode, no adapters, FP16 CUDA input, SM90, and M from 1 through
  16; and
- an exact one-row installation-time comparison with the original parent.

Any rejection restores or invokes the original parent path.  Unsupported
gate/up geometry uses independent child recovery.  The installer never treats
failure to fuse as a model error.

## Transient VRAM

The grouped P32 payload remains storage-neutral as documented in Phase 4.
Phase 5 changes only transient activation storage.

Before paired recovery, the live recovery chain holds two FP32 recovered
outputs before their separate FP16 casts.  Direct FP16 stores remove

\[
2MN\operatorname{sizeof}(\text{float})=8MN\ \text{bytes}.
\]

At the Llama 3.2 1B gate/up width \(N=8192\), this is 64 KiB at M1 and 1 MiB
at M16.  The two FP32 WGMMA inner outputs remain live because their exact
recovery still consumes them.

The precondition kernel does not materialize the separate FP16 gate/up product,
removing another

\[
MN\operatorname{sizeof}(\text{half})=2MN\ \text{bytes},
\]

or 16 KiB at M1 and 256 KiB at M16.  No persistent buffer, decoded-weight
cache, trailer, or split workspace is added.

## Correctness and timing contract

Low-level tests compare both new kernels bit-for-bit with their former
multi-launch sequences across W-independent transform geometry, M1/M2/M4/M8/
M16, multiple seeds, overflow cases, optional biases, non-default streams, and
CUDA Graph capture.  Runtime tests cover paired and independent recovery,
activation-fusion enable/disable, M17 fallback, uninstall, and a real
Transformers Llama decoder layer with exact logits and cached generation.

Performance is measured only on the physical H100.  The benchmark fails before
importing Torch unless that H100 is explicitly visible and idle across at least
three samples.  Warmed CUDA Graph replays are bracketed by CUDA events so CPU
or container starvation is outside the measured interval.

The H100 matrix and comparison with the previous committed benchmark are
recorded separately in `qvq_a41_r0_phase5_h100_benchmark.md`.
