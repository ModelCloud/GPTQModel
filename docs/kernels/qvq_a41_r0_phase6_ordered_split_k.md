# QVQ A41/R0 Phase 6: deterministic Hopper split-K

Phase 6 fills the H100 for narrow-output Llama projections without weakening
the exact A41/R0 execution contract.  The first target is the Llama 3.2 1B
down projection

\[
M\in\{1,2,4,8,16\},\qquad K=8192,\qquad N=2048.
\]

The existing Hopper P32 kernel owns 64 output columns per thread block.  An
unsplit launch therefore has only

\[
B_N=N/64=32
\]

blocks for a 132-multiprocessor H100.  With (S) K partitions the decoder grid
contains

\[
B(S)=B_N S=32S
\]

blocks.  Each block traverses

\[
K_S=K/S
\]

input channels.  Splits 1, 2, 4, 8, 16, and 32 are legal because each
partition contains an integral number of the kernel's 256-channel stages.

## Ordered arithmetic

Let (X\in\mathbb{R}^{16\times K}) be the padded FP16 input and let the exact
P32 decoder reconstruct (Q\in\mathbb{R}^{K\times N}).  Partition K into S
contiguous intervals.  Split block (s) computes one FP32 partial plane:

\[
P_s[m,n]=\sum_{k=sK/S}^{(s+1)K/S-1}
  \operatorname{fp32}(X[m,k])\operatorname{fp32}(Q[k,n]).
\]

No two blocks write the same address.  A second kernel loads the planes as
aligned `float4` vectors and evaluates the fixed left-to-right order

\[
Y[m,n]=(((P_0[m,n]+P_1[m,n])+P_2[m,n])+\cdots)+P_{S-1}[m,n].
\]

This differs from the former atomic path in two important ways:

- accumulation order is independent of thread-block scheduling; and
- CUDA Graph replay and eager execution produce bit-identical results.

The optimized split result is compared against its own ordered arithmetic
reference.  It is not required to equal the unsplit kernel bit-for-bit because
split-K changes FP32 parenthesization.  Both paths must remain within 2e-3
maximum absolute error of the reconstructed dense P32 matrix.

## Kernel isolation and specialization budget

The ordered path is a separate registered operation.  Existing H200 and H100
atomic launches retain their former kernel specialization and generated code.
The decoder adds one ordered-output specialization for each supported P32
transition width, W2 through W3.5.  The reduction adds fixed specializations
for split 2, 4, 8, 16, and 32, with a scalar ordered fallback for other legal
counts.

The fixed reducer maps one thread to four adjacent FP32 output values.  For
the Llama down shape it launches

\[
16\cdot2048/4=8192
\]

threads worth of vector work, independent of split count.  The split loop is
fully unrolled.

## Transient memory

The deterministic workspace contains S FP32 partial planes:

\[
V_{partial}=S\cdot16\cdot N\cdot4\ \text{bytes}.
\]

For N=2048 this is 128 KiB per split, or 2 MiB at split 16.  The workspace is
temporary allocator storage, is capturable by CUDA Graphs, and does not change
checkpoint or persistent model VRAM.  Production policy must select a split
from measured latency and may choose a smaller split when the marginal decoder
gain does not pay for workspace traffic.

## A41/R0 integration boundary

PR #98 already provides the grouped Hopper operation, shared input scale and
Hadamard transform, child-local alternative banks, and canonical-checkpoint to
transient-window compilation.  Phase 6 does not rebuild those components.
Instead it supplies the deterministic reduction primitive needed before a
group plan may safely assign different split counts to Q, K, and V segments.

The production order is:

1. measure and promote the Llama down split policy by rate and M;
2. preserve ordinary unsplit behavior for all unmeasured shapes and devices;
3. extend ordered partial addressing to grouped child segments; and
4. allow the R0 planner to retain each child's independently selected split
   and reduction mode.
