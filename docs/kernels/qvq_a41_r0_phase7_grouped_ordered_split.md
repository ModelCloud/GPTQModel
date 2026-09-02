# QVQ A41/R0 Phase 7: grouped ordered split scheduling

Phase 7 adds deterministic, child-specific K-dimension splitting to the
grouped Hopper query/key/value kernel.  It builds on the Phase-6 ordered
reducer without changing canonical checkpoints, the continuous-window P32
payload, the decoder, or any existing H200 operation.

For Llama 3.2 1B, the grouped attention input has

\[
M\in\{1,2,4,8,16\},\qquad K=2048
\]

and three children:

\[
N_Q=2048,\qquad N_K=N_V=512.
\]

The kernel owns 64 output columns per thread block.  Without K splitting the
three children provide only

\[
2048/64+512/64+512/64=32+8+8=48
\]

useful blocks for the 132-multiprocessor H100.

## Flattened useful-work schedule

The existing grouped operation launches a rectangular grid using the widest
child and largest split count.  Narrow children reject excess blocks after
launch.  The Phase-7 operation instead assigns each child (i):

\[
B_i=N_i/64,\qquad S_i=\text{child split count},
\]

and a prefix

\[
W_i=\sum_{j<i}B_jS_j.
\]

The one-dimensional grid contains exactly

\[
W=\sum_iB_iS_i
\]

blocks.  A block with flat index (w) selects the largest child whose prefix
does not exceed (w), then computes

\[
u=w-W_i,
\]

\[
s=\left\lfloor u/B_i\right\rfloor,
\qquad
b=u-sB_i.
\]

Here (s) is the K partition and (b) is the child's 64-column block.  This
removes deliberately empty blocks from the ordered path.

## Child-local partial planes

Each child retains its own P32 payload, alternative-bank selector, output
width, and split count.  Split (s) computes

\[
P_{i,s}[m,n]=
\sum_{k=sK/S_i}^{(s+1)K/S_i-1}
X[m,k]Q_i[k,n]
\]

in the existing FP32 accumulator.  The partial workspace is a packed sequence
of child-local planes:

\[
\operatorname{workspace}=
[P_{0,0},\ldots,P_{0,S_0-1},
 P_{1,0},\ldots,P_{1,S_1-1},\ldots].
\]

No two decoder blocks write the same address.  One fixed reducer per child
evaluates

\[
Y_i=(((P_{i,0}+P_{i,1})+P_{i,2})+\cdots)+P_{i,S_i-1}
\]

left to right using aligned four-FP32-value loads.  Therefore block scheduling
cannot change accumulation order.

The transient storage is

\[
V=4\cdot16\sum_iS_iN_i\ \text{bytes}.
\]

At query/key/value split 8 this is 1.5 MiB.  It is temporary allocator storage
and adds no checkpoint or persistent model memory.

## Correctness contract

The grouped output for child (i) must be bit-identical to the independently
launched ordered child kernel using the same (S_i).  Both must remain within
2e-3 maximum absolute error of the reconstructed dense P32 matrix.  Repeated
eager launches and warmed CUDA Graph replays must also be bit-identical.

Split arithmetic is not required to equal split 1 bit-for-bit because the
FP32 summation parentheses differ.  Quantized weights and decoded values are
unchanged.

## Isolation

Phase 7 registers a separate grouped ordered operation.  The former grouped
split-1 operation and H200 atomic operations keep their existing template
specializations and launch behavior.  Production dispatch remains unchanged
until a complete H100 rate/M policy has passed the benchmark and correctness
gates.
