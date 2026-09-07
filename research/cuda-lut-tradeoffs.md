# CUDA lookup tables: fewer instructions can cost more time

## Finding and scope

Replacing arithmetic with a LUT exchanges compute for address generation,
dependent memory access, storage and often synchronization. It can reduce
source operations yet increase latency. That is a useful QVQ hypothesis,
not a universal rule that GPUs dislike LUTs or shared memory.

NVIDIA documents serialization when distinct shared-memory words in a request
hit the same bank, and broadcasting when lanes read the same location. Constant
memory has a different rule: divergent addresses within a warp serialize.
See [CUDA Best Practices, shared and constant memory](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html).
**Sharing an address is not the same as a bank conflict.**

## Mechanisms to investigate

The following are engineering hypotheses to distinguish with measurements.

| Mechanism | Why a LUT can lose | Diagnostic / candidate remedy |
|---|---|---|
| Shared bank conflicts | Different table words compete within a transaction | Inspect actual lane addresses and access widths; test layout or lane-dependent replicas |
| Dependent lookup chain | Load/index/lookup/consumer leaves less independent work | Check dependency stalls and issue gaps; prefetch or interleave independent tiles |
| Shared-port pressure | Conflict-free LUT reads still consume bandwidth needed by fragments | Compare shared traffic and throughput; fuse/vectorize lookups where legal |
| Table footprint | LUT storage displaces staging buffers or reduces resident blocks | Measure shared allocation, registers and complete launch occupancy limits |
| Global/cache lookup | Scattered gathers add sectors, misses or dependent cache-hit latency | Inspect actual memory space and locality; compare cache policies at matched reuse |
| Initialization and barriers | Table setup is not amortized on small work | Include setup in timing; separate per-model, per-launch and per-block costs |
| Register selection | A dynamic array index may become selects, shuffles or local memory | Inspect generated SASS and spills; do not assume a C++ array stays cheaply indexed |
| Changed scheduling | Shorter source code can leave a longer critical path | Check eligible work, pipeline overlap and full operator latency |

For ordinary 32-bit shared accesses on a 32-bank layout, a useful model is
`bank = (byte_address / 4) % 32`. Check the actual target and instruction.
For wide/vector instructions, analyze their component transactions; necessary
multiple transactions are not automatically bank-conflict overhead.

Replication is useful only if the address mapping changes which banks the
competing lanes use. A copy at a bank-period-aligned offset with the same mapping
does not itself remove conflicts. Padding fixes some regular strides, not every
data-dependent lookup pattern. More replicas also cost storage.

## Existing QVQ evidence

[Experiments 5/6](https://github.com/ModelCloud/QvQ/blob/66565c27ed8a42639c0c2bbe55fdb4a8e677dca0/docs/experiments/p32-twenty/LOOKUP.md)
report exact decoded pairs over 94 projections at 1/4 warps:

| Materialized output | Index LUT: median direct/LUT | Pair LUT: median direct/LUT |
|---|---|---|
| FP32 | 0.9120x | 0.8161x |
| Packed half2 | 0.9714x | 0.8600x |

Below 1 means the LUT was slower. The 65,536-entry tables have approximately
128 KiB index and 256 KiB pair payloads. This is an unfused materialization
experiment, not a fused GEMM or model-speed result. The report explicitly leaves
causal profiling open.

The [implementation](https://github.com/ModelCloud/QvQ/blob/66565c27ed8a42639c0c2bbe55fdb4a8e677dca0/scripts/p32_twenty/lookup_decoder_gpu.py)
creates CUDA tensors and reads INDEX/PAIRS through Triton `tl.load`.
It does not explicitly stage these tables in CUDA shared memory. “Shared table”
in the experiment description must not be turned into a claim about physical
shared-memory residency or measured bank conflicts. Inspect generated code
and counters before attributing this slowdown.

## Evidence that LUTs can work

[FLUTE](flute.md) designs lookup layout and execution together.
[QUICK](quick.md) studies a related dequantization write-back/layout problem.
[LUT-GEMM](lut-gemm.md) uses a different LUT-based computation.
These are scoped counterexamples to a blanket rejection of lookup methods;
none validates a large P32 state LUT by analogy.

## Proposed QVQ decision

Keep direct arithmetic as the comparison baseline for this experiment.
A LUT candidate should show a plausible traffic/dependency advantage before
expanding the sweep. Measure direct, lookup and fused paths at matched packing,
output dtype, shapes and cache conditions. Record bank/sector behavior,
dependency stalls, registers, setup and end-to-end timing together.

Retain exact codebook/state semantics and the existing kernel accuracy gates.
Reject or retain candidates on measured complete-operator results, not
instruction reduction alone. See [metric interpretation](cuda-metrics-and-performance.md).
