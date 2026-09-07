# FLUTE: making lookup-table quantization GPU-efficient

## Primary references and findings

Han Guo et al.,
[Fast Matrix Multiplications for Lookup Table-Quantized LLMs, v3](https://arxiv.org/html/2407.10960v3),
especially §§3.1–3.3.
[Authors' open-source implementation](https://github.com/HanGuo97/flute).

FLUTE combines offline weight restructuring, vectorized table lookups,
bank-aware table replication and workload partitioning. The paper identifies
shared-memory traffic and conflicts as problems for naive lookup
dequantization, then designs around them. It reports speedups for its evaluated
LUT quantizers and shapes; LUTs are therefore not inherently slower than
arithmetic on GPUs.

## QVQ implication (proposed)

Study its mapping of table entries to lanes/banks and its fragment-compatible
weight layout. Copying only the idea of a LUT omits the surrounding design.

FLUTE's small low-bit codebook tables and QVQ's full 65,536-state tables have
different storage and reuse properties. Estimate table size, replication cost,
staging space and conflict behavior before transferring the technique.

Compare conflict mitigation against direct PGC arithmetic under the same
packing and complete operator. A reduced shared conflict count does not alone
pay for extra setup or data rearrangement. Keep state/codebook bit patterns
unchanged if the experiment claims lossless decoding.

See [QVQ LUT evidence](cuda-lut-tradeoffs.md) and
[pipeline throughput](cuda-pipeline-throughput.md). No FLUTE integration or
QVQ speedup is demonstrated by this note.
