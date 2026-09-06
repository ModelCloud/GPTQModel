# Experiment25: GPU bit planes over 32 independent streams

The Triton kernel packs 32 stream states into 16 uint32 bit planes, advances the
planes by shift-register logic, injects packed symbol bits, and reconstructs scalar
state outputs. It is an actual bit-plane representation, not the earlier per-lane
scalar-state checkpoint decoder. Packing/unpacking costs are included. Circular
initial states come from the exact final window; no additional checkpoint bits
or LUT are stored. Original F6 data remains read only.

All tiles in all 94 projections match canonical states exactly at one and four
warps. Work was partitioned without overlap across four GPUs. Median direct-window
latency divided by bit-sliced latency is 0.43257x: the complete state-only bit-plane
pipeline is roughly 2.31x slower overall. This does not establish performance of a
future fused codebook/MMA implementation. Generated opcode/utilization evidence
for this kernel remains outstanding; no LOP3 throughput claim is made.

Raw samples retain the shared harness's legacy `scan_warps*` key for the candidate;
`bit_sliced=true` and experiment25 identify the actual implementation. Both methods
use the same payload and int32 output scratch. No added serialized metadata.

[Summary](results/bitsliced/summary.json), with all four worker reports nearby.
Experiment25 remains partial pending the requested full-operator/model scorecard.
