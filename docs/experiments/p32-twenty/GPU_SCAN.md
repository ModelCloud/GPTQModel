# Experiment 21: GPU associative state scan

Implemented compact affine associative scan in Triton on the actual circular128
P32 stream with 16 state bits. Each transition composes a saturated shift and
16-bit suffix; the circular initial state is the final transfer's fixed point.
No 65,536-state transition table is materialized. Checkpoint bits remain fixed.

The same Triton harness compares direct circular-window extraction with scan at
4 and 8 warps, on every tile of all 94 saved P32 projections. All four configurations
match canonical GPU state reconstruction exactly. Integer equality ensures that
unchanged downstream bank/codebook mapping receives identical state indices.
This run does not execute or validate a new full linear operator.

Measured median direct/scan latency ratio (best of the two measured warp counts
per method and projection): **0.8477x**, range
0.7213–1.2208x. Below 1 means scan is slower. These are
state-materialization timings with five warmups and ten event samples; neither
method is fused with GEMM. They do not establish end-to-end speed or rejection of
all possible fused scan kernels. No automatic production selection was changed.

Both methods consume the same lossless window repack and allocate an int32 state
output. Reports include payload and scratch bytes; scan adds no serialized state
checkpoints or LUT. Complete model BPW remains that of the accepted snapshot,
with runtime repack/scratch accounted separately. Subsequent fused experiments
must account for their own storage and measure against the existing Ampere kernel.

[Raw measurements](results/gpu-scan/report.json). GPU 0, fixed F6 seed-7 snapshot
read only, no calibration or evaluation text involved in this state-only check.
This is partial experiment evidence, not completion of the full scorecard.
