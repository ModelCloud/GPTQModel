# Experiment 24: compact GPU super-symbol transfers

Groups of 2, 4, and 8 transitions are represented by a saturated shift and packed
suffix, followed by an associative boundary scan and local state reconstruction.
This is a factored affine super-symbol implementation, not the proposed exponential
transition/output LUT. It preserves the exact 16-bit state and circular boundary.
Each group size ran at 4 and 8 warps alongside direct circular-window extraction.

All configurations exactly matched canonical state reconstruction for every tile
of all 94 P32 projections. No checkpoint or quantization metadata was modified.
The initial experimental kernel had a negative circular gather index; its workers
failed with invalid accesses. After correcting the wrap and verifying fresh CUDA
contexts were healthy, the complete runs passed. Failed logs remain external.

The JSON summary reports same-device direct/candidate state-only latency ratios;
values below 1 mean the candidate is slower. Each method uses the best median
of the two measured warp configurations. These are unfused state-materialization
measurements and cannot establish full linear-layer speed, decoder/GEMM overlap,
or completion of the model scorecard. No exponential LUT or checkpoint metadata
is added; shared repack payload and int32 output scratch are explicitly reported.

[Summary](results/gpu-supersymbol/summary.json), with full per-projection measurements
in the same directory. Fused output lookup and MMA integration remain outstanding.
