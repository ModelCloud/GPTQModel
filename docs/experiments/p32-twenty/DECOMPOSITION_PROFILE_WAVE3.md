# Decoder decomposition profile wave 3

This queue wave ran `layer_baseline.py --profile` on all four GPUs, using the
same 12 real F6 seed-7 projections and the same nine row counts as
decomposition wave 2. Each job completed 27 rows, for 108 matched cases total;
all 108 candidate outputs passed the local reconstruction gate.

The profile hook warmed each case, verified GPU exclusivity, and captured one
planar inner call and one Ampere window inner call with CUDA profiler start/stop
markers. The reports preserve the teacher metrics and exact module/shape
mapping. They do not by themselves contain Nsight instruction counters or
Tensor Core occupancy counters; those require an external Nsight Systems or
Nsight Compute collection pass.

Raw reports:

- [worker 0](results/decomposition-profile-wave3/worker0.json)
- [worker 1](results/decomposition-profile-wave3/worker1.json)
- [worker 2](results/decomposition-profile-wave3/worker2.json)
- [worker 3](results/decomposition-profile-wave3/worker3.json)

This advances the matched profile coverage for experiment 1. External
instruction, register, occupancy, memory-traffic, and decoder/GEMM-overlap
counters remain open.
