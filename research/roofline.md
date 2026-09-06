# Roofline: performance bounds and operational intensity

## Primary reference and finding

Samuel Williams, Andrew Waterman and David Patterson,
[Roofline: An Insightful Visual Performance Model for Multicore
Architectures](https://doi.org/10.1145/1498765.1498785), CACM 52(4), 2009.
[DOE publication record](https://www.osti.gov/biblio/1407073).

Roofline relates attainable floating-point throughput to operational intensity
and memory bandwidth, capped by a compute ceiling. It helps identify whether
reducing traffic or improving compute utilization is a plausible priority.
It is an upper-bound model, not a latency predictor for every workload.

## Proposed QVQ application

Define counted operations and the memory level before calculating intensity.
Packed weights reduce bytes but require decoding, scales and metadata. Report
logical dense-equivalent FLOPs separately from actual executed work.

Include codebooks, correction factors, split partials and output traffic.
Account for cache reuse rather than substituting checkpoint size for measured
DRAM traffic. Use datatype-appropriate compute ceilings; a CUDA-core FP32 roof
does not characterize every Tensor Core kernel.

For small decode grids, launch latency and insufficient parallelism may dominate.
For trellis kernels, integer/address/control work may impose additional limits.
Use [Nsight](nsight-profiling.md) counters to test these hypotheses and measure
the complete operator after tuning. A point below a roof does not by itself
identify the cause, and a bandwidth reduction does not prove lower latency.
