<!--
SPDX-FileCopyrightText: Copyright (c) 2011-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Bottleneck Analysis Guide

Root-cause analysis and optimization strategies for each bottleneck type identified by SOL% classification.

## SOL% Classification

| Compute % | Memory % | Bottleneck | Primary Section |
|-----------|----------|------------|-----------------|
| >60 | <40 | **Compute-bound** | ComputeWorkloadAnalysis |
| <40 | >60 | **Memory-bound** | MemoryWorkloadAnalysis |
| <40 | <40 | **Latency-bound** | LaunchStats + Occupancy |
| 40-60 | 40-60 | **Balanced** | Profile deeper with detailed sections |

Additional signals:
- Duration <10us with many launches: **Launch-overhead bound** (use nsys first)
- Both <40% but occupancy >50%: **Instruction-bound** (check InstructionStats)

## SOL% Performance Levels

| SOL% | Level | Action |
|------|-------|--------|
| >80% | Excellent | Minor tuning only |
| 60-80% | Good | Targeted optimization |
| 40-60% | Fair | Significant optimization needed |
| <40% | Poor | Major rework needed |

## Compute-Bound Kernels

**Symptoms:** Compute throughput >60%, Memory throughput <40%. Heavy arithmetic operations.

**Key Metrics:**
- `sm__throughput.avg.pct_of_peak_sustained_elapsed` — compute throughput
- `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed` — tensor core usage
- `smsp__inst_executed_pipe_fp64.avg.pct_of_peak_sustained_elapsed` — FP64 usage

**Root Causes:**
- Heavy FP operations without tensor core usage
- Inefficient math (FP64 when FP32 suffices)
- Warp divergence in compute paths

**Optimization Priority:**
1. Enable tensor cores (FP16/BF16/FP8 operations)
2. Use faster math intrinsics (`--use_fast_math`)
3. Reduce warp divergence
4. Algorithmic improvements (reduce FLOPs)

## Memory-Bound Kernels

**Symptoms:** Memory throughput >60%, Compute throughput <40%. Low arithmetic intensity.

**Key Metrics:**
- `dram__throughput.avg.pct_of_peak_sustained_elapsed` — DRAM bandwidth
- `l1tex__t_sector_hit_rate.pct` — L1 cache hit rate
- `lts__t_sector_hit_rate.pct` — L2 cache hit rate

**Root Causes:**
- Large data movement per operation
- Poor cache utilization (many misses)
- Uncoalesced memory access
- Shared memory bank conflicts

**Optimization Priority:**
1. Kernel fusion to increase arithmetic intensity
2. Improve data locality and cache reuse
3. Use shared memory for frequently accessed data
4. Ensure coalesced global memory access
5. Lower precision formats (FP16, INT8) to reduce bandwidth

**Memory becomes limiting when:** hardware units are fully utilized (Mem Busy), communication bandwidth between units is exhausted (Max Bandwidth), or memory instruction issue rate is maxed (Mem Pipes Busy).

## Latency-Bound (Low Occupancy)

**Symptoms:** Both compute and memory SOL are low and the scheduler is not issuing enough useful work. Low achieved occupancy can contribute, but is not required; high-occupancy kernels can still be dependency- or instruction-limited.

**Key Metrics:**
- `sm__warps_active.avg.pct_of_peak_sustained_active` — active warps
- `launch__occupancy_limit_registers` — register limit
- `launch__occupancy_limit_shared_mem` — shared memory limit
- `launch__occupancy_limit_blocks` — block limit

**Root Causes:**
- Register pressure that reduces useful residency
- Large shared memory per block / one-CTA-per-SM residency
- Insufficient grid waves or poor work distribution
- Long dependency chains, barriers, or async-pipeline waits even when occupancy is nominal

**Diagnosis Table:**

| Symptom | Cause | Solution |
|---------|-------|----------|
| Theoretical occupancy is resource-limited | Register pressure | Reduce live ranges/fragments first; test a register cap only if it does not spill/regress |
| Theoretical occupancy < 50% | Shared memory | Reduce shared memory per block |
| Achieved << Theoretical | Workload imbalance | Adjust grid/block dimensions |
| Both throughputs <40% | Low occupancy | Check LaunchStats for limiting resource |

**Optimization Priority:**
1. Identify the limiting dependency/stall and whether additional resident warps would hide it
2. Reduce register live ranges or shared footprint when they are the measured residency limiter
3. Adjust block dimensions/grid waves for the actual architecture and workload
4. Test register caps only as measured experiments; forced spills can be slower than lower occupancy

## Instruction-Bound

**Symptoms:** High instruction count relative to useful compute. Warp divergence indicators.

**Key Metrics:**
- `smsp__inst_executed.sum` — total instructions
- `smsp__thread_inst_executed_per_inst_executed.ratio` — divergence indicator (ideal=32)
- `smsp__inst_executed_pipe_cbu.avg.pct_of_peak_sustained_elapsed` — control flow overhead

**Root Causes:**
- Excessive control flow (branches)
- Warp divergence
- Many low-throughput instructions
- Instruction cache misses

**Optimization Priority:**
1. Simplify control flow
2. Use predication instead of branches
3. Reorganize data to reduce divergence
4. Unroll only when it removes loop/control/address work without excessive code size, registers, or instruction-cache pressure

## Launch-Overhead Bound

**Symptoms:** Very short kernel durations (<10us), many launches, high CPU time between them.

**Key Metrics:**
- `gpu__time_duration.sum` — kernel duration
- Launch count and CPU-GPU gaps (from nsys trace)

**Note:** This is better diagnosed with `nsys` (Nsight Systems) which shows the system-level timeline. Use `ncu` to confirm individual kernel performance after identifying hot kernels with `nsys`.

**Optimization Priority:**
1. CUDA Graphs to batch kernel launches
2. Fuse small kernels together
3. Increase work per kernel launch
4. Persistent kernels for repeated small work

## Quick Optimization Map

| Bottleneck Type | First Try | Second Try | Third Try |
|-----------------|-----------|------------|-----------|
| Compute-bound | Enable tensor cores | Mixed precision | Algorithmic opt |
| Memory-bound | Kernel fusion | Improve locality | Shared memory |
| Latency-bound | Adjust block size | Reduce registers | Increase parallelism |
| Instruction-bound | Simplify control flow | Use predication | Loop unrolling |
| Launch-overhead | CUDA Graphs | Kernel fusion | Persistent kernels |

## ncu vs nsys

| Tool | Scope | Overhead | Purpose |
|------|-------|----------|---------|
| **nsys** | System-level | 5-10% | Find which kernels to optimize |
| **ncu** | Kernel-level | 10-100x slower | Understand why a kernel is slow |

Use nsys first to identify top kernels by GPU time, then ncu for deep analysis of those specific kernels.


## Low-GB/s decision rule for compressed kernels

Do not classify a quantized P32/GPTQ/QTIP-style kernel as a bad-memory-layout
kernel just because it reaches a small fraction of peak HBM bandwidth. First ask:

1. How many compressed bytes should the algorithm actually read per output?
2. Are those bytes served by L1/L2 after reuse?
3. How many integer/address/shared instructions execute per decoded value?
4. Are eligible warps low because of dependent LUT loads, barriers, or MMA waits?
5. Are shared-memory requests expanding into extra wavefronts?
6. Is the Tensor Core pipe underfed while DRAM is idle?

If the answers point to decode/scheduler/shared dependencies, increasing DRAM
traffic is not an optimization objective.
