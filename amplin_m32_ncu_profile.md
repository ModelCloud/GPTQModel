# Amplin M=32 remaining losses — Nsight Compute profile and strategy

Branch: `devin/1784971296-amplin-m32-ncu`  
GPU: NVIDIA PG506-230 (A100-class), sm_80, 124 SMs, 96 GiB  
Config: GPTQ 4-bit, group_size=128, desc_act=False, sym=True, FP16  
Profiled kernels (selected via `ncu -k`):

- `Amplin splitk16` = `amplin_mma_lane_m32_n64_splitk16_pipe2_interleaved_kernel`
- `Amplin tile2` = `amplin_mma_lane_mN_n64_tiled_fullk_kernel<__half, 32, 2, 1>`
- `Marlin` = `Marlin<__half, ...>` (the packed W4A16 prefill kernel)

## Baseline timing (FP16, GPU 0, `benchmark_amplin_m32_tile4.py`, `--iters 50 --warmup 10`)

| model | role | M | K | N | best Amplin | Amplin us | Marlin us | ratio |
|---|---|---:|---:|---:|---|---:|---:|---:|
| glm-5.2 | lm-head | 32 | 6144 | 154880 | tile2 | 533.4 | 454.4 | 1.17 |
| kimi-k2.5 | dense-up | 32 | 7168 | 18432 | splitk16 | 138.7 | 119.7 | 1.16 |
| kimi-k2.5 | lm-head | 32 | 7168 | 163840 | tile2 | 687.5 | 550.9 | 1.25 |

## NCU command used

```bash
ncu --section SpeedOfLight \
    --section MemoryWorkloadAnalysis \
    --section ComputeWorkloadAnalysis \
    --section Occupancy \
    --section LaunchStats \
    --section InstructionStats \
    --metrics smsp__average_warp_latency_issue_stalled_long_scoreboard,...
    -k 'regex:<kernel-name>' \
    --launch-count 1 \
    python /tmp/profile_m32_shape.py --m <M> --k <K> --n <N> --kernel <kernel>
```

## 1. Core throughput comparison

| kernel | shape | duration (us) | memory (GB/s) | memory % | DRAM % | compute (SM) % | SM Busy % | achieved occupancy |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Amplin splitk16 | kimi dense-up | 113.28 | 635.1 | 59.78 | 25.97 | 29.35 | 29.35 | 24.79% |
| Marlin | kimi dense-up | 70.34 | 1050.0 | 42.92 | 42.92 | 39.44 | 39.44 | 6.25% |
| Amplin tile2 | glm lm-head | 515.68 | 972.4 | 39.75 | 39.75 | 48.95 | 48.95 | 27.92% |
| Marlin | glm lm-head | 428.83 | 1180.0 | 48.24 | 48.24 | 46.20 | 46.20 | 6.25% |
| Amplin tile2 | kimi lm-head | 683.78 | 903.8 | 36.94 | 36.94 | 45.70 | 45.70 | 27.69% |
| Marlin | kimi lm-head | 523.23 | 1190.0 | 48.57 | 48.57 | 46.72 | 46.72 | 6.25% |

Observations:

- Marlin is faster even though its **occupancy is much lower** (6.25% vs 24-28%).
- Marlin achieves **35-65% higher memory throughput** in absolute GB/s on every shape.
- For `lm-head`, Amplin `tile2` is close to Marlin in memory throughput (~90%) but is killed by stalls.

## 2. Launch configuration / occupancy

| kernel | shape | grid | block | threads total | regs/thread | dynamic smem/block | shared config | block limit warps | waves/SM |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Amplin splitk16 | kimi dense-up | 288 | 512 | 147456 | 113 | 65.54 KB | 102.40 KB | 4 | 2.32 |
| Marlin | kimi dense-up | 124 | 128 | 15872 | 213 | 166.91 KB | 167.94 KB | 16 | 1.00 |
| Amplin tile2 | glm lm-head | 1210 | 128 | 154880 | 86 | 0 | 102.40 KB | 16 | 1.95 |
| Marlin | glm lm-head | 124 | 128 | 15872 | 213 | 166.91 KB | 167.94 KB | 16 | 1.00 |
| Amplin tile2 | kimi lm-head | 1280 | 128 | 163840 | 86 | 0 | 102.40 KB | 16 | 2.06 |
| Marlin | kimi lm-head | 124 | 128 | 15872 | 213 | 166.91 KB | 167.94 KB | 16 | 1.00 |

Key takeaways:

- `splitk16` is **limited to one block per SM** by both shared memory (65 KB) and registers (113). It launches 512 threads (16 warps) but only one block per SM.
- `tile2` has much lower register pressure (86) and small shared memory, so it can put ~2 waves on each SM, but it is still slow because each warp has too little independent work and too many dependencies.
- Marlin uses **one large block per SM** with 128 threads and 213 registers, consuming 167 KB shared. The block is large enough to keep the SM busy with independent `cp.async` memory and `mma` math in flight, despite low occupancy.

## 3. Warp stall comparison (ratio per warp instruction, lower is better)

| stall reason | Amplin splitk16 dense-up | Marlin dense-up | Amplin tile2 glm lm-head | Marlin glm lm-head | Amplin tile2 kimi lm-head | Marlin kimi lm-head |
|---|---:|---:|---:|---:|---:|---:|
| long_scoreboard | 20,687 | 3,740 | 69,284 | 20,701 | 82,414 | 24,479 |
| barrier | 2,049 | 3,501 | 10,814 | 13,263 | 12,868 | 15,138 |
| math_pipe_throttle | 3,006 | 75 | 42,183 | 334 | 49,444 | 351 |
| not_selected | 2,031 | 0 | 32,032 | 0 | 37,474 | 0 |
| lg_throttle | 3,270 | 17 | 2,988 | 18 | 3,428 | 18 |
| mio_throttle | 242 | 35 | 3,645 | 177 | 5,368 | 186 |
| no_instruction | 87 | 2,244 | 1,207 | 12,853 | 1,395 | 15,443 |
| membar | 0 | 147 | 0 | 153 | 0 | 152 |

Key takeaways:

- **Marlin `not_selected` stalls are 0** for all three shapes; Amplin has thousands. This means Marlin keeps enough warps eligible that the scheduler always has something to do, while Amplin warps are blocked on barriers/dependencies.
- **Amplin `math_pipe_throttle` is orders of magnitude larger** than Marlin, especially on `tile2` (42k-49k vs ~350). The per-instruction math pressure from Amplin's lane-based dequantization and `mma` scheduling is far higher.
- **High `long_scoreboard` on both** shows both kernels spend most stall cycles waiting for global/shared memory dependencies, but Marlin's ratio is much lower.
- **Marlin `barrier` / `membar` / `no_instruction` are higher** because its `cp.async` pipeline deliberately idles warps on memory barriers, but it hides this with independent warps from the same large block.

## 4. Instruction counts

| kernel | shape | issued instructions | executed instructions | avg issued per scheduler |
|---|---|---:|---:|---:|
| Amplin splitk16 | kimi dense-up | 19,669,204 | 19,643,904 | 39,655.65 |
| Marlin | kimi dense-up | 12,961,006 | 12,870,984 | 26,131.06 |
| Amplin tile2 | glm lm-head | 151,586,901 | 151,554,920 | 305,618.75 |
| Marlin | glm lm-head | 91,765,347 | 91,178,828 | 185,010.78 |
| Amplin tile2 | kimi lm-head | 186,937,858 | 186,905,600 | 376,890.84 |
| Marlin | kimi lm-head | 112,907,160 | 112,185,648 | 227,635.40 |

Observations:

- Marlin issues **~34% fewer instructions** for the same `dense-up` GEMM.
- For `lm-head`, Amplin `tile2` issues **1.65-2.0x more instructions** than Marlin despite the two kernels computing the same result. The extra instructions come from many small blocks (1210/1280 vs Marlin's 124) and the lane-dequant epilogue/serialisation overhead inside each block.

## 5. Memory request pattern deep-dive

Additional NCU metrics collected on the memory hierarchy (L1/TEX global load requests/sectors, L2 requests, DRAM sectors, `LDG` vs `LDGSTS` instruction mix, and shared-memory bank conflicts).

### 5.1 Global load instruction mix

| kernel | shape | `LDG` (global load) inst | `LDGSTS` (cp.async) inst | L1 global load requests | L1 load sectors | sectors/req | L1 write to L2 (MB) |
|---|---|---:|---:|---:|---:|---:|---:|
| Amplin splitk16 | kimi dense-up | 1,548,288 | 0 | 1,548,288 | 12,643,442 | 8.17 | 3.96 |
| Amplin tile2 | glm lm-head | 3,717,120 | 929,280 | 4,646,400 | 48,322,560 | 10.40 | 38.81 |
| Amplin tile2 | kimi lm-head | 4,587,520 | 1,146,880 | 5,734,400 | 59,637,760 | 10.40 | 41.71 |
| Marlin | kimi dense-up | 4,012 | 266,112 | 270,124 | 4,253,932 | 15.75 | 3.12 |
| Marlin | glm lm-head | 4,114 | 1,916,640 | 1,920,754 | 30,263,794 | 15.76 | 11.90 |
| Marlin | kimi lm-head | 4,114 | 2,365,440 | 2,369,554 | 37,335,794 | 15.76 | 12.48 |

Observations:

- **Marlin uses almost exclusively `LDGSTS` (cp.async) for global loads**: only ~4,000 regular `LDG` instructions vs 266k-2.4M `LDGSTS` instructions. Amplin `splitk16` uses **zero** `LDGSTS`; `tile2` uses some, but still 3.7M-4.6M `LDG` instructions.
- **Each Marlin load request is ~50% larger**: `sectors/req` is ~15.75 for Marlin (≈504 bytes/request, near-perfect 128-bit/thread `uint4`/int4 loads) vs 8.2-10.4 for Amplin (≈261-333 bytes/request). This means Amplin is using smaller or less coalesced load vectors.
- **Amplin `tile2` writes 3x more bytes to L2** than the actual output (38-42 MB vs ~10 MB output). Marlin's L2 write bytes are close to the output size (12-13 MB). Store coalescing/amplification is a real problem for `tile2`.

### 5.2 L1 / L2 / DRAM traffic

| kernel | shape | L1 hit sectors | L1 miss sectors | L2 requests | L2 hits | L2 misses | DRAM read sectors | DRAM write sectors |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Amplin splitk16 | kimi dense-up | 6,100,041 | 6,543,401 | 5,461,710 | 4,421,173 | 1,040,537 | 2,145,432 | 97,828 |
| Amplin tile2 | glm lm-head | 16,261,295 | 32,061,265 | 19,539,280 | 11,218,489 | 8,320,791 | 15,359,496 | 310,664 |
| Amplin tile2 | kimi lm-head | 20,068,960 | 39,568,800 | 23,851,284 | 13,664,930 | 10,186,354 | 18,968,000 | 346,984 |
| Marlin | kimi dense-up | 0 | 4,253,932 | 1,597,351 | 523,738 | 1,073,613 | 2,179,724 | 127,552 |
| Marlin | glm lm-head | 0 | 30,263,794 | 11,572,323 | 3,722,851 | 7,849,472 | 15,408,944 | 406,676 |
| Marlin | kimi lm-head | 0 | 37,335,794 | 14,260,289 | 4,594,625 | 9,665,664 | 19,001,464 | 428,840 |

Observations:

- **DRAM read sectors are essentially the same** between Amplin and Marlin for each shape, because both read the same weight/activation data. The difference is entirely in L1/L2 request amplification.
- **Marlin has 0 L1/TEX load hits** for the measured global loads (it bypasses L1 for weight streams, relying on `cp.async` directly to shared). This is actually efficient for weight-only inference: the same weights are read once and not cached.
- **Amplin `tile2` has 2.4-2.8x more L1 global load requests and 2.4-2.8x more L2 requests** than Marlin for the same shape. The extra L1/L2 traffic is wasted bandwidth.

### 5.3 Shared-memory bank conflicts

| kernel | shape | shared load bank conflicts | shared store bank conflicts | `ldsm` bank conflicts (SM) | `ldsm` bank conflicts (SMSP) |
|---|---|---:|---:|---:|---:|
| Amplin splitk16 | kimi dense-up | 0 | 0 | 0 | 0 |
| Amplin tile2 | glm lm-head | 0 | 0 | 0 | 0 |
| Amplin tile2 | kimi lm-head | 866 | 0 | 0 | 0 |
| Marlin | all | 0 | 0 | 0 | 0 |

Shared-memory bank conflicts are essentially nonexistent. The bottleneck is **not** bank conflicts; it is global load vector size and `cp.async` usage.

## 6. Strategy to close the Marlin gap

The deep-dive changes the priority. The two biggest levers are now:

1. **Replace regular `LDG` weight/activation loads with `LDGSTS` (`cp.async`) and `uint4` vector loads** so each global load request carries ~504 bytes like Marlin instead of ~260-330 bytes.
2. **Reduce the number of global load/store instructions** by increasing the per-block N tile and reusing the loaded A tile across more N columns, matching Marlin's one-large-block-per-SM design.

The original four levers, reordered by impact, are now:

### 6.1 Adopt `cp.async` (`LDGSTS`) with `uint4` vector loads for weights and activations

- Marlin's `LDGSTS` instruction count dominates (266k-2.4M) while regular `LDG` is only ~4k. Amplin `splitk16` has **zero** `LDGSTS`; `tile2` has 0.9M-1.1M `LDGSTS` but still 3.7M-4.6M `LDG`.
- Marlin's `sectors/req` is ~15.75 (≈504 bytes/request, 128-bit/thread). Amplin is 8.2-10.4 (≈260-330 bytes/request), meaning smaller load vectors or uncoalesced access.
- Replacing scalar/uint2 global loads with `LDGSTS.128`/`LDGSTS.64` for packed weights and `cp.async` for the A tile should cut L1/L2 request amplification by ~2x and raise memory throughput.

### 6.2 Move B dequantization into registers (LOP3) instead of shared-staged B

- `tile2` `math_pipe_throttle` is **42-49k inst/warp** vs Marlin **~350**. The lane dequantization pipeline is oversubscribing the math pipe and serializing warps.
- Re-implement the 4-bit unpack + scale multiply as LOP3/bitwise `half2` operations directly into tensor-core fragments, the way Marlin does. This removes the separate `MmaLaneDequant` shared-memory shuffle and should cut both `math_pipe_throttle` and `barrier` stalls.

### 6.3 Use a 4-stage `cp.async` pipeline for A and packed weights

- Marlin's `barrier` / `membar` stalls are high but hidden because each block has enough warps/independent work. Amplin `tile2` already uses shared-A but does **not** use `cp.async` staging with `cp_async_wait_group` across multiple buffers.
- Add a 2- or 4-stage `cp.async` pipeline in the `tiled_fullk` and `splitk` paths. This lets warps overlap global loads with `ldmatrix`/`mma` and reduces `long_scoreboard` wait time.

### 6.4 Use larger N tiles per block and fewer blocks per SM

- Marlin uses one 128-thread block per SM with a large N tile (N=128/256). Amplin `tile2` uses many tiny N64 blocks and 1.95-2.06 waves.
- For M=32, try an **N128 or N256 tile** with 8 warps (256 threads) while keeping register pressure under ~128 regs/thread so 2 blocks/SM is still possible. This increases per-block work, reduces block count, and gives the scheduler more independent warps.
- For `splitk16`, the 65 KB per-warp partials block is already limiting occupancy. Shrink partials to 32 KB or use a **warp-shuffle/tree reduction** to allow 2 CTAs per SM and hide latency.

### 6.5 Improve store coalescing and partial reduction layout

- `tile2` writes **38-42 MB to L2** when the actual FP16 output is only ~10 MB. The store epilogue is scattering partial outputs across many sectors.
- A larger N tile and a contiguous block-level output write (e.g. `uint4` stores, or a single reduction before writing) should bring L2 write bytes down to the ~12 MB Marlin achieves.

## 7. Recommended next implementation steps

1. **Prototype a new `mma_lane_m32_n128` kernel** built on the existing `tiled_fullk` skeleton but with:
   - 8 warps, N128/N256 tile, one block per SM like Marlin.
   - 4-stage `cp.async` (`LDGSTS`) for A and packed weights, using `uint4` vector loads to reach ~15+ sectors/request.
   - Register-level nibble unpack + `half2` scale multiply (LOP3) into `mma` fragments.
   - Coalesced `uint4` store epilogue to avoid L2 write amplification.
   - Keep group-size=128, desc_act=False, sym=True support and FP16/BF16.

2. **For `splitk16`, add a `cp.async`-based `uint4` load path** first:
   - Replace the scalar/uint2 global weight loads with `LDGSTS` + `uint4`.
   - Halve the per-warp FP32 partial buffer to 32 KB so 2 blocks fit per SM.
   - Benchmark against the existing `splitk16` and Marlin on `kimi-k2.5 dense-up`.

3. **Run the same NCU metrics** after each change and compare `sectors/req`, `l1tex__t_requests_pipe_lsu_mem_global_op_ld`, `l1tex__m_l1tex2xbar_write_bytes`, `memory (GB/s)`, `SM Busy %`, `not_selected`, `math_pipe_throttle`, and `long_scoreboard`. The target is to match or beat Marlin on:
   - `glm-5.2 lm-head` (M32 K6144 N154880)
   - `kimi-k2.5 dense-up` (M32 K7168 N18432)
   - `kimi-k2.5 lm-head` (M32 K7168 N163840)

## 8. Raw NCU reports

Saved under `ncu_reports/` in this branch (not committed, large binary):

- Memory-request pass: `memreq_splitk16_denseup.ncu-rep`, `memreq_tile2_glm.ncu-rep`, `memreq_tile2_kimi.ncu-rep`, `memreq_marlin_denseup.ncu-rep`, `memreq_marlin_glm.ncu-rep`, `memreq_marlin_kimi.ncu-rep`
- Instruction/bank-conflict pass: `bank_splitk16_denseup.ncu-rep`, `bank_tile2_glm.ncu-rep`, `bank_tile2_kimi.ncu-rep`, `bank_marlin_denseup.ncu-rep`, `bank_marlin_glm.ncu-rep`, `bank_marlin_kimi.ncu-rep`
- Core SOL/stall/occupancy: `ncu_splitk16_denseup.ncu-rep` / `_warp.ncu-rep`, `ncu_tile2_glm_lmhead.ncu-rep` / `_stalls.ncu-rep`, `ncu_tile2_kimi_lmhead.ncu-rep` / `_stalls.ncu-rep`, `ncu_marlin_kimi_denseup.ncu-rep` / `_stalls.ncu-rep` / `_inst.ncu-rep`, `ncu_marlin_glm_lmhead.ncu-rep` / `_stalls.ncu-rep`, `ncu_marlin_kimi_lmhead.ncu-rep` / `_stalls.ncu-rep`

These can be re-opened with `ncu -i <file>.ncu-rep` for deeper section-by-section inspection.

## 9. `run2` scale-pair dequant follow-up and Nsight Systems launch-gap check

After landing the `MmaLaneDequant::run2` change (scale broadcast hoisted, fused two-fragment dequant), the three M=32 loss shapes moved closer to Marlin but still did not pass it.

### Nsight Systems: no dispatch gaps
A focused `nsys profile` run of `mma_lane_m32_n64_tile2_shared_a` on `glm-5.2 lm-head M=32 K=6144 N=154880` (70 consecutive kernel launches, warmup+timed) showed essentially zero dead time between kernel instances:

```
instances 70
duration ns: min 508191  max 511807  avg 509607
gaps ns:     min 1056   max 94496   avg 2631
total gap %: 0.51%
```

The trace file is `/tmp/nsys_tile2_glm_lmhead.nsys-rep`. This confirms the bottleneck is **inside** the kernel, not launch overhead or stream scheduling.

### Updated NCU on `tile2` (`glm-5.2 lm-head`, post-`run2`)

| metric | value |
|---|---:|
| Duration | 505.95 us |
| Memory Throughput | 992.11 GB/s |
| Max Bandwidth | 40.55% |
| Compute (SM) Throughput | 51.35% |
| L1/TEX Hit Rate | 35.15% |
| L2 Hit Rate | 58.06% |
| Registers / thread | 86 |
| Shared Memory / block | 17.41 KB static + 0 dynamic |
| Waves Per SM | 1.95 |
| Theoretical Occupancy | 31.25% |
| Achieved Occupancy | 27.64% |
| `not_selected` (ratio) | 37,744.60 inst/warp |
| `math_pipe_throttle` | to be re-profiled with matching `--metrics` |

Key observations:
- Memory throughput is now ~992 GB/s, still below Marlin's ~1180 GB/s.
- The kernel is limited by occupancy (27.6% achieved) and `not_selected` stalls remain huge (37k inst/warp), indicating warps are ready but not being chosen due to instruction serialization or long-latency dependencies in the dequant/MMA loop.
- There is no launch-gap problem; the gap is the kernel's instruction mix and per-warp parallelism.

### Next route
The `run2` optimization was a 5-10% improvement. To close the remaining 14-22% gap, the next step is to increase the per-warp independent work and reduce `not_selected`:
1. **Larger N tile with shared A**: one A tile reused across 4/8 warps, so each warp does independent dequant on its own N slice.
2. **`cp.async` weight + scale staging**: convert the `LDG`/`LDGSTS` imbalance and allow the MMA math to overlap more memory transfers.
3. **Reduce `not_selected` stalls**: investigate instruction-level dependencies in `run2` / `MmaInstruction::run` and consider interleaving dequant for the next `k_step` with MMA of the current one.

## 2026-07-25: M=32 follow-up experiments (regressed, reverted)

### NCU re-profile of `mma_lane_m32_n64_tile2_shared_a` on `kimi-k2.5 lm-head`

Command:
```bash
ncu --section SpeedOfLight --section Occupancy --section LaunchStats \
    --section InstructionStats --section MemoryWorkloadAnalysis \
    --kernel-name "regex:.*tiled_fullk_kernel.*" \
    python /tmp/ncu_m32_tile2.py
```

Key raw metrics (M=32 K=7168 N=163840, FP16, GPU 0):

| metric | value |
|---|---:|
| Duration | 659.26 us |
| Memory Throughput | 938.24 GB/s |
| Max Bandwidth | 38.35% |
| Compute (SM) Throughput | 48.72% |
| L1/TEX Hit Rate | 34.94% |
| L2 Hit Rate | 57.92% |
| Registers / thread | 86 |
| Static Shared Memory / block | 17.41 KB |
| Waves Per SM | 2.06 |
| Theoretical Occupancy | 31.25% |
| Achieved Occupancy | 27.60% |
| Block Limit Registers | 5 |
| Block Limit Shared Mem | 5 |

Interpretation: the kernel is latency/memory-latency bound, not raw throughput bound. Occupancy is capped by both registers and shared memory. The partial-wave tail (40 blocks out of 1280) accounts for up to one third of the wave runtime if all blocks were uniform.

### Experiment A: `tile4_msplit` (BlockM=16, NTiles=4, `__launch_bounds__(128,8)`, M split across grid.y)

Added `mma_lane_m32_n64_tile4_msplit_shared_a` to reuse a smaller 16-row A tile across 4 independent N64 slices, expecting higher occupancy (smaller shared footprint) and more independent warps. Correctness passed, but it was slower than `tile2` on every M=32 shape and much slower on the loss shapes:

| model | role | M | K | N | tile2 | tile4_msplit | marlin |
|---|---|---:|---:|---:|---:|---:|---:|
| glm-5.2 | lm-head | 32 | 6144 | 154880 | 530.4 us | 827.6 us | 463.7 us |
| glm-5.2 | dense-up | 32 | 6144 | 12288 | 144.3 us | 230.0 us | 93.5 us |
| glm-5.2 | o-proj | 32 | 16384 | 6144 | 299.4 us | 529.3 us | 103.8 us |
| kimi-k2.5 | dense-up | 32 | 7168 | 18432 | 181.4 us | 261.5 us | 118.3 us |
| kimi-k2.5 | dense-down | 32 | 18432 | 7168 | 331.0 us | 580.2 us | 117.9 us |
| kimi-k2.5 | lm-head | 32 | 7168 | 163840 | 678.3 us | 980.5 us | 559.4 us |

The `__launch_bounds__(128,8)` forced the compiler below its natural register budget and produced spills/local-memory traffic, overwhelming the smaller A-tile benefit. Reverted.

### Experiment B: remove shared-A padding (`kSharedAK = kHmmaBlockK`)

Changed `tiled_fullk_kernel` to use a tight shared-A row stride (no +8 half padding), aiming to drop shared memory from 17.41 KB/block to 16 KB/block and raise the shared-mem block limit from 5 to 6 blocks/SM.

Result: `tile2` regressed on large-N M=32 shapes (e.g. glm-5.2 lm-head 530 us -> 571 us, kimi-k2.5 lm-head 678 us -> 684 us) and improved only marginally or not at all elsewhere. The bank-conflict cost of the un-padded layout outweighed the one-block occupancy gain. Reverted to `kSharedAK = kHmmaBlockK + 8`.

### Current M=32 status

The best existing Amplin kernels remain:
- `glm-5.2 lm-head`: `tile2` ~1.14x Marlin
- `kimi-k2.5 dense-up`: `splitk16` ~1.16x Marlin
- `kimi-k2.5 lm-head`: `tile2` ~1.21x Marlin

All three ordered micro-routes (interleaved-dequant, `cp.async` weight staging, larger N-tile shared-A) plus the occupancy-focused `tile4_msplit` and no-padding variants have failed to close the gap. The remaining gap appears to need either:
- a swizzled shared-A layout that removes padding *and* bank conflicts, or
- a Marlin-style 8-warp N256/512 `cp.async` pipeline with register-fragment LOP3 dequantization.
