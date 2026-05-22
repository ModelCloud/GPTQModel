# GrassHopper Kernel Findings

This file tracks GrassHopper/VecQuant3 CUDA kernel tuning results on the local
`NVIDIA PG506-230/232` `sm_80` datacenter GPUs. Profiling commands should set
`CUDA_DEVICE_ORDER=PCI_BUS_ID` and use `scripts/profile_grasshopper_nsight.py`
plus Nsight Compute for promoted changes.

## Promoted Changes

| Date | Commit | Area | Finding |
| --- | --- | --- | --- |
| 2026-05-21 | `0d5c3716` | Narrow GEMV occupancy | Narrow single-row 4/8-bit GEMV uses a 32-half2 K tile for `out_features < 2048`, improving occupancy for underfilled decode shapes. |
| 2026-05-21 | `79c13213` | Narrow GEMM batch tile | Narrow-output GEMM uses the existing 4-row batch tile for `batch >= 4`, avoiding separate row-wise GEMV launches on small-output batched decode. |
| 2026-05-21 | `d010721f` | Int8 LoRA scale indexing | For divisible LoRA-B layouts, scale indexing advances by row stride instead of dividing inside the rank loop. Nsight `gemv_lora_int8` 8192x1024 rank64 group128 improved `31.07 us -> 21.89 us`; host median improved `0.0915 ms -> 0.0889 ms`. Batch8 GEMM median improved `0.1363 ms -> 0.1173 ms`. |
| 2026-05-21 | `80d6bfdb` | Multi-group 4/8-bit decode | Multi-group tiles compute the quant group once per qweight row and reuse the fixed-group decoder. Nsight batch8 8192x1024 group128 improved `55.55 us -> 49.25 us`, instructions dropped `15.58M -> 11.91M`, and registers/thread dropped `32 -> 31`. Host medians improved: GEMV 8192x8192 group64 `0.1337 ms -> 0.1228 ms`; batch8 8192x1024 group128 `0.0974 ms -> 0.0887 ms`; batch8 8192x8192 group64 `0.2402 ms -> 0.2144 ms`. |
| 2026-05-22 | `683d9125` | Group32 batched multi-group decode | Batch GEMM group32 tiles iterate qweight rows by quant-group segment, avoiding per-row group recomputation only where each tile crosses many groups. Nsight batch8 4-bit 8192x8192 group32 improved `168.352 us -> 164.736 us`, instructions dropped `83.61M -> 80.30M`, registers/thread rose `31 -> 32`, and host median improved `0.2277 ms -> 0.2099 ms`. Nearby group64 remained neutral (`0.2141 ms -> 0.2137 ms`). |
| 2026-05-22 | `8b572f51` | Group64 8-bit batched multi-group decode | Batch GEMM now uses quant-group segment iteration for the 8-bit group64 dynamic-group path. Nsight batch8 8192x8192 improved `189.632 us -> 170.848 us`, instructions dropped `93.03M -> 80.94M`, registers/thread dropped `30 -> 28`, and host median improved `0.2493 ms -> 0.2187 ms` with final-source rebuild at `0.2320 ms`. Adjacent 4-bit group32 stayed `0.2099 ms`; 4-bit group64 remained comparable at `0.2150 ms`. |
| 2026-05-22 | `e8afa845` | Group64 8-bit fixed-group K tile | Batch GEMM 8-bit group64 now uses a 32-half2 K tile so each CTA covers exactly one quant group and takes the fixed-group decode path. Nsight improved the previous segment path `170.848 us -> 163.264 us`, host median improved `0.2320 ms -> 0.2127 ms`, and issue/throughput rose `78.96% -> 82.96%`. |
| 2026-05-22 | `0a9d71eb` | Group64 4-bit fixed-group K tile | Batch GEMM 4-bit group64 also uses the 32-half2 K tile so each CTA covers exactly one quant group and takes the fixed-group decode path. Nsight improved `168.224 us -> 155.936 us`, instructions dropped `83.61M -> 79.59M`, issue/throughput rose `82.91% -> 85.24%`, active warps rose `81.87% -> 89.51%`, and host median improved `0.2145 ms -> 0.2004 ms`. Adjacent checks stayed in range: 8-bit group64 median `0.2074 ms`; 4-bit group32 median `0.2111 ms`. |
| 2026-05-22 | `f5bb54ef` | Group128 3-bit wide batch tile | Batch GEMM 3-bit group128 uses the 8-row wide batch tile for batch8 wide-feature shapes, halving grid z-slices and split-K atomics from `2048 -> 1024` CTAs. Nsight improved `180.480 us -> 179.136 us`, instructions dropped `88.16M -> 78.05M`, registers/thread rose `40 -> 47`, and host median improved `0.2355 ms -> 0.2200 ms` with final-source rebuild at `0.2306 ms`. Adjacent 3-bit group64 stayed on the 4-row layout with median `0.2278 ms`. |

## Rejected Probes

| Date | Probe | Result |
| --- | --- | --- |
| 2026-05-21 | Skip output memset and direct-store when `gridDim.x == 1` | Rejected. 3-bit GEMV 256x8192 group128 median regressed `0.0684 ms -> 0.0733 ms`. |
| 2026-05-21 | Use a 256-half2 K tile for very wide 3-bit GEMV | Rejected. 3-bit GEMV 8192x8192 group128 median regressed `0.1047 ms -> 0.1263 ms`; halving split-K CTAs lost more parallelism than it saved in atomics. |
| 2026-05-21 | Use a 2-row batch tile for batch2 wide 4-bit GEMM | Rejected. Batch2 8192x8192 group128 median regressed `0.1133 ms -> 0.1175 ms`. |
| 2026-05-21 | Reuse dense LoRA `up` loads across batch rows | Rejected. Dense LoRA GEMV 8192x1024 rank64 was neutral/slower `0.0738 ms -> 0.0743 ms`; dense LoRA batch8 GEMM regressed `0.0923 ms -> 0.1013 ms`. |
| 2026-05-21 | Replace common int8 LoRA group divisions with runtime 32/64/128 shift branches | Rejected. Int8 LoRA GEMV 8192x1024 rank64 group128 median regressed `0.0740 ms -> 0.0868 ms`; the extra runtime branch outweighed the one-time division. |
| 2026-05-22 | Use a 16-row batch tile for wide-output batch16 4-bit GEMM | Rejected. Batch16 8192x8192 group128 median regressed `0.3468 ms -> 0.3524 ms`; halving the batch grid did not offset lower per-CTA efficiency. |
| 2026-05-22 | Use an 8-row batch tile for narrow-output batch16 4-bit GEMM | Rejected. Batch16 8192x1024 group128 median regressed `0.1129 ms -> 0.1269 ms`; the existing 4-row tile keeps better kernel efficiency for narrow output. |
| 2026-05-22 | Apply quant-group segment iteration to all 4/8-bit dynamic group sizes | Rejected. The broad change helped group32 but regressed batch8 4-bit 8192x8192 group64 median `0.2141 ms -> 0.2200 ms`; refined the promoted path to `GroupSize == 32` batched GEMM only. |
| 2026-05-22 | Apply group32 quant-group segment iteration to single-row GEMV | Rejected. 4-bit GEMV 8192x8192 group32 median regressed `0.1119 ms -> 0.1238 ms`; the extra loop structure hurt single-row decode despite reducing group checks. |
| 2026-05-22 | Use a 32-half2 K tile for batch8 8-bit group32 GEMM | Rejected. Batch8 8192x8192 group32 median regressed `0.2200 ms -> 0.2273 ms`; doubling split-K CTAs and atomics outweighed less per-CTA group work. |

## Promotion Rule

Promote a GrassHopper tuning change only when it has a positive measurement on
the affected shape and does not regress the nearby shape family. Prefer Nsight
Compute kernel time, instruction count, occupancy/register metrics, and a
repeat host median from `scripts/profile_grasshopper_nsight.py`.
