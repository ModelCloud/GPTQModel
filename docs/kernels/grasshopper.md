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

## Rejected Probes

| Date | Probe | Result |
| --- | --- | --- |
| 2026-05-21 | Skip output memset and direct-store when `gridDim.x == 1` | Rejected. 3-bit GEMV 256x8192 group128 median regressed `0.0684 ms -> 0.0733 ms`. |
| 2026-05-21 | Use a 256-half2 K tile for very wide 3-bit GEMV | Rejected. 3-bit GEMV 8192x8192 group128 median regressed `0.1047 ms -> 0.1263 ms`; halving split-K CTAs lost more parallelism than it saved in atomics. |
| 2026-05-21 | Use a 2-row batch tile for batch2 wide 4-bit GEMM | Rejected. Batch2 8192x8192 group128 median regressed `0.1133 ms -> 0.1175 ms`. |
| 2026-05-21 | Reuse dense LoRA `up` loads across batch rows | Rejected. Dense LoRA GEMV 8192x1024 rank64 was neutral/slower `0.0738 ms -> 0.0743 ms`; dense LoRA batch8 GEMM regressed `0.0923 ms -> 0.1013 ms`. |
| 2026-05-21 | Replace common int8 LoRA group divisions with runtime 32/64/128 shift branches | Rejected. Int8 LoRA GEMV 8192x1024 rank64 group128 median regressed `0.0740 ms -> 0.0868 ms`; the extra runtime branch outweighed the one-time division. |

## Promotion Rule

Promote a GrassHopper tuning change only when it has a positive measurement on
the affected shape and does not regress the nearby shape family. Prefer Nsight
Compute kernel time, instruction count, occupancy/register metrics, and a
repeat host median from `scripts/profile_grasshopper_nsight.py`.
