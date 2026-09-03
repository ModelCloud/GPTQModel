# Phase 62: rejected H100 paired-recovery high-stage geometry

Phase 62 swept the thread-block width of the exact N=8192 paired-recovery
high stage.  The accepted Phase-56 kernel uses four 64-thread blocks per
projection row.  Candidates retained the same 256 threads, column mapping,
FP32 butterfly sequence, overflow-preserving rounding, scale, bias, FP16
store boundary, workspace, and two-kernel graph topology, but scheduled them
as either eight 32-thread, two 128-thread, or one 256-thread block per
projection row.

None survived the complete-MLP promotion gate.  All candidate CUDA, Python,
runtime, telemetry, and test code was removed; production remains byte
identical to Phase 56.

## Isolated paired recovery

The physical 132-SM NVIDIA H100 passed the zero-utilization/zero-memory idle
gate.  Timing used warmed CUDA Graph replay measured by CUDA events.  The
baseline and candidate were compiled into the same extension and timed in the
same process.  Ratios above one favor the candidate.

| Threads/block | M | M/K/N per child | 64-thread control | Candidate | vs control | Better |
|--:|--:|:--|--:|--:|--:|:--:|
| 32 | 1 | 1/2048/8192 x2 | 6.090 us | 5.960 us | 1.0218x | Yes |
| 32 | 2 | 2/2048/8192 x2 | 6.140 us | 6.126 us | 1.0022x | Yes |
| 32 | 4 | 4/2048/8192 x2 | 6.131 us | 6.337 us | 0.9676x | No |
| 32 | 8 | 8/2048/8192 x2 | 6.635 us | 6.566 us | 1.0104x | Yes |
| 32 | 16 | 16/2048/8192 x2 | 7.173 us | 7.359 us | 0.9747x | No |
| 128 | 1 | 1/2048/8192 x2 | 5.998 us | 5.918 us | 1.0135x | Yes |
| 128 | 2 | 2/2048/8192 x2 | 6.146 us | 6.148 us | 0.9996x | No |
| 128 | 4 | 4/2048/8192 x2 | 6.175 us | 6.190 us | 0.9975x | No |
| 128 | 8 | 8/2048/8192 x2 | 6.635 us | 6.417 us | 1.0339x | Yes |
| 128 | 16 | 16/2048/8192 x2 | 7.227 us | 7.225 us | 1.0003x | Yes |
| 256 | 8 | 8/2048/8192 x2 | 6.563 us | 6.940 us | 0.9457x | No |

The 32-thread form is not robust across M.  The 256-thread form is decisively
slower.  The 128-thread form produced an apparently useful M8 component win,
so it advanced to an alternating-control complete-MLP test.

## Complete MLP rejection

The complete test includes grouped gate/up P32, paired recovery, exact SiLU,
down preconditioning, split-16 down P32, ordered reduction, and down recovery.
Each candidate was timed in an ABBA sequence with production controls before
and after it.  `Better` requires both candidate medians to beat both controls.
Marlin and Machete are figurative W4 latency baselines.

The 32-thread M1 candidate did not win any rate:

| W | M/K/N: gate/up x2; down | Candidate | vs Marlin W4 | vs Machete W4 | vs control | Better |
|--:|:--|--:|--:|--:|--:|:--:|
| 2 | 1/2048/8192 x2; 1/8192/2048 | 45.913 us | 0.630x | 1.090x | 1.0002x | No |
| 2.5 | 1/2048/8192 x2; 1/8192/2048 | 47.335 us | 0.611x | 1.057x | 0.9992x | No |
| 3 | 1/2048/8192 x2; 1/8192/2048 | 46.590 us | 0.621x | 1.074x | 0.9995x | No |
| 3.5 | 1/2048/8192 x2; 1/8192/2048 | 47.532 us | 0.609x | 1.052x | 0.9989x | No |

The first 300-sample 128-thread M8 run showed one provisional W2 win, but the
mandatory clean-rebuild confirmation used 500 samples and reversed it:

| W | M/K/N: gate/up x2; down | 64-thread control | 128-thread candidate | vs Marlin W4 | vs Machete W4 | vs control | Better |
|--:|:--|--:|--:|--:|--:|--:|:--:|
| 2 | 8/2048/8192 x2; 8/8192/2048 | 47.260 us | 47.304 us | 0.619x | 1.062x | 0.9991x | No |

The provisional measurement was correctly treated as an experiment rather
than a promotion.  The confirmation demonstrates why a sub-percent isolated
movement cannot be accepted without matched complete-operation repetition.

## Correctness and decision

- 133 focused H100 tests passed for the 128-thread candidate, including 120
  randomized exactness cases, both normalization modes, both low-stage
  schedules, overflow-preserving FP32 intermediates, CUDA Graph replay,
  argument guards, and a real-shape W2/M8 runtime/telemetry test.
- Every candidate was output-bit exact; the rejection is solely performance.
- No checkpoint bytes, quantization math, persistent VRAM, operation-local
  workspace size, launch count, or non-H100 dispatch changed.
- Compilation used no more than four Ninja jobs, one NVCC host thread, and one
  CUDA split-compile partition.

Phase 62 closes simple high-stage CTA-width redistribution.  Future work must
delete instructions, traffic, or a materialization boundary rather than
rescheduling the same sixteen warps.
