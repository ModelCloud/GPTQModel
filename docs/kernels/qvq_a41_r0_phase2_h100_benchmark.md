# QVQ A41/R0 Phase 2: H100 benchmark against pre-PR main

This benchmark compares the Phase-2 grouped P32 kernel at `a1cd1b02` with the
independent child kernels from pre-PR `origin/main` at `c0469004`.  It measures
only the exact P32 inner projections after the one shared input transform.

## Result

For the two grouped projection sites in one Llama 3.2 1B decoder layer, the
combined geometric-mean speedup is **1.233x** across W2/W2.5/W3/W3.5 and
M1/M2/M4/M8/M16.  Every combined QKV-plus-gate/up case is faster than pre-PR
main.  QKV benefits most: its geometric-mean speedup is **1.610x**, with all 20
QKV cases improving.  Gate/up improves by **1.071x** geometrically, with 14 of
20 individual cases improving.

| Rate | Group | Geomean vs pre-PR main | Better cases |
|---:|---|---:|---:|
| W2 | QKV | 1.553x | 5/5 |
| W2 | gate/up | 1.073x | 4/5 |
| W2.5 | QKV | 1.682x | 5/5 |
| W2.5 | gate/up | 1.077x | 3/5 |
| W3 | QKV | 1.568x | 5/5 |
| W3 | gate/up | 1.065x | 4/5 |
| W3.5 | QKV | 1.643x | 5/5 |
| W3.5 | gate/up | 1.071x | 3/5 |

The model-oriented sum of QKV and gate/up time is:

| Rate | M | Pre-PR main (us) | Phase 2 (us) | Speedup | Better than pre-PR main |
|---:|---:|---:|---:|---:|:---:|
| W2 | 1 | 52.950 | 42.764 | 1.238x | Yes |
| W2 | 2 | 50.692 | 45.484 | 1.115x | Yes |
| W2 | 4 | 58.191 | 53.323 | 1.091x | Yes |
| W2 | 8 | 100.984 | 73.770 | 1.369x | Yes |
| W2 | 16 | 107.111 | 81.946 | 1.307x | Yes |
| W2.5 | 1 | 64.060 | 46.919 | 1.365x | Yes |
| W2.5 | 2 | 60.336 | 53.844 | 1.121x | Yes |
| W2.5 | 4 | 66.711 | 59.607 | 1.119x | Yes |
| W2.5 | 8 | 101.070 | 72.902 | 1.386x | Yes |
| W2.5 | 16 | 107.006 | 81.359 | 1.315x | Yes |
| W3 | 1 | 57.792 | 47.143 | 1.226x | Yes |
| W3 | 2 | 57.714 | 52.009 | 1.110x | Yes |
| W3 | 4 | 69.803 | 63.958 | 1.091x | Yes |
| W3 | 8 | 101.944 | 74.570 | 1.367x | Yes |
| W3 | 16 | 106.557 | 81.534 | 1.307x | Yes |
| W3.5 | 1 | 64.343 | 46.820 | 1.374x | Yes |
| W3.5 | 2 | 60.963 | 54.634 | 1.116x | Yes |
| W3.5 | 4 | 67.240 | 63.111 | 1.065x | Yes |
| W3.5 | 8 | 101.413 | 73.176 | 1.386x | Yes |
| W3.5 | 16 | 105.882 | 81.352 | 1.302x | Yes |

## Detailed matrix

`Child N` lists the independent projection widths.  Effective TFLOP/s is the
dense-equivalent work `2 * M * K * sum(N_i) / latency`; it does not count P32
decoder integer instructions as floating-point operations.

| Rate | Group | M | K | Child N | Main us | Phase 2 us | Speedup | Better | Main eff. TFLOP/s | Phase 2 eff. TFLOP/s |
|---:|---|---:|---:|---|---:|---:|---:|:---:|---:|---:|
| W2 | QKV | 1 | 2048 | 2048/512/512 | 22.846 | 14.956 | 1.528x | Yes | 0.551 | 0.841 |
| W2 | QKV | 2 | 2048 | 2048/512/512 | 20.554 | 15.479 | 1.328x | Yes | 1.224 | 1.626 |
| W2 | QKV | 4 | 2048 | 2048/512/512 | 22.119 | 17.003 | 1.301x | Yes | 2.275 | 2.960 |
| W2 | QKV | 8 | 2048 | 2048/512/512 | 38.044 | 20.490 | 1.857x | Yes | 2.646 | 4.913 |
| W2 | QKV | 16 | 2048 | 2048/512/512 | 40.120 | 21.747 | 1.845x | Yes | 5.018 | 9.258 |
| W2 | gate/up | 1 | 2048 | 8192/8192 | 30.103 | 27.808 | 1.083x | Yes | 2.229 | 2.413 |
| W2 | gate/up | 2 | 2048 | 8192/8192 | 30.138 | 30.005 | 1.004x | Yes | 4.453 | 4.473 |
| W2 | gate/up | 4 | 2048 | 8192/8192 | 36.072 | 36.320 | 0.993x | No | 7.442 | 7.391 |
| W2 | gate/up | 8 | 2048 | 8192/8192 | 62.940 | 53.279 | 1.181x | Yes | 8.530 | 10.077 |
| W2 | gate/up | 16 | 2048 | 8192/8192 | 66.991 | 60.199 | 1.113x | Yes | 16.028 | 17.836 |
| W2.5 | QKV | 1 | 2048 | 2048/512/512 | 28.853 | 15.984 | 1.805x | Yes | 0.436 | 0.787 |
| W2.5 | QKV | 2 | 2048 | 2048/512/512 | 25.357 | 18.119 | 1.399x | Yes | 0.992 | 1.389 |
| W2.5 | QKV | 4 | 2048 | 2048/512/512 | 25.758 | 18.460 | 1.395x | Yes | 1.954 | 2.727 |
| W2.5 | QKV | 8 | 2048 | 2048/512/512 | 37.731 | 19.014 | 1.984x | Yes | 2.668 | 5.294 |
| W2.5 | QKV | 16 | 2048 | 2048/512/512 | 39.292 | 20.437 | 1.923x | Yes | 5.124 | 9.851 |
| W2.5 | gate/up | 1 | 2048 | 8192/8192 | 35.207 | 30.935 | 1.138x | Yes | 1.906 | 2.169 |
| W2.5 | gate/up | 2 | 2048 | 8192/8192 | 34.979 | 35.725 | 0.979x | No | 3.837 | 3.757 |
| W2.5 | gate/up | 4 | 2048 | 8192/8192 | 40.954 | 41.147 | 0.995x | No | 6.555 | 6.524 |
| W2.5 | gate/up | 8 | 2048 | 8192/8192 | 63.338 | 53.887 | 1.175x | Yes | 8.476 | 9.963 |
| W2.5 | gate/up | 16 | 2048 | 8192/8192 | 67.714 | 60.922 | 1.111x | Yes | 15.857 | 17.625 |
| W3 | QKV | 1 | 2048 | 2048/512/512 | 24.782 | 16.000 | 1.549x | Yes | 0.508 | 0.786 |
| W3 | QKV | 2 | 2048 | 2048/512/512 | 23.075 | 17.519 | 1.317x | Yes | 1.091 | 1.436 |
| W3 | QKV | 4 | 2048 | 2048/512/512 | 27.154 | 20.323 | 1.336x | Yes | 1.854 | 2.477 |
| W3 | QKV | 8 | 2048 | 2048/512/512 | 37.831 | 20.663 | 1.831x | Yes | 2.661 | 4.872 |
| W3 | QKV | 16 | 2048 | 2048/512/512 | 39.221 | 20.676 | 1.897x | Yes | 5.133 | 9.737 |
| W3 | gate/up | 1 | 2048 | 8192/8192 | 33.010 | 31.143 | 1.060x | Yes | 2.033 | 2.155 |
| W3 | gate/up | 2 | 2048 | 8192/8192 | 34.639 | 34.490 | 1.004x | Yes | 3.875 | 3.892 |
| W3 | gate/up | 4 | 2048 | 8192/8192 | 42.650 | 43.634 | 0.977x | No | 6.294 | 6.152 |
| W3 | gate/up | 8 | 2048 | 8192/8192 | 64.113 | 53.906 | 1.189x | Yes | 8.374 | 9.959 |
| W3 | gate/up | 16 | 2048 | 8192/8192 | 67.336 | 60.858 | 1.106x | Yes | 15.946 | 17.644 |
| W3.5 | QKV | 1 | 2048 | 2048/512/512 | 28.913 | 15.983 | 1.809x | Yes | 0.435 | 0.787 |
| W3.5 | QKV | 2 | 2048 | 2048/512/512 | 25.519 | 18.186 | 1.403x | Yes | 0.986 | 1.384 |
| W3.5 | QKV | 4 | 2048 | 2048/512/512 | 25.970 | 20.742 | 1.252x | Yes | 1.938 | 2.427 |
| W3.5 | QKV | 8 | 2048 | 2048/512/512 | 37.779 | 19.118 | 1.976x | Yes | 2.665 | 5.265 |
| W3.5 | QKV | 16 | 2048 | 2048/512/512 | 38.992 | 20.448 | 1.907x | Yes | 5.163 | 9.846 |
| W3.5 | gate/up | 1 | 2048 | 8192/8192 | 35.430 | 30.837 | 1.149x | Yes | 1.894 | 2.176 |
| W3.5 | gate/up | 2 | 2048 | 8192/8192 | 35.444 | 36.448 | 0.972x | No | 3.787 | 3.682 |
| W3.5 | gate/up | 4 | 2048 | 8192/8192 | 41.270 | 42.369 | 0.974x | No | 6.504 | 6.336 |
| W3.5 | gate/up | 8 | 2048 | 8192/8192 | 63.634 | 54.058 | 1.177x | Yes | 8.437 | 9.931 |
| W3.5 | gate/up | 16 | 2048 | 8192/8192 | 66.890 | 60.904 | 1.098x | Yes | 16.052 | 17.630 |

## Method and controls

- Device: the exclusive physical H100 at
  `GPU-f5ea03cf-efa4-9807-7de5-b174957a1348`, SM 9.0, 132 SMs,
  102,077,169,664 bytes of memory.  The H200 was not visible to either run.
- Workload: real Llama 3.2 1B QKV (`K=2048`, `N=2048/512/512`) and gate/up
  (`K=2048`, `N=8192/8192`) shapes at M1/M2/M4/M8/M16.
- Timing: one CUDA Graph contains the complete grouped operation or all
  independent child operations.  CUDA events bracket 20 graph replays per
  sample; the table reports the median of 50 samples after 20 warmups.
- Split plans are explicit and identical on both revisions: split 32 for
  M1/M2, split 40 for M4, then split 8 for QKV and split 3 for gate/up at
  M8/M16.  No autotuning is included in timing.
- Inputs, P32 window payloads, selector bytes, alternative-bank IDs, Torch,
  CUDA, and compilation flags are identical.  Grouped output is bit-exact to
  the independent feature-branch path.  Cross-process output checksums are
  also identical to pre-PR main for all 40 cases.
- The feature branch's independent-child diagnostic is 1.002x versus pre-PR
  main geometrically.  This near-unity control shows that the grouped results
  are not explained by a faster rewritten child kernel.
- A denser 100-sample, 30-replay repeat confirms that gate/up M2/M4 is near
  break-even and retains small regressions for some rates.  These cases need a
  grouped-specific split/scheduling pass; the current implementation
  deliberately preserves each child's pre-existing split policy.

The detached main checkout required a local host-only capability-gate change
so its embedded compute-80 PTX could JIT on the H100.  No baseline device
function, kernel body, launch geometry, split plan, or reduction math changed.
This makes the relative same-H100 result useful, while the absolute numbers
remain compatibility-path measurements rather than native Hopper or A100
performance.

Marlin and Machete are not included in this table.  This experiment isolates
the Phase-2 change against pre-PR QVQ main and times a grouped multi-output
operation.  A Marlin/Machete comparison requires summing the same independent
QKV and gate/up children under an explicitly matched W4 protocol; it should be
reported as a separate baseline rather than mixed into this change-isolation
matrix.

## Reproduction artifacts

- Benchmark driver: `scripts/benchmark_qvq_p32_ampere_grouped.py`
- Pre-PR main: `artifacts/a41_phase2_h100/pre_pr_main_plain.json`
- Phase-2 grouped: `artifacts/a41_phase2_h100/phase2_grouped.json`
- Feature-branch plain control:
  `artifacts/a41_phase2_h100/phase2_plain_diagnostic.json`
- Targeted gate/up repeats:
  `artifacts/a41_phase2_h100/pre_pr_main_gate_repeat.json` and
  `artifacts/a41_phase2_h100/phase2_grouped_gate_repeat.json`
