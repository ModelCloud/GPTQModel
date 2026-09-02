# QVQ A41/R0 Phase 3: native Hopper H100 benchmark

This benchmark compares the Phase-3 grouped TMA/register-sourced WGMMA kernel
at `a4dd3190` with independent child launches from `37edcaad`.  The two Hopper
source files at the baseline commit are byte-identical to pre-PR
`origin/main@c0469004`, so this is also the requested comparison with pre-PR
main.

## Result

For the grouped QKV and gate/up projection sites of one Llama 3.2 1B layer,
Phase 3 is **1.965x faster geometrically** across W2/W2.5/W3/W3.5 and
M1/M2/M4/M8/M16.  This is approximately 49.1% lower kernel latency.  All 40
individual group cases and all 20 combined cases improve.  The slowest
combined case is still 1.809x faster.

| Rate | Group | Geomean vs pre-PR main | Better cases |
|---:|---|---:|---:|
| W2 | QKV | 2.485x | 5/5 |
| W2 | gate/up | 1.387x | 5/5 |
| W2.5 | QKV | 2.340x | 5/5 |
| W2.5 | gate/up | 1.368x | 5/5 |
| W3 | QKV | 2.819x | 5/5 |
| W3 | gate/up | 1.468x | 5/5 |
| W3.5 | QKV | 2.918x | 5/5 |
| W3.5 | gate/up | 1.519x | 5/5 |

The model-oriented sum of QKV and gate/up kernel time is:

| Rate | M | Pre-PR main (us) | Phase 3 (us) | Speedup | Better than pre-PR main |
|---:|---:|---:|---:|---:|:---:|
| W2 | 1 | 114.573 | 60.637 | 1.889x | Yes |
| W2 | 2 | 113.334 | 60.118 | 1.885x | Yes |
| W2 | 4 | 113.581 | 60.286 | 1.884x | Yes |
| W2 | 8 | 113.293 | 60.196 | 1.882x | Yes |
| W2 | 16 | 112.589 | 60.324 | 1.866x | Yes |
| W2.5 | 1 | 113.493 | 62.468 | 1.817x | Yes |
| W2.5 | 2 | 112.403 | 62.125 | 1.809x | Yes |
| W2.5 | 4 | 112.874 | 62.098 | 1.818x | Yes |
| W2.5 | 8 | 112.917 | 61.823 | 1.826x | Yes |
| W2.5 | 16 | 112.462 | 62.040 | 1.813x | Yes |
| W3 | 1 | 112.294 | 54.746 | 2.051x | Yes |
| W3 | 2 | 112.294 | 54.540 | 2.059x | Yes |
| W3 | 4 | 112.254 | 54.514 | 2.059x | Yes |
| W3 | 8 | 111.114 | 54.596 | 2.035x | Yes |
| W3 | 16 | 112.322 | 54.602 | 2.057x | Yes |
| W3.5 | 1 | 115.574 | 54.626 | 2.116x | Yes |
| W3.5 | 2 | 115.798 | 54.254 | 2.134x | Yes |
| W3.5 | 4 | 114.826 | 54.336 | 2.113x | Yes |
| W3.5 | 8 | 115.636 | 54.312 | 2.129x | Yes |
| W3.5 | 16 | 115.657 | 54.322 | 2.129x | Yes |

## Detailed matrix

`Child N` lists the separately observable child projection widths.  Effective
TFLOP/s is dense-equivalent logical work,
`2 * logical_M * K * sum(N_i) / latency`.  It does not count P32 decoder
integer instructions as floating-point operations.  The current native Hopper
kernel pads every target M to kernel M16, so effective TFLOP/s deliberately
uses the real logical row count.

| Rate | Group | M | K | Child N | Main us | Phase 3 us | Speedup | Better | Main eff. TFLOP/s | Phase 3 eff. TFLOP/s |
|---:|---|---:|---:|---|---:|---:|---:|:---:|---:|---:|
| W2 | QKV | 1 | 2048 | 2048/512/512 | 68.203 | 27.448 | 2.485x | Yes | 0.184 | 0.458 |
| W2 | QKV | 2 | 2048 | 2048/512/512 | 67.434 | 26.966 | 2.501x | Yes | 0.373 | 0.933 |
| W2 | QKV | 4 | 2048 | 2048/512/512 | 67.571 | 27.045 | 2.498x | Yes | 0.745 | 1.861 |
| W2 | QKV | 8 | 2048 | 2048/512/512 | 67.468 | 27.045 | 2.495x | Yes | 1.492 | 3.722 |
| W2 | QKV | 16 | 2048 | 2048/512/512 | 66.573 | 27.202 | 2.447x | Yes | 3.024 | 7.401 |
| W2 | gate/up | 1 | 2048 | 8192/8192 | 46.370 | 33.189 | 1.397x | Yes | 1.447 | 2.022 |
| W2 | gate/up | 2 | 2048 | 8192/8192 | 45.901 | 33.152 | 1.385x | Yes | 2.924 | 4.049 |
| W2 | gate/up | 4 | 2048 | 8192/8192 | 46.010 | 33.242 | 1.384x | Yes | 5.834 | 8.075 |
| W2 | gate/up | 8 | 2048 | 8192/8192 | 45.825 | 33.151 | 1.382x | Yes | 11.716 | 16.195 |
| W2 | gate/up | 16 | 2048 | 8192/8192 | 46.016 | 33.122 | 1.389x | Yes | 23.334 | 32.418 |
| W2.5 | QKV | 1 | 2048 | 2048/512/512 | 67.794 | 28.761 | 2.357x | Yes | 0.186 | 0.438 |
| W2.5 | QKV | 2 | 2048 | 2048/512/512 | 66.651 | 28.783 | 2.316x | Yes | 0.378 | 0.874 |
| W2.5 | QKV | 4 | 2048 | 2048/512/512 | 67.164 | 28.560 | 2.352x | Yes | 0.749 | 1.762 |
| W2.5 | QKV | 8 | 2048 | 2048/512/512 | 67.176 | 28.534 | 2.354x | Yes | 1.499 | 3.528 |
| W2.5 | QKV | 16 | 2048 | 2048/512/512 | 66.726 | 28.741 | 2.322x | Yes | 3.017 | 7.005 |
| W2.5 | gate/up | 1 | 2048 | 8192/8192 | 45.699 | 33.707 | 1.356x | Yes | 1.468 | 1.991 |
| W2.5 | gate/up | 2 | 2048 | 8192/8192 | 45.752 | 33.342 | 1.372x | Yes | 2.934 | 4.026 |
| W2.5 | gate/up | 4 | 2048 | 8192/8192 | 45.710 | 33.538 | 1.363x | Yes | 5.873 | 8.004 |
| W2.5 | gate/up | 8 | 2048 | 8192/8192 | 45.741 | 33.289 | 1.374x | Yes | 11.737 | 16.128 |
| W2.5 | gate/up | 16 | 2048 | 8192/8192 | 45.735 | 33.299 | 1.373x | Yes | 23.477 | 32.245 |
| W3 | QKV | 1 | 2048 | 2048/512/512 | 66.831 | 23.786 | 2.810x | Yes | 0.188 | 0.529 |
| W3 | QKV | 2 | 2048 | 2048/512/512 | 66.677 | 23.573 | 2.829x | Yes | 0.377 | 1.068 |
| W3 | QKV | 4 | 2048 | 2048/512/512 | 66.786 | 23.506 | 2.841x | Yes | 0.754 | 2.141 |
| W3 | QKV | 8 | 2048 | 2048/512/512 | 65.686 | 23.582 | 2.785x | Yes | 1.532 | 4.269 |
| W3 | QKV | 16 | 2048 | 2048/512/512 | 66.802 | 23.597 | 2.831x | Yes | 3.014 | 8.532 |
| W3 | gate/up | 1 | 2048 | 8192/8192 | 45.462 | 30.960 | 1.468x | Yes | 1.476 | 2.168 |
| W3 | gate/up | 2 | 2048 | 8192/8192 | 45.618 | 30.967 | 1.473x | Yes | 2.942 | 4.334 |
| W3 | gate/up | 4 | 2048 | 8192/8192 | 45.469 | 31.008 | 1.466x | Yes | 5.904 | 8.657 |
| W3 | gate/up | 8 | 2048 | 8192/8192 | 45.427 | 31.014 | 1.465x | Yes | 11.818 | 17.310 |
| W3 | gate/up | 16 | 2048 | 8192/8192 | 45.521 | 31.005 | 1.468x | Yes | 23.588 | 34.631 |
| W3.5 | QKV | 1 | 2048 | 2048/512/512 | 68.763 | 23.618 | 2.912x | Yes | 0.183 | 0.533 |
| W3.5 | QKV | 2 | 2048 | 2048/512/512 | 68.945 | 23.404 | 2.946x | Yes | 0.365 | 1.075 |
| W3.5 | QKV | 4 | 2048 | 2048/512/512 | 67.982 | 23.618 | 2.878x | Yes | 0.740 | 2.131 |
| W3.5 | QKV | 8 | 2048 | 2048/512/512 | 68.745 | 23.487 | 2.927x | Yes | 1.464 | 4.286 |
| W3.5 | QKV | 16 | 2048 | 2048/512/512 | 68.806 | 23.489 | 2.929x | Yes | 2.926 | 8.571 |
| W3.5 | gate/up | 1 | 2048 | 8192/8192 | 46.811 | 31.008 | 1.510x | Yes | 1.434 | 2.164 |
| W3.5 | gate/up | 2 | 2048 | 8192/8192 | 46.853 | 30.850 | 1.519x | Yes | 2.865 | 4.351 |
| W3.5 | gate/up | 4 | 2048 | 8192/8192 | 46.844 | 30.718 | 1.525x | Yes | 5.730 | 8.739 |
| W3.5 | gate/up | 8 | 2048 | 8192/8192 | 46.891 | 30.825 | 1.521x | Yes | 11.449 | 17.417 |
| W3.5 | gate/up | 16 | 2048 | 8192/8192 | 46.850 | 30.833 | 1.519x | Yes | 22.919 | 34.825 |

## Method and controls

- Device: exclusive physical H100
  `GPU-f5ea03cf-efa4-9807-7de5-b174957a1348`, SM 9.0, 132 SMs,
  102,077,169,664 bytes of memory.  `CUDA_VISIBLE_DEVICES=1` hid the H200.
- Workload: Llama 3.2 1B QKV (`K=2048`, `N=2048/512/512`) and gate/up
  (`K=2048`, `N=8192/8192`) at logical M1/M2/M4/M8/M16.
- Kernel M is 16 for both sides.  Rows beyond logical M are identically zero,
  matching current Hopper inference dispatch.
- Every child uses the same explicit split-1 plan on both revisions.  Payloads,
  selector bytes, alternative-bank IDs, input values, compilation flags,
  Torch, and CUDA are identical.
- One CUDA Graph contains either every independent child launch or the one
  grouped launch.  CUDA events bracket 20 graph replays per sample; tables use
  the median of 50 samples after 20 warmups.
- All grouped children are bit-exact to independent children, and
  cross-process output checksums match the baseline for all 40 cases.
- The feature branch's separately compiled plain specialization is 0.996x the
  baseline geometrically.  This near-unity control rules out a rewritten plain
  child kernel as the source of the grouped speedup.
- A denser W3 repeat using 100 samples and 30 replays per sample reproduced
  approximately 2.0x combined speedup at every M.

The improvement comes from exposing all sibling CTAs in one launch.  Narrow K
and V no longer pay two independent launch/scheduling tails, and the scheduler
can fill the H100 from the whole QKV rectangle.  Gate/up similarly replaces two
serial wide grids with one segmented grid.  The decoder and WGMMA math inside
each active CTA remain unchanged.

Marlin and Machete are not included here because this experiment isolates the
Phase-3 change against pre-PR QVQ main.  A W4 baseline should sum the same
independent child shapes and be reported separately.

## Reproduction artifacts

- Driver: `scripts/benchmark_qvq_p32_hopper_grouped.py`
- Pre-Phase-3/plain baseline:
  `artifacts/a41_phase3_h100/pre_phase3_plain_hopper.json`
- Phase-3 grouped:
  `artifacts/a41_phase3_h100/phase3_grouped_hopper.json`
- Feature-branch plain control:
  `artifacts/a41_phase3_h100/phase3_plain_control.json`
- Denser W3 repeats:
  `artifacts/a41_phase3_h100/pre_phase3_w3_repeat.json` and
  `artifacts/a41_phase3_h100/phase3_grouped_w3_repeat.json`
