# QVQ V2B2-P32 continuous-window Hopper kernel progression

This ledger tracks the accuracy-safe standard-P32 kernel after LR32 development
was stopped.  The continuous-window representation is a lossless physical
permutation of canonical planar P32: it has the same word count, adds zero bits,
and reconstructs the identical K16 x N16 matrix.

## Measurement contract

- Device: physical GPU 0, NVIDIA H200, UUID
  `GPU-0c667065-5c47-38ce-0b0a-d211392ce9ea`, CC 9.0, 132 SMs.
- Model shapes: Qwen3.8-27B; the first optimization gate is M16 FP16 W3.
- Reference: each candidate is checked against its own dense standard-P32
  reconstruction with `max_abs <= 2e-3`.
- Timing: CUDA Graph external-event medians, 10 warmups and 40 measured launches.
- Comparator: Machete symmetric W4 group 128 is a performance reference, not a
  quality-equivalent format.
- CUTLASS/CuTe: 4.7.1, SM90a RS-WGMMA.

## Accepted progression

| Commit | Device | Rate | M | K | N | Kernel | Split | Median ms | Speedup vs planar P32 | xMachete | Max abs |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| `63daa348` | H200 | W3 | 16 | 5120 | 17408 | planar P32 production | auto | 2.49670 | 1.000x | 0.017x | 2.10e-5 |
| `942c5ae6` | H200 | W3 | 16 | 5120 | 17408 | window RS-WGMMA | 4 | 0.12288 | 20.318x | 0.337x | 6.20e-5 |
| `00872584` | H200 | W3 | 16 | 5120 | 17408 | window TMA RS-WGMMA | 4 | 0.07610 | 32.810x | 0.544x | 6.20e-5 |
| `8dea54a5` | H200 | W3 | 16 | 5120 | 17408 | window TMA RS-WGMMA | 10 | 0.07378 | 33.840x | 0.561x | 3.05e-5 |
| `63daa348` | H200 | W3 | 16 | 17408 | 5120 | planar P32 production | auto | 2.50002 | 1.000x | 0.017x | 3.24e-5 |
| `942c5ae6` | H200 | W3 | 16 | 17408 | 5120 | window RS-WGMMA | 4 | 0.15272 | 16.370x | 0.278x | 3.09e-4 |
| `00872584` | H200 | W3 | 16 | 17408 | 5120 | window TMA RS-WGMMA | 4 | 0.09253 | 27.019x | 0.459x | 3.09e-4 |
| `8dea54a5` | H200 | W3 | 16 | 17408 | 5120 | window TMA RS-WGMMA | 34 | 0.07389 | 33.835x | 0.574x | 6.48e-5 |

The direct-window mapping was also checked at M16/K256/N64 and the TMA path at
M16/K256/N256.  Their maximum errors were 2.38e-6 and 3.34e-6.  The full window
format suite at `63daa348` passes 21/21 tests across W1-W3.5, including CPU/CUDA
word and state identity.

## H200 NCU profile

Profiled head: `f36e1837`; kernel source: `00872584`.  Shape:
M16/K5120/N17408 W3 split 4.  Report:
`artifacts/h200_p32_window/profiles/qwen38_gate_w3_p32_tma_rs_wgmma_split4_f36e1837.ncu-rep`.

| Metric | Result |
|---|---:|
| NCU duration | 71.648 us |
| Grid / block | 1088 CTAs / 160 threads |
| Waves per SM | 1.18 |
| Warp instructions | 36.842M |
| Registers / thread | 56 |
| Static shared memory | 29.31 KiB |
| Local/shared spills | 0 / 0 |
| Theoretical / achieved occupancy | 54.69% / 41.59% |
| DRAM / L1 throughput | 10.36% / 78.44% |
| Shared wavefronts actual / ideal | 3.482M / 3.482M |
| Shared bank conflicts | 160 |
| Global PGC level loads | 2.785M |
| Shared window loads | 2.785M |
| TMA pipe utilization | 0.39% |
| No eligible warp | 40.28% |
| Long-scoreboard stall samples | 2,087 |
| Wait / math-throttle samples | 1,022 / 853 |

The dominant opcode mix is LOP3 8.863M (24.1%), IMAD 8.076M (21.9%), SHF
6.406M (17.4%), LDS 3.133M (8.5%), and LDG 2.785M (7.6%).  Every P32 state is
decoded exactly once, so recurrence deduplication is no longer available.  The
next experiments should target the random read-only PGC level loads and launch
tail without damaging the already conflict-free shared/TMA layout.

## Rejected experiments

| Head | Experiment | Gate M16/K5120/N17408 | Down M16/K17408/N5120 | Decision |
|---|---|---:|---:|---|
| `12b3a321` + working tree | Warp-distributed register PGC table | 0.10992 ms (0.687x baseline) | 0.14904 ms (0.620x baseline) | Rejected; exact, but an arbitrary lookup needs four requester-dependent shuffles and is 31-39% slower than the 99.7%-L1-hit read-only table. |

The Qwen3.8 MLP split sweep at `8dea54a5` accepted split 10 for gate/up and
split 34 for down.  These policies preserve K256 stage alignment, reduce the
partial-wave penalty seen in NCU, and are selected automatically for those two
exact shapes.  The default output is bitwise identical to explicitly requesting
the selected split.

## Coverage queue

| Priority | Coverage | State |
|---:|---|---|
| 1 | Warp-register PGC table versus read-only L1 | rejected; keep read-only L1 |
| 2 | W3 split/grid policy to reduce the 164-CTA tail | accepted for gate/down; remaining Qwen shapes pending |
| 3 | Generalize direct-window TMA RS-WGMMA to W2, W2.5, and W3.5 | pending |
| 4 | Qwen3.8 M1/M2/M4/M8 specializations | pending |
| 5 | Full seven-shape W2-W3.5 P32 versus Machete sweep | pending |
