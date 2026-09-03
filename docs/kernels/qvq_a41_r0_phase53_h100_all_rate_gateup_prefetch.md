# Phase 53: H100 all-rate grouped gate/up prefetch

Phase 53 re-sweeps the Phase-49 packed decode-prefetch representation for W2
and W3.5. Both rates now improve at every M, so all W2--W3.5 rates use the
prefetch path only for the measured Llama 3.2 1B grouped gate/up geometry.
Ordered-split QKV and split-16 down remain on their independently measured
schedules.

## Why the old decision changed

The original Phase-48 broad prefetch experiment copied an eight-element CuTe
tensor into the reusable WGMMA fragment. That form was not consistently
profitable for W2/W3.5, so production initially enabled it only for W2.5/W3.

Phase 49 changed the transient representation to four explicit 32-bit words:

\[
P_j=\operatorname{bits}(d_{2j})\;|\;
    \left(\operatorname{bits}(d_{2j+1})\ll16\right).
\]

That removed generic tensor-copy plumbing and changed the register/scheduler
tradeoff. Phase 53 therefore re-measures rather than carrying forward the old
rate gate. No P32 state, level, WGMMA, or reduction math changes.

The final launch policy is:

| Site | W2 | W2.5 | W3 | W3.5 |
|:--|:--:|:--:|:--:|:--:|
| Grouped gate/up, K=2048, N=(8192,8192), split 1 | **Prefetch** | **Prefetch** | **Prefetch** | **Prefetch** |
| Grouped QKV, ordered split `(8,8,8)` | Phase 47 | Phase 47 | Phase 47 | Phase 47 |
| Non-grouped down, split 16 | Phase 47 | Phase 47 | Phase 47 | Phase 47 |

## Rejected intermediate experiments

Two intervening variants were exact but not promoted:

1. Four inline PTX `PRMT` operations forced before the WGMMA dependency
   barrier regressed all five W3 cells by 0.6--1.4%. Extra live registers and
   moves cost more than the four post-barrier permutations.
2. A five-slot rotating operand ring eliminated the copy, but lost the
   successful level-load overlap. W2.5 regressed slightly and W3 regressed
   roughly 9--11%.
3. An algebraically equivalent unscaled PGC low-index address removed one
   apparent source operation but regressed W2.5 by 3.2--4.1% and W3 by 0.5%
   geometrically because ptxas produced a worse dependency schedule.

These results reinforce that generated scheduling and register liveness are
the optimization target; source-level instruction count alone is not a
promotion criterion.

## Isolated grouped gate/up result

Timing uses warmed CUDA Graph replay and CUDA events. All outputs are exact,
repeatable, and graph stable.

| W | M/K/N per child | Prior production | Packed prefetch | Better |
|--:|:--|--:|--:|:--:|
| 2 | 1/2048/8192 x2 | 27.442 us | 26.357 us | Yes |
| 2 | 2/2048/8192 x2 | 27.177 us | 26.194 us | Yes |
| 2 | 4/2048/8192 x2 | 27.564 us | 26.138 us | Yes |
| 2 | 8/2048/8192 x2 | 27.787 us | 26.436 us | Yes |
| 2 | 16/2048/8192 x2 | 28.614 us | 27.267 us | Yes |
| 3.5 | 1/2048/8192 x2 | 28.998 us | 27.243 us | Yes |
| 3.5 | 2/2048/8192 x2 | 28.842 us | 26.856 us | Yes |
| 3.5 | 4/2048/8192 x2 | 29.193 us | 26.723 us | Yes |
| 3.5 | 8/2048/8192 x2 | 29.603 us | 27.204 us | Yes |
| 3.5 | 16/2048/8192 x2 | 30.317 us | 27.975 us | Yes |

## Nsight Compute evidence

Both candidate profiles use Nsight Compute 2026.2.1 on the physical H100.
The baselines are the accepted deferred-wait kernels for the same M1 grouped
gate/up geometry.

| Metric | W2 prior | W2 prefetch | W3.5 prior | W3.5 prefetch |
|:--|--:|--:|--:|--:|
| Duration | 25.184 us | **23.904 us** | 26.976 us | **24.736 us** |
| Executed instructions | 10,394,892 | 10,234,392 | 10,800,390 | 10,667,544 |
| Shared-load instructions | 1,966,080 | 1,966,080 | 2,228,224 | 2,228,224 |
| Shared-load bank conflicts | 1,052,585 | 1,048,952 | 1,051,001 | 1,048,556 |
| Shared-load wavefronts | 3,394,571 | 3,397,803 | 3,386,057 | 3,392,279 |
| Eligible warps/cycle | 0.656 | **0.710** | 0.608 | **0.703** |
| Long-scoreboard cycles/instruction | 1.312 | **1.245** | 1.324 | **1.258** |
| Registers/thread | 53 | 66 | 57 | 71 |

The W2 kernel is 5.08% faster and W3.5 is 8.30% faster under NCU. Shared
loads and conflict/wavefront counts remain effectively unchanged. The gain
comes from the different register schedule: more eligible warps, less exposed
long-scoreboard latency, and modestly fewer executed instructions. The extra
registers do not change useful residency for this two-wave gate/up launch.

## Complete Llama 3.2 1B MLP

The formal run acquired three spaced 0% utilization / 0 MiB idle samples on
the 132-SM H100. It uses 30 warmups, 200 CUDA-event samples, and 50 warmed
CUDA Graph replays per sample. Marlin and Machete are figurative W4 baselines;
ratios above one mean QVQ is faster. `Better` compares with Phase 49.

| W | MKN: gate/up x2; down | QVQ | vs Marlin W4 | vs Machete W4 | Better |
|--:|:--|--:|--:|--:|:--:|
| 2 | 1/2048/8192 x2; 1/8192/2048 | 46.284 us | 0.632x | 1.080x | Yes |
| 2 | 2/2048/8192 x2; 2/8192/2048 | 46.771 us | 0.669x | 1.062x | Yes |
| 2 | 4/2048/8192 x2; 4/8192/2048 | 47.406 us | 0.665x | 1.061x | Yes |
| 2 | 8/2048/8192 x2; 8/8192/2048 | 48.185 us | 0.611x | 1.046x | Yes |
| 2 | 16/2048/8192 x2; 16/8192/2048 | 49.225 us | 0.663x | 1.025x | Yes |
| 2.5 | 1/2048/8192 x2; 1/8192/2048 | 47.906 us | 0.611x | 1.044x | Yes |
| 2.5 | 2/2048/8192 x2; 2/8192/2048 | 48.071 us | 0.651x | 1.034x | Yes |
| 2.5 | 4/2048/8192 x2; 4/8192/2048 | 48.808 us | 0.646x | 1.030x | Yes |
| 2.5 | 8/2048/8192 x2; 8/8192/2048 | 49.430 us | 0.595x | 1.019x | Yes |
| 2.5 | 16/2048/8192 x2; 16/8192/2048 | 50.505 us | 0.646x | 0.999x | Yes |
| 3 | 1/2048/8192 x2; 1/8192/2048 | 47.118 us | 0.621x | 1.061x | No |
| 3 | 2/2048/8192 x2; 2/8192/2048 | 47.244 us | 0.663x | 1.052x | No |
| 3 | 4/2048/8192 x2; 4/8192/2048 | 47.793 us | 0.659x | 1.052x | No |
| 3 | 8/2048/8192 x2; 8/8192/2048 | 48.145 us | 0.611x | 1.047x | No |
| 3 | 16/2048/8192 x2; 16/8192/2048 | 49.426 us | 0.660x | 1.021x | No |
| 3.5 | 1/2048/8192 x2; 1/8192/2048 | 47.764 us | 0.613x | 1.047x | Yes |
| 3.5 | 2/2048/8192 x2; 2/8192/2048 | 48.594 us | 0.644x | 1.022x | Yes |
| 3.5 | 4/2048/8192 x2; 4/8192/2048 | 48.868 us | 0.645x | 1.029x | Yes |
| 3.5 | 8/2048/8192 x2; 8/8192/2048 | 49.233 us | 0.598x | 1.024x | Yes |
| 3.5 | 16/2048/8192 x2; 16/8192/2048 | 50.058 us | 0.652x | 1.008x | Yes |

Phase 53 improves **1.0194x** geometrically and 15/20 strict cells. The
newly targeted W2 improves **1.0270x** and W3.5 improves **1.0516x**, each
with five wins. W3 is unchanged code and its 0.22% geometric movement is
cross-run variation. Overall QVQ reaches **1.0378x Machete W4** and
**0.6373x Marlin W4** geometrically.

## Correctness and scope

- 126 Hopper P32/grouped-P32 tests pass.
- Every targeted isolated and complete-MLP output is repeatable and CUDA
  Graph stable.
- The optimization adds no checkpoint bytes, persistent VRAM, shared memory,
  workspace, launch, or reduction-order change.
- Compilation used at most four Ninja jobs, one NVCC host thread, and one
  CUDA split-compile partition.

Artifacts:

- `artifacts/a41_phase53_h100/production_mlp_all_rate_packed_prefetch_vs_phase49.json`
- `artifacts/a41_phase53_h100/all_rate_prefetch_profile.json`
- Nsight Compute reports outside Git under
  `/root/qvq-profiler-artifacts/phase53-all-rate-prefetch/`

## Next phase

The complete MLP is now faster than Machete W4 geometrically at every rate;
only W2.5 M16 is a statistical tie at 0.999x. Further gate/up decode changes
should start from matched SASS and preserve the five-fragment physical
schedule. The next experiment should instead profile the full MLP component
breakdown at Phase-53 tip and target the largest remaining non-decode launch
or materialization boundary, rather than continuing speculative PGC algebra.
