# Phase 49: H100 prepacked decoded P32 fragment

Phase 49 replaces the generic eight-element tensor copy at the end of the
Phase-48 register-prefetch path with an explicit four-word packed fragment.
It preserves the exact decoded FP16 bit patterns while reducing generated
copy/packing work. The optimization remains restricted to grouped Llama 3.2
1B gate/up at W2.5 and W3.

## Register representation

Phase 48 loads eight decoded FP16 values into an independent register tensor:

\[
D=(d_0,d_1,\ldots,d_7),\qquad d_i\in\mathrm{FP16}.
\]

WGMMA consumes the register-sourced operand as four 32-bit pairs. Phase 49
expresses that representation directly:

\[
P_j=\operatorname{bits}(d_{2j})\;|\;
    \left(\operatorname{bits}(d_{2j+1})\ll16\right),
\qquad j=0,\ldots,3.
\]

After the old source fragment is no longer live, the four packed words become
the next WGMMA operand. This is a bitwise representation change only: there
is no floating-point operation, conversion, reordered decode, or new rounding
boundary.

## What SASS actually does

The original hypothesis was that all four pair-packing operations could move
before the WGMMA dependency barrier. Nsight Compute and `cuobjdump` show a
more nuanced result. The eight `LDS.U16` level loads remain before
`WARPGROUP.DEPBAR.LE gsb0, 0x3`, but the final four `PRMT` instructions remain
immediately after it because their destination registers are the old WGMMA
fragment and remain live through the barrier.

The explicit packed representation still removes the generic tensor-copy
plumbing and lowers register pressure:

| W3 M1 grouped gate/up metric | Phase 48 | Phase 49 | Change |
|:--|--:|--:|--:|
| Duration | 25.152 us | 24.352 us | **-3.18%** |
| Executed instructions | 11,529,740 | 10,481,152 | **-9.10%** |
| Shared-load instructions | 2,228,224 | 2,228,224 | unchanged |
| Shared-load bank conflicts | 1,048,848 | 1,048,887 | unchanged |
| Shared-load wavefronts | 3,396,281 | 3,393,248 | -0.09% |
| Eligible warps/cycle | 0.763 | 0.709 | -7.00% |
| Long-scoreboard cycles/instruction | 1.183 | 1.310 | +10.7% |
| Wait-stall ratio | 0.875 | 0.932 | +6.61% |
| Registers/thread | 67 | 65 | -2 |
| Shared memory | 46,848 B | 46,848 B | unchanged |

Unlike Phase 48, this phase wins primarily by deleting instructions rather
than improving scheduler eligibility. The unchanged shared-load count and
conflict count isolate the gain to register representation/copy overhead.
The measurements use Nsight Compute 2026.2.1, and the complete cubin SASS is
also exported with `cuobjdump` outside Git.

## Isolated grouped gate/up result

CUDA Graph replay timing uses 20 warmups, 100 samples, and 50 replays per
sample. Every output is repeatable, graph stable, and exact.

| M/K/N per child | W2.5 Phase 48 | W2.5 Phase 49 | Better | W3 Phase 48 | W3 Phase 49 | Better |
|:--|--:|--:|:--:|--:|--:|:--:|
| 1/2048/8192 x2 | 28.052 us | 27.512 us | Yes | 27.214 us | 26.331 us | Yes |
| 2/2048/8192 x2 | 27.583 us | 27.396 us | Yes | 26.957 us | 26.124 us | Yes |
| 4/2048/8192 x2 | 27.985 us | 27.343 us | Yes | 27.057 us | 26.077 us | Yes |
| 8/2048/8192 x2 | 28.409 us | 27.725 us | Yes | 27.626 us | 26.564 us | Yes |
| 16/2048/8192 x2 | 29.180 us | 28.498 us | Yes | 28.262 us | 27.235 us | Yes |

Geometric improvements are **1.0197x for W2.5** and **1.0361x for W3**.

## Complete Llama 3.2 1B MLP

The formal run acquired three spaced 0% utilization / 0 MiB idle samples on
the physical 132-SM H100. It uses 30 warmups, 200 CUDA-event samples, and 50
warmed CUDA Graph replays per sample. Marlin and Machete are figurative W4
baselines; ratios above one mean QVQ is faster. `Better` compares with Phase
48.

| W | MKN: gate/up x2; down | QVQ | vs Marlin W4 | vs Machete W4 | Better |
|--:|:--|--:|--:|--:|:--:|
| 2 | 1/2048/8192 x2; 1/8192/2048 | 47.709 us | 0.609x | 1.028x | Yes |
| 2 | 2/2048/8192 x2; 2/8192/2048 | 47.950 us | 0.651x | 1.020x | Yes |
| 2 | 4/2048/8192 x2; 4/8192/2048 | 48.731 us | 0.644x | 1.014x | No |
| 2 | 8/2048/8192 x2; 8/8192/2048 | 49.404 us | 0.594x | 1.005x | Yes |
| 2 | 16/2048/8192 x2; 16/8192/2048 | 50.484 us | 0.644x | 0.984x | Yes |
| 2.5 | 1/2048/8192 x2; 1/8192/2048 | 48.076 us | 0.605x | 1.020x | Yes |
| 2.5 | 2/2048/8192 x2; 2/8192/2048 | 48.150 us | 0.648x | 1.015x | Yes |
| 2.5 | 4/2048/8192 x2; 4/8192/2048 | 48.895 us | 0.642x | 1.011x | Yes |
| 2.5 | 8/2048/8192 x2; 8/8192/2048 | 49.586 us | 0.592x | 1.001x | Yes |
| 2.5 | 16/2048/8192 x2; 16/8192/2048 | 50.529 us | 0.643x | 0.983x | Yes |
| 3 | 1/2048/8192 x2; 1/8192/2048 | 47.002 us | 0.619x | 1.043x | Yes |
| 3 | 2/2048/8192 x2; 2/8192/2048 | 47.087 us | 0.663x | 1.038x | Yes |
| 3 | 4/2048/8192 x2; 4/8192/2048 | 47.688 us | 0.658x | 1.037x | Yes |
| 3 | 8/2048/8192 x2; 8/8192/2048 | 48.132 us | 0.610x | 1.031x | Yes |
| 3 | 16/2048/8192 x2; 16/8192/2048 | 49.283 us | 0.659x | 1.008x | Yes |
| 3.5 | 1/2048/8192 x2; 1/8192/2048 | 50.704 us | 0.573x | 0.967x | No |
| 3.5 | 2/2048/8192 x2; 2/8192/2048 | 50.955 us | 0.612x | 0.959x | No |
| 3.5 | 4/2048/8192 x2; 4/8192/2048 | 51.299 us | 0.612x | 0.964x | Yes |
| 3.5 | 8/2048/8192 x2; 8/8192/2048 | 51.592 us | 0.569x | 0.962x | No |
| 3.5 | 16/2048/8192 x2; 16/8192/2048 | 52.579 us | 0.618x | 0.945x | Yes |

The all-rate result improves **1.0071x** geometrically with 16/20 cells
better. All ten targeted W2.5/W3 cells improve. W3 improves **1.0186x** over
Phase 48 and reaches **1.0314x Machete W4** geometrically. Across all rates,
Phase 49 is **0.6226x Marlin W4** and **1.0014x Machete W4**.

## Correctness and scope

- 126 Hopper P32/grouped-P32 tests pass.
- All isolated and complete-MLP cases are repeatable and CUDA Graph stable.
- P32 math, FP16 bits, WGMMA issue order, split reduction, storage, workspace,
  and persistent VRAM are unchanged.
- Compilation used at most four Ninja jobs, one NVCC host thread, and one
  CUDA split-compile partition.

Artifacts:

- `artifacts/a41_phase49_h100/production_mlp_prepacked_prefetch_vs_phase48.json`
- `artifacts/a41_phase49_h100/prepacked_prefetch_profile.json`
- Nsight Compute and `cuobjdump` outputs outside Git under
  `/root/qvq-profiler-artifacts/phase49-prepacked-prefetch/`

## Next phase

Phase 50 should force the four `PRMT` pair constructions into independent
registers before `WARPGROUP.DEPBAR`, then either rename or move those four
words into the reusable WGMMA fragment. This directly tests whether replacing
four post-barrier permutations with four cheap moves improves latency enough
to offset the extra live registers. The experiment remains gated to W2.5/W3
grouped gate/up and must beat Phase 49 across the complete MLP before
promotion.
