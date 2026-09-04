# Phase 54: H100 Llama down decode prefetch

Phase 54 extends the packed register-prefetch decoder to the H100 Llama 3.2
1B down projection at exactly K=8192, N=2048, ordered split 16. The full MLP
improves 1.0059x geometrically with 19/20 strict wins. Every rate is now
faster than the Machete W4 full-MLP baseline at every measured M.

## Component-driven target

An in-process CUDA profile at the Phase-53 tip measured the W2.5/M16 complete
MLP. The single eager replay has profiler overhead, but its kernel shares
identify the next target without relying on CPU launch timing.

| Component | CUDA time | Share |
|:--|--:|--:|
| Grouped gate/up P32 | 22.751 us | 43.1% |
| Split-16 down P32 | 12.959 us | 24.5% |
| Gate/up recovery | 6.272 us | 11.9% |
| SwiGLU + down precondition | 4.000 us | 7.6% |
| Down reduction + recovery | 3.936 us | 7.5% |
| Input transform | 2.880 us | 5.5% |

The down P32 kernel was therefore the largest component not already using the
packed-prefetch decoder.

## Exact scheduling change

The split-16 down launch has:

\[
32\ N64\ blocks \times 16\ K\ partitions = 512\ \text{thread blocks}.
\]

Each block decodes one K partition in ascending K16 order. Phase 54 changes
only the local register schedule:

```text
state extraction + four PGC products
eight shared level loads into a temporary decoded fragment
WGMMA dependency wait
four packed register-pair writes
WGMMA issue
```

The sixteen partial planes, left-to-right reduction order, FP32 accumulation,
and fused down recovery remain unchanged. The decoded FP16 values are bit
identical.

## Narrow H100 promotion gate

The new specialization is selected only when all conditions hold:

```text
physical CUDA device name == NVIDIA H100
ordered split path
K == 8192
N == 2048
split count == 16
```

H200, unsplit calls, other shapes, other split counts, and grouped QKV retain
their existing kernels. The device gate uses CUDA device properties, not a
visible-device index or label supplied by the benchmark.

## Isolated down result

Timing uses 20 warmups, 100 CUDA-event samples, and 50 warmed CUDA Graph
replays per sample. Every row is repeatable, graph stable, and within the
dense-P32 error bound.

| W | M/K/N | Split-16 down |
|--:|:--|--:|
| 2 | 1/8192/2048 | 14.266 us |
| 2 | 2/8192/2048 | 14.345 us |
| 2 | 4/8192/2048 | 14.268 us |
| 2 | 8/8192/2048 | 14.260 us |
| 2 | 16/8192/2048 | 14.258 us |
| 2.5 | 1/8192/2048 | 14.408 us |
| 2.5 | 2/8192/2048 | 14.356 us |
| 2.5 | 4/8192/2048 | 14.317 us |
| 2.5 | 8/8192/2048 | 14.332 us |
| 2.5 | 16/8192/2048 | 14.422 us |
| 3 | 1/8192/2048 | 14.564 us |
| 3 | 2/8192/2048 | 14.583 us |
| 3 | 4/8192/2048 | 14.566 us |
| 3 | 8/8192/2048 | 14.557 us |
| 3 | 16/8192/2048 | 14.555 us |
| 3.5 | 1/8192/2048 | 14.598 us |
| 3.5 | 2/8192/2048 | 14.571 us |
| 3.5 | 4/8192/2048 | 14.567 us |
| 3.5 | 8/8192/2048 | 14.624 us |
| 3.5 | 16/8192/2048 | 14.630 us |

Against the most recent matched no-prefetch audit, W2.5 improves about
3.0--4.0% and W3 improves about 1.4--2.4%. The full MLP is the formal
promotion comparison for W2/W3.5 because their last committed down-only
control predates multiple decoder changes.

## Nsight Compute candidate

Nsight Compute 2026.2.1 measured the W2.5/M16 ordered split-16 main kernel:

| Metric | Value |
|:--|--:|
| Duration under NCU | 15.680 us |
| Executed instructions | 5,962,574 |
| Shared-load instructions | 1,114,112 |
| Shared-load bank conflicts | 539,052 |
| Shared-load wavefronts | 1,710,748 |
| Eligible warps/cycle | 0.962 |
| Long-scoreboard cycles/instruction | 3.121 |
| Registers/thread | 63 |
| Shared memory/block | 44,800 B |

The production runtime requests ordered partials, so it does not launch the
standalone reducer profiled by the isolated convenience API. It passes those
partial planes directly into the existing fused ordered reduction and output
recovery kernel.

## Complete Llama 3.2 1B MLP

The formal run acquired three spaced 0% utilization / 0 MiB idle samples on
the physical 132-SM H100. It uses 30 warmups, 200 CUDA-event samples, and 50
warmed CUDA Graph replays per sample. Marlin and Machete are figurative W4
baselines; ratios above one mean QVQ is faster. `Better` compares with Phase
53.

| W | MKN: gate/up x2; down | QVQ | vs Marlin W4 | vs Machete W4 | Better |
|--:|:--|--:|--:|--:|:--:|
| 2 | 1/2048/8192 x2; 1/8192/2048 | 46.227 us | 0.628x | 1.071x | Yes |
| 2 | 2/2048/8192 x2; 2/8192/2048 | 46.400 us | 0.670x | 1.075x | Yes |
| 2 | 4/2048/8192 x2; 4/8192/2048 | 47.144 us | 0.664x | 1.064x | Yes |
| 2 | 8/2048/8192 x2; 8/8192/2048 | 47.545 us | 0.615x | 1.055x | Yes |
| 2 | 16/2048/8192 x2; 16/8192/2048 | 48.844 us | 0.665x | 1.030x | Yes |
| 2.5 | 1/2048/8192 x2; 1/8192/2048 | 47.693 us | 0.609x | 1.038x | Yes |
| 2.5 | 2/2048/8192 x2; 2/8192/2048 | 47.798 us | 0.650x | 1.043x | Yes |
| 2.5 | 4/2048/8192 x2; 4/8192/2048 | 48.542 us | 0.644x | 1.033x | Yes |
| 2.5 | 8/2048/8192 x2; 8/8192/2048 | 49.137 us | 0.595x | 1.021x | Yes |
| 2.5 | 16/2048/8192 x2; 16/8192/2048 | 50.115 us | 0.648x | 1.004x | Yes |
| 3 | 1/2048/8192 x2; 1/8192/2048 | 46.739 us | 0.621x | 1.059x | Yes |
| 3 | 2/2048/8192 x2; 2/8192/2048 | 46.998 us | 0.661x | 1.061x | Yes |
| 3 | 4/2048/8192 x2; 4/8192/2048 | 47.560 us | 0.658x | 1.055x | Yes |
| 3 | 8/2048/8192 x2; 8/8192/2048 | 48.129 us | 0.607x | 1.042x | Yes |
| 3 | 16/2048/8192 x2; 16/8192/2048 | 49.218 us | 0.660x | 1.022x | Yes |
| 3.5 | 1/2048/8192 x2; 1/8192/2048 | 47.950 us | 0.605x | 1.032x | No |
| 3.5 | 2/2048/8192 x2; 2/8192/2048 | 47.875 us | 0.649x | 1.042x | Yes |
| 3.5 | 4/2048/8192 x2; 4/8192/2048 | 48.592 us | 0.644x | 1.032x | Yes |
| 3.5 | 8/2048/8192 x2; 8/8192/2048 | 48.883 us | 0.598x | 1.026x | Yes |
| 3.5 | 16/2048/8192 x2; 16/8192/2048 | 49.719 us | 0.653x | 1.012x | Yes |

Phase 54 improves **1.0059x** geometrically with 19/20 strict wins. Its
geometric ratios are **1.0405x Machete W4** and **0.6368x Marlin W4**. Every
individual cell is now faster than Machete; W3.5/M1's 0.39% movement against
Phase 53 is the only strict run-to-run regression.

## Correctness and resource scope

- 126 Hopper P32/grouped-P32 tests pass.
- Split-16 output is repeatable and CUDA Graph stable at every rate and M.
- Device, shape, split, and ordered-output gates fail closed to the prior
  specialization.
- Checkpoint bytes, persistent VRAM, shared-memory size, partial-plane size,
  reduction order, and launch count are unchanged.
- Compilation used at most four Ninja jobs, one NVCC host thread, and one
  CUDA split-compile partition.

Artifacts:

- `artifacts/a41_phase54_h100/production_mlp_down_prefetch_vs_phase53.json`
- `artifacts/a41_phase54_h100/down_prefetch_profile.json`
- Profiler outputs outside Git under
  `/root/qvq-profiler-artifacts/phase54-down-prefetch/` and
  `/root/qvq-profiler-artifacts/phase54-full-mlp-breakdown/`

## Next phase

The remaining 47--50 us complete MLP is now dominated by gate/up P32 and the
still-independent recovery/precondition boundaries. The next experiment
should re-profile the Phase-54 tip and test whether the 6.3 us gate/up
recovery can feed the 4.0 us SwiGLU/precondition pipeline without writing and
re-reading both full 8192-wide recovered tensors. Any fusion must preserve
the exact FP16 store/rounding boundary between recovery and SiLU.
