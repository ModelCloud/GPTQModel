# QVQ A41/R0 Phase 20: H100 W3 level-conflict limits

Phase 20 tested three ways to remove the remaining shared-memory bank
conflicts from the W3 grouped gate/up decoder. None passed the complete MLP
promotion gate. Production therefore remains byte-for-byte identical to
Phase 19 after revert commit `51b616ee`.

This is still a useful negative result. A lane-private level table can make
the isolated decoder substantially faster, but its larger opt-in shared
allocation changes whole-graph behavior enough to erase that gain. Removing
the table conflicts by giving up Tensor Memory Accelerator (TMA) double
buffering is also a net loss. The next decoder phase must retain both load
overlap and the existing compact launch resources.

## Production mapping

For codebook level `L[i]`, decoded byte `i`, and consumer lane `l`, the
Phase-19 table stores 32 FP16 values per index. One 32-bit word is shared by
each adjacent lane pair:

\[
T[i,2p]=L[i],
\qquad
T[i,2p+1]=L[i\mathbin{\oplus}(i\gg7)].
\]

The even half is the canonical low-byte view and the odd half is the exact
high-byte permutation. Its byte address is

\[
A(i,l)=A_0+(i\ll6)+(l\ll1),
\]

and its shared bank is

\[
bank(i,l)=\left(16i+\left\lfloor l/2\right\rfloor\right)\bmod32.
\]

The 16 KiB table plus two staged input/trellis/selector buffers consumes
45.824 KiB per block. The adjacent consumers in a lane pair can request two
different level indices from the same bank, which accounts for approximately
1.05 million bank conflicts in the profiled W3 M1 launch.

## Candidate A: one TMA stage and one word per lane

The first candidate doubled the W3 table to 32 KiB and stored one canonical
and one high-permuted FP16 value in a private word for each lane:

\[
T_{private}[i,l,0]=L[i],
\qquad
T_{private}[i,l,1]=L[i\mathbin{\oplus}(i\gg7)].
\]

Its addresses are

\[
A_{low}(i,l)=A_0+(i\ll7)+(l\ll2),
\]

\[
A_{high}(i,l)=A_{low}(i,l)+2.
\]

Therefore

\[
bank(i,l)=l,
\]

so every active lane owns a distinct bank. To remain below the default 48 KiB
block limit, the experiment reduced the TMA pipeline from two stages to one.

Nsight Compute confirmed that the table conflicts disappeared, but the
single-stage pipeline could no longer overlap the next weight/input transfer
with current decode. Long-scoreboard stalls rose from 1.38 to 2.20
instructions per issue-active cycle and replay latency regressed from 28.192
to 29.440 microseconds. The experiment was reverted.

## Candidate B: compact-table address swizzle

The second candidate kept two TMA stages and the 16 KiB table. For logical
word

\[
w=16i+\left\lfloor l/2\right\rfloor,
\]

it used the bijective mapping

\[
w'=w\mathbin{\oplus}((i\gg1)\mathbin{\&}31).
\]

The intent was to expose more index entropy to the bank selection while
retaining the lane-pair word. That intuition was wrong for the actual
per-lane random index stream. Bank conflicts increased from 1.05 million to
2.65 million, wavefronts increased from 3.35 million to 4.95 million, and the
extra address math raised executed warp instructions from 10.79 million to
14.46 million. Replay latency regressed to 35.940 microseconds and CUDA-event
latency regressed to 39.01--39.64 microseconds. The experiment was reverted.

## Candidate C: two TMA stages plus opt-in dynamic shared memory

The third candidate combined the conflict-free lane-private table with the
original two TMA stages. The total 61.824 KiB block allocation was moved to
Hopper opt-in dynamic shared memory and the kernel requested that capacity
with `cudaFuncSetAttribute`.

The isolated result was strong:

| M | Production decoder us | Candidate decoder us | Speedup | Exact | Better |
|---:|---:|---:|---:|:---:|:---:|
| 1 | 30.295 | 27.977 | 1.0828x | Yes | Yes |
| 2 | 30.104 | 27.615 | 1.0901x | Yes | Yes |
| 4 | 30.526 | 27.995 | 1.0904x | Yes | Yes |
| 8 | 30.870 | 28.356 | 1.0887x | Yes | Yes |
| 16 | 31.672 | 29.184 | 1.0852x | Yes | Yes |

The geometric-mean isolated speedup was **1.0875x**. Nsight Compute measured
1,050,805 to 3,066 shared-load conflicts, 3.35 million to 2.26 million
wavefronts, 0.588 to 0.66 eligible warps per cycle, and 28.192 to 25.630
microseconds replay latency.

The complete Llama 3.2 1B MLP did not improve. A same-window detached
Phase-19 executable provided the production QVQ measurements. Marlin and
Machete values below come from the candidate's strict-idle H100 matrix. The
effective throughput is the logical dense-equivalent FLOP count divided by
QVQ latency; it is not a claim that W3 and W4 perform equal compressed work.

| Rate | M | MKN (gate/up; down) | Candidate us | Effective TFLOP/s | vs Marlin W4 | vs Machete W4 | vs last benchmark | Better |
|---:|---:|:---|---:|---:|---:|---:|---:|:---:|
| W3 | 1 | 1x2048x8192 (x2); 1x8192x2048 | 61.841 | 1.628 | 0.473x | 0.800x | 0.9979x | No |
| W3 | 2 | 2x2048x8192 (x2); 2x8192x2048 | 60.743 | 3.314 | 0.513x | 0.811x | 0.9965x | No |
| W3 | 4 | 4x2048x8192 (x2); 4x8192x2048 | 60.456 | 6.660 | 0.519x | 0.824x | 0.9971x | No |
| W3 | 8 | 8x2048x8192 (x2); 8x8192x2048 | 60.498 | 13.311 | 0.485x | 0.826x | 0.9976x | No |
| W3 | 16 | 16x2048x8192 (x2); 16x8192x2048 | 61.638 | 26.130 | 0.529x | 0.813x | 0.9945x | No |

The matched production QVQ latencies were 61.663, 60.541, 60.162, 60.473,
and 61.573 microseconds. Candidate/production geometric speedup was
**0.9975x**, so all five full-MLP cells failed promotion. Moving every rate's
storage to dynamic shared memory also regressed unchanged W2.5 and W3.5
cells, reinforcing the rejection. A W3-only static/dynamic specialization was
compiled and checked for exactness, but it cannot repair the W3 complete-MLP
failure and was not promoted.

## Profiler evidence

All retained profiles used Nsight Compute 2026.2.1 with kernel replay on the
physical 132-SM H100 UUID
`GPU-f5ea03cf-efa4-9807-7de5-b174957a1348`. The profiler itself occupied 4
MiB, so profiling used the documented 4-MiB allowance; formal CUDA-event
benchmarks required zero MiB before admission.

| W3 M1 variant | Replay us | Warp instructions | Shared-load conflicts | Shared-load wavefronts | Eligible warps/cycle | Long scoreboard |
|:--|--:|--:|--:|--:|--:|--:|
| Production, two-stage lane-pair | 28.192 | 10,792,258 | 1,050,805 | 3,350,926 | 0.588 | 1.38 |
| One-stage lane-private | 29.440 | 10,634,502 | 1,352 | 2,264,558 | 0.54 | 2.20 |
| Two-stage swizzled lane-pair | 35.940 | 14,462,910 | 2,645,310 | 4,945,698 | 0.60 | 1.26 |
| Two-stage dynamic lane-private | 25.630 | 10,878,738 | 3,066 | 2,263,334 | 0.66 | 1.32 |

Binary reports remain outside Git under
`/root/qvq-profiler-artifacts/phase20-gateup`. Compact profiler metrics and
CUDA-event results are committed in `artifacts/a41_phase20_h100`.

## Correctness, storage, and decision

- Every candidate was bit-exact and CUDA Graph stable in the focused W3
  M1/M2/M4/M8/M16 benchmark.
- The dynamic lane-private candidate passed all 126 Hopper P32/grouped tests
  across W2/W2.5/W3/W3.5, ordered split reduction, dense-oracle bounds,
  repeatability, graph replay, and Llama shapes.
- No checkpoint, canonical P32 payload, selector, alternative bank, FP16
  level, WGMMA accumulation order, or persistent VRAM changed.
- All candidate production source was reverted. `git diff be4df5b1 --
  gptqmodel_ext/qvq/qvq_wgmma_cuda.cu` is empty after `51b616ee`.
- Compilation used at most four Ninja jobs, one QVQ NVCC host thread, and one
  split-compile partition. A detached baseline attempt exposed a Machete
  wrapper that tried `nvcc --threads 16`; that build was stopped before
  timing and replaced by a QVQ-only baseline driver.

Phase 20 therefore promotes **no kernel change**. The conflict-free layout is
useful evidence, but the next phase should delete or reuse decode work without
increasing block shared memory or weakening the two-stage TMA pipeline.
