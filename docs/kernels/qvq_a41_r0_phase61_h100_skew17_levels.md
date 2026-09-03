# Phase 61: rejected H100 17-word level-row skew

Phase 61 tested the smallest padded shared-level row that exposes all five
level-index bits to Hopper's 32 shared-memory banks. The candidate was exact,
but grouped gate/up regressed 17--21% and the complete W2.5 MLP regressed
roughly 10%. All candidate CUDA source was removed; production remains
byte-identical to Phase 56.

## Layout math

Production stores sixteen active 32-bit lane-pair words per level. Its
64-byte stride gives

\[
bank(i,p)=(16i+p)\bmod32,
\]

so only level-index parity changes the starting bank. The candidate appended
one unused 32-bit word:

\[
T_{61}[i]=[w_0,\ldots,w_{15},pad],
\qquad sizeof(T_{61}[i])=68\ bytes.
\]

The active word mapping became

\[
bank_{61}(i,p)=(17i+p)\bmod32.
\]

Because 17 and 32 are coprime, consecutive logical levels cycle through all
32 bank offsets. The consumer address used only shifts and additions:

\[
A(i,l)=A_0+(i\ll6)+(i\ll2)+(l\ll1).
\]

For the existing low-byte offset \(o=2i\), the equivalent expression was

\[
A(o,l)=A_0+(o\ll5)+(o\ll1)+(l\ll1).
\]

The canonical and high-permuted FP16 views, PGC values, fragment order,
WGMMA issue order, and FP32 accumulation were unchanged. The table grew only
from 16 to 17 KiB per block, with no checkpoint or persistent-VRAM change.

## Exactness and isolated H100 timing

Ten real Llama-shape W2.5 grouped-Hopper cases passed exact child parity,
dense-oracle tolerance, repeatability, and CUDA Graph stability. The physical
H100 passed three spaced 0% utilization / 0 MiB admission samples. Timing used
20 warmups, 100 CUDA-event samples, and 30 warmed graph replays per sample.

| W | M/K/N per child | Phase 56 production | 17-word skew | vs last | Better |
|--:|:--|--:|--:|--:|:--:|
| 2.5 | 1/2048/8192 x2 | 27.638 us | 33.380 us | 0.8280x | No |
| 2.5 | 2/2048/8192 x2 | 26.936 us | 32.243 us | 0.8354x | No |
| 2.5 | 4/2048/8192 x2 | 27.233 us | 32.438 us | 0.8396x | No |
| 2.5 | 8/2048/8192 x2 | 27.395 us | 32.677 us | 0.8384x | No |
| 2.5 | 16/2048/8192 x2 | 28.109 us | 33.543 us | 0.8380x | No |

## Complete Llama 3.2 1B MLP

The required full-operation comparison used the exact preserved candidate
binary after source reversion. Its SHA-256 is
`f0508c5506bd2c1dd44ff3080229c3ada57d499b8cfb851f576f2f9f7ac72c10`;
the source fingerprint captured by the candidate run is
`b6b90a58786e203420103264eb5bfb3b5f1e043de16ebe0895084d1fe747cb0f`.
The candidate binary was loaded before the ordinary lazy extension resolver,
so the rest of the production runtime and benchmark were unchanged.

Marlin and Machete are figurative W4 baselines. Ratios above one mean QVQ is
faster. `Better` compares the complete candidate MLP with the committed
Phase-56 W2.5 row.

| W | MKN: gate/up x2; down | QVQ candidate | vs Marlin W4 | vs Machete W4 | vs last | Better |
|--:|:--|--:|--:|--:|--:|:--:|
| 2.5 | 1/2048/8192 x2; 1/8192/2048 | 53.083 us | 0.555x | 0.956x | 0.891x | No |
| 2.5 | 2/2048/8192 x2; 2/8192/2048 | 52.814 us | 0.599x | 0.946x | 0.898x | No |
| 2.5 | 4/2048/8192 x2; 4/8192/2048 | 53.494 us | 0.590x | 0.936x | 0.899x | No |
| 2.5 | 8/2048/8192 x2; 8/8192/2048 | 54.030 us | 0.545x | 0.925x | 0.899x | No |
| 2.5 | 16/2048/8192 x2; 16/8192/2048 | 55.075 us | 0.593x | 0.907x | 0.903x | No |

## Nsight Compute and SASS result

Nsight Compute 2026.2.1 captured W2.5/M1 after a clean H100 admission. The
Phase-56 report is the matched production reference.

| Metric | Phase 56 | 17-word skew | Change |
|:--|--:|--:|--:|
| NCU duration | 24.800 us | 30.784 us | 0.806x |
| Executed warp instructions | 10,381,568 | 10,619,520 | +2.29% |
| Eligible warps/cycle | 0.741 | 0.532 | -28.2% |
| Long-scoreboard cycles/instruction | 0.799 | 1.066 | +33.4% |
| Registers/thread | 64 | 64 | unchanged |
| Dynamic shared memory | 54.016 KiB | 55.040 KiB | +1.024 KiB |

`cuobjdump` confirms no spills and the same 64 registers/thread. The static
W2.5 N128 function contains 1,816 instructions versus 2,128 in the production
binary, but this does not translate to throughput: the added level-index
address dependency sharply reduces eligible work and exposes shared-load
latency. The clean event and NCU results reject the candidate regardless of
whether its theoretical bank distribution is better.

The NCU report is outside Git at:

```text
/root/qvq-profiler-artifacts/phase61-skew17/candidate_w25_m1.ncu-rep
```

## Decision

- No Phase-61 CUDA/runtime candidate remains.
- Quantized values, output bits, graph topology, checkpoint bytes, and
  persistent VRAM remain unchanged.
- Compilation used no more than four Ninja jobs, one NVCC host thread, and one
  CUDA split-compile partition.
- Compact row padding is closed. A viable conflict reduction cannot put any
  additional level-index-dependent address operation on the consumer's
  critical path, even when the operation is only a shift and add.
