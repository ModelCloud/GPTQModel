# Phase 60: rejected H100 level-row bank hashing

Phase 60 tested two storage-neutral or transient-only mappings intended to
spread W2.5's remaining random level lookups across more shared-memory banks.
Both were output-exact but substantially slower. All candidate source was
removed; production remains byte-identical to Phase 56.

## Baseline alias

The accepted lane-pair table stores 32 FP16 entries per level index. Its
64-byte row stride gives shared-bank index

\[
bank(i,p)=(16i+p)\bmod32,
\]

where `i` is the PGC level index and `p` is the consumer lane pair. Only the
index parity changes the row's bank offset, which leaves approximately one
million extra shared-load wavefronts in the W2.5 N128 kernel.

## Candidate A: 80-byte padded rows

Candidate A padded each transient row from 32 to 40 FP16 values. Four aligned
`uint4` stores still initialized the active values and the final `uint4` was
unused padding. The new bank relation was

\[
bank_{80}(i,p)=(20i+p)\bmod32,
\]

which rotates through eight starting banks rather than two. The layout added
4 KiB of per-block shared memory but no checkpoint or persistent VRAM.

Twenty W2.5 grouped-Hopper tests passed exact child parity, ordered reduction,
dense-oracle tolerance, and real Llama QKV/gate-up shapes.

The physical H100 passed the strict 0% utilization / 0 MiB admission gate.
Timing used 20 warmups, 100 CUDA-event samples, and 30 warmed CUDA Graph
replays per sample. Ratios above one mean QVQ is faster than the figurative W4
baseline. `Better` compares with Phase-56 production.

| W | M/K/N per child | Production | 80-byte row | vs Marlin W4 | vs Machete W4 | vs last | Better |
|--:|:--|--:|--:|--:|--:|--:|:--:|
| 2.5 | 1/2048/8192 x2 | 29.649 us | 33.726 us | 0.408x | 0.856x | 0.8791x | No |
| 2.5 | 2/2048/8192 x2 | 29.420 us | 33.446 us | 0.426x | 0.856x | 0.8796x | No |
| 2.5 | 4/2048/8192 x2 | 29.737 us | 33.772 us | 0.426x | 0.839x | 0.8805x | No |
| 2.5 | 8/2048/8192 x2 | 30.102 us | 34.142 us | 0.397x | 0.827x | 0.8817x | No |
| 2.5 | 16/2048/8192 x2 | 30.930 us | 34.975 us | 0.431x | 0.810x | 0.8843x | No |

The padded row is **0.88105x** geometrically, or 11.9% slower. Replacing a
shift-only row address with a general stride multiply and enlarging the
working set costs much more than any conflict reduction can recover.

## Candidate B: index-hashed replicated word

Every accepted row already replicates the same low/high FP16 pair in all 16
physical words. Candidate B therefore kept the original 64-byte row and chose
an equivalent replicated word dynamically:

\[
word=p\mathbin{\oplus}((i\gg1)\mathbin{\&}15).
\]

The stored bits are identical for every selected word. This makes the
theoretical bank mapping depend on more PGC index bits without allocating or
initializing additional memory. All 69 grouped Hopper tests passed exactly.

| W | M/K/N per child | Production | Hashed word | vs Marlin W4 | vs Machete W4 | vs last | Better |
|--:|:--|--:|--:|--:|--:|--:|:--:|
| 2.5 | 1/2048/8192 x2 | 29.649 us | 37.793 us | 0.368x | 0.774x | 0.7845x | No |
| 2.5 | 2/2048/8192 x2 | 29.420 us | 37.265 us | 0.385x | 0.773x | 0.7895x | No |
| 2.5 | 4/2048/8192 x2 | 29.737 us | 37.759 us | 0.385x | 0.750x | 0.7875x | No |
| 2.5 | 8/2048/8192 x2 | 30.102 us | 38.188 us | 0.359x | 0.737x | 0.7883x | No |
| 2.5 | 16/2048/8192 x2 | 30.930 us | 38.962 us | 0.392x | 0.723x | 0.7938x | No |

The storage-neutral hash is **0.78872x** geometrically, or 21.1% slower. The
extra level-index-to-word dependency is repeated for every decoded value and
lands directly on the shared-load critical path. This is worse than accepting
the second shared wavefront.

Nsight Compute admission was attempted repeatedly for both mappings, but the
device reported a transient external 2% utilization sample on every attempt.
No contaminated conflict counter was accepted. The clean CUDA Graph results
are sufficiently decisive to reject both candidates without weakening the
profiling gate.

## Decision

- No Phase-60 CUDA/runtime candidate remains.
- P32 values, selector math, WGMMA order, output bits, checkpoint storage,
  persistent VRAM, and graph topology are unchanged.
- Compilation used no more than four Ninja jobs, one NVCC host thread, and one
  CUDA split-compile partition.
- Future conflict work must avoid adding any level-index-dependent address
  operation to the hot lookup chain. A useful design needs producer-side
  remapping with the consumer address remaining shift/add only, or a hardware
  lookup primitive that is measurably cheaper than the existing second
  shared wavefront.
