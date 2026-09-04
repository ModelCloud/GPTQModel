# Phase 59: H100 gate/up prefetch limits

Phase 59 tested three exact or intended-exact extensions of the grouped
gate/up decoder. None met the promotion gate. All candidate CUDA source was
removed, so production remains byte-identical to Phase 56.

## Candidate A: canonical read-only level table

The W2.5 N128 decoder's lane-pair shared table occupies 16 KiB and still
generates roughly one million shared-load bank conflicts. Candidate A skipped
that per-block table initialization and loaded the immutable 512-byte
canonical FP16 codebook through the read-only cache.

The candidate failed exact plain-child parity at all five target M values
before timing. The shared decoder contains an explicitly pre-permuted
high-byte table view; substituting the generic global fallback did not retain
the exact accepted child schedule. It was reverted without a benchmark.

## Candidate B: W3.5 N128 dual-consumer block

The accepted W2.5 kernel assigns adjacent N64 regions to two independent
consumer warpgroups sharing one producer and input transfer. Candidate B
enabled the same lossless N128 mapping for W3.5. P32 state extraction, PGC,
level lookup, WGMMA order, output recovery, and checkpoint bytes were
unchanged.

The physical H100 passed the strict 0% utilization / 0 MiB admission gate.
Timing used 20 warmups, 100 CUDA-event samples, and 30 warmed CUDA Graph
replays per sample. Ratios above one mean QVQ is faster than the figurative W4
baseline. `Better` compares with the matched N64 executable.

| W | M/K/N per child | N64 | N128 candidate | vs Marlin W4 | vs Machete W4 | vs last | Better |
|--:|:--|--:|--:|--:|--:|--:|:--:|
| 3.5 | 1/2048/8192 x2 | 29.219 us | 29.906 us | 0.462x | 0.977x | 0.9770x | No |
| 3.5 | 2/2048/8192 x2 | 28.940 us | 29.714 us | 0.476x | 0.963x | 0.9740x | No |
| 3.5 | 4/2048/8192 x2 | 29.442 us | 29.810 us | 0.480x | 0.948x | 0.9876x | No |
| 3.5 | 8/2048/8192 x2 | 29.904 us | 30.275 us | 0.451x | 0.929x | 0.9877x | No |
| 3.5 | 16/2048/8192 x2 | 30.655 us | 31.179 us | 0.485x | 0.901x | 0.9832x | No |

N128 is **0.98190x** geometrically with 0/5 wins. Its 64-block launch does
not provide enough independent block scheduling on the 132-SM H100 to repay
the shared producer work. W3.5 remains on N64.

## Candidate C: force pair construction before the dependency wait

Phase 49 observed that four FP16 pair-pack `PRMT` instructions remained after
each `WARPGROUP.DEPBAR`. Candidate C expressed each pair with volatile inline
PTX `mov.b32` before the wait. The output remained exact and the focused
timing appeared positive:

| W | M/K/N per child | Production | Inline-PTX candidate | vs Marlin W4 | vs Machete W4 | vs last | Better |
|--:|:--|--:|--:|--:|--:|--:|:--:|
| 2.5 | 1/2048/8192 x2 | 29.649 us | 29.421 us | 0.469x | 0.998x | 1.0078x | Yes |
| 2.5 | 2/2048/8192 x2 | 29.420 us | 29.116 us | 0.490x | 1.000x | 1.0104x | Yes |
| 2.5 | 4/2048/8192 x2 | 29.737 us | 29.498 us | 0.482x | 0.984x | 1.0081x | Yes |
| 2.5 | 8/2048/8192 x2 | 30.102 us | 29.987 us | 0.457x | 0.965x | 1.0038x | Yes |
| 2.5 | 16/2048/8192 x2 | 30.930 us | 30.778 us | 0.490x | 0.941x | 1.0049x | Yes |
| 3 | 1/2048/8192 x2 | 28.755 us | 28.575 us | 0.483x | 1.027x | 1.0063x | Yes |
| 3 | 2/2048/8192 x2 | 28.481 us | 28.323 us | 0.504x | 1.028x | 1.0056x | Yes |
| 3 | 4/2048/8192 x2 | 28.869 us | 28.787 us | 0.494x | 1.008x | 1.0029x | Yes |
| 3 | 8/2048/8192 x2 | 29.283 us | 29.140 us | 0.470x | 0.993x | 1.0049x | Yes |
| 3 | 16/2048/8192 x2 | 30.131 us | 29.846 us | 0.505x | 0.970x | 1.0096x | Yes |

The apparent geometric movements were 1.00700x and 1.00583x. `cuobjdump`
then proved that the production and candidate functions had identical SASS
hashes for both W2.5 N128 and W3 N64. In both binaries the four `PRMT`
instructions still followed the dependency barrier, with identical resource
usage. The timing movement was therefore run variation, not an executable
optimization. The source commit was immediately reverted and is not
production.

## Candidate D: spare decoded-fragment ring

Candidate D used a physical fragment ring of depth `D+1` while retaining at
most `D` outstanding WGMMA groups. The spare fragment received the next eight
decoded levels while all prior source fragments remained live:

\[
F_{j\bmod(D+1)}\leftarrow Decode(state_j),
\]

then the ordered wait retired the oldest group before the new WGMMA issue.
This avoids copying an independent decoded temporary into the newly reusable
source fragment.

The implementation was bit-exact in all ten real-shape tests. Generated
machine code also changed materially: W2.5 N128 fell from 64 to 56
registers/thread, and two of four pair-pack permutations moved ahead of each
dependency wait. Performance nevertheless regressed decisively:

| W | M/K/N per child | Production | Spare-ring candidate | vs Marlin W4 | vs Machete W4 | vs last | Better |
|--:|:--|--:|--:|--:|--:|--:|:--:|
| 2.5 | 1/2048/8192 x2 | 29.649 us | 33.314 us | 0.414x | 0.870x | 0.8900x | No |
| 2.5 | 2/2048/8192 x2 | 29.420 us | 32.860 us | 0.433x | 0.874x | 0.8953x | No |
| 2.5 | 4/2048/8192 x2 | 29.737 us | 33.267 us | 0.432x | 0.854x | 0.8939x | No |
| 2.5 | 8/2048/8192 x2 | 30.102 us | 33.699 us | 0.403x | 0.842x | 0.8933x | No |
| 2.5 | 16/2048/8192 x2 | 30.930 us | 34.525 us | 0.437x | 0.825x | 0.8959x | No |
| 3 | 1/2048/8192 x2 | 28.755 us | 31.587 us | 0.437x | 0.917x | 0.9104x | No |
| 3 | 2/2048/8192 x2 | 28.481 us | 31.223 us | 0.455x | 0.920x | 0.9122x | No |
| 3 | 4/2048/8192 x2 | 28.869 us | 31.966 us | 0.449x | 0.888x | 0.9031x | No |
| 3 | 8/2048/8192 x2 | 29.283 us | 32.300 us | 0.420x | 0.879x | 0.9066x | No |
| 3 | 16/2048/8192 x2 | 30.131 us | 33.015 us | 0.457x | 0.862x | 0.9126x | No |

The candidate is **0.89366x** geometrically for W2.5 and **0.90897x** for
W3. Moving two permutations and reducing allocated registers did not retain
the accepted decoder's dependency/issue overlap. A changed SASS stream is
necessary but not sufficient; this schedule is closed.

## Decision

- No Phase-59 CUDA or runtime candidate remains.
- Production stays on Phase-56 W2.5 N128 depth three and the existing N64
  paths for all other rates.
- Quantization math, checkpoint bytes, persistent VRAM, graph topology, and
  output ordering are unchanged.
- Compilation used no more than four Ninja jobs, one NVCC host thread, and one
  CUDA split-compile partition.
- Future work should not force pre-barrier packing by widening the fragment
  ring, and source-only PTX changes must be rejected when SASS is identical.
