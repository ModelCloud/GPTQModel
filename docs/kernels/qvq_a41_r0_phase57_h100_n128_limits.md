# Phase 57: H100 N128 limits

Phase 57 tested two follow-ups to the accepted W2.5 N128 gate/up kernel. Both
were exact but slower, so all candidate source was reverted. Production
remains byte-identical to the Phase-56 depth-three kernel.

## Candidate A: conflict-free private-lane levels

The accepted level table stores one 32-bit word per adjacent lane pair:

\[
T[i,p]=[L(i),L(i\oplus(i\gg7))],
\qquad p=\lfloor lane/2\rfloor.
\]

Its 64-byte index stride maps each pair to one of two banks depending on the
index parity. The two lanes in a pair can therefore serialize when they
request different words with the same parity.

The candidate instead stored one word per lane:

\[
T_{private}[i,l]=[L(i),L(i\oplus(i\gg7))].
\]

With byte address

\[
A(i,l)=A_0+(i\ll7)+(l\ll2),
\]

the shared bank is exactly \(l\). This mathematically eliminates random
decode-load conflicts without sign reconstruction or index approximation.

The price is doubling the block-local level table from 16 KiB to 32 KiB and
initializing 2,048 rather than 1,024 `uint4` entries. Total dynamic shared
memory rises from 54.016 KiB to approximately 70.016 KiB. It adds no
checkpoint or persistent VRAM.

### H100 result

The workload is W2.5 grouped gate/up P32 plus exact paired recovery. It uses
30 warmups, 200 CUDA-event samples, and 50 warmed CUDA Graph replays per
sample. `Better` compares with the accepted Phase-56 depth-three mapping.

| M/K/N per child | Phase 56 | Private-lane table | vs last | Better |
|:--|--:|--:|--:|:--:|
| 1/2048/8192 x2 | 27.080 us | 29.298 us | 0.924x | No |
| 2/2048/8192 x2 | 26.520 us | 28.930 us | 0.917x | No |
| 4/2048/8192 x2 | 26.913 us | 29.324 us | 0.918x | No |
| 8/2048/8192 x2 | 27.183 us | 29.579 us | 0.919x | No |
| 16/2048/8192 x2 | 28.151 us | 30.443 us | 0.925x | No |

The candidate is **0.9205x** geometrically, or 8.0% slower. The larger
initialization and shared working set cost more than the remaining pair-local
two-way conflicts. This closes the in-kernel replicated private-table design
for N128.

Nsight Compute admission was attempted three times but correctly refused
each run because the device intermittently reported 2% external utilization.
The candidate was already decisively slower under clean CUDA-event timing, so
no contaminated profile was accepted.

## Candidate B: N128 split-16 down

The accepted down projection uses:

\[
\frac{2048}{64}\times16=512
\]

N64 thread blocks. Candidate B gave adjacent N64 regions to two consumer
warpgroups, sharing one input TMA stage and decoder table:

\[
\frac{2048}{128}\times16=256
\]

thread blocks. Each split still wrote its original deterministic partial
plane and the ordered reducer was unchanged.

### H100 result

The candidate passed dense-P32 accuracy, ten-launch repeatability, and warmed
CUDA Graph stability for every M. `Better` compares with the unchanged
accepted N64 split-16 down kernel.

| W | M/K/N | Accepted N64 | N128 candidate | vs last | Better |
|--:|:--|--:|--:|--:|:--:|
| 2.5 | 1/8192/2048 | 14.408 us | 14.649 us | 0.984x | No |
| 2.5 | 2/8192/2048 | 14.356 us | 14.723 us | 0.975x | No |
| 2.5 | 4/8192/2048 | 14.317 us | 14.682 us | 0.975x | No |
| 2.5 | 8/8192/2048 | 14.332 us | 14.690 us | 0.976x | No |
| 2.5 | 16/8192/2048 | 14.422 us | 14.654 us | 0.984x | No |

The candidate is **0.9787x** geometrically, or 2.1% slower. Two H100 waves
are not enough to make the shared setup savings offset the loss of block-level
parallel scheduling. N128 remains restricted to W2.5 gate/up.

## Comparator policy

Neither candidate reached the component promotion gate, so running a full
Marlin/Machete matrix would consume the H100 only to reconfirm a known local
regression. The most recent production cross-kernel comparison remains the
Phase-56 table: QVQ is **1.0510x Machete W4** and **0.6411x Marlin W4**
geometrically across W2--W3.5 and M1/2/4/8/16.

## Decision

- No candidate CUDA or runtime code remains.
- Production output, graph topology, storage, and dispatch are unchanged.
- Candidate compilation and testing used no more than four Ninja jobs, one
  NVCC host thread, and one CUDA split-compile partition.
- Future work should not enlarge the per-block level table or reduce the
  split-16 down grid. A promising conflict-free design would need to preload
  an expanded table without per-block replication overhead, or eliminate
  level lookups through reuse that does not add random global traffic.
