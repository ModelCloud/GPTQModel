# Phase 65: rejected H100 adjacent-column recovery vectorization

Phase 65 tested two adjacent recovery/precondition columns per thread. The
candidate retained 32 blocks per row, reduced each block from 256 to 128
threads, loaded the recovery workspace as aligned `float2`, formed the first
FP16 butterfly directly as `half2`, kept five pair-index stages inside each
warp, and left only the last two low stages for shared memory.

The candidate was byte-exact in all 36 focused eager and CUDA Graph cases,
but it regressed every M value. All candidate CUDA, Python, schema, runtime,
and test code was removed; production remains Phase 64.

## Exact mapping

For adjacent columns `2p` and `2p+1`, the selected recovery tree evaluated
the same nodes and applied overflow-preserving rounding independently to both
lanes. The resulting FP16 down-precondition seeds were packed as:

```text
pair = [seed(2p), seed(2p+1)]
stage_bit_1 = [R(pair.low + pair.high), R(pair.low - pair.high)]
```

Pair-index bits 1, 2, 4, 8, and 16 then used full-warp XOR exchange. Pair
bits 32 and 64 used the existing shared-memory butterfly. No arithmetic was
reassociated and no new rounding boundary was introduced.

## Same-binary H100 result

Timing used 30 warmups, 300 CUDA-event samples, and 50 warmed CUDA Graph
replays per sample after three spaced 0% utilization / 0 MiB H100 admission
samples. The Phase-64 control and candidate were compiled into one extension.
M16 control uses the accepted paired-output-tile path; smaller M uses the
Phase-63 middle kernel selected by Phase 64.

| M | M/K/N gate/up x2 | Phase 64 | Adjacent columns | vs Phase 64 | Better |
|--:|:--|--:|--:|--:|:--:|
| 1 | 1/2048/8192 x2 | 7.759 us | 8.043 us | 0.9647x | No |
| 2 | 2/2048/8192 x2 | 7.882 us | 8.235 us | 0.9572x | No |
| 4 | 4/2048/8192 x2 | 8.006 us | 8.468 us | 0.9455x | No |
| 8 | 8/2048/8192 x2 | 8.771 us | 9.018 us | 0.9726x | No |
| 16 | 16/2048/8192 x2 | 9.878 us | 10.487 us | 0.9419x | No |

CUDA `cuobjdump` reports **72 registers/thread**, zero stack bytes, and zero
local bytes for the candidate. The accepted scalar-column Phase-63 kernel
uses 42 registers/thread, while the accepted Phase-64 paired-output kernel
uses 40. Holding two full 32-value FP32 recovery trees increases live state
enough that the reduced warp count and register footprint outweigh the
aligned load and low-transform instruction savings.

## Decision

- No Phase-65 candidate remains in production source.
- Quantization math, checkpoint bytes, operation workspace, persistent VRAM,
  and dispatch are unchanged.
- Adjacent-column vectorization is closed unless the recovery tree can first
  be streamed or shortened enough to avoid holding 64 FP32 values per thread.
- Compilation used at most four Ninja jobs, one NVCC host thread, and one
  CUDA split-compile partition.
