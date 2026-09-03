# A41/R0 Phase 21: H100 W3 fused state-to-level lookup

## Decision

Phase 21 promotes no executable change. Two W3 decode experiments were exact,
but neither improved the production kernel:

1. spelling the high-byte extraction as PTX `bfe.u32` produced the same SASS
   as the existing source; and
2. replacing PGC arithmetic and the shared level table with an exact 1 MiB
   state-to-level table reduced instructions and removed every shared-load
   bank conflict, but made the grouped gate/up kernel about four times slower.

All experimental Python and CUDA code was reverted after measurement. The
production source at the end of this phase is byte-identical to `34c84e50`.

## Scope and timing contract

Measurements used only the physical 132-SM NVIDIA H100 at PCI
`00000000:44:00.0`, UUID
`GPU-f5ea03cf-efa4-9807-7de5-b174957a1348`. Formal benchmarks required zero
MiB in use before admission and timed warmed CUDA Graph replays with CUDA
events. Nsight Compute 2026.2.1 used kernel replay and its documented 4-MiB
idle-memory allowance. Builds were limited to four Ninja jobs, one NVCC host
thread, and one split-compile partition.

## Control: explicit high-byte bit extraction

The first experiment replaced the source expression based on `__byte_perm`
with explicit `bfe.u32` PTX. Both sources compiled to the same binary shape:

| Metric | Production | Explicit BFE | Change |
|:--|--:|--:|--:|
| Static SASS instructions | 1,947 | 1,947 | 0 |
| Static `PRMT` | 198 | 198 | 0 |
| Static `BFE`/`UBFE` | 0 | 0 | 0 |
| Executed warp instructions | 10,792,258 | 10,792,258 | 0 |

CUDA-event times were neutral across M1--M16. The compiler had already chosen
the same permutation instruction, so this source spelling cannot reduce the
decoder.

## Exact fused lookup math

For W3, production first applies the selected repeated-byte bank mask and the
16-bit PGC mapping. Only the high mask byte survives the first XOR fold. For
state \(s\), bank row \(r\), and W3 masks

\[
M_r \in \{0,\;0x6900,\;0x5a00,\;0x3c00\},
\]

the experiment precomputed

\[
x_r(s)=s\oplus((s\gg8)\mathbin{\&}255)\oplus M_r,
\]

\[
p_r(s)=(40503x_r(s)+17011)\bmod 2^{16}.
\]

Production's two FP16 level indices are then exactly

\[
i_{high}=b\oplus(b\gg7),\qquad b=(p_r(s)\gg8)\mathbin{\&}255,
\]

and

\[
i_{low}=(p_r(s)\oplus(p_r(s)\gg7))\mathbin{\&}255.
\]

The candidate stored the two resulting FP16 values in one packed 32-bit
entry:

```text
table[4 banks][65536 states][high FP16, low FP16]
```

This is \(4\times65536\times4=1,048,576\) bytes. It does not change the
canonical checkpoint or P32 payload, but it would add 1 MiB of persistent
device storage per distinct codebook/device. Each decoded state used one
32-bit global load; four states supplied the eight WGMMA fragment values.

## Isolated grouped gate/up result

The candidate remained repeatable, CUDA Graph stable, and within
`1.17e-5` maximum absolute error of the dense P32 oracle. It was nevertheless
far slower for both Llama 3.2 1B gate/up matrices, each with K=2048 and
N=8192:

| M | MKN per child | Candidate grouped gate/up + recovery us | Exact | Graph stable |
|---:|:--|---:|:---:|:---:|
| 1 | 1x2048x8192 (x2) | 121.526 | Yes | Yes |
| 2 | 2x2048x8192 (x2) | 121.142 | Yes | Yes |
| 4 | 4x2048x8192 (x2) | 121.638 | Yes | Yes |
| 8 | 8x2048x8192 (x2) | 121.911 | Yes | Yes |
| 16 | 16x2048x8192 (x2) | 122.709 | Yes | Yes |

## Full Llama 3.2 1B MLP benchmark

The table compares W3 QVQ against freshly measured W4 Marlin and Machete on
the same H100. `vs last` is the Phase-19 production latency divided by the
candidate latency; values below 1 are regressions. Effective TFLOP/s uses the
logical dense-equivalent FLOP count and is included only as a common work-rate
normalization.

| Rate | M | MKN (gate/up; down) | Candidate us | Effective TFLOP/s | vs Marlin W4 | vs Machete W4 | vs last | Better than last |
|---:|---:|:--|---:|---:|---:|---:|---:|:---:|
| W3 | 1 | 1x2048x8192 (x2); 1x8192x2048 | 151.179 | 0.666 | 0.196x | 0.336x | 0.408x | No |
| W3 | 2 | 2x2048x8192 (x2); 2x8192x2048 | 150.942 | 1.334 | 0.211x | 0.334x | 0.401x | No |
| W3 | 4 | 4x2048x8192 (x2); 4x8192x2048 | 150.602 | 2.674 | 0.213x | 0.339x | 0.400x | No |
| W3 | 8 | 8x2048x8192 (x2); 8x8192x2048 | 150.662 | 5.345 | 0.199x | 0.335x | 0.401x | No |
| W3 | 16 | 16x2048x8192 (x2); 16x8192x2048 | 151.492 | 10.632 | 0.217x | 0.333x | 0.405x | No |

Geometric means were **0.207x Marlin W4**, **0.335x Machete W4**, and
**0.403x the last production benchmark**. All five cells regressed. The full
MLP result was bit-exact to the unfused QVQ control in every cell.

## Why fewer instructions lost badly

The lookup achieved its local goals but exchanged inexpensive shared reuse
for a scattered dependent cache access in every lane:

| W3 M1 NCU metric | Production | Fused lookup | Change |
|:--|--:|--:|--:|
| Replay duration | 28.192 us | 118.464 us | 4.20x slower |
| Executed warp instructions | 10,792,258 | 7,570,888 | -29.9% |
| Static SASS instructions | 1,947 | 1,472 | -24.4% |
| Shared-load bank conflicts | 1,050,805 | 0 | eliminated |
| L1/LSU wavefronts | 3,350,926 | 11,183,522 | 3.34x more |
| Eligible warps/scheduler/cycle | 0.588 | 0.071 | -87.8% |

The candidate generated 16.76 million global-load L1 sectors. Only 0.389
million hit in L1. Although 96.4% of L2 sectors hit, the scattered,
state-dependent loads still expanded into 16.70 million L2 read sectors and
left only 0.071 eligible warps per scheduler per cycle. Nsight reported 84.2%
memory-system throughput but only 4.7% DRAM throughput: the bottleneck was the
L1/L2 request and dependency path, not HBM capacity.

This demonstrates a stronger constraint than “remove bank conflicts.” A
replacement must preserve broadcast/reuse and issue independence. Turning
eight conflicted shared reads into four lane-random global reads removes
arithmetic but destroys coalescing and latency hiding.

## Artifacts

- `artifacts/a41_phase21_h100/rejected_w3_fused_lookup_gateup.json`
- `artifacts/a41_phase21_h100/rejected_w3_fused_lookup_full_mlp.json`
- `artifacts/a41_phase21_h100/w3_decode_experiment_summary.json`

Binary Nsight reports remain outside Git under
`/root/qvq-profiler-artifacts/phase21-gateup`.

## Next constraint

Phase 21 closes the large state-to-level-table direction for this H100
kernel. The next decoder experiment should keep the 256-entry level table in
shared memory and reduce work before the table address becomes lane-random,
or reuse decoded fragments across more WGMMA work without enlarging the
working set. It should not add per-state global lookup traffic.
