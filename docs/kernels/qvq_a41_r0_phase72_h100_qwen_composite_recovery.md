# Phase 72: native H100 Qwen composite recovery

Phase 72 replaces Qwen3.8-27B's graph-time dual composite output transform
with one exact native H100 kernel.  It improves all 15 W2--W3 by M1--M16
complete-MLP cells and lowers their geometric latency by 12.0 percent.

## Exact transform

The canonical factor selection for width 5120 is

$$
H_{5120}=H_{40}\otimes H_{128}.
$$

The old CUDA Graph path evaluated both of these complete branches because a
captured graph cannot make the Python finite-result decision:

```text
FP32 inner -> FP16 historical composite transform --+
                                                 finite? -> select
FP32 inner -> FP32 overflow-rescue transform -------+
```

The native kernel keeps one FP32 shared representation and applies the same
stage-local rule already used by the exact power-of-two recovery kernel:

$$
R(v)=
\begin{cases}
\operatorname{FP16}(v), & \operatorname{FP16}(v)\text{ is finite},\\
v, & \text{otherwise}.
\end{cases}
$$

It executes, in order:

1. the historical FP16 input boundary and division by
   `FP16(sqrt(5120))`;
2. seven ascending `H128` butterfly stages, applying `R` after each add/sub;
3. the canonical 40 by 40 base transform in FP32, followed by `R`;
4. historical FP16 `SV` multiplication and optional bias addition;
5. the final FP16 output store.

Ordinary finite values preserve every historical FP16 boundary.  An
overflowing intermediate remains FP32 only until cancellation or scaling
makes it representable again.  Tests cover finite random inputs and explicit
`+/-70000` cancellation cases, with exact FP16-bit equality to the previous
fallback.

Promotion is restricted to physical NVIDIA H100, FP32 inner output, output
Hadamard enabled, and the exact Qwen down geometry `17408 -> 5120`.
`h100_qwen_composite_down_recovery_launches` makes selection observable.
There is no checkpoint, persistent VRAM, or transient workspace increase.

## Correctness and graph safety

- Native recovery is bit-exact at M1 and M16, with and without bias.
- Explicit overflow-cancellation cases are finite and bit-exact.
- The complete Qwen MLP remains exact across eager execution and CUDA Graph
  replay.
- Mean absolute error is `1.503e-8` to `1.544e-8` against the same-payload
  dense-P32 Torch oracle; maximum error is `4.838e-8`.

## H100 benchmark

Physical H100 UUID `GPU-f5ea03cf-efa4-9807-7de5-b174957a1348`, CUDA Graph
replay timed with CUDA events, 20 warmups, 60 samples, and 50 replays/sample.
`last/new` compares with Phase 71.  Marlin and Machete are figurative W4
projection-sum baselines; ratios below one mean the W4 baseline is faster.

| W | M | MKN | new us | Marlin/new | Machete/new | last/new | better? |
|--:|--:|:--|--:|--:|--:|--:|:--:|
| 2 | 1 | 1x5120x17408 x2 + 1x17408x5120 | 201.799 | 0.284x | 0.557x | 1.1010x | yes |
| 2 | 2 | 2x5120x17408 x2 + 2x17408x5120 | 203.589 | 0.288x | 0.544x | 1.1092x | yes |
| 2 | 4 | 4x5120x17408 x2 + 4x17408x5120 | 204.748 | 0.286x | 0.541x | 1.1201x | yes |
| 2 | 8 | 8x5120x17408 x2 + 8x17408x5120 | 208.587 | 0.277x | 0.531x | 1.1511x | yes |
| 2 | 16 | 16x5120x17408 x2 + 16x17408x5120 | 210.692 | 0.320x | 0.527x | 1.1992x | yes |
| 2.5 | 1 | 1x5120x17408 x2 + 1x17408x5120 | 200.185 | 0.287x | 0.562x | 1.1062x | yes |
| 2.5 | 2 | 2x5120x17408 x2 + 2x17408x5120 | 201.769 | 0.290x | 0.549x | 1.1143x | yes |
| 2.5 | 4 | 4x5120x17408 x2 + 4x17408x5120 | 202.927 | 0.289x | 0.546x | 1.1255x | yes |
| 2.5 | 8 | 8x5120x17408 x2 + 8x17408x5120 | 207.281 | 0.279x | 0.535x | 1.1529x | yes |
| 2.5 | 16 | 16x5120x17408 x2 + 16x17408x5120 | 210.214 | 0.320x | 0.528x | 1.1955x | yes |
| 3 | 1 | 1x5120x17408 x2 + 1x17408x5120 | 204.626 | 0.280x | 0.550x | 1.1029x | yes |
| 3 | 2 | 2x5120x17408 x2 + 2x17408x5120 | 206.696 | 0.283x | 0.536x | 1.1088x | yes |
| 3 | 4 | 4x5120x17408 x2 + 4x17408x5120 | 207.598 | 0.282x | 0.534x | 1.1203x | yes |
| 3 | 8 | 8x5120x17408 x2 + 8x17408x5120 | 211.729 | 0.273x | 0.524x | 1.1516x | yes |
| 3 | 16 | 16x5120x17408 x2 + 16x17408x5120 | 214.699 | 0.314x | 0.517x | 1.1936x | yes |

The geometric speedup is **1.1363x over Phase 71**, with 15 of 15 wins, and
**1.1413x over Phase 70**.  Raw distilled result:
`artifacts/a41_phase72_h100/qwen38_27b_native_composite_recovery.json`.

## Next target

The remaining MLP time is dominated by grouped gate/up P32 decode and the
Qwen down P32 decode.  The down launch still uses unordered split-K atomics.
The next experiment should sweep deterministic child-local split schedules
and fuse the winning ordered reduction directly into this composite recovery
kernel, avoiding the reduced FP32 output materialization.
