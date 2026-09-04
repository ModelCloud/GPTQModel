# Phase 81: native H100 Qwen linear recovery

Phase 81 extends the exact native Qwen composite output recovery to the two
linear-input projections in the pinned `Qwen/Qwen3.8-27B` configuration:

```text
in_proj_qkv: 5120 -> 10240
in_proj_z:   5120 ->  6144
```

It also removes a redundant shared-memory handoff from every native Qwen
composite recovery, including the existing 5120-wide multilayer-perceptron
down projection. The production path remains restricted to the measured
physical NVIDIA H100 and batch rows 1 through 16.

## Exact transform math

The two new output widths use the canonical factorizations selected by the
existing Torch oracle:

$$
10240 = 40 \times 256,
\qquad
6144 = 12 \times 512.
$$

For $N=B L$, the transform is the same Kronecker-factor execution as the
existing implementation:

$$
H_N = H_B \otimes H_L.
$$

Define the established overflow-preserving FP16 boundary

$$
R(v)=
\begin{cases}
\operatorname{FP16}(v), & \operatorname{FP16}(v)\text{ is finite},\\
v, & \text{otherwise}.
\end{cases}
$$

For each logical row, the native kernel performs, in order:

1. $R(x)$ followed by $R(x / \operatorname{FP16}(\sqrt{N}))$;
2. every ascending $H_L$ butterfly, applying $R$ after each add/subtract;
3. the $B\times B$ base multiply in its original source order with FP32
   fused multiply-add, followed by $R$;
4. multiplication by the FP16-rounded output scale, optional addition of the
   FP16-rounded bias, and the final FP16 store.

This is exactly the historical finite path while retaining the FP32 rescue
for intermediate overflow. No butterfly, scale, bias, or rounding boundary
moves.

## Removed work

Previously, CUDA Graph capture had to retain the generic composite path:

```text
FP16 low transform -> Torch base matmul -> scale/bias
FP32 overflow-rescue transform ----------------------> select
```

Each child now uses one native kernel launch. Inside that kernel, the old
base stage wrote its rounded result to a second shared-memory array, executed
a block barrier, and had the same thread reload the same column for the
scale/bias epilogue. Because column ownership does not change, Phase 81 keeps
that rounded value in a register. Shared memory falls from $2N$ FP32 values
to $N$, and one full block barrier plus the high-buffer store/load disappear.

## Correctness and graph safety

- Native 5120, 6144, and 10240 recovery is bit-exact to the Torch oracle for
  M1 and M16, with and without bias.
- The 5120 overflow-cancellation tests remain finite and bit-exact.
- Ordered split-17 and split-34 down recovery remains bit-exact.
- The full grouped linear-input site is bit-exact across five CUDA Graph
  replays for W2, W2.5, and W3.
- Maximum linear-input error against the dense-P32 FP32 Torch oracle is
  `1.774e-5`; maximum mean absolute error is `2.479e-6`.
- Maximum complete-MLP error is `4.838e-8`; maximum mean absolute error is
  `1.544e-8`.

The selection is observable through
`h100_qwen_linear_composite_recovery_launches`.

## H100 linear-input benchmark

Physical H100 UUID `GPU-f5ea03cf-efa4-9807-7de5-b174957a1348`, CUDA Graph
replay timed with CUDA events, 20 warmups, 60 samples, and 50 replays per
sample. `last/new` compares with committed Phase 80. Marlin and Machete are
figurative W4 projection-sum baselines; ratios below one mean the W4 baseline
is faster.

| W | M | MKN | new us | Marlin/new | Machete/new | last/new | better? |
|--:|--:|:--|--:|--:|--:|--:|:--:|
| 2 | 1 | 1x5120x10240 + 1x5120x6144 | 122.526 | 0.1902x | 0.4065x | 1.2714x | yes |
| 2 | 2 | 2x5120x10240 + 2x5120x6144 | 124.140 | 0.1943x | 0.3965x | 1.3071x | yes |
| 2 | 4 | 4x5120x10240 + 4x5120x6144 | 124.364 | 0.1955x | 0.3950x | 1.3537x | yes |
| 2 | 8 | 8x5120x10240 + 8x5120x6144 | 124.470 | 0.1864x | 0.3942x | 1.4585x | yes |
| 2 | 16 | 16x5120x10240 + 16x5120x6144 | 124.459 | 0.2167x | 0.3969x | 1.5508x | yes |
| 2.5 | 1 | 1x5120x10240 + 1x5120x6144 | 122.838 | 0.1897x | 0.4055x | 1.2690x | yes |
| 2.5 | 2 | 2x5120x10240 + 2x5120x6144 | 124.752 | 0.1934x | 0.3945x | 1.3058x | yes |
| 2.5 | 4 | 4x5120x10240 + 4x5120x6144 | 125.062 | 0.1944x | 0.3928x | 1.3535x | yes |
| 2.5 | 8 | 8x5120x10240 + 8x5120x6144 | 124.937 | 0.1857x | 0.3928x | 1.4584x | yes |
| 2.5 | 16 | 16x5120x10240 + 16x5120x6144 | 124.812 | 0.2161x | 0.3958x | 1.5511x | yes |
| 3 | 1 | 1x5120x10240 + 1x5120x6144 | 118.292 | 0.1970x | 0.4211x | 1.2835x | yes |
| 3 | 2 | 2x5120x10240 + 2x5120x6144 | 120.652 | 0.2000x | 0.4079x | 1.3209x | yes |
| 3 | 4 | 4x5120x10240 + 4x5120x6144 | 121.060 | 0.2008x | 0.4058x | 1.3665x | yes |
| 3 | 8 | 8x5120x10240 + 8x5120x6144 | 120.881 | 0.1919x | 0.4059x | 1.4733x | yes |
| 3 | 16 | 16x5120x10240 + 16x5120x6144 | 120.991 | 0.2230x | 0.4083x | 1.5665x | yes |

The geometric speedup is **1.3889x over Phase 80**, with 15 of 15 wins.

## Complete MLP benchmark

The shared-memory/barrier removal also improves the existing 5120-wide down
recovery. `last/new` compares W2/W2.5 with Phase 79 and W3 with Phase 77, the
latest committed MLP baselines before this executable change.

| W | M | MKN | new us | Marlin/new | Machete/new | last/new | better? |
|--:|--:|:--|--:|--:|--:|--:|:--:|
| 2 | 1 | 1x5120x17408 x2 + 1x17408x5120 | 179.853 | 0.3191x | 0.6254x | 1.0111x | yes |
| 2 | 2 | 2x5120x17408 x2 + 2x17408x5120 | 181.313 | 0.3229x | 0.6113x | 1.0103x | yes |
| 2 | 4 | 4x5120x17408 x2 + 4x17408x5120 | 181.909 | 0.3224x | 0.6095x | 1.0101x | yes |
| 2 | 8 | 8x5120x17408 x2 + 8x17408x5120 | 182.284 | 0.3169x | 0.6081x | 1.0076x | yes |
| 2 | 16 | 16x5120x17408 x2 + 16x17408x5120 | 182.168 | 0.3698x | 0.6096x | 1.0103x | yes |
| 2.5 | 1 | 1x5120x17408 x2 + 1x17408x5120 | 180.665 | 0.3177x | 0.6226x | 1.0130x | yes |
| 2.5 | 2 | 2x5120x17408 x2 + 2x17408x5120 | 182.786 | 0.3203x | 0.6063x | 1.0071x | yes |
| 2.5 | 4 | 4x5120x17408 x2 + 4x17408x5120 | 183.027 | 0.3204x | 0.6057x | 1.0080x | yes |
| 2.5 | 8 | 8x5120x17408 x2 + 8x17408x5120 | 182.796 | 0.3160x | 0.6064x | 1.0111x | yes |
| 2.5 | 16 | 16x5120x17408 x2 + 16x17408x5120 | 183.286 | 0.3675x | 0.6059x | 1.0110x | yes |
| 3 | 1 | 1x5120x17408 x2 + 1x17408x5120 | 183.715 | 0.3124x | 0.6123x | 1.0107x | yes |
| 3 | 2 | 2x5120x17408 x2 + 2x17408x5120 | 185.689 | 0.3153x | 0.5969x | 1.0109x | yes |
| 3 | 4 | 4x5120x17408 x2 + 4x17408x5120 | 186.159 | 0.3150x | 0.5955x | 1.0130x | yes |
| 3 | 8 | 8x5120x17408 x2 + 8x17408x5120 | 186.449 | 0.3098x | 0.5945x | 1.0121x | yes |
| 3 | 16 | 16x5120x17408 x2 + 16x17408x5120 | 187.099 | 0.3600x | 0.5935x | 1.0081x | yes |

The complete-MLP geometric speedup is **1.0103x**, with 15 of 15 wins.
Raw distilled results are in
`artifacts/a41_phase81_h100/qwen38_27b_native_linear_recovery.json`.

## Next target

The new nearly flat M1--M16 curve confirms that the previous growth came from
generic transform/recovery orchestration, not the grouped inner kernel. The
linear-input site remains only 0.39--0.42x the Machete W4 projection sum, so
the next profile should split the remaining approximately 120 microseconds
between shared input transform, grouped decode, and the two native recoveries.
The next production change should delete a full materialization or merge the
two child recovery launches rather than micro-tune one butterfly.
