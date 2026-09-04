# Phase 74: native H100 Qwen composite input

Phase 74 replaces Qwen3.8-27B's staged 5120-wide shared input transform and
separate M16 padding with one exact H100 kernel.  The measured promotion is
restricted to M8 and M16; M1, M2, and M4 retain the previous staged path.

## Exact transform

Qwen3.8-27B's canonical 5120-wide factorization is

$$
H_{5120}=H_{40}\otimes H_{128}.
$$

For an input row $x$ and the shared child input scale $S_U$, the kernel
preserves the existing operation order and FP16 boundaries:

$$
t_0=\operatorname{FP16}(x\odot S_U),
$$

$$
t_1=\operatorname{FP16}
\left(t_0/\operatorname{FP16}(\sqrt{5120})\right),
$$

$$
t_2=H_{128}^{\mathrm{ascending\ FP16}}(t_1),
$$

$$
t=\operatorname{FP16}(H_{40}^{\mathrm{FP32\ FMA}}t_2).
$$

The seven H128 butterfly stages round every add and subtract to FP16.  Each
H40 output accumulates its 40 products sequentially in FP32 and rounds once
to FP16.  That is the same arithmetic tree as `matmul_hadU_stable`; the
low-level test requires exact FP16-bit equality for M1, M2, M4, M8, and M16.

The kernel always produces the WGMMA `[16, 5120]` input.  Rows beyond the
logical M are written as exact zero in the same launch, eliminating the old
allocation/fill/copy padding boundary.  It uses 20 KiB of dynamic shared
memory per CTA and is CUDA Graph safe.

## Dispatch policy

The path requires all of the following:

- physical NVIDIA H100, SM90;
- a legal grouped P32 site with shared input Hadamard enabled;
- input width 5120 and canonical H40 base;
- M8 or M16.

The original staged implementation is faster for M1, M2, and M4, so those
rows intentionally retain it.  `h100_qwen_composite_input_launches` exposes
native selection.  The same generic group gate covers QKV, linear-input, and
gate/up sites; there is no projection-role logic in the CUDA operation.

## Correctness and graph safety

- Native output is bit-exact to `matmul_hadU_stable` for all M1--M16 tests.
- Every padded row is exact zero.
- Captured output is bit-exact across CUDA Graph replay.
- Grouped Qwen attention and the complete folded MLP pass their graph tests.
- Across the production benchmarks, maximum absolute error against the
  dense-P32 Torch oracle is `1.774e-5`; the largest mean absolute error is
  `2.479e-6`.  MLP-only maximum error is `4.838e-8`.

## H100 benchmark

Physical H100 UUID `GPU-f5ea03cf-efa4-9807-7de5-b174957a1348`, CUDA Graph
replay timed entirely with CUDA events, 20 warmups, 60 samples, and 50
replays/sample.  `last/new` uses Phase 73 for MLP and the last committed
grouped Qwen site matrix for QKV/linear-input sites.  Marlin and Machete are
figurative W4 projection-sum baselines; a ratio below one means W4 is faster.

Only the M8/M16 rows below select the new kernel.  It wins all 18 affected
W2--W3 cells.

| Site | W | M | MKN | new us | Marlin/new | Machete/new | last/new | better? |
|:--|--:|--:|:--|--:|--:|--:|--:|:--:|
| MLP | 2 | 8 | 8x5120x17408 x2 + 8x17408x5120 | 206.723 | 0.279x | 0.536x | 1.0111x | yes |
| MLP | 2 | 16 | 16x5120x17408 x2 + 16x17408x5120 | 209.902 | 0.321x | 0.529x | 1.0051x | yes |
| MLP | 2.5 | 8 | 8x5120x17408 x2 + 8x17408x5120 | 206.288 | 0.280x | 0.537x | 1.0082x | yes |
| MLP | 2.5 | 16 | 16x5120x17408 x2 + 16x17408x5120 | 209.591 | 0.321x | 0.530x | 1.0067x | yes |
| MLP | 3 | 8 | 8x5120x17408 x2 + 8x17408x5120 | 208.986 | 0.276x | 0.530x | 1.0049x | yes |
| MLP | 3 | 16 | 16x5120x17408 x2 + 16x17408x5120 | 212.136 | 0.318x | 0.523x | 1.0017x | yes |
| QKV | 2 | 8 | 8x5120x12288 + 8x5120x1024 x2 | 133.848 | 0.384x | 0.479x | 1.0141x | yes |
| QKV | 2 | 16 | 16x5120x12288 + 16x5120x1024 x2 | 135.125 | 0.417x | 0.472x | 1.0120x | yes |
| QKV | 2.5 | 8 | 8x5120x12288 + 8x5120x1024 x2 | 134.844 | 0.381x | 0.475x | 1.0136x | yes |
| QKV | 2.5 | 16 | 16x5120x12288 + 16x5120x1024 x2 | 135.472 | 0.416x | 0.471x | 1.0149x | yes |
| QKV | 3 | 8 | 8x5120x12288 + 8x5120x1024 x2 | 136.425 | 0.376x | 0.470x | 1.0151x | yes |
| QKV | 3 | 16 | 16x5120x12288 + 16x5120x1024 x2 | 137.634 | 0.409x | 0.464x | 1.0093x | yes |
| linear inputs | 2 | 8 | 8x5120x10240 + 8x5120x6144 | 183.240 | 0.127x | 0.268x | 1.0108x | yes |
| linear inputs | 2 | 16 | 16x5120x10240 + 16x5120x6144 | 194.284 | 0.139x | 0.254x | 1.0074x | yes |
| linear inputs | 2.5 | 8 | 8x5120x10240 + 8x5120x6144 | 183.435 | 0.126x | 0.268x | 1.0110x | yes |
| linear inputs | 2.5 | 16 | 16x5120x10240 + 16x5120x6144 | 194.665 | 0.139x | 0.254x | 1.0110x | yes |
| linear inputs | 3 | 8 | 8x5120x10240 + 8x5120x6144 | 178.823 | 0.130x | 0.274x | 1.0143x | yes |
| linear inputs | 3 | 16 | 16x5120x10240 + 16x5120x6144 | 190.252 | 0.142x | 0.260x | 1.0095x | yes |

The affected-row geometric speedups are **1.0063x for MLP**, **1.0132x for
QKV**, and **1.0107x for linear-input sites**.  The raw distilled result is
`artifacts/a41_phase74_h100/qwen38_27b_native_composite_input.json`.

## Next target

The grouped gate/up P32 operation remains the dominant MLP component.  The
next experiment should measure legal split counts 5 and 10 for Qwen's
K=5120, N=17408 children before attempting a larger N128 consumer rewrite.
