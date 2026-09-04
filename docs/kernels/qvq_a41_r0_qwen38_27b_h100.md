# Qwen3.8-27B A41/R0 on H100

This note records the Qwen3.8-27B extension of PR #98.  The source geometry is
the official [`Qwen/Qwen3.8-27B` configuration](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/config.json)
at revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.

## Architecture contract

The Qwen model definitions, rather than the generic P32 runtime, declare the
three shared-input groups:

| Group | Child output widths | Shared input width |
|:--|:--|--:|
| full-attention input | `q_proj=12288`, `k_proj=1024`, `v_proj=1024` | 5120 |
| linear-attention input | `in_proj_qkv=10240`, `in_proj_z=6144` | 5120 |
| MLP input | `gate_proj=17408`, `up_proj=17408` | 5120 |

The remaining large projections are `o_proj: 6144 -> 5120` and
`down_proj: 17408 -> 5120`.  A declared sibling group receives one
layer-local input-sign seed during quantization, so its children have
bit-identical `SU`.  The runtime still checks the complete R0 legality
contract and falls back rather than grouping incompatible payloads.

Qwen's 17,408-wide MLP dimension is `17 * 1024`; there is no supported exact
17-point Hadamard base.  The architecture therefore quantizes gate/up without
their module-local output Hadamards and down without its module-local input
Hadamard.  This does **not** move a linear transform through SiLU.  Each linear
is quantized under its declared axes, and inference preserves the original
FP16 gate/up recovery, SiLU/product, and down-input scaling order.

The axis policy is stored in versioned quantization metadata and restored on
load.  Canonical checkpoint P32 payloads are unchanged.  Grouped continuous
windows remain a transient, lossless runtime representation and do not add
persistent checkpoint storage.

## Composite rotation math

For Qwen widths such as 5120 (`20 * 256`), 6144 (`12 * 512`), 10240
(`20 * 512`), and 12288 (`12 * 1024`), the reference implementation repeatedly
applies ascending pairwise butterflies until the active dimension is the
small base `K`:

```text
X [B, K * P]
  -> exact ascending power-of-two butterflies
L [B, K, P]
  -> H_K @ L[:, :, p] for every p
Y [B, K * P]
```

Here `P` is a power of two and `K` is 12 or 20.  The optimized implementation
replaces only the first arrow with one native CUDA launch in unnormalized
mode.  It retains the same butterfly order and element dtype, the same small
`H_K` PyTorch matmul, and the same normalization location.  Consequently it
is bit-exact to the previous Torch implementation for FP16, BF16, and FP32.

## H100 child schedules

The grouped launch preserves child-local reduction order and uses schedules
measured on the physical H100:

| Rate | full Q/K/V | linear inputs | gate/up |
|--:|:--:|:--:|:--:|
| W2 | `(10,20,20)` | `(10,20)` | `(10,10)` |
| W2.5 | `(10,20,20)` | `(10,20)` | `(10,10)` |
| W3 | `(10,20,20)` | `(4,20)` | `(10,10)` |
| W3.5 | `(4,20,20)` | `(4,4)` | `(5,5)` |

Each tuple is the ordered split-K count for the children listed in the
architecture table.  There is no split derived from concatenated output
width, and every child is reduced left-to-right in its own FP32 partial planes.

## CUDA Graph safety

Payload construction, canonical tensor comparisons, CPU-to-CUDA constant
copies, and cache destruction are forbidden during capture.  An eager warmup
builds and validates grouped payloads and device/dtype-specific composite
Hadamard bases.  A cold group encountered during capture fails closed to the
ordinary graph-safe children.  Tests cover warmed grouped full attention,
the folded full MLP, cold fallback, and composite transforms at every Qwen
width; captured replay is bit-exact to eager execution.

## H100 benchmark

Timing uses CUDA Graph replay timed by CUDA events on the physical
`NVIDIA H100` (`GPU-f5ea03cf-efa4-9807-7de5-b174957a1348`): 20 warmups, 60
samples, and 50 replays per sample.  `Marlin/QVQ` and `Machete/QVQ` are W4
baseline latency divided by QVQ latency; values below 1 mean the W4 baseline
is still faster.  `last/QVQ` compares with the previous Qwen grouped benchmark
before the composite rotation collapse.  Effective TFLOP/s counts the dense
logical matrix multiply work for every member; it is useful-work throughput,
not a claim about tensor-core issue rate.  Marlin/Machete site values are sums
of separately measured child kernels; the QVQ MLP additionally includes the
FP16 boundary, SiLU/product, scaling, rotations, and runtime coordination.

Aggregate geometric means:

| Site | vs previous Qwen group | wins | vs Marlin W4 | vs Machete W4 |
|:--|--:|--:|--:|--:|
| full Q/K/V | **2.417x** | 20/20 | 0.427x | 0.504x |
| linear inputs | **2.575x** | 20/20 | 0.143x | 0.289x |
| full MLP | **1.539x** | 20/20 | 0.233x | 0.434x |

Across all 60 cells, the new Qwen composite/grouped path improves over the
last benchmark.  No cell regresses.  The largest error against the canonical
dense-P32 Torch oracle is `1.774e-5`; the corresponding mean absolute errors
are approximately `2.3e-6` for attention groups and `1.5e-8` for the full MLP.

### Full Q/K/V

| W | M | MxKxN member shapes | QVQ us | eff. TFLOP/s | Marlin/QVQ | Machete/QVQ | last/QVQ | better? |
|-:|-:|:--|-:|-:|-:|-:|-:|:--:|
| 2 | 1 | 1x5120x12288 + 1x5120x1024 + 1x5120x1024 | 116.288 | 1.262 | 0.443x | 0.566x | 2.515x | yes |
| 2 | 2 | 2x5120x12288 + 2x5120x1024 + 2x5120x1024 | 123.348 | 2.380 | 0.461x | 0.521x | 2.442x | yes |
| 2 | 4 | 4x5120x12288 + 4x5120x1024 + 4x5120x1024 | 126.982 | 4.624 | 0.451x | 0.503x | 2.439x | yes |
| 2 | 8 | 8x5120x12288 + 8x5120x1024 + 8x5120x1024 | 135.739 | 8.652 | 0.378x | 0.472x | 2.359x | yes |
| 2 | 16 | 16x5120x12288 + 16x5120x1024 + 16x5120x1024 | 136.743 | 17.177 | 0.412x | 0.467x | 2.351x | yes |
| 2.5 | 1 | 1x5120x12288 + 1x5120x1024 + 1x5120x1024 | 117.637 | 1.248 | 0.438x | 0.560x | 2.481x | yes |
| 2.5 | 2 | 2x5120x12288 + 2x5120x1024 + 2x5120x1024 | 123.800 | 2.372 | 0.460x | 0.519x | 2.438x | yes |
| 2.5 | 4 | 4x5120x12288 + 4x5120x1024 + 4x5120x1024 | 128.142 | 4.582 | 0.447x | 0.499x | 2.424x | yes |
| 2.5 | 8 | 8x5120x12288 + 8x5120x1024 + 8x5120x1024 | 136.677 | 8.593 | 0.376x | 0.469x | 2.355x | yes |
| 2.5 | 16 | 16x5120x12288 + 16x5120x1024 + 16x5120x1024 | 137.492 | 17.083 | 0.410x | 0.464x | 2.337x | yes |
| 3 | 1 | 1x5120x12288 + 1x5120x1024 + 1x5120x1024 | 119.421 | 1.229 | 0.431x | 0.551x | 2.466x | yes |
| 3 | 2 | 2x5120x12288 + 2x5120x1024 + 2x5120x1024 | 125.699 | 2.336 | 0.453x | 0.511x | 2.420x | yes |
| 3 | 4 | 4x5120x12288 + 4x5120x1024 + 4x5120x1024 | 129.921 | 4.520 | 0.440x | 0.492x | 2.409x | yes |
| 3 | 8 | 8x5120x12288 + 8x5120x1024 + 8x5120x1024 | 138.481 | 8.481 | 0.371x | 0.463x | 2.340x | yes |
| 3 | 16 | 16x5120x12288 + 16x5120x1024 + 16x5120x1024 | 138.917 | 16.908 | 0.405x | 0.460x | 2.329x | yes |
| 3.5 | 1 | 1x5120x12288 + 1x5120x1024 + 1x5120x1024 | 113.918 | 1.289 | 0.452x | 0.578x | 2.536x | yes |
| 3.5 | 2 | 2x5120x12288 + 2x5120x1024 + 2x5120x1024 | 120.441 | 2.438 | 0.473x | 0.534x | 2.478x | yes |
| 3.5 | 4 | 4x5120x12288 + 4x5120x1024 + 4x5120x1024 | 124.599 | 4.713 | 0.459x | 0.513x | 2.467x | yes |
| 3.5 | 8 | 8x5120x12288 + 8x5120x1024 + 8x5120x1024 | 132.750 | 8.847 | 0.387x | 0.483x | 2.398x | yes |
| 3.5 | 16 | 16x5120x12288 + 16x5120x1024 + 16x5120x1024 | 133.818 | 17.552 | 0.421x | 0.477x | 2.379x | yes |

### Linear-attention inputs

| W | M | MxKxN member shapes | QVQ us | eff. TFLOP/s | Marlin/QVQ | Machete/QVQ | last/QVQ | better? |
|-:|-:|:--|-:|-:|-:|-:|-:|:--:|
| 2 | 1 | 1x5120x10240 + 1x5120x6144 | 157.236 | 1.067 | 0.148x | 0.317x | 2.666x | yes |
| 2 | 2 | 2x5120x10240 + 2x5120x6144 | 163.736 | 2.049 | 0.147x | 0.301x | 2.625x | yes |
| 2 | 4 | 4x5120x10240 + 4x5120x6144 | 170.267 | 3.941 | 0.143x | 0.289x | 2.597x | yes |
| 2 | 8 | 8x5120x10240 + 8x5120x6144 | 185.213 | 7.247 | 0.125x | 0.265x | 2.473x | yes |
| 2 | 16 | 16x5120x10240 + 16x5120x6144 | 195.717 | 13.716 | 0.138x | 0.252x | 2.390x | yes |
| 2.5 | 1 | 1x5120x10240 + 1x5120x6144 | 157.970 | 1.062 | 0.148x | 0.315x | 2.660x | yes |
| 2.5 | 2 | 2x5120x10240 + 2x5120x6144 | 164.828 | 2.036 | 0.146x | 0.299x | 2.609x | yes |
| 2.5 | 4 | 4x5120x10240 + 4x5120x6144 | 170.924 | 3.926 | 0.142x | 0.287x | 2.592x | yes |
| 2.5 | 8 | 8x5120x10240 + 8x5120x6144 | 185.460 | 7.237 | 0.125x | 0.265x | 2.475x | yes |
| 2.5 | 16 | 16x5120x10240 + 16x5120x6144 | 196.813 | 13.639 | 0.137x | 0.251x | 2.360x | yes |
| 3 | 1 | 1x5120x10240 + 1x5120x6144 | 153.769 | 1.091 | 0.152x | 0.324x | 2.704x | yes |
| 3 | 2 | 2x5120x10240 + 2x5120x6144 | 160.076 | 2.096 | 0.151x | 0.307x | 2.663x | yes |
| 3 | 4 | 4x5120x10240 + 4x5120x6144 | 166.634 | 4.027 | 0.146x | 0.295x | 2.635x | yes |
| 3 | 8 | 8x5120x10240 + 8x5120x6144 | 181.375 | 7.400 | 0.128x | 0.271x | 2.507x | yes |
| 3 | 16 | 16x5120x10240 + 16x5120x6144 | 192.053 | 13.977 | 0.140x | 0.257x | 2.415x | yes |
| 3.5 | 1 | 1x5120x10240 + 1x5120x6144 | 148.182 | 1.132 | 0.157x | 0.336x | 2.769x | yes |
| 3.5 | 2 | 2x5120x10240 + 2x5120x6144 | 154.520 | 2.172 | 0.156x | 0.319x | 2.722x | yes |
| 3.5 | 4 | 4x5120x10240 + 4x5120x6144 | 161.186 | 4.163 | 0.151x | 0.305x | 2.689x | yes |
| 3.5 | 8 | 8x5120x10240 + 8x5120x6144 | 175.756 | 7.637 | 0.132x | 0.279x | 2.555x | yes |
| 3.5 | 16 | 16x5120x10240 + 16x5120x6144 | 186.924 | 14.361 | 0.144x | 0.264x | 2.453x | yes |

### Full MLP

| W | M | MxKxN member shapes | QVQ us | eff. TFLOP/s | Marlin/QVQ | Machete/QVQ | last/QVQ | better? |
|-:|-:|:--|-:|-:|-:|-:|-:|:--:|
| 2 | 1 | 1x5120x17408 + 1x5120x17408 + 1x17408x5120 | 244.493 | 2.187 | 0.235x | 0.460x | 1.556x | yes |
| 2 | 2 | 2x5120x17408 + 2x5120x17408 + 2x17408x5120 | 250.586 | 4.268 | 0.234x | 0.442x | 1.554x | yes |
| 2 | 4 | 4x5120x17408 + 4x5120x17408 + 4x17408x5120 | 253.985 | 8.422 | 0.231x | 0.437x | 1.551x | yes |
| 2 | 8 | 8x5120x17408 + 8x5120x17408 + 8x17408x5120 | 262.998 | 16.267 | 0.220x | 0.421x | 1.529x | yes |
| 2 | 16 | 16x5120x17408 + 16x5120x17408 + 16x17408x5120 | 270.007 | 31.690 | 0.249x | 0.411x | 1.502x | yes |
| 2.5 | 1 | 1x5120x17408 + 1x5120x17408 + 1x17408x5120 | 244.521 | 2.187 | 0.235x | 0.460x | 1.558x | yes |
| 2.5 | 2 | 2x5120x17408 + 2x5120x17408 + 2x17408x5120 | 250.233 | 4.274 | 0.234x | 0.443x | 1.558x | yes |
| 2.5 | 4 | 4x5120x17408 + 4x5120x17408 + 4x17408x5120 | 254.025 | 8.421 | 0.231x | 0.436x | 1.553x | yes |
| 2.5 | 8 | 8x5120x17408 + 8x5120x17408 + 8x17408x5120 | 263.015 | 16.266 | 0.220x | 0.421x | 1.527x | yes |
| 2.5 | 16 | 16x5120x17408 + 16x5120x17408 + 16x17408x5120 | 270.414 | 31.642 | 0.249x | 0.411x | 1.499x | yes |
| 3 | 1 | 1x5120x17408 + 1x5120x17408 + 1x17408x5120 | 252.475 | 2.118 | 0.227x | 0.446x | 1.541x | yes |
| 3 | 2 | 2x5120x17408 + 2x5120x17408 + 2x17408x5120 | 258.750 | 4.134 | 0.226x | 0.428x | 1.538x | yes |
| 3 | 4 | 4x5120x17408 + 4x5120x17408 + 4x17408x5120 | 262.166 | 8.159 | 0.224x | 0.423x | 1.536x | yes |
| 3 | 8 | 8x5120x17408 + 8x5120x17408 + 8x17408x5120 | 271.302 | 15.769 | 0.213x | 0.409x | 1.514x | yes |
| 3 | 16 | 16x5120x17408 + 16x5120x17408 + 16x17408x5120 | 278.415 | 30.732 | 0.242x | 0.399x | 1.485x | yes |
| 3.5 | 1 | 1x5120x17408 + 1x5120x17408 + 1x17408x5120 | 236.361 | 2.263 | 0.243x | 0.476x | 1.580x | yes |
| 3.5 | 2 | 2x5120x17408 + 2x5120x17408 + 2x17408x5120 | 242.482 | 4.411 | 0.241x | 0.457x | 1.577x | yes |
| 3.5 | 4 | 4x5120x17408 + 4x5120x17408 + 4x17408x5120 | 246.020 | 8.695 | 0.238x | 0.451x | 1.570x | yes |
| 3.5 | 8 | 8x5120x17408 + 8x5120x17408 + 8x17408x5120 | 255.259 | 16.760 | 0.226x | 0.434x | 1.544x | yes |
| 3.5 | 16 | 16x5120x17408 + 16x5120x17408 + 16x17408x5120 | 262.812 | 32.557 | 0.256x | 0.423x | 1.511x | yes |

These results establish successful activation and a large improvement over the
previous Qwen path.  They do not establish parity with W4 Marlin or Machete:
the baseline ratios remain below 1, especially for linear attention and MLP.
