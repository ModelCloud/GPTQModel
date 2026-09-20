# Qwen3.8-Flash-Next P32 inference on H100: model projection scorecard

This phase validates the combined attention and expert work from phases 1 and
2 against the exact Qwen3.8-Flash-Next text-model topology. It answers the
model-level kernel question that isolated projection wins cannot: does the
complete weighted P32 projection stack remain above 2x at every supported rate
and decode row count?

The answer is yes. All 100 exact site/rate/M cells exceed 2x, and all 20
topology-weighted scorecards exceed 3.3x.

## Scope and weighting

The local checkpoint config has 48 text layers: 12 full-attention and 36
linear-attention layers. Every layer selects 10 routed experts and executes one
shared expert. The scorecard therefore weights measured site medians as:

- 12 full Q/K/V groups at widths `(12288, 512, 512)` from K=2560;
- 36 linear QKV/Z groups at widths `(10240, 6144)` from K=2560;
- 48 attention output projections at `K=6144, N=2560`;
- 528 gate/up expert groups at `K=2560, N=(640, 640)`;
- 528 expert down projections at `K=640, N=2560`.

The baseline is the former planar `qvq_cuda` V2B2-P32 projection path. The
candidate is the production exact contiguous-window dispatch. This is a
projection-kernel scorecard, not a full-model tokens/second claim: attention
cores, routing, activations, norms, communication, vision, and MTP are excluded.

## Complete result

The benchmark uses candidate/control/candidate CUDA Graph timing on an idle
132-SM H100, with 30 warmups, 300 samples, and 50 replays per sample.

| Rate | M1 | M2 | M4 | M8 | M16 |
|---|---:|---:|---:|---:|---:|
| W2 | 3.357x | 4.245x | 4.084x | 3.685x | 12.386x |
| W2.5 | 7.127x | 13.862x | 13.002x | 12.542x | 12.378x |
| W3 | 12.959x | 13.742x | 12.967x | 12.671x | 12.533x |
| W3.5 | 12.556x | 13.181x | 13.118x | 12.642x | 12.556x |

The text-model projection-stack geometric mean is **9.801x**. The minimum is
**3.357x** at W2/M1 and the maximum is **13.862x** at W2.5/M2. Full-attention
layers have a **9.143x** geometric mean and linear-attention layers have a
**10.025x** geometric mean.

The attention-only weighted stack has a **23.149x** geometric mean and ranges
from **5.908x** to **33.601x**. Per-site results are:

| Site | Geometric mean | Minimum | Maximum |
|---|---:|---:|---:|
| Full Q/K/V group | 20.033x | 4.665x | 30.912x |
| Linear QKV/Z group | 28.796x | 7.042x | 40.025x |
| Attention output | 17.107x | 5.129x | 29.389x |
| Expert gate/up | 6.769x | 2.918x | 9.549x |
| Expert down | 5.956x | 2.298x | 8.467x |

Thus every constituent projection family independently clears 2x; the weighted
result is not hiding a regressing site behind expert multiplicity.

## Accuracy and dtype gates

Every cell was bitwise repeatable under graph replay. The largest
candidate-versus-planar mean absolute error was `2.624e-6`, and the largest
absolute error was `2.170e-5`, versus gates of `4e-3` and `0.046875`.

The benchmark disables TF32 and samples FP32 and FP64 references over the full
K dimension for every child. The largest sampled FP64 mean errors were
`2.376e-7` for the planar baseline, `3.102e-6` for the contiguous candidate,
and `3.646e-7` for the FP32 anchor. The candidate's changed FP32 reduction
order was farther from FP64 in all 180 child aggregates, but the absolute drift
remains three orders of magnitude below the local acceptance gate. This is
retained under the rounding policy rather than mislabeled as bitwise parity.

The source checkpoint declares BF16 model activations. QVQ intentionally keeps
the model-facing BF16 contract while narrowing the transformed P32 operand to
FP16 for the native kernels. Both FP16 and BF16 full expert-MLP forwards pass
eager execution plus CUDA Graph capture/replay against the unfused path.

## Reproduction

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<H100-UUID> \
  python scripts/benchmark_qvq_p32_qwen38_flash_next_model_h100.py \
  --warmup 30 --samples 300 --replays-per-sample 50 \
  --fp64-columns 64 \
  --output /root/qvq-results/qwen38-flash-next-model-scorecard-production.json
```

The benchmark records a SHA-256 fingerprint over every production CUDA/Python
source that controls the measured routes, so an uncommitted or later-modified
kernel cannot silently inherit these results. The authoritative run records
commit `2ee51a8e0edf1f9909d1c9f86696b2f63ec3a94a` and source fingerprint
`04b97f107e5c8534dd54be43e5f95d34c9a02e2c6a43a6f7f4936dcfe1b4fcd9`.
