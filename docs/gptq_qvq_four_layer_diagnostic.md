# GPTQ 2--8-bit versus QVQ four-layer diagnostic

The four-layer GPTQ data below remains valid, but the unmeasured QVQ follow-up has moved to a matched two-layer
interim gate on a CUDA quantization host. Four-layer QVQ is deferred until the reference quantizer is faster. See
[the CUDA handoff and TODOs](qvq_todos.md).

## Outcome

The GPTQ parameter-search boundary has complete 2--8-bit symmetric and adjacent-asymmetric measurements. PGC16 QVQ
rows remain unmeasured, but are no longer blocked on missing quantization math: the repository now has batched Viterbi,
RHT, BlockLDLQ, two-pass tail-biting, planar packing, and a direct four-layer diagnostic hook. The W2--W8 model run and
its KLD/top-k evidence still need to be captured before QVQ rows can be added. Independent nearest-codebook rounding
is not QVQ and remains excluded.

The compact machine-readable GPTQ results are in [gptq_four_layer_2_8_grid.csv](gptq_four_layer_2_8_grid.csv). The
diagnostic JSON additionally records every projection's weight, local-output, live-output, layer-output, and final-logit
metrics. Reproduce it with `scripts/analyze_gptq_low_bit_grid.py` and `--bits 2 3 4 5 6 7 8 --symmetry both`.

## Method

- Model: local Llama 3.2 1B Instruct checkpoint, truncated to its first four decoder layers.
- Scope: 28 projections (`q/k/v/o` and `gate/up/down`), 243,269,632 target weights.
- GPTQ boundary: group size 128, activation-weighted scale search, adjacent integer zero-point search for asymmetric
  arms, fake-quantized dense reconstruction before packing or backend execution.
- Calibration/evaluation: four disjoint prompts each; final logits have shape `[4, 12, 128256]` (48 token positions).
- Intermediate KL: dense-standardized channel softmax. Final KL: raw vocabulary-logit softmax.
- Scheduling: four one-thread workers requested Darwin user-interactive QoS for P-cluster scheduling. macOS does not
  expose a supported hard CPU-affinity API, so QoS is the enforceable scheduling control used by this diagnostic.
- Versions: GPT-QModel `7.3.3+ultra+2e781f4c`, Transformers `5.14.1`, PyTorch `2.14.0.dev20260806`.

This is the same pre-pack parameter-search boundary as the earlier low-bit diagnostic. It isolates the affine grid and
propagated model error; it is not a complete packed GPTQ checkpoint or a backend accuracy test.

## GPTQ results

| Arm | W rel-L2 | Local KL | Live KL | Layer KL | Final KL | Logit RMSE | Logit cosine | Top-1 | Top-5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| W2 symmetric | 0.40844 | 0.123095 | 8.792973 | 31.643363 | 6.568686 | 2.76125 | 0.007531 | 8.33% | 6.67% |
| W2 asymmetric, adjacent | 0.38956 | 0.075950 | 0.286793 | 1.033310 | 1.092850 | 1.59868 | 0.674527 | 56.25% | 47.08% |
| W3 symmetric | 0.21504 | 0.025472 | 0.098959 | 0.255953 | 0.341199 | 0.93949 | 0.886193 | 64.58% | 61.67% |
| W3 asymmetric, adjacent | 0.20338 | 0.020928 | 0.085273 | 0.251026 | 0.280725 | 0.86209 | 0.904293 | 66.67% | 68.33% |
| W4 symmetric | 0.11320 | 0.005995 | 0.023142 | 0.032117 | 0.090392 | 0.47601 | 0.970657 | 72.92% | 80.42% |
| W4 asymmetric, adjacent | 0.10491 | 0.004642 | 0.020741 | 0.025361 | 0.075826 | 0.44070 | 0.974828 | 70.83% | 83.33% |
| W5 symmetric | 0.05866 | 0.001606 | 0.006624 | 0.008740 | 0.028337 | 0.25179 | 0.991759 | 89.58% | 90.42% |
| W5 asymmetric, adjacent | 0.05350 | 0.001177 | 0.005737 | 0.008674 | 0.023387 | 0.23706 | 0.992694 | 95.83% | 90.00% |
| W6 symmetric | 0.03046 | 0.000684 | 0.002545 | 0.003464 | 0.008926 | 0.13950 | 0.997471 | 95.83% | 94.58% |
| W6 asymmetric, adjacent | 0.02771 | 0.000344 | 0.001559 | 0.002091 | 0.007626 | 0.12786 | 0.997873 | 81.25% | 91.67% |
| W7 symmetric | 0.01594 | 0.000129 | 0.000629 | 0.001515 | 0.002637 | 0.07688 | 0.999232 | 93.75% | 94.58% |
| W7 asymmetric, adjacent | 0.01459 | 0.000104 | 0.000468 | 0.000807 | 0.002056 | 0.07095 | 0.999346 | 91.67% | 95.00% |
| W8 symmetric | 0.00843 | 0.000049 | 0.000230 | 0.000312 | 0.000901 | 0.04289 | 0.999761 | 97.92% | 96.67% |
| W8 asymmetric, adjacent | 0.00775 | 0.000036 | 0.000168 | 0.000189 | 0.000728 | 0.04160 | 0.999775 | 97.92% | 98.33% |

The adjacent asymmetric grid improves reconstruction and final KL at every width. Its largest benefit is W2: final KL
falls from 6.568686 to 1.092850 and top-1 rises from 8.33% to 56.25%. W2 asymmetric nevertheless remains substantially
worse than W3 asymmetric (final KL 0.280725, top-1 66.67%), so the zero-point correction prevents catastrophic grid
collapse but does not remove the two-bit capacity gap.

Top-1 is based on only 48 token positions and is visibly noisy at high bit widths: for example, W6 asymmetric has lower
weight error, KL, RMSE, and higher cosine than W6 symmetric, but lower top-1. KL and continuous error metrics are more
stable for ranking these small-error arms; both are retained rather than selecting whichever metric favors one arm.

## Why QVQ rows are still absent

QTIP was introduced by Albert Tseng, Qingyao Sun, David Hou, and Christopher De Sa in
[QTIP: Quantization with Trellises and Incoherence Processing](https://arxiv.org/abs/2406.11235), NeurIPS 2024
Spotlight. The method requires randomized Hadamard incoherence processing, BlockLDLQ error feedback, HYB trellis-coded
rounding, and the paper's two-pass tail-biting approximation.

The four-layer matrices contain 950,272 16-by-16 trellis tiles. The measured rate-aware MPS defaults sustain roughly
193--314 tiles/second depending on rate, including both tail-biting passes. W2 remains the limiting case at about 193
tiles/second, giving a straight-line estimate of roughly 1.37 hours of raw W2 trellis work before BlockLDLQ,
transform, model-forward, and serialization overhead. This is much better than the old scalar-oracle estimate but
still warrants one arm at a time with a committed result after each completed rate.

The direct diagnostic now uses the same dense baseline, prompts, module/layer hooks, and final raw-logit metrics as the
GPTQ arms. It additionally records reverse KL, Jensen--Shannon, total variation, Hellinger, entropy/cross-entropy,
top-5 set agreement, and top-1-in-top-5 containment. Until the run completes, QVQ KL, top-1, and top-5 are
unmeasured—not zero and not estimated from a scalar or nearest-codebook proxy.
