# Llama 3.2 1B LR32 versus P32 post-quantization quality

Date: 2026-08-30

## Result

LR32 is **not output-equivalent** to standard P32 at W2, and the current LR32 quantizer is worse against the dense model on every tested Llama 3.2 1B projection shape. This is a bounded layer-0 regression check, not a full-model acceptance run.

The production-default YAQA rounding result is the most relevant summary:

| Dense comparison | Standard P32 | LR32 | LR32 change |
|---|---:|---:|---:|
| Mean weight relative Euclidean error | 0.288386 | 0.327854 | +13.7% |
| Local module-output relative Euclidean error | 0.067596 | 0.079564 | +17.7% |
| Live module-output relative Euclidean error | 0.075628 | 0.088403 | +16.9% |
| Layer-output relative Euclidean error | 0.128841 | 0.147339 | +14.4% |
| Final-logit relative Euclidean error | 0.247275 | 0.284930 | +15.2% |
| Final-logit mean absolute error | 0.463263 | 0.533759 | +15.2% |
| Final-logit root-mean-square error | 0.598188 | 0.689372 | +15.2% |
| Final-logit cosine similarity (higher is better) | 0.969509 | 0.959983 | -0.009526 |
| Final-logit forward divergence | 0.010899 | 0.013535 | +24.2% |

All outputs and reconstructed weights were finite. Every module passed the quantizer's packed-payload round-trip verification.

## Test protocol

| Setting | Value |
|---|---|
| GPU | NVIDIA H100, physical device 1, UUID `GPU-f5ea03cf-efa4-9807-7de5-b174957a1348` |
| Model | `/monster/data/model/Llama-3.2-1B-Instruct` |
| Tested scope | Layer 0; q/k/v/o and gate/up/down projections |
| Rate | W2 plus 0.03125 selector bits per weight; 2.03125 effective bits per weight |
| Calibration | 8 real chat rows, 4,045 tokens |
| Evaluation | 8 disjoint real chat rows, 983 valid tokens, rows 8–15 |
| YAQA factor data | 8 additional disjoint real chat rows, 1,024 valid tokens, rows 16–23 |
| Hessian mode | Dense-frozen, so both formats receive the same source calibration geometry |
| Seeds | Quantization 18,240; YAQA factor capture 0 |
| Evaluation execution | CUDA forward and CUDA metric reductions |

The comparison evaluates dense reconstructed quantized weights. This isolates quantization quality from kernel speed. The quantizer also reconstructs the packed payload and verifies it against the selected dense quantized weight for every module.

## Production-default YAQA rounding

### Weight error by projection

| Projection | Shape N×K | P32 relative error | LR32 relative error | LR32 change |
|---|---:|---:|---:|---:|
| down | 2048×8192 | 0.274854 | 0.312540 | +13.7% |
| gate | 8192×2048 | 0.276508 | 0.313685 | +13.4% |
| up | 8192×2048 | 0.275736 | 0.312590 | +13.4% |
| key | 512×2048 | 0.307536 | 0.350437 | +14.0% |
| output | 2048×2048 | 0.281416 | 0.319779 | +13.6% |
| query | 2048×2048 | 0.278430 | 0.317532 | +14.0% |
| value | 512×2048 | 0.324220 | 0.368414 | +13.6% |

### Output error versus dense

| Output | Metric | Standard P32 | LR32 | LR32 change |
|---|---|---:|---:|---:|
| Local modules | Mean absolute error | 0.027125 | 0.031534 | +16.3% |
| Local modules | Root-mean-square error | 0.041946 | 0.049372 | +17.7% |
| Local modules | Relative Euclidean error | 0.067596 | 0.079564 | +17.7% |
| Local modules | Cosine similarity | 0.997786 | 0.996980 | -0.000806 |
| Live modules | Mean absolute error | 0.031745 | 0.036755 | +15.8% |
| Live modules | Root-mean-square error | 0.046930 | 0.054857 | +16.9% |
| Live modules | Relative Euclidean error | 0.075628 | 0.088403 | +16.9% |
| Live modules | Cosine similarity | 0.997205 | 0.996229 | -0.000976 |
| Final logits | Mean absolute error | 0.463263 | 0.533759 | +15.2% |
| Final logits | Root-mean-square error | 0.598188 | 0.689372 | +15.2% |
| Final logits | Relative Euclidean error | 0.247275 | 0.284930 | +15.2% |
| Final logits | Cosine similarity | 0.969509 | 0.959983 | -0.009526 |
| Final logits | Forward divergence | 0.010899 | 0.013535 | +24.2% |

## Block feedback rounding control

The non-YAQA control shows the same format-wide regression.

### Weight error by projection

| Projection | Shape N×K | P32 relative error | LR32 relative error | LR32 change |
|---|---:|---:|---:|---:|
| down | 2048×8192 | 0.315014 | 0.357891 | +13.6% |
| gate | 8192×2048 | 0.382536 | 0.433047 | +13.2% |
| up | 8192×2048 | 0.381866 | 0.432319 | +13.2% |
| key | 512×2048 | 0.362197 | 0.411588 | +13.6% |
| output | 2048×2048 | 0.357168 | 0.404080 | +13.1% |
| query | 2048×2048 | 0.366497 | 0.415743 | +13.4% |
| value | 512×2048 | 0.353848 | 0.400839 | +13.3% |

| Dense comparison | Standard P32 | LR32 | LR32 change |
|---|---:|---:|---:|
| Mean weight relative Euclidean error | 0.359875 | 0.407930 | +13.4% |
| Local module-output relative Euclidean error | 0.058754 | 0.072590 | +23.5% |
| Live module-output relative Euclidean error | 0.065894 | 0.080451 | +22.1% |
| Layer-output relative Euclidean error | 0.112246 | 0.132904 | +18.4% |
| Final-logit relative Euclidean error | 0.232173 | 0.258493 | +11.3% |
| Final-logit cosine similarity | 0.972993 | 0.966590 | -0.006403 |
| Final-logit forward divergence | 0.009551 | 0.027223 | +185.0% |

## Why the format changes the output

The format change is not merely a byte-level transpose.

Standard P32 quantizes each 16×16 weight tile as one 128-step trellis history. Its eight 16-step bank-selection segments retain continuous trellis state across segment boundaries. LR32 reorganizes the same 256 weights as a 32×8 tile, then quantizes eight independent 16-step rings. Each ring closes its own history, so LR32 removes all state continuity between the eight rings.

That shorter history changes the set of reachable quantized vectors and therefore changes the reconstructed weights. The nearly uniform 13–14% weight-error increase across all seven shapes, under both rounding methods, is strong evidence that the loss is caused by this shared LR32 history constraint rather than one projection or one calibration sample.

The next quantizer experiment should keep the LR32 packed/runtime layout but solve all eight 16-step segments as one continuous 128-step history with bank decisions every 16 steps. Runtime decoding uses the stored states and selectors and does not need to reproduce the encoder's recurrence. That experiment can test whether the low-conflict LR32 memory layout can retain standard P32 quantization quality without changing the CUDA kernel.

## Raw reports

- `artifacts/lr32_quality/llama32_1b_layer0_all_linear_w2_block_ldlq.json`
- `artifacts/lr32_quality/llama32_1b_layer0_all_linear_w2_block_ldlq_lr32.json`
- `artifacts/lr32_quality/llama32_1b_layer0_all_linear_w2_yaqa_p32.json`
- `artifacts/lr32_quality/llama32_1b_layer0_all_linear_w2_yaqa_lr32.json`
