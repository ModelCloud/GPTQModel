# Real ParoQuant QKV GSQ validation

Executed against `5417c5493` plus the archived experiment scripts on the SM80
PG506-230 (physical GPU 0, PCI DE:00.0). This is selected-projection validation,
not a complete native model export or final-logit recovery measurement.

The saved Llama 3.2 1B block-0 Q/K/V weights were checked exactly against the
dense safetensors checkpoint. Saved F6/seed7 calibration source hashes and
activation dimensions were verified. The 16 calibration documents were split
into 12 training (2,894 tokens) and four optimizer-validation (873 tokens).
The 32 evaluation documents (6,367 tokens) were untouched by optimization.
Identical token documents across these splits are rejected. Calibration inputs
use the saved source weights via square-root activation weighting.

Each arm ran the actual ParoQuant module processor with W4, group 128, eight
rotations, base seed 7, and default ten rotation plus ten finetuning epochs.
GSQ used 100 steps and seed 7. Config serialization was round-tripped before
execution. The initializer may use calibration validation; GSQ checkpoint
selection uses training reconstruction only.

| Projection | Shape | Held-out native output MSE, all three arms |
|---|---|---:|
| Q | 2048 × 2048 | 0.0006027474924634314 |
| K | 512 × 2048 | 0.0011034244717537793 |
| V | 512 × 2048 | 0.000028381133478102033 |

Both fixed-scale and learned-scale GSQ retained byte-identical packed tensors
to the baseline in all three projections. No GSQ improvement is claimed.
MSE compares native outputs with original dense FP32 weights on the same
FP16-cast activation inputs. It is not full-model KL or Top-K agreement.

All nine arms were packed into ParoLinear, saved, strictly reloaded and run on
all 32 held-out documents: **288 localized native checks passed**. The reference
reconstructs the affine grid from export tensors and applies inverse rotations
using exported metadata. Worst mean absolute drift was 0.0003560413 and worst
maximum drift was 0.008094788, below 0.002 and 0.046875 respectively.
The earlier synthetic native suite separately covers CUDA graph replay; this
real-data run checks eager execution and does not repeat graph coverage.

Evidence: `artifacts/gsq-paro/real-qkv-seed7-v1/` contains the report, provenance,
exact document split, configs, payloads, compressed executed scripts/log and
SHA256 manifest. Partial-layer artifacts remain in the experiment workspace.
Grouped optimization binding, final-model propagation and complete exports
remain pending.
