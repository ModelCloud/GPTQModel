# Per-module window accumulation policy wave 19

This wave tested a sensitivity-selected policy on all 12 captured P32
projections and all 12 row counts. The policy uses FP16 partial accumulation
with K256 FP32 promotion on the 11 projections that passed the K256 local
sweep. Layer-1 `mlp.down_proj` uses the exact FP32 fused path (`K=0`) as its
fallback.

All 144 cases passed both the FP32-teacher and production-window gates. The
maximum teacher mean error was `0.0027259162`, maximum teacher error
`0.0125326961`, maximum window mean drift `0.0006264964`, and maximum window
drift `0.0045776367`.

## Full-layer speedup

Values are `production-window full-layer latency / policy full-layer latency`;
values above 1.0 are faster. Each median is across the 12 projections.

| M | Policy speedup |
|---:|---:|
| 1 | 0.978x |
| 2 | 0.970x |
| 4 | 0.950x |
| 8 | 0.971x |
| 16 | 0.947x |
| 32 | 0.949x |
| 64 | 0.945x |
| 128 | 0.964x |
| 256 | 0.973x |
| 512 | 1.049x |
| 1024 | 1.101x |
| 2048 | 1.201x |

The policy is slower at small M because the current fused implementation still
pays the same launch and transform costs. At M=2048 it reaches a median 1.20x
linear-layer speedup, but this is a local layer sweep and does not establish a
full-model gain. The sensitive layer-1 down fallback preserves the local
accuracy gate.

| Projection group | Promotion policy | Cases passed |
|---|---:|---:|
| Layers 0–1 q/k/v, layer-0 gate/up/down, layer-1 gate/up | K256 | 132/132 |
| Layer-1 down | K0 exact FP32 | 12/12 |

The next required step is model integration with this dispatch table, followed
by C4/FineWeb, logits, ARC/GSM8K, CUDA Graph, and real prefill/decode latency.

Raw reports:

- [Layer-0 q](results/window-policy-wave19/model-layers-0-self-attn-q-proj)
- [Layer-0 k](results/window-policy-wave19/model-layers-0-self-attn-k-proj)
- [Layer-0 v](results/window-policy-wave19/model-layers-0-self-attn-v-proj)
- [Layer-0 gate](results/window-policy-wave19/model-layers-0-mlp-gate-proj)
- [Layer-0 up](results/window-policy-wave19/model-layers-0-mlp-up-proj)
- [Layer-0 down](results/window-policy-wave19/model-layers-0-mlp-down-proj)
- [Layer-1 q](results/window-policy-wave19/model-layers-1-self-attn-q-proj)
- [Layer-1 k](results/window-policy-wave19/model-layers-1-self-attn-k-proj)
- [Layer-1 v](results/window-policy-wave19/model-layers-1-self-attn-v-proj)
- [Layer-1 gate](results/window-policy-wave19/model-layers-1-mlp-gate-proj)
- [Layer-1 up](results/window-policy-wave19/model-layers-1-mlp-up-proj)
- [Layer-1 down fallback](results/window-policy-wave19/model-layers-1-mlp-down-proj)
