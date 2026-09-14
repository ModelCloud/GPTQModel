# Author quantizer arithmetic parity

Compared pinned IST-DASLab/GSQ `03fc16484c369e3127225615d5e03e8d3a6043e3` against explicit local backward.

Six CUDA fixtures: W2/W3/W4, FP32/BF16, 8x32 weights, group16, seed7 initialization and seed11 sampling, temperature .7, multiplier20. All forward/logit-gradient/scale-gradient maximum differences are zero.

This is matched-noise quantizer arithmetic only, not staged optimizer trajectory or model-quality parity. Executed on leased GPU-737e2423-874a-23a4-1126-dfbe3e77c294 (PG506-230).

/root/polly-work/qvq-gsq/artifacts/gsq-staged/author-parity-explicit-backward.json SHA256 ab4d4415f32354ddae7fb7c381f9c70b37c18953d1798398b1d317bc91b26103

/root/polly-work/qvq-gsq/artifacts/gsq-staged/logs/author-parity-explicit-backward.log SHA256 8b0df1b4ee706be90d3548819aa02be80e8383e934bad7f905ee431c0e307cae

```json
[
  {
    "bits": 2,
    "dtype": "torch.float32",
    "forward": 0.0,
    "logit_gradient": 0.0,
    "scale_gradient": 0.0
  },
  {
    "bits": 2,
    "dtype": "torch.bfloat16",
    "forward": 0.0,
    "logit_gradient": 0.0,
    "scale_gradient": 0.0
  },
  {
    "bits": 3,
    "dtype": "torch.float32",
    "forward": 0.0,
    "logit_gradient": 0.0,
    "scale_gradient": 0.0
  },
  {
    "bits": 3,
    "dtype": "torch.bfloat16",
    "forward": 0.0,
    "logit_gradient": 0.0,
    "scale_gradient": 0.0
  },
  {
    "bits": 4,
    "dtype": "torch.float32",
    "forward": 0.0,
    "logit_gradient": 0.0,
    "scale_gradient": 0.0
  },
  {
    "bits": 4,
    "dtype": "torch.bfloat16",
    "forward": 0.0,
    "logit_gradient": 0.0,
    "scale_gradient": 0.0
  }
]
```
