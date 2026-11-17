# QQQ default Hessian damping

QQQ damps the diagonal of its calibration Hessian with `damp_percent` before
the Cholesky solve. The default was `0.005`, ten times below GPTQ's `0.05` and
half of upstream `HandH1998/QQQ` (`fasterquant`, `percdamp=0.01`).

QQQ has no activation-group-aware calibration. A single massive activation —
Llama's BOS token — therefore dominates the Hessian, and a small damp lets that
one row distort every column. Raising the default to `0.05` restores the GPTQ
reference behaviour without adding a new code path. This is issue #4074 item 4.

## Change

| Mechanism | Before | After |
| --- | --- | --- |
| `_default_damp_percent(METHOD.QQQ)` | `0.005`, `gptqmodel/quantization/config.py:3816-3817` | `0.05` |
| `_default_damp_percent(METHOD.GPTQ)` | `0.05` | `0.05` (unchanged) |
| `_default_damp_auto_increment(METHOD.QQQ)` | `0.001` | `0.001` (unchanged) |
| Explicit `damp_percent=` | honoured | honoured (unchanged) |

This changes results for existing QQQ users who did not set `damp_percent`.
Packing and export are not the cause of the old numbers: the captured float `Q`
evaluates to the same PPL, and int8 double rounding adds only ~3e-4 relative
error.

## Measured effect

Llama-3.2-1B QQQ4, wikitext-2 PPL, 80 × 512 tokens:

| `damp_percent` | BOS kept | BOS removed |
| --- | --- | --- |
| `0.005` (old default) | 59.5 | 28.1 |
| `0.01` (upstream) | 55.5 | – |
| `0.05` (new default) | 34.4 (= reference GPTQ without AGA) | 25.1 |

The effect is model-dependent: on Qwen2.5 the lower damp is slightly better
(20.99 at `0.005` vs 21.30 at `0.05`). The measured PR numbers are in the
issue #4074 pull request; the default is chosen for the Llama-class case where
the BOS row dominates.

## Independent reproduction

The PR author re-ran Llama-3.2-1B QQQ4 (128 C4 calibration documents, bf16,
`group_size=128`, 80 × 512 wikitext-2 test tokens) on an RTX 4090. With
`CalibrationConfig(concat_size=512)` the result was 18.8 at `damp_percent=0.005` and
27.0 at `0.05`; with `concat_size=2048` it was 27.4 and 56.1. On
this harness the lower damp was better, i.e. the opposite direction from the
issue's Llama table. That matches the issue's own note that the effect depends
on the model and calibration. The default is raised anyway because it matches
GPTQ and the issue's measured Llama case, but it is a visible behaviour change
for existing QQQ users, not a no-op.

## Validation

`tests/test_qqq_config.py::test_qqq_default_damp_percent_matches_gptq` pins the
default for both `QQQConfig` and `QuantizeConfig(... method=METHOD.QQQ)`.
