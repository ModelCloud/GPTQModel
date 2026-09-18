# QVQ + YAQA P32 diversity-sampled family search

This quantization-stage optimization reduces full YAQA family histories while preserving a dual-precision quality gate. It does not change post-quant inference.

## Method

The P32 complementary families share one canonical bank but produce distinct sequential YAQA feedback histories. Full reselection executes canonical V2 plus all three complementary histories.

The optimized policy:

1. scores 256 evenly spaced 16x16 tiles under the local two-sided Hessian proxy;
2. keeps the lowest-scoring and highest-scoring complementary families;
3. runs complete sequential YAQA for canonical V2 and those two diversity endpoints;
4. selects the final payload with the unchanged full Kronecker-Fisher objective.

Keeping the local-proxy extremes is intentional. On the measured Llama layer, the full sequential winner was frequently the local proxy's worst family. Keeping the two locally best families missed that diversity and produced larger held-out regressions.

Configuration:

```json
{
  "yaqa": {
    "sample_strategy": "256_16x16",
    "sampled_family_candidates": 2,
    "sampled_family_selection": "diversity"
  }
}
```

The previous one-family sampled behavior remains available with `sampled_family_candidates=1` and `sampled_family_selection="lowest"`.

## Matched H100 results

Layer 0 of Llama 3.2 1B, 32 independent YAQA sequences, 11,992 valid tokens, batch 8, exact pruning, and identical seeds/data:

| Rate | Full search | Diversity search | Speedup | FP32 Fisher delta | FP64 Fisher delta |
|---|---:|---:|---:|---:|---:|
| W2.5 | 34.950 s | 26.613 s | 1.313x | 0.0000% | 0.0000% |
| W3 | 41.451 s | 32.268 s | 1.285x | -0.1245% | -0.1246% |
| W3.5 | 39.093 s | 29.623 s | 1.320x | +0.00525% | +0.00525% |

Negative objective delta is an improvement. The timing includes the opt-in FP64 oracle for both arms, so the speed comparison is matched.

W2.5 reproduced every module objective exactly. W3 changed only V and improved its FP64 Fisher objective by 0.1503%. W3.5 changed Q, gate, and down; its tiny aggregate Fisher regression was therefore checked against disjoint held-out activations rather than rejected from proxy ordering alone.

## Held-out gate

`scripts/validate_qvq_yaqa_p32_heldout.py` reconstructs both serialized P32 payloads, captures the corresponding dense-model projection inputs and outputs, and measures projection error on held-out data. It requires a passing disjointness manifest and records its SHA-256 binding.

The strict preflight checked calibration rows `[0, 32)` and YAQA rows `[64, 96)` against all 1,209 GSM8K Platinum test questions and passed with zero overlap.

On 256 GSM8K Platinum test questions:

- W3 diversity search changed aggregate held-out projection MSE by `+0.000670%`. Six modules were exact; V changed by `+0.1467%` while its FP32 and FP64 Fisher objectives improved.
- W3.5 diversity search improved aggregate held-out projection MSE by `0.04561%`. Q improved by `0.1393%`, gate improved by `0.03276%`, and down regressed by `0.2052%`.

The W3 change is accepted as a bounded rounding tradeoff: the practical aggregate change is six parts per million, the Fisher objective improves in both precisions, and no broad error expansion occurs. W3.5 is a double win despite its tiny Fisher-proxy regression because the independent held-out metric improves.

## Oracle controls

Set `GPTQMODEL_QVQ_DUAL_ORACLE=1` during a validation quantization run to emit `yaqa_kronecker_proxy_loss` and `yaqa_kronecker_proxy_loss_fp64` for every projection. This is intentionally opt-in because FP64 scoring is a validation cost, not part of normal quantization.

Acceptance requires:

1. matched model, data, seed, and rate;
2. FP32 and FP64 objective deltas reported per module and in aggregate;
3. exact payload differences enumerated;
4. a disjoint held-out activation report when either precision oracle regresses or operation ordering changes the payload;
5. bounded local regressions with no aggregate held-out degradation beyond the declared tolerance.
