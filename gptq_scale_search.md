# GPTQ scale-search evaluation log

This document records post-quantization quality and quantization-time measurements for GPTQ scale-search
objectives. The objective is to determine whether activation-, Hessian-, or hybrid-weighted scale selection improves
downstream model quality over disabled scale search, and whether different projection families prefer different
objectives.

## Llama 3.2 1B Instruct

### Test configuration

```text
+------------------+-----------------------------------------------------------------------+
| Model            | /monster/data/model/Llama-3.2-1B-Instruct                            |
| Quantization     | GPTQ W4, group size 128, symmetric, BF16 compute                     |
| Calibration      | 512 samples, 188,256 non-padding tokens, sequence length 2048         |
| Evaluation       | ARC-Challenge 1,172; GSM8K Platinum 1,209; greedy generation          |
| Inference        | Marlin; FlashAttention 2; non-paged attention                         |
| GPU              | NVIDIA PG506-230/232, Ampere sm_80, 96 GiB                           |
| Software         | GPT-QModel Ultra 7.2.0; Evalution 0.0.8; Transformers 5.14.1         |
|                  | Torch 2.12.0+cu130; Triton 3.7.0                                     |
+------------------+-----------------------------------------------------------------------+
```

The reported mean is the unweighted arithmetic mean of ARC accuracy, ARC normalized accuracy, and GSM8K
Platinum accuracy. It is a compact comparison statistic, not a replacement for the individual task scores.

### Full-model four-arm test

Scale search used the same objective for every quantized projection.

```text
+--------------------+----------+------------+----------+----------+
| Metric             | Disabled | Activation | Hessian  | Hybrid   |
+--------------------+----------+------------+----------+----------+
| ARC accuracy       | 0.301195 |   0.310580 | 0.307167 | 0.323379 |
| ARC normalized acc | 0.339590 |   0.343003 | 0.338737 | 0.342150 |
| GSM8K Platinum acc | 0.387924 |   0.434243 | 0.428453 | 0.419355 |
+--------------------+----------+------------+----------+----------+
| Mean               | 0.342903 |   0.362609 | 0.358119 | 0.361628 |
+--------------------+----------+------------+----------+----------+
| Module quant time  | 177.252s |   184.749s | 280.324s | 312.026s |
+--------------------+----------+------------+----------+----------+
```

Full-model findings:

- Activation had the highest mean score: `+0.019706`, or `+5.747%`, over disabled scale search.
- Hybrid had the strongest raw ARC accuracy, but activation was better on GSM8K and on the aggregate mean.
- Hessian and hybrid were materially more expensive than activation.

### Projection-scoped seven-arm test

The disabled arm was shared. Each enabled arm applied scale search only to one projection family while all other
projections retained `scale_search=None`:

- QKV only: `self_attn.q_proj`, `self_attn.k_proj`, and `self_attn.v_proj`.
- MLP only: `mlp.gate_proj`, `mlp.up_proj`, and `mlp.down_proj`.
- `self_attn.o_proj` remained disabled in both scopes to keep the requested QKV and non-attention groups disjoint.

The seven independent arms ran on GPUs 0-6 with per-process OpenMP/BLAS thread limits. The disabled scores exactly
reproduced the earlier full-model disabled arm.

#### QKV projections only

```text
+--------------------+----------+------------+----------------------+----------+----------------------+----------+----------------------+
| Metric             | Disabled | Activation | Activation delta     | Hessian  | Hessian delta        | Hybrid   | Hybrid delta         |
+--------------------+----------+------------+----------------------+----------+----------------------+----------+----------------------+
| ARC accuracy       | 0.301195 |   0.319113 | +0.017918 / +5.949%  | 0.319966 | +0.018771 / +6.232%  | 0.325085 | +0.023891 / +7.932%  |
| ARC normalized acc | 0.339590 |   0.350683 | +0.011092 / +3.266%  | 0.366041 | +0.026451 / +7.789%  | 0.351536 | +0.011945 / +3.518%  |
| GSM8K Platinum acc | 0.387924 |   0.405294 | +0.017370 / +4.478%  | 0.408602 | +0.020678 / +5.330%  | 0.391232 | +0.003309 / +0.853%  |
+--------------------+----------+------------+----------------------+----------+----------------------+----------+----------------------+
| Mean               | 0.342903 |   0.358363 | +0.015460 / +4.509%  | 0.364870 | +0.021967 / +6.406%  | 0.355951 | +0.013048 / +3.805%  |
+--------------------+----------+------------+----------------------+----------+----------------------+----------+----------------------+
```

QKV finding: Hessian is the preferred objective. It won ARC normalized accuracy, GSM8K, and the aggregate mean.
Hybrid won raw ARC but was the weakest enabled method on the mean and barely improved GSM8K.

#### MLP projections only

```text
+--------------------+----------+------------+-----------------------+----------+-----------------------+----------+----------------------+
| Metric             | Disabled | Activation | Activation delta      | Hessian  | Hessian delta         | Hybrid   | Hybrid delta         |
+--------------------+----------+------------+-----------------------+----------+-----------------------+----------+----------------------+
| ARC accuracy       | 0.301195 |   0.333618 | +0.032423 / +10.765%  | 0.308874 | +0.007679 /  +2.550%  | 0.321672 | +0.020478 / +6.799%  |
| ARC normalized acc | 0.339590 |   0.352389 | +0.012799 /  +3.769%  | 0.340444 | +0.000853 /  +0.251%  | 0.359215 | +0.019625 / +5.779%  |
| GSM8K Platinum acc | 0.387924 |   0.411911 | +0.023987 /  +6.183%  | 0.427626 | +0.039702 / +10.235%  | 0.416873 | +0.028950 / +7.463%  |
+--------------------+----------+------------+-----------------------+----------+-----------------------+----------+----------------------+
| Mean               | 0.342903 |   0.365972 | +0.023070 /  +6.728%  | 0.358981 | +0.016078 /  +4.689%  | 0.365920 | +0.023017 / +6.712%  |
+--------------------+----------+------------+-----------------------+----------+-----------------------+----------+----------------------+
```

MLP finding: activation and hybrid were effectively tied on the mean (`0.000052` apart). Activation strongly favored
raw ARC, hybrid favored normalized ARC and GSM8K, and Hessian produced the best GSM8K result but weak ARC gains.
Activation is the practical default because it achieved the nominally highest mean at substantially lower cost.

#### Projection-scoped quantization cost

These are summed per-module quantization times. They were captured during the concurrent seven-GPU run, so they are
most useful as directional cost measurements rather than isolated single-worker benchmarks.

```text
+----------------+------------+-------------------+
| Scope/method   | Quant time | Versus disabled   |
+----------------+------------+-------------------+
| Disabled       |   197.266s | 1.000x            |
| QKV activation |   199.794s | 1.013x /  +1.3%   |
| QKV Hessian    |   251.548s | 1.275x / +27.5%   |
| QKV hybrid     |   251.998s | 1.277x / +27.7%   |
| MLP activation |   212.636s | 1.078x /  +7.8%   |
| MLP Hessian    |   239.170s | 1.212x / +21.2%   |
| MLP hybrid     |   266.419s | 1.351x / +35.1%   |
+----------------+------------+-------------------+
```

### Correlated scale-search optimization

The original Hessian and hybrid implementation evaluated all 80 clipping candidates serially. Every candidate
issued a separate FP32 `error @ Hessian` matrix multiplication, making eager CUDA launch overhead dominate even for
large output projections. The optimized implementation:

- Batches a bounded number of candidates into each quadratic scorer call.
- Uses one larger FP32 GEMM per candidate chunk while preserving exact group-local Hessian scoring.
- Folds hybrid's 50/50 diagonal shrinkage into its prepared Hessian once, removing a second candidate-sized
  square/reduction allocation.
- Retains activation/MSE's existing 8M-element, 16-candidate workspace policy.
- Uses a 16M-element correlated-objective workspace, producing chunks of 80, 64, and 16 candidates for
  512x128, 2048x128, and 8192x128 groups respectively.
- Preserves the scalar implementation's first-candidate tie behavior and exact selected scales, zeros, and quantized
  weights in CPU/FP32 and Ampere CUDA/BF16 A/B tests.

Ampere `sm_80` microbenchmark after optimization:

```text
+------------+------------+-------------+-------------+-----------------------+
| Shape      | Activation | Hessian     | Hybrid      | Correlated peak memory|
+------------+------------+-------------+-------------+-----------------------+
| 512x128    |    2.476ms |     1.164ms |     1.159ms |              69.3 MiB |
| 2048x128   |    2.412ms |     2.279ms |     2.346ms |             203.0 MiB |
| 8192x128   |    5.975ms |     6.603ms |     5.965ms |             205.3 MiB |
+------------+------------+-------------+-------------+-----------------------+
```

Relative to the old scalar correlated path, Hessian improved by `24.9x`, `12.8x`, and `4.5x` at these shapes;
hybrid improved by `30.4x`, `15.1x`, and `6.3x`. FP32 measurements matching GPTQ's internal weight dtype showed the
same pattern: Hessian/hybrid were faster than activation at 2048x128 and about 16-17% slower at 8192x128.

The same seven-arm Llama quantization-only test after optimization produced:

```text
+----------------+------------+-------------------+----------------------+
| Scope/method   | Quant time | Versus disabled   | Versus same scope act|
+----------------+------------+-------------------+----------------------+
| Disabled       |   170.769s | 1.000x            | -                    |
| QKV activation |   180.253s | 1.056x /  +5.6%   | 1.000x               |
| QKV Hessian    |   193.426s | 1.133x / +13.3%   | 1.073x / +7.3%       |
| QKV hybrid     |   173.841s | 1.018x /  +1.8%   | 0.964x / -3.6%       |
| MLP activation |   177.269s | 1.038x /  +3.8%   | 1.000x               |
| MLP Hessian    |   182.478s | 1.069x /  +6.9%   | 1.029x / +2.9%       |
| MLP hybrid     |   176.423s | 1.033x /  +3.3%   | 0.995x / -0.5%       |
+----------------+------------+-------------------+----------------------+
```

Raw times vary between concurrent runs, so within-run ratios are the most reliable comparison. The costly correlated
gap versus activation fell from `+25.9%` to `+7.3%` for QKV Hessian and from `+12.5%` to `+2.9%` for MLP Hessian.
Hybrid is now within run-to-run noise of activation in both scopes.

## Qwen3 8B projection-scope replication

The projection-scope test was repeated on the complete local `/monster/data/model/Qwen3-8B` checkpoint. It covered
ten unique configurations: one shared disabled baseline plus activation, Hessian, and hybrid scale search applied
independently to QKV, O, or MLP projections. Quantization used GPTQ W4, group size 128, symmetric weights, BF16
compute, 512 calibration samples, 181,796 non-padding tokens,
and sequence length 2048. Evaluation used all 1,172 ARC-Challenge and 1,209 GSM8K Platinum examples. Qwen evaluation
used plain, non-chat prompts to avoid applying the model's thinking chat template; GSM8K generation used a fixed
batch size of 16 in every arm with zero invalid answers or OOM retries.
Permanent full-coverage checkpoint paths and reuse notes are indexed in `gptq_qwen3_8b_test_models.md`.

### Consolidated raw metrics

The mean is the unweighted arithmetic mean of the three raw task metrics in this table.

```text
+----------------+--------------+----------------+-----------------+----------+-------------------+
| Scope/method   | ARC accuracy | ARC normalized | GSM8K Platinum | Mean     | Mean vs disabled  |
+----------------+--------------+----------------+-----------------+----------+-------------------+
| Disabled       |     0.537543 |       0.541809 |        0.913151 | 0.664168 | -                 |
| QKV activation |     0.534130 |       0.552048 |        0.908189 | 0.664789 | +0.000621/+0.094% |
| QKV Hessian    |     0.542662 |       0.550341 |        0.898263 | 0.663755 | -0.000412/-0.062% |
| QKV hybrid     |     0.540102 |       0.543515 |        0.907361 | 0.663660 | -0.000508/-0.076% |
| O activation   |     0.553754 |       0.552901 |        0.904880 | 0.670512 | +0.006344/+0.955% |
| O Hessian      |     0.522184 |       0.534130 |        0.908189 | 0.654834 | -0.009333/-1.405% |
| O hybrid       |     0.534983 |       0.541809 |        0.904053 | 0.660282 | -0.003886/-0.585% |
| MLP activation |     0.539249 |       0.536689 |        0.917287 | 0.664409 | +0.000241/+0.036% |
| MLP Hessian    |     0.538396 |       0.541809 |        0.929694 | 0.669966 | +0.005799/+0.873% |
| MLP hybrid     |     0.542662 |       0.534130 |        0.896609 | 0.657800 | -0.006367/-0.959% |
+----------------+--------------+----------------+-----------------+----------+-------------------+
```

### Meaning of the disabled baseline

`Disabled` is the historical out-of-the-box GPTQ behavior from before `ScaleSearchConfig` was added. In the parent
of the initial Ultra scale-search port (`2993cd97`), `GPTQConfig` had `mse=0.0` and no `scale_search` field; the zero
MSE value skipped the clipping/scale-search loop. The initial port itself also declared `scale_search=None`, so its
default remained disabled. A user could still explicitly set the legacy `mse` parameter above zero to enable the
older weight-MSE search, but that was not the default.

Commit `44a82003` changed the behavior on this branch: omitting `scale_search` now resolves to `activation`, while
an explicit `GPTQConfig(scale_search=None)` remains the supported opt-out and is exactly what the disabled test arm
uses. Therefore, `Disabled` is the correct legacy-default comparison, but it is no longer the default on the current
`scale-search-default` branch.

### QKV projections only

```text
+--------------------+----------+------------+----------------------+----------+----------------------+----------+----------------------+
| Metric             | Disabled | Activation | Activation delta     | Hessian  | Hessian delta        | Hybrid   | Hybrid delta         |
+--------------------+----------+------------+----------------------+----------+----------------------+----------+----------------------+
| ARC accuracy       | 0.537543 |   0.534130 | -0.003413 / -0.635%  | 0.542662 | +0.005119 / +0.952%  | 0.540102 | +0.002560 / +0.476%  |
| ARC normalized acc | 0.541809 |   0.552048 | +0.010239 / +1.890%  | 0.550341 | +0.008532 / +1.575%  | 0.543515 | +0.001706 / +0.315%  |
| GSM8K Platinum acc | 0.913151 |   0.908189 | -0.004963 / -0.543%  | 0.898263 | -0.014888 / -1.630%  | 0.907361 | -0.005790 / -0.634%  |
+--------------------+----------+------------+----------------------+----------+----------------------+----------+----------------------+
| Mean               | 0.664168 |   0.664789 | +0.000621 / +0.094%  | 0.663755 | -0.000412 / -0.062%  | 0.663660 | -0.000508 / -0.076%  |
+--------------------+----------+------------+----------------------+----------+----------------------+----------+----------------------+
```

QKV finding: no enabled objective clearly beat disabled search. Activation had the nominally highest mean by only
`0.000621`; Hessian produced the strongest raw and normalized ARC combination but lost `0.014888` on GSM8K. This
does not reproduce Llama's QKV Hessian aggregate win.

### Attention output projection only

This scope applied scale search only to `self_attn.o_proj`; QKV and MLP projections retained disabled search. The
disabled arm was rerun concurrently and exactly reproduced all three earlier disabled scores.

```text
+--------------------+----------+------------+----------------------+----------+-----------------------+----------+----------------------+
| Metric             | Disabled | Activation | Activation delta     | Hessian  | Hessian delta         | Hybrid   | Hybrid delta         |
+--------------------+----------+------------+----------------------+----------+-----------------------+----------+----------------------+
| ARC accuracy       | 0.537543 |   0.553754 | +0.016212 / +3.016%  | 0.522184 | -0.015358 / -2.857%   | 0.534983 | -0.002560 / -0.476%  |
| ARC normalized acc | 0.541809 |   0.552901 | +0.011092 / +2.047%  | 0.534130 | -0.007679 / -1.417%   | 0.541809 | +0.000000 / +0.000%  |
| GSM8K Platinum acc | 0.913151 |   0.904880 | -0.008271 / -0.906%  | 0.908189 | -0.004963 / -0.543%   | 0.904053 | -0.009098 / -0.996%  |
+--------------------+----------+------------+----------------------+----------+-----------------------+----------+----------------------+
| Mean               | 0.664168 |   0.670512 | +0.006344 / +0.955%  | 0.654834 | -0.009333 / -1.405%   | 0.660282 | -0.003886 / -0.585%  |
+--------------------+----------+------------+----------------------+----------+-----------------------+----------+----------------------+
```

O-projection finding: activation is the clear preference. It improved both ARC metrics substantially and produced
the highest mean despite losing `0.008271` on GSM8K. Hessian was the worst objective because it regressed both ARC
metrics, while hybrid also finished below disabled on the mean.

The O-only concurrent module-time sums were:

```text
+--------------+------------+-------------------+
| Method       | Quant time | Versus disabled   |
+--------------+------------+-------------------+
| Disabled     |   774.797s | 1.000x            |
| O activation |   764.946s | 0.987x / -1.3%    |
| O Hessian    |   776.563s | 1.002x / +0.2%    |
| O hybrid     |   792.214s | 1.022x / +2.2%    |
+--------------+------------+-------------------+
```

These timing differences are small and include cross-GPU noise. There is no evidence that isolated O-projection
activation or Hessian search materially increases total Qwen3 8B quantization time.

### MLP projections only

```text
+--------------------+----------+------------+----------------------+----------+----------------------+----------+----------------------+
| Metric             | Disabled | Activation | Activation delta     | Hessian  | Hessian delta        | Hybrid   | Hybrid delta         |
+--------------------+----------+------------+----------------------+----------+----------------------+----------+----------------------+
| ARC accuracy       | 0.537543 |   0.539249 | +0.001706 / +0.317%  | 0.538396 | +0.000853 / +0.159%  | 0.542662 | +0.005119 / +0.952%  |
| ARC normalized acc | 0.541809 |   0.536689 | -0.005119 / -0.945%  | 0.541809 | +0.000000 / +0.000%  | 0.534130 | -0.007679 / -1.417%  |
| GSM8K Platinum acc | 0.913151 |   0.917287 | +0.004136 / +0.453%  | 0.929694 | +0.016543 / +1.812%  | 0.896609 | -0.016543 / -1.812%  |
+--------------------+----------+------------+----------------------+----------+----------------------+----------+----------------------+
| Mean               | 0.664168 |   0.664409 | +0.000241 / +0.036%  | 0.669966 | +0.005799 / +0.873%  | 0.657800 | -0.006367 / -0.959%  |
+--------------------+----------+------------+----------------------+----------+----------------------+----------+----------------------+
```

MLP finding: Hessian was the clear aggregate winner, driven by a `+0.016543` GSM8K gain without giving up ARC.
Hybrid moved GSM8K by exactly the same magnitude in the opposite direction and had the lowest mean. The Llama
preference for activation/hybrid on the MLP mean therefore does not generalize to Qwen3 8B.

Taken together, the Qwen3 8B scoped results suggest a projection-specific preference: attention QKV projections
nominally favor activation scale search (`0.664789` mean), attention O projections clearly favor activation
(`0.670512`), and non-attention MLP projections favor Hessian (`0.669966`). The QKV activation margin over disabled
is small (`+0.000621`, or `+0.094%`), while the O activation and MLP Hessian margins are materially larger at
`+0.006344` (`+0.955%`) and `+0.005799` (`+0.873%`). This conclusion is specific to Qwen3 8B and should not be
generalized across architectures without another scoped evaluation.

### Full-coverage QKVO-activation/MLP-Hessian test

The independently preferred Qwen objectives were combined into a full-coverage policy. Q, K, V, and O projections
used activation scale search; all remaining quantized projections used Hessian scale search. Qwen3 8B quantizes
exactly seven projection types per layer, so this maps QKVO to activation and MLP gate/up/down to Hessian without
leaving any quantized projection unmatched. The disabled baseline was rerun concurrently and reproduced its earlier
scores exactly.

```python
scale_search = ScaleSearchConfig.HESSIAN
dynamic = {
    r"+:^model\.layers\.\d+\.self_attn\.(?:q_proj|k_proj|v_proj|o_proj)$": {
        "scale_search": ScaleSearchConfig.ACTIVATION,
    },
}
```

```text
+--------------------+----------+-----------------+----------------------+----------------------+
| Metric             | Disabled | Combined policy | Delta                | Relative             |
+--------------------+----------+-----------------+----------------------+----------------------+
| ARC accuracy       | 0.537543 |        0.542662 | +0.005119            | +0.952%              |
| ARC normalized acc | 0.541809 |        0.544369 | +0.002560            | +0.472%              |
| GSM8K Platinum acc | 0.913151 |        0.917287 | +0.004136            | +0.453%              |
+--------------------+----------+-----------------+----------------------+----------------------+
| Mean               | 0.664168 |        0.668106 | +0.003938            | +0.593%              |
| Module quant time  | 805.122s |        817.814s | +12.692s             | +1.576% / 1.016x     |
+--------------------+----------+-----------------+----------------------+----------------------+
```

Full-coverage finding: the combined policy improved all three quality metrics and the aggregate mean for only
`+1.6%` quantization time. The scoped gains did not add linearly: its `0.668106` mean is below isolated O activation
(`0.670512`) and isolated MLP Hessian (`0.669966`). Nonetheless, unlike either isolated preference, the combined
policy moved ARC, normalized ARC, and GSM8K in the same positive direction, making it the strongest balanced
Qwen-specific policy tested here.

### Full-model objective sweep

To complete the full-coverage comparison without rerunning existing arms, three additional policies applied the
same objective to every Qwen projection: all activation, all Hessian, and all hybrid. The disabled and mixed-policy
values below are reused from their completed runs; only the three missing global-objective arms were executed.

```text
+----------------------------+--------------+----------------+-----------------+----------+---------------------+
| Full-coverage policy       | ARC accuracy | ARC normalized | GSM8K Platinum | Mean     | Mean vs disabled    |
+----------------------------+--------------+----------------+-----------------+----------+---------------------+
| Disabled                   |     0.537543 |       0.541809 |        0.913151 | 0.664168 | -                   |
| All activation             |     0.548635 |       0.550341 |        0.911497 | 0.670158 | +0.005990 / +0.902% |
| All Hessian                |     0.543515 |       0.546075 |        0.913978 | 0.667856 | +0.003689 / +0.555% |
| All hybrid                 |     0.534983 |       0.530717 |        0.913978 | 0.659893 | -0.004275 / -0.644% |
| QKVO activation/MLP Hessian|     0.542662 |       0.544369 |        0.917287 | 0.668106 | +0.003938 / +0.593% |
+----------------------------+--------------+----------------+-----------------+----------+---------------------+
```

```text
+----------------------------+------------+----------------+--------------+------------+
| Full-coverage policy       | ARC delta  | ARC norm delta | GSM8K delta | Mean delta |
+----------------------------+------------+----------------+--------------+------------+
| All activation             |  +0.011092 |      +0.008532 |    -0.001654 |  +0.005990 |
| All Hessian                |  +0.005973 |      +0.004266 |    +0.000827 |  +0.003689 |
| All hybrid                 |  -0.002560 |      -0.011092 |    +0.000827 |  -0.004275 |
| QKVO activation/MLP Hessian|  +0.005119 |      +0.002560 |    +0.004136 |  +0.003938 |
+----------------------------+------------+----------------+--------------+------------+
```

All-activation produced the highest mean among full-coverage policies and the strongest ARC scores, but it lost
`0.001654` on GSM8K. The mixed policy had a lower mean but was the strongest balanced choice because it improved all
three metrics and produced the best GSM8K score. All-Hessian also improved every metric, though less than the mixed
policy on mean and GSM8K. All-hybrid is not competitive: its small GSM8K gain did not offset regressions in both ARC
metrics.

The global arms were run concurrently without repeating disabled. Ratios use the most recent disabled module-time
sum and are directional because they cross concurrent runs and GPUs.

```text
+----------------------------+------------+--------------------+
| Full-coverage policy       | Quant time | Versus disabled    |
+----------------------------+------------+--------------------+
| Disabled                   |   805.122s | 1.000x             |
| All activation             |   831.257s | 1.032x /  +3.2%   |
| All Hessian                |   798.155s | 0.991x /  -0.9%   |
| All hybrid                 |   899.539s | 1.117x / +11.7%   |
| QKVO activation/MLP Hessian|   817.814s | 1.016x /  +1.6%   |
+----------------------------+------------+--------------------+
```

### Qwen3 8B quantization cost

The seven arms quantized concurrently on GPUs 0-6. The table reports the sum of all 252 per-module quantization
times, so within-scope Hessian-versus-activation ratios are more meaningful than absolute cross-GPU times.

```text
+----------------+------------+-------------------+-----------------------+
| Scope/method   | Quant time | Versus disabled   | Versus same scope act |
+----------------+------------+-------------------+-----------------------+
| Disabled       |   740.606s | 1.000x            | -                     |
| QKV activation |   786.340s | 1.062x /  +6.2%   | 1.000x                |
| QKV Hessian    |   847.753s | 1.145x / +14.5%   | 1.078x / +7.8%        |
| QKV hybrid     |   748.341s | 1.010x /  +1.0%   | 0.952x / -4.8%        |
| MLP activation |   766.342s | 1.035x /  +3.5%   | 1.000x                |
| MLP Hessian    |   791.584s | 1.069x /  +6.9%   | 1.033x / +3.3%        |
| MLP hybrid     |   765.664s | 1.034x /  +3.4%   | 0.999x / -0.1%        |
+----------------+------------+-------------------+-----------------------+
```

The optimized Hessian overhead versus activation remained bounded at `+7.8%` for QKV and `+3.3%` for MLP on the
8B model. Hessian and hybrid execute the same correlated-scoring shapes; the much wider concurrent timing spread
between them is hardware/run noise rather than an algorithmic hybrid speed advantage.

### Current recommendation

Activation remains the safest general default and produced the highest Qwen full-coverage mean. Projection-specific
objective selection is model-dependent: Llama favored QKV Hessian and MLP activation, whereas Qwen nominally favored
QKV activation, clearly favored O activation, and more clearly favored MLP Hessian. For Qwen, choose all-activation
when maximizing the three-metric mean, or global Hessian with a QKVO activation override when avoiding a regression
on any measured task is more important. Global hybrid should not be used. The strongest untested Llama-specific
mixed configuration remains:

```python
scale_search = None
dynamic = {
    r"+:^model\.layers\.\d+\.self_attn\.(?:q_proj|k_proj|v_proj)$": {
        "scale_search": "hessian",
    },
    r"+:^model\.layers\.\d+\.mlp\.(?:gate_proj|up_proj|down_proj)$": {
        "scale_search": "activation",
    },
}
```

This mixed configuration requires a full end-to-end evaluation because the benefits of independently selected
objectives may not compose linearly, and it should not be promoted to a model-independent default.

### Validation

```text
Projection-scoped GPU arms: 7 passed
Optimized quant-only arms:   7 passed
Qwen3 8B scoped GPU arms:   10 passed
Qwen3 8B combined GPU arms:  2 passed
Qwen3 8B global GPU arms:    3 passed
Scale-search unit tests:     32 passed
Ruff:                        passed
git diff --check:            passed
```

The projection-scope unit guards verify disjoint QKV, O, and MLP overrides and prove that the full-coverage Qwen
policy assigns activation to QKVO and Hessian to every MLP projection. The GPU A/B test verifies finite post-quant
ARC and GSM8K results for every arm and supports sharding one arm per GPU through
`GPTQMODEL_SCALE_SEARCH_SCOPE_ARM`.
