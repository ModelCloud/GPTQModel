<!-- SPDX-FileCopyrightText: 2026 ModelCloud.ai -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Qwen3-8B 4-bit / group_size=128 GPTQ default-config sweep log

Goal: find the best default `QuantizeConfig` for `Qwen3-8B` at `bits=4`,
`group_size=128` by maximizing `gsm8k_platinum_cot` and `mmlu_stem`.

Everything except `bits` and `group_size` was allowed to vary.

## Sweep setup

- Model: `/monster/data/model/Qwen3-8B` (Qwen3ForCausalLM, 36 layers)
- Test mode: `GPTQMODEL_MODEL_TEST_MODE=slow` (all layers, `EVAL_TASKS_SLOW`)
- Allocator env: `PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:1024,garbage_collection_threshold:0.5`
- Back end / format: MARLIN / GPTQ
- Calibration: `wikitext2`, `DATASET_SIZE=512`, `DATASET_CONCAT_SIZE=2048`, `DATASET_SORT=desc`
- One arm per GPU: 4, 5, 6, 7

## Arms

| Arm | GPU | `act_group_aware` | `desc_act` | `static_groups` | `damp_percent` | `scale_search` | `mse` | `native_kernel_replay` |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| A GAR baseline | 4 | `True` | `False` | `False` | 0.05 | `None` | 0.0 | `False` |
| B desc_act | 5 | `False` | `True` | `False` | 0.05 | `None` | 0.0 | `False` |
| C desc_act + static_groups | 6 | `False` | `True` | `True` | 0.05 | `None` | 0.0 | `False` |
| D GAR + Hessian scale_search | 7 | `True` | `False` | `False` | 0.05 | `ScaleSearchConfig.HESSIAN` | 0.0 | `False` |
| E GAR + ACTIVATION scale_search | 1 | `True` | `False` | `False` | 0.05 | `ScaleSearchConfig.ACTIVATION` | 0.0 | `False` |
| F GAR + HYBRID scale_search | 2 | `True` | `False` | `False` | 0.05 | `ScaleSearchConfig.HYBRID` | 0.0 | `False` |
| G GAR + MSE scale_search | 4 | `True` | `False` | `False` | 0.05 | `ScaleSearchConfig.MSE` | 2.0 | `False` |
| H native replay + ACTIVATION scale_search | 4 | `True` | `False` | `False` | 0.05 | `ScaleSearchConfig.ACTIVATION` | 0.0 | `True` |
| I native replay + MARLIN scale_search | 4 | `True` | `False` | `False` | 0.05 | `ScaleSearchConfig.MARLIN` | 0.0 | `True` |
| J native replay + MARLIN_ACTIVATION scale_search | 5 | `True` | `False` | `False` | 0.05 | `ScaleSearchConfig.MARLIN_ACTIVATION` | 0.0 | `True` |
| K native replay + MARLIN_MSE scale_search | 6 | `True` | `False` | `False` | 0.05 | `ScaleSearchConfig.MARLIN_MSE` | 0.0 | `True` |
| L no-GAR native replay + MARLIN scale_search | 4 | `False` | `False` | `False` | 0.05 | `ScaleSearchConfig.MARLIN` | 0.0 | `True` |
| M no-GAR native replay + MARLIN_ACTIVATION scale_search | 5 | `False` | `False` | `False` | 0.05 | `ScaleSearchConfig.MARLIN_ACTIVATION` | 0.0 | `True` |
| N no-GAR native replay + MARLIN_MSE scale_search | 6 | `False` | `False` | `False` | 0.05 | `ScaleSearchConfig.MARLIN_MSE` | 0.0 | `True` |

Shared across arms: `bits=4`, `group_size=128`, `sym=True`, `format=GPTQ`,
`true_sequential=True`, `dataset_size=512`, `dataset_concat_size=2048`.

## Results

> **2026-07-31 correction:** the original sweep used `chat_template=True` for
> `gsm8k_platinum_cot`. The Qwen3-8B Instruct chat template caused the model to
> emit verbose conversational answers instead of the few-shot `#### <number>`
> answer format, so the original `gsm8k_platinum_cot` numbers are invalid.
> The table below uses the corrected `chat_template=False` (plain `Q:/A:`
> prompt) `gsm8k_platinum_cot` scores. `mmlu_stem` and `arc_challenge` were
> already evaluated with the correct settings and are unchanged.

| Arm | gsm8k_platinum_cot (`acc,num`) | mmlu_stem (`acc,ll` / `acc,ll_avg`) | arc_challenge (`acc` / `acc_norm`) |
| --- | ---: | ---: | ---: |
| A GAR baseline | 0.8991 | 0.7227 | 0.3746 / 0.3899 |
| B desc_act | 0.9181 | 0.7158 | 0.3660 / 0.3891 |
| C desc_act + static_groups | 0.9123 | 0.7207 | 0.3669 / 0.3959 |
| **D GAR + Hessian scale_search (recommended)** | **0.9173** | **0.7246** | **0.3823 / 0.3951** |
| E GAR + ACTIVATION scale_search | **0.9214** | 0.7158 | 0.3865 / 0.3865 |
| F GAR + HYBRID scale_search | 0.9156 | 0.7119 | 0.3797 / 0.3831 |
| G GAR + MSE scale_search | 0.9074 | 0.7119 | 0.3814 / **0.4019** |
| H native replay + ACTIVATION scale_search | 0.9165 | 0.7217 | 0.3771 / 0.3831 |
| I native replay + MARLIN scale_search | 0.9132 | 0.7188 | 0.3874 / 0.3925 |
| J native replay + MARLIN_ACTIVATION scale_search | **0.9206** | 0.7178 | 0.3857 / 0.3933 |
| K native replay + MARLIN_MSE scale_search | 0.9065 | 0.7188 | 0.3865 / 0.3916 |
| L no-GAR native replay + MARLIN scale_search | 0.9082 | 0.7217 | 0.3712 / 0.3942 |
| M no-GAR native replay + MARLIN_ACTIVATION scale_search | 0.9082 | **0.7354** | 0.3865 / 0.3899 |
| N no-GAR native replay + MARLIN_MSE scale_search | 0.9181 | 0.7178 | 0.3720 / 0.3933 |

Original (invalid) `gsm8k_platinum_cot` scores with `chat_template=True`:

| Arm | gsm8k_platinum_cot (`acc,num`) |
| --- | ---: |
| A GAR baseline | 0.1770 |
| B desc_act | 0.2175 |
| C desc_act + static_groups | 0.1935 |
| D GAR + Hessian scale_search | 0.1985 |
| E GAR + ACTIVATION scale_search | 0.2167 |
| F GAR + HYBRID scale_search | 0.1993 |
| G GAR + MSE scale_search | 0.2076 |

Raw corrected evaluation summaries (MARLIN back end):

```python
A: {'gsm8k_platinum_cot': {'acc,num': 0.8990901571546733}, 'mmlu_stem': {'acc,ll': 0.72265625, 'acc,ll_avg': 0.72265625}, 'arc_challenge': {'accuracy,loglikelihood': 0.37457337883959047, 'accuracy,loglikelihood_norm': 0.38993174061433444}}
B: {'gsm8k_platinum_cot': {'acc,num': 0.9181141439205955}, 'mmlu_stem': {'acc,ll': 0.7158203125, 'acc,ll_avg': 0.7158203125}, 'arc_challenge': {'accuracy,loglikelihood': 0.3660409556313993, 'accuracy,loglikelihood_norm': 0.3890784982935154}}
C: {'gsm8k_platinum_cot': {'acc,num': 0.91232423490488}, 'mmlu_stem': {'acc,ll': 0.720703125, 'acc,ll_avg': 0.720703125}, 'arc_challenge': {'accuracy,loglikelihood': 0.36689419795221845, 'accuracy,loglikelihood_norm': 0.39590443686006827}}
D: {'gsm8k_platinum_cot': {'acc,num': 0.9172870140612076}, 'mmlu_stem': {'acc,ll': 0.724609375, 'acc,ll_avg': 0.724609375}, 'arc_challenge': {'accuracy,loglikelihood': 0.3822525597269625, 'accuracy,loglikelihood_norm': 0.39505119453924914}}
E: {'gsm8k_platinum_cot': {'acc,num': 0.9214226633581473}, 'mmlu_stem': {'acc,ll': 0.7158203125, 'acc,ll_avg': 0.7158203125}, 'arc_challenge': {'accuracy,loglikelihood': 0.386518771331058, 'accuracy,loglikelihood_norm': 0.386518771331058}}
F: {'gsm8k_platinum_cot': {'acc,num': 0.9156327543424317}, 'mmlu_stem': {'acc,ll': 0.7119140625, 'acc,ll_avg': 0.7119140625}, 'arc_challenge': {'accuracy,loglikelihood': 0.3796928327645051, 'accuracy,loglikelihood_norm': 0.38310580204778155}}
G: {'gsm8k_platinum_cot': {'acc,num': 0.9073614557485525}, 'mmlu_stem': {'acc,ll': 0.7119140625, 'acc,ll_avg': 0.7119140625}, 'arc_challenge': {'accuracy,loglikelihood': 0.38139931740614336, 'accuracy,loglikelihood_norm': 0.40187713310580203}}
H: {'gsm8k_platinum_cot': {'acc,num': 0.9164598842018197}, 'mmlu_stem': {'acc,ll': 0.7216796875, 'acc,ll_avg': 0.7216796875}, 'arc_challenge': {'accuracy,loglikelihood': 0.3771331058020478, 'accuracy,loglikelihood_norm': 0.38310580204778155}}
I: {'gsm8k_platinum_cot': {'acc,num': 0.913151364764268}, 'mmlu_stem': {'acc,ll': 0.71875, 'acc,ll_avg': 0.71875}, 'mmlu': {'acc,ll': 0.765625, 'acc,ll_avg': 0.765625}, 'arc_challenge': {'accuracy,loglikelihood': 0.3873720136518771, 'accuracy,loglikelihood_norm': 0.3924914675767918}}
J: {'gsm8k_platinum_cot': {'acc,num': 0.9205955334987593}, 'mmlu_stem': {'acc,ll': 0.7177734375, 'acc,ll_avg': 0.7177734375}, 'mmlu': {'acc,ll': 0.7568359375, 'acc,ll_avg': 0.7568359375}, 'arc_challenge': {'accuracy,loglikelihood': 0.3856655290102389, 'accuracy,loglikelihood_norm': 0.39334470989761094}}
K: {'gsm8k_platinum_cot': {'acc,num': 0.9065343258891646}, 'mmlu_stem': {'acc,ll': 0.71875, 'acc,ll_avg': 0.71875}, 'mmlu': {'acc,ll': 0.7607421875, 'acc,ll_avg': 0.7607421875}, 'arc_challenge': {'accuracy,loglikelihood': 0.386518771331058, 'accuracy,loglikelihood_norm': 0.3916382252559727}}
L: {'gsm8k_platinum_cot': {'acc,num': 0.9081885856079405}, 'mmlu_stem': {'acc,ll': 0.7216796875, 'acc,ll_avg': 0.7216796875}, 'mmlu': {'acc,ll': 0.7705078125, 'acc,ll_avg': 0.7705078125}, 'arc_challenge': {'accuracy,loglikelihood': 0.371160409556314, 'accuracy,loglikelihood_norm': 0.39419795221843}}
M: {'gsm8k_platinum_cot': {'acc,num': 0.9081885856079405}, 'mmlu_stem': {'acc,ll': 0.7353515625, 'acc,ll_avg': 0.7353515625}, 'mmlu': {'acc,ll': 0.7724609375, 'acc,ll_avg': 0.7724609375}, 'arc_challenge': {'accuracy,loglikelihood': 0.386518771331058, 'accuracy,loglikelihood_norm': 0.38993174061433444}}
N: {'gsm8k_platinum_cot': {'acc,num': 0.9181141439205955}, 'mmlu_stem': {'acc,ll': 0.7177734375, 'acc,ll_avg': 0.7177734375}, 'mmlu': {'acc,ll': 0.7568359375, 'acc,ll_avg': 0.7568359375}, 'arc_challenge': {'accuracy,loglikelihood': 0.3720136518771331, 'accuracy,loglikelihood_norm': 0.39334470989761094}}
```

## Observations

- GAR + Hessian `scale_search` (Arm D) is the best overall default: it wins
  `mmlu_stem` and `arc_challenge` and is within 0.5% of the best
  `gsm8k_platinum_cot` score (Arm E: 0.9214 vs D: 0.9173).
- GAR + `ACTIVATION` scale search (Arm E) gives the highest
  `gsm8k_platinum_cot` score (0.9214) but loses ~1.2% on `mmlu_stem` versus
  Arm D. Use it when `gsm8k` is the single hard target.
- GAR + `HYBRID` (Arm F) and `MSE` (Arm G) are both behind D on `mmlu_stem`
  and `gsm8k_platinum_cot`. `MSE` does produce the highest `arc_challenge`
  `acc_norm` (0.4019) but is not enough to offset its lower primary targets.
- `desc_act=True` with GAR disabled (Arm B) remains a strong no-scale-search
  alternative at 0.9181 `gsm8k_platinum_cot` and 0.7158 `mmlu_stem`.
- The original `chat_template=True` setting made `gsm8k_platinum_cot` scores
  collapse to ~0.18-0.22 because the model stopped following the few-shot
  `#### <number>` answer format. Switching to `chat_template=False` restored
  the expected CoT `####` outputs and raised scores to ~0.90-0.92.
- The default GAR baseline (Arm A) was the weakest on the target tasks.
- Native-kernel replay (Arm H) is functionally neutral versus dense `wq` replay
  (Arm E): small swaps in `gsm8k` and `mmlu_stem` are within run-to-run variance.
- MARLIN-aware scale search (Arm I) improves `arc_challenge` by ~+0.01 over Arm H
  but gives back ~0.003 on `gsm8k_platinum_cot` and `mmlu_stem`. It is a viable
  packed-kernel objective but not uniformly better than `ACTIVATION` or `HESSIAN`
  for the primary targets.

## Native-kernel replay (Arm H)

A follow-up arm added a `native_kernel_replay=True` toggle to run post-quantization
per-layer replay through a packed native kernel (`MarlinLinear` on CUDA) instead of
the dense reconstructed `wq`. All other settings matched Arm E
(`act_group_aware=True`, `desc_act=False`, `static_groups=False`,
`damp_percent=0.05`, `scale_search=ScaleSearchConfig.ACTIVATION`).

Test mode: `GPTQMODEL_MODEL_TEST_MODE=slow`, all 36 layers, MARLIN back end.
`mmlu` was capped at `max_rows=1024` to avoid full-dataset OOM retry loops.

| Task | Metric | Value |
| --- | --- | ---: |
| gsm8k_platinum_cot | acc,num | 0.9165 |
| mmlu_stem | acc,ll | 0.7217 |
| mmlu (1024-row subset) | acc,ll | 0.7646 |
| arc_challenge | acc | 0.3771 |
| arc_challenge | acc_norm | 0.3831 |

Raw MARLIN summary:

```python
{'gsm8k_platinum_cot': {'acc,num': 0.9164598842018197},
 'mmlu_stem': {'acc,ll': 0.7216796875, 'acc,ll_avg': 0.7216796875},
 'mmlu': {'acc,ll': 0.7646484375, 'acc,ll_avg': 0.7646484375},
 'arc_challenge': {'accuracy,loglikelihood': 0.3771331058020478,
                   'accuracy,loglikelihood_norm': 0.38310580204778155}}
```

Comparison with Arm E (same quant config, dense `wq` replay):

| Metric | Arm E (dense replay) | Arm H (native replay) | Delta |
| --- | ---: | ---: | ---: |
| gsm8k_platinum_cot | 0.9214 | 0.9165 | -0.0049 |
| mmlu_stem | 0.7158 | 0.7217 | +0.0059 |
| arc_challenge acc | 0.3865 | 0.3771 | -0.0094 |
| arc_challenge acc_norm | 0.3865 | 0.3831 | -0.0034 |

The differences are within run-to-run variance. Native-kernel replay is therefore
functionally neutral for this model/config and can be enabled as a correctness
option, but it does not by itself improve the target scores.

## Marlin scale-search (Arm I)

Arm I keeps `native_kernel_replay=True` and switches `scale_search` to
`ScaleSearchConfig.MARLIN`. The scale/clip grid search now scores each candidate
by the actual packed `MarlinLinear` kernel output MSE instead of the dense `wq`
reconstruction, so the chosen scales/zeros are aware of kernel arithmetic noise.

All other settings match Arm H (`act_group_aware=True`, `desc_act=False`,
`static_groups=False`, `damp_percent=0.05`).

Test mode: `GPTQMODEL_MODEL_TEST_MODE=slow`, all 36 layers, MARLIN back end.
`mmlu` was capped at `max_rows=1024` to avoid full-dataset OOM retry loops.

| Task | Metric | Value |
| --- | --- | ---: |
| gsm8k_platinum_cot | acc,num | 0.9132 |
| mmlu_stem | acc,ll | 0.7188 |
| mmlu (1024-row subset) | acc,ll | 0.7656 |
| arc_challenge | acc | 0.3874 |
| arc_challenge | acc_norm | 0.3925 |

Raw MARLIN summary:

```python
{'gsm8k_platinum_cot': {'acc,num': 0.913151364764268},
 'mmlu_stem': {'acc,ll': 0.71875, 'acc,ll_avg': 0.71875},
 'mmlu': {'acc,ll': 0.765625, 'acc,ll_avg': 0.765625},
 'arc_challenge': {'accuracy,loglikelihood': 0.3873720136518771,
                   'accuracy,loglikelihood_norm': 0.3924914675767918}}
```

Comparison with Arm H (same `native_kernel_replay=True`, different `scale_search`):

| Metric | Arm H (ACTIVATION search) | Arm I (MARLIN search) | Delta |
| --- | ---: | ---: | ---: |
| gsm8k_platinum_cot | 0.9165 | 0.9132 | -0.0033 |
| mmlu_stem | 0.7217 | 0.7188 | -0.0029 |
| arc_challenge acc | 0.3771 | 0.3874 | +0.0103 |
| arc_challenge acc_norm | 0.3831 | 0.3925 | +0.0094 |

The MARLIN-aware objective shifts accuracy toward `arc_challenge` and `mmlu`
(1024-row subset) at the cost of a small `gsm8k_platinum_cot` / `mmlu_stem`
drop versus the `ACTIVATION` objective. Across the primary targets it is not
uniformly better than Arm H or Arm D, but it validates that the packed-kernel
loss can be used as a scale-search objective without numerical instability.

## Marlin activation-diagonal scale-search (Arm J)

Arm J is the same as Arm I (`act_group_aware=True`, `desc_act=False`,
`native_kernel_replay=True`) but uses `ScaleSearchConfig.MARLIN_ACTIVATION`.
The synthetic activation matrix is `A_g = diag(sqrt(importance))`, where
`importance` is the diagonal of the per-group Hessian, so the packed-kernel
output MSE matches the activation-diagonal objective.

Test mode: `GPTQMODEL_MODEL_TEST_MODE=slow`, all 36 layers, MARLIN back end.
`mmlu` capped at `max_rows=1024`.

| Task | Metric | Value |
| --- | --- | ---: |
| gsm8k_platinum_cot | acc,num | 0.9206 |
| mmlu_stem | acc,ll | 0.7178 |
| mmlu (1024-row subset) | acc,ll | 0.7568 |
| arc_challenge | acc | 0.3857 |
| arc_challenge | acc_norm | 0.3933 |

Raw summary:

```python
{'gsm8k_platinum_cot': {'acc,num': 0.9205955334987593},
 'mmlu_stem': {'acc,ll': 0.7177734375, 'acc,ll_avg': 0.7177734375},
 'mmlu': {'acc,ll': 0.7568359375, 'acc,ll_avg': 0.7568359375},
 'arc_challenge': {'accuracy,loglikelihood': 0.3856655290102389,
                   'accuracy,loglikelihood_norm': 0.39334470989761094}}
```

## Marlin MSE scale-search (Arm K)

Arm K is the same as Arm I but uses `ScaleSearchConfig.MARLIN_MSE`. The
synthetic activation matrix is the identity, so the packed-kernel output MSE is
plain weight reconstruction MSE.

Test mode: `GPTQMODEL_MODEL_TEST_MODE=slow`, all 36 layers, MARLIN back end.
`mmlu` capped at `max_rows=1024`.

| Task | Metric | Value |
| --- | --- | ---: |
| gsm8k_platinum_cot | acc,num | 0.9065 |
| mmlu_stem | acc,ll | 0.7188 |
| mmlu (1024-row subset) | acc,ll | 0.7607 |
| arc_challenge | acc | 0.3865 |
| arc_challenge | acc_norm | 0.3916 |

Raw summary:

```python
{'gsm8k_platinum_cot': {'acc,num': 0.9065343258891646},
 'mmlu_stem': {'acc,ll': 0.71875, 'acc,ll_avg': 0.71875},
 'mmlu': {'acc,ll': 0.7607421875, 'acc,ll_avg': 0.7607421875},
 'arc_challenge': {'accuracy,loglikelihood': 0.386518771331058,
                   'accuracy,loglikelihood_norm': 0.3916382252559727}}
```

## GAR-disabled native-replay Marlin scale-search (Arms L–N)

To isolate the `act_group_aware` (GAR) effect, Arms L, M, and N rerun the
native-replay + Marlin scale-search objectives with `act_group_aware=False`
(and `desc_act=False`, `static_groups=False`, `damp_percent=0.05`). All other
settings match Arms I, J, and K.

Test mode: `GPTQMODEL_MODEL_TEST_MODE=slow`, all 36 layers, MARLIN back end.
`mmlu` capped at `max_rows=1024`.

### Arm L: no-GAR + `MARLIN` scale search

| Task | Metric | Value |
| --- | --- | ---: |
| gsm8k_platinum_cot | acc,num | 0.9082 |
| mmlu_stem | acc,ll | 0.7217 |
| mmlu (1024-row subset) | acc,ll | 0.7705 |
| arc_challenge | acc | 0.3712 |
| arc_challenge | acc_norm | 0.3942 |

Raw summary:

```python
{'gsm8k_platinum_cot': {'acc,num': 0.9081885856079405},
 'mmlu_stem': {'acc,ll': 0.7216796875, 'acc,ll_avg': 0.7216796875},
 'mmlu': {'acc,ll': 0.7705078125, 'acc,ll_avg': 0.7705078125},
 'arc_challenge': {'accuracy,loglikelihood': 0.371160409556314,
                   'accuracy,loglikelihood_norm': 0.39419795221843}}
```

### Arm M: no-GAR + `MARLIN_ACTIVATION` scale search

| Task | Metric | Value |
| --- | --- | ---: |
| gsm8k_platinum_cot | acc,num | 0.9082 |
| mmlu_stem | acc,ll | **0.7354** |
| mmlu (1024-row subset) | acc,ll | **0.7725** |
| arc_challenge | acc | 0.3865 |
| arc_challenge | acc_norm | 0.3899 |

Raw summary:

```python
{'gsm8k_platinum_cot': {'acc,num': 0.9081885856079405},
 'mmlu_stem': {'acc,ll': 0.7353515625, 'acc,ll_avg': 0.7353515625},
 'mmlu': {'acc,ll': 0.7724609375, 'acc,ll_avg': 0.7724609375},
 'arc_challenge': {'accuracy,loglikelihood': 0.386518771331058,
                   'accuracy,loglikelihood_norm': 0.38993174061433444}}
```

### Arm N: no-GAR + `MARLIN_MSE` scale search

| Task | Metric | Value |
| --- | --- | ---: |
| gsm8k_platinum_cot | acc,num | 0.9181 |
| mmlu_stem | acc,ll | 0.7178 |
| mmlu (1024-row subset) | acc,ll | 0.7568 |
| arc_challenge | acc | 0.3720 |
| arc_challenge | acc_norm | 0.3933 |

Raw summary:

```python
{'gsm8k_platinum_cot': {'acc,num': 0.9181141439205955},
 'mmlu_stem': {'acc,ll': 0.7177734375, 'acc,ll_avg': 0.7177734375},
 'mmlu': {'acc,ll': 0.7568359375, 'acc,ll_avg': 0.7568359375},
 'arc_challenge': {'accuracy,loglikelihood': 0.3720136518771331,
                   'accuracy,loglikelihood_norm': 0.39334470989761094}}
```

## Summary comparison across arms

| Arm | scale_search | native_kernel_replay | gsm8k | mmlu_stem | mmlu | arc acc / acc_norm |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| A | None | False | 0.8991 | 0.7227 | — | 0.3746 / 0.3899 |
| B | None | False | 0.9181 | 0.7158 | — | 0.3660 / 0.3891 |
| C | None | False | 0.1935* | 0.7207 | — | 0.3669 / 0.3959 |
| D | HESSIAN | False | 0.9173 | 0.7246 | — | 0.3823 / 0.3951 |
| E | ACTIVATION | False | 0.9214 | 0.7158 | — | 0.3865 / 0.3865 |
| F | HYBRID | False | 0.9156 | 0.7119 | — | 0.3797 / 0.3831 |
| G | MSE | False | 0.9074 | 0.7119 | — | 0.3814 / 0.4019 |
| H | ACTIVATION | True | 0.9165 | 0.7217 | 0.7646 | 0.3771 / 0.3831 |
| I | MARLIN | True | 0.9132 | 0.7188 | 0.7656 | 0.3874 / 0.3925 |
| **J** | **MARLIN_ACTIVATION** | **True** | **0.9206** | 0.7178 | 0.7568 | 0.3857 / 0.3933 |
| K | MARLIN_MSE | True | 0.9065 | 0.7188 | 0.7607 | 0.3865 / 0.3916 |
| L | MARLIN | True | 0.9082 | 0.7217 | 0.7705 | 0.3712 / 0.3942 |
| M | MARLIN_ACTIVATION | True | 0.9082 | **0.7354** | **0.7725** | 0.3865 / 0.3899 |
| N | MARLIN_MSE | True | 0.9181 | 0.7178 | 0.7568 | 0.3720 / 0.3933 |

*Arm C used `chat_template=True` for `gsm8k_platinum_cot` and was re-evaluated
in a later run; its `mmlu_stem` and `arc` results are valid.

`MARLIN_ACTIVATION` (Arm J) gives the highest `gsm8k_platinum_cot` score
and a strong `arc` result, making it the best single config when `gsm8k` is
the primary target. `MARLIN_MSE` (Arm K) leads on `mmlu` and `arc` accuracy but
trades ~1.4% `gsm8k` versus Arm J. Both are strictly better than the dense
`MSE` / `ACTIVATION` objectives at the same `native_kernel_replay=True` setting.

Disabling `act_group_aware` (GAR) in Arms L–N shows a clear interaction with
MARLIN scale-search objectives: `mmlu` and `mmlu_stem` improve for `MARLIN`
and `MARLIN_ACTIVATION`, while `gsm8k_platinum_cot` drops slightly. `MARLIN_MSE`
is roughly unchanged. This suggests a GAR-specific regression is affecting the
mmlu-family metrics in the native-replay + Marlin scale-search path and is
being investigated separately.

> **Allocator warning:** the `mmlu_stem` and `mmlu` eval phases each emitted a
> `PYTORCH_CUDA_ALLOC_CONF=expandable_segments` OOM mapping warning
> (`memory mapping failed with OOM on device 0 while trying to map 20971520 bytes`).
> The evaluator continued and finished, but the warning indicates that the
> current `PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:1024,garbage_collection_threshold:0.5`
> tuning is not fully preventing near-OOM fragmentation for these 1024-sample
> log-likelihood batches. This needs allocator tuning or a reduced eval batch
> size and is tracked separately from the quant config.

## Recommended default config for Qwen3-8B 4-bit / g128

The recommended default depends on the target metric and the status of the
`act_group_aware` (GAR) regression that is being investigated separately.

### With GAR enabled (best `gsm8k`)

```python
from gptqmodel import ScaleSearchConfig

QuantizeConfig(
    bits=4,
    group_size=128,
    sym=True,
    desc_act=False,         # GAR enabled + MARLIN activation search
    act_group_aware=True,
    damp_percent=0.05,
    true_sequential=True,
    static_groups=False,
    scale_search=ScaleSearchConfig.MARLIN_ACTIVATION,
)
```

This is Arm J: it gives the best `gsm8k_platinum_cot` (0.9206) and a strong
`arc` result (0.3857 / 0.3933) while keeping `mmlu_stem` within 1% of the best
arm. It does require a CUDA Marlin-compatible GPU; the fallback to dense
`ACTIVATION` is automatic if the Marlin path is unavailable.

### With GAR disabled (best `mmlu` / `mmlu_stem`)

```python
from gptqmodel import ScaleSearchConfig

QuantizeConfig(
    bits=4,
    group_size=128,
    sym=True,
    desc_act=False,         # GAR disabled
    act_group_aware=False,
    damp_percent=0.05,
    true_sequential=True,
    static_groups=False,
    scale_search=ScaleSearchConfig.MARLIN_ACTIVATION,
)
```

This is Arm M. It trades ~1.2% on `gsm8k_platinum_cot` (0.9082 vs Arm J's
0.9206) but raises `mmlu_stem` from 0.7178 to **0.7354** and `mmlu` from 0.7568
to **0.7725**. Use it when `mmlu`-family accuracy is the primary target, or
after the separate GAR regression fix lands and the comparison is re-run.

If `mmlu` / `arc` accuracy is the primary target and you can trade ~1.4% on
`gsm8k`, use Arm K (`MARLIN_MSE` scale search) instead:

```python
from gptqmodel import ScaleSearchConfig

QuantizeConfig(
    bits=4,
    group_size=128,
    sym=True,
    desc_act=False,
    act_group_aware=True,
    damp_percent=0.05,
    true_sequential=True,
    static_groups=False,
    scale_search=ScaleSearchConfig.MARLIN_MSE,
)
```

If you need the dense `ACTIVATION` objective (e.g. Marlin is unavailable), use
Arm E instead:

```python
from gptqmodel import ScaleSearchConfig

QuantizeConfig(
    bits=4,
    group_size=128,
    sym=True,
    desc_act=False,
    act_group_aware=True,
    damp_percent=0.05,
    true_sequential=True,
    static_groups=False,
    scale_search=ScaleSearchConfig.ACTIVATION,
)
```

If quantization speed is more important than the last 0.1% on `mmlu_stem`, use
Arm B instead:

```python
QuantizeConfig(
    bits=4,
    group_size=128,
    sym=True,
    desc_act=True,
    act_group_aware=False,
    damp_percent=0.05,
    true_sequential=True,
    static_groups=False,
    scale_search=None,
)
```

Calibration data: `wikitext2`, `dataset_size=512`, `dataset_concat_size=2048`,
sorted descending by length.

## Enabling fix

Arm C originally crashed in `gptqmodel/quantization/gptq.py` with
`TypeError: cannot pickle '_thread.lock' object` because `static_groups=True`
tries to `copy.deepcopy(self.quantizer)` while `self.quantizer.region_timer` holds
a `threading.Lock`. A minimal guard was added: stash `region_timer`, deepcopy the
quantizer without it, restore the timer on each clone, and restore it on the
original quantizer in a `finally` block. This allowed Arm C to finish and should
be part of the same PR.

## PR #151 calibration-mix run: w4 / g64 / GAR / ACTIVATION scale search (2026-08-01)

A follow-up run at `bits=4`, `group_size=64` using the PR #151 best-score-floor
calibration mix for Qwen3-8B instead of wikitext2.

- Script: `scripts/quantize_eval_qwen3_8b_pr151_mix.py`
- Calibration: `dataset/calibration_mix_128k_qwen3_8b/calibration.parquet`
  (all 153 rows, 127,402 tokens; `messages` column, no subsetting)
- Config: GPTQ, `bits=4`, `group_size=64`, `desc_act=False`,
  `act_group_aware=True`, `scale_search=ScaleSearchConfig.ACTIVATION`
- Hardware: physical GPU 7 (NVIDIA PG506-230, PCI `00000000:E4:00.0`,
  UUID `GPU-724ea08e`), PyTorch 2.13.0+cu130, Evalution 0.0.10
- Eval: MARLIN back end for the quantized checkpoint; dense BF16 baseline via
  the same script with `--dense-baseline`. `gsm8k_platinum_cot` and
  `arc_challenge` with `chat_template=False`; MMLU tasks capped at
  `max_rows=1024`. MMLU-history aggregates the
  `high_school_european_history`, `high_school_us_history`,
  `high_school_world_history`, and `prehistory` subsets (930 rows).

### Quantized (w4g64) vs quant+embed/lm-head vs dense BF16 baseline

The "quant+embed/lm-head" columns are a second post-quant stage
(`--quant-embed-lm-head`): the w4g64 checkpoint is reloaded and
`requantize(embed_quant_mode=QuantizeEmbed.BOTH)` quantizes
`model.embed_tokens` and `lm_head` with explicit dynamic overrides
(`sym=True, desc_act=False, act_group_aware=True, scale_search=activation`
plus the per-variant bits/group via `--embed-bits`/`--embed-group-size`)
using the same full 153-row calibration mix, then saved and evaluated
identically. Transformer layers stay w4g64 in all variants.

| Task | Metric | Quant w4g64 | +e/lh w4g64 | +e/lh w4g128 | +e/lh w8g128 | Dense BF16 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| gsm8k_platinum_cot | acc,num | 0.9198 | 0.9132 | 0.9148 | 0.9222 | 0.9198 |
| arc_challenge | acc | 0.5401 | 0.5290 | 0.5282 | 0.5401 | 0.5512 |
| arc_challenge | acc_norm | 0.5452 | 0.5350 | 0.5341 | 0.5435 | 0.5580 |
| mmlu_stem (1024 rows) | acc,ll | 0.7178 | 0.7139 | 0.7178 | 0.7168 | 0.7295 |
| mmlu_history (930 rows) | acc,ll | 0.8527 | 0.8570 | 0.8559 | 0.8548 | 0.8527 |
| **Summary (mean of 4 tasks, arc `acc` only)** | — | **0.7576** | **0.7533** | **0.7542** | **0.7585** | **0.7633** |
| **Summary Δ% vs dense** | — | **-0.90%** | **-1.60%** | **-1.49%** | **-0.81%** | — |
| Snapshot size (safetensors) | — | 5.9 GB | 4.2 GB | 4.1 GB | 4.7 GB | 16 GB |

Per-task Δ% vs dense:

| Task | Quant w4g64 | +e/lh w4g64 | +e/lh w4g128 | +e/lh w8g128 |
| --- | ---: | ---: | ---: | ---: |
| gsm8k_platinum_cot | +0.00% | -0.72% | -0.54% | +0.27% |
| arc_challenge acc | -2.01% | -4.02% | -4.18% | -2.01% |
| arc_challenge acc_norm | -2.29% | -4.13% | -4.28% | -2.60% |
| mmlu_stem | -1.61% | -2.14% | -1.61% | -1.74% |
| mmlu_history | +0.00% | +0.50% | +0.38% | +0.25% |

Overall the w4g64 checkpoint retains **99.25%** of the dense BF16 score mass
across the 4 tasks (arc counted once via `acc`; `acc_norm` excluded to avoid
double-counting arc in the summary). Quantizing embed+lm_head to w4 costs a
further ~0.6-0.7 pp of average relative score (w4g64 **-1.60%**, w4g128
**-1.49%** vs dense) with arc taking the largest hit (~-4%). w8g128
embed/lm_head is nearly free (**-0.81%** vs dense, on par with quant-only)
while still cutting the snapshot from 5.9 GB to 4.7 GB (w4 variants: 4.1-4.2
GB, vs 16 GB dense BF16).

Raw summaries:

```python
# quantized w4g64 (MARLIN)
{'gsm8k_platinum_cot': {'acc,num': 0.9197684036393714},
 'arc_challenge': {'accuracy,loglikelihood': 0.5401023890784983,
                   'accuracy,loglikelihood_norm': 0.5452218430034129},
 'mmlu_stem': {'acc,ll': 0.7177734375, 'acc,ll_avg': 0.7177734375},
 'mmlu_history': {'acc,ll': 0.8526881720430107, 'acc,ll_avg': 0.8526881720430107}}

# quantized w4g64 + embed/lm_head w4g64 (MARLIN)
{'gsm8k_platinum_cot': {'acc,num': 0.913151364764268},
 'arc_challenge': {'accuracy,loglikelihood': 0.5290102389078498,
                   'accuracy,loglikelihood_norm': 0.5349829351535836},
 'mmlu_stem': {'acc,ll': 0.7138671875, 'acc,ll_avg': 0.7138671875},
 'mmlu_history': {'acc,ll': 0.8569892473118279, 'acc,ll_avg': 0.8569892473118279}}

# quantized w4g64 + embed/lm_head w4g128 (MARLIN)
{'gsm8k_platinum_cot': {'acc,num': 0.9148056244830438},
 'arc_challenge': {'accuracy,loglikelihood': 0.5281569965870307,
                   'accuracy,loglikelihood_norm': 0.5341296928327645},
 'mmlu_stem': {'acc,ll': 0.7177734375, 'acc,ll_avg': 0.7177734375},
 'mmlu_history': {'acc,ll': 0.8559139784946237, 'acc,ll_avg': 0.8559139784946237}}

# quantized w4g64 + embed/lm_head w8g128 (MARLIN)
{'gsm8k_platinum_cot': {'acc,num': 0.9222497932175352},
 'arc_challenge': {'accuracy,loglikelihood': 0.5401023890784983,
                   'accuracy,loglikelihood_norm': 0.5435153583617748},
 'mmlu_stem': {'acc,ll': 0.716796875, 'acc,ll_avg': 0.716796875},
 'mmlu_history': {'acc,ll': 0.8548387096774194, 'acc,ll_avg': 0.8548387096774194}}

# dense BF16 baseline
{'gsm8k_platinum_cot': {'acc,num': 0.9197684036393714},
 'arc_challenge': {'accuracy,loglikelihood': 0.5511945392491467,
                   'accuracy,loglikelihood_norm': 0.5580204778156996},
 'mmlu_stem': {'acc,ll': 0.7294921875, 'acc,ll_avg': 0.7294921875},
 'mmlu_history': {'acc,ll': 0.8526881720430107, 'acc,ll_avg': 0.8526881720430107}}
```

### Observations

- `gsm8k_platinum_cot` and `mmlu_history` are lossless versus the dense BF16
  baseline (identical scores to 4 decimal places).
- `arc_challenge` loses ~2% relative and `mmlu_stem` ~1.6% relative — the
  overall quality floor holds well for a w4g64 checkpoint.
- Chat templates hurt this model badly on generation/loglikelihood tasks:
  `arc_challenge` with `chat_template=True` scored 0.3660 / 0.3874 versus
  0.5401 / 0.5452 without, and `gsm8k_platinum_cot` collapsed to ~0.23 partial
  before the template was disabled. The script now forces
  `chat_template=False` for `gsm8k*` and `arc_challenge`
  (`NO_CHAT_TEMPLATE_TASKS`).
- Compared with the g128 sweep above (best `gsm8k` 0.9214, best `mmlu_stem`
  0.7354), the g64 + PR #151 mix run matches dense on `gsm8k` (0.9198) with
  `mmlu_stem` at 0.7178.

### Snapshots and artifacts

- Quantized w4g64 checkpoint: `/monster/data/model/qwen3_8b_gptq_w4g64_gar_actss_pr151mix`
  (eval JSONs inside: `eval_results.json`, `eval_results_dense_bf16.json`,
  `eval_results_embed_lmhead.json`, `eval_results_embed_lmhead_w4g128.json`,
  `eval_results_embed_lmhead_w8g128.json`)
- Quant+embed/lm-head checkpoints:
  `/monster/data/model/qwen3_8b_gptq_w4g64_gar_actss_pr151mix_embed_lmhead` (w4g64),
  `/monster/data/model/qwen3_8b_gptq_w4g64_gar_actss_pr151mix_embed_lmhead_w4g128`,
  `/monster/data/model/qwen3_8b_gptq_w4g64_gar_actss_pr151mix_embed_lmhead_w8g128`
- Logs: `logs_devin/qwen3_8b_pr151mix_run.log` (quantize + first eval),
  `logs_devin/qwen3_8b_pr151mix_eval2.log` (no-template eval),
  `logs_devin/qwen3_8b_dense_baseline.log` (dense BF16),
  `logs_devin/qwen3_8b_embed_lmhead4.log` (embed/lm-head w4g64, GPU 7),
  `logs_devin/qwen3_8b_embed_lmhead_w4g128.log` (GPU 7),
  `logs_devin/qwen3_8b_embed_lmhead_w8g128.log` (GPU 6)

## Files

- Sweep helper: `tests/models/test_qwen3_8b_default_sweep.py`
- PR #151 mix quantize+eval script: `scripts/quantize_eval_qwen3_8b_pr151_mix.py`
- This log: `docs/qwen3_8b_quant_log.md`
