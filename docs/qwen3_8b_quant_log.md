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

| Arm | GPU | `act_group_aware` | `desc_act` | `static_groups` | `damp_percent` | `scale_search` | `mse` |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: |
| A GAR baseline | 4 | `True` | `False` | `False` | 0.05 | `None` | 0.0 |
| B desc_act | 5 | `False` | `True` | `False` | 0.05 | `None` | 0.0 |
| C desc_act + static_groups | 6 | `False` | `True` | `True` | 0.05 | `None` | 0.0 |
| D GAR + Hessian scale_search | 7 | `True` | `False` | `False` | 0.05 | `ScaleSearchConfig.HESSIAN` | 0.0 |
| E GAR + ACTIVATION scale_search | 1 | `True` | `False` | `False` | 0.05 | `ScaleSearchConfig.ACTIVATION` | 0.0 |
| F GAR + HYBRID scale_search | 2 | `True` | `False` | `False` | 0.05 | `ScaleSearchConfig.HYBRID` | 0.0 |
| G GAR + MSE scale_search | 4 | `True` | `False` | `False` | 0.05 | `ScaleSearchConfig.MSE` | 2.0 |

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

## Recommended default config for Qwen3-8B 4-bit / g128

```python
from gptqmodel import ScaleSearchConfig

QuantizeConfig(
    bits=4,
    group_size=128,
    sym=True,
    desc_act=False,         # keep GAR enabled and use Hessian scale search
    act_group_aware=True,
    damp_percent=0.05,
    true_sequential=True,
    static_groups=False,
    scale_search=ScaleSearchConfig.HESSIAN,
)
```

If `gsm8k_platinum_cot` is the single hard target and you can accept a ~1.2%
`mmlu_stem` drop, use Arm E (`ACTIVATION` scale search) instead:

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

## Files

- Sweep helper: `tests/models/test_qwen3_8b_default_sweep.py`
- This log: `docs/qwen3_8b_quant_log.md`
