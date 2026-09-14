# Matched GSM8K Platinum: Llama 3.2 1B Instruct

All 1,209 test questions, eight-shot CoT, chat template, greedy generation, 256 new-token limit,
FP16/eager, batch size 32, seed 7. Exact rendered prompts, targets and input IDs match.

| Arm | Correct | Accuracy | Invalid answers |
|---|---:|---:|---:|
| dense | 593/1209 | 49.0488% | 0 |
| baseline | 460/1209 | 38.0480% | 0 |
| staged | 403/1209 | 33.3333% | 1 |

The W4/group128 control uses the staged-path GPTQ initializer with GSQ disabled.
The treatment adds staged adamw training and learned scales (10 attention/MLP epochs, 2000 Q/K updates). Both use 128 calibration
documents and are experimental; this is not the package-default GPTQ recipe or full paper reproduction.

```json
{
  "arms": {
    "dense": {
      "correct": 593,
      "total": 1209,
      "accuracy": 0.4904880066170389,
      "invalid": 0,
      "run_sha256": "6850f45259b3cada0a81d2a4caafee69304e97f10bfab6c6c69350e0925189ee",
      "raw_sha256": "0985579f54044682fa8b6fb6237f261b979316ea6d07f7ffd860373528e05a2a",
      "qvq_commit": "5cc76af5c715f6089663d00e5cc5cefb61445afd",
      "evaluation_source_sha256": "7d571319fdfd7eb72fdc9a68861b0977b7604c4d01273a9827b3a485ee2ef78d",
      "seconds": 185.9880379885435,
      "model": "/monster/data/model/Llama-3.2-1B-Instruct"
    },
    "baseline": {
      "correct": 460,
      "total": 1209,
      "accuracy": 0.380479735318445,
      "invalid": 0,
      "run_sha256": "17d8db5cfb69b9139ef49c86c8807383e5e3bed8064948f757a5d9d94727fbe5",
      "raw_sha256": "a49cba24a69ad975c50a75631c358705fb0a31e79c28aec490e69d966a0fe2a1",
      "qvq_commit": "6c4f2d825b2e9a79ffc2f46f1fc4191b23548a25",
      "evaluation_source_sha256": "7d571319fdfd7eb72fdc9a68861b0977b7604c4d01273a9827b3a485ee2ef78d",
      "seconds": 199.02333253342658,
      "model": "/root/polly-work/qvq-gsq/artifacts/gsq-staged/full-model-w4-nm128-signed-baseline-v1/model"
    },
    "staged": {
      "correct": 403,
      "total": 1209,
      "accuracy": 0.3333333333333333,
      "invalid": 1,
      "run_sha256": "8e41447031f103c00e9d4b955289e6dfb38dc940708a8e63a62b0624a9cae1b3",
      "raw_sha256": "d170ac1e55a8c3c2bafcbfe6cb32f0c1c476724e036d500eaafd5544215f0c2e",
      "qvq_commit": "6c4f2d825b2e9a79ffc2f46f1fc4191b23548a25",
      "evaluation_source_sha256": "7d571319fdfd7eb72fdc9a68861b0977b7604c4d01273a9827b3a485ee2ef78d",
      "seconds": 195.47891049832106,
      "model": "/root/polly-work/qvq-gsq/artifacts/gsq-staged/full-model-w4-nm128-signed-adamw-e10-v1/model"
    }
  },
  "quantization_recipe": {
    "bits": 4,
    "group_size": 128,
    "source_model": "/monster/data/model/Llama-3.2-1B-Instruct"
  },
  "reused_dense_reference": true,
  "training_configs": {
    "baseline": {
      "enabled": false,
      "initializer": "gptq_signed",
      "seed": 7,
      "epochs": 5,
      "batch_size": 64,
      "microbatch_size": 16,
      "qk_steps": 2000,
      "damp_percent": 0.01,
      "assignment_lr": 0.0001,
      "scale_lr": 5e-05,
      "weight_decay": 1.0,
      "betas": [
        0.9,
        0.95
      ],
      "temperature": [
        2.0,
        0.05
      ],
      "multiplier": [
        100.0,
        500.0
      ],
      "warmup_steps": 0,
      "min_lr": 0.1,
      "decay": "cosine"
    },
    "staged": {
      "enabled": true,
      "initializer": "gptq_signed",
      "optimizer": "adamw",
      "seed": 7,
      "epochs": 10,
      "batch_size": 64,
      "microbatch_size": 16,
      "qk_steps": 2000,
      "damp_percent": 0.01,
      "assignment_lr": 0.0001,
      "scale_lr": 5e-05,
      "weight_decay": 1.0,
      "betas": [
        0.9,
        0.95
      ],
      "temperature": [
        2.0,
        0.05
      ],
      "multiplier": [
        100.0,
        500.0
      ],
      "warmup_steps": 0,
      "min_lr": 0.1,
      "decay": "cosine"
    }
  },
  "paired_staged_minus_baseline": -0.04714640198511166,
  "calibration_samples": 128,
  "calibration_inputs_sha256": "893d981d8398efe64e111a22a6996f3e23f073b6460f124fc061097c8bb19dda",
  "paired_bootstrap_ci95": [
    -0.07361455748552523,
    -0.021505376344086023
  ],
  "bootstrap_seed": 7,
  "bootstrap_draws": 10000,
  "baseline_only_correct": 155,
  "staged_only_correct": 98,
  "exact_prompts_targets_and_input_ids": true,
  "prompt_ids_sha256": "3a66bc49447c3e335051f131323f18a0444202de2cd828d4fb842fda4abd73cb"
}
```
