# Matched GSM8K Platinum: Llama 3.2 1B Instruct

All 1,209 test questions, eight-shot CoT, chat template, greedy generation, 256 new-token limit,
FP16/eager, batch size 32, seed 7. Exact rendered prompts, targets and input IDs match.

| Arm | Correct | Accuracy | Invalid answers |
|---|---:|---:|---:|
| dense | 593/1209 | 49.0488% | 0 |
| baseline | 0/1209 | 0.0000% | 1170 |
| staged | 20/1209 | 1.6543% | 482 |

The W2/group128 control uses the staged-path GPTQ initializer with GSQ disabled.
The treatment adds staged Lion training and learned scales. Both use 128 calibration
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
      "seconds": 185.9880379885435,
      "model": "/monster/data/model/Llama-3.2-1B-Instruct"
    },
    "baseline": {
      "correct": 0,
      "total": 1209,
      "accuracy": 0.0,
      "invalid": 1170,
      "run_sha256": "b07853cb9d87511c0d31fe1e9dbf341ca7fe449a50ab0e2dbbf9c8e2a69a53eb",
      "raw_sha256": "5f3a03b37c203bd719b1b73266bc8854e711886d6fcb367eb47758d24ec2c554",
      "seconds": 229.62789281550795,
      "model": "/root/polly-work/qvq-gsq/artifacts/gsq-staged/full-model-w2-nm128-signed-baseline-v1/model"
    },
    "staged": {
      "correct": 20,
      "total": 1209,
      "accuracy": 0.016542597187758478,
      "invalid": 482,
      "run_sha256": "fae1b3eb85a00db515f03208fc344858713b49c187199c1bdbc1573bea9e71a4",
      "raw_sha256": "0990825e411f998914d0e8267ef7e38b220ab2173a1a8d9dfdea1508eee299ad",
      "seconds": 217.27876761555672,
      "model": "/root/polly-work/qvq-gsq/artifacts/gsq-staged/full-model-w2-nm128-signed-staged-v1/model"
    }
  },
  "paired_staged_minus_baseline": 0.016542597187758478,
  "calibration_samples": 128,
  "calibration_inputs_sha256": "893d981d8398efe64e111a22a6996f3e23f073b6460f124fc061097c8bb19dda",
  "paired_bootstrap_ci95": [
    0.009925558312655087,
    0.023986765922249794
  ],
  "bootstrap_seed": 7,
  "bootstrap_draws": 10000,
  "baseline_only_correct": 0,
  "staged_only_correct": 20,
  "exact_prompts_targets_and_input_ids": true,
  "prompt_ids_sha256": "3a66bc49447c3e335051f131323f18a0444202de2cd828d4fb842fda4abd73cb"
}
```
