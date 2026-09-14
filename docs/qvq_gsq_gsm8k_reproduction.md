# QVQ PR 212: GSQ/GPTQ GSM8K reproduction report

This report records the full-checkpoint W3 and W4 experiments used to assess
whether staged GSQ improves packed scalar GPTQ. It includes model paths,
dataset paths and hashes, configs, commands, evaluator version, artifacts,
and paired results.

## Executive result

The experiments used Llama-3.2-1B-Instruct and the same 1,209-row
GSM8K-Platinum test set. At W3, GSQ substantially improves the matched GPTQ
control. At W4, the current staged recipe still regresses.

| Arm | W3 | W4 |
| --- | ---: | ---: |
| Dense | 583/1209 = **48.221671%** | 583/1209 = **48.221671%** |
| Packed GPTQ | 158/1209 = **13.068652%** | 447/1209 = **36.972705%** |
| Packed GPTQ + staged GSQ | 342/1209 = **28.287841%** | 417/1209 = **34.491315%** |

Relative to matched GPTQ:

- **W3:** GSQ gains 184 correct answers and +15.219189 percentage points.
- **W4:** GSQ loses 30 correct answers and -2.481390 percentage points.

Paired sample transitions:

| Comparison | Recovered by GSQ | Lost from GPTQ | Net |
| --- | ---: | ---: | ---: |
| W3 GPTQ → GSQ | 239 | 55 | +184 |
| W4 GPTQ → GSQ | 122 | 152 | -30 |

The supported conclusion is bit-dependent: GSQ can substantially recover
ultra-low-bit W3 GPTQ task accuracy, but the current W4 integration is not yet
an improvement and should not be presented as one.

## Source and runtime

- Repository: `ModelCloud/QvQ`
- PR: [212](https://github.com/ModelCloud/QvQ/pull/212)
- Tested branch: `codex/gsq-p32-window`
- Tested commit: `0fd58200` (`docs: define complementary QVQ GSQ integration`)
- Python environment: `/root/venv-py3.14t-gil0`
- Python: 3.14 free-threaded build with the GIL enabled
- GPTQModel: `7.4.0+ultra+0fd58200`
- Transformers: `5.5.4`
- Torch: `2.13.0+cu130`
- Triton: `3.7.1`
- Evalution: **`0.0.17`**, installed from PyPI
- Datasets: `5.0.1`
- LogBar: `0.4.14`
- PyPcre: `0.6.2`

Install/verify the evaluator:

```bash
python -m pip install --upgrade Evalution
python -m pip index versions Evalution
python - <<'PY'
from importlib.metadata import version
import evalution
print(version("Evalution"))
print(evalution.__file__)
PY
```

The installed version was `0.0.17`.

## Model paths

All paths below are local, durable validation-host artifacts.

Dense reference:

```text
/monster/data/model/Llama-3.2-1B-Instruct
```

W4 packed GPTQ control:

```text
/monster/data/model/qvq/modelcloud-qvq__llama-3.2-1b-instruct__gsq-w4-baseline__gptq-v2__calib128-unweighted__seed7__20260909__commit5cc76af5c715__a804113c065f/gptq-v2
```

W4 staged GSQ checkpoint:

```text
/monster/data/model/qvq/modelcloud-qvq__llama-3.2-1b-instruct__gsq-w4-staged__gptq-v2__calib128-unweighted__seed7__20260909__commit5cc76af5c715__de50d6afd3f2/gptq-v2
```

W3 packed GPTQ control generated for this report:

```text
/monster/data/model/qvq/gsq-pr212-full-w3-llama32-1b-20260914/baseline-b1/model
```

W3 staged GSQ checkpoint generated for this report:

```text
/monster/data/model/qvq/gsq-pr212-full-w3-llama32-1b-20260914/staged-b1/model
```

Both W3 checkpoints are GPTQ-V2 exports with `bits=3`, `group_size=128`.
Both passed 32/32 exact held-out reload checks with `reload_max_abs=0.0`.

## Calibration data

The explicit W3 input artifact is:

```text
/monster/data/model/qvq/gsq-pr212-w3-inputs-nm128-seed7/inputs.json
/monster/data/model/qvq/gsq-pr212-w3-inputs-nm128-seed7/provenance.json
```

Calibration details:

- 128 training documents and 32 held-out documents;
- seed `7`, token cap `2048`, and 47,005 training tokens;
- unweighted native variable-length documents with no padding;
- source model `/monster/data/model/Llama-3.2-1B-Instruct`;
- disjointness passed against locked D300 and all 1,209 GSM8K-Platinum
  questions.

Original calibration source:

```text
/monster/data/model/dataset/nm-calibration/llm.parquet
```

Recorded source SHA-256:

```text
26122fd822e64d2fc704b0fe84af7a2df8a24a4455d347e16a6b5a7484f5cbef
```

Recorded calibration input SHA-256:

```text
893d981d8398efe64e111a22a6996f3e23f073b6460f124fc061097c8bb19dda
```

## GSM8K test data and evaluator protocol

Every dense, W3 GPTQ, W3 GSQ, W4 GPTQ, and W4 GSQ run used:

```text
/monster/data/model/qvq/modelcloud-qvq__llama-3.2-1b-instruct__gsq-w2-staged__gptq-v2__yaqa-nm16-unweighted__seed7__20260909__commit5cc76af5c715__e267c3af8d75/evaluations/gsm8k-platinum-16/dataset
```

Dataset file and SHA-256:

```text
test.parquet
7a2de6410ded2b7995de2c4d92c72df2e1049735ec490e804d52b451cb95aff7
```

The directory contains 1,209 test rows. Each run used:

```text
task                = gsm8k_platinum_cot
apply_chat_template = true
fewshot_seed        = 7
max_new_tokens      = 256
stream              = false
gen_kwargs          = do_sample=false,temperature=0.0
batch_size          = 32
dtype               = float16
device              = cuda:0
attn_implementation = eager
seed                = 7
backend             = native GPTQModel Torch backend
```

All compared raw outputs had identical sample indices, targets, and rendered
prompts at all 1,209 positions.

## Quantization and GSQ configuration

Common packed scalar format:

```text
method            = gptq
quant_method      = gptq
format            = gptq_v2
checkpoint_format = gptq_v2
group_size        = 128
sym               = true
desc_act          = false
lm_head           = false
pack_dtype        = int32
initializer       = gptq_signed
```

GSQ training configuration:

```text
initializer    = gptq_signed
optimizer      = lion
seed           = 7
epochs         = 5
qk_steps       = 2000
damp_percent   = 0.01
assignment_lr  = 0.0001
scale_lr       = 0.00005
weight_decay   = 1.0
betas          = (0.9, 0.95)
temperature    = (2.0, 0.05)
multiplier     = (100.0, 500.0)
warmup_steps   = 0
min_lr         = 0.1
decay          = cosine
```

The W3 current-branch run used `batch_size=1` and `microbatch_size=1` for
variable-length capture. An initial W3 attempt with batch size 64 was rejected
by the current capture guard before quantization and was not scored. The
existing W4 checkpoint was produced with its recorded earlier recipe of
`batch_size=64` and `microbatch_size=16`.

## GPU environment

The validation GPU was:

```text
UUID:  GPU-cb9e7784-cf50-203d-4f0d-5c622a89b1f2
Name:  NVIDIA PG506-230
VRAM:  98304 MiB
```

Environment used for quantization and evaluation:

```bash
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=GPU-cb9e7784-cf50-203d-4f0d-5c622a89b1f2
export GPU_ALLOCATOR_LEASE_ID=manual-local-free-gpu-20260914
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export GPTQ_CACHE_DEQUANTIZED_WEIGHTS=1
export OMP_NUM_THREADS=4
export MAX_JOBS=4
```

The allocator daemon was unavailable, so the GPU was used through a manually
declared local lease. Preflight confirmed the UUID was idle with 4 MiB used
and 0% utilization before each run. W3 GSQ peaked at approximately 20 GiB of
GPU allocation.

## Exact commands

Run from `/root/work/qvq-pr212` after setting the environment above.

### W3 generation

Baseline GPTQ:

```bash
python scripts/validate_gsq_full_llama.py \
  --inputs /monster/data/model/qvq/gsq-pr212-w3-inputs-nm128-seed7 \
  --output /monster/data/model/qvq/gsq-pr212-full-w3-llama32-1b-20260914/baseline-b1 \
  --arm baseline --bits 3 --train-samples 128 \
  --initializer gptq_signed --batch-size 1 --microbatch-size 1 \
  --epochs 5 --qk-steps 2000 --train-precision float32
```

Staged GSQ:

```bash
python scripts/validate_gsq_full_llama.py \
  --inputs /monster/data/model/qvq/gsq-pr212-w3-inputs-nm128-seed7 \
  --output /monster/data/model/qvq/gsq-pr212-full-w3-llama32-1b-20260914/staged-b1 \
  --arm staged --bits 3 --train-samples 128 \
  --initializer gptq_signed --batch-size 1 --microbatch-size 1 \
  --epochs 5 --qk-steps 2000 --train-precision float32
```

### Dense evaluation

The dense reference was evaluated once because it is identical for W3 and W4:

```bash
python scripts/evaluate_gsq_gsm8k.py \
  --model /monster/data/model/Llama-3.2-1B-Instruct \
  --output /monster/data/model/qvq/gsq-pr212-evalution017-gsm8k-platinum-llama32-1b-w4-20260914/dense \
  --dataset /monster/data/model/qvq/modelcloud-qvq__llama-3.2-1b-instruct__gsq-w2-staged__gptq-v2__yaqa-nm16-unweighted__seed7__20260909__commit5cc76af5c715__e267c3af8d75/evaluations/gsm8k-platinum-16/dataset \
  --arm dense-evalution017 --batch-size 32
```

### W3 evaluation

```bash
python scripts/evaluate_gsq_gsm8k.py \
  --model /monster/data/model/qvq/gsq-pr212-full-w3-llama32-1b-20260914/baseline-b1/model \
  --output /monster/data/model/qvq/gsq-pr212-evalution017-gsm8k-platinum-llama32-1b-w3-20260914/gptq \
  --dataset /monster/data/model/qvq/modelcloud-qvq__llama-3.2-1b-instruct__gsq-w2-staged__gptq-v2__yaqa-nm16-unweighted__seed7__20260909__commit5cc76af5c715__e267c3af8d75/evaluations/gsm8k-platinum-16/dataset \
  --arm packed-gptq-w3-g128-evalution017 --batch-size 32
```

```bash
python scripts/evaluate_gsq_gsm8k.py \
  --model /monster/data/model/qvq/gsq-pr212-full-w3-llama32-1b-20260914/staged-b1/model \
  --output /monster/data/model/qvq/gsq-pr212-evalution017-gsm8k-platinum-llama32-1b-w3-20260914/gsq \
  --dataset /monster/data/model/qvq/modelcloud-qvq__llama-3.2-1b-instruct__gsq-w2-staged__gptq-v2__yaqa-nm16-unweighted__seed7__20260909__commit5cc76af5c715__e267c3af8d75/evaluations/gsm8k-platinum-16/dataset \
  --arm packed-gptq-gsq-w3-g128-evalution017 --batch-size 32
```

### W4 evaluation of existing checkpoints

```bash
python scripts/evaluate_gsq_gsm8k.py \
  --model /monster/data/model/qvq/modelcloud-qvq__llama-3.2-1b-instruct__gsq-w4-baseline__gptq-v2__calib128-unweighted__seed7__20260909__commit5cc76af5c715__a804113c065f/gptq-v2 \
  --output /monster/data/model/qvq/gsq-pr212-evalution017-gsm8k-platinum-llama32-1b-w4-20260914/gptq \
  --dataset /monster/data/model/qvq/modelcloud-qvq__llama-3.2-1b-instruct__gsq-w2-staged__gptq-v2__yaqa-nm16-unweighted__seed7__20260909__commit5cc76af5c715__e267c3af8d75/evaluations/gsm8k-platinum-16/dataset \
  --arm packed-gptq-w4-g128-evalution017 --batch-size 32
```

```bash
python scripts/evaluate_gsq_gsm8k.py \
  --model /monster/data/model/qvq/modelcloud-qvq__llama-3.2-1b-instruct__gsq-w4-staged__gptq-v2__calib128-unweighted__seed7__20260909__commit5cc76af5c715__de50d6afd3f2/gptq-v2 \
  --output /monster/data/model/qvq/gsq-pr212-evalution017-gsm8k-platinum-llama32-1b-w4-20260914/gsq \
  --dataset /monster/data/model/qvq/modelcloud-qvq__llama-3.2-1b-instruct__gsq-w2-staged__gptq-v2__yaqa-nm16-unweighted__seed7__20260909__commit5cc76af5c715__e267c3af8d75/evaluations/gsm8k-platinum-16/dataset \
  --arm packed-gptq-gsq-w4-g128-evalution017 --batch-size 32
```

## Run hashes and audit paths

Each result directory contains `run.json`, `raw.json`, and `evaluation.md`.
`run.json` records the evaluator source, model and dataset hashes, complete
argv, metric, and raw-output SHA-256.

| Arm | `raw.json` SHA-256 | Result directory |
| --- | --- | --- |
| Dense | `49bf7ef0b503f3d32156a51138dfb4585aed735a8b7ba57f3faa15b886b3a798` | `/monster/data/model/qvq/gsq-pr212-evalution017-gsm8k-platinum-llama32-1b-w4-20260914/dense` |
| W3 GPTQ | `f637859157696e102713b19596a24d94cc7aa8e172e3fbc7ea5387b34bec1282` | `/monster/data/model/qvq/gsq-pr212-evalution017-gsm8k-platinum-llama32-1b-w3-20260914/gptq` |
| W3 GSQ | `aadfd6e098a428e8c358de425b0f12ffaece2859c1ed68088dc76d7881575328` | `/monster/data/model/qvq/gsq-pr212-evalution017-gsm8k-platinum-llama32-1b-w3-20260914/gsq` |
| W4 GPTQ | `87ed5d5064b7bd6fa8e00a7aed89f45939c1af7d29d93eb4cc67ff0be6098dc7` | `/monster/data/model/qvq/gsq-pr212-evalution017-gsm8k-platinum-llama32-1b-w4-20260914/gptq` |
| W4 GSQ | `3e432d548036ac68de8a89877f9474f611f306133e06ac3a90791ad4bba05484` | `/monster/data/model/qvq/gsq-pr212-evalution017-gsm8k-platinum-llama32-1b-w4-20260914/gsq` |

Do not compare these scores with older runs made using Evalution versions
before `0.0.17`; the evaluation matrix was rerun under the current PyPI
version for evaluator consistency.
