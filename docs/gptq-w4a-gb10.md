<!-- SPDX-FileCopyrightText: 2026 ModelCloud.ai
SPDX-License-Identifier: Apache-2.0 -->

# Experimental GPTQ W4AFP8 on GB10

This backend targets NVIDIA GB10 (SM121), Linux, and Llama decoder models.
Weights remain native GPTQ INT4 in INT32-packed checkpoint tensors. Activations
use E4M3 FP8 with a dynamic FP32 scale per token. This release supports symmetric
128-element GPTQ groups, `desc_act=False`, FP16/BF16 model dtype, and an
unquantized output head. Triton with FP8 support is required.

## Lifecycle and arithmetic

```python
import torch
from gptqmodel import BACKEND, GPTQModel, QuantizeConfig

config = QuantizeConfig(
    bits=4, group_size=128, sym=True, desc_act=False,
    activation="w4afp8", offload_to_disk=False,
)
model = GPTQModel.load("unsloth/Llama-3.2-1B-Instruct", config,
                       device="cuda", dtype=torch.bfloat16)
model.quantize(calibration, backend=BACKEND.GPTQ_W4AFP8)
model.save("llama-w4afp8")
model = GPTQModel.load("llama-w4afp8", backend=BACKEND.AUTO, device="cuda")
```

Supply representative tokenized or text records as `calibration`. Default policy
version 3 replays activation rounding before Hessian capture and propagates the
same rounding policy through successive GPTQ stages. Selected decoder layers
must contain all seven quantized projections. Partial projection selection is
rejected. Optional Hadamard rotation is applied before encoding its GEMM operand.

Packing and serialization retain `qweight`, `qzeros`, `scales`, and `g_idx`.
Post-init stages centered INT4 codes in a nonpersistent E4M3 cache. Every integer
from -8 through 7 is exactly representable. Forward calls reuse this cache;
loading or repacking invalidates it. Runtime memory therefore includes the FP8
weight cache in addition to the INT4 checkpoint tensors.

The fused GEMM computes each 128-wide group with FP8 operands and FP32
accumulation, applies that group's original GPTQ scale, and then sums groups.
The per-token activation scale applies to the result. Arbitrary GPTQ scales are
never rounded into hardware scale formats or moved outside the group reduction.

## Activation boundary contract (version 3)

| Boundary or operation | Representation |
| --- | --- |
| Decoder-to-decoder handoff and residual carrier | E4M3 codes plus FP32 token scales |
| RMSNorm output to Q/K/V or gate/up | One shared encoded carrier; no separate repacking per projection |
| All selected Linear inputs | Encoded FP8; plain BF16/FP16 input raises an error |
| Linear output into attention, activation functions, or residual arithmetic | Model dtype; avoids an extra rounding immediately before wider arithmetic |
| RMSNorm and residual arithmetic | Decoded FP32 internally, then encoded output |
| RoPE, attention, KV cache, and MLP activation/product | Wider arithmetic; encoded again at the next GEMM input |
| Final norm and unquantized output head | Model dtype |

This contract provides encoded inter-layer transport and FP8 GEMM consumption.
It does not claim FP8 attention/KV-cache arithmetic or a fully fused decoder.
Version 2 remains readable; version 1 requires explicit migration and quality
revalidation. Ordinary weight-only GPTQ loads do not select this backend.

## Validation and memory limits

Use `tests/models/run_w4a_gb10_safe.sh` for GB10 runs. It serializes runs, sets
cgroup memory limits with swap disabled, and watches host and cgroup headroom
with a **2 GiB** reserve. A user systemd session and delegated cgroup memory
controller are required. Set `GPTQMODEL_TEST_PYTHON` to the intended interpreter.

```bash
export GPTQMODEL_TEST_PYTHON=/path/to/venv/bin/python
bash tests/models/run_w4a_gb10_safe.sh tests/kernels/test_w4afp8_gb10.py
bash tests/models/run_w4a_gb10_safe.sh tests/kernels/test_w4a_stream.py
bash tests/models/run_w4a_gb10_safe.sh tests/models/test_w4a_replay_stream.py
bash tests/models/run_w4a_gb10_safe.sh tests/models/test_w4a_tiny_llama_lifecycle.py
bash tests/models/run_w4a_gb10_safe.sh dtype-audit \
  --checkpoint llama-w4afp8 --variant w4afp8 --output audit.json
```

The dtype audit requires every decoder layer by default, checks pointer reuse
between layers and shared projections, and checks cached generation. The tiny
Llama test exercises quantize/save/reload with and without rotation. Dedicated
Llama 3.2 1B tests require all 16 layers and 112 packed projections. Override the
source with `GPTQMODEL_LLAMA3_2_MODEL`, calibration parquet with
`GPTQMODEL_CALIBRATION_PARQUET`, and calibration size with
`GPTQMODEL_W4A_CALIBRATION_ROWS`. `GPTQMODEL_W4A_TEST_PHASE=quant-save` and
`GPTQMODEL_W4AFP8_SAVE_PATH` separate quantization and reload memory peaks.

## Full-row quality regression

Create a W4A16 reference view with identical packed tensors, then evaluate both
lanes using the same prompts and settings. The tool enforces complete test splits:
1,209 GSM8K Platinum rows and 1,172 ARC-Challenge rows. GSM8K Platinum is the
acceptance gate; ARC is a diagnostic. Keep the generated per-row JSON results as
reusable baselines. Use `freeze-date` when a later run must match a baseline's
date-dependent chat prompt, and verify prompt equality during comparison.

```bash
python -m tests.models.w4a_quality_regression prepare \
  --checkpoint llama-w4afp8 --reference llama-w4a16
bash tests/models/run_w4a_gb10_safe.sh quality-eval \
  --checkpoint llama-w4afp8 --reference llama-w4a16 --variant w4a16 \
  --task gsm8k_platinum_cot --output baseline.json
bash tests/models/run_w4a_gb10_safe.sh quality-eval \
  --checkpoint llama-w4afp8 --variant w4afp8 \
  --task gsm8k_platinum_cot --output fp8.json
python -m tests.models.w4a_quality_regression compare \
  --w4a16 baseline.json --w4a-float fp8.json --task gsm8k_platinum_cot \
  --allowed-drop-pp 2 --output paired.json
```

The fresh version-3 full-model experiment used 512 calibration records (188,256
nonpadding tokens, concatenation size 2,048). Its same-weight W4A16 reference
scored **432/1,209 (35.7320%)**; W4AFP8 scored **441/1,209 (36.4764%)**. The paired
delta was +0.7444 percentage points, with 95% CI [-1.5431, +3.0319], inside the
provisional 2-point regression budget. There were 95 baseline-only correct and
104 FP8-only correct answers. This is evidence against a material activation
regression for this checkpoint, not evidence of a statistically significant
improvement or parity with other weight-only checkpoints. The full-row run
predates the FP8-only extraction; rerun it when numerical behavior changes.

The kernel tests use independent Torch arithmetic, exact packed-code checks,
1e-6 quantization tolerances, and at most 2e-3 inference tolerances. The optional
microbenchmark uses CUDA graph replay; it does not establish end-to-end speedup.

### FP8-only branch validation

After rebasing on upstream `6d3232c`, GB10 checks passed 75 tests (one optional
rotated-checkpoint test skipped): 14 FP8 kernel/config tests, six encoded-stream
tests, 12 calibration-replay tests, two tiny-model lifecycle tests, one saved
Llama 1B reload test, and 40 stage/rotation tests. Kernel checks cover FP16/BF16
GEMM inputs, zero rows, outliers, random per-column scales, partial output tiles,
and FP8 threshold ties/neighbors in FP16, BF16, and FP32. Threshold tests compare
complete code bytes, including signed zero. The saved-model audit also passed
all 112 projections and 16 layers, with shared carrier pointers and cached
text generation. Full GSM8K results above are retained from the prior full run;
these focused checks do not constitute a repeated full benchmark.
