# SwiGLU W2 quantization experiment

Date: 2026-08-26

This report records the first matched Smooth-SwiGLU versus no-Smooth control on
the cached Llama 3.2 1B Instruct model. The purpose is to test whether the
offline `up_proj`/`down_proj` reparameterization improves the actual nonlinear
MLP output rather than only improving a local linear reconstruction proxy.

## Matched setup

- Model: `meta-llama/Llama-3.2-1B-Instruct`
- Hardware: Apple M4 Max; MLX GPU inference; MPS quantization
- Quantizer: QVQ, W2, `format=qvq`, `rounding=block_ldlq`
- Scope: first decoder layer only; all seven attention/MLP projections in that layer
- Calibration: 8 identical rows from `dataset/calibration_mix_128k_qwen3_0.6b/calibration.parquet`
- Calibration token count: 2,722 non-padding tokens
- Smooth run: group size 16, maximum Smooth-SwiGLU scale-search calibration 64 tokens
- Control: `smooth_swiglu=None`
- Rate: 2.0 BPW for both checkpoints
- Evaluation: identical dense MLX reference, prompt, token IDs, and output shape
- Prompt: `Explain in one sentence why unit tests are useful.`
- Logit tensor shape: `[1, 12, 128256]`

Smooth-SwiGLU rescales only the dense `up_proj` rows and matching `down_proj`
columns before QVQ quantization. `gate_proj` is not rescaled. The dense model
function remains unchanged before quantization.

## Logit comparison

Relative L2 is defined as:

```text
||logits_quantized - logits_dense||_2 / ||logits_dense||_2
```

| Metric | No Smooth-SwiGLU | Smooth-SwiGLU | Absolute change | Smooth change |
|---|---:|---:|---:|---:|
| Relative L2 | 0.1312788093 | **0.1197898642** | -0.0114889451 | **8.7516% lower** |
| RMSE | 0.3996324725 | **0.3636707660** | -0.0359617065 | **8.9987% lower** |
| Maximum absolute error | 3.8955078125 | **3.2822265625** | -0.61328125 | **15.7433% lower** |
| Cosine similarity | 0.9916922450 | **0.9930137396** | +0.0013214946 | higher |

This is positive evidence that Smooth-SwiGLU reduces propagated first-layer
logit error at the same nominal BPW and runtime graph.

## Important guardrail result

The single-prompt top-1 result is mixed:

| Model | Dense top-1 token | Quantized top-1 token | Match |
|---|---:|---:|---:|
| No Smooth-SwiGLU | 8113 | 8113 | yes |
| Smooth-SwiGLU | 8113 | 2435 | no |

Therefore this experiment demonstrates a logit-distribution improvement, not
yet a task-quality improvement. The top-1 mismatch is a guardrail and must be
checked across multiple prompts and layers before Smooth-SwiGLU becomes a
default policy.

## Runtime and artifact checks

The no-Smooth control completed successfully through QVQ save, public MLX
reload, and MLX forward:

| Measurement | No Smooth-SwiGLU |
|---|---:|
| Quantization time | 64.6773 s |
| Save time | 1.1504 s |
| Dense MLX load | 0.9472 s |
| Public MLX reload | 4.3912 s |
| MLX forward | 0.1028 s |
| Checkpoint size | 2272.25 MB |

The Smooth run also completed save/reload/forward and produced the same
estimated 2.0 BPW checkpoint class. Smooth-SwiGLU adds no inference operation,
tensor, or stored scale requirement: the transformed weights are quantized and
stored directly.

## Interpretation and next gate

The result supports retaining Smooth-SwiGLU as an offline QVQ search
candidate. It does not justify claiming universal improvement because this is
one prompt and one quantized decoder layer. The next confirmation should use
multiple disjoint prompts and selected early/late layers, then a full-model
teacher-forced logit and generation comparison. Report both propagated logit
metrics and decision metrics; do not select Smooth-SwiGLU on local module loss
alone.

## Reduced YAQA follow-up on Apple M4 Max

Date: 2026-08-26

The first experiment above is a small proof-of-mechanism test. The following
matched control is larger and uses the Llama 3.2 1B QVQ campaign configuration:

- nominal rate: flat W2 (`2.0` BPW; the serialized QVQ estimate is `2.03125`
  BPW including segmented-bank selector metadata);
- YAQA regularization: `0.15` selected by an explicit rate override;
- quantized scope: decoder layer 0 only, all seven attention/MLP projections;
- calibration: rows `0:128`, with YAQA from rows `128:310` of the local
  `calibration_mix_128k_qwen3_0.6b` parquet;
- Smooth calibration statistics: 512 tokens per MLP, group size 16, candidate
  exponents `[-1, -0.5, 0, 0.5, 1]`;
- held-out evaluation: rows `438:503`, 65 rows, 15,062 tokens after truncating
  each row to 256 tokens;
- evaluation backend: native MLX GPU teacher-forced forward, with a dense MLX
  reference converted to FP16 so both paths use the same comparison dtype.

The first reduced numbers were collected before the bridge stopped forcing MLX
to CPU. They are retained in the execution notes only as a CPU reference; the
GPU rerun below is the authoritative result for this protocol.

| Metric | No Smooth | Smooth | Smooth minus control |
|---|---:|---:|---:|
| Relative logit L2 | **0.496332398** | 0.497496762 | +0.23% |
| RMSE | **1.483880545** | 1.487361633 | +0.23% |
| Maximum absolute error | **20.027344** | 20.501953 | +2.37% |
| Cosine similarity | **0.886480202** | 0.885346320 | -0.001134 |
| Top-1 agreement | **0.795578** | 0.792989 | -0.259 percentage points |

This larger reduced test does not show an improvement from the current
analytical Smooth proxy. In this run every selected group scale reached the
configured upper bound (`scale_min = scale_max = 2.0`), so the result is also
evidence that the proxy/search range is not yet well calibrated for this
campaign. It is not evidence that the exact reparameterization is incorrect.
The next meaningful test is QVQ-aware scale selection and the atomic
gate/up/down candidate-search arm, followed by full-depth held-out replay.

The Smooth checkpoint has a complete quantization manifest at
`/tmp/qvq-swiglu-reg015-smooth-mps-95e206a4/qvq_quantize_run.json`. The
no-Smooth payloads were written successfully, but its manifest write hit the
pre-fix MPS device-reporting bug after quantization; therefore the no-Smooth
numbers above come from a direct MLX reload/forward comparison and not from a
published task-evaluation report. Both checkpoint payload directories remain
available locally.

## Post-MLX-bridge rerun

Date: 2026-08-26

The following measurements were rerun after the MLX bridge was changed to use
`mx.gpu` when available. They use the same dense Llama 3.2 1B Instruct source
and native MLX GPU teacher-forced comparison, but are reported separately
because the current calibration/tokenization path does not reproduce the old
2,722-token tiny artifact or the old 15,062-token reduced slice exactly.

### Tiny one-layer proof

- fresh artifacts: `/tmp/qvq-swiglu-tiny-nosmooth-87f97f04` and
  `/tmp/qvq-swiglu-tiny-smooth-87f97f04`;
- QVQ W2 `block_ldlq`, first decoder layer, seven projections;
- 8 calibration rows, 2,970 non-padding calibration tokens reported by the
  current run;
- Smooth group size 16 and 64-token scale-search limit;
- prompt: `Explain in one sentence why unit tests are useful.`;
- output shape: `[1, 12, 128256]`.

| Metric | No Smooth | Smooth-SwiGLU | Smooth minus control |
|---|---:|---:|---:|
| Relative L2 | 0.123125224 | 0.125049626 | +1.56% |
| RMSE | 0.374812615 | 0.380670801 | +1.56% |
| Maximum absolute error | 3.718750000 | 3.531250000 | **-5.04%** |
| Cosine similarity | 0.992489318 | 0.992296272 | -0.000193 |
| Last-token top-1 | match | match | unchanged |

The corrected Smooth run completed through lazy-source materialization,
function-parity checking, QVQ save, native MLX reload, and GPU forward. The
current tiny run therefore validates the corrected offline path, but its
relative-L2 result is not an improvement. The historical 0.131278809 to
0.119789864 improvement remains valid for its original artifact/protocol and
is retained above as historical evidence rather than mixed into this rerun.

### Reduced 65-row MLX slice

This is a fresh native-MLX-GPU evaluation of the preserved matched YAQA
artifacts `/tmp/qvq-swiglu-reg015-nosmooth-mps-95e206a4` and
`/tmp/qvq-swiglu-reg015-smooth-mps-95e206a4`. The current tokenizer produced
65 held-out rows and 14,969 truncated tokens (maximum 256 tokens per row).

| Metric | No Smooth | Smooth-SwiGLU | Smooth minus control |
|---|---:|---:|---:|
| Relative logit L2 | 0.380534681 | 0.382467134 | +0.51% |
| RMSE | 1.201429318 | 1.207530485 | +0.51% |
| Maximum absolute error | 18.085449219 | 16.311523438 | **-9.81%** |
| Cosine similarity | 0.930606950 | 0.929755969 | -0.000851 |
| Top-1 agreement | 81.2412% | 81.2680% | **+0.0267 pp** |

Both artifacts loaded with the public QVQ MLX backend on `Device(gpu, 0)` and
completed all 65 teacher-forced forwards. These numbers supersede neither the
historical 15,062-token table nor the tiny historical proof; they are the
post-bridge rerun to use when assessing the current MLX path.

### MLX/runtime status

Native MLX QVQ reload, GPU full-model forward, and short deterministic
generation pass for both the Smooth and no-Smooth payloads. The MLX bridge
selects `Device(gpu, 0)` on this Apple host and falls back to CPU only when no
MLX GPU is available. The public Torch/MPS evaluator still segfaulted while
loading the quantized checkpoint, so no Torch/MPS quality number is being
reported as valid.

## Orthogonal mechanism tests on the M4 Max

Date: 2026-08-26

These tests implement the next mechanism-isolation arms while the full
Atomic/Smooth+Atomic jobs are pending. They use the local cached
`meta-llama/Llama-3.2-1B-Instruct` model, QVQ `qvq_v2b2_p32`, flat W2 as the
default rate, YAQA regularization `0.15`, seed `0`, ordinary rows `0:128`,
YAQA rows `128:310`, and only decoder layer `0`. V+O and Gate+Down each used
the matched 182-sequence YAQA capture (`68,486` valid output samples).

The held-out comparison uses rows `438:503`, the same chat-template rendering
with `add_generation_prompt=False`, truncation at 256 tokens, and native MLX
GPU teacher-forced forward. The current tokenizer produced 14,938 tokens.
Dense outputs were MLX bfloat16 and both outputs were cast to MLX float32
before streaming metric reduction; this avoids retaining the vocabulary-sized
logits for all rows at once. `Device(gpu, 0)` was confirmed for every forward.

The flat-W2 control is the preserved artifact
`/tmp/qvq-swiglu-reg015-nosmooth-mps-95e206a4`. New artifacts are:

- V+O W2.5: `/tmp/qvq-mechanism-vo-w25-mps-a14`
- Gate+Down W2.5: `/tmp/qvq-mechanism-gate-down-w25-mps-a14`
- Up+Down W2.5: `/tmp/qvq-mechanism-up-down-w25-mps-a14` (quantization was
  restarted after a monitor interruption and completed successfully)
- O-only W2.5: `/tmp/qvq-mechanism-o-w25-mps-0011250f`
- V-only W2.5: `/tmp/qvq-mechanism-v-w25-mps-0011250f`

### Local layer-0 held-out logits

| Arm | W2.5 projections | Relative L2 | RMSE | Max abs. error | Cosine | Top-1 agreement |
|---|---|---:|---:|---:|---:|---:|
| Flat-W2 control | none | 0.494589233 | 1.478702173 | 18.593750 | 0.887347785 | 80.4124% |
| V+O | `self_attn.v_proj`, `self_attn.o_proj` | **0.493095418** | **1.474236026** | **18.156250** | **0.888034137** | **80.4325%** |
| Gate+Down | `mlp.gate_proj`, `mlp.down_proj` | **0.491829011** | **1.470449775** | 18.789063 | **0.888827608** | **80.6132%** |
| Up+Down | `mlp.up_proj`, `mlp.down_proj` | **0.491072044** | **1.468186628** | 19.578125 | **0.889104352** | **80.5262%** |
| O-only | `self_attn.o_proj` | **0.494193906** | **1.477520240** | **17.949219** | **0.887693717** | 80.1848% |
| V-only | `self_attn.v_proj` | 0.494626580 | 1.478813831 | 19.031250 | 0.887281293 | **80.4994%** |

Relative to the local control, V+O changes relative L2 by `-0.302%`, RMSE
by `-0.302%`, max error by `-2.35%`, cosine by `+0.000686`, and top-1 by
`+0.0201` percentage points. Gate+Down changes relative L2 by `-0.558%`,
RMSE by `-0.558%`, cosine by `+0.001480`, and top-1 by `+0.2008` percentage
points; its maximum error is `+1.05%` higher. Up+Down changes relative L2 by
`-0.711%`, RMSE by `-0.711%`, cosine by `+0.001757`, and top-1 by `+0.1138`
percentage points; its maximum error is `+5.29%` higher. These are layer-0
propagated logit measurements, not task scores.

Relative to the same flat-W2 control, O-only changes relative L2 and RMSE by
`-0.080%`, max error by `-3.47%`, cosine by `+0.000346`, and top-1 by
`-0.2276` percentage points. V-only changes relative L2 and RMSE by
`+0.008%`, max error by `+2.35%`, cosine by `-0.000067`, and top-1 by
`+0.0870` percentage points. Thus, on this layer-0 slice, O-only reduces the
largest error but does not improve aggregate top-1; V-only is effectively
neutral on L2 and slightly improves top-1. These local controls do not replace
the matched full-depth task evaluation.

The newly completed single-projection MLX reports also measured mean absolute
logit error: O-only `1.087624615` and V-only `1.088175197`. The earlier
control report did not retain this extra reduction, so no control-relative
claim is made for mean absolute error. Each new report contains the exact
row/token counts (`65` rows, `14,938` tokens, `1,915,888,128` logit elements),
the MLX device (`Device(gpu, 0)`), dense/compare dtypes (`bfloat16`/`float32`),
and the reproducible checkpoint path under `/tmp` listed above.

### Canonical task-score status

The existing matched full-depth control remains D300 `17.9792%` (Exact32
`3/300`) and GSM8K `23.8213%`, as recorded in the campaign ledger. The new
remote queue has since completed the canonical full-depth V+O recipe (arm
`d71136`, effective `2.0663` BPW): D300 `19.1771%` (Exact32 `3/300`, mean
first divergence `5.0633`) and GSM8K `25.4756%` (`308/1209`). Relative to
the reg-0.15 flat-W2 control, that is `+1.1979` D300 percentage points and
`+1.6543` GSM8K percentage points. The local layer-0 V+O checkpoint above is
a separate MPS artifact used for mechanism metrics, while `d71136` is the
full-depth CUDA task artifact. The later full-depth single-projection task
reports publish GSM8K `22.9942%` for O-only (`09674b`) and `21.4227%` for
V-only (`0cd45d`); V-only also publishes D300 `17.2917%`. O-only D300 and
complete paired D300 metadata were not present in the pulled ledger snapshot,
so they are not inferred here. Gate+Down and Up+Down publish GSM8K `27.9569%`
and `28.1224%`, respectively, but their complete D300 metadata is likewise
not in the local ledger. These are full-depth CUDA task artifacts and must not
be conflated with the local layer-0 native-MLX measurements above.

### Configuration files for the queued/full campaign

The reproducible CUDA and MPS configs are in `scripts/configs/`:

- `llama32_1b_v2b2_p32_yaqa_reg015_vo_w25{,_mps}.json`
- `llama32_1b_v2b2_p32_yaqa_reg015_o_w25{,_mps}.json`
- `llama32_1b_v2b2_p32_yaqa_reg015_v_w25{,_mps}.json`
- `llama32_1b_v2b2_p32_yaqa_reg015_gate_down_w25{,_mps}.json`
- `llama32_1b_v2b2_p32_yaqa_reg015_up_down_w25{,_mps}.json`

The local M4 Max O-only and V-only controls are now complete and represented
above. Smooth-alpha and local-only Atomic are not exposed as runnable MLX
experiment configs yet; the available Atomic config still performs the
validated propagated final-logit replay rather than a local-only selector.
The five completed mixed-precision arms provide the first interaction map:
both MLP pairs improve aggregate teacher-forced fidelity on this slice, with
Up+Down giving the lowest relative L2; O-only gives the smallest maximum
error; and V-only gives the highest top-1 among the two single-projection
attention controls.
