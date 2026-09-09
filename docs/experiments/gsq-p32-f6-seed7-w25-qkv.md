# W2.5 full QKV: YAQA alone versus refinement

## Scope

This follow-up sets the complete block-0 Q/K/V projections to W2.5 and keeps
the other 109 F6 projections unchanged. It is **not** a uniformly W2.5 whole
model. It follows the [original-rate full-QKV verification](gsq-p32-f6-seed7-full-qkv.md)
with identical dense teacher, calibration documents, held-out token IDs and
FP32 reference execution.

GSQ-inspired refinement enhances the existing YAQA pipeline experimentally;
it does not replace YAQA. The comparison is fresh W2.5 YAQA alone versus that
same baseline followed by GSQ-inspired or deterministic candidate refinement.

## Fresh rate-matched baseline

All three target projections are freshly quantized from dense weights using
the existing `capture_yaqa_sketch_b` and `quantize_qvq_linear` APIs. The run uses
16 complete-model backward calibration sequences from the original verified
YAQA/NM corpus: 3,767 real token samples, YAQA/NM weights 1.25/1.0 (4,279 weighted
tokens), seed 7, FP32 exact batched Gram factors, and activation checkpointing
over all 16 decoder blocks. No synthetic or identity output Hessian is used.

The W2.5 quantizer uses RHT, seed/input-sign seed 7, `pgc16-v1`, two-bank P32,
YAQA family reselection, full family scoring and the F6 W2.5 regularization
0.02. Output scales and banks are selected by this fresh YAQA baseline and then
held fixed for both refinement arms. The original minimum of 10,178 sequences
is explicitly reduced to 16 for this bounded experiment, not silently claimed
as a full repeat of the F6 quantization campaign.

Because both calibration scope and Q/V bit rates differ from the historical
F6 quantization, the original F6 score is a contextual reference only. Compare
refinements to the **fresh W2.5 YAQA baseline** to isolate their effect.

## Full-model held-out results

Evaluation uses 32 locked divergence documents and 6,367 valid token positions,
with the same 256-token cap and disjointness bindings as the prior full-QKV
run. All entries below are token-weighted. Top-N is set overlap/N, not task
accuracy. MSE here measures final logits, not local projection outputs.

| Arm | Final KLD | Logit MSE | Logit NMSE | Top-1 | Top-5 | Top-10 |
|---|---:|---:|---:|---:|---:|---:|
| W2.5 YAQA alone | 0.110186534 | 0.446268583 | 0.059553561 | 86.0374% | 83.0407% | 82.7815% |
| YAQA + GSQ-inspired | 0.110245976 | 0.446262958 | 0.059552374 | 85.9274% | 83.0470% | 82.7831% |
| YAQA + deterministic | 0.110549340 | 0.443950337 | 0.059232865 | 86.0374% | 83.0093% | 82.7831% |

GSQ-inspired refinement worsens KLD by approximately 0.054% and Top-1 by
0.110 percentage points. Its paired **document-mean** KLD delta is +0.00008024,
95% bootstrap interval [+0.00005943, +0.00010551]. Top-1's document-mean delta
is -0.0017195, interval [-0.0031798, -0.0005734]. These are small measured negative
effects on this split. MSE, NMSE, Top-5 and Top-10 changes are noise-consistent.
Document-mean bootstrap intervals and token-weighted aggregate scores are
different estimands; the raw report labels them separately.

Deterministic refinement lowers logit MSE but increases KLD; its document-mean
Top-5 change is negative. Lower local or logit MSE does not establish better
probability-distribution preservation.

## Local held-out output MSE

These values average each full projection's MSE across held-out documents.

| Full projection [out,in] | YAQA alone | GSQ-inspired | Deterministic |
|---|---:|---:|---:|
| Q [2048,2048] | 0.003041142 | 0.003040816 | 0.003028734 |
| K [512,2048] | 0.005128840 | 0.005127742 | 0.005105810 |
| V [512,2048] | 0.0002197245 | 0.0002197245 | 0.0002197385 |

GSQ's local calibration guard retains the V baseline exactly. Q/K fitting gains
do not translate into improved final KLD here. This directly demonstrates why
the earlier isolated slice result was insufficient.

## Verification and decision

- Fresh YAQA results match full window-decode reconstruction exactly.
- All complete projection exports pass planar/window round-trip and payload
  reload checks. Full-model candidate evaluation uses the reloaded payloads.
- Each arm has identical W2.5 payload bytes: Q 1,310,720, K 327,680, V 327,680;
  banks and scales remain fixed across matched arms.
- All layer outputs and model logits remain finite.
- The unmodified F6 reference repeats the prior full-model metrics exactly.
- GPU 0, NVIDIA PG506-230/SM80, exclusive lease and idle preflight passed;
  lease released after execution. The existing CUDA quantization extension was
  compiled for SM80 with `MAX_JOBS=4`; exact build flags are retained locally.

**Decision: no promotion of this W2.5 refinement.** The effect sizes are small,
but this run establishes no propagated-model advantage and has adverse KLD/
Top-1 evidence. This conclusion concerns the current fixed-scale, 33-candidate
GSQ-inspired implementation, not the original scalar GSQ algorithm or future
joint-scale/path refinements. RCO remains independent and unimplemented.

## Artifacts

The [raw report](../../artifacts/gsq-p32/full-qkv-w25-seed7-v2/report.json) records
source/shard hashes, full Fisher statistics, complete projection dimensions,
per-document metrics and uncertainty. Local artifacts additionally retain input
IDs, FP32 teacher logits, real Fisher factors, complete target payload exports,
executed sources, the CUDA build recipe and log. The first attempt stopped on
an invalid partial fixed-transform API call before quantization; it is retained
as failed evidence and contributes no results. The successful run uses the
normal seeded transform initialization.

```bash
PYTHONPATH=. /root/venv-py3.14t/bin/python -m scripts.validate_qvq_gsq_layers \
  --prepare --target-bits 2.5 --output artifacts/gsq-p32/full-qkv-w25-seed7-v2
PYTHONPATH=. MAX_JOBS=4 /root/venv-py3.14t/bin/python -m gpu_allocator.cli run -n 1 --style uuid -- \
  /root/venv-py3.14t/bin/python -u -m scripts.validate_qvq_gsq_layers \
  --target-bits 2.5 --output artifacts/gsq-p32/full-qkv-w25-seed7-v2
```

Use a new output directory for another run. The partial-layer experiment is
not published as a complete model snapshot. Uniform whole-model W2.5,
native-kernel inference parity and task evaluation are not claimed.
