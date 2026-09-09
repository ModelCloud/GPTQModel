# GSQ-inspired refinement: complete F6 seed-7 QKV verification

## Scope and provenance

This supersedes the 32x32 slice as the relevant evidence for this experiment.
The complete first decoder block's Q, K and V projection matrices were refined;
every evaluation passed through the entire 16-block model. The remaining 109
quantized projections retained the exact F6 seed-7 payloads. This is not a
re-quantization of every layer or a reproduction of scalar GSQ's joint scale
learning.

- Snapshot: `modelcloud-qvq__llama-3.2-1b-instruct__f6-p32__qvq-p32-gguf-exl3__yaqa125x__seed7__20260904__commit5c5979194dc0__aff65a505e88`.
- Historical quantization commit: `0f6cb786c071d00cbba76e7da9f29df80a8e5521`.
- Experiment code base: `5d9a9d5e54a3383647cf7c045b1161dd5b316955` plus the
  source files hashed in the raw report. Exact executed sources are retained
  locally alongside the run; subsequent CLI provenance hardening does not
  change the executed quantization math.
- F6 config: 112 quantized projections, 94 P32; Q/K/V at W2/W2.5/W3.5,
  `pgc16-v1`, RHT, YAQA seed 7, dense endpoints and the snapshot's mixed-rate
  overrides. All saved non-quantized parameters were copied from the snapshot.
- Calibration: the original `yaqa182-nm10000.parquet`, SHA-256
  `5a2429da9754040e16baf47c569c14267b92126f37caafd4fafc8d5bfb5c3f39`.
  Seed-7 stratified selection of eight YAQA and eight NM documents, 3,767 tokens
  with a 256-token cap per document. Objective weights retain YAQA 1.25 / NM 1.0.
  This balanced bounded sample is not the historical 182/10,000 corpus mixture
  or a repeat of all 10,178 original calibration rows.
- Evaluation: seed-7 selection of 32 locked divergence documents, 6,367 tokens,
  with the same 256-token cap. Evaluation file SHA-256 and the historical
  disjointness-audit hash match the snapshot's recorded bindings. Refinement
  never uses these documents for fitting or checkpoint selection.
- Prompts: tokenizer chat template, no generation prompt; exact token IDs,
  selected document indices and hashes are retained in the local `inputs.json`.
- Physical GPU 0: NVIDIA PG506-230, UUID
  `GPU-737e2423-874a-23a4-1126-dfbe3e77c294`, PCI `00000000:DE:00.0`, SM80,
  124 SMs, 96 GiB advertised memory. Exclusive allocator lease and three idle
  preflight samples passed; the lease was released after execution.
- PyTorch `2.15.0.dev20260817+cu130`, CUDA 13.0, eager attention, FP32 weights,
  activations and matmuls, TF32 disabled. This is the canonical QVQ reference
  backend, not a native FP16 kernel speed or deployment-parity claim.

## Controlled comparison

Each tile receives 33 valid circular histories: its original payload and 32
seeded one-bit mutations. GSQ-inspired refinement optimizes categorical logits
for 100 steps and retains the best hard calibration checkpoint. The deterministic
control searches the same candidates with three full input-block coordinate
sweeps, using the complete output residual. Both preserve scales and banks.

Calibration hooks capture full projection inputs from the dense model. The
objective uses the full transformed input dimension. Constant absolute SV is
checked explicitly: under this condition orthogonality makes inner normalized
MSE equivalent to full deployed-output normalized MSE. Held-out layer MSE is
measured after both RHT transforms and saved SU/SV scales against the dense
projection on identical inputs.

## Full-projection results

These are document-mean, held-out output MSE values; lower is better.

| Projection | Full weight shape [out,in] | Rate | F6 baseline | GSQ-inspired | Deterministic |
|---|---|---:|---:|---:|---:|
| Block 0 Q | 2048 x 2048 | W2 | 0.004186952 | 0.004185688 | 0.004187359 |
| Block 0 K | 512 x 2048 | W2.5 | 0.003827039 | 0.003826581 | 0.003827920 |
| Block 0 V | 512 x 2048 | W3.5 | 0.00003317514 | 0.00003317514 | 0.00003317702 |

The GSQ-inspired hard-checkpoint guard retained the original V payload. Gains
on full Q/K matrices are tiny; the earlier 9.68% slice result does not generalize
to a comparable full-layer improvement.

## Propagated full-model results

All arms run the full F6 model against the identical FP32 dense teacher. Values
below are weighted by valid token count. MSE here is final-logit MSE, distinct
from the projection-output MSE above. Top-N is top-N set intersection divided
by N, with stable vocabulary-index tie breaking; it is not task accuracy.

| Arm | Final KLD | Logit MSE | Logit NMSE | Top-1 | Top-5 | Top-10 |
|---|---:|---:|---:|---:|---:|---:|
| F6 seed-7 | 0.108126780 | 0.440472766 | 0.058726416 | 85.4877% | 83.0313% | 82.9260% |
| GSQ-inspired QKV | 0.107935677 | 0.440426234 | 0.058720476 | 85.4720% | 83.0218% | 82.9292% |
| Deterministic QKV | 0.107884147 | 0.440383814 | 0.058710793 | 85.4563% | 83.0250% | 82.8334% |

GSQ-inspired refinement reduces token-weighted KLD by approximately **0.177%**.
The paired document-mean KLD delta is -0.0002335, with a 95% document bootstrap
interval of [-0.0002930, -0.0001854]. This is a small positive effect on this
split. Document-mean MSE, NMSE and Top-1/5/10 intervals cross zero: their changes
are noise-consistent. The raw report distinguishes document-mean uncertainty
from token-weighted aggregate scores; they are different estimands.

Deterministic search slightly improves KLD further, but its Top-10 document-mean
delta is -0.0010665 (about -0.107 percentage points), with interval
[-0.0014175, -0.0007235]. This is a small measured negative effect on this split.
Do not rank the methods using only KLD or only their local fitting losses.

## Verification and decision

- All selected baseline planar/window decoded weights matched exactly.
- Every refined full projection passed exact window/planar round-trip and saved
  payload reload checks. Full-model candidate evaluation uses reloaded payloads.
- Payload shapes and bytes, rates, banks, SU/SV and endpoint precision remain
  unchanged. Baseline serialized tensor shards occupy 915,535,582 bytes; no
  additional inference tensors are introduced by refinement.
- All layer outputs and full-model logits remained finite.
- Eight focused CPU tests pass, including the full-output/inner-objective
  equivalence check; Ruff and whitespace checks pass.

**Decision: keep experimental, no default promotion.** This verifies complete
real projections with original calibration data and propagated full-model
metrics, but establishes only a small local-candidate benefit. Joint scale
learning, more layers, further independent data, task evaluation, and the native
packed inference backend remain follow-up work. The result does not establish
the benefit of the original scalar GSQ algorithm.

## Artifacts and reproduction

Raw [report](../../artifacts/gsq-p32/full-qkv-seed7/report.json) includes per-document
metrics, paired intervals, full quantization/data bindings and shard hashes.
Local `artifacts/gsq-p32/full-qkv-seed7/` also retains the exact inputs, dense
teacher logits, full projection payload exports, executed source copies and log.
Those partial tensor artifacts are not published as a complete model snapshot.

```bash
PYTHONPATH=. /root/venv-py3.14t/bin/python -m scripts.validate_qvq_gsq_layers \
  --gsq --prepare --output artifacts/gsq-p32/full-qkv-seed7
PYTHONPATH=. /root/venv-py3.14t/bin/python -m gpu_allocator.cli run -n 1 --style uuid -- \
  /root/venv-py3.14t/bin/python -u -m scripts.validate_qvq_gsq_layers \
  --gsq --output artifacts/gsq-p32/full-qkv-seed7
```

Use a new output directory for another run. This does not overwrite the F6
snapshot or re-run its original 3.75-hour quantization.
