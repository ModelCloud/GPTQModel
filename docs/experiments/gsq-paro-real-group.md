# Real grouped ParoQuant GSQ pilot

Real Llama 3.2 1B decoder layer 0, selected Q/K/V compute-block optimization,
seed7, W4/group128/krot8. Two rotation and two finetuning epochs initialize
each arm; GSQ uses 100 steps. The rest of the decoder stays dense. This is a
bounded grouped pilot, not the default ten-epoch recipe or a complete-model
recovery measurement.

Exact dense embeddings for the locked F6/seed7 token documents feed the actual
decoder before normalization. Calibration is 12 training / four initializer
validation documents; 32 held-out documents never select checkpoints. Inputs
are unweighted: scaling inputs by source weights would change the nonlinear
decoder computation. Layer-0 clean/noisy inputs coincide. This does not verify
later-layer paired noisy propagation.

| Arm | Held-out full decoder output MSE | Payload vs baseline |
|---|---:|---|
| Baseline | 1.580617086139014e-6 | Baseline |
| Fixed GSQ | 1.5965070349405958e-6 | Q, K and V changed |
| Learned-scale GSQ | 1.580617086139014e-6 | All identical |

Fixed GSQ lowers each local calibration objective but increases held-out decoder
output MSE by 1.0053% (absolute 1.58899588e-8). No recovery/promotion is claimed.
This single aggregate pilot has no paired uncertainty estimate; its practical
model-quality impact is unestablished. Final-logit KL/Top-K are not measured.

Actual ParoLinear Q/K/V payloads were packed, saved, strictly reloaded and
installed into the decoder. All 288 native projection checks pass against the
export-coordinate reference on identical inputs. Worst mean absolute drift:
0.00038643842; maximum: 0.01028251648. Full decoder MSE compares FP16 native
inference with dense FP32 decoder outputs on the same FP16-cast inputs; this
propagated metric is separate from the native projection acceptance gates.

Executed on leased SM80 PG506-230, physical GPU0, PCI DE:00.0. Evidence:
`artifacts/gsq-paro/real-group-compute-block-seed7-v1/` contains reports, configs,
payloads, compressed executed scripts/log, and SHA256 manifest. Prepared exact
inputs are in `artifacts/gsq-paro/real-group-layer0-seed7-v1/`. These selected-layer
artifacts are not published full-model snapshots.

## Layer-scope follow-up

The separate live `layer` optimizer path completed the same selected-QKV
experiment on the same data, rates and epoch budget. Baseline held-out decoder
MSE: 1.5046040714217985e-6. Fixed GSQ: 1.5184046743833112e-6
(+0.9172%). Learned scales: 1.5046040714217985e-6, with identical
baseline payloads. Fixed GSQ changes Q/K/V payloads. All 288 native checks pass;
worst mean drift 0.0003840324644, max
0.009357452393. Results confirm the same local-versus-layer
tradeoff observed in compute-block scope; no recovery is claimed. This remains
a layer-0 selected-QKV experiment, not full-layer quantization of every linear.
Evidence: `artifacts/gsq-paro/real-group-layer-scope-seed7-v1/`.
