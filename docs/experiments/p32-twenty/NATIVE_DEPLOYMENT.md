# Native export reload and deployment boundary validation

All 16 standalone exports (four projections, four ranks) reloaded and reproduced
all stored FP32 output mean/max errors exactly across nine row counts. The reload
harness then fed identical FP16 inputs to each replacement and the FP32 canonical
teacher. This revealed that the original FP32-input fit does not satisfy the
production FP16 boundary: even down rank128 fails max error at every M.

Down rank128 MAE remains within 0.003, but max error is about 0.12292 versus the
0.046875 gate. Canonical teacher output rounding to FP16 contributes at most
0.00357, so the failure cannot be excused as unavoidable output rounding. Input
rounding before BF16 base execution and correction changes the deployed operator.
The next fit explicitly uses rounded FP16 inputs and recomputes the canonical
teacher on those same values. No gate exception has been approved.

A diagnostic full-model run replaced only layer 0 down-projection rank128 in the
window model, preserving output dtype. C4 PPL is 27.02072721849153 versus window
26.9915252378136. This small slice is not a model-quality acceptance decision.
The concurrently recorded baseline and candidate are on different GPUs; their
timings must not be used as a matched-device speed ratio. The fixed source
checkpoint remains read only; only explicit runtime modules are replaced.

Calibration was expanded to 16 independent historical Fisher sequences of 512
tokens (8192 total), captured over the same 14 projections. The prepared source
manifest already passed the saved historical dataset hash. This remains a bounded
subset of the verified source, not a recreation of historical quantization. C4 and
ARC data are excluded from fitting. Four refits with the actual FP16 boundary are
running; final results and model integration remain outstanding.

[Raw evidence](results/native-deployment/down_proj-reload.json), with the model
reports and expanded capture manifest in the same directory. The standalone
FP32-input success reported earlier remains valid only at that tested boundary.
