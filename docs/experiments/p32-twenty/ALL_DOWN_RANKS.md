# All-down rank screening (experiment37)

The remaining 15 P32 down projections were independently fitted with native W4A16
bases and FP32 output-aware corrections at ranks0/2/4/6/8/12/16. They use the same
8192-token historical Fisher capture and expanded 4096-token C4 capture, with
canonical FP32 references recomputed on identical FP16-rounded inputs.

**No tested rank passes all nine canonical cases on any layer1–15.** Their
current selection is therefore window. Layer0 retains its separately studied
joint-base rank8/12 candidates; these findings cannot be generalized to other
layers. The lowest-rank all-down fallback map currently changes only layer0.

The 105 rank exports include complete serialized byte counts. Across 945 reload
cases, recorded MAE/max metrics reproduce with zero drift. This checks numerical
metrics, not stored elementwise-output equality, and does not satisfy every
acceptance requirement by itself. Broader per-document replay and full-model
replacement remain prerequisites for any future passing candidate.

This initial screen uses a newly quantized native base for each layer; it does
not alter the F6 teacher and does not repeat the layer0 fixed-base experiment.
Calibration composition, joint optimization and larger/structured residuals may
change the result. The current rank-grid candidates are not promoted.

The layer harness's window timing here uses its older FP32-transform composition,
not production QVQLinear's fused transforms. Those timing ratios must not be used
as production-window speedups or inherited by a model result. Production-matched
measurements are required if a new candidate becomes viable.

[Raw per-layer results, serialized BPW and reload metrics](results/all-down-ranks).
Progressive replacement stages that keep all failing layers as window have the
same layer0-only operator map; rerunning them cannot establish a benefit from
replacing additional down projections.
