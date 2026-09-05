# Signed-basis real-weight screen (16/28)

The layer0 down folded teacher is fit with greedy signed planes and per-tile
FP32 scales, weighted by diagonal historical activation energy. This is not a
full activation-output regression or jointly learned quantizer. Packed sign
planes, scales and metadata are serialized; reported BPW is actual file size.
Execution here reconstructs dense FP32 weights for screening. No packed-GEMM
speed, model quality or deployment claim is made.

The 16-weight tile run completed ranks1/2/4/8/12/16, each with exact serialized
reconstruction reload equality. Every rank fails all nine canonical max-error
cases. At rank16, worst MAE is 0.00193608 but max is 0.47400622, with approximately
48 BPW including scales. The folded dense control passes (max1.90735e-5), so
folding is not responsible for this large approximation error. Other tile sizes
remain in the queue. The read teacher-shard hashes remained unchanged.

[Raw tile16 results](results/structured/signed-tile16.json).

Limitations: calibration uses a diagonal energy approximation; no window output
comparison, original-BF16 comparison, full-model run, or packed runtime profiling
has yet been performed. This result rules out only the tested greedy representation,
not signed-basis quantization generally. The 3e-3/0.046875 gates remain unchanged.
