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

## Uniform-weight control

Uniform-weight controls completed for tiles 16/32/64/128 and ranks 1/2/4/8/12/16.
Every candidate failed all nine local cases; the tile16 rank16 maximum error was
approximately 2.4783, with about 48 BPW. Changing diagonal activation weighting
to uniform weighting therefore does not rescue this greedy representation. These
are screening results only; no packed GPU kernel or model claim follows.

## Sparse Walsh screen (experiment 30)

The four tile sizes 16/32/64/128 completed with largest-magnitude coefficient
retention at keep values up to the full tile. Every reduced keep value failed
all nine local max-error cases. Full keep reconstructs the folded weights to the
FP32 control (nine of nine cases), but costs about 40 BPW because this screen
stores FP32 values and byte indices; it is therefore not a useful compressed
representation. At tile128/keep64, for example, retained spectral energy is
0.92642 but maximum error is about 7.60531. This rejects the tested sparse
Walsh export as a practical approximation. No packed GPU execution or model
quality claim follows.

[Raw tile reports](results/structured/sparse-walsh-tile16.json),
[tile32](results/structured/sparse-walsh-tile32.json),
[tile64](results/structured/sparse-walsh-tile64.json), and
[tile128](results/structured/sparse-walsh-tile128.json).
