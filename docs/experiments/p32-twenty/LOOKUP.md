# Experiments5/6: codebook-index and pair lookup tables

The direct path extracts circular states, applies the original bank XOR and PGC16
mix, and loads the two codebook values. The index-table path replaces the PGC16
mix with a 65536-entry uint16 lookup. The pair-table path replaces mix and codebook
loads with a 65536-entry pair lookup. Bank XOR remains explicit. Both consume the
same read-only lossless window representation; no quantization bits change.

FP32-output variants exactly match canonical decoded pairs on every tile of all 94 P32
projections at one and four warps. Median direct/candidate latency ratios are
0.9120x for index LUT and 0.8161x for pair LUT: neither is faster overall in this
unfused materialization test. This is consistent with a tradeoff between reduced
integer arithmetic and extra table traffic, but profiling has not yet established
the cause. No linear-layer or model speed claim follows from these timings.

The shared index/pair tables are about 128/256 KiB respectively. The accepted
PGC16-v1 level table is FP16; the first pair LUT stores those exact half values
and promotes them to FP32 output. It is not a float32-storage LUT. Reports count the
actual versioned table file, including metadata, once per model and add it to the
complete existing snapshot storage inventory. Runtime scratch and repack are
separate. A packed half2 variant is running to test direct packed pair output. It preserves
the already-FP16 canonical codebook values; this packing does not add codebook
quantization error. Neither variant establishes exact full forward arithmetic.

[FP32 summary](results/lookup/fp32-summary.json), with raw per-projection samples.
The full fused/model scorecard remains outstanding for experiments5/6.
