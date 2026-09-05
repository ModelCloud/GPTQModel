# Experiments5/6: codebook-index and pair lookup tables

The direct path extracts circular states, applies the original bank XOR and PGC16
mix, and loads the two codebook values. The index-table path replaces the PGC16
mix with a 65536-entry uint16 lookup. The pair-table path replaces mix and codebook
loads with a 65536-entry pair lookup. Bank XOR remains explicit. Both consume the
same read-only lossless window representation; no quantization bits change.

FP32 variants exactly match canonical decoded pairs on every tile of all 94 P32
projections at one and four warps. Median direct/candidate latency ratios are
0.9120x for index LUT and 0.8161x for pair LUT: neither is faster overall in this
unfused materialization test. This is consistent with a tradeoff between reduced
integer arithmetic and extra table traffic, but profiling has not yet established
the cause. No linear-layer or model speed claim follows from these timings.

The shared index/pair tables are about 128/512 KiB respectively. Reports count the
actual versioned table file, including metadata, once per model and add it to the
complete existing snapshot storage inventory. Runtime scratch and repack are
separate. A packed half2 variant is running to test 256-KiB pair output directly;
it must be compared with the same rounded FP16 codebook, not labeled FP32-exact.

[FP32 summary](results/lookup/fp32-summary.json), with raw per-projection samples.
The full fused/model scorecard remains outstanding for experiments5/6.
