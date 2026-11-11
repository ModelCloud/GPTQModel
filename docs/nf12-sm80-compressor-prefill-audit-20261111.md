# SM80 real-compressor NF12 prefill audit (2026-11-11)

The released `layers.20.attn.compressor.wkv.weight` is BF16 `[512,5120]`.
The benchmark uses the same BF16 `[512,5120]` activations and compares the
complete NF12 call with dense Torch `F.linear` in alternating CUDA graph
replays on GPU 7 (PG506-230, SM80, PyTorch 2.15, GIL disabled). The stored NF12
weight restores every BF16 bit. The existing
`scripts/bench_nf12_real_compressor_prefill.py` is the reproducible gate.

| Path | NF12 | Paired dense BF16 | Speedup | Decision agreement |
| --- | ---: | ---: | ---: | --- |
| Current decode plus cuBLAS | 30.04 µs | 24.45 µs | 0.814× | Bitwise output and top-8 exact |
| Best tested native fused MMA, 64×128 tile, split-K 3 | 37.72 µs | 24.45 µs | 0.648× | 53/512 top-8 rows differ |

The native sweep covered 14 tile configurations with split-K 1, 2, 4 and 8,
then six promising tiles with split-K 3, 5, 6 and 7. The 64×128, split-K 3
case was the fastest of those fused trials. The fused path's maximum BF16
output error was 0.00390625, and over 107,000 of 262,144 output bit patterns
differed from dense on this fixture. That difference comes from MMA reduction
order; the packed weight itself remains lossless. The fused experiment was
kept in an isolated worktree and was not added to the PR.

Nsight Compute 2026.2.1 confirms the current execution plan. NF12 launches a
packed `decode_kernel`, then the same cuBLAS Ampere BF16 GEMM and split-K
reduction kernels that the dense reference launches. Profiled kernel durations
were 7.65 µs for decode, 21.18 µs for NF12's GEMM and 8.35 µs for its split-K
reduce; dense GEMM and reduce were 21.34 and 8.67 µs. Profiler replay times
are diagnostic and must not be summed as unprofiled call latency. Decode uses
34 registers per thread and no local spills. Its executed SASS includes 61,440
`PRMT`, 40,960 `SHFL.BFLY`, and 122,880 `SEL` instructions.

The profiled 64×128 split-K 2 fused trial launched 64 MMA CTAs over two K
splits, then a separate reduction kernel. The MMA image used 182 registers per
thread, achieved 12.51% occupancy, and spent 69.92% of scheduler cycles with
no eligible warp. The reduction also occupied a launch; the extra split-K
sweep did not recover parity. A direct Triton per-element packed-gather
prototype took at least 411 µs. Its gather pattern is an implementation
bottleneck, not evidence that a vectorized fused design cannot work.

The result leaves the 1.05× real-compressor gate open. Faster decode alone
cannot be described as an NF12 prefill win while the complete call still adds
work before the identical cuBLAS GEMM. A new candidate needs a packed-tile
producer and Tensor Core consumer with sufficient CTA residency, no separate
split reduction, and the required BF16 decision oracle; then it must pass the
unprofiled full-call and three-lane serving gates.
