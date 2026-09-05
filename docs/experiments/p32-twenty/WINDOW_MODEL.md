# Window inner kernel in real Llama inference

Fixed historical F6 seed-7, 94 P32 inner replacements; production transforms and
output boundaries retained. Serialized checkpoint is unchanged; reports include
additional runtime window/codebook cache bytes. TF32 disabled, eager attention.

Two same-device comparisons ran on physical GPUs 2 and 3. GPU 2 uses window for
all row counts; GPU 3 retains planar below M=32. Warmed prefill has 3 warmups and
10 CUDA-event samples per arm. Decode contains 32 growing-cache steps including
the cold first step, so treat its ratio as preliminary.

| Prefill tokens | GPU 2 speedup | GPU 3 speedup |
|---:|---:|---:|
|128|3.396x|3.681x|
|512|7.156x|7.171x|
|2048|7.445x|7.477x|

These are full-model forward ratios against the existing planar runtime, not
isolated GEMM ratios. The earlier baseline and later candidate were on the same
GPU but separated in time. Repeated baseline measurements are in progress to
check temporal effects; no confidence interval or production promotion is claimed.

On 16 bounded C4 documents (4080 next-token positions), window PPL is
26.9915252378136, planar 26.989785799529503, and canonical FP32 P32
26.99102051460363. Original BF16 PPL is 25.3876977933493; this slice does not
establish P32's advantage over BF16. Logits comparison and downstream checks
remain pending. This evaluation text is never used for correction fitting.

Corrected profiling selected exactly layer 1 down-projection M=1 and layer 0
gate-projection M=16. Main-kernel executed warp-instruction counts fall from
61,474,560 to 11,538,176 and 94,476,800 to 10,974,720 respectively. These totals
are not integer/FP32 instruction breakdowns. Nsight replay timings are diagnostic,
not the unprofiled speed score. The original .ncu-rep files remain outside Git at
/root/p32-resumed. JSON summaries include units, registers, shared memory,
occupancy and Tensor Core activity. Post-profile unprofiled sweeps pass all 18
cases under the approved MAE <=0.003 and max <=0.046875 gates.

[Raw reports and paired summary](results/window-model/paired-summary.json).
These are partial experiment 1/10 integration results; no experiment is marked
fully complete by these bounded checks.

## Follow-up evidence

Repeated same-device planar baselines completed after the candidate runs; raw
reports are stored alongside the first baselines. Saved window-versus-FP32 teacher
logits comparison completed on all 4096 positions: mean KL 3.2836589359317e-05,
top-1 agreement 0.995361328125, top-5 0.995849609375, top-10 0.9955078125.
These are propagated diagnostics, not localized kernel gates.

The harness now supports live-model Evalution ARC-Challenge or GSM8K evaluation,
with explicit bounded row counts. Results go only to the external output directory,
avoiding the general evaluation CLI's checkpoint publication path. Four matched
128-row ARC-Challenge runs (BF16, FP32 teacher, planar, window) are in progress.
