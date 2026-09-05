# Targeted rank8 recovery and fused correction

The fixed source remains the passing joint-rank16 W4A16 base, read from its
existing export. Historical F6 seed7 checkpoint shards are never modified.
This extends experiments 33/34/39/40 and retains every earlier failure.

## A passing rank8 fit

Tail weight alpha=1 (weights 2 for calibration errors above the initial 99.9th
percentile, 1 otherwise), 80 Adam steps, selected by calibration objective,
passes all 9 original cases against both canonical FP32 and production window.
FP16 factors have maximum window drift **0.0465087891**, below 0.046875;
complete serialized operator BPW is **4.3297781944**.
Both FP32 and FP16 factors pass the 25-case expanded document/context replay.
No error exception or threshold change was needed.

Alpha19 and alpha99, also 80 steps, pass 8/9 and 0/9 window cases respectively.
Increasing tail weight does not monotonically improve held-out extremes.
The original alpha9/40-step export still fails its original M2048 window gate,
although it passes 25/25 expanded cases. Expanded contexts do not erase that
recorded development-case failure.

## Fused expansion/add and CUDA Graphs

`rank8_runtime.py` fuses `(XA)B`, its FP16 rounding, native-output FP32 addition,
and final FP16 store into a Triton expansion/add kernel. The W4A16 native GEMM
remains a separate call; this is **not fusion into the native GEMM itself**.
The physical correction MMA rank is 16 while logical/storage rank is 8.
Both window and candidate have matched eager and CUDA Graph measurements;
all timing includes native base, correction, conversions and final output.

| M | Graphed full-linear speedup vs graphed window, alpha1 |
|---:|---:|
|1|2.54x initially; 2.63x after profiling|
|16|1.19x in both runs|
|2048|1.13x in both runs|

Both complete 72-case alpha1 runs pass both local references; every graph replay
matches its eager counterpart bit-for-bit. Fused versus separate output differs
by at most about 1.91e-6 at M2048, and is exact at M1. These are single real
layer0-down measurements on sm80, **not a 2x model inference result**. Model runs
below use the separate recovered operator; integrated fused-model checks remain.

Nsight captures at M1/M16/M2048 include executed, source-correlated SASS. All 74
kernel opcode-count sums match Nsight's executed-instruction totals. At M16:

- Separate recovered operator: 11 kernels, 3,577,296 executed warp instructions.
- Fused expansion/add path: 5 kernels, 3,425,232 instructions (4.25% fewer).
- Expansion/add uses 26 registers/thread, 2.560 KB shared/block, no spills,
  with measured active-warp occupancy about 6.19% for this small grid.
- SASS confirms FP32-output HMMA followed by FP16 conversion, base addition and
  final store. The intermediate correction copy/conversion/add launches disappear.
- The layout conversion still emits substantial integer address/shuffle logic and
  shared stores/loads. This is a remaining optimization opportunity; the native
  GEMM is unchanged and remains the dominant work at larger M.

Per-kernel opcode histograms, resource/traffic/conflict counters, units and artifact
hashes are in [the profile records](results/rank8-targeted/profile-M16-profile.json).
Full `.ncu-rep`, raw CSV and SASS remain under `/root/p32-rank8-targeted`.
Profiler durations are not substituted for the unprofiled CUDA-event speed score.

## Blockwise FP32

Correction input-projection variants promote/reduce at K16/32/64/128/256, with
FP32 partial and final sums, then FP16 hidden output. They retain alpha1 accuracy
but are slower than cuBLAS input projection plus fused expansion/add. At M16,
the profiled K64 projection takes 176.32 us with only 6.25% active-warp occupancy.
Its small output rank leaves too little parallel work in this current unsplit-K
implementation. Native base accumulation was not changed; integrated base-level
promotion is still a separate outstanding experiment.

## Targeted sparse recovery

The original failing element is token768, output channel1666. That localization
uses the original development case. Sparse support and coefficients are fitted
**only on the historical Fisher capture**, using normalized residual correlation
and FP64 least squares (rcond1e-5). No ARC/GSM8K data enters fitting.

Budgets 0/1/4/8/16/32/64/128/256/512 were tested. Budgets16–128 pass all original
window cases. The 16-exception operator uses **4.3300433159 serialized BPW**.
Every budget passes the expanded replay, but budget256 has a reload/repeat
mismatch and is ineligible pending investigation. Budget512 fails two original
cases despite passing the expanded replay. Runtime uses gather/multiply/index-add
before final output rounding; it is not a native fused sparse epilogue. Atomic
accumulation and extra launches require further profiling/robustness checks.

## Model quality and joint optimization

Alpha1 rank8 FP16 has C4 PPL **26.97691913** versus window **26.99152524** on the
small 16-document slice. Full ARC scores are 389/1172 raw and 429/1172 normalized,
versus window 390/1172 and 428/1172. Paired exact tests give p=1 for both; these
small changes establish no quality win. See [full ARC results](FULL_ARC.md).

Rank8 joint native/refit runs with 1/2/4/8 iterations selected steps1/2/4/5 by
calibration MSE. Their original canonical gate counts are 7/9,7/9,7/9,9/9;
selected step5 has max error0.03875141. Its standalone export reload and model
checks are queued. This changes the native base and is experiment40, not the
fixed-base rank sweep. Production-window comparisons and broader checks remain.

Raw numerical, storage, replay and timing evidence is in
[results/rank8-targeted](results/rank8-targeted). No production default is changed.
