# Experiment 2: existing-kernel row-group ablation

All five limits (1, 2, 4, 8, 16 activation rows per window-kernel call) were measured
on real FP32-teacher activations of layer 0 MLP gate-projection across
M=1,2,4,8,16,32,128,512,2048. **45/45 cases pass** MAE <=0.003 and max <=0.046875.
All consume the same read-only F6 seed-7 weights and lossless window repack.

This experiment schedules the existing Ampere kernel on activation chunks, then
concatenates results. It measures actual decoder reuse together with launch,
allocation, concatenation, and dispatch effects. It is not a controlled in-kernel
reuse-factor sweep and must not be used to attribute the entire gain to decoder
instructions. The current window kernel already supports 16-row MMA tiles.
Each report includes same-device planar comparisons and full transform timing;
row-group configurations ran on different GPUs, so their absolute times are not
a matched-device cross-configuration speed comparison.

No metadata is added to the checkpoint. Runtime output/chunk allocation costs
are included in latency, but allocator-resident bytes still need explicit profiling.
Full-model candidate integration, further projections, and a controlled fused
kernel sweep remain required before marking experiment 2 complete.

[Measurements](results/row-reuse/reuse16.json); sibling reports cover all five limits.
