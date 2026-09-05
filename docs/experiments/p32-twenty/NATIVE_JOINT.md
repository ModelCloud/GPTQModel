# Experiment 20: two joint native/correction rounds at rank16

Starting from the one-shot W4A16 base and rank16 output fit, each round quantizes
W_teacher-AB into a fresh independent native base and refits AB from the actual
new base's calibration-output residual. Calibration is 8192 verified historical
Fisher tokens with the actual FP16 input/output boundary. The same activation
SVD is reused, with the same 1e-5 cutoff. No teacher tensors are mutated.

Selection uses minimum actual calibration-output MSE among steps 0, 1 and 2;
held-out C4 metrics never select an iteration. q/k retain step0 because later
rounds worsen calibration MSE. Gate/down select step2. Every candidate has rank16
and the same group128 tiled INT4 packing. Serialized operator BPW is unchanged:
q 4.7554, k 5.5216, gate/down 4.5639.

Down calibration MSE improves from 4.8890e-6 to 4.7912e-6; held-out MAE improves
from 0.00183859 to 0.00182597 and max from 0.04452360 to 0.04010366. It still
passes all nine row counts, including a separate trusted-export reload check.
Gate held-out MAE improves only from 0.0112720 to 0.0111169; q/k/gate still fail
the local gate. No inference exception or model-wide promotion is approved.

All four final candidates were timed against same-device planar/window operators
at all nine required M values. These are layer measurements; model validation and
native profiling are running separately. Per-layer rank allocation, larger rank
budgets, full model BPW and uncertainty/quality checks remain outstanding.
This is partial experiment20 evidence, not a complete joint optimization study.

[Reports](results/native-joint/down_proj.json) include every iteration's calibration
and held-out errors, selection, final timings, exact serialized-byte counts and
scope limitations. The original historical F6 seed-7 checkpoint remains read only.

## First model and profile follow-up

Joint down-only rank16 C4 PPL is **26.989909349246798**, versus window
26.9915252378136 and one-shot down-only rank16 27.035818912832937. This is a
bounded 4080-next-token slice, not a statistically established quality gain.
No same-device whole-model speed ratio is inferred from these different runs.

The warmed M=16 recovered operator profile contains nine kernels: dtype conversion,
tinygemm INT4-storage/BF16 arithmetic, FP32 correction GEMMs/reduction, addition,
and final FP16 conversion. The tinygemm kernel reports 3.67% Tensor Core activity
in this replay. Full metric values and units are retained. This confirms Tensor
Core work in the native base, not INT4-activation IMMA or high overall utilization.
Instruction-class counts and an end-to-end timeline remain outstanding.

GSM8K cot runs for FP32 teacher, original BF16, window and joint down-only are now
running on four GPUs, with the same first 128 examples and 256-new-token cap.
Any scores will be reported as that bounded/capped evaluation, not full GSM8K.
