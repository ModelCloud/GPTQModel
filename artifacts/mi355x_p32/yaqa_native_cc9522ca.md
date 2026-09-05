# Native ROCm YAQA: verified first implementation, not the 100x goal

Code revision: `cc9522caaf4d0db9adab64cbc60622b4b029b456`, based on
`origin/main` `bc36a5ba` through the ROCm reference guard at `7beba298`.
The default branch is main; this remote has no master. Work branch:
`codex/qvq-rocm-quantization-wip`.

## Scope and activation

`GPTQMODEL_QVQ_AMD_NATIVE_QUANTIZATION=1` enables the experimental native
V2/P32 quantizer on runtime-probed gfx950, for contiguous FP32 sequences,
FP16/FP32 codebooks, W2/W2.5/W3/W3.5 and batches 1–256. Canonical V2 and
two-bank V2B2 use the same exact survivor/traceback implementation. Other
shapes, precisions and architectures keep their existing fallbacks. Strict
pruning policies still reject the unpruned ROCm route.

This is opt-in, not a default promotion. It covers native Viterbi, two-pass
tail biting, public dispatch and a small complete YAQA family-reselection
solve. Large-matrix factored YAQA feedback still has NVIDIA-specific native
calls; it is not ported by this change. Real Qwen model calibration,
quantize/save/reload and model-quality evaluation are not certified.
The 100x end-to-end model-quantization target is **not achieved**.

## Equivalent public-call timings

Three warmed event samples per implementation, same input tensors, including
public validation and dispatch. Reference remains eager ROCm, explicitly
disabling the native opt-in. Every native output matches its independent
reference exactly. Full samples/fingerprints are in
`yaqa_native_cc9522ca_public_paired.json`.

```text
batch  bits   eager_ms  native_ms  speedup
    1   2.0     17.716      2.187    8.10x
    1   2.5     17.532      2.197    7.98x
    1   3.0     17.458      2.162    8.07x
    1   3.5     17.612      2.199    8.01x
    4   2.0     17.682      2.234    7.91x
    4   2.5     17.005      2.189    7.77x
    4   3.0     17.042      2.173    7.84x
    4   3.5     17.057      2.180    7.82x
   16   2.0     17.034      2.234    7.63x
   16   2.5     17.688      2.190    8.08x
   16   3.0     17.603      2.209    7.97x
   16   3.5     17.668      2.211    7.99x
   64   2.0     17.876      4.605    3.88x
   64   2.5     17.922      4.437    4.04x
   64   3.0     18.003      5.267    3.42x
   64   3.5     17.955      4.578    3.92x
  256   2.0     49.164     14.946    3.29x
  256   2.5     48.784     15.204    3.21x
  256   3.0     48.157     18.596    2.59x
  256   3.5     48.908     15.476    3.16x
```

The separate complete 32x32 YAQA solve uses production PGC16 codebooks,
synthetic weights and SPD Hessians, full family reselection and full sampling.
It compares all four returned tensors exactly with native mode off/on and
alternates timing order. See `yaqa_native_cc9522ca_pipeline.json` for final
post-profile synchronized wall timings. This is algorithm/kernel evidence,
not synthetic evidence of model quality. The initial measured solve improved
about 6.7x; the final artifact is authoritative.

Graph replay experiments are deliberately separate. They exclude graph
preparation and public validation and are not the production entry point.
Batch-1 trusted replay was approximately 0.36–0.49ms; that is not a full
YAQA speedup. The native recurrence has 128 survivor launches plus one
traceback launch, versus 2350 launches in the prior eager public trace.
The 129 count excludes public validation/setup kernels.

## Exact recurrence and algebra audit

Let P=2^(2*bits), S=65536/P, and F[t,b,s] be the original cost. Retain
R[t,b,q]=min_p F[t,b,p*S+q], rather than materializing all F between steps.
Each emission adds the prior survivor R[t-1,b,s>>shift]. Before the next
segment boundary, merge the bank survivors and retain the bank-major prefix
index. P32 has 16 V2 steps, not 32; 128 steps produce eight selectors.

The transformation reuses already-rounded cost values. No precision reduction,
new pruning policy, reassociation or altered tie rule is accepted. Prefix
ties choose the lowest prefix; bank-boundary ties choose bank then prefix;
terminal ties compare the **original bank*65536+state**, not the compressed
suffix index. Traceback restores every state and selector. Start/end overlap
masks, nonnegative weighted costs and the original FP32 work dtype remain.

Emission follows the reference evaluation order:
`max((x0*x0+x1*x1)+(c0*c0+c1*c1)-2*fma(x1,c1,x0*c0),0)`.
Only the explicit two-term dot-product FMA contracts; compilation uses
`enable_fp_fusion=False`, no reduced-mantissa math and no denormal flush opt-in.
LLVM SSA inspection confirms separate norm/add/sub rounding and an explicit
`llvm.fma.f32`. Byte-exact states, banks and packed words—not an inference
tolerance—are the acceptance gate.

The SSA/address pass checked constant shifts/masks, common state/codebook
addresses, int16 pointer loads, final int64 widening, norm computation,
reduction permutations and scratch use. The compiler already folds constant
power-of-two division/modulo and vectorizes some adjacent loads. Norm hoisting
was tested explicitly rather than assuming fewer multiplies means faster.

## Executed AMD ISA/counter profiles after cc9522ca

Hardware: MI355X VF, gfx950, 256 CUs, physical GPU0/BDF 0000:83:00.0;
Torch 2.13.0+rocm10.0.0, HIP 7.15.26333, Triton
3.8.0+git4cff872c.rocm10.0.0. Full identity/driver data is in timing reports.
Strict three-sample 0%-utilization/no-foreign-PID preflights and pre-timing
PID rechecks were used. Transient post-run utilization failures were retried
later without weakening the gate.

AMD has no NVIDIA SASS. The equivalent audit used executed rocprofv3 counters,
source-correlated gfx950 assembly, LLVM SSA and exact HSACO/IR hashes. The
checked-in FP32/FP16 audit JSONs record each compiled variant and each profiler
kernel ID. Runtime LDS allocation is `metadata.shared`; the static assembly
group-segment field alone is insufficient.

```text
variant                 VGPR range  static ISA range  private bytes  dynamic LDS bytes
FP32 survivor              16–24         134–477             0          16–128
FP32 traceback             17–74         269–640             0             16
FP16 weighted survivor     14–21         160–494             0          16–128
FP16 closed traceback      20–87         289–726             0             16
```

All four rates were executed for FP32 open/unweighted and FP16 closed/weighted.
No spills occur in these inspected variants. The canonical one-bank variant
has exact functional checks but is not separately certified by these two-bank
counter captures. There was no prior committed native AMD quantizer to compare
instruction-for-instruction; the prior committed implementation is the eager
Torch operator sequence, documented in `yaqa_rocm_7beba298_mapping.json`.

Example W2 FP32 issued VALU counts per launch are 290816 (initial), 311296
(ordinary), 348160 (bank merge), and 10591 (traceback). These are wave
instruction metrics, not FLOPs. `MeanOccupancyPerCU` is waves/CU, not percent;
`LDSBankConflict` is the SDK's derived metric; `SQ_WAIT_INST_LDS` is reported
in its SDK units, not a normalized stall percentage. Achieved bandwidth and
general scheduler-stall percentages were not measured.

Raw captures and compiler caches:

- `/tmp/qvq-yaqa-cc9522ca-fp32-pmc`, `/tmp/qvq-yaqa-cc9522ca-fp32-cache`
- `/tmp/qvq-yaqa-cc9522ca-fp16-pmc`, `/tmp/qvq-yaqa-cc9522ca-fp16-cache`

Both captures used `sudo -n`, process-local
`LD_LIBRARY_PATH=/opt/rocm/core-10.0/lib:/opt/rocm/core-10.0/lib/rocprofiler-sdk`,
`rocprofv3 --pmc SQ_INSTS_VALU SQ_INSTS_SALU SQ_INSTS_LDS SQ_INSTS_MFMA
MeanOccupancyPerCU LDSBankConflict SQ_WAIT_INST_LDS`, CSV output, and kernel
filter `_survivor_step|_traceback`. The benchmark used `--native --batch-sizes 1
--bits 2 2.5 3 3.5 --iterations 2`, adding `--codebook-dtype fp16 --closed
--weighted` for the second capture. Profiled latency is not used as speedup.

## Rejected deduplication and fusion experiments

1. **Hoisted codebook norms:** exact across all 12 B1/4/16 and rate cases,
   but slower in every tested warmed replay case (speedup 0.884–0.990x).
   Two FP32 products and their sum were removed from each state emission,
   but a norm load and preparation kernels were added. On W2.5 the ordinary
   survivor's static ISA increased 161→199 even as VGPRs fell 19→15; on W3.5
   it fell 455→447 but latency still regressed. This is measured evidence
   against retaining the change, not a license to reassociate the metric.
   See norm baseline/rejected timing and audit JSONs; raw profile/cache:
   `/tmp/qvq-yaqa-norm-pmc`, `/tmp/qvq-yaqa-norm-audit-cache`.
   Rejected source: `/tmp/qvq-yaqa-rejected-norm-candidate.py`.
2. **Single-CTA persistent Gluon:** exact W2/B1, but ~25.6ms; 3416 private
   bytes/thread and 10333 static instructions. Register spills overwhelm
   launch savings. Not retained.
3. **Cooperative Gluon:** exact initial W2/B1 (~1.24ms), with a legal HIP
   cooperative launch. A Q128 sweep passed all eight B1/B4 rate cases but
   ranged ~0.83–2.99ms and did not justify global-barrier complexity.
   It is not a public dispatch path and has been removed.
4. **Volatile-load barrier polling:** stalled its own benchmark; terminated
   only PID 937337, process exit143. GPU returned to 0% utilization with no
   KFD PIDs. No reset, foreign-process kill or claimed successful profile.
   Exact root cause was not established; do not reuse this barrier. The
   working atomic-polling variant is distinct from this rejected experiment.

The Gluon experiment snapshots are outside the retained source at
`/tmp/qvq-yaqa-rejected-experiments.py`; their executed audit summaries are
checked in. AITER/Primus-Turbo/rocBLAS dense GEMM dispatch does not implement
the required min-plus trellis reduction. No GPL YAQA/QTIP code was copied.

## Validation and next work

The final post-profile focused suite passed **88 tests**, with 14 existing
Python 3.14 deprecation warnings, in 9.84 seconds. It covers FP16/FP32, open/closed and weighted solves,
zero-weight ties including extreme overlaps, canonical/banked tail biting,
packed words, non-default streams, mutated-input graph replay, full small
YAQA selection and strict pruning. Final test output is
`/tmp/qvq-yaqa-final-guard-tests.log`. Core sweeps additionally pass exact
outputs through batches 1,2,4,8,16,32,64,128,256 at all four rates.
No quantization decision or quality threshold was relaxed. Ruff passes for the
native module, changed benchmark/audit scripts and new tests; `git diff --check`
passes. The large existing `qvq.py` file still has three unrelated Ruff findings
(UP035, UP037, B023); they were not silently fixed as part of this kernel work.

Next priorities are amortizing the Python launch chain with an explicitly
owned prepared executor, testing exact adjacent-step fusion, and porting the
large-matrix factored feedback path. Model-sized calibration/serialization
validation must precede default promotion or a model-level speedup claim.
Generic reassociation of squared distance is not an accepted shortcut.
