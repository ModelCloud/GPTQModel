# Guarded FlyDSL FP16 HGEMM trial

Experimental dispatch revision3f3dc857, preceding production5dc2c9da.
Overall target remains1.5x across364 cases versus c89459e3.
Production dispatch is unchanged; `--flydsl-hgemm` is benchmark-only.

## Full post-profile result: do not promote

All364 requested cases completed with strict idle/pre-timing checks,
warmup20/iterations50 and alternating paired CUDA-event timing.
The candidate dispatched192 cases to FlyDSL and preserved all other paths.
352 cases pass canonical max-absolute0.002; the12 unchanged large gate/up
fallbacks remain exact-baseline-equal but above the canonical threshold.
All192 changed cases pass, maximum error0.0018539429. All232 applicable
graph and stream checks pass, including the192 FlyDSL cases.

The candidate geometric mean is **0.860972764x** versus c89459e3, with
36/364 at least1.5x: no new target-reaching cases. The public wrapper is
slower overall, despite initial two-sample pilot observations. Those pilot
timings were compilation/correctness probes, not accepted speed evidence.

| Shape | Geomean vs c89459e3 | Cases >=1.5x |
|---|---:|---:|
| full_q_gate | 0.805407 | 0 |
| full_kv | 0.449059 | 0 |
| attn_out | 2.062225 | 28 |
| linear_qkv | 0.753726 | 0 |
| linear_z | 0.653909 | 0 |
| mlp_gate_up | 0.884117 | 0 |
| mlp_down | 1.079022 | 8 |

Full evidence: `flydsl_3f3dc857_full.json`. This is a synthetic kernel
experiment, not model-quality evidence or production promotion.

## Contract and setup

AITER installed revision7440ef72503e1c3fadc5be85a5c74eb7c9c34841,
FlyDSL0.3.2. Wrapper and kernel source hashes are embedded in each report.
Only M>=8, contiguous FP16 X, physical contiguous N-by-K folded FP16 weight,
same-type output, no residual and no composite recovery are routed. Small
attention retains its existing residual-disabled rule. Public layer mutation
guards remain active. No new per-forward dense weight copy is introduced.
The wrapper rejects combination with raw ceiling modes.

Normal tile128x128x64, four waves, two stages, splitK1, B in LDS.
The installed selection_filter requires256x256x64, eight waves, two stages,
splitK1 whenever min(M,N,K)>=4096; this rule is followed, not bypassed.
Initial128-tile M4096 rejection and root-owned default cache rejection were
resolved before the successful boundary run. Cache override is process-local
`FLYDSL_RUNTIME_CACHE_DIR=/tmp/qvq-flydsl-runtime-cache`; no installation or
system permissions changed. Boundary run48/48 passed,44 dispatched and44
graph/stream passes, before the device-selection commit.

## Executed profiles at full-Q K5120,N12288

rocprofv3 collected all seven metrics in one capture after3f3dc857. Baseline
is5dc2c9da at matching shapes. Reference/preprocessing FP32 GEMMs are not
mixed into the two HHS baseline symbols below. Issued counts are identical
across repeated captures of each symbol (baseline16, candidate28 records per
metric); counts are per invocation, not summed over unequal sample counts.

| Metric | M128 library | M128 FlyDSL | M4096 library | M4096 FlyDSL |
|---|---:|---:|---:|---:|
| VALU | 1,254,144 | 1,702,272 | 35,384,320 | 44,261,376 |
| SALU | 1,310,976 | 756,480 | 10,607,104 | 12,257,280 |
| LDS | 740,352 | 519,168 | 7,919,616 | 12,681,216 |
| MFMA | 983,040 | 983,040 | 31,503,360 | 31,457,280 |
| LDS bytes | 116,224 | 65,536 | 135,168 | 131,072 |
| Threads | 256 | 256 | 256 | 512 |
| MeanOccupancyPerCU | 2.23078 | 1.19734 | 3.51907 | 7.02998 |
| LDSBankConflict mean | 3.20762 | 0.12309 | 0.11803 | 0.80169 |
| SQ_WAIT_INST_LDS mean | 830,247.06 | 437,027.93 | 6,041,852 | 15,843,021.93 |

All report zero scratch. Occupancy is waves/CU, not percentage. Bank conflict
is the installed derived metric, not conflicts per access. LDS waiting uses
four-wave-cycle units; overall scheduling stalls/bandwidth/overlap are not
measured. More occupancy at M4096 does not prove more speed.

Raw rocprof VGPR/AccumVGPR/SGPR fields are32/160/112 and120/384/96 for
the baselines,4/172/112 and76/132/112 for FlyDSL. Embedded ELF notes report
FlyDSL VGPR/AGPR/SGPR132/0/34 and202/0/29, with zero register spills.
These metadata sources differ; preserve both rather than infer a register
partition or occupancy limit from the raw fields. JSONs retain raw profiler
fields; ELF-note values are recorded here.

## Exact binary / SSA / algebra audit

`analyze_qvq_flydsl_isa.py` decodes the embedded binary in the final compiler
MLIR, resolves its exact function symbol/byte extent and disassembles only
that extent. It excludes trailing section padding. CPU tests cover hex,
escaped quote/backslash, missing/invalid headers and ambiguous binaries.
Both extracted binaries match normal runtime-cache pickle payload hashes,
inspected with pickletools without unpickling or executing cache objects:

- 128 tile:992bbdb0bce19386f98ca7cb2e70e0d47f1738db6a6f61641391d87121e432e0
- 256 tile:d1ed153844f49f896417eaeaa252a99bf9f9ce535a7523aa498c24e72916b2f7

Exact static function totals646 and1085 instructions, with64/128 MFMA,
40/64 ds_read_b128,64/128 ds_write_b16,8/16 buffer_store_dwordx4,
23/50 waitcnt, four barriers and three nops each. The large tile additionally
has eight s_setprio instructions. The scalar FP16 LDS stores are the output
layout staging, followed by wide global stores; they are not split-K scratch.
Do not remove required hazard nops, waits or reuse barriers based on count.

Source uses FP16 operands and FP32 MFMA accumulators, narrowing once to FP16
before LDS output staging. SplitK1 and one K-wave eliminate atomic split-K
reduction and cross-K-slice output addition. No atomic opcode survives.
HAS_BIAS=false eliminates bias arithmetic. Constant N/K remove N/K-tail math;
runtime M still emits eight/sixteen EXEC-guarded output-store paths, even at
divisible M. Address work includes XOR swizzling, repeated output row/index
formation and wide-buffer bounds construction. These are next reduction
opportunities, not automatically removable work: one compiled kernel serves
runtime M values including M8, so its M guards are currently necessary.

The two baseline HHS functions resolve exactly in code object
04b15e474ada427872ec96867888a4828cad9135f6fa0c2bb813381f592824fd.
Whole-symbol totals20260/218107 include alternate argument/epilogue branches,
so their difference from FlyDSL totals is not an executed-instruction saving.
The counter table above is the executed comparison. Seven other selected
reference/candidate symbols are explicitly marked absent from that library
code object; they are not silently attributed to it.

## Next action and verification

Do not select this public wrapper as a winner. First separate wrapper cost
from device time with cached compiled-launcher metadata and a matched trace,
while retaining fresh input/output pointers, current-stream binding and
public layer guards. The wrapper currently normalizes streams twice and
retrieves semaphore/signal tensors even for splitK1. These are source-observed
operations, not yet a measured explanation for the regression. Small-M tile
occupancy and large-M LDS waits also need attention. Do not enable splitK>1
without a new numerical audit: the inspected split-K branch narrows partials
before output atomic addition, unlike this splitK1 configuration.

After the profile and full warmed sweep, all1080 AMD/experimental/analyzer
tests passed (14existing deprecation warnings,14.03seconds), including six
new CPU decoder cases. Ruff and whitespace checks passed on changed files.

Profile JSON:`flydsl_3f3dc857_profile.json`; exact ISA/counters:
`flydsl_3f3dc857_isa.json` and `flydsl_3f3dc857_baseline_isa.json`.
Raw `/tmp/qvq-flydsl-profile/raw/ubuntu2404-mi350x/896635_counter_collection.csv`;
IR `/tmp/qvq-flydsl-profile-ir`; exact ISA `/tmp/qvq-flydsl-exact-isa`.
Profiler uses sudo process-local ROCm library paths, FLYDSL_DUMP_IR=1 and
fresh cache, regex `.*hgemm.*|Cijk.*`, full_q_gate M128/4096,warmup1/iterations2.
Post sweep uses --flydsl-hgemm --butterfly none --baseline-amd-commit c89459e3
--full-sweep --warmup20 --iterations50 (space-separated CLI values), output
`/tmp/qvq-flydsl-post-full-retry/report.json`, terminal exit0.
The first post sweep failed its foreign-PID check and is not used. No foreign
process was killed or idle restriction relaxed. Test log:
`/tmp/qvq-flydsl-post-tests.log`. No production change or goal completion claimed.
