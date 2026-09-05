# Cached FlyDSL launcher: verified stopping point

Implementation edd4862f; comparison0e3f4e02 production. User requested stopping
optimization and preparing the verified branch for merge. No further tile,
precision, dispatch-selection or compiler experiment is enabled.

The benchmark-only --flydsl-direct option retains the same splitK1 kernels,
FP16 operands/output and FP32 accumulation. It caches the compiled callable,
owning weight tensor/pointer metadata, and per-device/current-stream metadata.
Each call supplies fresh input/output pointers and runtime M. Public layer
guards remain active. Entries are cleared at shape/rate boundaries and cleanup.
No direct launcher is imported into production.

Correction to the initial investigation: the installed _get_split_k_tensors
has an lru_cache(maxsize=128) decorator. Its two zero allocations are NOT
per-call work after warmup. There is no demonstrated removal of two steady-
state GPU launches. The matched trace shows one GEMM for each path.

After commit, Torch CPU/GPU mapping traces covered full-Q and full-KV M128,
four rates. Full-KV baseline GPU durations were12.919..13.479us versus
44.719..48.880us for FlyDSL. Full-Q baseline42.319..44.360us versus
52.760..54.279us for FlyDSL. These perturbed, single-dispatch trace durations
locate the device-side gap; warmed event timing below determines performance.
Host annotation durations include profiler overhead and are not speed claims.

Normal runtime-cache payloads still match both previously profiled FlyDSL
binary hashes exactly (992bbdb0...1e432e0 and d1ed1538...916b2f7; full hashes
in flydsl_3f3dc857_isa.json). No tile, codegen option, MFMA evaluation order,
output conversion, mask, address algebra or GPU opcode changed. The preceding
executed PMC/ISA/SSA audit therefore remains applicable; this phase adds
executed launch mapping rather than inventing new instruction deltas.

Post-profile warmed paired verification: full-Q/full-KV, M8/128/4096,
rates2/2.5/3/3.5, warmup20/iterations50. All24 cases passed,20 dispatched,
all20 graph/stream checks passed, maximum canonical error0.0016229153.
Strict idle/pre-timing gates passed and the process exited0.

| Shape | M8 speedup | M128 speedup | M4096 speedup |
|---|---:|---:|---:|
| full_q_gate | 0.60705 | 0.81129 | 0.88317 |
| full_kv | 0.23361 | 0.27784 | 1.00024 |

These geometric means over four rates are versus current production0e3f4e02,
not versus the previous public FlyDSL wrapper. Four full-KV M4096 cases use
the unchanged fallback. No speedup or promotion is claimed. No new full364
sweep was necessary for promotion because this candidate is not promoted.
The earlier full public-FlyDSL sweep remains0.86097x overall versus c89459e3;
production remains36/364 target-reaching cases, with the documented12 unchanged
large gate/up canonical-error exceptions. The1.5x-all-cases goal is unmet.

Final stopping-point suite:1132passed,14existing Python3.14 deprecation
warnings,14.34seconds. Files: tests/test_qvq_p32_amd.py,
tests/test_qvq_p32_amd_butterfly_experiment.py, tests/test_qvq_flydsl_isa.py,
tests/test_torch_ops_jit_extension.py. Log /tmp/qvq-merge-ready-tests.log.
Changed-script/analyzer lint and whitespace checks passed. No optimization
jobs remain running. Existing unrelated untracked artifacts are preserved.

Reports: flydsl_direct_edd4862f_post.json and
flydsl_direct_edd4862f_mapping.json. Raw traces:
/tmp/qvq-flydsl-direct-mapping/*_m128.json. Post log:
/tmp/qvq-flydsl-direct-post.log. Exact software/config/source fingerprints
are embedded in the reports. Mapping uses --torch-profile-dir with warmup2/
iterations2; post uses --flydsl-hgemm --flydsl-direct --butterfly none
--baseline-amd-commit0e3f4e02 with the shapes/M values above (CLI flags and
values separated by spaces).

Merge preparation fetched origin/main3a171111. The tested branch already
contains main83c8fc33; the newer upstream FP8 replay changes were not merged
after the user's stop instruction. GitHub reported MERGEABLE/CLEAN against
the current base. PR116 is open and non-draft; no merge was performed and no
claim of validating the combined merge result is made.
