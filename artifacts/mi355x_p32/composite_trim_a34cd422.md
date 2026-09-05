# Composite recovery padding experiment

Production remains unchanged. Baseline for this local experiment is `23d06be7`;
the overall 1.5x target remains relative to `c89459e3`. Integrated origin/main
`9dcaf07e` in merge `789e6393`. Experimental implementation: `a34cd422`.

## Math and scope

For the 40-by-128 down-projection recovery, split the real base dimension
into 32+8 (tail padded to 16) instead of padding the entire dimension to 64.
Compute `B_lo (X_lo H) + B_hi (X_hi H)`, preserving all real terms, FP32 IEEE
products/accumulation, normalization and SV scale, and final FP16 output.
This is equivalent real algebra with changed floating-point evaluation, not
a bitwise claim. No precision reduction, packed representation change, or
production dispatch change. Tests include canary rows and nonunit SV.

Exploratory sweep: M=32,128,512,2048,4096; K=17408; N=5120; all four rates;
warmup 10, 30 paired alternating event samples; canonical max-absolute gate
2e-3. Eight-wave variant passes 20/20 but regresses; four-wave variant also
passes 20/20 and shows small gains at larger M, not a 1.5x improvement.
These synthetic kernel tests do not establish real-model quality.

## Executed instruction audit

MI355X gfx950, physical GPU0, BDF 0000:83:00.0, unique ID
0x333ef6e01ec019b3, 256 CUs. Torch 2.13.0+rocm10.0.0,
Triton 3.8.0+git4cff872c.rocm10.0.0. Strict three-sample idle gates passed.
Matched M4096, N5120 recovery: 8192 CTAs; workgroup 512 baseline/8-wave,
256 four-wave. rocprofv3 SQ counters are issued wave instructions, not FLOPs.

| Metric | Retained | Trim 8 waves | Trim 4 waves |
| --- | ---: | ---: | ---: |
| Static MFMA | 96 | 184 | 144 |
| Issued MFMA | 6291456 | 12058624 | 4718592 |
| Issued VALU | 14352384 | 23658496 | 11960320 |
| Issued SALU | 2490368 | 2883584 | 1572864 |
| Issued LDS | 6356992 | 10551296 | 4194304 |
| VGPR descriptor | 116 | 205 | 158 |
| Dynamic LDS bytes | 32768 | 40960 | 32768 |
| Static barriers | 9 | 14 | 15 |
| Scratch bytes | 0 | 0 | 0 |

Four-wave layout removes 25% of issued MFMA despite a larger static body:
144*4 versus 96*8 instructions per CTA. Eight-wave lowering duplicates
enough work to erase the source-level savings. Four-wave SSA/address review
still finds two independently staged input subsets and layout transfers;
the extra static barriers and live registers are important remaining costs.
No measured occupancy, scheduler stalls or bank-conflict counters were
collected; do not infer those metrics from register/LDS counts.

Baseline JIT hash: 70065b317c7f2ef0f05f2ad5bd70bcaad88ce84e64949c79d15005c889d23605.
Four-wave JIT hash: 90c4cb73d67ddad810147ed030a6971be00765c77e6bf12f0d02aeeefcfcf329.
Four-wave audit was collected after experiment commit a34cd422.
Post-profile focused tests: 274 passed, 14 warnings in 11.66s
(`/tmp/qvq-trim-tests.log`); Ruff and whitespace checks passed.
Post-profile four-wave paired timing again passes all 20 cases, with
1.018-1.058x speedup at M512/2048/4096 and mixed M32/128 results.
Compact timing evidence is preserved alongside this note in
`composite_trim_a34_w8.json` and `composite_trim_a34_w4_post.json`.

Merged production regression sweep (experiment disabled): 364/364 cases
pass, 36/364 at least 1.5x against c89459e3, geometric mean 1.22217x.
All requested M and seven Qwen shape groups across four rates are covered.
The full target is not met. Report: `qwen38_27b_merged_a34_full.json`;
raw `/tmp/qvq-merged-a34-full/report.json`. Command uses `--full-sweep
--butterfly none --baseline-forward-commit c89459e3 --baseline-amd-commit
c89459e3 --warmup 10 --iterations 30`. No target threshold or precision
contract was relaxed. This does not certify newly merged NVIDIA kernels
on this AMD-only host.

## Reproduction and raw evidence

Benchmark: `python scripts/benchmark_qvq_p32_amd_butterfly.py --butterfly none
--trim-composite --recovery-warps 4 --baseline-amd-commit 23d06be7
--shapes mlp_down --m-values 32 128 512 2048 4096 --warmup 10 --iterations 30
--output /tmp/qvq-trim-composite-w4-post/report.json`.

Profiler uses sudo with process-local LD_LIBRARY_PATH set to
`/opt/rocm/core-10.0/lib:/opt/rocm/core-10.0/lib/rocprofiler-sdk`,
`rocprofv3 --pmc SQ_INSTS_MFMA SQ_INSTS_VALU SQ_INSTS_SALU SQ_INSTS_LDS
--kernel-include-regex 'composite_trim_kernel|_qvq_p32_composite_recovery.*'
--output-format csv`, same benchmark with M4096, warmup1, iterations2.

- Eight-wave counters: /tmp/qvq-trim-profile/raw/ubuntu2404-mi350x/744910_counter_collection.csv
- Four-wave counters: /tmp/qvq-trim-profile-w4/raw/ubuntu2404-mi350x/747507_counter_collection.csv
- Four-wave TTIR/TTGIR/LLVM/AMDGCN: /tmp/qvq-trim-profile-w4-cache/SDCMW46WPXNNQEAUP3IDBJUXDPQAOZOHPZV7CLYNAKXO57H46MUQ/
- Initial eight-wave timing: /tmp/qvq-trim-composite/report.json
- Post-profile eight-wave timing: /tmp/qvq-trim-composite-post/report.json
- Initial four-wave timing: /tmp/qvq-trim-composite-w4/report.json
- Post-profile four-wave timing: /tmp/qvq-trim-composite-w4-post/report.json

Next: test explicit layouts that avoid padded/replicated wave work and
register transfers, then measure the complete sweep before promotion.
