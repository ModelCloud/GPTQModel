# Full-K GEMV and dispatch ceiling experiment

Experimental commit cdaab451; production unchanged at bd09d89b. Overall
performance target stays 1.5x in every requested case versus c89459e3.

The raw dispatch ceiling bypasses module guards but calls the unchanged
cached production operator (no bias in these fixtures). It is not a safe
production interface: mutation, eligibility, and module semantics still
require guards. Across four shapes, M1/4/16/32 and four rates, all 64
cases passed the existing canonical gate. Sample W3 full-KV speedups
versus c89459e3 were 1.375/1.428/1.370/1.448x, still short of 1.5x.
Gate/up and down cases had substantially less improvement. Removing
Python guards alone cannot justify a full-target completion claim.

## Math change

Retained M1 GEMV sums FP32 products within each K tile and adds the
partial sums serially. The experiment loads the full K extent, masks
padding to the next power of two, and performs one FP32 reduction.
FP16 weight/input storage, optional residual addition in FP32, and final
FP16 output remain unchanged. This changes floating-point reduction
order; it is real-arithmetic equivalent, not bitwise equivalent.
No quantization format, scale, rate, packed data, or precision gate changed.

K=5120/6144, all seven Qwen shape groups at M1, four rates. The down
path does not call this GEMV and remains unchanged. Both block-N=2 and
block-N=4 pass all 28 cases. Initial gate/up improvement was about
16-18%; several other projections regressed. Neither variant is enabled
in production. Synthetic tests are kernel evidence, not model-quality
certification. Tests cover both K values, N tiles 2/4/8, residual on/off,
padding and canary output rows: total focused suite 286 passed,
26 warnings in 12.83s after profiling. Ruff and whitespace checks passed.
Post-profile paired repeats pass 28/28 for both N2 and N4, with strict idle
and pre-timing gates. N2 gate/up speedups remain 1.163-1.179x against
bd09d89b. Compact results: `full_k_gemv_n2_cdaab451_post.json`,
`full_k_gemv_n4_cdaab451_post.json`, and
`direct_dispatch_ceiling_cdaab451.json`. Production's last full 364-case
certification remains unchanged; this experiment does not meet or replace
the all-case 1.5x target.

## Post-commit ISA and executed counters

Physical GPU0: MI355X gfx950, BDF 0000:83:00.0, unique ID
0x333ef6e01ec019b3, 256 CUs. Strict three-sample idle gates passed.
Torch 2.13.0+rocm10.0.0; HIP 7.15.26333;
Triton 3.8.0+git4cff872c.rocm10.0.0.

Gate/up M1,K5120,N17408; four waves per CTA:

| Metric | Retained | Full K, N2 | Full K, N4 |
| --- | ---: | ---: | ---: |
| CTAs | 4352 | 8704 | 4352 |
| Static instructions | 163 | 256 | 397 |
| Issued LDS | 905216 | 69632 | 69632 |
| Issued SALU | 487424 | 557056 | 261120 |
| Issued VALU | 5344256 | 6963200 | 5587968 |
| VGPR descriptor | 31 | 76 | 138 |
| Dynamic LDS bytes | 2048 | 32 | 64 |
| Scratch bytes | 0 | 0 | 0 |

The conditional K-loop branch disappears; an unconditional entry branch
remains. Most repeated LDS reductions disappear (92.3% fewer issued LDS),
but extra CTAs and full-K per-thread work increase VALU. Static counts
must not be substituted for executed counts: baseline loops execute
multiple times. Full-K kernel has no residual load/add instructions when
the constexpr residual flag is false. No spill storage was generated.

For full-KV N1024, the retained kernel uses N16 tiles and is already
efficient. Its issued LDS/SALU/VALU counts are 26368/18944/353280,
versus 4096/32768/409600 for full-K N2 and 4096/15360/328704 for N4.
Thus lower LDS counts alone do not establish lower end-to-end latency.
Do not infer occupancy, scheduler stalls, bandwidth, or bank conflicts:
those counters were not collected. The rocprof N4 VGPR field reports 12,
in disagreement with the compiler descriptor's 138; retain the discrepancy
rather than claiming twelve-register occupancy. Its cause is not established.

JIT hashes:

- Retained gate/up: 3944ec954205c55385aabe510d5936e5d723c0135fb1ae84bef5b7098c2513eb
- Retained full-KV: 6e50e9dcdce372f88d15761a89fee744d5e7e848a02aef39e4b01c39aea42ea2
- Full K N2: cac086e2d6b81729f92abede0faf595ff3f00c49f6477ffef295652b6156e4a5
- Full K N4: 4977788c72e9d392c9cab27a9e208be4d6c1f67ef047ee8d677c79f532b7a517

## Reproduction

Benchmark: `python scripts/benchmark_qvq_p32_amd_butterfly.py --butterfly none
--gemv-full-k --gemv-block-n 2 --baseline-amd-commit bd09d89b --full-sweep
--m-values 1 --warmup 10 --iterations 30 --output /tmp/qvq-fullk-bn2-post/report.json`.
Repeat with block-N=4. Direct-ceiling command uses `--folded-direct-ceiling`
instead of `--gemv-full-k`, both baseline commit flags c89459e3, shapes
full_kv/full_q_gate/mlp_gate_up/mlp_down and M1/4/16/32.

Profiler: sudo rocprofv3, process-local LD_LIBRARY_PATH
`/opt/rocm/core-10.0/lib:/opt/rocm/core-10.0/lib/rocprofiler-sdk`,
`--pmc SQ_INSTS_VALU SQ_INSTS_SALU SQ_INSTS_LDS --kernel-include-regex
'folded_gemv_full_k_kernel|_qvq_p32_folded_gemv.*' --output-format csv`.
Benchmark arguments: shapes mlp_gate_up/full_kv, M1, warmup1, iterations2.
Both N2 and N4 counters were captured after cdaab451.

- N2 counters: /tmp/qvq-fullk-profile/raw/ubuntu2404-mi350x/755424_counter_collection.csv
- N4 counters: /tmp/qvq-fullk-profile4/raw/ubuntu2404-mi350x/756359_counter_collection.csv
- N2 compiler artifacts: /tmp/qvq-fullk-profile-cache/ZLAINYWWXALST6JKX3PA7L2ZL7Z7ADCJ6ZDX77XSSVSSWYKW4SSQ/
- N4 compiler artifacts: /tmp/qvq-fullk-profile4-cache/JF3XRDDS5HJZFSOKWJ5J4IEL4TLMD5T66BD65DLHPR47KMVXUULQ/
- Tests: /tmp/qvq-fullk-tests.log

Next investigate FP16 packed-product instructions with FP32 accumulation
and remaining conversion/reduction work, alongside separately measured
host dispatch savings. Preserve all guards; do not turn the raw ceiling
into an unguarded production path. Any packed-product idea remains an
unverified hypothesis until ISA support and canonical correctness pass.

Source-correlated follow-up: AMDGCN lines mapped to input/weight loads
(source lines 16/17) still issue out-of-bounds buffer loads for padding,
convert their zero results with `v_cvt_f32_f16`, then multiply them using
`v_pk_mul_f32` at source line 20. The compiler did not fully erase padded
arithmetic. A concrete next exact-real-algebra candidate is splitting
K5120 into 4096+1024 (K6144 into 4096+2048), retaining FP32 reductions.
This must be measured because an extra reduction can offset saved work.
