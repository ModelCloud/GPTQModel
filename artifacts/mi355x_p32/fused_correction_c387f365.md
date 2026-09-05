# Shared-activation correction GEMM: rejected configurations

Experimental device revision `c387f365`; paired retained baseline `62db1106`.
Production dispatch is unchanged. The full objective remains 1.5x across all
364 cases versus `c89459e3`; the last production sweep still has only 36/364
meeting that target. These targeted experiments are not a replacement full sweep.

## Post-profile results

All configurations use Gluon on gfx950, four waves, MFMA16x16x32 FP16 inputs
with FP32 accumulators, and `num_stages=2`. The latter does not imply that
Gluon automatically software-pipelines the load/MFMA loop.

```text
shape/config                  M       speedup (geomean over four rates)
attention, dual, 64x64x64      64      0.40178x
attention, dual, 64x64x64    1024      0.53782x
attention, dual, 64x64x64    4096      0.52103x
gate/up, dual, 128x128x128   1024      0.63022x
gate/up, dual, 128x128x128   2048      0.53061x
gate/up, dual, 128x128x128   4096      0.48984x
gate/up, single, 64x128x64   1024      0.94146x
gate/up, single, 64x128x64   2048      0.81107x
gate/up, single, 64x128x64   4096      0.70315x
```

Tile order is BM,BN,BK. Attention K6144,N5120 uses the real folded-forward
wrapper with only its execute function replaced. Gate/up K5120,N17408 uses
the explicitly labeled raw folded-residual ceiling: it precomputes high/low
folded weights and bypasses production guards and preprocessing. It is not a
deployable dispatch path, and even that optimistic raw comparison loses.
The changed tile dimensions confound an isolated single-versus-dual accumulator
claim: this table compares complete configurations, not one variable alone.

All 36 measured cases pass canonical FP32 max-absolute <=0.002. Maximum errors
are 0.0019392967 for attention and 0.0018806458 for both gate/up configurations.
All 36 stream and graph checks pass. Gate/up additionally passes fresh-input
and previous-output ownership checks. Post-profile focused tests: 374 passed,
90 warnings, 15.99 seconds. New tests cover both accumulator modes, FP16/FP32
output, K256/5120/6144, tiles64/128, M65/N67 tails, and output canaries.
Ruff and `git diff --check` pass. These synthetic checks are not model-quality
evidence; no numerical or performance promotion is made.

## Math / SSA audit

Both paths load the same activation tile once and reuse its converted MFMA
operand for high and low weights. Dual mode computes separate FP32 sums and
adds them in the epilogue. Single mode interleaves high/low MFMA updates into
one FP32 accumulator. Both preserve all real-number terms, but both change
floating-point evaluation relative to two library GEMMs. Neither bitwise
equivalence nor accuracy follows from that identity; independent canonical
checks above are the gate. The output cast remains at the established boundary.

In single mode, the unused correction accumulator and final vector addition
are removed by constexpr specialization. The separate high/low weight loads
share source index algebra, but each still requires layout conversion into
MFMA ownership. Generated code contains LDS writes/reads, barriers, and waits
for those conversions inside the K loop. The shared X operand survives source
reuse, but it does not eliminate the two weight streams or double-MFMA work.

The installed compiler source
`/opt/python314/lib/python3.14/site-packages/triton/backends/amd/compiler.py`
distinguishes normal Triton scheduling/pipelining from `gluon_to_ttgir`, which
runs the explicit warp-pipeline pass but not normal automatic loop scheduling.
This kernel has no explicit warp-pipeline or asynchronous-copy stages. A
launch-level `num_stages=2` therefore is not evidence of an overlapped pipeline.
The generated ISA corroborates repeated load/conversion/wait/MFMA regions.

Next algebraic opportunity: remove load/store tail predicates when constexpr
M or N is exactly divisible by its tile, keeping the existing masked tail
specialization. The current code retains those predicates despite exact grid
coverage. Next architectural opportunity: explicit load/MFMA overlap and
cheaper direct-to-LDS operand preparation. Neither is implemented in this commit.

## Executed-instruction audit after commit

Counters are per dispatch, not static instructions or FLOPs. All repeated
candidate dispatches of each specialization have identical counts.

```text
M1024 configuration       VALU        SALU         LDS       MFMA
attention dual64       43335680    12953600     8888320    7864320
gate/up dual128        70467584    17229568    12673024   22282240
gate/up single64x128  106362880    29558784    21028864   22282240
```

The retained gate/up main GEMM issues 11151360 MFMA and 13588000 VALU.
Its Hadamard/preprocessing kernels are outside this filtered main-GEMM counter
comparison; end-to-end benchmark timings include the retained wrapper. The
nearly doubled candidate MFMA count is expected because the folded high/low
strategy has two matrix products. No claim of reduced total arithmetic is made.

```text
configuration        static ISA  VGPR metadata  AGPR metadata  LDS bytes  scratch
attention dual64         441          126             0          16384       0
gate/up dual128         1292          378           122          32768       0
gate/up single64x128     583          164             0          32768       0
```

These are complete function-body static counts, without multiplying loop
iterations. No spills are reported. Preserve ELF/JIT resource metadata rather
than treating rocprof's compact VGPR field as authoritative for large register
counts. Occupancy, bank conflicts, scheduler eligibility, stall reasons, and
achieved bandwidth were not measured. Register pressure and missing overlap
are hypotheses for follow-up, not measured stall attribution.

The retained MT144x256x64 main GEMM resolves to the installed hipBLASLt gfx950
code object (same SHA256 as the preceding correction audit). Its whole-symbol
ISA summary is included separately: 63902 static instructions, including
alternative/tail paths. Five unrelated symbols are explicitly unresolved in
that supplied object. Do not compare static whole-library alternatives with
dynamic loop counts as though they were equivalent units.

## Reproduction and artifacts

Post-profile benchmarks use baseline62db1106, warmup20/iterations50, strict
three-sample idle preflight and pre-timing recheck on physical GPU0, MI355X VF,
BDF0000:83:00.0, unique0x333ef6e01ec019b3, gfx950,256CUs. All three runs are
valid. Full hardware/software/shape/rate/source hashes are embedded in JSON.

Shared flags: `--butterfly none --fused-correction --full-sweep`.
Attention uses `--shapes attn_out --m-values 64 1024 4096` and default tiles.
Gate/up uses `--shapes mlp_gate_up --m-values 1024 2048 4096
--folded-residual-ceiling`, with the tile flags from the table and
`--fused-interleave` for single mode.

Executed profiles use sudo rocprofv3 with process-local ROCm/core and profiler
LD_LIBRARY_PATH, CSV output, mangled names, SQ_INSTS_VALU/SALU/LDS/MFMA,
regex `folded_residual_gemm_gluon_kernel.*|Cijk.*`, M1024, warmup1/iterations2.
Raw CSVs are under `/tmp/qvq-fused-{single,dual,small}-profile/raw`.
Compiler artifacts are in `/tmp/qvq-fused-profile-cache`; exact JIT keys,
binary hashes, static opcodes, resource fields, CSV hashes, and captured
baseline/candidate counters are preserved in the ISA JSON.

Raw post-profile reports are `/tmp/qvq-fused-{single,dual,small}-post/report.json`;
their copies are committed alongside this audit. Production remains at62db1106's
device behavior. Do not select any of these losing candidates in default dispatch.
