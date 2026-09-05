# Private FP32 correction output reuse

Retained implementation: `a2599ed9`, preceding commit `36a21393`.
Overall target baseline remains `c89459e3`, across all 364 requested MKNR cases.
No packed weights, precision, correction term, output dtype, or dispatch eligibility changed.

## Result

All 28 changed attention-output cases (K6144, N5120, M64 through4096, four
rates) improve versus 36a21393. Per-case speedups range 1.02389–1.09291x;
geometric mean 1.05207x. Every output is exactly equal to the preceding
operator on identical inputs. All 28 stream, graph, fresh-input accuracy,
and previous-output ownership checks pass. Canonical maximum absolute error
is 0.0019459724, below the unchanged 0.002 threshold.

```text
M      speedup geomean over rates 2/2.5/3/3.5
64     1.04945x
128    1.04496x
256    1.02657x
512    1.03933x
1024   1.05193x
2048   1.08231x
4096   1.07093x
```

The post-profile full sweep is valid and complete: 352 canonical FP32 checks
pass (maximum absolute error 0.0019612312), plus 12 unchanged large gate/up
fallback cases exactly match c89459e3. Those fallback cases are not claimed
to pass the canonical threshold: their recorded maximum canonical error is
0.0035161972. This distinction is explicit in each report row.

Current production reaches >=1.5x in 36/364 cases, geometric mean 1.16688x in
this paired run. The full goal remains unmet. Do not substitute the earlier
AITER prototype's 57/364 result for this production measurement or combine
geometric means from separate runs.

```text
shape         cases  passed contract  >=1.5x  geomean vs c89459e3
full_q_gate      52         52            0     1.0043x
full_kv          52         52            0     1.0018x
attn_out         52         52           28     2.6572x
linear_qkv       52         52            0     1.0024x
linear_z         52         52            0     1.0014x
mlp_gate_up      52         52            0     1.0061x
mlp_down         52         52            8     1.0909x
```

## Algebra / ownership audit

Unchanged evaluation is `primary = mm(X, high, FP32)` followed by
`addmm(primary, X, residual, FP32)` and the existing output cast. Previously,
addmm allocated a second FP32 output and copied primary into it before GEMM.
The new call uses `out=primary`. That buffer is freshly allocated and private
to this forward, never a cache, input, or previously returned output. Both
GEMM reduction orders and FP32 epilogues remain unchanged. This is reuse of
already-rounded values, not algebraic reassociation or reduced precision.

When gradients are enabled and any operand requires gradients, the original
out-of-place addmm remains in use. The new tests exercise this branch for each
operand independently, both output dtypes, unchanged inputs, and independent
outputs from repeated calls. They verify forward/fallback selection, not new
backward support beyond the existing Torch operator contract.

An initial test incorrectly assumed bitwise odd symmetry under `X -> -X`.
That assertion failed even though same-input outputs matched. It now compares
the changed input against an independent out-of-place evaluation on that same
input; the numerical tolerance was not changed.

## Post-commit mapping and issued instructions

Matched Torch mapping traces at M64,1024,4096 and all four rates show four
kernel launches becoming three: primary GEMM, correction GEMM, output cast.
The only removed launch is `__amd_rocclr_copyBuffer.kd`. Copy payloads removed
are respectively 1.25,20,80 MiB, equivalent to one FP32 M-by-N tensor. This
also removes one simultaneously live M-by-N FP32 temporary. These are byte
counts from the mapping traces and the tensor ownership proof, not an allocator
peak-memory measurement. Trace duration is not used for the speedup claim.

rocprofv3 counters at M1024 identify 16 baseline primary/copy/correction pairs
and 28 candidate primary/correction pairs. Every pair in its group has identical
counts. The different sample counts come from candidate-only stream/graph and
fresh-input validation; they are not summed into a misleading total speedup.

```text
stage                  VALU      SALU       LDS       MFMA
primary (both)       4595712   1385472   2666496    3932160
correction (both)    4678656   1425408   2676736    3932160
copy (baseline only)  120832     67584         0          0
copy (candidate)          0         0         0          0
```

Both GEMMs resolve to the same exact hipBLASLt MT160x128x128 gfx950 function
in the same code object, SHA256
`424e296a25535a3257f966cd4892015140a101691ae3b74117b15a4573d6e973`.
The complete function range has 37211 static instructions and 260 static
`v_mfma_f32_16x16x32_f16` instructions, including alternate paths. These are
not dynamic counts. The disassembly retains beta epilogue loads/FMA/stores;
no GEMM math, mask, conversion, address, or permutation instruction stream
was rewritten. Buffer aliasing removes the preceding whole copy stream.

Reported GEMM resources: 256 threads, 79872-byte LDS, no scratch, 112 VGPR,
208 accumulation VGPR, 96 SGPR. No before/after resource change is claimed.
Occupancy, bank conflicts, scheduler eligibility, stall reasons, and achieved
bandwidth were not measured. The runtime copy's static code object was not
resolved by llvm-objdump offloading extraction; its removed issued counters
and matched launch are measured, but no static copy-opcode count is claimed.
The ISA analyzer explicitly marks six unrelated/copy/cast symbols absent from
the supplied GEMM code object rather than pretending to disassemble them.

## Validation and reproduction

350 focused tests passed, 14 warnings, 11.28 seconds, after the profile.
Ruff and `git diff --check` pass. GPU kernel tests executed on MI355X VF gfx950.
Both post-profile benchmarks passed three-sample idle preflight and pre-timing
process recheck. Physical GPU0: BDF0000:83:00.0, unique0x333ef6e01ec019b3,
256 CUs. Full software/config/source fingerprints are embedded in reports;
the harness now fingerprints its own source as well as kernel/production files.

Full sweep:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID HIP_VISIBLE_DEVICES=0 python scripts/benchmark_qvq_p32_amd_butterfly.py \
  --butterfly none --baseline-amd-commit c89459e3 --full-sweep \
  --warmup 10 --iterations 30 --output /tmp/qvq-inplace-a2599ed9-full/report.json
```

Direct comparison uses baseline36a21393, `--shapes attn_out`, M64 through4096,
warmup20/iterations50. Mapping uses `--torch-profile-dir`, M64/1024/4096,
warmup2/iterations2; its perturbed timings are not final performance evidence.

Instruction capture uses sudo rocprofv3 with process-local
`LD_LIBRARY_PATH=/opt/rocm/core-10.0/lib:/opt/rocm/core-10.0/lib/rocprofiler-sdk`,
`--mangled-kernels --output-format csv`, counters SQ_INSTS_VALU/SALU/LDS/MFMA,
regex `Cijk.*|.*copyBuffer.*|.*float16_copy_kernel.*`, M1024,
warmup1/iterations2, baseline36a21393. Raw CSV:
`/tmp/qvq-inplace-committed-profile/raw/ubuntu2404-mi350x/798288_counter_collection.csv`.
The profile JSON preserves counter pairing and complete mapping kernel names.

The GEMM bundle is installed under the ROCm SDK libraries' hipblaslt/library/gfx950:
`TensileLibrary_HH_SH_HA_Bias_SAV_UA_Type_HS_HPA_Contraction_l_Alik_Bljk_Cijk_Dijk_ID75a0_gfx950.co`.
Unbundle with clang-offload-bundler, target `hipv4-amdgcn-amd-amdhsa--gfx950`,
then run `scripts/analyze_qvq_amd_library_isa.py` against that CSV. Extracted
binary and whole-function disassembly are in `/tmp/qvq-inplace-a2599ed9-isa`.

## Next decision

Retain this exact-value reuse, but do not call it the requested 1.5x result.
Next pursue a fused primary/residual GEMM that shares activation loads and
avoids intermediate traffic while retaining FP32 accumulation and correction.
It must pass independent canonical checks because fusion can change rounding.
Large gate/up remains a major gap; do not hide its unchanged fallback cases.
