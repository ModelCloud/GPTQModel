# Additional 1.5x investigation: not achieved

Production kernel baseline remains `c89459e3` (documentation HEAD `b6549447`).
Commit `04d3586b` adds isolated experimental kernels, a paired benchmark, and tests;
it changes no production dispatch. Measurements use physical GPU 0, MI355X VF,
gfx950, BDF `0000:83:00.0`, ID `0x333ef6e01ec019b3`, **256 runtime-reported CUs**.

## Measured outcome

Final unprofiled, post-commit/post-profile public-forward comparison, M1024,
K17408, N5120, FP16 input/output, FP32 intermediate, 20 warmups and 100 samples
per path, alternating AB/BA order:

| Rate | Retained baseline ms | Split butterfly ms | Speedup |
|---|---:|---:|---:|
| W2 | 0.232724 | 0.222424 | 1.046x |
| W2.5 | 0.231904 | 0.221884 | 1.045x |
| W3 | 0.234264 | 0.223664 | 1.047x |
| W3.5 | 0.233885 | 0.223384 | 1.047x |

All four canonical error gates and same-input non-default-stream/graph replay
checks passed. These are synthetic algebra fixtures, not real-model quality tests.
The benchmark's scoped `torch.mm` substitution adds Python overhead; these are
experimental public-forward results, not a certified production integration win.

The complete 364-case sweep passed: all seven Qwen3.8-27B shape groups, four rates,
and M=1,2,4,8,16,32,64,128,256,512,1024,2048,4096. Its 316 folded-dispatch cases
passed canonical FP32 max-absolute error <=0.002 (worst 0.001958371); the other 48
cases matched the unchanged baseline exactly. All-case geometric mean is only
1.00136x because this experiment changes four cases. A prior run was interrupted
after 134 cases by the strict new-PID gate; it is not counted as a completed sweep.

## Profile and SSA/algebra review

Post-commit rocprofv3 trace: dense power-Hadamard p50 22.961 us, butterfly p50
9.761 us, unchanged base recovery p50 18.120 us. The unchanged main GEMM averages
221.756 us. Trace summaries include warmup and stream/graph checks; use the
unprofiled event results above for speed claims.

The Walsh matrix admits seven sum/difference stages. A four-row tile avoids the
LDS traffic and three barriers found in the eight-row gather tile. Split/join
ownership changes the gather prototype's generated code as follows:

| Static metric | Gather, four rows | Split/join, four rows |
|---|---:|---:|
| Instructions | 93 | 92 |
| `ds_bpermute` | 14 | 2 |
| `s_waitcnt` | 9 | 3 |
| VGPR descriptor count | 16 | 11 |
| SGPR descriptor count | 17 | 17 |
| LDS / scratch bytes | 0 / 0 | 0 / 0 |

This is a routing/dependency reduction, not a large total-instruction reduction.
Conditional masks increased from 13 to 17 and ten scalar nops remain. DPP and
permlane operations replace most gathers. The post-commit JIT preserves these
counts; raw TTIR/TTGIR/LLVM/AMDGCN are in the recorded cache directory.

Executed instruction counts, occupancy, conflicts and stalls are **unavailable**:
the bounded SQ_WAVES attempt failed with
`aqlprofile API table load failed: HSA_STATUS_ERROR`. Only task-owned stuck
profiler processes were terminated. Kernel traces succeeded under sudo; the
earlier unprivileged trace hit a ring-buffer mmap shutdown failure. No counter
values were inferred from static ISA. This profiling limitation prevents promotion
under the repository workflow; runtime timing/correctness evidence is nevertheless
available and is not merely compilation evidence.

## Other candidates and decision

- AITER hipBLASLt FP32 solution 65827: paired 1.034–1.047x on the four targeted
  cases, stream/graph checks passed. Build-specific index and private wrapper are
  unsuitable as unconditional production defaults; provisional production edits
  were removed. It remains an explicit benchmark knob.
- AITER plus the first eight-row gather butterfly: paired 1.063–1.079x; exploratory,
  not a claim for the final split/join combination (not yet measured together).
- FP16 main output: rejected by the public canonical gate. An earlier explanation
  blaming SU/SV was wrong: those fixtures used unit scales, and the public layer
  always casts its result back to input dtype. Raw FP32 helper results do not waive
  that FP16 public-boundary failure.
- Earlier FlyDSL/Gluon, sharded GEMM and split-partial probes did not establish a
  retained win. The installed FlyDSL path required same-type output; do not assume
  arbitrary FP32 output support from the backend lookup alone.

Next pursue the **main FP32-output GEMM**, especially tile/layout coverage. The
traced 256x256 tile gives 80 CTAs for this matrix against 256 CUs. This motivates
smaller-tile/pipeline experiments but does not prove low achieved occupancy.
Recovery-only changes cannot deliver 1.5x: even eliminating its approximately
41 us entirely would leave the approximately 222 us main GEMM. Do not relax the
public FP16 output or canonical tolerance to obtain the target.

## Reproduction and evidence

```bash
python scripts/benchmark_qvq_p32_amd_butterfly.py --full-sweep \
  --warmup 10 --iterations 30 --output /tmp/butterfly-sweep/report.json
python scripts/benchmark_qvq_p32_amd_butterfly.py --warmup 20 --iterations 100 \
  --output /tmp/butterfly-target/report.json
# Optional, only with the recorded matching AITER/hipBLASLt installation:
python scripts/benchmark_qvq_p32_amd_butterfly.py --butterfly none \
  --hipblaslt-solution 65827 --output /tmp/aiter-target/report.json
```

Artifacts alongside this note:

- `qwen38_27b_butterfly_full_exploratory.json`: complete paired matrix.
- `qwen38_27b_butterfly_post04_timing.json`: final unprofiled target run.
- `qwen38_27b_butterfly_post04_profile.json`: exact commands, paths, JIT hash,
  resources, static opcode counts, trace distributions, counter failure.
- `qwen38_27b_m1024_aiter_paired_exploratory.json` and
  `qwen38_27b_m1024_aiter_gather_paired_exploratory.json`: provisional library probes.

Production dispatch is intentionally unchanged; the additional 1.5x objective
remains open. Post-profile validation: 247 GPU tests passed (14 warnings), Ruff
passed, and the diff whitespace check passed.
