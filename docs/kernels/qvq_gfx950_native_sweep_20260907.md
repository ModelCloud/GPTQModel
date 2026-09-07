# gfx950 native P32 sweep (2026-09-07)

This is a guarded MI355X measurement of the existing combined HIP window decode
plus rocBLAS path against the pinned QVQ AOT baseline. It is synthetic data
using the canonical P32 codebook and an independent FP64 reference; it is not
a model-quality claim. Each run used three idle samples before touching the
GPU, ten warmups and ten synchronized event samples. All reported rows passed
finite output, MAE <= 0.003 and max <= 0.006 checks, including graph replay.

| M | K | N | transition bits | native decode+GEMM (us) | baseline (us) | speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5120 | 1024 | 5 | 28.660 | 89.960 | 3.14x |
| 2 | 5120 | 1024 | 5 | 28.320 | 90.300 | 3.19x |
| 4 | 5120 | 1024 | 5 | 27.480 | 91.720 | 3.34x |
| 8 | 5120 | 1024 | 5 | 25.520 | 132.960 | 5.21x |
| 2048 | 5120 | 1024 | 5 | 56.900 | 175.160 | 3.08x |
| 4096 | 5120 | 1024 | 5 | 71.480 | 241.841 | 3.38x |

The benchmark now supports `--autotune`, bounded warmups and bounded timing
iterations. A one-sample sweep selected solution `66914` but produced a slower
end-to-end result (30.08 us), demonstrating that a low-sample winner must not
be promoted blindly. With three samples, the installed rocBLAS list exposed a
negative internal entry (`-9`) that cannot be replayed through the explicit
configuration ABI. The candidate enumeration path now filters negative IDs;
the rebuilt filtered DSO exposed 681 replayable candidates and selected
solution `67155` at 16.0 us median GEMM, with 27.18 us complete latency.

Solution zero remains a valid standard-algorithm candidate when it wins a
measurement; it is not an autotune failure sentinel. The benchmark and native
plan test now handle that case explicitly.

The raw JSON measurements are retained under
`/home/ubuntu/qvq-gfx950-runtime/next-*.json`. The filtered host DSO was built
from this commit with `clang++ -std=c++20 -O3 -fPIC` and the ROCm 10.0 headers;
no device decoder source or generated ISA changed in this pass.

## rocprofv3 / ISA pass

`rocprofv3 --kernel-trace --stats` on the M=1 row recorded 64 native decoder
dispatches (908.604 us aggregate), seven rocBLAS GEMM dispatches (1,116.603 us)
and 44 baseline AOT dispatches (3,919.214 us). Profiling perturbs latency, so
those aggregates are attribution data rather than the timing table above. The
same profile showed the native decoder's 256-thread launch and 20 VGPR / 32
SGPR runtime allocation; the baseline's largest dispatch used 512 threads.

Static `llvm-objdump -d` on the unchanged gfx950 decoder object reports 109
instructions for the W4 specialization and 114 each for W5/W6/W7. This change
only filters host-side rocBLAS candidate IDs; it generates no new device ISA,
so there is no SASS/SSA arithmetic delta to promote in this pass.
