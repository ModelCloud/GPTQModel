# QVQ end-to-end nsys re-profile — post round-2 optimizations (PR #8 + PR #11)

Date: 2026-08-24. Branch `perf/qvq-nsys-reprofile`, base `origin/main @ fcd679e8` (includes the PR #7 NVTX
profiler harness, the PR #8 fused `qvq_fused_w2_family_grid_kernel`, the PR #11 round-2 squared-difference /
weight-fold optimizations, and the PR #12 test fixes). Baseline for all comparisons:
`docs/qvq_nsys_profile_llama32_1b.md` (PR #7, pre-optimization main).

Status: capture in progress — this revision commits the environment proof; measurement sections follow.

## Environment proof (phase 1)

`scripts/setup_qvq_profile_env.sh` run unchanged in this worktree on 2026-08-24:

* venv `.venv-qvq-profile`: Python 3.12, `torch 2.13.0+cu132` (cu130 index), CUDA 13.3 toolkit at
  `/usr/local/cuda`, cusparse/cublas/... header shim at `.venv-qvq-profile/cuda-shim-include`.
* QVQ CUDA JIT extension built against **this checkout** (`fcd679e8`) in 155 s
  (`~/.cache/gptqmodel/torch_extensions/qvq_cuda/2770b928e051f758`), `prewarm_qvq_cuda() → True`.
* Device: NVIDIA PG506-230 (98 GB), nsys 2026.4.1.
* Harness-hash JIT variant warmed with an unprofiled `--layers 1` run at `GPTQMODEL_QVQ_NVCC_THREADS=8`
  (same value used for the captures) so no nvcc runs under nsys.
