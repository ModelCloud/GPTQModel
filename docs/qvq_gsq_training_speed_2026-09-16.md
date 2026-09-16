# QVQ P32 staged-GSQ training speed, 2026-09-16

## Scope

This experiment accelerates the differentiable RHT reconstruction used while
optimizing legal W3/P32 QVQ candidates. It does not change the candidate bank,
GSQ objective, Gumbel draws, schedule, optimizer, update count, hard export, or
held-out acceptance rule.

The controlled workload is Llama 3.2 1B layer 0, all seven projections in
paper stage order, on one NVIDIA H100:

- Q and K: 2,000 updates each.
- Joint V/O attention: 20 epochs over four logical batches (80 updates).
- Joint gate/up/down MLP: 20 epochs over four logical batches (80 updates).
- 16,384 FineWeb-Edu training tokens, 4,096 disjoint validation tokens, and
  16,384 separately offset Q/K metric tokens.
- 33 legal P32 candidates, batch size 8, microbatch size 4, deterministic
  algorithms, and `CUBLAS_WORKSPACE_CONFIG=:4096:8`.

The measured `fit_seconds` is therefore per complete decoder layer, not per
projection and not per model.

## Bottleneck and implementation

Nsight Systems showed that eager FP32 Hadamard butterfly autograd dominated
the original path with small copies, fills, additions, launches, and host
synchronizations. The optimized forward uses the vendored fused Hadamard
kernel. A custom descending-stage CUDA kernel implements backward because the
eager reference's autograd traverses butterfly stages in reverse order; using
the ordinary self-adjoint transform changes FP32 rounding.

The fused path is enabled only for CUDA FP32 power-of-two widths supported by
the native extension. Unsupported configurations retain the eager oracle. The
validation CLIs expose `--disable-fast-training-hadamard` for matched A/B runs
and report the selected backend.

## Result

| Path | Fit time, run 1 | Fit time, run 2 | Speedup versus baseline |
| --- | ---: | ---: | ---: |
| Eager FP32 oracle | 80.053 s | — | 1.000x |
| Fused exact forward/reverse backward | 38.071 s | 38.031 s | 2.103x–2.105x |

Both optimized runs produced all 14 exported state tensors bitwise equal to
the eager baseline. They also exactly matched every recorded stage loss, best
epoch, changed-tile count, scale delta, and final held-out block result:

- Validation loss: `3.337860107421875e-05` to
  `2.8312206268310547e-05`.
- Held-out improvement: `15.178571428571429%`.
- Strict quantization/evaluation split disjointness: true.

At 16 decoder layers, serial GSQ fitting extrapolates to about 10.2 minutes;
this excludes one-time calibration/capture and candidate-construction costs.

## Second optimization phase

The next profile showed that repeated held-out Q/K checkpoints were dominated
by reconstructing already-validated P32 candidates and by serialized host
dispatch. The second phase therefore:

- keeps strict P32 pack/unpack validation on public export while using exact
  sparse candidate values for intermediate held-out materialization;
- stores those sparse values directly, rather than reconstructing them as
  `baseline + delta`, which could introduce a new FP32 rounding boundary;
- maps sparse tile coordinates into the Hadamard matrix layout once at module
  construction, eliminating a per-update tile permutation and its adjoint;
- fits independent Q and K projections concurrently on separate CUDA streams,
  with separate identically seeded generators and unchanged per-projection
  update/validation order; and
- uses Python 3.14 free-threaded mode (`PYTHON_GIL=0`) so the two host dispatch
  threads are not serialized by the GIL.

The same complete seven-projection layer workload measured 18.031 and 17.951
seconds in two idle-gated runs. This is 2.109x–2.119x faster than the prior
38.031-second path and 4.440x–4.460x faster than the 80.053-second eager
baseline. A 16-layer serial extrapolation is about 4.8 minutes.

Every one of the 14 exported state tensors is bitwise equal to the accepted
38.031-second oracle. Stage losses, validation losses, best epochs,
changed-tile counts, learned scales, and the 15.178571428571429% held-out
block improvement also match exactly. The formal validator now records the
physical-GPU idle samples, pre-timing ownership recheck, and GIL state in its
JSON output.

Artifacts for this phase are under
`/root/qvq-results/gsq-next-2x-20260916/`; `nogil-r1` and `nogil-r2` are the
two formal full-layer repetitions.

## SM90 audit

Nsight Compute on a `2048 x 2048` FP32 reverse transform reports:

- 26.91 microseconds kernel duration.
- 57.99% SM throughput and 67.83% memory throughput.
- 18 registers per thread, zero local memory, and no divergent branches.
- 628.47 GB/s DRAM bandwidth.

The SM90 SASS contains one FP32 multiply before shared-memory materialization,
then FP32 add/subtract butterflies separated by barriers. It contains no FP64
instructions and no fused multiply-add across the required normalization
rounding boundary.

Artifacts are under `/root/qvq-results/gsq-2x-20260916/`, including the eager
baseline, both optimized layer runs, Nsight Systems reports, and
`ncu-reverse-hadamard/profile.ncu-rep`.
