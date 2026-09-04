# AMD GPU Kernel Log

This log contains only AMD GPU kernel work. It records successful and failed
experiments for the QVQ V2B2-P32 inference kernel targeting AMD Instinct
MI355X (`gfx950`). Performance from a contaminated GPU is retained for
engineering comparison but is never presented as an official result.

## Target and correctness contract

- GPU: AMD Instinct MI355X VF, `gfx950:sramecc+:xnack-`, 256 compute units
- PCI bus: `0000:83:00.0`
- GPU unique ID: `0x333ef6e01ec019b3`
- Driver: `7.1.3.31500000`
- Software: Python 3.14.7, PyTorch 2.13.0+rocm10.0.0, HIP 7.15.26333,
  Triton 3.8.0
- Formats: canonical QVQ V2B2-P32 W2, W2.5, W3, and W3.5
- Requested rows: M=1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048,
  and 4096
- Accuracy gate: maximum absolute error no greater than `2e-3` against an
  FP32 dense oracle reconstructed from the same packed P32 payload

## 2026-09-04 experiments

| Result | Experiment | Accuracy | Exploratory performance and decision |
| --- | --- | --- | --- |
| FAIL (environment) | Strict three-sample idle preflight | Not run | GPU utilization was 0%, but PIDs 1463574, 1463764, and 1463785 retained 254936.4 MiB (86%) VRAM. The strict benchmark correctly failed closed. The processes are outside this namespace and were not killed. |
| PASS, REJECTED | Direct planar recurrence decode, 64-column output tile | Passed focused P32 checks | W3, K=N=4096: about 0.213 ms at M1 and 0.973 ms at M4096. Rejected because continuous-window decoding reduced decode overhead. Results were exploratory because the idle gate was blocked. |
| PASS, REJECTED | Direct planar recurrence decode, 128-column output tile | Passed focused P32 checks | W3, K=N=4096: about 0.384 ms at M1, 0.284 ms at M32, and 0.994 ms at M4096. Rejected as slower than the 64-column tile. Results were exploratory. |
| FAIL, FIXED | First continuous-window decoder used signed words during funnel shifts | Maximum absolute error observed up to about 2.87, above `2e-3` | Root cause was arithmetic right shift of signed `int32` packed words. Casting both loaded words to Triton `uint32` before shifts fixed the decode. No performance result from the incorrect kernel was accepted. |
| PASS | Continuous-window decode, 64-column output tile, four M regimes | 62/62 focused tests passed; 10 bitwise-identical repeats per case; small K=N=256 sweep worst maximum absolute error about `7.15e-6` | Selected as the initial kernel. Tile regimes are 16x64 for M<=16, 32x64 for M<=64, 64x64 for M<=256, and 128x64 for larger M. |
| PASS (exploratory) | Full requested M/rate sweep, K=N=4096, 5 warmups and 20 timed iterations | All 52 cases passed; worst maximum absolute error `1.2397766e-5` | The run is explicitly invalidated by foreign VRAM residency. P32 median ranges were W2 0.142-0.786 ms, W2.5 0.160-0.850 ms, W3 0.167-0.954 ms, and W3.5 0.167-0.950 ms. Cached dense FP16 GEMM was 0.013-0.107 ms and is recorded as an ideal uncompressed ceiling, not the production fallback. |
| PASS (exploratory) | Expanded K=N=4096 sweep with production fallback, 5/20 kernel and 1/5 fallback warmup/iterations | All 52 cases passed; worst maximum absolute error `1.2397766e-5` | P32 was 3.66x-24.11x faster than reconstruct-plus-GEMM and peaked at 165.2 effective TFLOP/s. Packed storage was 7.88x, 6.32x, 5.28x, and 4.53x smaller than dense FP16 for W2 through W3.5. This run remains explicitly invalidated by the same foreign residency. |
| FAIL (test harness), FIXED | First branch-coverage invocation used a dotted `--source` module | Tests did not start | Coverage imported the side-effectful package while resolving the source and Python 3.14 then rejected a second NumPy extension import before segfaulting. Using an absolute `--include` file filter avoided the duplicate import. |
| FAIL (test harness), FIXED | Public `QVQLinear.forward()` integration assertion compared FP16 output metadata to an FP32 oracle | Numeric values reached the assertion, but 3 integration cases failed on dtype equality | Public forward intentionally restores the activation dtype after scaling. The assertion now promotes the output to FP32 before applying the `2e-3` oracle gate. This was not a kernel numeric failure. |
| PASS | Expanded correctness and contract suite | 183/183 tests passed | Covers three independent seeds for every requested M/rate pair, 10 repeat calls per pair, adversarial signs and magnitudes, all alternate bank IDs, FP16/FP32 output, public module dispatch, a non-default stream, and invalid contracts. Python branch coverage is 100% after excluding the Triton JIT body that is compiled and validated by the GPU oracle cases. |
| FAIL (pre-existing test portability) | Existing `test_qvq_v2b2_p32.py`, `test_qvq_v2b2_p32_window.py`, and `test_qvq.py` regression selection on ROCm | 853 passed, 139 skipped, 125 failed | The CUDA-marked failures are not in the new inference path. They gate only on `torch.cuda.is_available()`, which is true on ROCm, then require NVIDIA-only QVQ/diagnostic CUDA extensions or NVIDIA telemetry. The extension reports `QVQ CUDA requires NVIDIA CUDA; ROCm is not supported`. Five representative non-matrix failures were rerun separately and confirmed the same ROCm/NVIDIA capability mismatch. |

The initial exploratory sweep is stored in
`artifacts/mi355x_p32/initial_gfx950.json`. The expanded sweep is stored in
`artifacts/mi355x_p32/fallback_gfx950.json`; it also records effective
throughput, packed/dense storage, and enrolls the benchmark's host-visible ROCm
context before rejecting newly arriving PIDs ahead of every timed case.

## Implementation notes

The selected Triton kernel decodes the storage-neutral continuous-window P32
layout in registers, applies packed binary bank selection and the exact PGC16
mix, loads canonical FP16 levels, and accumulates with `tl.dot` into FP32. The
integration dispatch is limited to ROCm `gfx950`, inference mode, FP16 input,
V2B2-P32 vector size 2, and transition widths 4 through 7. Unsupported devices
and formats retain the existing reference or CUDA paths.
