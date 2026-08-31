# QVQ exact-P32 Ampere kernel progression

This ledger tracks the exact continuous-window P32 inference kernel for
A100-class `sm_80` GPUs. It records accepted forward progress and failed or
discarded experiments so later tuning does not repeat unsafe variants.

## Contract and target

- Source base: GitHub `main` at `d290c56d1d6fac4e6bbd9a9dc02795bb2e02658f`.
- Device: physical GPU 0, `NVIDIA PG506-230`, UUID
  `GPU-14ab23f1-a785-e9df-bbb5-215547154e3c`, CC 8.0, 124 SMs, 96 GiB.
- Software: PyTorch 2.13.0+cu130; CUDA runtime 13.0; NVCC 13.3.
- Input and levels: FP16. Accumulation and output: FP32.
- Rates: exact standard-P32 W2, W2.5, W3, and W3.5 continuous-window payloads.
- Rows: M1 through M16; tuned measurements cover M1 and M16.
- Accuracy gate: identical packed payload and bank metadata, dense exact-P32
  reference, and maximum absolute inference drift `<= 2e-3`.
- Timing: 10 warmups and 50 per-iteration CUDA-event samples. Every formal run
  passed a three-sample 0% utilization/8 MiB idle gate and rejected foreign
  compute processes again before each timed pair.

The Ampere path explicitly rejects devices other than CC 8.0. It contains no
TMA, WGMMA, thread-block clusters, or distributed shared memory.

Marlin is another important teacher for this path because it is a very fast
Ampere-native weight-only kernel. Its warp partitioning, shared-memory layout,
software pipelining, vectorized movement, occupancy tradeoffs, and shape-aware
dispatch are relevant to future P32 tuning. Its packed-weight decode and
quantization contract are different, however, so those parts cannot be copied
directly into the exact continuous-window P32 representation.

## Accepted design

The kernel carries forward the parts of the Hopper work that do not depend on
Hopper-only hardware:

1. Consume the storage-neutral continuous-window P32 representation directly.
2. Assign four WMMA warps to four adjacent N16 tiles so they share one staged
   M16 x K16 activation tile.
3. Double-buffer activation and window-word staging with Ampere `cp.async`.
4. Decode states 64 pairs apart together because they share one funnel-shift
   amount and a compile-time word distance.
5. Accumulate in FP32 and use bounded split-K to expose enough CTA work.
6. Use an explicit block barrier after MMA before reusing a stage buffer.

Future Ampere experiments should compare the generated instruction schedule,
register pressure, shared-memory bank behavior, and CTA swizzle against Marlin
as well as carrying forward architecture-independent lessons from the Hopper
kernel.

There are eight primary device specializations: four transition widths times
full-M16 and partial-row paths. One runtime split reducer is shared by all
rates. Unknown shapes use a live-SM-derived fallback; the seven measured
Qwen3.8-27B shapes use recorded split counts without embedding the local
124-SM inventory.

## Accepted correctness

`tests/test_qvq_p32_ampere.py` passes 10/10 cases on the target GPU:

- W2-W3.5 at M1 and M16;
- an N80 partial N64 block;
- split-K reconstruction;
- long-K accumulation;
- repeated launches on a non-default CUDA stream;
- output shape, FP32 dtype, finite values, exact repeatability, and the
  `2e-3` maximum-error contract.

The full formal benchmark adds 56 dense-reference cases across seven
Qwen3.8-27B projection shapes, four rates, and M1/M16. All 56 pass.

## Accepted performance

Artifacts:

- `artifacts/a100_p32_window/qwen38_m16_p32_ampere.json`
- `artifacts/a100_p32_window/qwen38_m1_p32_ampere.json`

The comparator is the current canonical planar P32 CUDA GEMV built from the
same checkout. It is quality-equivalent, unlike a W4 kernel comparison.

| Regime | Cases | Geomean speedup vs planar P32 | Speedup range | Worst max abs |
|---|---:|---:|---:|---:|
| M16 | 28 | 15.134x | 4.192x-21.198x | 2.823e-4 |
| M1 | 28 | 6.933x | 0.981x-21.226x | 2.632e-4 |

M16 per-shape speedup ranges across W2-W3.5:

| Shape | K | N | Split | Speedup range |
|---|---:|---:|---:|---:|
| Full Q+gate | 5120 | 12288 | 5 | 19.908x-20.821x |
| Full K/V | 5120 | 1024 | 8 | 4.192x-4.360x |
| Attention out | 6144 | 5120 | 8 | 15.218x-16.145x |
| Linear QKV | 5120 | 10240 | 6 | 19.644x-20.155x |
| Linear Z | 5120 | 6144 | 8 | 19.904x-20.083x |
| MLP gate/up | 5120 | 17408 | 8 | 20.080x-21.198x |
| MLP down | 17408 | 5120 | 8 | 15.626x-17.723x |

The only measured regression is M1/W2 full K/V: 0.053248 ms versus the planar
kernel's 0.052224 ms (`0.981x`). A future production dispatcher should retain
the planar path for this narrow low-rate case unless a new result clears it.

## Failed and discarded experiments

| Experiment | Observation | Decision |
|---|---|---|
| WMMA shared operands with 16-byte/natural alignment | M1 failed with `cudaErrorMisalignedAddress` before comparison. | Rejected. All WMMA shared operands and stores now have explicit 32-byte alignment. |
| Double buffering without a post-MMA block barrier | Some low-occupancy cases passed, but denser split grids produced multi-unit output corruption. Fast warps could overwrite a stage still consumed by slower warps under independent thread scheduling. | Rejected and all timings discarded. Added `__syncthreads()` at the producer/consumer handoff. |
| Split counts above eight during the unsafe-buffer experiment | Long-K splits 9-16 showed increasing corruption. | Not accepted. Public validation remains capped at eight even after the barrier fix; expand only with a new exhaustive correctness gate. |
| Generic two-wave split heuristic | Correct on the first formal matrix but left substantial performance unused on long-K and wide-N shapes. | Replaced by measured Qwen3.8 splits plus a live-SM fallback for unknown shapes. |
| Full repository QVQ comparator build | Failed because unrelated YAQA translation units require cuBLAS/cuSPARSE developer headers absent from this local toolkit. | Benchmark builds the current `qvq_gemv_cuda.cu` alone. The GEMV file now uses the lightweight current-stream header and remains source-identical to production GEMV. |

## Reproduction

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=GPU-14ab23f1-a785-e9df-bbb5-215547154e3c
export TORCH_CUDA_ARCH_LIST=8.0
export MAX_JOBS=8 NINJAFLAGS=-j8 CMAKE_BUILD_PARALLEL_LEVEL=8 NVCC_THREADS=2

python -m pytest -q tests/test_qvq_p32_ampere.py -s
python scripts/benchmark_qvq_p32_ampere.py --physical-gpu 0 --m-values 16 --warmup 10 --iterations 50
python scripts/benchmark_qvq_p32_ampere.py --physical-gpu 0 --m-values 1 --warmup 10 --iterations 50
```
