# QVQ V2B2-P32-LR CUDA kernel progression

This is the durable performance ledger for the local-ring (LR32) CUDA GEMV in
`gptqmodel_ext/qvq/qvq_gemv_cuda.cu`. Update it whenever an LR kernel or launch
policy changes. Keep regressions and reverted experiments in the history: they
are useful constraints for later optimization work.

## Measurement contract

- A result is identified by source commit, physical device index, PCI bus ID,
  UUID, compute capability, and the SM count queried from CUDA. Device
  properties are cached once per CUDA ordinal by `qvq_cuda_device_config()`.
- Unless a row says otherwise, timings use FP16, M=16, 10 warm-up launches, 60
  measured launches, CUDA events, and the automatic split policy.
- `speedup` is `non_lr_native median / lr_native_auto median`. `LR gain` is the
  previous LR median divided by the new LR median.
- Numerical acceptance requires maximum absolute error <=2e-3 for both LR and
  non-LR paths against the dense reference.
- The >=2x target applies to substantial model shapes. `m1_narrow` is retained
  as a launch-bound canary but is reported separately from the throughput gate.
  The stretch target is >=4x versus non-LR.
- An idle gate requires three clean `nvidia-smi` samples, 0% utilization, and
  at most 8 MiB allocated before a worker starts.

## Devices

| Status | Physical IDs | Device | CC | CUDA-reported SMs | Notes |
|---|---:|---|---:|---:|---|
| tested | 0, 1, 3, 4, 5, 6, 7, 8 | RTX 4090 | 8.9 | 128 each | Eight Ada devices; UUID-pinned workers |
| tested | 2 | RTX 5090 | 12.0 | 170 | One Blackwell device; UUID-pinned worker |
| pending | none on host | A100 | 8.0 | query required | Do not assume an SM count from the product name; no A100 result is claimed |

## Shape matrix

The benchmark's historical shape names describe their model role. In the
tables below every row was intentionally measured at M=16.

| Shape | M | K | N | Role |
|---|---:|---:|---:|---|
| `m1_narrow` | 16 | 2,048 | 256 | launch-bound canary |
| `m1_wide` | 16 | 2,048 | 8,192 | wide projection A |
| `m4_wide` | 16 | 2,048 | 8,192 | wide projection B / repeatability |
| `m16_mid` | 16 | 8,192 | 2,048 | split-K projection |
| `mlp_down` | 16 | 4,096 | 11,008 | MLP down projection |

## Commit progression

| Commit | Device/rate | Change or experiment | Measured outcome | Disposition |
|---|---|---|---|---|
| `bd625c15` | 4090/5090, W2-W3.5 | Last full matrix before the merged-main baseline | 4090 higher rates were near parity; 5090 W2 was near 2x and W2.5-W3.5 were 5.26-6.36x on substantial shapes | historical reference |
| `24601c7a` | all nine, W2 | Exact merged `origin/main` tip from PR #59 | 45/45 accurate; 5.163x all-device geomean versus non-LR | comparison baseline |
| `9b67f25c` | 5090, W2 | Broadcast one uniform bank mask per warp | 2.10-2.73x LR gain versus merged main on substantial shapes; Ada unchanged | accepted |
| `85181bd6`, `4ea1c0b7` | 4090/5090, W2 | Double-buffered `cp.async` input staging | 5090 regressed 13-24%; Ada collapsed to about 0.93 ms wide and 2.46 ms down | rejected; reverted by `a412bf9c` |
| `3ac9986e` | 4090/5090, W2 | Form ring state by bit permutation | Exact output but performance-neutral | rejected; reverted by `5d68b9f2` |
| `fae1a12a` | 4090/5090, W2 | Batch 16 K32 tiles for ROWS=16 | Ada gained 6-8%; 5090 gained another 5-34%, shape-dependent | accepted |
| `13a6d3a6` | 4090/5090, W2 | Extend batch 16 to small rows | Some Ada small-row paths regressed | narrowed by `bcf4d05d` |
| `bcf4d05d` | all nine, W2 | Architecture/row-specific batch depth | 5.999x all-device geomean versus non-LR; 5090 LR geomean gained 2.316x versus `24601c7a` | accepted |
| `0480f5c8` | 4090/5090, W3.5 | Broadcast the TB7 bank mask on Ada | 4090 substantial shapes rose from about 1x to 2.28-2.53x; 5090 remained 4.99-5.85x | accepted |
| `73c3cc86` | 5090, W2-W3.5 | Direct higher-rate recurrence in the shared specialization | Recovered W2.5/W3/W3.5 non-split rates, but compiler rescheduled W2 and W2.5 split-K; W2 wide/down fell near 2x and W2.5 mid to 1.41x | superseded |
| `294f15f4` | 4090/5090, W2-W3.5 | Isolate rate code generation; preserve batch 16 for Blackwell split-K | Higher rates recovered to 4.98-6.50x, but the specialization used TB2 instead of W2's TB4; W2 non-split remained 1.95-2.03x | superseded |
| `e4b1006c` | 4090/5090, W2-W3.5 | Attach the Blackwell shuffle/broadcast/deep-batch path to W2's actual TB4 width | 5090: all 16 substantial cells pass >=4x, 4.054x geomean including canaries, max error 1.53e-4; 4090 W3.5 remains 2.43-2.53x | accepted candidate |

## RTX 4090 accepted M=16 matrix

This combines W2-W3 from `bcf4d05d` and the accepted W3.5 Ada specialization
from `0480f5c8`. The prior column is `bd625c15`; W2 uses the same historical
lineage as merged main and is included for completeness.

| W | Shape | M | K | N | Result commit | LR ms | non-LR ms | Speedup | Prior LR ms | LR gain | Gate |
|---:|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---|
| 2 | `m1_narrow` | 16 | 2,048 | 256 | `bcf4d05d` | 0.0246 | 0.0369 | 1.50x | 0.0237 | 0.96x | canary |
| 2 | `m1_wide` | 16 | 2,048 | 8,192 | `bcf4d05d` | 0.1024 | 0.9432 | 9.21x | 0.1085 | 1.06x | pass >=4x |
| 2 | `m4_wide` | 16 | 2,048 | 8,192 | `bcf4d05d` | 0.1024 | 0.9441 | 9.22x | 0.1079 | 1.05x | pass >=4x |
| 2 | `m16_mid` | 16 | 8,192 | 2,048 | `bcf4d05d` | 0.1034 | 0.9196 | 8.89x | 0.1109 | 1.07x | pass >=4x |
| 2 | `mlp_down` | 16 | 4,096 | 11,008 | `bcf4d05d` | 0.2611 | 2.4699 | 9.46x | 0.2796 | 1.07x | pass >=4x |
| 2.5 | `m1_narrow` | 16 | 2,048 | 256 | `bcf4d05d` | 0.0243 | 0.0369 | 1.52x | 0.0358 | 1.47x | canary |
| 2.5 | `m1_wide` | 16 | 2,048 | 8,192 | `bcf4d05d` | 0.1044 | 0.9421 | 9.02x | 0.9062 | 8.68x | pass >=4x |
| 2.5 | `m4_wide` | 16 | 2,048 | 8,192 | `bcf4d05d` | 0.1044 | 0.9411 | 9.01x | 0.9011 | 8.63x | pass >=4x |
| 2.5 | `m16_mid` | 16 | 8,192 | 2,048 | `bcf4d05d` | 0.1065 | 0.9267 | 8.70x | 0.9073 | 8.52x | pass >=4x |
| 2.5 | `mlp_down` | 16 | 4,096 | 11,008 | `bcf4d05d` | 0.2673 | 2.3736 | 8.88x | 2.4084 | 9.01x | pass >=4x |
| 3 | `m1_narrow` | 16 | 2,048 | 256 | `bcf4d05d` | 0.0252 | 0.0369 | 1.46x | 0.0369 | 1.46x | canary |
| 3 | `m1_wide` | 16 | 2,048 | 8,192 | `bcf4d05d` | 0.1311 | 0.9441 | 7.20x | 0.8359 | 6.38x | pass >=4x |
| 3 | `m4_wide` | 16 | 2,048 | 8,192 | `bcf4d05d` | 0.1311 | 0.9523 | 7.27x | 0.8387 | 6.40x | pass >=4x |
| 3 | `m16_mid` | 16 | 8,192 | 2,048 | `bcf4d05d` | 0.1341 | 0.9227 | 6.88x | 0.8704 | 6.49x | pass >=4x |
| 3 | `mlp_down` | 16 | 4,096 | 11,008 | `bcf4d05d` | 0.3410 | 2.2760 | 6.67x | 2.2344 | 6.55x | pass >=4x |
| 3.5 | `m1_narrow` | 16 | 2,048 | 256 | `0480f5c8` | 0.0307 | 0.0369 | 1.20x | 0.0358 | 1.17x | canary |
| 3.5 | `m1_wide` | 16 | 2,048 | 8,192 | `0480f5c8` | 0.3741 | 0.9404 | 2.51x | 0.8366 | 2.24x | pass >=2x |
| 3.5 | `m4_wide` | 16 | 2,048 | 8,192 | `0480f5c8` | 0.3738 | 0.9472 | 2.53x | 0.8346 | 2.23x | pass >=2x |
| 3.5 | `m16_mid` | 16 | 8,192 | 2,048 | `0480f5c8` | 0.3779 | 0.9226 | 2.44x | 0.8387 | 2.22x | pass >=2x |
| 3.5 | `mlp_down` | 16 | 4,096 | 11,008 | `0480f5c8` | 0.9974 | 2.2743 | 2.28x | 2.2211 | 2.23x | pass >=2x |

## RTX 5090 accepted M=16 matrix

All rows below come from `e4b1006c`, source fingerprint
`176b415b6b673bea64ab91d4c1e56f97e72c8ff1dd90597a690718a9ba4069fc`.
Every substantial shape clears the 4x stretch target; narrow rows remain
launch-bound canaries.

| W | Shape | M | K | N | Result commit | LR ms | non-LR ms | Speedup | Status |
|---:|---|---:|---:|---:|---|---:|---:|---:|---|
| 2 | `m1_narrow` | 16 | 2,048 | 256 | `e4b1006c` | 0.0248 | 0.0279 | 1.13x | canary |
| 2 | `m1_wide` | 16 | 2,048 | 8,192 | `e4b1006c` | 0.0959 | 0.5001 | 5.21x | pass >=4x |
| 2 | `m4_wide` | 16 | 2,048 | 8,192 | `e4b1006c` | 0.0955 | 0.4963 | 5.20x | pass >=4x |
| 2 | `m16_mid` | 16 | 8,192 | 2,048 | `e4b1006c` | 0.0916 | 0.4993 | 5.45x | pass >=4x |
| 2 | `mlp_down` | 16 | 4,096 | 11,008 | `e4b1006c` | 0.2347 | 1.3477 | 5.74x | pass >=4x |
| 2.5 | `m1_narrow` | 16 | 2,048 | 256 | `e4b1006c` | 0.0249 | 0.0279 | 1.12x | canary |
| 2.5 | `m1_wide` | 16 | 2,048 | 8,192 | `e4b1006c` | 0.0836 | 0.4988 | 5.96x | pass >=4x |
| 2.5 | `m4_wide` | 16 | 2,048 | 8,192 | `e4b1006c` | 0.0838 | 0.5064 | 6.04x | pass >=4x |
| 2.5 | `m16_mid` | 16 | 8,192 | 2,048 | `e4b1006c` | 0.0897 | 0.4977 | 5.55x | pass >=4x |
| 2.5 | `mlp_down` | 16 | 4,096 | 11,008 | `e4b1006c` | 0.2085 | 1.3563 | 6.50x | pass >=4x |
| 3 | `m1_narrow` | 16 | 2,048 | 256 | `e4b1006c` | 0.0243 | 0.0279 | 1.15x | canary |
| 3 | `m1_wide` | 16 | 2,048 | 8,192 | `e4b1006c` | 0.0978 | 0.5056 | 5.17x | pass >=4x |
| 3 | `m4_wide` | 16 | 2,048 | 8,192 | `e4b1006c` | 0.0979 | 0.5007 | 5.11x | pass >=4x |
| 3 | `m16_mid` | 16 | 8,192 | 2,048 | `e4b1006c` | 0.1019 | 0.4984 | 4.89x | pass >=4x |
| 3 | `mlp_down` | 16 | 4,096 | 11,008 | `e4b1006c` | 0.2450 | 1.3584 | 5.54x | pass >=4x |
| 3.5 | `m1_narrow` | 16 | 2,048 | 256 | `e4b1006c` | 0.0249 | 0.0289 | 1.16x | canary |
| 3.5 | `m1_wide` | 16 | 2,048 | 8,192 | `e4b1006c` | 0.0877 | 0.5076 | 5.79x | pass >=4x |
| 3.5 | `m4_wide` | 16 | 2,048 | 8,192 | `e4b1006c` | 0.0886 | 0.5076 | 5.73x | pass >=4x |
| 3.5 | `m16_mid` | 16 | 8,192 | 2,048 | `e4b1006c` | 0.0938 | 0.4973 | 5.30x | pass >=4x |
| 3.5 | `mlp_down` | 16 | 4,096 | 11,008 | `e4b1006c` | 0.2202 | 1.3554 | 6.15x | pass >=4x |

## Coverage and targeting queue

| Priority | Device/rate/shape | Current state | Next evidence needed |
|---:|---|---|---|
| 1 | RTX 5090 W2-W3.5, all substantial M=16 shapes | `e4b1006c`: all 16 pass >=4x | retain in expanded row/dtype coverage |
| 2 | all nine installed GPUs, W2-W3.5 | W2 has all-device coverage; higher rates have representative 4090/5090 coverage | final UUID-pinned all-device FP16 M=16 sweep |
| 3 | RTX 4090/5090, M=1/4/8/32 and BF16 | earlier full matrix exists at `bd625c15`, not yet repeated for this branch head | expanded accuracy/performance regression |
| 4 | A100 W2-W3.5, all M/K/N cells | no A100 installed; SM count unknown | query properties once by device ordinal, then run the same matrix |
| 5 | launch-bound narrow shapes | typically 0.98-1.52x | reduce launch/split overhead without regressing substantial shapes |
| 6 | hardware-counter attribution | Nsight Systems works; Nsight Compute reports `ERR_NVGPUCTRPERM`, with `RmProfilingAdminOnly: 1` | rerun NCU after the driver exposes counters |

## Reproduction

Representative device sweep:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
GPTQMODEL_QVQ_CUDA_BUILD_ROOT=/tmp/qvq-jit-lr-gpu2 \
python scripts/benchmark_qvq_cuda_lr.py \
  --physical-gpu 2 --bits 2 2.5 3 3.5 --dtype float16 --m 16 \
  --warmup 10 --iterations 60 --out-dir artifacts/<stamp>
```

All installed devices in parallel:

```bash
python scripts/benchmark_qvq_cuda_lr.py \
  --all --bits 2 2.5 3 3.5 --dtype float16 --m 16 \
  --warmup 10 --iterations 60 --out-dir artifacts/<stamp>
```

Artifact directories are intentionally local/untracked. Copy accepted medians,
source commit, fingerprint, device identity, SM count, accuracy, and split count
into this ledger before artifacts are retired.
