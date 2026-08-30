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
| tested | 0 (current Hopper host) | H200 | 9.0 | 132 | PCI `00000000:1C:00.0`; UUID `GPU-0c667065-5c47-38ce-0b0a-d211392ce9ea`; 143,771 MiB |
| tested | 1 (current Hopper host) | H100 | 9.0 | 132 | PCI `00000000:44:00.0`; UUID `GPU-f5ea03cf-efa4-9807-7de5-b174957a1348`; 97,871 MiB |
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
| `e4b1006c` | 4090/5090, W2-W3.5 | Attach the Blackwell shuffle/broadcast/deep-batch path to W2's actual TB4 width | 5090: all 16 substantial cells pass >=4x, 4.054x geomean including canaries, max error 1.53e-4; 4090 W3.5 remains 2.43-2.53x | accepted source |
| `18389b4d` | all nine, W2-W3.5 | Final UUID-pinned simultaneous sweep of `e4b1006c` plus the ledger | 180/180 accurate; 4.457x overall geomean; all 144 substantial cells >=2x and 112 >=4x | accepted system gate |
| `ca4f8039` | H200/H100, W2-W3.5 | Exact merged PR #60 baseline on the two-GPU Hopper host | H200 W3 was 1.00-1.05x versus non-LR on substantial M16 shapes; H100 W3 was 0.98-1.04x. Other requested rates already exceeded 2x | Hopper comparison baseline |
| `611d249e` | H200, W3 | Select the pre-existing TB6 vector-staging kernel on CC 9.x | Accurate, but only 1-2% faster than `ca4f8039`; W3 remained near parity with non-LR | rejected; reverted by `b7d0e256` |
| `0080d53e` | H200/H100, W3 | Broadcast the uniform TB6 bank mask once per warp on CC 9.x | H200 LR gained 2.15-2.29x and H100 gained 2.11-2.28x versus `ca4f8039`; all substantial M16 W2-W3.5 cells exceed 2x on both devices | accepted source |
| `f6e417d9` | H200, W3 M32 | Batch 16 K32 tiles for Hopper ROWS=32 | Wide gained about 2% and down about 3%, but mid regressed about 1.5%; registers rose to 80/thread and shared memory to 36,368 bytes | rejected; reverted by `3aea27bd` |
| `3aea27bd` | H200/H100, W2-W3.5 | Restore the accepted `0080d53e` source after the M32 experiment | Source-equivalent to `0080d53e`; 28 CUDA tests and 50 host tests passed; NCU access restored for final attribution | accepted validation head before ledger stamp |
| `82cfba57` | H100, W2/W2.5 M1-M16 | Storage-neutral cooperative local-ring prototype reconstructs each unique transition once per warp and advances four adjacent states by recurrence | Up to 11.40x faster than production; M16 substantial shapes gained 1.29-2.23x with unchanged checkpoint storage | accepted prototype evidence |
| `41ffd8db` | H200, W2/W2.5 M16 | Pulled prototype head reproduced on physical GPU 0 before production integration | All eight shape/rate cells accurate; 1.28-2.26x faster than production. NCU gate/up W2: 105.48M to 18.48M instructions (-82.5%) | authoritative new-head baseline |
| `f7f29083` | H200, W2/W2.5 M16 | Integrate cooperative TB4/TB5 reconstruction into the production Hopper WMMA path; retain compile-time-specialized TB6 unpack | Production matches the isolated prototype within 0.1%; 36 CUDA cases pass, including FP16/FP32 output and split-1/split-3; W3 timing unchanged | accepted and pushed |
| `f7ae7daf` | H200, W2/W2.5 M16 | Store each decoded half2 pair contiguously in a TB4/TB5-only column-major WMMA tile; TB6 remains row-major | All eight cells gain 1-3%; shared-store conflicts halve; 36 CUDA tests pass and W3 remains unchanged | accepted and pushed |
| `8b659d41` | H100, W2/W2.5 M1-M16 | Extract four contiguous packed transitions once and derive the cooperative states without repeated planar extraction | Remote H100 evidence reduced the remaining TB4/TB5 integer decode work without changing checkpoint storage or arithmetic | accepted upstream evidence |
| `2a71cad5` | H100, W2/W2.5 M16 | Replace the padded N16 WMMA consumer with a row-major native-N8 MMA path using `ldmatrix.x2.trans` | Improved many H100 cells but retained a shared decoded-weight round trip and regressed some H100 geometries | superseded by direct-fragment path |
| `6844450b` | H200, W2/W2.5 M16 | Map the exact Amplin `m16n8k16` fragments, store four decoded half2 planes, load B fragments directly, and store accumulator fragments without padding/reduction | All eight cells accurate; isolated medians fell to 0.0128-0.0518 ms before the packed-transition merge | accepted and pushed |
| `1521292b` | H200, W2/W2.5 M16 | Merge packed transition extraction with the direct native-N8 fragment consumer | 0.0113-0.0448 ms, 14.20M executed instructions on gate/up W2, zero shared-store conflicts, and 69/69 CUDA tests | accepted and pushed |
| `3997021c` / `ac440de3` | H200, W2/W2.5 M16 | XOR-swizzle the four 16-byte activation segments by row and remap native-N8 A-fragment loads | 0.0115-0.0400 ms; `ldmatrix` conflicts fell 66.7% and total load conflicts 42.8%; 69/69 CUDA tests | accepted, merged, and pushed |
| `6f4e4b72` / `e944e528` | H200, W2/W2.5 M16 | Pad native-N8 activation rows from 32 to 40 half values; merge the upstream H100 lead and validate locally on H200 GPU 0 | MLP gained 2.6-5.8%; NCU gate/up fell 6.9% with 35.9% fewer load conflicts; 69/69 CUDA tests passed | accepted, merged, and pushed |
| `603a3e64` uncommitted A/B | H200, W2/W2.5 M16 | Replace each synchronous 16-byte activation copy with `cp.async`, then immediately commit/wait before the existing barrier | Accurate, but Q/O rose to 0.024 ms and MLP to 0.069-0.071 ms, roughly 1.5-1.9x slower | rejected; source restored before next experiment |

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

## Hopper accepted M=16 matrix

The H200 baseline and accepted runs use physical GPU 0, CC 9.0, 132 SMs,
fingerprints `e7b6a889d9b3ca292743622a7096f5950b677b0fb9ed45fcb87a7e73691efa3e`
and `70cac1d75faa027f34edb6c420eae3d78f5a72621829222d910b3d604f65b5fc`,
respectively. The accepted source is `0080d53e`; final validation head
`3aea27bd` is source-identical after reverting the rejected M32 experiment.

### H200 exact matrix

| W | Shape | M | K | N | Result commit | LR ms | non-LR ms | Speedup | Prior LR ms | LR gain | Gate |
|---:|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---|
| 2 | `m1_narrow` | 16 | 2,048 | 256 | `0080d53e` | 0.0293 | 0.0292 | 0.99x | 0.0297 | 1.01x | canary |
| 2 | `m1_wide` | 16 | 2,048 | 8,192 | `0080d53e` | 0.1308 | 0.4950 | 3.78x | 0.1307 | 1.00x | pass >=2x |
| 2 | `m4_wide` | 16 | 2,048 | 8,192 | `0080d53e` | 0.1309 | 0.4958 | 3.79x | 0.1308 | 1.00x | pass >=2x |
| 2 | `m16_mid` | 16 | 8,192 | 2,048 | `0080d53e` | 0.1329 | 0.4843 | 3.64x | 0.1329 | 1.00x | pass >=2x |
| 2 | `mlp_down` | 16 | 4,096 | 11,008 | `0080d53e` | 0.3389 | 1.3177 | 3.89x | 0.3389 | 1.00x | pass >=2x |
| 2.5 | `m1_narrow` | 16 | 2,048 | 256 | `0080d53e` | 0.0361 | 0.0289 | 0.80x | 0.0342 | 0.95x | canary |
| 2.5 | `m1_wide` | 16 | 2,048 | 8,192 | `0080d53e` | 0.1320 | 0.4938 | 3.74x | 0.1320 | 1.00x | pass >=2x |
| 2.5 | `m4_wide` | 16 | 2,048 | 8,192 | `0080d53e` | 0.1321 | 0.4983 | 3.77x | 0.1320 | 1.00x | pass >=2x |
| 2.5 | `m16_mid` | 16 | 8,192 | 2,048 | `0080d53e` | 0.1369 | 0.4842 | 3.54x | 0.1369 | 1.00x | pass >=2x |
| 2.5 | `mlp_down` | 16 | 4,096 | 11,008 | `0080d53e` | 0.3420 | 1.3439 | 3.93x | 0.3421 | 1.00x | pass >=2x |
| 3 | `m1_narrow` | 16 | 2,048 | 256 | `0080d53e` | 0.0393 | 0.0288 | 0.73x | 0.0361 | 0.92x | canary |
| 3 | `m1_wide` | 16 | 2,048 | 8,192 | `0080d53e` | 0.2213 | 0.4989 | 2.25x | 0.4815 | 2.18x | pass >=2x |
| 3 | `m4_wide` | 16 | 2,048 | 8,192 | `0080d53e` | 0.2201 | 0.4917 | 2.23x | 0.4825 | 2.19x | pass >=2x |
| 3 | `m16_mid` | 16 | 8,192 | 2,048 | `0080d53e` | 0.2254 | 0.4832 | 2.14x | 0.4837 | 2.15x | pass >=2x |
| 3 | `mlp_down` | 16 | 4,096 | 11,008 | `0080d53e` | 0.5553 | 1.3283 | 2.39x | 1.2693 | 2.29x | pass >=2x |
| 3.5 | `m1_narrow` | 16 | 2,048 | 256 | `0080d53e` | 0.0400 | 0.0289 | 0.72x | 0.0366 | 0.92x | canary |
| 3.5 | `m1_wide` | 16 | 2,048 | 8,192 | `0080d53e` | 0.2167 | 0.4956 | 2.29x | 0.2164 | 1.00x | pass >=2x |
| 3.5 | `m4_wide` | 16 | 2,048 | 8,192 | `0080d53e` | 0.2098 | 0.4988 | 2.38x | 0.2101 | 1.00x | pass >=2x |
| 3.5 | `m16_mid` | 16 | 8,192 | 2,048 | `0080d53e` | 0.2136 | 0.4861 | 2.28x | 0.2135 | 1.00x | pass >=2x |
| 3.5 | `mlp_down` | 16 | 4,096 | 11,008 | `0080d53e` | 0.5420 | 1.3397 | 2.47x | 0.5415 | 1.00x | pass >=2x |

All 20 H200 pairs pass the `2e-3` accuracy gate with worst-case absolute
error `1.53e-4`. All 16 substantial cells exceed 2x versus non-LR; the four
narrow cells remain explicitly separated launch canaries.

### H200 cooperative W2/W2.5 production integration

Commit `f7f29083` is based on pulled head `41ffd8db` and ports the
storage-neutral prototype from `82cfba57` into production. These are isolated
H200 FP16-input, FP32-output M16 medians with 10 warm-ups and 40 measured CUDA
graph launches. `Prior LR` is production at `41ffd8db`; Marlin and Machete are
their W4 baselines measured in the same run. The lower-rate QVQ payload remains
W2/W2.5, so cross-rate parity ratios are directional throughput references.

| W | Shape | M | K | N | Result commit | LR ms | Prior LR ms | LR gain | xMarlin W4 | xMachete W4 | Max abs |
|---:|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 2 | `attn_qo` | 16 | 2,048 | 2,048 | `f7f29083` | 0.0209 | 0.0430 | 2.06x | 0.752x | 0.732x | 1.43e-5 |
| 2.5 | `attn_qo` | 16 | 2,048 | 2,048 | `f7f29083` | 0.0221 | 0.0441 | 1.99x | 0.708x | 0.689x | 1.43e-5 |
| 2 | `attn_kv` | 16 | 2,048 | 512 | `f7f29083` | 0.0137 | 0.0185 | 1.35x | 1.745x | 1.048x | 7.63e-6 |
| 2.5 | `attn_kv` | 16 | 2,048 | 512 | `f7f29083` | 0.0144 | 0.0187 | 1.30x | 1.654x | 0.993x | 8.11e-6 |
| 2 | `mlp_gate_up` | 16 | 2,048 | 8,192 | `f7f29083` | 0.0577 | 0.1304 | 2.26x | 0.189x | 0.310x | 7.63e-5 |
| 2.5 | `mlp_gate_up` | 16 | 2,048 | 8,192 | `f7f29083` | 0.0607 | 0.1318 | 2.17x | 0.180x | 0.295x | 6.29e-5 |
| 2 | `mlp_down` | 16 | 8,192 | 2,048 | `f7f29083` | 0.0587 | 0.1322 | 2.25x | 0.309x | 0.382x | 9.92e-5 |
| 2.5 | `mlp_down` | 16 | 8,192 | 2,048 | `f7f29083` | 0.0627 | 0.1361 | 2.17x | 0.289x | 0.358x | 1.09e-4 |

The K/V W2 path now matches or exceeds Machete W4. Attention Q/O is within
about 27-31%; MLP remains 2.6-3.4x slower than Machete despite the decode
instruction collapse. That residual gap is therefore in the padded WMMA/shared
consumer and launch structure, not the eliminated redundant state windows.

Commit `f7ae7daf` packs each cooperative TB4/TB5 decoded pair as one `half2`
in the column-major WMMA B tile. The arithmetic and TB6 path are unchanged.
These medians use the same H200 protocol; `Prior LR` is the production result
from `f7f29083` measured before the layout change.

| W | Shape | M | K | N | Result commit | LR ms | Prior LR ms | LR gain | xMarlin W4 | xMachete W4 | Max abs |
|---:|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 2 | `attn_qo` | 16 | 2,048 | 2,048 | `f7ae7daf` | 0.0204 | 0.0209 | 1.020x | 0.766x | 0.734x | 1.43e-5 |
| 2.5 | `attn_qo` | 16 | 2,048 | 2,048 | `f7ae7daf` | 0.0217 | 0.0221 | 1.018x | 0.720x | 0.690x | 1.43e-5 |
| 2 | `attn_kv` | 16 | 2,048 | 512 | `f7ae7daf` | 0.0134 | 0.0137 | 1.018x | 1.777x | 1.117x | 7.63e-6 |
| 2.5 | `attn_kv` | 16 | 2,048 | 512 | `f7ae7daf` | 0.0143 | 0.0144 | 1.007x | 1.666x | 1.047x | 8.11e-6 |
| 2 | `mlp_gate_up` | 16 | 2,048 | 8,192 | `f7ae7daf` | 0.0560 | 0.0577 | 1.030x | 0.194x | 0.318x | 7.63e-5 |
| 2.5 | `mlp_gate_up` | 16 | 2,048 | 8,192 | `f7ae7daf` | 0.0592 | 0.0607 | 1.027x | 0.183x | 0.301x | 6.29e-5 |
| 2 | `mlp_down` | 16 | 8,192 | 2,048 | `f7ae7daf` | 0.0572 | 0.0587 | 1.025x | 0.317x | 0.387x | 9.92e-5 |
| 2.5 | `mlp_down` | 16 | 8,192 | 2,048 | `f7ae7daf` | 0.0614 | 0.0627 | 1.021x | 0.296x | 0.361x | 1.09e-4 |

All eight cells improve by 1-3% and the K/V no-regression gate now exceeds
Machete W4 for both rates. The MLP gap remains 2.6-3.3x, so this is an accepted
layout improvement rather than the final consumer architecture.

Head `ac440de3` combines the packed-transition extractor, direct native
N8 MMA fragments, and activation-row XOR swizzle. These are the exact medians
at merged/pushed head `ac440de3`; `Prior LR` is the unswizzled direct-fragment
head `1521292b`. The W3/TB6 specialization is compile-time unchanged.

| W | Shape | M | K | N | Result commit | LR ms | Prior LR ms | LR gain | xMarlin W4 | xMachete W4 | Max abs |
|---:|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 2 | `attn_qo` | 16 | 2,048 | 2,048 | `ac440de3` | 0.015584 | 0.016032 | 1.029x | 1.016x | 0.995x | 1.43e-5 |
| 2.5 | `attn_qo` | 16 | 2,048 | 2,048 | `ac440de3` | 0.015920 | 0.017152 | 1.077x | 0.995x | 0.974x | 1.43e-5 |
| 2 | `attn_kv` | 16 | 2,048 | 512 | `ac440de3` | 0.011488 | 0.011328 | 0.986x | 2.047x | 1.259x | 7.63e-6 |
| 2.5 | `attn_kv` | 16 | 2,048 | 512 | `ac440de3` | 0.011680 | 0.012128 | 1.038x | 2.014x | 1.238x | 8.11e-6 |
| 2 | `mlp_gate_up` | 16 | 2,048 | 8,192 | `ac440de3` | 0.037424 | 0.038864 | 1.038x | 0.295x | 0.478x | 7.63e-5 |
| 2.5 | `mlp_gate_up` | 16 | 2,048 | 8,192 | `ac440de3` | 0.037808 | 0.042912 | 1.135x | 0.292x | 0.473x | 6.29e-5 |
| 2 | `mlp_down` | 16 | 8,192 | 2,048 | `ac440de3` | 0.039200 | 0.040768 | 1.040x | 0.461x | 0.564x | 9.92e-5 |
| 2.5 | `mlp_down` | 16 | 8,192 | 2,048 | `ac440de3` | 0.040000 | 0.044832 | 1.121x | 0.452x | 0.552x | 1.09e-4 |

Relative to the first packed-pair production head `f7ae7daf`, the complete
native-N8 path is 1.17-1.56x faster on K/V and MLP and 1.31-1.37x faster on
Q/O. Q/O is now within 2.6% of Machete W4 and K/V is 1.24-1.26x faster. The
remaining MLP gap is 1.77-2.11x, making async staging, shared-load pressure,
and the much narrower N tile the next structural targets.

The stride-40 follow-up at current pushed head `e944e528` adds eight padding
halves per activation row to break the residual native-N8 bank periodicity.
`Prior LR` below is `ac440de3`; all measurements are H200 GPU 0 only with 20
warm-ups and 60 CUDA-event launches.

| W | Shape | M | K | N | LR ms | Prior LR ms | LR gain | xMarlin W4 | xMachete W4 | Max abs |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | `attn_qo` | 16 | 2,048 | 2,048 | 0.016064 | 0.015584 | 0.970x | 0.970x | 0.955x | 1.43e-5 |
| 2.5 | `attn_qo` | 16 | 2,048 | 2,048 | 0.015904 | 0.015920 | 1.001x | 0.980x | 0.965x | 1.43e-5 |
| 2 | `attn_kv` | 16 | 2,048 | 512 | 0.011616 | 0.011488 | 0.989x | 2.047x | 1.256x | 7.63e-6 |
| 2.5 | `attn_kv` | 16 | 2,048 | 512 | 0.011632 | 0.011680 | 1.004x | 2.044x | 1.254x | 8.11e-6 |
| 2 | `mlp_gate_up` | 16 | 2,048 | 8,192 | 0.036192 | 0.037424 | 1.034x | 0.296x | 0.496x | 7.63e-5 |
| 2.5 | `mlp_gate_up` | 16 | 2,048 | 8,192 | 0.035872 | 0.037808 | 1.054x | 0.299x | 0.500x | 6.29e-5 |
| 2 | `mlp_down` | 16 | 8,192 | 2,048 | 0.038224 | 0.039200 | 1.026x | 0.470x | 0.577x | 9.92e-5 |
| 2.5 | `mlp_down` | 16 | 8,192 | 2,048 | 0.037792 | 0.040000 | 1.058x | 0.475x | 0.583x | 1.09e-4 |

The two W2 attention deltas are 0.13-0.48 us regressions; W2.5 attention is
flat-to-better, while every MLP cell moves forward. The change is accepted for
the throughput target because its NCU improvement is structural and the small
attention paths remain within 4.5% of Machete or 1.25x faster. Future changes
must recover Q/O W2 without surrendering the MLP gain.

### H100 same-CC regression

The H100 run uses physical GPU 1, CC 9.0, 132 SMs, and the same accepted
fingerprint. All 16 substantial cells exceed 2x and worst-case absolute error
is `1.53e-4`.

| W | Substantial cells | LR ms range | Geomean speedup | Min-max speedup | Gate |
|---:|---:|---:|---:|---:|---|
| 2 | 4 | 0.1298-0.3365 | 3.710x | 3.53-3.79x | all >=2x |
| 2.5 | 4 | 0.1310-0.3396 | 3.660x | 3.45-3.78x | all >=2x |
| 3 | 4 | 0.2164-0.5380 | 2.239x | 2.13-2.37x | all >=2x |
| 3.5 | 4 | 0.2053-0.5280 | 2.342x | 2.26-2.41x | all >=2x |

The exact W3 progression, which is the path changed by this branch, is:

| Shape | M | K | N | LR ms | non-LR ms | Speedup | Prior LR ms | LR gain | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `m1_narrow` | 16 | 2,048 | 256 | 0.0340 | 0.0285 | 0.84x | 0.0327 | 0.96x | canary |
| `m1_wide` | 16 | 2,048 | 8,192 | 0.2234 | 0.4902 | 2.19x | 0.4724 | 2.11x | pass >=2x |
| `m4_wide` | 16 | 2,048 | 8,192 | 0.2164 | 0.4896 | 2.26x | 0.4700 | 2.17x | pass >=2x |
| `m16_mid` | 16 | 8,192 | 2,048 | 0.2189 | 0.4663 | 2.13x | 0.4745 | 2.17x | pass >=2x |
| `mlp_down` | 16 | 4,096 | 11,008 | 0.5380 | 1.2775 | 2.37x | 1.2279 | 2.28x | pass >=2x |

### H200 W3 expanded rows and dtypes

This sweep holds W3 fixed and covers both production input dtypes across all
compiled row specializations. Each row summarizes the four substantial K/N
cells; narrow launch canaries are excluded from these geomeans.

| Dtype | M | Cells | LR ms range | Geomean speedup | Min-max speedup | Max abs | Gate |
|---|---:|---:|---:|---:|---:|---:|---|
| FP16 | 1 | 4 | 0.2111-0.5864 | 2.225x | 2.05-2.31x | 1.83e-4 | 4/4 >=2x |
| FP16 | 4 | 4 | 0.2023-0.5242 | 2.418x | 2.37-2.50x | 1.22e-4 | 4/4 >=2x |
| FP16 | 8 | 4 | 0.2021-0.5278 | 2.420x | 2.38-2.49x | 1.79e-4 | 4/4 >=2x |
| FP16 | 16 | 4 | 0.2205-0.5516 | 2.257x | 2.16-2.39x | 1.37e-4 | 4/4 >=2x |
| FP16 | 32 | 4 | 0.2811-0.7353 | 2.026x | 1.76-2.35x | 3.74e-4 | 3/4 >=2x |
| BF16 | 1 | 4 | 0.2102-0.5816 | 2.223x | 2.05-2.31x | 1.83e-4 | 4/4 >=2x |
| BF16 | 4 | 4 | 0.2020-0.5269 | 2.423x | 2.38-2.49x | 1.22e-4 | 4/4 >=2x |
| BF16 | 8 | 4 | 0.2019-0.5286 | 2.418x | 2.37-2.49x | 1.45e-4 | 4/4 >=2x |
| BF16 | 16 | 4 | 0.2196-0.5507 | 2.241x | 2.12-2.40x | 1.22e-4 | 4/4 >=2x |
| BF16 | 32 | 4 | 0.2775-0.7253 | 2.042x | 1.76-2.38x | 3.05e-4 | 3/4 >=2x |

The two remaining sub-2x substantial cells are W3 M32/K8192/N2048 in FP16
and BF16. A split-count sweep found split 4 already optimal; the rejected
16-tile ROWS=32 batch did not close the gap.

### Hopper validation

- H200 CUDA dispatch/correctness: 69/69 tests passed at current stride-40
  native-N8 head `e944e528`, covering W1-W3.5, FP16/BF16,
  M1/M2/M4/M8/M16/M17,
  split-K, typed output, streams, determinism, validation errors, the
  cooperative W2/W2.5 Hopper path at split 1/3, and an explicit W3 unchanged
  regression.
- Host benchmark/layout/profiler suites: 50 passed and 141 platform skips.
- H200 accepted M16 sweep: 20/20 LR/non-LR pairs passed accuracy; all 16
  substantial W2-W3.5 cells passed >=2x.
- H100 rows in this document are retained historical/upstream evidence. After
  the H200-only instruction, every new local benchmark, CUDA test, and NCU
  capture uses physical GPU 0 only; no current acceptance claim depends on the
  installed H100.

## Final all-device aggregate

The simultaneous run at `18389b4d`, fingerprint
`7cd6585215bd3b95557a910b3b787d597c3ca3989e98e40901665d2c5b380c5a`,
contains 180 LR/non-LR pairs (540 timed path rows). The aggregate below excludes
the narrow canary and covers the four substantial K/N cells in the shape table.
Running all nine boards together exposes system power/thermal contention and is
therefore kept distinct from the isolated-device matrix above.

| Device group | W | M | K/N cells | Cells | LR ms range | Geomean speedup | Min-max speedup | Gate |
|---|---:|---:|---|---:|---:|---:|---:|---|
| 8x RTX 4090 | 2 | 16 | wide A/B, mid, down | 32 | 0.1014-0.2621 | 9.099x | 8.541-9.461x | all >=4x |
| 8x RTX 4090 | 2.5 | 16 | wide A/B, mid, down | 32 | 0.1044-0.2684 | 8.936x | 8.588-9.199x | all >=4x |
| 8x RTX 4090 | 3 | 16 | wide A/B, mid, down | 32 | 0.1300-0.3432 | 7.094x | 6.691-7.267x | all >=4x |
| 8x RTX 4090 | 3.5 | 16 | wide A/B, mid, down | 32 | 0.3707-1.0071 | 2.459x | 2.269-2.520x | all >=2x |
| 1x RTX 5090 | 2 | 16 | wide A/B, mid, down | 4 | 0.0939-0.2311 | 5.434x | 5.120-5.893x | all >=4x |
| 1x RTX 5090 | 2.5 | 16 | wide A/B, mid, down | 4 | 0.0977-0.2413 | 5.285x | 4.944-5.723x | all >=4x |
| 1x RTX 5090 | 3 | 16 | wide A/B, mid, down | 4 | 0.1224-0.3065 | 4.205x | 4.091-4.455x | all >=4x |
| 1x RTX 5090 | 3.5 | 16 | wide A/B, mid, down | 4 | 0.1020-0.2536 | 5.045x | 4.728-5.450x | all >=4x |

Against exact merged main `24601c7a`, the accepted W2 kernel's substantial LR
geomean improves by 2.722x on the 5090 and 1.063-1.071x on each 4090. The Ada
baseline already ran W2 at roughly 9x versus non-LR; the large Ada gains in this
series are W2.5 (about 8.6-9.2x versus non-LR), W3 (about 6.7-7.3x), and W3.5
(about 2.3-2.5x).

## Hopper profiler attribution

Nsight Compute 2026.2.1 access was restored on the Hopper host after the
initial `ERR_NVGPUCTRPERM` failure. The formal target was the accepted W3
FP16 M16/K4096/N11008 non-split LR kernel on H200. NCU timings include replay
and instrumentation overhead and are not used as final latency evidence; the
CUDA-event matrices above remain authoritative.

| Metric group | Metric | Value | Interpretation |
|---|---|---:|---|
| SOL | Compute throughput | 60.05% | substantial compute use, but below the cache-request ceiling |
| SOL | Memory throughput | 80.27% | primary SOL limiter |
| SOL | DRAM throughput | 0.90% | not DRAM-bandwidth bound |
| cache | L1/TEX hit rate | 84.67% | most staged loads hit on chip |
| cache | L2 hit rate | 98.71% | the 80.94% memory-request rate is almost entirely L2-resident |
| occupancy | theoretical / achieved | 50.00% / 45.89% | four blocks/SM by registers; shared memory permits six |
| resources | registers / static shared | 59/thread / 18,960 bytes/block | zero local- or shared-spilling requests |
| scheduler | issue active | 64.12% | 0.64 issued warp per scheduler per active cycle |
| scheduler | active / eligible warps | 7.33 / 1.90 per scheduler | enough resident warps, but dependencies reduce eligibility |
| stalls | long scoreboard / not selected | 1.98 / 1.96 cycles per issued instruction | cache dependency and normal arbitration dominate |
| stalls | wait / MIO throttle | 1.55 / 1.23 cycles per issued instruction | secondary dependency/shared-memory pressure |
| stalls | short scoreboard / barrier | 1.14 / 1.04 cycles per issued instruction | staging synchronization remains a smaller target |

The post-broadcast W3 kernel is therefore L2 request/scoreboard limited, not
launch-bound, DRAM-bound, or spill-bound. The raw local artifacts are
`artifacts/h200_lr_final_3aea27bd/profiles/h200_w3_m16_down_sol.csv` and
`artifacts/h200_lr_final_3aea27bd/profiles/h200_w3_m16_down_raw.csv`.

The pulled cooperative prototype was also captured with the NCU full set on
H200 at W2/M16/K2048/N8192. Both captures use the same payload and source head;
NCU duration is shown only for attribution, while the CUDA-event medians above
remain the latency gate.

| Metric | `41ffd8db` scalar production | Cooperative / `f7f29083` production | Delta |
|---|---:|---:|---:|
| executed instructions | 105,476,096 | 18,484,224 | -82.47% |
| NCU duration | 130.336 us | 58.464 us | -55.14% |
| grid / block | 1,024 / 256 | 256 / 128 | four N8 tiles per block |
| registers/thread | 59 | 96 | +37 |
| static shared/block | 18,960 B | 39,520 B | +20,560 B |
| achieved / theoretical occupancy | 42.46% / 50.00% | 12.14% / 31.25% | lower residency and tail utilization |
| issue active | 81.30% | 35.09% | consumer dependencies now exposed |
| shared-load bank conflicts | 1,173,341 | 3,154,223 | 2.69x |
| shared-store bank conflicts | 33,173 | 1,581,056 | 47.66x |

This validates the intended decode deduplication and changes the next target:
do not revisit scalar planar extraction. Preserve cooperative recurrence and
remove the N8-to-N16 padding, decoded-weight shared round trip, and conflicting
WMMA fragment loads. Machete's one-block-per-SM TMA/WGMMA pipeline is the
longer-term structure once the native N8 consumer is proven.

The packed-pair follow-up was captured on the same H200 W2/M16/K2048/N8192
case. It reduces the decoded-weight store cost but moves some conflict pressure
to WMMA fragment loads:

| Metric | `f7f29083` row-major pairs | `f7ae7daf` packed column-major pairs | Delta |
|---|---:|---:|---:|
| executed instructions | 18,484,224 | 18,549,760 | +0.35% |
| NCU duration | 58.464 us | 56.608 us | -3.18% |
| issue active | 35.09% | 36.66% | +1.57 points |
| shared-load bank conflicts | 3,154,223 | 4,226,889 | +34.00% |
| shared-store bank conflicts | 1,581,056 | 794,624 | -49.75% |
| shared-store wavefronts | 2,297,856 | 1,241,088 | -45.99% |

The net latency win is accepted, but the increased load conflicts rule out
further blind layout permutations. The next consumer change must be validated
against a dense native-N8 MMA microtest before replacing production WMMA.

The direct native-N8 consumer removes N16 padding, the WMMA B-fragment load,
and the accumulator reduction. The subsequent activation XOR swizzle targets
the native A `ldmatrix` addresses, then stride 40 breaks the remaining row
periodicity. The table compares exact gate/up W2 M16/K2048/N8192 H200 captures
at `1521292b`, `ac440de3`, and current pushed head `e944e528`:

| Metric | Direct native N8 | + XOR swizzle | + stride 40 | Stride delta |
|---|---:|---:|---:|---:|
| executed instructions | 14,197,760 | 14,480,384 | 14,445,568 | -0.24% |
| NCU duration | 39.136 us | 39.296 us | 36.576 us | -6.92% |
| registers/thread | 70 | 80 | 92 | +12 |
| static shared/block | 35,424 B | 35,424 B | 41,568 B | +6,144 B |
| achieved / theoretical occupancy | 12.10% / 37.50% | 12.09% / 37.50% | 12.10% / 31.25% | achieved unchanged |
| issue active | 41.05% | 40.74% | 40.84% | +0.10 points |
| shared-load bank conflicts | 2,553,359 | 1,459,994 | 935,560 | -35.92% |
| shared-store bank conflicts | 0 | 0 | 131,072 | new staging cost |
| shared-load wavefronts | 3,995,151 | 2,901,786 | 2,377,352 | -18.07% |
| total excessive shared wavefronts | 2,507,700 | 1,459,124 | 1,065,908 | -26.95% |
| `ldmatrix` load conflicts | 1,572,864 | 524,288 | not isolated | see total above |
| `ldmatrix` load wavefronts | 2,097,152 | 1,048,576 | not isolated | see total above |

The swizzle alone exchanges conflict stalls for address instructions and ten
registers, leaving NCU replay duration flat. Stride 40 then converts the
conflict reduction into a 6.9% instrumented latency win while adding another
12 registers and 6 KiB of shared memory. Theoretical occupancy falls one block
per SM, but achieved occupancy stays 12.10% because the 256-block grid remains
tail-limited. Padding also introduces 131k store conflicts, so the next layout
should preserve the load win while making the global-to-shared stage contiguous
again. Raw captures are
`/tmp/ncu_h200_w2_native_n8_packed_1521292b.csv` and
`/tmp/ncu_h200_w2_native_n8_swizzle_ac440de3.csv`, plus
`/tmp/ncu_h200_w2_native_n8_stride40_e944e528.csv` on this host.

## Coverage and targeting queue

| Priority | Device/rate/shape | Current state | Next evidence needed |
|---:|---|---|---|
| 1 | H200 W2/W2.5 M16 MLP | `e944e528`: stride 40 reaches 0.496-0.583x Machete W4; NCU reports 14.45M instructions, 0.94M load conflicts, and 0.13M new store conflicts | make staging contiguous/conflict-free, then overlap fetch/dequant/MMA with a Hopper TMA or staged pipeline and wider N tile |
| 2 | H200 W2/W2.5 M16 attention/KV | Q/O reaches 0.955-0.965x Machete; K/V reaches 1.254-1.256x; Q/O W2 regressed 0.48 us versus `ac440de3` | recover the Q/O W2 delta while enforcing K/V as the no-regression gate |
| 3 | H200 W3 M32/K8192/N2048, FP16/BF16 | 1.76x versus non-LR; split 4 is best; 16-tile batch rejected | reduce L2 request/scoreboard pressure without raising ROWS32 registers |
| 4 | RTX 5090 W2-W3.5, all substantial M=16 shapes | `e4b1006c`: all 16 pass >=4x | retain in expanded row/dtype coverage |
| 5 | all nine prior Ada/Blackwell GPUs, W2-W3.5 | `18389b4d`: all 144 substantial cells >=2x, 112 >=4x | retain as the prior-host acceptance gate |
| 6 | RTX 4090 W3.5 substantial shapes | 2.269-2.520x in the simultaneous run | target the 4x stretch goal |
| 7 | RTX 4090/5090, M=1/4/8/32 and BF16 | earlier full matrix exists at `bd625c15`, not yet repeated for PR #60 source | expanded accuracy/performance regression |
| 8 | A100 W2-W3.5, all M/K/N cells | no A100 installed; SM count unknown | query properties once by device ordinal, then run the same matrix |
| 9 | launch-bound narrow shapes | 0.72-1.04x on Hopper M16 | reduce launch/split overhead without regressing substantial shapes |
| 10 | H200 W3, FP16/BF16 M1/M4/M8/M16 | accepted TB6 path remains compile-time unchanged at `ac440de3`; explicit W3 regression passes | retain as a rate-specific no-regression gate while optimizing TB4/TB5 |

## Reproduction

Current H200 sweep:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=GPU-0c667065-5c47-38ce-0b0a-d211392ce9ea \
GPTQMODEL_QVQ_CUDA_BUILD_ROOT=/tmp/qvq-jit-hopper-current \
python scripts/benchmark_qvq_cuda_lr.py \
  --physical-gpu 0 --bits 2 2.5 3 3.5 --dtype float16 --m 16 \
  --warmup 10 --iterations 60 --out-dir artifacts/<stamp>
```

Prior RTX 5090 representative sweep:

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
