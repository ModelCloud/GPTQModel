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
| `e0c29e60` | H200, W2-W3.5 M1/M2/M4/M8/M16 | Full four-rate, four-K/N projection matrix against matched Machete/Marlin W4 | 120/120 rows passed their dense-reference gates; QVQ/Machete geomeans were 0.787x, 0.790x, 0.241x, and 0.197x from W2 through W3.5 | accepted full-M/K/N targeting baseline |
| `54e2e3ac` / `efe8ae70` | H200, W3 M1-M16 | Permit the existing cooperative Hopper TB6 kernel for every M<=16 instead of only the ROWS16 dispatch | All 20 W3 cells accurate; Q/O M1-M8 gained 2.74-2.99x and MLP M1-M8 gained 3.51-4.37x versus `e0c29e60`; K/V and M16 stayed within normal variance | accepted, merged, and pushed |
| `f157104d` | H200, W3 M1-M16 | Feed TB6 decoded half2 pairs to the same direct native-N8 MMA consumer used by W2/W2.5 | All 20 cells accurate; Q/O gained 1.22-1.25x and MLP gained 1.38-1.40x versus `54e2e3ac`; exact-head NCU fell to 17.38M instructions and 33.0 us | accepted and pushed |
| `e0f55ecb` / `e55997ce` | H200, W3 M1-M16 | Exchange one previous four-edge pack per lane instead of six overlapping per-edge TB6 shuffles | All 20 cells accurate; Q/O gained 3.3-5.1%, gate/up 4.1-4.7%, and down 4.6-6.3%; K/V scalar controls stayed within 0.8% | accepted, merged, and pushed |
| `a12c1e35` | H200, W3 M1-M16 | Expand four TB6 low nibbles and high dibits with two broadword mask/shift stages instead of four scalar extracts | All 20 cells accurate; Q/O gained up to 3.5%, gate/up 2.1-3.0%, and down 2.9-4.9%; NCU fell to 16.14M instructions and 31.0 us | accepted and pushed |
| `1671e6a5` / `009533a6` | H200, W3 M1-M16 gate/up | Widen the native-N8 output block from four to eight warps only for K2,048/N8,192; compact the otherwise dead padded shared storage in that specialization | All five gate/up rows gained 1.9-3.3% at 0.03184-0.03254 ms; merged-head NCU fell to 14.53M instructions with zero spilling | accepted, merged, and pushed |
| `65743ae8` | H200, W3/W3.5 | Repair the concurrent W3.5 native-N8 merge by supplying its four-tile launcher argument | Full 20-cell W3 matrix remained accurate; W3.5 M16/K2,048/N8,192 passed at 0.0392 ms with 6.29e-05 max error | accepted integration fix and pushed |
| `65743ae8` uncommitted A/B | H200, W3 gate/up | Machete-inspired two-stage `cp.async`: split the N64 batch into alternating eight-tile buffers and issue batch i+1 before decoding/MMA of batch i | Accurate, but gate/up regressed from 0.0318-0.0325 ms to 0.0593-0.0641 ms; extra synchronization/bookkeeping could not be amortized by each short split-K partition | rejected; source restored before next experiment |
| `34aac964` uncommitted A/B | H200, W3 gate/up | Compose the earlier XOR activation-segment swizzle with the accepted stride-40 native-N8 layout | Accurate but 3.2-3.6% slower; NCU shared-load conflicts rose from 0.936M to 1.985M and memory throughput fell from 210.8 to 200.7 GB/s | rejected; source restored before next experiment |
| `548c5c73` | H200, W3 gate/up | Bypass the random shared level-table lookup in the N64 specialization and use Hopper's read-only/L1 path | All five M1-M16 rows gained 0.3-0.9%; shared-load bank conflicts fell 936,250 to 483 and L1 hit rate rose 11.3% to 93.9%; 108 CUDA tests passed | accepted and pushed |
| `7788a8f6` uncommitted A/B | H200, W3 gate/up | Replace the decoded-B shared transpose and two warp fences with direct register shuffles into the MMA B-fragment layout | Accurate, but 16 scalar routes per K32 tile raised M1-M16 from 0.0316-0.0324 to 0.0360-0.0368 ms (12-14%) | rejected; source restored before next experiment |
| `8159988f` uncommitted A/B | H200, W3 gate/up | Widen N64 to N128; reduce K batch 16 to 14 to fit Hopper's 48 KiB static-shared limit; sweep split 2/4 | N128 split-2 was accurate but 4-6% slower at 0.0333-0.0340 ms; split-4 was mixed across M and did not recover N64 | rejected; source restored before next experiment |
| `f24b2227` uncommitted A/B | H200, W3 gate/up | Reuse the merged W3.5-M1 producer-lane permutation for W3 N64 while preserving the stride-40 shared layout and `ldmatrix` addresses | Accurate, but scattered global activation fetches raised M1-M16 from 0.0316-0.0324 to 0.0322-0.0335 ms (1.6-3.3%) | rejected; source restored before next experiment |
| `1e15b299` uncommitted A/B | H200, W3 gate/up | Sweep split 1/2/4/8 after the read-only level-table change, then set automatic split-1 for an exact CUDA Graph replay | The direct-event diagnostic favored split-1, but the production contract regressed 17-20% to 0.0378-0.0390 ms; split-2's second CTA wave remains necessary | rejected; source restored before next experiment |
| `11f7f40b` | H200, W3 K/V | Lower the Hopper cooperative dispatch threshold from N2,048 to N512 for every rate, reusing the native-N8 TB6 kernel | K/V M1-M16 fell from 0.0207-0.0244 to 0.01142-0.01165 ms (1.81-2.09x) and reached 1.246-1.306x Machete; 108 CUDA tests passed | accepted and pushed |

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

### H200 full W2-W3.5 M/K/N matrix versus Machete

Pushed head `e0c29e60`, source fingerprint
`f911d8d851dd21a01e813c7d789ffaa8e9e86a89a98b6c8b9d410b962e50f05b`,
was measured on H200 GPU 0 at M1/M2/M4/M8/M16 across all four distinct Llama
3.2 1B K/N geometries. Every one of the 80 QVQ rows passed the `2e-3` dense-
reference gate; all 40 matched W4 rows also passed their reference gate.
`xMachete` is Machete W4 latency divided by QVQ LR latency.

| W | QVQ rows | xMachete geomean | Min-max | Rows >= Machete | Worst max abs |
|---:|---:|---:|---:|---:|---:|
| 2 | 20 | 0.787x | 0.491-1.289x | 8/20 | 1.01e-4 |
| 2.5 | 20 | 0.790x | 0.502-1.266x | 7/20 | 1.13e-4 |
| 3 | 20 | 0.241x | 0.085-0.716x | 0/20 | 1.22e-4 |
| 3.5 | 20 | 0.197x | 0.074-0.718x | 0/20 | 1.72e-5 |

| W | M1 geo | M2 geo | M4 geo | M8 geo | M16 geo |
|---:|---:|---:|---:|---:|---:|
| 2 | 0.832x | 0.787x | 0.775x | 0.774x | 0.770x |
| 2.5 | 0.825x | 0.789x | 0.783x | 0.780x | 0.774x |
| 3 | 0.201x | 0.205x | 0.202x | 0.199x | 0.493x |
| 3.5 | 0.181x | 0.208x | 0.205x | 0.203x | 0.192x |

Geomeans by K/N geometry across M expose the rate-specific priorities:

| W | Q/O 2048x2048 | K/V 2048x512 | Gate/up 2048x8192 | Down 8192x2048 |
|---:|---:|---:|---:|---:|
| 2 | 1.010x | 1.260x | 0.499x | 0.606x |
| 2.5 | 1.002x | 1.255x | 0.510x | 0.608x |
| 3 | 0.322x | 0.679x | 0.114x | 0.136x |
| 3.5 | 0.258x | 0.665x | 0.085x | 0.104x |

The complete per-row latency, P95, payload rate, xMarlin, xMachete, and error
record is in
`artifacts/h200_lr_next4x/full_w2_w35_mkn_e0c29e60/gpu0.json` and its Markdown
rendering. The next high-impact path is W3.5 MLP at every M, followed by W3
M1-M8 MLP. W3 M16 already dispatches the specialized Hopper cooperative path
and is 3-4x faster than the scalar lower-M path, but still only 0.361-0.374x
Machete on MLP.

### H200 W3 small-row cooperative dispatch

Source commit `54e2e3ac`, merged/pushed as `efe8ae70`, removes the host-side
restriction that sent W3 M1-M8 to the scalar LR kernel. The measured source
fingerprint was
`ddc8cabd3ff4e75e0b2a3ca481fbb1ca7dc657011fb57dc6f51b3990445845a8`.
These H200 GPU 0 results use 10 warm-ups and 60 graph-timed launches. `Prior`
is the exact full-matrix result above. K/V remains scalar because N=512 is
below the cooperative TB6 threshold.

| Shape | M | K | N | LR ms | Prior ms | LR gain | xMachete W4 | Max abs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `attn_qo` | 1 | 2,048 | 2,048 | 0.019808 | 0.059200 | 2.989x | 0.767x | 8.58e-06 |
| `attn_qo` | 2 | 2,048 | 2,048 | 0.021296 | 0.058304 | 2.738x | 0.723x | 1.24e-05 |
| `attn_qo` | 4 | 2,048 | 2,048 | 0.021248 | 0.059104 | 2.782x | 0.718x | 1.34e-05 |
| `attn_qo` | 8 | 2,048 | 2,048 | 0.021376 | 0.060320 | 2.822x | 0.722x | 1.53e-05 |
| `attn_qo` | 16 | 2,048 | 2,048 | 0.021680 | 0.022016 | 1.015x | 0.708x | 1.34e-05 |
| `attn_kv` | 1 | 2,048 | 512 | 0.021312 | 0.021808 | 1.023x | 0.664x | 1.91e-06 |
| `attn_kv` | 2 | 2,048 | 512 | 0.020464 | 0.021024 | 1.027x | 0.714x | 3.81e-06 |
| `attn_kv` | 4 | 2,048 | 512 | 0.020768 | 0.021488 | 1.035x | 0.700x | 4.77e-06 |
| `attn_kv` | 8 | 2,048 | 512 | 0.021824 | 0.022192 | 1.017x | 0.650x | 4.77e-06 |
| `attn_kv` | 16 | 2,048 | 512 | 0.024320 | 0.024448 | 1.005x | 0.599x | 3.81e-06 |
| `mlp_gate_up` | 1 | 2,048 | 8,192 | 0.048512 | 0.211952 | 4.369x | 0.367x | 2.48e-05 |
| `mlp_gate_up` | 2 | 2,048 | 8,192 | 0.048624 | 0.207920 | 4.276x | 0.366x | 2.67e-05 |
| `mlp_gate_up` | 4 | 2,048 | 8,192 | 0.048832 | 0.208960 | 4.279x | 0.370x | 2.57e-05 |
| `mlp_gate_up` | 8 | 2,048 | 8,192 | 0.049120 | 0.211488 | 4.306x | 0.364x | 3.43e-05 |
| `mlp_gate_up` | 16 | 2,048 | 8,192 | 0.049600 | 0.049696 | 1.002x | 0.365x | 3.05e-05 |
| `mlp_down` | 1 | 8,192 | 2,048 | 0.048512 | 0.208224 | 4.292x | 0.466x | 4.58e-05 |
| `mlp_down` | 2 | 8,192 | 2,048 | 0.059296 | 0.208032 | 3.508x | 0.372x | 8.58e-05 |
| `mlp_down` | 4 | 8,192 | 2,048 | 0.059392 | 0.209232 | 3.523x | 0.379x | 8.58e-05 |
| `mlp_down` | 8 | 8,192 | 2,048 | 0.059504 | 0.212528 | 3.572x | 0.375x | 9.35e-05 |
| `mlp_down` | 16 | 8,192 | 2,048 | 0.059872 | 0.059280 | 0.990x | 0.375x | 1.22e-04 |

NCU explains the gain rather than merely correlating with it: W3 M4
K2048/N8192 scalar executes 74.15M instructions in 201.6 us, while the
cooperative M16 implementation executes 19.68M in 46.5 us. Small M can reuse
that same zero-padded M16 arithmetic, avoiding the redundant TB6 decode and
cutting the measured MLP latency by up to 4.37x.

Commit `f157104d` then makes native-N8 MMA rate-independent for every
cooperative TB4/TB5/TB6 specialization. This removes TB6's eight unused N16
columns, the padded shared B-fragment load, and the shared accumulator store.
All cells below passed the dense-reference gate; K/V is included as an
unchanged scalar control.

| Shape | M | K | N | LR ms | Prior ms | LR gain | xMachete W4 | Max abs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `attn_qo` | 1 | 2,048 | 2,048 | 0.016192 | 0.019808 | 1.223x | 0.948x | 8.58e-06 |
| `attn_qo` | 2 | 2,048 | 2,048 | 0.017008 | 0.021296 | 1.252x | 0.929x | 1.24e-05 |
| `attn_qo` | 4 | 2,048 | 2,048 | 0.017136 | 0.021248 | 1.240x | 0.902x | 1.34e-05 |
| `attn_qo` | 8 | 2,048 | 2,048 | 0.017152 | 0.021376 | 1.246x | 0.920x | 1.53e-05 |
| `attn_qo` | 16 | 2,048 | 2,048 | 0.017312 | 0.021680 | 1.252x | 0.891x | 1.34e-05 |
| `attn_kv` | 1 | 2,048 | 512 | 0.021392 | 0.021312 | 0.996x | 0.698x | 1.91e-06 |
| `attn_kv` | 2 | 2,048 | 512 | 0.020720 | 0.020464 | 0.988x | 0.713x | 3.81e-06 |
| `attn_kv` | 4 | 2,048 | 512 | 0.021072 | 0.020768 | 0.986x | 0.705x | 4.77e-06 |
| `attn_kv` | 8 | 2,048 | 512 | 0.022080 | 0.021824 | 0.988x | 0.668x | 4.77e-06 |
| `attn_kv` | 16 | 2,048 | 512 | 0.024592 | 0.024320 | 0.989x | 0.602x | 3.81e-06 |
| `mlp_gate_up` | 1 | 2,048 | 8,192 | 0.034992 | 0.048512 | 1.386x | 0.513x | 2.48e-05 |
| `mlp_gate_up` | 2 | 2,048 | 8,192 | 0.035040 | 0.048624 | 1.388x | 0.511x | 2.67e-05 |
| `mlp_gate_up` | 4 | 2,048 | 8,192 | 0.035120 | 0.048832 | 1.390x | 0.512x | 2.57e-05 |
| `mlp_gate_up` | 8 | 2,048 | 8,192 | 0.035456 | 0.049120 | 1.385x | 0.504x | 3.43e-05 |
| `mlp_gate_up` | 16 | 2,048 | 8,192 | 0.035936 | 0.049600 | 1.380x | 0.501x | 3.05e-05 |
| `mlp_down` | 1 | 8,192 | 2,048 | 0.035056 | 0.048512 | 1.384x | 0.631x | 4.58e-05 |
| `mlp_down` | 2 | 8,192 | 2,048 | 0.042592 | 0.059296 | 1.392x | 0.513x | 8.58e-05 |
| `mlp_down` | 4 | 8,192 | 2,048 | 0.042752 | 0.059392 | 1.389x | 0.519x | 8.58e-05 |
| `mlp_down` | 8 | 8,192 | 2,048 | 0.042736 | 0.059504 | 1.392x | 0.517x | 9.35e-05 |
| `mlp_down` | 16 | 8,192 | 2,048 | 0.042928 | 0.059872 | 1.395x | 0.517x | 1.22e-04 |

On the exact pushed binary at M16/K2048/N8192, W2 executes 14.44M
instructions and W3 executes 17.38M, only 20.3% more after sharing the native
consumer. Both use 92 registers/thread and 31.25% theoretical occupancy. The
remaining rate gap is therefore the TB6 two-plane extraction and its repeated
per-edge shuffles, not tensor compute or accumulator materialization.

Commit `e0f55ecb`, merged/pushed as `e55997ce`, replaces those six overlapping
TB6 shuffles with one exchange of the previous four-edge pack. The same packed
recurrence is now shared by W2, W2.5, and W3; only TB6's two-plane extraction
remains rate-specific.

| Shape | M | K | N | LR ms | Prior ms | LR gain | xMachete W4 | Max abs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `attn_qo` | 1 | 2,048 | 2,048 | 0.015408 | 0.016192 | 1.051x | 1.001x | 8.58e-06 |
| `attn_qo` | 2 | 2,048 | 2,048 | 0.016320 | 0.017008 | 1.042x | 0.966x | 1.24e-05 |
| `attn_qo` | 4 | 2,048 | 2,048 | 0.016480 | 0.017136 | 1.040x | 0.938x | 1.34e-05 |
| `attn_qo` | 8 | 2,048 | 2,048 | 0.016544 | 0.017152 | 1.037x | 0.952x | 1.53e-05 |
| `attn_qo` | 16 | 2,048 | 2,048 | 0.016752 | 0.017312 | 1.033x | 0.926x | 1.34e-05 |
| `attn_kv` | 1 | 2,048 | 512 | 0.021520 | 0.021392 | 0.994x | 0.684x | 1.91e-06 |
| `attn_kv` | 2 | 2,048 | 512 | 0.020720 | 0.020720 | 1.000x | 0.712x | 3.81e-06 |
| `attn_kv` | 4 | 2,048 | 512 | 0.021248 | 0.021072 | 0.992x | 0.694x | 4.77e-06 |
| `attn_kv` | 8 | 2,048 | 512 | 0.021984 | 0.022080 | 1.004x | 0.672x | 4.77e-06 |
| `attn_kv` | 16 | 2,048 | 512 | 0.024544 | 0.024592 | 1.002x | 0.606x | 3.81e-06 |
| `mlp_gate_up` | 1 | 2,048 | 8,192 | 0.033424 | 0.034992 | 1.047x | 0.535x | 2.48e-05 |
| `mlp_gate_up` | 2 | 2,048 | 8,192 | 0.033648 | 0.035040 | 1.041x | 0.532x | 2.67e-05 |
| `mlp_gate_up` | 4 | 2,048 | 8,192 | 0.033712 | 0.035120 | 1.042x | 0.531x | 2.57e-05 |
| `mlp_gate_up` | 8 | 2,048 | 8,192 | 0.033936 | 0.035456 | 1.045x | 0.527x | 3.43e-05 |
| `mlp_gate_up` | 16 | 2,048 | 8,192 | 0.034528 | 0.035936 | 1.041x | 0.517x | 3.05e-05 |
| `mlp_down` | 1 | 8,192 | 2,048 | 0.033520 | 0.035056 | 1.046x | 0.660x | 4.58e-05 |
| `mlp_down` | 2 | 8,192 | 2,048 | 0.040224 | 0.042592 | 1.059x | 0.554x | 8.58e-05 |
| `mlp_down` | 4 | 8,192 | 2,048 | 0.040224 | 0.042752 | 1.063x | 0.551x | 8.58e-05 |
| `mlp_down` | 8 | 8,192 | 2,048 | 0.040320 | 0.042736 | 1.060x | 0.558x | 9.35e-05 |
| `mlp_down` | 16 | 8,192 | 2,048 | 0.040480 | 0.042928 | 1.060x | 0.547x | 1.22e-04 |

Commit `a12c1e35` fuses each four-edge TB6 planar expansion. Exact per-cell
results remain in the preserved benchmark JSON; this compact table identifies
the M/K/N ranges and progression without repeating the unchanged K/V controls.

| Shape | M values | K | N | LR ms range | Gain versus `e0f55ecb` | xMachete range | Worst max abs |
|---|---|---:|---:|---:|---:|---:|---:|
| `attn_qo` | 1,2,4,8,16 | 2,048 | 2,048 | 0.015264-0.016224 | 1.009-1.035x | 0.963-1.020x | 1.53e-05 |
| `attn_kv` | 1,2,4,8,16 | 2,048 | 512 | 0.020624-0.024464 | 0.994-1.008x | 0.602-0.714x | 4.77e-06 |
| `mlp_gate_up` | 1,2,4,8,16 | 2,048 | 8,192 | 0.032704-0.033760 | 1.021-1.030x | 0.536-0.547x | 3.43e-05 |
| `mlp_down` | 1,2,4,8,16 | 8,192 | 2,048 | 0.032576-0.038928 | 1.029-1.049x | 0.570-0.678x | 1.22e-04 |

On the same M16/K2048/N8192 counter gate, broadword expansion reduces W3
from 16.45M to 16.14M instructions and NCU duration from 32.0 to 31.0 us.
Registers return from 89 to 92/thread, so further scalar decode algebra has
diminishing returns; wider N reuse is now the higher-impact W3 target.

Commit `1671e6a5`, merged with the concurrent Hopper work at `009533a6`,
groups eight native N8 output tiles in one W3 gate/up block. The specialization
halves the block grid while preserving the same total thread count, reuses each
activation stage across N64, and compacts storage that the native-N8 consumer
does not address. It is guarded to `K <= 2,048 && N >= 8,192`; the existing
four-warp path remains selected for Q/O and down projection geometries.

| Shape | M values | K | N | LR ms range | Gain versus `a12c1e35` | Worst max abs |
|---|---|---:|---:|---:|---:|---:|
| `mlp_gate_up` | 1,2,4,8,16 | 2,048 | 8,192 | 0.031840-0.032544 | 1.019-1.033x | 3.43e-05 |

The exact merged-head NCU capture at W3 M16/K2048/N8192 reports 14.53M
executed instructions, 30.59 us, 64 registers/thread, 37.50 KiB static shared
memory, and zero local spilling. The earlier N32 broadword kernel used 16.14M
instructions, so output reuse removes another 10.0% of executed work and makes
W3's instruction count essentially equal to W2's 14.44M. Remaining evidence is
45.02% cycles with no eligible warp and approximately 1.00M excessive shared
wavefronts; DRAM throughput is only 4.38%.

An alternating two-stage eight-tile `cp.async` A/B then tested the direct
Machete pipeline lesson without increasing staged shared capacity. It was
accurate but nearly doubled gate/up latency to 0.0593-0.0641 ms. Unlike
Machete's persistent producer/consumer WGMMA mainloop, this warp-synchronous
MMA kernel paid an extra block barrier per eight K32 tiles, while each split-K
partition held only 32 tiles. That experiment is rejected; future overlap work
must avoid increasing block-wide synchronization frequency.

Source-correlated NCU showed that the accepted stride-40 `ldmatrix.x4` A load
was already conflict-free. Eight divergent `LDS.U16` level-table lookups caused
approximately 934k of the remaining 936k shared-load conflicts. Commit
`548c5c73` instead reads that immutable 512-byte table through Hopper's
read-only/L1 path for W3 N64 only and removes its two now-unneeded setup
barriers. Across M1/M2/M4/M8/M16, gate/up medians become
0.031600/0.031792/0.031792/0.032032/0.032416 ms, a consistent 0.3-0.9% gain
from the merged N64 baseline with worst max error 3.43e-05.

Commit `11f7f40b` removes the obsolete N2,048 minimum for Hopper's TB6/TB7
cooperative path. W3 K/V at K2,048/N512 now uses the same native-N8 tensor-core
consumer instead of the scalar LR kernel. Exact H200 M1/M2/M4/M8/M16 medians
are 0.011424/0.011440/0.011504/0.011632/0.011648 ms, respectively, with worst
max error 1.05e-05. This is 1.81-2.09x faster than the preceding scalar rows
and 1.246-1.306x the matched Machete W4 throughput. The full W3 matrix remained
accuracy-clean and the exact-head CUDA suite passed 108/108 cases.

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

The accepted W3 N64 gate/up specialization was profiled at pushed head
`65743ae8` on physical H200 GPU 0. It executes 14,526,464 instructions in an
NCU-replayed 30.59 us, uses 64 registers/thread and 37.50 KiB shared/block, and
has no local spilling. Its 256 total split-K blocks form 0.48 waves/SM; achieved
active warps are 14.61/SM, issue availability is 54.98%, and 936,250 shared-load
plus 65,536 shared-store bank conflicts remain. The local report is
`/tmp/ncu-h200-w3-gate-out8-65743ae8.ncu-rep`; CUDA-event medians above remain
the acceptance timing.

The rejected stride-40 plus XOR composition kept 64 registers/thread and the
same shared allocation, but raised shared-load conflicts from 936,250 to
1,984,923 (2.12x), increased NCU duration from 30.59 to 32.13 us, reduced
memory throughput from 210.75 to 200.67 GB/s, and raised no-eligible cycles
from 45.02% to 48.10%. This proves the two permutations cannot be composed
blindly: the next activation layout must be solved against the native
`ldmatrix.x4` lane-to-bank mapping as one transform.

The accepted read-only level-table capture at `548c5c73` changes shared-load
conflicts from 936,250 to 483 (-99.95%), shared-load wavefronts from 2,443,578
to 983,523 (-59.75%), and L1 hit rate from 11.31% to 93.94%. It deliberately
executes slightly more work (14.53M to 14.86M instructions), yet NCU duration
falls 30.592 to 30.400 us, issue availability rises 54.98% to 56.46%, and
measured memory throughput rises 210.75 to 212.10 GB/s. Registers remain 64,
static shared memory falls 37.50 to 36.99 KiB, and no local spilling appears.
The report is `/tmp/ncu-h200-w3-global-levels-09ab284a.ncu-rep`. This is the
desired Machete-style trade: spend cheap cache-resident instructions to remove
serialized shared-memory traffic.

## Coverage and targeting queue

| Priority | Device/rate/shape | Current state | Next evidence needed |
|---:|---|---|---|
| 1 | H200 W3 M1-M16 MLP | gate/up N64 is 0.0316-0.0324 ms; random level-table conflicts are eliminated, but only 212 GB/s is measured and down remains 0.0327-0.0390 ms | widen/repack the remaining aligned weight fetch and move toward persistent producer/consumer WGMMA without adding per-eight-tile block barriers |
| 2 | H200 W3.5 M1-M16 MLP | concurrent native-N8 path now reaches 0.0392 ms for the H200 M16 gate canary | run the full H200 W3.5 matrix from the merged head after the current W3 focus and profile its remaining TB7 arithmetic |
| 3 | H200 W2/W2.5 M1-M16 MLP | 0.499-0.608x Machete geomean; stride 40 reports 14.45M instructions and 1.07M total excessive shared wavefronts | preserve native N8; use a truly overlapped TMA/async design or wider N tile, not immediate `cp.async` wait |
| 4 | H200 W2/W2.5 attention/KV | Q/O is 1.002-1.010x and K/V 1.255-1.260x Machete across M | enforce as the rate-specific latency/no-regression gate |
| 5 | H200 W3 M16 and M32 | M16 is 0.361-0.709x Machete by K/N; prior M32 K8192/N2048 is 1.76x versus non-LR | reduce TB6 cache/scoreboard pressure; retain split 4 for the M32 down-like case |
| 6 | RTX 5090 W2-W3.5, all substantial M=16 shapes | historical `e4b1006c`: all 16 pass >=4x | retain as historical coverage; current local work remains H200-only |
| 7 | all nine prior Ada/Blackwell GPUs, W2-W3.5 | historical `18389b4d`: all 144 substantial cells >=2x, 112 >=4x | retain as the prior-host acceptance gate |
| 8 | A100 W2-W3.5, all M/K/N cells | no A100 installed; SM count unknown | pending hardware; do not infer from Hopper results |
| 9 | launch-bound narrow shapes | W2/W2.5 attention is already at or above Machete across M | avoid trading these wins for MLP throughput |

## Reproduction

Current H200 sweep:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=GPU-0c667065-5c47-38ce-0b0a-d211392ce9ea \
GPTQMODEL_QVQ_CUDA_BUILD_ROOT=/tmp/qvq-jit-hopper-current \
python scripts/benchmark_qvq_lr_vs_gptq_llama32_1b.py \
  --physical-gpu 0 --shapes attn_qo attn_kv mlp_gate_up mlp_down \
  --m 1 2 4 8 16 --qvq-bits 2 2.5 3 3.5 --dtype float16 \
  --warmup 10 --iterations 60 --output artifacts/<stamp>/gpu0.json
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
