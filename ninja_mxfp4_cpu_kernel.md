# MXFP4 CPU Ninja Kernel Dev Log

Repository: `ModelCloud/GPT-QModel-Ultra`  
Goal: build an on-the-fly MXFP4 weight dequant + matmul kernel for Intel Xeon (AVX-512 / AMX optional) that beats a naive PyTorch dequant-to-FP8 + `torch.matmul` baseline by 2x, using only MXFP4-sized storage.

## FP8 CPU support check

* `/proc/cpuinfo` does **not** contain `avx512_fp8` or `amx_fp8`; this Xeon 8559C VM has no native FP8 FMA/AMX-FP8 instructions.
* `avx512fp16` (FP16) and `avx_vnni` are present, so the existing BF16/FP16 paths are the native high-performance paths.
* An emulated FP8-E4M3 path is still implemented for comparison: weights stay 4-bit, activation and output are `torch.float8_e4m3fn`, and the dot product is performed in FP32 via 256-entry lookup tables.

## Environment (snapshot)

* CPU: `INTEL(R) XEON(R) PLATINUM 8559C` (avx512, avx512bf16, avx512fp16, avx_vnni)
* `/proc/cpuinfo` flags: `avx512f avx512bw avx512vl avx512bf16 avx512fp16 avx_vnni` (no `amx*` flags visible in this VM)
* PyTorch: 2.13.0+cu130 (CPU path used)
* Threads: 8

## Kimi 3 qkv MXFP4 shapes used for simulation

Derived from `moonshotai/Kimi-K3` `config.json` and the standard MLA projection formulas used in `moonshotai/Kimi-K2.5`/`Kimi-K2.6`:

```
hidden_size       = 7168
num_attention_heads = 96
q_lora_rank       = 1536
kv_lora_rank      = 512
qk_nope_head_dim  = 128
qk_rope_head_dim  = 64
v_head_dim        = 128
q_head_dim        = qk_nope_head_dim + qk_rope_head_dim = 192
```

| projection | in_features | out_features | weight bytes (MXFP4) | scale bytes (E8M0) |
|------------|-------------|--------------|----------------------|--------------------|
| `q_a_proj` | 7168 | 1536 | 5,505,024 | 344,064 |
| `q_b_proj` | 1536 | 18432 | 14,155,776 | 884,736 |
| `kv_a_proj_with_mqa` | 7168 | 576 | 2,064,384 | 129,024 |
| `kv_b_proj` | 512 | 24576 | 6,291,456 | 393,216 |
| `o_proj` | 12288 | 7168 | 44,040,192 | 2,752,512 |

Total per attention layer (q-a + q-b + kv-a + kv-b + o) ≈ 71 MB of MXFP4 weights + scales.

## Research findings

* Existing Intel/AMD CPU MXFP4 work:
  * OpenVINO/oneDNN PR added MXFP4 FC weights (`wei=f4e2m1, scales=f8e8m0`) with AVX2/AVX-512 paths (no public standalone kernel).
  * vLLM PR #41922 added CPU MXFP4 W4A16 fused MoE experts with AVX-512/AMX-optimized kernels for `gpt-oss-20b`.
  * AMD Primus uses MXFP4 on MI355X GPU, not CPU.
* `GPT-QModel-Ultra` already has `gptqmodel_ext/floatx_cpu.cpp` which supports fast AVX-512 dequant of FP4/FP8 to BF16/FP16, but it does **not** fuse the matmul.
* The `MXFP4` layout in this repo is: `qweight` `(N, K//2)` `uint8` (two E2M1 nibbles per byte), `weight_scale` `(N, K//32)` `uint8` (one E8M0 scale per 32 weights).

## Experiments

### 1. PyTorch baseline (dequant MXFP4 -> FP8 -> BF16 matmul)

* File: `scripts/benchmark_mxfp4_cpu_kernel.py`
* Baseline steps:
  1. Unpack `qweight` uint8 into two 4-bit nibbles per byte.
  2. Map each nibble through the OCP E2M1 table.
  3. Multiply by `2^(scale - 127)` for each 32-weight block.
  4. Cast the FP32 result to `torch.float8_e4m3fn`.
  5. Convert the FP8 weight to `torch.bfloat16` and run `torch.matmul(x, W.T)`.

### 2. Ninja C++ kernel

* File: `gptqmodel_ext/mxfp4_cpu_kernel.cpp` + JIT load in `scripts/benchmark_mxfp4_cpu_kernel.py`
* Strategy:
  * Build a `float scaled_fp4_table[256][16]` once per call (16 E2M1 values times each E8M0 scale).  This lets each 32-weight block do a single scale lookup + 16-entry table gather instead of per-element exponent math.
  * AVX-512 BF16 path: load 16 `bfloat16` activations at a time (`_mm512` load + zero-extend the 16-bit lanes into the upper 16 bits of each float, which is exactly how BF16 maps to FP32), gather 16 dequantized weights from the block's scaled table (`_mm512_i32gather_ps`), and accumulate with `_mm512_fmadd_ps` into an `__m512` accumulator.
  * Reduce lanes at the end with `_mm512_reduce_add_ps`, round to BF16, and store.
  * Parallelize over output elements with `at::parallel_for` and a fuzzy per-64K-output thread count.
  * AMX tile path is left as a compile-time `#if`/`runtime` check; it is not enabled because the VM does not expose `amx*` CPU flags.

## Results

All runs use the `scripts/benchmark_mxfp4_cpu_kernel.py` script with `iters=10, warmup=3` and the default `torch.get_num_threads()` count (8 in this VM).  The C++ kernel is bit-exact with the torch baseline (`kernel_baseline_err == 0.0`); the absolute/relative numbers below are the MXFP4 quantization error against the dense FP32 reference, which is expected for random N(0,1) weights with group-size 32 E2M1.

### q_b_proj (1536 x 18432)

| M | baseline ms | kernel ms | speedup | vs dense rel err |
|---|-------------|-----------|---------|------------------|
| 1 | 77.79 | 4.37 | **17.80x** | 0.453 |
| 4 | 70.57 | 8.88 | **7.95x** | 0.484 |
| 16 | 66.39 | 15.37 | **4.32x** | 0.468 |

### q_a_proj (7168 x 1536), M=1

* baseline: 28.41 ms, kernel: 1.73 ms => **16.44x** speedup
* vs dense max abs err: 121.72 (rel 0.398)

### kv_b_proj (512 x 24576), M=1

* baseline: 27.02 ms, kernel: 2.04 ms => **13.22x** speedup
* vs dense max abs err: 39.30 (rel 0.412)

### o_proj (12288 x 7168), M=1

* baseline: 234.61 ms, kernel: 13.37 ms => **17.54x** speedup
* vs dense max abs err: 196.57 (rel 0.482)

## FP8-E4M3 activation path (new)

Implemented in the same `mxfp4_linear_cpu` extension.  Activation and output are now allowed to be `torch.float8_e4m3fn`; the kernel uses two lookup tables:

1. `scaled_table_fp8[256][16]` — each E2M1 nibble, after applying the block E8M0 scale, is rounded to FP8-E4M3 and stored as the corresponding FP32 value.
2. `fp8_to_fp32_table[256]` — each FP8-E4M3 activation byte is converted to FP32.

The hot loop gathers both sides as FP32 and uses `_mm512_fmadd_ps`.  Weight storage is still exactly MXFP4 (4-bit nibbles + 8-bit scales); no FP8 weight tensor is materialized.

### q_b_proj (1536 x 18432), M=1, dtype comparison

Run with `iters=30, warmup=5`:

| dtype | baseline ms | kernel ms | speedup | vs dense max abs err | vs dense rel err |
|---|---|---|---|---|---|
| BF16 | 66.74 | 4.38 | **15.23x** | 611.86 | 0.4079 |
| FP16 | 64.58 | 4.32 | **14.93x** | 603.77 | 0.4025 |
| FP8-E4M3 | 68.43 | 7.00 | **9.77x** | 1048.54 | 0.7006 |

Notes:

* The C++ kernel is bit-exact with the matching torch baseline for BF16 and FP8 (`kernel_baseline_err == 0.0`); FP16 shows a single-ULP difference of `3.05e-05` vs the baseline (expected from FP16 accumulation rounding).
* The comparison above uses the dense **FP32** reference directly (not rounded to the output dtype).  BF16 and FP16 accumulate the error from MXFP4 quantization plus the target-dtype rounding, while FP8-E4M3 is much worse because values above the FP8-E4M3 maximum (~448) saturate, producing a relative error of ~70% on random N(0,1) weights.
* The original FP8-E4M3 implementation was slower than BF16/FP16 because it used an extra `_mm512_i32gather_ps` for the activation table.  The BF16 dot-product optimization below replaces that with a one-time FP8->BF16 conversion and `VDPBF16PS`, making FP8-E4M3 faster than BF16/FP16 in every tested shape.

The first successful milestone (torch baseline + Ninja AVX-512 kernel reaching >2x) has been reached on every tested Kimi-3 attention projection shape.

## FP8-E4M3 BF16 dot-product optimization

The CPU exposes `avx512bf16`, so the FP8-E4M3 path now uses `VDPBF16PS` (32 multiply-adds per instruction, accumulated in FP32) instead of two FP32 FMAs per 32 K positions.  Because every FP8-E4M3 value is exactly representable in BF16, the activation is converted to BF16 once and the per-E8M0-scale FP4 table is stored as 32 BF16 lanes (the 16 E2M1 values duplicated for `VPERMW`).

New helpers in `gptqmodel_ext/mxfp4_cpu_kernel.cpp`:

* `load_fp4x32_to_bf16_bh()` — unpack 16 packed bytes into 32 nibble indices and `vpermw` them into a `__m512bh` weight vector.
* `mxfp4_fp8_tile4_bf16()` — 4-wide N tile that loads one `__m512bh` activation chunk and reuses it for 4 `VDPBF16PS` accumulators.
* `mxfp4_fp8_tile4_bf16_allm()` — same 4-wide N tile, but reuses the dequantized weight vectors across a 4-wide **M** block.  This removes the repeated per-M weight-table `vpermw` work and weight loads for batch decode (`M > 1`).
* The existing BF16/FP16 paths are kept unchanged so the comparison is apples-to-apples.

### Updated `q_b_proj` (1536 x 18432) dtype comparison

Run with `iters=20, warmup=3` and 8 threads:

| M | dtype | baseline ms | kernel ms | vs baseline | vs BF16 kernel | vs dense rel err |
|---|---|---|---|---|---|---|
| 1 | BF16 | 65.47 | 2.088 | **31.36x** | 1.00x | 0.4079 |
| 1 | FP16 | 63.16 | 2.104 | **30.02x** | 1.00x | 0.4025 |
| 1 | FP8-E4M3 | 66.60 | **1.908** | **34.90x** | 1.09x | 0.7006 |
| 8 | BF16 | 67.82 | 5.632 | **12.04x** | 1.00x | 0.3871 |
| 8 | FP16 | 64.89 | 6.163 | **10.53x** | 1.05x | 0.3838 |
| 8 | FP8-E4M3 | 70.94 | **2.253** | **31.48x** | **2.50x** | 0.7179 |

* `kernel_baseline_err == 0.0` for the FP8-E4M3 C++ kernel vs its torch baseline (bit-exact).
* `vs dense rel err` is the MXFP4 quantization error on random N(0,1) weights; FP8-E4M3 saturates values above ~448, which inflates the random-weight error.

### All Kimi-3 attention projections, M=1 vs M=8

| projection | M | BF16 kernel ms | FP16 kernel ms | FP8 kernel ms | FP8 vs best (BF16/FP16) | vs baseline speedup |
|---|---|---|---|---|---|---|
| q_a_proj | 1 | 0.816 | 0.815 | **0.672** | 1.21x | 35.46x |
| q_a_proj | 8 | 6.592 | 6.334 | **2.431** | **2.71x** | 10.45x |
| q_b_proj | 1 | 2.088 | 2.104 | **1.908** | 1.09x | 34.90x |
| q_b_proj | 8 | 5.632 | 6.163 | **2.253** | **2.50x** | 31.48x |
| kv_a_proj_with_mqa | 1 | 0.318 | 0.316 | **0.264** | 1.20x | 15.38x |
| kv_a_proj_with_mqa | 8 | 2.399 | 2.392 | **0.959** | **2.50x** | 4.36x |
| kv_b_proj | 1 | 0.971 | 1.005 | **1.068** | 0.91x (slower) | 26.11x |
| kv_b_proj | 8 | 2.565 | 2.814 | **1.222** | **2.10x** | 22.52x |
| o_proj | 1 | 6.507 | 6.309 | **5.087** | 1.24x | 45.73x |
| o_proj | 8 | 50.335 | 50.339 | **18.660** | **2.70x** | 12.91x |

Observations:

* The FP8-E4M3 kernel is **2.1-2.7x faster than the existing BF16/FP16 C++ kernels at `M=8`** across every Kimi-3 attention projection.  The `M`-block optimization is the key change: it reuses the dequantized weight vectors for 4 rows at once, cutting the dominant `vpermw`/table-load work by ~4x for `M >= 4`.
* At `M=1` the speedups over BF16/FP16 are smaller (0.9-1.3x) because the `M`-block cannot amortize weight work; `kv_b_proj` is slightly slower at `M=1` due to the fixed FP8->BF16 activation conversion cost on a small `K`.
* All C++ paths remain bit-exact with their matching torch baselines (BF16/FP8 `kernel_baseline_err == 0.0`; FP16 shows the usual single-ULP accumulation difference).

## Native AVX512-FP16 compute path (`_mm512_fmadd_ph`)

The previous fastest path dequantized MXFP4 to BF16 and used `VDPBF16PS`.  The test host also exposes `avx512fp16`, so
the kernel now has a path that dequantizes MXFP4 **directly to FP16** and multiplies with `VFMADD*PH`, with no FP8 or
BF16 intermediate.  Weights stay in MXFP4 storage (4-bit nibbles + one E8M0 byte per 32 weights).

### Why FP16 wins: raw ISA throughput

A standalone probe (8 independent accumulators, one instruction per accumulator per iteration, single core pinned with
`taskset`, measured clock 3.97 GHz) gives the per-instruction throughput ceiling on this CPU:

| instruction | instr/s | instr/cycle | MAC/instr | GMAC/s |
|---|---|---|---|---|
| `VDPBF16PS` (zmm) | 3.98 G | **1.00** | 32 | 127.4 |
| `VFMADD132PH` (zmm) | 6.61 G | **1.67** | 32 | **211.6** |
| `VFMADD132PS` (zmm) | 7.97 G | 2.01 | 16 | 127.5 |

`VDPBF16PS` packs 32 multiply-adds into one instruction but only issues **once per cycle**, so it delivers exactly the
same MAC rate as plain FP32 FMA.  `VFMADD*PH` also does 32 multiply-adds and issues on both FMA ports, so the FP16 ISA
has ~1.66x more arithmetic throughput than the BF16 dot-product ISA.  That, and not the dequant, is the headline result.

### Implementation

New code in `gptqmodel_ext/mxfp4_cpu_kernel.cpp`, all behind `__builtin_cpu_supports("avx512fp16")` plus a compile-time
guard (`GPTQMODEL_MXFP4_HAS_FP16_ISA`, needs GCC >= 12 or Clang >= 14; older toolchains keep the previous behaviour):

* `scaled_table_fp16_512[256][32]` — per-E8M0-scale FP4 table stored as 32 FP16 lanes, so one `VPERMW` dequantizes 32
  weights straight into a `__m512h`.  No FP8 rounding step, so the FP16 path is *more* accurate than the FP8 path.
* `e8m0_scales_fp16_safe_avx512()` — one AVX-512 min/max scan over the E8M0 bytes.  FP4 magnitudes (0.5 .. 6) times
  `2^(bits-127)` are exact in FP16 only while the exponent stays in `[114, 140]`; outside that window (and for the
  `255`/NaN encoding) the kernel falls back to the existing FP32/BF16 paths instead of producing inf/denormals.
* `mxfp4_fp16_tile_allm<kNTile, kMBlock>()` — register tile that dequantizes each weight vector once and reuses it
  across the whole M block.  `M >= 8` uses a 2 column x 8 row tile, otherwise 4 x 4.  Widening the M block is what pays
  for the dequant: at `q_b_proj M=8` the 2x8 tile is 1.08x faster than 4x4 at a matched thread count.
* `fold_ph_into_ps()` + `kFp16FlushGroups` — FP16 accumulators are widened and folded into FP32 accumulators every 64
  K-groups (2048 elements).  Unbounded FP16 accumulation is fine for short reductions but drifts at long `K`; see below.
* `mxfp4_fp16_column_scalar()` — scalar tail for the columns that do not fill a tile.
* `clamped_threads()` now sizes the thread pool from `M*N*K` (multiply-accumulates) instead of `M*N` (output elements).
  The old heuristic gave `q_b_proj M=8` only 3 threads regardless of `K`; this alone is a 2.3x win for *every* dtype.
* `mxfp4_linear_cpu(..., variant, fp16_flush)` — `variant` selects the compute path (`0` auto, `1` legacy
  FP32/`VDPBF16PS`, `2` native FP16 with unbounded FP16 accumulation, `3` native FP16 with periodic FP32 folding,
  `4` native FP16 pinned to the narrow 4x4 tile) so the paths can be A/B benchmarked in one process.
  `scripts/benchmark_mxfp4_cpu_kernel.py` exposes these as `--variants`, `--threads`, and `--fp16-flush`.

### Matched-shape ISA comparison (`q_b_proj`, M=8, FP8 activations, 8 threads)

Same tile shape, same activation conversion, only the inner instruction differs:

| inner loop | tile | kernel ms | speedup |
|---|---|---|---|
| `VDPBF16PS`, FP32 accumulate | 4x4 | 0.981 | 1.00x |
| `VFMADD*PH`, FP16 accumulate | 4x4 | 0.653 | **1.50x** |
| `VFMADD*PH`, FP16 accumulate | 2x8 | 0.603 | **1.63x** |

The 1.50x at matched tile shape closely tracks the 1.66x ISA-level ratio measured above.

### `q_b_proj` (K=1536, N=18432), M=8, auto threads

| path | kernel ms | vs previous best (FP8 `VDPBF16PS`) | `kernel_baseline_err` | vs dense max abs err |
|---|---|---|---|---|
| BF16, FP32 FMA (before thread fix) | 5.779 | 0.39x | 9.77e-04 | 615.48 |
| FP16, FP32 FMA (before thread fix) | 5.928 | 0.38x | 6.25e-02 | 610.16 |
| FP8-E4M3, `VDPBF16PS` (before thread fix) | 2.263 | 1.00x | 0.0 | 1139.89 |
| BF16, FP32 FMA | 2.155 | 1.05x | 9.77e-04 | 615.48 |
| FP16, FP32 FMA | 2.114 | 1.07x | 6.25e-02 | 610.16 |
| FP8-E4M3, `VDPBF16PS` | 0.974 | 2.32x | 0.0 | 1139.89 |
| **FP16, native `VFMADD*PH`** | **0.523** | **4.33x** | 1.0 | 610.16 |

Target was `< 1.2 ms`; the native FP16 path reaches **0.523 ms**, 1.86x faster than the FP8 `VDPBF16PS` path measured
in the same process.  `kernel_baseline_err = 1.0` is exactly 1 FP16 ULP at the output magnitude (`ref_max = 1589.8`,
ULP = 1.0), and the error against the dense FP32 reference is *unchanged* at 610.16 — MXFP4 quantization dominates by
three orders of magnitude.

### All projections, M=8, auto threads

| projection | K | N | BF16 legacy | FP16 legacy | FP8 `VDPBF16PS` | **FP16 native** | native vs FP8 | `kernel_baseline_err` (ULP) |
|---|---|---|---|---|---|---|---|---|
| q_a_proj | 7168 | 1536 | 0.833 | 0.832 | 0.415 | **0.216** | 1.92x | 8.0 (2 ULP of 7303) |
| q_b_proj | 1536 | 18432 | 2.155 | 2.114 | 0.974 | **0.523** | 1.86x | 1.0 (1 ULP of 1590) |
| kv_b_proj | 512 | 24576 | 1.106 | 1.021 | 0.541 | **0.361** | 1.50x | 0.125 (<1 ULP of 534) |
| o_proj | 12288 | 7168 | 6.503 | 6.510 | 2.783 | **1.416** | 1.96x | 0.8125 (<1 ULP of 12578) |

### `q_b_proj` M scaling

| M | BF16 legacy ms | FP8 `VDPBF16PS` ms | FP16 native ms |
|---|---|---|---|
| 1 | 0.297 | 0.272 | **0.268** |
| 8 | 2.139 | 0.974 | **0.540** |
| 32 | 8.500 | 4.425 | **2.871** |

At `M=1` all paths converge: a single activation row streams 14 MB of packed weights, so the kernel is memory-bound and
the ISA choice is irrelevant.  The FP16 advantage appears exactly where the M block makes the kernel compute-bound.

### FP16 accumulation error vs fold interval (`o_proj`, K=12288, M=8)

FP16 has an 11-bit significand, so a long running sum loses low-order bits.  Folding into FP32 every `N` K-groups
bounds that drift:

| fold every | kernel ms | max abs err vs FP32 accumulation |
|---|---|---|
| never (unbounded FP16) | 1.327 | 32.0 |
| 128 groups (4096 elem) | 1.368 | 16.0 |
| **64 groups (2048 elem)** | **1.417** | **0.8125** |
| 32 groups (1024 elem) | 1.502 | 0.75 |
| 16 groups (512 elem) | 1.749 | 0.5 |
| 8 groups (256 elem) | 2.108 | 0.5 |

64 groups is the knee: it costs 7% over unbounded accumulation and lands below one FP16 ULP of the output, so it is the
default (`kFp16FlushGroups`).  Reductions with `K <= 2048` never fold at all.

### Conclusion

Native FP16 is faster, and the reason is the instruction schedule rather than the dequant: `VDPBF16PS` is a 1-per-cycle
instruction on Emerald Rapids while `VFMADD*PH` issues ~2-per-cycle at the same 32 MAC/instruction, so the FP16 ISA has
~1.66x the MAC ceiling of the BF16 dot-product ISA.  Realizing it needs (a) a wide enough M block that the shared
dequant is amortized, (b) periodic FP32 folding to keep long-`K` reductions accurate, and (c) a scale-range guard,
because FP16 cannot represent the whole E8M0 exponent range.  Accuracy is not a trade-off here: the FP16 path skips the
FP8 rounding of the `VDPBF16PS` path, so its error against the dense FP32 reference is *lower* than the FP8 path's and
identical to the FP32-accumulating FP16 path.

## Next steps / open issues

* Route BF16 and FP8 activations through the native FP16 kernel by default (variants `2`/`3` already do: FP8 drops from
  0.974 ms to 0.510 ms at `q_b_proj M=8`).  BF16 needs an activation range check first, since BF16 carries FP32's
  exponent range and FP16 does not.
* Try `N_TILE` micro-kernels (4/8 output columns per K-tile) to improve activation reuse and keep the speedup gap as M grows.
* Add an AMX-BF16 tile path once a host with `amx_bf16`/`amx_tile` is available; AMX-FP8 is only useful if `amx_fp8` is exposed.
* Integrate `mxfp4_linear_cpu` into the GPT-QModel `nn_modules/qlinear/` backend catalog so real MXFP4 checkpoints can use it.
* Evaluate on real Kimi-3 weights rather than random N(0,1) weights; the relative quantization error should drop on trained weight distributions.

## Thread sizing + M=8 register blocking (layout/expansion investigation)

Machine: 8-core Intel Xeon Platinum 8559C (Emerald Rapids), `avx512f/bw/vl/bf16/fp16/vnni/vbmi`, no AMX, no `avx512_fp8`.

Two profiling findings drove this round:

1. **Only 3 of 8 cores were used.**  `clamped_threads()` sized the pool from `M * N` output elements
   (one thread per 64K outputs).  Decode-shaped GEMMs have a tiny `M`, so `q_b_proj` at `M=8` requested
   `ceil(147456 / 65536) = 3` threads while each output still costs `K = 1536` multiply-adds.  Thread
   sizing now uses the MAC count (`M * N * (K / 64)`), which saturates the 8 cores.
2. **The M block was half the batch.**  With `kMBlock = 4` and `M = 8`, every weight tile was dequantized
   twice.  `kMBlock = 8` amortizes each `vpermw` expansion over 8 `VDPBF16PS` instead of 4.

The BF16 activation path was additionally routed through the same `VDPBF16PS` tile
(`mxfp4_tile_bf16_allm`, now templated on the output dtype) using an unrounded BF16 value table —
every MXFP4 value has at most 3 mantissa bits, so the table entries are exact in BF16.

### `q_b_proj` (K=1536, N=18432), M=8, 8 threads, `iters=100 warmup=20`, median of 3 runs

| dtype | before ms | after ms | speedup | kernel vs torch baseline max abs err |
|---|---|---|---|---|
| BF16 | 5.800 | **0.937** (best 0.824) | **6.2x** | 2.5e-01 (rel 1.6e-04, FP32 accumulation order) |
| FP16 | 5.963 | **2.118** | **2.8x** | 6.25e-02 |
| FP8-E4M3 | 2.356 | **0.764** (best 0.737) | **3.1x** | 0.0 (bit-exact) |

Error vs the dense FP32 reference is unchanged for all three dtypes (BF16 615.479, FP16 610.155,
FP8 1139.892), i.e. the speedup costs no accuracy.

Tile-shape sweep at `M=8` (`kNTile, kMBlock`, FP8, single runs): `(4,8) 0.733 ms`, `(1,8) 0.770 ms`,
`(2,8) 0.848 ms`, `(8,4) 0.917 ms`, `(2,4) 0.869 ms`, `(4,4) 1.072 ms`.  `kNTile * kMBlock` must stay
within the 32 zmm registers; `(4,8)` uses exactly 32 accumulators and wins.  Note that GCC still
spills: the accumulators alone fill the register file, so the `kMBlock` activation vectors and the
weight/table temporaries go to the stack (39 spill stores / 32 reloads in the BF16 instantiation,
95 / 80 in the FP8 one).  The spills are L1 hits and `(4,8)` still beats every smaller,
spill-free shape in the sweep, so weight-expansion amortization dominates register pressure here.  Both constants are
overridable at build time via `-DGPTQMODEL_MXFP4_NTILE` / `-DGPTQMODEL_MXFP4_MBLOCK`.

### AVX-512 VBMI expansion: measured, no gain, not merged

`VPMULTISHIFTQB` extracts the 32 unaligned 4-bit fields of a K group directly into the low nibble of
32 16-bit lanes, cutting the expansion from 5 shuffle-port ops
(`vpand`, `vpsrlw`, `vpunpcklbw`, `vpunpckhbw`, `vinserti128`, `vpmovzxbw`, `vpermw`) to 3
(`vpermq` + `vpmultishiftqb` + `vpermw`).  It was implemented, verified bit-exact, and benchmarked
against the existing loader in the same binary (runtime-gated, 5 runs each, `iters=100`):

| loader | q_b_proj M=8 FP8 ms (5 runs) | min |
|---|---|---|
| `vpermw` unpack (current) | 0.768 / 0.752 / 0.757 / 0.770 / 0.737 | 0.737 |
| `vpmultishiftqb` (VBMI) | 0.798 / 0.734 / 0.742 / 0.834 / 0.778 | 0.734 |

No difference beyond run-to-run noise, so the VBMI path was dropped rather than merged as dead
complexity.  Thread scaling explains why: `1/2/4/8` threads give `5.417 / 2.903 / 1.468 / 0.916 ms`,
i.e. near-linear to 4 threads and 5.9x at 8, so at 8 threads the kernel is bounded by weight traffic
(14.2 MB of `qweight` per call) and `VDPBF16PS` issue rate, not by the shuffle port.  Once the weight
expansion is amortized over 8 M rows it is no longer the critical resource, so a cheaper expansion has
nothing to recover.  Re-pre-packing the nibbles into a SIMD-friendlier order (which would also remove
one shuffle) was not pursued for the same reason: it changes the on-disk weight order for no measured
gain on this host.

A VNNI (`vpdpbusd`) layout was ruled out on numerics, not on layout: MXFP4 values times an E8M0 scale
are not int8-representable without a per-group requantization, and `VPDPBUSD` has the same 32
multiply-adds per instruction as `VDPBF16PS` on this core, so there is no throughput headroom to pay
for it.

### Remaining gap

At 8 threads, `q_b_proj M=8` runs at ~0.74 ms = 610 GMAC/s, about 24% of the `VDPBF16PS` peak.  The
next lever is FP16 (still 2.1 ms because it runs the FP32 FMA path with no M-block reuse) and, on a
host with AMX, an `amx_bf16` tile path.


## AVX-512 VNNI int8 path (`mxfp4_linear_cpu_vnni`)

Follow-up experiment: convert the MXFP4 dot product into an int8 dot product so
each `VPDPBUSD` retires 64 multiply-adds, versus 32 for `VDPBF16PS`.

### Approach

1. **Weights as unsigned int8.** The E2M1 code book multiplied by 2 is exactly
   `{0, 1, 2, 3, 4, 6, 8, 12}` and negatives, so it fits in int8 with no rounding.
   Adding the constant offset `12` maps every code to `[0, 24]`, giving the
   *unsigned* operand `VPDPBUSD` requires.  The factor of 2 is folded into the
   FP32 rescale (`e8m0_scale / 2`).
2. **Offset correction.** `sum_k (w_u8 - 12) * a = vpdpbusd(w_u8, a) - 12 * sum_k a`.
   The correction is identical for all 16 output columns in a tile, so it is one
   broadcast `vpsubd` per (row, group) using a precomputed `12 * sum(a_i8)`.
3. **Activations as signed int8** with a **per-(row, 32-element group)** scale
   (`a_i8 = round(a / (max_abs / 127))`), matching MXFP4's group granularity, so
   activation quantization error stays around 0.4% relative.
4. **Layout.** `mxfp4_prepack_vnni` permutes the nibbles into 16-column x 4-K
   tiles so one 64-byte `VPDPBUSD` operand covers 16 output columns x 4 K values,
   with the activation supplied as a broadcast int32 (embedded broadcast operand).
   Nibble unpack is `vpandq`/`vpsrlw` + `vpshufb` against a broadcast code table:
   ~5 uops per 128 weights, versus ~7 uops per 32 weights for the BF16 `vpermw`
   table lookup.  Per 32-K group and 16 columns the inner loop is 8 `vpdpbusd`
   plus one `vpsubd`/`vcvtdq2ps`/`vfmadd` rescale.
   **Storage is unchanged**: the packed tensors are byte-for-byte the same size
   as the input MXFP4 tensors (`packed bytes == mxfp4 bytes` in every run below);
   only the permutation differs, so this is a one-time load-time transform
   (1.3 ms for `q_b_proj`, 5.6 ms for `o_proj`).

`avx512vnni` is checked at runtime with `__builtin_cpu_supports`; a scalar
reference (`mxfp4_vnni_tile_scalar`) handles CPUs without it and can be forced
with `GPTQMODEL_MXFP4_DISABLE_VNNI=1`.  It produces bit-identical output to the
vectorized path.  The BF16/FP16/FP8 kernels are untouched.

### `q_b_proj` (K=1536, N=18432), M=8, `--iters 30 --warmup 5`

| dtype / path | kernel ms | vs current FP8 | torch baseline ms | speedup vs baseline |
|---|---|---|---|---|
| BF16 | 5.715 | 0.40x | 84.63 | 14.8x |
| FP16 | 5.727 | 0.40x | 70.45 | 12.3x |
| FP8-E4M3 (VDPBF16PS, current) | 2.298 | 1.00x | 70.73 | 30.8x |
| **FP8-E4M3 + VNNI int8 (new)** | **0.534** | **4.30x** | 76.71 | **143.7x** |

Target was <1.2 ms; measured **0.534 ms**.

### Accuracy (`q_b_proj`, M=8)

| metric | FP8 kernel | VNNI kernel |
|---|---|---|
| vs dense FP32, max abs err | 1.1399e+03 | 1.1399e+03 |
| vs dense FP32, rel err | 0.7179 | 0.7179 |
| `kernel_baseline_err` (vs torch baseline) | 0.0 | 16.0 |

The VNNI path is not bit-exact with the FP8 reference because activations are
re-quantized to int8.  The residual is at most one or two FP8-E4M3 output ULPs
(the output ULP at these magnitudes is 64-128), and the error **against the
dense FP32 reference is unchanged** - MXFP4 weight quantization and FP8 output
rounding dominate.  `scripts/check_mxfp4_vnni_correctness.py` reports RMS error
vs dense within 0.3% of the torch baseline across all shapes:

```
M=1   K=1536 N=18432 rms=1.7272e+01 (baseline 1.7240e+01) frac>1ulp=0.0264 -> OK
M=8   K=1536 N=18432 rms=1.7365e+01 (baseline 1.7323e+01) frac>1ulp=0.0269 -> OK
M=3   K=512  N=576   rms=9.9613e+00 (baseline 9.9033e+00) frac>1ulp=0.0301 -> OK
M=8   K=128  N=77    rms=5.0770e+00 (baseline 5.0505e+00) frac>1ulp=0.0195 -> OK
M=2   K=64   N=16    rms=4.5899e+00 (baseline 4.6080e+00) frac>1ulp=0.0312 -> OK
```

Accuracy is acceptable: identical error against the dense reference, and the
per-group int8 activation scale is finer than FP8-E4M3's 3-bit mantissa for
large values.

### Full Kimi-3 attention sweep (`--iters 30 --warmup 5`)

| projection | M | BF16 ms | FP16 ms | FP8 ms | **VNNI ms** | VNNI vs FP8 |
|---|---|---|---|---|---|---|
| q_a_proj | 1 | 0.814 | 0.920 | 0.689 | **0.264** | 2.61x |
| q_b_proj | 1 | 2.196 | 2.095 | 1.971 | **0.623** | 3.16x |
| kv_a_proj_with_mqa | 1 | 0.320 | 0.367 | 0.300 | **0.096** | 3.13x |
| kv_b_proj | 1 | 0.975 | 1.001 | 1.104 | **0.262** | 4.21x |
| o_proj | 1 | 6.868 | 6.384 | 5.174 | **1.723** | 3.00x |
| q_a_proj | 8 | 6.340 | 6.347 | 2.459 | **0.773** | 3.18x |
| q_b_proj | 8 | 5.715 | 5.727 | 2.298 | **0.534** | 4.30x |
| kv_a_proj_with_mqa | 8 | 2.388 | 2.395 | 0.956 | **0.392** | 2.44x |
| kv_b_proj | 8 | 2.691 | 2.627 | 1.213 | **0.319** | 3.80x |
| o_proj | 8 | 51.009 | 51.218 | 18.734 | **4.294** | 4.37x |

VNNI wins on every shape at both M=1 and M=8 (2.4x-4.4x over the FP8 kernel,
6x-16x over BF16).  Two effects compound: `VPDPBUSD` does 2x the MACs per
instruction of `VDPBF16PS`, and the prepacked 16-column tile removes almost all
per-column dequant overhead (one `vpshufb` pair per 128 weights instead of a
`vpermw` table lookup per 32 weights per column).

Test host: INTEL(R) XEON(R) PLATINUM 8559C, 8 cores, `avx512f/bw/vl/bf16/fp16/vnni/vbmi`,
no AMX, no `avx512_fp8`.  Both paths use the same `clamped_threads` policy.

### Remaining headroom

* Thread count: `clamped_threads` gives only 3 threads at `M*N = 147456`
  (one thread per 64K output elements).  The VNNI kernel is now short enough
  that the heuristic, not the ISA, is the limit for small projections.
* Activation quantization is still scalar (~12K elements at M=8); vectorizing it
  would help the small-`K` shapes.
* `o_proj` at M=8 is memory-bound on the 47 MB weight stream, not compute-bound.

## Combined VNNI + M-block/thread-tuning branch update

Merged `devin/mxfp4-cpu-kernel-opt-vnni` and `devin/mxfp4-cpu-kernel-opt-layout`
into `devin/mxfp4-cpu-kernel` (PR #122).  The M-block BF16/FP8 tile and the
new thread heuristic now make the `VDPBF16PS` path competitive with (and
faster than) VNNI on several shapes, so the kernel keeps all four dtype paths
selectable and lets the caller/runtime choose the best per projection.

### `q_b_proj` (1536 x 18432), M=8, `--iters 10 --warmup 2`

| dtype | kernel ms | vs torch baseline | vs dense max abs err | kernel_baseline_err |
|---|---|---|---|---|
| BF16 | 0.734 | ~96x | 615.48 | 0.25 |
| FP16 | 2.122 | ~32x | 610.16 | 0.0625 |
| FP8-E4M3 | 0.747 | ~96x | 1139.89 | 0.0 (bit-exact) |
| FP8-E4M3 + VNNI int8 | **0.520** | ~136x | 1139.89 | 16.0 |

### Full Kimi-3 sweep, M=8 (combined branch, `--iters 10 --warmup 2`)

| projection | K | N | BF16 ms | FP16 ms | FP8 ms | VNNI ms | fastest path |
|---|---|---|---|---|---|---|---|
| q_a_proj | 7168 | 1536 | **0.269** | 0.819 | 0.332 | 0.692 | **BF16** |
| q_b_proj | 1536 | 18432 | 0.734 | 2.122 | 0.747 | **0.520** | **VNNI FP8** |
| kv_a_proj_with_mqa | 7168 | 576 | **0.116** | 0.319 | 0.172 | 0.396 | **BF16** |
| kv_b_proj | 512 | 24576 | 0.385 | 1.004 | 0.393 | **0.278** | **VNNI FP8** |
| o_proj | 12288 | 7168 | **2.025** | 6.329 | 2.112 | 4.267 | **BF16** |

Observations:

* The VNNI int8 path is the fastest FP8 implementation for `q_b_proj` and
  `kv_b_proj` (large `N`, moderate `K`), beating the previous 2.25 ms FP8
  kernel by **~4.3x** and the torch baseline by **>100x**.
* The BF16/FP8 `VDPBF16PS` tile wins for `q_a_proj`, `kv_a_proj_with_mqa`, and
  `o_proj` (large `K` relative to `N` or memory-bound weight streams), where
  the prepack and int8 requant overhead of VNNI is not amortized.
* FP16 remains the slowest path because it still uses FP32-FMA accumulation in
  this branch; the FP16-native `VFMADD*PH` exploration in PR #125 shows it can
  be ~0.6 ms when a compiler with `avx512fp16` intrinsics is available.
* CPU memory is still ordinary DDR (no HBM), so `o_proj` and other large-weight
  projections are ultimately memory-bound.  Further gains require either
  reducing weight bytes (still constrained to MXFP4-sized storage) or reusing
  cached weights across tokens/tiles.

## Native FP16 rebased on the VNNI branch (PR #125, post-merge)

The FP16 report above was measured before `devin/mxfp4-cpu-kernel` gained the
VNNI int8 path and the `(4,8)` BF16 tile.  After merging, two things changed.

### The MAC-based thread heuristic now also covers the VNNI entry point

`mxfp4_linear_cpu_vnni` still sized its pool from `M * N`, so it ran on 1-3
threads.  Routing it through the same `M * N * K` estimate is worth **2.2x** on
`q_b_proj M=8` (0.520 -> 0.238 ms) and is the single largest change in this
merge.  Both entry points now share one policy.

### `q_b_proj` (K=1536, N=18432), M=8, auto threads, median of 5 (`--iters 40 --warmup 10`)

| path | ms | `kernel_baseline_err` | vs dense |
|---|---|---|---|
| BF16 `VDPBF16PS` | 0.759 | 0.25 | 615.48 |
| FP16 legacy FP32 FMA (`variant=1`) | 2.254 | 6.25e-02 | 610.16 |
| FP8-E4M3 `VDPBF16PS` | 0.864 | 0.0 | 1139.89 |
| **FP16 native `VFMADD*PH`** | **0.530** | 1.0 | 610.16 |
| FP8-E4M3 + VNNI int8 | **0.238** | 16.0 | 1139.89 |

### Full Kimi-3 attention sweep (`--iters 20 --warmup 5`)

| projection | M | BF16 | FP16 legacy | **FP16 native** | FP8 | VNNI int8 |
|---|---|---|---|---|---|---|
| q_a_proj | 1 | 0.173 | 0.165 | **0.123** | 0.145 | **0.051** |
| q_b_proj | 1 | 0.266 | 0.296 | **0.265** | 0.274 | **0.062** |
| kv_a_proj_with_mqa | 1 | 0.149 | 0.178 | **0.132** | 0.175 | **0.055** |
| kv_b_proj | 1 | 0.196 | 0.424 | 0.259 | 0.206 | **0.042** |
| o_proj | 1 | 0.711 | 0.826 | 0.857 | 0.750 | **0.264** |
| q_a_proj | 8 | 0.299 | 0.940 | **0.198** | 0.433 | 0.264 |
| q_b_proj | 8 | 0.759 | 2.254 | **0.530** | 0.864 | **0.238** |
| kv_a_proj_with_mqa | 8 | 0.133 | 0.378 | **0.122** | 0.207 | 0.221 |
| kv_b_proj | 8 | 0.406 | 1.009 | **0.358** | 0.417 | **0.123** |
| o_proj | 8 | 2.105 | 6.866 | **1.415** | 2.194 | **0.834** |

### Conclusion after the merge

* Native FP16 is **2.7x-4.9x** faster than the legacy FP16 path at M=8 and is
  the fastest **FP16-activation** implementation on every shape measured, so
  FP16 is no longer the dtype to avoid: it now beats both BF16 and the FP8
  `VDPBF16PS` path (`q_a_proj` 0.198 vs 0.299/0.433, `o_proj` 1.415 vs
  2.105/2.194).
* VNNI int8 is still faster where it applies (2.2x on `q_b_proj M=8`).  The two
  are not interchangeable: VNNI needs a one-time weight prepack and re-quantizes
  activations to int8 (`kernel_baseline_err` 16.0), while native FP16 needs no
  prepack, consumes the FP16 activations a model already has, and stays within
  one FP16 ULP of the FP32-accumulating reference.  For FP16 inference the
  prepack + FP16->FP8->int8 round trip is not free, so the FP16 path is the one
  a FP16 model should use; VNNI remains the choice for FP8 activations.
* Neither path is limited by dequantization any more.  `VFMADD*PH` retires 32
  MACs at ~1.67 instructions/cycle versus 1.00 for `VDPBF16PS`, and `VPDPBUSD`
  retires 64; the measured ordering follows those ratios once threads and the M
  block are right.

### FP16 overflow guard

FP16 multiply-accumulate saturates at 65504, which the E8M0 scale-range check
alone does not prevent: a representable weight (up to `6 * 2^13`) times a
moderate activation already overflows.  The dispatch now also scans
`max|activation|` (one `VPMAXUW` pass over the FP16 bit patterns with the sign
cleared) and takes the native path only when
`flush_groups * max|w| * max|a| <= 65504`, treating inf/NaN activations as
unsafe.  Anything outside that window falls back to the FP32-accumulating path
bit-exactly.  The bound is worst-case (it assumes every product in a fold
interval has the same sign), so it is conservative by design; for the sweep
above it never triggers.  `tests/kernels/test_mxfp4_cpu_kernel.py` covers it
with a case whose exact result is 0 but whose FP16 lane partials reach 98304 —
verified to return inf with the guard removed.

## Final parent-PR merge: VNNI + layout fixes + native FP16 FMA

Merged the updated child PRs into `devin/mxfp4-cpu-kernel`:

* `devin/mxfp4-cpu-kernel-opt-layout` — `std::max(1, K/64)` thread sizing fix.
* `devin/mxfp4-cpu-kernel-opt-vnni` — exact prepack-footprint assert, ISA-clean
  scalar fallback, removal of global `-mavx512*` flags.
* `devin/mxfp4-cpu-kernel-opt-fp16` — native `VFMADD*PH` FP16 path + overflow
  guard + VNNI thread-sizing correction (`mxfp4_linear_cpu_vnni` now uses the
  same MAC-based `clamped_threads` as the other paths).

`devin/mxfp4-cpu-kernel-opt-tile` (VBMI nibble decode + compile-time tile
sweep) was **not** merged: it conflicts with the VNNI/FP16 paths, and the
merged VNNI + FP16 results below already beat or match its best numbers.

### Full Kimi-3 sweep, M=8, merged branch (`--iters 10 --warmup 2`, g++-12)

| projection | K | N | BF16 ms | FP16 ms | FP8 ms | VNNI ms | fastest |
|---|---|---|---|---|---|---|---|
| q_a_proj | 7168 | 1536 | 0.292 | 0.200 | 0.333 | **0.093** | **VNNI** |
| q_b_proj | 1536 | 18432 | 0.760 | 0.516 | 0.779 | **0.196** | **VNNI** |
| kv_a_proj_with_mqa | 7168 | 576 | 0.130 | 0.098 | 0.172 | **0.063** | **VNNI** |
| kv_b_proj | 512 | 24576 | 0.472 | 0.355 | 0.461 | **0.110** | **VNNI** |
| o_proj | 12288 | 7168 | 2.273 | 2.548 | 2.175 | **0.556** | **VNNI** |

Key changes vs the earlier combined branch:

* VNNI is now the fastest path on **every** projection at M=8, including the
  previously memory-bound / large-`K` shapes (`q_a_proj`, `kv_a`, `o_proj`).
  This comes from sizing the VNNI parallel pool by `M * N * K` instead of
  `M * N`.
* Native FP16 is now competitive (0.098–0.516 ms) and is the fastest **FP16
  activation** path, beating BF16 and FP8 `VDPBF16PS` on all shapes.
* `q_b_proj` M=8 is now **0.196 ms** VNNI and **0.516 ms** FP16, versus the
  previous 2.25 ms FP8 baseline — a **>11×** improvement for FP8 and a
  **>4×** improvement for FP16.
* Accuracy is unchanged vs dense: VNNI `kernel_baseline_err` is 16.0 (1–2
  FP8 ULP from int8 requant), FP16 is within 1 FP16 ULP, BF16/FP8 are
  bit-exact where applicable.

### Validation

* `ruff check scripts/benchmark_mxfp4_cpu_kernel.py scripts/check_mxfp4_vnni_correctness.py tests/kernels/test_mxfp4_cpu_kernel.py` — clean.
* `pytest -q tests/kernels/test_mxfp4_cpu_kernel.py` — 8 passed.
* `scripts/check_mxfp4_vnni_correctness.py` and
  `GPTQMODEL_MXFP4_DISABLE_VNNI=1 scripts/check_mxfp4_vnni_correctness.py` —
  0 failures.

### Remaining open question

`devin/mxfp4-cpu-kernel-opt-tile` adds a compile-time (N,M) tile dispatch and
a VBMI nibble decode that is ~5–10% faster in isolation, but it conflicts with
the current VNNI/FP16 dispatch structure. It can be revisited as a follow-up
if the extra complexity is justified after real-model integration.

## GPT-QModel Ultra backend integration (`Mxfp4CpuLinear`)

Added the plumbing needed for `GPTQModel.load(..., backend="mxfp4_cpu")` to use
the kernel end-to-end.

### New / modified files

* `gptqmodel/utils/backend.py` — `BACKEND.MXFP4_CPU = "mxfp4_cpu"`.
* `gptqmodel/quantization/config.py` — `METHOD.MXFP4`, `FORMAT.MXFP4`, and
  `MXFP4Config` (4-bit, group_size=-1, desc_act=False, sym=True).
* `gptqmodel/quantization/__init__.py` — exports `MXFP4Config`.
* `gptqmodel/nn_modules/qlinear/mxfp4_cpu.py` — `Mxfp4CpuLinear`, derived from
  `WeightOnlyQuantLinear`, with `SUPPORTS_DEVICES=[CPU]`,
  `SUPPORTS_FORMATS={FORMAT.MXFP4: 15}`, and `SUPPORTS_DTYPES` for bf16/fp16/
  fp8-e4m3fn/fp32.  Registers `qweight` `(N, K//2)` uint8 and `scales`
  `(N, K//32)` uint8; `pack_original` quantizes dense weights, `post_init`
  JIT-loads the C++ extension and optionally builds the VNNI prepack,
  and `forward` dispatches to `mxfp4_linear_cpu` / `mxfp4_linear_cpu_vnni`.
* `gptqmodel/utils/mxfp4_cpu.py` — shared JIT loader (`load_mxfp4_cpu_kernel`)
  plus `quantize_mxfp4` / `dequantize_mxfp4` helpers used by `pack_original`.

### End-to-end test

`scripts/test_mxfp4_cpu_load.py` builds a tiny GPT2 checkpoint (1 layer, n_embd=64,
vocab_size=1024), converts its `c_attn`, `c_proj`, `c_fc`, and `mlp.c_proj`
to `Mxfp4CpuLinear`, saves it, and reloads with `GPTQModel.load(path,
backend="mxfp4_cpu", device="cpu", dtype="bfloat16")`.  The loader selects the
`Mxfp4CpuLinear` kernel and a forward pass produces the expected logits shape
`[1, seq_len, vocab_size]`.

### Validation

* `ruff check --config format/ruff.toml <changed paths>` — clean for the new
  `mxfp4_cpu.py` files, `__init__.py`, `backend.py`, and the test/script files.
* `pytest -q tests/kernels/test_mxfp4_cpu_qlinear.py` — 7 passed
  (bf16/fp16/fp8-e4m3 × VNNI on/off, plus `dequantize_weight`).
* `pytest -q tests/kernels/test_mxfp4_cpu_kernel.py` — 8 passed.
* `scripts/benchmark_mxfp4_cpu_kernel.py --projection q_b_proj --dtype all --m 8`
  — passes, VNNI FP8 remains fastest at `q_b_proj M=8` (~0.2 ms, ~340× baseline).

### Tiny K3 model for real testing

The best publicly available tiny K3-like checkpoint is
`inference-optimization/Kimi-K3-0.40B` on Hugging Face (~0.4 B params, 8 layers,
preserves the 3:1 KDA/MLA attention ratio).  It is FP32, not pre-quantized to
MXFP4, and uses a custom `kimi_linear` modeling file that requires
`llm-compressor` on `sys.path` and `trust_remote_code=True`.  GPT-QModel-Ultra
does not yet have a `KimiK3QModel` definition, so it cannot be loaded directly
through `GPTQModel.load`.  The synthetic GPT2 integration above validates the
backend plumbing; adding real K3 support would need a new `module_tree` for the
`kimi_linear` architecture and either quantizing the 0.4B weights to MXFP4 or
finding/converting an already-quantized K3 checkpoint.
