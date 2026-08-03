# Planar (gptq_p) GPU kernel log

Running log of kernel design decisions, forward/backward progress, and all test
and benchmark results for the planar GPTQ GPU kernels (3/5/6/7-bit).

## Hardware / software snapshot

- Host: `amd-one` outpost (NVIDIA, not AMD: 8x NVIDIA PG506 A100-class 96GB)
- GPUs: `nvidia-smi` PCI-bus order: 8x `NVIDIA PG506-230/232`, 96 GiB each,
  compute capability 8.0 (Ampere sm_80)
- Torch 2.13.0+cu130, CUDA 13.0, Triton 3.7.1
- Python 3.14.5 free-threading (`/root/gptqmodel/.venv`)
- GPU leasing through `gpu_allocator` CLI (`http://127.0.0.1:17351`)

## Design

### Format recap (docs/gptq_planar.md)

Every 32 logical codes occupy `bits` adjacent int32 words, low plane first.
Plane layouts: 3=(2)+(1), 5=(4)+(1), 6=(4)+(2), 7=(4)+(2)+(1). Within a plane
of width `w`, word `i` holds codes `[i*(32//w), (i+1)*(32//w))` at shifts
`w*j`. `qweight` packs along rows, `qzeros` along columns. Zeros are v2
semantics (no +1 bias).

### Bandwidth-first strategy

The planar decode is purely memory-bound: per 32 codes we read exactly `bits`
int32 words (the information-theoretic minimum), and the decode is fixed
shifts/masks + OR-merge — branch-free and uniform per lane. The priority
order is therefore:

1. **Coalesced packed-word loads.** In the dequant kernel adjacent threads in
   a warp cover adjacent output columns, so each plane-word load is a fully
   coalesced 128B transaction per 32 lanes; words re-read by multiple rows of
   the same 32-code block hit L2/L1.
2. **No intermediate materialization on the decode path** (registers only,
   single fp16 store per element).
3. **Fused dequant+matmul for decode-shape (small-M) GEMM** so packed words
   are read once per tile instead of writing + re-reading a dense fp16 W
   (traffic ratio fp16 W vs packed = 16/bits, i.e. 2.3x-5.3x less DRAM
   traffic at 7-3 bits).

### Tiering

- Tier 1 (this PR): Triton kernels — `planar_dequant` (dequant to fp16/bf16,
  then cuBLAS matmul; best for prefill) and `planar_gemm` (fused
  dequant+matmul, used for small-M decode shapes).
- Tier 2 (future): native CUDA kernel modeled on Marlin (cp.async multi-stage
  pipeline, LOP3 register decode, mma.sync). Not in this PR.

## Progress log

### 2026-08-02: environment + design

- Confirmed repo state: planar CPU reference merged on main (PR #157/#158/#159).
- Chose Triton-first tier; decode math per element:
  `block = k // 32; idx = k % 32; word_row = block*bits + p_off + idx // (32//w);
  shift = w * (idx % (32//w))`, OR-merged over planes with bit offsets equal to
  cumulative plane widths.
- qzeros mirror the same decode along columns.

### 2026-08-02: Tier-1 Triton kernels implemented

New module `gptqmodel/nn_modules/triton_utils/planar.py`:

- `planar_dequant_kernel`: 1D elementwise decode (same launch geometry family
  as the tuned continuous `dequant_kernel`), fp32 math, fp16/bf16 store.
- `planar_gemm_kernel`: fused dequant+matmul with split-K (`tl.atomic_add`
  into an fp32 workspace) for small-M decode shapes.

Wiring:

- `TorchLinear._can_use_triton_dequant` now routes planar modules (bits
  3/5/6/7) to `planar_dequant` on CUDA instead of the eager unpack path;
  CPU/non-CUDA fallback and continuous decode are untouched.
- `TritonV2Linear` gains `FORMAT.GPTQ_P` + bits 5/6/7 support; planar forward
  uses the fused GEMM for M <= 32 and dequant+cuBLAS above. The continuous
  3-bit fused path and its contract are unchanged (planar modules skip the
  continuous 3-bit `post_init` checks, which do not apply to planar words).

### What worked / what didn't

- WORKED: bit-exact dequant vs `planar_unpack_rows/cols` reference for all of
  3/5/6/7 bits, desc_act on/off, uneven group tails (fp16). fp32 decode +
  single fp16 round matches Torch's fp16 elementwise rounding exactly.
- WORKED: fused GEMM matches dequant reference within fp16/bf16 matmul
  tolerance for M in {1, 8, 33, 128} and desc_act/sym variants.
- FAILED then fixed: first fused-GEMM config space (BLOCK_N/BLOCK_K up to
  128x128, num_stages 3) exceeded Ampere's 164KB shared-memory limit at
  N=11008 (`OutOfResources: Required: 168192, Hardware limit: 166912`).
  Two causes: (1) decode gathers buffer whole `[BLOCK_K, BLOCK_N]` int32
  tiles per plane per stage, so tile budgets must assume ~4B/element/plane;
  (2) the autotune cache was keyed on (N, K) only while BLOCK_M is passed
  explicitly, so a config tuned at BLOCK_M=16 (M=1) was replayed OOM at
  BLOCK_M=32 (M=512). Fixed by shrinking the config space (BLOCK_K=64,
  BLOCK_N<=128) and adding BLOCK_M to the autotune key.
- OBSERVED: without split-K, decode-shape (M=1) fused GEMM only launched
  N/BLOCK_N programs (~32-64) on a 124-SM device and lost to
  dequant+cuBLAS despite reading 2.3-5.3x less weight data. Split-K
  (1/4/8 via `tl.atomic_add` into an fp32 workspace) improved M=1 latency
  (~0.40 -> ~0.33 ms at 4096x4096) but the fused path STILL loses to
  dequant+cuBLAS at every measured shape/bit. Root cause: the decode gathers
  load one int32 word per output element, so each packed word is re-fetched
  32/width times; the effective weight-side traffic is ~4B/element/plane
  instead of `bits/8` B/element, erasing the theoretical bandwidth win.
  The fused path is therefore DISABLED by default
  (`GPTQMODEL_PLANAR_FUSED_MAX_M=0`); production routing always uses
  Triton dequant + cuBLAS. A future rewrite should stage each 32-code
  block's `bits` words once in shared/registers and expand in-register
  (Marlin-style), which is also the natural shape for the native CUDA tier.
- DEFERRED: native CUDA (Marlin-style cp.async + LOP3 register decode) is a
  follow-up tier; Triton was validated first per the tiering plan.
- NOT ADDED (by design): no dense-weight caching between forwards/layers —
  every matmul decodes transiently, preserving the VRAM savings.

### Test results (2026-08-02, NVIDIA PG506-230 96GB, sm_80, lease via gpu_allocator)

- `tests/test_planar_triton_kernels.py`: **64 passed** at runtime on GPU
  (dequant bit-exactness, fused matmul tolerance, desc_act/sym, batched
  shapes, TorchLinear routing, TritonV2 forward; fp16 + bf16).
- Existing CPU suites (`test_planar_bits_567.py`, `test_planar_format_gptq_p.py`,
  `test_torch_kernel_accuracy.py`): **146 passed** — no regressions in the
  planar CPU fallback or continuous layouts.
- Commands:
  ```
  python -m gpu_allocator.cli run -n 1 --style uuid -- python -m pytest -q tests/test_planar_triton_kernels.py
  python -m pytest -q tests/test_planar_bits_567.py tests/test_planar_format_gptq_p.py tests/test_torch_kernel_accuracy.py
  ```

### Benchmarks

`scripts/benchmark_planar_kernels.py` (median CUDA-event ms, 50 iters after
warmup, fp16, group_size=128, NVIDIA PG506-230 / A100-class sm_80, 124 SMs,
95.2 GiB, torch 2.13.0+cu130, triton 3.7.1, CUDA 13.0; idle preflight before
import and re-check before the timed region; leased via gpu_allocator).

Columns: `torch-eager` = merged Torch planar CPU-style decode running on GPU,
`tri-dequant` = new Triton planar dequant + cuBLAS (the production path),
`tri-fused` = experimental fused split-K GEMM (opt-in only),
`marlin-4bit` = 4-bit Marlin native CUDA kernel as the speed ceiling.

```
| bits |         K x N |     M |  torch-eager |  tri-dequant |  tri-fused | marlin-4bit | best-vs-eager |
|------|---------------|-------|--------------|--------------|------------|-------------|---------------|
|    3 |  4096 x  4096 |     1 |        1.944 |        0.183 |      0.331 |       0.082 |        10.60x |
|    3 |  4096 x  4096 |    16 |        1.950 |        0.184 |      0.337 |       0.082 |        10.58x |
|    3 |  4096 x  4096 |   512 |        1.992 |        0.237 |      2.896 |       0.143 |         8.42x |
|    3 |  4096 x  4096 |  2048 |        2.182 |        0.416 |     10.371 |       0.373 |         5.25x |
|    3 |  4096 x 11008 |     1 |        4.384 |        0.414 |      0.608 |       0.081 |        10.60x |
|    3 |  4096 x 11008 |    16 |        4.395 |        0.413 |      0.617 |       0.084 |        10.65x |
|    3 |  4096 x 11008 |   512 |        4.497 |        0.528 |      7.169 |       0.271 |         8.52x |
|    3 |  4096 x 11008 |  2048 |        4.949 |        0.994 |     27.367 |       0.902 |         4.98x |
|    3 | 11008 x  4096 |     1 |        4.421 |        0.346 |      0.648 |       0.083 |        12.77x |
|    3 | 11008 x  4096 |    16 |        4.411 |        0.343 |      0.660 |       0.084 |        12.86x |
|    3 | 11008 x  4096 |   512 |        4.542 |        0.470 |      7.398 |       0.263 |         9.66x |
|    3 | 11008 x  4096 |  2048 |        5.032 |        0.964 |     28.965 |       0.878 |         5.22x |
|    5 |  4096 x  4096 |     1 |        1.971 |        0.184 |      0.333 |       0.082 |        10.69x |
|    5 |  4096 x  4096 |    16 |        1.981 |        0.186 |      0.339 |       0.083 |        10.63x |
|    5 |  4096 x  4096 |   512 |        2.017 |        0.240 |      2.808 |       0.142 |         8.42x |
|    5 |  4096 x  4096 |  2048 |        2.190 |        0.416 |     10.046 |       0.372 |         5.27x |
|    5 |  4096 x 11008 |     1 |        4.441 |        0.407 |      0.607 |       0.083 |        10.92x |
|    5 |  4096 x 11008 |    16 |        4.425 |        0.403 |      0.617 |       0.083 |        10.97x |
|    5 |  4096 x 11008 |   512 |        4.552 |        0.522 |      6.985 |       0.271 |         8.72x |
|    5 |  4096 x 11008 |  2048 |        5.019 |        0.993 |     26.617 |       0.906 |         5.06x |
|    5 | 11008 x  4096 |     1 |        4.475 |        0.342 |      0.647 |       0.085 |        13.09x |
|    5 | 11008 x  4096 |    16 |        4.481 |        0.341 |      0.659 |       0.085 |        13.14x |
|    5 | 11008 x  4096 |   512 |        4.596 |        0.469 |      7.185 |       0.265 |         9.80x |
|    5 | 11008 x  4096 |  2048 |        5.090 |        0.958 |     27.898 |       0.884 |         5.31x |
|    6 |  4096 x  4096 |     1 |        2.009 |        0.214 |      0.303 |       0.085 |         9.39x |
|    6 |  4096 x  4096 |    16 |        1.960 |        0.216 |      0.307 |       0.084 |         9.07x |
|    6 |  4096 x  4096 |   512 |        2.014 |        0.267 |      2.549 |       0.144 |         7.54x |
|    6 |  4096 x  4096 |  2048 |        2.195 |        0.447 |      9.056 |       0.373 |         4.91x |
|    6 |  4096 x 11008 |     1 |        4.494 |        0.470 |      0.544 |       0.082 |         9.57x |
|    6 |  4096 x 11008 |    16 |        4.475 |        0.470 |      0.561 |       0.082 |         9.52x |
|    6 |  4096 x 11008 |   512 |        4.598 |        0.586 |      6.309 |       0.270 |         7.84x |
|    6 |  4096 x 11008 |  2048 |        5.060 |        1.058 |     24.133 |       0.905 |         4.78x |
|    6 | 11008 x  4096 |     1 |        4.528 |        0.421 |      0.581 |       0.083 |        10.76x |
|    6 | 11008 x  4096 |    16 |        4.488 |        0.419 |      0.589 |       0.084 |        10.72x |
|    6 | 11008 x  4096 |   512 |        4.612 |        0.548 |      6.562 |       0.264 |         8.41x |
|    6 | 11008 x  4096 |  2048 |        5.102 |        1.041 |     25.488 |       0.892 |         4.90x |
|    7 |  4096 x  4096 |     1 |        2.656 |        0.241 |      0.406 |       0.085 |        11.04x |
|    7 |  4096 x  4096 |    16 |        2.639 |        0.242 |      0.410 |       0.085 |        10.92x |
|    7 |  4096 x  4096 |   512 |        2.696 |        0.293 |      3.814 |       0.145 |         9.20x |
|    7 |  4096 x  4096 |  2048 |        2.897 |        0.473 |     13.761 |       0.373 |         6.12x |
|    7 |  4096 x 11008 |     1 |        6.027 |        0.508 |      0.782 |       0.084 |        11.87x |
|    7 |  4096 x 11008 |    16 |        6.001 |        0.510 |      0.794 |       0.084 |        11.76x |
|    7 |  4096 x 11008 |   512 |        6.120 |        0.629 |      9.378 |       0.272 |         9.73x |
|    7 |  4096 x 11008 |  2048 |        6.574 |        1.095 |     36.041 |       0.904 |         6.01x |
|    7 | 11008 x  4096 |     1 |        6.032 |        0.484 |      0.843 |       0.085 |        12.45x |
|    7 | 11008 x  4096 |    16 |        6.010 |        0.484 |      0.852 |       0.085 |        12.42x |
|    7 | 11008 x  4096 |   512 |        6.148 |        0.611 |      9.741 |       0.264 |        10.06x |
|    7 | 11008 x  4096 |  2048 |        6.637 |        1.106 |     38.026 |       0.879 |         6.00x |
```

Command:

```
PYTHONPATH=<worktree> python -m gpu_allocator.cli run -n 1 --style uuid -- \
    python -u scripts/benchmark_planar_kernels.py
```

Interpretation:

- The production path (`tri-dequant` + cuBLAS) is 4.8-13.1x faster than the
  merged Torch planar decode across all bits/shapes, with the largest wins at
  decode shapes (M=1/16) where the Torch unpack dominates end-to-end latency.
- Marlin 4-bit remains ~2-6x faster still. That gap is the headroom for the
  native CUDA tier: Marlin avoids the dense fp16 weight round-trip through
  DRAM entirely (register decode + tensor-core MMA), which the current Triton
  fused kernel cannot match because of its gather-based decode (see above).
- Latency is nearly flat in `bits` for the dequant path (it is dominated by
  the dense fp16 write/read), which is exactly the planar design goal: adding
  planes adds coalesced word rows, not divergent decode work.
- Failed attempts kept for the record: two `OutOfResources` shared-memory
  crashes of the fused kernel (168192 and 231680 bytes vs the 166912-byte
  hardware limit) before the config-space and autotune-key fixes; and a first
  Marlin baseline that used a non-existent `MarlinLinear.pack_block` (fixed
  by packing via `TorchLinear.pack_block` and copying the GPTQ-layout
  tensors before `post_init()`).

### Status summary

- GPU accuracy tests: 64/64 passed at runtime (no skips, no compile-only).
- CPU regression suites: 146/146 passed.
- Production GPU routing: Triton planar dequant + cuBLAS (3/5/6/7-bit).
- Fused split-K GEMM: implemented, correct, benchmarked; opt-in via
  `GPTQMODEL_PLANAR_FUSED_MAX_M` because it does not yet win.
- Native CUDA/HIP Marlin-style kernel: deferred (documented follow-up).
- No dense-weight caching between layers/modules anywhere in the new paths.

## Nsight Compute iteration round 2 (post-PR #160, same A100-class PG506-230)

Bandwidth-first optimization of the production dequant kernel, driven by
`ncu --set full` under GPU-allocator leases. Profiling target:
`scripts/profile_planar_dequant.py` (3-bit, 4096x4096, fp16, g=128 unless
noted). All numbers are per-kernel from ncu.

| kernel version | duration | DRAM SOL | mem tput | SM/ALU | occupancy | note |
|---|---|---|---|---|---|---|
| v1 elementwise, X_BLOCK=1024/1 warp (as merged in PR) | 90.5 us | 8.9% | 217 GB/s | 71.7% (ALU-bound) | 17.6% | inherited launch geometry from continuous dequant |
| v2 elementwise, autotuned blocks/warps | 82.1 us | 9.8% | ~240 GB/s | 81.0% ALU | 57.0% | occupancy fixed, still integer-ALU bound: per-element div/mod/address math dominates |
| v3 row-per-program + separate zeros-decode kernel | 49.6 us (+12 us zeros) | 16.5% | 403 GB/s | 26.6% ALU | 85.5% | scalar row math removed the ALU wall; became L2-traffic bound (54% L2, 19% L1 hit) |
| v4 32-row-block-per-program (shipped) | 30.6 us (+4 us zeros) | 25.9% | 634 GB/s | 51.6% | 39.8% | plane word rows stay L1-resident across the static 32-row unroll; every shift/word offset is a compile-time constant |

Net: ~2.6x faster kernel (90.5 -> ~35 us including the tiny zeros kernel).
The fp16 dense write floor for 4096x4096 is ~23 us at ~1.4 TB/s, so v4 is
within ~1.5x of the format's bandwidth floor on this GPU.

What worked:
- Killing per-element index ALU: v2 profiling showed 81% integer-pipe
  utilization with only 9% DRAM SOL — the kernel was decoding addresses, not
  moving bytes. Moving row/plane/shift math to scalars (v3) and then to
  compile-time constants via `tl.static_range(32)` (v4) freed the ALU.
- Hoisting the qzeros decode into a separate tiny kernel writing a dense
  int32 [groups, N] buffer (transient, freed with the dequant output; NOT a
  cross-layer cache). The main kernel then does pure coalesced row loads.
- Block-per-program L1 reuse: each packed word row is read from L2/DRAM once
  per program instead of once per output row (plane loads hit L1 across the
  32-row unroll).

Failed/rejected along the way:
- `PF2: tl.constexpr` defined inside `tl.static_range` raised
  `constexpr cannot be reassigned` on Triton 3.7.1 for 3-plane (7-bit)
  decode; fixed by hoisting `PF2` above the loop.
- A pure row-per-program kernel (v3) without the block unroll left the
  kernel L2-bound because scale/zero rows were re-fetched per row with no
  intra-program reuse.

End-to-end benchmark after v4 (same methodology as the first table; 50
CUDA-event iters, fp16, g=128, idle preflight + lease):

| bits |         K x N |     M |  torch-eager |  tri-dequant |  tri-fused | marlin-4bit | best-vs-eager |
|------|---------------|-------|--------------|--------------|------------|-------------|---------------|
|    3 |  4096 x  4096 |     1 |        2.008 |        0.215 |      0.370 |       0.086 |         9.34x |
|    3 |  4096 x  4096 |    16 |        2.016 |        0.219 |      0.374 |       0.085 |         9.22x |
|    3 |  4096 x  4096 |   512 |        2.075 |        0.281 |      3.335 |       0.158 |         7.40x |
|    3 |  4096 x  4096 |  2048 |        2.264 |        0.471 |     12.014 |       0.422 |         4.81x |
|    3 |  4096 x 11008 |     1 |        4.464 |        0.278 |      0.688 |       0.086 |        16.09x |
|    3 |  4096 x 11008 |    16 |        4.468 |        0.278 |      0.699 |       0.086 |        16.10x |
|    3 |  4096 x 11008 |   512 |        4.586 |        0.404 |      8.280 |       0.305 |        11.34x |
|    3 |  4096 x 11008 |  2048 |        5.098 |        0.926 |     31.681 |       1.027 |         5.51x |
|    3 | 11008 x  4096 |     1 |        4.520 |        0.285 |      0.736 |       0.085 |        15.88x |
|    3 | 11008 x  4096 |    16 |        4.499 |        0.279 |      0.743 |       0.085 |        16.12x |
|    3 | 11008 x  4096 |   512 |        4.640 |        0.423 |      8.577 |       0.298 |        10.97x |
|    3 | 11008 x  4096 |  2048 |        5.223 |        0.982 |     33.618 |       0.977 |         5.32x |
|    5 |  4096 x  4096 |     1 |        2.014 |        0.214 |      0.372 |       0.086 |         9.41x |
|    5 |  4096 x  4096 |    16 |        2.038 |        0.218 |      0.377 |       0.083 |         9.34x |
|    5 |  4096 x  4096 |   512 |        2.085 |        0.278 |      3.241 |       0.159 |         7.50x |
|    5 |  4096 x  4096 |  2048 |        2.278 |        0.468 |     11.663 |       0.422 |         4.87x |
|    5 |  4096 x 11008 |     1 |        4.513 |        0.289 |      0.692 |       0.086 |        15.60x |
|    5 |  4096 x 11008 |    16 |        4.533 |        0.284 |      0.707 |       0.089 |        15.95x |
|    5 |  4096 x 11008 |   512 |        4.679 |        0.420 |      8.119 |       0.307 |        11.13x |
|    5 |  4096 x 11008 |  2048 |        5.167 |        0.949 |     30.810 |       1.028 |         5.45x |
|    5 | 11008 x  4096 |     1 |        4.549 |        0.293 |      0.733 |       0.086 |        15.53x |
|    5 | 11008 x  4096 |    16 |        4.552 |        0.289 |      0.745 |       0.087 |        15.74x |
|    5 | 11008 x  4096 |   512 |        4.697 |        0.432 |      8.297 |       0.298 |        10.87x |
|    5 | 11008 x  4096 |  2048 |        5.242 |        0.989 |     32.178 |       0.978 |         5.30x |
|    6 |  4096 x  4096 |     1 |        2.065 |        0.225 |      0.340 |       0.086 |         9.17x |
|    6 |  4096 x  4096 |    16 |        2.034 |        0.224 |      0.354 |       0.091 |         9.07x |
|    6 |  4096 x  4096 |   512 |        2.181 |        0.285 |      2.966 |       0.163 |         7.66x |
|    6 |  4096 x  4096 |  2048 |        2.300 |        0.482 |     10.559 |       0.423 |         4.77x |
|    6 |  4096 x 11008 |     1 |        4.550 |        0.284 |      0.613 |       0.083 |        16.04x |
|    6 |  4096 x 11008 |    16 |        4.548 |        0.285 |      0.632 |       0.084 |        15.98x |
|    6 |  4096 x 11008 |   512 |        4.675 |        0.410 |      7.272 |       0.304 |        11.41x |
|    6 |  4096 x 11008 |  2048 |        5.205 |        0.959 |     27.865 |       1.027 |         5.42x |
|    6 | 11008 x  4096 |     1 |        4.589 |        0.292 |      0.654 |       0.085 |        15.72x |
|    6 | 11008 x  4096 |    16 |        4.583 |        0.286 |      0.674 |       0.084 |        16.04x |
|    6 | 11008 x  4096 |   512 |        4.765 |        0.451 |      7.590 |       0.298 |        10.58x |
|    6 | 11008 x  4096 |  2048 |        5.283 |        0.993 |     29.633 |       0.980 |         5.32x |
|    7 |  4096 x  4096 |     1 |        2.697 |        0.213 |      0.449 |       0.084 |        12.66x |
|    7 |  4096 x  4096 |    16 |        2.698 |        0.218 |      0.455 |       0.083 |        12.37x |
|    7 |  4096 x  4096 |   512 |        2.753 |        0.274 |      4.401 |       0.157 |        10.03x |
|    7 |  4096 x  4096 |  2048 |        2.956 |        0.468 |     15.962 |       0.421 |         6.31x |
|    7 |  4096 x 11008 |     1 |        6.111 |        0.294 |      0.884 |       0.082 |        20.79x |
|    7 |  4096 x 11008 |    16 |        6.095 |        0.295 |      0.895 |       0.084 |        20.67x |
|    7 |  4096 x 11008 |   512 |        6.231 |        0.421 |     10.840 |       0.305 |        14.80x |
|    7 |  4096 x 11008 |  2048 |        6.752 |        0.945 |     41.710 |       1.027 |         7.14x |
|    7 | 11008 x  4096 |     1 |        6.141 |        0.302 |      0.958 |       0.086 |        20.33x |
|    7 | 11008 x  4096 |    16 |        6.133 |        0.298 |      0.971 |       0.086 |        20.58x |
|    7 | 11008 x  4096 |   512 |        6.286 |        0.445 |     11.276 |       0.298 |        14.13x |
|    7 | 11008 x  4096 |  2048 |        6.850 |        1.002 |     44.721 |       1.266 |         6.83x |

Interpretation vs round 1: the big-shape dequant path improved ~1.2-1.7x
(e.g. 7-bit 11008x4096 M=1: 0.484 -> 0.302 ms; 3-bit 11008x4096 M=16:
0.343 -> 0.279 ms) and speedups over torch-eager now reach 16-21x at
decode shapes. 4096x4096 M=1 is roughly flat (extra zeros-kernel launch
offsets the faster main kernel at the smallest shape). Remaining headroom
to Marlin is the dense fp16 round-trip, unchanged: still the native-tier
follow-up.

GPU tests after the rewrite: 64/64 passed at runtime under lease
(one intermediate failure recorded above: the constexpr-reassignment
compile error at 7-bit, fixed before landing).

## Round 3 (ncu-driven): fused decode-regime GEMV (`planar_gemv`, v5)

Goal: remove the dense fp16 weight round-trip entirely at decode shapes.
`planar_gemv` reuses the 32-row-block static-unroll decode from the v4
dequant kernel but accumulates `x[m,k] * w[k,n]` with broadcast fp32 FMAs
(no `tl.dot`, so BLOCK_M can be 1), split-K across SMs with fp32 atomic-add
combine. Weight-side DRAM traffic is only the packed words (`bits/8` B/elem).

ncu iterations (3-bit 4096x4096 M=1, A100-class PG506-230, 124 SM):

| version | change | duration | mem SOL | occupancy |
|---------|--------|----------|---------|-----------|
| gemv v1 | BLOCK_N=256, SPLIT_K=4 (grid 64) | 158.1 us | 9.9% | 6.3% |
| gemv v2 | deep split-K (BLOCK_N 128/256, SPLIT_K 16/32; grid 512) | 48.8 us | 40.2% | 24.9% |
| gemv v3 | GROUP_UNIFORM hoist: one scales+zeros vector per 32-block instead of 32 of each (canonical monotone g_idx, checked once and cached on the tensor; desc_act falls back to per-row loads) | 29.2 us | 42.5% | 24.6% |

Kernel-side the GEMV (29 us + 6 us zeros) now beats dequant+cuBLAS
(4 us zeros + 30.6 us dequant + GEMV matmul) at M=1. Wall-clock in the
eager microbenchmark is launch-overhead-bound (~0.15 ms/call), so the
end-to-end win only shows at the larger shapes:

| bits |         K x N |   M | tri-dequant | tri-gemv | verdict |
|------|---------------|-----|-------------|----------|---------|
|    3 |  4096 x  4096 |   1 |       0.205 |    0.229 | ~flat   |
|    3 | 11008 x  4096 |   1 |       0.275 |    0.230 | 1.20x   |
|    5 |  4096 x 11008 |   1 |       0.279 |    0.232 | 1.20x   |
|    6 |  4096 x 11008 |   1 |       0.279 |    0.225 | 1.24x   |
|    7 | 11008 x  4096 |   1 |       0.289 |    0.237 | 1.22x   |
|    7 |  4096 x 11008 |  16 |       0.289 |    0.470 | loses   |

(all values ms, fp16, 50-iter median; full table in the benchmark run below)

Routing: `GPTQMODEL_PLANAR_GEMV_MAX_M` (default 0 = opt-in, same policy as
the fused GEMM). At M=16 the broadcast-FMA accumulator loses to cuBLAS, and
at the smallest shape the extra launch cancels the kernel win, so the
default path stays dequant+cuBLAS; the GEMV is the right building block for
CUDA-graph/persistent decode serving where launch overhead amortizes.
No dense weight, zeros, or output caching across calls (the dense zeros
buffer stays transient per call).

GPU tests after adding the GEMV: 96/96 passed at runtime under lease
(32 new: `test_planar_gemv_matches_reference` fp16/bf16 x M in {1,4,16},
`test_planar_gemv_desc_act_sym` exercising the non-GROUP_UNIFORM branch).

Full benchmark rerun with tri-gemv column (fp16, 50-iter median ms, NVIDIA PG506-230 cc8.0 124SM, torch 2.13.0+cu130, triton 3.7.1):

| bits |         K x N |     M |  torch-eager |  tri-dequant |  tri-fused |   tri-gemv | marlin-4bit | best-vs-eager |
|------|---------------|-------|--------------|--------------|------------|------------|-------------|---------------|
|    3 |  4096 x  4096 |     1 |        1.959 |        0.205 |      0.334 |      0.229 |       0.082 |         9.57x |
|    3 |  4096 x  4096 |    16 |        1.944 |        0.208 |      0.336 |      0.321 |       0.081 |         9.35x |
|    3 |  4096 x  4096 |   512 |        1.998 |        0.263 |      2.891 |        n/a |       0.142 |         7.61x |
|    3 |  4096 x  4096 |  2048 |        2.186 |        0.441 |     10.353 |        n/a |       0.372 |         4.95x |
|    3 |  4096 x 11008 |     1 |        4.387 |        0.271 |      0.608 |      0.231 |       0.084 |        18.96x |
|    3 |  4096 x 11008 |    16 |        4.384 |        0.272 |      0.616 |      0.439 |       0.084 |        16.09x |
|    3 |  4096 x 11008 |   512 |        4.513 |        0.394 |      7.168 |        n/a |       0.270 |        11.45x |
|    3 |  4096 x 11008 |  2048 |        4.953 |        0.859 |     27.366 |        n/a |       0.904 |         5.77x |
|    3 | 11008 x  4096 |     1 |        4.399 |        0.275 |      0.646 |      0.230 |       0.084 |        19.13x |
|    3 | 11008 x  4096 |    16 |        4.392 |        0.271 |      0.656 |      0.455 |       0.083 |        16.18x |
|    3 | 11008 x  4096 |   512 |        4.526 |        0.403 |      7.408 |        n/a |       0.266 |        11.22x |
|    3 | 11008 x  4096 |  2048 |        5.017 |        0.904 |     28.947 |        n/a |       0.878 |         5.55x |
|    5 |  4096 x  4096 |     1 |        1.996 |        0.217 |      0.336 |      0.232 |       0.085 |         9.20x |
|    5 |  4096 x  4096 |    16 |        1.968 |        0.218 |      0.338 |      0.323 |       0.082 |         9.02x |
|    5 |  4096 x  4096 |   512 |        2.006 |        0.263 |      2.813 |        n/a |       0.143 |         7.62x |
|    5 |  4096 x  4096 |  2048 |        2.185 |        0.448 |     10.060 |        n/a |       0.372 |         4.88x |
|    5 |  4096 x 11008 |     1 |        4.427 |        0.279 |      0.606 |      0.232 |       0.084 |        19.09x |
|    5 |  4096 x 11008 |    16 |        4.422 |        0.280 |      0.615 |      0.447 |       0.083 |        15.82x |
|    5 |  4096 x 11008 |   512 |        4.547 |        0.395 |      6.986 |        n/a |       0.270 |        11.50x |
|    5 |  4096 x 11008 |  2048 |        5.011 |        0.868 |     26.611 |        n/a |       0.899 |         5.77x |
|    5 | 11008 x  4096 |     1 |        4.458 |        0.283 |      0.646 |      0.231 |       0.084 |        19.26x |
|    5 | 11008 x  4096 |    16 |        4.463 |        0.280 |      0.663 |      0.537 |       0.084 |        15.97x |
|    5 | 11008 x  4096 |   512 |        4.619 |        0.421 |      7.212 |        n/a |       0.266 |        10.97x |
|    5 | 11008 x  4096 |  2048 |        5.101 |        0.914 |     27.908 |        n/a |       0.911 |         5.58x |
|    6 |  4096 x  4096 |     1 |        1.961 |        0.208 |      0.301 |      0.231 |       0.085 |         9.43x |
|    6 |  4096 x  4096 |    16 |        1.960 |        0.215 |      0.306 |      0.326 |       0.082 |         9.14x |
|    6 |  4096 x  4096 |   512 |        2.016 |        0.265 |      2.553 |        n/a |       0.143 |         7.60x |
|    6 |  4096 x  4096 |  2048 |        2.204 |        0.446 |      9.063 |        n/a |       0.373 |         4.94x |
|    6 |  4096 x 11008 |     1 |        4.472 |        0.279 |      0.543 |      0.225 |       0.081 |        19.85x |
|    6 |  4096 x 11008 |    16 |        4.461 |        0.279 |      0.558 |      0.450 |       0.083 |        16.01x |
|    6 |  4096 x 11008 |   512 |        4.581 |        0.395 |      6.299 |        n/a |       0.270 |        11.59x |
|    6 |  4096 x 11008 |  2048 |        5.046 |        0.869 |     24.078 |        n/a |       0.910 |         5.80x |
|    6 | 11008 x  4096 |     1 |        4.493 |        0.288 |      0.582 |      0.237 |       0.085 |        19.00x |
|    6 | 11008 x  4096 |    16 |        4.489 |        0.283 |      0.590 |      0.476 |       0.088 |        15.88x |
|    6 | 11008 x  4096 |   512 |        4.621 |        0.415 |      6.557 |        n/a |       0.265 |        11.13x |
|    6 | 11008 x  4096 |  2048 |        5.111 |        0.909 |     25.494 |        n/a |       0.896 |         5.62x |
|    7 |  4096 x  4096 |     1 |        2.643 |        0.209 |      0.402 |      0.232 |       0.084 |        12.65x |
|    7 |  4096 x  4096 |    16 |        2.637 |        0.217 |      0.408 |      0.324 |       0.084 |        12.15x |
|    7 |  4096 x  4096 |   512 |        2.690 |        0.266 |      3.818 |        n/a |       0.145 |        10.10x |
|    7 |  4096 x  4096 |  2048 |        2.869 |        0.455 |     13.754 |        n/a |       0.375 |         6.31x |
|    7 |  4096 x 11008 |     1 |        6.030 |        0.288 |      0.783 |      0.243 |       0.086 |        24.85x |
|    7 |  4096 x 11008 |    16 |        6.022 |        0.289 |      0.795 |      0.470 |       0.086 |        20.85x |
|    7 |  4096 x 11008 |   512 |        6.154 |        0.407 |      9.373 |        n/a |       0.273 |        15.14x |
|    7 |  4096 x 11008 |  2048 |        6.611 |        0.880 |     36.006 |        n/a |       0.904 |         7.52x |
|    7 | 11008 x  4096 |     1 |        6.036 |        0.289 |      0.843 |      0.237 |       0.084 |        25.52x |
|    7 | 11008 x  4096 |    16 |        6.027 |        0.286 |      0.855 |      0.489 |       0.087 |        21.10x |
|    7 | 11008 x  4096 |   512 |        6.158 |        0.417 |      9.737 |        n/a |       0.265 |        14.76x |
|    7 | 11008 x  4096 |  2048 |        6.661 |        0.916 |     37.948 |        n/a |       0.903 |         7.27x |

## Round 4: native CUDA register-decode GEMV ("pangolin", `gptqmodel_ext/planar/`)

Goal: close the launch/scheduling overhead gap the Triton GEMV could not —
a Marlin-style native CUDA kernel with register-level plane decode.

Design (`planar_gemv_kernel.cu`, JIT-built via `TorchOpsJitExtension` as
`gptqmodel.utils.pangolin`, torch library namespace `gptqmodel_pangolin`):

- One warp per 32-column block per K-chunk; each lane owns one output column.
- Per 32-row logical block a lane loads exactly `bits` packed `int32` words
  (coalesced across the warp: adjacent lanes read adjacent columns) and
  decodes all 32 codes with compile-time shifts/masks (`PlaneSpec<Bits>` +
  `plane_code<Width>`, `if constexpr` for the optional third plane) —
  branch-free and uniform per lane.
- Activations are loaded once per warp and broadcast with `__shfl_sync`;
  weight-side DRAM traffic stays at `bits/8` B/elem, no dense fp16
  round-trip and no dense weight/zeros buffer at all (zeros decode in
  registers from the packed planar `qzeros` words).
- Split-K sized from SM count (`2*SMs / column_blocks`), fp32 `atomicAdd`
  combine; direct store when `gridDim.y == 1`. fp32 accumulation, output
  cast back to the input dtype (fp16/bf16 supported).
- Group metadata (scale/zero) is hoisted per 32-row block, which requires
  block-uniform `g_idx`; the Python router checks this once per tensor and
  falls back to the Triton paths for desc_act-style shuffled `g_idx`.

Failed attempt: template instantiation of `plane_code<0>` for the absent
third plane at 3/5/6-bit caused a compile-time division by zero; runtime
`if` is not enough, the guard must be `if constexpr (Spec::kW2 > 0)`.

Correctness: 72/72 standalone checks (bits 3/5/6/7 x fp16/bf16 x M 1/2/4 x
{256x128, 384x64, 4096x4096} x group 32/128) bit-compared against
`planar_unpack_rows/cols` dequant reference; pytest coverage added as
`test_pangolin_gemv_matches_reference` / `test_pangolin_gemv_sym_metadata`.

Benchmark (fp16, 50-iter median ms, NVIDIA PG506-230 cc8.0 124SM,
torch 2.13.0+cu130, CUDA 13.0):

| bits |         K x N |   M | tri-dequant | tri-gemv | pangolin | vs tri-gemv |
|------|---------------|-----|-------------|----------|-----------|-------------|
|    3 |  4096 x  4096 |   1 |       0.207 |    0.233 |     0.058 |       4.00x |
|    3 |  4096 x 11008 |   1 |       0.267 |    0.232 |     0.056 |       4.13x |
|    3 | 11008 x  4096 |   1 |       0.271 |    0.234 |     0.078 |       3.01x |
|    5 |  4096 x  4096 |   1 |       0.203 |    0.232 |     0.058 |       3.98x |
|    5 |  4096 x 11008 |   1 |       0.275 |    0.236 |     0.057 |       4.11x |
|    5 | 11008 x  4096 |   1 |       0.276 |    0.232 |     0.077 |       3.03x |
|    6 |  4096 x  4096 |   1 |       0.206 |    0.234 |     0.060 |       3.88x |
|    6 |  4096 x 11008 |   1 |       0.275 |    0.234 |     0.058 |       4.02x |
|    6 | 11008 x  4096 |   1 |       0.281 |    0.236 |     0.079 |       2.99x |
|    7 |  4096 x  4096 |   1 |       0.209 |    0.232 |     0.059 |       3.91x |
|    7 |  4096 x 11008 |   1 |       0.282 |    0.244 |     0.071 |       3.45x |
|    7 | 11008 x  4096 |   1 |       0.286 |    0.239 |     0.090 |       2.65x |

M=2/4 hold the same 2.4-4.1x band (full 36-row table in the benchmark
script output). This is a wall-clock 3.5-4.7x over the previous default
(tri-dequant + cuBLAS) at decode shapes and finally lands in Marlin's
latency neighborhood.

Routing: `TritonV2Linear._forward_pangolin` — enabled by default for
planar modules when M <= 4, CUDA fp16/bf16 with matching scales dtype,
block-uniform g_idx, and the JIT extension builds (cc >= 8.0 gate in
`pangolin_supported`); everything else falls back to the Triton paths
unchanged. Kill switch: `GPTQMODEL_PANGOLIN_DISABLE=1`.

Also fixed in this round (Devin Review): `planar_gemm_kernel` masked only
M and K, so an out_features divisible by 32 but not by BLOCK_N=64 read and
wrote past the end of scales/qzeros/qweight/output. Added `mask_n` to the
tile and output masks plus regression test
`test_planar_matmul_n_not_multiple_of_block` (N=160); and made the
`torch.utils.weak` dependency an explicit import.

ncu on the native kernel (3-bit 4096x4096 M=1, fp16):

| version | change | duration | DRAM SOL | compute SOL | grid |
|---------|--------|----------|----------|-------------|------|
| pangolin v1 | split-K target 2 blocks/SM | 26.1 us | 11.9% | 38.7% | (128, 2) |
| pangolin v2 | split-K target 8 blocks/SM | 18.6 us | 16.7% | 57.0% | (128, 8) |

The deeper split also flattens the shape imbalance: 11008x4096 M=1 dropped
0.078-0.098 -> 0.067-0.075 ms wall-clock across bits while 4096-deep shapes
stayed within noise. Remaining SOL headroom is latency-bound at this size
(0.06 ms wall vs 0.019 ms kernel = launch overhead dominates).

Full benchmark with the marlin ceiling (fp16, 50-iter median ms, same box):

| bits |         K x N | M | tri-dequant | tri-gemv | pangolin | marlin-4bit |
|------|---------------|---|-------------|----------|-----------|-------------|
|    3 |  4096 x  4096 | 1 |       0.211 |    0.244 |     0.063 |       0.084 |
|    3 |  4096 x 11008 | 1 |       0.274 |    0.244 |     0.059 |       0.084 |
|    3 | 11008 x  4096 | 1 |       0.281 |    0.236 |     0.080 |       0.083 |
|    5 |  4096 x  4096 | 1 |       0.214 |    0.251 |     0.065 |       0.083 |
|    5 |  4096 x 11008 | 1 |       0.287 |    0.244 |     0.061 |       0.087 |
|    5 | 11008 x  4096 | 1 |       0.400 |    0.248 |     0.080 |       0.084 |
|    6 |  4096 x  4096 | 1 |       0.215 |    0.250 |     0.065 |       0.083 |
|    6 |  4096 x 11008 | 1 |       0.285 |    0.246 |     0.061 |       0.084 |
|    6 | 11008 x  4096 | 1 |       0.290 |    0.253 |     0.083 |       0.087 |
|    7 |  4096 x  4096 | 1 |       0.217 |    0.247 |     0.065 |       0.084 |
|    7 |  4096 x 11008 | 1 |       0.307 |    0.259 |     0.072 |       0.085 |
|    7 | 11008 x  4096 | 1 |       0.301 |    0.260 |     0.098 |       0.087 |

(pangolin column from the pre-retune build; the v2 split retune brings the
11008x4096 rows to 0.067-0.075 ms.) At decode shapes the native planar GEMV
now matches or beats the 4-bit Marlin ceiling at 3/5/6/7 bits — the last
2-3x gap is closed. M=16+ still routes to tri-dequant + cuBLAS.

Tests after routing pangolin as the default decode path (M <= 4):
128/128 GPU tests passed at runtime under lease (32 new pangolin
correctness cases + the N=160 fused-GEMM regression), 146/146 CPU
regression tests passed with CUDA hidden.

### Pangolin v3 — fused split-K finalize (rename: planarlin -> Pangolin)

Wall-clock at M=1 was launch-bound: the op ran 3 kernels (`output.zero_()`,
gemv, `output.to(fp16)`). v3 fuses all of it into the single gemv launch:

- split-K accumulates into a transient fp32 workspace (allocated with a
  per-column-block completion counter appended, one `torch::zeros` call);
- after its atomicAdd each block does `__threadfence()` + an atomic counter
  bump; the last split-K block for a column block converts the workspace to
  the output dtype in-kernel (classic release/acquire finalize);
- the split_k == 1 path writes the output dtype directly.

Kernel time is unchanged within noise (18.6 -> 19.6 us ncu, the finalize
conversion moved inside), but wall-clock dropped ~20%:

| bits | K x N | M | v2 (3 launches) | v3 (fused) |
|------|-------|---|-----------------|------------|
| 3 | 4096x4096  | 1 | 0.062 | 0.050 |
| 3 | 4096x11008 | 1 | 0.066 | 0.063 |
| 3 | 11008x4096 | 1 | 0.067 | 0.063 |
| 5 | 4096x4096  | 1 | 0.060 | 0.052 |
| 6 | 4096x4096  | 1 | 0.063 | 0.052 |
| 7 | 4096x4096  | 1 | 0.062 | 0.051 |
| 7 | 11008x4096 | 1 | 0.075 | 0.075 |

(median CUDA-event ms; fp16; same A100-class box.) All 128 GPU tests
re-passed at runtime after the change. Remaining overhead is Python/dispatch
plus the workspace `torch::zeros`; a persistent-workspace or CUDA-graph
integration is the next step if decode serving needs more.

### Pangolin v4 — atomic-free split-K (per-slice workspace)

v3 still paid a `torch::zeros` on the full M x N workspace plus fp32 global
atomics on every partial. v4 removes both:

- workspace becomes `[split_k, M, N]`; each split-K block plain-stores its
  partial into its own slice (no atomics, no zero-init of the accumulator);
- only the tiny per-column-block counter tail (`N/32` ints) is cleared, via
  `cudaMemsetAsync` on the launch stream;
- the last-arriving block for a column block sums the `split_k` slices and
  converts to the output dtype (same threadfence + counter finalize as v3).

| bits | K x N | M | v3 | v4 |
|------|-------|---|----|----|
| 3 | 4096x4096  | 1 | 0.050 | 0.045 |
| 3 | 4096x11008 | 1 | 0.063 | 0.060 |
| 3 | 11008x4096 | 1 | 0.063 | 0.059 |
| 5 | 4096x4096  | 1 | 0.052 | 0.046 |
| 6 | 4096x4096  | 1 | 0.052 | 0.044 |
| 7 | 4096x4096  | 1 | 0.051 | 0.045 |
| 7 | 4096x11008 | 1 | 0.071 | 0.071 |
| 7 | 11008x4096 | 1 | 0.075 | 0.070 |

(median CUDA-event ms, fp16, same A100-class box.) Cumulative decode-path
history at 3-bit 4096x4096 M=1: tri-dequant 0.207 -> tri-gemv 0.233 ->
pangolin v1 0.058 -> v3 0.050 -> v4 0.045 ms, vs Marlin 4-bit ~0.083.
All 132 GPU tests re-passed at runtime (includes the new zero-row guard
coverage). Workspace is transient per call; no dense weights are cached.

## Round 5: format-aware validation, /32 auto-padding, Pangolin M<=8, Laguna S 2.1 shapes

### Format-aware validation (planar 3-bit vs continuous 3-bit)

`validate()`/`_validate()` were format-blind: `bits` alone cannot distinguish
planar 3-bit (`FORMAT.GPTQ_P`) from continuous 3-bit, so planar-only rules
(and the new auto-pad allowance) could leak onto continuous 3-bit modules.
Fixed by threading `format` through the whole chain:

- `BaseQuantLinear.validate/_validate`, `GroupedQuantLinear._validate`, and
  `GPTQQuantLinear._validate` accept `format` (Marlin/Humming overrides accept
  and ignore it);
- `GroupedQuantLinear.__init__` forwards `format` into constructor-time
  `validate_kwargs` so `BaseQuantLinear.__init__` validation sees it;
- kernel auto-selection (`utils/importer.py`, both dynamic-contract and
  single-select branches) and `utils/model.py` module construction now pass
  `format=format`;
- `TritonV2Linear.validate` applies its continuous 3-bit fused-path
  requirements only when `format != FORMAT.GPTQ_P`.

Planar classification everywhere is `bits in {5,6,7} or (bits == 3 and
format == FORMAT.GPTQ_P)`.

### /32 auto-padding at pack time (once), not per kernel call

Planar packing stores whole 32-code blocks, so non-/32 logical dims (Laguna S
2.1 attention heads 3072x48 and 3072x72) previously failed validation. Fixing
this inside kernels would add per-forward tail logic on every path; instead
the padding happens once on the packed side:

- `GPTQQuantLinear.__init__` computes `padded_in_features`/
  `padded_out_features` (next /32 multiple). K padding under `desc_act` is
  rejected (padded rows would reorder into real groups).
- `_register_gptq_buffers` allocates the packed buffers at padded K/N; the
  group count stays `ceil(logical_K / group_size)` and padded K rows join the
  last real group. `bias` stays logical.
- `pack_block`/`pack_gpu` pad the dense weight with zero rows/columns, scales
  with neutral 1.0, zero points with 0, and `g_idx` with the last real group
  before planar packing. Padded weights quantize to the group zero point, so
  padded rows and columns dequantize to exactly 0.0 (asserted in tests).
- forward pads the activation K transiently when `padded_in_features !=
  in_features` and slices the output back to logical N; Pangolin, Triton
  GEMV/GEMM/dequant, and the Torch fallback all keep seeing aligned shapes
  with zero per-call tail handling.

New GPU tests: `test_planar_auto_pad_forward` (3072x48, 3072x72, 112x64 =
K-pad; M=1/8/33; bits 3/5/6/7), `test_pangolin_auto_pad_routing` (padded
buffers route through the native GEMV), and
`test_continuous_3bit_not_planar` (format-aware classification regression).
All passed at runtime on the A100-class box (173/173 in the focused file).

### Pangolin M extension 4 -> 8

`kMaxM` raised to 8 with dispatch cases M=5..8 (register accumulators scale
linearly per row; still one column block per warp, split-K unchanged).
Runtime-validated bit-exact vs the CPU planar reference for bits 3/5/6/7 at
M=5/6/7/8 (max fp16 err <= 9.8e-4). `PANGOLIN_MAX_M = 8` in
`utils/pangolin.py` routes decode shapes up to M=8 to the native kernel.

### Laguna S 2.1 benchmark (M = 1/4/8/16/32/64)

`scripts/benchmark_planar_kernels.py --shapes laguna` benchmarks the 11
Laguna S 2.1 linear shapes (including non-/32 3072x48 / 3072x72) at
M=1,4,8,16,32,64; results below.

### Round 5 Laguna S 2.1 benchmark results (completed run)

Command: `scripts/benchmark_planar_kernels.py --shapes laguna --skip-fused --iters 30` (tri-fused skipped: its per-shape Triton GEMM compiles take minutes each on this box and it is opt-in/slower than tri-dequant everywhere).

```
device      : NVIDIA PG506-230 (cc 8.0, 124 SMs, 95.2 GiB)
torch       : 2.13.0+cu130 cuda 13.0
triton      : 3.7.1
dtype       : float16, iters=30 (median ms)
torch-eager/tri-dequant/tri-fused are per-call transient dequant (no dense weight caching).

| bits |         K x N |     M |  torch-eager |  tri-dequant |  tri-fused |   tri-gemv |   pangolin | marlin-4bit | best-vs-eager |
|------|---------------|-------|--------------|--------------|------------|------------|------------|-------------|---------------|
|    3 |  1024 x  3072 |     1 |        0.809 |        0.216 |        n/a |      0.252 |      0.049 |       0.088 |        16.63x |
|    3 |  1024 x  3072 |     4 |        0.834 |        0.215 |        n/a |      0.251 |      0.049 |       0.088 |        16.96x |
|    3 |  1024 x  3072 |     8 |        0.849 |        0.217 |        n/a |      0.249 |      0.049 |       0.087 |        17.45x |
|    3 |  1024 x  3072 |    16 |        0.840 |        0.215 |        n/a |      0.248 |        n/a |       0.087 |         3.91x |
|    3 |  1024 x  3072 |    32 |        0.826 |        0.213 |        n/a |        n/a |        n/a |       0.086 |         3.88x |
|    3 |  1024 x  3072 |    64 |        0.840 |        0.214 |        n/a |        n/a |        n/a |       0.086 |         3.92x |
|    3 |  3072 x    48 |     1 |        0.822 |        0.218 |        n/a |      0.245 |      0.047 |         n/a |        17.46x |
|    3 |  3072 x    48 |     4 |        0.829 |        0.219 |        n/a |      0.245 |      0.047 |         n/a |        17.61x |
|    3 |  3072 x    48 |     8 |        0.830 |        0.217 |        n/a |      0.244 |      0.047 |         n/a |        17.62x |
|    3 |  3072 x    48 |    16 |        0.827 |        0.218 |        n/a |      0.244 |        n/a |         n/a |         3.79x |
|    3 |  3072 x    48 |    32 |        0.822 |        0.217 |        n/a |        n/a |        n/a |         n/a |         3.79x |
|    3 |  3072 x    48 |    64 |        0.827 |        0.219 |        n/a |        n/a |        n/a |         n/a |         3.77x |
|    3 |  3072 x    72 |     1 |        0.817 |        0.219 |        n/a |      0.247 |      0.047 |         n/a |        17.35x |
|    3 |  3072 x    72 |     4 |        0.817 |        0.217 |        n/a |      0.249 |      0.048 |         n/a |        16.98x |
|    3 |  3072 x    72 |     8 |        0.828 |        0.220 |        n/a |      0.247 |      0.048 |         n/a |        17.40x |
|    3 |  3072 x    72 |    16 |        0.816 |        0.219 |        n/a |      0.252 |        n/a |         n/a |         3.73x |
|    3 |  3072 x    72 |    32 |        0.823 |        0.219 |        n/a |        n/a |        n/a |         n/a |         3.75x |
|    3 |  3072 x    72 |    64 |        0.821 |        0.218 |        n/a |        n/a |        n/a |         n/a |         3.77x |
|    3 |  3072 x  1024 |     1 |        0.806 |        0.218 |        n/a |      0.245 |      0.046 |       0.085 |        17.50x |
|    3 |  3072 x  1024 |     4 |        0.823 |        0.217 |        n/a |      0.247 |      0.047 |       0.085 |        17.47x |
|    3 |  3072 x  1024 |     8 |        0.833 |        0.221 |        n/a |      0.246 |      0.047 |       0.085 |        17.87x |
|    3 |  3072 x  1024 |    16 |        0.822 |        0.217 |        n/a |      0.249 |        n/a |       0.084 |         3.79x |
|    3 |  3072 x  1024 |    32 |        0.825 |        0.218 |        n/a |        n/a |        n/a |       0.086 |         3.79x |
|    3 |  3072 x  1024 |    64 |        0.826 |        0.214 |        n/a |        n/a |        n/a |       0.097 |         3.87x |
|    3 |  3072 x  6144 |     1 |        2.149 |        0.215 |        n/a |      0.255 |      0.047 |       0.084 |        45.63x |
|    3 |  3072 x  6144 |     4 |        2.140 |        0.214 |        n/a |      0.254 |      0.052 |       0.086 |        40.98x |
|    3 |  3072 x  6144 |     8 |        2.139 |        0.216 |        n/a |      0.265 |      0.070 |       0.084 |        30.72x |
|    3 |  3072 x  6144 |    16 |        2.165 |        0.212 |        n/a |      0.337 |        n/a |       0.088 |        10.21x |
|    3 |  3072 x  6144 |    32 |        2.137 |        0.212 |        n/a |        n/a |        n/a |       0.084 |        10.08x |
|    3 |  3072 x  6144 |    64 |        2.146 |        0.214 |        n/a |        n/a |        n/a |       0.084 |        10.05x |
|    3 |  3072 x  9216 |     1 |        2.945 |        0.230 |        n/a |      0.247 |      0.049 |       0.084 |        60.55x |
|    3 |  3072 x  9216 |     4 |        2.936 |        0.231 |        n/a |      0.246 |      0.059 |       0.085 |        49.43x |
|    3 |  3072 x  9216 |     8 |        2.941 |        0.230 |        n/a |      0.289 |      0.083 |       0.084 |        35.46x |
|    3 |  3072 x  9216 |    16 |        2.937 |        0.230 |        n/a |      0.385 |        n/a |       0.084 |        12.75x |
|    3 |  3072 x  9216 |    32 |        2.945 |        0.231 |        n/a |        n/a |        n/a |       0.084 |        12.73x |
|    3 |  3072 x  9216 |    64 |        2.945 |        0.234 |        n/a |        n/a |        n/a |       0.087 |        12.58x |
|    3 |  3072 x 12288 |     1 |        3.750 |        0.259 |        n/a |      0.253 |      0.055 |       0.087 |        67.81x |
|    3 |  3072 x 12288 |     4 |        3.761 |        0.261 |        n/a |      0.262 |      0.071 |       0.088 |        53.22x |
|    3 |  3072 x 12288 |     8 |        3.757 |        0.261 |        n/a |      0.323 |      0.101 |       0.084 |        37.06x |
|    3 |  3072 x 12288 |    16 |        3.748 |        0.256 |        n/a |      0.440 |        n/a |       0.085 |        14.64x |
|    3 |  3072 x 12288 |    32 |        3.742 |        0.259 |        n/a |        n/a |        n/a |       0.084 |        14.47x |
|    3 |  3072 x 12288 |    64 |        3.763 |        0.259 |        n/a |        n/a |        n/a |       0.091 |        14.52x |
|    3 |  6144 x  3072 |     1 |        2.147 |        0.222 |        n/a |      0.249 |      0.047 |       0.084 |        45.59x |
|    3 |  6144 x  3072 |     4 |        2.148 |        0.218 |        n/a |      0.249 |      0.052 |       0.084 |        41.53x |
|    3 |  6144 x  3072 |     8 |        2.147 |        0.219 |        n/a |      0.268 |      0.072 |       0.085 |        29.96x |
|    3 |  6144 x  3072 |    16 |        2.145 |        0.218 |        n/a |      0.323 |        n/a |       0.086 |         9.86x |
|    3 |  6144 x  3072 |    32 |        2.146 |        0.215 |        n/a |        n/a |        n/a |       0.083 |         9.98x |
|    3 |  6144 x  3072 |    64 |        2.144 |        0.218 |        n/a |        n/a |        n/a |       0.088 |         9.83x |
|    3 |  9216 x  3072 |     1 |        2.970 |        0.244 |        n/a |      0.254 |      0.050 |       0.085 |        59.19x |
|    3 |  9216 x  3072 |     4 |        2.966 |        0.244 |        n/a |      0.252 |      0.062 |       0.086 |        47.48x |
|    3 |  9216 x  3072 |     8 |        2.977 |        0.242 |        n/a |      0.296 |      0.088 |       0.085 |        33.80x |
|    3 |  9216 x  3072 |    16 |        2.965 |        0.245 |        n/a |      0.379 |        n/a |       0.088 |        12.12x |
|    3 |  9216 x  3072 |    32 |        2.980 |        0.249 |        n/a |        n/a |        n/a |       0.088 |        11.98x |
|    3 |  9216 x  3072 |    64 |        2.965 |        0.244 |        n/a |        n/a |        n/a |       0.096 |        12.17x |
|    3 | 12288 x  3072 |     1 |        3.798 |        0.275 |        n/a |      0.247 |      0.055 |       0.084 |        68.68x |
|    3 | 12288 x  3072 |     4 |        3.789 |        0.272 |        n/a |      0.261 |      0.072 |       0.086 |        52.86x |
|    3 | 12288 x  3072 |     8 |        3.799 |        0.274 |        n/a |      0.323 |      0.105 |       0.086 |        36.02x |
|    3 | 12288 x  3072 |    16 |        3.778 |        0.269 |        n/a |      0.424 |        n/a |       0.087 |        14.03x |
|    3 | 12288 x  3072 |    32 |        3.789 |        0.272 |        n/a |        n/a |        n/a |       0.088 |        13.94x |
|    3 | 12288 x  3072 |    64 |        3.794 |        0.274 |        n/a |        n/a |        n/a |       0.102 |        13.82x |
|    3 |  3072 x   256 |     1 |        0.816 |        0.216 |        n/a |      0.245 |      0.046 |       0.086 |        17.91x |
|    3 |  3072 x   256 |     4 |        0.805 |        0.218 |        n/a |      0.250 |      0.047 |       0.092 |        17.09x |
|    3 |  3072 x   256 |     8 |        0.831 |        0.223 |        n/a |      0.246 |      0.047 |       0.088 |        17.64x |
|    3 |  3072 x   256 |    16 |        0.819 |        0.220 |        n/a |      0.244 |        n/a |       0.091 |         3.72x |
|    3 |  3072 x   256 |    32 |        0.820 |        0.220 |        n/a |        n/a |        n/a |       0.101 |         3.73x |
|    3 |  3072 x   256 |    64 |        0.817 |        0.219 |        n/a |        n/a |        n/a |       0.127 |         3.74x |
|    5 |  1024 x  3072 |     1 |        0.796 |        0.214 |        n/a |      0.244 |      0.045 |       0.084 |        17.67x |
|    5 |  1024 x  3072 |     4 |        0.802 |        0.214 |        n/a |      0.245 |      0.045 |       0.083 |        17.80x |
|    5 |  1024 x  3072 |     8 |        0.799 |        0.213 |        n/a |      0.244 |      0.046 |       0.084 |        17.33x |
|    5 |  1024 x  3072 |    16 |        0.795 |        0.211 |        n/a |      0.246 |        n/a |       0.084 |         3.77x |
|    5 |  1024 x  3072 |    32 |        0.800 |        0.211 |        n/a |        n/a |        n/a |       0.084 |         3.78x |
|    5 |  1024 x  3072 |    64 |        0.799 |        0.214 |        n/a |        n/a |        n/a |       0.086 |         3.74x |
|    5 |  3072 x    48 |     1 |        0.802 |        0.217 |        n/a |      0.246 |      0.045 |         n/a |        17.81x |
|    5 |  3072 x    48 |     4 |        0.798 |        0.217 |        n/a |      0.245 |      0.045 |         n/a |        17.70x |
|    5 |  3072 x    48 |     8 |        0.795 |        0.216 |        n/a |      0.245 |      0.045 |         n/a |        17.64x |
|    5 |  3072 x    48 |    16 |        0.800 |        0.220 |        n/a |      0.246 |        n/a |         n/a |         3.63x |
|    5 |  3072 x    48 |    32 |        0.798 |        0.218 |        n/a |        n/a |        n/a |         n/a |         3.67x |
|    5 |  3072 x    48 |    64 |        0.804 |        0.219 |        n/a |        n/a |        n/a |         n/a |         3.68x |
|    5 |  3072 x    72 |     1 |        0.796 |        0.213 |        n/a |      0.243 |      0.045 |         n/a |        17.66x |
|    5 |  3072 x    72 |     4 |        0.798 |        0.215 |        n/a |      0.245 |      0.046 |         n/a |        17.31x |
|    5 |  3072 x    72 |     8 |        0.799 |        0.216 |        n/a |      0.247 |      0.047 |         n/a |        16.96x |
|    5 |  3072 x    72 |    16 |        0.808 |        0.216 |        n/a |      0.247 |        n/a |         n/a |         3.75x |
|    5 |  3072 x    72 |    32 |        0.815 |        0.221 |        n/a |        n/a |        n/a |         n/a |         3.69x |
|    5 |  3072 x    72 |    64 |        0.815 |        0.218 |        n/a |        n/a |        n/a |         n/a |         3.74x |
|    5 |  3072 x  1024 |     1 |        0.805 |        0.215 |        n/a |      0.245 |      0.046 |       0.084 |        17.47x |
|    5 |  3072 x  1024 |     4 |        0.807 |        0.221 |        n/a |      0.243 |      0.047 |       0.083 |        17.13x |
|    5 |  3072 x  1024 |     8 |        0.806 |        0.220 |        n/a |      0.247 |      0.047 |       0.084 |        17.12x |
|    5 |  3072 x  1024 |    16 |        0.806 |        0.220 |        n/a |      0.244 |        n/a |       0.084 |         3.67x |
|    5 |  3072 x  1024 |    32 |        0.803 |        0.217 |        n/a |        n/a |        n/a |       0.085 |         3.70x |
|    5 |  3072 x  1024 |    64 |        0.801 |        0.211 |        n/a |        n/a |        n/a |       0.097 |         3.79x |
|    5 |  3072 x  6144 |     1 |        2.134 |        0.217 |        n/a |      0.246 |      0.046 |       0.083 |        46.30x |
|    5 |  3072 x  6144 |     4 |        2.134 |        0.215 |        n/a |      0.247 |      0.051 |       0.084 |        41.68x |
|    5 |  3072 x  6144 |     8 |        2.129 |        0.215 |        n/a |      0.262 |      0.070 |       0.083 |        30.58x |
|    5 |  3072 x  6144 |    16 |        2.129 |        0.214 |        n/a |      0.333 |        n/a |       0.083 |         9.95x |
|    5 |  3072 x  6144 |    32 |        2.132 |        0.213 |        n/a |        n/a |        n/a |       0.085 |        10.01x |
|    5 |  3072 x  6144 |    64 |        2.132 |        0.215 |        n/a |        n/a |        n/a |       0.085 |         9.91x |
|    5 |  3072 x  9216 |     1 |        2.973 |        0.240 |        n/a |      0.245 |      0.047 |       0.084 |        63.12x |
|    5 |  3072 x  9216 |     4 |        2.963 |        0.241 |        n/a |      0.247 |      0.060 |       0.084 |        49.04x |
|    5 |  3072 x  9216 |     8 |        2.971 |        0.238 |        n/a |      0.291 |      0.083 |       0.083 |        35.81x |
|    5 |  3072 x  9216 |    16 |        2.959 |        0.237 |        n/a |      0.385 |        n/a |       0.084 |        12.51x |
|    5 |  3072 x  9216 |    32 |        2.970 |        0.238 |        n/a |        n/a |        n/a |       0.086 |        12.50x |
|    5 |  3072 x  9216 |    64 |        2.963 |        0.239 |        n/a |        n/a |        n/a |       0.087 |        12.42x |
|    5 |  3072 x 12288 |     1 |        3.800 |        0.265 |        n/a |      0.252 |      0.055 |       0.087 |        68.71x |
|    5 |  3072 x 12288 |     4 |        3.802 |        0.266 |        n/a |      0.261 |      0.071 |       0.087 |        53.81x |
|    5 |  3072 x 12288 |     8 |        3.803 |        0.264 |        n/a |      0.330 |      0.101 |       0.087 |        37.51x |
|    5 |  3072 x 12288 |    16 |        3.808 |        0.266 |        n/a |      0.449 |        n/a |       0.088 |        14.30x |
|    5 |  3072 x 12288 |    32 |        3.805 |        0.267 |        n/a |        n/a |        n/a |       0.088 |        14.24x |
|    5 |  3072 x 12288 |    64 |        3.812 |        0.268 |        n/a |        n/a |        n/a |       0.095 |        14.24x |
|    5 |  6144 x  3072 |     1 |        2.152 |        0.219 |        n/a |      0.243 |      0.046 |       0.083 |        46.70x |
|    5 |  6144 x  3072 |     4 |        2.230 |        0.219 |        n/a |      0.257 |      0.051 |       0.083 |        43.56x |
|    5 |  6144 x  3072 |     8 |        2.189 |        0.217 |        n/a |      0.264 |      0.071 |       0.083 |        30.99x |
|    5 |  6144 x  3072 |    16 |        2.157 |        0.217 |        n/a |      0.325 |        n/a |       0.083 |         9.94x |
|    5 |  6144 x  3072 |    32 |        2.154 |        0.218 |        n/a |        n/a |        n/a |       0.083 |         9.88x |
|    5 |  6144 x  3072 |    64 |        2.162 |        0.219 |        n/a |        n/a |        n/a |       0.088 |         9.86x |
|    5 |  9216 x  3072 |     1 |        2.986 |        0.244 |        n/a |      0.247 |      0.047 |       0.084 |        63.40x |
|    5 |  9216 x  3072 |     4 |        3.017 |        0.246 |        n/a |      0.254 |      0.060 |       0.084 |        49.94x |
|    5 |  9216 x  3072 |     8 |        2.977 |        0.253 |        n/a |      0.293 |      0.086 |       0.086 |        34.61x |
|    5 |  9216 x  3072 |    16 |        2.997 |        0.245 |        n/a |      0.376 |        n/a |       0.097 |        12.22x |
|    5 |  9216 x  3072 |    32 |        2.992 |        0.246 |        n/a |        n/a |        n/a |       0.091 |        12.17x |
|    5 |  9216 x  3072 |    64 |        2.986 |        0.245 |        n/a |        n/a |        n/a |       0.095 |        12.20x |
|    5 | 12288 x  3072 |     1 |        3.831 |        0.274 |        n/a |      0.254 |      0.056 |       0.088 |        68.03x |
|    5 | 12288 x  3072 |     4 |        3.839 |        0.276 |        n/a |      0.269 |      0.072 |       0.087 |        53.56x |
|    5 | 12288 x  3072 |     8 |        3.833 |        0.279 |        n/a |      0.327 |      0.104 |       0.089 |        36.70x |
|    5 | 12288 x  3072 |    16 |        3.839 |        0.279 |        n/a |      0.431 |        n/a |       0.087 |        13.78x |
|    5 | 12288 x  3072 |    32 |        3.835 |        0.276 |        n/a |        n/a |        n/a |       0.088 |        13.90x |
|    5 | 12288 x  3072 |    64 |        3.835 |        0.280 |        n/a |        n/a |        n/a |       0.103 |        13.69x |
|    5 |  3072 x   256 |     1 |        0.831 |        0.222 |        n/a |      0.252 |      0.049 |       0.089 |        16.92x |
|    5 |  3072 x   256 |     4 |        0.855 |        0.226 |        n/a |      0.253 |      0.049 |       0.096 |        17.39x |
|    5 |  3072 x   256 |     8 |        0.850 |        0.228 |        n/a |      0.250 |      0.049 |       0.089 |        17.30x |
|    5 |  3072 x   256 |    16 |        0.838 |        0.224 |        n/a |      0.252 |        n/a |       0.093 |         3.74x |
|    5 |  3072 x   256 |    32 |        0.849 |        0.226 |        n/a |        n/a |        n/a |       0.103 |         3.75x |
|    5 |  3072 x   256 |    64 |        0.850 |        0.223 |        n/a |        n/a |        n/a |       0.130 |         3.81x |
|    6 |  1024 x  3072 |     1 |        0.847 |        0.211 |        n/a |      0.247 |      0.047 |       0.085 |        17.99x |
|    6 |  1024 x  3072 |     4 |        0.806 |        0.209 |        n/a |      0.247 |      0.048 |       0.085 |        16.94x |
|    6 |  1024 x  3072 |     8 |        0.801 |        0.209 |        n/a |      0.246 |      0.048 |       0.085 |        16.64x |
|    6 |  1024 x  3072 |    16 |        0.805 |        0.208 |        n/a |      0.246 |        n/a |       0.085 |         3.87x |
|    6 |  1024 x  3072 |    32 |        0.802 |        0.212 |        n/a |        n/a |        n/a |       0.084 |         3.78x |
|    6 |  1024 x  3072 |    64 |        0.803 |        0.209 |        n/a |        n/a |        n/a |       0.084 |         3.83x |
|    6 |  3072 x    48 |     1 |        0.795 |        0.214 |        n/a |      0.243 |      0.047 |         n/a |        16.87x |
|    6 |  3072 x    48 |     4 |        0.806 |        0.216 |        n/a |      0.244 |      0.046 |         n/a |        17.49x |
|    6 |  3072 x    48 |     8 |        0.797 |        0.216 |        n/a |      0.249 |      0.047 |         n/a |        16.91x |
|    6 |  3072 x    48 |    16 |        0.808 |        0.221 |        n/a |      0.244 |        n/a |         n/a |         3.66x |
|    6 |  3072 x    48 |    32 |        0.805 |        0.217 |        n/a |        n/a |        n/a |         n/a |         3.71x |
|    6 |  3072 x    48 |    64 |        0.805 |        0.218 |        n/a |        n/a |        n/a |         n/a |         3.69x |
|    6 |  3072 x    72 |     1 |        0.795 |        0.212 |        n/a |      0.239 |      0.045 |         n/a |        17.64x |
|    6 |  3072 x    72 |     4 |        0.786 |        0.212 |        n/a |      0.240 |      0.046 |         n/a |        17.26x |
|    6 |  3072 x    72 |     8 |        0.795 |        0.214 |        n/a |      0.241 |      0.045 |         n/a |        17.64x |
|    6 |  3072 x    72 |    16 |        0.794 |        0.216 |        n/a |      0.242 |        n/a |         n/a |         3.68x |
|    6 |  3072 x    72 |    32 |        0.792 |        0.216 |        n/a |        n/a |        n/a |         n/a |         3.67x |
|    6 |  3072 x    72 |    64 |        0.802 |        0.217 |        n/a |        n/a |        n/a |         n/a |         3.70x |
|    6 |  3072 x  1024 |     1 |        0.824 |        0.219 |        n/a |      0.245 |      0.045 |       0.083 |        18.30x |
|    6 |  3072 x  1024 |     4 |        0.825 |        0.217 |        n/a |      0.244 |      0.046 |       0.085 |        17.90x |
|    6 |  3072 x  1024 |     8 |        0.821 |        0.217 |        n/a |      0.245 |      0.047 |       0.084 |        17.43x |
|    6 |  3072 x  1024 |    16 |        0.828 |        0.219 |        n/a |      0.247 |        n/a |       0.085 |         3.78x |
|    6 |  3072 x  1024 |    32 |        0.820 |        0.215 |        n/a |        n/a |        n/a |       0.086 |         3.82x |
|    6 |  3072 x  1024 |    64 |        0.816 |        0.215 |        n/a |        n/a |        n/a |       0.096 |         3.80x |
|    6 |  3072 x  6144 |     1 |        2.152 |        0.218 |        n/a |      2.125 |      0.402 |       0.748 |         9.87x |
|    6 |  3072 x  6144 |     4 |        6.811 |        1.844 |        n/a |      0.251 |      0.052 |       0.084 |       130.42x |
|    6 |  3072 x  6144 |     8 |        2.162 |        0.219 |        n/a |      0.261 |      0.070 |       0.084 |        31.04x |
|    6 |  3072 x  6144 |    16 |        2.151 |        0.216 |        n/a |      0.336 |        n/a |       0.086 |         9.96x |
|    6 |  3072 x  6144 |    32 |        2.153 |        0.215 |        n/a |        n/a |        n/a |       0.086 |        10.04x |
|    6 |  3072 x  6144 |    64 |        2.151 |        0.216 |        n/a |        n/a |        n/a |       0.086 |         9.98x |
|    6 |  3072 x  9216 |     1 |        3.035 |        0.241 |        n/a |      0.254 |      0.048 |       0.085 |        63.06x |
|    6 |  3072 x  9216 |     4 |        2.999 |        0.239 |        n/a |      0.248 |      0.061 |       0.085 |        49.22x |
|    6 |  3072 x  9216 |     8 |        3.015 |        0.238 |        n/a |      0.292 |      0.083 |       0.086 |        36.35x |
|    6 |  3072 x  9216 |    16 |        2.994 |        0.237 |        n/a |      0.386 |        n/a |       0.087 |        12.63x |
|    6 |  3072 x  9216 |    32 |        2.993 |        0.237 |        n/a |        n/a |        n/a |       0.086 |        12.65x |
|    6 |  3072 x  9216 |    64 |        3.002 |        0.240 |        n/a |        n/a |        n/a |       0.087 |        12.53x |
|    6 |  3072 x 12288 |     1 |        3.836 |        0.266 |        n/a |      0.247 |      0.056 |       0.085 |        68.11x |
|    6 |  3072 x 12288 |     4 |        3.836 |        0.266 |        n/a |      0.257 |      0.072 |       0.084 |        53.52x |
|    6 |  3072 x 12288 |     8 |        3.842 |        0.266 |        n/a |      0.330 |      0.102 |       0.085 |        37.52x |
|    6 |  3072 x 12288 |    16 |        3.839 |        0.265 |        n/a |      0.445 |        n/a |       0.084 |        14.48x |
|    6 |  3072 x 12288 |    32 |        3.835 |        0.268 |        n/a |        n/a |        n/a |       0.085 |        14.29x |
|    6 |  3072 x 12288 |    64 |        3.836 |        0.267 |        n/a |        n/a |        n/a |       0.093 |        14.35x |
|    6 |  6144 x  3072 |     1 |        2.175 |        0.226 |        n/a |      0.252 |      0.049 |       0.086 |        44.25x |
|    6 |  6144 x  3072 |     4 |        2.192 |        0.229 |        n/a |      0.262 |      0.054 |       0.089 |        40.40x |
|    6 |  6144 x  3072 |     8 |        2.185 |        0.224 |        n/a |      0.263 |      0.072 |       0.086 |        30.48x |
|    6 |  6144 x  3072 |    16 |        2.172 |        0.224 |        n/a |      0.324 |        n/a |       0.085 |         9.68x |
|    6 |  6144 x  3072 |    32 |        2.164 |        0.223 |        n/a |        n/a |        n/a |       0.086 |         9.72x |
|    6 |  6144 x  3072 |    64 |        2.164 |        0.219 |        n/a |        n/a |        n/a |       0.090 |         9.87x |
|    6 |  9216 x  3072 |     1 |        3.014 |        0.244 |        n/a |      0.248 |      0.048 |       0.084 |        62.62x |
|    6 |  9216 x  3072 |     4 |        3.012 |        0.246 |        n/a |      0.249 |      0.061 |       0.086 |        49.02x |
|    6 |  9216 x  3072 |     8 |        3.014 |        0.244 |        n/a |      0.291 |      0.086 |       0.085 |        35.04x |
|    6 |  9216 x  3072 |    16 |        3.023 |        0.246 |        n/a |      0.374 |        n/a |       0.084 |        12.28x |
|    6 |  9216 x  3072 |    32 |        3.003 |        0.246 |        n/a |        n/a |        n/a |       0.086 |        12.22x |
|    6 |  9216 x  3072 |    64 |        3.007 |        0.246 |        n/a |        n/a |        n/a |       0.094 |        12.24x |
|    6 | 12288 x  3072 |     1 |        3.851 |        0.268 |        n/a |      0.245 |      0.055 |       0.084 |        69.65x |
|    6 | 12288 x  3072 |     4 |        3.852 |        0.273 |        n/a |      0.266 |      0.073 |       0.084 |        52.99x |
|    6 | 12288 x  3072 |     8 |        3.850 |        0.273 |        n/a |      0.318 |      0.104 |       0.083 |        36.86x |
|    6 | 12288 x  3072 |    16 |        3.855 |        0.273 |        n/a |      0.435 |        n/a |       0.085 |        14.13x |
|    6 | 12288 x  3072 |    32 |        3.868 |        0.275 |        n/a |        n/a |        n/a |       0.085 |        14.07x |
|    6 | 12288 x  3072 |    64 |        3.855 |        0.273 |        n/a |        n/a |        n/a |       0.100 |        14.13x |
|    6 |  3072 x   256 |     1 |        0.809 |        0.214 |        n/a |      0.246 |      0.047 |       0.085 |        17.17x |
|    6 |  3072 x   256 |     4 |        0.820 |        0.218 |        n/a |      0.245 |      0.047 |       0.093 |        17.59x |
|    6 |  3072 x   256 |     8 |        0.814 |        0.223 |        n/a |      0.247 |      0.047 |       0.088 |        17.28x |
|    6 |  3072 x   256 |    16 |        0.806 |        0.217 |        n/a |      0.247 |        n/a |       0.092 |         3.71x |
|    6 |  3072 x   256 |    32 |        0.813 |        0.216 |        n/a |        n/a |        n/a |       0.101 |         3.76x |
|    6 |  3072 x   256 |    64 |        0.815 |        0.217 |        n/a |        n/a |        n/a |       0.126 |         3.75x |
|    7 |  1024 x  3072 |     1 |        1.016 |        0.211 |        n/a |      0.249 |      0.047 |       0.085 |        21.57x |
|    7 |  1024 x  3072 |     4 |        1.023 |        0.211 |        n/a |      0.251 |      0.046 |       0.084 |        22.21x |
|    7 |  1024 x  3072 |     8 |        1.016 |        0.212 |        n/a |      0.258 |      0.047 |       0.085 |        21.57x |
|    7 |  1024 x  3072 |    16 |        1.054 |        0.212 |        n/a |      0.252 |        n/a |       0.087 |         4.97x |
|    7 |  1024 x  3072 |    32 |        1.033 |        0.212 |        n/a |        n/a |        n/a |       0.085 |         4.86x |
|    7 |  1024 x  3072 |    64 |        1.031 |        0.211 |        n/a |        n/a |        n/a |       0.084 |         4.87x |
|    7 |  3072 x    48 |     1 |        1.015 |        0.213 |        n/a |      0.248 |      0.047 |         n/a |        21.55x |
|    7 |  3072 x    48 |     4 |        1.015 |        0.216 |        n/a |      0.249 |      0.047 |         n/a |        21.55x |
|    7 |  3072 x    48 |     8 |        1.018 |        0.216 |        n/a |      0.244 |      0.047 |         n/a |        21.61x |
|    7 |  3072 x    48 |    16 |        1.004 |        0.218 |        n/a |      0.252 |        n/a |         n/a |         4.61x |
|    7 |  3072 x    48 |    32 |        1.030 |        0.222 |        n/a |        n/a |        n/a |         n/a |         4.63x |
|    7 |  3072 x    48 |    64 |        1.029 |        0.218 |        n/a |        n/a |        n/a |         n/a |         4.73x |
|    7 |  3072 x    72 |     1 |        0.998 |        0.215 |        n/a |      0.246 |      0.046 |         n/a |        21.67x |
|    7 |  3072 x    72 |     4 |        1.018 |        0.214 |        n/a |      0.243 |      0.047 |         n/a |        21.62x |
|    7 |  3072 x    72 |     8 |        0.999 |        0.214 |        n/a |      0.246 |      0.047 |         n/a |        21.22x |
|    7 |  3072 x    72 |    16 |        1.003 |        0.215 |        n/a |      0.248 |        n/a |         n/a |         4.68x |
|    7 |  3072 x    72 |    32 |        1.025 |        0.219 |        n/a |        n/a |        n/a |         n/a |         4.68x |
|    7 |  3072 x    72 |    64 |        1.022 |        0.217 |        n/a |        n/a |        n/a |         n/a |         4.71x |
|    7 |  3072 x  1024 |     1 |        1.015 |        0.217 |        n/a |      0.245 |      0.048 |       0.085 |        21.09x |
|    7 |  3072 x  1024 |     4 |        1.009 |        0.217 |        n/a |      0.251 |      0.046 |       0.084 |        21.89x |
|    7 |  3072 x  1024 |     8 |        0.997 |        0.219 |        n/a |      0.248 |      0.047 |       0.085 |        21.16x |
|    7 |  3072 x  1024 |    16 |        1.010 |        0.220 |        n/a |      0.248 |        n/a |       0.086 |         4.59x |
|    7 |  3072 x  1024 |    32 |        1.019 |        0.217 |        n/a |        n/a |        n/a |       0.086 |         4.69x |
|    7 |  3072 x  1024 |    64 |        1.025 |        0.216 |        n/a |        n/a |        n/a |       0.097 |         4.75x |
|    7 |  3072 x  6144 |     1 |        2.892 |        0.221 |        n/a |      0.250 |      0.048 |       0.086 |        60.09x |
|    7 |  3072 x  6144 |     4 |        2.872 |        0.215 |        n/a |      0.253 |      0.055 |       0.086 |        51.94x |
|    7 |  3072 x  6144 |     8 |        2.883 |        0.218 |        n/a |      0.267 |      0.071 |       0.086 |        40.51x |
|    7 |  3072 x  6144 |    16 |        2.897 |        0.218 |        n/a |      0.342 |        n/a |       0.085 |        13.28x |
|    7 |  3072 x  6144 |    32 |        2.888 |        0.220 |        n/a |        n/a |        n/a |       0.086 |        13.15x |
|    7 |  3072 x  6144 |    64 |        2.896 |        0.219 |        n/a |        n/a |        n/a |       0.086 |        13.21x |
|    7 |  3072 x  9216 |     1 |        4.034 |        0.243 |        n/a |      0.258 |      0.053 |       0.085 |        75.75x |
|    7 |  3072 x  9216 |     4 |        4.009 |        0.241 |        n/a |      0.253 |      0.065 |       0.086 |        62.15x |
|    7 |  3072 x  9216 |     8 |        4.015 |        0.243 |        n/a |      0.298 |      0.086 |       0.086 |        46.67x |
|    7 |  3072 x  9216 |    16 |        4.015 |        0.243 |        n/a |      0.391 |        n/a |       0.085 |        16.51x |
|    7 |  3072 x  9216 |    32 |        4.019 |        0.242 |        n/a |        n/a |        n/a |       0.086 |        16.63x |
|    7 |  3072 x  9216 |    64 |        4.014 |        0.241 |        n/a |        n/a |        n/a |       0.086 |        16.68x |
|    7 |  3072 x 12288 |     1 |        5.129 |        0.269 |        n/a |      0.255 |      0.061 |       0.082 |        83.48x |
|    7 |  3072 x 12288 |     4 |        5.129 |        0.269 |        n/a |      0.264 |      0.078 |       0.083 |        65.91x |
|    7 |  3072 x 12288 |     8 |        5.127 |        0.267 |        n/a |      0.343 |      0.104 |       0.084 |        49.08x |
|    7 |  3072 x 12288 |    16 |        5.119 |        0.269 |        n/a |      0.460 |        n/a |       0.084 |        19.05x |
|    7 |  3072 x 12288 |    32 |        5.124 |        0.271 |        n/a |        n/a |        n/a |       0.084 |        18.88x |
|    7 |  3072 x 12288 |    64 |        5.128 |        0.269 |        n/a |        n/a |        n/a |       0.091 |        19.04x |
|    7 |  6144 x  3072 |     1 |        2.899 |        0.229 |        n/a |      0.249 |      0.049 |       0.087 |        58.99x |
|    7 |  6144 x  3072 |     4 |        2.900 |        0.227 |        n/a |      0.258 |      0.055 |       0.087 |        52.44x |
|    7 |  6144 x  3072 |     8 |        2.899 |        0.226 |        n/a |      0.272 |      0.072 |       0.087 |        40.44x |
|    7 |  6144 x  3072 |    16 |        2.896 |        0.226 |        n/a |      0.338 |        n/a |       0.087 |        12.80x |
|    7 |  6144 x  3072 |    32 |        2.902 |        0.228 |        n/a |        n/a |        n/a |       0.088 |        12.71x |
|    7 |  6144 x  3072 |    64 |        2.910 |        0.228 |        n/a |        n/a |        n/a |       0.091 |        12.77x |
|    7 |  9216 x  3072 |     1 |        4.031 |        0.255 |        n/a |      0.251 |      0.052 |       0.085 |        77.20x |
|    7 |  9216 x  3072 |     4 |        4.027 |        0.255 |        n/a |      0.253 |      0.066 |       0.084 |        60.98x |
|    7 |  9216 x  3072 |     8 |        4.031 |        0.259 |        n/a |      0.298 |      0.086 |       0.085 |        46.87x |
|    7 |  9216 x  3072 |    16 |        4.023 |        0.254 |        n/a |      0.386 |        n/a |       0.084 |        15.81x |
|    7 |  9216 x  3072 |    32 |        4.015 |        0.258 |        n/a |        n/a |        n/a |       0.086 |        15.56x |
|    7 |  9216 x  3072 |    64 |        4.008 |        0.255 |        n/a |        n/a |        n/a |       0.094 |        15.72x |
|    7 | 12288 x  3072 |     1 |        5.195 |        0.285 |        n/a |      0.247 |      0.061 |       0.083 |        84.55x |
|    7 | 12288 x  3072 |     4 |        5.177 |        0.287 |        n/a |      0.269 |      0.079 |       0.083 |        65.66x |
|    7 | 12288 x  3072 |     8 |        5.168 |        0.286 |        n/a |      0.333 |      0.106 |       0.084 |        48.53x |
|    7 | 12288 x  3072 |    16 |        5.166 |        0.287 |        n/a |      0.451 |        n/a |       0.085 |        17.98x |
|    7 | 12288 x  3072 |    32 |        5.171 |        0.287 |        n/a |        n/a |        n/a |       0.086 |        18.03x |
|    7 | 12288 x  3072 |    64 |        5.175 |        0.290 |        n/a |        n/a |        n/a |       0.101 |        17.86x |
|    7 |  3072 x   256 |     1 |        1.013 |        0.217 |        n/a |      0.255 |      0.047 |       0.087 |        21.50x |
|    7 |  3072 x   256 |     4 |        1.040 |        0.219 |        n/a |      0.250 |      0.047 |       0.093 |        22.08x |
|    7 |  3072 x   256 |     8 |        1.020 |        0.220 |        n/a |      0.247 |      0.047 |       0.088 |        21.65x |
|    7 |  3072 x   256 |    16 |        1.030 |        0.225 |        n/a |      0.250 |        n/a |       0.092 |         4.58x |
|    7 |  3072 x   256 |    32 |        1.032 |        0.220 |        n/a |        n/a |        n/a |       0.101 |         4.70x |
|    7 |  3072 x   256 |    64 |        1.032 |        0.219 |        n/a |        n/a |        n/a |       0.126 |         4.71x |
```

Notes:
- pangolin column populated for M<=8 (native decode gate, PANGOLIN_MAX_M=8); tri-gemv for M<=16; larger M uses tri-dequant + cuBLAS.
- marlin-4bit ceiling is n/a for N=48/72 only (Marlin packer needs out_features % 32).
- 3072x48 and 3072x72 rows exercise the new /32 auto-padding (padded N=64/96); Pangolin runs them at the same ~0.047 ms as aligned shapes.
- Triton first-call compile latency per new shape is minutes on this Python 3.14t box (MLIR/LLVM, single-threaded); results above are steady-state CUDA-event medians after warmup.

### Padding unit tests (tests/test_planar_padding.py)

Dedicated pad/unpad/slice validation, 60/60 passed on GPU box:
- pack-stage: aligned buffer shapes, neutral padded scales (1.0), g_idx tail reuse, dequantized padded region exactly zero (N-pad, K-pad, both).
- load-stage: register_buffers=True shapes match pack-stage padding; state_dict round-trip gives bit-identical forwards.
- forward: CPU + GPU (TritonV2) outputs logical [batch, out_features], dense-reference accurate, deterministic across repeats.

## Round 6 — SUPPORTS_FORMAT_BIT_MAP + continuous gptq_v2:3 -> planar relayout for Pangolin

Design:
- Replaced the separate `SUPPORTS_BITS` / `SUPPORTS_FORMATS` kernel declarations with a combined
  `SUPPORTS_FORMAT_BIT_MAP: Dict[FORMAT, FormatSupport(priority, bits)]` so every kernel declares the exact
  format:bit combos it supports (planar gptq_p 3/5/6/7 vs continuous gptq/gptq_v2 2/3/4/8 was previously
  inexpressible). Legacy `SUPPORTS_FORMATS`/`SUPPORTS_BITS` are auto-derived in `__init_subclass__` for
  back-compat consumers (importer support tree, quantizer). All 31 qlinear backends migrated.
- Validation is now format-aware: `_validate()`/`_validate_dynamic_bits()` check `supported_bits(format)`,
  so e.g. gptq:5 is rejected while gptq_p:5 passes on the same kernel.
- gptq_p at 2/4/8-bit reuses the continuous word layout (planar flag False), so its map entry is
  (2,3,4,5,6,7,8) — first attempt with (3,5,6,7) broke the dual-layout 2/4/8 tests and was corrected.
- New `PackableQuantLinear.convert_to_planar()`: one-time in-place relayout of continuous gptq_v2 3-bit
  qweight/qzeros into the planar layout (decode continuous codes -> planar_pack_rows/cols). Runtime-only;
  on-disk checkpoint stays continuous. TritonV2 `post_init` calls it (gated: CUDA cc>=8.0, /32 dims,
  int32 pack, v2 zeros, block-uniform g_idx, Pangolin ext available, kill switch respected) so the native
  Pangolin GEMV now legitimately serves gptq_v2:3 checkpoints.

What worked / what didn't:
- Mutating `.data` of packed buffers during conversion tripped inference-mode version-counter checks;
  fixed by replacing the buffer tensors outright.
- TritonV2 continuous 3-bit fixture needed `sym=True` (fused path requirement).

Test results (A100 box, allocator lease):
- tests/test_format_bit_map.py + test_planar_padding.py + test_planar_triton_kernels.py +
  test_planar_bits_567.py + test_planar_format_gptq_p.py + test_torch_kernel_accuracy.py +
  test_planar_model_e2e.py: 400/400 passed (ran on GPU).
- CUDA relayout tests confirm: post_init converts once, Pangolin routes gptq_v2:3, output matches the
  continuous reference, and repeated forwards are stable (no reconversion).
