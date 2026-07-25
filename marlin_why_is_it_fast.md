# Why Marlin is fast on large M/N GPTQ W4A16 GEMMs

This file explains the design choices that let Marlin (Frantar et al.,
*MARLIN: Mixed-Precision Auto-Regressive Parallel Inference on Large Language
Models*, arXiv:2408.11743, 2024) outperform simpler 4-warp / shared-B-dequant
pipelines for the M=16/32, large-N shapes that still lose to Marlin in the
Amplin sweep. It is a design reference, not a code plan.

## 1. The high-level claim

Marlin shows that a 4-bit quantized LLM decode can stay **memory-bound**, and
therefore get close to the ideal 4x speedup, for batch sizes up to 16-32, and
still get a useful speedup up to batch 64-128. This is not obvious: larger
batches raise arithmetic intensity, so naively the kernel should become
compute-bound and the quantization advantage should shrink. Marlin avoids that
trap by making the kernel almost always limited by how fast weights can be
streamed, while using the GPU's execution pipelines to hide dequantization.

## 2. Core observation from the paper

From arXiv:2408.11743:

> "...modern GPUs, such as the ones from NVIDIA's Ampere family, typically have
> a FLOP-to-byte ratio in the range of 100 ... the cost of reading the LLM
> weights dwarfs that of the arithmetic operations, and their footprint greatly
> exceeds the cache size."

This means the game is not raw FLOPS, it is **bytes moved per useful FLOP**.
If the kernel can (a) read 4-bit weights instead of 16-bit weights, (b)
decompress them cheaply in registers, and (c) reuse activations enough to
keep DRAM/L2 busy, it wins. Marlin is built around those three ideas.

## 3. The five techniques that matter most

### 3.1 Weights stay compressed until they reach registers

Marlin never writes a fully dequantized FP16 weight matrix to shared or global
memory. It loads 4-bit packed weights with `cp.async` (16-byte `int4` vectors),
then uses bitwise LOP3 instructions (see `gptqmodel_ext/marlin/dequant.h`) to
turn a 32-bit packed word directly into FP16/BF16 `half2`/`nv_bfloat162`
fragments. The constants `0x6400...`, `0x2c00...` and `0xd400...` are the
standard FasterTransformer int4-to-FP16/BF16 conversion magic numbers; the
subtraction/addition of `0x64086408` folds in the symmetric `-8` zero point.

Result: dequantization is a handful of bit and half2 instructions per 8
weights, and the output is already in the tensor-core fragment format that the
next `mma` instruction expects.

### 3.2 A, B and scales are fetched asynchronously

`gptqmodel_ext/marlin/marlin.cuh` exposes thin `cp_async4*` wrappers around
`cp.async.cg.shared.global`. In `gptqmodel_ext/marlin/marlin_template.h` the
main loop is a 4-stage asynchronous pipeline:

- `fetch_to_shared()` issues `cp.async` copies for the next A tile, the next
  compressed B tile, and the next scale tile into a ring of shared buffers.
- `fetch_to_registers()` `ldmatrix`s the A tile from shared into a `FragA`
  and reads the raw compressed B word into `frag_b_quant`.
- The inner `mma` loop dequantizes two `FragB` halves and issues one or two
  `mma.sync.aligned.m16n8k16`/`m16n8k8` PTX instructions, accumulating in
  FP32.
- `cp_async_wait<stages - 2>()` and `__syncthreads()` keep the pipeline full.

Because the next memory transfer is issued while the current math is running,
DRAM latency is hidden even though the occupancy is low.

### 3.3 Tile shapes are chosen per batch regime

`gptqmodel_ext/marlin/gptq_marlin.cu` selects the CTA shape at runtime:

```cpp
thread_config_t small_batch_thread_configs[] = {
    {128, 128, 256},  // thread_k, thread_n, num_threads
    {64, 128, 128},
    {128, 64, 128}};
thread_config_t large_batch_thread_configs[] = {
    {64, 256, 256},
    {64, 128, 128},
    {128, 64, 128}};
```

`num_threads = 256` = 8 warps. Small batches use a larger K tile (`128`)
to amortize the A load across many output columns; large batches use a larger
N tile (`256`) to increase arithmetic intensity and keep the tensor cores fed.

Each configuration is validated against the device's shared-memory budget in
`is_valid_config()`; Marlin uses `cudaFuncSetAttribute` to raise the per-block
dynamic shared limit when needed.

### 3.4 One big block per SM, not many small ones

Marlin typically launches one block per SM for the main GEMM. Each block uses a
large contiguous N slice (up to 256 columns), a K tile of 64 or 128, and up to
~167 KB of shared memory. This is the opposite of a tiny-tile strategy.

Why it works:

- A large N slice means each A row is reused many times while it is hot in L2.
- A single block per SM avoids inter-block reduction traffic and keeps the
  memory controller streaming.
- 8 warps × 255 registers give enough register pressure to keep many in-flight
  `mma`s and `cp.async`s active.

### 3.5 Multi-warp partials are reduced in shared memory

`gptqmodel_ext/marlin/marlin_template.h` splits K across multiple warps. Each
warp computes a partial accumulator for the same output tile. At the end a
parallel logarithmic reduction writes partials through `sh_red` and then the
final tile is written to global output. This lets Marlin use more warps per
output tile without extra global memory traffic.

## 4. What Nsight Compute says

The Marlin README and paper both emphasize that the kernel is deliberately
**not** optimized for high SM occupancy in the classical sense. A typical NCU
snapshot for a large layer on an Ampere GPU looks roughly like:

| metric | typical value | meaning |
|---|---|---|
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | ~40-50% | tensor/vector pipes are busy, not idle |
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | ~15-20% | DRAM is not the bottleneck; L2 catches most reuse |
| `l2_hit_rate` | ~60-70% | A-tile reuse is working |
| active warps per SM | 8-16 | low occupancy, but enough to hide `mma` latency |
| shared memory per block | ~100-170 KB | large A/B/scales buffers |
| pipeline stages | 4 | `cp.async` ring buffer |

The key insight is that **low occupancy is okay when the warps that are resident
issue tensor-core instructions back-to-back and the memory pipeline is kept
busy with `cp.async`**. A 4-warp kernel with the same total work per block
cannot do that; it simply does not have enough parallel instruction streams to
feed both the memory and math pipes.

## 5. Why the current 4-warp Amplin pipeline loses

The `gemm_hmma_m32_n128_pipeline4` variant implemented on the
`devin/1784898994-marlin-style-cpasync` branch is correct but loses for the
same reasons the paper warns about:

1. **Only 4 warps per CTA**. Marlin uses 8. With 4 warps there are not enough
   independent instruction streams to hide `cp.async` latency and `mma`
   latency at the same time.
2. **B is dequantized into shared memory**. The current code unpacks each
   `uint4` weight word, converts each nibble with a float multiply, and stores
   the resulting FP16/BF16 column into `shared_b`. That is many more
   instructions than Marlin's LOP3-to-fragment path, and it creates a
   `__syncthreads` dependency on every pipeline stage.
3. **N tile is too small**. The N=128 block (and the N=64 variant) processes
   too few output columns per A-tile load. Marlin's large-batch config uses
   N=256.
4. **Barriers dominate**. Nsight Compute on the Amplin 4-warp pipeline showed
   SM throughput around 20% and `barrier`/`long_scoreboard` stalls as the top
   stall reasons. That is the signature of a kernel waiting on
   `__syncthreads` and `cp.async` with too few warps, not a kernel that is
   compute or memory starved.

## 6. Take-away for Amplin

To beat Marlin on the remaining M=16/32, large-N shapes the kernel must:

- Use **8 warps per block** (256 threads), not 4, and accept the higher
  register/shared-memory footprint.
- Dequantize weights **directly into tensor-core fragments in registers**,
  using LOP3 or byte-perm magic, not into a shared FP16/B buffer.
- Keep activations in **shared memory** and load them with `ldmatrix`, while
  `cp.async` streams the next activations in the background.
- Pick **large N tiles** (128-256 columns) for large M and small K tiles
  (64) to keep A-tile reuse high and to fit the shared-memory budget.
- Use **multi-warp partial K-split reduction** in shared memory rather than
  global partials for large K.
- Avoid global reductions; when K is large enough to need split-K, do it with
  a persistent grid or cooperative blocks and a single final reduction.

In short, Marlin is fast because it treats the problem as a **memory
streaming + tensor-core dequantization** problem, not a `wmma::load_matrix_sync`
from a dequantized shared buffer problem. The 4-warp pipeline was a useful
first step, but closing the remaining gap requires moving to the same 8-warp,
register-fragment, large-N-tile design.

## 7. References

- Frantar, E., Castro, R. L., Chen, J., Hoefler, T., & Alistarh, D.
  *MARLIN: Mixed-Precision Auto-Regressive Parallel Inference on Large
  Language Models*. arXiv:2408.11743, 2024.
  https://arxiv.org/abs/2408.11743
- Original implementation: https://github.com/IST-DASLab/marlin
- Local vendored sources in this repo:
  - `gptqmodel_ext/marlin/marlin_template.h` — main kernel template and
    scheduling.
  - `gptqmodel_ext/marlin/dequant.h` — LOP3 int4/int8/FP4/FP8 to FP16/BF16
    dequantization.
  - `gptqmodel_ext/marlin/marlin_mma.h` — `mma.sync.aligned.m16n8k16` PTX
    wrappers.
  - `gptqmodel_ext/marlin/marlin_dtypes.cuh` — `FragA/B/C/S` definitions.
  - `gptqmodel_ext/marlin/gptq_marlin.cu` — runtime tile/config selection and
    `is_valid_config()`.
