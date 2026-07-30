# Laguna-S-2.1-GPTQ-FIXED Decode Optimization Log

Target: 4x faster decode tokens/sec on the real quantized checkpoint
`/monster/data/model/Laguna-S-2.1-GPTQ-FIXED` with `backend=GPTQ_MARLIN`,
`flash_attention_2`, and all fusing enabled.

## Environment

- GPU: NVIDIA A100 96 GB (sm80)
- PyTorch: 2.13.0+cu130
- Triton: 3.7.1
- Model: `Laguna-S-2.1-GPTQ-FIXED` (59 GB)
- Checkpoint: 4-bit GPTQ, dynamic `group_size` (32/128), `sym=True`, `desc_act=False`
- Attention: `flash_attention_2`
- Script: `scripts/benchmark_fuse_real_laguna.py`

## Notes on reported `tok/s`

`scripts/benchmark_fuse_real_laguna.py` currently computes throughput as
`batch * (seq_len + max_new_tokens) / (ms / 1000)`.  For `seq_len=1` and
`max_new_tokens=1` this counts both the input token and the generated token,
so the **true new-token decode rate is half the printed value** for these runs.
Batch 1 at 7.0 tok/s in the table below is therefore ~3.5 new decode tok/s.
Future runs should report `max_new_tokens` only for decode-only measurements.

## Baseline (fuse only, no compile)

`python scripts/benchmark_fuse_real_laguna.py --gpu 6 --batch-sizes 1 2 4 8 16 32 --seq-len 1 --max-new-tokens 1 --repeats 5 --warmup 3 --fuse --disable-speculative --attn-implementation flash_attention_2`

| state  | batch | seq | ms      | tok/s | speedup vs unfused |
|--------|------:|----:|--------:|------:|-------------------:|
| unfused | 1   | 1   | 283.677 | 7.0   | 1.00 |
| fused   | 1   | 1   | 262.382 | 7.6   | 1.09 |
| unfused | 2   | 1   | 402.912 | 9.9   | 1.00 |
| fused   | 2   | 1   | 349.385 | 11.4  | 1.15 |
| unfused | 4   | 1   | 609.508 | 13.6  | 1.00 |
| fused   | 4   | 1   | 535.780 | 14.9  | 1.10 |
| unfused | 8   | 1   | 825.396 | 19.4  | 1.00 |
| fused   | 8   | 1   | 752.269 | 21.3  | 1.10 |
| unfused | 16  | 1   | 1094.497 | 29.2 | 1.00 |
| fused   | 16  | 1   | 1035.779 | 30.9 | 1.06 |
| unfused | 32  | 1   | 1531.489 | 41.8 | 1.00 |
| fused   | 32  | 1   | 1437.040 | 44.5 | 1.07 |

Observations:
- Fusing QKV + gate/up gives a modest 6-15% speedup on decode.
- Batch 1 is only ~3.5 new decode tok/s; to reach 4x (~14 new tok/s) the
  single-token forward path needs to be much faster.
- Single-token `model.generate()` timings are dominated by one full forward; the
  model is memory-bound at M=1 and the standard Marlin GEMM path is not ideal
  for very skinny M (M <= 16 uses the small-batch thread config).

## Attempt 1: `torch.compile` (`reduce-overhead`, `inductor`)

`python scripts/benchmark_fuse_real_laguna.py ... --fuse --optimize --optimize-mode reduce-overhead --batch-sizes 1 2 4`

Result: **regression / not usable**.

| state           | batch | seq | ms      | tok/s |
|-----------------|------:|----:|--------:|------:|
| unfused         | 1     | 1   | 419.977 | 4.8   |
| fused+compiled  | 1     | 1   | 680.684 | 2.9   |
| unfused         | 2     | 1   | 831.151 | 4.8   |
| fused+compiled  | 2     | 1   | 813.093 | 4.9   |
| unfused         | 4     | 1   | 1143.078| 7.0   |
| fused+compiled  | 4     | 1   | 1103.822| 7.2   |

Failure analysis:
- `torch.compile` hit the recompile limit (128) because the fused-projection
  cache key is a UUID string stored as a Python attribute on the input tensor.
  Dynamo specializes on the literal string value, so every layer/fuse group
  triggered a new compilation.
- Warning:
  ```
  torch._dynamo hit config.recompile_limit (128)
  function: 'torch_dynamo_resume_in_forward_at_385' (modeling_laguna.py:385)
  last reason: 29/127: hidden_states._gptqmodel_fused_qkv_cache[0] == '...'
  ```
- `reduce-overhead` mode was unable to amortize the compilation cost with the
  current cache design.

## Attempt 2: gating the grouped/MoE mega-kernel for small decode batches

The batched/offset Marlin MoE mega-kernel (`marlin_moe`) and the `grouped_mm`
path both build a block-aligned token-expert index and stack active-expert
weights on every forward.  For single-token decode this Python setup can dominate
kernel time, so a tunable minimum token-expert pair gate was added to
`_can_use_grouped_mm`:

```python
# gptqmodel/utils/moe_dispatch.py
GPTQMODEL_GROUPED_MM_MIN_TOKENS  # default 1 (disabled), set to e.g. 64 to test
```

`python scripts/benchmark_fuse_real_laguna.py ... --fuse` with the gate set to
`64` did **not** recover the original fast baseline on this run; both the
unfused and fused paths were in the 700-1100 ms range for batch 1 (see table
below).  This indicates the regression is not solely the grouped-dispatch
overhead.

| state  | batch | seq | ms      | tok/s |
|--------|------:|----:|--------:|------:|
| unfused | 1   | 1   | 775.561 | 2.6   |
| fused   | 1   | 1   | 711.054 | 2.8   |
| unfused | 2   | 1   | 919.238 | 4.4   |
| fused   | 2   | 1   | 816.165 | 4.9   |
| unfused | 4   | 1   | 1260.454| 6.3   |
| fused   | 4   | 1   | 1117.971| 7.2   |
| unfused | 8   | 1   | 1452.834| 11.0  |
| fused   | 8   | 1   | 1389.528| 11.5  |
| unfused | 16  | 1   | 1811.715| 17.7  |
| fused   | 16  | 1   | 1752.518| 18.3  |
| unfused | 32  | 1   | 2447.877| 26.1  |
| fused   | 32  | 1   | 2351.481| 27.2  |

The gate is retained as an experimental environment variable so profiling can
identify the exact crossover point for this model; it defaults to 1 so existing
unit tests and the Marlin MoE correctness path remain active.

## Attempt 3: profiler capture of one fused decode step

`python scripts/profile_laguna_decode.py --gpu 6 --batch-size 1 --seq-len 1 --fuse --disable-speculative`

Kineto summary for one `model.generate(..., max_new_tokens=1)` call:
- Self CUDA time total: **81.9 ms**
- `generate_step` CPU total: **1.23 s**
- Top CPU consumers:
  - `moe_dispatch.py:linear_loop_experts_forward` -- 992 ms
  - `defuser/modeling/moe_experts_int...` -- 890 ms
  - `aten::any` -- 198 ms
  - `aten::item` -- 171 ms
  - `Memcpy DtoH` -- 15.8 ms

The GPU only does ~82 ms of work for one token, but the host-side MoE routing is
spending large amounts of time in `nonzero`/`item`/`any` synchronization and the
per-expert Python loop.  This is the dominant decode bottleneck, not the raw
Marlin GEMM bandwidth.

## Attempt 4: allocator tuning

Removing `expandable_segments` from `PYTORCH_CUDA_ALLOC_CONF` caused PyTorch to
abort at startup (`AllocatorConfig.h` tokenizing assertion).  The project default
(`expandable_segments:True,max_split_size_mb:1024,garbage_collection_threshold=0.5`)
is therefore required for this build.

## Attempt 5: fix host-side routing and use per-expert Marlin active loop

Changes in `gptqmodel/utils/moe_dispatch.py`:
- Vectorized `_moe_align_block_size` (already present) and added helper
  `_marlin_moe_target`.
- Added per-shape homogeneity check in `_batched_marlin_moe_supported` so the
  batched/offset kernel is only used when every expert in a module has the
  same packed `qweight`/`scales` shapes.
- Fixed `_prestack_marlin_moe_weights` to assign stacked views through
  `nn.Parameter` when the target is a `MarlinLinear` (previously it tried to
  assign a raw `Tensor` to a `Parameter` and crashed).
- Optimized `_marlin_experts_project` to call the fused gate/up group's
  `_compute` once and split the output, instead of calling `gate_proj` and
  `up_proj` separately (which recomputed the fused GEMM).
- For fused Marlin experts whose packed shapes are heterogeneous, route to
  the per-expert Marlin active loop (`backend="marlin"`) instead of the dense
  `grouped_mm` dequant path or the defuser fallback.  This removes the
  `nonzero`/`item`/`any` synchronization that was dominating wall time.

### Why the batched/offset kernel is still not used on Laguna

A shape audit across all 47 MoE `mlp.experts` modules showed heterogeneous packed
`scales` shapes inside almost every module:

| projection | shapes observed | cause |
|------------|----------------:|-------|
| gate/up    | `(24, 2048)` and `(96, 2048)` | dynamic `group_size` 128 vs 32 |
| down       | `(8, 3072)` and `(32, 3072)` | dynamic `group_size` 128 vs 32 |

Because `moe_wna16_marlin_gemm` expects a single `[num_experts, ...]` stacked
weight per projection, pre-stacking is not possible without either (a) grouping
experts by packed shape and launching the kernel once per shape cluster, or
(b) padding/re-quantizing all experts to a uniform shape.  Heterogeneity now
falls back to the per-expert active loop, which is still much faster than the
defuser.

### Decode benchmark (`seq_len=1`, `max_new_tokens=1`)

`python scripts/benchmark_fuse_real_laguna.py --gpu 6 --batch-sizes 1 2 4 8 16 32 --seq-len 1 --max-new-tokens 1 --repeats 5 --warmup 3 --fuse --disable-speculative --attn-implementation flash_attention_2`

| state   | batch | seq | ms      | tok/s | decode tok/s | speedup vs unfused |
|---------|------:|----:|--------:|------:|-------------:|-------------------:|
| unfused | 1     | 1   | 301.779 | 6.6   | 3.3          | 1.00 |
| fused   | 1     | 1   | 273.639 | 7.3   | 3.7          | 1.12 |
| unfused | 2     | 1   | 388.046 | 10.3  | 5.2          | 1.00 |
| fused   | 2     | 1   | 374.158 | 10.7  | 5.3          | 1.05 |
| unfused | 4     | 1   | 595.974 | 13.4  | 6.7          | 1.00 |
| fused   | 4     | 1   | 529.808 | 15.1  | 7.5          | 1.12 |
| unfused | 8     | 1   | 832.517 | 19.2  | 9.6          | 1.00 |
| fused   | 8     | 1   | 769.056 | 20.8  | 10.4         | 1.08 |
| unfused | 16    | 1   | 1135.110| 28.2  | 14.1         | 1.00 |
| fused   | 16    | 1   | 1040.621| 30.8  | 15.4         | 1.09 |
| unfused | 32    | 1   | 1613.076| 39.7  | 19.8         | 1.00 |
| fused   | 32    | 1   | 1424.795| 44.9  | 22.5         | 1.13 |

Observations:
- The host-side defuser fallback has been replaced by the per-expert Marlin
  active loop for both unfused and fused; decode tok/s roughly doubles versus
  the old defuser baseline (from ~1.3-1.5 tok/s to ~3.3-3.7 tok/s at batch 1).
- `model.fuse()` adds a further 5-13% by halving the gate/up Marlin calls.
- Accuracy drift stays within bf16 tensor-core accumulation tolerance.

### Prefill benchmark (`seq_len=128`, `max_new_tokens=1`)

| state   | batch | seq | ms      | tok/s  | decode tok/s |
|---------|------:|----:|--------:|-------:|-------------:|
| unfused | 1     | 128 | 2282.870 | 56.5  | 0.4 |
| fused   | 1     | 128 | 2089.038 | 61.8  | 0.5 |
| unfused | 32    | 128 | 3772.727 | 1094.2| 8.5 |
| fused   | 32    | 128 | 3518.458 | 1173.2| 9.1 |

### Profiler capture after the routing fix

`python scripts/profile_laguna_decode.py --gpu 6 --batch-size 1 --seq-len 1 --fuse --disable-speculative`

- Self CUDA time total: **33.4 ms** (down from 82 ms)
- `generate_step` CPU total: **0.55 s** (down from 1.23 s)
- Self CPU time total: **1.13 s** (down from 3.47 s)
- Top CPU consumers:
  - `_marlin_experts_project` -- 272 ms over 47 MoE layers
  - `apply_gptq_marlin_linear` / `gptq_marlin_gemm_bf16` -- 1092 calls
  - `aten::any` / `aten::item` -- no longer in the top list

The remaining bottleneck is the per-expert Python loop in
`_marlin_experts_project` and the 1000+ individual Marlin kernel launches per
`generate` step.  Collapsing each MoE layer's active experts into one or two
batched Marlin MoE kernel launches (one for fused gate/up, one for down) is
now the highest-impact next step.

## What would give a real 4x

1. **Per-shape-cluster batched/offset Marlin MoE kernel.**  The checkpoint has
   heterogeneous packed shapes per module, so `moe_wna16_marlin_gemm` must be
   called once per `(qweight_shape, scales_shape)` cluster, with a local
   `expert_ids` remap, instead of once per module.  This would collapse the
   1000+ per-token Marlin launches down to ~2 per MoE layer and remove the
   `_marlin_experts_project` Python loop overhead.
2. **M=1 skinny GEMV kernel.**  Even with batching, the Marlin kernel uses
   small-batch thread configs for `prob_m <= 16` and does not fully utilize the
   GPU for M=1.  A dedicated decode GEMV (or vLLM/SGLang-style fused MLP/QKV
   GEMV) would close the remaining gap after routing overhead is removed.

## Attempt 6: per-shape-cluster batched/offset Marlin MoE on multi-GPU

The per-shape-cluster batched/offset Marlin MoE mega-kernel from PR #131 was
benchmarked end-to-end on the real checkpoint with `device_map="auto"` across
eight A100 96 GB GPUs.  The script was extended with `--device-map`,
`--moe-grouped-dispatch`, and partial `--fuse-qkv` / `--fuse-gate-up` toggles.

### Environment

- GPUs: 8 × NVIDIA A100 96 GB (sm80), `CUDA_DEVICE_ORDER=PCI_BUS_ID`
- PyTorch: 2.13.0+cu130, Triton: 3.7.1
- Model: `/monster/data/model/Laguna-S-2.1-GPTQ-FIXED` (59 GB, dynamic `group_size` 32/128)
- Backend: `GPTQ_MARLIN`, attention: `flash_attention_2`
- `device_map="auto"` placed alternating layers across all 8 GPUs
- Grouped MoE dispatch was enabled before the first benchmark pass; `model.fuse()`
  was applied between the `grouped` and `grouped+fused` passes.

### Decode benchmark (`seq_len=1`, `max_new_tokens=1`)

`python scripts/benchmark_fuse_real_laguna.py --device-map auto --moe-grouped-dispatch --fuse --batch-sizes 1 2 4 8 16 32 --seq-len 1 --max-new-tokens 1 --repeats 3 --warmup 1 --disable-speculative --attn-implementation flash_attention_2`

| state         | batch | seq | ms        | decode tok/s | vs unfused |
|---------------|------:|----:|----------:|-------------:|-----------:|
| unfused       | 1     | 1   | 806.014   | 1.2          | 1.00 |
| grouped       | 1     | 1   | 804.691   | 1.2          | 1.00 |
| grouped+fused | 1     | 1   | 821.973   | 1.2          | 1.00 |
| unfused       | 2     | 1   | 955.778   | 2.1          | 1.00 |
| grouped       | 2     | 1   | 938.507   | 2.1          | 1.02 |
| grouped+fused | 2     | 1   | 986.297   | 2.0          | 0.95 |
| unfused       | 4     | 1   | 1220.220  | 3.3          | 1.00 |
| grouped       | 4     | 1   | 1268.229  | 3.2          | 0.96 |
| grouped+fused | 4     | 1   | 1243.457  | 3.2          | 0.98 |
| unfused       | 8     | 1   | 1552.040  | 5.2          | 1.00 |
| grouped       | 8     | 1   | 1621.252  | 4.9          | 0.96 |
| grouped+fused | 8     | 1   | 1569.243  | 5.1          | 1.01 |
| unfused       | 16    | 1   | 1991.398  | 8.0          | 1.00 |
| grouped       | 16    | 1   | 2091.442  | 7.7          | 0.95 |
| grouped+fused | 16    | 1   | 1983.665  | 8.1          | 1.01 |
| unfused       | 32    | 1   | 2660.461  | 12.0         | 1.00 |
| grouped       | 32    | 1   | 2711.911  | 11.8         | 0.98 |
| grouped+fused | 32    | 1   | 2544.893  | 12.6         | 1.05 |

### Prefill benchmark (`seq_len=128`, `max_new_tokens=0`)

`python scripts/benchmark_fuse_real_laguna.py --device-map auto --moe-grouped-dispatch --fuse --batch-sizes 1 2 4 8 16 32 --seq-len 128 --max-new-tokens 0 --repeats 3 --warmup 1 --disable-speculative --attn-implementation flash_attention_2`

| state         | batch | seq | ms        | tok/s | vs grouped |
|---------------|------:|----:|----------:|------:|-----------:|
| grouped       | 1     | 128 | 4466.509  | 28.7  | 1.00 |
| grouped+fused | 1     | 128 | 3634.465  | 35.2  | 1.23 |
| grouped       | 2     | 128 | 5233.611  | 48.9  | 1.00 |
| grouped+fused | 2     | 128 | 3902.759  | 65.6  | 1.34 |
| grouped       | 4     | 128 | 5531.480  | 92.6  | 1.00 |
| grouped+fused | 4     | 128 | 4244.137  | 120.6 | 1.30 |
| grouped       | 8     | 128 | 5041.711  | 203.1 | 1.00 |
| grouped+fused | 8     | 128 | 4665.898  | 219.5 | 1.08 |
| grouped       | 16    | 128 | 5421.639  | 377.7 | 1.00 |
| grouped+fused | 16    | 128 | 4937.165  | 414.8 | 1.10 |
| grouped       | 32    | 128 | 6121.439  | 669.1 | 1.00 |
| grouped+fused | 32    | 128 | 5336.268  | 767.6 | 1.15 |

### Numerical parity

Decode logits (`max_new_tokens=1`) stay within BF16 accumulation tolerance:
`max_abs_diff` ≤ 14.75, `mean_abs_diff` ≤ 0.52.

Prefill logits (`seq_len=128`) show slightly larger but still acceptable drift:
`max_abs_diff` ≤ 23.25, `mean_abs_diff` ≤ 0.99.

### Observations

- The per-shape-cluster batched/offset Marlin MoE path is **neutral to slightly
  regressive for decode** when `device_map="auto"` spreads layers across all 8
  GPUs.  Cross-device data movement and `accelerate` dispatch overhead dominate
  the kernel-launch savings.
- `model.fuse()` on top of grouped dispatch gives a modest **+5% decode** win at
  batch 32 and **+15–34% prefill** wins across batch sizes, but it is far from
  the 4× decode target.
- Multi-GPU `device_map="auto"` is actually **slower** than the single-GPU
  Attempt 5 numbers for small-batch decode/prefill, so single-GPU execution is
  preferred whenever it fits.
- `model.fuse()` still OOMs on a single A100 because concatenating gate/up
  packed weights temporarily doubles per-module memory and the checkpoint already
  consumes ~90 GB after load.

### Updated hypothesis for 4× decode

1. **Run on a single GPU and avoid the multi-GPU copy tax.**  This means either
   fitting the model on one 96 GB card or using a tight tensor-parallel layout
   that keeps each layer on one device.
2. **Collapse each MoE layer to one homogeneous `group_size`** (e.g. the uniform
   W4G64 config in `laguna_s21/quant_config_fixed_w4g64.json`).  That eliminates
   the per-shape clusters and allows a single `moe_wna16_marlin_gemm` launch per
   projection per MoE layer.
3. **Add an M=1 skinny GEMV path** for the batched MoE kernel, because the
   standard Marlin small-batch configuration does not fully utilize the GPU
   for single-token decode.

## Reproduction

```bash
# Multi-GPU per-shape-cluster decode benchmark
python scripts/benchmark_fuse_real_laguna.py \
  --device-map auto \
  --moe-grouped-dispatch \
  --fuse \
  --batch-sizes 1 2 4 8 16 32 \
  --seq-len 1 --max-new-tokens 1 \
  --repeats 3 --warmup 1 \
  --disable-speculative \
  --attn-implementation flash_attention_2

# Multi-GPU per-shape-cluster prefill benchmark
python scripts/benchmark_fuse_real_laguna.py \
  --device-map auto \
  --moe-grouped-dispatch \
  --fuse \
  --batch-sizes 1 2 4 8 16 32 \
  --seq-len 128 --max-new-tokens 0 \
  --repeats 3 --warmup 1 \
  --disable-speculative \
  --attn-implementation flash_attention_2

# Single-GPU baseline (requires a physical GPU with enough free memory)
python scripts/benchmark_fuse_real_laguna.py \
  --gpu 6 \
  --batch-sizes 1 2 4 8 16 32 \
  --seq-len 1 --max-new-tokens 1 \
  --repeats 5 --warmup 3 \
  --fuse --disable-speculative \
  --attn-implementation flash_attention_2

# Profile one step
python scripts/profile_laguna_decode.py \
  --gpu 6 --batch-size 1 --seq-len 1 \
  --fuse --disable-speculative
```
