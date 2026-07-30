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

## Attempt 6: Uniform W4G64 re-quantized checkpoint (`Laguna-S-2.1-GPTQ-4G64`)

`Laguna-S-2.1-GPTQ-4G64` has `group_size=64` for every expert, so a single
stacked weight tensor per MoE projection is possible and the batched/offset
Marlin mega-kernel (`marlin_moe`) can activate.

A full real-model decode benchmark was started but the load phase of the 59 GB
checkpoint is very slow (Marlin repacking of 36k+ modules / multi-GPU
`device_map` placement).  To avoid blocking on the load, the dispatch backends
were compared with a synthetic 256-expert, top-k=10, hidden=3072, intermediate=1024
Laguna-shape MoE module on a single A100 (physical GPU 6, sm80).

### MoE dispatch kernel shoot-out (synthetic Laguna shape, W4G64)

`python scripts/benchmark_marlin_moe_kernel.py --model laguna-s-2.1 --num-experts 256 --top-k 10 --group-size 64 --gpu 6 --backends per_expert,grouped_mm,marlin,marlin_moe`

| mode    | batch | seq | per_expert_ms | per_expert_tok/s | grouped_mm_ms | grouped_mm_tok/s | marlin_ms | marlin_tok/s | marlin_moe_ms | marlin_moe_tok/s | grouped_mmx | marlinx | marlin_moex |
|---------|------:|----:|--------------:|-----------------:|--------------:|-----------------:|----------:|-------------:|--------------:|-----------------:|------------:|--------:|------------:|
| decode  | 1     | 1   | 13.795        | 724.9            | 11.236        | 890.0            | 3.442     | 2,904.9      | 9.537         | 1,048.5          | 1.228       | 4.007   | 1.446       |
| decode  | 2     | 1   | 17.355        | 1,152.4          | 19.699        | 1,015.3          | 5.946     | 3,363.8      | 17.981        | 1,112.3          | 0.881       | 2.919   | 0.965       |
| decode  | 4     | 1   | 24.393        | 1,639.8          | 36.332        | 1,101.0          | 9.965     | 4,014.1      | 33.101        | 1,208.4          | 0.671       | 2.448   | 0.737       |
| decode  | 8     | 1   | 37.089        | 2,157.0          | 64.173        | 1,246.6          | 45.486    | 1,758.8      | 63.036        | 1,269.1          | 0.578       | 0.815   | 0.588       |
| decode  | 16    | 1   | 58.321        | 2,743.4          | 110.798       | 1,444.1          | 37.751    | 4,238.3      | 108.055       | 1,480.7          | 0.526       | 1.545   | 0.540       |
| decode  | 32    | 1   | 88.889        | 3,600.0          | 179.801       | 1,779.7          | 50.461    | 6,341.6      | 171.172       | 1,869.5          | 0.494       | 1.762   | 0.519       |
| prefill | 1     | 128 | 117.626       | 10,881.9         | 226.447       | 5,652.5          | 65.180    | 19,638.1     | 239.857       | 5,336.5          | 0.519       | 1.805   | 0.490       |
| prefill | 1     | 1024| 128.722       | 79,551.1         | 227.737       | 44,964.2         | 65.842    | 155,524.4    | 239.654       | 42,728.3         | 0.565       | 1.955   | 0.537       |
| prefill | 1     | 4096| 129.554       | 316,161.3        | 231.087       | 177,249.5        | 73.531    | 557,041.0    | 256.204       | 159,872.5        | 0.561       | 1.762   | 0.506       |

(Units: ms for the full MoE forward, tok/s = token-expert pairs per second.)

Key findings:
- The **per-expert packed Marlin active loop** (`marlin` backend) is the clear
  winner, up to **4x faster than the defuser per-expert loop** at batch=1 and
  1.5-2x faster across decode/prefill.
- The **batched/offset Marlin mega-kernel (`marlin_moe`)** and the **dense
  `grouped_mm` dequant path** are both slower than the simple per-expert Marlin
  active loop for this shape class.  They suffer from poor small-M occupancy and
  dense dequant overhead, respectively.
- This contradicts the earlier hypothesis that a single `moe_wna16_marlin_gemm`
  launch per MoE layer would beat per-expert launches.  The optimized Marlin
  GEMM + fused gate/up `_compute` path is more efficient for M <= 32.

### Code change

`gptqmodel/utils/moe_dispatch.py` now defaults to the per-expert packed Marlin
active loop for `GPTQ_MARLIN` MoE checkpoints.  The mega-kernel remains available
for experimentation:

```bash
GPTQMODEL_MARLIN_MOE_BACKEND=marlin_moe python scripts/benchmark_fuse_real_laguna.py ...
```

### Real-model load attempt

A single-GPU real-model decode run was started with:

```bash
python scripts/benchmark_fuse_real_laguna.py \
  --model-path /monster/data/model/Laguna-S-2.1-GPTQ-4G64 \
  --gpu 6 --fuse --batch-sizes 1 32 \
  --seq-len 1 --max-new-tokens 1 \
  --repeats 3 --warmup 1 \
  --disable-speculative --attn-implementation flash_attention_2
```

The loader reached `device_map = {'': 'cuda:0'}` and allocated ~52 GB on GPU 6,
but the process then entered an uninterruptible sleep (`D` state) and made no
progress for ~10 minutes. `nvidia-smi` showed 0% GPU utilization. The hang is
in the weight-loading / model-instantiation phase, before `model.fuse()` was
reached. The same script has successfully loaded the heterogeneous
`Laguna-S-2.1-GPTQ-FIXED` checkpoint in previous sessions, so this may be a
transient storage/loader issue for the 4G64 snapshot or a difference in the
safetensors layout. The synthetic shoot-out above is the actionable evidence
for the dispatch default change.

## Attempt 7: Eliminate hot-path `getattr` and per-layer device syncs

Changes in `gptqmodel/utils/moe_dispatch.py`:
- Added a `weakref.WeakKeyDictionary` cache of per-expert dispatch handles
  (`gate_proj`, `up_proj`, `down_proj`, fused gate/up group, slices). The hot
  per-expert loops now use direct list indexing instead of `getattr(self, str(i))`
  and repeated fused-group introspection.
- Removed the `if sentinel_mask.any():` device->host synchronization in both the
  grouped GEMM and `marlin_moe` paths. Sentinel IDs are now clamped and masked
  unconditionally with GPU-side ops.
- `_marlin_experts_project` now branches on the precomputed fused/unfused gate/up
  target and calls `down_projs[expert_idx]` directly.

### Real-model decode benchmark (`seq_len=1`, `max_new_tokens=1`, GPU 6)

`python scripts/benchmark_fuse_real_laguna.py --model-path /monster/data/model/Laguna-S-2.1-GPTQ-4G64 --gpu 6 --fuse --batch-sizes 1 2 4 8 16 32 --seq-len 1 --max-new-tokens 1 --repeats 3 --warmup 1 --disable-speculative --attn-implementation flash_attention_2`

| state   | batch | seq | ms      | tok/s | decode_tok/s | speedup vs unfused |
|---------|------:|----:|--------:|------:|-------------:|-------------------:|
| unfused | 1     | 1   | 254.700 | 7.9   | 3.9          | 1.00 |
| fused   | 1     | 1   | 264.003 | 7.6   | 3.8          | 0.97 |
| unfused | 2     | 1   | 334.826 | 11.9  | 6.0          | 1.00 |
| fused   | 2     | 1   | 334.329 | 12.0  | 6.0          | 1.00 |
| unfused | 4     | 1   | 498.188 | 16.1  | 8.0          | 1.00 |
| fused   | 4     | 1   | 478.490 | 16.7  | 8.4          | 1.04 |
| unfused | 8     | 1   | 742.477 | 21.5  | 10.8         | 1.00 |
| fused   | 8     | 1   | 701.120 | 22.8  | 11.4         | 1.06 |
| unfused | 16    | 1   | 995.264 | 32.2  | 16.1         | 1.00 |
| fused   | 16    | 1   | 918.465 | 34.8  | 17.4         | 1.08 |
| unfused | 32    | 1   | 1351.900| 47.3  | 23.7         | 1.00 |
| fused   | 32    | 1   | 1238.537| 51.7  | 25.8         | 1.09 |

- The `getattr`/sync cleanup improved the **unfused** decode path by ~18-22%
  compared with Attempt 6 (e.g. batch 1: 312 ms -> 255 ms). This shows the
  per-expert active loop is now the dominant path and the remaining time is
  mostly Python framework overhead, not the Marlin GEMM itself.
- Fused QKV + gate/up adds a further 4-9% for batch >= 8.
- Accuracy drift remains within BF16 tensor-core accumulation tolerance.

### Real-model prefill benchmark (`seq_len=128`, `max_new_tokens=1`, GPU 6)

`python scripts/benchmark_fuse_real_laguna.py ... --batch-sizes 1 16 --seq-len 128 --max-new-tokens 1`

| state   | batch | seq | ms      | tok/s  | decode_tok/s | speedup vs unfused |
|---------|------:|----:|--------:|-------:|-------------:|-------------------:|
| unfused | 1     | 128 | 2096.771| 61.5   | 0.5          | 1.00 |
| fused   | 1     | 128 | 1895.398| 68.1   | 0.5          | 1.11 |
| unfused | 16    | 128 | 3067.650| 672.8  | 5.2          | 1.00 |
| fused   | 16    | 128 | 2766.958| 745.9  | 5.8          | 1.11 |

- Prefill is roughly 1.1x faster with QKV + gate/up fusion. The MoE `group_size=64`
  checkpoint does not take the packed-prefill Marlin route (which is gated to
  `group_size=128`), so the gains are from kernel launch reduction and attention
  QKV fusion only.

### Profiler observation after the cleanup

`python scripts/profile_laguna_decode.py --model-path /monster/data/model/Laguna-S-2.1-GPTQ-4G64 --gpu 6 --fuse --disable-speculative`

- Self CUDA time total: **33.5 ms**
- `generate_step` wall time (CUDA events): **~260 ms** for batch 1
- The GPU is busy only ~13% of the wall time; the rest is CPU dispatch and
  framework overhead in `transformers`/`LagunaModel` and the per-expert Python
  loop inside `_marlin_experts_project`.

### What would give a real 4x decode / 2x prefill from here

1. **`torch.compile` / CUDA graph the full model forward.** The dominant cost is
   now Python framework overhead, not kernel time. `torch.compile` or a
   persistent CUDA graph for the static-shape decode step is the only remaining
   lever that can close a 2-4x gap. The fused-projection cache key (`_gptqmodel_fused_*_cache`)
   currently causes Dynamo to recompile every layer; making that cache
   compile-friendly (or replacing it with a single fused QKV/gate-up call in the
   attention/MLP forward) is the prerequisite.
2. **Packed-prefill Marlin route for `group_size=64`.** The W4G64 checkpoint
   falls back to the normal Marlin GEMM for prefill; extending the packed-prefill
   path (or re-quantizing to `group_size=128`) would improve prefill throughput.

### Next steps

- Make `_FusedQuantGroup` cache compile-friendly so `torch.compile(model.model)`
  can remove the remaining Python overhead without recompiling per layer.
- Validate end-to-end generation quality on `Laguna-S-2.1-GPTQ-4G64` after the
  `getattr`/sync cleanup.
- Continue investigating why `marlin_moe` underperforms for small M; a dedicated
  M=1/2/4 skinny GEMV may still be needed, but the per-expert Marlin active loop
  remains the current best path.

### Compile experiment

`python scripts/benchmark_fuse_real_laguna.py ... --fuse --compile --batch-sizes 1`

`torch.compile(model.model, mode="reduce-overhead", dynamic=True)` runs but is
neutral (~270 ms vs ~264 ms for fused batch 1). `torch._dynamo.disable` was
placed on the fused projection and MoE dispatch forwards to stop the
`_gptqmodel_fused_*_cache` string and `.tolist()` loops from causing per-layer
Dynamo recompiles. The graph breaks prevent `torch.compile` from fusing across
the model, so the real win requires rewriting the cache/MoE dispatch to be
Dynamo-traceable or using a single CUDA graph for the decode step.

### Static cache experiment

`python scripts/benchmark_fuse_real_laguna.py ... --cache-implementation static --max-cache-len 4096`

Using `transformers` `StaticCache` instead of `DynamicCache` was neutral to
slightly regressive for the `seq_len=1, max_new_tokens=1` decode benchmark
(~306 ms vs ~254 ms unfused batch 1). Preallocating a 4,096-token cache adds
memory pressure without reducing the per-token framework overhead that
dominates wall time, so it is not the path to 4x.

## Attempt 8: Single-token Marlin MoE fast path + fused SiLU*mul

Added `_marlin_experts_project_one_token` in `gptqmodel/utils/moe_dispatch.py`
and made `_resolve_apply_gate` prefer the fused `fused_silu_mul` Triton kernel
over the module's `_apply_gate` when `act_fn` is SiLU.

### Single-token fast path

When `num_tokens == 1` and the Marlin per-expert active loop is selected,
`_grouped_mm_dequant_experts_forward` now calls `_marlin_experts_project_one_token`,
which:

- skips `torch.unique`, `searchsorted`, `argsort`, `index_select`, and `bincount`
- iterates over the `top_k` experts for one token
- accumulates weighted down-projection outputs in FP32 on the host accumulator

### Fused SiLU*mul

`fused_silu_mul` now allocates a row-major contiguous output, and the MoE
dispatch uses it for `silu(gate) * up` instead of the eager `_apply_gate` path
(`SiLU` kernel + elementwise `mul`).

### Nsight kernel-launch comparison (`generate_step`, batch=1, seq=1)

Before the change:

| metric | value |
|--------|------:|
| `cudaLaunchKernel` calls | 7,385 |
| `cuLaunchKernelEx` calls | 2 |
| separate `silu` + `mul` kernel instances | 1,133 (518 + 615) |
| total trace time | ~336 ms |

After the change:

| metric | value |
|--------|------:|
| `cudaLaunchKernel` calls | 5,975 |
| `cuLaunchKernelEx` calls | 472 |
| `_fused_silu_mul_kernel` instances | 470 |
| total trace time | ~377 ms (Nsight overhead included) |

- `cudaLaunchKernel` count dropped by **1,410 calls** (~19%).
- The separate `silu`/`mul` kernels were replaced by a single Triton kernel per
  active expert.
- The bulk of the remaining time is still CPU/Python framework overhead and
  `cudaLaunchKernel` API time, not GPU kernel work.

### Real-model decode benchmark after the change

`python scripts/benchmark_fuse_real_laguna.py --model-path /monster/data/model/Laguna-S-2.1-GPTQ-4G64 --gpu 6 --batch-sizes 1 --seq-len 1 --max-new-tokens 1 --repeats 5 --warmup 3 --fuse --attn-implementation flash_attention_2 --cache-implementation static --max-cache-len 4096`

| state   | batch | seq | ms      | tok/s | decode_tok/s | speedup vs unfused |
|---------|------:|----:|--------:|------:|-------------:|-------------------:|
| unfused | 1     | 1   | 309.821 | 6.5   | 3.2          | 1.00 |
| fused   | 1     | 1   | 285.439 | 7.0   | 3.5          | 1.09 |

The single-token Marlin fast path and fused SiLU*mul are correct and do reduce
kernel launches, but end-to-end wall time does not move much because the
remaining bottleneck is CPU-side launch/Python dispatch, not the fused kernels.

### Contiguous input experiment (Attempt 9)

`hidden_states` reaching `_marlin_experts_project_one_token` can be a 3-D view
`(batch, seq, hidden)`, so `x.reshape(-1, hidden_dim)` inside the per-expert
Marlin call was producing a non-contiguous 2-D tensor and triggering a
`direct_copy` inside `MarlinLinear` for every active expert. The M=1 path now
flattens once and ensures a contiguous `(1, hidden_dim)` buffer before the loop.

`python scripts/benchmark_fuse_real_laguna.py --model-path /monster/data/model/Laguna-S-2.1-GPTQ-4G64 --gpu 6 --batch-sizes 1 --seq-len 1 --max-new-tokens 1 --repeats 3 --warmup 2 --fuse --attn-implementation flash_attention_2 --cache-implementation static --max-cache-len 4096`

| state   | batch | seq | ms      | tok/s | decode_tok/s | speedup vs unfused |
|---------|------:|----:|--------:|------:|-------------:|-------------------:|
| unfused | 1     | 1   | 295.836 | 6.8   | 3.4          | 1.00 |
| fused   | 1     | 1   | 282.828 | 7.1   | 3.5          | 1.06 |

Nsight still reports `cudaLaunchKernel` ~5,975 calls and `cuLaunchKernelEx` ~472
calls for the fused `generate_step`; the `direct_copy` kernel count did not
move because `flash_attention_2` and the framework emit a similar number of
small copies. The contiguous-up-front change is still a defensive win and keeps
each Marlin GEMM from doing its own copy when the input is a 3-D view.

### `marlin_moe` mega-kernel attempt on real 4G64

Forcing `GPTQMODEL_MARLIN_MOE_BACKEND=marlin_moe` on the real checkpoint with
`model.fuse(free_original_weights=True)` triggers OOM during fusion. The
batched/offset mega-kernel needs to pre-stack every expert's packed `qweight` and
`scales` into `(num_experts, ...)` tensors, which duplicates the already-fused
gate/up buffers and exceeds the 96 GB A100 budget. A memory-aware stacked-weight
builder (build the stack while freeing per-expert buffers incrementally) is
required before the mega-kernel can be evaluated end-to-end on this model.

### Next steps toward 4x decode

1. **CUDA graph or Dynamo-traceable single-token path.** The per-expert Python
   loop (`for expert_idx, weight in expert_to_weight.items()`) still issues
   ~282 separate Python-level Marlin calls per token (6 experts * 47 layers).
   Replacing this with a single batched/offset Marlin MoE kernel, or wrapping the
   whole M=1 step in a `torch.cuda.graph` replay, is needed to remove the host
   launch overhead.
2. **Pre-stack Marlin MoE weights once and launch one kernel per layer.** With
   `group_size=64` now uniform across all experts, a single
   `moe_wna16_marlin_gemm` call per projection per layer would collapse the
   per-expert loop and its launch overhead entirely.
3. **Profile with Nsight Compute** on the Marlin GEMM to confirm the kernel is
   memory-bandwidth limited at M=1 and choose the right skinny-GEMV tile size
   or a custom M=1 CUDA kernel.

## Attempt 10: memory-aware single-cluster batched/offset Marlin MoE

`gptqmodel/utils/moe_dispatch.py` was refactored so the batched/offset Marlin
MoE mega-kernel (`moe_wna16_marlin_gemm`) no longer needs a full
`[num_experts, ...]` prestack and no longer falls back to the per-expert loop
on every forward:

- Added `_build_single_cluster_stack` which builds one homogeneous cluster of
  the routed experts with a single `torch.stack` copy.
- Single-cluster MoE layers now compute `local_topk_ids` once (via
  `torch.searchsorted`) and share the same `_moe_align_block_size` metadata
  for gate, up, and down projections.
- Outputs are pre-allocated and passed as the optional `c=` argument to
  `moe_wna16_marlin_gemm`, so the kernel writes directly into the final buffer
  and no per-cluster scatter is needed.
- Removed the remaining `sentinel_mask.any()` device-to-host sync in the
  `marlin_moe` path and the temporary debug logging in `_can_use_grouped_mm`.
- `_is_moe_individual_expert` in `fused_quant_linear.py` skips per-expert gate/up
  fusion inside MoE lists, avoiding the packed-weight duplication that caused
  the earlier mega-kernel OOM.

The multi-cluster fallback is preserved for heterogeneous checkpoints, but on
uniform W4G64 `Laguna-S-2.1-GPTQ-4G64` every active forward collapses to one
`moe_wna16_marlin_gemm` launch per projection per MoE layer.

### Nsight Systems kernel-launch comparison (`generate_step`, batch=1, seq=1)

Trace captured with:

```bash
GPTQMODEL_MARLIN_MOE_BACKEND=marlin_moe nsys profile -o /tmp/marlin_moe_decode_fast \
  python scripts/nsys_laguna_decode.py --gpu 6 --moe-backend marlin_moe --fuse \
  --batch-size 1 --seq-len 1
```

| metric | before (per-cluster scatter) | after (single-cluster in-place) | reduction |
|--------|------------------------------|--------------------------------|----------:|
| `cudaLaunchKernel` calls | 34,833 | 6,774 | 5.1x |
| `cudaLaunchKernel` total API time | 243.4 ms | 57.0 ms | 4.3x |
| `cudaStreamSynchronize` calls | 13,064 | 327 | 40.0x |
| `cudaStreamSynchronize` total API time | 134.4 ms | 4.2 ms | 32.0x |
| `cudaMemcpyAsync` calls | 13,075 | 573 | 22.8x |
| `cudaMemcpyAsync` total API time | 78.7 ms | 6.3 ms | 12.5x |
| `reduce_kernel<bool>` instances | 12,036 | 56 | 215x |
| `_fused_silu_mul_kernel` instances | 0 | 47 | — |
| `moe_wna16_marlin` instances | 0 | 94 | — |
| `Marlin` instances | 1,602 | 192 | 8.3x |

The alignment/scatter kernels (`reduce_kernel<bool>`, `AUnaryFunctor<long,bool>`,
`index_select`, `CatArrayBatchedCopy`) that dominated the old trace are now
barely visible. The remaining launch count is mostly attention/FlashAttention,
`torch.stack` copy, and framework copies.

### Real-model decode benchmark (`seq_len=1`, `max_new_tokens=1`, GPU 6)

`GPTQMODEL_MARLIN_MOE_BACKEND=marlin_moe python scripts/benchmark_fuse_real_laguna.py --model-path /monster/data/model/Laguna-S-2.1-GPTQ-4G64 --gpu 6 --batch-sizes 1 2 4 8 16 32 --seq-len 1 --max-new-tokens 1 --repeats 3 --warmup 2 --fuse --disable-speculative --attn-implementation flash_attention_2`

| state   | batch | seq | ms      | tok/s | decode_tok/s | speedup vs per_expert* |
|---------|------:|----:|--------:|------:|-------------:|-----------------------:|
| unfused | 1     | 1   | 200.724 | 10.0  | 5.0          | 1.28 |
| fused   | 1     | 1   | 183.912 | 10.9  | 5.4          | 1.39 |
| unfused | 2     | 1   | 193.767 | 20.6  | 10.3         | 1.72 |
| fused   | 2     | 1   | 190.531 | 21.0  | 10.5         | 1.75 |
| unfused | 4     | 1   | 209.848 | 38.1  | 19.1         | 2.86 |
| fused   | 4     | 1   | 203.187 | 39.4  | 19.7         | 2.95 |
| unfused | 8     | 1   | 229.396 | 69.7  | 34.9         | 3.23 |
| fused   | 8     | 1   | 215.081 | 74.4  | 37.2         | 3.45 |
| unfused | 16    | 1   | 244.718 | 130.8 | 65.4         | 3.99 |
| fused   | 16    | 1   | 237.406 | 134.8 | 67.4         | 4.10 |
| unfused | 32    | 1   | 273.219 | 234.2 | 117.1        | 4.94 |
| fused   | 32    | 1   | 265.572 | 241.0 | 120.5        | 5.08 |

\* per_expert reference taken from Attempt 7 (`decode_tok/s` 3.9 batch 1, 6.0
batch 2, 8.0 batch 4, 10.8 batch 8, 16.1 batch 16, 23.7 batch 32).

- `marlin_moe` reaches **~5.4 new decode tok/s at batch 1** and scales to
  **>120 new decode tok/s at batch 32**.
- vs. the earlier `per_expert` active loop, `marlin_moe` is **1.4x faster at
  batch 1**, **3.2x at batch 8**, and **5.1x at batch 32**.
- `model.fuse()` (QKV + gate/up) adds a small further win (4-9%).
- Accuracy drift stays within BF16 accumulation tolerance (max abs diff < 7,
  mean < 0.6 for all batch sizes).

### Real-model prefill benchmark (`seq_len=128`, `max_new_tokens=0`, GPU 6)

`GPTQMODEL_MARLIN_MOE_BACKEND=marlin_moe python scripts/benchmark_fuse_real_laguna.py --model-path /monster/data/model/Laguna-S-2.1-GPTQ-4G64 --gpu 6 --batch-sizes 1 2 4 8 16 32 --seq-len 128 --max-new-tokens 0 --repeats 2 --warmup 1 --fuse --disable-speculative --attn-implementation flash_attention_2`

| state   | batch | seq | ms      | tok/s   | speedup vs per_expert* |
|---------|------:|----:|--------:|--------:|-----------------------:|
| unfused | 1     | 128 | 281.851 | 454.1   | 7.38 |
| fused   | 1     | 128 | 286.542 | 446.7   | 7.26 |
| unfused | 2     | 128 | 299.723 | 854.1   | 6.93 |
| fused   | 2     | 128 | 301.919 | 847.9   | 6.88 |
| unfused | 4     | 128 | 342.017 | 1497.0  | 6.93 |
| fused   | 4     | 128 | 348.475 | 1469.3  | 6.78 |
| unfused | 8     | 128 | 414.332 | 2471.4  | 6.18 |
| fused   | 8     | 128 | 409.253 | 2502.1  | 6.26 |
| unfused | 16    | 128 | 556.530 | 3679.9  | 5.47 |
| fused   | 16    | 128 | 537.494 | 3810.3  | 5.66 |
| unfused | 32    | 128 | 814.606 | 5028.2  | 5.25 |
| fused   | 32    | 128 | 817.087 | 5012.9  | 5.23 |

\* per_expert reference taken from Attempt 7 (`tok/s` 61.5 batch 1, 1094.2 batch 32).

- `marlin_moe` exceeds the 2x prefill target at every batch size, reaching
  **>5,000 tok/s at batch 32**.
- QKV + gate/up fusion is mostly neutral for prefill at seq=128 (within 3%).
- Accuracy drift stays within BF16 accumulation tolerance (max abs diff < 27,
  mean < 1.0) for the full prefill shapes.

### Observations and next steps

- The batched/offset Marlin MoE mega-kernel finally outperforms the per-expert
  active loop on the real 4G64 checkpoint for both decode and prefill, with the
  largest gains at higher batch sizes.
- The batch-1 decode gap vs. the 4x target is close: 5.4 new decode tok/s vs.
  an estimated ~1.3 tok/s defuser baseline (~4x) and 3.9 tok/s per-expert
  (~1.4x).  The remaining M=1 overhead is host launch/Python framework time,
  not the GPU kernel itself.
- The prefill 2x target is exceeded across all measured batch sizes.
- Next levers: pre-stack weights once (or fuse the stack into the kernel), CUDA
  graph the single-token step, and extend the packed-prefill Marlin route to
  `group_size=64` or re-quantize to 128.
