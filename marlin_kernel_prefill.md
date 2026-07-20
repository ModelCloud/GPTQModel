# Marlin packed-weight prefill kernel: result and findings

## Outcome

GPTQModel has a purpose-built large-M W4A16 prefill path that consumes
Marlin-packed INT4 weights and group scales directly. It never materializes or
caches the complete dequantized weight state. Decode stays on ordinary Marlin.

Automatic routing is enabled when a `MarlinLinear` module is loaded; no shell
export is required. The inference path first distinguishes prefill-like work by
flattened row count, then the native CUDA dispatcher checks GPU architecture,
quantization contract, and a conservative measured-shape table. A failed check
falls through to ordinary Marlin. `GPTQMODEL_MARLIN_PACKED_PREFILL=0` is an
optional opt-out, not a required opt-in.

## Automatic inference policy

The Python boundary sends packed-prefill candidates only when `M >= 1024`.
Explicit transformer inputs with sequence length one are always treated as
decode, even with a larger flattened batch. The CUDA boundary then requires:

- NVIDIA compute capability 8.0;
- symmetric GPTQ U4B8 weights with group size 128;
- full K, with no activation ordering or zero point; and
- one of the measured winning M/K/N shapes below.

| M | K | N | Kernel config |
|---:|---:|---:|---:|
| 1024 | 2048 | 8192 | 1 |
| 2048 | 2048 | 2048 | 1 |
| 2048 | 2048 | 8192 | 2 |

Everything else, including token-by-token decode, uses ordinary Marlin. The
optional `GPTQMODEL_MARLIN_PACKED_PREFILL_MIN_ROWS` and
`GPTQMODEL_MARLIN_PACKED_PREFILL_CONFIG` variables remain tuning/debug controls;
normal inference does not need them.

## Kernel design

`MarlinPrefill` is a separately generated CUDA kernel family. A CTA owns one
complete M/N output tile and traverses all of K. Packed INT4 words are loaded
and dequantized in registers immediately before tensor-core MMA. Because no
second CTA contributes to the same output tile, the path avoids ordinary
decode-oriented Marlin's striped cross-CTA K reduction and workspace locks.
It also writes no dense weight tensor to global memory.

Four launch shapes are compiled for BF16 and FP16:

| Config | Threads | M tile | N tile | K stage |
|---:|---:|---:|---:|---:|
| 1 | 128 | 64 | 128 | 64 |
| 2 | 128 | 64 | 256 | 64 |
| 3 | 256 | 32 | 512 | 64 |
| 4 | 64 | 64 | 128 | 64 |

## Test case

- GPU: PCI-ordered index 7, NVIDIA PG506-230, compute capability 8.0, 124 SMs,
  96 GiB
- Dtype: BF16; FP16 build and correctness were also checked
- Requested model root: `/monster/data/model/llama-3.2-1B-instruct`
- Existing case-sensitive root: `/monster/data/model/Llama-3.2-1B-Instruct`
- Checkpoint:
  `gptq_4bits_10-26_15-59-54_maxlen2048_ns128_descFalse_damp0.005`
- Quantization: GPTQ 4-bit, group size 128, symmetric, no activation order

The checkpoint contains 112 Marlin projections:

| K | N | Count |
|---:|---:|---:|
| 2048 | 512 | 32 |
| 2048 | 2048 | 32 |
| 2048 | 8192 | 32 |
| 8192 | 2048 | 16 |

### No-environment automatic-routing check

After changing the inference default, the model was loaded again on GPU 7 with
`GPTQMODEL_MARLIN_PACKED_PREFILL`, its row threshold, and its config override
all explicitly removed from the process environment. All 112 `MarlinLinear`
modules reported packed-prefill routing enabled on load, with the default
threshold 1024 and auto config 0.

A short selected/fallback projection smoke run then confirmed the actual
forward boundary: M=1 and unsupported K/N shapes were bit-identical to ordinary
Marlin, while the three auto-table shapes executed the packed path and remained
within tolerance (worst max absolute difference 0.03125). Thus an ordinary
`GPTQModel.load(..., backend=BACKEND.MARLIN)` now reaches the native automatic
selector without an export.

## Performance results

Projection timings are medians of 11 samples with 200 launches per sample.
Weighted time includes all 112 projections, including ordinary-Marlin
fallbacks.

| M | Selected KxN | Ordinary Marlin (ms) | Packed prefill (ms) | Projection speedup | Weighted speedup |
|---:|:---|---:|---:|---:|---:|
| 1024 | 2048x8192 | 0.17858 | 0.16610 | 1.075x | 1.014x |
| 2048 | 2048x2048 | 0.09261 | 0.09175 | 1.009x | included below |
| 2048 | 2048x8192 | 0.36217 | 0.30690 | 1.180x | 1.083x |

End-to-end prefill used 31 alternating trials after three warmups. Rows below
1024 deliberately execute the identical CUDA route and show measurement noise.

| Prompt tokens | Route | Ordinary Marlin (ms) | Automatic candidate (ms) | Paired ratio | Candidate wins |
|---:|:---|---:|---:|---:|---:|
| 128 | ordinary fallback | 27.085 | 26.744 | 1.025x | 18/31 |
| 512 | ordinary fallback | 26.888 | 28.283 | 0.961x | 10/31 |
| 1024 | packed selected | 30.768 | 30.737 | 1.008x | 18/31 |
| 2048 | packed selected | 76.072 | 73.783 | 1.030x | 30/31 |

At 2048 prompt tokens, separately timed decode had a 0.994x paired ratio and
all generated token IDs matched. The kernel therefore improves the measured
large prefill without changing decode dispatch.

## Correctness and memory

- All 20 final automatic-dispatch projection cases passed
  `atol=0.05, rtol=0.005`; worst BF16 max absolute difference was 0.03125.
- M=1 decode and all automatic fallbacks were bit-identical.
- A 96-case, two-seed sweep over all four configs and every projection shape
  passed; worst max difference was 0.0625.
- FP16 selected-shape checks had a worst max difference of 0.00390625.
- A transient dense reference for K=2048, N=8192 differed by at most 0.015625
  at M=1024 and matched at M=2048. That tensor is validation-only.
- Full Evalution `arc_challenge` evaluation (1,172 examples) was identical:
  raw accuracy 0.3506825938566553 and normalized accuracy
  0.3728668941979522 for both routes.

| Memory measurement | Bytes |
|:---|---:|
| Loaded model allocation | 1,028,322,304 |
| Packed qweights within model | 486,539,264 |
| Persistent prefill cache | 0 |
| Model-load peak allocation | 1,822,432,256 |

The rejected dense-cache prototype added 1.8125 GiB, which is why it was
removed. The production kernel keeps the quantized checkpoint resident and
dequantizes only fragments being consumed by MMA.

## Qwen3 8B follow-up

The packed kernel was also tested on 2026-07-20 with a complete, already
quantized Qwen3 8B checkpoint found under `/tmp`:

- checkpoint used for timing:
  `/tmp/llama3_2_scale_search_disabled_8labmwez/disabled`;
- independently load/forward-checked checkpoint:
  `/tmp/llama3_2_scale_search_qkvo_activation_else_hessian_4c6e2yyq/qkvo_activation_else_hessian`;
- model: `Qwen3ForCausalLM`, hidden size 4096, intermediate size 12288,
  36 transformer layers;
- quantization: GPTQ 4-bit, group size 128, symmetric, no activation order,
  INT32-packed weights; and
- device/dtype: PCI-ordered GPU 7, NVIDIA PG506-230, compute capability 8.0,
  124 SMs, 96 GiB, BF16 activations.

At test time, both checkpoint directories contained the config, tokenizer,
index, and two complete safetensor shards (4,271,572,432 and 1,832,344,848
bytes). The second checkpoint loaded as 252 `MarlinLinear` modules and
completed a forward pass with finite logits of shape `[1, 8, 151936]`. These
were transient host artifacts, were subsequently cleaned from `/tmp`, and are
not part of this commit.

The 252 projections have four shapes:

```text
K      N      Projection count
-----  -----  ----------------
4096   1024                 72
4096   4096                 72
4096  12288                 72
12288  4096                 36
```

### Current automatic behavior

Normal inference already enters the automatic native dispatcher without
`GPTQMODEL_MARLIN_PACKED_PREFILL=1`. However, the committed selector currently
contains only the measured Llama 3.2 K=2048 shapes listed above. Every Qwen3 8B
K=4096/12288 projection therefore falls back to ordinary Marlin. In other
words, automatic switching works, but the current production table gives this
model no packed-prefill speedup yet.

To determine whether Qwen entries are worth adding, the benchmark applied a
temporary exact-shape policy without modifying production code:

```text
M     K      N      Packed config  Result
----  -----  -----  -------------  ----------------
1024   4096  12288              1  select packed
2048   4096   4096              1  select packed
2048   4096  12288              2  select packed
2048  12288   4096              1  select packed
all other Qwen3 shapes               ordinary fallback
```

### Projection performance and accuracy

Each projection result is the median of 11 alternating-route samples after 20
warmups, with 200 launches per sample. Accuracy is against ordinary Marlin in
BF16; all selected cases passed `atol=0.05, rtol=0.005`.

```text
M     K      N      Config  Ordinary ms  Packed ms  Speedup  Max abs   Mean abs
----  -----  -----  ------  -----------  ---------  -------  --------  ---------
1024   4096  12288       1     0.497989   0.459892   1.083x  0.031250  1.319e-07
2048   4096   4096       1     0.331340   0.312355   1.061x  0.031250  2.054e-07
2048   4096  12288       2     1.011712   0.873220   1.159x  0.031250  2.226e-06
2048  12288   4096       1     1.052733   0.964508   1.091x  0.062500  1.523e-06
```

Weighted over all 252 projections, including fallback shapes:

```text
M     Ordinary projected ms  Policy projected ms  Speedup
----  ---------------------  -------------------  -------
1024              70.610798            67.867806   1.040x
2048             141.086481           126.572051   1.115x
```

Shape-specific gating is necessary. K=4096, N=1024 regressed with configs 1
and 2 (0.802x/0.794x at M=1024 and 0.840x/0.687x at M=2048). At M=1024,
K=4096, N=4096 with config 1 reached only 0.987x, and K=12288, N=4096 with
config 1 reached 0.941x. Configs 3 and 4 broadly regressed as well. Those cases
must remain on ordinary Marlin.

### End-to-end inference

End-to-end prefill used three warmups and 31 alternating-route trials; decode
used four generated tokens. Generated token IDs matched for every comparison.

```text
M     Ordinary prefill ms  Policy prefill ms  Paired speedup  Wins   Ordinary decode ms/token  Policy decode ms/token  Decode ratio  Tokens equal
----  -------------------  -----------------  --------------  -----  ------------------------  ----------------------  ------------  ------------
1024           121.293823         118.443008          1.024x  29/31                 61.209599               61.374977        0.991x  yes
2048           287.603699         273.669128          1.051x  31/31                 59.470848               59.158783        0.997x  yes
```

A separately materialized BF16 dense reference was used only for validation on
`model.layers.0.mlp.gate_proj` at M=1024, K=4096, N=12288. Ordinary Marlin and
packed config 1 each differed from dense by at most 0.03125 and passed
`atol=0.05, rtol=0.005`; packed versus ordinary also differed by at most
0.03125. No dense tensor was retained by inference.

The result supports extending the native automatic selector with the four
winning Qwen shape/config entries above. Until that selector change is made,
Qwen3 8B safely uses ordinary Marlin automatically; users should not force the
packed path globally because several Qwen shapes regress.

## Reproduce

No packed-prefill environment variable is needed:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. \
python scripts/benchmark_marlin_prefill_decode.py \
  --m-values 1,128,512,1024,2048 \
  --prompt-lengths 128,512,1024,2048 \
  --decode-tokens 4 \
  --warmup 20 \
  --iterations 200 \
  --layer-samples 11 \
  --model-warmup 3 \
  --model-runs 31 \
  --json-out /tmp/marlin_packed_prefill_ab.json
```

The detailed implementation notes are also in
[`docs/kernels/marlin_prefill_decode.md`](docs/kernels/marlin_prefill_decode.md).
