# Packed Marlin prefill/decode split

## Result

GPTQModel now has an experimental, purpose-built large-M W4A16 Marlin kernel.
It consumes the checkpoint's Marlin-packed INT4 weights and permuted group
scales directly. It does not construct, retain, or cache a dense FP16/BF16
weight matrix.

No environment variable is required. Inference enables conservative automatic
routing when each `MarlinLinear` module is constructed. To opt out before model
loading:

```bash
export GPTQMODEL_MARLIN_PACKED_PREFILL=0
```

The default policy considers flattened `M >= 1024`. Explicit transformer
decode inputs with sequence length one stay on ordinary Marlin. The CUDA
selector only dispatches measured winning shapes; every unsupported contract
or unproven shape falls through to ordinary Marlin.

Optional tuning controls are:

```bash
export GPTQMODEL_MARLIN_PACKED_PREFILL_MIN_ROWS=1024
export GPTQMODEL_MARLIN_PACKED_PREFILL_CONFIG=0  # 0=auto, 1..4=force a tile
```

The implementation is automatic but conservative because its auto table is
specific to the tested 124-SM `sm_80` target and checkpoint shapes. The native
dispatcher verifies the hardware and quantization contract before consulting
that table, then falls through to ordinary Marlin whenever no entry matches.

## Kernel design

Ordinary Marlin is optimized to keep tiny-M decode busy. It stripes work across
K and uses workspace locks/reduction when more than one CTA contributes to an
output tile. That scheduling is valuable when M does not expose enough output
tiles.

`MarlinPrefill` is a separately generated CUDA kernel family. For large M:

- one CTA owns one M/N output tile;
- that CTA traverses the complete K dimension;
- packed INT4 words are loaded and dequantized into registers immediately
  before tensor-core MMA;
- no dense weight is written to global memory;
- no cross-CTA K reduction or workspace lock is needed; and
- a partial final M tile is bounds checked.

Four validated launch shapes are compiled for BF16 and FP16:

| Config | Threads | M tile | N tile | K stage |
|---:|---:|---:|---:|---:|
| 1 | 128 | 64 | 128 | 64 |
| 2 | 128 | 64 | 256 | 64 |
| 3 | 256 | 32 | 512 | 64 |
| 4 | 64 | 64 | 128 | 64 |

Auto dispatch is deliberately narrow. On the test checkpoint it selects:

| M | K | N | Config |
|---:|---:|---:|---:|
| 1024 | 2048 | 8192 | 1 |
| 2048 | 2048 | 2048 | 1 |
| 2048 | 2048 | 8192 | 2 |

The native contract is full-K symmetric GPTQ U4B8, group size 128, no
activation order, no zero point, and compute capability 8.0. Other contracts
retain the existing Marlin path. This is consistent with the original Marlin
design's focus on near-ideal weight streaming at small-to-medium batch sizes;
see the [official Marlin repository](https://github.com/IST-DASLab/marlin) and
[paper](https://arxiv.org/abs/2408.11743).

## Test setup

- Physical PCI-ordered GPU index: 7 (`CUDA_VISIBLE_DEVICES=7`)
- GPU: NVIDIA PG506-230, compute capability 8.0, 124 SMs, 96 GiB
- Dtype: BF16, with an additional FP16 build/correctness check
- Requested model root: `/monster/data/model/llama-3.2-1B-instruct`
- Case-sensitive root present on the host:
  `/monster/data/model/Llama-3.2-1B-Instruct`
- Checkpoint:
  `gptq_4bits_10-26_15-59-54_maxlen2048_ns128_descFalse_damp0.005`
- Quantization: GPTQ 4-bit, group size 128, symmetric, no activation order

The model has 112 Marlin projections:

| K | N | Count |
|---:|---:|---:|
| 2048 | 512 | 32 |
| 2048 | 2048 | 32 |
| 2048 | 8192 | 32 |
| 8192 | 2048 | 16 |

## Performance

Projection timings are medians of 11 samples, each containing 200 launches.
The selected large projection improves by 1.075x at M=1024 and 1.180x at
M=2048. Model-weighted projection time includes all 112 projections and their
fallbacks.

| M | Selected KxN | Selected Marlin (ms) | Packed prefill (ms) | Selected speedup | Weighted speedup |
|---:|---:|---:|---:|---:|---:|
| 1024 | 2048x8192 | 0.17858 | 0.16610 | 1.075x | 1.014x |
| 2048 | 2048x8192 | 0.36217 | 0.30690 | 1.180x | 1.083x |

End-to-end results use 31 alternating trials after three warmups. Four decode
steps were timed separately. Rows below the threshold execute the same ordinary
Marlin CUDA path and are included to expose run-to-run timing noise.

| Prompt | CUDA route | Marlin prefill (ms) | Feature-on prefill (ms) | Paired ratio | Feature wins |
|---:|:---|---:|---:|---:|---:|
| 128 | ordinary fallback | 27.085 | 26.744 | 1.025x | 18/31 |
| 512 | ordinary fallback | 26.888 | 28.283 | 0.961x | 10/31 |
| 1024 | packed selected | 30.768 | 30.737 | 1.008x | 18/31 |
| 2048 | packed selected | 76.072 | 73.783 | 1.030x | 30/31 |

The 2048-token decode paired ratio was 0.994x. Decode dispatch is unchanged,
and generated token IDs matched for every prompt length.

## Memory and correctness

The 112 packed `qweight` tensors occupy 486,539,264 bytes. Enabling packed
prefill adds zero persistent model bytes:

| Measurement | Bytes |
|:---|---:|
| Loaded model allocation | 1,028,322,304 |
| Packed qweights within model | 486,539,264 |
| Persistent prefill cache | 0 |
| Model-load peak allocation | 1,822,432,256 |

All 20 final auto-dispatch projection cases passed
`atol=0.05, rtol=0.005`; the worst BF16 max absolute delta from ordinary
Marlin was 0.03125. M=1 decode and every auto fallback were bit-identical.
An additional 96-case, two-seed sweep across all four manual configs and every
model projection shape also passed (worst max delta 0.0625). FP16
selected-shape checks had a worst max delta of 0.00390625.

A transient dense-reference check used the GPTQ Torch backend to dequantize one
`K=2048, N=8192` projection only for validation. Packed prefill differed by at
most 0.015625 at M=1024 and matched the dense BF16 GEMM exactly at M=2048.
This dense tensor is not part of the Marlin runtime.

Evalution `arc_challenge` (1,172 samples, batch 64) produced identical current
baseline and feature-on metrics:

| Route | Accuracy | Normalized accuracy |
|:---|---:|---:|
| Ordinary Marlin | 0.350683 | 0.372867 |
| Packed prefill | 0.350683 | 0.372867 |

## Rejected designs

- A persistent BF16 weight cache was rejected because it duplicated all
  quantized projections and added 1.8125 GiB for this model.
- A conventional kernel that expanded a 128x128 INT4 tile into shared memory
  was numerically correct but achieved only about 0.23x weighted Marlin
  throughput because of low occupancy and repeated expansion.
- Two experimental direct-grid tile shapes showed intermittent corruption and
  were removed before the production family was generated.
- Smaller-M routes were removed from auto dispatch when paired model testing
  showed no stable benefit.

## Reproduce

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

The script records paired layer outputs, per-sample timings, generated token
IDs, projection inventory, and CUDA memory measurements in the JSON artifact.
