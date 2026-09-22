# SM90 rank-8 Hadamard epilogue

## Scope and numerical contract

This optimization targets the FP16 Hadamard/scale epilogue following the FP32
P32 base output and rank-8 correction. It does not change quantized weights,
the rank-8 projection, correction accumulation order, normalization placement,
or FP16 rounding boundaries.

The decode path stores adjacent FP16 values as `half2`, executes pair-index
stages below 32 with warp shuffles, and retains shared memory only for wider
stages. Each butterfly lane is converted to FP32 for scalar add/sub and rounded
explicitly back to FP16. Prefill and other matrices above M=128 retain the
compatibility kernel.

The initial native `__hadd2`/`__hsub2` prototype was not promoted. Its apparent
token-59 regression was first confounded by comparing a B1 run with a row from
a B128 run. A subsequent matched B128 test still found real stream drift. The
retained implementation restores the compatibility arithmetic while preserving
the packed layout and shuffle/data-movement savings.

## Oracle identity

The retained gate used:

- QVQ base `a66427ee331ec1e5e192a85e30c690cd3fe16d0b`;
- ZML-Ultra `5cc861f170acae9fc740cdc653bfb8e1f7c29666`;
- inference-ultra `ed538ce38ecdf090057cc4e270f7d0e70a3c8dd9`;
- the same F6 seed-7 Llama 3.2 1B snapshot, tokenizer, GSM8K-Platinum row keys,
  prompt tokens, B128 engine/request geometry, prefill bucket 960, context
  8192, generation limit, allocator policy, and H100 for both arms.

Control-versus-control and candidate-versus-candidate each produced 128/128
identical streams. The all-active B128 numerical gate used 8,192 physical KV
pages. The 544-page cold gate exhausts on untouched origin/main too, so it is a
separate latest-code serving/configuration blocker and was not attributed to
this kernel.

## Kernel results

CUDA-event timings are medians of nine samples, each containing 1,000 launches
after 100 warmup launches. M=128, rank=8, and both normalization modes compare
byte-identical inputs and output at the same operator boundary.

| N | Normalize first | Compatibility | Packed FP32 | Speedup | FP16 output |
|---:|:---:|---:|---:|---:|:---:|
| 512 | no | 4.817 us | 4.159 us | 1.158x | bitwise exact |
| 512 | yes | 4.401 us | 4.294 us | 1.025x | bitwise exact |
| 2,048 | no | 8.158 us | 5.640 us | 1.446x | bitwise exact |
| 2,048 | yes | 7.506 us | 5.651 us | 1.328x | bitwise exact |
| 8,192 | no | 23.132 us | 13.089 us | 1.767x | bitwise exact |
| 8,192 | yes | 20.415 us | 13.606 us | 1.500x | bitwise exact |

Runtime smoke coverage executes M=128 at N=16, 32, 64, 512, 2,048, and
8,192 against the scalar compatibility oracle. N=16/32 use the active-lane
shuffle mask rather than assuming a full warp.

## Full model result

The matched 1,209-row GSM8K-Platinum gate used KV=8,192 so every B128 request
could run without changing admission between arms.

| Metric | Compatibility | Packed FP32 | Change |
|---|---:|---:|---:|
| Exact token streams | 1,209/1,209 | 1,209/1,209 | no drift |
| Correct / invalid | 542 / 0 | 542 / 0 | unchanged |
| Prefill | 43.234 s | 43.230 s | unchanged |
| Decode | 54.631 s | 53.206 s | 2.678% faster |
| Useful decode | 2,096.5 tok/s | 2,152.7 tok/s | 2.678% faster |
| Padded decode | 5,695.8 tok/s | 5,848.4 tok/s | 2.678% faster |
| Per-stream decode | 44.50 tok/s | 45.69 tok/s | 2.678% faster |
| Wall time | 100.455 s | 99.114 s | 1.354% faster |

The model-level result is consistent with the epilogue's previously measured
single-digit share of decode time. Microkernel speedup must not be reported as
model-level speedup.
