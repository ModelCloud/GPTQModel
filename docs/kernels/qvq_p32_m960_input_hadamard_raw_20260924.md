# H100 M960/N8192 input Hadamard: framework-neutral raw ABI

The Llama 3.2 1B QVQ down projection prepares its FP16 input with an SU
product, FP16-rounded sqrt normalization, and 13 ascending FP16-rounded
Hadamard butterfly stages. ZML's StableHLO path emits 13 short butterfly
kernels for M960/N8192. The new SM90 raw ABI executes the same boundaries in
two kernels, splitting after the first eight tile-local stages. It reads the
compressed P32 projection separately; it does **not** materialize a dense
weight matrix or introduce a weight cache.

`qvq_hadamard_input_raw_launch` accepts caller-owned input, SU, output, a
versioned `{rows,width}` config, and the caller's CUDA stream. Its workspace
may be the output buffer itself: the high-stage thread loads all 32 values it
owns before writing any of them. The kernel allocates no GPU memory, performs
no synchronization on the host, and does not autotune. Only N8192 and
1–960 rows are compiled; ZML's production dispatch is restricted further to
FP16 M960 on a single SM90 device. All other cases retain StableHLO.

## Correctness and resource gate

- FP16 outputs match the independent staged reference bit-for-bit at rows 1,
  16, and 960. Changed-input CUDA graph replay passes. Output/workspace alias
  is covered by all three shapes.
- The matched full GSM8K-Platinum B128/M960 queue produces 1,209/1,209 exact
  token streams versus the prior baseline, 542/1,209 correct, zero invalid.
- NCU basic report on the final aliased M960 kernels:

| Kernel | GPU duration | Registers/thread | Shared/block | Local memory |
| --- | ---: | ---: | ---: | ---: |
| Low eight stages | 51.968 µs | 24 | 1.552 KiB | 0 |
| High five stages | 12.800 µs | 40 | 1.024 KiB | 0 |

The NCU report is local at
`/root/qvq-profiler-artifacts/hadamard-input-20260924/ncu-final.ncu-rep`.
`cuobjdump --dump-resource-usage` independently reports zero stack/local
memory for both kernels. Profile binary files are not committed.

## Matched B128 full-suite gate

Same optimized ZML runner, checkpoint, dataset, B128/M960 queue, GPU-local CPU
set, FA2, rank-8 enabled, and warmed graph. Only the pinned QVQ raw library
selection changes between the merged control and candidate. Candidate values
below are the final alias-safe candidate-after-control run; two earlier
candidate runs gave 38,214–38,228 useful prefill tok/s with the same exact
token streams.

| Metric | Merged control | Final candidate |
| --- | ---: | ---: |
| Useful prefill tok/s | 37,277.20 | **38,257.17** |
| Padded prefill tok/s | 42,995.91 | **44,126.22** |
| Useful decode tok/s | 10,188.00 | 10,202.74 |
| Padded decode tok/s | 11,871.34 | 11,888.51 |
| Padded decode tok/s per stream | 92.74 | 92.88 |
| Wall seconds | 38.806 | 38.100 |
| Correct / invalid / exact streams | 542 / 0 / 1,209 | 542 / 0 / 1,209 |

Useful prefill improves 2.63% in the matched comparison while decode is flat
within the observed run-to-run band. This is a forward increment, **not** the
overall 2× prefill target. The raw JSON records remain local under
`/var/tmp/qvq-had-input-*-20260924.json`.

The original 13-stage trace was ~85 µs for the M960/N8192 butterfly sequence;
the native two-stage path is ~65 µs under NCU and ~61 µs in an isolated warm
CUDA graph. Those are component measurements, not substitutes for the full
queue result above.
