# Window model integration wave 20

This bounded model wave exercised the fused-window promotion paths through the
Llama 3.2 1B model on the exact F6 seed-7 snapshot. The production window arm
is the same-wave reference. Each arm used 16 fixed model input rows, nine
prefill sizes, and a 32-token growing-KV decode run. The reports contain ten
warmed samples per prefill size and 32 decode samples.

The runs completed on separate A100 GPUs with the following configuration:

| Arm | GPU index | Window configuration | PPL |
|---|---:|---|---:|
| window | 7 | production window | 26.9915252 |
| K0 | 0 | exact FP32 fused accumulation | 26.9936504 |
| K16 | 1 | FP16 partials, FP32 promotion every K16 | 26.9968585 |
| K32 | 2 | FP16 partials, FP32 promotion every K32 | 26.9968193 |
| K64 | 3 | FP16 partials, FP32 promotion every K64 | 26.9919274 |
| K128 | 4 | FP16 partials, FP32 promotion every K128 | 26.9889928 |
| K256 | 5 | FP16 partials, FP32 promotion every K256 | 26.9876831 |
| policy | 6 | K256 except layer-1 down uses exact K0 | 26.9933601 |

All eight reports completed with 16 quality rows and 10 performance rows. The
PPL differences are small on this bounded input set and are not a downstream
quality conclusion. ARC, GSM8K, broader held-out replay, and a repeated matched
timing run are still required.

## Full-model timing medians

Values are milliseconds. Prefill rows are full-model forward passes for the
listed token count. Decode is the report median for prompt 128 and 32 new
tokens. The production-window M=1 control contains an anomalous 331.4 ms
median; its raw samples show the first sample at 49.4 ms followed by roughly
330 ms samples. That makes the apparent M=1 candidate speedups unusable until
the matched repeat wave confirms or rejects the control behavior.

| Arm | M=1 | M=2 | M=4 | M=8 | M=16 | M=32 | M=128 | M=512 | M=2048 | decode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| window | 331.399 | 41.565 | 39.997 | 40.422 | 41.305 | 41.600 | 42.294 | 75.925 | 289.881 | 43.384 |
| K0 | 46.752 | 48.492 | 47.021 | 46.390 | 46.324 | 45.732 | 45.901 | 56.907 | 386.220 | 46.081 |
| K16 | 49.517 | 47.765 | 48.517 | 49.039 | 48.437 | 47.694 | 50.113 | 70.303 | 246.680 | 48.142 |
| K32 | 47.741 | 46.936 | 46.866 | 47.121 | 47.410 | 46.015 | 46.820 | 380.639 | 218.101 | 48.748 |
| K64 | 47.792 | 47.620 | 49.619 | 48.177 | 47.782 | 49.838 | 47.480 | 59.537 | 215.090 | 49.969 |
| K128 | 49.607 | 48.460 | 47.539 | 46.406 | 47.757 | 47.769 | 49.268 | 59.429 | 213.222 | 47.858 |
| K256 | 48.372 | 54.316 | 46.167 | 49.629 | 47.067 | 49.245 | 47.486 | 385.363 | 211.896 | 46.072 |
| policy | 47.865 | 47.406 | 47.863 | 46.851 | 46.857 | 48.277 | 47.294 | 412.753 | 212.852 | 47.467 |

Using the raw medians, the fused arms are slower than the same-wave control at
most small and medium sizes because the control anomaly only affects M=1 and
the fused model path retains extra boundary work. At M=2048, K16/K32/K64/K128/
K256/policy have apparent ratios of 1.175x/1.329x/1.348x/1.360x/1.368x/1.362x
against the control. Decode ratios are below 1.0x for every fused arm. These
are single-run model measurements and require the repeat wave before use as a
performance claim.

The model hook replaced the P32 inner linear operation while retaining the
production transform and output boundaries. It did not run ARC or GSM8K in
this wave, and it did not provide Nsight counters for Tensor Core activity,
registers, occupancy, shared memory, L2 traffic, or decoder/GEMM overlap.

Raw reports:

- [window](results/window-model-wave20/window-gpu7.json)
- [K0](results/window-model-wave20/k0-gpu0.json)
- [K16](results/window-model-wave20/k16-gpu1.json)
- [K32](results/window-model-wave20/k32-gpu2.json)
- [K64](results/window-model-wave20/k64-gpu3.json)
- [K128](results/window-model-wave20/k128-gpu4.json)
- [K256](results/window-model-wave20/k256-gpu5.json)
- [policy](results/window-model-wave20/policy-gpu6.json)
