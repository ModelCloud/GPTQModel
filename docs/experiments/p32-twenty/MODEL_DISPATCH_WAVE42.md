# Full-model BM64 dispatcher wave 42

Wave 42 reran four production-window controls and four BM64/BN32 arms after
fixing the subthreshold dispatcher. With `--ampere-min-rows 512`, M<512 now
uses the production window kernel rather than the dense reconstructed inner
path. All eight jobs ran on distinct physical A100-class GPU UUIDs.

The pooled median prefill latency was:

| M | Window | BM64/BN32 dispatch | Speedup |
|---:|---:|---:|---:|
| 1 | 40.303 ms | 40.772 ms | 0.988× |
| 2 | 40.434 ms | 40.263 ms | 1.004× |
| 4 | 40.625 ms | 40.699 ms | 0.998× |
| 8 | 40.119 ms | 39.716 ms | 1.010× |
| 16 | 41.260 ms | 41.895 ms | 0.985× |
| 32 | 41.340 ms | 40.044 ms | 1.032× |
| 128 | 41.258 ms | 40.995 ms | 1.006× |
| 512 | 76.150 ms | 53.969 ms | 1.411× |
| 2048 | 289.670 ms | 203.357 ms | 1.425× |

Decode medians were 41.024 ms for window and 39.558 ms for the dispatched
arm. All four runs of each arm reported C4-style PPL 26.991525 on the fixed
16-document slice. No downstream benchmark was used for calibration or run
in this timing wave.

The M<512 result is now a valid dispatcher comparison. It confirms that the
direct kernel should be selected only at large prefill, with the conservative
initial policy `M >= 512`; it does not support a global replacement or the 3×
target. The four control and four candidate reports, including per-sample
latencies and model logs, are in [raw wave 42 results](results/model-dispatch-wave42/).

Wave 41's full-model Nsight traces remain useful for kernel composition, but
its subthreshold BM64 arm used the old dense fallback and must not be used for
low-M dispatcher claims. The corrected wave 42 is the authoritative timing
record.
