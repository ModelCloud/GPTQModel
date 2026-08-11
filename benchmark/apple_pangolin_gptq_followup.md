# Apple Pangolin and GPTQ follow-up

Measured from `origin/main` at `1871be2` on an Apple M4 Max with Python
3.10.11, PyTorch `2.14.0.dev20260806`, and MLX 0.32.0. Runs used performance
QoS via `taskpolicy -t 1 -l 1` and six-thread BLAS/OpenMP limits. Kernel
measurements used 10 warmups followed by seven samples of 10 calls; every
sample was synchronized with `torch.mps.synchronize()` or `mx.eval()` and the
median per-call latency is reported. Inputs use FP16 activations/scales,
int32-packed weights, and group size 128.

## Pangolin inference

The MPS M=16 baseline is the former pair of M=8 launches. The MLX baseline is
the former independent-row kernel. Every old/new output was bit-identical.

| Backend and shape | b2 | b3 | b4 | b5 | b6 | b7 | b8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| MPS M16, K=N=1024 | 2.31x | 1.51x | 1.45x | 1.50x | 1.49x | 1.53x | 1.45x |
| MPS M16, K=N=4096 | 1.48x | 1.58x | 1.51x | 1.56x | 1.58x | 1.59x | 1.54x |
| MLX M9, K=N=1024 | 2.87x | 3.42x | 1.96x | 3.37x | 3.38x | 3.23x | 1.96x |
| MLX M16, K=N=1024 | 3.56x | 5.94x | 3.31x | 6.00x | 5.91x | 5.87x | 3.18x |
| MLX M17, K=N=1024 | 2.07x | 3.62x | 2.04x | 3.50x | 3.50x | 3.26x | 2.00x |
| MLX M32, K=N=1024 | 3.55x | 6.20x | 3.38x | 5.96x | 5.84x | 5.57x | 3.09x |

Division-free continuous MPS decode also improves the high-frequency small-M
path without changing its layout or accumulation order:

| K=N=4096 | b2 | b4 | b8 |
|---|---:|---:|---:|
| M1 | 1.78x | 1.06x | 1.13x |
| M3 | 1.47x | 1.25x | 1.08x |

The retained caches contain only one immutable MPS shader library and at most
three immutable MLX kernels (plus at most one fixed error string per mode).
They do not retain activations, packed weights, decoded weights, or outputs.
Operand-lifetime and concurrent-launch tests cover the new multirow path.

## GPTQ Hessian lifecycle

The optimized path removes a host barrier after an unchunked MPS `X.T @ X`.
Chunked workspace reuse and non-MPS backends retain their existing barriers;
`GPTQMODEL_MPS_ASYNC_HESSIAN=0` restores the old MPS behavior.

| Tokens | Columns | Batches | Sync accumulation | Async accumulation | Accumulation speedup | Sync full lifecycle | Async full lifecycle | Full speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 128 | 16 | 2.806 ms | 0.687 ms | 4.08x | 7.104 ms | 4.978 ms | 1.43x |
| 32 | 128 | 16 | 2.785 ms | 0.713 ms | 3.91x | 7.103 ms | 4.952 ms | 1.43x |
| 128 | 128 | 16 | 2.930 ms | 0.761 ms | 3.85x | 7.324 ms | 5.022 ms | 1.46x |
| 128 | 512 | 8 | 1.628 ms | 0.492 ms | 3.31x | 12.023 ms | 10.745 ms | 1.12x |
| 128 | 1024 | 4 | 1.012 ms | 0.528 ms | 1.92x | 20.184 ms | 19.489 ms | 1.04x |

Synchronized and asynchronous end-to-end GPTQ outputs were bit-identical for
2--8 bits. Tests also cover immediate host reads, repeated accumulation into
one output, input and temporary destruction before completion, 64 fresh-output
GC cycles with bounded live allocator usage, and independent Python threads.

Rejected experiments were not retained: routing M16 through the 256-thread M32
kernel regressed 21--31%, while applying division replacement to all continuous
paths regressed the shared M4--32 kernels by roughly 40--48%.
