# Integrated window wave 7 partial result

GPU6 completed its BM64/BN32/split1 assignment against the immutable F6 seed-7
checkpoint: three real projections, 12 row counts, 36/36 local correctness
passes. The selected projection medians at M=1/16/512/2048 were:

| Projection | M=1 | M=16 | M=512 | M=2048 |
|---|---:|---:|---:|---:|
| layer-0 v | 0.966x | 0.877x | 0.975x | 0.957x |
| layer-1 q | 0.990x | 0.985x | 0.901x | 1.155x |
| layer-1 up | 1.019x | 0.918x | 1.176x | 1.290x |

The other three BM64/BN32 assignments were still running when this partial
result was collected. Raw report: [GPU 6](results/window-fused-wave7-bm64/gpu6.json).
