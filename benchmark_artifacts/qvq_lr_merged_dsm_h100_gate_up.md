# QVQ LR versus W4 GPTQ kernels — Llama 3.2 1B shapes

This is a kernel-throughput comparison on the H100 host from PR #62. W4 GPTQ is a figurative performance baseline, not a quality-equivalent arm.

## Measurement contract

- Commit: `90df1c560c347b06bbfb5cb69a0821f7452a4570`; benchmark SHA256: `ae13dac8f763d972716bba127790a3bffea2c83f7d2e91d7479728527011b6ca`
- GPU: physical `1`, `NVIDIA H100`, PCI `00000000:44:00.0`, UUID `GPU-f5ea03cf-efa4-9807-7de5-b174957a1348`, CC `9.0`, 132 SMs
- Torch/CUDA: `2.15.0.dev20260828+cu130` / `13.0`; input dtype: `float16`
- M values: `[1, 2, 4, 8, 16]`; warmup: `20`; measured launches: `60`
- QVQ: `qvq_v2b2_p32_lr`, rates `[2.0, 2.5, 3.0, 3.5]`, native `P32 is LR packing geometry, not a GPTQ affine scale group.`
- GPTQ: symmetric W4, group `128`, no activation order; Marlin and Machete use the same W4 source payload
- Latency is CUDA-event median/P95 from one CUDA Graph replay containing every measured launch. CPU scheduling and host launch gaps are outside each timed interval. Logical TFLOP/s is `2*M*K*N / median_ms`; payload GB/s is packed payload bytes divided by median latency.
- Every row passed its dense-reference shape/dtype/finite gate. QVQ uses `max_abs <= 2e-3`; GPTQ uses `atol=2e-2, rtol=2e-2`.

`xMarlin` and `xMachete` are baseline median latency divided by the row median, so values above 1.0x are faster.

| Shape | Roles | M | K | N | Kernel | W | Group | Median ms | P95 ms | Logical TFLOP/s | Payload GB/s | xMarlin | xMachete | Max abs |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | qvq_lr | 2 | P32 | 0.0305 | 0.0309 | 1.100 | 139.69 | 0.322x | 0.592x | 4.86e-05 |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | qvq_lr | 2 | P32 | 0.0306 | 0.0310 | 2.190 | 139.03 | 0.331x | 0.586x | 5.15e-05 |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | qvq_lr | 2 | P32 | 0.0308 | 0.0311 | 4.360 | 138.38 | 0.335x | 0.589x | 5.91e-05 |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | qvq_lr | 2 | P32 | 0.0312 | 0.0316 | 8.608 | 136.60 | 0.321x | 0.578x | 5.53e-05 |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | qvq_lr | 2 | P32 | 0.0315 | 0.0319 | 17.033 | 135.15 | 0.346x | 0.575x | 5.34e-05 |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | qvq_lr | 2.5 | P32 | 0.0312 | 0.0316 | 1.077 | 170.40 | 0.315x | 0.579x | 5.72e-05 |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | qvq_lr | 2.5 | P32 | 0.0313 | 0.0318 | 2.142 | 169.45 | 0.324x | 0.573x | 6.1e-05 |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | qvq_lr | 2.5 | P32 | 0.0316 | 0.0319 | 4.254 | 168.24 | 0.327x | 0.575x | 5.53e-05 |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | qvq_lr | 2.5 | P32 | 0.0319 | 0.0323 | 8.422 | 166.55 | 0.314x | 0.565x | 5.34e-05 |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | qvq_lr | 2.5 | P32 | 0.0324 | 0.0328 | 16.595 | 164.08 | 0.337x | 0.560x | 6.1e-05 |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | qvq_lr | 3 | P32 | 0.0299 | 0.0303 | 1.122 | 212.58 | 0.329x | 0.604x | 2.67e-05 |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | qvq_lr | 3 | P32 | 0.0300 | 0.0304 | 2.233 | 211.56 | 0.338x | 0.597x | 3.24e-05 |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | qvq_lr | 3 | P32 | 0.0301 | 0.0306 | 4.453 | 210.89 | 0.342x | 0.601x | 2.86e-05 |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | qvq_lr | 3 | P32 | 0.0304 | 0.0309 | 8.839 | 209.33 | 0.330x | 0.593x | 2.96e-05 |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | qvq_lr | 3 | P32 | 0.0306 | 0.0310 | 17.540 | 207.69 | 0.357x | 0.592x | 3.34e-05 |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | qvq_lr | 3.5 | P32 | 0.0321 | 0.0325 | 1.046 | 230.96 | 0.306x | 0.563x | 5.91e-05 |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | qvq_lr | 3.5 | P32 | 0.0323 | 0.0327 | 2.076 | 229.13 | 0.314x | 0.555x | 4.29e-05 |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | qvq_lr | 3.5 | P32 | 0.0326 | 0.0329 | 4.116 | 227.11 | 0.316x | 0.556x | 5.72e-05 |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | qvq_lr | 3.5 | P32 | 0.0333 | 0.0337 | 8.058 | 222.31 | 0.301x | 0.541x | 6.1e-05 |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | qvq_lr | 3.5 | P32 | 0.0337 | 0.0340 | 15.948 | 219.98 | 0.324x | 0.538x | 6.68e-05 |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | gptq_marlin | 4 | 128 | 0.0098 | 0.0102 | 3.416 | 880.57 | 1.000x | 1.837x | 0.000218 |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | gptq_marlin | 4 | 128 | 0.0101 | 0.0106 | 6.616 | 852.79 | 1.000x | 1.770x | 0.000231 |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | gptq_marlin | 4 | 128 | 0.0103 | 0.0107 | 13.026 | 839.55 | 1.000x | 1.759x | 0.000319 |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | gptq_marlin | 4 | 128 | 0.0100 | 0.0103 | 26.801 | 863.69 | 1.000x | 1.799x | 0.000259 |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | gptq_marlin | 4 | 128 | 0.0109 | 0.0112 | 49.200 | 792.77 | 1.000x | 1.660x | 0.000311 |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | gptq_machete | 4 | 128 | 0.0180 | 0.0188 | 1.859 | 479.77 | 0.544x | 1.000x | 0.000218 |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | gptq_machete | 4 | 128 | 0.0180 | 0.0188 | 3.738 | 482.34 | 0.565x | 1.000x | 0.000231 |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | gptq_machete | 4 | 128 | 0.0181 | 0.0190 | 7.404 | 477.66 | 0.568x | 1.000x | 0.000272 |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | gptq_machete | 4 | 128 | 0.0180 | 0.0189 | 14.900 | 480.63 | 0.556x | 1.000x | 0.000259 |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | gptq_machete | 4 | 128 | 0.0181 | 0.0191 | 29.642 | 478.08 | 0.602x | 1.000x | 0.000311 |

The four shape rows preserve all seven Llama 3.2 1B projection roles: `q_proj/o_proj` (2048×2048), `k_proj/v_proj` (2048×512), `gate_proj/up_proj` (2048×8192), and `down_proj` (8192×2048).
