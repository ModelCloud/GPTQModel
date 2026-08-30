# QVQ LR versus W4 GPTQ kernels — Llama 3.2 1B shapes

This is a kernel-throughput comparison on the H100 host from PR #62. W4 GPTQ is a figurative performance baseline, not a quality-equivalent arm.

## Measurement contract

- Commit: `30c62b5336e90dc8db66fb9568b94a5713e5da98`; benchmark SHA256: `ae13dac8f763d972716bba127790a3bffea2c83f7d2e91d7479728527011b6ca`
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
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | qvq_lr | 2 | P32 | 0.0307 | 0.0310 | 1.093 | 138.81 | 0.322x | 0.584x | 4.86e-05 |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | qvq_lr | 2 | P32 | 0.0307 | 0.0311 | 2.189 | 138.96 | 0.333x | 0.592x | 5.15e-05 |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | qvq_lr | 2 | P32 | 0.0308 | 0.0312 | 4.351 | 138.09 | 0.335x | 0.586x | 5.91e-05 |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | qvq_lr | 2 | P32 | 0.0312 | 0.0317 | 8.599 | 136.46 | 0.321x | 0.577x | 5.53e-05 |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | qvq_lr | 2 | P32 | 0.0316 | 0.0320 | 17.007 | 134.94 | 0.345x | 0.570x | 5.34e-05 |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | qvq_lr | 2.5 | P32 | 0.0313 | 0.0317 | 1.072 | 169.53 | 0.315x | 0.573x | 5.72e-05 |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | qvq_lr | 2.5 | P32 | 0.0314 | 0.0317 | 2.136 | 168.93 | 0.325x | 0.577x | 6.1e-05 |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | qvq_lr | 2.5 | P32 | 0.0315 | 0.0319 | 4.263 | 168.59 | 0.328x | 0.574x | 5.53e-05 |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | qvq_lr | 2.5 | P32 | 0.0317 | 0.0323 | 8.465 | 167.39 | 0.316x | 0.568x | 5.34e-05 |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | qvq_lr | 2.5 | P32 | 0.0322 | 0.0328 | 16.652 | 164.65 | 0.338x | 0.558x | 6.1e-05 |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | qvq_lr | 3 | P32 | 0.0284 | 0.0288 | 1.180 | 223.46 | 0.347x | 0.630x | 2.67e-05 |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | qvq_lr | 3 | P32 | 0.0285 | 0.0290 | 2.358 | 223.33 | 0.359x | 0.637x | 3.24e-05 |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | qvq_lr | 3 | P32 | 0.0286 | 0.0291 | 4.689 | 222.09 | 0.361x | 0.632x | 2.86e-05 |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | qvq_lr | 3 | P32 | 0.0286 | 0.0291 | 9.373 | 221.96 | 0.350x | 0.628x | 2.96e-05 |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | qvq_lr | 3 | P32 | 0.0291 | 0.0295 | 18.447 | 218.42 | 0.374x | 0.618x | 3.34e-05 |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | qvq_lr | 3.5 | P32 | 0.0321 | 0.0326 | 1.044 | 230.50 | 0.307x | 0.558x | 5.91e-05 |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | qvq_lr | 3.5 | P32 | 0.0324 | 0.0328 | 2.070 | 228.45 | 0.315x | 0.560x | 4.29e-05 |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | qvq_lr | 3.5 | P32 | 0.0327 | 0.0331 | 4.104 | 226.44 | 0.316x | 0.553x | 5.72e-05 |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | qvq_lr | 3.5 | P32 | 0.0334 | 0.0337 | 8.027 | 221.46 | 0.300x | 0.538x | 6.1e-05 |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | qvq_lr | 3.5 | P32 | 0.0337 | 0.0340 | 15.948 | 219.98 | 0.324x | 0.535x | 6.68e-05 |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | gptq_marlin | 4 | 128 | 0.0099 | 0.0103 | 3.399 | 876.29 | 1.000x | 1.817x | 0.000218 |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | gptq_marlin | 4 | 128 | 0.0102 | 0.0106 | 6.574 | 847.45 | 1.000x | 1.777x | 0.000231 |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | gptq_marlin | 4 | 128 | 0.0103 | 0.0107 | 12.985 | 836.95 | 1.000x | 1.749x | 0.000319 |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | gptq_marlin | 4 | 128 | 0.0100 | 0.0104 | 26.801 | 863.69 | 1.000x | 1.797x | 0.000259 |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | gptq_marlin | 4 | 128 | 0.0109 | 0.0113 | 49.272 | 793.94 | 1.000x | 1.652x | 0.000311 |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | gptq_machete | 4 | 128 | 0.0179 | 0.0190 | 1.871 | 482.77 | 0.550x | 1.000x | 0.000218 |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | gptq_machete | 4 | 128 | 0.0181 | 0.0190 | 3.699 | 477.23 | 0.563x | 1.000x | 0.000231 |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | gptq_machete | 4 | 128 | 0.0181 | 0.0189 | 7.424 | 478.92 | 0.572x | 1.000x | 0.000272 |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | gptq_machete | 4 | 128 | 0.0180 | 0.0190 | 14.913 | 481.05 | 0.556x | 1.000x | 0.000259 |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | gptq_machete | 4 | 128 | 0.0180 | 0.0190 | 29.826 | 481.05 | 0.605x | 1.000x | 0.000311 |

The four shape rows preserve all seven Llama 3.2 1B projection roles: `q_proj/o_proj` (2048×2048), `k_proj/v_proj` (2048×512), `gate_proj/up_proj` (2048×8192), and `down_proj` (8192×2048).
