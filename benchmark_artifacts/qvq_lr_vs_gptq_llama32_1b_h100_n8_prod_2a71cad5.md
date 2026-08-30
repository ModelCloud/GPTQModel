# QVQ LR versus W4 GPTQ kernels — Llama 3.2 1B shapes

This is a kernel-throughput comparison on the H100 host from PR #62. W4 GPTQ is a figurative performance baseline, not a quality-equivalent arm.

## Measurement contract

- Commit: `2a71cad5b1d418a764149fcad08b37fd0d511d32`; benchmark SHA256: `ae13dac8f763d972716bba127790a3bffea2c83f7d2e91d7479728527011b6ca`
- GPU: physical `1`, `NVIDIA H100`, PCI `00000000:44:00.0`, UUID `GPU-f5ea03cf-efa4-9807-7de5-b174957a1348`, CC `9.0`, 132 SMs
- Torch/CUDA: `2.15.0.dev20260828+cu130` / `13.0`; input dtype: `float16`
- M values: `[1, 2, 4, 8, 16]`; warmup: `20`; measured launches: `60`
- QVQ: `qvq_v2b2_p32_lr`, rates `[2.0, 2.5]`, native `P32 is LR packing geometry, not a GPTQ affine scale group.`
- GPTQ: symmetric W4, group `128`, no activation order; Marlin and Machete use the same W4 source payload
- Latency is CUDA-event median/P95 from one CUDA Graph replay containing every measured launch. CPU scheduling and host launch gaps are outside each timed interval. Logical TFLOP/s is `2*M*K*N / median_ms`; payload GB/s is packed payload bytes divided by median latency.
- Every row passed its dense-reference shape/dtype/finite gate. QVQ uses `max_abs <= 2e-3`; GPTQ uses `atol=2e-2, rtol=2e-2`.

`xMarlin` and `xMachete` are baseline median latency divided by the row median, so values above 1.0x are faster.

| Shape | Roles | M | K | N | Kernel | W | Group | Median ms | P95 ms | Logical TFLOP/s | Payload GB/s | xMarlin | xMachete | Max abs |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| attn_qo | q_proj/o_proj | 1 | 2048 | 2048 | qvq_lr | 2 | P32 | 0.0168 | 0.0176 | 0.501 | 63.57 | 0.870x | 0.921x | 7.63e-06 |
| attn_qo | q_proj/o_proj | 2 | 2048 | 2048 | qvq_lr | 2 | P32 | 0.0185 | 0.0192 | 0.909 | 57.68 | 0.884x | 0.828x | 1.34e-05 |
| attn_qo | q_proj/o_proj | 4 | 2048 | 2048 | qvq_lr | 2 | P32 | 0.0182 | 0.0192 | 1.841 | 58.44 | 0.899x | 0.847x | 1.43e-05 |
| attn_qo | q_proj/o_proj | 8 | 2048 | 2048 | qvq_lr | 2 | P32 | 0.0185 | 0.0192 | 3.631 | 57.63 | 0.805x | 0.833x | 1.34e-05 |
| attn_qo | q_proj/o_proj | 16 | 2048 | 2048 | qvq_lr | 2 | P32 | 0.0186 | 0.0192 | 7.232 | 57.38 | 0.852x | 0.839x | 1.34e-05 |
| attn_qo | q_proj/o_proj | 1 | 2048 | 2048 | qvq_lr | 2.5 | P32 | 0.0171 | 0.0177 | 0.490 | 77.59 | 0.852x | 0.902x | 7.63e-06 |
| attn_qo | q_proj/o_proj | 2 | 2048 | 2048 | qvq_lr | 2.5 | P32 | 0.0194 | 0.0199 | 0.864 | 68.38 | 0.841x | 0.788x | 1.43e-05 |
| attn_qo | q_proj/o_proj | 4 | 2048 | 2048 | qvq_lr | 2.5 | P32 | 0.0196 | 0.0202 | 1.708 | 67.54 | 0.834x | 0.786x | 1.14e-05 |
| attn_qo | q_proj/o_proj | 8 | 2048 | 2048 | qvq_lr | 2.5 | P32 | 0.0196 | 0.0201 | 3.430 | 67.82 | 0.760x | 0.787x | 1.34e-05 |
| attn_qo | q_proj/o_proj | 16 | 2048 | 2048 | qvq_lr | 2.5 | P32 | 0.0199 | 0.0205 | 6.732 | 66.57 | 0.793x | 0.781x | 1.34e-05 |
| attn_qo | q_proj/o_proj | 1 | 2048 | 2048 | gptq_marlin | 4 | 128 | 0.0146 | 0.0150 | 0.576 | 148.37 | 1.000x | 1.058x | 0.000264 |
| attn_qo | q_proj/o_proj | 2 | 2048 | 2048 | gptq_marlin | 4 | 128 | 0.0163 | 0.0166 | 1.028 | 132.52 | 1.000x | 0.937x | 0.000229 |
| attn_qo | q_proj/o_proj | 4 | 2048 | 2048 | gptq_marlin | 4 | 128 | 0.0164 | 0.0167 | 2.048 | 132.00 | 1.000x | 0.942x | 0.000197 |
| attn_qo | q_proj/o_proj | 8 | 2048 | 2048 | gptq_marlin | 4 | 128 | 0.0149 | 0.0152 | 4.510 | 145.34 | 1.000x | 1.034x | 0.000242 |
| attn_qo | q_proj/o_proj | 16 | 2048 | 2048 | gptq_marlin | 4 | 128 | 0.0158 | 0.0161 | 8.490 | 136.81 | 1.000x | 0.985x | 0.00028 |
| attn_qo | q_proj/o_proj | 1 | 2048 | 2048 | gptq_machete | 4 | 128 | 0.0154 | 0.0163 | 0.544 | 140.75 | 0.945x | 1.000x | 0.000264 |
| attn_qo | q_proj/o_proj | 2 | 2048 | 2048 | gptq_machete | 4 | 128 | 0.0153 | 0.0161 | 1.097 | 141.92 | 1.067x | 1.000x | 0.000229 |
| attn_qo | q_proj/o_proj | 4 | 2048 | 2048 | gptq_machete | 4 | 128 | 0.0154 | 0.0164 | 2.173 | 140.60 | 1.061x | 1.000x | 0.000197 |
| attn_qo | q_proj/o_proj | 8 | 2048 | 2048 | gptq_machete | 4 | 128 | 0.0154 | 0.0162 | 4.360 | 141.04 | 0.967x | 1.000x | 0.000242 |
| attn_qo | q_proj/o_proj | 16 | 2048 | 2048 | gptq_machete | 4 | 128 | 0.0156 | 0.0164 | 8.621 | 139.45 | 1.015x | 1.000x | 0.00028 |
| attn_kv | k_proj/v_proj | 1 | 2048 | 512 | qvq_lr | 2 | P32 | 0.0126 | 0.0133 | 0.167 | 21.17 | 1.785x | 1.172x | 5.72e-06 |
| attn_kv | k_proj/v_proj | 2 | 2048 | 512 | qvq_lr | 2 | P32 | 0.0127 | 0.0133 | 0.330 | 20.93 | 2.045x | 1.171x | 7.63e-06 |
| attn_kv | k_proj/v_proj | 4 | 2048 | 512 | qvq_lr | 2 | P32 | 0.0128 | 0.0134 | 0.658 | 20.88 | 2.065x | 1.181x | 6.68e-06 |
| attn_kv | k_proj/v_proj | 8 | 2048 | 512 | qvq_lr | 2 | P32 | 0.0127 | 0.0134 | 1.324 | 21.01 | 1.827x | 1.159x | 7.63e-06 |
| attn_kv | k_proj/v_proj | 16 | 2048 | 512 | qvq_lr | 2 | P32 | 0.0129 | 0.0134 | 2.605 | 20.67 | 1.906x | 1.168x | 7.63e-06 |
| attn_kv | k_proj/v_proj | 1 | 2048 | 512 | qvq_lr | 2.5 | P32 | 0.0133 | 0.0139 | 0.158 | 24.98 | 1.690x | 1.110x | 5.72e-06 |
| attn_kv | k_proj/v_proj | 2 | 2048 | 512 | qvq_lr | 2.5 | P32 | 0.0129 | 0.0136 | 0.324 | 25.66 | 2.012x | 1.152x | 8.11e-06 |
| attn_kv | k_proj/v_proj | 4 | 2048 | 512 | qvq_lr | 2.5 | P32 | 0.0134 | 0.0141 | 0.627 | 24.80 | 1.969x | 1.126x | 6.68e-06 |
| attn_kv | k_proj/v_proj | 8 | 2048 | 512 | qvq_lr | 2.5 | P32 | 0.0129 | 0.0139 | 1.301 | 25.73 | 1.795x | 1.139x | 7.63e-06 |
| attn_kv | k_proj/v_proj | 16 | 2048 | 512 | qvq_lr | 2.5 | P32 | 0.0130 | 0.0135 | 2.576 | 25.47 | 1.885x | 1.155x | 9.54e-06 |
| attn_kv | k_proj/v_proj | 1 | 2048 | 512 | gptq_marlin | 4 | 128 | 0.0224 | 0.0230 | 0.093 | 24.09 | 1.000x | 0.656x | 0.000167 |
| attn_kv | k_proj/v_proj | 2 | 2048 | 512 | gptq_marlin | 4 | 128 | 0.0260 | 0.0265 | 0.161 | 20.78 | 1.000x | 0.573x | 0.000155 |
| attn_kv | k_proj/v_proj | 4 | 2048 | 512 | gptq_marlin | 4 | 128 | 0.0263 | 0.0267 | 0.319 | 20.53 | 1.000x | 0.572x | 0.000167 |
| attn_kv | k_proj/v_proj | 8 | 2048 | 512 | gptq_marlin | 4 | 128 | 0.0232 | 0.0236 | 0.725 | 23.35 | 1.000x | 0.634x | 0.000206 |
| attn_kv | k_proj/v_proj | 16 | 2048 | 512 | gptq_marlin | 4 | 128 | 0.0245 | 0.0249 | 1.367 | 22.03 | 1.000x | 0.613x | 0.000194 |
| attn_kv | k_proj/v_proj | 1 | 2048 | 512 | gptq_machete | 4 | 128 | 0.0147 | 0.0155 | 0.142 | 37.25 | 1.523x | 1.000x | 0.000167 |
| attn_kv | k_proj/v_proj | 2 | 2048 | 512 | gptq_machete | 4 | 128 | 0.0149 | 0.0158 | 0.282 | 36.85 | 1.747x | 1.000x | 0.000155 |
| attn_kv | k_proj/v_proj | 4 | 2048 | 512 | gptq_machete | 4 | 128 | 0.0151 | 0.0158 | 0.557 | 36.45 | 1.749x | 1.000x | 0.000167 |
| attn_kv | k_proj/v_proj | 8 | 2048 | 512 | gptq_machete | 4 | 128 | 0.0147 | 0.0155 | 1.142 | 37.37 | 1.576x | 1.000x | 0.000206 |
| attn_kv | k_proj/v_proj | 16 | 2048 | 512 | gptq_machete | 4 | 128 | 0.0150 | 0.0157 | 2.231 | 36.49 | 1.632x | 1.000x | 0.000194 |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | qvq_lr | 2 | P32 | 0.0502 | 0.0505 | 0.668 | 84.79 | 0.196x | 0.356x | 5.72e-05 |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | qvq_lr | 2 | P32 | 0.0503 | 0.0506 | 1.333 | 84.63 | 0.201x | 0.356x | 5.15e-05 |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | qvq_lr | 2 | P32 | 0.0506 | 0.0511 | 2.654 | 84.23 | 0.201x | 0.353x | 5.72e-05 |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | qvq_lr | 2 | P32 | 0.0502 | 0.0505 | 5.352 | 84.93 | 0.197x | 0.358x | 7.63e-05 |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | qvq_lr | 2 | P32 | 0.0503 | 0.0506 | 10.669 | 84.66 | 0.214x | 0.357x | 6.1e-05 |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | qvq_lr | 2.5 | P32 | 0.0486 | 0.0490 | 0.691 | 109.28 | 0.202x | 0.369x | 4.86e-05 |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | qvq_lr | 2.5 | P32 | 0.0488 | 0.0492 | 1.375 | 108.78 | 0.207x | 0.367x | 5.53e-05 |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | qvq_lr | 2.5 | P32 | 0.0491 | 0.0495 | 2.736 | 108.21 | 0.207x | 0.364x | 6.29e-05 |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | qvq_lr | 2.5 | P32 | 0.0492 | 0.0495 | 5.452 | 107.82 | 0.201x | 0.365x | 5.34e-05 |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | qvq_lr | 2.5 | P32 | 0.0493 | 0.0498 | 10.880 | 107.58 | 0.218x | 0.364x | 6.87e-05 |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | gptq_marlin | 4 | 128 | 0.0098 | 0.0102 | 3.416 | 880.57 | 1.000x | 1.822x | 0.000272 |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | gptq_marlin | 4 | 128 | 0.0101 | 0.0105 | 6.637 | 855.49 | 1.000x | 1.772x | 0.000299 |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | gptq_marlin | 4 | 128 | 0.0101 | 0.0105 | 13.231 | 852.79 | 1.000x | 1.762x | 0.000274 |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | gptq_marlin | 4 | 128 | 0.0099 | 0.0104 | 27.148 | 874.87 | 1.000x | 1.816x | 0.000278 |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | gptq_marlin | 4 | 128 | 0.0108 | 0.0111 | 49.932 | 804.57 | 1.000x | 1.673x | 0.000327 |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | gptq_machete | 4 | 128 | 0.0179 | 0.0188 | 1.874 | 483.63 | 0.549x | 1.000x | 0.000272 |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | gptq_machete | 4 | 128 | 0.0179 | 0.0187 | 3.745 | 483.20 | 0.564x | 1.000x | 0.000299 |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | gptq_machete | 4 | 128 | 0.0179 | 0.0187 | 7.510 | 484.50 | 0.568x | 1.000x | 0.000274 |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | gptq_machete | 4 | 128 | 0.0180 | 0.0188 | 14.953 | 482.34 | 0.551x | 1.000x | 0.000278 |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | gptq_machete | 4 | 128 | 0.0180 | 0.0188 | 29.853 | 481.48 | 0.598x | 1.000x | 0.000327 |
| mlp_down | down_proj | 1 | 8192 | 2048 | qvq_lr | 2 | P32 | 0.0391 | 0.0398 | 0.858 | 108.98 | 0.423x | 0.567x | 4.2e-05 |
| mlp_down | down_proj | 2 | 8192 | 2048 | qvq_lr | 2 | P32 | 0.0478 | 0.0486 | 1.403 | 89.07 | 0.381x | 0.463x | 9.92e-05 |
| mlp_down | down_proj | 4 | 8192 | 2048 | qvq_lr | 2 | P32 | 0.0481 | 0.0488 | 2.792 | 88.63 | 0.380x | 0.463x | 8.39e-05 |
| mlp_down | down_proj | 8 | 8192 | 2048 | qvq_lr | 2 | P32 | 0.0482 | 0.0490 | 5.570 | 88.39 | 0.347x | 0.461x | 9.16e-05 |
| mlp_down | down_proj | 16 | 8192 | 2048 | qvq_lr | 2 | P32 | 0.0485 | 0.0491 | 11.078 | 87.90 | 0.372x | 0.457x | 0.000101 |
| mlp_down | down_proj | 1 | 8192 | 2048 | qvq_lr | 2.5 | P32 | 0.0408 | 0.0414 | 0.822 | 130.01 | 0.405x | 0.542x | 4.2e-05 |
| mlp_down | down_proj | 2 | 8192 | 2048 | qvq_lr | 2.5 | P32 | 0.0500 | 0.0508 | 1.342 | 106.17 | 0.365x | 0.443x | 9.92e-05 |
| mlp_down | down_proj | 4 | 8192 | 2048 | qvq_lr | 2.5 | P32 | 0.0500 | 0.0508 | 2.683 | 106.13 | 0.365x | 0.445x | 9.92e-05 |
| mlp_down | down_proj | 8 | 8192 | 2048 | qvq_lr | 2.5 | P32 | 0.0501 | 0.0508 | 5.358 | 105.96 | 0.334x | 0.443x | 0.000113 |
| mlp_down | down_proj | 16 | 8192 | 2048 | qvq_lr | 2.5 | P32 | 0.0508 | 0.0515 | 10.578 | 104.60 | 0.356x | 0.436x | 0.000103 |
| mlp_down | down_proj | 1 | 8192 | 2048 | gptq_marlin | 4 | 128 | 0.0165 | 0.0168 | 2.028 | 522.89 | 1.000x | 1.338x | 0.000346 |
| mlp_down | down_proj | 2 | 8192 | 2048 | gptq_marlin | 4 | 128 | 0.0182 | 0.0186 | 3.679 | 474.27 | 1.000x | 1.214x | 0.000545 |
| mlp_down | down_proj | 4 | 8192 | 2048 | gptq_marlin | 4 | 128 | 0.0183 | 0.0187 | 7.346 | 473.44 | 1.000x | 1.217x | 0.000622 |
| mlp_down | down_proj | 8 | 8192 | 2048 | gptq_marlin | 4 | 128 | 0.0167 | 0.0170 | 16.039 | 516.89 | 1.000x | 1.327x | 0.000546 |
| mlp_down | down_proj | 16 | 8192 | 2048 | gptq_marlin | 4 | 128 | 0.0180 | 0.0185 | 29.747 | 479.32 | 1.000x | 1.227x | 0.000539 |
| mlp_down | down_proj | 1 | 8192 | 2048 | gptq_machete | 4 | 128 | 0.0221 | 0.0230 | 1.515 | 392.14 | 0.747x | 1.000x | 0.000346 |
| mlp_down | down_proj | 2 | 8192 | 2048 | gptq_machete | 4 | 128 | 0.0221 | 0.0231 | 3.031 | 392.14 | 0.824x | 1.000x | 0.000545 |
| mlp_down | down_proj | 4 | 8192 | 2048 | gptq_machete | 4 | 128 | 0.0222 | 0.0234 | 6.035 | 390.45 | 0.822x | 1.000x | 0.000622 |
| mlp_down | down_proj | 8 | 8192 | 2048 | gptq_machete | 4 | 128 | 0.0222 | 0.0231 | 12.087 | 391.01 | 0.754x | 1.000x | 0.000546 |
| mlp_down | down_proj | 16 | 8192 | 2048 | gptq_machete | 4 | 128 | 0.0221 | 0.0231 | 24.245 | 392.14 | 0.815x | 1.000x | 0.000539 |

The four shape rows preserve all seven Llama 3.2 1B projection roles: `q_proj/o_proj` (2048×2048), `k_proj/v_proj` (2048×512), `gate_proj/up_proj` (2048×8192), and `down_proj` (8192×2048).
