# QVQ LR versus W4 GPTQ kernels — Llama 3.2 1B shapes

This is a kernel-throughput comparison on the H100 host from PR #62. W4 GPTQ is a figurative performance baseline, not a quality-equivalent arm.

## Measurement contract

- Commit: `56c635b2f84aeb859a34b71d721cc07cbd60654e`; benchmark SHA256: `ae13dac8f763d972716bba127790a3bffea2c83f7d2e91d7479728527011b6ca`
- GPU: physical `1`, `NVIDIA H100`, PCI `00000000:44:00.0`, UUID `GPU-f5ea03cf-efa4-9807-7de5-b174957a1348`, CC `9.0`, 132 SMs
- Torch/CUDA: `2.15.0.dev20260828+cu130` / `13.0`; input dtype: `float16`
- M values: `[1, 2, 4, 8, 16]`; warmup: `20`; measured launches: `60`
- QVQ: `qvq_v2b2_p32_lr`, rates `[3.5]`, native `P32 is LR packing geometry, not a GPTQ affine scale group.`
- GPTQ: symmetric W4, group `128`, no activation order; Marlin and Machete use the same W4 source payload
- Latency is CUDA-event median/P95 from one CUDA Graph replay containing every measured launch. CPU scheduling and host launch gaps are outside each timed interval. Logical TFLOP/s is `2*M*K*N / median_ms`; payload GB/s is packed payload bytes divided by median latency.
- Every row passed its dense-reference shape/dtype/finite gate. QVQ uses `max_abs <= 2e-3`; GPTQ uses `atol=2e-2, rtol=2e-2`.

`xMarlin` and `xMachete` are baseline median latency divided by the row median, so values above 1.0x are faster.

| Shape | Roles | M | K | N | Kernel | W | Group | Median ms | P95 ms | Logical TFLOP/s | Payload GB/s | xMarlin | xMachete | Max abs |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| attn_qo | q_proj/o_proj | 1 | 2048 | 2048 | qvq_lr | 3.5 | P32 | 0.0151 | 0.0159 | 0.554 | 122.32 | 0.956x | 1.021x | 7.63e-06 |
| attn_qo | q_proj/o_proj | 2 | 2048 | 2048 | qvq_lr | 3.5 | P32 | 0.0158 | 0.0164 | 1.065 | 117.47 | 1.031x | 0.973x | 1.29e-05 |
| attn_qo | q_proj/o_proj | 4 | 2048 | 2048 | qvq_lr | 3.5 | P32 | 0.0157 | 0.0163 | 2.138 | 117.95 | 1.040x | 0.987x | 1.34e-05 |
| attn_qo | q_proj/o_proj | 8 | 2048 | 2048 | qvq_lr | 3.5 | P32 | 0.0159 | 0.0164 | 4.220 | 116.41 | 0.925x | 0.964x | 1.53e-05 |
| attn_qo | q_proj/o_proj | 16 | 2048 | 2048 | qvq_lr | 3.5 | P32 | 0.0162 | 0.0167 | 8.306 | 114.57 | 0.966x | 0.952x | 1.43e-05 |
| attn_qo | q_proj/o_proj | 1 | 2048 | 2048 | gptq_marlin | 4 | 128 | 0.0145 | 0.0149 | 0.580 | 149.52 | 1.000x | 1.069x | 0.000264 |
| attn_qo | q_proj/o_proj | 2 | 2048 | 2048 | gptq_marlin | 4 | 128 | 0.0163 | 0.0166 | 1.032 | 133.04 | 1.000x | 0.943x | 0.000229 |
| attn_qo | q_proj/o_proj | 4 | 2048 | 2048 | gptq_marlin | 4 | 128 | 0.0163 | 0.0167 | 2.056 | 132.52 | 1.000x | 0.949x | 0.000197 |
| attn_qo | q_proj/o_proj | 8 | 2048 | 2048 | gptq_marlin | 4 | 128 | 0.0147 | 0.0150 | 4.564 | 147.08 | 1.000x | 1.042x | 0.000242 |
| attn_qo | q_proj/o_proj | 16 | 2048 | 2048 | gptq_marlin | 4 | 128 | 0.0156 | 0.0160 | 8.595 | 138.49 | 1.000x | 0.986x | 0.00028 |
| attn_qo | q_proj/o_proj | 1 | 2048 | 2048 | gptq_machete | 4 | 128 | 0.0155 | 0.0164 | 0.543 | 140.46 | 0.936x | 1.000x | 0.000264 |
| attn_qo | q_proj/o_proj | 2 | 2048 | 2048 | gptq_machete | 4 | 128 | 0.0153 | 0.0162 | 1.095 | 141.63 | 1.061x | 1.000x | 0.000229 |
| attn_qo | q_proj/o_proj | 4 | 2048 | 2048 | gptq_machete | 4 | 128 | 0.0155 | 0.0164 | 2.166 | 140.17 | 1.054x | 1.000x | 0.000197 |
| attn_qo | q_proj/o_proj | 8 | 2048 | 2048 | gptq_machete | 4 | 128 | 0.0153 | 0.0162 | 4.378 | 141.63 | 0.959x | 1.000x | 0.000242 |
| attn_qo | q_proj/o_proj | 16 | 2048 | 2048 | gptq_machete | 4 | 128 | 0.0154 | 0.0163 | 8.720 | 141.04 | 1.015x | 1.000x | 0.00028 |
| attn_kv | k_proj/v_proj | 1 | 2048 | 512 | qvq_lr | 3.5 | P32 | 0.0224 | 0.0230 | 0.094 | 20.68 | 0.995x | 0.660x | 2.38e-06 |
| attn_kv | k_proj/v_proj | 2 | 2048 | 512 | qvq_lr | 3.5 | P32 | 0.0204 | 0.0211 | 0.205 | 22.67 | 1.270x | 0.746x | 4.29e-06 |
| attn_kv | k_proj/v_proj | 4 | 2048 | 512 | qvq_lr | 3.5 | P32 | 0.0207 | 0.0212 | 0.405 | 22.32 | 1.255x | 0.737x | 3.81e-06 |
| attn_kv | k_proj/v_proj | 8 | 2048 | 512 | qvq_lr | 3.5 | P32 | 0.0215 | 0.0221 | 0.780 | 21.52 | 1.067x | 0.687x | 3.81e-06 |
| attn_kv | k_proj/v_proj | 16 | 2048 | 512 | qvq_lr | 3.5 | P32 | 0.0240 | 0.0247 | 1.397 | 19.27 | 1.009x | 0.632x | 4.29e-06 |
| attn_kv | k_proj/v_proj | 1 | 2048 | 512 | gptq_marlin | 4 | 128 | 0.0223 | 0.0232 | 0.094 | 24.28 | 1.000x | 0.663x | 0.000167 |
| attn_kv | k_proj/v_proj | 2 | 2048 | 512 | gptq_marlin | 4 | 128 | 0.0259 | 0.0265 | 0.162 | 20.86 | 1.000x | 0.588x | 0.000155 |
| attn_kv | k_proj/v_proj | 4 | 2048 | 512 | gptq_marlin | 4 | 128 | 0.0260 | 0.0265 | 0.322 | 20.78 | 1.000x | 0.587x | 0.000167 |
| attn_kv | k_proj/v_proj | 8 | 2048 | 512 | gptq_marlin | 4 | 128 | 0.0229 | 0.0234 | 0.731 | 23.56 | 1.000x | 0.644x | 0.000206 |
| attn_kv | k_proj/v_proj | 16 | 2048 | 512 | gptq_marlin | 4 | 128 | 0.0242 | 0.0247 | 1.385 | 22.32 | 1.000x | 0.627x | 0.000194 |
| attn_kv | k_proj/v_proj | 1 | 2048 | 512 | gptq_machete | 4 | 128 | 0.0148 | 0.0157 | 0.142 | 37.17 | 1.508x | 1.000x | 0.000167 |
| attn_kv | k_proj/v_proj | 2 | 2048 | 512 | gptq_machete | 4 | 128 | 0.0152 | 0.0160 | 0.275 | 36.03 | 1.702x | 1.000x | 0.000155 |
| attn_kv | k_proj/v_proj | 4 | 2048 | 512 | gptq_machete | 4 | 128 | 0.0153 | 0.0160 | 0.549 | 35.92 | 1.703x | 1.000x | 0.000167 |
| attn_kv | k_proj/v_proj | 8 | 2048 | 512 | gptq_machete | 4 | 128 | 0.0148 | 0.0156 | 1.136 | 37.17 | 1.554x | 1.000x | 0.000206 |
| attn_kv | k_proj/v_proj | 16 | 2048 | 512 | gptq_machete | 4 | 128 | 0.0152 | 0.0160 | 2.210 | 36.15 | 1.595x | 1.000x | 0.000194 |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | qvq_lr | 3.5 | P32 | 0.0377 | 0.0381 | 0.889 | 196.29 | 0.261x | 0.480x | 4.86e-05 |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | qvq_lr | 3.5 | P32 | 0.0379 | 0.0382 | 1.773 | 195.62 | 0.268x | 0.475x | 6.68e-05 |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | qvq_lr | 3.5 | P32 | 0.0380 | 0.0383 | 3.532 | 194.88 | 0.268x | 0.472x | 6.48e-05 |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | qvq_lr | 3.5 | P32 | 0.0382 | 0.0387 | 7.029 | 193.90 | 0.260x | 0.470x | 6.29e-05 |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | qvq_lr | 3.5 | P32 | 0.0384 | 0.0389 | 13.969 | 192.69 | 0.279x | 0.467x | 5.91e-05 |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | gptq_marlin | 4 | 128 | 0.0099 | 0.0102 | 3.404 | 877.71 | 1.000x | 1.838x | 0.000272 |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | gptq_marlin | 4 | 128 | 0.0101 | 0.0105 | 6.616 | 852.79 | 1.000x | 1.773x | 0.000299 |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | gptq_marlin | 4 | 128 | 0.0102 | 0.0105 | 13.190 | 850.11 | 1.000x | 1.761x | 0.000274 |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | gptq_marlin | 4 | 128 | 0.0099 | 0.0102 | 27.060 | 872.05 | 1.000x | 1.810x | 0.000278 |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | gptq_marlin | 4 | 128 | 0.0107 | 0.0111 | 50.081 | 806.97 | 1.000x | 1.673x | 0.000327 |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | gptq_machete | 4 | 128 | 0.0181 | 0.0190 | 1.853 | 478.08 | 0.544x | 1.000x | 0.000272 |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | gptq_machete | 4 | 128 | 0.0180 | 0.0189 | 3.732 | 481.48 | 0.564x | 1.000x | 0.000299 |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | gptq_machete | 4 | 128 | 0.0179 | 0.0188 | 7.490 | 483.20 | 0.568x | 1.000x | 0.000274 |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | gptq_machete | 4 | 128 | 0.0180 | 0.0187 | 14.953 | 482.34 | 0.553x | 1.000x | 0.000278 |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | gptq_machete | 4 | 128 | 0.0179 | 0.0188 | 29.933 | 482.77 | 0.598x | 1.000x | 0.000327 |
| mlp_down | down_proj | 1 | 8192 | 2048 | qvq_lr | 3.5 | P32 | 0.0344 | 0.0349 | 0.975 | 215.18 | 0.483x | 0.653x | 5.15e-05 |
| mlp_down | down_proj | 2 | 8192 | 2048 | qvq_lr | 3.5 | P32 | 0.0402 | 0.0407 | 1.670 | 184.25 | 0.453x | 0.561x | 8.01e-05 |
| mlp_down | down_proj | 4 | 8192 | 2048 | qvq_lr | 3.5 | P32 | 0.0402 | 0.0408 | 3.339 | 184.25 | 0.456x | 0.555x | 9.54e-05 |
| mlp_down | down_proj | 8 | 8192 | 2048 | qvq_lr | 3.5 | P32 | 0.0403 | 0.0409 | 6.658 | 183.67 | 0.417x | 0.560x | 9.92e-05 |
| mlp_down | down_proj | 16 | 8192 | 2048 | qvq_lr | 3.5 | P32 | 0.0406 | 0.0414 | 13.210 | 182.22 | 0.447x | 0.547x | 0.000103 |
| mlp_down | down_proj | 1 | 8192 | 2048 | gptq_marlin | 4 | 128 | 0.0166 | 0.0171 | 2.020 | 520.88 | 1.000x | 1.354x | 0.000346 |
| mlp_down | down_proj | 2 | 8192 | 2048 | gptq_marlin | 4 | 128 | 0.0182 | 0.0185 | 3.686 | 475.11 | 1.000x | 1.238x | 0.000545 |
| mlp_down | down_proj | 4 | 8192 | 2048 | gptq_marlin | 4 | 128 | 0.0183 | 0.0188 | 7.320 | 471.79 | 1.000x | 1.216x | 0.000622 |
| mlp_down | down_proj | 8 | 8192 | 2048 | gptq_marlin | 4 | 128 | 0.0168 | 0.0172 | 15.978 | 514.93 | 1.000x | 1.343x | 0.000546 |
| mlp_down | down_proj | 16 | 8192 | 2048 | gptq_marlin | 4 | 128 | 0.0182 | 0.0185 | 29.537 | 475.94 | 1.000x | 1.224x | 0.000539 |
| mlp_down | down_proj | 1 | 8192 | 2048 | gptq_machete | 4 | 128 | 0.0225 | 0.0234 | 1.493 | 386.28 | 0.739x | 1.000x | 0.000346 |
| mlp_down | down_proj | 2 | 8192 | 2048 | gptq_machete | 4 | 128 | 0.0225 | 0.0233 | 2.977 | 385.18 | 0.808x | 1.000x | 0.000545 |
| mlp_down | down_proj | 4 | 8192 | 2048 | gptq_machete | 4 | 128 | 0.0223 | 0.0232 | 6.022 | 389.61 | 0.823x | 1.000x | 0.000622 |
| mlp_down | down_proj | 8 | 8192 | 2048 | gptq_machete | 4 | 128 | 0.0226 | 0.0234 | 11.899 | 384.91 | 0.745x | 1.000x | 0.000546 |
| mlp_down | down_proj | 16 | 8192 | 2048 | gptq_machete | 4 | 128 | 0.0222 | 0.0233 | 24.140 | 390.45 | 0.817x | 1.000x | 0.000539 |

The four shape rows preserve all seven Llama 3.2 1B projection roles: `q_proj/o_proj` (2048×2048), `k_proj/v_proj` (2048×512), `gate_proj/up_proj` (2048×8192), and `down_proj` (8192×2048).
