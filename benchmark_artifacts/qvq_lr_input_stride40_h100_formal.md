# QVQ LR versus W4 GPTQ kernels — Llama 3.2 1B shapes

This is a kernel-throughput comparison on the H100 host from PR #62. W4 GPTQ is a figurative performance baseline, not a quality-equivalent arm.

## Measurement contract

- Commit: `3df1d2dfe4b452f3d450963002dcbb77c9b44888`; benchmark SHA256: `ae13dac8f763d972716bba127790a3bffea2c83f7d2e91d7479728527011b6ca`
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
| attn_qo | q_proj/o_proj | 1 | 2048 | 2048 | qvq_lr | 2 | P32 | 0.0145 | 0.0152 | 0.578 | 73.38 | 1.001x | 1.062x | 7.63e-06 |
| attn_qo | q_proj/o_proj | 2 | 2048 | 2048 | qvq_lr | 2 | P32 | 0.0154 | 0.0160 | 1.092 | 69.33 | 1.055x | 1.015x | 1.34e-05 |
| attn_qo | q_proj/o_proj | 4 | 2048 | 2048 | qvq_lr | 2 | P32 | 0.0153 | 0.0161 | 2.198 | 69.77 | 1.069x | 1.019x | 1.43e-05 |
| attn_qo | q_proj/o_proj | 8 | 2048 | 2048 | qvq_lr | 2 | P32 | 0.0154 | 0.0159 | 4.355 | 69.12 | 0.964x | 1.012x | 1.34e-05 |
| attn_qo | q_proj/o_proj | 16 | 2048 | 2048 | qvq_lr | 2 | P32 | 0.0155 | 0.0160 | 8.648 | 68.62 | 1.012x | 1.000x | 1.34e-05 |
| attn_qo | q_proj/o_proj | 1 | 2048 | 2048 | qvq_lr | 2.5 | P32 | 0.0147 | 0.0151 | 0.572 | 90.55 | 0.991x | 1.051x | 7.63e-06 |
| attn_qo | q_proj/o_proj | 2 | 2048 | 2048 | qvq_lr | 2.5 | P32 | 0.0152 | 0.0161 | 1.103 | 87.22 | 1.065x | 1.024x | 1.43e-05 |
| attn_qo | q_proj/o_proj | 4 | 2048 | 2048 | qvq_lr | 2.5 | P32 | 0.0154 | 0.0160 | 2.180 | 86.22 | 1.060x | 1.010x | 1.14e-05 |
| attn_qo | q_proj/o_proj | 8 | 2048 | 2048 | qvq_lr | 2.5 | P32 | 0.0153 | 0.0160 | 4.387 | 86.76 | 0.971x | 1.020x | 1.34e-05 |
| attn_qo | q_proj/o_proj | 16 | 2048 | 2048 | qvq_lr | 2.5 | P32 | 0.0156 | 0.0162 | 8.595 | 84.98 | 1.006x | 0.994x | 1.34e-05 |
| attn_qo | q_proj/o_proj | 1 | 2048 | 2048 | gptq_marlin | 4 | 128 | 0.0145 | 0.0150 | 0.577 | 148.86 | 1.000x | 1.061x | 0.000264 |
| attn_qo | q_proj/o_proj | 2 | 2048 | 2048 | gptq_marlin | 4 | 128 | 0.0162 | 0.0165 | 1.035 | 133.43 | 1.000x | 0.962x | 0.000229 |
| attn_qo | q_proj/o_proj | 4 | 2048 | 2048 | gptq_marlin | 4 | 128 | 0.0163 | 0.0167 | 2.056 | 132.52 | 1.000x | 0.953x | 0.000197 |
| attn_qo | q_proj/o_proj | 8 | 2048 | 2048 | gptq_marlin | 4 | 128 | 0.0148 | 0.0151 | 4.520 | 145.66 | 1.000x | 1.051x | 0.000242 |
| attn_qo | q_proj/o_proj | 16 | 2048 | 2048 | gptq_marlin | 4 | 128 | 0.0157 | 0.0160 | 8.542 | 137.65 | 1.000x | 0.988x | 0.00028 |
| attn_qo | q_proj/o_proj | 1 | 2048 | 2048 | gptq_machete | 4 | 128 | 0.0154 | 0.0163 | 0.544 | 140.89 | 0.943x | 1.000x | 0.000264 |
| attn_qo | q_proj/o_proj | 2 | 2048 | 2048 | gptq_machete | 4 | 128 | 0.0156 | 0.0165 | 1.077 | 139.30 | 1.040x | 1.000x | 0.000229 |
| attn_qo | q_proj/o_proj | 4 | 2048 | 2048 | gptq_machete | 4 | 128 | 0.0156 | 0.0165 | 2.158 | 139.59 | 1.049x | 1.000x | 0.000197 |
| attn_qo | q_proj/o_proj | 8 | 2048 | 2048 | gptq_machete | 4 | 128 | 0.0156 | 0.0165 | 4.302 | 139.16 | 0.952x | 1.000x | 0.000242 |
| attn_qo | q_proj/o_proj | 16 | 2048 | 2048 | gptq_machete | 4 | 128 | 0.0155 | 0.0164 | 8.648 | 139.88 | 1.012x | 1.000x | 0.00028 |
| attn_kv | k_proj/v_proj | 1 | 2048 | 512 | qvq_lr | 2 | P32 | 0.0115 | 0.0122 | 0.183 | 23.18 | 1.923x | 1.287x | 5.72e-06 |
| attn_kv | k_proj/v_proj | 2 | 2048 | 512 | qvq_lr | 2 | P32 | 0.0115 | 0.0122 | 0.365 | 23.14 | 2.225x | 1.270x | 7.63e-06 |
| attn_kv | k_proj/v_proj | 4 | 2048 | 512 | qvq_lr | 2 | P32 | 0.0116 | 0.0122 | 0.724 | 22.98 | 2.240x | 1.265x | 6.68e-06 |
| attn_kv | k_proj/v_proj | 8 | 2048 | 512 | qvq_lr | 2 | P32 | 0.0116 | 0.0122 | 1.452 | 23.05 | 1.975x | 1.277x | 7.63e-06 |
| attn_kv | k_proj/v_proj | 16 | 2048 | 512 | qvq_lr | 2 | P32 | 0.0116 | 0.0122 | 2.905 | 23.05 | 2.089x | 1.273x | 7.63e-06 |
| attn_kv | k_proj/v_proj | 1 | 2048 | 512 | qvq_lr | 2.5 | P32 | 0.0116 | 0.0122 | 0.181 | 28.60 | 1.905x | 1.274x | 5.72e-06 |
| attn_kv | k_proj/v_proj | 2 | 2048 | 512 | qvq_lr | 2.5 | P32 | 0.0116 | 0.0122 | 0.363 | 28.68 | 2.213x | 1.263x | 8.11e-06 |
| attn_kv | k_proj/v_proj | 4 | 2048 | 512 | qvq_lr | 2.5 | P32 | 0.0116 | 0.0122 | 0.724 | 28.64 | 2.240x | 1.265x | 6.68e-06 |
| attn_kv | k_proj/v_proj | 8 | 2048 | 512 | qvq_lr | 2.5 | P32 | 0.0112 | 0.0120 | 1.502 | 29.71 | 2.043x | 1.321x | 7.63e-06 |
| attn_kv | k_proj/v_proj | 16 | 2048 | 512 | qvq_lr | 2.5 | P32 | 0.0113 | 0.0118 | 2.979 | 29.45 | 2.142x | 1.305x | 9.54e-06 |
| attn_kv | k_proj/v_proj | 1 | 2048 | 512 | gptq_marlin | 4 | 128 | 0.0221 | 0.0224 | 0.095 | 24.47 | 1.000x | 0.669x | 0.000167 |
| attn_kv | k_proj/v_proj | 2 | 2048 | 512 | gptq_marlin | 4 | 128 | 0.0256 | 0.0261 | 0.164 | 21.12 | 1.000x | 0.571x | 0.000155 |
| attn_kv | k_proj/v_proj | 4 | 2048 | 512 | gptq_marlin | 4 | 128 | 0.0260 | 0.0262 | 0.323 | 20.83 | 1.000x | 0.565x | 0.000167 |
| attn_kv | k_proj/v_proj | 8 | 2048 | 512 | gptq_marlin | 4 | 128 | 0.0228 | 0.0234 | 0.735 | 23.70 | 1.000x | 0.647x | 0.000206 |
| attn_kv | k_proj/v_proj | 16 | 2048 | 512 | gptq_marlin | 4 | 128 | 0.0241 | 0.0245 | 1.391 | 22.41 | 1.000x | 0.609x | 0.000194 |
| attn_kv | k_proj/v_proj | 1 | 2048 | 512 | gptq_machete | 4 | 128 | 0.0148 | 0.0156 | 0.142 | 37.13 | 1.495x | 1.000x | 0.000167 |
| attn_kv | k_proj/v_proj | 2 | 2048 | 512 | gptq_machete | 4 | 128 | 0.0146 | 0.0150 | 0.287 | 37.57 | 1.752x | 1.000x | 0.000155 |
| attn_kv | k_proj/v_proj | 4 | 2048 | 512 | gptq_machete | 4 | 128 | 0.0147 | 0.0152 | 0.572 | 37.45 | 1.771x | 1.000x | 0.000167 |
| attn_kv | k_proj/v_proj | 8 | 2048 | 512 | gptq_machete | 4 | 128 | 0.0148 | 0.0154 | 1.137 | 37.21 | 1.547x | 1.000x | 0.000206 |
| attn_kv | k_proj/v_proj | 16 | 2048 | 512 | gptq_machete | 4 | 128 | 0.0147 | 0.0154 | 2.282 | 37.33 | 1.641x | 1.000x | 0.000194 |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | qvq_lr | 2 | P32 | 0.0349 | 0.0354 | 0.962 | 122.13 | 0.282x | 0.513x | 5.72e-05 |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | qvq_lr | 2 | P32 | 0.0349 | 0.0352 | 1.921 | 121.96 | 0.292x | 0.517x | 5.15e-05 |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | qvq_lr | 2 | P32 | 0.0351 | 0.0356 | 3.825 | 121.40 | 0.290x | 0.512x | 5.72e-05 |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | qvq_lr | 2 | P32 | 0.0355 | 0.0360 | 7.561 | 119.98 | 0.280x | 0.507x | 7.63e-05 |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | qvq_lr | 2 | P32 | 0.0359 | 0.0361 | 14.960 | 118.70 | 0.301x | 0.498x | 6.1e-05 |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | qvq_lr | 2.5 | P32 | 0.0343 | 0.0346 | 0.978 | 154.75 | 0.286x | 0.521x | 4.86e-05 |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | qvq_lr | 2.5 | P32 | 0.0345 | 0.0348 | 1.945 | 153.81 | 0.295x | 0.523x | 5.53e-05 |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | qvq_lr | 2.5 | P32 | 0.0347 | 0.0351 | 3.873 | 153.17 | 0.294x | 0.518x | 6.29e-05 |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | qvq_lr | 2.5 | P32 | 0.0350 | 0.0354 | 7.675 | 151.77 | 0.285x | 0.514x | 5.34e-05 |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | qvq_lr | 2.5 | P32 | 0.0353 | 0.0357 | 15.224 | 150.53 | 0.307x | 0.507x | 6.87e-05 |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | gptq_marlin | 4 | 128 | 0.0098 | 0.0102 | 3.416 | 880.57 | 1.000x | 1.821x | 0.000272 |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | gptq_marlin | 4 | 128 | 0.0102 | 0.0105 | 6.584 | 848.78 | 1.000x | 1.772x | 0.000299 |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | gptq_marlin | 4 | 128 | 0.0102 | 0.0105 | 13.190 | 850.11 | 1.000x | 1.764x | 0.000274 |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | gptq_marlin | 4 | 128 | 0.0100 | 0.0102 | 26.973 | 869.25 | 1.000x | 1.807x | 0.000278 |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | gptq_marlin | 4 | 128 | 0.0108 | 0.0112 | 49.637 | 799.81 | 1.000x | 1.652x | 0.000327 |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | gptq_machete | 4 | 128 | 0.0179 | 0.0188 | 1.876 | 484.06 | 0.549x | 1.000x | 0.000272 |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | gptq_machete | 4 | 128 | 0.0181 | 0.0190 | 3.715 | 479.35 | 0.564x | 1.000x | 0.000299 |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | gptq_machete | 4 | 128 | 0.0180 | 0.0189 | 7.476 | 482.34 | 0.567x | 1.000x | 0.000274 |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | gptq_machete | 4 | 128 | 0.0180 | 0.0189 | 14.926 | 481.48 | 0.553x | 1.000x | 0.000278 |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | gptq_machete | 4 | 128 | 0.0179 | 0.0188 | 30.040 | 484.50 | 0.605x | 1.000x | 0.000327 |
| mlp_down | down_proj | 1 | 8192 | 2048 | qvq_lr | 2 | P32 | 0.0313 | 0.0320 | 1.072 | 136.11 | 0.527x | 0.714x | 4.2e-05 |
| mlp_down | down_proj | 2 | 8192 | 2048 | qvq_lr | 2 | P32 | 0.0376 | 0.0384 | 1.783 | 113.20 | 0.484x | 0.589x | 9.92e-05 |
| mlp_down | down_proj | 4 | 8192 | 2048 | qvq_lr | 2 | P32 | 0.0378 | 0.0385 | 3.553 | 112.77 | 0.487x | 0.589x | 8.39e-05 |
| mlp_down | down_proj | 8 | 8192 | 2048 | qvq_lr | 2 | P32 | 0.0379 | 0.0386 | 7.076 | 112.29 | 0.438x | 0.587x | 9.16e-05 |
| mlp_down | down_proj | 16 | 8192 | 2048 | qvq_lr | 2 | P32 | 0.0380 | 0.0387 | 14.122 | 112.05 | 0.478x | 0.585x | 0.000101 |
| mlp_down | down_proj | 1 | 8192 | 2048 | qvq_lr | 2.5 | P32 | 0.0319 | 0.0325 | 1.052 | 166.47 | 0.517x | 0.700x | 4.2e-05 |
| mlp_down | down_proj | 2 | 8192 | 2048 | qvq_lr | 2.5 | P32 | 0.0372 | 0.0380 | 1.806 | 142.82 | 0.490x | 0.597x | 9.92e-05 |
| mlp_down | down_proj | 4 | 8192 | 2048 | qvq_lr | 2.5 | P32 | 0.0371 | 0.0378 | 3.614 | 142.95 | 0.495x | 0.599x | 9.92e-05 |
| mlp_down | down_proj | 8 | 8192 | 2048 | qvq_lr | 2.5 | P32 | 0.0374 | 0.0381 | 7.182 | 142.03 | 0.445x | 0.595x | 0.000113 |
| mlp_down | down_proj | 16 | 8192 | 2048 | qvq_lr | 2.5 | P32 | 0.0378 | 0.0386 | 14.206 | 140.46 | 0.481x | 0.589x | 0.000103 |
| mlp_down | down_proj | 1 | 8192 | 2048 | gptq_marlin | 4 | 128 | 0.0165 | 0.0169 | 2.036 | 524.92 | 1.000x | 1.355x | 0.000346 |
| mlp_down | down_proj | 2 | 8192 | 2048 | gptq_marlin | 4 | 128 | 0.0182 | 0.0186 | 3.686 | 475.11 | 1.000x | 1.218x | 0.000545 |
| mlp_down | down_proj | 4 | 8192 | 2048 | gptq_marlin | 4 | 128 | 0.0184 | 0.0189 | 7.294 | 470.15 | 1.000x | 1.209x | 0.000622 |
| mlp_down | down_proj | 8 | 8192 | 2048 | gptq_marlin | 4 | 128 | 0.0166 | 0.0171 | 16.147 | 520.38 | 1.000x | 1.339x | 0.000546 |
| mlp_down | down_proj | 16 | 8192 | 2048 | gptq_marlin | 4 | 128 | 0.0182 | 0.0185 | 29.537 | 475.94 | 1.000x | 1.224x | 0.000539 |
| mlp_down | down_proj | 1 | 8192 | 2048 | gptq_machete | 4 | 128 | 0.0223 | 0.0231 | 1.502 | 388.77 | 0.738x | 1.000x | 0.000346 |
| mlp_down | down_proj | 2 | 8192 | 2048 | gptq_machete | 4 | 128 | 0.0222 | 0.0231 | 3.026 | 391.57 | 0.821x | 1.000x | 0.000545 |
| mlp_down | down_proj | 4 | 8192 | 2048 | gptq_machete | 4 | 128 | 0.0222 | 0.0230 | 6.035 | 390.45 | 0.827x | 1.000x | 0.000622 |
| mlp_down | down_proj | 8 | 8192 | 2048 | gptq_machete | 4 | 128 | 0.0223 | 0.0231 | 12.061 | 390.17 | 0.747x | 1.000x | 0.000546 |
| mlp_down | down_proj | 16 | 8192 | 2048 | gptq_machete | 4 | 128 | 0.0223 | 0.0232 | 24.123 | 390.17 | 0.817x | 1.000x | 0.000539 |

The four shape rows preserve all seven Llama 3.2 1B projection roles: `q_proj/o_proj` (2048×2048), `k_proj/v_proj` (2048×512), `gate_proj/up_proj` (2048×8192), and `down_proj` (8192×2048).
