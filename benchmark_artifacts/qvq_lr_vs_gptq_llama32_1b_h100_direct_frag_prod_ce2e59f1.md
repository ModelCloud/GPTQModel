# QVQ LR versus W4 GPTQ kernels — Llama 3.2 1B shapes

This is a kernel-throughput comparison on the H100 host from PR #62. W4 GPTQ is a figurative performance baseline, not a quality-equivalent arm.

## Measurement contract

- Commit: `ce2e59f1e79928e50064f2b2f9d62554af49df53`; benchmark SHA256: `ae13dac8f763d972716bba127790a3bffea2c83f7d2e91d7479728527011b6ca`
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
| attn_qo | q_proj/o_proj | 1 | 2048 | 2048 | qvq_lr | 2 | P32 | 0.0154 | 0.0159 | 0.544 | 69.12 | 0.947x | 0.993x | 7.63e-06 |
| attn_qo | q_proj/o_proj | 2 | 2048 | 2048 | qvq_lr | 2 | P32 | 0.0161 | 0.0167 | 1.044 | 66.29 | 1.012x | 0.940x | 1.34e-05 |
| attn_qo | q_proj/o_proj | 4 | 2048 | 2048 | qvq_lr | 2 | P32 | 0.0160 | 0.0167 | 2.093 | 66.43 | 1.020x | 0.957x | 1.43e-05 |
| attn_qo | q_proj/o_proj | 8 | 2048 | 2048 | qvq_lr | 2 | P32 | 0.0161 | 0.0168 | 4.169 | 66.16 | 0.918x | 0.948x | 1.34e-05 |
| attn_qo | q_proj/o_proj | 16 | 2048 | 2048 | qvq_lr | 2 | P32 | 0.0165 | 0.0169 | 8.144 | 64.62 | 0.949x | 0.935x | 1.34e-05 |
| attn_qo | q_proj/o_proj | 1 | 2048 | 2048 | qvq_lr | 2.5 | P32 | 0.0158 | 0.0163 | 0.532 | 84.12 | 0.925x | 0.970x | 7.63e-06 |
| attn_qo | q_proj/o_proj | 2 | 2048 | 2048 | qvq_lr | 2.5 | P32 | 0.0172 | 0.0178 | 0.974 | 77.01 | 0.943x | 0.877x | 1.43e-05 |
| attn_qo | q_proj/o_proj | 4 | 2048 | 2048 | qvq_lr | 2.5 | P32 | 0.0172 | 0.0179 | 1.951 | 77.16 | 0.951x | 0.892x | 1.14e-05 |
| attn_qo | q_proj/o_proj | 8 | 2048 | 2048 | qvq_lr | 2.5 | P32 | 0.0174 | 0.0180 | 3.862 | 76.38 | 0.851x | 0.878x | 1.34e-05 |
| attn_qo | q_proj/o_proj | 16 | 2048 | 2048 | qvq_lr | 2.5 | P32 | 0.0175 | 0.0181 | 7.689 | 76.03 | 0.896x | 0.883x | 1.34e-05 |
| attn_qo | q_proj/o_proj | 1 | 2048 | 2048 | gptq_marlin | 4 | 128 | 0.0146 | 0.0150 | 0.575 | 148.21 | 1.000x | 1.048x | 0.000264 |
| attn_qo | q_proj/o_proj | 2 | 2048 | 2048 | gptq_marlin | 4 | 128 | 0.0163 | 0.0168 | 1.032 | 133.04 | 1.000x | 0.929x | 0.000229 |
| attn_qo | q_proj/o_proj | 4 | 2048 | 2048 | gptq_marlin | 4 | 128 | 0.0164 | 0.0168 | 2.052 | 132.26 | 1.000x | 0.938x | 0.000197 |
| attn_qo | q_proj/o_proj | 8 | 2048 | 2048 | gptq_marlin | 4 | 128 | 0.0148 | 0.0152 | 4.539 | 146.29 | 1.000x | 1.032x | 0.000242 |
| attn_qo | q_proj/o_proj | 16 | 2048 | 2048 | gptq_marlin | 4 | 128 | 0.0156 | 0.0161 | 8.586 | 138.35 | 1.000x | 0.986x | 0.00028 |
| attn_qo | q_proj/o_proj | 1 | 2048 | 2048 | gptq_machete | 4 | 128 | 0.0153 | 0.0163 | 0.548 | 141.92 | 0.954x | 1.000x | 0.000264 |
| attn_qo | q_proj/o_proj | 2 | 2048 | 2048 | gptq_machete | 4 | 128 | 0.0151 | 0.0159 | 1.111 | 143.73 | 1.076x | 1.000x | 0.000229 |
| attn_qo | q_proj/o_proj | 4 | 2048 | 2048 | gptq_machete | 4 | 128 | 0.0153 | 0.0163 | 2.187 | 141.48 | 1.066x | 1.000x | 0.000197 |
| attn_qo | q_proj/o_proj | 8 | 2048 | 2048 | gptq_machete | 4 | 128 | 0.0153 | 0.0162 | 4.397 | 142.22 | 0.969x | 1.000x | 0.000242 |
| attn_qo | q_proj/o_proj | 16 | 2048 | 2048 | gptq_machete | 4 | 128 | 0.0154 | 0.0163 | 8.711 | 140.89 | 1.015x | 1.000x | 0.00028 |
| attn_kv | k_proj/v_proj | 1 | 2048 | 512 | qvq_lr | 2 | P32 | 0.0115 | 0.0121 | 0.182 | 23.11 | 1.919x | 1.274x | 5.72e-06 |
| attn_kv | k_proj/v_proj | 2 | 2048 | 512 | qvq_lr | 2 | P32 | 0.0116 | 0.0121 | 0.362 | 22.98 | 2.215x | 1.256x | 7.63e-06 |
| attn_kv | k_proj/v_proj | 4 | 2048 | 512 | qvq_lr | 2 | P32 | 0.0116 | 0.0121 | 0.722 | 22.92 | 2.245x | 1.259x | 6.68e-06 |
| attn_kv | k_proj/v_proj | 8 | 2048 | 512 | qvq_lr | 2 | P32 | 0.0116 | 0.0122 | 1.446 | 22.95 | 1.981x | 1.274x | 7.63e-06 |
| attn_kv | k_proj/v_proj | 16 | 2048 | 512 | qvq_lr | 2 | P32 | 0.0118 | 0.0123 | 2.853 | 22.64 | 2.053x | 1.245x | 7.63e-06 |
| attn_kv | k_proj/v_proj | 1 | 2048 | 512 | qvq_lr | 2.5 | P32 | 0.0124 | 0.0130 | 0.168 | 26.65 | 1.776x | 1.179x | 5.72e-06 |
| attn_kv | k_proj/v_proj | 2 | 2048 | 512 | qvq_lr | 2.5 | P32 | 0.0123 | 0.0129 | 0.342 | 27.04 | 2.091x | 1.185x | 8.11e-06 |
| attn_kv | k_proj/v_proj | 4 | 2048 | 512 | qvq_lr | 2.5 | P32 | 0.0124 | 0.0131 | 0.675 | 26.69 | 2.098x | 1.176x | 6.68e-06 |
| attn_kv | k_proj/v_proj | 8 | 2048 | 512 | qvq_lr | 2.5 | P32 | 0.0120 | 0.0129 | 1.394 | 27.57 | 1.910x | 1.229x | 7.63e-06 |
| attn_kv | k_proj/v_proj | 16 | 2048 | 512 | qvq_lr | 2.5 | P32 | 0.0121 | 0.0126 | 2.781 | 27.50 | 2.001x | 1.214x | 9.54e-06 |
| attn_kv | k_proj/v_proj | 1 | 2048 | 512 | gptq_marlin | 4 | 128 | 0.0221 | 0.0228 | 0.095 | 24.45 | 1.000x | 0.664x | 0.000167 |
| attn_kv | k_proj/v_proj | 2 | 2048 | 512 | gptq_marlin | 4 | 128 | 0.0257 | 0.0261 | 0.163 | 21.07 | 1.000x | 0.567x | 0.000155 |
| attn_kv | k_proj/v_proj | 4 | 2048 | 512 | gptq_marlin | 4 | 128 | 0.0261 | 0.0266 | 0.322 | 20.73 | 1.000x | 0.561x | 0.000167 |
| attn_kv | k_proj/v_proj | 8 | 2048 | 512 | gptq_marlin | 4 | 128 | 0.0230 | 0.0233 | 0.730 | 23.53 | 1.000x | 0.643x | 0.000206 |
| attn_kv | k_proj/v_proj | 16 | 2048 | 512 | gptq_marlin | 4 | 128 | 0.0241 | 0.0245 | 1.390 | 22.39 | 1.000x | 0.606x | 0.000194 |
| attn_kv | k_proj/v_proj | 1 | 2048 | 512 | gptq_machete | 4 | 128 | 0.0147 | 0.0154 | 0.143 | 37.41 | 1.507x | 1.000x | 0.000167 |
| attn_kv | k_proj/v_proj | 2 | 2048 | 512 | gptq_machete | 4 | 128 | 0.0145 | 0.0154 | 0.288 | 37.74 | 1.765x | 1.000x | 0.000155 |
| attn_kv | k_proj/v_proj | 4 | 2048 | 512 | gptq_machete | 4 | 128 | 0.0146 | 0.0154 | 0.574 | 37.53 | 1.783x | 1.000x | 0.000167 |
| attn_kv | k_proj/v_proj | 8 | 2048 | 512 | gptq_machete | 4 | 128 | 0.0148 | 0.0158 | 1.135 | 37.13 | 1.554x | 1.000x | 0.000206 |
| attn_kv | k_proj/v_proj | 16 | 2048 | 512 | gptq_machete | 4 | 128 | 0.0146 | 0.0155 | 2.292 | 37.49 | 1.649x | 1.000x | 0.000194 |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | qvq_lr | 2 | P32 | 0.0372 | 0.0377 | 0.902 | 114.46 | 0.269x | 0.482x | 5.72e-05 |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | qvq_lr | 2 | P32 | 0.0372 | 0.0377 | 1.803 | 114.46 | 0.278x | 0.482x | 5.15e-05 |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | qvq_lr | 2 | P32 | 0.0374 | 0.0378 | 3.591 | 113.97 | 0.275x | 0.484x | 5.72e-05 |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | qvq_lr | 2 | P32 | 0.0377 | 0.0381 | 7.127 | 113.10 | 0.266x | 0.476x | 7.63e-05 |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | qvq_lr | 2 | P32 | 0.0381 | 0.0385 | 14.104 | 111.91 | 0.285x | 0.474x | 6.1e-05 |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | qvq_lr | 2.5 | P32 | 0.0413 | 0.0417 | 0.812 | 128.40 | 0.242x | 0.434x | 4.86e-05 |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | qvq_lr | 2.5 | P32 | 0.0415 | 0.0420 | 1.617 | 127.90 | 0.249x | 0.433x | 5.53e-05 |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | qvq_lr | 2.5 | P32 | 0.0418 | 0.0422 | 3.208 | 126.87 | 0.246x | 0.433x | 6.29e-05 |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | qvq_lr | 2.5 | P32 | 0.0420 | 0.0425 | 6.384 | 126.25 | 0.238x | 0.427x | 5.34e-05 |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | qvq_lr | 2.5 | P32 | 0.0423 | 0.0427 | 12.691 | 125.48 | 0.256x | 0.427x | 6.87e-05 |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | gptq_marlin | 4 | 128 | 0.0100 | 0.0104 | 3.350 | 863.69 | 1.000x | 1.792x | 0.000272 |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | gptq_marlin | 4 | 128 | 0.0103 | 0.0107 | 6.493 | 836.95 | 1.000x | 1.737x | 0.000299 |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | gptq_marlin | 4 | 128 | 0.0103 | 0.0107 | 13.046 | 840.86 | 1.000x | 1.759x | 0.000274 |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | gptq_marlin | 4 | 128 | 0.0100 | 0.0104 | 26.844 | 865.08 | 1.000x | 1.794x | 0.000278 |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | gptq_marlin | 4 | 128 | 0.0108 | 0.0112 | 49.490 | 797.45 | 1.000x | 1.664x | 0.000327 |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | gptq_machete | 4 | 128 | 0.0180 | 0.0189 | 1.869 | 482.34 | 0.558x | 1.000x | 0.000272 |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | gptq_machete | 4 | 128 | 0.0180 | 0.0190 | 3.738 | 482.34 | 0.576x | 1.000x | 0.000299 |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | gptq_machete | 4 | 128 | 0.0181 | 0.0192 | 7.417 | 478.50 | 0.569x | 1.000x | 0.000274 |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | gptq_machete | 4 | 128 | 0.0179 | 0.0191 | 14.966 | 482.77 | 0.558x | 1.000x | 0.000278 |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | gptq_machete | 4 | 128 | 0.0180 | 0.0191 | 29.747 | 479.77 | 0.601x | 1.000x | 0.000327 |
| mlp_down | down_proj | 1 | 8192 | 2048 | qvq_lr | 2 | P32 | 0.0347 | 0.0353 | 0.966 | 122.63 | 0.472x | 0.636x | 4.2e-05 |
| mlp_down | down_proj | 2 | 8192 | 2048 | qvq_lr | 2 | P32 | 0.0401 | 0.0408 | 1.672 | 106.16 | 0.452x | 0.549x | 9.92e-05 |
| mlp_down | down_proj | 4 | 8192 | 2048 | qvq_lr | 2 | P32 | 0.0401 | 0.0407 | 3.347 | 106.24 | 0.453x | 0.555x | 8.39e-05 |
| mlp_down | down_proj | 8 | 8192 | 2048 | qvq_lr | 2 | P32 | 0.0404 | 0.0411 | 6.650 | 105.53 | 0.411x | 0.552x | 9.16e-05 |
| mlp_down | down_proj | 16 | 8192 | 2048 | qvq_lr | 2 | P32 | 0.0405 | 0.0412 | 13.242 | 105.07 | 0.444x | 0.545x | 0.000101 |
| mlp_down | down_proj | 1 | 8192 | 2048 | qvq_lr | 2.5 | P32 | 0.0348 | 0.0356 | 0.963 | 152.33 | 0.470x | 0.634x | 4.2e-05 |
| mlp_down | down_proj | 2 | 8192 | 2048 | qvq_lr | 2.5 | P32 | 0.0442 | 0.0450 | 1.517 | 120.03 | 0.410x | 0.498x | 9.92e-05 |
| mlp_down | down_proj | 4 | 8192 | 2048 | qvq_lr | 2.5 | P32 | 0.0443 | 0.0452 | 3.031 | 119.86 | 0.410x | 0.502x | 9.92e-05 |
| mlp_down | down_proj | 8 | 8192 | 2048 | qvq_lr | 2.5 | P32 | 0.0446 | 0.0452 | 6.013 | 118.92 | 0.372x | 0.499x | 0.000113 |
| mlp_down | down_proj | 16 | 8192 | 2048 | qvq_lr | 2.5 | P32 | 0.0447 | 0.0454 | 12.009 | 118.75 | 0.402x | 0.495x | 0.000103 |
| mlp_down | down_proj | 1 | 8192 | 2048 | gptq_marlin | 4 | 128 | 0.0164 | 0.0168 | 2.048 | 528.00 | 1.000x | 1.348x | 0.000346 |
| mlp_down | down_proj | 2 | 8192 | 2048 | gptq_marlin | 4 | 128 | 0.0181 | 0.0187 | 3.699 | 476.78 | 1.000x | 1.213x | 0.000545 |
| mlp_down | down_proj | 4 | 8192 | 2048 | gptq_marlin | 4 | 128 | 0.0182 | 0.0185 | 7.384 | 475.94 | 1.000x | 1.224x | 0.000622 |
| mlp_down | down_proj | 8 | 8192 | 2048 | gptq_marlin | 4 | 128 | 0.0166 | 0.0170 | 16.163 | 520.88 | 1.000x | 1.341x | 0.000546 |
| mlp_down | down_proj | 16 | 8192 | 2048 | gptq_marlin | 4 | 128 | 0.0180 | 0.0184 | 29.853 | 481.02 | 1.000x | 1.230x | 0.000539 |
| mlp_down | down_proj | 1 | 8192 | 2048 | gptq_machete | 4 | 128 | 0.0221 | 0.0230 | 1.520 | 393.28 | 0.742x | 1.000x | 0.000346 |
| mlp_down | down_proj | 2 | 8192 | 2048 | gptq_machete | 4 | 128 | 0.0220 | 0.0230 | 3.048 | 394.42 | 0.824x | 1.000x | 0.000545 |
| mlp_down | down_proj | 4 | 8192 | 2048 | gptq_machete | 4 | 128 | 0.0222 | 0.0233 | 6.035 | 390.45 | 0.817x | 1.000x | 0.000622 |
| mlp_down | down_proj | 8 | 8192 | 2048 | gptq_machete | 4 | 128 | 0.0223 | 0.0232 | 12.053 | 389.89 | 0.746x | 1.000x | 0.000546 |
| mlp_down | down_proj | 16 | 8192 | 2048 | gptq_machete | 4 | 128 | 0.0221 | 0.0230 | 24.280 | 392.71 | 0.813x | 1.000x | 0.000539 |

The four shape rows preserve all seven Llama 3.2 1B projection roles: `q_proj/o_proj` (2048×2048), `k_proj/v_proj` (2048×512), `gate_proj/up_proj` (2048×8192), and `down_proj` (8192×2048).
