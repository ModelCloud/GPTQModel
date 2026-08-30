# QVQ LR versus W4 GPTQ kernels — Llama 3.2 1B shapes

This is a kernel-throughput comparison on the H100 host from PR #62. W4 GPTQ is a figurative performance baseline, not a quality-equivalent arm.

## Measurement contract

- Commit: `041ab457e7847203d7ea685be769439dcca87933`; benchmark SHA256: `ae13dac8f763d972716bba127790a3bffea2c83f7d2e91d7479728527011b6ca`
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
| attn_qo | q_proj/o_proj | 1 | 2048 | 2048 | qvq_lr | 2 | P32 | 0.0144 | 0.0149 | 0.581 | 73.71 | 1.003x | 1.071x | 7.63e-06 |
| attn_qo | q_proj/o_proj | 2 | 2048 | 2048 | qvq_lr | 2 | P32 | 0.0152 | 0.0158 | 1.101 | 69.92 | 1.068x | 1.013x | 1.34e-05 |
| attn_qo | q_proj/o_proj | 4 | 2048 | 2048 | qvq_lr | 2 | P32 | 0.0152 | 0.0158 | 2.205 | 69.99 | 1.081x | 1.017x | 1.43e-05 |
| attn_qo | q_proj/o_proj | 8 | 2048 | 2048 | qvq_lr | 2 | P32 | 0.0154 | 0.0161 | 4.355 | 69.12 | 0.964x | 1.003x | 1.34e-05 |
| attn_qo | q_proj/o_proj | 16 | 2048 | 2048 | qvq_lr | 2 | P32 | 0.0155 | 0.0161 | 8.648 | 68.62 | 1.008x | 0.998x | 1.34e-05 |
| attn_qo | q_proj/o_proj | 1 | 2048 | 2048 | qvq_lr | 2.5 | P32 | 0.0146 | 0.0152 | 0.573 | 90.65 | 0.990x | 1.057x | 7.63e-06 |
| attn_qo | q_proj/o_proj | 2 | 2048 | 2048 | qvq_lr | 2.5 | P32 | 0.0153 | 0.0161 | 1.093 | 86.49 | 1.060x | 1.005x | 1.43e-05 |
| attn_qo | q_proj/o_proj | 4 | 2048 | 2048 | qvq_lr | 2.5 | P32 | 0.0152 | 0.0160 | 2.205 | 87.22 | 1.081x | 1.017x | 1.14e-05 |
| attn_qo | q_proj/o_proj | 8 | 2048 | 2048 | qvq_lr | 2.5 | P32 | 0.0153 | 0.0159 | 4.383 | 86.67 | 0.970x | 1.009x | 1.34e-05 |
| attn_qo | q_proj/o_proj | 16 | 2048 | 2048 | qvq_lr | 2.5 | P32 | 0.0156 | 0.0163 | 8.586 | 84.90 | 1.001x | 0.991x | 1.34e-05 |
| attn_qo | q_proj/o_proj | 1 | 2048 | 2048 | qvq_lr | 3 | P32 | 0.0154 | 0.0159 | 0.543 | 102.93 | 0.939x | 1.002x | 8.58e-06 |
| attn_qo | q_proj/o_proj | 2 | 2048 | 2048 | qvq_lr | 3 | P32 | 0.0166 | 0.0172 | 1.010 | 95.69 | 0.980x | 0.929x | 1.24e-05 |
| attn_qo | q_proj/o_proj | 4 | 2048 | 2048 | qvq_lr | 3 | P32 | 0.0167 | 0.0174 | 2.015 | 95.42 | 0.988x | 0.929x | 1.34e-05 |
| attn_qo | q_proj/o_proj | 8 | 2048 | 2048 | qvq_lr | 3 | P32 | 0.0167 | 0.0174 | 4.018 | 95.14 | 0.889x | 0.925x | 1.53e-05 |
| attn_qo | q_proj/o_proj | 16 | 2048 | 2048 | qvq_lr | 3 | P32 | 0.0168 | 0.0175 | 7.966 | 94.33 | 0.929x | 0.919x | 1.34e-05 |
| attn_qo | q_proj/o_proj | 1 | 2048 | 2048 | qvq_lr | 3.5 | P32 | 0.0620 | 0.0628 | 0.135 | 29.87 | 0.234x | 0.250x | 4.77e-06 |
| attn_qo | q_proj/o_proj | 2 | 2048 | 2048 | qvq_lr | 3.5 | P32 | 0.0557 | 0.0564 | 0.301 | 33.21 | 0.292x | 0.277x | 3.81e-06 |
| attn_qo | q_proj/o_proj | 4 | 2048 | 2048 | qvq_lr | 3.5 | P32 | 0.0559 | 0.0568 | 0.600 | 33.11 | 0.294x | 0.277x | 4.77e-06 |
| attn_qo | q_proj/o_proj | 8 | 2048 | 2048 | qvq_lr | 3.5 | P32 | 0.0570 | 0.0580 | 1.177 | 32.48 | 0.260x | 0.271x | 4.77e-06 |
| attn_qo | q_proj/o_proj | 16 | 2048 | 2048 | qvq_lr | 3.5 | P32 | 0.0616 | 0.0627 | 2.178 | 30.05 | 0.254x | 0.251x | 4.77e-06 |
| attn_qo | q_proj/o_proj | 1 | 2048 | 2048 | gptq_marlin | 4 | 128 | 0.0145 | 0.0148 | 0.579 | 149.19 | 1.000x | 1.067x | 0.000264 |
| attn_qo | q_proj/o_proj | 2 | 2048 | 2048 | gptq_marlin | 4 | 128 | 0.0163 | 0.0167 | 1.031 | 132.91 | 1.000x | 0.948x | 0.000229 |
| attn_qo | q_proj/o_proj | 4 | 2048 | 2048 | gptq_marlin | 4 | 128 | 0.0164 | 0.0168 | 2.040 | 131.49 | 1.000x | 0.941x | 0.000197 |
| attn_qo | q_proj/o_proj | 8 | 2048 | 2048 | gptq_marlin | 4 | 128 | 0.0148 | 0.0152 | 4.520 | 145.66 | 1.000x | 1.041x | 0.000242 |
| attn_qo | q_proj/o_proj | 16 | 2048 | 2048 | gptq_marlin | 4 | 128 | 0.0156 | 0.0161 | 8.577 | 138.21 | 1.000x | 0.990x | 0.00028 |
| attn_qo | q_proj/o_proj | 1 | 2048 | 2048 | gptq_machete | 4 | 128 | 0.0155 | 0.0164 | 0.542 | 140.31 | 0.937x | 1.000x | 0.000264 |
| attn_qo | q_proj/o_proj | 2 | 2048 | 2048 | gptq_machete | 4 | 128 | 0.0154 | 0.0163 | 1.088 | 140.75 | 1.055x | 1.000x | 0.000229 |
| attn_qo | q_proj/o_proj | 4 | 2048 | 2048 | gptq_machete | 4 | 128 | 0.0155 | 0.0163 | 2.169 | 140.31 | 1.063x | 1.000x | 0.000197 |
| attn_qo | q_proj/o_proj | 8 | 2048 | 2048 | gptq_machete | 4 | 128 | 0.0155 | 0.0163 | 4.342 | 140.46 | 0.961x | 1.000x | 0.000242 |
| attn_qo | q_proj/o_proj | 16 | 2048 | 2048 | gptq_machete | 4 | 128 | 0.0155 | 0.0164 | 8.666 | 140.17 | 1.010x | 1.000x | 0.00028 |
| attn_kv | k_proj/v_proj | 1 | 2048 | 512 | qvq_lr | 2 | P32 | 0.0116 | 0.0122 | 0.181 | 22.95 | 1.892x | 1.248x | 5.72e-06 |
| attn_kv | k_proj/v_proj | 2 | 2048 | 512 | qvq_lr | 2 | P32 | 0.0116 | 0.0121 | 0.361 | 22.92 | 2.176x | 1.274x | 7.63e-06 |
| attn_kv | k_proj/v_proj | 4 | 2048 | 512 | qvq_lr | 2 | P32 | 0.0114 | 0.0120 | 0.737 | 23.40 | 2.277x | 1.302x | 6.68e-06 |
| attn_kv | k_proj/v_proj | 8 | 2048 | 512 | qvq_lr | 2 | P32 | 0.0114 | 0.0120 | 1.473 | 23.37 | 1.982x | 1.270x | 7.63e-06 |
| attn_kv | k_proj/v_proj | 16 | 2048 | 512 | qvq_lr | 2 | P32 | 0.0115 | 0.0121 | 2.925 | 23.21 | 2.100x | 1.294x | 7.63e-06 |
| attn_kv | k_proj/v_proj | 1 | 2048 | 512 | qvq_lr | 2.5 | P32 | 0.0116 | 0.0123 | 0.180 | 28.48 | 1.885x | 1.243x | 5.72e-06 |
| attn_kv | k_proj/v_proj | 2 | 2048 | 512 | qvq_lr | 2.5 | P32 | 0.0117 | 0.0123 | 0.357 | 28.25 | 2.153x | 1.260x | 8.11e-06 |
| attn_kv | k_proj/v_proj | 4 | 2048 | 512 | qvq_lr | 2.5 | P32 | 0.0117 | 0.0124 | 0.714 | 28.25 | 2.206x | 1.262x | 6.68e-06 |
| attn_kv | k_proj/v_proj | 8 | 2048 | 512 | qvq_lr | 2.5 | P32 | 0.0116 | 0.0125 | 1.440 | 28.48 | 1.938x | 1.242x | 7.63e-06 |
| attn_kv | k_proj/v_proj | 16 | 2048 | 512 | qvq_lr | 2.5 | P32 | 0.0120 | 0.0124 | 2.800 | 27.68 | 2.011x | 1.239x | 9.54e-06 |
| attn_kv | k_proj/v_proj | 1 | 2048 | 512 | qvq_lr | 3 | P32 | 0.0208 | 0.0215 | 0.101 | 19.07 | 1.054x | 0.695x | 1.91e-06 |
| attn_kv | k_proj/v_proj | 2 | 2048 | 512 | qvq_lr | 3 | P32 | 0.0202 | 0.0208 | 0.208 | 19.71 | 1.254x | 0.734x | 3.81e-06 |
| attn_kv | k_proj/v_proj | 4 | 2048 | 512 | qvq_lr | 3 | P32 | 0.0204 | 0.0211 | 0.410 | 19.43 | 1.267x | 0.725x | 4.77e-06 |
| attn_kv | k_proj/v_proj | 8 | 2048 | 512 | qvq_lr | 3 | P32 | 0.0214 | 0.0222 | 0.785 | 18.59 | 1.056x | 0.677x | 4.77e-06 |
| attn_kv | k_proj/v_proj | 16 | 2048 | 512 | qvq_lr | 3 | P32 | 0.0240 | 0.0247 | 1.397 | 16.54 | 1.003x | 0.618x | 3.81e-06 |
| attn_kv | k_proj/v_proj | 1 | 2048 | 512 | qvq_lr | 3.5 | P32 | 0.0223 | 0.0230 | 0.094 | 20.75 | 0.984x | 0.649x | 2.38e-06 |
| attn_kv | k_proj/v_proj | 2 | 2048 | 512 | qvq_lr | 3.5 | P32 | 0.0206 | 0.0211 | 0.204 | 22.51 | 1.230x | 0.720x | 4.29e-06 |
| attn_kv | k_proj/v_proj | 4 | 2048 | 512 | qvq_lr | 3.5 | P32 | 0.0209 | 0.0216 | 0.402 | 22.17 | 1.241x | 0.710x | 3.81e-06 |
| attn_kv | k_proj/v_proj | 8 | 2048 | 512 | qvq_lr | 3.5 | P32 | 0.0216 | 0.0223 | 0.777 | 21.44 | 1.046x | 0.670x | 3.81e-06 |
| attn_kv | k_proj/v_proj | 16 | 2048 | 512 | qvq_lr | 3.5 | P32 | 0.0239 | 0.0247 | 1.404 | 19.36 | 1.008x | 0.621x | 4.29e-06 |
| attn_kv | k_proj/v_proj | 1 | 2048 | 512 | gptq_marlin | 4 | 128 | 0.0220 | 0.0223 | 0.096 | 24.63 | 1.000x | 0.660x | 0.000167 |
| attn_kv | k_proj/v_proj | 2 | 2048 | 512 | gptq_marlin | 4 | 128 | 0.0253 | 0.0259 | 0.166 | 21.39 | 1.000x | 0.585x | 0.000155 |
| attn_kv | k_proj/v_proj | 4 | 2048 | 512 | gptq_marlin | 4 | 128 | 0.0259 | 0.0265 | 0.324 | 20.87 | 1.000x | 0.572x | 0.000167 |
| attn_kv | k_proj/v_proj | 8 | 2048 | 512 | gptq_marlin | 4 | 128 | 0.0226 | 0.0230 | 0.743 | 23.95 | 1.000x | 0.641x | 0.000206 |
| attn_kv | k_proj/v_proj | 16 | 2048 | 512 | gptq_marlin | 4 | 128 | 0.0241 | 0.0246 | 1.393 | 22.44 | 1.000x | 0.616x | 0.000194 |
| attn_kv | k_proj/v_proj | 1 | 2048 | 512 | gptq_machete | 4 | 128 | 0.0145 | 0.0154 | 0.145 | 37.90 | 1.516x | 1.000x | 0.000167 |
| attn_kv | k_proj/v_proj | 2 | 2048 | 512 | gptq_machete | 4 | 128 | 0.0148 | 0.0156 | 0.283 | 37.09 | 1.708x | 1.000x | 0.000155 |
| attn_kv | k_proj/v_proj | 4 | 2048 | 512 | gptq_machete | 4 | 128 | 0.0148 | 0.0158 | 0.566 | 37.05 | 1.748x | 1.000x | 0.000167 |
| attn_kv | k_proj/v_proj | 8 | 2048 | 512 | gptq_machete | 4 | 128 | 0.0145 | 0.0154 | 1.160 | 37.95 | 1.561x | 1.000x | 0.000206 |
| attn_kv | k_proj/v_proj | 16 | 2048 | 512 | gptq_machete | 4 | 128 | 0.0148 | 0.0160 | 2.260 | 36.97 | 1.623x | 1.000x | 0.000194 |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | qvq_lr | 2 | P32 | 0.0348 | 0.0351 | 0.966 | 122.58 | 0.286x | 0.518x | 5.72e-05 |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | qvq_lr | 2 | P32 | 0.0348 | 0.0352 | 1.926 | 122.24 | 0.295x | 0.509x | 5.15e-05 |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | qvq_lr | 2 | P32 | 0.0349 | 0.0353 | 3.841 | 121.90 | 0.294x | 0.516x | 5.72e-05 |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | qvq_lr | 2 | P32 | 0.0352 | 0.0356 | 7.626 | 121.02 | 0.285x | 0.513x | 7.63e-05 |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | qvq_lr | 2 | P32 | 0.0354 | 0.0358 | 15.156 | 120.25 | 0.308x | 0.508x | 6.1e-05 |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | qvq_lr | 2.5 | P32 | 0.0342 | 0.0345 | 0.982 | 155.33 | 0.291x | 0.527x | 4.86e-05 |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | qvq_lr | 2.5 | P32 | 0.0342 | 0.0346 | 1.963 | 155.25 | 0.300x | 0.518x | 5.53e-05 |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | qvq_lr | 2.5 | P32 | 0.0344 | 0.0349 | 3.903 | 154.39 | 0.299x | 0.524x | 6.29e-05 |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | qvq_lr | 2.5 | P32 | 0.0348 | 0.0352 | 7.717 | 152.61 | 0.288x | 0.519x | 5.34e-05 |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | qvq_lr | 2.5 | P32 | 0.0352 | 0.0355 | 15.266 | 150.94 | 0.310x | 0.511x | 6.87e-05 |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | qvq_lr | 3 | P32 | 0.0347 | 0.0353 | 0.968 | 183.43 | 0.287x | 0.520x | 2.48e-05 |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | qvq_lr | 3 | P32 | 0.0348 | 0.0354 | 1.929 | 182.76 | 0.295x | 0.510x | 2.67e-05 |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | qvq_lr | 3 | P32 | 0.0348 | 0.0356 | 3.860 | 182.84 | 0.295x | 0.518x | 2.57e-05 |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | qvq_lr | 3 | P32 | 0.0351 | 0.0356 | 7.640 | 180.93 | 0.285x | 0.514x | 3.43e-05 |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | qvq_lr | 3 | P32 | 0.0355 | 0.0361 | 15.121 | 179.05 | 0.307x | 0.507x | 3.05e-05 |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | qvq_lr | 3.5 | P32 | 0.2363 | 0.2373 | 0.142 | 31.34 | 0.042x | 0.076x | 6.2e-06 |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | qvq_lr | 3.5 | P32 | 0.1962 | 0.1999 | 0.342 | 37.75 | 0.052x | 0.090x | 4.77e-06 |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | qvq_lr | 3.5 | P32 | 0.1992 | 0.2010 | 0.674 | 37.19 | 0.052x | 0.090x | 5.72e-06 |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | qvq_lr | 3.5 | P32 | 0.1996 | 0.2014 | 1.345 | 37.10 | 0.050x | 0.090x | 5.96e-06 |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | qvq_lr | 3.5 | P32 | 0.2065 | 0.2094 | 2.600 | 35.86 | 0.053x | 0.087x | 5.25e-06 |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | gptq_marlin | 4 | 128 | 0.0100 | 0.0103 | 3.372 | 869.25 | 1.000x | 1.810x | 0.000272 |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | gptq_marlin | 4 | 128 | 0.0103 | 0.0106 | 6.533 | 842.17 | 1.000x | 1.726x | 0.000299 |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | gptq_marlin | 4 | 128 | 0.0103 | 0.0107 | 13.066 | 842.17 | 1.000x | 1.754x | 0.000274 |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | gptq_marlin | 4 | 128 | 0.0100 | 0.0104 | 26.801 | 863.69 | 1.000x | 1.802x | 0.000278 |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | gptq_marlin | 4 | 128 | 0.0109 | 0.0113 | 49.272 | 793.94 | 1.000x | 1.651x | 0.000327 |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | gptq_machete | 4 | 128 | 0.0180 | 0.0189 | 1.862 | 480.63 | 0.552x | 1.000x | 0.000272 |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | gptq_machete | 4 | 128 | 0.0177 | 0.0186 | 3.785 | 488.43 | 0.579x | 1.000x | 0.000299 |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | gptq_machete | 4 | 128 | 0.0180 | 0.0189 | 7.450 | 480.63 | 0.570x | 1.000x | 0.000274 |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | gptq_machete | 4 | 128 | 0.0180 | 0.0191 | 14.873 | 479.77 | 0.555x | 1.000x | 0.000278 |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | gptq_machete | 4 | 128 | 0.0180 | 0.0189 | 29.853 | 481.48 | 0.606x | 1.000x | 0.000327 |
| mlp_down | down_proj | 1 | 8192 | 2048 | qvq_lr | 2 | P32 | 0.0310 | 0.0318 | 1.081 | 137.24 | 0.532x | 0.718x | 4.2e-05 |
| mlp_down | down_proj | 2 | 8192 | 2048 | qvq_lr | 2 | P32 | 0.0374 | 0.0380 | 1.794 | 113.88 | 0.484x | 0.582x | 9.92e-05 |
| mlp_down | down_proj | 4 | 8192 | 2048 | qvq_lr | 2 | P32 | 0.0377 | 0.0382 | 3.564 | 113.10 | 0.478x | 0.592x | 8.39e-05 |
| mlp_down | down_proj | 8 | 8192 | 2048 | qvq_lr | 2 | P32 | 0.0377 | 0.0383 | 7.112 | 112.86 | 0.438x | 0.588x | 9.16e-05 |
| mlp_down | down_proj | 16 | 8192 | 2048 | qvq_lr | 2 | P32 | 0.0379 | 0.0386 | 14.152 | 112.29 | 0.474x | 0.590x | 0.000101 |
| mlp_down | down_proj | 1 | 8192 | 2048 | qvq_lr | 2.5 | P32 | 0.0316 | 0.0323 | 1.062 | 168.07 | 0.523x | 0.705x | 4.2e-05 |
| mlp_down | down_proj | 2 | 8192 | 2048 | qvq_lr | 2.5 | P32 | 0.0370 | 0.0375 | 1.814 | 143.50 | 0.490x | 0.588x | 9.92e-05 |
| mlp_down | down_proj | 4 | 8192 | 2048 | qvq_lr | 2.5 | P32 | 0.0371 | 0.0377 | 3.620 | 143.19 | 0.486x | 0.602x | 9.92e-05 |
| mlp_down | down_proj | 8 | 8192 | 2048 | qvq_lr | 2.5 | P32 | 0.0371 | 0.0378 | 7.228 | 142.95 | 0.445x | 0.598x | 0.000113 |
| mlp_down | down_proj | 16 | 8192 | 2048 | qvq_lr | 2.5 | P32 | 0.0376 | 0.0382 | 14.278 | 141.18 | 0.478x | 0.596x | 0.000103 |
| mlp_down | down_proj | 1 | 8192 | 2048 | qvq_lr | 3 | P32 | 0.0347 | 0.0353 | 0.966 | 183.09 | 0.476x | 0.641x | 4.58e-05 |
| mlp_down | down_proj | 2 | 8192 | 2048 | qvq_lr | 3 | P32 | 0.0421 | 0.0427 | 1.593 | 150.90 | 0.430x | 0.517x | 8.58e-05 |
| mlp_down | down_proj | 4 | 8192 | 2048 | qvq_lr | 3 | P32 | 0.0422 | 0.0428 | 3.181 | 150.67 | 0.427x | 0.529x | 8.58e-05 |
| mlp_down | down_proj | 8 | 8192 | 2048 | qvq_lr | 3 | P32 | 0.0421 | 0.0428 | 6.369 | 150.84 | 0.393x | 0.527x | 9.35e-05 |
| mlp_down | down_proj | 16 | 8192 | 2048 | qvq_lr | 3 | P32 | 0.0425 | 0.0432 | 12.633 | 149.59 | 0.423x | 0.527x | 0.000122 |
| mlp_down | down_proj | 1 | 8192 | 2048 | qvq_lr | 3.5 | P32 | 0.2238 | 0.2252 | 0.150 | 33.09 | 0.074x | 0.100x | 1.14e-05 |
| mlp_down | down_proj | 2 | 8192 | 2048 | qvq_lr | 3.5 | P32 | 0.1998 | 0.2034 | 0.336 | 37.06 | 0.091x | 0.109x | 1.24e-05 |
| mlp_down | down_proj | 4 | 8192 | 2048 | qvq_lr | 3.5 | P32 | 0.2005 | 0.2032 | 0.669 | 36.93 | 0.090x | 0.111x | 1.34e-05 |
| mlp_down | down_proj | 8 | 8192 | 2048 | qvq_lr | 3.5 | P32 | 0.2015 | 0.2037 | 1.332 | 36.75 | 0.082x | 0.110x | 1.72e-05 |
| mlp_down | down_proj | 16 | 8192 | 2048 | qvq_lr | 3.5 | P32 | 0.2070 | 0.2097 | 2.594 | 35.78 | 0.087x | 0.108x | 1.34e-05 |
| mlp_down | down_proj | 1 | 8192 | 2048 | gptq_marlin | 4 | 128 | 0.0165 | 0.0169 | 2.032 | 523.91 | 1.000x | 1.349x | 0.000346 |
| mlp_down | down_proj | 2 | 8192 | 2048 | gptq_marlin | 4 | 128 | 0.0181 | 0.0186 | 3.705 | 477.63 | 1.000x | 1.201x | 0.000545 |
| mlp_down | down_proj | 4 | 8192 | 2048 | gptq_marlin | 4 | 128 | 0.0180 | 0.0184 | 7.450 | 480.17 | 1.000x | 1.238x | 0.000622 |
| mlp_down | down_proj | 8 | 8192 | 2048 | gptq_marlin | 4 | 128 | 0.0165 | 0.0169 | 16.226 | 522.89 | 1.000x | 1.342x | 0.000546 |
| mlp_down | down_proj | 16 | 8192 | 2048 | gptq_marlin | 4 | 128 | 0.0180 | 0.0184 | 29.853 | 481.02 | 1.000x | 1.246x | 0.000539 |
| mlp_down | down_proj | 1 | 8192 | 2048 | gptq_machete | 4 | 128 | 0.0223 | 0.0233 | 1.507 | 389.89 | 0.741x | 1.000x | 0.000346 |
| mlp_down | down_proj | 2 | 8192 | 2048 | gptq_machete | 4 | 128 | 0.0218 | 0.0227 | 3.084 | 399.06 | 0.832x | 1.000x | 0.000545 |
| mlp_down | down_proj | 4 | 8192 | 2048 | gptq_machete | 4 | 128 | 0.0223 | 0.0232 | 6.018 | 389.33 | 0.808x | 1.000x | 0.000622 |
| mlp_down | down_proj | 8 | 8192 | 2048 | gptq_machete | 4 | 128 | 0.0222 | 0.0231 | 12.087 | 391.01 | 0.745x | 1.000x | 0.000546 |
| mlp_down | down_proj | 16 | 8192 | 2048 | gptq_machete | 4 | 128 | 0.0224 | 0.0232 | 23.967 | 387.66 | 0.803x | 1.000x | 0.000539 |

The four shape rows preserve all seven Llama 3.2 1B projection roles: `q_proj/o_proj` (2048×2048), `k_proj/v_proj` (2048×512), `gate_proj/up_proj` (2048×8192), and `down_proj` (8192×2048).
