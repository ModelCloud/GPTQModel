# QVQ LR versus W4 GPTQ kernels — Llama 3.2 1B shapes

This is a kernel-throughput comparison on the H100 host from PR #62. W4 GPTQ is a figurative performance baseline, not a quality-equivalent arm.

## Measurement contract

- Commit: `e0c29e604db571b9afe91cba9b15e1e88e6d6631`; benchmark SHA256: `ae13dac8f763d972716bba127790a3bffea2c83f7d2e91d7479728527011b6ca`
- GPU: physical `0`, `NVIDIA H200`, PCI `00000000:1C:00.0`, UUID `GPU-0c667065-5c47-38ce-0b0a-d211392ce9ea`, CC `9.0`, 132 SMs
- Torch/CUDA: `2.15.0.dev20260828+cu130` / `13.0`; input dtype: `float16`
- M values: `[1, 2, 4, 8, 16]`; warmup: `10`; measured launches: `60`
- QVQ: `qvq_v2b2_p32_lr`, rates `[2.0, 2.5, 3.0, 3.5]`, native `P32 is LR packing geometry, not a GPTQ affine scale group.`
- GPTQ: symmetric W4, group `128`, no activation order; Marlin and Machete use the same W4 source payload
- Latency is CUDA-event median/P95 from one CUDA Graph replay containing every measured launch. CPU scheduling and host launch gaps are outside each timed interval. Logical TFLOP/s is `2*M*K*N / median_ms`; payload GB/s is packed payload bytes divided by median latency.
- Every row passed its dense-reference shape/dtype/finite gate. QVQ uses `max_abs <= 2e-3`; GPTQ uses `atol=2e-2, rtol=2e-2`.

`xMarlin` and `xMachete` are baseline median latency divided by the row median, so values above 1.0x are faster.

| Shape | Roles | M | K | N | Kernel | W | Group | Median ms | P95 ms | Logical TFLOP/s | Payload GB/s | xMarlin | xMachete | Max abs |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| attn_qo | q_proj/o_proj | 1 | 2048 | 2048 | qvq_lr | 2 | P32 | 0.0146 | 0.0156 | 0.576 | 73.14 | 1.003x | 1.065x | 7.63e-06 |
| attn_qo | q_proj/o_proj | 2 | 2048 | 2048 | qvq_lr | 2 | P32 | 0.0157 | 0.0164 | 1.070 | 67.92 | 1.034x | 1.011x | 1.34e-05 |
| attn_qo | q_proj/o_proj | 4 | 2048 | 2048 | qvq_lr | 2 | P32 | 0.0157 | 0.0165 | 2.140 | 67.92 | 1.041x | 0.986x | 1.43e-05 |
| attn_qo | q_proj/o_proj | 8 | 2048 | 2048 | qvq_lr | 2 | P32 | 0.0158 | 0.0164 | 4.258 | 67.57 | 0.938x | 1.007x | 1.34e-05 |
| attn_qo | q_proj/o_proj | 16 | 2048 | 2048 | qvq_lr | 2 | P32 | 0.0159 | 0.0166 | 8.448 | 67.03 | 0.991x | 0.983x | 1.34e-05 |
| attn_qo | q_proj/o_proj | 1 | 2048 | 2048 | qvq_lr | 2.5 | P32 | 0.0150 | 0.0156 | 0.560 | 88.62 | 0.975x | 1.035x | 7.63e-06 |
| attn_qo | q_proj/o_proj | 2 | 2048 | 2048 | qvq_lr | 2.5 | P32 | 0.0156 | 0.0163 | 1.073 | 84.90 | 1.037x | 1.014x | 1.43e-05 |
| attn_qo | q_proj/o_proj | 4 | 2048 | 2048 | qvq_lr | 2.5 | P32 | 0.0157 | 0.0166 | 2.140 | 84.64 | 1.041x | 0.986x | 1.14e-05 |
| attn_qo | q_proj/o_proj | 8 | 2048 | 2048 | qvq_lr | 2.5 | P32 | 0.0160 | 0.0165 | 4.207 | 83.19 | 0.927x | 0.995x | 1.34e-05 |
| attn_qo | q_proj/o_proj | 16 | 2048 | 2048 | qvq_lr | 2.5 | P32 | 0.0159 | 0.0169 | 8.422 | 83.28 | 0.988x | 0.980x | 1.34e-05 |
| attn_qo | q_proj/o_proj | 1 | 2048 | 2048 | qvq_lr | 3 | P32 | 0.0592 | 0.0602 | 0.142 | 26.85 | 0.247x | 0.262x | 3.34e-06 |
| attn_qo | q_proj/o_proj | 2 | 2048 | 2048 | qvq_lr | 3 | P32 | 0.0583 | 0.0594 | 0.288 | 27.26 | 0.278x | 0.272x | 4.77e-06 |
| attn_qo | q_proj/o_proj | 4 | 2048 | 2048 | qvq_lr | 3 | P32 | 0.0591 | 0.0603 | 0.568 | 26.89 | 0.276x | 0.262x | 4.77e-06 |
| attn_qo | q_proj/o_proj | 8 | 2048 | 2048 | qvq_lr | 3 | P32 | 0.0603 | 0.0614 | 1.113 | 26.35 | 0.245x | 0.263x | 5.72e-06 |
| attn_qo | q_proj/o_proj | 16 | 2048 | 2048 | qvq_lr | 3 | P32 | 0.0220 | 0.0226 | 6.096 | 72.19 | 0.715x | 0.709x | 1.34e-05 |
| attn_qo | q_proj/o_proj | 1 | 2048 | 2048 | qvq_lr | 3.5 | P32 | 0.0646 | 0.0655 | 0.130 | 28.67 | 0.226x | 0.240x | 4.77e-06 |
| attn_qo | q_proj/o_proj | 2 | 2048 | 2048 | qvq_lr | 3.5 | P32 | 0.0580 | 0.0589 | 0.289 | 31.91 | 0.279x | 0.273x | 3.81e-06 |
| attn_qo | q_proj/o_proj | 4 | 2048 | 2048 | qvq_lr | 3.5 | P32 | 0.0584 | 0.0596 | 0.575 | 31.70 | 0.279x | 0.265x | 4.77e-06 |
| attn_qo | q_proj/o_proj | 8 | 2048 | 2048 | qvq_lr | 3.5 | P32 | 0.0596 | 0.0604 | 1.125 | 31.04 | 0.248x | 0.266x | 4.77e-06 |
| attn_qo | q_proj/o_proj | 16 | 2048 | 2048 | qvq_lr | 3.5 | P32 | 0.0633 | 0.0644 | 2.122 | 29.26 | 0.249x | 0.247x | 4.77e-06 |
| attn_qo | q_proj/o_proj | 1 | 2048 | 2048 | gptq_marlin | 4 | 128 | 0.0146 | 0.0149 | 0.574 | 148.05 | 1.000x | 1.061x | 0.000264 |
| attn_qo | q_proj/o_proj | 2 | 2048 | 2048 | gptq_marlin | 4 | 128 | 0.0162 | 0.0165 | 1.035 | 133.43 | 1.000x | 0.978x | 0.000229 |
| attn_qo | q_proj/o_proj | 4 | 2048 | 2048 | gptq_marlin | 4 | 128 | 0.0163 | 0.0166 | 2.056 | 132.52 | 1.000x | 0.947x | 0.000197 |
| attn_qo | q_proj/o_proj | 8 | 2048 | 2048 | gptq_marlin | 4 | 128 | 0.0148 | 0.0152 | 4.539 | 146.29 | 1.000x | 1.074x | 0.000242 |
| attn_qo | q_proj/o_proj | 16 | 2048 | 2048 | gptq_marlin | 4 | 128 | 0.0157 | 0.0161 | 8.525 | 137.37 | 1.000x | 0.992x | 0.00028 |
| attn_qo | q_proj/o_proj | 1 | 2048 | 2048 | gptq_machete | 4 | 128 | 0.0155 | 0.0165 | 0.541 | 140.02 | 0.942x | 1.000x | 0.000264 |
| attn_qo | q_proj/o_proj | 2 | 2048 | 2048 | gptq_machete | 4 | 128 | 0.0159 | 0.0169 | 1.058 | 136.91 | 1.022x | 1.000x | 0.000229 |
| attn_qo | q_proj/o_proj | 4 | 2048 | 2048 | gptq_machete | 4 | 128 | 0.0155 | 0.0165 | 2.171 | 140.46 | 1.056x | 1.000x | 0.000197 |
| attn_qo | q_proj/o_proj | 8 | 2048 | 2048 | gptq_machete | 4 | 128 | 0.0159 | 0.0168 | 4.228 | 136.77 | 0.931x | 1.000x | 0.000242 |
| attn_qo | q_proj/o_proj | 16 | 2048 | 2048 | gptq_machete | 4 | 128 | 0.0156 | 0.0167 | 8.595 | 139.02 | 1.008x | 1.000x | 0.00028 |
| attn_kv | k_proj/v_proj | 1 | 2048 | 512 | qvq_lr | 2 | P32 | 0.0119 | 0.0126 | 0.176 | 22.34 | 1.855x | 1.262x | 5.72e-06 |
| attn_kv | k_proj/v_proj | 2 | 2048 | 512 | qvq_lr | 2 | P32 | 0.0117 | 0.0125 | 0.359 | 22.79 | 2.195x | 1.289x | 7.63e-06 |
| attn_kv | k_proj/v_proj | 4 | 2048 | 512 | qvq_lr | 2 | P32 | 0.0120 | 0.0126 | 0.700 | 22.22 | 2.171x | 1.254x | 6.68e-06 |
| attn_kv | k_proj/v_proj | 8 | 2048 | 512 | qvq_lr | 2 | P32 | 0.0120 | 0.0126 | 1.393 | 22.10 | 1.899x | 1.251x | 7.63e-06 |
| attn_kv | k_proj/v_proj | 16 | 2048 | 512 | qvq_lr | 2 | P32 | 0.0121 | 0.0127 | 2.774 | 22.01 | 2.024x | 1.246x | 7.63e-06 |
| attn_kv | k_proj/v_proj | 1 | 2048 | 512 | qvq_lr | 2.5 | P32 | 0.0120 | 0.0125 | 0.174 | 27.57 | 1.838x | 1.250x | 5.72e-06 |
| attn_kv | k_proj/v_proj | 2 | 2048 | 512 | qvq_lr | 2.5 | P32 | 0.0119 | 0.0125 | 0.353 | 27.91 | 2.156x | 1.266x | 8.11e-06 |
| attn_kv | k_proj/v_proj | 4 | 2048 | 512 | qvq_lr | 2.5 | P32 | 0.0120 | 0.0126 | 0.700 | 27.68 | 2.171x | 1.254x | 6.68e-06 |
| attn_kv | k_proj/v_proj | 8 | 2048 | 512 | qvq_lr | 2.5 | P32 | 0.0119 | 0.0126 | 1.406 | 27.80 | 1.917x | 1.263x | 7.63e-06 |
| attn_kv | k_proj/v_proj | 16 | 2048 | 512 | qvq_lr | 2.5 | P32 | 0.0121 | 0.0127 | 2.767 | 27.36 | 2.018x | 1.243x | 9.54e-06 |
| attn_kv | k_proj/v_proj | 1 | 2048 | 512 | qvq_lr | 3 | P32 | 0.0218 | 0.0224 | 0.096 | 18.22 | 1.014x | 0.690x | 1.91e-06 |
| attn_kv | k_proj/v_proj | 2 | 2048 | 512 | qvq_lr | 3 | P32 | 0.0210 | 0.0217 | 0.200 | 18.90 | 1.219x | 0.716x | 3.81e-06 |
| attn_kv | k_proj/v_proj | 4 | 2048 | 512 | qvq_lr | 3 | P32 | 0.0215 | 0.0220 | 0.390 | 18.49 | 1.211x | 0.699x | 4.77e-06 |
| attn_kv | k_proj/v_proj | 8 | 2048 | 512 | qvq_lr | 3 | P32 | 0.0222 | 0.0228 | 0.756 | 17.90 | 1.031x | 0.679x | 4.77e-06 |
| attn_kv | k_proj/v_proj | 16 | 2048 | 512 | qvq_lr | 3 | P32 | 0.0244 | 0.0254 | 1.372 | 16.25 | 1.001x | 0.616x | 3.81e-06 |
| attn_kv | k_proj/v_proj | 1 | 2048 | 512 | qvq_lr | 3.5 | P32 | 0.0238 | 0.0245 | 0.088 | 19.41 | 0.928x | 0.631x | 2.38e-06 |
| attn_kv | k_proj/v_proj | 2 | 2048 | 512 | qvq_lr | 3.5 | P32 | 0.0210 | 0.0218 | 0.200 | 22.07 | 1.222x | 0.718x | 4.29e-06 |
| attn_kv | k_proj/v_proj | 4 | 2048 | 512 | qvq_lr | 3.5 | P32 | 0.0215 | 0.0222 | 0.390 | 21.54 | 1.211x | 0.699x | 3.81e-06 |
| attn_kv | k_proj/v_proj | 8 | 2048 | 512 | qvq_lr | 3.5 | P32 | 0.0224 | 0.0230 | 0.751 | 20.71 | 1.024x | 0.674x | 3.81e-06 |
| attn_kv | k_proj/v_proj | 16 | 2048 | 512 | qvq_lr | 3.5 | P32 | 0.0248 | 0.0256 | 1.354 | 18.68 | 0.988x | 0.608x | 4.29e-06 |
| attn_kv | k_proj/v_proj | 1 | 2048 | 512 | gptq_marlin | 4 | 128 | 0.0221 | 0.0226 | 0.095 | 24.45 | 1.000x | 0.680x | 0.000167 |
| attn_kv | k_proj/v_proj | 2 | 2048 | 512 | gptq_marlin | 4 | 128 | 0.0256 | 0.0262 | 0.164 | 21.09 | 1.000x | 0.587x | 0.000155 |
| attn_kv | k_proj/v_proj | 4 | 2048 | 512 | gptq_marlin | 4 | 128 | 0.0260 | 0.0264 | 0.322 | 20.78 | 1.000x | 0.577x | 0.000167 |
| attn_kv | k_proj/v_proj | 8 | 2048 | 512 | gptq_marlin | 4 | 128 | 0.0229 | 0.0235 | 0.733 | 23.63 | 1.000x | 0.659x | 0.000206 |
| attn_kv | k_proj/v_proj | 16 | 2048 | 512 | gptq_marlin | 4 | 128 | 0.0245 | 0.0250 | 1.371 | 22.09 | 1.000x | 0.616x | 0.000194 |
| attn_kv | k_proj/v_proj | 1 | 2048 | 512 | gptq_machete | 4 | 128 | 0.0150 | 0.0162 | 0.139 | 36.49 | 1.470x | 1.000x | 0.000167 |
| attn_kv | k_proj/v_proj | 2 | 2048 | 512 | gptq_machete | 4 | 128 | 0.0151 | 0.0161 | 0.279 | 36.45 | 1.702x | 1.000x | 0.000155 |
| attn_kv | k_proj/v_proj | 4 | 2048 | 512 | gptq_machete | 4 | 128 | 0.0150 | 0.0161 | 0.558 | 36.53 | 1.732x | 1.000x | 0.000167 |
| attn_kv | k_proj/v_proj | 8 | 2048 | 512 | gptq_machete | 4 | 128 | 0.0151 | 0.0160 | 1.113 | 36.42 | 1.518x | 1.000x | 0.000206 |
| attn_kv | k_proj/v_proj | 16 | 2048 | 512 | gptq_machete | 4 | 128 | 0.0151 | 0.0162 | 2.226 | 36.42 | 1.624x | 1.000x | 0.000194 |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | qvq_lr | 2 | P32 | 0.0356 | 0.0358 | 0.943 | 119.71 | 0.281x | 0.507x | 5.72e-05 |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | qvq_lr | 2 | P32 | 0.0357 | 0.0360 | 1.881 | 119.39 | 0.288x | 0.496x | 5.15e-05 |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | qvq_lr | 2 | P32 | 0.0358 | 0.0361 | 3.745 | 118.86 | 0.284x | 0.506x | 5.72e-05 |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | qvq_lr | 2 | P32 | 0.0366 | 0.0369 | 7.336 | 116.41 | 0.272x | 0.491x | 7.63e-05 |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | qvq_lr | 2 | P32 | 0.0363 | 0.0367 | 14.782 | 117.29 | 0.300x | 0.494x | 6.1e-05 |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | qvq_lr | 2.5 | P32 | 0.0348 | 0.0352 | 0.965 | 152.61 | 0.287x | 0.519x | 4.86e-05 |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | qvq_lr | 2.5 | P32 | 0.0349 | 0.0352 | 1.922 | 152.05 | 0.294x | 0.507x | 5.53e-05 |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | qvq_lr | 2.5 | P32 | 0.0351 | 0.0355 | 3.823 | 151.22 | 0.290x | 0.517x | 6.29e-05 |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | qvq_lr | 2.5 | P32 | 0.0357 | 0.0361 | 7.510 | 148.51 | 0.279x | 0.502x | 5.34e-05 |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | qvq_lr | 2.5 | P32 | 0.0357 | 0.0361 | 15.047 | 148.78 | 0.306x | 0.503x | 6.87e-05 |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | qvq_lr | 3 | P32 | 0.2120 | 0.2126 | 0.158 | 29.99 | 0.047x | 0.085x | 6.68e-06 |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | qvq_lr | 3 | P32 | 0.2079 | 0.2104 | 0.323 | 30.57 | 0.049x | 0.085x | 4.77e-06 |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | qvq_lr | 3 | P32 | 0.2090 | 0.2124 | 0.642 | 30.42 | 0.049x | 0.087x | 4.89e-06 |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | qvq_lr | 3 | P32 | 0.2115 | 0.2152 | 1.269 | 30.06 | 0.047x | 0.085x | 4.77e-06 |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | qvq_lr | 3 | P32 | 0.0497 | 0.0506 | 10.803 | 127.92 | 0.220x | 0.361x | 3.05e-05 |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | qvq_lr | 3.5 | P32 | 0.2439 | 0.2448 | 0.138 | 30.36 | 0.041x | 0.074x | 6.2e-06 |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | qvq_lr | 3.5 | P32 | 0.2027 | 0.2048 | 0.331 | 36.53 | 0.051x | 0.087x | 4.77e-06 |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | qvq_lr | 3.5 | P32 | 0.2028 | 0.2047 | 0.662 | 36.52 | 0.050x | 0.089x | 5.72e-06 |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | qvq_lr | 3.5 | P32 | 0.2037 | 0.2073 | 1.318 | 36.35 | 0.049x | 0.088x | 5.96e-06 |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | qvq_lr | 3.5 | P32 | 0.2098 | 0.2140 | 2.559 | 35.30 | 0.052x | 0.086x | 5.25e-06 |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | gptq_marlin | 4 | 128 | 0.0100 | 0.0104 | 3.361 | 866.46 | 1.000x | 1.808x | 0.000272 |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | gptq_marlin | 4 | 128 | 0.0103 | 0.0106 | 6.533 | 842.17 | 1.000x | 1.723x | 0.000299 |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | gptq_marlin | 4 | 128 | 0.0102 | 0.0105 | 13.190 | 850.11 | 1.000x | 1.783x | 0.000274 |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | gptq_marlin | 4 | 128 | 0.0100 | 0.0103 | 26.930 | 867.85 | 1.000x | 1.801x | 0.000278 |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | gptq_marlin | 4 | 128 | 0.0109 | 0.0112 | 49.200 | 792.77 | 1.000x | 1.645x | 0.000327 |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | gptq_machete | 4 | 128 | 0.0180 | 0.0190 | 1.859 | 479.77 | 0.553x | 1.000x | 0.000272 |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | gptq_machete | 4 | 128 | 0.0177 | 0.0186 | 3.792 | 489.32 | 0.580x | 1.000x | 0.000299 |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | gptq_machete | 4 | 128 | 0.0181 | 0.0190 | 7.397 | 477.23 | 0.561x | 1.000x | 0.000274 |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | gptq_machete | 4 | 128 | 0.0180 | 0.0188 | 14.953 | 482.34 | 0.555x | 1.000x | 0.000278 |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | gptq_machete | 4 | 128 | 0.0180 | 0.0191 | 29.906 | 482.34 | 0.608x | 1.000x | 0.000327 |
| mlp_down | down_proj | 1 | 8192 | 2048 | qvq_lr | 2 | P32 | 0.0314 | 0.0322 | 1.068 | 135.63 | 0.518x | 0.702x | 4.2e-05 |
| mlp_down | down_proj | 2 | 8192 | 2048 | qvq_lr | 2 | P32 | 0.0375 | 0.0382 | 1.789 | 113.58 | 0.480x | 0.592x | 9.92e-05 |
| mlp_down | down_proj | 4 | 8192 | 2048 | qvq_lr | 2 | P32 | 0.0380 | 0.0384 | 3.532 | 112.10 | 0.475x | 0.578x | 8.39e-05 |
| mlp_down | down_proj | 8 | 8192 | 2048 | qvq_lr | 2 | P32 | 0.0380 | 0.0387 | 7.055 | 111.96 | 0.433x | 0.582x | 9.16e-05 |
| mlp_down | down_proj | 16 | 8192 | 2048 | qvq_lr | 2 | P32 | 0.0381 | 0.0388 | 14.104 | 111.91 | 0.472x | 0.582x | 0.000101 |
| mlp_down | down_proj | 1 | 8192 | 2048 | qvq_lr | 2.5 | P32 | 0.0320 | 0.0325 | 1.048 | 165.72 | 0.507x | 0.689x | 4.2e-05 |
| mlp_down | down_proj | 2 | 8192 | 2048 | qvq_lr | 2.5 | P32 | 0.0373 | 0.0379 | 1.799 | 142.33 | 0.482x | 0.595x | 9.92e-05 |
| mlp_down | down_proj | 4 | 8192 | 2048 | qvq_lr | 2.5 | P32 | 0.0373 | 0.0380 | 3.594 | 142.15 | 0.483x | 0.588x | 9.92e-05 |
| mlp_down | down_proj | 8 | 8192 | 2048 | qvq_lr | 2.5 | P32 | 0.0377 | 0.0382 | 7.127 | 140.94 | 0.438x | 0.588x | 0.000113 |
| mlp_down | down_proj | 16 | 8192 | 2048 | qvq_lr | 2.5 | P32 | 0.0379 | 0.0386 | 14.170 | 140.11 | 0.475x | 0.584x | 0.000103 |
| mlp_down | down_proj | 1 | 8192 | 2048 | qvq_lr | 3 | P32 | 0.2082 | 0.2111 | 0.161 | 30.53 | 0.078x | 0.106x | 1.41e-05 |
| mlp_down | down_proj | 2 | 8192 | 2048 | qvq_lr | 3 | P32 | 0.2080 | 0.2099 | 0.323 | 30.56 | 0.086x | 0.107x | 1.14e-05 |
| mlp_down | down_proj | 4 | 8192 | 2048 | qvq_lr | 3 | P32 | 0.2092 | 0.2117 | 0.641 | 30.38 | 0.086x | 0.105x | 1.34e-05 |
| mlp_down | down_proj | 8 | 8192 | 2048 | qvq_lr | 3 | P32 | 0.2125 | 0.2152 | 1.263 | 29.91 | 0.078x | 0.104x | 1.86e-05 |
| mlp_down | down_proj | 16 | 8192 | 2048 | qvq_lr | 3 | P32 | 0.0593 | 0.0599 | 9.057 | 107.24 | 0.303x | 0.374x | 0.000122 |
| mlp_down | down_proj | 1 | 8192 | 2048 | qvq_lr | 3.5 | P32 | 0.2312 | 0.2334 | 0.145 | 32.02 | 0.070x | 0.095x | 1.14e-05 |
| mlp_down | down_proj | 2 | 8192 | 2048 | qvq_lr | 3.5 | P32 | 0.2050 | 0.2081 | 0.327 | 36.13 | 0.088x | 0.108x | 1.24e-05 |
| mlp_down | down_proj | 4 | 8192 | 2048 | qvq_lr | 3.5 | P32 | 0.2066 | 0.2104 | 0.650 | 35.84 | 0.087x | 0.106x | 1.34e-05 |
| mlp_down | down_proj | 8 | 8192 | 2048 | qvq_lr | 3.5 | P32 | 0.2072 | 0.2118 | 1.295 | 35.73 | 0.080x | 0.107x | 1.72e-05 |
| mlp_down | down_proj | 16 | 8192 | 2048 | qvq_lr | 3.5 | P32 | 0.2115 | 0.2140 | 2.539 | 35.02 | 0.085x | 0.105x | 1.34e-05 |
| mlp_down | down_proj | 1 | 8192 | 2048 | gptq_marlin | 4 | 128 | 0.0163 | 0.0166 | 2.064 | 532.16 | 1.000x | 1.357x | 0.000346 |
| mlp_down | down_proj | 2 | 8192 | 2048 | gptq_marlin | 4 | 128 | 0.0180 | 0.0184 | 3.732 | 481.02 | 1.000x | 1.235x | 0.000545 |
| mlp_down | down_proj | 4 | 8192 | 2048 | gptq_marlin | 4 | 128 | 0.0180 | 0.0184 | 7.443 | 479.74 | 1.000x | 1.218x | 0.000622 |
| mlp_down | down_proj | 8 | 8192 | 2048 | gptq_marlin | 4 | 128 | 0.0165 | 0.0170 | 16.289 | 524.92 | 1.000x | 1.344x | 0.000546 |
| mlp_down | down_proj | 16 | 8192 | 2048 | gptq_marlin | 4 | 128 | 0.0180 | 0.0183 | 29.853 | 481.02 | 1.000x | 1.231x | 0.000539 |
| mlp_down | down_proj | 1 | 8192 | 2048 | gptq_machete | 4 | 128 | 0.0221 | 0.0232 | 1.521 | 393.56 | 0.737x | 1.000x | 0.000346 |
| mlp_down | down_proj | 2 | 8192 | 2048 | gptq_machete | 4 | 128 | 0.0222 | 0.0229 | 3.022 | 391.01 | 0.810x | 1.000x | 0.000545 |
| mlp_down | down_proj | 4 | 8192 | 2048 | gptq_machete | 4 | 128 | 0.0220 | 0.0228 | 6.110 | 395.28 | 0.821x | 1.000x | 0.000622 |
| mlp_down | down_proj | 8 | 8192 | 2048 | gptq_machete | 4 | 128 | 0.0221 | 0.0229 | 12.122 | 392.14 | 0.744x | 1.000x | 0.000546 |
| mlp_down | down_proj | 16 | 8192 | 2048 | gptq_machete | 4 | 128 | 0.0221 | 0.0231 | 24.245 | 392.14 | 0.812x | 1.000x | 0.000539 |

The four shape rows preserve all seven Llama 3.2 1B projection roles: `q_proj/o_proj` (2048×2048), `k_proj/v_proj` (2048×512), `gate_proj/up_proj` (2048×8192), and `down_proj` (8192×2048).
