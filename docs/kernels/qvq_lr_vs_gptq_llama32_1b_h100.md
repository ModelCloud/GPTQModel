# QVQ LR versus W4 GPTQ kernels — Llama 3.2 1B shapes

This is a kernel-throughput comparison on the H100 host from PR #62. W4 GPTQ is a figurative performance baseline, not a quality-equivalent arm.

## Measurement contract

- Commit: `6bb7d21edc4a4fc93ac456fd0d7917e893f35ca7`; benchmark SHA256: `e0f85cbae361e0e1ac1258ce42be3487db9dbafdad061b85fc03e16a8f85c1d7`
- GPU: physical `1`, `NVIDIA H100`, PCI `00000000:44:00.0`, UUID `GPU-f5ea03cf-efa4-9807-7de5-b174957a1348`, CC `9.0`, 132 SMs
- Torch/CUDA: `2.15.0.dev20260828+cu130` / `13.0`; input dtype: `float16`
- M values: `[1, 2, 4, 8, 16, 32]`; warmup: `10`; measured launches: `60`
- QVQ: `qvq_v2b2_p32_lr`, rates `[2.0, 2.5, 3.0, 3.5]`, native `P32 is LR packing geometry, not a GPTQ affine scale group.`
- GPTQ: symmetric W4, group `128`, no activation order; Marlin and Machete use the same W4 source payload
- Latency is CUDA-event median/P95 from one CUDA Graph replay containing every measured launch. CPU scheduling and host launch gaps are outside each timed interval. Logical TFLOP/s is `2*M*K*N / median_ms`; payload GB/s is packed payload bytes divided by median latency.
- Every row passed its dense-reference shape/dtype/finite gate. QVQ uses `max_abs <= 2e-3`; GPTQ uses `atol=2e-2, rtol=2e-2`.

`xMarlin` and `xMachete` are baseline median latency divided by the row median, so values above 1.0x are faster.

| Shape | Roles | M | K | N | Kernel | W | Group | Median ms | P95 ms | Logical TFLOP/s | Payload GB/s | xMarlin | xMachete | Max abs |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| attn_qo | q_proj/o_proj | 1 | 2048 | 2048 | qvq_lr | 2 | P32 | 0.1381 | 0.1399 | 0.061 | 7.71 | 0.105x | 0.111x | 3.93e-06 |
| attn_qo | q_proj/o_proj | 2 | 2048 | 2048 | qvq_lr | 2 | P32 | 0.0308 | 0.0333 | 0.544 | 34.56 | 0.524x | 0.507x | 4.77e-06 |
| attn_qo | q_proj/o_proj | 4 | 2048 | 2048 | qvq_lr | 2 | P32 | 0.0300 | 0.0307 | 1.117 | 35.44 | 0.542x | 0.517x | 3.81e-06 |
| attn_qo | q_proj/o_proj | 8 | 2048 | 2048 | qvq_lr | 2 | P32 | 0.0315 | 0.0324 | 2.128 | 33.77 | 0.466x | 0.494x | 5.96e-06 |
| attn_qo | q_proj/o_proj | 16 | 2048 | 2048 | qvq_lr | 2 | P32 | 0.0435 | 0.0452 | 3.089 | 24.51 | 0.361x | 0.353x | 5.72e-06 |
| attn_qo | q_proj/o_proj | 32 | 2048 | 2048 | qvq_lr | 2 | P32 | 0.1379 | 0.1413 | 1.946 | 7.72 | 0.248x | 0.116x | 4.77e-06 |
| attn_qo | q_proj/o_proj | 1 | 2048 | 2048 | qvq_lr | 2.5 | P32 | 0.1204 | 0.1218 | 0.070 | 11.02 | 0.121x | 0.127x | 4.29e-06 |
| attn_qo | q_proj/o_proj | 2 | 2048 | 2048 | qvq_lr | 2.5 | P32 | 0.1209 | 0.1218 | 0.139 | 10.98 | 0.134x | 0.129x | 4.98e-06 |
| attn_qo | q_proj/o_proj | 4 | 2048 | 2048 | qvq_lr | 2.5 | P32 | 0.1217 | 0.1244 | 0.276 | 10.91 | 0.134x | 0.128x | 4.77e-06 |
| attn_qo | q_proj/o_proj | 8 | 2048 | 2048 | qvq_lr | 2.5 | P32 | 0.1240 | 0.1263 | 0.541 | 10.70 | 0.119x | 0.126x | 5.72e-06 |
| attn_qo | q_proj/o_proj | 16 | 2048 | 2048 | qvq_lr | 2.5 | P32 | 0.0456 | 0.0473 | 2.944 | 29.11 | 0.344x | 0.337x | 5.3e-06 |
| attn_qo | q_proj/o_proj | 32 | 2048 | 2048 | qvq_lr | 2.5 | P32 | 0.1359 | 0.1380 | 1.976 | 9.77 | 0.252x | 0.117x | 5.72e-06 |
| attn_qo | q_proj/o_proj | 1 | 2048 | 2048 | qvq_lr | 3 | P32 | 0.0568 | 0.0589 | 0.148 | 28.00 | 0.256x | 0.270x | 3.34e-06 |
| attn_qo | q_proj/o_proj | 2 | 2048 | 2048 | qvq_lr | 3 | P32 | 0.0576 | 0.0598 | 0.291 | 27.58 | 0.280x | 0.271x | 4.77e-06 |
| attn_qo | q_proj/o_proj | 4 | 2048 | 2048 | qvq_lr | 3 | P32 | 0.0586 | 0.0606 | 0.572 | 27.10 | 0.278x | 0.265x | 4.77e-06 |
| attn_qo | q_proj/o_proj | 8 | 2048 | 2048 | qvq_lr | 3 | P32 | 0.0595 | 0.0621 | 1.128 | 26.71 | 0.247x | 0.262x | 5.72e-06 |
| attn_qo | q_proj/o_proj | 16 | 2048 | 2048 | qvq_lr | 3 | P32 | 0.0258 | 0.0282 | 5.201 | 61.58 | 0.608x | 0.595x | 1.34e-05 |
| attn_qo | q_proj/o_proj | 32 | 2048 | 2048 | qvq_lr | 3 | P32 | 0.0749 | 0.0771 | 3.583 | 21.21 | 0.457x | 0.213x | 4.77e-06 |
| attn_qo | q_proj/o_proj | 1 | 2048 | 2048 | qvq_lr | 3.5 | P32 | 0.0653 | 0.0709 | 0.128 | 28.35 | 0.222x | 0.235x | 4.77e-06 |
| attn_qo | q_proj/o_proj | 2 | 2048 | 2048 | qvq_lr | 3.5 | P32 | 0.0566 | 0.0585 | 0.296 | 32.70 | 0.285x | 0.276x | 3.81e-06 |
| attn_qo | q_proj/o_proj | 4 | 2048 | 2048 | qvq_lr | 3.5 | P32 | 0.0562 | 0.0569 | 0.597 | 32.94 | 0.290x | 0.276x | 4.77e-06 |
| attn_qo | q_proj/o_proj | 8 | 2048 | 2048 | qvq_lr | 3.5 | P32 | 0.0572 | 0.0583 | 1.173 | 32.35 | 0.257x | 0.272x | 4.77e-06 |
| attn_qo | q_proj/o_proj | 16 | 2048 | 2048 | qvq_lr | 3.5 | P32 | 0.0638 | 0.0658 | 2.104 | 29.02 | 0.246x | 0.241x | 4.77e-06 |
| attn_qo | q_proj/o_proj | 32 | 2048 | 2048 | qvq_lr | 3.5 | P32 | 0.0756 | 0.0779 | 3.549 | 24.48 | 0.452x | 0.211x | 4.77e-06 |
| attn_qo | q_proj/o_proj | 1 | 2048 | 2048 | gptq_marlin | 4 | 128 | 0.0145 | 0.0155 | 0.577 | 148.86 | 1.000x | 1.055x | 0.000264 |
| attn_qo | q_proj/o_proj | 2 | 2048 | 2048 | gptq_marlin | 4 | 128 | 0.0162 | 0.0165 | 1.038 | 133.83 | 1.000x | 0.966x | 0.000229 |
| attn_qo | q_proj/o_proj | 4 | 2048 | 2048 | gptq_marlin | 4 | 128 | 0.0163 | 0.0168 | 2.060 | 132.78 | 1.000x | 0.953x | 0.000197 |
| attn_qo | q_proj/o_proj | 8 | 2048 | 2048 | gptq_marlin | 4 | 128 | 0.0147 | 0.0151 | 4.564 | 147.08 | 1.000x | 1.060x | 0.000242 |
| attn_qo | q_proj/o_proj | 16 | 2048 | 2048 | gptq_marlin | 4 | 128 | 0.0157 | 0.0160 | 8.560 | 137.93 | 1.000x | 0.980x | 0.00028 |
| attn_qo | q_proj/o_proj | 32 | 2048 | 2048 | gptq_marlin | 4 | 128 | 0.0342 | 0.0348 | 7.847 | 63.22 | 1.000x | 0.466x | 0.000269 |
| attn_qo | q_proj/o_proj | 1 | 2048 | 2048 | gptq_machete | 4 | 128 | 0.0153 | 0.0163 | 0.547 | 141.63 | 0.948x | 1.000x | 0.000264 |
| attn_qo | q_proj/o_proj | 2 | 2048 | 2048 | gptq_machete | 4 | 128 | 0.0156 | 0.0164 | 1.074 | 139.02 | 1.035x | 1.000x | 0.000229 |
| attn_qo | q_proj/o_proj | 4 | 2048 | 2048 | gptq_machete | 4 | 128 | 0.0155 | 0.0163 | 2.162 | 139.88 | 1.049x | 1.000x | 0.000197 |
| attn_qo | q_proj/o_proj | 8 | 2048 | 2048 | gptq_machete | 4 | 128 | 0.0156 | 0.0164 | 4.306 | 139.30 | 0.944x | 1.000x | 0.000242 |
| attn_qo | q_proj/o_proj | 16 | 2048 | 2048 | gptq_machete | 4 | 128 | 0.0154 | 0.0163 | 8.738 | 141.33 | 1.021x | 1.000x | 0.00028 |
| attn_qo | q_proj/o_proj | 32 | 2048 | 2048 | gptq_machete | 4 | 128 | 0.0160 | 0.0169 | 16.828 | 136.09 | 2.144x | 1.000x | 0.000269 |
| attn_kv | k_proj/v_proj | 1 | 2048 | 512 | qvq_lr | 2 | P32 | 0.0451 | 0.0458 | 0.047 | 5.90 | 0.481x | 0.328x | 3.34e-06 |
| attn_kv | k_proj/v_proj | 2 | 2048 | 512 | qvq_lr | 2 | P32 | 0.0131 | 0.0138 | 0.319 | 20.27 | 1.988x | 1.147x | 2.86e-06 |
| attn_kv | k_proj/v_proj | 4 | 2048 | 512 | qvq_lr | 2 | P32 | 0.0136 | 0.0143 | 0.616 | 19.55 | 1.904x | 1.116x | 4.77e-06 |
| attn_kv | k_proj/v_proj | 8 | 2048 | 512 | qvq_lr | 2 | P32 | 0.0145 | 0.0152 | 1.157 | 18.37 | 1.596x | 1.023x | 4.77e-06 |
| attn_kv | k_proj/v_proj | 16 | 2048 | 512 | qvq_lr | 2 | P32 | 0.0186 | 0.0194 | 1.800 | 14.28 | 1.297x | 0.814x | 4.29e-06 |
| attn_kv | k_proj/v_proj | 32 | 2048 | 512 | qvq_lr | 2 | P32 | 0.0473 | 0.0482 | 1.420 | 5.63 | 0.709x | 0.328x | 5.72e-06 |
| attn_kv | k_proj/v_proj | 1 | 2048 | 512 | qvq_lr | 2.5 | P32 | 0.0411 | 0.0415 | 0.051 | 8.08 | 0.528x | 0.360x | 2.38e-06 |
| attn_kv | k_proj/v_proj | 2 | 2048 | 512 | qvq_lr | 2.5 | P32 | 0.0372 | 0.0389 | 0.113 | 8.91 | 0.701x | 0.405x | 3.81e-06 |
| attn_kv | k_proj/v_proj | 4 | 2048 | 512 | qvq_lr | 2.5 | P32 | 0.0376 | 0.0385 | 0.223 | 8.82 | 0.689x | 0.404x | 3.81e-06 |
| attn_kv | k_proj/v_proj | 8 | 2048 | 512 | qvq_lr | 2.5 | P32 | 0.0382 | 0.0390 | 0.440 | 8.69 | 0.606x | 0.389x | 3.81e-06 |
| attn_kv | k_proj/v_proj | 16 | 2048 | 512 | qvq_lr | 2.5 | P32 | 0.0190 | 0.0196 | 1.767 | 17.47 | 1.273x | 0.799x | 4.77e-06 |
| attn_kv | k_proj/v_proj | 32 | 2048 | 512 | qvq_lr | 2.5 | P32 | 0.0465 | 0.0479 | 1.442 | 7.13 | 0.719x | 0.333x | 5.25e-06 |
| attn_kv | k_proj/v_proj | 1 | 2048 | 512 | qvq_lr | 3 | P32 | 0.0208 | 0.0214 | 0.101 | 19.07 | 1.041x | 0.710x | 1.91e-06 |
| attn_kv | k_proj/v_proj | 2 | 2048 | 512 | qvq_lr | 3 | P32 | 0.0200 | 0.0206 | 0.210 | 19.91 | 1.309x | 0.755x | 3.81e-06 |
| attn_kv | k_proj/v_proj | 4 | 2048 | 512 | qvq_lr | 3 | P32 | 0.0203 | 0.0210 | 0.413 | 19.55 | 1.276x | 0.748x | 4.77e-06 |
| attn_kv | k_proj/v_proj | 8 | 2048 | 512 | qvq_lr | 3 | P32 | 0.0212 | 0.0218 | 0.792 | 18.76 | 1.092x | 0.700x | 4.77e-06 |
| attn_kv | k_proj/v_proj | 16 | 2048 | 512 | qvq_lr | 3 | P32 | 0.0238 | 0.0245 | 1.407 | 16.67 | 1.014x | 0.636x | 3.81e-06 |
| attn_kv | k_proj/v_proj | 32 | 2048 | 512 | qvq_lr | 3 | P32 | 0.0302 | 0.0323 | 2.219 | 13.14 | 1.107x | 0.513x | 4.77e-06 |
| attn_kv | k_proj/v_proj | 1 | 2048 | 512 | qvq_lr | 3.5 | P32 | 0.0222 | 0.0228 | 0.095 | 20.87 | 0.978x | 0.667x | 2.38e-06 |
| attn_kv | k_proj/v_proj | 2 | 2048 | 512 | qvq_lr | 3.5 | P32 | 0.0204 | 0.0209 | 0.206 | 22.71 | 1.281x | 0.739x | 4.29e-06 |
| attn_kv | k_proj/v_proj | 4 | 2048 | 512 | qvq_lr | 3.5 | P32 | 0.0208 | 0.0213 | 0.404 | 22.29 | 1.248x | 0.732x | 3.81e-06 |
| attn_kv | k_proj/v_proj | 8 | 2048 | 512 | qvq_lr | 3.5 | P32 | 0.0217 | 0.0223 | 0.774 | 21.36 | 1.068x | 0.685x | 3.81e-06 |
| attn_kv | k_proj/v_proj | 16 | 2048 | 512 | qvq_lr | 3.5 | P32 | 0.0240 | 0.0247 | 1.401 | 19.32 | 1.009x | 0.633x | 4.29e-06 |
| attn_kv | k_proj/v_proj | 32 | 2048 | 512 | qvq_lr | 3.5 | P32 | 0.0302 | 0.0309 | 2.220 | 15.31 | 1.108x | 0.513x | 4.77e-06 |
| attn_kv | k_proj/v_proj | 1 | 2048 | 512 | gptq_marlin | 4 | 128 | 0.0217 | 0.0222 | 0.097 | 24.92 | 1.000x | 0.682x | 0.000167 |
| attn_kv | k_proj/v_proj | 2 | 2048 | 512 | gptq_marlin | 4 | 128 | 0.0261 | 0.0267 | 0.161 | 20.71 | 1.000x | 0.577x | 0.000155 |
| attn_kv | k_proj/v_proj | 4 | 2048 | 512 | gptq_marlin | 4 | 128 | 0.0259 | 0.0265 | 0.324 | 20.86 | 1.000x | 0.586x | 0.000167 |
| attn_kv | k_proj/v_proj | 8 | 2048 | 512 | gptq_marlin | 4 | 128 | 0.0231 | 0.0236 | 0.725 | 23.37 | 1.000x | 0.641x | 0.000206 |
| attn_kv | k_proj/v_proj | 16 | 2048 | 512 | gptq_marlin | 4 | 128 | 0.0242 | 0.0248 | 1.388 | 22.36 | 1.000x | 0.627x | 0.000194 |
| attn_kv | k_proj/v_proj | 32 | 2048 | 512 | gptq_marlin | 4 | 128 | 0.0335 | 0.0338 | 2.004 | 16.15 | 1.000x | 0.463x | 0.000265 |
| attn_kv | k_proj/v_proj | 1 | 2048 | 512 | gptq_machete | 4 | 128 | 0.0148 | 0.0157 | 0.142 | 37.09 | 1.466x | 1.000x | 0.000167 |
| attn_kv | k_proj/v_proj | 2 | 2048 | 512 | gptq_machete | 4 | 128 | 0.0151 | 0.0160 | 0.278 | 36.42 | 1.732x | 1.000x | 0.000155 |
| attn_kv | k_proj/v_proj | 4 | 2048 | 512 | gptq_machete | 4 | 128 | 0.0152 | 0.0159 | 0.552 | 36.11 | 1.705x | 1.000x | 0.000167 |
| attn_kv | k_proj/v_proj | 8 | 2048 | 512 | gptq_machete | 4 | 128 | 0.0148 | 0.0155 | 1.131 | 37.01 | 1.560x | 1.000x | 0.000206 |
| attn_kv | k_proj/v_proj | 16 | 2048 | 512 | gptq_machete | 4 | 128 | 0.0152 | 0.0158 | 2.212 | 36.19 | 1.594x | 1.000x | 0.000194 |
| attn_kv | k_proj/v_proj | 32 | 2048 | 512 | gptq_machete | 4 | 128 | 0.0155 | 0.0162 | 4.328 | 35.40 | 2.160x | 1.000x | 0.000265 |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | qvq_lr | 2 | P32 | 0.4813 | 0.4828 | 0.070 | 8.85 | 0.020x | 0.037x | 7.15e-06 |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | qvq_lr | 2 | P32 | 0.0882 | 0.0886 | 0.761 | 48.30 | 0.116x | 0.205x | 4.77e-06 |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | qvq_lr | 2 | P32 | 0.0889 | 0.0892 | 1.509 | 47.90 | 0.115x | 0.201x | 5.01e-06 |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | qvq_lr | 2 | P32 | 0.0903 | 0.0907 | 2.973 | 47.17 | 0.111x | 0.198x | 5.72e-06 |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | qvq_lr | 2 | P32 | 0.1296 | 0.1300 | 4.144 | 32.88 | 0.084x | 0.138x | 5.72e-06 |
| mlp_gate_up | gate_proj/up_proj | 32 | 2048 | 8192 | qvq_lr | 2 | P32 | 0.4922 | 0.4950 | 2.181 | 8.65 | 0.032x | 0.039x | 1.19e-05 |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | qvq_lr | 2.5 | P32 | 0.4844 | 0.4854 | 0.069 | 10.96 | 0.020x | 0.037x | 6.08e-06 |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | qvq_lr | 2.5 | P32 | 0.4557 | 0.4596 | 0.147 | 11.65 | 0.022x | 0.040x | 4.77e-06 |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | qvq_lr | 2.5 | P32 | 0.4557 | 0.4605 | 0.295 | 11.65 | 0.023x | 0.039x | 4.77e-06 |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | qvq_lr | 2.5 | P32 | 0.4570 | 0.4619 | 0.587 | 11.61 | 0.022x | 0.039x | 4.89e-06 |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | qvq_lr | 2.5 | P32 | 0.1309 | 0.1314 | 4.103 | 40.56 | 0.083x | 0.137x | 5.72e-06 |
| mlp_gate_up | gate_proj/up_proj | 32 | 2048 | 8192 | qvq_lr | 2.5 | P32 | 0.4932 | 0.4962 | 2.177 | 10.76 | 0.032x | 0.039x | 1.34e-05 |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | qvq_lr | 3 | P32 | 0.2059 | 0.2064 | 0.163 | 30.88 | 0.048x | 0.088x | 6.68e-06 |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | qvq_lr | 3 | P32 | 0.1927 | 0.1953 | 0.348 | 32.99 | 0.053x | 0.094x | 4.77e-06 |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | qvq_lr | 3 | P32 | 0.1943 | 0.1974 | 0.691 | 32.72 | 0.053x | 0.092x | 4.89e-06 |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | qvq_lr | 3 | P32 | 0.1957 | 0.1997 | 1.372 | 32.48 | 0.051x | 0.091x | 4.77e-06 |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | qvq_lr | 3 | P32 | 0.0673 | 0.0677 | 7.978 | 94.46 | 0.161x | 0.266x | 5.91e-05 |
| mlp_gate_up | gate_proj/up_proj | 32 | 2048 | 8192 | qvq_lr | 3 | P32 | 0.2227 | 0.2241 | 4.821 | 28.54 | 0.072x | 0.086x | 1.34e-05 |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | qvq_lr | 3.5 | P32 | 0.2362 | 0.2371 | 0.142 | 31.35 | 0.042x | 0.076x | 6.2e-06 |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | qvq_lr | 3.5 | P32 | 0.1966 | 0.1990 | 0.341 | 37.68 | 0.052x | 0.092x | 4.77e-06 |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | qvq_lr | 3.5 | P32 | 0.1984 | 0.2007 | 0.676 | 37.32 | 0.052x | 0.090x | 5.72e-06 |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | qvq_lr | 3.5 | P32 | 0.2004 | 0.2021 | 1.339 | 36.95 | 0.050x | 0.089x | 5.96e-06 |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | qvq_lr | 3.5 | P32 | 0.2056 | 0.2072 | 2.611 | 36.02 | 0.053x | 0.087x | 5.25e-06 |
| mlp_gate_up | gate_proj/up_proj | 32 | 2048 | 8192 | qvq_lr | 3.5 | P32 | 0.2271 | 0.2283 | 4.729 | 32.62 | 0.070x | 0.084x | 1.36e-05 |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | gptq_marlin | 4 | 128 | 0.0099 | 0.0104 | 3.404 | 877.71 | 1.000x | 1.828x | 0.000272 |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | gptq_marlin | 4 | 128 | 0.0102 | 0.0105 | 6.554 | 844.80 | 1.000x | 1.769x | 0.000299 |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | gptq_marlin | 4 | 128 | 0.0103 | 0.0107 | 13.087 | 843.48 | 1.000x | 1.741x | 0.000274 |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | gptq_marlin | 4 | 128 | 0.0100 | 0.0104 | 26.887 | 866.46 | 1.000x | 1.792x | 0.000278 |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | gptq_marlin | 4 | 128 | 0.0108 | 0.0113 | 49.490 | 797.45 | 1.000x | 1.652x | 0.000327 |
| mlp_gate_up | gate_proj/up_proj | 32 | 2048 | 8192 | gptq_marlin | 4 | 128 | 0.0159 | 0.0163 | 67.378 | 542.84 | 1.000x | 1.197x | 0.000315 |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | gptq_machete | 4 | 128 | 0.0180 | 0.0189 | 1.862 | 480.63 | 0.547x | 1.000x | 0.000272 |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | gptq_machete | 4 | 128 | 0.0181 | 0.0191 | 3.705 | 478.08 | 0.565x | 1.000x | 0.000299 |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | gptq_machete | 4 | 128 | 0.0179 | 0.0186 | 7.517 | 484.93 | 0.574x | 1.000x | 0.000274 |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | gptq_machete | 4 | 128 | 0.0179 | 0.0190 | 15.006 | 484.06 | 0.558x | 1.000x | 0.000278 |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | gptq_machete | 4 | 128 | 0.0179 | 0.0189 | 29.959 | 483.20 | 0.605x | 1.000x | 0.000327 |
| mlp_gate_up | gate_proj/up_proj | 32 | 2048 | 8192 | gptq_machete | 4 | 128 | 0.0191 | 0.0200 | 56.299 | 454.01 | 0.836x | 1.000x | 0.000315 |
| mlp_down | down_proj | 1 | 8192 | 2048 | qvq_lr | 2 | P32 | 0.5184 | 0.5218 | 0.065 | 8.22 | 0.036x | 0.049x | 1.34e-05 |
| mlp_down | down_proj | 2 | 8192 | 2048 | qvq_lr | 2 | P32 | 0.0914 | 0.0920 | 0.734 | 46.60 | 0.224x | 0.274x | 9.54e-06 |
| mlp_down | down_proj | 4 | 8192 | 2048 | qvq_lr | 2 | P32 | 0.0945 | 0.0981 | 1.420 | 45.07 | 0.202x | 0.239x | 1.05e-05 |
| mlp_down | down_proj | 8 | 8192 | 2048 | qvq_lr | 2 | P32 | 0.0962 | 0.0988 | 2.790 | 44.28 | 0.192x | 0.277x | 1.67e-05 |
| mlp_down | down_proj | 16 | 8192 | 2048 | qvq_lr | 2 | P32 | 0.1348 | 0.1380 | 3.983 | 31.60 | 0.143x | 0.188x | 1.34e-05 |
| mlp_down | down_proj | 32 | 8192 | 2048 | qvq_lr | 2 | P32 | 0.4968 | 0.5017 | 2.161 | 8.57 | 0.079x | 0.050x | 1.19e-05 |
| mlp_down | down_proj | 1 | 8192 | 2048 | qvq_lr | 2.5 | P32 | 0.4566 | 0.4602 | 0.073 | 11.63 | 0.041x | 0.055x | 1.25e-05 |
| mlp_down | down_proj | 2 | 8192 | 2048 | qvq_lr | 2.5 | P32 | 0.4595 | 0.4655 | 0.146 | 11.55 | 0.044x | 0.054x | 9.54e-06 |
| mlp_down | down_proj | 4 | 8192 | 2048 | qvq_lr | 2.5 | P32 | 0.4594 | 0.4655 | 0.292 | 11.56 | 0.042x | 0.049x | 1.14e-05 |
| mlp_down | down_proj | 8 | 8192 | 2048 | qvq_lr | 2.5 | P32 | 0.4595 | 0.4653 | 0.584 | 11.55 | 0.040x | 0.058x | 1.72e-05 |
| mlp_down | down_proj | 16 | 8192 | 2048 | qvq_lr | 2.5 | P32 | 0.1388 | 0.1418 | 3.868 | 38.25 | 0.139x | 0.183x | 1.34e-05 |
| mlp_down | down_proj | 32 | 8192 | 2048 | qvq_lr | 2.5 | P32 | 0.4982 | 0.5036 | 2.155 | 10.66 | 0.079x | 0.050x | 1.34e-05 |
| mlp_down | down_proj | 1 | 8192 | 2048 | qvq_lr | 3 | P32 | 0.1992 | 0.2010 | 0.168 | 31.91 | 0.093x | 0.127x | 1.41e-05 |
| mlp_down | down_proj | 2 | 8192 | 2048 | qvq_lr | 3 | P32 | 0.1972 | 0.1992 | 0.340 | 32.23 | 0.104x | 0.127x | 1.14e-05 |
| mlp_down | down_proj | 4 | 8192 | 2048 | qvq_lr | 3 | P32 | 0.1974 | 0.2008 | 0.680 | 32.20 | 0.097x | 0.114x | 1.34e-05 |
| mlp_down | down_proj | 8 | 8192 | 2048 | qvq_lr | 3 | P32 | 0.2017 | 0.2042 | 1.331 | 31.52 | 0.092x | 0.132x | 1.86e-05 |
| mlp_down | down_proj | 16 | 8192 | 2048 | qvq_lr | 3 | P32 | 0.0723 | 0.0751 | 7.427 | 87.94 | 0.266x | 0.351x | 0.000122 |
| mlp_down | down_proj | 32 | 8192 | 2048 | qvq_lr | 3 | P32 | 0.2284 | 0.2297 | 4.701 | 27.83 | 0.173x | 0.109x | 1.24e-05 |
| mlp_down | down_proj | 1 | 8192 | 2048 | qvq_lr | 3.5 | P32 | 0.2240 | 0.2257 | 0.150 | 33.06 | 0.083x | 0.113x | 1.14e-05 |
| mlp_down | down_proj | 2 | 8192 | 2048 | qvq_lr | 3.5 | P32 | 0.2005 | 0.2038 | 0.335 | 36.93 | 0.102x | 0.125x | 1.24e-05 |
| mlp_down | down_proj | 4 | 8192 | 2048 | qvq_lr | 3.5 | P32 | 0.2025 | 0.2053 | 0.663 | 36.57 | 0.094x | 0.112x | 1.34e-05 |
| mlp_down | down_proj | 8 | 8192 | 2048 | qvq_lr | 3.5 | P32 | 0.2042 | 0.2076 | 1.314 | 36.26 | 0.091x | 0.130x | 1.72e-05 |
| mlp_down | down_proj | 16 | 8192 | 2048 | qvq_lr | 3.5 | P32 | 0.2085 | 0.2116 | 2.575 | 35.51 | 0.092x | 0.122x | 1.34e-05 |
| mlp_down | down_proj | 32 | 8192 | 2048 | qvq_lr | 3.5 | P32 | 0.2330 | 0.2389 | 4.609 | 31.79 | 0.169x | 0.107x | 1.29e-05 |
| mlp_down | down_proj | 1 | 8192 | 2048 | gptq_marlin | 4 | 128 | 0.0186 | 0.0205 | 1.803 | 464.89 | 1.000x | 1.357x | 0.000346 |
| mlp_down | down_proj | 2 | 8192 | 2048 | gptq_marlin | 4 | 128 | 0.0204 | 0.0231 | 3.284 | 423.39 | 1.000x | 1.225x | 0.000545 |
| mlp_down | down_proj | 4 | 8192 | 2048 | gptq_marlin | 4 | 128 | 0.0191 | 0.0219 | 7.026 | 452.82 | 1.000x | 1.183x | 0.000622 |
| mlp_down | down_proj | 8 | 8192 | 2048 | gptq_marlin | 4 | 128 | 0.0185 | 0.0210 | 14.513 | 467.71 | 1.000x | 1.439x | 0.000546 |
| mlp_down | down_proj | 16 | 8192 | 2048 | gptq_marlin | 4 | 128 | 0.0193 | 0.0204 | 27.869 | 449.06 | 1.000x | 1.316x | 0.000539 |
| mlp_down | down_proj | 32 | 8192 | 2048 | gptq_marlin | 4 | 128 | 0.0394 | 0.0417 | 27.225 | 219.34 | 1.000x | 0.630x | 0.00057 |
| mlp_down | down_proj | 1 | 8192 | 2048 | gptq_machete | 4 | 128 | 0.0252 | 0.0276 | 1.329 | 343.93 | 0.737x | 1.000x | 0.000346 |
| mlp_down | down_proj | 2 | 8192 | 2048 | gptq_machete | 4 | 128 | 0.0250 | 0.0271 | 2.682 | 347.01 | 0.816x | 1.000x | 0.000545 |
| mlp_down | down_proj | 4 | 8192 | 2048 | gptq_machete | 4 | 128 | 0.0226 | 0.0237 | 5.941 | 384.36 | 0.846x | 1.000x | 0.000622 |
| mlp_down | down_proj | 8 | 8192 | 2048 | gptq_machete | 4 | 128 | 0.0266 | 0.0310 | 10.082 | 326.15 | 0.695x | 1.000x | 0.000546 |
| mlp_down | down_proj | 16 | 8192 | 2048 | gptq_machete | 4 | 128 | 0.0254 | 0.0282 | 21.170 | 342.41 | 0.760x | 1.000x | 0.000539 |
| mlp_down | down_proj | 32 | 8192 | 2048 | gptq_machete | 4 | 128 | 0.0248 | 0.0266 | 43.240 | 349.69 | 1.588x | 1.000x | 0.00057 |

The four shape rows preserve all seven Llama 3.2 1B projection roles: `q_proj/o_proj` (2048×2048), `k_proj/v_proj` (2048×512), `gate_proj/up_proj` (2048×8192), and `down_proj` (8192×2048).
