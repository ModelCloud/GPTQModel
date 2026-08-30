# QVQ LR versus W4 GPTQ kernels — Llama 3.2 1B shapes

This is a kernel-throughput comparison on the H100 host from PR #62. W4 GPTQ is a figurative performance baseline, not a quality-equivalent arm.

## Measurement contract

- Commit: `0f0cba61d5317fd1ab0f9f98f94116bcba6687ce`; benchmark SHA256: `ae13dac8f763d972716bba127790a3bffea2c83f7d2e91d7479728527011b6ca`
- GPU: physical `1`, `NVIDIA H100`, PCI `00000000:44:00.0`, UUID `GPU-f5ea03cf-efa4-9807-7de5-b174957a1348`, CC `9.0`, 132 SMs
- Torch/CUDA: `2.15.0.dev20260828+cu130` / `13.0`; input dtype: `float16`
- M values: `[1, 2, 4, 8, 16]`; warmup: `10`; measured launches: `60`
- QVQ: `qvq_v2b2_p32_lr`, rates `[2.0, 2.5, 3.0, 3.5]`, native `P32 is LR packing geometry, not a GPTQ affine scale group.`
- GPTQ: symmetric W4, group `128`, no activation order; Marlin and Machete use the same W4 source payload
- Latency is CUDA-event median/P95 from one CUDA Graph replay containing every measured launch. CPU scheduling and host launch gaps are outside each timed interval. Logical TFLOP/s is `2*M*K*N / median_ms`; payload GB/s is packed payload bytes divided by median latency.
- Every row passed its dense-reference shape/dtype/finite gate. QVQ uses `max_abs <= 2e-3`; GPTQ uses `atol=2e-2, rtol=2e-2`.

`xMarlin` and `xMachete` are baseline median latency divided by the row median, so values above 1.0x are faster.

For the focused M sweep, the per-row comparison against the previous CUDA-Graph run (including the required `yes`/`no` regression flag) is in [qvq_lr_vs_gptq_llama32_1b_h100_M1_2_4_8_16_comparison.md](qvq_lr_vs_gptq_llama32_1b_h100_M1_2_4_8_16_comparison.md).

| Shape | Roles | M | K | N | Kernel | W | Group | Median ms | P95 ms | Logical TFLOP/s | Payload GB/s | xMarlin | xMachete | Max abs |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| attn_qo | q_proj/o_proj | 1 | 2048 | 2048 | qvq_lr | 2 | P32 | 0.1352 | 0.1361 | 0.062 | 7.88 | 0.107x | 0.113x | 3.93e-06 |
| attn_qo | q_proj/o_proj | 2 | 2048 | 2048 | qvq_lr | 2 | P32 | 0.0294 | 0.0300 | 0.571 | 36.25 | 0.553x | 0.533x | 4.77e-06 |
| attn_qo | q_proj/o_proj | 4 | 2048 | 2048 | qvq_lr | 2 | P32 | 0.0302 | 0.0308 | 1.110 | 35.24 | 0.537x | 0.505x | 3.81e-06 |
| attn_qo | q_proj/o_proj | 8 | 2048 | 2048 | qvq_lr | 2 | P32 | 0.0317 | 0.0322 | 2.117 | 33.60 | 0.464x | 0.491x | 5.96e-06 |
| attn_qo | q_proj/o_proj | 16 | 2048 | 2048 | qvq_lr | 2 | P32 | 0.0432 | 0.0438 | 3.110 | 24.68 | 0.362x | 0.357x | 5.72e-06 |
| attn_qo | q_proj/o_proj | 1 | 2048 | 2048 | qvq_lr | 2.5 | P32 | 0.1201 | 0.1214 | 0.070 | 11.05 | 0.121x | 0.127x | 4.29e-06 |
| attn_qo | q_proj/o_proj | 2 | 2048 | 2048 | qvq_lr | 2.5 | P32 | 0.1206 | 0.1222 | 0.139 | 11.01 | 0.135x | 0.130x | 4.98e-06 |
| attn_qo | q_proj/o_proj | 4 | 2048 | 2048 | qvq_lr | 2.5 | P32 | 0.1210 | 0.1224 | 0.277 | 10.97 | 0.134x | 0.126x | 4.77e-06 |
| attn_qo | q_proj/o_proj | 8 | 2048 | 2048 | qvq_lr | 2.5 | P32 | 0.1223 | 0.1238 | 0.549 | 10.85 | 0.120x | 0.127x | 5.72e-06 |
| attn_qo | q_proj/o_proj | 16 | 2048 | 2048 | qvq_lr | 2.5 | P32 | 0.0440 | 0.0446 | 3.053 | 30.18 | 0.355x | 0.350x | 5.3e-06 |
| attn_qo | q_proj/o_proj | 1 | 2048 | 2048 | qvq_lr | 3 | P32 | 0.0561 | 0.0567 | 0.150 | 28.35 | 0.259x | 0.273x | 3.34e-06 |
| attn_qo | q_proj/o_proj | 2 | 2048 | 2048 | qvq_lr | 3 | P32 | 0.0556 | 0.0564 | 0.302 | 28.58 | 0.292x | 0.282x | 4.77e-06 |
| attn_qo | q_proj/o_proj | 4 | 2048 | 2048 | qvq_lr | 3 | P32 | 0.0562 | 0.0574 | 0.597 | 28.28 | 0.289x | 0.272x | 4.77e-06 |
| attn_qo | q_proj/o_proj | 8 | 2048 | 2048 | qvq_lr | 3 | P32 | 0.0570 | 0.0580 | 1.177 | 27.87 | 0.258x | 0.273x | 5.72e-06 |
| attn_qo | q_proj/o_proj | 16 | 2048 | 2048 | qvq_lr | 3 | P32 | 0.0222 | 0.0229 | 6.048 | 71.61 | 0.704x | 0.694x | 1.34e-05 |
| attn_qo | q_proj/o_proj | 1 | 2048 | 2048 | qvq_lr | 3.5 | P32 | 0.0621 | 0.0629 | 0.135 | 29.80 | 0.233x | 0.246x | 4.77e-06 |
| attn_qo | q_proj/o_proj | 2 | 2048 | 2048 | qvq_lr | 3.5 | P32 | 0.0557 | 0.0566 | 0.301 | 33.23 | 0.291x | 0.281x | 3.81e-06 |
| attn_qo | q_proj/o_proj | 4 | 2048 | 2048 | qvq_lr | 3.5 | P32 | 0.0561 | 0.0571 | 0.598 | 33.01 | 0.289x | 0.272x | 4.77e-06 |
| attn_qo | q_proj/o_proj | 8 | 2048 | 2048 | qvq_lr | 3.5 | P32 | 0.0572 | 0.0579 | 1.174 | 32.39 | 0.258x | 0.272x | 4.77e-06 |
| attn_qo | q_proj/o_proj | 16 | 2048 | 2048 | qvq_lr | 3.5 | P32 | 0.0617 | 0.0626 | 2.174 | 29.99 | 0.253x | 0.249x | 4.77e-06 |
| attn_qo | q_proj/o_proj | 1 | 2048 | 2048 | gptq_marlin | 4 | 128 | 0.0145 | 0.0150 | 0.579 | 149.19 | 1.000x | 1.055x | 0.000264 |
| attn_qo | q_proj/o_proj | 2 | 2048 | 2048 | gptq_marlin | 4 | 128 | 0.0162 | 0.0166 | 1.033 | 133.17 | 1.000x | 0.965x | 0.000229 |
| attn_qo | q_proj/o_proj | 4 | 2048 | 2048 | gptq_marlin | 4 | 128 | 0.0162 | 0.0166 | 2.068 | 133.30 | 1.000x | 0.941x | 0.000197 |
| attn_qo | q_proj/o_proj | 8 | 2048 | 2048 | gptq_marlin | 4 | 128 | 0.0147 | 0.0151 | 4.559 | 146.92 | 1.000x | 1.058x | 0.000242 |
| attn_qo | q_proj/o_proj | 16 | 2048 | 2048 | gptq_marlin | 4 | 128 | 0.0156 | 0.0160 | 8.595 | 138.49 | 1.000x | 0.986x | 0.00028 |
| attn_qo | q_proj/o_proj | 1 | 2048 | 2048 | gptq_machete | 4 | 128 | 0.0153 | 0.0162 | 0.548 | 141.92 | 0.948x | 1.000x | 0.000264 |
| attn_qo | q_proj/o_proj | 2 | 2048 | 2048 | gptq_machete | 4 | 128 | 0.0157 | 0.0165 | 1.071 | 138.59 | 1.037x | 1.000x | 0.000229 |
| attn_qo | q_proj/o_proj | 4 | 2048 | 2048 | gptq_machete | 4 | 128 | 0.0153 | 0.0162 | 2.198 | 142.22 | 1.063x | 1.000x | 0.000197 |
| attn_qo | q_proj/o_proj | 8 | 2048 | 2048 | gptq_machete | 4 | 128 | 0.0156 | 0.0165 | 4.311 | 139.45 | 0.946x | 1.000x | 0.000242 |
| attn_qo | q_proj/o_proj | 16 | 2048 | 2048 | gptq_machete | 4 | 128 | 0.0154 | 0.0163 | 8.720 | 141.04 | 1.015x | 1.000x | 0.00028 |
| attn_kv | k_proj/v_proj | 1 | 2048 | 512 | qvq_lr | 2 | P32 | 0.0452 | 0.0460 | 0.046 | 5.89 | 0.482x | 0.325x | 3.34e-06 |
| attn_kv | k_proj/v_proj | 2 | 2048 | 512 | qvq_lr | 2 | P32 | 0.0133 | 0.0140 | 0.315 | 19.98 | 1.916x | 1.103x | 2.86e-06 |
| attn_kv | k_proj/v_proj | 4 | 2048 | 512 | qvq_lr | 2 | P32 | 0.0138 | 0.0144 | 0.609 | 19.33 | 1.856x | 1.062x | 4.77e-06 |
| attn_kv | k_proj/v_proj | 8 | 2048 | 512 | qvq_lr | 2 | P32 | 0.0147 | 0.0151 | 1.145 | 18.17 | 1.539x | 1.004x | 4.77e-06 |
| attn_kv | k_proj/v_proj | 16 | 2048 | 512 | qvq_lr | 2 | P32 | 0.0187 | 0.0196 | 1.794 | 14.23 | 1.271x | 0.783x | 4.29e-06 |
| attn_kv | k_proj/v_proj | 1 | 2048 | 512 | qvq_lr | 2.5 | P32 | 0.0411 | 0.0418 | 0.051 | 8.07 | 0.529x | 0.357x | 2.38e-06 |
| attn_kv | k_proj/v_proj | 2 | 2048 | 512 | qvq_lr | 2.5 | P32 | 0.0371 | 0.0380 | 0.113 | 8.94 | 0.688x | 0.396x | 3.81e-06 |
| attn_kv | k_proj/v_proj | 4 | 2048 | 512 | qvq_lr | 2.5 | P32 | 0.0375 | 0.0382 | 0.223 | 8.84 | 0.681x | 0.390x | 3.81e-06 |
| attn_kv | k_proj/v_proj | 8 | 2048 | 512 | qvq_lr | 2.5 | P32 | 0.0381 | 0.0388 | 0.440 | 8.70 | 0.591x | 0.386x | 3.81e-06 |
| attn_kv | k_proj/v_proj | 16 | 2048 | 512 | qvq_lr | 2.5 | P32 | 0.0189 | 0.0195 | 1.777 | 17.57 | 1.259x | 0.775x | 4.77e-06 |
| attn_kv | k_proj/v_proj | 1 | 2048 | 512 | qvq_lr | 3 | P32 | 0.0209 | 0.0214 | 0.100 | 19.03 | 1.042x | 0.703x | 1.91e-06 |
| attn_kv | k_proj/v_proj | 2 | 2048 | 512 | qvq_lr | 3 | P32 | 0.0202 | 0.0208 | 0.208 | 19.66 | 1.264x | 0.728x | 3.81e-06 |
| attn_kv | k_proj/v_proj | 4 | 2048 | 512 | qvq_lr | 3 | P32 | 0.0206 | 0.0212 | 0.408 | 19.32 | 1.244x | 0.711x | 4.77e-06 |
| attn_kv | k_proj/v_proj | 8 | 2048 | 512 | qvq_lr | 3 | P32 | 0.0215 | 0.0220 | 0.782 | 18.52 | 1.051x | 0.686x | 4.77e-06 |
| attn_kv | k_proj/v_proj | 16 | 2048 | 512 | qvq_lr | 3 | P32 | 0.0238 | 0.0244 | 1.411 | 16.71 | 1.000x | 0.616x | 3.81e-06 |
| attn_kv | k_proj/v_proj | 1 | 2048 | 512 | qvq_lr | 3.5 | P32 | 0.0221 | 0.0228 | 0.095 | 20.93 | 0.984x | 0.664x | 2.38e-06 |
| attn_kv | k_proj/v_proj | 2 | 2048 | 512 | qvq_lr | 3.5 | P32 | 0.0203 | 0.0209 | 0.206 | 22.78 | 1.257x | 0.724x | 4.29e-06 |
| attn_kv | k_proj/v_proj | 4 | 2048 | 512 | qvq_lr | 3.5 | P32 | 0.0208 | 0.0213 | 0.404 | 22.30 | 1.232x | 0.705x | 3.81e-06 |
| attn_kv | k_proj/v_proj | 8 | 2048 | 512 | qvq_lr | 3.5 | P32 | 0.0213 | 0.0223 | 0.787 | 21.70 | 1.058x | 0.690x | 3.81e-06 |
| attn_kv | k_proj/v_proj | 16 | 2048 | 512 | qvq_lr | 3.5 | P32 | 0.0238 | 0.0245 | 1.409 | 19.44 | 0.999x | 0.615x | 4.29e-06 |
| attn_kv | k_proj/v_proj | 1 | 2048 | 512 | gptq_marlin | 4 | 128 | 0.0218 | 0.0222 | 0.096 | 24.85 | 1.000x | 0.675x | 0.000167 |
| attn_kv | k_proj/v_proj | 2 | 2048 | 512 | gptq_marlin | 4 | 128 | 0.0255 | 0.0260 | 0.164 | 21.17 | 1.000x | 0.576x | 0.000155 |
| attn_kv | k_proj/v_proj | 4 | 2048 | 512 | gptq_marlin | 4 | 128 | 0.0256 | 0.0260 | 0.328 | 21.15 | 1.000x | 0.572x | 0.000167 |
| attn_kv | k_proj/v_proj | 8 | 2048 | 512 | gptq_marlin | 4 | 128 | 0.0226 | 0.0230 | 0.744 | 23.97 | 1.000x | 0.652x | 0.000206 |
| attn_kv | k_proj/v_proj | 16 | 2048 | 512 | gptq_marlin | 4 | 128 | 0.0238 | 0.0242 | 1.411 | 22.74 | 1.000x | 0.616x | 0.000194 |
| attn_kv | k_proj/v_proj | 1 | 2048 | 512 | gptq_machete | 4 | 128 | 0.0147 | 0.0157 | 0.143 | 37.37 | 1.481x | 1.000x | 0.000167 |
| attn_kv | k_proj/v_proj | 2 | 2048 | 512 | gptq_machete | 4 | 128 | 0.0147 | 0.0154 | 0.285 | 37.33 | 1.737x | 1.000x | 0.000155 |
| attn_kv | k_proj/v_proj | 4 | 2048 | 512 | gptq_machete | 4 | 128 | 0.0146 | 0.0154 | 0.574 | 37.53 | 1.748x | 1.000x | 0.000167 |
| attn_kv | k_proj/v_proj | 8 | 2048 | 512 | gptq_machete | 4 | 128 | 0.0147 | 0.0156 | 1.140 | 37.29 | 1.533x | 1.000x | 0.000206 |
| attn_kv | k_proj/v_proj | 16 | 2048 | 512 | gptq_machete | 4 | 128 | 0.0146 | 0.0156 | 2.292 | 37.49 | 1.624x | 1.000x | 0.000194 |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | qvq_lr | 2 | P32 | 0.4809 | 0.4825 | 0.070 | 8.86 | 0.021x | 0.037x | 7.15e-06 |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | qvq_lr | 2 | P32 | 0.0882 | 0.0885 | 0.761 | 48.32 | 0.117x | 0.203x | 4.77e-06 |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | qvq_lr | 2 | P32 | 0.0888 | 0.0892 | 1.511 | 47.97 | 0.116x | 0.201x | 5.01e-06 |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | qvq_lr | 2 | P32 | 0.0903 | 0.0908 | 2.974 | 47.19 | 0.109x | 0.197x | 5.72e-06 |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | qvq_lr | 2 | P32 | 0.1294 | 0.1298 | 4.149 | 32.92 | 0.083x | 0.139x | 5.72e-06 |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | qvq_lr | 2.5 | P32 | 0.4842 | 0.4852 | 0.069 | 10.96 | 0.020x | 0.037x | 6.08e-06 |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | qvq_lr | 2.5 | P32 | 0.4553 | 0.4588 | 0.147 | 11.66 | 0.023x | 0.039x | 4.77e-06 |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | qvq_lr | 2.5 | P32 | 0.4561 | 0.4610 | 0.294 | 11.64 | 0.023x | 0.039x | 4.77e-06 |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | qvq_lr | 2.5 | P32 | 0.4562 | 0.4595 | 0.588 | 11.64 | 0.022x | 0.039x | 4.89e-06 |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | qvq_lr | 2.5 | P32 | 0.1307 | 0.1313 | 4.107 | 40.61 | 0.082x | 0.138x | 5.72e-06 |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | qvq_lr | 3 | P32 | 0.2060 | 0.2068 | 0.163 | 30.86 | 0.048x | 0.087x | 6.68e-06 |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | qvq_lr | 3 | P32 | 0.1945 | 0.1987 | 0.345 | 32.68 | 0.053x | 0.092x | 4.77e-06 |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | qvq_lr | 3 | P32 | 0.1952 | 0.1983 | 0.688 | 32.57 | 0.053x | 0.092x | 4.89e-06 |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | qvq_lr | 3 | P32 | 0.1975 | 0.2015 | 1.359 | 32.19 | 0.050x | 0.090x | 4.77e-06 |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | qvq_lr | 3 | P32 | 0.0598 | 0.0602 | 8.981 | 106.35 | 0.180x | 0.301x | 5.91e-05 |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | qvq_lr | 3.5 | P32 | 0.2364 | 0.2372 | 0.142 | 31.33 | 0.042x | 0.076x | 6.2e-06 |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | qvq_lr | 3.5 | P32 | 0.1966 | 0.1988 | 0.341 | 37.66 | 0.052x | 0.091x | 4.77e-06 |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | qvq_lr | 3.5 | P32 | 0.1986 | 0.2014 | 0.676 | 37.29 | 0.052x | 0.090x | 5.72e-06 |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | qvq_lr | 3.5 | P32 | 0.1991 | 0.2011 | 1.348 | 37.19 | 0.049x | 0.089x | 5.96e-06 |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | qvq_lr | 3.5 | P32 | 0.2066 | 0.2097 | 2.599 | 35.84 | 0.052x | 0.087x | 5.25e-06 |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | gptq_marlin | 4 | 128 | 0.0099 | 0.0102 | 3.388 | 873.46 | 1.000x | 1.806x | 0.000272 |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | gptq_marlin | 4 | 128 | 0.0103 | 0.0106 | 6.513 | 839.55 | 1.000x | 1.739x | 0.000299 |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | gptq_marlin | 4 | 128 | 0.0103 | 0.0105 | 13.066 | 842.17 | 1.000x | 1.741x | 0.000274 |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | gptq_marlin | 4 | 128 | 0.0099 | 0.0102 | 27.236 | 877.71 | 1.000x | 1.805x | 0.000278 |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | gptq_marlin | 4 | 128 | 0.0108 | 0.0111 | 49.858 | 803.38 | 1.000x | 1.670x | 0.000327 |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | gptq_machete | 4 | 128 | 0.0179 | 0.0187 | 1.876 | 484.06 | 0.554x | 1.000x | 0.000272 |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | gptq_machete | 4 | 128 | 0.0179 | 0.0188 | 3.745 | 483.20 | 0.575x | 1.000x | 0.000299 |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | gptq_machete | 4 | 128 | 0.0179 | 0.0188 | 7.503 | 484.06 | 0.574x | 1.000x | 0.000274 |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | gptq_machete | 4 | 128 | 0.0178 | 0.0188 | 15.087 | 486.68 | 0.554x | 1.000x | 0.000278 |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | gptq_machete | 4 | 128 | 0.0180 | 0.0189 | 29.853 | 481.48 | 0.599x | 1.000x | 0.000327 |
| mlp_down | down_proj | 1 | 8192 | 2048 | qvq_lr | 2 | P32 | 0.5170 | 0.5186 | 0.065 | 8.24 | 0.031x | 0.043x | 1.34e-05 |
| mlp_down | down_proj | 2 | 8192 | 2048 | qvq_lr | 2 | P32 | 0.0910 | 0.0916 | 0.737 | 46.81 | 0.197x | 0.242x | 9.54e-06 |
| mlp_down | down_proj | 4 | 8192 | 2048 | qvq_lr | 2 | P32 | 0.0918 | 0.0925 | 1.462 | 46.39 | 0.196x | 0.241x | 1.05e-05 |
| mlp_down | down_proj | 8 | 8192 | 2048 | qvq_lr | 2 | P32 | 0.0935 | 0.0942 | 2.870 | 45.55 | 0.177x | 0.238x | 1.67e-05 |
| mlp_down | down_proj | 16 | 8192 | 2048 | qvq_lr | 2 | P32 | 0.1316 | 0.1324 | 4.079 | 32.37 | 0.137x | 0.168x | 1.34e-05 |
| mlp_down | down_proj | 1 | 8192 | 2048 | qvq_lr | 2.5 | P32 | 0.4557 | 0.4592 | 0.074 | 11.65 | 0.036x | 0.048x | 1.25e-05 |
| mlp_down | down_proj | 2 | 8192 | 2048 | qvq_lr | 2.5 | P32 | 0.4575 | 0.4631 | 0.147 | 11.60 | 0.039x | 0.048x | 9.54e-06 |
| mlp_down | down_proj | 4 | 8192 | 2048 | qvq_lr | 2.5 | P32 | 0.4580 | 0.4619 | 0.293 | 11.59 | 0.039x | 0.048x | 1.14e-05 |
| mlp_down | down_proj | 8 | 8192 | 2048 | qvq_lr | 2.5 | P32 | 0.4601 | 0.4660 | 0.583 | 11.54 | 0.036x | 0.048x | 1.72e-05 |
| mlp_down | down_proj | 16 | 8192 | 2048 | qvq_lr | 2.5 | P32 | 0.1354 | 0.1361 | 3.964 | 39.20 | 0.133x | 0.164x | 1.34e-05 |
| mlp_down | down_proj | 1 | 8192 | 2048 | qvq_lr | 3 | P32 | 0.1989 | 0.2006 | 0.169 | 31.96 | 0.081x | 0.111x | 1.41e-05 |
| mlp_down | down_proj | 2 | 8192 | 2048 | qvq_lr | 3 | P32 | 0.1961 | 0.1989 | 0.342 | 32.42 | 0.092x | 0.113x | 1.14e-05 |
| mlp_down | down_proj | 4 | 8192 | 2048 | qvq_lr | 3 | P32 | 0.1977 | 0.2016 | 0.679 | 32.16 | 0.091x | 0.112x | 1.34e-05 |
| mlp_down | down_proj | 8 | 8192 | 2048 | qvq_lr | 3 | P32 | 0.1985 | 0.2034 | 1.352 | 32.03 | 0.083x | 0.112x | 1.86e-05 |
| mlp_down | down_proj | 16 | 8192 | 2048 | qvq_lr | 3 | P32 | 0.0628 | 0.0634 | 8.555 | 101.30 | 0.287x | 0.353x | 0.000122 |
| mlp_down | down_proj | 1 | 8192 | 2048 | qvq_lr | 3.5 | P32 | 0.2231 | 0.2257 | 0.150 | 33.19 | 0.073x | 0.099x | 1.14e-05 |
| mlp_down | down_proj | 2 | 8192 | 2048 | qvq_lr | 3.5 | P32 | 0.2000 | 0.2018 | 0.336 | 37.04 | 0.090x | 0.110x | 1.24e-05 |
| mlp_down | down_proj | 4 | 8192 | 2048 | qvq_lr | 3.5 | P32 | 0.2006 | 0.2028 | 0.669 | 36.91 | 0.090x | 0.110x | 1.34e-05 |
| mlp_down | down_proj | 8 | 8192 | 2048 | qvq_lr | 3.5 | P32 | 0.2017 | 0.2036 | 1.331 | 36.71 | 0.082x | 0.110x | 1.72e-05 |
| mlp_down | down_proj | 16 | 8192 | 2048 | qvq_lr | 3.5 | P32 | 0.2071 | 0.2098 | 2.592 | 35.75 | 0.087x | 0.107x | 1.34e-05 |
| mlp_down | down_proj | 1 | 8192 | 2048 | gptq_marlin | 4 | 128 | 0.0162 | 0.0167 | 2.072 | 534.26 | 1.000x | 1.363x | 0.000346 |
| mlp_down | down_proj | 2 | 8192 | 2048 | gptq_marlin | 4 | 128 | 0.0180 | 0.0184 | 3.738 | 481.88 | 1.000x | 1.229x | 0.000545 |
| mlp_down | down_proj | 4 | 8192 | 2048 | gptq_marlin | 4 | 128 | 0.0180 | 0.0185 | 7.450 | 480.17 | 1.000x | 1.228x | 0.000622 |
| mlp_down | down_proj | 8 | 8192 | 2048 | gptq_marlin | 4 | 128 | 0.0165 | 0.0170 | 16.257 | 523.91 | 1.000x | 1.346x | 0.000546 |
| mlp_down | down_proj | 16 | 8192 | 2048 | gptq_marlin | 4 | 128 | 0.0180 | 0.0185 | 29.800 | 480.17 | 1.000x | 1.231x | 0.000539 |
| mlp_down | down_proj | 1 | 8192 | 2048 | gptq_machete | 4 | 128 | 0.0221 | 0.0231 | 1.521 | 393.56 | 0.734x | 1.000x | 0.000346 |
| mlp_down | down_proj | 2 | 8192 | 2048 | gptq_machete | 4 | 128 | 0.0221 | 0.0230 | 3.042 | 393.56 | 0.814x | 1.000x | 0.000545 |
| mlp_down | down_proj | 4 | 8192 | 2048 | gptq_machete | 4 | 128 | 0.0221 | 0.0229 | 6.066 | 392.42 | 0.814x | 1.000x | 0.000622 |
| mlp_down | down_proj | 8 | 8192 | 2048 | gptq_machete | 4 | 128 | 0.0222 | 0.0231 | 12.079 | 390.73 | 0.743x | 1.000x | 0.000546 |
| mlp_down | down_proj | 16 | 8192 | 2048 | gptq_machete | 4 | 128 | 0.0222 | 0.0228 | 24.210 | 391.57 | 0.812x | 1.000x | 0.000539 |

The four shape rows preserve all seven Llama 3.2 1B projection roles: `q_proj/o_proj` (2048×2048), `k_proj/v_proj` (2048×512), `gate_proj/up_proj` (2048×8192), and `down_proj` (8192×2048).
