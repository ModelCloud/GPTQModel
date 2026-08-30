# QVQ LR versus W4 GPTQ kernels — Llama 3.2 1B shapes

This is a kernel-throughput comparison on the H100 host from PR #62. W4 GPTQ is a figurative performance baseline, not a quality-equivalent arm.

## Measurement contract

- Commit: `168cefcbef6ea0732bb5b77ecb32604d488c24ae`; benchmark SHA256: `ae13dac8f763d972716bba127790a3bffea2c83f7d2e91d7479728527011b6ca`
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
| attn_qo | q_proj/o_proj | 1 | 2048 | 2048 | qvq_lr | 2 | P32 | 0.1356 | 0.1364 | 0.062 | 7.85 | 0.107x | 0.114x | 3.93e-06 |
| attn_qo | q_proj/o_proj | 2 | 2048 | 2048 | qvq_lr | 2 | P32 | 0.0298 | 0.0304 | 0.564 | 35.78 | 0.544x | 0.523x | 4.77e-06 |
| attn_qo | q_proj/o_proj | 4 | 2048 | 2048 | qvq_lr | 2 | P32 | 0.0306 | 0.0312 | 1.098 | 34.85 | 0.534x | 0.505x | 3.81e-06 |
| attn_qo | q_proj/o_proj | 8 | 2048 | 2048 | qvq_lr | 2 | P32 | 0.0322 | 0.0328 | 2.085 | 33.08 | 0.460x | 0.482x | 5.96e-06 |
| attn_qo | q_proj/o_proj | 16 | 2048 | 2048 | qvq_lr | 2 | P32 | 0.0438 | 0.0444 | 3.067 | 24.34 | 0.360x | 0.352x | 5.72e-06 |
| attn_qo | q_proj/o_proj | 1 | 2048 | 2048 | qvq_lr | 2.5 | P32 | 0.1202 | 0.1218 | 0.070 | 11.04 | 0.121x | 0.128x | 4.29e-06 |
| attn_qo | q_proj/o_proj | 2 | 2048 | 2048 | qvq_lr | 2.5 | P32 | 0.1210 | 0.1225 | 0.139 | 10.97 | 0.134x | 0.129x | 4.98e-06 |
| attn_qo | q_proj/o_proj | 4 | 2048 | 2048 | qvq_lr | 2.5 | P32 | 0.1208 | 0.1225 | 0.278 | 10.99 | 0.135x | 0.128x | 4.77e-06 |
| attn_qo | q_proj/o_proj | 8 | 2048 | 2048 | qvq_lr | 2.5 | P32 | 0.1219 | 0.1230 | 0.551 | 10.89 | 0.122x | 0.127x | 5.72e-06 |
| attn_qo | q_proj/o_proj | 16 | 2048 | 2048 | qvq_lr | 2.5 | P32 | 0.0441 | 0.0446 | 3.045 | 30.11 | 0.357x | 0.350x | 5.3e-06 |
| attn_qo | q_proj/o_proj | 1 | 2048 | 2048 | qvq_lr | 3 | P32 | 0.0559 | 0.0568 | 0.150 | 28.44 | 0.261x | 0.276x | 3.34e-06 |
| attn_qo | q_proj/o_proj | 2 | 2048 | 2048 | qvq_lr | 3 | P32 | 0.0556 | 0.0566 | 0.302 | 28.60 | 0.291x | 0.280x | 4.77e-06 |
| attn_qo | q_proj/o_proj | 4 | 2048 | 2048 | qvq_lr | 3 | P32 | 0.0563 | 0.0571 | 0.596 | 28.21 | 0.290x | 0.274x | 4.77e-06 |
| attn_qo | q_proj/o_proj | 8 | 2048 | 2048 | qvq_lr | 3 | P32 | 0.0571 | 0.0579 | 1.176 | 27.84 | 0.260x | 0.272x | 5.72e-06 |
| attn_qo | q_proj/o_proj | 16 | 2048 | 2048 | qvq_lr | 3 | P32 | 0.0213 | 0.0220 | 6.288 | 74.46 | 0.738x | 0.723x | 1.34e-05 |
| attn_qo | q_proj/o_proj | 1 | 2048 | 2048 | qvq_lr | 3.5 | P32 | 0.0621 | 0.0631 | 0.135 | 29.83 | 0.235x | 0.249x | 4.77e-06 |
| attn_qo | q_proj/o_proj | 2 | 2048 | 2048 | qvq_lr | 3.5 | P32 | 0.0559 | 0.0568 | 0.300 | 33.15 | 0.290x | 0.278x | 3.81e-06 |
| attn_qo | q_proj/o_proj | 4 | 2048 | 2048 | qvq_lr | 3.5 | P32 | 0.0561 | 0.0572 | 0.598 | 32.98 | 0.291x | 0.275x | 4.77e-06 |
| attn_qo | q_proj/o_proj | 8 | 2048 | 2048 | qvq_lr | 3.5 | P32 | 0.0572 | 0.0580 | 1.174 | 32.39 | 0.259x | 0.271x | 4.77e-06 |
| attn_qo | q_proj/o_proj | 16 | 2048 | 2048 | qvq_lr | 3.5 | P32 | 0.0617 | 0.0626 | 2.175 | 30.00 | 0.255x | 0.250x | 4.77e-06 |
| attn_qo | q_proj/o_proj | 1 | 2048 | 2048 | gptq_marlin | 4 | 128 | 0.0146 | 0.0150 | 0.576 | 148.54 | 1.000x | 1.059x | 0.000264 |
| attn_qo | q_proj/o_proj | 2 | 2048 | 2048 | gptq_marlin | 4 | 128 | 0.0162 | 0.0166 | 1.036 | 133.57 | 1.000x | 0.960x | 0.000229 |
| attn_qo | q_proj/o_proj | 4 | 2048 | 2048 | gptq_marlin | 4 | 128 | 0.0163 | 0.0166 | 2.056 | 132.52 | 1.000x | 0.945x | 0.000197 |
| attn_qo | q_proj/o_proj | 8 | 2048 | 2048 | gptq_marlin | 4 | 128 | 0.0148 | 0.0151 | 4.529 | 145.97 | 1.000x | 1.048x | 0.000242 |
| attn_qo | q_proj/o_proj | 16 | 2048 | 2048 | gptq_marlin | 4 | 128 | 0.0157 | 0.0161 | 8.525 | 137.37 | 1.000x | 0.980x | 0.00028 |
| attn_qo | q_proj/o_proj | 1 | 2048 | 2048 | gptq_machete | 4 | 128 | 0.0154 | 0.0164 | 0.544 | 140.75 | 0.944x | 1.000x | 0.000264 |
| attn_qo | q_proj/o_proj | 2 | 2048 | 2048 | gptq_machete | 4 | 128 | 0.0156 | 0.0165 | 1.079 | 139.59 | 1.041x | 1.000x | 0.000229 |
| attn_qo | q_proj/o_proj | 4 | 2048 | 2048 | gptq_machete | 4 | 128 | 0.0154 | 0.0159 | 2.175 | 140.75 | 1.058x | 1.000x | 0.000197 |
| attn_qo | q_proj/o_proj | 8 | 2048 | 2048 | gptq_machete | 4 | 128 | 0.0155 | 0.0164 | 4.324 | 139.88 | 0.955x | 1.000x | 0.000242 |
| attn_qo | q_proj/o_proj | 16 | 2048 | 2048 | gptq_machete | 4 | 128 | 0.0154 | 0.0162 | 8.702 | 140.75 | 1.021x | 1.000x | 0.00028 |
| attn_kv | k_proj/v_proj | 1 | 2048 | 512 | qvq_lr | 2 | P32 | 0.0453 | 0.0458 | 0.046 | 5.88 | 0.486x | 0.329x | 3.34e-06 |
| attn_kv | k_proj/v_proj | 2 | 2048 | 512 | qvq_lr | 2 | P32 | 0.0132 | 0.0139 | 0.317 | 20.10 | 1.917x | 1.147x | 2.86e-06 |
| attn_kv | k_proj/v_proj | 4 | 2048 | 512 | qvq_lr | 2 | P32 | 0.0137 | 0.0144 | 0.612 | 19.42 | 1.907x | 1.117x | 4.77e-06 |
| attn_kv | k_proj/v_proj | 8 | 2048 | 512 | qvq_lr | 2 | P32 | 0.0146 | 0.0152 | 1.147 | 18.21 | 1.578x | 1.022x | 4.77e-06 |
| attn_kv | k_proj/v_proj | 16 | 2048 | 512 | qvq_lr | 2 | P32 | 0.0187 | 0.0193 | 1.794 | 14.23 | 1.299x | 0.815x | 4.29e-06 |
| attn_kv | k_proj/v_proj | 1 | 2048 | 512 | qvq_lr | 2.5 | P32 | 0.0412 | 0.0418 | 0.051 | 8.06 | 0.535x | 0.362x | 2.38e-06 |
| attn_kv | k_proj/v_proj | 2 | 2048 | 512 | qvq_lr | 2.5 | P32 | 0.0372 | 0.0380 | 0.113 | 8.91 | 0.682x | 0.408x | 3.81e-06 |
| attn_kv | k_proj/v_proj | 4 | 2048 | 512 | qvq_lr | 2.5 | P32 | 0.0376 | 0.0385 | 0.223 | 8.82 | 0.695x | 0.407x | 3.81e-06 |
| attn_kv | k_proj/v_proj | 8 | 2048 | 512 | qvq_lr | 2.5 | P32 | 0.0383 | 0.0388 | 0.438 | 8.67 | 0.603x | 0.390x | 3.81e-06 |
| attn_kv | k_proj/v_proj | 16 | 2048 | 512 | qvq_lr | 2.5 | P32 | 0.0189 | 0.0195 | 1.771 | 17.51 | 1.283x | 0.805x | 4.77e-06 |
| attn_kv | k_proj/v_proj | 1 | 2048 | 512 | qvq_lr | 3 | P32 | 0.0208 | 0.0213 | 0.101 | 19.10 | 1.058x | 0.717x | 1.91e-06 |
| attn_kv | k_proj/v_proj | 2 | 2048 | 512 | qvq_lr | 3 | P32 | 0.0201 | 0.0208 | 0.209 | 19.75 | 1.263x | 0.756x | 3.81e-06 |
| attn_kv | k_proj/v_proj | 4 | 2048 | 512 | qvq_lr | 3 | P32 | 0.0204 | 0.0210 | 0.410 | 19.43 | 1.279x | 0.749x | 4.77e-06 |
| attn_kv | k_proj/v_proj | 8 | 2048 | 512 | qvq_lr | 3 | P32 | 0.0214 | 0.0221 | 0.784 | 18.56 | 1.078x | 0.698x | 4.77e-06 |
| attn_kv | k_proj/v_proj | 16 | 2048 | 512 | qvq_lr | 3 | P32 | 0.0240 | 0.0246 | 1.396 | 16.53 | 1.011x | 0.634x | 3.81e-06 |
| attn_kv | k_proj/v_proj | 1 | 2048 | 512 | qvq_lr | 3.5 | P32 | 0.0224 | 0.0229 | 0.094 | 20.65 | 0.982x | 0.665x | 2.38e-06 |
| attn_kv | k_proj/v_proj | 2 | 2048 | 512 | qvq_lr | 3.5 | P32 | 0.0204 | 0.0210 | 0.205 | 22.64 | 1.242x | 0.743x | 4.29e-06 |
| attn_kv | k_proj/v_proj | 4 | 2048 | 512 | qvq_lr | 3.5 | P32 | 0.0208 | 0.0215 | 0.403 | 22.24 | 1.256x | 0.736x | 3.81e-06 |
| attn_kv | k_proj/v_proj | 8 | 2048 | 512 | qvq_lr | 3.5 | P32 | 0.0215 | 0.0222 | 0.780 | 21.52 | 1.073x | 0.695x | 3.81e-06 |
| attn_kv | k_proj/v_proj | 16 | 2048 | 512 | qvq_lr | 3.5 | P32 | 0.0239 | 0.0245 | 1.404 | 19.36 | 1.017x | 0.638x | 4.29e-06 |
| attn_kv | k_proj/v_proj | 1 | 2048 | 512 | gptq_marlin | 4 | 128 | 0.0220 | 0.0228 | 0.095 | 24.56 | 1.000x | 0.677x | 0.000167 |
| attn_kv | k_proj/v_proj | 2 | 2048 | 512 | gptq_marlin | 4 | 128 | 0.0254 | 0.0261 | 0.165 | 21.29 | 1.000x | 0.599x | 0.000155 |
| attn_kv | k_proj/v_proj | 4 | 2048 | 512 | gptq_marlin | 4 | 128 | 0.0261 | 0.0267 | 0.321 | 20.68 | 1.000x | 0.586x | 0.000167 |
| attn_kv | k_proj/v_proj | 8 | 2048 | 512 | gptq_marlin | 4 | 128 | 0.0231 | 0.0234 | 0.727 | 23.43 | 1.000x | 0.648x | 0.000206 |
| attn_kv | k_proj/v_proj | 16 | 2048 | 512 | gptq_marlin | 4 | 128 | 0.0243 | 0.0249 | 1.381 | 22.25 | 1.000x | 0.627x | 0.000194 |
| attn_kv | k_proj/v_proj | 1 | 2048 | 512 | gptq_machete | 4 | 128 | 0.0149 | 0.0158 | 0.141 | 36.81 | 1.476x | 1.000x | 0.000167 |
| attn_kv | k_proj/v_proj | 2 | 2048 | 512 | gptq_machete | 4 | 128 | 0.0152 | 0.0161 | 0.276 | 36.11 | 1.671x | 1.000x | 0.000155 |
| attn_kv | k_proj/v_proj | 4 | 2048 | 512 | gptq_machete | 4 | 128 | 0.0153 | 0.0161 | 0.548 | 35.85 | 1.707x | 1.000x | 0.000167 |
| attn_kv | k_proj/v_proj | 8 | 2048 | 512 | gptq_machete | 4 | 128 | 0.0149 | 0.0158 | 1.123 | 36.73 | 1.544x | 1.000x | 0.000206 |
| attn_kv | k_proj/v_proj | 16 | 2048 | 512 | gptq_machete | 4 | 128 | 0.0152 | 0.0162 | 2.201 | 36.00 | 1.594x | 1.000x | 0.000194 |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | qvq_lr | 2 | P32 | 0.4812 | 0.4825 | 0.070 | 8.85 | 0.020x | 0.037x | 7.15e-06 |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | qvq_lr | 2 | P32 | 0.0882 | 0.0885 | 0.761 | 48.32 | 0.116x | 0.207x | 4.77e-06 |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | qvq_lr | 2 | P32 | 0.0889 | 0.0892 | 1.509 | 47.90 | 0.115x | 0.202x | 5.01e-06 |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | qvq_lr | 2 | P32 | 0.0905 | 0.0908 | 2.966 | 47.07 | 0.111x | 0.200x | 5.72e-06 |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | qvq_lr | 2 | P32 | 0.1296 | 0.1300 | 4.141 | 32.86 | 0.084x | 0.141x | 5.72e-06 |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | qvq_lr | 2.5 | P32 | 0.4845 | 0.4858 | 0.069 | 10.96 | 0.020x | 0.037x | 6.08e-06 |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | qvq_lr | 2.5 | P32 | 0.4560 | 0.4596 | 0.147 | 11.64 | 0.022x | 0.040x | 4.77e-06 |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | qvq_lr | 2.5 | P32 | 0.4568 | 0.4600 | 0.294 | 11.62 | 0.022x | 0.039x | 4.77e-06 |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | qvq_lr | 2.5 | P32 | 0.4557 | 0.4595 | 0.589 | 11.65 | 0.022x | 0.040x | 4.89e-06 |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | qvq_lr | 2.5 | P32 | 0.1308 | 0.1314 | 4.103 | 40.57 | 0.083x | 0.140x | 5.72e-06 |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | qvq_lr | 3 | P32 | 0.2061 | 0.2072 | 0.163 | 30.84 | 0.048x | 0.087x | 6.68e-06 |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | qvq_lr | 3 | P32 | 0.1935 | 0.1960 | 0.347 | 32.85 | 0.053x | 0.094x | 4.77e-06 |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | qvq_lr | 3 | P32 | 0.1956 | 0.1995 | 0.686 | 32.51 | 0.052x | 0.092x | 4.89e-06 |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | qvq_lr | 3 | P32 | 0.1969 | 0.2012 | 1.363 | 32.28 | 0.051x | 0.092x | 4.77e-06 |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | qvq_lr | 3 | P32 | 0.0567 | 0.0571 | 9.476 | 112.20 | 0.193x | 0.322x | 5.91e-05 |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | qvq_lr | 3.5 | P32 | 0.2364 | 0.2372 | 0.142 | 31.33 | 0.042x | 0.076x | 6.2e-06 |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | qvq_lr | 3.5 | P32 | 0.1963 | 0.1998 | 0.342 | 37.73 | 0.052x | 0.093x | 4.77e-06 |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | qvq_lr | 3.5 | P32 | 0.1986 | 0.2009 | 0.676 | 37.28 | 0.052x | 0.091x | 5.72e-06 |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | qvq_lr | 3.5 | P32 | 0.1994 | 0.2016 | 1.346 | 37.14 | 0.050x | 0.091x | 5.96e-06 |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | qvq_lr | 3.5 | P32 | 0.2065 | 0.2091 | 2.600 | 35.86 | 0.053x | 0.088x | 5.25e-06 |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | gptq_marlin | 4 | 128 | 0.0099 | 0.0102 | 3.404 | 877.71 | 1.000x | 1.825x | 0.000272 |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | gptq_marlin | 4 | 128 | 0.0102 | 0.0106 | 6.554 | 844.80 | 1.000x | 1.784x | 0.000299 |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | gptq_marlin | 4 | 128 | 0.0102 | 0.0106 | 13.107 | 844.80 | 1.000x | 1.756x | 0.000274 |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | gptq_marlin | 4 | 128 | 0.0100 | 0.0104 | 26.715 | 860.94 | 1.000x | 1.806x | 0.000278 |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | gptq_marlin | 4 | 128 | 0.0109 | 0.0113 | 49.200 | 792.77 | 1.000x | 1.673x | 0.000327 |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | gptq_machete | 4 | 128 | 0.0180 | 0.0190 | 1.866 | 481.48 | 0.548x | 1.000x | 0.000272 |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | gptq_machete | 4 | 128 | 0.0183 | 0.0191 | 3.673 | 473.89 | 0.560x | 1.000x | 0.000299 |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | gptq_machete | 4 | 128 | 0.0180 | 0.0188 | 7.463 | 481.48 | 0.569x | 1.000x | 0.000274 |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | gptq_machete | 4 | 128 | 0.0181 | 0.0191 | 14.795 | 477.23 | 0.554x | 1.000x | 0.000278 |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | gptq_machete | 4 | 128 | 0.0183 | 0.0192 | 29.408 | 474.31 | 0.598x | 1.000x | 0.000327 |
| mlp_down | down_proj | 1 | 8192 | 2048 | qvq_lr | 2 | P32 | 0.5168 | 0.5187 | 0.065 | 8.24 | 0.031x | 0.043x | 1.34e-05 |
| mlp_down | down_proj | 2 | 8192 | 2048 | qvq_lr | 2 | P32 | 0.0913 | 0.0920 | 0.735 | 46.68 | 0.198x | 0.241x | 9.54e-06 |
| mlp_down | down_proj | 4 | 8192 | 2048 | qvq_lr | 2 | P32 | 0.0921 | 0.0928 | 1.457 | 46.23 | 0.197x | 0.244x | 1.05e-05 |
| mlp_down | down_proj | 8 | 8192 | 2048 | qvq_lr | 2 | P32 | 0.0938 | 0.0945 | 2.861 | 45.39 | 0.177x | 0.238x | 1.67e-05 |
| mlp_down | down_proj | 16 | 8192 | 2048 | qvq_lr | 2 | P32 | 0.1320 | 0.1328 | 4.068 | 32.28 | 0.137x | 0.170x | 1.34e-05 |
| mlp_down | down_proj | 1 | 8192 | 2048 | qvq_lr | 2.5 | P32 | 0.4566 | 0.4608 | 0.073 | 11.63 | 0.036x | 0.049x | 1.25e-05 |
| mlp_down | down_proj | 2 | 8192 | 2048 | qvq_lr | 2.5 | P32 | 0.4577 | 0.4620 | 0.147 | 11.60 | 0.039x | 0.048x | 9.54e-06 |
| mlp_down | down_proj | 4 | 8192 | 2048 | qvq_lr | 2.5 | P32 | 0.4575 | 0.4617 | 0.293 | 11.60 | 0.040x | 0.049x | 1.14e-05 |
| mlp_down | down_proj | 8 | 8192 | 2048 | qvq_lr | 2.5 | P32 | 0.4588 | 0.4641 | 0.585 | 11.57 | 0.036x | 0.049x | 1.72e-05 |
| mlp_down | down_proj | 16 | 8192 | 2048 | qvq_lr | 2.5 | P32 | 0.1358 | 0.1365 | 3.955 | 39.10 | 0.133x | 0.165x | 1.34e-05 |
| mlp_down | down_proj | 1 | 8192 | 2048 | qvq_lr | 3 | P32 | 0.1987 | 0.2001 | 0.169 | 31.99 | 0.082x | 0.113x | 1.41e-05 |
| mlp_down | down_proj | 2 | 8192 | 2048 | qvq_lr | 3 | P32 | 0.1968 | 0.1990 | 0.341 | 32.30 | 0.092x | 0.112x | 1.14e-05 |
| mlp_down | down_proj | 4 | 8192 | 2048 | qvq_lr | 3 | P32 | 0.1981 | 0.2008 | 0.678 | 32.09 | 0.092x | 0.113x | 1.34e-05 |
| mlp_down | down_proj | 8 | 8192 | 2048 | qvq_lr | 3 | P32 | 0.1986 | 0.2026 | 1.351 | 32.00 | 0.084x | 0.112x | 1.86e-05 |
| mlp_down | down_proj | 16 | 8192 | 2048 | qvq_lr | 3 | P32 | 0.0589 | 0.0596 | 9.113 | 107.91 | 0.307x | 0.380x | 0.000122 |
| mlp_down | down_proj | 1 | 8192 | 2048 | qvq_lr | 3.5 | P32 | 0.2232 | 0.2251 | 0.150 | 33.18 | 0.073x | 0.101x | 1.14e-05 |
| mlp_down | down_proj | 2 | 8192 | 2048 | qvq_lr | 3.5 | P32 | 0.1996 | 0.2022 | 0.336 | 37.10 | 0.091x | 0.110x | 1.24e-05 |
| mlp_down | down_proj | 4 | 8192 | 2048 | qvq_lr | 3.5 | P32 | 0.2008 | 0.2026 | 0.669 | 36.89 | 0.090x | 0.112x | 1.34e-05 |
| mlp_down | down_proj | 8 | 8192 | 2048 | qvq_lr | 3.5 | P32 | 0.2014 | 0.2039 | 1.333 | 36.78 | 0.083x | 0.111x | 1.72e-05 |
| mlp_down | down_proj | 16 | 8192 | 2048 | qvq_lr | 3.5 | P32 | 0.2076 | 0.2099 | 2.586 | 35.67 | 0.087x | 0.108x | 1.34e-05 |
| mlp_down | down_proj | 1 | 8192 | 2048 | gptq_marlin | 4 | 128 | 0.0162 | 0.0167 | 2.066 | 532.68 | 1.000x | 1.383x | 0.000346 |
| mlp_down | down_proj | 2 | 8192 | 2048 | gptq_marlin | 4 | 128 | 0.0181 | 0.0185 | 3.712 | 478.47 | 1.000x | 1.216x | 0.000545 |
| mlp_down | down_proj | 4 | 8192 | 2048 | gptq_marlin | 4 | 128 | 0.0181 | 0.0186 | 7.397 | 476.78 | 1.000x | 1.239x | 0.000622 |
| mlp_down | down_proj | 8 | 8192 | 2048 | gptq_marlin | 4 | 128 | 0.0166 | 0.0169 | 16.147 | 520.38 | 1.000x | 1.344x | 0.000546 |
| mlp_down | down_proj | 16 | 8192 | 2048 | gptq_marlin | 4 | 128 | 0.0181 | 0.0185 | 29.694 | 478.47 | 1.000x | 1.239x | 0.000539 |
| mlp_down | down_proj | 1 | 8192 | 2048 | gptq_machete | 4 | 128 | 0.0225 | 0.0234 | 1.494 | 386.55 | 0.723x | 1.000x | 0.000346 |
| mlp_down | down_proj | 2 | 8192 | 2048 | gptq_machete | 4 | 128 | 0.0220 | 0.0229 | 3.053 | 394.99 | 0.822x | 1.000x | 0.000545 |
| mlp_down | down_proj | 4 | 8192 | 2048 | gptq_machete | 4 | 128 | 0.0225 | 0.0232 | 5.971 | 386.28 | 0.807x | 1.000x | 0.000622 |
| mlp_down | down_proj | 8 | 8192 | 2048 | gptq_machete | 4 | 128 | 0.0223 | 0.0231 | 12.018 | 388.77 | 0.744x | 1.000x | 0.000546 |
| mlp_down | down_proj | 16 | 8192 | 2048 | gptq_machete | 4 | 128 | 0.0224 | 0.0232 | 23.967 | 387.66 | 0.807x | 1.000x | 0.000539 |

The four shape rows preserve all seven Llama 3.2 1B projection roles: `q_proj/o_proj` (2048×2048), `k_proj/v_proj` (2048×512), `gate_proj/up_proj` (2048×8192), and `down_proj` (8192×2048).
