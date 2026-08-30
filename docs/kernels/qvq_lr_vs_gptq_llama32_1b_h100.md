# QVQ LR versus W4 GPTQ kernels — Llama 3.2 1B shapes

This is a kernel-throughput comparison on the H100 host from PR #62. W4 GPTQ is a figurative performance baseline, not a quality-equivalent arm.

## Measurement contract

- Commit: `2b5be7aabcbf7b2ee00303e23b8ba0ac6427c926`; benchmark SHA256: `ae13dac8f763d972716bba127790a3bffea2c83f7d2e91d7479728527011b6ca`
- GPU: physical `1`, `NVIDIA H100`, PCI `00000000:44:00.0`, UUID `GPU-f5ea03cf-efa4-9807-7de5-b174957a1348`, CC `9.0`, 132 SMs
- Torch/CUDA: `2.15.0.dev20260828+cu130` / `13.0`; input dtype: `float16`
- M values: `[1, 2, 4, 8, 16]`; warmup: `10`; measured launches: `60`
- QVQ: `qvq_v2b2_p32_lr`, rates `[2.0, 2.5, 3.0, 3.5]`, native `P32 is LR packing geometry, not a GPTQ affine scale group.`
- GPTQ: symmetric W4, group `128`, no activation order; Marlin and Machete use the same W4 source payload
- Latency is CUDA-event median/P95 from one CUDA Graph replay containing every measured launch. CPU scheduling and host launch gaps are outside each timed interval. Logical TFLOP/s is `2*M*K*N / median_ms`; payload GB/s is packed payload bytes divided by median latency.
- Every row passed its dense-reference shape/dtype/finite gate. QVQ uses `max_abs <= 2e-3`; GPTQ uses `atol=2e-2, rtol=2e-2`.

`xMarlin` and `xMachete` are baseline median latency divided by the row median, so values above 1.0x are faster.

For the focused M sweep, the per-row comparison against the previous CUDA-Graph run (including the required `yes`/`no` regression flag) is in [qvq_lr_vs_gptq_llama32_1b_h100_M1_2_4_8_16_comparison.md](qvq_lr_vs_gptq_llama32_1b_h100_M1_2_4_8_16_comparison.md).

## Latest PR-tip validation

After the report above, PR tip `31e2f142` (kernel change `1d0c8d19`) was checked on the same H100. W3 at M=16 is not publishable: all four geometries fail the dense-reference gate. This is recorded separately from the last complete matrix so the valid timings are not mixed with invalid output.

| Shape | M | K | N | W | Result |
|---|---:|---:|---:|---:|---|
| attn_qo | 16 | 2048 | 2048 | 3 | fail: non-finite output |
| attn_kv | 16 | 2048 | 512 | 3 | fail: max abs 7.37 |
| mlp_gate_up | 16 | 2048 | 8192 | 3 | fail: max abs 7.58 |
| mlp_down | 16 | 8192 | 2048 | 3 | fail: max abs 14.76 |

| Shape | Roles | M | K | N | Kernel | W | Group | Median ms | P95 ms | Logical TFLOP/s | Payload GB/s | xMarlin | xMachete | Max abs |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| attn_qo | q_proj/o_proj | 1 | 2048 | 2048 | qvq_lr | 2 | P32 | 0.1349 | 0.1358 | 0.062 | 7.90 | 0.108x | 0.113x | 3.93e-06 |
| attn_qo | q_proj/o_proj | 2 | 2048 | 2048 | qvq_lr | 2 | P32 | 0.0292 | 0.0297 | 0.575 | 36.51 | 0.555x | 0.530x | 4.77e-06 |
| attn_qo | q_proj/o_proj | 4 | 2048 | 2048 | qvq_lr | 2 | P32 | 0.0299 | 0.0305 | 1.122 | 35.61 | 0.543x | 0.512x | 3.81e-06 |
| attn_qo | q_proj/o_proj | 8 | 2048 | 2048 | qvq_lr | 2 | P32 | 0.0314 | 0.0320 | 2.136 | 33.89 | 0.471x | 0.495x | 5.96e-06 |
| attn_qo | q_proj/o_proj | 16 | 2048 | 2048 | qvq_lr | 2 | P32 | 0.0430 | 0.0436 | 3.123 | 24.78 | 0.365x | 0.356x | 5.72e-06 |
| attn_qo | q_proj/o_proj | 1 | 2048 | 2048 | qvq_lr | 2.5 | P32 | 0.1199 | 0.1216 | 0.070 | 11.07 | 0.121x | 0.128x | 4.29e-06 |
| attn_qo | q_proj/o_proj | 2 | 2048 | 2048 | qvq_lr | 2.5 | P32 | 0.1207 | 0.1223 | 0.139 | 11.00 | 0.134x | 0.128x | 4.98e-06 |
| attn_qo | q_proj/o_proj | 4 | 2048 | 2048 | qvq_lr | 2.5 | P32 | 0.1212 | 0.1231 | 0.277 | 10.95 | 0.134x | 0.126x | 4.77e-06 |
| attn_qo | q_proj/o_proj | 8 | 2048 | 2048 | qvq_lr | 2.5 | P32 | 0.1220 | 0.1239 | 0.550 | 10.88 | 0.121x | 0.127x | 5.72e-06 |
| attn_qo | q_proj/o_proj | 16 | 2048 | 2048 | qvq_lr | 2.5 | P32 | 0.0440 | 0.0447 | 3.050 | 30.16 | 0.357x | 0.348x | 5.3e-06 |
| attn_qo | q_proj/o_proj | 1 | 2048 | 2048 | qvq_lr | 3 | P32 | 0.0559 | 0.0565 | 0.150 | 28.41 | 0.260x | 0.273x | 3.34e-06 |
| attn_qo | q_proj/o_proj | 2 | 2048 | 2048 | qvq_lr | 3 | P32 | 0.0554 | 0.0561 | 0.303 | 28.69 | 0.292x | 0.279x | 4.77e-06 |
| attn_qo | q_proj/o_proj | 4 | 2048 | 2048 | qvq_lr | 3 | P32 | 0.0563 | 0.0573 | 0.596 | 28.23 | 0.288x | 0.272x | 4.77e-06 |
| attn_qo | q_proj/o_proj | 8 | 2048 | 2048 | qvq_lr | 3 | P32 | 0.0569 | 0.0577 | 1.179 | 27.92 | 0.260x | 0.273x | 5.72e-06 |
| attn_qo | q_proj/o_proj | 16 | 2048 | 2048 | qvq_lr | 3 | P32 | 0.0232 | 0.0237 | 5.785 | 68.50 | 0.677x | 0.659x | 1.34e-05 |
| attn_qo | q_proj/o_proj | 1 | 2048 | 2048 | qvq_lr | 3.5 | P32 | 0.0621 | 0.0629 | 0.135 | 29.83 | 0.234x | 0.246x | 4.77e-06 |
| attn_qo | q_proj/o_proj | 2 | 2048 | 2048 | qvq_lr | 3.5 | P32 | 0.0555 | 0.0564 | 0.302 | 33.37 | 0.292x | 0.279x | 3.81e-06 |
| attn_qo | q_proj/o_proj | 4 | 2048 | 2048 | qvq_lr | 3.5 | P32 | 0.0560 | 0.0566 | 0.599 | 33.04 | 0.290x | 0.273x | 4.77e-06 |
| attn_qo | q_proj/o_proj | 8 | 2048 | 2048 | qvq_lr | 3.5 | P32 | 0.0571 | 0.0579 | 1.176 | 32.43 | 0.260x | 0.272x | 4.77e-06 |
| attn_qo | q_proj/o_proj | 16 | 2048 | 2048 | qvq_lr | 3.5 | P32 | 0.0616 | 0.0623 | 2.177 | 30.03 | 0.255x | 0.248x | 4.77e-06 |
| attn_qo | q_proj/o_proj | 1 | 2048 | 2048 | gptq_marlin | 4 | 128 | 0.0145 | 0.0150 | 0.577 | 148.86 | 1.000x | 1.053x | 0.000264 |
| attn_qo | q_proj/o_proj | 2 | 2048 | 2048 | gptq_marlin | 4 | 128 | 0.0162 | 0.0165 | 1.037 | 133.70 | 1.000x | 0.956x | 0.000229 |
| attn_qo | q_proj/o_proj | 4 | 2048 | 2048 | gptq_marlin | 4 | 128 | 0.0162 | 0.0166 | 2.068 | 133.30 | 1.000x | 0.943x | 0.000197 |
| attn_qo | q_proj/o_proj | 8 | 2048 | 2048 | gptq_marlin | 4 | 128 | 0.0148 | 0.0152 | 4.529 | 145.97 | 1.000x | 1.050x | 0.000242 |
| attn_qo | q_proj/o_proj | 16 | 2048 | 2048 | gptq_marlin | 4 | 128 | 0.0157 | 0.0161 | 8.551 | 137.79 | 1.000x | 0.975x | 0.00028 |
| attn_qo | q_proj/o_proj | 1 | 2048 | 2048 | gptq_machete | 4 | 128 | 0.0153 | 0.0162 | 0.548 | 141.92 | 0.950x | 1.000x | 0.000264 |
| attn_qo | q_proj/o_proj | 2 | 2048 | 2048 | gptq_machete | 4 | 128 | 0.0155 | 0.0163 | 1.084 | 140.31 | 1.046x | 1.000x | 0.000229 |
| attn_qo | q_proj/o_proj | 4 | 2048 | 2048 | gptq_machete | 4 | 128 | 0.0153 | 0.0159 | 2.194 | 141.92 | 1.061x | 1.000x | 0.000197 |
| attn_qo | q_proj/o_proj | 8 | 2048 | 2048 | gptq_machete | 4 | 128 | 0.0156 | 0.0165 | 4.315 | 139.59 | 0.953x | 1.000x | 0.000242 |
| attn_qo | q_proj/o_proj | 16 | 2048 | 2048 | gptq_machete | 4 | 128 | 0.0153 | 0.0162 | 8.775 | 141.92 | 1.026x | 1.000x | 0.00028 |
| attn_kv | k_proj/v_proj | 1 | 2048 | 512 | qvq_lr | 2 | P32 | 0.0451 | 0.0460 | 0.047 | 5.90 | 0.493x | 0.327x | 3.34e-06 |
| attn_kv | k_proj/v_proj | 2 | 2048 | 512 | qvq_lr | 2 | P32 | 0.0133 | 0.0140 | 0.314 | 19.95 | 1.942x | 1.101x | 2.86e-06 |
| attn_kv | k_proj/v_proj | 4 | 2048 | 512 | qvq_lr | 2 | P32 | 0.0136 | 0.0142 | 0.616 | 19.55 | 1.922x | 1.072x | 4.77e-06 |
| attn_kv | k_proj/v_proj | 8 | 2048 | 512 | qvq_lr | 2 | P32 | 0.0145 | 0.0151 | 1.157 | 18.37 | 1.586x | 1.010x | 4.77e-06 |
| attn_kv | k_proj/v_proj | 16 | 2048 | 512 | qvq_lr | 2 | P32 | 0.0186 | 0.0193 | 1.808 | 14.34 | 1.310x | 0.790x | 4.29e-06 |
| attn_kv | k_proj/v_proj | 1 | 2048 | 512 | qvq_lr | 2.5 | P32 | 0.0409 | 0.0416 | 0.051 | 8.12 | 0.543x | 0.360x | 2.38e-06 |
| attn_kv | k_proj/v_proj | 2 | 2048 | 512 | qvq_lr | 2.5 | P32 | 0.0372 | 0.0378 | 0.113 | 8.93 | 0.697x | 0.395x | 3.81e-06 |
| attn_kv | k_proj/v_proj | 4 | 2048 | 512 | qvq_lr | 2.5 | P32 | 0.0374 | 0.0382 | 0.224 | 8.86 | 0.699x | 0.390x | 3.81e-06 |
| attn_kv | k_proj/v_proj | 8 | 2048 | 512 | qvq_lr | 2.5 | P32 | 0.0381 | 0.0389 | 0.440 | 8.71 | 0.603x | 0.384x | 3.81e-06 |
| attn_kv | k_proj/v_proj | 16 | 2048 | 512 | qvq_lr | 2.5 | P32 | 0.0189 | 0.0195 | 1.779 | 17.59 | 1.289x | 0.777x | 4.77e-06 |
| attn_kv | k_proj/v_proj | 1 | 2048 | 512 | qvq_lr | 3 | P32 | 0.0210 | 0.0216 | 0.100 | 18.96 | 1.060x | 0.703x | 1.91e-06 |
| attn_kv | k_proj/v_proj | 2 | 2048 | 512 | qvq_lr | 3 | P32 | 0.0201 | 0.0208 | 0.208 | 19.72 | 1.287x | 0.729x | 3.81e-06 |
| attn_kv | k_proj/v_proj | 4 | 2048 | 512 | qvq_lr | 3 | P32 | 0.0206 | 0.0212 | 0.407 | 19.28 | 1.270x | 0.708x | 4.77e-06 |
| attn_kv | k_proj/v_proj | 8 | 2048 | 512 | qvq_lr | 3 | P32 | 0.0215 | 0.0220 | 0.781 | 18.50 | 1.071x | 0.682x | 4.77e-06 |
| attn_kv | k_proj/v_proj | 16 | 2048 | 512 | qvq_lr | 3 | P32 | 0.0240 | 0.0244 | 1.401 | 16.59 | 1.015x | 0.612x | 3.81e-06 |
| attn_kv | k_proj/v_proj | 1 | 2048 | 512 | qvq_lr | 3.5 | P32 | 0.0222 | 0.0228 | 0.094 | 20.81 | 0.999x | 0.663x | 2.38e-06 |
| attn_kv | k_proj/v_proj | 2 | 2048 | 512 | qvq_lr | 3.5 | P32 | 0.0203 | 0.0209 | 0.207 | 22.81 | 1.278x | 0.724x | 4.29e-06 |
| attn_kv | k_proj/v_proj | 4 | 2048 | 512 | qvq_lr | 3.5 | P32 | 0.0207 | 0.0214 | 0.406 | 22.39 | 1.266x | 0.706x | 3.81e-06 |
| attn_kv | k_proj/v_proj | 8 | 2048 | 512 | qvq_lr | 3.5 | P32 | 0.0216 | 0.0222 | 0.777 | 21.44 | 1.065x | 0.678x | 3.81e-06 |
| attn_kv | k_proj/v_proj | 16 | 2048 | 512 | qvq_lr | 3.5 | P32 | 0.0238 | 0.0244 | 1.410 | 19.45 | 1.022x | 0.616x | 4.29e-06 |
| attn_kv | k_proj/v_proj | 1 | 2048 | 512 | gptq_marlin | 4 | 128 | 0.0222 | 0.0228 | 0.094 | 24.35 | 1.000x | 0.664x | 0.000167 |
| attn_kv | k_proj/v_proj | 2 | 2048 | 512 | gptq_marlin | 4 | 128 | 0.0259 | 0.0263 | 0.162 | 20.86 | 1.000x | 0.567x | 0.000155 |
| attn_kv | k_proj/v_proj | 4 | 2048 | 512 | gptq_marlin | 4 | 128 | 0.0262 | 0.0265 | 0.320 | 20.66 | 1.000x | 0.557x | 0.000167 |
| attn_kv | k_proj/v_proj | 8 | 2048 | 512 | gptq_marlin | 4 | 128 | 0.0230 | 0.0234 | 0.730 | 23.52 | 1.000x | 0.637x | 0.000206 |
| attn_kv | k_proj/v_proj | 16 | 2048 | 512 | gptq_marlin | 4 | 128 | 0.0243 | 0.0248 | 1.380 | 22.23 | 1.000x | 0.603x | 0.000194 |
| attn_kv | k_proj/v_proj | 1 | 2048 | 512 | gptq_machete | 4 | 128 | 0.0147 | 0.0155 | 0.142 | 37.25 | 1.507x | 1.000x | 0.000167 |
| attn_kv | k_proj/v_proj | 2 | 2048 | 512 | gptq_machete | 4 | 128 | 0.0147 | 0.0155 | 0.286 | 37.37 | 1.765x | 1.000x | 0.000155 |
| attn_kv | k_proj/v_proj | 4 | 2048 | 512 | gptq_machete | 4 | 128 | 0.0146 | 0.0155 | 0.575 | 37.61 | 1.794x | 1.000x | 0.000167 |
| attn_kv | k_proj/v_proj | 8 | 2048 | 512 | gptq_machete | 4 | 128 | 0.0146 | 0.0155 | 1.146 | 37.49 | 1.570x | 1.000x | 0.000206 |
| attn_kv | k_proj/v_proj | 16 | 2048 | 512 | gptq_machete | 4 | 128 | 0.0147 | 0.0155 | 2.289 | 37.45 | 1.659x | 1.000x | 0.000194 |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | qvq_lr | 2 | P32 | 0.4811 | 0.4825 | 0.070 | 8.85 | 0.020x | 0.037x | 7.15e-06 |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | qvq_lr | 2 | P32 | 0.0882 | 0.0885 | 0.761 | 48.32 | 0.116x | 0.203x | 4.77e-06 |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | qvq_lr | 2 | P32 | 0.0888 | 0.0893 | 1.511 | 47.95 | 0.115x | 0.201x | 5.01e-06 |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | qvq_lr | 2 | P32 | 0.0903 | 0.0907 | 2.972 | 47.16 | 0.110x | 0.199x | 5.72e-06 |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | qvq_lr | 2 | P32 | 0.1294 | 0.1298 | 4.148 | 32.91 | 0.083x | 0.140x | 5.72e-06 |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | qvq_lr | 2.5 | P32 | 0.4846 | 0.4856 | 0.069 | 10.95 | 0.020x | 0.037x | 6.08e-06 |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | qvq_lr | 2.5 | P32 | 0.4557 | 0.4597 | 0.147 | 11.65 | 0.022x | 0.039x | 4.77e-06 |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | qvq_lr | 2.5 | P32 | 0.4566 | 0.4602 | 0.294 | 11.62 | 0.022x | 0.039x | 4.77e-06 |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | qvq_lr | 2.5 | P32 | 0.4568 | 0.4614 | 0.588 | 11.62 | 0.022x | 0.039x | 4.89e-06 |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | qvq_lr | 2.5 | P32 | 0.1313 | 0.1318 | 4.089 | 40.43 | 0.082x | 0.138x | 5.72e-06 |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | qvq_lr | 3 | P32 | 0.2060 | 0.2068 | 0.163 | 30.85 | 0.048x | 0.087x | 6.68e-06 |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | qvq_lr | 3 | P32 | 0.1943 | 0.1987 | 0.345 | 32.72 | 0.053x | 0.092x | 4.77e-06 |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | qvq_lr | 3 | P32 | 0.1954 | 0.1982 | 0.687 | 32.54 | 0.052x | 0.091x | 4.89e-06 |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | qvq_lr | 3 | P32 | 0.1967 | 0.2003 | 1.364 | 32.31 | 0.050x | 0.091x | 4.77e-06 |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | qvq_lr | 3 | P32 | 0.0621 | 0.0626 | 8.644 | 102.35 | 0.173x | 0.292x | 5.91e-05 |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | qvq_lr | 3.5 | P32 | 0.2360 | 0.2367 | 0.142 | 31.38 | 0.042x | 0.076x | 6.2e-06 |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | qvq_lr | 3.5 | P32 | 0.1956 | 0.2004 | 0.343 | 37.87 | 0.052x | 0.091x | 4.77e-06 |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | qvq_lr | 3.5 | P32 | 0.1991 | 0.2014 | 0.674 | 37.20 | 0.051x | 0.090x | 5.72e-06 |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | qvq_lr | 3.5 | P32 | 0.1996 | 0.2014 | 1.345 | 37.10 | 0.050x | 0.090x | 5.96e-06 |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | qvq_lr | 3.5 | P32 | 0.2062 | 0.2086 | 2.603 | 35.91 | 0.052x | 0.088x | 5.25e-06 |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | gptq_marlin | 4 | 128 | 0.0099 | 0.0101 | 3.404 | 877.71 | 1.000x | 1.810x | 0.000272 |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | gptq_marlin | 4 | 128 | 0.0102 | 0.0105 | 6.574 | 847.45 | 1.000x | 1.749x | 0.000299 |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | gptq_marlin | 4 | 128 | 0.0102 | 0.0105 | 13.107 | 844.80 | 1.000x | 1.744x | 0.000274 |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | gptq_marlin | 4 | 128 | 0.0099 | 0.0102 | 27.060 | 872.05 | 1.000x | 1.811x | 0.000278 |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | gptq_marlin | 4 | 128 | 0.0108 | 0.0112 | 49.932 | 804.57 | 1.000x | 1.688x | 0.000327 |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | gptq_machete | 4 | 128 | 0.0178 | 0.0189 | 1.881 | 485.37 | 0.552x | 1.000x | 0.000272 |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | gptq_machete | 4 | 128 | 0.0179 | 0.0187 | 3.758 | 484.93 | 0.572x | 1.000x | 0.000299 |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | gptq_machete | 4 | 128 | 0.0179 | 0.0187 | 7.517 | 484.93 | 0.573x | 1.000x | 0.000274 |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | gptq_machete | 4 | 128 | 0.0180 | 0.0188 | 14.940 | 481.91 | 0.552x | 1.000x | 0.000278 |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | gptq_machete | 4 | 128 | 0.0181 | 0.0190 | 29.589 | 477.23 | 0.593x | 1.000x | 0.000327 |
| mlp_down | down_proj | 1 | 8192 | 2048 | qvq_lr | 2 | P32 | 0.5170 | 0.5199 | 0.065 | 8.24 | 0.031x | 0.043x | 1.34e-05 |
| mlp_down | down_proj | 2 | 8192 | 2048 | qvq_lr | 2 | P32 | 0.0910 | 0.0916 | 0.737 | 46.79 | 0.197x | 0.237x | 9.54e-06 |
| mlp_down | down_proj | 4 | 8192 | 2048 | qvq_lr | 2 | P32 | 0.0919 | 0.0925 | 1.461 | 46.38 | 0.196x | 0.242x | 1.05e-05 |
| mlp_down | down_proj | 8 | 8192 | 2048 | qvq_lr | 2 | P32 | 0.0935 | 0.0943 | 2.871 | 45.56 | 0.176x | 0.237x | 1.67e-05 |
| mlp_down | down_proj | 16 | 8192 | 2048 | qvq_lr | 2 | P32 | 0.1317 | 0.1327 | 4.076 | 32.34 | 0.136x | 0.170x | 1.34e-05 |
| mlp_down | down_proj | 1 | 8192 | 2048 | qvq_lr | 2.5 | P32 | 0.4558 | 0.4596 | 0.074 | 11.65 | 0.035x | 0.049x | 1.25e-05 |
| mlp_down | down_proj | 2 | 8192 | 2048 | qvq_lr | 2.5 | P32 | 0.4575 | 0.4628 | 0.147 | 11.60 | 0.039x | 0.047x | 9.54e-06 |
| mlp_down | down_proj | 4 | 8192 | 2048 | qvq_lr | 2.5 | P32 | 0.4570 | 0.4611 | 0.294 | 11.61 | 0.039x | 0.049x | 1.14e-05 |
| mlp_down | down_proj | 8 | 8192 | 2048 | qvq_lr | 2.5 | P32 | 0.4589 | 0.4636 | 0.585 | 11.57 | 0.036x | 0.048x | 1.72e-05 |
| mlp_down | down_proj | 16 | 8192 | 2048 | qvq_lr | 2.5 | P32 | 0.1353 | 0.1359 | 3.968 | 39.23 | 0.132x | 0.165x | 1.34e-05 |
| mlp_down | down_proj | 1 | 8192 | 2048 | qvq_lr | 3 | P32 | 0.1988 | 0.2008 | 0.169 | 31.97 | 0.081x | 0.111x | 1.41e-05 |
| mlp_down | down_proj | 2 | 8192 | 2048 | qvq_lr | 3 | P32 | 0.1969 | 0.1990 | 0.341 | 32.28 | 0.091x | 0.110x | 1.14e-05 |
| mlp_down | down_proj | 4 | 8192 | 2048 | qvq_lr | 3 | P32 | 0.1977 | 0.2020 | 0.679 | 32.16 | 0.091x | 0.113x | 1.34e-05 |
| mlp_down | down_proj | 8 | 8192 | 2048 | qvq_lr | 3 | P32 | 0.1986 | 0.2025 | 1.352 | 32.02 | 0.083x | 0.112x | 1.86e-05 |
| mlp_down | down_proj | 16 | 8192 | 2048 | qvq_lr | 3 | P32 | 0.0704 | 0.0711 | 7.621 | 90.24 | 0.254x | 0.317x | 0.000122 |
| mlp_down | down_proj | 1 | 8192 | 2048 | qvq_lr | 3.5 | P32 | 0.2240 | 0.2252 | 0.150 | 33.06 | 0.072x | 0.099x | 1.14e-05 |
| mlp_down | down_proj | 2 | 8192 | 2048 | qvq_lr | 3.5 | P32 | 0.2002 | 0.2032 | 0.335 | 37.00 | 0.090x | 0.108x | 1.24e-05 |
| mlp_down | down_proj | 4 | 8192 | 2048 | qvq_lr | 3.5 | P32 | 0.2003 | 0.2023 | 0.670 | 36.97 | 0.090x | 0.111x | 1.34e-05 |
| mlp_down | down_proj | 8 | 8192 | 2048 | qvq_lr | 3.5 | P32 | 0.2015 | 0.2035 | 1.332 | 36.75 | 0.082x | 0.110x | 1.72e-05 |
| mlp_down | down_proj | 16 | 8192 | 2048 | qvq_lr | 3.5 | P32 | 0.2069 | 0.2093 | 2.594 | 35.79 | 0.086x | 0.108x | 1.34e-05 |
| mlp_down | down_proj | 1 | 8192 | 2048 | gptq_marlin | 4 | 128 | 0.0162 | 0.0166 | 2.076 | 535.32 | 1.000x | 1.371x | 0.000346 |
| mlp_down | down_proj | 2 | 8192 | 2048 | gptq_marlin | 4 | 128 | 0.0179 | 0.0184 | 3.742 | 482.31 | 1.000x | 1.204x | 0.000545 |
| mlp_down | down_proj | 4 | 8192 | 2048 | gptq_marlin | 4 | 128 | 0.0180 | 0.0185 | 7.443 | 479.74 | 1.000x | 1.234x | 0.000622 |
| mlp_down | down_proj | 8 | 8192 | 2048 | gptq_marlin | 4 | 128 | 0.0164 | 0.0168 | 16.336 | 526.46 | 1.000x | 1.348x | 0.000546 |
| mlp_down | down_proj | 16 | 8192 | 2048 | gptq_marlin | 4 | 128 | 0.0179 | 0.0182 | 30.013 | 483.61 | 1.000x | 1.249x | 0.000539 |
| mlp_down | down_proj | 1 | 8192 | 2048 | gptq_machete | 4 | 128 | 0.0222 | 0.0232 | 1.514 | 391.86 | 0.729x | 1.000x | 0.000346 |
| mlp_down | down_proj | 2 | 8192 | 2048 | gptq_machete | 4 | 128 | 0.0216 | 0.0225 | 3.107 | 402.01 | 0.830x | 1.000x | 0.000545 |
| mlp_down | down_proj | 4 | 8192 | 2048 | gptq_machete | 4 | 128 | 0.0223 | 0.0231 | 6.031 | 390.17 | 0.810x | 1.000x | 0.000622 |
| mlp_down | down_proj | 8 | 8192 | 2048 | gptq_machete | 4 | 128 | 0.0221 | 0.0231 | 12.122 | 392.14 | 0.742x | 1.000x | 0.000546 |
| mlp_down | down_proj | 16 | 8192 | 2048 | gptq_machete | 4 | 128 | 0.0223 | 0.0232 | 24.036 | 388.77 | 0.801x | 1.000x | 0.000539 |

The four shape rows preserve all seven Llama 3.2 1B projection roles: `q_proj/o_proj` (2048×2048), `k_proj/v_proj` (2048×512), `gate_proj/up_proj` (2048×8192), and `down_proj` (8192×2048).
