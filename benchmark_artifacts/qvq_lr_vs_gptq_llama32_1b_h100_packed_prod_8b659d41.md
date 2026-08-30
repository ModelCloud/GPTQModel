# QVQ LR versus W4 GPTQ kernels — Llama 3.2 1B shapes

This is a kernel-throughput comparison on the H100 host from PR #62. W4 GPTQ is a figurative performance baseline, not a quality-equivalent arm.

## Measurement contract

- Commit: `8b659d419138263144d0199fe46e7613eaa9f68d`; benchmark SHA256: `ae13dac8f763d972716bba127790a3bffea2c83f7d2e91d7479728527011b6ca`
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
| attn_qo | q_proj/o_proj | 1 | 2048 | 2048 | qvq_lr | 2 | P32 | 0.0185 | 0.0191 | 0.454 | 57.68 | 0.786x | 0.834x | 7.63e-06 |
| attn_qo | q_proj/o_proj | 2 | 2048 | 2048 | qvq_lr | 2 | P32 | 0.0190 | 0.0196 | 0.882 | 55.98 | 0.854x | 0.792x | 1.34e-05 |
| attn_qo | q_proj/o_proj | 4 | 2048 | 2048 | qvq_lr | 2 | P32 | 0.0188 | 0.0196 | 1.783 | 56.60 | 0.864x | 0.820x | 1.43e-05 |
| attn_qo | q_proj/o_proj | 8 | 2048 | 2048 | qvq_lr | 2 | P32 | 0.0190 | 0.0198 | 3.525 | 55.93 | 0.776x | 0.794x | 1.34e-05 |
| attn_qo | q_proj/o_proj | 16 | 2048 | 2048 | qvq_lr | 2 | P32 | 0.0196 | 0.0203 | 6.853 | 54.38 | 0.794x | 0.792x | 1.34e-05 |
| attn_qo | q_proj/o_proj | 1 | 2048 | 2048 | qvq_lr | 2.5 | P32 | 0.0183 | 0.0189 | 0.457 | 72.38 | 0.791x | 0.839x | 7.63e-06 |
| attn_qo | q_proj/o_proj | 2 | 2048 | 2048 | qvq_lr | 2.5 | P32 | 0.0191 | 0.0200 | 0.880 | 69.64 | 0.853x | 0.791x | 1.43e-05 |
| attn_qo | q_proj/o_proj | 4 | 2048 | 2048 | qvq_lr | 2.5 | P32 | 0.0191 | 0.0200 | 1.758 | 69.53 | 0.852x | 0.808x | 1.14e-05 |
| attn_qo | q_proj/o_proj | 8 | 2048 | 2048 | qvq_lr | 2.5 | P32 | 0.0191 | 0.0198 | 3.507 | 69.35 | 0.772x | 0.790x | 1.34e-05 |
| attn_qo | q_proj/o_proj | 16 | 2048 | 2048 | qvq_lr | 2.5 | P32 | 0.0194 | 0.0199 | 6.933 | 68.55 | 0.803x | 0.802x | 1.34e-05 |
| attn_qo | q_proj/o_proj | 1 | 2048 | 2048 | gptq_marlin | 4 | 128 | 0.0145 | 0.0148 | 0.578 | 149.03 | 1.000x | 1.061x | 0.000264 |
| attn_qo | q_proj/o_proj | 2 | 2048 | 2048 | gptq_marlin | 4 | 128 | 0.0163 | 0.0167 | 1.032 | 133.04 | 1.000x | 0.927x | 0.000229 |
| attn_qo | q_proj/o_proj | 4 | 2048 | 2048 | gptq_marlin | 4 | 128 | 0.0163 | 0.0167 | 2.064 | 133.04 | 1.000x | 0.949x | 0.000197 |
| attn_qo | q_proj/o_proj | 8 | 2048 | 2048 | gptq_marlin | 4 | 128 | 0.0148 | 0.0151 | 4.544 | 146.44 | 1.000x | 1.024x | 0.000242 |
| attn_qo | q_proj/o_proj | 16 | 2048 | 2048 | gptq_marlin | 4 | 128 | 0.0156 | 0.0159 | 8.630 | 139.06 | 1.000x | 0.998x | 0.00028 |
| attn_qo | q_proj/o_proj | 1 | 2048 | 2048 | gptq_machete | 4 | 128 | 0.0154 | 0.0164 | 0.545 | 141.04 | 0.943x | 1.000x | 0.000264 |
| attn_qo | q_proj/o_proj | 2 | 2048 | 2048 | gptq_machete | 4 | 128 | 0.0151 | 0.0160 | 1.113 | 144.03 | 1.079x | 1.000x | 0.000229 |
| attn_qo | q_proj/o_proj | 4 | 2048 | 2048 | gptq_machete | 4 | 128 | 0.0154 | 0.0163 | 2.175 | 140.75 | 1.054x | 1.000x | 0.000197 |
| attn_qo | q_proj/o_proj | 8 | 2048 | 2048 | gptq_machete | 4 | 128 | 0.0151 | 0.0158 | 4.438 | 143.58 | 0.977x | 1.000x | 0.000242 |
| attn_qo | q_proj/o_proj | 16 | 2048 | 2048 | gptq_machete | 4 | 128 | 0.0155 | 0.0165 | 8.648 | 139.88 | 1.002x | 1.000x | 0.00028 |
| attn_kv | k_proj/v_proj | 1 | 2048 | 512 | qvq_lr | 2 | P32 | 0.0125 | 0.0131 | 0.167 | 21.22 | 1.764x | 1.186x | 5.72e-06 |
| attn_kv | k_proj/v_proj | 2 | 2048 | 512 | qvq_lr | 2 | P32 | 0.0125 | 0.0131 | 0.335 | 21.25 | 2.079x | 1.208x | 7.63e-06 |
| attn_kv | k_proj/v_proj | 4 | 2048 | 512 | qvq_lr | 2 | P32 | 0.0126 | 0.0132 | 0.666 | 21.14 | 2.084x | 1.215x | 6.68e-06 |
| attn_kv | k_proj/v_proj | 8 | 2048 | 512 | qvq_lr | 2 | P32 | 0.0126 | 0.0132 | 1.334 | 21.17 | 1.836x | 1.181x | 7.63e-06 |
| attn_kv | k_proj/v_proj | 16 | 2048 | 512 | qvq_lr | 2 | P32 | 0.0129 | 0.0134 | 2.605 | 20.67 | 1.886x | 1.180x | 7.63e-06 |
| attn_kv | k_proj/v_proj | 1 | 2048 | 512 | qvq_lr | 2.5 | P32 | 0.0130 | 0.0136 | 0.162 | 25.60 | 1.707x | 1.148x | 5.72e-06 |
| attn_kv | k_proj/v_proj | 2 | 2048 | 512 | qvq_lr | 2.5 | P32 | 0.0128 | 0.0135 | 0.326 | 25.82 | 2.027x | 1.178x | 8.11e-06 |
| attn_kv | k_proj/v_proj | 4 | 2048 | 512 | qvq_lr | 2.5 | P32 | 0.0129 | 0.0136 | 0.649 | 25.66 | 2.030x | 1.183x | 6.68e-06 |
| attn_kv | k_proj/v_proj | 8 | 2048 | 512 | qvq_lr | 2.5 | P32 | 0.0124 | 0.0133 | 1.350 | 26.69 | 1.857x | 1.194x | 7.63e-06 |
| attn_kv | k_proj/v_proj | 16 | 2048 | 512 | qvq_lr | 2.5 | P32 | 0.0127 | 0.0131 | 2.648 | 26.18 | 1.917x | 1.199x | 9.54e-06 |
| attn_kv | k_proj/v_proj | 1 | 2048 | 512 | gptq_marlin | 4 | 128 | 0.0221 | 0.0229 | 0.095 | 24.43 | 1.000x | 0.672x | 0.000167 |
| attn_kv | k_proj/v_proj | 2 | 2048 | 512 | gptq_marlin | 4 | 128 | 0.0260 | 0.0265 | 0.161 | 20.76 | 1.000x | 0.581x | 0.000155 |
| attn_kv | k_proj/v_proj | 4 | 2048 | 512 | gptq_marlin | 4 | 128 | 0.0262 | 0.0267 | 0.320 | 20.60 | 1.000x | 0.583x | 0.000167 |
| attn_kv | k_proj/v_proj | 8 | 2048 | 512 | gptq_marlin | 4 | 128 | 0.0231 | 0.0236 | 0.727 | 23.42 | 1.000x | 0.643x | 0.000206 |
| attn_kv | k_proj/v_proj | 16 | 2048 | 512 | gptq_marlin | 4 | 128 | 0.0243 | 0.0247 | 1.382 | 22.26 | 1.000x | 0.626x | 0.000194 |
| attn_kv | k_proj/v_proj | 1 | 2048 | 512 | gptq_machete | 4 | 128 | 0.0149 | 0.0156 | 0.141 | 36.89 | 1.487x | 1.000x | 0.000167 |
| attn_kv | k_proj/v_proj | 2 | 2048 | 512 | gptq_machete | 4 | 128 | 0.0151 | 0.0159 | 0.277 | 36.26 | 1.721x | 1.000x | 0.000155 |
| attn_kv | k_proj/v_proj | 4 | 2048 | 512 | gptq_machete | 4 | 128 | 0.0153 | 0.0160 | 0.548 | 35.88 | 1.715x | 1.000x | 0.000167 |
| attn_kv | k_proj/v_proj | 8 | 2048 | 512 | gptq_machete | 4 | 128 | 0.0148 | 0.0157 | 1.130 | 36.97 | 1.555x | 1.000x | 0.000206 |
| attn_kv | k_proj/v_proj | 16 | 2048 | 512 | gptq_machete | 4 | 128 | 0.0152 | 0.0162 | 2.208 | 36.11 | 1.598x | 1.000x | 0.000194 |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | qvq_lr | 2 | P32 | 0.0521 | 0.0536 | 0.644 | 81.77 | 0.227x | 0.399x | 5.72e-05 |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | qvq_lr | 2 | P32 | 0.0530 | 0.0561 | 1.266 | 80.39 | 0.226x | 0.406x | 5.15e-05 |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | qvq_lr | 2 | P32 | 0.0524 | 0.0552 | 2.562 | 81.32 | 0.230x | 0.397x | 5.72e-05 |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | qvq_lr | 2 | P32 | 0.0512 | 0.0516 | 5.245 | 83.23 | 0.229x | 0.414x | 7.63e-05 |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | qvq_lr | 2 | P32 | 0.0533 | 0.0556 | 10.073 | 79.93 | 0.237x | 0.409x | 6.1e-05 |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | qvq_lr | 2.5 | P32 | 0.0479 | 0.0485 | 0.700 | 110.74 | 0.246x | 0.433x | 4.86e-05 |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | qvq_lr | 2.5 | P32 | 0.0492 | 0.0506 | 1.364 | 107.89 | 0.244x | 0.437x | 5.53e-05 |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | qvq_lr | 2.5 | P32 | 0.0496 | 0.0509 | 2.707 | 107.06 | 0.243x | 0.419x | 6.29e-05 |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | qvq_lr | 2.5 | P32 | 0.0500 | 0.0522 | 5.369 | 106.17 | 0.234x | 0.424x | 5.34e-05 |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | qvq_lr | 2.5 | P32 | 0.0506 | 0.0520 | 10.615 | 104.96 | 0.250x | 0.431x | 6.87e-05 |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | gptq_marlin | 4 | 128 | 0.0118 | 0.0144 | 2.842 | 732.62 | 1.000x | 1.759x | 0.000272 |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | gptq_marlin | 4 | 128 | 0.0120 | 0.0147 | 5.592 | 720.90 | 1.000x | 1.793x | 0.000299 |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | gptq_marlin | 4 | 128 | 0.0120 | 0.0137 | 11.140 | 718.02 | 1.000x | 1.726x | 0.000274 |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | gptq_marlin | 4 | 128 | 0.0117 | 0.0138 | 22.920 | 738.62 | 1.000x | 1.809x | 0.000278 |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | gptq_marlin | 4 | 128 | 0.0127 | 0.0154 | 42.420 | 683.53 | 1.000x | 1.722x | 0.000327 |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | gptq_machete | 4 | 128 | 0.0208 | 0.0235 | 1.616 | 416.94 | 0.569x | 1.000x | 0.000272 |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | gptq_machete | 4 | 128 | 0.0215 | 0.0250 | 3.118 | 402.37 | 0.558x | 1.000x | 0.000299 |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | gptq_machete | 4 | 128 | 0.0208 | 0.0239 | 6.453 | 416.30 | 0.579x | 1.000x | 0.000274 |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | gptq_machete | 4 | 128 | 0.0212 | 0.0260 | 12.672 | 408.75 | 0.553x | 1.000x | 0.000278 |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | gptq_machete | 4 | 128 | 0.0218 | 0.0258 | 24.636 | 397.35 | 0.581x | 1.000x | 0.000327 |
| mlp_down | down_proj | 1 | 8192 | 2048 | qvq_lr | 2 | P32 | 0.0437 | 0.0454 | 0.767 | 97.38 | 0.377x | 0.582x | 4.2e-05 |
| mlp_down | down_proj | 2 | 8192 | 2048 | qvq_lr | 2 | P32 | 0.0497 | 0.0515 | 1.350 | 85.69 | 0.362x | 0.499x | 9.92e-05 |
| mlp_down | down_proj | 4 | 8192 | 2048 | qvq_lr | 2 | P32 | 0.0503 | 0.0553 | 2.669 | 84.71 | 0.360x | 0.505x | 8.39e-05 |
| mlp_down | down_proj | 8 | 8192 | 2048 | qvq_lr | 2 | P32 | 0.0500 | 0.0516 | 5.369 | 85.20 | 0.331x | 0.503x | 9.16e-05 |
| mlp_down | down_proj | 16 | 8192 | 2048 | qvq_lr | 2 | P32 | 0.0495 | 0.0519 | 10.849 | 86.08 | 0.362x | 0.517x | 0.000101 |
| mlp_down | down_proj | 1 | 8192 | 2048 | qvq_lr | 2.5 | P32 | 0.0464 | 0.0496 | 0.724 | 114.48 | 0.356x | 0.549x | 4.2e-05 |
| mlp_down | down_proj | 2 | 8192 | 2048 | qvq_lr | 2.5 | P32 | 0.0515 | 0.0540 | 1.302 | 103.00 | 0.349x | 0.481x | 9.92e-05 |
| mlp_down | down_proj | 4 | 8192 | 2048 | qvq_lr | 2.5 | P32 | 0.0500 | 0.0509 | 2.684 | 106.17 | 0.362x | 0.508x | 9.92e-05 |
| mlp_down | down_proj | 8 | 8192 | 2048 | qvq_lr | 2.5 | P32 | 0.0502 | 0.0510 | 5.345 | 105.69 | 0.329x | 0.500x | 0.000113 |
| mlp_down | down_proj | 16 | 8192 | 2048 | qvq_lr | 2.5 | P32 | 0.0507 | 0.0515 | 10.592 | 104.73 | 0.354x | 0.505x | 0.000103 |
| mlp_down | down_proj | 1 | 8192 | 2048 | gptq_marlin | 4 | 128 | 0.0165 | 0.0168 | 2.032 | 523.91 | 1.000x | 1.542x | 0.000346 |
| mlp_down | down_proj | 2 | 8192 | 2048 | gptq_marlin | 4 | 128 | 0.0180 | 0.0184 | 3.732 | 481.02 | 1.000x | 1.378x | 0.000545 |
| mlp_down | down_proj | 4 | 8192 | 2048 | gptq_marlin | 4 | 128 | 0.0181 | 0.0186 | 7.410 | 477.63 | 1.000x | 1.402x | 0.000622 |
| mlp_down | down_proj | 8 | 8192 | 2048 | gptq_marlin | 4 | 128 | 0.0165 | 0.0171 | 16.241 | 523.40 | 1.000x | 1.521x | 0.000546 |
| mlp_down | down_proj | 16 | 8192 | 2048 | gptq_marlin | 4 | 128 | 0.0179 | 0.0182 | 29.959 | 482.74 | 1.000x | 1.428x | 0.000539 |
| mlp_down | down_proj | 1 | 8192 | 2048 | gptq_machete | 4 | 128 | 0.0255 | 0.0280 | 1.318 | 341.12 | 0.649x | 1.000x | 0.000346 |
| mlp_down | down_proj | 2 | 8192 | 2048 | gptq_machete | 4 | 128 | 0.0248 | 0.0268 | 2.708 | 350.37 | 0.726x | 1.000x | 0.000545 |
| mlp_down | down_proj | 4 | 8192 | 2048 | gptq_machete | 4 | 128 | 0.0254 | 0.0286 | 5.286 | 341.98 | 0.713x | 1.000x | 0.000622 |
| mlp_down | down_proj | 8 | 8192 | 2048 | gptq_machete | 4 | 128 | 0.0251 | 0.0282 | 10.679 | 345.46 | 0.658x | 1.000x | 0.000546 |
| mlp_down | down_proj | 16 | 8192 | 2048 | gptq_machete | 4 | 128 | 0.0256 | 0.0294 | 20.985 | 339.41 | 0.700x | 1.000x | 0.000539 |

The four shape rows preserve all seven Llama 3.2 1B projection roles: `q_proj/o_proj` (2048×2048), `k_proj/v_proj` (2048×512), `gate_proj/up_proj` (2048×8192), and `down_proj` (8192×2048).
