# QVQ LR versus W4 GPTQ kernels — Llama 3.2 1B shapes

This is a kernel-throughput comparison on the H100 host from PR #62. W4 GPTQ is a figurative performance baseline, not a quality-equivalent arm.

## Measurement contract

- Commit: `835329b09694c4b0dd9d6b8976227167aaf976ee`; benchmark SHA256: `7845a29f5e1d0445cbf5672f0c14fab016132ac0de3b9d22967b61ef244754e6`
- GPU: physical `1`, `NVIDIA H100`, PCI `00000000:44:00.0`, UUID `GPU-f5ea03cf-efa4-9807-7de5-b174957a1348`, CC `9.0`, 132 SMs
- Torch/CUDA: `2.15.0.dev20260828+cu130` / `13.0`; input dtype: `float16`
- M values: `[1, 2, 4, 8, 16, 32]`; warmup: `10`; measured launches: `60`
- QVQ: `qvq_v2b2_p32_lr`, rates `[2.0, 2.5, 3.0, 3.5]`, native `P32 is LR packing geometry, not a GPTQ affine scale group.`
- GPTQ: symmetric W4, group `128`, no activation order; Marlin and Machete use the same W4 source payload
- Latency is CUDA-event median/P95. Logical TFLOP/s is `2*M*K*N / median_ms`; payload GB/s is packed payload bytes divided by median latency.
- Every row passed its dense-reference shape/dtype/finite gate. QVQ uses `max_abs <= 2e-3`; GPTQ uses `atol=2e-2, rtol=2e-2`.

`xMarlin` and `xMachete` are baseline median latency divided by the row median, so values above 1.0x are faster.

| Shape | Roles | M | K | N | Kernel | W | Group | Median ms | P95 ms | Logical TFLOP/s | Payload GB/s | xMarlin | xMachete | Max abs |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| attn_qo | q_proj/o_proj | 1 | 2048 | 2048 | qvq_lr | 2 | P32 | 0.1379 | 0.1398 | 0.061 | 7.73 | 0.252x | 0.303x | 3.93e-06 |
| attn_qo | q_proj/o_proj | 2 | 2048 | 2048 | qvq_lr | 2 | P32 | 0.0704 | 0.0829 | 0.238 | 15.12 | 0.524x | 0.591x | 4.77e-06 |
| attn_qo | q_proj/o_proj | 4 | 2048 | 2048 | qvq_lr | 2 | P32 | 0.0643 | 0.0731 | 0.522 | 16.57 | 0.584x | 0.636x | 3.81e-06 |
| attn_qo | q_proj/o_proj | 8 | 2048 | 2048 | qvq_lr | 2 | P32 | 0.0643 | 0.0747 | 1.043 | 16.55 | 0.496x | 0.619x | 5.96e-06 |
| attn_qo | q_proj/o_proj | 16 | 2048 | 2048 | qvq_lr | 2 | P32 | 0.0662 | 0.0768 | 2.028 | 16.09 | 0.496x | 0.611x | 5.72e-06 |
| attn_qo | q_proj/o_proj | 32 | 2048 | 2048 | qvq_lr | 2 | P32 | 0.1388 | 0.1420 | 1.933 | 7.67 | 0.265x | 0.285x | 4.77e-06 |
| attn_qo | q_proj/o_proj | 1 | 2048 | 2048 | qvq_lr | 2.5 | P32 | 0.1238 | 0.1270 | 0.068 | 10.72 | 0.280x | 0.338x | 4.29e-06 |
| attn_qo | q_proj/o_proj | 2 | 2048 | 2048 | qvq_lr | 2.5 | P32 | 0.1250 | 0.1285 | 0.134 | 10.62 | 0.295x | 0.333x | 4.98e-06 |
| attn_qo | q_proj/o_proj | 4 | 2048 | 2048 | qvq_lr | 2.5 | P32 | 0.1252 | 0.1285 | 0.268 | 10.60 | 0.300x | 0.327x | 4.77e-06 |
| attn_qo | q_proj/o_proj | 8 | 2048 | 2048 | qvq_lr | 2.5 | P32 | 0.1257 | 0.1297 | 0.534 | 10.56 | 0.254x | 0.317x | 5.72e-06 |
| attn_qo | q_proj/o_proj | 16 | 2048 | 2048 | qvq_lr | 2.5 | P32 | 0.0809 | 0.1004 | 1.658 | 16.40 | 0.406x | 0.500x | 5.3e-06 |
| attn_qo | q_proj/o_proj | 32 | 2048 | 2048 | qvq_lr | 2.5 | P32 | 0.1396 | 0.1427 | 1.923 | 9.51 | 0.264x | 0.283x | 5.72e-06 |
| attn_qo | q_proj/o_proj | 1 | 2048 | 2048 | qvq_lr | 3 | P32 | 0.0581 | 0.0598 | 0.144 | 27.33 | 0.596x | 0.719x | 3.34e-06 |
| attn_qo | q_proj/o_proj | 2 | 2048 | 2048 | qvq_lr | 3 | P32 | 0.0575 | 0.0588 | 0.292 | 27.64 | 0.642x | 0.724x | 4.77e-06 |
| attn_qo | q_proj/o_proj | 4 | 2048 | 2048 | qvq_lr | 3 | P32 | 0.0579 | 0.0594 | 0.580 | 27.47 | 0.649x | 0.707x | 4.77e-06 |
| attn_qo | q_proj/o_proj | 8 | 2048 | 2048 | qvq_lr | 3 | P32 | 0.0592 | 0.0603 | 1.135 | 26.87 | 0.540x | 0.673x | 5.72e-06 |
| attn_qo | q_proj/o_proj | 16 | 2048 | 2048 | qvq_lr | 3 | P32 | 0.0636 | 0.0647 | 2.109 | 24.98 | 0.516x | 0.636x | 4.77e-06 |
| attn_qo | q_proj/o_proj | 32 | 2048 | 2048 | qvq_lr | 3 | P32 | 0.0741 | 0.0756 | 3.621 | 21.44 | 0.496x | 0.533x | 4.77e-06 |
| attn_qo | q_proj/o_proj | 1 | 2048 | 2048 | qvq_lr | 3.5 | P32 | 0.0647 | 0.0665 | 0.130 | 28.63 | 0.536x | 0.647x | 4.77e-06 |
| attn_qo | q_proj/o_proj | 2 | 2048 | 2048 | qvq_lr | 3.5 | P32 | 0.0580 | 0.0595 | 0.289 | 31.89 | 0.636x | 0.717x | 3.81e-06 |
| attn_qo | q_proj/o_proj | 4 | 2048 | 2048 | qvq_lr | 3.5 | P32 | 0.0584 | 0.0598 | 0.574 | 31.68 | 0.642x | 0.700x | 4.77e-06 |
| attn_qo | q_proj/o_proj | 8 | 2048 | 2048 | qvq_lr | 3.5 | P32 | 0.0598 | 0.0616 | 1.123 | 30.98 | 0.534x | 0.666x | 4.77e-06 |
| attn_qo | q_proj/o_proj | 16 | 2048 | 2048 | qvq_lr | 3.5 | P32 | 0.0748 | 0.0864 | 1.794 | 24.75 | 0.439x | 0.541x | 4.77e-06 |
| attn_qo | q_proj/o_proj | 32 | 2048 | 2048 | qvq_lr | 3.5 | P32 | 0.0755 | 0.0772 | 3.555 | 24.52 | 0.487x | 0.523x | 4.77e-06 |
| attn_qo | q_proj/o_proj | 1 | 2048 | 2048 | gptq_marlin | 4 | 128 | 0.0347 | 0.0487 | 0.242 | 62.38 | 1.000x | 1.206x | 0.000264 |
| attn_qo | q_proj/o_proj | 2 | 2048 | 2048 | gptq_marlin | 4 | 128 | 0.0369 | 0.0708 | 0.455 | 58.62 | 1.000x | 1.128x | 0.000229 |
| attn_qo | q_proj/o_proj | 4 | 2048 | 2048 | gptq_marlin | 4 | 128 | 0.0375 | 0.0781 | 0.894 | 57.64 | 1.000x | 1.090x | 0.000197 |
| attn_qo | q_proj/o_proj | 8 | 2048 | 2048 | gptq_marlin | 4 | 128 | 0.0319 | 0.0394 | 2.101 | 67.72 | 1.000x | 1.246x | 0.000242 |
| attn_qo | q_proj/o_proj | 16 | 2048 | 2048 | gptq_marlin | 4 | 128 | 0.0328 | 0.0444 | 4.086 | 65.84 | 1.000x | 1.232x | 0.00028 |
| attn_qo | q_proj/o_proj | 32 | 2048 | 2048 | gptq_marlin | 4 | 128 | 0.0368 | 0.0801 | 7.294 | 58.77 | 1.000x | 1.074x | 0.000269 |
| attn_qo | q_proj/o_proj | 1 | 2048 | 2048 | gptq_machete | 4 | 128 | 0.0418 | 0.0552 | 0.201 | 51.92 | 0.829x | 1.000x | 0.000264 |
| attn_qo | q_proj/o_proj | 2 | 2048 | 2048 | gptq_machete | 4 | 128 | 0.0416 | 0.0913 | 0.403 | 52.16 | 0.887x | 1.000x | 0.000229 |
| attn_qo | q_proj/o_proj | 4 | 2048 | 2048 | gptq_machete | 4 | 128 | 0.0409 | 0.0971 | 0.820 | 53.08 | 0.917x | 1.000x | 0.000197 |
| attn_qo | q_proj/o_proj | 8 | 2048 | 2048 | gptq_machete | 4 | 128 | 0.0398 | 0.0583 | 1.686 | 54.56 | 0.803x | 1.000x | 0.000242 |
| attn_qo | q_proj/o_proj | 16 | 2048 | 2048 | gptq_machete | 4 | 128 | 0.0405 | 0.0843 | 3.317 | 53.65 | 0.812x | 1.000x | 0.00028 |
| attn_qo | q_proj/o_proj | 32 | 2048 | 2048 | gptq_machete | 4 | 128 | 0.0395 | 0.0569 | 6.792 | 54.93 | 0.931x | 1.000x | 0.000269 |
| attn_kv | k_proj/v_proj | 1 | 2048 | 512 | qvq_lr | 2 | P32 | 0.0480 | 0.0537 | 0.044 | 5.55 | 0.670x | 0.828x | 3.34e-06 |
| attn_kv | k_proj/v_proj | 2 | 2048 | 512 | qvq_lr | 2 | P32 | 0.0347 | 0.0414 | 0.121 | 7.68 | 0.915x | 1.176x | 2.86e-06 |
| attn_kv | k_proj/v_proj | 4 | 2048 | 512 | qvq_lr | 2 | P32 | 0.0360 | 0.0429 | 0.233 | 7.40 | 0.865x | 1.099x | 4.77e-06 |
| attn_kv | k_proj/v_proj | 8 | 2048 | 512 | qvq_lr | 2 | P32 | 0.0347 | 0.0416 | 0.484 | 7.68 | 0.905x | 1.147x | 4.77e-06 |
| attn_kv | k_proj/v_proj | 16 | 2048 | 512 | qvq_lr | 2 | P32 | 0.0355 | 0.0441 | 0.945 | 7.50 | 0.880x | 1.131x | 4.29e-06 |
| attn_kv | k_proj/v_proj | 32 | 2048 | 512 | qvq_lr | 2 | P32 | 0.0501 | 0.0517 | 1.339 | 5.31 | 0.715x | 0.798x | 5.72e-06 |
| attn_kv | k_proj/v_proj | 1 | 2048 | 512 | qvq_lr | 2.5 | P32 | 0.0433 | 0.0541 | 0.048 | 7.67 | 0.743x | 0.918x | 2.38e-06 |
| attn_kv | k_proj/v_proj | 2 | 2048 | 512 | qvq_lr | 2.5 | P32 | 0.0400 | 0.0455 | 0.105 | 8.29 | 0.792x | 1.019x | 3.81e-06 |
| attn_kv | k_proj/v_proj | 4 | 2048 | 512 | qvq_lr | 2.5 | P32 | 0.0405 | 0.0487 | 0.207 | 8.18 | 0.768x | 0.976x | 3.81e-06 |
| attn_kv | k_proj/v_proj | 8 | 2048 | 512 | qvq_lr | 2.5 | P32 | 0.0405 | 0.0501 | 0.414 | 8.19 | 0.775x | 0.982x | 3.81e-06 |
| attn_kv | k_proj/v_proj | 16 | 2048 | 512 | qvq_lr | 2.5 | P32 | 0.0361 | 0.0464 | 0.929 | 9.18 | 0.865x | 1.112x | 4.77e-06 |
| attn_kv | k_proj/v_proj | 32 | 2048 | 512 | qvq_lr | 2.5 | P32 | 0.0681 | 0.0950 | 0.985 | 4.87 | 0.526x | 0.587x | 5.25e-06 |
| attn_kv | k_proj/v_proj | 1 | 2048 | 512 | qvq_lr | 3 | P32 | 0.0385 | 0.0587 | 0.055 | 10.33 | 0.836x | 1.032x | 1.91e-06 |
| attn_kv | k_proj/v_proj | 2 | 2048 | 512 | qvq_lr | 3 | P32 | 0.0355 | 0.0451 | 0.118 | 11.19 | 0.893x | 1.148x | 3.81e-06 |
| attn_kv | k_proj/v_proj | 4 | 2048 | 512 | qvq_lr | 3 | P32 | 0.0404 | 0.0782 | 0.208 | 9.83 | 0.770x | 0.979x | 4.77e-06 |
| attn_kv | k_proj/v_proj | 8 | 2048 | 512 | qvq_lr | 3 | P32 | 0.0366 | 0.0504 | 0.459 | 10.86 | 0.858x | 1.088x | 4.77e-06 |
| attn_kv | k_proj/v_proj | 16 | 2048 | 512 | qvq_lr | 3 | P32 | 0.0352 | 0.0482 | 0.954 | 11.29 | 0.888x | 1.142x | 3.81e-06 |
| attn_kv | k_proj/v_proj | 32 | 2048 | 512 | qvq_lr | 3 | P32 | 0.0356 | 0.0859 | 1.883 | 11.15 | 1.005x | 1.122x | 4.77e-06 |
| attn_kv | k_proj/v_proj | 1 | 2048 | 512 | qvq_lr | 3.5 | P32 | 0.0390 | 0.0540 | 0.054 | 11.86 | 0.824x | 1.018x | 2.38e-06 |
| attn_kv | k_proj/v_proj | 2 | 2048 | 512 | qvq_lr | 3.5 | P32 | 0.0359 | 0.0450 | 0.117 | 12.89 | 0.883x | 1.135x | 4.29e-06 |
| attn_kv | k_proj/v_proj | 4 | 2048 | 512 | qvq_lr | 3.5 | P32 | 0.0365 | 0.0445 | 0.230 | 12.68 | 0.853x | 1.083x | 3.81e-06 |
| attn_kv | k_proj/v_proj | 8 | 2048 | 512 | qvq_lr | 3.5 | P32 | 0.0354 | 0.0440 | 0.474 | 13.07 | 0.886x | 1.123x | 3.81e-06 |
| attn_kv | k_proj/v_proj | 16 | 2048 | 512 | qvq_lr | 3.5 | P32 | 0.0356 | 0.0457 | 0.942 | 12.99 | 0.877x | 1.128x | 4.29e-06 |
| attn_kv | k_proj/v_proj | 32 | 2048 | 512 | qvq_lr | 3.5 | P32 | 0.0351 | 0.0455 | 1.914 | 13.20 | 1.022x | 1.141x | 4.77e-06 |
| attn_kv | k_proj/v_proj | 1 | 2048 | 512 | gptq_marlin | 4 | 128 | 0.0321 | 0.0393 | 0.065 | 16.82 | 1.000x | 1.235x | 0.000167 |
| attn_kv | k_proj/v_proj | 2 | 2048 | 512 | gptq_marlin | 4 | 128 | 0.0317 | 0.0391 | 0.132 | 17.06 | 1.000x | 1.286x | 0.000155 |
| attn_kv | k_proj/v_proj | 4 | 2048 | 512 | gptq_marlin | 4 | 128 | 0.0311 | 0.0473 | 0.269 | 17.36 | 1.000x | 1.270x | 0.000167 |
| attn_kv | k_proj/v_proj | 8 | 2048 | 512 | gptq_marlin | 4 | 128 | 0.0314 | 0.0412 | 0.534 | 17.22 | 1.000x | 1.268x | 0.000206 |
| attn_kv | k_proj/v_proj | 16 | 2048 | 512 | gptq_marlin | 4 | 128 | 0.0312 | 0.0428 | 1.074 | 17.30 | 1.000x | 1.286x | 0.000194 |
| attn_kv | k_proj/v_proj | 32 | 2048 | 512 | gptq_marlin | 4 | 128 | 0.0358 | 0.0378 | 1.873 | 15.09 | 1.000x | 1.116x | 0.000265 |
| attn_kv | k_proj/v_proj | 1 | 2048 | 512 | gptq_machete | 4 | 128 | 0.0397 | 0.0465 | 0.053 | 13.82 | 0.809x | 1.000x | 0.000167 |
| attn_kv | k_proj/v_proj | 2 | 2048 | 512 | gptq_machete | 4 | 128 | 0.0408 | 0.0479 | 0.103 | 13.46 | 0.777x | 1.000x | 0.000155 |
| attn_kv | k_proj/v_proj | 4 | 2048 | 512 | gptq_machete | 4 | 128 | 0.0396 | 0.0486 | 0.212 | 13.88 | 0.787x | 1.000x | 0.000167 |
| attn_kv | k_proj/v_proj | 8 | 2048 | 512 | gptq_machete | 4 | 128 | 0.0398 | 0.0481 | 0.422 | 13.79 | 0.789x | 1.000x | 0.000206 |
| attn_kv | k_proj/v_proj | 16 | 2048 | 512 | gptq_machete | 4 | 128 | 0.0402 | 0.0461 | 0.835 | 13.66 | 0.778x | 1.000x | 0.000194 |
| attn_kv | k_proj/v_proj | 32 | 2048 | 512 | gptq_machete | 4 | 128 | 0.0400 | 0.0464 | 1.678 | 13.73 | 0.896x | 1.000x | 0.000265 |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | qvq_lr | 2 | P32 | 0.4833 | 0.4853 | 0.069 | 8.81 | 0.065x | 0.084x | 7.15e-06 |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | qvq_lr | 2 | P32 | 0.0907 | 0.0926 | 0.740 | 46.99 | 0.335x | 0.451x | 4.77e-06 |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | qvq_lr | 2 | P32 | 0.0909 | 0.0927 | 1.476 | 46.85 | 0.329x | 0.449x | 5.01e-06 |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | qvq_lr | 2 | P32 | 0.0924 | 0.0942 | 2.905 | 46.10 | 0.326x | 0.437x | 5.72e-06 |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | qvq_lr | 2 | P32 | 0.1318 | 0.1334 | 4.074 | 32.32 | 0.232x | 0.304x | 5.72e-06 |
| mlp_gate_up | gate_proj/up_proj | 32 | 2048 | 8192 | qvq_lr | 2 | P32 | 0.4944 | 0.4975 | 2.172 | 8.62 | 0.063x | 0.081x | 1.19e-05 |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | qvq_lr | 2.5 | P32 | 0.4861 | 0.4875 | 0.069 | 10.92 | 0.064x | 0.083x | 6.08e-06 |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | qvq_lr | 2.5 | P32 | 0.4579 | 0.4630 | 0.147 | 11.59 | 0.066x | 0.089x | 4.77e-06 |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | qvq_lr | 2.5 | P32 | 0.4598 | 0.4639 | 0.292 | 11.55 | 0.065x | 0.089x | 4.77e-06 |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | qvq_lr | 2.5 | P32 | 0.4600 | 0.4650 | 0.584 | 11.54 | 0.065x | 0.088x | 4.89e-06 |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | qvq_lr | 2.5 | P32 | 0.1332 | 0.1359 | 4.029 | 39.84 | 0.229x | 0.301x | 5.72e-06 |
| mlp_gate_up | gate_proj/up_proj | 32 | 2048 | 8192 | qvq_lr | 2.5 | P32 | 0.4956 | 0.5019 | 2.166 | 10.71 | 0.063x | 0.081x | 1.34e-05 |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | qvq_lr | 3 | P32 | 0.2077 | 0.2090 | 0.162 | 30.61 | 0.150x | 0.195x | 6.68e-06 |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | qvq_lr | 3 | P32 | 0.1956 | 0.1981 | 0.343 | 32.51 | 0.155x | 0.209x | 4.77e-06 |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | qvq_lr | 3 | P32 | 0.1967 | 0.1993 | 0.682 | 32.32 | 0.152x | 0.208x | 4.89e-06 |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | qvq_lr | 3 | P32 | 0.1985 | 0.2026 | 1.352 | 32.03 | 0.152x | 0.203x | 4.77e-06 |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | qvq_lr | 3 | P32 | 0.2073 | 0.2104 | 2.590 | 30.66 | 0.147x | 0.193x | 6.68e-06 |
| mlp_gate_up | gate_proj/up_proj | 32 | 2048 | 8192 | qvq_lr | 3 | P32 | 0.2254 | 0.2275 | 4.764 | 28.21 | 0.139x | 0.178x | 1.34e-05 |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | qvq_lr | 3.5 | P32 | 0.2382 | 0.2399 | 0.141 | 31.09 | 0.131x | 0.170x | 6.2e-06 |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | qvq_lr | 3.5 | P32 | 0.1973 | 0.2016 | 0.340 | 37.53 | 0.154x | 0.207x | 4.77e-06 |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | qvq_lr | 3.5 | P32 | 0.1991 | 0.2019 | 0.674 | 37.20 | 0.150x | 0.205x | 5.72e-06 |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | qvq_lr | 3.5 | P32 | 0.2021 | 0.2068 | 1.328 | 36.65 | 0.149x | 0.200x | 5.96e-06 |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | qvq_lr | 3.5 | P32 | 0.2083 | 0.2107 | 2.578 | 35.56 | 0.147x | 0.193x | 5.25e-06 |
| mlp_gate_up | gate_proj/up_proj | 32 | 2048 | 8192 | qvq_lr | 3.5 | P32 | 0.2299 | 0.2320 | 4.671 | 32.21 | 0.136x | 0.174x | 1.36e-05 |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | gptq_marlin | 4 | 128 | 0.0312 | 0.0380 | 1.075 | 277.27 | 1.000x | 1.295x | 0.000272 |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | gptq_marlin | 4 | 128 | 0.0304 | 0.0408 | 2.210 | 284.86 | 1.000x | 1.346x | 0.000299 |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | gptq_marlin | 4 | 128 | 0.0299 | 0.0391 | 4.488 | 289.28 | 1.000x | 1.366x | 0.000274 |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | gptq_marlin | 4 | 128 | 0.0301 | 0.0356 | 8.910 | 287.13 | 1.000x | 1.340x | 0.000278 |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | gptq_marlin | 4 | 128 | 0.0306 | 0.0369 | 17.559 | 282.93 | 1.000x | 1.311x | 0.000327 |
| mlp_gate_up | gate_proj/up_proj | 32 | 2048 | 8192 | gptq_marlin | 4 | 128 | 0.0312 | 0.0379 | 34.362 | 276.84 | 1.000x | 1.284x | 0.000315 |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | gptq_machete | 4 | 128 | 0.0404 | 0.0452 | 0.831 | 214.33 | 0.772x | 1.000x | 0.000272 |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | gptq_machete | 4 | 128 | 0.0409 | 0.0476 | 1.642 | 211.90 | 0.743x | 1.000x | 0.000299 |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | gptq_machete | 4 | 128 | 0.0408 | 0.0493 | 3.286 | 211.98 | 0.732x | 1.000x | 0.000274 |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | gptq_machete | 4 | 128 | 0.0404 | 0.0502 | 6.650 | 214.50 | 0.746x | 1.000x | 0.000278 |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | gptq_machete | 4 | 128 | 0.0401 | 0.0469 | 13.390 | 215.96 | 0.763x | 1.000x | 0.000327 |
| mlp_gate_up | gate_proj/up_proj | 32 | 2048 | 8192 | gptq_machete | 4 | 128 | 0.0401 | 0.0454 | 26.769 | 215.87 | 0.779x | 1.000x | 0.000315 |
| mlp_down | down_proj | 1 | 8192 | 2048 | qvq_lr | 2 | P32 | 0.5211 | 0.5236 | 0.064 | 8.17 | 0.050x | 0.065x | 1.34e-05 |
| mlp_down | down_proj | 2 | 8192 | 2048 | qvq_lr | 2 | P32 | 0.0946 | 0.1120 | 0.709 | 45.01 | 0.272x | 0.375x | 9.54e-06 |
| mlp_down | down_proj | 4 | 8192 | 2048 | qvq_lr | 2 | P32 | 0.0960 | 0.0992 | 1.398 | 44.37 | 0.269x | 0.349x | 1.05e-05 |
| mlp_down | down_proj | 8 | 8192 | 2048 | qvq_lr | 2 | P32 | 0.0972 | 0.1027 | 2.761 | 43.81 | 0.327x | 0.345x | 1.67e-05 |
| mlp_down | down_proj | 16 | 8192 | 2048 | qvq_lr | 2 | P32 | 0.1350 | 0.1379 | 3.977 | 31.55 | 0.192x | 0.247x | 1.34e-05 |
| mlp_down | down_proj | 32 | 8192 | 2048 | qvq_lr | 2 | P32 | 0.4972 | 0.5023 | 2.160 | 8.57 | 0.075x | 0.071x | 1.19e-05 |
| mlp_down | down_proj | 1 | 8192 | 2048 | qvq_lr | 2.5 | P32 | 0.4610 | 0.4645 | 0.073 | 11.52 | 0.057x | 0.074x | 1.25e-05 |
| mlp_down | down_proj | 2 | 8192 | 2048 | qvq_lr | 2.5 | P32 | 0.4602 | 0.4661 | 0.146 | 11.53 | 0.056x | 0.077x | 9.54e-06 |
| mlp_down | down_proj | 4 | 8192 | 2048 | qvq_lr | 2.5 | P32 | 0.4609 | 0.4659 | 0.291 | 11.52 | 0.056x | 0.073x | 1.14e-05 |
| mlp_down | down_proj | 8 | 8192 | 2048 | qvq_lr | 2.5 | P32 | 0.4629 | 0.4690 | 0.580 | 11.47 | 0.069x | 0.072x | 1.72e-05 |
| mlp_down | down_proj | 16 | 8192 | 2048 | qvq_lr | 2.5 | P32 | 0.1394 | 0.1436 | 3.850 | 38.07 | 0.186x | 0.239x | 1.34e-05 |
| mlp_down | down_proj | 32 | 8192 | 2048 | qvq_lr | 2.5 | P32 | 0.4996 | 0.5059 | 2.149 | 10.63 | 0.074x | 0.071x | 1.34e-05 |
| mlp_down | down_proj | 1 | 8192 | 2048 | qvq_lr | 3 | P32 | 0.1992 | 0.2004 | 0.168 | 31.91 | 0.131x | 0.171x | 1.41e-05 |
| mlp_down | down_proj | 2 | 8192 | 2048 | qvq_lr | 3 | P32 | 0.1971 | 0.2004 | 0.341 | 32.25 | 0.131x | 0.180x | 1.14e-05 |
| mlp_down | down_proj | 4 | 8192 | 2048 | qvq_lr | 3 | P32 | 0.1973 | 0.1998 | 0.680 | 32.21 | 0.131x | 0.170x | 1.34e-05 |
| mlp_down | down_proj | 8 | 8192 | 2048 | qvq_lr | 3 | P32 | 0.2011 | 0.2031 | 1.335 | 31.61 | 0.158x | 0.167x | 1.86e-05 |
| mlp_down | down_proj | 16 | 8192 | 2048 | qvq_lr | 3 | P32 | 0.2090 | 0.2109 | 2.568 | 30.41 | 0.124x | 0.160x | 1.53e-05 |
| mlp_down | down_proj | 32 | 8192 | 2048 | qvq_lr | 3 | P32 | 0.2277 | 0.2286 | 4.716 | 27.92 | 0.163x | 0.156x | 1.24e-05 |
| mlp_down | down_proj | 1 | 8192 | 2048 | qvq_lr | 3.5 | P32 | 0.2244 | 0.2256 | 0.150 | 33.00 | 0.116x | 0.152x | 1.14e-05 |
| mlp_down | down_proj | 2 | 8192 | 2048 | qvq_lr | 3.5 | P32 | 0.1994 | 0.2024 | 0.337 | 37.14 | 0.129x | 0.178x | 1.24e-05 |
| mlp_down | down_proj | 4 | 8192 | 2048 | qvq_lr | 3.5 | P32 | 0.2009 | 0.2043 | 0.668 | 36.86 | 0.128x | 0.167x | 1.34e-05 |
| mlp_down | down_proj | 8 | 8192 | 2048 | qvq_lr | 3.5 | P32 | 0.2031 | 0.2060 | 1.322 | 36.47 | 0.157x | 0.165x | 1.72e-05 |
| mlp_down | down_proj | 16 | 8192 | 2048 | qvq_lr | 3.5 | P32 | 0.2101 | 0.2135 | 2.556 | 35.25 | 0.123x | 0.159x | 1.34e-05 |
| mlp_down | down_proj | 32 | 8192 | 2048 | qvq_lr | 3.5 | P32 | 0.2330 | 0.2342 | 4.608 | 31.78 | 0.160x | 0.153x | 1.29e-05 |
| mlp_down | down_proj | 1 | 8192 | 2048 | gptq_marlin | 4 | 128 | 0.0261 | 0.0518 | 1.283 | 330.89 | 1.000x | 1.302x | 0.000346 |
| mlp_down | down_proj | 2 | 8192 | 2048 | gptq_marlin | 4 | 128 | 0.0258 | 0.0464 | 2.604 | 335.61 | 1.000x | 1.379x | 0.000545 |
| mlp_down | down_proj | 4 | 8192 | 2048 | gptq_marlin | 4 | 128 | 0.0258 | 0.0291 | 5.201 | 335.20 | 1.000x | 1.296x | 0.000622 |
| mlp_down | down_proj | 8 | 8192 | 2048 | gptq_marlin | 4 | 128 | 0.0318 | 0.0499 | 8.439 | 271.97 | 1.000x | 1.055x | 0.000546 |
| mlp_down | down_proj | 16 | 8192 | 2048 | gptq_marlin | 4 | 128 | 0.0259 | 0.0305 | 20.738 | 334.16 | 1.000x | 1.289x | 0.000539 |
| mlp_down | down_proj | 32 | 8192 | 2048 | gptq_marlin | 4 | 128 | 0.0372 | 0.0376 | 28.889 | 232.75 | 1.000x | 0.956x | 0.00057 |
| mlp_down | down_proj | 1 | 8192 | 2048 | gptq_machete | 4 | 128 | 0.0340 | 0.0645 | 0.986 | 255.16 | 0.768x | 1.000x | 0.000346 |
| mlp_down | down_proj | 2 | 8192 | 2048 | gptq_machete | 4 | 128 | 0.0355 | 0.0654 | 1.888 | 244.36 | 0.725x | 1.000x | 0.000545 |
| mlp_down | down_proj | 4 | 8192 | 2048 | gptq_machete | 4 | 128 | 0.0335 | 0.0587 | 4.012 | 259.55 | 0.771x | 1.000x | 0.000622 |
| mlp_down | down_proj | 8 | 8192 | 2048 | gptq_machete | 4 | 128 | 0.0336 | 0.0398 | 8.001 | 258.81 | 0.948x | 1.000x | 0.000546 |
| mlp_down | down_proj | 16 | 8192 | 2048 | gptq_machete | 4 | 128 | 0.0334 | 0.0658 | 16.086 | 260.17 | 0.776x | 1.000x | 0.000539 |
| mlp_down | down_proj | 32 | 8192 | 2048 | gptq_machete | 4 | 128 | 0.0355 | 0.0648 | 30.216 | 244.36 | 1.046x | 1.000x | 0.00057 |

The four shape rows preserve all seven Llama 3.2 1B projection roles: `q_proj/o_proj` (2048×2048), `k_proj/v_proj` (2048×512), `gate_proj/up_proj` (2048×8192), and `down_proj` (8192×2048).

## Focused Nsight Compute check

Nsight Compute 2026.2.1 successfully captured the native `qvq_gemv_local_ring_kernel` on the same H100 (permission check fixed). This is a profiler sanity check, not a replacement for the CUDA-event matrix above.

| Shape | M | K | N | Kernel | W | Median ms | Logical TFLOP/s | xMarlin | xMachete |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| mlp_gate_up | 16 | 2048 | 8192 | qvq_lr | 3 | 0.2446 | 2.195 | 0.454x | 0.633x |

Capture command: `ncu --set basic --target-processes all --kernel-name-base function --kernel-name 'regex:.*qvq.*' --launch-count 1` with the benchmark restricted to `mlp_gate_up`, `M=16`, `W3`.
