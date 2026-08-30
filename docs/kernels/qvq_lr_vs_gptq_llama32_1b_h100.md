# QVQ LR versus W4 GPTQ kernels — Llama 3.2 1B shapes

This is a kernel-throughput comparison on the H100 host from PR #62. W4 GPTQ is a figurative performance baseline, not a quality-equivalent arm.

## Measurement contract

- Commit: `4ba8642ca8c1b314d4041339d4b1f3edc286a0ce`; benchmark SHA256: `7845a29f5e1d0445cbf5672f0c14fab016132ac0de3b9d22967b61ef244754e6`
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
| attn_qo | q_proj/o_proj | 1 | 2048 | 2048 | qvq_lr | 2 | P32 | 0.1356 | 0.1366 | 0.062 | 7.85 | 0.258x | 0.319x | 3.93e-06 |
| attn_qo | q_proj/o_proj | 2 | 2048 | 2048 | qvq_lr | 2 | P32 | 0.0299 | 0.0340 | 0.562 | 35.67 | 1.217x | 1.446x | 4.77e-06 |
| attn_qo | q_proj/o_proj | 4 | 2048 | 2048 | qvq_lr | 2 | P32 | 0.0306 | 0.0312 | 1.096 | 34.78 | 1.182x | 1.399x | 3.81e-06 |
| attn_qo | q_proj/o_proj | 8 | 2048 | 2048 | qvq_lr | 2 | P32 | 0.0322 | 0.0324 | 2.087 | 33.11 | 1.089x | 1.347x | 5.96e-06 |
| attn_qo | q_proj/o_proj | 16 | 2048 | 2048 | qvq_lr | 2 | P32 | 0.0437 | 0.0440 | 3.073 | 24.38 | 0.818x | 0.984x | 5.72e-06 |
| attn_qo | q_proj/o_proj | 32 | 2048 | 2048 | qvq_lr | 2 | P32 | 0.1362 | 0.1382 | 1.971 | 7.82 | 0.273x | 0.318x | 4.77e-06 |
| attn_qo | q_proj/o_proj | 1 | 2048 | 2048 | qvq_lr | 2.5 | P32 | 0.1209 | 0.1221 | 0.069 | 10.98 | 0.289x | 0.358x | 4.29e-06 |
| attn_qo | q_proj/o_proj | 2 | 2048 | 2048 | qvq_lr | 2.5 | P32 | 0.1214 | 0.1226 | 0.138 | 10.94 | 0.299x | 0.356x | 4.98e-06 |
| attn_qo | q_proj/o_proj | 4 | 2048 | 2048 | qvq_lr | 2.5 | P32 | 0.1218 | 0.1230 | 0.275 | 10.89 | 0.297x | 0.352x | 4.77e-06 |
| attn_qo | q_proj/o_proj | 8 | 2048 | 2048 | qvq_lr | 2.5 | P32 | 0.1228 | 0.1238 | 0.546 | 10.81 | 0.285x | 0.353x | 5.72e-06 |
| attn_qo | q_proj/o_proj | 16 | 2048 | 2048 | qvq_lr | 2.5 | P32 | 0.0446 | 0.0449 | 3.009 | 29.75 | 0.801x | 0.963x | 5.3e-06 |
| attn_qo | q_proj/o_proj | 32 | 2048 | 2048 | qvq_lr | 2.5 | P32 | 0.1365 | 0.1380 | 1.967 | 9.72 | 0.272x | 0.317x | 5.72e-06 |
| attn_qo | q_proj/o_proj | 1 | 2048 | 2048 | qvq_lr | 3 | P32 | 0.0564 | 0.0570 | 0.149 | 28.19 | 0.620x | 0.768x | 3.34e-06 |
| attn_qo | q_proj/o_proj | 2 | 2048 | 2048 | qvq_lr | 3 | P32 | 0.0557 | 0.0567 | 0.301 | 28.51 | 0.652x | 0.774x | 4.77e-06 |
| attn_qo | q_proj/o_proj | 4 | 2048 | 2048 | qvq_lr | 3 | P32 | 0.0564 | 0.0573 | 0.595 | 28.17 | 0.642x | 0.760x | 4.77e-06 |
| attn_qo | q_proj/o_proj | 8 | 2048 | 2048 | qvq_lr | 3 | P32 | 0.0578 | 0.0593 | 1.161 | 27.50 | 0.606x | 0.749x | 5.72e-06 |
| attn_qo | q_proj/o_proj | 16 | 2048 | 2048 | qvq_lr | 3 | P32 | 0.0618 | 0.0626 | 2.173 | 25.73 | 0.578x | 0.696x | 4.77e-06 |
| attn_qo | q_proj/o_proj | 32 | 2048 | 2048 | qvq_lr | 3 | P32 | 0.0729 | 0.0733 | 3.683 | 21.81 | 0.510x | 0.594x | 4.77e-06 |
| attn_qo | q_proj/o_proj | 1 | 2048 | 2048 | qvq_lr | 3.5 | P32 | 0.0628 | 0.0633 | 0.134 | 29.49 | 0.557x | 0.690x | 4.77e-06 |
| attn_qo | q_proj/o_proj | 2 | 2048 | 2048 | qvq_lr | 3.5 | P32 | 0.0563 | 0.0568 | 0.298 | 32.88 | 0.645x | 0.767x | 3.81e-06 |
| attn_qo | q_proj/o_proj | 4 | 2048 | 2048 | qvq_lr | 3.5 | P32 | 0.0567 | 0.0571 | 0.592 | 32.64 | 0.638x | 0.755x | 4.77e-06 |
| attn_qo | q_proj/o_proj | 8 | 2048 | 2048 | qvq_lr | 3.5 | P32 | 0.0576 | 0.0582 | 1.165 | 32.13 | 0.608x | 0.752x | 4.77e-06 |
| attn_qo | q_proj/o_proj | 16 | 2048 | 2048 | qvq_lr | 3.5 | P32 | 0.0622 | 0.0629 | 2.158 | 29.76 | 0.574x | 0.691x | 4.77e-06 |
| attn_qo | q_proj/o_proj | 32 | 2048 | 2048 | qvq_lr | 3.5 | P32 | 0.0739 | 0.0745 | 3.631 | 25.04 | 0.503x | 0.585x | 4.77e-06 |
| attn_qo | q_proj/o_proj | 1 | 2048 | 2048 | gptq_marlin | 4 | 128 | 0.0350 | 0.0400 | 0.240 | 61.83 | 1.000x | 1.238x | 0.000264 |
| attn_qo | q_proj/o_proj | 2 | 2048 | 2048 | gptq_marlin | 4 | 128 | 0.0363 | 0.0395 | 0.462 | 59.52 | 1.000x | 1.188x | 0.000229 |
| attn_qo | q_proj/o_proj | 4 | 2048 | 2048 | gptq_marlin | 4 | 128 | 0.0362 | 0.0397 | 0.927 | 59.73 | 1.000x | 1.183x | 0.000197 |
| attn_qo | q_proj/o_proj | 8 | 2048 | 2048 | gptq_marlin | 4 | 128 | 0.0350 | 0.0389 | 1.916 | 61.75 | 1.000x | 1.237x | 0.000242 |
| attn_qo | q_proj/o_proj | 16 | 2048 | 2048 | gptq_marlin | 4 | 128 | 0.0357 | 0.0398 | 3.758 | 60.56 | 1.000x | 1.203x | 0.00028 |
| attn_qo | q_proj/o_proj | 32 | 2048 | 2048 | gptq_marlin | 4 | 128 | 0.0372 | 0.0410 | 7.222 | 58.19 | 1.000x | 1.164x | 0.000269 |
| attn_qo | q_proj/o_proj | 1 | 2048 | 2048 | gptq_machete | 4 | 128 | 0.0433 | 0.0488 | 0.194 | 50.14 | 0.808x | 1.000x | 0.000264 |
| attn_qo | q_proj/o_proj | 2 | 2048 | 2048 | gptq_machete | 4 | 128 | 0.0432 | 0.0466 | 0.389 | 50.29 | 0.842x | 1.000x | 0.000229 |
| attn_qo | q_proj/o_proj | 4 | 2048 | 2048 | gptq_machete | 4 | 128 | 0.0428 | 0.0463 | 0.783 | 50.66 | 0.845x | 1.000x | 0.000197 |
| attn_qo | q_proj/o_proj | 8 | 2048 | 2048 | gptq_machete | 4 | 128 | 0.0433 | 0.0470 | 1.549 | 50.12 | 0.809x | 1.000x | 0.000242 |
| attn_qo | q_proj/o_proj | 16 | 2048 | 2048 | gptq_machete | 4 | 128 | 0.0430 | 0.0455 | 3.124 | 50.53 | 0.831x | 1.000x | 0.00028 |
| attn_qo | q_proj/o_proj | 32 | 2048 | 2048 | gptq_machete | 4 | 128 | 0.0433 | 0.0458 | 6.205 | 50.18 | 0.859x | 1.000x | 0.000269 |
| attn_kv | k_proj/v_proj | 1 | 2048 | 512 | qvq_lr | 2 | P32 | 0.0458 | 0.0467 | 0.046 | 5.81 | 0.822x | 0.940x | 3.34e-06 |
| attn_kv | k_proj/v_proj | 2 | 2048 | 512 | qvq_lr | 2 | P32 | 0.0363 | 0.0396 | 0.116 | 7.33 | 1.025x | 1.180x | 2.86e-06 |
| attn_kv | k_proj/v_proj | 4 | 2048 | 512 | qvq_lr | 2 | P32 | 0.0367 | 0.0388 | 0.228 | 7.25 | 1.027x | 1.177x | 4.77e-06 |
| attn_kv | k_proj/v_proj | 8 | 2048 | 512 | qvq_lr | 2 | P32 | 0.0378 | 0.0409 | 0.444 | 7.04 | 0.989x | 1.135x | 4.77e-06 |
| attn_kv | k_proj/v_proj | 16 | 2048 | 512 | qvq_lr | 2 | P32 | 0.0410 | 0.0435 | 0.818 | 6.49 | 0.907x | 1.049x | 4.29e-06 |
| attn_kv | k_proj/v_proj | 32 | 2048 | 512 | qvq_lr | 2 | P32 | 0.0480 | 0.0487 | 1.399 | 5.55 | 0.780x | 0.908x | 5.72e-06 |
| attn_kv | k_proj/v_proj | 1 | 2048 | 512 | qvq_lr | 2.5 | P32 | 0.0421 | 0.0484 | 0.050 | 7.88 | 0.894x | 1.022x | 2.38e-06 |
| attn_kv | k_proj/v_proj | 2 | 2048 | 512 | qvq_lr | 2.5 | P32 | 0.0428 | 0.0485 | 0.098 | 7.76 | 0.871x | 1.002x | 3.81e-06 |
| attn_kv | k_proj/v_proj | 4 | 2048 | 512 | qvq_lr | 2.5 | P32 | 0.0421 | 0.0467 | 0.199 | 7.88 | 0.896x | 1.027x | 3.81e-06 |
| attn_kv | k_proj/v_proj | 8 | 2048 | 512 | qvq_lr | 2.5 | P32 | 0.0424 | 0.0469 | 0.396 | 7.83 | 0.882x | 1.012x | 3.81e-06 |
| attn_kv | k_proj/v_proj | 16 | 2048 | 512 | qvq_lr | 2.5 | P32 | 0.0417 | 0.0455 | 0.805 | 7.96 | 0.892x | 1.032x | 4.77e-06 |
| attn_kv | k_proj/v_proj | 32 | 2048 | 512 | qvq_lr | 2.5 | P32 | 0.0472 | 0.0486 | 1.421 | 7.03 | 0.793x | 0.922x | 5.25e-06 |
| attn_kv | k_proj/v_proj | 1 | 2048 | 512 | qvq_lr | 3 | P32 | 0.0413 | 0.0469 | 0.051 | 9.61 | 0.912x | 1.042x | 1.91e-06 |
| attn_kv | k_proj/v_proj | 2 | 2048 | 512 | qvq_lr | 3 | P32 | 0.0411 | 0.0451 | 0.102 | 9.67 | 0.906x | 1.043x | 3.81e-06 |
| attn_kv | k_proj/v_proj | 4 | 2048 | 512 | qvq_lr | 3 | P32 | 0.0410 | 0.0453 | 0.205 | 9.70 | 0.921x | 1.055x | 4.77e-06 |
| attn_kv | k_proj/v_proj | 8 | 2048 | 512 | qvq_lr | 3 | P32 | 0.0408 | 0.0445 | 0.411 | 9.74 | 0.916x | 1.051x | 4.77e-06 |
| attn_kv | k_proj/v_proj | 16 | 2048 | 512 | qvq_lr | 3 | P32 | 0.0405 | 0.0435 | 0.828 | 9.80 | 0.918x | 1.061x | 3.81e-06 |
| attn_kv | k_proj/v_proj | 32 | 2048 | 512 | qvq_lr | 3 | P32 | 0.0425 | 0.0477 | 1.580 | 9.35 | 0.881x | 1.025x | 4.77e-06 |
| attn_kv | k_proj/v_proj | 1 | 2048 | 512 | qvq_lr | 3.5 | P32 | 0.0424 | 0.0490 | 0.049 | 10.91 | 0.888x | 1.015x | 2.38e-06 |
| attn_kv | k_proj/v_proj | 2 | 2048 | 512 | qvq_lr | 3.5 | P32 | 0.0422 | 0.0450 | 0.099 | 10.98 | 0.883x | 1.016x | 4.29e-06 |
| attn_kv | k_proj/v_proj | 4 | 2048 | 512 | qvq_lr | 3.5 | P32 | 0.0421 | 0.0458 | 0.199 | 11.00 | 0.896x | 1.027x | 3.81e-06 |
| attn_kv | k_proj/v_proj | 8 | 2048 | 512 | qvq_lr | 3.5 | P32 | 0.0428 | 0.0462 | 0.392 | 10.83 | 0.874x | 1.003x | 3.81e-06 |
| attn_kv | k_proj/v_proj | 16 | 2048 | 512 | qvq_lr | 3.5 | P32 | 0.0418 | 0.0456 | 0.803 | 11.07 | 0.890x | 1.029x | 4.29e-06 |
| attn_kv | k_proj/v_proj | 32 | 2048 | 512 | qvq_lr | 3.5 | P32 | 0.0425 | 0.0461 | 1.578 | 10.88 | 0.880x | 1.024x | 4.77e-06 |
| attn_kv | k_proj/v_proj | 1 | 2048 | 512 | gptq_marlin | 4 | 128 | 0.0377 | 0.0416 | 0.056 | 14.35 | 1.000x | 1.143x | 0.000167 |
| attn_kv | k_proj/v_proj | 2 | 2048 | 512 | gptq_marlin | 4 | 128 | 0.0372 | 0.0413 | 0.113 | 14.53 | 1.000x | 1.151x | 0.000155 |
| attn_kv | k_proj/v_proj | 4 | 2048 | 512 | gptq_marlin | 4 | 128 | 0.0377 | 0.0412 | 0.222 | 14.34 | 1.000x | 1.146x | 0.000167 |
| attn_kv | k_proj/v_proj | 8 | 2048 | 512 | gptq_marlin | 4 | 128 | 0.0374 | 0.0407 | 0.449 | 14.47 | 1.000x | 1.147x | 0.000206 |
| attn_kv | k_proj/v_proj | 16 | 2048 | 512 | gptq_marlin | 4 | 128 | 0.0372 | 0.0411 | 0.902 | 14.53 | 1.000x | 1.157x | 0.000194 |
| attn_kv | k_proj/v_proj | 32 | 2048 | 512 | gptq_marlin | 4 | 128 | 0.0374 | 0.0419 | 1.793 | 14.45 | 1.000x | 1.164x | 0.000265 |
| attn_kv | k_proj/v_proj | 1 | 2048 | 512 | gptq_machete | 4 | 128 | 0.0431 | 0.0483 | 0.049 | 12.75 | 0.875x | 1.000x | 0.000167 |
| attn_kv | k_proj/v_proj | 2 | 2048 | 512 | gptq_machete | 4 | 128 | 0.0428 | 0.0468 | 0.098 | 12.81 | 0.869x | 1.000x | 0.000155 |
| attn_kv | k_proj/v_proj | 4 | 2048 | 512 | gptq_machete | 4 | 128 | 0.0432 | 0.0458 | 0.194 | 12.70 | 0.872x | 1.000x | 0.000167 |
| attn_kv | k_proj/v_proj | 8 | 2048 | 512 | gptq_machete | 4 | 128 | 0.0429 | 0.0469 | 0.391 | 12.80 | 0.872x | 1.000x | 0.000206 |
| attn_kv | k_proj/v_proj | 16 | 2048 | 512 | gptq_machete | 4 | 128 | 0.0430 | 0.0467 | 0.780 | 12.76 | 0.865x | 1.000x | 0.000194 |
| attn_kv | k_proj/v_proj | 32 | 2048 | 512 | gptq_machete | 4 | 128 | 0.0436 | 0.0466 | 1.541 | 12.60 | 0.859x | 1.000x | 0.000265 |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | qvq_lr | 2 | P32 | 0.4807 | 0.4824 | 0.070 | 8.86 | 0.068x | 0.092x | 7.15e-06 |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | qvq_lr | 2 | P32 | 0.0883 | 0.0886 | 0.760 | 48.23 | 0.369x | 0.504x | 4.77e-06 |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | qvq_lr | 2 | P32 | 0.0891 | 0.0894 | 1.507 | 47.83 | 0.362x | 0.495x | 5.01e-06 |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | qvq_lr | 2 | P32 | 0.0905 | 0.0907 | 2.967 | 47.09 | 0.359x | 0.484x | 5.72e-06 |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | qvq_lr | 2 | P32 | 0.1298 | 0.1301 | 4.137 | 32.83 | 0.252x | 0.336x | 5.72e-06 |
| mlp_gate_up | gate_proj/up_proj | 32 | 2048 | 8192 | qvq_lr | 2 | P32 | 0.4917 | 0.4956 | 2.184 | 8.66 | 0.073x | 0.092x | 1.19e-05 |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | qvq_lr | 2.5 | P32 | 0.4845 | 0.4859 | 0.069 | 10.96 | 0.067x | 0.091x | 6.08e-06 |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | qvq_lr | 2.5 | P32 | 0.4571 | 0.4603 | 0.147 | 11.61 | 0.071x | 0.097x | 4.77e-06 |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | qvq_lr | 2.5 | P32 | 0.4567 | 0.4616 | 0.294 | 11.62 | 0.071x | 0.096x | 4.77e-06 |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | qvq_lr | 2.5 | P32 | 0.4594 | 0.4641 | 0.584 | 11.56 | 0.071x | 0.095x | 4.89e-06 |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | qvq_lr | 2.5 | P32 | 0.1310 | 0.1315 | 4.099 | 40.53 | 0.250x | 0.333x | 5.72e-06 |
| mlp_gate_up | gate_proj/up_proj | 32 | 2048 | 8192 | qvq_lr | 2.5 | P32 | 0.4932 | 0.4963 | 2.177 | 10.76 | 0.073x | 0.092x | 1.34e-05 |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | qvq_lr | 3 | P32 | 0.2064 | 0.2069 | 0.163 | 30.80 | 0.157x | 0.215x | 6.68e-06 |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | qvq_lr | 3 | P32 | 0.1928 | 0.1950 | 0.348 | 32.97 | 0.169x | 0.231x | 4.77e-06 |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | qvq_lr | 3 | P32 | 0.1939 | 0.1960 | 0.692 | 32.79 | 0.166x | 0.227x | 4.89e-06 |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | qvq_lr | 3 | P32 | 0.1953 | 0.2012 | 1.375 | 32.55 | 0.166x | 0.224x | 4.77e-06 |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | qvq_lr | 3 | P32 | 0.2044 | 0.2067 | 2.627 | 31.11 | 0.160x | 0.213x | 6.68e-06 |
| mlp_gate_up | gate_proj/up_proj | 32 | 2048 | 8192 | qvq_lr | 3 | P32 | 0.2228 | 0.2238 | 4.820 | 28.53 | 0.161x | 0.203x | 1.34e-05 |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | qvq_lr | 3.5 | P32 | 0.2363 | 0.2371 | 0.142 | 31.34 | 0.137x | 0.187x | 6.2e-06 |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | qvq_lr | 3.5 | P32 | 0.1952 | 0.1978 | 0.344 | 37.93 | 0.167x | 0.228x | 4.77e-06 |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | qvq_lr | 3.5 | P32 | 0.1969 | 0.1991 | 0.682 | 37.61 | 0.164x | 0.224x | 5.72e-06 |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | qvq_lr | 3.5 | P32 | 0.1997 | 0.2023 | 1.344 | 37.08 | 0.162x | 0.219x | 5.96e-06 |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | qvq_lr | 3.5 | P32 | 0.2058 | 0.2080 | 2.608 | 35.98 | 0.159x | 0.212x | 5.25e-06 |
| mlp_gate_up | gate_proj/up_proj | 32 | 2048 | 8192 | qvq_lr | 3.5 | P32 | 0.2276 | 0.2286 | 4.717 | 32.54 | 0.157x | 0.199x | 1.36e-05 |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | gptq_marlin | 4 | 128 | 0.0325 | 0.0365 | 1.034 | 266.47 | 1.000x | 1.364x | 0.000272 |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | gptq_marlin | 4 | 128 | 0.0326 | 0.0366 | 2.057 | 265.17 | 1.000x | 1.365x | 0.000299 |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | gptq_marlin | 4 | 128 | 0.0323 | 0.0356 | 4.159 | 268.06 | 1.000x | 1.365x | 0.000274 |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | gptq_marlin | 4 | 128 | 0.0324 | 0.0364 | 8.277 | 266.74 | 1.000x | 1.349x | 0.000278 |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | gptq_marlin | 4 | 128 | 0.0328 | 0.0359 | 16.392 | 264.13 | 1.000x | 1.330x | 0.000327 |
| mlp_gate_up | gate_proj/up_proj | 32 | 2048 | 8192 | gptq_marlin | 4 | 128 | 0.0358 | 0.0388 | 29.986 | 241.59 | 1.000x | 1.265x | 0.000315 |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | gptq_machete | 4 | 128 | 0.0443 | 0.0479 | 0.758 | 195.51 | 0.733x | 1.000x | 0.000272 |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | gptq_machete | 4 | 128 | 0.0445 | 0.0557 | 1.507 | 194.39 | 0.732x | 1.000x | 0.000299 |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | gptq_machete | 4 | 128 | 0.0440 | 0.0499 | 3.047 | 196.58 | 0.733x | 1.000x | 0.000274 |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | gptq_machete | 4 | 128 | 0.0437 | 0.0464 | 6.137 | 197.95 | 0.741x | 1.000x | 0.000278 |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | gptq_machete | 4 | 128 | 0.0436 | 0.0476 | 12.327 | 198.82 | 0.752x | 1.000x | 0.000327 |
| mlp_gate_up | gate_proj/up_proj | 32 | 2048 | 8192 | gptq_machete | 4 | 128 | 0.0453 | 0.0479 | 23.713 | 191.23 | 0.791x | 1.000x | 0.000315 |
| mlp_down | down_proj | 1 | 8192 | 2048 | qvq_lr | 2 | P32 | 0.5187 | 0.5209 | 0.065 | 8.21 | 0.049x | 0.064x | 1.34e-05 |
| mlp_down | down_proj | 2 | 8192 | 2048 | qvq_lr | 2 | P32 | 0.0916 | 0.0918 | 0.733 | 46.51 | 0.278x | 0.358x | 9.54e-06 |
| mlp_down | down_proj | 4 | 8192 | 2048 | qvq_lr | 2 | P32 | 0.0924 | 0.0927 | 1.452 | 46.08 | 0.275x | 0.355x | 1.05e-05 |
| mlp_down | down_proj | 8 | 8192 | 2048 | qvq_lr | 2 | P32 | 0.0941 | 0.0943 | 2.853 | 45.28 | 0.272x | 0.349x | 1.67e-05 |
| mlp_down | down_proj | 16 | 8192 | 2048 | qvq_lr | 2 | P32 | 0.1324 | 0.1328 | 4.056 | 32.19 | 0.194x | 0.250x | 1.34e-05 |
| mlp_down | down_proj | 32 | 8192 | 2048 | qvq_lr | 2 | P32 | 0.4951 | 0.4981 | 2.169 | 8.60 | 0.075x | 0.067x | 1.19e-05 |
| mlp_down | down_proj | 1 | 8192 | 2048 | qvq_lr | 2.5 | P32 | 0.4579 | 0.4612 | 0.073 | 11.59 | 0.056x | 0.072x | 1.25e-05 |
| mlp_down | down_proj | 2 | 8192 | 2048 | qvq_lr | 2.5 | P32 | 0.4570 | 0.4619 | 0.147 | 11.62 | 0.056x | 0.072x | 9.54e-06 |
| mlp_down | down_proj | 4 | 8192 | 2048 | qvq_lr | 2.5 | P32 | 0.4574 | 0.4612 | 0.293 | 11.61 | 0.056x | 0.072x | 1.14e-05 |
| mlp_down | down_proj | 8 | 8192 | 2048 | qvq_lr | 2.5 | P32 | 0.4595 | 0.4636 | 0.584 | 11.55 | 0.056x | 0.071x | 1.72e-05 |
| mlp_down | down_proj | 16 | 8192 | 2048 | qvq_lr | 2.5 | P32 | 0.1362 | 0.1366 | 3.942 | 38.98 | 0.188x | 0.243x | 1.34e-05 |
| mlp_down | down_proj | 32 | 8192 | 2048 | qvq_lr | 2.5 | P32 | 0.4961 | 0.4996 | 2.164 | 10.70 | 0.075x | 0.067x | 1.34e-05 |
| mlp_down | down_proj | 1 | 8192 | 2048 | qvq_lr | 3 | P32 | 0.1993 | 0.2005 | 0.168 | 31.90 | 0.128x | 0.166x | 1.41e-05 |
| mlp_down | down_proj | 2 | 8192 | 2048 | qvq_lr | 3 | P32 | 0.1970 | 0.1993 | 0.341 | 32.27 | 0.129x | 0.167x | 1.14e-05 |
| mlp_down | down_proj | 4 | 8192 | 2048 | qvq_lr | 3 | P32 | 0.1974 | 0.2000 | 0.680 | 32.20 | 0.129x | 0.166x | 1.34e-05 |
| mlp_down | down_proj | 8 | 8192 | 2048 | qvq_lr | 3 | P32 | 0.2003 | 0.2029 | 1.340 | 31.74 | 0.128x | 0.164x | 1.86e-05 |
| mlp_down | down_proj | 16 | 8192 | 2048 | qvq_lr | 3 | P32 | 0.2082 | 0.2107 | 2.579 | 30.53 | 0.123x | 0.159x | 1.53e-05 |
| mlp_down | down_proj | 32 | 8192 | 2048 | qvq_lr | 3 | P32 | 0.2272 | 0.2282 | 4.726 | 27.98 | 0.163x | 0.145x | 1.24e-05 |
| mlp_down | down_proj | 1 | 8192 | 2048 | qvq_lr | 3.5 | P32 | 0.2241 | 0.2253 | 0.150 | 33.04 | 0.114x | 0.148x | 1.14e-05 |
| mlp_down | down_proj | 2 | 8192 | 2048 | qvq_lr | 3.5 | P32 | 0.1990 | 0.2025 | 0.337 | 37.21 | 0.128x | 0.165x | 1.24e-05 |
| mlp_down | down_proj | 4 | 8192 | 2048 | qvq_lr | 3.5 | P32 | 0.2004 | 0.2035 | 0.670 | 36.95 | 0.127x | 0.164x | 1.34e-05 |
| mlp_down | down_proj | 8 | 8192 | 2048 | qvq_lr | 3.5 | P32 | 0.2030 | 0.2058 | 1.322 | 36.48 | 0.126x | 0.162x | 1.72e-05 |
| mlp_down | down_proj | 16 | 8192 | 2048 | qvq_lr | 3.5 | P32 | 0.2096 | 0.2125 | 2.562 | 35.34 | 0.122x | 0.158x | 1.34e-05 |
| mlp_down | down_proj | 32 | 8192 | 2048 | qvq_lr | 3.5 | P32 | 0.2326 | 0.2340 | 4.616 | 31.84 | 0.159x | 0.142x | 1.29e-05 |
| mlp_down | down_proj | 1 | 8192 | 2048 | gptq_marlin | 4 | 128 | 0.0256 | 0.0310 | 1.311 | 337.92 | 1.000x | 1.294x | 0.000346 |
| mlp_down | down_proj | 2 | 8192 | 2048 | gptq_marlin | 4 | 128 | 0.0254 | 0.0280 | 2.638 | 340.05 | 1.000x | 1.289x | 0.000545 |
| mlp_down | down_proj | 4 | 8192 | 2048 | gptq_marlin | 4 | 128 | 0.0255 | 0.0284 | 5.273 | 339.83 | 1.000x | 1.290x | 0.000622 |
| mlp_down | down_proj | 8 | 8192 | 2048 | gptq_marlin | 4 | 128 | 0.0256 | 0.0284 | 10.499 | 338.34 | 1.000x | 1.283x | 0.000546 |
| mlp_down | down_proj | 16 | 8192 | 2048 | gptq_marlin | 4 | 128 | 0.0257 | 0.0295 | 20.919 | 337.08 | 1.000x | 1.288x | 0.000539 |
| mlp_down | down_proj | 32 | 8192 | 2048 | gptq_marlin | 4 | 128 | 0.0370 | 0.0375 | 29.026 | 233.85 | 1.000x | 0.892x | 0.00057 |
| mlp_down | down_proj | 1 | 8192 | 2048 | gptq_machete | 4 | 128 | 0.0331 | 0.0362 | 1.013 | 262.18 | 0.773x | 1.000x | 0.000346 |
| mlp_down | down_proj | 2 | 8192 | 2048 | gptq_machete | 4 | 128 | 0.0328 | 0.0362 | 2.046 | 264.74 | 0.776x | 1.000x | 0.000545 |
| mlp_down | down_proj | 4 | 8192 | 2048 | gptq_machete | 4 | 128 | 0.0328 | 0.0367 | 4.088 | 264.48 | 0.775x | 1.000x | 0.000622 |
| mlp_down | down_proj | 8 | 8192 | 2048 | gptq_machete | 4 | 128 | 0.0328 | 0.0368 | 8.184 | 264.74 | 0.780x | 1.000x | 0.000546 |
| mlp_down | down_proj | 16 | 8192 | 2048 | gptq_machete | 4 | 128 | 0.0331 | 0.0378 | 16.241 | 262.69 | 0.776x | 1.000x | 0.000539 |
| mlp_down | down_proj | 32 | 8192 | 2048 | gptq_machete | 4 | 128 | 0.0330 | 0.0367 | 32.530 | 263.07 | 1.121x | 1.000x | 0.00057 |

The four shape rows preserve all seven Llama 3.2 1B projection roles: `q_proj/o_proj` (2048×2048), `k_proj/v_proj` (2048×512), `gate_proj/up_proj` (2048×8192), and `down_proj` (8192×2048).

## Focused Nsight Compute check

Nsight Compute 2026.2.1 successfully captured the native `qvq_gemv_local_ring_kernel` on the same H100 (permission check fixed). This is a profiler sanity check, not a replacement for the CUDA-event matrix above.

| Shape | M | K | N | Kernel | W | Median ms | Logical TFLOP/s | xMarlin | xMachete |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| mlp_gate_up | 16 | 2048 | 8192 | qvq_lr | 3 | 0.2446 | 2.195 | 0.454x | 0.633x |

Capture command: `ncu --set basic --target-processes all --kernel-name-base function --kernel-name 'regex:.*qvq.*' --launch-count 1` with the benchmark restricted to `mlp_gate_up`, `M=16`, `W3`.
