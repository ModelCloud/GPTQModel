# QVQ LR H100 comparison — focused M sweep

QVQ V2B2-P32-LR versus matched W4 GPTQ Marlin/Machete baselines for the Llama 3.2 1B projection shapes.

## Measurement contract

- Current benchmark commit: `3df1d2dfe4b452f3d450963002dcbb77c9b44888`
- Previous benchmark commit: `ce2e59f1e79928e50064f2b2f9d62554af49df53`
- GPU: physical `1`, `NVIDIA H100`, PCI `00000000:44:00.0`, UUID `GPU-f5ea03cf-efa4-9807-7de5-b174957a1348`, CC `9.0`, 132 SMs
- M values: `[1, 2, 4, 8, 16]`; QVQ rates: `[2.0, 2.5]`
- Timing: one CUDA Graph replay with internal external CUDA events; host launch gaps excluded.
- `xMarlin`/`xMachete` = baseline median latency / QVQ median latency; above 1.0x is faster.
- `Better than last` is `yes` only when current QVQ median latency is strictly lower than the previous run; equal/regressed is `no`.

| Shape | Roles | M | K | N | W | Median ms | Logical TFLOP/s | xMarlin | xMachete | Better than last |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| attn_kv | k_proj/v_proj | 1 | 2048 | 512 | 2 | 0.0115 | 0.183 | 1.923x | 1.287x | yes |
| attn_kv | k_proj/v_proj | 1 | 2048 | 512 | 2.5 | 0.0116 | 0.181 | 1.905x | 1.274x | yes |
| attn_kv | k_proj/v_proj | 2 | 2048 | 512 | 2 | 0.0115 | 0.365 | 2.225x | 1.270x | yes |
| attn_kv | k_proj/v_proj | 2 | 2048 | 512 | 2.5 | 0.0116 | 0.363 | 2.213x | 1.263x | yes |
| attn_kv | k_proj/v_proj | 4 | 2048 | 512 | 2 | 0.0116 | 0.724 | 2.240x | 1.265x | yes |
| attn_kv | k_proj/v_proj | 4 | 2048 | 512 | 2.5 | 0.0116 | 0.724 | 2.240x | 1.265x | yes |
| attn_kv | k_proj/v_proj | 8 | 2048 | 512 | 2 | 0.0116 | 1.452 | 1.975x | 1.277x | yes |
| attn_kv | k_proj/v_proj | 8 | 2048 | 512 | 2.5 | 0.0112 | 1.502 | 2.043x | 1.321x | yes |
| attn_kv | k_proj/v_proj | 16 | 2048 | 512 | 2 | 0.0116 | 2.905 | 2.089x | 1.273x | yes |
| attn_kv | k_proj/v_proj | 16 | 2048 | 512 | 2.5 | 0.0113 | 2.979 | 2.142x | 1.305x | yes |
| attn_qo | q_proj/o_proj | 1 | 2048 | 2048 | 2 | 0.0145 | 0.578 | 1.001x | 1.062x | yes |
| attn_qo | q_proj/o_proj | 1 | 2048 | 2048 | 2.5 | 0.0147 | 0.572 | 0.991x | 1.051x | yes |
| attn_qo | q_proj/o_proj | 2 | 2048 | 2048 | 2 | 0.0154 | 1.092 | 1.055x | 1.015x | yes |
| attn_qo | q_proj/o_proj | 2 | 2048 | 2048 | 2.5 | 0.0152 | 1.103 | 1.065x | 1.024x | yes |
| attn_qo | q_proj/o_proj | 4 | 2048 | 2048 | 2 | 0.0153 | 2.198 | 1.069x | 1.019x | yes |
| attn_qo | q_proj/o_proj | 4 | 2048 | 2048 | 2.5 | 0.0154 | 2.180 | 1.060x | 1.010x | yes |
| attn_qo | q_proj/o_proj | 8 | 2048 | 2048 | 2 | 0.0154 | 4.355 | 0.964x | 1.012x | yes |
| attn_qo | q_proj/o_proj | 8 | 2048 | 2048 | 2.5 | 0.0153 | 4.387 | 0.971x | 1.020x | yes |
| attn_qo | q_proj/o_proj | 16 | 2048 | 2048 | 2 | 0.0155 | 8.648 | 1.012x | 1.000x | yes |
| attn_qo | q_proj/o_proj | 16 | 2048 | 2048 | 2.5 | 0.0156 | 8.595 | 1.006x | 0.994x | yes |
| mlp_down | down_proj | 1 | 8192 | 2048 | 2 | 0.0313 | 1.072 | 0.527x | 0.714x | yes |
| mlp_down | down_proj | 1 | 8192 | 2048 | 2.5 | 0.0319 | 1.052 | 0.517x | 0.700x | yes |
| mlp_down | down_proj | 2 | 8192 | 2048 | 2 | 0.0376 | 1.783 | 0.484x | 0.589x | yes |
| mlp_down | down_proj | 2 | 8192 | 2048 | 2.5 | 0.0372 | 1.806 | 0.490x | 0.597x | yes |
| mlp_down | down_proj | 4 | 8192 | 2048 | 2 | 0.0378 | 3.553 | 0.487x | 0.589x | yes |
| mlp_down | down_proj | 4 | 8192 | 2048 | 2.5 | 0.0371 | 3.614 | 0.495x | 0.599x | yes |
| mlp_down | down_proj | 8 | 8192 | 2048 | 2 | 0.0379 | 7.076 | 0.438x | 0.587x | yes |
| mlp_down | down_proj | 8 | 8192 | 2048 | 2.5 | 0.0374 | 7.182 | 0.445x | 0.595x | yes |
| mlp_down | down_proj | 16 | 8192 | 2048 | 2 | 0.0380 | 14.122 | 0.478x | 0.585x | yes |
| mlp_down | down_proj | 16 | 8192 | 2048 | 2.5 | 0.0378 | 14.206 | 0.481x | 0.589x | yes |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | 2 | 0.0349 | 0.962 | 0.282x | 0.513x | yes |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | 2.5 | 0.0343 | 0.978 | 0.286x | 0.521x | yes |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | 2 | 0.0349 | 1.921 | 0.292x | 0.517x | yes |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | 2.5 | 0.0345 | 1.945 | 0.295x | 0.523x | yes |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | 2 | 0.0351 | 3.825 | 0.290x | 0.512x | yes |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | 2.5 | 0.0347 | 3.873 | 0.294x | 0.518x | yes |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | 2 | 0.0355 | 7.561 | 0.280x | 0.507x | yes |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | 2.5 | 0.0350 | 7.675 | 0.285x | 0.514x | yes |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | 2 | 0.0359 | 14.960 | 0.301x | 0.498x | yes |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | 2.5 | 0.0353 | 15.224 | 0.307x | 0.507x | yes |

The four unique geometries preserve all seven Llama 3.2 1B roles: q_proj/o_proj (2048×2048), k_proj/v_proj (2048×512), gate_proj/up_proj (2048×8192), and down_proj (8192×2048).
