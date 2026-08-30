# QVQ LR H100 comparison — focused M sweep

QVQ V2B2-P32-LR versus matched W4 GPTQ Marlin/Machete baselines for the Llama 3.2 1B projection shapes.

## Measurement contract

- Current benchmark commit: `ce2e59f1e79928e50064f2b2f9d62554af49df53`
- Previous benchmark commit: `2a71cad5b1d418a764149fcad08b37fd0d511d32`
- GPU: physical `1`, `NVIDIA H100`, PCI `00000000:44:00.0`, UUID `GPU-f5ea03cf-efa4-9807-7de5-b174957a1348`, CC `9.0`, 132 SMs
- M values: `[1, 2, 4, 8, 16]`; QVQ rates: `[2.0, 2.5]`
- Timing: one CUDA Graph replay with internal external CUDA events; host launch gaps excluded.
- `xMarlin`/`xMachete` = baseline median latency / QVQ median latency; above 1.0x is faster.
- `Better than last` is `yes` only when current QVQ median latency is strictly lower than the previous run; equal/regressed is `no`.

| Shape | Roles | M | K | N | W | Median ms | Logical TFLOP/s | xMarlin | xMachete | Better than last |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| attn_kv | k_proj/v_proj | 1 | 2048 | 512 | 2 | 0.0115 | 0.182 | 1.919x | 1.274x | yes |
| attn_kv | k_proj/v_proj | 1 | 2048 | 512 | 2.5 | 0.0124 | 0.168 | 1.776x | 1.179x | yes |
| attn_kv | k_proj/v_proj | 2 | 2048 | 512 | 2 | 0.0116 | 0.362 | 2.215x | 1.256x | yes |
| attn_kv | k_proj/v_proj | 2 | 2048 | 512 | 2.5 | 0.0123 | 0.342 | 2.091x | 1.185x | yes |
| attn_kv | k_proj/v_proj | 4 | 2048 | 512 | 2 | 0.0116 | 0.722 | 2.245x | 1.259x | yes |
| attn_kv | k_proj/v_proj | 4 | 2048 | 512 | 2.5 | 0.0124 | 0.675 | 2.098x | 1.176x | yes |
| attn_kv | k_proj/v_proj | 8 | 2048 | 512 | 2 | 0.0116 | 1.446 | 1.981x | 1.274x | yes |
| attn_kv | k_proj/v_proj | 8 | 2048 | 512 | 2.5 | 0.0120 | 1.394 | 1.910x | 1.229x | yes |
| attn_kv | k_proj/v_proj | 16 | 2048 | 512 | 2 | 0.0118 | 2.853 | 2.053x | 1.245x | yes |
| attn_kv | k_proj/v_proj | 16 | 2048 | 512 | 2.5 | 0.0121 | 2.781 | 2.001x | 1.214x | yes |
| attn_qo | q_proj/o_proj | 1 | 2048 | 2048 | 2 | 0.0154 | 0.544 | 0.947x | 0.993x | yes |
| attn_qo | q_proj/o_proj | 1 | 2048 | 2048 | 2.5 | 0.0158 | 0.532 | 0.925x | 0.970x | yes |
| attn_qo | q_proj/o_proj | 2 | 2048 | 2048 | 2 | 0.0161 | 1.044 | 1.012x | 0.940x | yes |
| attn_qo | q_proj/o_proj | 2 | 2048 | 2048 | 2.5 | 0.0172 | 0.974 | 0.943x | 0.877x | yes |
| attn_qo | q_proj/o_proj | 4 | 2048 | 2048 | 2 | 0.0160 | 2.093 | 1.020x | 0.957x | yes |
| attn_qo | q_proj/o_proj | 4 | 2048 | 2048 | 2.5 | 0.0172 | 1.951 | 0.951x | 0.892x | yes |
| attn_qo | q_proj/o_proj | 8 | 2048 | 2048 | 2 | 0.0161 | 4.169 | 0.918x | 0.948x | yes |
| attn_qo | q_proj/o_proj | 8 | 2048 | 2048 | 2.5 | 0.0174 | 3.862 | 0.851x | 0.878x | yes |
| attn_qo | q_proj/o_proj | 16 | 2048 | 2048 | 2 | 0.0165 | 8.144 | 0.949x | 0.935x | yes |
| attn_qo | q_proj/o_proj | 16 | 2048 | 2048 | 2.5 | 0.0175 | 7.689 | 0.896x | 0.883x | yes |
| mlp_down | down_proj | 1 | 8192 | 2048 | 2 | 0.0347 | 0.966 | 0.472x | 0.636x | yes |
| mlp_down | down_proj | 1 | 8192 | 2048 | 2.5 | 0.0348 | 0.963 | 0.470x | 0.634x | yes |
| mlp_down | down_proj | 2 | 8192 | 2048 | 2 | 0.0401 | 1.672 | 0.452x | 0.549x | yes |
| mlp_down | down_proj | 2 | 8192 | 2048 | 2.5 | 0.0442 | 1.517 | 0.410x | 0.498x | yes |
| mlp_down | down_proj | 4 | 8192 | 2048 | 2 | 0.0401 | 3.347 | 0.453x | 0.555x | yes |
| mlp_down | down_proj | 4 | 8192 | 2048 | 2.5 | 0.0443 | 3.031 | 0.410x | 0.502x | yes |
| mlp_down | down_proj | 8 | 8192 | 2048 | 2 | 0.0404 | 6.650 | 0.411x | 0.552x | yes |
| mlp_down | down_proj | 8 | 8192 | 2048 | 2.5 | 0.0446 | 6.013 | 0.372x | 0.499x | yes |
| mlp_down | down_proj | 16 | 8192 | 2048 | 2 | 0.0405 | 13.242 | 0.444x | 0.545x | yes |
| mlp_down | down_proj | 16 | 8192 | 2048 | 2.5 | 0.0447 | 12.009 | 0.402x | 0.495x | yes |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | 2 | 0.0372 | 0.902 | 0.269x | 0.482x | yes |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | 2.5 | 0.0413 | 0.812 | 0.242x | 0.434x | yes |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | 2 | 0.0372 | 1.803 | 0.278x | 0.482x | yes |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | 2.5 | 0.0415 | 1.617 | 0.249x | 0.433x | yes |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | 2 | 0.0374 | 3.591 | 0.275x | 0.484x | yes |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | 2.5 | 0.0418 | 3.208 | 0.246x | 0.433x | yes |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | 2 | 0.0377 | 7.127 | 0.266x | 0.476x | yes |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | 2.5 | 0.0420 | 6.384 | 0.238x | 0.427x | yes |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | 2 | 0.0381 | 14.104 | 0.285x | 0.474x | yes |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | 2.5 | 0.0423 | 12.691 | 0.256x | 0.427x | yes |

The four unique geometries preserve all seven Llama 3.2 1B roles: q_proj/o_proj (2048×2048), k_proj/v_proj (2048×512), gate_proj/up_proj (2048×8192), and down_proj (8192×2048).
