# QVQ LR H100 comparison — focused M sweep

QVQ V2B2-P32-LR versus matched W4 GPTQ Marlin/Machete baselines for the Llama 3.2 1B projection shapes.

## Measurement contract

- Current benchmark commit: `0f0cba61d5317fd1ab0f9f98f94116bcba6687ce`
- Previous benchmark commit: `2b5be7aabcbf7b2ee00303e23b8ba0ac6427c926`
- GPU: physical `1`, `NVIDIA H100`, PCI `00000000:44:00.0`, UUID `GPU-f5ea03cf-efa4-9807-7de5-b174957a1348`, CC `9.0`, 132 SMs
- M values: `[1, 2, 4, 8, 16]`; QVQ rates: `[2.0, 2.5, 3.0, 3.5]`
- Timing: one CUDA Graph replay with internal external CUDA events; host launch gaps excluded.
- `xMarlin`/`xMachete` = baseline median latency / QVQ median latency; above 1.0x is faster.
- `Better than last` is `yes` only when current QVQ median latency is strictly lower than the previous run; equal/regressed is `no`.

| Shape | Roles | M | K | N | W | Median ms | Logical TFLOP/s | xMarlin | xMachete | Better than last |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| attn_kv | k_proj/v_proj | 1 | 2048 | 512 | 2 | 0.0452 | 0.046 | 0.482x | 0.325x | no |
| attn_kv | k_proj/v_proj | 1 | 2048 | 512 | 2.5 | 0.0411 | 0.051 | 0.529x | 0.357x | no |
| attn_kv | k_proj/v_proj | 1 | 2048 | 512 | 3 | 0.0209 | 0.100 | 1.042x | 0.703x | yes |
| attn_kv | k_proj/v_proj | 1 | 2048 | 512 | 3.5 | 0.0221 | 0.095 | 0.984x | 0.664x | yes |
| attn_kv | k_proj/v_proj | 2 | 2048 | 512 | 2 | 0.0133 | 0.315 | 1.916x | 1.103x | yes |
| attn_kv | k_proj/v_proj | 2 | 2048 | 512 | 2.5 | 0.0371 | 0.113 | 0.688x | 0.396x | yes |
| attn_kv | k_proj/v_proj | 2 | 2048 | 512 | 3 | 0.0202 | 0.208 | 1.264x | 0.728x | no |
| attn_kv | k_proj/v_proj | 2 | 2048 | 512 | 3.5 | 0.0203 | 0.206 | 1.257x | 0.724x | no |
| attn_kv | k_proj/v_proj | 4 | 2048 | 512 | 2 | 0.0138 | 0.609 | 1.856x | 1.062x | no |
| attn_kv | k_proj/v_proj | 4 | 2048 | 512 | 2.5 | 0.0375 | 0.223 | 0.681x | 0.390x | no |
| attn_kv | k_proj/v_proj | 4 | 2048 | 512 | 3 | 0.0206 | 0.408 | 1.244x | 0.711x | yes |
| attn_kv | k_proj/v_proj | 4 | 2048 | 512 | 3.5 | 0.0208 | 0.404 | 1.232x | 0.705x | no |
| attn_kv | k_proj/v_proj | 8 | 2048 | 512 | 2 | 0.0147 | 1.145 | 1.539x | 1.004x | no |
| attn_kv | k_proj/v_proj | 8 | 2048 | 512 | 2.5 | 0.0381 | 0.440 | 0.591x | 0.386x | no |
| attn_kv | k_proj/v_proj | 8 | 2048 | 512 | 3 | 0.0215 | 0.782 | 1.051x | 0.686x | yes |
| attn_kv | k_proj/v_proj | 8 | 2048 | 512 | 3.5 | 0.0213 | 0.787 | 1.058x | 0.690x | yes |
| attn_kv | k_proj/v_proj | 16 | 2048 | 512 | 2 | 0.0187 | 1.794 | 1.271x | 0.783x | no |
| attn_kv | k_proj/v_proj | 16 | 2048 | 512 | 2.5 | 0.0189 | 1.777 | 1.259x | 0.775x | no |
| attn_kv | k_proj/v_proj | 16 | 2048 | 512 | 3 | 0.0238 | 1.411 | 1.000x | 0.616x | yes |
| attn_kv | k_proj/v_proj | 16 | 2048 | 512 | 3.5 | 0.0238 | 1.409 | 0.999x | 0.615x | no |
| attn_qo | q_proj/o_proj | 1 | 2048 | 2048 | 2 | 0.1352 | 0.062 | 0.107x | 0.113x | no |
| attn_qo | q_proj/o_proj | 1 | 2048 | 2048 | 2.5 | 0.1201 | 0.070 | 0.121x | 0.127x | no |
| attn_qo | q_proj/o_proj | 1 | 2048 | 2048 | 3 | 0.0561 | 0.150 | 0.259x | 0.273x | no |
| attn_qo | q_proj/o_proj | 1 | 2048 | 2048 | 3.5 | 0.0621 | 0.135 | 0.233x | 0.246x | no |
| attn_qo | q_proj/o_proj | 2 | 2048 | 2048 | 2 | 0.0294 | 0.571 | 0.553x | 0.533x | no |
| attn_qo | q_proj/o_proj | 2 | 2048 | 2048 | 2.5 | 0.1206 | 0.139 | 0.135x | 0.130x | yes |
| attn_qo | q_proj/o_proj | 2 | 2048 | 2048 | 3 | 0.0556 | 0.302 | 0.292x | 0.282x | no |
| attn_qo | q_proj/o_proj | 2 | 2048 | 2048 | 3.5 | 0.0557 | 0.301 | 0.291x | 0.281x | no |
| attn_qo | q_proj/o_proj | 4 | 2048 | 2048 | 2 | 0.0302 | 1.110 | 0.537x | 0.505x | no |
| attn_qo | q_proj/o_proj | 4 | 2048 | 2048 | 2.5 | 0.1210 | 0.277 | 0.134x | 0.126x | yes |
| attn_qo | q_proj/o_proj | 4 | 2048 | 2048 | 3 | 0.0562 | 0.597 | 0.289x | 0.272x | yes |
| attn_qo | q_proj/o_proj | 4 | 2048 | 2048 | 3.5 | 0.0561 | 0.598 | 0.289x | 0.272x | no |
| attn_qo | q_proj/o_proj | 8 | 2048 | 2048 | 2 | 0.0317 | 2.117 | 0.464x | 0.491x | no |
| attn_qo | q_proj/o_proj | 8 | 2048 | 2048 | 2.5 | 0.1223 | 0.549 | 0.120x | 0.127x | no |
| attn_qo | q_proj/o_proj | 8 | 2048 | 2048 | 3 | 0.0570 | 1.177 | 0.258x | 0.273x | no |
| attn_qo | q_proj/o_proj | 8 | 2048 | 2048 | 3.5 | 0.0572 | 1.174 | 0.258x | 0.272x | no |
| attn_qo | q_proj/o_proj | 16 | 2048 | 2048 | 2 | 0.0432 | 3.110 | 0.362x | 0.357x | no |
| attn_qo | q_proj/o_proj | 16 | 2048 | 2048 | 2.5 | 0.0440 | 3.053 | 0.355x | 0.350x | yes |
| attn_qo | q_proj/o_proj | 16 | 2048 | 2048 | 3 | 0.0222 | 6.048 | 0.704x | 0.694x | yes |
| attn_qo | q_proj/o_proj | 16 | 2048 | 2048 | 3.5 | 0.0617 | 2.174 | 0.253x | 0.249x | no |
| mlp_down | down_proj | 1 | 8192 | 2048 | 2 | 0.5170 | 0.065 | 0.031x | 0.043x | no |
| mlp_down | down_proj | 1 | 8192 | 2048 | 2.5 | 0.4557 | 0.074 | 0.036x | 0.048x | yes |
| mlp_down | down_proj | 1 | 8192 | 2048 | 3 | 0.1989 | 0.169 | 0.081x | 0.111x | no |
| mlp_down | down_proj | 1 | 8192 | 2048 | 3.5 | 0.2231 | 0.150 | 0.073x | 0.099x | yes |
| mlp_down | down_proj | 2 | 8192 | 2048 | 2 | 0.0910 | 0.737 | 0.197x | 0.242x | yes |
| mlp_down | down_proj | 2 | 8192 | 2048 | 2.5 | 0.4575 | 0.147 | 0.039x | 0.048x | no |
| mlp_down | down_proj | 2 | 8192 | 2048 | 3 | 0.1961 | 0.342 | 0.092x | 0.113x | yes |
| mlp_down | down_proj | 2 | 8192 | 2048 | 3.5 | 0.2000 | 0.336 | 0.090x | 0.110x | yes |
| mlp_down | down_proj | 4 | 8192 | 2048 | 2 | 0.0918 | 1.462 | 0.196x | 0.241x | yes |
| mlp_down | down_proj | 4 | 8192 | 2048 | 2.5 | 0.4580 | 0.293 | 0.039x | 0.048x | no |
| mlp_down | down_proj | 4 | 8192 | 2048 | 3 | 0.1977 | 0.679 | 0.091x | 0.112x | no |
| mlp_down | down_proj | 4 | 8192 | 2048 | 3.5 | 0.2006 | 0.669 | 0.090x | 0.110x | no |
| mlp_down | down_proj | 8 | 8192 | 2048 | 2 | 0.0935 | 2.870 | 0.177x | 0.238x | no |
| mlp_down | down_proj | 8 | 8192 | 2048 | 2.5 | 0.4601 | 0.583 | 0.036x | 0.048x | no |
| mlp_down | down_proj | 8 | 8192 | 2048 | 3 | 0.1985 | 1.352 | 0.083x | 0.112x | yes |
| mlp_down | down_proj | 8 | 8192 | 2048 | 3.5 | 0.2017 | 1.331 | 0.082x | 0.110x | no |
| mlp_down | down_proj | 16 | 8192 | 2048 | 2 | 0.1316 | 4.079 | 0.137x | 0.168x | yes |
| mlp_down | down_proj | 16 | 8192 | 2048 | 2.5 | 0.1354 | 3.964 | 0.133x | 0.164x | no |
| mlp_down | down_proj | 16 | 8192 | 2048 | 3 | 0.0628 | 8.555 | 0.287x | 0.353x | yes |
| mlp_down | down_proj | 16 | 8192 | 2048 | 3.5 | 0.2071 | 2.592 | 0.087x | 0.107x | no |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | 2 | 0.4809 | 0.070 | 0.021x | 0.037x | yes |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | 2.5 | 0.4842 | 0.069 | 0.020x | 0.037x | yes |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | 3 | 0.2060 | 0.163 | 0.048x | 0.087x | yes |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | 3.5 | 0.2364 | 0.142 | 0.042x | 0.076x | no |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | 2 | 0.0882 | 0.761 | 0.117x | 0.203x | no |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | 2.5 | 0.4553 | 0.147 | 0.023x | 0.039x | yes |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | 3 | 0.1945 | 0.345 | 0.053x | 0.092x | no |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | 3.5 | 0.1966 | 0.341 | 0.052x | 0.091x | no |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | 2 | 0.0888 | 1.511 | 0.116x | 0.201x | yes |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | 2.5 | 0.4561 | 0.294 | 0.023x | 0.039x | yes |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | 3 | 0.1952 | 0.688 | 0.053x | 0.092x | yes |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | 3.5 | 0.1986 | 0.676 | 0.052x | 0.090x | yes |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | 2 | 0.0903 | 2.974 | 0.109x | 0.197x | yes |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | 2.5 | 0.4562 | 0.588 | 0.022x | 0.039x | yes |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | 3 | 0.1975 | 1.359 | 0.050x | 0.090x | no |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | 3.5 | 0.1991 | 1.348 | 0.049x | 0.089x | yes |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | 2 | 0.1294 | 4.149 | 0.083x | 0.139x | yes |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | 2.5 | 0.1307 | 4.107 | 0.082x | 0.138x | yes |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | 3 | 0.0598 | 8.981 | 0.180x | 0.301x | yes |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | 3.5 | 0.2066 | 2.599 | 0.052x | 0.087x | no |

The four unique geometries preserve all seven Llama 3.2 1B roles: q_proj/o_proj (2048×2048), k_proj/v_proj (2048×512), gate_proj/up_proj (2048×8192), and down_proj (8192×2048).
