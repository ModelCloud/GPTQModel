# QVQ LR H100 comparison — focused M sweep

QVQ V2B2-P32-LR versus matched W4 GPTQ Marlin/Machete baselines for the Llama 3.2 1B projection shapes.

## Measurement contract

- Current benchmark commit: `2b5be7aabcbf7b2ee00303e23b8ba0ac6427c926`
- Previous benchmark commit: `6bb7d21edc4a4fc93ac456fd0d7917e893f35ca7`
- GPU: physical `1`, `NVIDIA H100`, PCI `00000000:44:00.0`, UUID `GPU-f5ea03cf-efa4-9807-7de5-b174957a1348`, CC `9.0`, 132 SMs
- M values: `[1, 2, 4, 8, 16]`; QVQ rates: `[2.0, 2.5, 3.0, 3.5]`
- Timing: one CUDA Graph replay with internal external CUDA events; host launch gaps excluded.
- `xMarlin`/`xMachete` = baseline median latency / QVQ median latency; above 1.0x is faster.
- `Better than last` is `yes` only when current QVQ median latency is strictly lower than the previous run; equal/regressed is `no`.

| Shape | Roles | M | K | N | W | Median ms | Logical TFLOP/s | xMarlin | xMachete | Better than last |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| attn_kv | k_proj/v_proj | 1 | 2048 | 512 | 2 | 0.0451 | 0.047 | 0.493x | 0.327x | no |
| attn_kv | k_proj/v_proj | 1 | 2048 | 512 | 2.5 | 0.0409 | 0.051 | 0.543x | 0.360x | yes |
| attn_kv | k_proj/v_proj | 1 | 2048 | 512 | 3 | 0.0210 | 0.100 | 1.060x | 0.703x | no |
| attn_kv | k_proj/v_proj | 1 | 2048 | 512 | 3.5 | 0.0222 | 0.094 | 0.999x | 0.663x | no |
| attn_kv | k_proj/v_proj | 2 | 2048 | 512 | 2 | 0.0133 | 0.314 | 1.942x | 1.101x | no |
| attn_kv | k_proj/v_proj | 2 | 2048 | 512 | 2.5 | 0.0372 | 0.113 | 0.697x | 0.395x | yes |
| attn_kv | k_proj/v_proj | 2 | 2048 | 512 | 3 | 0.0201 | 0.208 | 1.287x | 0.729x | no |
| attn_kv | k_proj/v_proj | 2 | 2048 | 512 | 3.5 | 0.0203 | 0.207 | 1.278x | 0.724x | yes |
| attn_kv | k_proj/v_proj | 4 | 2048 | 512 | 2 | 0.0136 | 0.616 | 1.922x | 1.072x | yes |
| attn_kv | k_proj/v_proj | 4 | 2048 | 512 | 2.5 | 0.0374 | 0.224 | 0.699x | 0.390x | yes |
| attn_kv | k_proj/v_proj | 4 | 2048 | 512 | 3 | 0.0206 | 0.407 | 1.270x | 0.708x | no |
| attn_kv | k_proj/v_proj | 4 | 2048 | 512 | 3.5 | 0.0207 | 0.406 | 1.266x | 0.706x | yes |
| attn_kv | k_proj/v_proj | 8 | 2048 | 512 | 2 | 0.0145 | 1.157 | 1.586x | 1.010x | no |
| attn_kv | k_proj/v_proj | 8 | 2048 | 512 | 2.5 | 0.0381 | 0.440 | 0.603x | 0.384x | yes |
| attn_kv | k_proj/v_proj | 8 | 2048 | 512 | 3 | 0.0215 | 0.781 | 1.071x | 0.682x | no |
| attn_kv | k_proj/v_proj | 8 | 2048 | 512 | 3.5 | 0.0216 | 0.777 | 1.065x | 0.678x | yes |
| attn_kv | k_proj/v_proj | 16 | 2048 | 512 | 2 | 0.0186 | 1.808 | 1.310x | 0.790x | yes |
| attn_kv | k_proj/v_proj | 16 | 2048 | 512 | 2.5 | 0.0189 | 1.779 | 1.289x | 0.777x | yes |
| attn_kv | k_proj/v_proj | 16 | 2048 | 512 | 3 | 0.0240 | 1.401 | 1.015x | 0.612x | no |
| attn_kv | k_proj/v_proj | 16 | 2048 | 512 | 3.5 | 0.0238 | 1.410 | 1.022x | 0.616x | yes |
| attn_qo | q_proj/o_proj | 1 | 2048 | 2048 | 2 | 0.1349 | 0.062 | 0.108x | 0.113x | yes |
| attn_qo | q_proj/o_proj | 1 | 2048 | 2048 | 2.5 | 0.1199 | 0.070 | 0.121x | 0.128x | yes |
| attn_qo | q_proj/o_proj | 1 | 2048 | 2048 | 3 | 0.0559 | 0.150 | 0.260x | 0.273x | yes |
| attn_qo | q_proj/o_proj | 1 | 2048 | 2048 | 3.5 | 0.0621 | 0.135 | 0.234x | 0.246x | yes |
| attn_qo | q_proj/o_proj | 2 | 2048 | 2048 | 2 | 0.0292 | 0.575 | 0.555x | 0.530x | yes |
| attn_qo | q_proj/o_proj | 2 | 2048 | 2048 | 2.5 | 0.1207 | 0.139 | 0.134x | 0.128x | yes |
| attn_qo | q_proj/o_proj | 2 | 2048 | 2048 | 3 | 0.0554 | 0.303 | 0.292x | 0.279x | yes |
| attn_qo | q_proj/o_proj | 2 | 2048 | 2048 | 3.5 | 0.0555 | 0.302 | 0.292x | 0.279x | yes |
| attn_qo | q_proj/o_proj | 4 | 2048 | 2048 | 2 | 0.0299 | 1.122 | 0.543x | 0.512x | yes |
| attn_qo | q_proj/o_proj | 4 | 2048 | 2048 | 2.5 | 0.1212 | 0.277 | 0.134x | 0.126x | yes |
| attn_qo | q_proj/o_proj | 4 | 2048 | 2048 | 3 | 0.0563 | 0.596 | 0.288x | 0.272x | yes |
| attn_qo | q_proj/o_proj | 4 | 2048 | 2048 | 3.5 | 0.0560 | 0.599 | 0.290x | 0.273x | yes |
| attn_qo | q_proj/o_proj | 8 | 2048 | 2048 | 2 | 0.0314 | 2.136 | 0.471x | 0.495x | yes |
| attn_qo | q_proj/o_proj | 8 | 2048 | 2048 | 2.5 | 0.1220 | 0.550 | 0.121x | 0.127x | yes |
| attn_qo | q_proj/o_proj | 8 | 2048 | 2048 | 3 | 0.0569 | 1.179 | 0.260x | 0.273x | yes |
| attn_qo | q_proj/o_proj | 8 | 2048 | 2048 | 3.5 | 0.0571 | 1.176 | 0.260x | 0.272x | yes |
| attn_qo | q_proj/o_proj | 16 | 2048 | 2048 | 2 | 0.0430 | 3.123 | 0.365x | 0.356x | yes |
| attn_qo | q_proj/o_proj | 16 | 2048 | 2048 | 2.5 | 0.0440 | 3.050 | 0.357x | 0.348x | yes |
| attn_qo | q_proj/o_proj | 16 | 2048 | 2048 | 3 | 0.0232 | 5.785 | 0.677x | 0.659x | yes |
| attn_qo | q_proj/o_proj | 16 | 2048 | 2048 | 3.5 | 0.0616 | 2.177 | 0.255x | 0.248x | yes |
| mlp_down | down_proj | 1 | 8192 | 2048 | 2 | 0.5170 | 0.065 | 0.031x | 0.043x | yes |
| mlp_down | down_proj | 1 | 8192 | 2048 | 2.5 | 0.4558 | 0.074 | 0.035x | 0.049x | yes |
| mlp_down | down_proj | 1 | 8192 | 2048 | 3 | 0.1988 | 0.169 | 0.081x | 0.111x | yes |
| mlp_down | down_proj | 1 | 8192 | 2048 | 3.5 | 0.2240 | 0.150 | 0.072x | 0.099x | yes |
| mlp_down | down_proj | 2 | 8192 | 2048 | 2 | 0.0910 | 0.737 | 0.197x | 0.237x | yes |
| mlp_down | down_proj | 2 | 8192 | 2048 | 2.5 | 0.4575 | 0.147 | 0.039x | 0.047x | yes |
| mlp_down | down_proj | 2 | 8192 | 2048 | 3 | 0.1969 | 0.341 | 0.091x | 0.110x | yes |
| mlp_down | down_proj | 2 | 8192 | 2048 | 3.5 | 0.2002 | 0.335 | 0.090x | 0.108x | yes |
| mlp_down | down_proj | 4 | 8192 | 2048 | 2 | 0.0919 | 1.461 | 0.196x | 0.242x | yes |
| mlp_down | down_proj | 4 | 8192 | 2048 | 2.5 | 0.4570 | 0.294 | 0.039x | 0.049x | yes |
| mlp_down | down_proj | 4 | 8192 | 2048 | 3 | 0.1977 | 0.679 | 0.091x | 0.113x | no |
| mlp_down | down_proj | 4 | 8192 | 2048 | 3.5 | 0.2003 | 0.670 | 0.090x | 0.111x | yes |
| mlp_down | down_proj | 8 | 8192 | 2048 | 2 | 0.0935 | 2.871 | 0.176x | 0.237x | yes |
| mlp_down | down_proj | 8 | 8192 | 2048 | 2.5 | 0.4589 | 0.585 | 0.036x | 0.048x | yes |
| mlp_down | down_proj | 8 | 8192 | 2048 | 3 | 0.1986 | 1.352 | 0.083x | 0.112x | yes |
| mlp_down | down_proj | 8 | 8192 | 2048 | 3.5 | 0.2015 | 1.332 | 0.082x | 0.110x | yes |
| mlp_down | down_proj | 16 | 8192 | 2048 | 2 | 0.1317 | 4.076 | 0.136x | 0.170x | yes |
| mlp_down | down_proj | 16 | 8192 | 2048 | 2.5 | 0.1353 | 3.968 | 0.132x | 0.165x | yes |
| mlp_down | down_proj | 16 | 8192 | 2048 | 3 | 0.0704 | 7.621 | 0.254x | 0.317x | yes |
| mlp_down | down_proj | 16 | 8192 | 2048 | 3.5 | 0.2069 | 2.594 | 0.086x | 0.108x | yes |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | 2 | 0.4811 | 0.070 | 0.020x | 0.037x | yes |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | 2.5 | 0.4846 | 0.069 | 0.020x | 0.037x | no |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | 3 | 0.2060 | 0.163 | 0.048x | 0.087x | no |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | 3.5 | 0.2360 | 0.142 | 0.042x | 0.076x | yes |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | 2 | 0.0882 | 0.761 | 0.116x | 0.203x | yes |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | 2.5 | 0.4557 | 0.147 | 0.022x | 0.039x | yes |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | 3 | 0.1943 | 0.345 | 0.053x | 0.092x | no |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | 3.5 | 0.1956 | 0.343 | 0.052x | 0.091x | yes |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | 2 | 0.0888 | 1.511 | 0.115x | 0.201x | yes |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | 2.5 | 0.4566 | 0.294 | 0.022x | 0.039x | no |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | 3 | 0.1954 | 0.687 | 0.052x | 0.091x | no |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | 3.5 | 0.1991 | 0.674 | 0.051x | 0.090x | no |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | 2 | 0.0903 | 2.972 | 0.110x | 0.199x | no |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | 2.5 | 0.4568 | 0.588 | 0.022x | 0.039x | yes |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | 3 | 0.1967 | 1.364 | 0.050x | 0.091x | no |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | 3.5 | 0.1996 | 1.345 | 0.050x | 0.090x | yes |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | 2 | 0.1294 | 4.148 | 0.083x | 0.140x | yes |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | 2.5 | 0.1313 | 4.089 | 0.082x | 0.138x | no |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | 3 | 0.0621 | 8.644 | 0.173x | 0.292x | yes |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | 3.5 | 0.2062 | 2.603 | 0.052x | 0.088x | no |

The four unique geometries preserve all seven Llama 3.2 1B roles: q_proj/o_proj (2048×2048), k_proj/v_proj (2048×512), gate_proj/up_proj (2048×8192), and down_proj (8192×2048).
