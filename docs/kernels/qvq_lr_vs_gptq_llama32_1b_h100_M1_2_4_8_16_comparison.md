# QVQ LR H100 comparison — focused M sweep

QVQ V2B2-P32-LR versus matched W4 GPTQ Marlin/Machete baselines for the Llama 3.2 1B projection shapes.

## Measurement contract

- Current benchmark commit: `168cefcbef6ea0732bb5b77ecb32604d488c24ae`
- Previous benchmark commit: `0f0cba61d5317fd1ab0f9f98f94116bcba6687ce`
- GPU: physical `1`, `NVIDIA H100`, PCI `00000000:44:00.0`, UUID `GPU-f5ea03cf-efa4-9807-7de5-b174957a1348`, CC `9.0`, 132 SMs
- M values: `[1, 2, 4, 8, 16]`; QVQ rates: `[2.0, 2.5, 3.0, 3.5]`
- Timing: one CUDA Graph replay with internal external CUDA events; host launch gaps excluded.
- `xMarlin`/`xMachete` = baseline median latency / QVQ median latency; above 1.0x is faster.
- `Better than last` is `yes` only when current QVQ median latency is strictly lower than the previous run; equal/regressed is `no`.

| Shape | Roles | M | K | N | W | Median ms | Logical TFLOP/s | xMarlin | xMachete | Better than last |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| attn_kv | k_proj/v_proj | 1 | 2048 | 512 | 2 | 0.0453 | 0.046 | 0.486x | 0.329x | no |
| attn_kv | k_proj/v_proj | 1 | 2048 | 512 | 2.5 | 0.0412 | 0.051 | 0.535x | 0.362x | no |
| attn_kv | k_proj/v_proj | 1 | 2048 | 512 | 3 | 0.0208 | 0.101 | 1.058x | 0.717x | yes |
| attn_kv | k_proj/v_proj | 1 | 2048 | 512 | 3.5 | 0.0224 | 0.094 | 0.982x | 0.665x | no |
| attn_kv | k_proj/v_proj | 2 | 2048 | 512 | 2 | 0.0132 | 0.317 | 1.917x | 1.147x | yes |
| attn_kv | k_proj/v_proj | 2 | 2048 | 512 | 2.5 | 0.0372 | 0.113 | 0.682x | 0.408x | no |
| attn_kv | k_proj/v_proj | 2 | 2048 | 512 | 3 | 0.0201 | 0.209 | 1.263x | 0.756x | yes |
| attn_kv | k_proj/v_proj | 2 | 2048 | 512 | 3.5 | 0.0204 | 0.205 | 1.242x | 0.743x | no |
| attn_kv | k_proj/v_proj | 4 | 2048 | 512 | 2 | 0.0137 | 0.612 | 1.907x | 1.117x | yes |
| attn_kv | k_proj/v_proj | 4 | 2048 | 512 | 2.5 | 0.0376 | 0.223 | 0.695x | 0.407x | no |
| attn_kv | k_proj/v_proj | 4 | 2048 | 512 | 3 | 0.0204 | 0.410 | 1.279x | 0.749x | yes |
| attn_kv | k_proj/v_proj | 4 | 2048 | 512 | 3.5 | 0.0208 | 0.403 | 1.256x | 0.736x | no |
| attn_kv | k_proj/v_proj | 8 | 2048 | 512 | 2 | 0.0146 | 1.147 | 1.578x | 1.022x | yes |
| attn_kv | k_proj/v_proj | 8 | 2048 | 512 | 2.5 | 0.0383 | 0.438 | 0.603x | 0.390x | no |
| attn_kv | k_proj/v_proj | 8 | 2048 | 512 | 3 | 0.0214 | 0.784 | 1.078x | 0.698x | yes |
| attn_kv | k_proj/v_proj | 8 | 2048 | 512 | 3.5 | 0.0215 | 0.780 | 1.073x | 0.695x | no |
| attn_kv | k_proj/v_proj | 16 | 2048 | 512 | 2 | 0.0187 | 1.794 | 1.299x | 0.815x | no |
| attn_kv | k_proj/v_proj | 16 | 2048 | 512 | 2.5 | 0.0189 | 1.771 | 1.283x | 0.805x | no |
| attn_kv | k_proj/v_proj | 16 | 2048 | 512 | 3 | 0.0240 | 1.396 | 1.011x | 0.634x | no |
| attn_kv | k_proj/v_proj | 16 | 2048 | 512 | 3.5 | 0.0239 | 1.404 | 1.017x | 0.638x | no |
| attn_qo | q_proj/o_proj | 1 | 2048 | 2048 | 2 | 0.1356 | 0.062 | 0.107x | 0.114x | no |
| attn_qo | q_proj/o_proj | 1 | 2048 | 2048 | 2.5 | 0.1202 | 0.070 | 0.121x | 0.128x | no |
| attn_qo | q_proj/o_proj | 1 | 2048 | 2048 | 3 | 0.0559 | 0.150 | 0.261x | 0.276x | yes |
| attn_qo | q_proj/o_proj | 1 | 2048 | 2048 | 3.5 | 0.0621 | 0.135 | 0.235x | 0.249x | yes |
| attn_qo | q_proj/o_proj | 2 | 2048 | 2048 | 2 | 0.0298 | 0.564 | 0.544x | 0.523x | no |
| attn_qo | q_proj/o_proj | 2 | 2048 | 2048 | 2.5 | 0.1210 | 0.139 | 0.134x | 0.129x | no |
| attn_qo | q_proj/o_proj | 2 | 2048 | 2048 | 3 | 0.0556 | 0.302 | 0.291x | 0.280x | yes |
| attn_qo | q_proj/o_proj | 2 | 2048 | 2048 | 3.5 | 0.0559 | 0.300 | 0.290x | 0.278x | no |
| attn_qo | q_proj/o_proj | 4 | 2048 | 2048 | 2 | 0.0306 | 1.098 | 0.534x | 0.505x | no |
| attn_qo | q_proj/o_proj | 4 | 2048 | 2048 | 2.5 | 0.1208 | 0.278 | 0.135x | 0.128x | yes |
| attn_qo | q_proj/o_proj | 4 | 2048 | 2048 | 3 | 0.0563 | 0.596 | 0.290x | 0.274x | no |
| attn_qo | q_proj/o_proj | 4 | 2048 | 2048 | 3.5 | 0.0561 | 0.598 | 0.291x | 0.275x | no |
| attn_qo | q_proj/o_proj | 8 | 2048 | 2048 | 2 | 0.0322 | 2.085 | 0.460x | 0.482x | no |
| attn_qo | q_proj/o_proj | 8 | 2048 | 2048 | 2.5 | 0.1219 | 0.551 | 0.122x | 0.127x | yes |
| attn_qo | q_proj/o_proj | 8 | 2048 | 2048 | 3 | 0.0571 | 1.176 | 0.260x | 0.272x | no |
| attn_qo | q_proj/o_proj | 8 | 2048 | 2048 | 3.5 | 0.0572 | 1.174 | 0.259x | 0.271x | no |
| attn_qo | q_proj/o_proj | 16 | 2048 | 2048 | 2 | 0.0438 | 3.067 | 0.360x | 0.352x | no |
| attn_qo | q_proj/o_proj | 16 | 2048 | 2048 | 2.5 | 0.0441 | 3.045 | 0.357x | 0.350x | no |
| attn_qo | q_proj/o_proj | 16 | 2048 | 2048 | 3 | 0.0213 | 6.288 | 0.738x | 0.723x | yes |
| attn_qo | q_proj/o_proj | 16 | 2048 | 2048 | 3.5 | 0.0617 | 2.175 | 0.255x | 0.250x | yes |
| mlp_down | down_proj | 1 | 8192 | 2048 | 2 | 0.5168 | 0.065 | 0.031x | 0.043x | yes |
| mlp_down | down_proj | 1 | 8192 | 2048 | 2.5 | 0.4566 | 0.073 | 0.036x | 0.049x | no |
| mlp_down | down_proj | 1 | 8192 | 2048 | 3 | 0.1987 | 0.169 | 0.082x | 0.113x | yes |
| mlp_down | down_proj | 1 | 8192 | 2048 | 3.5 | 0.2232 | 0.150 | 0.073x | 0.101x | no |
| mlp_down | down_proj | 2 | 8192 | 2048 | 2 | 0.0913 | 0.735 | 0.198x | 0.241x | no |
| mlp_down | down_proj | 2 | 8192 | 2048 | 2.5 | 0.4577 | 0.147 | 0.039x | 0.048x | no |
| mlp_down | down_proj | 2 | 8192 | 2048 | 3 | 0.1968 | 0.341 | 0.092x | 0.112x | no |
| mlp_down | down_proj | 2 | 8192 | 2048 | 3.5 | 0.1996 | 0.336 | 0.091x | 0.110x | yes |
| mlp_down | down_proj | 4 | 8192 | 2048 | 2 | 0.0921 | 1.457 | 0.197x | 0.244x | no |
| mlp_down | down_proj | 4 | 8192 | 2048 | 2.5 | 0.4575 | 0.293 | 0.040x | 0.049x | yes |
| mlp_down | down_proj | 4 | 8192 | 2048 | 3 | 0.1981 | 0.678 | 0.092x | 0.113x | no |
| mlp_down | down_proj | 4 | 8192 | 2048 | 3.5 | 0.2008 | 0.669 | 0.090x | 0.112x | no |
| mlp_down | down_proj | 8 | 8192 | 2048 | 2 | 0.0938 | 2.861 | 0.177x | 0.238x | no |
| mlp_down | down_proj | 8 | 8192 | 2048 | 2.5 | 0.4588 | 0.585 | 0.036x | 0.049x | yes |
| mlp_down | down_proj | 8 | 8192 | 2048 | 3 | 0.1986 | 1.351 | 0.084x | 0.112x | no |
| mlp_down | down_proj | 8 | 8192 | 2048 | 3.5 | 0.2014 | 1.333 | 0.083x | 0.111x | yes |
| mlp_down | down_proj | 16 | 8192 | 2048 | 2 | 0.1320 | 4.068 | 0.137x | 0.170x | no |
| mlp_down | down_proj | 16 | 8192 | 2048 | 2.5 | 0.1358 | 3.955 | 0.133x | 0.165x | no |
| mlp_down | down_proj | 16 | 8192 | 2048 | 3 | 0.0589 | 9.113 | 0.307x | 0.380x | yes |
| mlp_down | down_proj | 16 | 8192 | 2048 | 3.5 | 0.2076 | 2.586 | 0.087x | 0.108x | no |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | 2 | 0.4812 | 0.070 | 0.020x | 0.037x | no |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | 2.5 | 0.4845 | 0.069 | 0.020x | 0.037x | no |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | 3 | 0.2061 | 0.163 | 0.048x | 0.087x | no |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | 3.5 | 0.2364 | 0.142 | 0.042x | 0.076x | no |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | 2 | 0.0882 | 0.761 | 0.116x | 0.207x | no |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | 2.5 | 0.4560 | 0.147 | 0.022x | 0.040x | no |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | 3 | 0.1935 | 0.347 | 0.053x | 0.094x | yes |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | 3.5 | 0.1963 | 0.342 | 0.052x | 0.093x | yes |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | 2 | 0.0889 | 1.509 | 0.115x | 0.202x | no |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | 2.5 | 0.4568 | 0.294 | 0.022x | 0.039x | no |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | 3 | 0.1956 | 0.686 | 0.052x | 0.092x | no |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | 3.5 | 0.1986 | 0.676 | 0.052x | 0.091x | no |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | 2 | 0.0905 | 2.966 | 0.111x | 0.200x | no |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | 2.5 | 0.4557 | 0.589 | 0.022x | 0.040x | yes |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | 3 | 0.1969 | 1.363 | 0.051x | 0.092x | yes |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | 3.5 | 0.1994 | 1.346 | 0.050x | 0.091x | no |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | 2 | 0.1296 | 4.141 | 0.084x | 0.141x | no |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | 2.5 | 0.1308 | 4.103 | 0.083x | 0.140x | no |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | 3 | 0.0567 | 9.476 | 0.193x | 0.322x | yes |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | 3.5 | 0.2065 | 2.600 | 0.053x | 0.088x | yes |

The four unique geometries preserve all seven Llama 3.2 1B roles: q_proj/o_proj (2048×2048), k_proj/v_proj (2048×512), gate_proj/up_proj (2048×8192), and down_proj (8192×2048).
