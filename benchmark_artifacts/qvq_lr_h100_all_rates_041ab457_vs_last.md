# QVQ LR H100 comparison — focused M sweep

QVQ V2B2-P32-LR versus matched W4 GPTQ Marlin/Machete baselines for the Llama 3.2 1B projection shapes.

## Measurement contract

- Current benchmark commit: `041ab457e7847203d7ea685be769439dcca87933`
- Previous benchmark commits, in lookup priority: `3df1d2dfe4b452f3d450963002dcbb77c9b44888`, `168cefcbef6ea0732bb5b77ecb32604d488c24ae`
- GPU: physical `1`, `NVIDIA H100`, PCI `00000000:44:00.0`, UUID `GPU-f5ea03cf-efa4-9807-7de5-b174957a1348`, CC `9.0`, 132 SMs
- M values: `[1, 2, 4, 8, 16]`; QVQ rates: `[2.0, 2.5, 3.0, 3.5]`
- Timing: one CUDA Graph replay with internal external CUDA events; host launch gaps excluded.
- `xMarlin`/`xMachete` = baseline median latency / QVQ median latency; above 1.0x is faster.
- `Better than last` is `yes` only when current QVQ median latency is strictly lower than the previous run; equal/regressed is `no`.

| Shape | Roles | M | K | N | W | Median ms | Logical TFLOP/s | xMarlin | xMachete | Better than last |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| attn_kv | k_proj/v_proj | 1 | 2048 | 512 | 2 | 0.0116 | 0.181 | 1.892x | 1.248x | no |
| attn_kv | k_proj/v_proj | 1 | 2048 | 512 | 2.5 | 0.0116 | 0.180 | 1.885x | 1.243x | no |
| attn_kv | k_proj/v_proj | 1 | 2048 | 512 | 3 | 0.0208 | 0.101 | 1.054x | 0.695x | no |
| attn_kv | k_proj/v_proj | 1 | 2048 | 512 | 3.5 | 0.0223 | 0.094 | 0.984x | 0.649x | yes |
| attn_kv | k_proj/v_proj | 2 | 2048 | 512 | 2 | 0.0116 | 0.361 | 2.176x | 1.274x | no |
| attn_kv | k_proj/v_proj | 2 | 2048 | 512 | 2.5 | 0.0117 | 0.357 | 2.153x | 1.260x | no |
| attn_kv | k_proj/v_proj | 2 | 2048 | 512 | 3 | 0.0202 | 0.208 | 1.254x | 0.734x | no |
| attn_kv | k_proj/v_proj | 2 | 2048 | 512 | 3.5 | 0.0206 | 0.204 | 1.230x | 0.720x | no |
| attn_kv | k_proj/v_proj | 4 | 2048 | 512 | 2 | 0.0114 | 0.737 | 2.277x | 1.302x | yes |
| attn_kv | k_proj/v_proj | 4 | 2048 | 512 | 2.5 | 0.0117 | 0.714 | 2.206x | 1.262x | no |
| attn_kv | k_proj/v_proj | 4 | 2048 | 512 | 3 | 0.0204 | 0.410 | 1.267x | 0.725x | no |
| attn_kv | k_proj/v_proj | 4 | 2048 | 512 | 3.5 | 0.0209 | 0.402 | 1.241x | 0.710x | no |
| attn_kv | k_proj/v_proj | 8 | 2048 | 512 | 2 | 0.0114 | 1.473 | 1.982x | 1.270x | yes |
| attn_kv | k_proj/v_proj | 8 | 2048 | 512 | 2.5 | 0.0116 | 1.440 | 1.938x | 1.242x | no |
| attn_kv | k_proj/v_proj | 8 | 2048 | 512 | 3 | 0.0214 | 0.785 | 1.056x | 0.677x | yes |
| attn_kv | k_proj/v_proj | 8 | 2048 | 512 | 3.5 | 0.0216 | 0.777 | 1.046x | 0.670x | no |
| attn_kv | k_proj/v_proj | 16 | 2048 | 512 | 2 | 0.0115 | 2.925 | 2.100x | 1.294x | yes |
| attn_kv | k_proj/v_proj | 16 | 2048 | 512 | 2.5 | 0.0120 | 2.800 | 2.011x | 1.239x | no |
| attn_kv | k_proj/v_proj | 16 | 2048 | 512 | 3 | 0.0240 | 1.397 | 1.003x | 0.618x | yes |
| attn_kv | k_proj/v_proj | 16 | 2048 | 512 | 3.5 | 0.0239 | 1.404 | 1.008x | 0.621x | no |
| attn_qo | q_proj/o_proj | 1 | 2048 | 2048 | 2 | 0.0144 | 0.581 | 1.003x | 1.071x | yes |
| attn_qo | q_proj/o_proj | 1 | 2048 | 2048 | 2.5 | 0.0146 | 0.573 | 0.990x | 1.057x | yes |
| attn_qo | q_proj/o_proj | 1 | 2048 | 2048 | 3 | 0.0154 | 0.543 | 0.939x | 1.002x | yes |
| attn_qo | q_proj/o_proj | 1 | 2048 | 2048 | 3.5 | 0.0620 | 0.135 | 0.234x | 0.250x | yes |
| attn_qo | q_proj/o_proj | 2 | 2048 | 2048 | 2 | 0.0152 | 1.101 | 1.068x | 1.013x | yes |
| attn_qo | q_proj/o_proj | 2 | 2048 | 2048 | 2.5 | 0.0153 | 1.093 | 1.060x | 1.005x | no |
| attn_qo | q_proj/o_proj | 2 | 2048 | 2048 | 3 | 0.0166 | 1.010 | 0.980x | 0.929x | yes |
| attn_qo | q_proj/o_proj | 2 | 2048 | 2048 | 3.5 | 0.0557 | 0.301 | 0.292x | 0.277x | yes |
| attn_qo | q_proj/o_proj | 4 | 2048 | 2048 | 2 | 0.0152 | 2.205 | 1.081x | 1.017x | yes |
| attn_qo | q_proj/o_proj | 4 | 2048 | 2048 | 2.5 | 0.0152 | 2.205 | 1.081x | 1.017x | yes |
| attn_qo | q_proj/o_proj | 4 | 2048 | 2048 | 3 | 0.0167 | 2.015 | 0.988x | 0.929x | yes |
| attn_qo | q_proj/o_proj | 4 | 2048 | 2048 | 3.5 | 0.0559 | 0.600 | 0.294x | 0.277x | yes |
| attn_qo | q_proj/o_proj | 8 | 2048 | 2048 | 2 | 0.0154 | 4.355 | 0.964x | 1.003x | no |
| attn_qo | q_proj/o_proj | 8 | 2048 | 2048 | 2.5 | 0.0153 | 4.383 | 0.970x | 1.009x | no |
| attn_qo | q_proj/o_proj | 8 | 2048 | 2048 | 3 | 0.0167 | 4.018 | 0.889x | 0.925x | yes |
| attn_qo | q_proj/o_proj | 8 | 2048 | 2048 | 3.5 | 0.0570 | 1.177 | 0.260x | 0.271x | yes |
| attn_qo | q_proj/o_proj | 16 | 2048 | 2048 | 2 | 0.0155 | 8.648 | 1.008x | 0.998x | no |
| attn_qo | q_proj/o_proj | 16 | 2048 | 2048 | 2.5 | 0.0156 | 8.586 | 1.001x | 0.991x | no |
| attn_qo | q_proj/o_proj | 16 | 2048 | 2048 | 3 | 0.0168 | 7.966 | 0.929x | 0.919x | yes |
| attn_qo | q_proj/o_proj | 16 | 2048 | 2048 | 3.5 | 0.0616 | 2.178 | 0.254x | 0.251x | yes |
| mlp_down | down_proj | 1 | 8192 | 2048 | 2 | 0.0310 | 1.081 | 0.532x | 0.718x | yes |
| mlp_down | down_proj | 1 | 8192 | 2048 | 2.5 | 0.0316 | 1.062 | 0.523x | 0.705x | yes |
| mlp_down | down_proj | 1 | 8192 | 2048 | 3 | 0.0347 | 0.966 | 0.476x | 0.641x | yes |
| mlp_down | down_proj | 1 | 8192 | 2048 | 3.5 | 0.2238 | 0.150 | 0.074x | 0.100x | no |
| mlp_down | down_proj | 2 | 8192 | 2048 | 2 | 0.0374 | 1.794 | 0.484x | 0.582x | yes |
| mlp_down | down_proj | 2 | 8192 | 2048 | 2.5 | 0.0370 | 1.814 | 0.490x | 0.588x | yes |
| mlp_down | down_proj | 2 | 8192 | 2048 | 3 | 0.0421 | 1.593 | 0.430x | 0.517x | yes |
| mlp_down | down_proj | 2 | 8192 | 2048 | 3.5 | 0.1998 | 0.336 | 0.091x | 0.109x | no |
| mlp_down | down_proj | 4 | 8192 | 2048 | 2 | 0.0377 | 3.564 | 0.478x | 0.592x | yes |
| mlp_down | down_proj | 4 | 8192 | 2048 | 2.5 | 0.0371 | 3.620 | 0.486x | 0.602x | yes |
| mlp_down | down_proj | 4 | 8192 | 2048 | 3 | 0.0422 | 3.181 | 0.427x | 0.529x | yes |
| mlp_down | down_proj | 4 | 8192 | 2048 | 3.5 | 0.2005 | 0.669 | 0.090x | 0.111x | yes |
| mlp_down | down_proj | 8 | 8192 | 2048 | 2 | 0.0377 | 7.112 | 0.438x | 0.588x | yes |
| mlp_down | down_proj | 8 | 8192 | 2048 | 2.5 | 0.0371 | 7.228 | 0.445x | 0.598x | yes |
| mlp_down | down_proj | 8 | 8192 | 2048 | 3 | 0.0421 | 6.369 | 0.393x | 0.527x | yes |
| mlp_down | down_proj | 8 | 8192 | 2048 | 3.5 | 0.2015 | 1.332 | 0.082x | 0.110x | no |
| mlp_down | down_proj | 16 | 8192 | 2048 | 2 | 0.0379 | 14.152 | 0.474x | 0.590x | yes |
| mlp_down | down_proj | 16 | 8192 | 2048 | 2.5 | 0.0376 | 14.278 | 0.478x | 0.596x | yes |
| mlp_down | down_proj | 16 | 8192 | 2048 | 3 | 0.0425 | 12.633 | 0.423x | 0.527x | yes |
| mlp_down | down_proj | 16 | 8192 | 2048 | 3.5 | 0.2070 | 2.594 | 0.087x | 0.108x | yes |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | 2 | 0.0348 | 0.966 | 0.286x | 0.518x | yes |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | 2.5 | 0.0342 | 0.982 | 0.291x | 0.527x | yes |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | 3 | 0.0347 | 0.968 | 0.287x | 0.520x | yes |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | 3.5 | 0.2363 | 0.142 | 0.042x | 0.076x | yes |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | 2 | 0.0348 | 1.926 | 0.295x | 0.509x | yes |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | 2.5 | 0.0342 | 1.963 | 0.300x | 0.518x | yes |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | 3 | 0.0348 | 1.929 | 0.295x | 0.510x | yes |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | 3.5 | 0.1962 | 0.342 | 0.052x | 0.090x | yes |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | 2 | 0.0349 | 3.841 | 0.294x | 0.516x | yes |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | 2.5 | 0.0344 | 3.903 | 0.299x | 0.524x | yes |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | 3 | 0.0348 | 3.860 | 0.295x | 0.518x | yes |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | 3.5 | 0.1992 | 0.674 | 0.052x | 0.090x | no |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | 2 | 0.0352 | 7.626 | 0.285x | 0.513x | yes |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | 2.5 | 0.0348 | 7.717 | 0.288x | 0.519x | yes |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | 3 | 0.0351 | 7.640 | 0.285x | 0.514x | yes |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | 3.5 | 0.1996 | 1.345 | 0.050x | 0.090x | no |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | 2 | 0.0354 | 15.156 | 0.308x | 0.508x | yes |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | 2.5 | 0.0352 | 15.266 | 0.310x | 0.511x | yes |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | 3 | 0.0355 | 15.121 | 0.307x | 0.507x | yes |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | 3.5 | 0.2065 | 2.600 | 0.053x | 0.087x | no |

The four unique geometries preserve all seven Llama 3.2 1B roles: q_proj/o_proj (2048×2048), k_proj/v_proj (2048×512), gate_proj/up_proj (2048×8192), and down_proj (8192×2048).
