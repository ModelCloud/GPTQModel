# QVQ LR H100 comparison — focused M sweep

QVQ V2B2-P32-LR versus matched W4 GPTQ Marlin/Machete baselines for the Llama 3.2 1B projection shapes.

## Measurement contract

- Current benchmark commit: `56c635b2f84aeb859a34b71d721cc07cbd60654e`
- Previous benchmark commits, in lookup priority: `041ab457e7847203d7ea685be769439dcca87933`
- GPU: physical `1`, `NVIDIA H100`, PCI `00000000:44:00.0`, UUID `GPU-f5ea03cf-efa4-9807-7de5-b174957a1348`, CC `9.0`, 132 SMs
- M values: `[1, 2, 4, 8, 16]`; QVQ rates: `[3.5]`
- Timing: one CUDA Graph replay with internal external CUDA events; host launch gaps excluded.
- `xMarlin`/`xMachete` = baseline median latency / QVQ median latency; above 1.0x is faster.
- `Better than last` is `yes` only when current QVQ median latency is strictly lower than the previous run; equal/regressed is `no`.

| Shape | Roles | M | K | N | W | Median ms | Logical TFLOP/s | xMarlin | xMachete | Better than last |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| attn_kv | k_proj/v_proj | 1 | 2048 | 512 | 3.5 | 0.0224 | 0.094 | 0.995x | 0.660x | no |
| attn_kv | k_proj/v_proj | 2 | 2048 | 512 | 3.5 | 0.0204 | 0.205 | 1.270x | 0.746x | yes |
| attn_kv | k_proj/v_proj | 4 | 2048 | 512 | 3.5 | 0.0207 | 0.405 | 1.255x | 0.737x | yes |
| attn_kv | k_proj/v_proj | 8 | 2048 | 512 | 3.5 | 0.0215 | 0.780 | 1.067x | 0.687x | yes |
| attn_kv | k_proj/v_proj | 16 | 2048 | 512 | 3.5 | 0.0240 | 1.397 | 1.009x | 0.632x | no |
| attn_qo | q_proj/o_proj | 1 | 2048 | 2048 | 3.5 | 0.0151 | 0.554 | 0.956x | 1.021x | yes |
| attn_qo | q_proj/o_proj | 2 | 2048 | 2048 | 3.5 | 0.0158 | 1.065 | 1.031x | 0.973x | yes |
| attn_qo | q_proj/o_proj | 4 | 2048 | 2048 | 3.5 | 0.0157 | 2.138 | 1.040x | 0.987x | yes |
| attn_qo | q_proj/o_proj | 8 | 2048 | 2048 | 3.5 | 0.0159 | 4.220 | 0.925x | 0.964x | yes |
| attn_qo | q_proj/o_proj | 16 | 2048 | 2048 | 3.5 | 0.0162 | 8.306 | 0.966x | 0.952x | yes |
| mlp_down | down_proj | 1 | 8192 | 2048 | 3.5 | 0.0344 | 0.975 | 0.483x | 0.653x | yes |
| mlp_down | down_proj | 2 | 8192 | 2048 | 3.5 | 0.0402 | 1.670 | 0.453x | 0.561x | yes |
| mlp_down | down_proj | 4 | 8192 | 2048 | 3.5 | 0.0402 | 3.339 | 0.456x | 0.555x | yes |
| mlp_down | down_proj | 8 | 8192 | 2048 | 3.5 | 0.0403 | 6.658 | 0.417x | 0.560x | yes |
| mlp_down | down_proj | 16 | 8192 | 2048 | 3.5 | 0.0406 | 13.210 | 0.447x | 0.547x | yes |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | 3.5 | 0.0377 | 0.889 | 0.261x | 0.480x | yes |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | 3.5 | 0.0379 | 1.773 | 0.268x | 0.475x | yes |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | 3.5 | 0.0380 | 3.532 | 0.268x | 0.472x | yes |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | 3.5 | 0.0382 | 7.029 | 0.260x | 0.470x | yes |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | 3.5 | 0.0384 | 13.969 | 0.279x | 0.467x | yes |

The four unique geometries preserve all seven Llama 3.2 1B roles: q_proj/o_proj (2048×2048), k_proj/v_proj (2048×512), gate_proj/up_proj (2048×8192), and down_proj (8192×2048).
