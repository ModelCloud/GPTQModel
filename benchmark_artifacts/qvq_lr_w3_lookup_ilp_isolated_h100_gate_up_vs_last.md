# QVQ LR H100 comparison — focused M sweep

QVQ V2B2-P32-LR versus matched W4 GPTQ Marlin/Machete baselines for the Llama 3.2 1B projection shapes.

## Measurement contract

- Current benchmark commit: `30c62b5336e90dc8db66fb9568b94a5713e5da98`
- Previous benchmark commits, in lookup priority: `90df1c560c347b06bbfb5cb69a0821f7452a4570`
- GPU: physical `1`, `NVIDIA H100`, PCI `00000000:44:00.0`, UUID `GPU-f5ea03cf-efa4-9807-7de5-b174957a1348`, CC `9.0`, 132 SMs
- M values: `[1, 2, 4, 8, 16]`; QVQ rates: `[2.0, 2.5, 3.0, 3.5]`
- Timing: one CUDA Graph replay with internal external CUDA events; host launch gaps excluded.
- `xMarlin`/`xMachete` = baseline median latency / QVQ median latency; above 1.0x is faster.
- `Better than last` is `yes` only when current QVQ median latency is strictly lower than the previous run; equal/regressed is `no`.

| Shape | Roles | M | K | N | W | Median ms | Logical TFLOP/s | xMarlin | xMachete | Better than last |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | 2 | 0.0307 | 1.093 | 0.322x | 0.584x | no |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | 2.5 | 0.0313 | 1.072 | 0.315x | 0.573x | no |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | 3 | 0.0284 | 1.180 | 0.347x | 0.630x | yes |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | 3.5 | 0.0321 | 1.044 | 0.307x | 0.558x | no |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | 2 | 0.0307 | 2.189 | 0.333x | 0.592x | no |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | 2.5 | 0.0314 | 2.136 | 0.325x | 0.577x | no |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | 3 | 0.0285 | 2.358 | 0.359x | 0.637x | yes |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | 3.5 | 0.0324 | 2.070 | 0.315x | 0.560x | no |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | 2 | 0.0308 | 4.351 | 0.335x | 0.586x | no |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | 2.5 | 0.0315 | 4.263 | 0.328x | 0.574x | yes |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | 3 | 0.0286 | 4.689 | 0.361x | 0.632x | yes |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | 3.5 | 0.0327 | 4.104 | 0.316x | 0.553x | no |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | 2 | 0.0312 | 8.599 | 0.321x | 0.577x | no |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | 2.5 | 0.0317 | 8.465 | 0.316x | 0.568x | yes |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | 3 | 0.0286 | 9.373 | 0.350x | 0.628x | yes |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | 3.5 | 0.0334 | 8.027 | 0.300x | 0.538x | no |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | 2 | 0.0316 | 17.007 | 0.345x | 0.570x | no |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | 2.5 | 0.0322 | 16.652 | 0.338x | 0.558x | yes |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | 3 | 0.0291 | 18.447 | 0.374x | 0.618x | yes |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | 3.5 | 0.0337 | 15.948 | 0.324x | 0.535x | no |

The four unique geometries preserve all seven Llama 3.2 1B roles: q_proj/o_proj (2048×2048), k_proj/v_proj (2048×512), gate_proj/up_proj (2048×8192), and down_proj (8192×2048).
