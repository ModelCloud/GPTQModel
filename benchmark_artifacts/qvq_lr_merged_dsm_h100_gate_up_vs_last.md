# QVQ LR H100 comparison — focused M sweep

QVQ V2B2-P32-LR versus matched W4 GPTQ Marlin/Machete baselines for the Llama 3.2 1B projection shapes.

## Measurement contract

- Current benchmark commit: `90df1c560c347b06bbfb5cb69a0821f7452a4570`
- Previous benchmark commits, in lookup priority: `9b49ce5931960e8130eb5d94c684700d6bc1c4c6`
- GPU: physical `1`, `NVIDIA H100`, PCI `00000000:44:00.0`, UUID `GPU-f5ea03cf-efa4-9807-7de5-b174957a1348`, CC `9.0`, 132 SMs
- M values: `[1, 2, 4, 8, 16]`; QVQ rates: `[2.0, 2.5, 3.0, 3.5]`
- Timing: one CUDA Graph replay with internal external CUDA events; host launch gaps excluded.
- `xMarlin`/`xMachete` = baseline median latency / QVQ median latency; above 1.0x is faster.
- `Better than last` is `yes` only when current QVQ median latency is strictly lower than the previous run; equal/regressed is `no`.

| Shape | Roles | M | K | N | W | Median ms | Logical TFLOP/s | xMarlin | xMachete | Better than last |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | 2 | 0.0305 | 1.100 | 0.322x | 0.592x | yes |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | 2.5 | 0.0312 | 1.077 | 0.315x | 0.579x | yes |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | 3 | 0.0299 | 1.122 | 0.329x | 0.604x | yes |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | 3.5 | 0.0321 | 1.046 | 0.306x | 0.563x | yes |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | 2 | 0.0306 | 2.190 | 0.331x | 0.586x | yes |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | 2.5 | 0.0313 | 2.142 | 0.324x | 0.573x | yes |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | 3 | 0.0300 | 2.233 | 0.338x | 0.597x | yes |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | 3.5 | 0.0323 | 2.076 | 0.314x | 0.555x | no |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | 2 | 0.0308 | 4.360 | 0.335x | 0.589x | yes |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | 2.5 | 0.0316 | 4.254 | 0.327x | 0.575x | no |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | 3 | 0.0301 | 4.453 | 0.342x | 0.601x | yes |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | 3.5 | 0.0326 | 4.116 | 0.316x | 0.556x | yes |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | 2 | 0.0312 | 8.608 | 0.321x | 0.578x | yes |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | 2.5 | 0.0319 | 8.422 | 0.314x | 0.565x | no |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | 3 | 0.0304 | 8.839 | 0.330x | 0.593x | yes |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | 3.5 | 0.0333 | 8.058 | 0.301x | 0.541x | yes |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | 2 | 0.0315 | 17.033 | 0.346x | 0.575x | yes |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | 2.5 | 0.0324 | 16.595 | 0.337x | 0.560x | no |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | 3 | 0.0306 | 17.540 | 0.357x | 0.592x | yes |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | 3.5 | 0.0337 | 15.948 | 0.324x | 0.538x | yes |

The four unique geometries preserve all seven Llama 3.2 1B roles: q_proj/o_proj (2048×2048), k_proj/v_proj (2048×512), gate_proj/up_proj (2048×8192), and down_proj (8192×2048).
