# QVQ LR H100 comparison — focused M sweep

QVQ V2B2-P32-LR versus matched W4 GPTQ Marlin/Machete baselines for the Llama 3.2 1B projection shapes.

## Measurement contract

- Current benchmark commit: `9b49ce5931960e8130eb5d94c684700d6bc1c4c6`
- Previous benchmark commits, in lookup priority: `3f8b70436b0edea5f53004355b61326e1db0337e`
- GPU: physical `1`, `NVIDIA H100`, PCI `00000000:44:00.0`, UUID `GPU-f5ea03cf-efa4-9807-7de5-b174957a1348`, CC `9.0`, 132 SMs
- M values: `[1, 2, 4, 8, 16]`; QVQ rates: `[2.0, 2.5, 3.0, 3.5]`
- Timing: one CUDA Graph replay with internal external CUDA events; host launch gaps excluded.
- `xMarlin`/`xMachete` = baseline median latency / QVQ median latency; above 1.0x is faster.
- `Better than last` is `yes` only when current QVQ median latency is strictly lower than the previous run; equal/regressed is `no`.

| Shape | Roles | M | K | N | W | Median ms | Logical TFLOP/s | xMarlin | xMachete | Better than last |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | 2 | 0.0310 | 1.081 | 0.323x | 0.578x | yes |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | 2.5 | 0.0312 | 1.074 | 0.321x | 0.574x | yes |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | 3 | 0.0304 | 1.103 | 0.329x | 0.590x | yes |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | 3.5 | 0.0321 | 1.045 | 0.312x | 0.559x | yes |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | 2 | 0.0311 | 2.155 | 0.327x | 0.580x | yes |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | 2.5 | 0.0314 | 2.140 | 0.325x | 0.576x | yes |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | 3 | 0.0307 | 2.185 | 0.332x | 0.588x | yes |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | 3.5 | 0.0323 | 2.076 | 0.315x | 0.558x | yes |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | 2 | 0.0312 | 4.302 | 0.328x | 0.579x | yes |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | 2.5 | 0.0315 | 4.260 | 0.325x | 0.573x | yes |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | 3 | 0.0304 | 4.408 | 0.336x | 0.593x | yes |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | 3.5 | 0.0328 | 4.098 | 0.313x | 0.552x | yes |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | 2 | 0.0316 | 8.490 | 0.317x | 0.565x | yes |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | 2.5 | 0.0319 | 8.422 | 0.314x | 0.560x | yes |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | 3 | 0.0308 | 8.706 | 0.325x | 0.579x | yes |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | 3.5 | 0.0335 | 8.020 | 0.299x | 0.533x | yes |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | 2 | 0.0320 | 16.777 | 0.343x | 0.564x | yes |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | 2.5 | 0.0323 | 16.611 | 0.340x | 0.558x | yes |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | 3 | 0.0312 | 17.190 | 0.351x | 0.578x | yes |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | 3.5 | 0.0338 | 15.903 | 0.325x | 0.535x | yes |

The four unique geometries preserve all seven Llama 3.2 1B roles: q_proj/o_proj (2048×2048), k_proj/v_proj (2048×512), gate_proj/up_proj (2048×8192), and down_proj (8192×2048).
