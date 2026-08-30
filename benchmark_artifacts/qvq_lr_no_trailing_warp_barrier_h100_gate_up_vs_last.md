# QVQ LR H100 comparison — focused M sweep

QVQ V2B2-P32-LR versus matched W4 GPTQ Marlin/Machete baselines for the Llama 3.2 1B projection shapes.

## Measurement contract

- Current benchmark commit: `3f8b70436b0edea5f53004355b61326e1db0337e`
- Previous benchmark commits, in lookup priority: `2ef2eac6d9c98ece16d91578526e375e5829a143`, `041ab457e7847203d7ea685be769439dcca87933`
- GPU: physical `1`, `NVIDIA H100`, PCI `00000000:44:00.0`, UUID `GPU-f5ea03cf-efa4-9807-7de5-b174957a1348`, CC `9.0`, 132 SMs
- M values: `[1, 2, 4, 8, 16]`; QVQ rates: `[2.0, 2.5, 3.0, 3.5]`
- Timing: one CUDA Graph replay with internal external CUDA events; host launch gaps excluded.
- `xMarlin`/`xMachete` = baseline median latency / QVQ median latency; above 1.0x is faster.
- `Better than last` is `yes` only when current QVQ median latency is strictly lower than the previous run; equal/regressed is `no`.

| Shape | Roles | M | K | N | W | Median ms | Logical TFLOP/s | xMarlin | xMachete | Better than last |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | 2 | 0.0334 | 1.003 | 0.297x | 0.534x | yes |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | 2.5 | 0.0331 | 1.014 | 0.300x | 0.539x | yes |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | 3 | 0.0309 | 1.087 | 0.321x | 0.578x | yes |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | 3.5 | 0.0343 | 0.979 | 0.289x | 0.521x | yes |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | 2 | 0.0335 | 2.003 | 0.307x | 0.538x | yes |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | 2.5 | 0.0332 | 2.018 | 0.309x | 0.542x | yes |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | 3 | 0.0311 | 2.158 | 0.330x | 0.580x | yes |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | 3.5 | 0.0346 | 1.941 | 0.297x | 0.522x | yes |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | 2 | 0.0337 | 3.987 | 0.306x | 0.535x | yes |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | 2.5 | 0.0335 | 4.004 | 0.307x | 0.537x | yes |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | 3 | 0.0311 | 4.320 | 0.332x | 0.580x | yes |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | 3.5 | 0.0349 | 3.848 | 0.295x | 0.517x | yes |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | 2 | 0.0340 | 7.895 | 0.295x | 0.525x | yes |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | 2.5 | 0.0339 | 7.921 | 0.296x | 0.527x | yes |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | 3 | 0.0313 | 8.573 | 0.320x | 0.570x | yes |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | 3.5 | 0.0357 | 7.517 | 0.280x | 0.500x | yes |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | 2 | 0.0341 | 15.738 | 0.321x | 0.525x | yes |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | 2.5 | 0.0343 | 15.665 | 0.319x | 0.523x | yes |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | 3 | 0.0316 | 16.972 | 0.346x | 0.567x | yes |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | 3.5 | 0.0354 | 15.156 | 0.309x | 0.506x | yes |

The four unique geometries preserve all seven Llama 3.2 1B roles: q_proj/o_proj (2048×2048), k_proj/v_proj (2048×512), gate_proj/up_proj (2048×8192), and down_proj (8192×2048).
