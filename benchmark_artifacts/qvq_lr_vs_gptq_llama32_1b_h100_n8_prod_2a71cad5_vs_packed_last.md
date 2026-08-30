# QVQ LR H100 comparison — focused M sweep

QVQ V2B2-P32-LR versus matched W4 GPTQ Marlin/Machete baselines for the Llama 3.2 1B projection shapes.

## Measurement contract

- Current benchmark commit: `2a71cad5b1d418a764149fcad08b37fd0d511d32`
- Previous benchmark commit: `8b659d419138263144d0199fe46e7613eaa9f68d`
- GPU: physical `1`, `NVIDIA H100`, PCI `00000000:44:00.0`, UUID `GPU-f5ea03cf-efa4-9807-7de5-b174957a1348`, CC `9.0`, 132 SMs
- M values: `[1, 2, 4, 8, 16]`; QVQ rates: `[2.0, 2.5]`
- Timing: one CUDA Graph replay with internal external CUDA events; host launch gaps excluded.
- `xMarlin`/`xMachete` = baseline median latency / QVQ median latency; above 1.0x is faster.
- `Better than last` is `yes` only when current QVQ median latency is strictly lower than the previous run; equal/regressed is `no`.

| Shape | Roles | M | K | N | W | Median ms | Logical TFLOP/s | xMarlin | xMachete | Better than last |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| attn_kv | k_proj/v_proj | 1 | 2048 | 512 | 2 | 0.0126 | 0.167 | 1.785x | 1.172x | no |
| attn_kv | k_proj/v_proj | 1 | 2048 | 512 | 2.5 | 0.0133 | 0.158 | 1.690x | 1.110x | no |
| attn_kv | k_proj/v_proj | 2 | 2048 | 512 | 2 | 0.0127 | 0.330 | 2.045x | 1.171x | no |
| attn_kv | k_proj/v_proj | 2 | 2048 | 512 | 2.5 | 0.0129 | 0.324 | 2.012x | 1.152x | no |
| attn_kv | k_proj/v_proj | 4 | 2048 | 512 | 2 | 0.0128 | 0.658 | 2.065x | 1.181x | no |
| attn_kv | k_proj/v_proj | 4 | 2048 | 512 | 2.5 | 0.0134 | 0.627 | 1.969x | 1.126x | no |
| attn_kv | k_proj/v_proj | 8 | 2048 | 512 | 2 | 0.0127 | 1.324 | 1.827x | 1.159x | no |
| attn_kv | k_proj/v_proj | 8 | 2048 | 512 | 2.5 | 0.0129 | 1.301 | 1.795x | 1.139x | no |
| attn_kv | k_proj/v_proj | 16 | 2048 | 512 | 2 | 0.0129 | 2.605 | 1.906x | 1.168x | no |
| attn_kv | k_proj/v_proj | 16 | 2048 | 512 | 2.5 | 0.0130 | 2.576 | 1.885x | 1.155x | no |
| attn_qo | q_proj/o_proj | 1 | 2048 | 2048 | 2 | 0.0168 | 0.501 | 0.870x | 0.921x | yes |
| attn_qo | q_proj/o_proj | 1 | 2048 | 2048 | 2.5 | 0.0171 | 0.490 | 0.852x | 0.902x | yes |
| attn_qo | q_proj/o_proj | 2 | 2048 | 2048 | 2 | 0.0185 | 0.909 | 0.884x | 0.828x | yes |
| attn_qo | q_proj/o_proj | 2 | 2048 | 2048 | 2.5 | 0.0194 | 0.864 | 0.841x | 0.788x | no |
| attn_qo | q_proj/o_proj | 4 | 2048 | 2048 | 2 | 0.0182 | 1.841 | 0.899x | 0.847x | yes |
| attn_qo | q_proj/o_proj | 4 | 2048 | 2048 | 2.5 | 0.0196 | 1.708 | 0.834x | 0.786x | no |
| attn_qo | q_proj/o_proj | 8 | 2048 | 2048 | 2 | 0.0185 | 3.631 | 0.805x | 0.833x | yes |
| attn_qo | q_proj/o_proj | 8 | 2048 | 2048 | 2.5 | 0.0196 | 3.430 | 0.760x | 0.787x | no |
| attn_qo | q_proj/o_proj | 16 | 2048 | 2048 | 2 | 0.0186 | 7.232 | 0.852x | 0.839x | yes |
| attn_qo | q_proj/o_proj | 16 | 2048 | 2048 | 2.5 | 0.0199 | 6.732 | 0.793x | 0.781x | no |
| mlp_down | down_proj | 1 | 8192 | 2048 | 2 | 0.0391 | 0.858 | 0.423x | 0.567x | yes |
| mlp_down | down_proj | 1 | 8192 | 2048 | 2.5 | 0.0408 | 0.822 | 0.405x | 0.542x | yes |
| mlp_down | down_proj | 2 | 8192 | 2048 | 2 | 0.0478 | 1.403 | 0.381x | 0.463x | yes |
| mlp_down | down_proj | 2 | 8192 | 2048 | 2.5 | 0.0500 | 1.342 | 0.365x | 0.443x | yes |
| mlp_down | down_proj | 4 | 8192 | 2048 | 2 | 0.0481 | 2.792 | 0.380x | 0.463x | yes |
| mlp_down | down_proj | 4 | 8192 | 2048 | 2.5 | 0.0500 | 2.683 | 0.365x | 0.445x | no |
| mlp_down | down_proj | 8 | 8192 | 2048 | 2 | 0.0482 | 5.570 | 0.347x | 0.461x | yes |
| mlp_down | down_proj | 8 | 8192 | 2048 | 2.5 | 0.0501 | 5.358 | 0.334x | 0.443x | yes |
| mlp_down | down_proj | 16 | 8192 | 2048 | 2 | 0.0485 | 11.078 | 0.372x | 0.457x | yes |
| mlp_down | down_proj | 16 | 8192 | 2048 | 2.5 | 0.0508 | 10.578 | 0.356x | 0.436x | no |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | 2 | 0.0502 | 0.668 | 0.196x | 0.356x | yes |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | 2.5 | 0.0486 | 0.691 | 0.202x | 0.369x | no |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | 2 | 0.0503 | 1.333 | 0.201x | 0.356x | yes |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | 2.5 | 0.0488 | 1.375 | 0.207x | 0.367x | yes |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | 2 | 0.0506 | 2.654 | 0.201x | 0.353x | yes |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | 2.5 | 0.0491 | 2.736 | 0.207x | 0.364x | yes |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | 2 | 0.0502 | 5.352 | 0.197x | 0.358x | yes |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | 2.5 | 0.0492 | 5.452 | 0.201x | 0.365x | yes |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | 2 | 0.0503 | 10.669 | 0.214x | 0.357x | yes |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | 2.5 | 0.0493 | 10.880 | 0.218x | 0.364x | yes |

The four unique geometries preserve all seven Llama 3.2 1B roles: q_proj/o_proj (2048×2048), k_proj/v_proj (2048×512), gate_proj/up_proj (2048×8192), and down_proj (8192×2048).
