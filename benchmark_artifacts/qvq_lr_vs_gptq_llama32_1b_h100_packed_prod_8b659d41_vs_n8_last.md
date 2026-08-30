# QVQ LR H100 comparison — focused M sweep

QVQ V2B2-P32-LR versus matched W4 GPTQ Marlin/Machete baselines for the Llama 3.2 1B projection shapes.

## Measurement contract

- Current benchmark commit: `8b659d419138263144d0199fe46e7613eaa9f68d`
- Previous benchmark commit: `41ffd8dbcae3fa11f0e59fa41d2e5138c32af470`
- GPU: physical `1`, `NVIDIA H100`, PCI `00000000:44:00.0`, UUID `GPU-f5ea03cf-efa4-9807-7de5-b174957a1348`, CC `9.0`, 132 SMs
- M values: `[1, 2, 4, 8, 16]`; QVQ rates: `[2.0, 2.5]`
- Timing: one CUDA Graph replay with internal external CUDA events; host launch gaps excluded.
- `xMarlin`/`xMachete` = baseline median latency / QVQ median latency; above 1.0x is faster.
- `Better than last` is `yes` only when current QVQ median latency is strictly lower than the previous run; equal/regressed is `no`.

| Shape | Roles | M | K | N | W | Median ms | Logical TFLOP/s | xMarlin | xMachete | Better than last |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| attn_kv | k_proj/v_proj | 1 | 2048 | 512 | 2 | 0.0125 | 0.167 | 1.764x | 1.186x | no |
| attn_kv | k_proj/v_proj | 1 | 2048 | 512 | 2.5 | 0.0130 | 0.162 | 1.707x | 1.148x | no |
| attn_kv | k_proj/v_proj | 2 | 2048 | 512 | 2 | 0.0125 | 0.335 | 2.079x | 1.208x | no |
| attn_kv | k_proj/v_proj | 2 | 2048 | 512 | 2.5 | 0.0128 | 0.326 | 2.027x | 1.178x | yes |
| attn_kv | k_proj/v_proj | 4 | 2048 | 512 | 2 | 0.0126 | 0.666 | 2.084x | 1.215x | no |
| attn_kv | k_proj/v_proj | 4 | 2048 | 512 | 2.5 | 0.0129 | 0.649 | 2.030x | 1.183x | no |
| attn_kv | k_proj/v_proj | 8 | 2048 | 512 | 2 | 0.0126 | 1.334 | 1.836x | 1.181x | no |
| attn_kv | k_proj/v_proj | 8 | 2048 | 512 | 2.5 | 0.0124 | 1.350 | 1.857x | 1.194x | yes |
| attn_kv | k_proj/v_proj | 16 | 2048 | 512 | 2 | 0.0129 | 2.605 | 1.886x | 1.180x | no |
| attn_kv | k_proj/v_proj | 16 | 2048 | 512 | 2.5 | 0.0127 | 2.648 | 1.917x | 1.199x | yes |
| attn_qo | q_proj/o_proj | 1 | 2048 | 2048 | 2 | 0.0185 | 0.454 | 0.786x | 0.834x | no |
| attn_qo | q_proj/o_proj | 1 | 2048 | 2048 | 2.5 | 0.0183 | 0.457 | 0.791x | 0.839x | no |
| attn_qo | q_proj/o_proj | 2 | 2048 | 2048 | 2 | 0.0190 | 0.882 | 0.854x | 0.792x | no |
| attn_qo | q_proj/o_proj | 2 | 2048 | 2048 | 2.5 | 0.0191 | 0.880 | 0.853x | 0.791x | yes |
| attn_qo | q_proj/o_proj | 4 | 2048 | 2048 | 2 | 0.0188 | 1.783 | 0.864x | 0.820x | no |
| attn_qo | q_proj/o_proj | 4 | 2048 | 2048 | 2.5 | 0.0191 | 1.758 | 0.852x | 0.808x | no |
| attn_qo | q_proj/o_proj | 8 | 2048 | 2048 | 2 | 0.0190 | 3.525 | 0.776x | 0.794x | no |
| attn_qo | q_proj/o_proj | 8 | 2048 | 2048 | 2.5 | 0.0191 | 3.507 | 0.772x | 0.790x | no |
| attn_qo | q_proj/o_proj | 16 | 2048 | 2048 | 2 | 0.0196 | 6.853 | 0.794x | 0.792x | no |
| attn_qo | q_proj/o_proj | 16 | 2048 | 2048 | 2.5 | 0.0194 | 6.933 | 0.803x | 0.802x | no |
| mlp_down | down_proj | 1 | 8192 | 2048 | 2 | 0.0437 | 0.767 | 0.377x | 0.582x | no |
| mlp_down | down_proj | 1 | 8192 | 2048 | 2.5 | 0.0464 | 0.724 | 0.356x | 0.549x | no |
| mlp_down | down_proj | 2 | 8192 | 2048 | 2 | 0.0497 | 1.350 | 0.362x | 0.499x | no |
| mlp_down | down_proj | 2 | 8192 | 2048 | 2.5 | 0.0515 | 1.302 | 0.349x | 0.481x | yes |
| mlp_down | down_proj | 4 | 8192 | 2048 | 2 | 0.0503 | 2.669 | 0.360x | 0.505x | no |
| mlp_down | down_proj | 4 | 8192 | 2048 | 2.5 | 0.0500 | 2.684 | 0.362x | 0.508x | yes |
| mlp_down | down_proj | 8 | 8192 | 2048 | 2 | 0.0500 | 5.369 | 0.331x | 0.503x | no |
| mlp_down | down_proj | 8 | 8192 | 2048 | 2.5 | 0.0502 | 5.345 | 0.329x | 0.500x | yes |
| mlp_down | down_proj | 16 | 8192 | 2048 | 2 | 0.0495 | 10.849 | 0.362x | 0.517x | no |
| mlp_down | down_proj | 16 | 8192 | 2048 | 2.5 | 0.0507 | 10.592 | 0.354x | 0.505x | yes |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | 2 | 0.0521 | 0.644 | 0.227x | 0.399x | no |
| mlp_gate_up | gate_proj/up_proj | 1 | 2048 | 8192 | 2.5 | 0.0479 | 0.700 | 0.246x | 0.433x | yes |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | 2 | 0.0530 | 1.266 | 0.226x | 0.406x | no |
| mlp_gate_up | gate_proj/up_proj | 2 | 2048 | 8192 | 2.5 | 0.0492 | 1.364 | 0.244x | 0.437x | no |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | 2 | 0.0524 | 2.562 | 0.230x | 0.397x | no |
| mlp_gate_up | gate_proj/up_proj | 4 | 2048 | 8192 | 2.5 | 0.0496 | 2.707 | 0.243x | 0.419x | no |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | 2 | 0.0512 | 5.245 | 0.229x | 0.414x | no |
| mlp_gate_up | gate_proj/up_proj | 8 | 2048 | 8192 | 2.5 | 0.0500 | 5.369 | 0.234x | 0.424x | no |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | 2 | 0.0533 | 10.073 | 0.237x | 0.409x | no |
| mlp_gate_up | gate_proj/up_proj | 16 | 2048 | 8192 | 2.5 | 0.0506 | 10.615 | 0.250x | 0.431x | no |

The four unique geometries preserve all seven Llama 3.2 1B roles: q_proj/o_proj (2048×2048), k_proj/v_proj (2048×512), gate_proj/up_proj (2048×8192), and down_proj (8192×2048).
