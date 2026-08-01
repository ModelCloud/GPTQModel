# Group Aware Reordering (GAR)

Group Aware Reordering (`act_group_aware`) is an activation-driven column reordering
scheme for GPTQ. It sorts columns inside each `group_size` block and sorts the blocks
themselves by the diagonal of the calibration Hessian, with the goal of placing the
most activation-sensitive weights in the earliest columns so GPTQ's error feedback
can protect them first.

```python
from gptqmodel import GPTQModel, QuantizeConfig

quant_config = QuantizeConfig(
    bits=4,
    group_size=128,
    desc_act=False,      # GAR and desc_act cannot both be True
    act_group_aware=True,
)
```

## When GAR helps

For the default `group_size=128`, GAR is usually neutral or helpful and is the
recommended activation ordering when `desc_act=False`. It has no inference-time cost
because all permutations are reversed before the checkpoint is saved.

## When GAR can hurt

With very small group sizes (`group_size <= 32`), GAR sorts columns into many small,
activation-homogeneous groups. Because each group contains only a few columns, the
ordering becomes tightly coupled to the calibration data. When the calibration
distribution differs from the downstream task distribution, the quantization can
overfit the calibration Hessian and underperform the natural channel ordering.

Empirical results on `Llama-3.2-1B-Instruct` (4-bit GPTQ, `imatrix_22 + nm_1024`
calibration):

| group_size | act_group_aware | gsm8k | arc acc | arc acc_norm | combined |
|---|---|---|---|---|---|
| 128 | True | 0.4251 | 0.3072 | 0.3413 | 1.0736 |
| 128 | False | 0.3730 | 0.2961 | 0.3328 | 1.0019 |
| 64 | True | 0.4400 | 0.2969 | 0.3345 | 1.0714 |
| 64 | False | 0.3929 | 0.3140 | 0.3430 | 1.0498 |
| 32 | True | 0.4285 | 0.3063 | 0.3396 | 1.0744 |
| 32 | False | **0.4376** | **0.3131** | **0.3498** | **1.1005** |

- With `act_group_aware=False`, smaller groups win: `gp32 > gp64 > gp128`.
- With `act_group_aware=True`, `gp32` loses its advantage and `gp128` becomes
  competitive, even though raw quantization error (`quant_log` loss, RTN RMSE,
  and packed dequant RMSE) all improve monotonically for smaller groups.

The reversal is therefore a calibration/Hessian interaction, not a bug in the
quantization math or packing.

## Automatic safeguard

Starting with this release, `act_group_aware` is automatically disabled at
quantization time for `group_size <= 32` when it was not explicitly requested.
A warning is emitted so the behavior is visible:

```text
QuantizeConfig: group_size=32 <= 32; auto-disabling `act_group_aware` because
activation-aware reordering overfits the calibration Hessian for small groups.
```

If you explicitly set `act_group_aware=True`, the safeguard is bypassed and GAR
remains enabled. This is useful for experiments and tests, but for production
workloads with `group_size <= 32` the empirical results recommend leaving GAR
off.

## References

* T. Gafni, A. Karnieli, Y. Hanani, "Dual Precision Quantization for Efficient
  and Accurate Deep Neural Networks Inference," CVPRW 2025.
