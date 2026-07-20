# Quantization method map

Use the source as the authority; this reference is a navigation aid.

| Concern | Primary source | Notes |
| --- | --- | --- |
| Method and format enums | `gptqmodel/quantization/config.py` | Includes GPTQ, AWQ, QQQ, GGUF, FP8, bitsandbytes, EXL3, Paro, RTN, and multiple export layouts. |
| Method/format compatibility | `QUANT_METHOD_FORMAT_MAPPING` in `config.py` | Some formats are shared or ambiguous; use the config method as well as the format. |
| Calibration dispatch | `_quantize_with_calibration` in `gptqmodel/models/base.py` | Dispatches GPTQ, AWQ, QQQ, EXL3, ParoQuant, and related processors. |
| Weight-only lifecycle | `gptqmodel/looper/weight_only_processor.py` | Used for calibration-free formats such as GGUF, FP8, and bitsandbytes, plus lifecycle-compatible RTN paths. |
| GPTQ math | `gptqmodel/looper/gptq_processor.py` | Hessian-driven quantization and GPTQ-specific state. |
| AWQ math | `gptqmodel/looper/awq_processor.py` | Activation statistics, scale search, and weight restoration require AWQ-specific testing. |
| Public protocol | `gptqmodel/quantization/protocol.py` | Confirm support rather than assuming every config is protocol-addressable. |
| Packed execution | `gptqmodel/nn_modules/qlinear/` | Capabilities and tensor layouts vary by method, format, and backend. |

## Required test matrix

| Layer | Minimum evidence |
| --- | --- |
| Config | Invalid combinations reject; aliases normalize; `to_dict` and reload preserve public fields. |
| Math | Quantize/dequantize output is finite and within a stated tolerance of the dense reference. |
| State | Shared caches do not leak between layers or models; restored weights match expected lifecycle state. |
| Layout | Packed tensor shapes, zero convention, group indexing, and activation ordering match the consumer. |
| Persistence | Save/reload preserves config and produces equivalent inference. |
| Integration | Auto backend selection chooses only declared support; explicit unsupported requests fail clearly. |

For AWQ, verify whether a consumer expresses `zero_point` as the inverse of `sym`; do not transfer that convention to GPTQ without evidence. For GPTQ, exercise both activation-order states if supported. For any dynamic per-layer configuration, include at least one overridden layer and one default layer.
