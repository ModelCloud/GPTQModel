# Supported methods and enhancement inventory

This is a source-audited navigation map, not a universal hardware/model support
matrix. Audit revision:
[`4cbd1bc`](https://github.com/ModelCloud/QvQ/tree/4cbd1bc564c5faff7ce9fd0911b16a42ffefbeac).

Authority: [config enums and mappings](https://github.com/ModelCloud/QvQ/blob/4cbd1bc564c5faff7ce9fd0911b16a42ffefbeac/gptqmodel/quantization/config.py),
[model dispatch](https://github.com/ModelCloud/QvQ/blob/4cbd1bc564c5faff7ce9fd0911b16a42ffefbeac/gptqmodel/models/base.py), and
[weight-only processor](https://github.com/ModelCloud/QvQ/blob/4cbd1bc564c5faff7ce9fd0911b16a42ffefbeac/gptqmodel/looper/weight_only_processor.py).
All ten METHOD members in that revision appear below.

## Method/config coverage

| METHOD member | Research note | Observed scope |
|---|---|---|
| GPTQ | [GPTQ](gptq.md) | Hessian-driven quantizer; also the serialized method used by RTNConfig |
| AWQ | [AWQ](awq.md) | Dedicated activation-aware weight quantization processor |
| QQQ | [QQQ](qqq.md) | Dedicated W4A8 algorithm/processor family |
| PARO | [ParoQuant](paroquant.md) | Learned transform/quantization optimization and dedicated processor |
| EXL3 | [EXL3](exl3.md) | Vendored ExLlamaV3 calibration/packing; local lifecycle exclusions apply |
| QVQ | [QVQ enhancements](qvq-enhancements.md), [QTIP](qtip.md), [P32](p32.md) | Trellis quantizer, transforms, banks and runtime contracts |
| FP8 | [FP8](fp8.md) | Direct weight packing; separate from QVQ activation and KV policies |
| BITSANDBYTES | [bitsandbytes](bitsandbytes.md) | Direct weight packing through the backend integration |
| GGUF | [GGUF](gguf.md) | Container/tensor-type family with direct packing and multiple consumers |
| MXFP4 | [MXFP4](mxfp4.md) | Config and CPU weight-only module; complete lifecycle coverage needs caller verification |

[RTN](rtn.md) is a separate configuration/lifecycle, not a METHOD enum member.
A supported enum is not proof of every quantize/save/load/backend combination.

## Enhancement coverage

| Concern | Notes |
|---|---|
| Upstream input mismatch | [GPTAQ](gptaq.md) |
| First-order latent-weight deviation | [FOEM](foem.md) |
| Group ordering | [GAR](gar.md) |
| Rotation and smoothing | [Rotation/scaling](rotation-and-smoothing.md), [AWQ](awq.md), [ParoQuant](paroquant.md) |
| Statistical fidelity and rate allocation | [SLQ](slq.md) |
| Local / runtime-output low-rank correction | [EoRA](eora.md) |
| Model-sensitive rounding | [YAQA](yaqa.md) |
| SwiGLU, SU/SV alignment, propagation, adjacent rounding, exact Viterbi pruning | [QVQ enhancements](qvq-enhancements.md) |
| A4 and KV numeric calibration | [NVFP4 hybrid PTQ](nvfp4-hybrid-ptq.md), [FP8](fp8.md) |
| Inference-time recurrent intervention | [Recirculation](recirculation.md); QVQ recovery benefit remains unproven |

## Avoid false equivalences

- Method, checkpoint format, packer, backend, activation dtype and KV dtype are
  independent declarations. Marlin, Machete, Triton and ExLlama execution names
  do not by themselves identify the quantization research method.
- GPTAQ's historical GPTQv2 name is not FORMAT.GPTQ_V2.
- A bitsandbytes NF4/FP4, MXFP4, scalar INT4 or QVQ W4 label is not NVFP4.
- SmoothQuant is related scientific context, not a separately verified METHOD
  in this tree. Native NVFP4 integration is not claimed by adding a paper note.
- Helper availability does not imply that it ran, is enabled by default, or
  composes with every other helper. Record exclusions and inspect actual dispatch.
- Refer to [the index's evidence policy](README.md#evidence-convention) before
  using these notes as support for a quality or performance claim.
