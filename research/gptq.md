# GPTQ: second-order weight quantization

## Sources and finding

Frantar et al., [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained
Transformers](https://arxiv.org/abs/2210.17323), and the
[original implementation](https://github.com/IST-DASLab/gptq).

GPTQ uses approximate second-order information to compensate unquantized weights
as other weights are rounded. Its calibration objective is layer output
reconstruction. It is a weight quantization method; “activation-aware” calibration
does not mean activations are stored in four bits.

## Repository evidence

[gptqmodel/quantization/gptq.py](https://github.com/ModelCloud/QvQ/blob/4cbd1bc564c5faff7ce9fd0911b16a42ffefbeac/gptqmodel/quantization/gptq.py) implements the math;
[gptqmodel/looper/gptq_processor.py](https://github.com/ModelCloud/QvQ/blob/4cbd1bc564c5faff7ce9fd0911b16a42ffefbeac/gptqmodel/looper/gptq_processor.py) integrates it.
[gptqmodel/quantization/config.py](https://github.com/ModelCloud/QvQ/blob/4cbd1bc564c5faff7ce9fd0911b16a42ffefbeac/gptqmodel/quantization/config.py) exposes damping, scale-search and
adaptive-clipping choices. Those choices are repository behavior, not all
defaults or claims of the original paper.

RTN can serialize as METHOD.GPTQ too; a method label alone does not prove the
Hessian-driven solve ran. GPTQ_V2 is also a checkpoint-format name and must not
be confused with [GPTAQ's former GPTQv2 algorithm name](gptaq.md).

## Recovery implications

Use a matched [RTN](rtn.md) control to isolate the compensation benefit.
Record grouping, symmetry, ordering, damping, scale search and packing convention.
Compare [GPTAQ](gptaq.md), [FOEM](foem.md), [GAR](gar.md) and [EoRA](eora.md)
as separate interventions. Refit after changing the actual packed operator.

An input Hessian captures local sensitivity; it does not certify full-model
teacher agreement or NVFP4 activation recovery. Quantize/save/reload and evaluate
disjoint real inputs before attributing a quality change to the algorithm.
