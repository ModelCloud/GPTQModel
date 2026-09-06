# AWQ: activation-aware weight scaling

## Sources and finding

Lin et al., [AWQ: Activation-aware Weight Quantization for LLM Compression and
Acceleration](https://arxiv.org/abs/2306.00978);
[authors' project](https://hanlab.mit.edu/projects/awq).

AWQ uses activation statistics to identify salient weight channels and searches
equivalent channel rescalings before weight quantization. The method targets
weight-only compression without gradient training. Protecting salient channels
through scaling does not require leaving those weights in mixed precision.

## Repository evidence

[gptqmodel/looper/awq_processor.py](https://github.com/ModelCloud/QvQ/blob/4cbd1bc564c5faff7ce9fd0911b16a42ffefbeac/gptqmodel/looper/awq_processor.py) owns the AWQ lifecycle; the
[AWQ package](https://github.com/ModelCloud/QvQ/blob/4cbd1bc564c5faff7ce9fd0911b16a42ffefbeac/gptqmodel/quantization/awq/__init__.py) and
[configuration](https://github.com/ModelCloud/QvQ/blob/4cbd1bc564c5faff7ce9fd0911b16a42ffefbeac/gptqmodel/quantization/config.py) are entry points.
[model dispatch](https://github.com/ModelCloud/QvQ/blob/4cbd1bc564c5faff7ce9fd0911b16a42ffefbeac/gptqmodel/models/base.py) selects AWQ-specific
processing and export handling.

AWQ checkpoint variants and backend packing are separate from scale search.
Trace the selected consumer's zero-point/symmetry convention rather than assuming
it matches GPTQ. Marlin or ExLlama execution is a backend choice, not a new AWQ
scientific method.

## Recovery implications

AWQ's channel rescaling is not an NVFP4 per-block activation scale and not an
additive EoRA correction. If combining them, fit correction against the operator
after AWQ transforms and export. Preserve compensating transforms in adjacent
modules, biases and fused projections.

Compare the same calibration/evaluation split and actual activation precision.
An AWQ W4A16 checkpoint does not establish native W4A4 support simply because
its weights are four bits.
