# GPTAQ: asymmetric calibration (formerly GPTQv2)

## Sources and finding

[GPTAQ: Efficient Finetuning-Free Quantization for Asymmetric Calibration](https://arxiv.org/abs/2504.02692);
[official implementation](https://github.com/Intelligent-Computing-Lab-Panda/GPTAQ).

GPTAQ explicitly accounts for the mismatch between inputs propagated through
earlier quantized layers and the full-precision model's inputs/outputs. This
asymmetric calibration addresses accumulated upstream error without requiring
fine-tuning. “Asymmetric” here is not a claim about integer zero-points.

## Repository evidence

[gptqmodel/quantization/gptaq.py](https://github.com/ModelCloud/QvQ/blob/4cbd1bc564c5faff7ce9fd0911b16a42ffefbeac/gptqmodel/quantization/gptaq.py) extends GPTQ, consumes native input
captures, and accumulates input second moments plus a native-minus-current
cross term. [lifecycle](https://github.com/ModelCloud/QvQ/blob/4cbd1bc564c5faff7ce9fd0911b16a42ffefbeac/gptqmodel/models/base.py) supplies native
capture when configured. EXL3 explicitly excludes this route.

The historical GPTQv2 name must not be confused with FORMAT.GPTQ_V2, a serialized
checkpoint layout. The algorithm and format are independently specified.

## Recovery implications

Keep native and propagated captures aligned by sequence/token and module.
Replacing current inputs with A4 requires tracing whether that exact quantized
operand enters the statistics; the presence of native capture alone is insufficient.

This is upstream-error-aware weight compensation, whereas [EoRA](eora.md)
adds factors after an operator is selected. Compare each alone and combined
under matched calibration rather than assuming their gains add.
