# QQQ: W4A8 smoothing and compensation

## Sources and finding

Zhang et al., [QQQ: Quality Quattuor-Bit Quantization for Large Language Models,
v3](https://arxiv.org/html/2406.09904v3);
[authors' code](https://github.com/HandH1998/QQQ).

QQQ combines adaptive smoothing and Hessian-based compensation for four-bit
weights and eight-bit activations. The paper co-designs W4A8 kernels to address
both prefill and decode. Reported speedups depend on the measured kernels,
shapes and hardware; they are not transferable performance guarantees.

## Repository evidence

[gptqmodel/quantization/qqq.py](https://github.com/ModelCloud/QvQ/blob/4cbd1bc564c5faff7ce9fd0911b16a42ffefbeac/gptqmodel/quantization/qqq.py) and
[gptqmodel/looper/qqq_processor.py](https://github.com/ModelCloud/QvQ/blob/4cbd1bc564c5faff7ce9fd0911b16a42ffefbeac/gptqmodel/looper/qqq_processor.py) implement the dedicated route;
[dispatch](https://github.com/ModelCloud/QvQ/blob/4cbd1bc564c5faff7ce9fd0911b16a42ffefbeac/gptqmodel/models/base.py) selects QQQ separately from
GPTQ and AWQ. Check the QQQ consumer's scale tensors and workspace lifetime when
changing packing or execution.

## Recovery implications

This is a useful comparison for activation-aware native low-bit inference, but
W4A8 is neither W4A4 nor NVFP4. Replacing its activation quantizer changes the
error distribution and scale contract.

Ablate smoothing, weight compensation and runtime activation rounding separately.
Capture the actual native output before fitting [EoRA-like recovery](eora.md).
Report full-operator conversion/correction overhead and both token regimes.
