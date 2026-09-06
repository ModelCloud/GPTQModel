# FOEM: first-order error compensation

## Sources and finding

Zheng et al., [First-Order Error Matters: Accurate Compensation for Quantized
Large Language Models](https://ojs.aaai.org/index.php/AAAI/article/view/40123),
AAAI 2026, DOI 10.1609/aaai.v40i34.40123.

The paper argues that progressive compensation moves latent weights away from
the original point, making omission of first-order terms problematic. FOEM
approximates that contribution from weight deviations and second-order structure.
It is compensation during PTQ, not an EoRA adapter or evidence that activation
scales were gradient-trained.

## Repository evidence

[gptqmodel/quantization/foem.py](https://github.com/ModelCloud/QvQ/blob/4cbd1bc564c5faff7ce9fd0911b16a42ffefbeac/gptqmodel/quantization/foem.py) extends GPTQ. Its update includes
the latent-versus-original weight deviation multiplied by `beta`.
[FOEMConfig](https://github.com/ModelCloud/QvQ/blob/4cbd1bc564c5faff7ce9fd0911b16a42ffefbeac/gptqmodel/quantization/config.py) separates
`alpha` (the GPTAQ component) and `beta` (FOEM's component), and documents
zero-coefficient reductions to the corresponding baselines.

Some source headers retain GPTQv2 ancestry; the configuration explicitly cites
the FOEM paper. Record this as a repository adaptation, not an identical copy
of every paper experiment.

## Recovery implications

Ablate alpha and beta independently, retain the original weight reference,
and compare exported results on held-out data. Native activation capture and
FOEM's latent-weight term address different error sources.
Combining FOEM with [GPTAQ](gptaq.md), rotation or [EoRA](eora.md) needs
measurement; the best coefficient is not universal.
