# Rotation, equivalent scaling and lossy smoothing

## Primary references

- [QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs](https://arxiv.org/abs/2404.00456).
- [SmoothQuant: Accurate and Efficient Post-Training Quantization for Large
  Language Models](https://arxiv.org/abs/2211.10438).

QuaRot uses rotations to redistribute outliers while preserving the
unquantized function under matched transforms. SmoothQuant uses equivalent
channel scaling to move activation quantization difficulty toward weights.
Neither principle means arbitrary scaling through a nonlinearity is valid.

## Repository evidence

[gptqmodel/quantization/rotation/rotation.py](https://github.com/ModelCloud/QvQ/blob/4cbd1bc564c5faff7ce9fd0911b16a42ffefbeac/gptqmodel/quantization/rotation/rotation.py) explicitly credits
QuaRot and includes norm fusion, embedding/head and attention/MLP rotations.
QVQ also has its own transform planner/runtime:
[gptqmodel/quantization/qvq_transform_planner.py](https://github.com/ModelCloud/QvQ/blob/4cbd1bc564c5faff7ce9fd0911b16a42ffefbeac/gptqmodel/quantization/qvq_transform_planner.py) and
[gptqmodel/quantization/qvq_transform_runtime.py](https://github.com/ModelCloud/QvQ/blob/4cbd1bc564c5faff7ce9fd0911b16a42ffefbeac/gptqmodel/quantization/qvq_transform_runtime.py).

[gptqmodel/quantization/fallback_smooth.py](https://github.com/ModelCloud/QvQ/blob/4cbd1bc564c5faff7ce9fd0911b16a42ffefbeac/gptqmodel/quantization/fallback_smooth.py) contains smoothing/clipping
and MSE helpers. Their presence is not proof that the complete published
SmoothQuant W8A8 pipeline is exposed as a standalone supported method.
Some helpers deliberately alter values; distinguish these from invertible
reparameterization.

## Recovery implications

Track the basis and scale at every boundary, including normalization, biases,
residual additions, gated products and tied parameters. Validate dense
equivalence before introducing quantization and measure FP rounding separately.

Collect scales and fit factors in the coordinates that runtime consumes.
Online transforms belong in timing and cannot always be folded away.
See [SwiGLU and QVQ enhancements](qvq-enhancements.md) for a concrete case
where the gate must remain unchanged.
