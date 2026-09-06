# YAQA: model-preserving adaptive rounding

## Naming and sources

This is the YAQA method interpreted from the request's “VAQA”; the repository
uses `qvq_yaqa`. No separate VAQA paper is identified.

- Tseng, Sun and De Sa,
  [Model-Preserving Adaptive Rounding, arXiv:2505.22988v1](https://arxiv.org/html/2505.22988v1).
- [Authors' implementation](https://github.com/Cornell-RelaxML/yaqa-quantization).
- QVQ [Fisher collector](https://github.com/ModelCloud/QvQ/blob/4cbd1bc564c5faff7ce9fd0911b16a42ffefbeac/gptqmodel/quantization/qvq_yaqa.py)
  and [quantizer](https://github.com/ModelCloud/QvQ/blob/4cbd1bc564c5faff7ce9fd0911b16a42ffefbeac/gptqmodel/quantization/qvq.py).

## Source findings

Local layer reconstruction ignores downstream sensitivity. YAQA approximates
each layer's Hessian of full-model teacher KL with input/output Kronecker
factors, then uses a quantizer-independent adaptive rounding algorithm.
This changes the geometry of rounding, not merely the datatype.

The paper distinguishes the real Fisher of the output distribution from
empirical Fisher based on task labels. These are not interchangeable. Sketch A
and Sketch B are approximation choices, not a full dense model Hessian.
Reported KL improvements are empirical and depend on model, quantizer and
experimental recipe; they are not a guaranteed QVQ recovery percentage.
[Paper, §§1–3](https://arxiv.org/html/2505.22988v1)

## Repository evidence

`qvq_yaqa.py` describes full-model real-Fisher collection for YAQA-v3. Its
Sketch-B code explicitly treats one complete sequence as a sample; valid token
count is not a count of independent sequences. Preserve both counts and the
collection protocol when comparing calibration runs.

The source includes explicit checks for activation-aware collection. At the
audited revision, the QVQ activation-module branch requires 8 bits. Its existence
must not be presented as established NVFP4 A4 support.

## QVQ implications

YAQA asks which representable weight candidate better preserves model behavior.
[EoRA](eora.md) asks how a small additive branch can compensate for an installed
operator. [NVFP4 scales](nvfp4-hybrid-ptq.md) define the numeric representation.
These may be combined, but they solve different problems.

For W4A4 research, verify whether the collected objective sees the intended
activation quantization, transformations and propagated inputs. A statistic
collected on an A16 teacher is not automatically a measurement of the deployed
A4 residual. Any change in collection semantics needs its own evidence.

Keep calibration manifests and held-out evaluation disjoint. Record damping,
sketch type, sampling, normalization, sequence/token counts and current versus
teacher model state. Do not silently relabel an existing experiment under a
different calibration protocol.

Use local reconstruction error to diagnose changes, then measure propagated
teacher KL/agreement and downstream task scores separately. A gain in one proxy
does not prove a universal task gain. Broader live-gradient or propagation-aware
recovery should be labeled as a QVQ adaptation unless its objective and algorithm
actually match the cited YAQA method.
