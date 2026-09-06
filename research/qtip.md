# QTIP: trellis quantization and incoherence processing

## Sources

- Tseng, Sun, Hou and De Sa,
  [QTIP: Quantization with Trellises and Incoherence Processing,
  arXiv:2406.11235v1](https://arxiv.org/html/2406.11235v1).
- [Authors' code](https://github.com/Cornell-RelaxML/qtip).
- QVQ [reference implementation](https://github.com/ModelCloud/QvQ/blob/4cbd1bc564c5faff7ce9fd0911b16a42ffefbeac/gptqmodel/quantization/qvq.py).

## Source findings

QTIP is weight-only PTQ. Trellis-coded quantization represents a weight sequence
as a path through code states, avoiding a conventional vector codebook whose
size grows exponentially with vector dimension. Viterbi finds paths under an
additive distortion objective.

Random Hadamard incoherence processing makes weights more suitable for Gaussian
codes. The bitshift trellis permits parallel decoding from bit histories;
stateful encoding does not imply mandatory serial inference. Lookup-only,
computed and hybrid codes trade arithmetic against lookup storage.

The paper uses an activation-weighted layer reconstruction objective and
discusses BlockLDLQ integration. Its results do not establish NVFP4 activation
calibration, FP8 KV calibration, or the exact QVQ P32 format.
[Paper, §§1–3 and appendix A.2](https://arxiv.org/html/2406.11235v1)

## Repository interpretation

The QVQ math module identifies QTIP and [YAQA](yaqa.md) as its published
foundations. Follow its actual decoder and metadata contract for compatibility;
an algorithmic relationship does not make upstream QTIP checkpoints and QVQ
checkpoints interchangeable.

[P32](p32.md) adds repository-specific segmentation/bank and packing choices.
Its continuous-window layout exposes the bit-history dependency without changing
the decoded weights. “Trellis” alone is insufficient to infer bank count,
transition width, codebook version, boundary handling or hardware support.

## QVQ implications and proposed checks

- Treat the quantizer, rounding objective and runtime representation separately.
  QTIP-inspired codes can be selected with a different objective without
  automatically changing the decoder.
- Compare decoder optimizations against the real direct-state-extraction
  implementation. Do not manufacture a serial baseline to claim a speedup.
- Preserve transform conventions and metadata when fitting [EoRA](eora.md);
  factor coordinates must match the runtime activation coordinates.
- Establish bitwise payload round-trip and decoded-weight identity for lossless
  repacks, then separately check floating-point kernel outputs. Identical weights
  do not force identical accumulation order.
- Measure decoder instruction cost, memory traffic, transforms and complete
  linear latency across prefill and decode. Weight-only throughput evidence
  does not establish W4A4 throughput.
- If converting decoded trellis weights to a hardware-native format, evaluate
  that extra quantization error against the actual runtime, not just the
  original trellis dequantization reference.

These checks are implementation guidance, not claims of new paper results.
