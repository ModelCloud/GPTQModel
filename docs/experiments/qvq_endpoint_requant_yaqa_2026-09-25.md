# Endpoint YAQA requantization on an already quantized QVQ model

This change separates endpoint-only YAQA capture from the original dense-model
decoder pass. When `requantize(..., embed_quant_config=QuantizeEmbedConfig(
embed_quant_mode=QuantizeEmbed.OUTPUT, embed_only=True))` targets an untied
`nn.Linear` head, YAQA seeds the autograd graph at that head's input. The
existing quantized decoder runs forward to produce its activations but does
not need a backward path. Decoder projections receive no new Fisher factors
and are not requantized. The ordinary dense-source YAQA rule remains in force
for decoder targets. Tied word embeddings are untied before YAQA preparation,
not after it.

The head's input and output Fisher factors still use the existing exact
Sketch-B/VAQA and GSQ math. This works for a bounded head and is tested with
a nondifferentiable decoder at W2.5, W3, W3.5 P32 and W4 planar, including
an actual GSQ optimization step. W4 is **not** a P32 rate in the current
format/kernel contract.

## Llama 3.2 1B Seed-7 feasibility result

The source checkpoint ties a `128256 × 2048` FP16 input embedding and
`lm_head` (~501 MiB). The original YAQA calibration source is
`yaqa182-nm10000.parquet`: 10,178 sequences, Seed 7, SHA256
`5a2429da9754040e16baf47c569c14267b92126f37caafd4fafc8d5bfb5c3f39`.
An audit found zero normalized GSM8K-Platinum test questions in the combined
YAQA calibration text. The 1,209-row test set remains held out.

The head's full FP32 output Gram would occupy
`128256² × 4 = 65,798,406,144 bytes = 61.28 GiB` before its transformed
copy, GSQ workspace, model, and CUDA runtime. Even the existing projected
collector materializes this Gram in the current solver. Endpoint preparation
therefore fails early once this Gram exceeds 1 GiB. A real-checkpoint smoke
reached that guard after automatically untying the head; it did not create a
new quantized artifact.

To make the production sweep possible, the next solver must retain a bounded
output Fisher representation through RHT, VAQA tile selection, GSQ objective,
and the dual FP32/FP64 oracle. Merely relaxing the dense-source guard or
selecting `streaming_projected` does not solve the output allocation. Output
channel blocking is one possible route, but it changes the weight format and
requires matching ZML head dispatch. A compressed input embedding additionally
needs a token-row lookup kernel. Quantizing only the head leaves the original
dense input embedding resident and adds a separate compressed head payload.

No GSM8K score or inference speedup is claimed for this lifecycle change.
The W2.5–W4 sweep and the 543/1209 full-suite quality gate remain pending
the bounded solver and serving integration.
