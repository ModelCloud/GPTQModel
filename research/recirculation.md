# Recirculation: inference-time state feedback

## Sources

- Mozer, Siddiqui, Sawyer, Sanyal and Liu,
  [Recirculation, arXiv:2608.17981v1](https://arxiv.org/html/2608.17981v1)
  (Google DeepMind and University of Texas, Austin).
- [ModelCloud/Recirculation](https://github.com/ModelCloud/Recirculation),
  the implementation corresponding to the user's repository reference.
- [Reproduction README at f18c176](https://github.com/ModelCloud/Recirculation/blob/f18c176b5285daef41a5788a251341d61e9395d8/README.md).

## Source findings

Recirculation mixes a small, norm-matched amount of a token's deep residual
representation into a shallower representation. Recurrence spans depth and token
steps; it is not simply reading logits after looping the current token through
extra layers. Readout uses the first iteration.

The default mixture is convex, with source L2 norm matched to the destination.
The paper also studies ramping and adaptive coefficients with frozen original
weights. It reports improved Gemma3 perplexity and task results.

Its near-zero added generation-latency claim does not mean zero extra computation
or ordinary parallel prefill: sequential state updating constrains prefill.
The paper is an inference architecture intervention, not a quantization scale
recipe or a demonstration of QVQ PTQ recovery.
[Paper, §§2, 4 and 5](https://arxiv.org/html/2608.17981v1)

## Repository evidence and reproduction boundary

ModelCloud's README calls the project an independent, best-effort reproduction,
not the authors' official implementation or authoritative reference.

Its documented schedule forms a mixture from token `t`'s first-pass source and
destination outputs, replays that same token from `destination + 1` through the
upper stack, replaces its upper-layer KV entries, and retains first-pass logits
for readout. The corrected state must precede the next token's upper-stack work.

The implementation defines the zero-source-norm edge case and a zero-based
optional ramp. It withdraws results from an earlier delayed cross-token
intervention. Those old results must not be reused as recirculation evidence.
The reported dense FP16 experiments do not establish quantized recovery.
[Implementation record](https://github.com/ModelCloud/Recirculation/blob/f18c176b5285daef41a5788a251341d61e9395d8/README.md)

## Proposed QVQ investigation

Treat recirculation as an optional behavior-changing intervention. It differs
from [EoRA](eora.md), which adds a fitted approximation of operator error, and
from [NVFP4/KV scales](nvfp4-hybrid-ptq.md), which define numeric encoding.

Use a matched four-arm comparison:

| Weights/runtime | Ordinary inference | Recirculation |
|---|---|---|
| Dense reference | Baseline | Dense intervention effect |
| QVQ | Quantization effect | Combined effect and interaction |

Keep prompts, scoring, context, seeds and precision policy matched. Tune path and
coefficients on separate data. Report both teacher agreement and task utility:
a useful intervention may intentionally depart from the original teacher.

Before claiming PTQ recovery, determine whether recirculation reduces the
quantized-versus-dense gap under matched inference semantics, rather than merely
improving both models. Repeat with correction off/on only after the four-arm
baseline is understood.

Verify same-token replay, first-pass readout, cache replacement, sequence reset,
batch isolation and prefix-cache reuse. A cached prefix must carry all recurrent
state required by its continuation, not just ordinary KV entries. If combining
with quantized caches, establish when corrected entries are re-encoded and which
scales are used.

Measure prefill, decode, replay work, scheduling and additional state separately.
No QVQ recirculation integration or quality benefit is claimed by this note.
