# H100 P32 output Hadamard: native packed FP16 for base-only prefill

Date: 2026-09-24 UTC. QVQ base: `196049e32ac61b83123ec09559ccccbdab442e7f`.

The packed output-Hadamard butterfly previously converted each FP16 pair to
FP32, added/subtracted, then rounded back to FP16. For the `RankCount == 0`
specialization, native `__hadd2`/`__hsub2` retain the per-stage FP16 boundary
while shortening the instruction sequence. Rank-8-on specializations keep the
old FP32-conversion arithmetic. No dense weight cache, extra VRAM allocation,
new ABI, or environment-controlled dispatch is involved.

## Correctness and isolated latency

- Production P32 runtime smoke passed the base-only and Rank-8 Hadamard paths
  at M128 and N16/32/64/512/2048/8192, bitwise FP16 against the scalar
  oracle.
- Changed-input CUDA graph replay passed bitwise FP16 against the old library
  at M960 and N512/2048/8192, across four random cases and one finite
  signed-zero/subnormal-sized case, with changed scale vectors.
- Alternating same-process CUDA-event measurements of the base-only epilogue:

| M960 N | Control | Candidate | Speedup |
| ---: | ---: | ---: | ---: |
| 512 | 5.566 us | 4.758 us | 1.17x |
| 2048 | 14.510 us | 10.964 us | 1.32x |
| 8192 | 53.507 us | 38.786 us | 1.38x |

The Rank-8 fused projector/Hadamard and Rank-8 Hadamard specializations have
identical SASS opcode histograms between the control and scoped candidate.
This is not a claim that complete SASS binaries or run-to-run timings are
identical.

## Full continuous B128 GSM8K-Platinum gate

Same H100, 1,209 ordered requests, model seed7, ZML runner binary, B128/M960,
FA2, 544-page KV pool, Rank-8 prefill off/decode on, automatic terminal
dead-row elimination, GPU-local CPU cores 0,1,3,4,12,13. The only source
difference is the QVQ library. Control library SHA-256:
`e2477ef7e3d8f331b75bb53a440230b1b1874b0ecfb14c195f3f6dd723553a08`.
Scoped candidate library SHA-256:
`b109d8ffbaaef7e87e5f02f0ef1097c19f019c48e5de47ce65fe16b01d1bf879`.
The reference record is
`/var/tmp/b128-prefill-fa2-compact-commit-full-20260924.json` (SHA-256
`bc1f47aa974032dab8c4b8a40a8d053fd645a78558f96df794da301f9812208b`).

Useful throughput counts real tokens; padded throughput counts all issued
B128 slots, including padding. Every arm produced 1,209/1,209 exact reference
token streams, 543 correct, and zero invalid outputs.

| Arm | Useful prefill tok/s | Padded prefill tok/s | Useful decode tok/s | Padded decode tok/s | Padded decode/stream tok/s | Wall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Control earlier 1 | 71,700 | 82,700 | 10,710 | 12,477 | 97.47 | 25.393 s |
| Control earlier 2 | 73,072 | 84,282 | 10,711 | 12,479 | 97.49 | 25.134 s |
| Scoped candidate 1 | 73,473 | 84,745 | 10,555 | 12,296 | 96.07 | 25.245 s |
| Scoped candidate 2 | 73,588 | 84,877 | 10,575 | 12,320 | 96.25 | 25.196 s |
| Adjacent control after | 71,793 | 82,807 | 10,594 | 12,342 | 96.42 | 25.537 s |

Versus the adjacent control, padded/useful prefill improved 2.34–2.50%.
The decode spread is 0.18–0.37% lower; the Rank-8-on arithmetic is unchanged,
so we do not attribute this small measurement difference to a decode-kernel
improvement or regression. The full-suite evidence supports a modest prefill
advance, not the outstanding 120,000 padded tok/s B128 target.

Full JSON records (not committed):
`/var/tmp/h2-native-prefill-only-candidate-full-20260924.json`,
`/var/tmp/h2-native-prefill-only-candidate-repeat-full-20260924.json`, and
`/var/tmp/h2-native-prefill-only-control-after-full-20260924.json`.
