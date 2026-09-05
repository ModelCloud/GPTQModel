# Accumulation isolation (experiments 7/8)

The first three real F6 seed-7 projections (layer 0 q, gate, down) passed all
432 localized cases at MAE <=0.003 and max <=0.046875. Each projection uses the
nine requested row counts, FP32 accumulation, and promotion intervals
16/32/64/128/256 for FP16, emulated BF16, and blockwise FP32 partials.

The teacher reconstructs the original checkpoint in FP32 and applies FP32
SU/Hadamard/GEMM/Hadamard/SV on identical captured C4 inputs. Candidate MMA
operands are FP16; canonical raw codebook values were verified exactly
representable in FP16. This is an arithmetic isolation with resident decoded
weights: timing excludes decoding and is **not full P32 operator speed**.
C4 activations are evaluation inputs, never fitting data.

| Projection | Worst FP16-partial MAE | Worst emulated BF16-partial MAE |
|---|---:|---:|
| layer 0 q | 0.00042762 | 0.00245424 |
| layer 0 gate | 0.00011595 | 0.00064544 |
| layer 0 down | 0.00038685 | 0.00219116 |

FP16 partials compile to native FP16-output MMA. The BF16 experiment instead
rounds FP32 MMA results to BF16 after each K16 step; it must not be described as
native BF16 accumulation. Static PTX hashes/opcode flags are retained alongside
raw case metrics and timing samples in [results/accumulation](results/accumulation).
Executed instruction profiling, SASS review, integrated decoding, complete BPW,
and model-quality confirmation remain outstanding. No kernel is promoted.

On q projection, the fastest FP16-partial variants were approximately 0.962x,
1.000x, and 1.001x the FP32 baseline at M=1,16,2048 respectively. These measurements
do not establish a speed benefit. Layer 1 gate/down follow-ups are running.
