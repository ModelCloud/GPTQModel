# Hopper M960 SwiGLU input transform

The optional `qvq_hadamard_input_swiglu_raw_launch` ABI combines Llama's
SwiGLU activation with the QVQ down projection's FP16 SU/Hadamard input
transform. It consumes the gate and up FP16 tensors directly. No dense weight
copy, persistent activation cache, or hidden CUDA stream is created. The caller
owns all buffers and launches on its supplied stream, including graph replay.

This specialization accepts only M960, K8192, FP16 gate/up/scale/output on
SM90. ZML selects it automatically for the matching P32 prefill down
projection when Rank-8 prefill is off. Other shapes, decode, and full Rank-8
prefill keep the established path.

## Arithmetic contract

The source StableHLO `logistic(f16)` lowers to an FP16 exponential, FP16 add,
and FP16 divide. The fused kernel rounds at each of those boundaries, then
rounds `gate * sigmoid`, `silu * up`, the SU product, the normalized value,
and every ascending Hadamard butterfly result to FP16. Leaving the logistic
intermediates in FP32 changed 13 of 128 matched B128 token streams in the
first prototype. Restoring their FP16 rounds removed that drift. The direct
M960×8192 ZML StableHLO oracle reported zero differing output elements; the
full 231-test ZML CUDA target passed.

## Measured production effect

The matched H100 native graph changed from 678 to 663 GPU activity nodes and
7.858 to 7.723 ms graph span. The separate production SwiGLU kernel and
N8192 input Hadamard averaged 14.99 and 33.51 µs per layer respectively;
the fused kernel averaged 45.46 µs. Both old and fused Hadamard kernels used
32 registers/thread, 17,408 bytes shared memory/CTA, and zero local memory.
The graph result includes overlap and cannot be inferred by adding those
individual kernel times.

On full GSM8K-Platinum, B128/M960, 1,209 requests, the crossed two-run means
were 93,524 → 94,512 useful and 107,871 → 109,011 padded prefill tok/s
(+1.06%). Both fused runs matched all 1,209 control token streams, with 543
correct and zero invalid in every arm. Mean padded decode was 12,427 →
12,391 tok/s (−0.29% within observed run variation). The target of 120,000
padded prefill tok/s remains open. The Inference-Ultra benchmark record has
the exact artifact hashes, server settings, individual runs, and quality
pairing.

The isolated CUDA microbenchmark is only a kernel diagnostic. Its standalone
SwiGLU reference uses a different launch geometry from XLA, so its larger
speed ratio must not be quoted as a serving gain.
