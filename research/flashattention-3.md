# FlashAttention-3: asynchronous attention and low precision

## Primary reference and findings

Jay Shah et al.,
[FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision,
v1](https://arxiv.org/html/2407.08608v1), especially §§3–4.

The paper develops Hopper attention schedules that overlap data movement,
matrix multiplication and softmax work using asynchronous execution and warp
specialization. Its low-precision path also addresses numerical error with
block quantization and incoherent processing. Reported speed and accuracy
results are specific to the attention implementations and evaluated hardware.

## Proposed QVQ implications

Study producer/consumer scheduling when packed decode, loads and matrix work
can overlap. Identify which work uses which execution resources and prove buffer
lifetimes before adding stages. See [CUDA execution](cuda-execution.md).

Do not transfer the paper's attention speedups to P32 projection kernels, or
its low-precision results to an NVFP4 A4 policy. Different operators, formats
and architectures require separate evidence. The current QVQ SM80 runtime is
not a Hopper implementation.

Keep attention algorithm changes distinct from weight-recovery changes.
If a new attention path alters propagated activations, re-evaluate downstream
[EoRA](eora.md) and [YAQA](yaqa.md) assumptions. Measure matched attention error,
full-model quality and prefill/decode latency independently.
