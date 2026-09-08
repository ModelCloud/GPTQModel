# ZML: model construction and compiler integration

## Primary sources and findings

[ZML](https://github.com/zml/zml) is an inference stack built with Zig and
OpenXLA/MLIR. Its [concepts guide](https://docs.zml.ai/learn/concepts/)
distinguishes Shape metadata, symbolic Tensor operations during compilation,
CPU Slice data and accelerator Buffer allocations. Model compilation can use
weight shapes before loading actual weight buffers. The resulting executable
is then run with device-resident inputs and parameters.

This distinction matters: a symbolic model describes computation; allocating
buffers alone does not expose that computation to optimization.

## QVQ evidence and proposed integration

The [P32 runtime ABI](https://github.com/ModelCloud/QvQ/blob/263ed4baf7be5e9547b4e731c9031bef5f48cf69/docs/kernels/qvq_p32_runtime_abi.md) accepts caller-owned buffers, workspace and a
CUDA stream without depending on Zig, PJRT or XLA. It gives integrations
responsibility for graph construction and tuning. This is an integration
boundary, not evidence of a completed upstream ZML backend.

Proposed design: represent normalization, transforms, scale application,
correction branches and legal reductions as graph operations where their
semantics can be expressed. Use a registered native call for packed P32
operations that require the existing kernel. Specify layouts, shapes, buffer
lifetimes and stream ownership explicitly.

External tuning is part of that call contract. ZML must be able to select the
P32 shape, N16 group boundaries, split wave, stage count, static-N policy,
threads/warps, and N tiles per block through the version-3 `qvq_p32` ABI. The
`*_tuned` entry points accept this policy and the launch-plan entry point
returns the exact graph-visible launches. A zero tuning field means native
QvQ policy; a nonzero field is validated and rejected if the loaded binary has
no matching specialization. In the current binary, N tiles per block is tied
to the selected warp count (`4 * warps` for scalar M1–4 and `warps` for block
M5–16), so ZML can tune the available warp/thread choices and pass the derived
N-tile value, but cannot request an unsupported independent tile count. This
prevents a bridge from silently measuring a different geometry. Calls through
the legacy entry points leave tuning policy with QvQ, including its local
autotuning.

The compiler cannot inspect arbitrary CUDA kernel internals through a call
name. Partial outputs can expose a subsequent reduction, but do not expose
packed decoding inside the product kernel. Preserve the ABI's ordered FP32
reduction requirement and untouched-region rules.

For QKV or gate/up projections, investigate shared-input work only when the
activation domain and scale policies agree. Output projection depends on
attention results; a shared layer boundary does not remove that dependency.
Keep recurrent state and cache updates explicit when considering
[recirculation](recirculation.md).

Validate the complete installed operator against its matched reference before
reusing [EoRA](eora.md) factors. Record ZML, XLA and runtime revisions, precision
policy and selected executable. Compilation visibility is an opportunity,
not proof of a latency or quality improvement.

See [XLA](xla.md) and [StableHLO](stablehlo.md) for the compiler and semantic
contracts. No new integration or benchmark was executed for this note.
