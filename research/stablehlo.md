# StableHLO: portable operation semantics

The requested “stablehalo” is interpreted as **StableHLO**, the OpenXLA
operation specification.

## Primary source findings

The [overview](https://openxla.org/stablehlo) describes a portability layer
between frameworks and compilers. StableHLO defines operations and types;
it is not itself a backend compiler, kernel library or autotuner.

The [specification](https://openxla.org/stablehlo/spec) gives a `composite`
operation a decomposition that must preserve its semantics. An opaque
`custom_call` serves a different extensibility role. Dot algorithms describe
operand precision, accumulation type and whether imprecise intermediate
accumulation is permitted. An unsupported specified algorithm must produce an
error rather than silently substitute another. Uniform quantized types encode
integer storage with scales and zero points.

## Proposed QVQ representation

An exact decomposition is useful as a semantic reference only if it expresses
the actual operation, including layout decoding, scales, transforms and casts.
A mathematically similar dense dot is not automatically a faithful
decomposition of a deployed low-precision kernel.

Do not relabel a P32 trellis payload or NVFP4 block representation as ordinary
uniform integer quantization. State the encoding and scale metadata explicitly.
Use a custom boundary if the operation cannot be represented faithfully with
available semantics; document its contract separately.

A declared FP32 accumulator does not by itself establish identical results to
the native reduction sequence. Check operand conversion, intermediate
accumulation, reassociation and output conversion against the required parity
gate. These details matter when fitting [EoRA](eora.md) or comparing
[NVFP4 recovery](nvfp4-hybrid-ptq.md).

## Repository boundary

The [P32 ABI](https://github.com/ModelCloud/QvQ/blob/263ed4baf7be5e9547b4e731c9031bef5f48cf69/docs/kernels/qvq_p32_runtime_abi.md) documents caller-visible split outputs and ordered
FP32 reconstruction. Graph lowering must preserve that contract, including
which outputs are initialized. A generic reduction must not be assumed to
preserve the native ordering without inspecting and validating its lowering.

No StableHLO serialization, native-call registration or portable P32 execution
is demonstrated by this note. See [ZML](zml.md), [XLA](xla.md) and
[SSA/SASS analysis](ssa-sass.md) for the surrounding stack.
