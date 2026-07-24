# CUDA-Q quantization feasibility probe

This directory contains a bounded research probe, not a production quantization backend.

## Decision

The useful quantum mapping is a GPTQ-like adaptive-rounding subproblem. With a fixed scale and one
floor/ceiling decision per weight, the Hessian-weighted error is a QUBO:

```text
min z in {0,1}^n  (w - lower - scale * z)^T H (w - lower - scale * z)
```

This is mathematically compatible with QAOA. It is not scalable to a normal GPTQ group on a
single-GPU state-vector simulator: a group of 128 weights requires 128 logical qubits even with the
restricted floor/ceiling encoding. The dense Hessian also creates all-to-all Ising couplings.

EoRA is not a useful CUDA-Q target. It must return explicit classical `A` and `B` factors after a
covariance-weighted truncated SVD. Quantum PCA/SVD or linear-system routines return quantum states
or sampled observables; loading dense classical matrices and reading the factors back removes the
claimed asymptotic advantage. The existing Cholesky plus randomized/cuSOLVER SVD path is the
appropriate production implementation.

## Environment

The setup is isolated from GPT-QModel's Python environment:

```bash
bash scripts/quantum_quantization/setup_cudaq.sh
```

It uses regular CPython 3.13 and pins CUDA-Q 0.15.0. The repository shell currently uses a
free-threaded CPython 3.14 build, for which the CUDA-Q CUDA 13 binary is unavailable.

Run the probe on physical GPU index 6 only:

```bash
bash scripts/quantum_quantization/run_gpu6.sh
```

The launcher resolves physical index 6 to its GPU UUID before setting `CUDA_VISIBLE_DEVICES`.
CUDA-Q therefore sees exactly one device, as logical GPU 0. Override the toy parameters after `--`,
for example:

```bash
bash scripts/quantum_quantization/run_gpu6.sh -- --qubits 10 --layers 4 --shots 20000
```

Run the classical FP64 AdjacentExact CUDA solver against RTN and production GPTQ across
2/3/4/8-bit diagonal, block, dense, ill-conditioned, symmetric, asymmetric, and outlier cases:

```bash
bash scripts/quantum_quantization/run_adjacent_exact_benchmark.sh
```

This launcher also resolves physical GPU 6 to its current UUID. It writes full hardware, build,
quality, and time-to-solution data to
`scripts/quantum_quantization/results/adjacent_exact_benchmark.json`.

Run the native full-coupling 64/128-variable branch-and-bound sweep with:

```bash
bash scripts/quantum_quantization/run_adjacent_native_benchmark.sh
```

This second launcher also resolves physical GPU 6 dynamically to a UUID. It compares symmetric
2/3/4/8-bit RTN, native AdjacentExact, and Classic GPTQ across fully dense, signed-correlated, and
ill-conditioned group-64/128 problems. The default premium sweep uses 256 prefix workers, up to
50,000 nodes per worker, and five timing repeats. It writes candidates, lower bounds, certificate
states, raw timing samples, and hardware/build metadata to
`scripts/quantum_quantization/results/adjacent_native_benchmark.json`.

The probe deliberately rejects more than 16 qubits. Its purpose is to verify the QUBO/Ising mapping
and compare QAOA with exhaustive search, not to imply a full-group simulation.

Run the 2-bit and 3-bit group-size-32 accuracy experiment with:

```bash
bash scripts/quantum_quantization/run_gpu6.sh --group32
```

This second probe uses FP64, a deterministic block-correlated 32-weight group, exact classical
blockwise enumeration, and depth-5 CUDA-Q QAOA. It runs both independent block samples and a final
combined 32-qubit state-vector sample. Use `--skip-full-state` after `--` for a cheap block-only
debug run.

## Real GPTQ bridge

Run the implemented GPTQ-to-CUDA-Q path with:

```bash
bash scripts/quantum_quantization/run_gptq_group32.sh
```

The launcher isolates physical GPU 6 by UUID and uses two environments in sequence:

1. GPT-QModel captures a real float32 Hessian through `GPTQ.add_batch`, obtains the row scale and
   zero point through `Quantizer.find_params`, builds the adjacent QUBO with
   `gptqmodel.quantization.adjacent`, and records the classic GPTQ result.
2. The isolated CUDA-Q environment validates the transported QUBO coefficients, maps them to Ising
   coefficients, and executes the FP64 QAOA comparison.

The Torch implementation is solver-agnostic. `quantize_adjacent_rows` accepts a callback that returns
one binary state per row group, so classical, CUDA-Q, or future hardware solvers can be substituted
without adding CUDA-Q to GPT-QModel's normal dependencies. It keeps the existing scale and zero
point, and its output codes remain compatible with the existing packer. This research path is not
enabled through `QuantizeConfig`.

The real-GPTQ one-row result was:

| Bits | RTN error | Classic GPTQ | Adjacent exact / CUDA-Q blockwise | Full32 best |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 3.843299390 | 1.789156808 | 1.405059805 | 1.559596550 |
| 3 | 1.488330530 | 0.212253659 | 0.236763813 | 0.294160467 |

Adjacent rounding improved on classic GPTQ for this 2-bit row but not at 3-bit. Classic GPTQ is not
restricted to the original weight's two neighboring codes because its sequential error feedback
updates the remaining working weights; this is an important limitation of the 32-qubit encoding.

## Single-GPU state-vector capacity

Direct CUDA-Q allocation and execution probes on GPU 6 established these boundaries:

| Precision | Maximum | Raw state | CUDA-Q GPU use | Next size |
| --- | ---: | ---: | ---: | --- |
| FP32 complex amplitudes | 33 qubits | 64.0 GiB | 66,049 MiB | 34 qubits requires 128 GiB and failed |
| FP64 complex amplitudes | 32 qubits | 64.0 GiB | 66,049 MiB | 33 qubits requires 128 GiB and failed |

Host-memory state-vector spill was disabled during the probes. Reproduce the accuracy-first FP64
boundary with:

```bash
CUDAQ_MAX_GPU_MEMORY_GB=94 CUDAQ_MAX_CPU_MEMORY_GB=0 \
    bash scripts/quantum_quantization/run_gpu6.sh \
    --capacity --precision fp64 --qubits 32
```

A 32-weight GPTQ adaptive-rounding group fits exactly in FP64 because every weight contributes one
binary floor/ceiling decision, regardless of whether the fixed codebook is 2-bit or 3-bit. If every
quantized code is allowed rather than only the adjacent two, binary encoding instead needs 64 qubits
for 2-bit or 96 qubits for 3-bit and does not fit this state-vector backend.

The completed group-size-32 run is recorded in [`quantum.md`](../../quantum.md). CUDA-Q blockwise
postselection matched the certified classical optimum at both bit widths. A single 20,000-shot
32-qubit sample improved over round-to-nearest but did not hit the joint optimum because the
per-block success probabilities multiply. This validates feasibility and the error objective, not a
quantum advantage.

## GPU 6 result

The default probe was run on physical GPU 6, UUID
`GPU-737e2423-874a-23a4-1126-dfbe3e77c294`:

```text
device             NVIDIA PG506-230, compute capability 8.0, 124 SMs, 96 GiB
driver / toolkit   610.43.02 / CUDA 13.0
PyTorch baseline   2.13.0+cu130
CUDA-Q             0.15.0, CUDA 13 `nvidia` state-vector target
problem            8 rounding bits, dense activation covariance, QAOA p=3
mapping error      3.331e-16 max absolute error over all 256 states

method             bits      Hessian-weighted cost
round-to-nearest   01000101  0.021548514867
classical optimum  01000001  0.007442758107
QAOA best sample   01000001  0.007442758107
```

The best QAOA sample was optimal, but that is not evidence of useful optimization by itself. With
10,000 shots, QAOA assigned 1.06% probability to the optimum versus 0.390625% for uniform random
sampling, only a 2.714x enrichment. Its expected cost was `0.052432266079`, worse than the single
round-to-nearest candidate. Exhaustive search checked all 256 states directly.

The result validates the algebra and CUDA-Q execution. It does not justify scaling QAOA into GPTQ:
the toy already required 1,500 classical objective evaluations, while a real 128-variable dense
group is outside the single-GPU state-vector regime.

## Primary references

- GPTQ: <https://arxiv.org/abs/2210.17323>
- EoRA: <https://arxiv.org/abs/2410.21271>
- CUDA-Q simulator limits: <https://nvidia.github.io/cuda-quantum/latest/using/backends/simulators.html>
- HHL output contract: <https://arxiv.org/abs/0811.3171>
- Quantum singular-value transformation: <https://arxiv.org/abs/1806.01838>
