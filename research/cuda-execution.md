# CUDA execution: SMs, CUDA cores, Tensor Cores and asynchronous pipelines

## Hardware and synchronization references

An SM (streaming multiprocessor) schedules warps and contains execution units
and shared resources. CUDA cores execute ordinary arithmetic; Tensor Cores
perform supported matrix operations. An SM is not another name for a CUDA core.
See the [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html).

The [Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)
describes TMA as an extension of Ampere asynchronous copies: bulk tensor movement
can avoid register staging and support producer/consumer warp specialization.
TMA is a transfer mechanism, not a matrix multiply or dequantizer.

The [asynchronous copy guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html)
distinguishes elementwise LDGSTS copies and bulk TMA transfers.
The [barrier guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-barriers.html)
describes arrival phases and transaction tracking. Issuing a transfer is not
proof that its destination can be consumed or reused.

## Architecture boundaries

| Feature | Interpretation for QVQ research |
|---|---|
| SM80 / Ampere | Explore supported MMA and asynchronous global-to-shared copies; no Hopper TMA assumption |
| SM90 / Hopper | TMA and architecture-specific asynchronous matrix pipelines enable additional designs |
| Other targets | Check exact instruction, datatype, layout and capability requirements independently |

The [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)
is the authority for instruction-level targets and synchronization semantics.
Do not treat every MMA as asynchronous or reuse one generation's wait sequence
for another. Tensor Core availability does not establish native support for a
particular packed QVQ or NVFP4 representation.

## Proposed QVQ investigations

Pipeline packed loads, decoding and matrix work only after defining buffer
ownership. Specify when each stage is filled, ready, consumed and reusable.
Match barrier participation, phase and expected transfer completion; protect
shared storage until both readers and asynchronous engines are finished.
Handle tail tiles and exceptional paths without dropping required participants.

More stages can hide latency but consume shared memory and registers. Separate
producer warps may help overlap while reducing available compute warps.
Measure decode, transfer and matrix throughput rather than assuming TMA removes
the limiting work. Packed payload alignment and tensor-map eligibility need
their own checks.

The [current P32 ABI](https://github.com/ModelCloud/QvQ/blob/263ed4baf7be5e9547b4e731c9031bef5f48cf69/docs/kernels/qvq_p32_runtime_abi.md) intentionally targets SM80 and requires
distinct gated libraries for additional architectures. This note proposes no
SM90 implementation and reports no new measurements.

Read [FlashAttention-3](flashattention-3.md) and
[task-based tensor computations](task-based-tensor-computations.md) as concrete
research on overlap and scheduling. Their results motivate experiments; they
do not establish QVQ gains. Validate with [Nsight](nsight-profiling.md) and retain
the existing parity gates when changing accumulation or reductions.
