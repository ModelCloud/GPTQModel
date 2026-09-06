# Task-Based Tensor Computations on Modern GPUs

## Primary reference and finding

[Task-Based Tensor Computations on Modern GPUs, arXiv:2504.07004v1](https://arxiv.org/html/2504.07004v1),
especially the programming model and evaluation sections.

The paper studies a task-based programming approach to coordinating asynchronous
GPU units, including Hopper TMA and Tensor Cores. It treats execution as
dependent work across specialized units rather than assuming every operation
finishes synchronously in the issuing thread. Its evaluated implementations
provide evidence for that system, not a QVQ integration.

## Proposed QVQ experiment

Describe packed transfer, state decoding, matrix computation and reduction as
tasks with explicit dependencies and storage lifetimes. Determine whether
decoding can supply the consumer fast enough and whether the schedule's extra
buffers reduce occupancy. Compare with the existing kernel at matched shapes,
precision and packing.

Do not confuse task scheduling with arbitrary algebraic reordering. A consumer
still needs completed producer data, and reduction order remains part of the
numeric contract. A schedule must protect storage until asynchronous users
finish, even when issuing threads have moved on.

This is related background for [CUDA asynchronous pipelines](cuda-execution.md)
and [FlashAttention-3](flashattention-3.md), not evidence that QVQ implements the
paper's system. Use [Nsight](nsight-profiling.md) to measure actual overlap and
apply the repository's existing correctness gates before performance claims.
