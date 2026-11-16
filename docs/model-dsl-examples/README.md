<!-- SPDX-FileCopyrightText: 2026 ModelCloud.ai -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Model computation DSL: architecture samples

These samples stress the [DSL proposal](../model-computation-dsl.md) against six
concrete architectures. They are source-backed **design excerpts**, not complete
serialized model programs, new model support, or executed numerical tests.
Every shown block declares tensor inputs/outputs and its external resource
requirements, including an explicit empty set. A compiler must expand every
reused body, layer record, parameter binding and resource-use record before
execution; an abbreviated example is never an executable fallback.

Selections were checked on 2026-11-16. "Latest" is resolved to the IDs below,
not a moving alias inside a compiled artifact. Kimi's published model list has
no model named Flash; K3.1 Pro is the selected open-weight model. Its high-speed
API variant does not establish a separate checkpoint architecture. Muse Spark
Open is newer than the Glimmer model currently represented in this repository.

| Requested sample | Selected checkpoint and pinned revision | Stress on the DSL |
| --- | --- | --- |
| [Simple Llama](llama32.md) | `meta-llama/Llama-3.2-1B-Instruct`, `9213176726f574b556790deb65791e0c5aa438b6` | Conventional GQA, Llama-3 RoPE, typed per-layer KV and a tied embedding/head. |
| [Qwen 3.8 MoE](qwen38.md) | `Qwen/Qwen3.8-2.4T-A95B`, `4f885e51d6bbfee5866446d56d3f96f336c39723` | 69 recurrent + 23 full-attention layers, different state types, gated shared experts and reference/config compatibility. |
| [DeepSeek V4.1 Flash](deepseek41.md) | `deepseek-ai/DeepSeek-V4.1-Flash`, `314113f8a7dadb388215396ba5fbee2c610a04d6` | Cross-layer compressed KV/index/candidate resources, hyper-connections, Engram, vision and separate draft entry point. |
| [MiniMax Flash](minimax31.md) | `MiniMaxAI/MiniMax-M3.1-Flash`, `ce7f4df87d2e05262f5775f373dac4772f134041` | Dense/MoE variation, sparse block selection with its own key history, multimodal prefill and MXFP8 storage. |
| [Latest Muse](muse-spark.md) | `meta-models/Muse-Spark-Open-816B-A42B`, `4c0e167c8b7a18e1dd5ca76763ecb5d53c6a9601` | Local RoPE/full NoPE schedule, attention/residual gates, pooled MoE, distinct normalization bodies and image/audio entry paths. |
| [Latest Kimi](kimi31.md) | `moonshotai/Kimi-K3.1-Pro`, `d9e99b0792a5e93b78ea9bea62a68167e36ddd9e` | KDA + gated DSA, cross-layer AttnRes, latent MoE, native quantization and transactional calibration state. |

The Qwen choice is the large explicit MoE checkpoint; neither the 27B model
nor the separately named Flash-Next is silently substituted. The source
references on each page distinguish checkpoint facts, candidate reference
implementation behavior and unresolved qualification.

## Contracts these examples add to the common design

| Concern | Required common representation |
| --- | --- |
| Who needs what and when | Every block has typed resource ports and guarded semantic use points. Exact instance bindings refer to request-global IDs/views, not an ambient request container. |
| Different attention states | A heterogeneous record of KV, recurrent matrices, convolution histories, indexes and metadata; no universal untyped cache object. |
| Cross-layer sharing | Resource producer and reader instances are explicit. Keep a resource until its last future consumer and asynchronous completion, independently of subsets. |
| Residual state across layers | AttnRes and hyper-connection edges are real values/resources, not inferred additions to a layer's hidden-state argument. |
| Selection vs weighting | Router/index scores, correction biases, top-k, normalization, dispatch and combine are individual operations with numerical and tie contracts. |
| Native precision | Separate logical dtype, packed weight dtype, scale encoding, compute/accumulation and rounding. Persistent expanded weight caches remain forbidden. |
| Branches and entry points | Text, multimodal prefill, decode, draft, verify and calibration name their own required resources and joins. Unused resources are not passed to every block. |
| Incomplete evidence | Missing config, body, conversion, tie policy or backend qualification blocks execution of the affected specialization. Documentation coverage is not runtime support. |

In every page, `A=bf16` denotes the proposed reference activation specialization,
not an assertion that native quantized activations have no intermediate casts.
`C` is bounded request capacity, `S` the valid history length, `T` the current
chunk and `N=B*T`. Request metadata includes typed positions `i64[B,C]`,
validity `bool[B,C]` and lengths `i64[B]` where relevant. Resources default to
no aliasing; any alternative layout, compression, paging or prefix sharing
requires an explicit descriptor and validated binding. Resource access
shorthand `at a..b` denotes named semantic use/completion points, never a
wall-clock estimate. All operation precision and storage records must be
fully resolved in the artifact; these excerpts leave that expansion visible
as an implementation requirement.

Unless a page specifies an expanded padding policy, these excerpts exercise
contiguous unpadded token intervals (`valid=true`). Padding, packed-document
resets and all-masked queries require an explicit body and qualification; a
validity argument does not authorize silently skipping recurrent updates.

## Evidence boundaries

Public checkpoint configs were read for Qwen, DeepSeek, MiniMax and Kimi.
Llama and Muse checkpoint config downloads require gated access unavailable
in this session. Llama dimensions are corroborated by a pinned public mirror;
Muse dimensions come from the official public card and pinned Transformers
configuration/reference source. The samples identify these boundaries rather
than claiming a gated checkpoint/config or weights audit.

All Transformers source links use revision
`696c46bacf291909624c4f8d62e8016281aa762e`. Kimi additionally uses a pinned
independent NVIDIA NeMo AutoModel implementation. None of these sources is an
allowed runtime dependency of the proposed compiled DSL artifact.
