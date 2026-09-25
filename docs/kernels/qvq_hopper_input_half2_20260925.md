# Native FP16 packed input Hadamard, 2026-09-25 UTC

## Change

The SM90 P32 raw input-Hadamard path uses native `half2` add/subtract for
each rounded butterfly. The previous implementation converted both halves
to FP32, added or subtracted, then converted each result back to FP16.
No projection weights, ABI, launch geometry, rank-8 policy, or output
epilogue changes. This affects the width-2,048 and width-8,192 raw paths,
including the exact fused SwiGLU/down-input variant from PR #374.

## Correctness and isolated speed

The raw ABI's FP16 oracle and changed-input CUDA Graph replay passed for
widths 2,048 and 8,192 with rows 1, 16, and 960; all 11 raw-ABI tests
passed. On H100, the isolated warmed 960-row CUDA Graph replay compared
64 launches per graph, 16 interleaved timing rounds, and byte-identical
inputs/scale/output. Outputs were bitwise equal:

| Width | Previous median µs | Native `half2` median µs | Speedup |
| ---: | ---: | ---: | ---: |
| 2,048 | 9.032 | 5.283 | 1.71× |
| 8,192 | 30.560 | 16.102 | 1.90× |

These are isolated kernel timings, not serving throughput. The direct
production ZML operator test and full suite provide the model gates.

## Matched full GSM8K-Platinum gate

The four arms used the same 1,209 dataset rows, order, rendered prompts,
checkpoint, B128/M960 continuous serving, 8,192 logical context, 736
physical KV pages, paged FA2, Rank-8 prefill off/decode on, eight Rank-8
projector warps, 45% BFC pool, GPU-local CPU set `0,1,3,4,12,13`, server
source, XLA autotune cache, and runner executable bytes. Only the raw QVQ
shared library differed. Sample pairing checked zero-based dataset index,
rendered prompt, and input IDs before comparing generated IDs.

| Arm | Useful prefill tok/s | Padded prefill tok/s | Useful decode tok/s | Padded decode tok/s | Padded decode tok/s/stream | Wall s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Control A | 93,731.4 | 108,110.8 | 10,481.0 | 12,391.8 | 96.81 | 22.213 |
| Native `half2` A | 99,049.4 | 114,244.6 | 10,491.6 | 12,404.4 | 96.91 | 21.626 |
| Control B | 94,327.4 | 108,798.2 | 10,456.2 | 12,362.5 | 96.58 | 22.172 |
| Native `half2` B | 98,909.5 | 114,083.2 | 10,564.2 | 12,490.2 | 97.58 | 21.566 |
| Control mean | 94,029.4 | 108,454.5 | 10,468.6 | 12,377.2 | 96.70 | 22.192 |
| Native `half2` mean | **98,979.4** | **114,163.9** | **10,527.9** | **12,447.3** | **97.24** | **21.596** |

Mean useful and padded prefill throughput improved **5.26%**. Wall time
fell 2.69%. Decode rose 0.57% in this crossed pair, though no decode code
changed. Both candidate runs produced **1,209/1,209 identical token streams**
against their corresponding controls, scored 543 correct, and had zero
invalid. The 120,000 padded prefill tok/s target remains open.

## Artifact identity

- Source control base: QVQ merge `fd992992fd5570e1a8a84f0a6bc6c4819750ca5b`.
  ZML source: merge `5ec560736cf8c2323c289a484ceb475e9b234322`,
  built with `--override_repository=qvq` pointing to this exact source
  tree. Inference server source: `fd48ded` (same server files as merge
  `3992f175e83436de581697ef38d82678c21218ce`).
- Build: `./bazel.sh build -c opt
  //examples/llm:llama_paged_token_runner --@zml//platforms:cuda=true
  --override_repository=qvq=/root/work/wt-qvq-prefill-input-hadd2-20260925
  --jobs=4`. The control and candidate runner executable SHA256 both equal
  `4e1e4d4c0c34a8e9d43d62061c351addfc6f56aed10228e7e2ce76881a7cde69`;
  the loaded raw-library SHA256 values are
  `ab27750ce3f2ed16e6082e2c763e92af669d59ca22f85a76c51c24bb97ddfa41`
  and `72682a74150d2154aa4401065b1fec8e59891f339a507e337df45caf18a29ef5`.
- Model:
  `/monster/data/model/qvq/modelcloud-qvq__llama-3.2-1b-instruct__f6-p32-r8-zml-cuda-v3__qvq-p32__yaqa125x__seed7__20260908`.
  Dataset Arrow SHA256:
  `b4c541a3b63d3d5045acc16dc64370b411384b2eb994b1d7bafa30d677fe4720`.
  Reference JSON SHA256:
  `29b1a19543c589ffc8a8d2383544217e01d327bfc039cc5ec575865ac9b8fc2e`.
- Server: `ZML_LLAMA_ATTENTION=fa2`,
  `--context=8192 --paged-batch-size=128 --paged-prefill-len=960
  --kv-pool-pages=736 --memory-fraction=0.45 --cpu-affinity=gpu-local
  --runner-arg=--rank8-project-warps=8`.
  Evaluator: `--rows=1209 --request-batch-size=1209
  --require-gpu-local-cpu-affinity --expected-rank8-project-warps=8`.
  XLA cache: `/var/tmp/zml-m960-no-regress-autotune-jQ6zN5`.
- Raw JSON outside Git:
  `/var/tmp/b128-{control-a,hadd2-a,control-b,hadd2-b}-probe-1209-20260925.json`.
  SHA256 in that order:

```text
fd3ab77d37642565defb20ffd4182136175f6b9059bdf77c98f5f33bf7e4bd38
1fc9e2d0be9f7f570b3ddd944e379da8cace6d5e784b5e3cd126e95ca8cb7778
a6ae85966dd98d136e84c25a8ff698b5ef7d9922f88775366c3b9d52895b50ed
68489256bdd8c85a9831aa8b75af76793294636478e17f18748c41fb994d9672
```

The scalar-to-native arithmetic screen was the only candidate code change.
No dense weight cache or additional model allocation was introduced.
