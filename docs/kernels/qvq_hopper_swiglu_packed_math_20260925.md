# Exact packed SwiGLU math for SM90 prefill, 2026-09-25 UTC

## Change and FP16 contract

The M960×8192 fused SwiGLU/down-input Hadamard path now evaluates the
logistic exponential with `__expf`, rounds it to FP16, then evaluates
`1 + exponential`, reciprocal, SiLU, and up-product as native `half2`
operations. Each operation still rounds at the original FP16 StableHLO
boundary. No projection, compressed weights, ABI, launch geometry, or
Rank-8 policy changes.

An exhaustive SM90 device screen over **every finite FP16 gate value**
found zero rounded-exponential differences between `expf` and `__expf`,
and zero rounded-sigmoid differences between FP32 reciprocal/FP16 store
and packed `h2rcp`. A second exhaustive screen across those gate values
and eight representative up values found zero final SwiGLU differences.
The raw M960×8192 output-Hadamard oracle also passed bitwise and on
changed-input CUDA Graph replay. The focused raw ABI suite passed 12/12.
The alternate `h2exp` native-half exponential needed four bit-pattern
corrections and ran slower, so it was removed from the implementation.

An isolated warmed M960×8192 CUDA Graph microbenchmark used 64 launches
per graph and 16 interleaved rounds, with byte-identical random FP16 gate,
up, and scale inputs. The original kernel measured 27.674 µs median;
the packed-math candidate measured 22.052 µs (**1.255× faster**), with
zero differing FP16 output elements. This is a local kernel result, not
the serving score.

## Matched full GSM8K-Platinum gate

The A/B/A/B arms held all 1,209 rows, order, rendered prompts, checkpoint,
B128/M960 continuous serving, 8,192 logical context, 736 physical KV
pages, paged FA2, Rank-8 prefill off/decode on, Rank-8 projector W8,
45% BFC pool, XLA cache, server code, and GPU-local CPU set
`0,1,3,4,12,13` fixed. The native runner executable was byte-identical;
only the loaded QVQ library changed. Paired output comparison verified
zero-based dataset index, rendered prompt, and input IDs before comparing
generated IDs.

| Arm | Useful prefill tok/s | Padded prefill tok/s | Useful decode tok/s | Padded decode tok/s | Padded decode per stream tok/s | Wall s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Control A | 98,883.5 | 114,053.2 | 10,460.9 | 12,368.1 | 96.63 | 21.671 |
| Candidate A | 100,229.9 | 115,606.2 | 10,488.1 | 12,400.2 | 96.88 | 21.506 |
| Control B | 98,779.8 | 113,933.6 | 10,455.1 | 12,361.2 | 96.57 | 21.687 |
| Candidate B | 99,941.4 | 115,273.4 | 10,454.9 | 12,360.9 | 96.57 | 21.565 |
| **Control mean** | **98,831.6** | **113,993.4** | **10,458.0** | **12,364.6** | **96.60** | **21.679** |
| **Candidate mean** | **100,085.6** | **115,439.8** | **10,471.5** | **12,380.6** | **96.72** | **21.536** |

Useful and padded prefill improved **1.27%** across the crossed runs;
wall time fell 0.66%. Decode was unchanged within run variation.
Both candidate arms produced **1,209/1,209 identical generated token
streams** against their paired controls, scored 543 correct, and had
zero invalid outputs. The 120,000 padded prefill tok/s target remains open.

## Reproduction and retained artifacts

- Source base: merged QVQ `c1c15ff872ddda1bd9a06d1de86498d9680b10ab`.
  ZML source: merged `a6da700e467de16098253bc504ff863d8f704e05`,
  built with `--override_repository=qvq` pointing to this source tree.
  Inference server source is unchanged from merge
  `3992f175e83436de581697ef38d82678c21218ce`.
- Build: `./bazel.sh build -c opt
  //examples/llm:llama_paged_token_runner --@zml//platforms:cuda=true
  --override_repository=qvq=/root/work/wt-qvq-prefill-swiglu-fast-exp-20260925
  --jobs=4`. Both control/candidate runner executable SHA256 values were
  `4e1e4d4c0c34a8e9d43d62061c351addfc6f56aed10228e7e2ce76881a7cde69`.
  Control QVQ library SHA256:
  `72682a74150d2154aa4401065b1fec8e59891f339a507e337df45caf18a29ef5`.
  Candidate QVQ library SHA256:
  `d7c9c6da81bdf3068a39ede9609e1c966961c3e2f743a82e9c9437f35da19c41`.
- Model:
  `/monster/data/model/qvq/modelcloud-qvq__llama-3.2-1b-instruct__f6-p32-r8-zml-cuda-v3__qvq-p32__yaqa125x__seed7__20260908`.
  Dataset Arrow SHA256:
  `b4c541a3b63d3d5045acc16dc64370b411384b2eb994b1d7bafa30d677fe4720`.
  Reference JSON SHA256:
  `29b1a19543c589ffc8a8d2383544217e01d327bfc039cc5ec575865ac9b8fc2e`.
- Server: `ZML_LLAMA_ATTENTION=fa2`, `--context=8192
  --paged-batch-size=128 --paged-prefill-len=960 --kv-pool-pages=736
  --memory-fraction=0.45 --cpu-affinity=gpu-local
  --runner-arg=--rank8-project-warps=8`. Evaluator: `--rows=1209
  --request-batch-size=1209 --require-gpu-local-cpu-affinity
  --expected-rank8-project-warps=8`. XLA cache:
  `/var/tmp/zml-m960-no-regress-autotune-jQ6zN5`.

Raw full-suite results remain outside Git at
`/var/tmp/b128-packed-{control-a,candidate-a,control-b,candidate-b}-probe-1209-20260925.json`.
SHA256 in that order:

```text
a44a772d6891b494d36cffbe49cf9d592a5c8b1e387bf70e5967d39c1fe0e87b
65738440a79e1a73a21e2567648d5b5a7342a6af0359ec878c4121a63e0ef9e1
7ff5344fa1615ccae7562c07e246a431f13769a41c3a25f48281bf7204fb1202
a18e985554631c254ebd21b143134709046c889e39f2bb27d722b16644a4754b
```

The exhaustive arithmetic screen, microbenchmark, and raw Nsight binaries
remain under `/var/tmp`, outside Git. The accepted source change is only
the fused SwiGLU arithmetic and its focused ABI regression test.
