# F6/seed7 QVQ: ZML versus Transformers parity finding

Date: 2026-09-10
Status: confirmed local source of the first tensor divergence
Scope: Llama-3.2-1B-Instruct, F6/P32 seed 7, GSM8K-Platinum, SM80, layer-0
Q projection only. This report does not claim that one canary explains the
complete 1,209-example score difference.

## Finding

The first non-exact boundary is the layer-0 QVQ q_inner output. Transformers
and ZML receive the same captured prepared input and the same loaded payload,
but select different QVQ implementations:

- Transformers calls legacy planar qvq_cuda.gemv with V2B2-P32 bank_mode=3.
- ZML calls zml.qvq_p32_partials, which launches the P32 continuous-window
  block/WMMA implementation.

The decoded payload and input are the same. The difference is the floating-point
accumulation schedule, not tokenization, padding, input preparation, trellis
layout, Hadamard, RMSNorm, attention, or a generic XLA GEMM rewrite.

## Matched intervention

The probe used one full prepared tensor of shape [832, 2048] (830 logical
tokens and two physical pad rows), one layer-0 Q payload, transition bits 4,
V2B2-P32 bank mode 3, bank alternative ID 1, FP16 input, FP32 output, and
split_count=1.

| Arm / comparison | MAE | Maximum absolute | Different values | Exact |
|---|---:|---:|---:|---|
| Legacy GEMV vs Transformers full raw_inner | 0 | 0 | 0 / 1,703,936 | yes |
| Window P32 vs ZML selected q_inner | 0 | 0 | 0 / 2,048 | yes |
| Window P32 vs legacy GEMV | 1.6540302e-4 | 1.6174316e-3 | 1,702,027 / 1,703,936 | no |
| Legacy GEMV vs FP64 dense decoded-weight oracle | 2.5319814e-6 | 4.5776367e-5 | 1,126,454 / 1,703,936 | no |
| Window P32 vs FP64 dense decoded-weight oracle | 1.6540529e-4 | 1.6174316e-3 | 1,702,008 / 1,703,936 | no |

The legacy kernel is approximately 65x lower MAE than the window kernel against
the FP64 dense oracle. Isolated one-launch timing was 4.202 ms for legacy GEMV
and 0.394 ms for window P32; this is not a full-model throughput claim.

## Mathematical boundary and source

The legacy kernel performs scalar FP32 fused updates over per-thread K subsets,
then combines the two K slots and warp partials:

    t_i[j+1] = fl32_fma(x[k_ij], decode(w[k_ij]), t_i[j])
    p_i = fl32(t_i[0] + t_i[1])
    y = fl32(p_0 + p_1 + ... + p_7)  # ascending warp order

In this checkout, the scalar update is in
gptqmodel_ext/qvq/qvq_gemv_cuda.cu:491-495; the K-slot shuffle is at
:502-509; and the ascending warp reduction is at :513-523. Python maps
V2B2-P32 to bank_mode=3 and invokes the legacy operator at
gptqmodel/utils/qvq_cuda.py:1994-1996.

The window kernel uses Tensor Core MMA tiles:

    a_g[r+1] = MMA_f32_accumulate(x_g[r:r+16], decode(W_g[r:r+16]), a_g[r])
    y = a_0 + a_1 + ...

The instruction
mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 is emitted at
gptqmodel_ext/qvq/p32/qvq_p32_cuda.cu:128-171, invoked per row group at
:570-594, and stored at :601-609. Its fragment accumulation tree and
traversal are not required to round like the scalar-FMA tree, even though both
produce FP32 output.

## Provenance

Run ID: f6-seed7-residual-20260910
Arm ID: direct-qvq-kernel-v3-row149
QVQ commit: 03c326d1314f707016f5159c9ccfa81dabe78f9a
ZML-Ultra commit used by the capture: ffac15de3b2bf3d39a038edacf39570dd2913221
Torch: 2.13.0; Transformers: 5.15.1; GPTQModel: 7.4.0+ultra; Tokenicer: 0.0.14
GPU: GPU-724ea08e-67c3-c0ce-29bb-e6c48e7dde28 (SM80)
Build limits: MAX_JOBS=32, CMAKE_BUILD_PARALLEL_LEVEL=32, NVCC_THREADS=2

Model path:

    /monster/data/model/qvq/modelcloud-qvq__llama-3.2-1b-instruct__f6-p32__qvq-p32-gguf-exl3__yaqa125x__seed7__20260904__commit5c5979194dc0__aff65a505e88/qvq-p32

Direct intervention artifacts:

    /monster/data/model/qvq/modelcloud-qvq__llama-3.2-1b-instruct__f6-p32__qvq-p32-gguf-exl3__yaqa125x__seed7__20260904__commit5c5979194dc0__aff65a505e88/qvq-p32/evaluations/f6-seed7-residual-20260910__direct-qvq-kernel-v3.json
    /monster/data/model/qvq/modelcloud-qvq__llama-3.2-1b-instruct__f6-p32__qvq-p32-gguf-exl3__yaqa125x__seed7__20260904__commit5c5979194dc0__aff65a505e88/qvq-p32/evaluations/f6-seed7-residual-20260910__direct-qvq-kernel-v3.md

Reproducer:

    /root/venv-py3.14t-gil0/bin/python scripts/compare_f6_direct_qvq_kernels.py --model=/monster/data/model/qvq/modelcloud-qvq__llama-3.2-1b-instruct__f6-p32__qvq-p32-gguf-exl3__yaqa125x__seed7__20260904__commit5c5979194dc0__aff65a505e88/qvq-p32 --transformers-row=/monster/data/model/qvq/modelcloud-qvq__llama-3.2-1b-instruct__f6-p32__qvq-p32-gguf-exl3__yaqa125x__seed7__20260904__commit5c5979194dc0__aff65a505e88/qvq-p32/evaluations/f6-seed7-residual-20260910__transformers-full-q-prefill-v12-row149/row-149.pt --zml-capture=/monster/data/model/qvq/modelcloud-qvq__llama-3.2-1b-instruct__f6-p32__qvq-p32-gguf-exl3__yaqa125x__seed7__20260904__commit5c5979194dc0__aff65a505e88/qvq-p32/evaluations/f6-seed7-residual-20260910__zml-fixed-qvq-production-prep-149-657-v10-native-window.json --output=/monster/data/model/qvq/modelcloud-qvq__llama-3.2-1b-instruct__f6-p32__qvq-p32-gguf-exl3__yaqa125x__seed7__20260904__commit5c5979194dc0__aff65a505e88/qvq-p32/evaluations/f6-seed7-residual-20260910__direct-qvq-kernel-v3.json --run-id=f6-seed7-residual-20260910 --arm-id=direct-qvq-kernel-v3-row149 --warmups=2

## Fix boundary

This finding belongs at the QVQ/ZML dispatch and arithmetic-contract boundary,
not in upstream XLA. If exact Transformers parity is required, route ZML to the
legacy-compatible path or add a separately validated parity mode. Any faster
WMMA, fusion, grouping, or StableHLO lowering must pass the same-input raw
q_inner comparison and a matched target-workload speed/quality review. The full
benchmark impact remains a separate end-to-end measurement.
